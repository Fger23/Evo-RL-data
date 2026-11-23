"""Check episode-step RLT scheduling through real loopback recording RPCs."""

import pickle
import threading

import grpc
import numpy as np
import pytest
from test_recording_client import MockRTC, due_action, observation, running_client, wait_until

from lerobot.inference_time_rtc.configs import RTCRecoveryConfig, RTCRLTConfig
from lerobot.inference_time_rtc.protocol import (
    CHECKPOINT_SELECTION_HEADER,
    CURRENT_POSE_RECOVERY_HEADER,
    POLICY_TYPE_HEADER,
    PROTOCOL_VERSION,
    RLT_HEADER,
    RLT_PHASE_HEADER,
    SESSION_HEADER,
    RTCActionChunk,
    rlt_row_mask,
)
from lerobot.transport import services_pb2


class ScheduledRTC(MockRTC):
    """Use the real interval mask; gates make reply ordering deterministic."""

    def __init__(self):
        super().__init__()
        self.results = {}
        self.request_gates = {}
        self.advertise_phase = True

    def allow(self, request_id):
        self.request_gates.setdefault(request_id, threading.Event()).set()

    def Ready(self, request, context):  # noqa: N802
        metadata = dict(context.invocation_metadata())
        if metadata.get("rtc-command") == "close" and metadata.get(SESSION_HEADER) == self.owner:
            self.pending = None
            self.owner = None
        ready = (
            ("rtc-protocol", PROTOCOL_VERSION),
            (POLICY_TYPE_HEADER, "pi05"),
            (CURRENT_POSE_RECOVERY_HEADER, "1"),
            (CHECKPOINT_SELECTION_HEADER, "1"),
            (RLT_HEADER, "1"),
        )
        if self.advertise_phase:
            ready += ((RLT_PHASE_HEADER, "1"),)
        context.send_initial_metadata(ready)
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        assert self.pending is not None and not self.busy
        self.busy = True
        captured, self.pending = self.pending, None
        gate = self.request_gates.setdefault(captured.request_id, threading.Event())
        try:
            while (
                captured.request_id in self.block_requests
                and not self.release.is_set()
                and not gate.wait(0.01)
            ):
                if not context.is_active():
                    context.abort(grpc.StatusCode.CANCELLED, "Cancelled scheduled mock inference")
            instructions = self.instructions[-1]
            count = instructions.actions_per_chunk
            reference = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + captured.request_id * 1000
            if captured.prev_chunk_left_over is not None:
                prefix = captured.prev_chunk_left_over
                reference[: len(prefix)] = prefix
            mask = rlt_row_mask(instructions.rlt, captured, count, instructions.rtc.execution_horizon)
            actions = reference.copy()
            actions[mask] += 50000
            result = RTCActionChunk(
                captured.request_id,
                captured.start_step,
                actions,
                actions + 100,
                rlt_applied=bool(mask.any()),
                rlt_action_mask=mask,
                rlt_reference_actions=reference + 100,
            )
            result.validate(2)
            self.results[captured.request_id] = result
            self.finished.append(captured.request_id)
            return services_pb2.Actions(data=pickle.dumps(result))
        finally:
            self.busy = False


def scheduled_config(start=100, end=200):
    return RTCRLTConfig(enabled=True, checkpoint="/models/rlt/test_actor.pt", start_step=start, end_step=end)


def start_at(client, episode_step):
    client.start()
    client.reset()
    assert client.get_action(observation(), "task", episode_step) is None
    wait_until(lambda: client.action_ready or client.error is not None)
    assert client.error is None


def test_phase_change_refreshes_above_threshold_and_reservations_track_each_actor_row():
    service = ScheduledRTC()
    with running_client(service, rlt=scheduled_config(100, 106)) as (client, _):
        start_at(client, 0)
        assert not service.results[0].rlt_action_mask.any()
        original_buffer = client._buffer
        # The episode counter is 100 while the buffer clock is still zero.
        assert due_action(client, 100) is not None
        assert client.last_reservation.step == 0
        assert not client.last_reservation.rlt_applied
        assert client._buffer.qsize() > client.cfg.rtc.queue_threshold
        wait_until(lambda: client._buffer.last_request_id == 1 or client.error is not None)
        assert client.error is None
        assert client._buffer is original_buffer
        request = service.observations[1]
        assert request.episode_step == 100 and request.start_step == 0
        assert request.prev_chunk_left_over is not None
        np.testing.assert_array_equal(
            service.results[1].rlt_action_mask,
            np.array([False, False, False, False, True, True, False, False, False, False]),
        )
        assert len(service.instructions) == 1
        flags = []
        for episode_step in range(101, 106):
            assert due_action(client, episode_step) is not None
            flags.append(client.last_reservation.rlt_applied)
        assert flags == [False, False, False, True, True]


def test_end_boundary_forces_plain_vla_refresh_even_while_queue_is_above_threshold():
    service = ScheduledRTC()
    with running_client(service, rlt=scheduled_config(100, 106)) as (client, _):
        start_at(client, 105)
        assert client._rlt_phase is True
        np.testing.assert_array_equal(service.results[0].rlt_action_mask, [True] + [False] * 9)
        previous_buffer = client._buffer
        assert due_action(client, 106) is not None
        assert client._rlt_phase is False
        assert client._buffer.qsize() > client.cfg.rtc.queue_threshold
        wait_until(lambda: client._buffer.last_request_id == 1 or client.error is not None)
        assert client.error is None
        assert client._buffer is previous_buffer
        request = service.observations[1]
        assert request.episode_step == 106 and request.start_step == 0
        assert request.prev_chunk_left_over is not None
        assert not service.results[1].rlt_applied
        assert not service.results[1].rlt_action_mask.any()
        assert len(service.instructions) == 1


@pytest.mark.parametrize("resume_step,expected_active", [(150, True), (240, False)])
def test_takeover_reset_preserves_the_supplied_episode_timestep(resume_step, expected_active):
    service = ScheduledRTC()
    with running_client(service, rlt=scheduled_config()) as (client, _):
        start_at(client, 120)
        assert service.observations[0].episode_step == 120
        assert due_action(client, 120) is not None
        assert client.last_reservation.rlt_applied
        client.pause()
        assert client.get_action(observation(), "human", 130) is None
        client.reset()
        assert client.get_action(observation(), "resumed", resume_step) is None
        wait_until(lambda: client.action_ready or client.error is not None)
        assert client.error is None
        resumed = service.observations[-1]
        assert resumed.episode_step == resume_step
        assert resumed.start_step == 0 and resumed.prev_chunk_left_over is None
        assert resumed.observation["task"] == "resumed"
        assert client._rlt_phase is expected_active
        assert len(service.instructions) == 2
        assert due_action(client, resume_step + 1) is not None
        assert client.last_reservation.rlt_applied is expected_active


def test_phase_change_discards_old_inflight_reply_without_clearing_queue_or_prefix():
    service = ScheduledRTC()
    service.block_requests.update({1, 2})
    with running_client(service, rlt=scheduled_config()) as (client, _):
        start_at(client, 0)
        for episode_step in range(4):
            assert due_action(client, episode_step) is not None
        wait_until(lambda: service.busy)
        assert len(service.observations) == 2
        previous_buffer = client._buffer
        _, expected_prefix, _ = previous_buffer.snapshot()
        old_generation = client._generation
        assert due_action(client, 100) is not None
        assert client._generation == old_generation + 1
        assert client._buffer is previous_buffer and client._buffer.last_request_id == 0
        assert client._buffer.qsize() == 5
        service.allow(1)
        wait_until(lambda: len(service.observations) == 3)
        assert 1 in service.finished
        assert client._buffer.last_request_id == 0  # Old phase's reply was consumed but not installed.
        assert client._buffer.qsize() == 5
        refreshed = service.observations[2]
        assert refreshed.episode_step == 100 and refreshed.start_step == 4
        np.testing.assert_array_equal(refreshed.prev_chunk_left_over, expected_prefix)
        service.allow(2)
        wait_until(lambda: client._buffer.last_request_id == 2 or client.error is not None)
        assert client.error is None
        assert client._buffer is previous_buffer
        assert len(service.instructions) == 1


def test_recovery_request_uses_fresh_observation_episode_step_and_row_provenance():
    service = ScheduledRTC()
    service.block_requests.update({1, 3})
    with running_client(
        service,
        rlt=scheduled_config(),
        recovery=RTCRecoveryConfig(enabled=True, prefix_steps=3),
    ) as (client, _):
        start_at(client, 0)
        for episode_step in range(4):
            assert due_action(client, episode_step) is not None
        wait_until(lambda: service.busy)
        for episode_step in range(4, 10):
            assert due_action(client, episode_step) is not None
        assert due_action(client, 10) is None
        assert client.motion_paused
        fresh = {"j1.pos": 14.0, "j2.pos": 25.0}
        assert client.get_action(fresh, "fresh recovery", 120) is None
        service.allow(1)
        wait_until(lambda: client._pending_recovery is not None or client.error is not None)
        assert client.error is None
        recovery = service.observations[2]
        assert recovery.episode_step == 120 and recovery.start_step == 11
        assert recovery.observation == {**fresh, "task": "fresh recovery"}
        assert recovery.recovery_prefix_steps == recovery.inference_delay == 3
        assert recovery.prev_chunk_left_over is None
        np.testing.assert_array_equal(service.results[2].rlt_action_mask, np.array([False] * 3 + [True] * 7))
        flags = []
        for episode_step in range(121, 125):
            assert due_action(client, episode_step) is not None
            flags.append(client.last_reservation.rlt_applied)
            assert client.last_reservation.request_id == 2
        assert flags == [False, False, False, True]


def test_scheduled_rlt_requires_interval_capability_before_any_observation():
    service = ScheduledRTC()
    service.advertise_phase = False
    with running_client(service, rlt=scheduled_config()) as (client, _):
        with pytest.raises(RuntimeError, match="updated RTC server|step interval|interval"):
            client.start()
        assert not service.instructions and not service.observations


def test_failed_old_phase_rpc_releases_pending_observation_before_refreshing_session():
    class FailedHandoffRTC(ScheduledRTC):
        def __init__(self):
            super().__init__()
            self.failed_rpc_started = threading.Event()
            self.fail_rpc = threading.Event()

        def GetActions(self, request, context):  # noqa: N802
            if self.pending is not None and self.pending.request_id == 1:
                # Simulate an upload that succeeded, followed by a GetActions
                # transport failure before the pending observation was consumed.
                self.failed_rpc_started.set()
                while not self.fail_rpc.wait(0.01) and not self.release.is_set():
                    if not context.is_active():
                        context.abort(grpc.StatusCode.CANCELLED, "Cancelled handoff test")
                context.abort(grpc.StatusCode.UNAVAILABLE, "Simulated handoff failure")
            return super().GetActions(request, context)

    service = FailedHandoffRTC()
    with running_client(service, rlt=scheduled_config()) as (client, _):
        start_at(client, 0)
        for step in range(4):
            assert due_action(client, step) is not None
        wait_until(service.failed_rpc_started.is_set)
        previous_buffer = client._buffer
        assert due_action(client, 100) is not None
        service.fail_rpc.set()
        wait_until(lambda: client._buffer.last_request_id == 2 or client.error is not None)
        assert client.error is None
        assert len(service.instructions) == 2
        assert client._buffer is previous_buffer
        assert service.observations[-1].episode_step == 100
        assert service.observations[-1].prev_chunk_left_over is not None
