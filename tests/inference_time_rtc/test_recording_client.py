"""Exercise RTC recording handoff with in-process RPCs and forbidden hardware calls."""

import pickle
import threading
import time
from concurrent import futures
from contextlib import contextmanager

import grpc
import numpy as np
import pytest

from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCRecoveryConfig
from lerobot.inference_time_rtc.protocol import (
    CHECKPOINT_PATH_HEADER,
    CHECKPOINT_SELECTION_HEADER,
    CURRENT_POSE_RECOVERY_HEADER,
    POLICY_TYPE_HEADER,
    PROTOCOL_VERSION,
    SESSION_HEADER,
    RTCActionChunk,
    RTCObservation,
    SessionInstructions,
)
from lerobot.inference_time_rtc.recording_client import RTCPolicyActionClient
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.transport import services_pb2, services_pb2_grpc


class FakeRobot:
    observation_features = {"j1.pos": float, "j2.pos": float}
    action_features = {"j1.pos": float, "j2.pos": float}

    def connect(self):
        raise AssertionError("The adapter must leave hardware connection to the recording loop")

    def disconnect(self):
        raise AssertionError("The adapter must leave hardware disconnection to the recording loop")

    def get_observation(self):
        raise AssertionError("The adapter must use the recording loop's observation")

    def send_action(self, action):
        raise AssertionError("The adapter must leave dispatch to the recording loop")


class MockRTC(services_pb2_grpc.AsyncInferenceServicer):
    def __init__(self):
        self.instructions = []
        self.observations = []
        self.finished = []
        self.pending = None
        self.busy = False
        self.owner = None
        self.block_requests = set()
        self.release = threading.Event()
        self.invalid_first_result = False
        self.advertise_selection = True

    def Ready(self, request, context):  # noqa: N802
        metadata = dict(context.invocation_metadata())
        if metadata.get("rtc-command") == "close" and metadata.get(SESSION_HEADER) == self.owner:
            self.pending = None
            self.owner = None
        ready = (
            ("rtc-protocol", PROTOCOL_VERSION), (POLICY_TYPE_HEADER, "pi05"),
            (CURRENT_POSE_RECOVERY_HEADER, "1"),
        )
        if self.advertise_selection:
            ready += ((CHECKPOINT_SELECTION_HEADER, "1"),)
        context.send_initial_metadata(ready)
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        if self.busy or self.pending is not None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "A request is still in flight")
        instructions = pickle.loads(request.data)
        assert isinstance(instructions, SessionInstructions)
        self.instructions.append(instructions)
        self.owner = dict(context.invocation_metadata())[SESSION_HEADER]
        context.send_initial_metadata(
            (
                ("rtc-chunk-size", str(instructions.actions_per_chunk)),
                ("rtc-policy-device", instructions.policy_device),
                (CHECKPOINT_PATH_HEADER, b"/models/chosen/pretrained_model"),
            )
        )
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        assert self.owner == dict(context.invocation_metadata())[SESSION_HEADER]
        observation = pickle.loads(b"".join(message.data for message in request_iterator))
        assert isinstance(observation, RTCObservation)
        assert not self.busy and self.pending is None
        self.observations.append(observation)
        self.pending = observation
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        assert self.pending is not None and not self.busy
        self.busy = True
        observation, self.pending = self.pending, None
        try:
            while observation.request_id in self.block_requests and not self.release.wait(0.01):
                if not context.is_active():
                    context.abort(grpc.StatusCode.CANCELLED, "Cancelled test inference")
            count = self.instructions[-1].actions_per_chunk
            raw = np.arange(count * 2, dtype=np.float32).reshape(count, 2) + observation.request_id * 1000
            result = RTCActionChunk(observation.request_id, observation.start_step, raw, raw + 100)
            if self.invalid_first_result and observation.request_id == 0:
                result = {"incompatible": True}
            self.finished.append(observation.request_id)
            return services_pb2.Actions(data=pickle.dumps(result))
        finally:
            self.busy = False


@contextmanager
def running_client(service=None, **overrides):
    service = service or MockRTC()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    options = {
        "enable": True,
        "backend": "rtc",
        "server_address": f"127.0.0.1:{port}",
        "policy_type": "pi05",
        "pretrained_name_or_path": "/models/chosen/pretrained_model",
        "policy_device": "cpu",
        "actions_per_chunk": 10,
        "rtc": InferenceTimeRTCConfig(
            execution_horizon=4, inference_delay=1, queue_threshold=6, auto_delay=False
        ),
        "connect_timeout_s": 1,
        "rpc_timeout_s": 3,
    }
    options.update(overrides)
    client = RTCPolicyActionClient(RemotePolicyRecordConfig(**options), FakeRobot(), fps=20)
    try:
        yield client, service
    finally:
        service.release.set()
        client.stop()
        server.stop(0).wait()


def wait_until(predicate, timeout=3):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("Timed out waiting for the mock RTC worker")


def observation():
    return {"j1.pos": 1.0, "j2.pos": 2.0}


def due_action(client, timestep=0):
    client._next_command_at = None
    return client.get_action(observation(), "task", timestep)


def bootstrap(client):
    client.start()
    client.reset()
    assert client.get_action(observation(), "task", 0) is None
    wait_until(lambda: client.action_ready)


def test_async_configuration_retains_original_defaults():
    cfg = RemotePolicyRecordConfig(enable=True, policy_type="pi05", pretrained_name_or_path="model")
    assert cfg.backend == "async"
    assert cfg.actions_per_chunk == 1
    assert cfg.policy_device == "cpu"


@pytest.mark.parametrize(
    "overrides",
    [
        {"backend": "unknown"},
        {"rpc_timeout_s": 0},
        {"connect_timeout_s": float("nan")},
        {"actions_per_chunk": 5},
        {"action_sg_window": 6},
        {"policy_device": "cuda:-1"},
    ],
)
def test_invalid_recording_options_are_rejected(overrides):
    options = {
        "enable": True,
        "backend": "rtc",
        "policy_type": "pi05",
        "pretrained_name_or_path": "model",
        "actions_per_chunk": 50,
    }
    options.update(overrides)
    with pytest.raises(ValueError):
        RemotePolicyRecordConfig(**options)


def test_start_prepares_model_without_touching_hardware_or_duplicate_initial_warmup():
    with running_client() as (client, service):
        bootstrap(client)
        assert len(service.instructions) == 1
        instructions = service.instructions[0]
        assert instructions.actions_per_chunk == 10 and instructions.fps == 20
        assert instructions.pretrained_name_or_path == "/models/chosen/pretrained_model"
        assert client.policy_id == "/models/chosen/pretrained_model"
        assert due_action(client) == {"j1.pos": 100.0, "j2.pos": 101.0}
        assert client.last_reservation.step == 0


def test_get_action_and_takeover_never_wait_for_blocked_rpc():
    service = MockRTC()
    service.block_requests.add(0)
    with running_client(service) as (client, service):
        client.start()
        started = time.perf_counter()
        assert client.get_action(observation(), "task", 0) is None
        assert time.perf_counter() - started < 0.1
        wait_until(lambda: service.busy)
        started = time.perf_counter()
        client.pause()
        assert time.perf_counter() - started < 0.1
        assert client.motion_paused
        assert client.get_action(observation(), "task", 1) is None


def test_takeover_discards_inflight_result_then_restarts_without_old_prefix():
    service = MockRTC()
    service.block_requests.add(0)
    with running_client(service) as (client, service):
        client.start()
        assert client.get_action(observation(), "old task", 0) is None
        wait_until(lambda: service.busy)
        client.pause()
        client.reset()
        assert client.get_action(observation(), "fresh task", 10) is None
        assert len(service.instructions) == 1
        assert not client.action_ready
        service.release.set()
        wait_until(lambda: client.action_ready)
        assert len(service.instructions) == 2
        assert len(service.observations) == 2
        fresh = service.observations[1]
        assert fresh.start_step == 0 and fresh.prev_chunk_left_over is None
        assert fresh.observation["task"] == "fresh task"
        assert due_action(client, 11) == {"j1.pos": 1100.0, "j2.pos": 1101.0}
        assert client.error is None


def test_observations_are_copied_before_rpc_serialization():
    with running_client() as (client, service):
        client.start()
        image = np.zeros((3, 4, 3), dtype=np.uint8)
        obs = {**observation(), "camera": image}
        assert client.get_action(obs, "task", 0) is None
        image[:] = 255
        wait_until(lambda: bool(service.observations))
        np.testing.assert_array_equal(service.observations[0].observation["camera"], np.zeros_like(image))


def test_slow_frames_skip_expired_targets_without_catchup_dispatch():
    with running_client() as (client, service):
        bootstrap(client)
        assert due_action(client) == {"j1.pos": 100.0, "j2.pos": 101.0}
        client._next_command_at = time.monotonic() - 0.125
        command = client.get_action(observation(), "task", 1)
        assert command == {"j1.pos": 106.0, "j2.pos": 107.0}
        assert client.last_reservation.step == 3
        assert client.last_reservation.chunk_index == 3


def test_refill_uses_model_space_prefix_and_accounts_for_the_reserved_slot():
    with running_client() as (client, service):
        bootstrap(client)
        for timestep in range(4):
            assert due_action(client, timestep) is not None
        wait_until(lambda: client._buffer.last_request_id == 1)
        refill = service.observations[1]
        assert refill.start_step == 3
        np.testing.assert_array_equal(refill.prev_chunk_left_over[0], [6, 7])
        command = due_action(client, 4)
        assert command == {"j1.pos": 1102.0, "j2.pos": 1103.0}
        assert client.last_reservation.request_id == 1
        assert client.last_reservation.chunk_index == 1


def test_exhaustion_latches_motion_while_inference_continues_until_explicit_reset():
    service = MockRTC()
    service.block_requests.add(1)
    with running_client(service) as (client, service):
        bootstrap(client)
        for timestep in range(4):
            assert due_action(client, timestep) is not None
        wait_until(lambda: service.busy)
        for timestep in range(4, 10):
            assert due_action(client, timestep) is not None
        assert due_action(client, 10) is None
        assert client.motion_paused
        service.release.set()
        wait_until(lambda: 1 in service.finished)
        assert client.get_action(observation(), "task", 11) is None
        assert client.motion_paused and not client.action_ready
        client.pause()
        client.reset()
        assert client.get_action(observation(), "task", 12) is None
        wait_until(lambda: client.action_ready)
        assert not client.motion_paused
        assert service.observations[-1].prev_chunk_left_over is None
        assert due_action(client, 13) is not None


@pytest.mark.parametrize("outcome", ["resume", "takeover", "expired"])
def test_rollout_recovers_with_fresh_pose_and_full_chunk_or_defers_to_takeover(outcome, monkeypatch):
    service = MockRTC()
    service.block_requests.update({1, 2})
    with running_client(
        service,
        actions_per_chunk=50,
        rtc=InferenceTimeRTCConfig(execution_horizon=20, queue_threshold=30),
        recovery=RTCRecoveryConfig(enabled=True),
    ) as (client, service):
        bootstrap(client)
        for timestep in range(20):
            assert due_action(client, timestep) is not None
        wait_until(lambda: len(service.observations) == 2)
        for timestep in range(20, 50):
            assert due_action(client, timestep) is not None
        assert due_action(client, 50) is None
        assert client.motion_paused

        fresh = {"j1.pos": 14.0, "j2.pos": 25.0}
        assert client.get_action(fresh, "task", 51) is None
        service.release.set()
        wait_until(lambda: client._pending_recovery is not None)
        recovery = service.observations[2]
        assert recovery.observation == {**fresh, "task": "task"}
        assert recovery.prev_chunk_left_over is None
        assert recovery.recovery_prefix_steps == recovery.inference_delay == 6
        assert recovery.start_step == 51

        if outcome == "takeover":
            client.pause()
            client.reset()
            assert client.get_action(observation(), "task", 52) is None
            wait_until(lambda: client.action_ready)
            assert service.observations[3].recovery_prefix_steps == 0
            assert due_action(client, 53) == {"j1.pos": 3100.0, "j2.pos": 3101.0}
        elif outcome == "expired":
            expired_snapshot, _ = client._pending_recovery
            expired = client._recovery_expired
            monkeypatch.setattr(
                client, "_recovery_expired", lambda snapshot: snapshot is expired_snapshot or expired(snapshot)
            )
            assert client.get_action(fresh, "task", 52) is None
            assert client.motion_paused
            wait_until(lambda: client._pending_recovery is not None)
            assert client.get_action(fresh, "task", 53) == {"j1.pos": 3100.0, "j2.pos": 3101.0}
            assert service.observations[3].recovery_prefix_steps == 6
            assert client._buffer.qsize() == 49
        else:
            assert client.get_action(fresh, "task", 52) == {"j1.pos": 2100.0, "j2.pos": 2101.0}
            assert client.last_reservation.chunk_index == 0
            assert client.last_reservation.request_id == 2
            assert client._buffer.qsize() == 49
            assert len(service.instructions) == 1


def test_late_rollout_chunk_leaves_old_actions_to_drain_before_recovery():
    service = MockRTC()
    service.block_requests.add(1)
    with running_client(service, recovery=RTCRecoveryConfig(enabled=True, prefix_steps=3)) as (client, service):
        bootstrap(client)
        for timestep in range(4):
            assert due_action(client, timestep) is not None
        wait_until(lambda: len(service.observations) == 2)
        for timestep in range(4, 7):
            assert due_action(client, timestep) is not None
        service.release.set()
        wait_until(lambda: client._buffer.draining)
        for timestep in range(7, 10):
            assert due_action(client, timestep) is not None
            assert client.last_reservation.request_id == 0
        assert len(service.observations) == 2
        assert due_action(client, 10) is None
        assert due_action(client, 11) is None
        wait_until(lambda: client._pending_recovery is not None)
        assert due_action(client, 12) == {"j1.pos": 2100.0, "j2.pos": 2101.0}


def test_invalid_result_holds_and_allows_human_takeover_before_recovery():
    service = MockRTC()
    service.invalid_first_result = True
    with running_client(service) as (client, service):
        client.start()
        assert client.get_action(observation(), "task", 0) is None
        wait_until(lambda: client.error is not None)
        assert client.motion_paused
        assert client.get_action(observation(), "task", 1) is None
        client.pause()
        client.reset()
        assert client.get_action(observation(), "task", 2) is None
        wait_until(lambda: client.action_ready)
        assert client.error is None


def test_missing_model_selection_capability_fails_before_hardware():
    service = MockRTC()
    service.advertise_selection = False
    with running_client(service) as (client, service):
        with pytest.raises(RuntimeError, match="checkpoint_root"):
            client.start()
        assert not service.instructions and not service.observations
