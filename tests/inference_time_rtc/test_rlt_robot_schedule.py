"""Check phase handoff on the standalone client's command clock, without hardware."""

import time

import numpy as np

from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCClientConfig, RTCRLTConfig
from lerobot.inference_time_rtc.protocol import RTCActionChunk, rlt_row_mask
from lerobot.inference_time_rtc.robot_client import RTCRobotClient
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig


class NoHardware:
    action_features = {"j1.pos": float, "j2.pos": float}
    observation_features = action_features

    def get_observation(self):
        raise AssertionError("The schedule test must use synthetic observations")


def make_client():
    cfg = RTCClientConfig(
        robot=PiperFollowerConfig(port="unused"),
        rtc=InferenceTimeRTCConfig(
            execution_horizon=4, inference_delay=1, queue_threshold=6, auto_delay=False,
        ),
        rlt=RTCRLTConfig(enabled=True, checkpoint="/models/actor.pt", start_step=4, end_step=12),
        actions_per_chunk=10,
    )
    client = RTCRobotClient(cfg, robot=NoHardware())
    client.chunk_size = 10
    client.server_policy_device = "cpu"
    client.request_id = 1
    initial = np.zeros((10, 2), np.float32)
    client.buffer.merge(RTCActionChunk(0, 0, initial, initial.copy()))
    client._capture_observation = lambda: ({}, time.monotonic_ns(), time.monotonic_ns())
    return client


def test_old_phase_reply_is_consumed_then_discarded_without_clearing_prefix():
    client = make_client()
    requests = []

    def exchange(request, begin, ready):
        requests.append(request)
        client.request_id += 1
        if len(requests) == 1:
            # Control crosses the start boundary while the first RPC is in flight.
            with client._dispatch_lock:
                for _ in range(4):
                    client.buffer.reserve()
                client._update_rlt_phase(4)
        raw = np.zeros((10, 2), np.float32)
        mask = rlt_row_mask(client.config.rlt, request, 10, 4)
        raw[mask] = 0.25
        return RTCActionChunk(
            request.request_id, request.start_step, raw, raw.copy(),
            rlt_applied=bool(mask.any()), rlt_action_mask=mask,
        ), 0.01

    client._exchange_chunk = exchange
    try:
        client.request_chunk()
        assert requests[0].episode_step == 0
        assert client.buffer.last_request_id == 0
        step, prefix, started = client.buffer.snapshot()
        assert started and step == 4 and len(prefix) == 6
        np.testing.assert_array_equal(prefix, np.zeros((6, 2)))
        assert client._phase_refresh and client.refill_event.is_set()
        client.request_chunk()
        assert requests[1].episode_step == 4
        np.testing.assert_array_equal(requests[1].prev_chunk_left_over, prefix)
        assert client.buffer.last_request_id == 2
        assert not client._phase_refresh
        # The continuation still protects four prefix rows before the actor.
        np.testing.assert_array_equal(client.buffer.original[:4], np.zeros((4, 2)))
        assert not client.buffer.reserve().rlt_applied
    finally:
        client.stop()


def test_skip_slots_cross_the_interval_but_paused_clock_does_not_advance():
    client = make_client()
    try:
        with client._dispatch_lock:
            client._update_rlt_phase(0)
            client._update_rlt_phase(5)
            reservation = client.buffer.reserve(skipped_steps=5)
        assert reservation.step == 5 and client._rlt_phase
        with client._dispatch_lock:
            client.buffer.reserve(skipped_steps=5, pause_on_exhaustion=True)
            assert client.buffer.motion_paused
            step = client.buffer.snapshot()[0]
            client._update_rlt_phase(step)
            epoch = client._rlt_epoch
            for _ in range(10):
                assert client.buffer.reserve(pause_on_exhaustion=True) is None
                client._update_rlt_phase(client.buffer.snapshot()[0])
        assert client.buffer.snapshot()[0] == step
        assert client._rlt_epoch == epoch
    finally:
        client.stop()
