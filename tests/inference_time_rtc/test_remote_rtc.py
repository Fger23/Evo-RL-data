"""Exercise real gRPC transport without model weights or connected hardware."""

import json
import logging
import pickle
import threading
import time
from concurrent import futures
from contextlib import contextmanager
from types import SimpleNamespace

import draccus
import grpc
import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature, RTCAttentionSchedule
from lerobot.inference_time_rtc import policy_server as server_module
from lerobot.inference_time_rtc.action_buffer import RTCActionBuffer
from lerobot.inference_time_rtc.compile_warmup import CompileWarmupConfig, warmup
from lerobot.inference_time_rtc.configs import (
    InferenceTimeRTCConfig,
    RTCClientConfig,
    RTCRecoveryConfig,
    RTCServerConfig,
)
from lerobot.inference_time_rtc.logging_utils import log_actions_to_file
from lerobot.inference_time_rtc.observation_codec import LosslessJPEGImage, LosslessPNGImage
from lerobot.inference_time_rtc.policy_server import RTCPolicyServer
from lerobot.inference_time_rtc.protocol import (
    PROTOCOL_VERSION,
    SESSION_HEADER,
    RTCActionChunk,
    RTCObservation,
)
from lerobot.inference_time_rtc.recording_client import RTCPolicyActionClient
from lerobot.inference_time_rtc.robot_client import RTCRobotClient
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.transport.utils import send_bytes_in_chunks
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


def chunk(request_id=0, start_step=0, length=50):
    raw = np.arange(length * 2, dtype=np.float32).reshape(length, 2)
    return RTCActionChunk(request_id, start_step, raw, raw + 100)


def test_bootstrap_and_normalized_prefix():
    buffer = RTCActionBuffer()
    for _ in range(100):
        assert buffer.advance() is None
    assert buffer.merge(chunk()) == 0
    np.testing.assert_array_equal(buffer.advance(), [100, 101])
    step, prefix, started = buffer.snapshot()
    assert step == 1 and started
    np.testing.assert_array_equal(prefix[0], [2, 3])
    prefix[:] = -1
    assert buffer.snapshot()[1][0, 0] == 2  # snapshots cannot mutate the queue


def test_merge_uses_elapsed_command_slots_and_never_averages():
    buffer = RTCActionBuffer()
    buffer.merge(chunk())
    buffer.advance()
    step, _, _ = buffer.snapshot()
    buffer.advance(skipped_steps=2)
    assert buffer.merge(chunk(1, step)) == 3
    np.testing.assert_array_equal(buffer.advance(), [106, 107])
    assert buffer.merge(chunk(0, 0)) is None  # late results never overwrite a newer chunk


def test_queue_underrun_still_advances_timeline():
    buffer = RTCActionBuffer()
    buffer.merge(chunk(length=2))
    buffer.advance()
    buffer.advance()
    assert buffer.advance() is None
    assert buffer.snapshot()[0] == 3
    assert buffer.snapshot()[1] is None
    with pytest.raises(RuntimeError, match="Entire RTC chunk expired"):
        buffer.merge(chunk(1, 0, length=2))


@pytest.mark.parametrize(
    "prefix_length,actual_delay,horizon",
    [(35, 29, 25), (21, 27, 21), (35, 25, 25)],
)
def test_late_splice_is_rejected_without_changing_pending_actions(prefix_length, actual_delay, horizon):
    buffer = RTCActionBuffer()
    buffer.merge(chunk())
    for _ in range(50 - prefix_length):
        buffer.advance()
    start_step, _, _ = buffer.snapshot()
    for _ in range(actual_delay):
        buffer.advance()
    before = buffer.snapshot()
    with pytest.raises(RuntimeError, match="RTC prefix continuity lost"):
        buffer.merge(chunk(1, start_step), prefix_horizon=horizon)
    assert buffer.last_request_id == 0
    assert buffer.snapshot()[0] == before[0]
    if before[1] is None:
        assert buffer.snapshot()[1] is None
    else:
        np.testing.assert_array_equal(buffer.snapshot()[1], before[1])
    assert buffer.last_splice.actual_delay == actual_delay
    assert buffer.last_splice.effective_horizon == horizon


def test_splice_diagnostics_compare_old_and_new_at_same_future_slot():
    buffer = RTCActionBuffer()
    buffer.merge(chunk())
    for _ in range(15):
        buffer.advance()
    start_step, _, _ = buffer.snapshot()
    for _ in range(24):
        buffer.advance()
    old_target = buffer.processed[39].copy()
    incoming = chunk(1, start_step)
    assert buffer.merge(incoming, prefix_horizon=25) == 24
    np.testing.assert_array_equal(
        buffer.last_splice.overlap_delta, incoming.processed_actions[24] - old_target
    )
    np.testing.assert_array_equal(buffer.advance(), incoming.processed_actions[24])


@pytest.mark.parametrize(
    "options",
    [
        {"enabled": False},
        {"execution_horizon": 0},
        {"queue_threshold": 19},
        {"inference_delay": -1},
        {"inference_delay": 20},
        {"latency_window": 0},
        {"observation_image_codec": "jpeg"},
        {"max_guidance_weight": float("nan")},
        {"max_guidance_weight": float("inf")},
    ],
)
def test_invalid_rtc_options(options):
    with pytest.raises(ValueError):
        InferenceTimeRTCConfig(**options)


def test_nested_linux_cli_and_chunk_constraints():
    cfg = draccus.parse(
        RTCClientConfig,
        args=[
            "--robot.type=bi_piper_follower",
            "--robot.left_arm_config.port=can3",
            "--robot.right_arm_config.port=can2",
            "--rtc.enabled=true",
            "--rtc.execution_horizon=20",
            "--rtc.prefix_attention_schedule=EXP",
            "--rtc.auto_delay=false",
            "--rtc.inference_delay=8",
            "--rtc.observation_image_codec=lossless_png",
            "--recovery.enabled=true",
            "--actions_per_chunk=50",
            "--policy_device=cuda:2",
        ],
    )
    assert cfg.robot.left_arm_config.port == "can3"
    assert cfg.policy_device == "cuda:2"
    assert cfg.rtc.prefix_attention_schedule == RTCAttentionSchedule.EXP
    assert not cfg.rtc.auto_delay and cfg.rtc.inference_delay == 8
    assert cfg.rtc.observation_image_codec == "lossless_png"
    assert cfg.recovery.enabled
    with pytest.raises(ValueError, match="queue_threshold"):
        cfg.rtc.validate_chunk_size(30)


class IdentityProcessor:
    def __call__(self, value):
        return value

    def reset(self):
        pass


class MutatingPostprocessor(IdentityProcessor):
    def __call__(self, value):
        return value.add_(100)


class OffsetNormalizer(IdentityProcessor):
    def __call__(self, batch):
        if ACTION in batch:
            batch[ACTION].sub_(100)
        return batch


class MockPolicy:
    def __init__(self):
        self.config = SimpleNamespace(
            chunk_size=50,
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
            image_features={},
            rtc_config=None,
        )
        self.calls = []
        self.delay = 0
        self.fail = False
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        return self

    def init_rtc_processor(self):
        self.rtc_processor = self.config.rtc_config

    def reset(self):
        pass

    def predict_action_chunk(self, observation, **kwargs):
        self.calls.append(kwargs)
        time.sleep(self.delay)
        if self.fail:
            raise RuntimeError("model failure")
        return torch.arange(100, dtype=torch.float32).reshape(1, 50, 2)


class MockRobot:
    observation_features = {"j1": float, "j2": float}
    action_features = {"j1": float, "j2": float}

    def __init__(self):
        self.connected = False
        self.actions = []
        self.capture_delay = 0

    def connect(self):
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        time.sleep(self.capture_delay)
        return {"j1": 0.0, "j2": 0.0}

    def send_action(self, action):
        assert self.connected
        self.actions.append((time.monotonic(), action))
        return action


@pytest.fixture
def remote():
    policy = MockPolicy()
    service = RTCPolicyServer(
        RTCServerConfig("unused", policy_device="cpu"),
        components=(policy, IdentityProcessor(), MutatingPostprocessor()),
    )
    with serve_mock_rtc(service) as running:
        yield running


@contextmanager
def serve_mock_rtc(service):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    stub = services_pb2_grpc.AsyncInferenceStub(channel)
    try:
        yield service, service.policy, stub, f"127.0.0.1:{port}"
    finally:
        service.shutdown_event.set()
        channel.close()
        server.stop(0).wait()


def client(remote, **kwargs):
    robot = MockRobot()
    cfg = RTCClientConfig(robot=PiperFollowerConfig(port="unused"), server_address=remote[3], **kwargs)
    return RTCRobotClient(cfg, robot=robot)


def test_grpc_config_prefix_and_processed_actions_round_trip(remote):
    service, policy, stub, _ = remote
    robot_client = client(remote)
    try:
        robot_client.start()
        assert service.policy.config.rtc_config is not robot_client.config.rtc
        assert policy.rtc_processor.enabled
        robot_client.request_chunk()
        np.testing.assert_array_equal(robot_client.buffer.advance(), [100, 101])
        assert not robot_client.latencies  # cold startup excluded
        robot_client.request_chunk()
        prefix = policy.calls[1]["prev_chunk_left_over"]
        assert prefix.device.type == "cpu"
        np.testing.assert_array_equal(prefix[0].numpy(), [2, 3])
        assert len(robot_client.latencies) == 1
        with pytest.raises(grpc.RpcError) as error:
            stub.GetActions(services_pb2.Empty(), metadata=robot_client.metadata, timeout=2)
        assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
    finally:
        robot_client.stop()
    assert not robot_client.robot.connected
    assert service.session_id is None


@pytest.mark.parametrize("rollout", [False, True], ids=["inference", "rollout"])
@pytest.mark.parametrize(
    ("codec", "packet_type"),
    [("lossless_jpeg", LosslessJPEGImage), ("lossless_png", LosslessPNGImage)],
    ids=["jpeg", "png"],
)
def test_lossless_observation_round_trip(tmp_path, monkeypatch, rollout, codec, packet_type):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    rgb = np.empty((63, 81, 3), dtype=np.uint8)
    rgb[:] = (16, 129, 255)
    rgb[10:25, 7:41] = (211, 10, 72)
    observation = {"j1": 0.0, "j2": 0.0, "camera": rgb.copy()}
    robot = MockRobot()
    robot.observation_features = {**robot.observation_features, "camera": rgb.shape}
    robot.get_observation = lambda: observation
    policy = MockPolicy()
    policy.config.image_features[f"{OBS_IMAGES}.camera"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, *rgb.shape[:2])
    )
    service = RTCPolicyServer(
        RTCServerConfig(str(tmp_path), policy_device="cpu", checkpoint_root=str(tmp_path)),
        components=(policy, IdentityProcessor(), MutatingPostprocessor()),
    )
    decode = server_module.decode_observation_images
    restored_pixels = []

    def inspect_upload(wire_images):
        image = wire_images["camera"]
        assert isinstance(image, packet_type)
        assert len(image.data) < rgb.nbytes
        restored = decode(wire_images)
        restored_pixels.append(restored["camera"])
        return restored

    monkeypatch.setattr(server_module, "decode_observation_images", inspect_upload)
    with serve_mock_rtc(service) as running:
        options = {
            "server_address": running[3],
            "pretrained_name_or_path": str(tmp_path),
            "policy_device": "cpu",
            "rtc": InferenceTimeRTCConfig(observation_image_codec=codec),
        }
        if rollout:
            cfg = RemotePolicyRecordConfig(
                enable=True, backend="rtc", policy_type="pi05", actions_per_chunk=50, **options
            )
            rtc_client = RTCPolicyActionClient(cfg, robot, fps=25)
        else:
            cfg = RTCClientConfig(robot=PiperFollowerConfig(port="unused"), **options)
            rtc_client = RTCRobotClient(cfg, robot=robot)
        try:
            rtc_client.start()
            if rollout:
                assert rtc_client.get_action(observation, "Move the card", 0) is None
                deadline = time.monotonic() + 3
                while not rtc_client.action_ready and rtc_client.error is None and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert rtc_client.error is None
                assert rtc_client.get_action(observation, "Move the card", 1) == {"j1": 100.0, "j2": 101.0}
            else:
                rtc_client.request_chunk()
                np.testing.assert_array_equal(rtc_client.buffer.advance(), [100, 101])
            assert len(restored_pixels) == 1
            np.testing.assert_array_equal(restored_pixels[0], rgb, strict=True)
            np.testing.assert_array_equal(observation["camera"], rgb, strict=True)
        finally:
            rtc_client.stop()


@pytest.mark.parametrize(("codec", "image_format"), [("lossless_jpeg", "JPEG"), ("lossless_png", "PNG")])
def test_lossless_codec_requires_server_support_before_robot_connection(
    remote, monkeypatch, codec, image_format
):
    monkeypatch.setattr(server_module, f"{codec}_available", lambda **kwargs: False)
    robot_client = client(remote, rtc=InferenceTimeRTCConfig(observation_image_codec=codec))
    try:
        with pytest.raises(RuntimeError, match=f"server must support lossless {image_format}"):
            robot_client.start()
        assert not robot_client.robot.connected
        assert remote[0].session_id is None
    finally:
        robot_client.stop()


def test_client_checkpoint_selects_committed_model_and_reuses_it_on_reconnect(tmp_path, monkeypatch):
    first_path = tmp_path / "checkpoint_a"
    second_path = tmp_path / "checkpoint_b"
    for path in (first_path, second_path):
        path.mkdir()
        (path / "config.json").write_text("{}", encoding="utf-8")
    loaded = []

    def load(config):
        loaded.append(config.pretrained_name_or_path)
        policy = MockPolicy()
        if config.pretrained_name_or_path == str(second_path):
            policy.config.chunk_size = 40
        return policy, IdentityProcessor(), MutatingPostprocessor()

    monkeypatch.setattr(server_module, "load_policy", load)
    service = RTCPolicyServer(RTCServerConfig(checkpoint_root=str(tmp_path), policy_device="cpu"))
    assert service.policy is None and loaded == []

    with serve_mock_rtc(service) as running:
        clients = [
            client(running, policy_type="pi05", pretrained_name_or_path=str(selected))
            for selected in (first_path, first_path, second_path)
        ]
        first, same, second = clients
        try:
            first.start()
            assert first.server_checkpoint == str(first_path)
            first.request_chunk()
            assert first.buffer.qsize() == 50 and loaded == [str(first_path)]
            with pytest.raises(grpc.RpcError) as error:
                second.start()
            assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
            assert loaded == [str(first_path)] and not second.robot.connected
            first.stop()

            same.start()
            assert same.server_checkpoint == str(first_path) and loaded == [str(first_path)]
            same.stop()

            second.start()
            assert second.server_checkpoint == str(second_path) and second.chunk_size == 40
            assert loaded == [str(first_path), str(second_path)]
            second.request_chunk()
            assert second.buffer.qsize() == 40
        finally:
            for robot_client in clients:
                robot_client.stop()


def test_client_checkpoint_rejects_path_outside_root_and_requires_selection(tmp_path, monkeypatch):
    root = tmp_path / "allowed"
    root.mkdir()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")

    def unexpected_load(config):
        pytest.fail(f"An invalid checkpoint was loaded: {config.pretrained_name_or_path}")

    monkeypatch.setattr(server_module, "load_policy", unexpected_load)
    service = RTCPolicyServer(RTCServerConfig(checkpoint_root=str(root), policy_device="cpu"))
    with serve_mock_rtc(service) as running:
        no_checkpoint = client(running)
        outside = client(running, pretrained_name_or_path=str(tmp_path))
        try:
            with pytest.raises(ValueError, match="pretrained_name_or_path"):
                no_checkpoint.start()
            with pytest.raises(grpc.RpcError) as error:
                outside.start()
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "under server checkpoint_root" in error.value.details()
            assert not outside.connection_attempted and not no_checkpoint.connection_attempted
            assert service.session_id is None and service.policy is None
        finally:
            no_checkpoint.stop()
            outside.stop()


def test_client_checkpoint_needs_capability_before_robot_connection(remote):
    robot_client = client(remote, pretrained_name_or_path="/model")
    try:
        with pytest.raises(RuntimeError, match="checkpoint_root"):
            robot_client.start()
        assert not robot_client.connection_attempted
    finally:
        robot_client.stop()


def test_duplicate_requests_session_ownership_and_reconnect(remote):
    first = client(remote)
    second = client(remote)
    try:
        first.start()
        with pytest.raises(grpc.RpcError) as error:
            second.start()
        assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
        first.request_chunk()
        request = RTCObservation(0, 0, {"j1": 0, "j2": 0}, None, 10)
        with pytest.raises(grpc.RpcError) as error:
            remote[2].SendObservations(
                send_bytes_in_chunks(pickle.dumps(request), services_pb2.Observation),
                metadata=first.metadata,
                timeout=2,
            )
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        first.stop()
        second.start()
        second.request_chunk()
        assert second.buffer.qsize() == 50
    finally:
        first.stop()
        second.stop()


def test_old_setup_is_rejected_without_changing_existing_config(remote):
    with pytest.raises(grpc.RpcError) as error:
        remote[2].SendPolicyInstructions(
            services_pb2.PolicySetup(data=pickle.dumps({"old": True})),
            metadata=((SESSION_HEADER, "test"),),
            timeout=2,
        )
    assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
    assert remote[0].instructions is None


def test_control_continues_during_camera_capture(remote):
    robot_client = client(remote, fps=100)
    robot_client.start()
    robot_client.request_chunk()
    robot_client.robot.capture_delay = 0.12
    worker = threading.Thread(target=robot_client.request_chunk)
    control = threading.Thread(target=robot_client.control_loop)
    try:
        worker.start()
        control.start()
        worker.join(timeout=3)
        assert not worker.is_alive()
        # Many commands were issued while the observation thread was blocked.
        assert len(robot_client.robot.actions) >= 6
        assert robot_client.buffer.last_request_id == 1
        assert robot_client.buffer.snapshot()[0] > 0
    finally:
        robot_client.stop_event.set()
        control.join(timeout=2)
        robot_client.stop()


def test_worker_model_error_stops_client_and_releases_session(remote):
    robot_client = client(remote, duration_s=2)
    remote[1].fail = True
    with pytest.raises(RuntimeError, match="RTC inference stopped"):
        robot_client.run()
    assert not robot_client.robot.connected
    assert remote[0].session_id is None
    assert not robot_client.worker.is_alive()


def test_client_rejects_logged_late_splice_and_stops_worker(remote, caplog):
    robot_client = client(
        remote,
        rtc=InferenceTimeRTCConfig(
            inference_delay=22, execution_horizon=25, queue_threshold=35, auto_delay=False
        ),
    )
    try:
        robot_client.start()
        robot_client.request_chunk()
        for _ in range(15):
            robot_client.buffer.advance()

        def capture_while_29_slots_elapse():
            for _ in range(29):
                robot_client.buffer.advance()
            return {"j1": 0.0, "j2": 0.0}

        robot_client.robot.get_observation = capture_while_29_slots_elapse
        with caplog.at_level("INFO"):
            robot_client._inference_loop()
        assert robot_client.stop_event.is_set()
        assert "actual_delay=29 >= effective_horizon=25" in str(robot_client.error)
        assert robot_client.buffer.last_request_id == 0
        assert "max_abs_overlap_delta=" in caplog.text
        assert "joint=j1" in caplog.text
        assert not robot_client.robot.actions
    finally:
        robot_client.stop()


def test_control_queue_exhaustion_pauses_motion_and_continues_inference(remote, monkeypatch, caplog):
    robot_client = client(remote, fps=50)
    policy = remote[1]
    in_flight = threading.Event()
    release_reply = threading.Event()
    paused = threading.Event()
    continued = threading.Event()
    completed = []
    predict = policy.predict_action_chunk
    reserve = robot_client.buffer.reserve
    request_chunk = robot_client.request_chunk

    def block_second_prediction(observation, **kwargs):
        if len(policy.calls) == 1:
            in_flight.set()
            assert release_reply.wait(timeout=5)
        return predict(observation, **kwargs)

    def observe_pause(*args, **kwargs):
        reservation = reserve(*args, **kwargs)
        if robot_client.buffer.motion_paused:
            paused.set()
        return reservation

    def record_completion():
        request_chunk()
        completed.append(robot_client.request_id - 1)
        if len(completed) >= 3:
            continued.set()

    monkeypatch.setattr(policy, "predict_action_chunk", block_second_prediction)
    monkeypatch.setattr(robot_client.buffer, "reserve", observe_pause)
    with caplog.at_level(logging.INFO), futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            robot_client.start()
            robot_client.request_chunk()
            # Leave one slot for control to consume before the stalled refill returns.
            robot_client.buffer.advance(skipped_steps=48)
            assert robot_client.buffer.qsize() == 1
            monkeypatch.setattr(robot_client, "request_chunk", record_completion)
            robot_client.worker = threading.Thread(target=robot_client._inference_loop, daemon=True)
            robot_client.worker.start()
            assert in_flight.wait(timeout=3)
            control = executor.submit(robot_client.control_loop)
            assert paused.wait(timeout=3)
            sent_before_pause = list(robot_client.robot.actions)

            release_reply.set()
            assert continued.wait(timeout=3)
            assert completed[:3] == [1, 2, 3]
            assert robot_client.buffer.motion_paused
            assert robot_client.buffer.last_request_id == 0
            assert not robot_client.stop_event.is_set()
            assert not control.done()
            assert robot_client.robot.actions == sent_before_pause
            assert len(policy.calls[1]["prev_chunk_left_over"]) == 1
            assert policy.calls[2]["prev_chunk_left_over"] is None
            assert policy.calls[3]["prev_chunk_left_over"] is None
            for request_id in (1, 2, 3):
                assert f"RTC inference-only result request={request_id}" in caplog.text
        finally:
            release_reply.set()
            robot_client.stop()
        control.result(timeout=3)
    assert not robot_client.robot.connected
    assert remote[0].session_id is None


def recovery_client(remote):
    robot = MockRobot()
    robot.action_features = {"joint.pos": float, "gripper.pos": float}
    rgb = np.full((8, 8, 3), 50, dtype=np.uint8)
    robot.observation_features = {**robot.action_features, "camera": rgb.shape}
    robot.positions = {"joint.pos": 102.0, "gripper.pos": 103.0}
    robot.get_observation = lambda: {**robot.positions, "camera": rgb.copy()}
    remote[1].config.image_features[f"{OBS_IMAGES}.camera"] = PolicyFeature(
        type=FeatureType.VISUAL, shape=(3, *rgb.shape[:2])
    )
    remote[0].preprocessor = OffsetNormalizer()
    config = RTCClientConfig(
        robot=PiperFollowerConfig(port="unused"),
        server_address=remote[3],
        fps=50,
        recovery=RTCRecoveryConfig(
            enabled=True, prefix_steps=3, max_observation_age_s=0.8, max_joint_delta=1.0, max_gripper_delta=1.0
        ),
        rtc=InferenceTimeRTCConfig(execution_horizon=4, queue_threshold=12, inference_delay=1, auto_delay=False),
    )
    return RTCRobotClient(config, robot=robot), rgb


def test_recovery_after_exhaustion_discards_the_inflight_result_and_uses_measured_pose(remote, monkeypatch):
    rtc_client, rgb = recovery_client(remote)
    policy = remote[1]
    in_flight = threading.Event()
    release = threading.Event()
    resumed = threading.Event()
    captured = []
    constant = torch.tensor([2.0, 3.0]).repeat(50, 1).unsqueeze(0)
    hold = torch.tensor([11.0, 13.0]).repeat(50, 1).unsqueeze(0)

    def predict(observation, **kwargs):
        assert ACTION not in observation
        captured.append((observation[f"{OBS_IMAGES}.camera"].clone(), kwargs.get("prev_chunk_left_over")))
        if len(captured) == 2:
            in_flight.set()
            assert release.wait(timeout=3)
            return constant + 50
        return hold + 0.25 if len(captured) > 2 else constant

    send = rtc_client.robot.send_action

    def send_action(action):
        reported = send(action)
        if action["joint.pos"] == 111.25:
            resumed.set()
            rtc_client.stop_event.set()
        return reported

    monkeypatch.setattr(policy, "predict_action_chunk", predict)
    monkeypatch.setattr(rtc_client.robot, "send_action", send_action)
    with futures.ThreadPoolExecutor(max_workers=1) as executor:
        try:
            rtc_client.start()
            rtc_client.request_chunk()
            rtc_client.buffer.advance(skipped_steps=48)
            rtc_client.worker = threading.Thread(target=rtc_client._inference_loop, daemon=True)
            rtc_client.worker.start()
            assert in_flight.wait(timeout=3)
            control = executor.submit(rtc_client.control_loop)
            deadline = time.monotonic() + 3
            while not rtc_client.buffer.motion_paused and time.monotonic() < deadline:
                time.sleep(0.005)
            assert rtc_client.buffer.motion_paused
            rgb[:] = 120
            rtc_client.robot.positions.update({"joint.pos": 111.0, "gripper.pos": 113.0})
            release.set()
            assert resumed.wait(timeout=3)
            control.result(timeout=2)
            assert rtc_client.buffer.last_request_id == 2
            assert [action["joint.pos"] for _, action in rtc_client.robot.actions] == [102.0, 111.25]
            assert len(captured) == 3
            np.testing.assert_array_equal(captured[2][1].numpy(), np.tile([11.0, 13.0], (3, 1)))
            torch.testing.assert_close(captured[2][0], torch.full_like(captured[2][0], 120 / 255))
            assert rtc_client.error is None
        finally:
            release.set()
            rtc_client.stop()


@pytest.mark.parametrize(
    "outcome",
    ["resume", "gripper_only", "jump", "bridge_jump", "expired", "drift", "invalid_pose", "invalid_transform"],
)
def test_late_splice_drains_before_recovery_restarts_with_a_complete_chunk(remote, monkeypatch, outcome):
    rtc_client, _ = recovery_client(remote)
    policy = remote[1]
    prefixes = []
    constant = torch.tensor([2.0, 3.0]).repeat(50, 1).unsqueeze(0)
    hold = torch.tensor([10.0, 20.0]).repeat(50, 1).unsqueeze(0)
    old_chunk = constant + torch.arange(50).view(1, 50, 1) * 0.01

    def predict(observation, **kwargs):
        prefixes.append(kwargs.get("prev_chunk_left_over"))
        if len(prefixes) == 1:
            return old_chunk
        if len(prefixes) == 2:
            return constant + 50
        if outcome == "drift":
            rtc_client.robot.positions["joint.pos"] += 2.0
        recovered = hold + 0.25
        if outcome == "gripper_only":
            rtc_client.robot.positions["gripper.pos"] += 2.0
            recovered[0, :3, 1] += 8.0
        if outcome == "jump":
            recovered[0, 0, 0] += 2.0
        if outcome == "bridge_jump":
            recovered[0, 3, 0] += 2.0
        return recovered

    monkeypatch.setattr(policy, "predict_action_chunk", predict)
    try:
        rtc_client.start()
        rtc_client.request_chunk()
        if outcome == "gripper_only":
            send = rtc_client.robot.send_action

            def send_without_gripper_report(action):
                sent = send(action)
                return {"joint.pos": sent["joint.pos"]}

            monkeypatch.setattr(rtc_client.robot, "send_action", send_without_gripper_report)
        rtc_client._dispatch(rtc_client.buffer.reserve(), 0)
        capture = rtc_client.robot.get_observation

        def capture_late():
            rtc_client.buffer.advance(skipped_steps=4)
            return capture()

        monkeypatch.setattr(rtc_client.robot, "get_observation", capture_late)
        rtc_client.request_chunk()
        assert rtc_client.buffer.draining
        assert not rtc_client.buffer.motion_paused
        assert rtc_client.buffer.qsize() == 44
        rtc_client.request_chunk()
        rtc_client._request_recovery()
        assert len(prefixes) == 2
        old = rtc_client.buffer.reserve()
        assert (old.request_id, old.step) == (0, 6)
        assert rtc_client.buffer.reserve(
            skipped_steps=rtc_client.buffer.qsize(), pause_on_exhaustion=True
        ) is None
        assert rtc_client.buffer.motion_paused
        assert rtc_client.buffer.last_request_id == 0
        monkeypatch.setattr(rtc_client.robot, "get_observation", capture)
        rtc_client.robot.positions.update({"joint.pos": 110.0, "gripper.pos": 120.0})
        if outcome == "invalid_pose":
            rtc_client.robot.positions["gripper.pos"] = float("nan")
        if outcome == "invalid_transform":
            remote[0].preprocessor = IdentityProcessor()
            with pytest.raises(grpc.RpcError, match="could not encode the measured positions"):
                rtc_client._request_recovery()
            assert len(prefixes) == 2
            assert not rtc_client._commit_recovery()
            assert rtc_client.buffer.motion_paused
            assert len(rtc_client.robot.actions) == 1
            return
        rtc_client._request_recovery()
        if outcome == "invalid_pose":
            assert len(prefixes) == 2
        else:
            torch.testing.assert_close(prefixes[2], hold[0, :3])
        if outcome == "expired":
            monkeypatch.setattr(rtc_client, "_recovery_expired", lambda _: True)
        should_resume = outcome in {"resume", "gripper_only", "jump", "bridge_jump", "drift"}
        assert rtc_client._commit_recovery() == should_resume
        assert rtc_client.buffer.motion_paused != should_resume
        assert rtc_client.buffer.draining != should_resume
        assert rtc_client.buffer.last_request_id == (2 if should_resume else 0)
        if should_resume:
            assert rtc_client.buffer.qsize() == 50
            first = rtc_client.buffer.reserve()
            assert (first.step, first.chunk_index) == (51, 0)
            np.testing.assert_array_equal(
                first.action, [112.25 if outcome == "jump" else 110.25, 128.25 if outcome == "gripper_only" else 120.25]
            )
            if outcome == "gripper_only":
                after_prefix = rtc_client.buffer.reserve(skipped_steps=2)
                np.testing.assert_array_equal(after_prefix.action, [110.25, 120.25])
            if outcome == "bridge_jump":
                after_prefix = rtc_client.buffer.reserve(skipped_steps=2)
                np.testing.assert_array_equal(after_prefix.action, [112.25, 120.25])
        assert len(rtc_client.robot.actions) == 1
    finally:
        rtc_client.stop()


def test_allow_late_splice_accepts_remaining_actions_and_keeps_timing(remote, caplog):
    robot_client = client(
        remote,
        allow_late_splice=True,
        rtc=InferenceTimeRTCConfig(
            inference_delay=15, execution_horizon=28, queue_threshold=35, auto_delay=False
        ),
    )
    try:
        robot_client.start()
        robot_client.request_chunk()
        robot_client.buffer.advance(skipped_steps=14)

        def capture_while_30_slots_elapse():
            robot_client.buffer.advance(skipped_steps=29)
            return {"j1": 0.0, "j2": 0.0}

        robot_client.robot.get_observation = capture_while_30_slots_elapse
        with caplog.at_level(logging.INFO):
            robot_client.request_chunk()
        assert not robot_client.stop_event.is_set()
        assert robot_client.buffer.qsize() == 20
        assert robot_client.buffer.last_splice.actual_delay == 30
        assert robot_client.buffer.last_splice.effective_horizon == 28
        reservation = robot_client.buffer.reserve()
        assert reservation.request_id == 1 and reservation.step == 45 and reservation.chunk_index == 30
        assert "RTC late splice accepted" in caplog.text
        timings = [
            json.loads(record.getMessage().removeprefix("RTC rpc_timing "))
            for record in caplog.records
            if record.getMessage().startswith("RTC rpc_timing ")
        ]
        (client_timing,) = [timing for timing in timings if timing["request_id"] == 1]
        assert client_timing["observation_bytes"] > 0
        for key in (
            "observation_ms",
            "serialize_ms",
            "send_observations_rpc_ms",
            "rpc_handoff_ms",
            "get_actions_rpc_ms",
            "deserialize_response_ms",
            "total_ms",
        ):
            assert client_timing[key] >= 0
        session = dict(robot_client.metadata)[SESSION_HEADER]
        assert client_timing["session_id"] == session
        for stage, keys in (
            ("observation", ("receive_ms", "decode_ms", "observation_bytes")),
            ("actions", ("pending_wait_ms", "pipeline_ms", "get_actions_ms")),
        ):
            prefix = f"RTC {stage} timing session={session} request=1 "
            (message,) = [
                record.getMessage() for record in caplog.records if record.getMessage().startswith(prefix)
            ]
            fields = dict(field.split("=", 1) for field in message.removeprefix(prefix).split())
            assert all(float(fields[key]) >= 0 for key in keys)
            if stage == "observation":
                assert int(fields["observation_bytes"]) == client_timing["observation_bytes"]
    finally:
        robot_client.stop()


def test_allow_late_splice_cannot_accept_an_entirely_expired_chunk():
    buffer = RTCActionBuffer()
    buffer.merge(chunk())
    buffer.advance(skipped_steps=49)
    with pytest.raises(RuntimeError, match="Entire RTC chunk expired"):
        buffer.merge(chunk(request_id=1), prefix_horizon=28, allow_late_splice=True)
    assert buffer.last_request_id == 0


def test_bootstrap_wait_is_not_treated_as_queue_exhaustion(remote):
    robot_client = client(remote, fps=100, duration_s=0.03)
    try:
        robot_client.start()
        robot_client.control_loop()
        assert not robot_client.robot.actions
        assert not robot_client.stop_event.is_set()
    finally:
        robot_client.stop()


def test_first_chunk_arriving_after_empty_reservation_is_not_an_underrun(remote, monkeypatch):
    robot_client = client(remote, fps=100, duration_s=0.1)
    reserve = robot_client.buffer.reserve

    def reserve_then_receive(*args, **kwargs):
        action = reserve(*args, **kwargs)
        if not robot_client.buffer.started:
            robot_client.buffer.merge(chunk())
        return action

    monkeypatch.setattr(robot_client.buffer, "reserve", reserve_then_receive)
    try:
        robot_client.start()
        robot_client.control_loop()
        assert robot_client.robot.actions
        assert not robot_client.stop_event.is_set()
    finally:
        robot_client.stop()


def test_rtc_action_log_preserves_full_chunk_and_driver_reported_target(
    remote, tmp_path, monkeypatch, caplog
):
    robot_client = client(remote)
    log_path = tmp_path / "rtc_client.log"
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    caplog.set_level(logging.INFO, logger="lerobot.inference_time_rtc.actions")

    def send_one(action):
        robot_client.stop_event.set()
        return {**action, "j2": action["j2"] + 0.125}

    monkeypatch.setattr(robot_client.robot, "send_action", send_one)
    try:
        robot_client.start()
        with log_actions_to_file(log_path):
            robot_client.request_chunk()
            robot_client.control_loop()
    finally:
        robot_client.stop()
        root_logger.removeHandler(file_handler)
        file_handler.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    (received,) = [
        json.loads(line.removeprefix("RTC action_chunk "))
        for line in lines
        if line.startswith("RTC action_chunk ")
    ]
    (sent,) = [
        json.loads(line.removeprefix("RTC sent_action "))
        for line in lines
        if line.startswith("RTC sent_action ")
    ]
    expected = chunk()
    assert received["request_id"] == sent["request_id"] == 0
    assert received["action_names"] == ["j1", "j2"]
    np.testing.assert_array_equal(received["original_actions"], expected.original_actions)
    np.testing.assert_array_equal(received["processed_actions"], expected.processed_actions)
    assert sent["step"] == sent["chunk_index"] == sent["skipped_steps"] == 0
    assert sent["requested_action"] == {"j1": 100.0, "j2": 101.0}
    assert sent["sent_action"] == {"j1": 100.0, "j2": 101.125}
    assert received["received_monotonic_ns"] <= sent["sent_monotonic_ns"]


def test_auto_delay_includes_recent_latency_and_can_be_fixed(remote):
    robot_client = client(remote)
    robot_client.chunk_size = 50
    robot_client.latencies.extend([0.2, 0.4])
    assert robot_client._delay_steps() == 12
    robot_client.config.rtc.auto_delay = False
    assert robot_client._delay_steps() == 10
    robot_client.channel.close()


def test_nonfinite_and_mismatched_actions_are_rejected():
    result = chunk()
    result.original_actions[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        result.validate()
    with pytest.raises(ValueError, match="matching"):
        RTCActionChunk(0, 0, np.zeros((2, 2)), np.zeros((2, 3))).validate()


def test_ordinary_server_protocol_rejected_before_robot_connect(remote):
    class OrdinaryService(services_pb2_grpc.AsyncInferenceServicer):
        def Ready(self, request, context):  # noqa: N802
            return services_pb2.Empty()

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(OrdinaryService(), server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    robot_client = client(remote)
    robot_client.channel.close()
    robot_client.channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    robot_client.stub = services_pb2_grpc.AsyncInferenceStub(robot_client.channel)
    try:
        with pytest.raises(RuntimeError, match="requires lerobot.inference_time_rtc"):
            robot_client.run()
        assert not robot_client.robot.connected
    finally:
        server.stop(0).wait()


def test_protocol_capability_is_explicit(remote):
    _, call = remote[2].Ready.with_call(services_pb2.Empty(), timeout=2)
    assert dict(call.initial_metadata())["rtc-protocol"] == PROTOCOL_VERSION


def test_real_rtc_guidance_runs_inside_remote_no_grad_path(remote):
    class FlowPolicy(MockPolicy):
        def init_rtc_processor(self):
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

        def predict_action_chunk(self, observation, **kwargs):
            value = torch.ones(1, 50, 2)
            velocity = self.rtc_processor.denoise_step(
                x_t=value,
                prev_chunk_left_over=kwargs["prev_chunk_left_over"],
                inference_delay=kwargs["inference_delay"],
                execution_horizon=kwargs["execution_horizon"],
                time=0.5,
                original_denoise_step_partial=lambda x: x * 0.5,
            )
            return value - 0.1 * velocity

    remote[0].policy = FlowPolicy()
    robot_client = client(remote)
    try:
        robot_client.start()
        robot_client.request_chunk()
        first = robot_client.buffer.snapshot()[1]
        robot_client.buffer.advance()
        robot_client.request_chunk()
        guided = robot_client.buffer.snapshot()[1]
        assert not np.allclose(first[0], guided[0])
        assert np.isfinite(guided).all()
    finally:
        robot_client.stop()


def test_expired_abandoned_pending_request_does_not_block_new_session(remote):
    first = client(remote)
    second = client(remote)
    try:
        first.start()
        request = RTCObservation(0, 0, {"j1": 0, "j2": 0}, None, 10)
        remote[2].SendObservations(
            send_bytes_in_chunks(pickle.dumps(request), services_pb2.Observation),
            metadata=first.metadata,
            timeout=2,
        )
        remote[0].last_activity -= remote[0].config.session_timeout_s + 1
        second.start()
        second.request_chunk()
        assert second.buffer.qsize() == 50
    finally:
        first.stop()
        second.stop()


def test_partial_robot_connection_is_cleaned_up(remote):
    robot_client = client(remote)

    def fail_after_connect():
        robot_client.robot.connected = True
        raise RuntimeError("camera unavailable")

    robot_client.robot.connect = fail_after_connect
    with pytest.raises(RuntimeError, match="camera unavailable"):
        robot_client.run()
    assert not robot_client.robot.connected


def test_successful_run_stops_worker_and_releases_robot(remote):
    robot_client = client(remote, fps=100, duration_s=0.35)
    robot_client.run()
    assert len(robot_client.robot.actions) > 5
    assert not robot_client.robot.connected
    assert not robot_client.worker.is_alive()
    assert remote[0].session_id is None


def test_offline_warmup_covers_first_and_guided_chunks(remote, monkeypatch):
    service, policy, _, _ = remote
    camera = f"{OBS_IMAGES}.fixed"
    policy.config.input_features = {OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))}
    policy.config.image_features = {camera: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 4, 6))}
    predict = policy.predict_action_chunk

    def check_observation(observation, **kwargs):
        assert observation[OBS_STATE].shape == (1, 2)
        assert observation[camera].shape == (1, 3, 4, 6)
        assert observation["task"] == "banknotes"
        return predict(observation, **kwargs)

    monkeypatch.setattr(policy, "predict_action_chunk", check_observation)
    warmup(service, CompileWarmupConfig("unused", task="banknotes", prefix_steps=[35, 25], repeats=2))
    prefixes = [call["prev_chunk_left_over"] for call in policy.calls]
    assert prefixes[:2] == [None, None]
    assert [tuple(prefix.shape) for prefix in prefixes[2:]] == [(35, 2), (35, 2), (25, 2), (25, 2)]
    np.testing.assert_array_equal(prefixes[2][0], [30, 31])
    assert service.session_id is None and service.pending is None


@pytest.mark.parametrize(("preload_device", "compile_model"), [(None, False), ("cpu", True)])
def test_model_preload_overrides_runtime_settings_and_tokenizer(monkeypatch, preload_device, compile_model):
    saved_config = SimpleNamespace(
        type="pi05",
        device="cuda:0",
        dtype="float32",
        compile_model=True,
        gradient_checkpointing=True,
        rtc_config=InferenceTimeRTCConfig(),
    )
    model = torch.nn.Linear(2, 2)
    loaded = {}

    def from_pretrained(path, config):
        loaded["path"] = path
        loaded["config"] = config
        model.config = config
        return model

    def processors(config, **kwargs):
        loaded["processors"] = kwargs
        return IdentityProcessor(), MutatingPostprocessor()

    def read_config(path, **kwargs):
        loaded["config_overrides"] = kwargs
        return saved_config

    monkeypatch.setattr(server_module.PreTrainedConfig, "from_pretrained", read_config)
    monkeypatch.setattr(
        server_module, "get_policy_class", lambda name: SimpleNamespace(from_pretrained=from_pretrained)
    )
    monkeypatch.setattr(server_module, "make_pre_post_processors", processors)
    policy, _, _ = server_module.load_policy(
        RTCServerConfig(
            "checkpoint",
            policy_device=preload_device,
            tokenizer_name_or_path="/linux/tokenizer",
            policy_dtype="bfloat16" if compile_model else None,
            compile_model=compile_model,
            compile_mode="default",
        )
    )
    assert loaded["path"] == "checkpoint"
    assert policy.weight.device.type == "cpu" and not policy.weight.requires_grad
    assert loaded["config"].compile_model is compile_model and not loaded["config"].gradient_checkpointing
    assert loaded["config"].dtype == ("bfloat16" if compile_model else "float32")
    if compile_model:
        assert loaded["config"].compile_mode == "default"
    assert loaded["config"].rtc_config is None
    assert loaded["config_overrides"] == {"cli_overrides": ["--device=cpu"]}
    assert loaded["processors"]["preprocessor_overrides"]["tokenizer_processor"] == {
        "tokenizer_name": "/linux/tokenizer"
    }


@pytest.mark.parametrize("device", ["cuda:-1", "cuda:1x", "gpu:0", "", "mps"])
def test_invalid_client_device_syntax(device):
    with pytest.raises(ValueError, match="policy_device"):
        RTCClientConfig(robot=PiperFollowerConfig(port="unused"), policy_device=device)


def test_server_device_validation_has_no_silent_cpu_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    assert RTCPolicyServer._resolve_device("cuda:2") == "cuda:2"
    assert RTCPolicyServer._resolve_device("cuda") == "cuda:0"
    with pytest.raises(ValueError, match="unavailable"):
        RTCPolicyServer._resolve_device("cuda:3")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is unavailable"):
        RTCPolicyServer._resolve_device("cuda:0")
    assert RTCPolicyServer._resolve_device("cpu") == "cpu"


def mock_gpu_moves(remote, monkeypatch):
    calls = []
    monkeypatch.setattr(remote[0], "_resolve_device", lambda device: device)
    monkeypatch.setattr(remote[0], "_clear_cuda_cache", lambda device: None)

    def processors(config, policy_config, device):
        calls.append(device)
        return IdentityProcessor(), MutatingPostprocessor()

    monkeypatch.setattr(server_module, "make_processors", processors)
    return calls


def test_client_selects_gpu_and_reconnect_can_choose_another(remote, monkeypatch):
    processors = mock_gpu_moves(remote, monkeypatch)
    first = client(remote, policy_device="cuda:2")
    second = client(remote, policy_device="cuda:1")
    try:
        first.start()
        assert first.server_policy_device == remote[0].device == "cuda:2"
        assert remote[0].policy.config.device == "cuda:2"
        assert first.robot.connected
        with pytest.raises(grpc.RpcError) as error:
            second.start()
        assert error.value.code() == grpc.StatusCode.ALREADY_EXISTS
        assert not second.robot.connected
        assert remote[0].device == "cuda:2"
        first.stop()
        second.start()
        assert second.server_policy_device == "cuda:1"
        assert remote[1].moves == ["cuda:2", "cpu", "cuda:1"]
        assert processors == ["cuda:2", "cuda:1"]
    finally:
        first.stop()
        second.stop()


def test_invalid_server_gpu_is_rejected_before_robot_connect(remote, monkeypatch):
    def invalid(device):
        raise ValueError("Server GPU is unavailable")

    monkeypatch.setattr(remote[0], "_resolve_device", invalid)
    robot_client = client(remote, policy_device="cuda:999")
    try:
        with pytest.raises(grpc.RpcError) as error:
            robot_client.start()
        assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
        assert not robot_client.connection_attempted
        assert not robot_client.robot.connected
        assert remote[0].session_id is None
    finally:
        robot_client.stop()


def test_gpu_oom_restores_cpu_and_allows_new_session(remote, monkeypatch):
    mock_gpu_moves(remote, monkeypatch)
    original_to = remote[1].to

    def move(device):
        if device == "cuda:2":
            raise torch.cuda.OutOfMemoryError("simulated OOM")
        return original_to(device)

    monkeypatch.setattr(remote[1], "to", move)
    failed = client(remote, policy_device="cuda:2")
    recovered = client(remote, policy_device="cpu")
    try:
        with pytest.raises(grpc.RpcError) as error:
            failed.start()
        assert error.value.code() == grpc.StatusCode.RESOURCE_EXHAUSTED
        assert not failed.connection_attempted
        assert remote[0].device == "cpu" and remote[0].session_id is None
        recovered.start()
        recovered.request_chunk()
        assert recovered.buffer.qsize() == 50
    finally:
        failed.stop()
        recovered.stop()


def test_default_device_is_decided_at_handshake(remote, monkeypatch):
    assert RTCServerConfig("unused").policy_device is None
    remote[0].config.policy_device = None
    requested = []

    def resolve(device):
        requested.append(device)
        return "cpu"

    monkeypatch.setattr(remote[0], "_resolve_device", resolve)
    robot_client = client(remote)
    try:
        robot_client.start()
        assert requested == ["cuda:0"]
    finally:
        robot_client.stop()
