"""Run RLT handshakes and action transport over loopback gRPC without hardware."""

import time
from concurrent import futures
from contextlib import contextmanager
from types import SimpleNamespace

import grpc
import numpy as np
import pytest
import torch

from lerobot.async_inference.helpers import map_robot_keys_to_lerobot_features
from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference_time_rtc import policy_server as server_module
from lerobot.inference_time_rtc.action_filter import filter_joint_targets
from lerobot.inference_time_rtc.configs import (
    InferenceTimeRTCConfig,
    RTCClientConfig,
    RTCRLTConfig,
    RTCServerConfig,
)
from lerobot.inference_time_rtc.policy_server import RTCPolicyServer
from lerobot.inference_time_rtc.protocol import (
    RLT_HEADER,
    RLT_OUTPUT_SG_HEADER,
    RLT_PHASE_HEADER,
    RTCObservation,
    SessionInstructions,
    rlt_row_mask,
)
from lerobot.inference_time_rtc.recording_client import RTCPolicyActionClient
from lerobot.inference_time_rtc.rlt.networks import RLTHeadBundle, RLTHeadConfig
from lerobot.inference_time_rtc.rlt.runtime import build_contract
from lerobot.inference_time_rtc.robot_client import RTCRobotClient
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.rtc.modeling_rtc import RTCProcessor
from lerobot.processor import NormalizerProcessorStep
from lerobot.robots.piper_follower.config_piper_follower import PiperFollowerConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.transport import services_pb2, services_pb2_grpc
from lerobot.utils.constants import ACTION, OBS_STATE


class NoHardwareRobot:
    observation_features = {"joint.pos": float, "gripper.pos": float}
    action_features = observation_features

    def __init__(self):
        self.connected = False
        self.connect_calls = 0

    def connect(self):
        self.connect_calls += 1
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_observation(self):
        return {"joint.pos": 0.1, "gripper.pos": 0.2}


class TinyVLA:
    def __init__(self):
        self.config = PI05Config(
            device="cpu",
            max_action_dim=2,
            max_state_dim=2,
            input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))},
            output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        )
        self.model = SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(
                paligemma=SimpleNamespace(config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8)))
            )
        )
        self.calls = []
        self.plain_calls = []
        self.rtc_processor = None
        step = torch.arange(50, dtype=torch.float32)
        self.actions = torch.stack((0.2 * torch.sin(step * 2) + step / 100, step / 200), dim=-1)[None]

    def init_rtc_processor(self):
        self.rtc_processor = RTCProcessor(self.config.rtc_config)

    def reset(self):
        pass

    def predict_action_chunk(self, observation, **kwargs):
        self.plain_calls.append(kwargs)
        assert observation[OBS_STATE].shape == (1, 2)
        return self.actions.clone()

    def predict_action_chunk_with_features(self, observation, **kwargs):
        self.calls.append(kwargs)
        assert observation[OBS_STATE].shape == (1, 2)
        return (
            self.actions.clone(),
            torch.arange(48, dtype=torch.float32).reshape(1, 6, 8),
            torch.ones(1, 6, dtype=torch.bool),
        )


class TinyPreprocessor:
    def __init__(self):
        self.steps = [
            NormalizerProcessorStep(
                features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
                norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
                stats={ACTION: {"mean": torch.zeros(2), "std": torch.full((2,), 2.0)}},
            )
        ]

    def __call__(self, observation):
        result = dict(observation)
        result[OBS_STATE] = torch.as_tensor(result[OBS_STATE], dtype=torch.float32).reshape(1, 2)
        return result

    def reset(self):
        pass


class DoublePostprocessor:
    def __call__(self, value):
        return value * 2

    def reset(self):
        pass


@contextmanager
def serve(service):
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(service, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    channel = grpc.insecure_channel(f"127.0.0.1:{port}")
    try:
        yield f"127.0.0.1:{port}", services_pb2_grpc.AsyncInferenceStub(channel)
    finally:
        service.shutdown_event.set()
        channel.close()
        server.stop(0).wait()


@pytest.fixture
def setup_rlt(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"type":"pi05"}', encoding="utf-8")
    (base / "policy_preprocessor.json").write_text('{"normalization":"test_std_2"}', encoding="utf-8")
    policy = TinyVLA()
    service = RTCPolicyServer(
        RTCServerConfig(str(base.resolve()), policy_device="cpu", checkpoint_root=str(tmp_path.resolve())),
        components=(policy, TinyPreprocessor(), DoublePostprocessor()),
    )
    robot = NoHardwareRobot()
    rtc = InferenceTimeRTCConfig(auto_delay=False, inference_delay=16)
    rlt = RTCRLTConfig(
        enabled=True, checkpoint=str((tmp_path / "test_actor.pt").resolve()), start_step=0, end_step=1000
    )
    instructions = SessionInstructions(
        map_robot_keys_to_lerobot_features(robot),
        tuple(robot.action_features),
        30,
        rtc,
        actions_per_chunk=50,
        policy_device="cpu",
        pretrained_name_or_path=str(base.resolve()),
        policy_type="pi05",
        rlt=rlt,
        action_sg_window=0,
    )
    head = RLTHeadBundle(
        RLTHeadConfig(
            token_dim=8,
            token_heads=2,
            token_pool_size=4,
            enc_layers=1,
            dec_layers=1,
            chunk_size=50,
            action_dim=2,
            proprio_dim=2,
            actor_hidden_dim=8,
            critic_hidden_dim=8,
        ),
        build_contract(service, instructions),
    )
    # A synthetic artifact exercises validation/transport only. No robot SDK or
    # physical actions are used; these counters do not assert task performance.
    head.training_updates = {"token": 1, "critic": 1, "actor": 1}
    with torch.no_grad():
        for parameter in head.actor.parameters():
            parameter.zero_()
        target = torch.stack((torch.sin(torch.arange(50) * 2.0) * 0.3, torch.full((50,), 0.15)), dim=-1)
        head.actor.net[-1].bias.copy_(target.flatten())
    head.save(rlt.checkpoint)
    options = {
        "pretrained_name_or_path": str(base.resolve()),
        "policy_type": "pi05",
        "policy_device": "cpu",
        "actions_per_chunk": 50,
        "action_sg_window": 0,
        "rtc": rtc,
        "rlt": rlt,
    }
    return service, policy, robot, options, target.numpy() * 2


def test_real_grpc_rlt_first_and_continuation_preserve_actor_targets(setup_rlt, monkeypatch):
    service, policy, robot, options, actor_targets = setup_rlt
    with serve(service) as (address, stub):
        _, ready = stub.Ready.with_call(services_pb2.Empty(), timeout=5)
        assert RLT_HEADER == "rtc-rlt-direct-vla"
        assert dict(ready.initial_metadata())[RLT_HEADER] == "1"
        assert dict(ready.initial_metadata())[RLT_PHASE_HEADER] == "1"
        assert "rtc-rlt-after-sg" not in dict(ready.initial_metadata())
        client = RTCRobotClient(
            RTCClientConfig(
                robot=PiperFollowerConfig(port="unused"), server_address=address, fps=30, **options
            ),
            robot=robot,
        )
        received = []
        exchange = client._exchange_chunk

        def capture_response(*args):
            result, elapsed = exchange(*args)
            received.append(result)
            return result, elapsed

        monkeypatch.setattr(client, "_exchange_chunk", capture_response)
        try:
            client.start()
            assert robot.connected and robot.connect_calls == 1
            # The window-zero configuration remains backwards compatible.
            assert options["action_sg_window"] == 0
            assert client.config.action_sg_window == 0
            assert service.instructions.action_sg_window == 0
            assert service.instructions.policy_device == "cpu"
            assert client.server_policy_device == "cpu"
            client.request_chunk()
            first = received[0]
            assert first.rlt_applied and first.applied_sg_window == 0
            expected_reference = policy.actions[0].numpy() * 2
            np.testing.assert_array_equal(first.rlt_reference_actions, expected_reference)
            assert first.sg_reference_actions is None
            smoothed_reference, _ = filter_joint_targets(expected_reference, (0,), 7)
            assert not np.allclose(first.rlt_reference_actions, smoothed_reference)
            np.testing.assert_allclose(first.processed_actions, actor_targets, atol=1e-6)
            np.testing.assert_array_equal(client.buffer.processed, first.processed_actions)
            assert client.buffer.applied_sg_window == 0
            twice_filtered, _ = filter_joint_targets(actor_targets, (0,), 7)
            assert not np.allclose(twice_filtered, client.buffer.processed)
            for _ in range(5):
                client.buffer.advance()
            expected_prefix = first.original_actions[5:].copy()
            client.request_chunk()
            assert len(policy.calls) == 2
            assert policy.calls[0]["prev_chunk_left_over"] is None
            prepared = policy.calls[1]["prepared_rtc_prefix"]
            np.testing.assert_allclose(prepared.actions[0, :45].numpy(), expected_prefix, atol=1e-6)
            second = received[1]
            np.testing.assert_allclose(
                second.processed_actions[:20], second.rlt_reference_actions[:20], atol=1e-6
            )
            np.testing.assert_array_equal(client.buffer.processed, second.processed_actions)
        finally:
            client.stop()
        assert not robot.connected and service.session_id is None


def test_real_grpc_rollout_records_rlt_policy_source(setup_rlt):
    service, _policy, robot, options, actor_targets = setup_rlt
    with serve(service) as (address, _stub):
        client = RTCPolicyActionClient(
            RemotePolicyRecordConfig(enable=True, backend="rtc", server_address=address, **options),
            robot,
            fps=30,
        )
        try:
            client.start()
            assert not robot.connected  # Recording loop owns hardware connection.
            assert options["action_sg_window"] == 0
            assert client.cfg.action_sg_window == 0
            assert service.instructions.action_sg_window == 0
            assert (
                client.policy_id == f"{options['pretrained_name_or_path']}::rlt={options['rlt'].checkpoint}"
            )
            observation = robot.get_observation()
            assert client.get_action(observation, "test", 0) is None
            deadline = time.monotonic() + 5
            while not client.action_ready and client.error is None and time.monotonic() < deadline:
                time.sleep(0.005)
            assert client.error is None
            assert client.action_ready
            np.testing.assert_allclose(client._buffer.processed, actor_targets, atol=1e-6)
            assert client._buffer.applied_sg_window == 0
            action = client.get_action(observation, "test", 1)
            assert action == dict(zip(robot.action_features, actor_targets[0].tolist(), strict=True))
            assert client.last_reservation.request_id == 0
        finally:
            client.stop()


@pytest.mark.parametrize("recording", [False, True])
def test_rlt_with_output_sg_over_real_grpc(setup_rlt, recording):
    service, _policy, robot, options, actor_targets = setup_rlt
    options["action_sg_window"] = 7
    head = RLTHeadBundle.load(options["rlt"].checkpoint)
    head.contract.update(sg_window=7, sg_stage="client_output")
    head.save(options["rlt"].checkpoint)
    expected, _ = filter_joint_targets(actor_targets, (0,), 7)
    with serve(service) as (address, stub):
        _, ready = stub.Ready.with_call(services_pb2.Empty(), timeout=5)
        assert dict(ready.initial_metadata())[RLT_OUTPUT_SG_HEADER] == "1"
        if recording:
            client = RTCPolicyActionClient(
                RemotePolicyRecordConfig(enable=True, backend="rtc", server_address=address, **options),
                robot, fps=30,
            )
        else:
            client = RTCRobotClient(
                RTCClientConfig(robot=PiperFollowerConfig(port="unused"), server_address=address, **options),
                robot=robot,
            )
        try:
            client.start()
            if recording:
                client.get_action(robot.get_observation(), "test", 0)
                deadline = time.monotonic() + 5
                while not client.action_ready and client.error is None and time.monotonic() < deadline:
                    time.sleep(0.005)
                assert client.error is None and client.action_ready
                buffer = client._buffer
            else:
                client.request_chunk()
                buffer = client.buffer
            assert service.instructions.action_sg_window == 7
            assert buffer.applied_sg_window == 7
            np.testing.assert_allclose(buffer.processed, expected, atol=1e-6)
            np.testing.assert_array_equal(buffer.processed[:, 1], actor_targets[:, 1])
            assert not np.allclose(buffer.processed[:, 0], actor_targets[:, 0])
        finally:
            client.stop()


def test_old_server_cannot_silently_disable_requested_rlt_output_sg(setup_rlt, monkeypatch):
    service, _policy, robot, options, _target = setup_rlt
    options["action_sg_window"] = 7
    monkeypatch.setattr(server_module, "RLT_OUTPUT_SG_HEADER", "unsupported-output-sg")
    with serve(service) as (address, _stub):
        client = RTCRobotClient(
            RTCClientConfig(robot=PiperFollowerConfig(port="unused"), server_address=address, **options), robot=robot
        )
        try:
            with pytest.raises(RuntimeError, match="client output S-G"):
                client.start()
            assert robot.connect_calls == 0
        finally:
            client.stop()


def test_legacy_sg_rlt_server_is_rejected_before_robot_connect(setup_rlt, monkeypatch):
    service, _policy, robot, options, _target = setup_rlt
    monkeypatch.setattr(server_module, "RLT_HEADER", "rtc-rlt-after-sg")
    with serve(service) as (address, _stub):
        client = RTCRobotClient(
            RTCClientConfig(
                robot=PiperFollowerConfig(port="unused"), server_address=address, fps=30, **options
            ),
            robot=robot,
        )
        try:
            with pytest.raises(RuntimeError, match="updated RTC server"):
                client.start()
            assert not robot.connected and robot.connect_calls == 0
            assert service.session_id is None and service.rlt_head is None
        finally:
            client.stop()


@pytest.mark.parametrize("recording", [False, True])
def test_server_missing_interval_capability_is_rejected_before_hardware(setup_rlt, monkeypatch, recording):
    service, _policy, robot, options, _target = setup_rlt
    monkeypatch.setattr(server_module, "RLT_PHASE_HEADER", "rtc-rlt-interval-obsolete")
    with serve(service) as (address, _stub):
        if recording:
            client = RTCPolicyActionClient(
                RemotePolicyRecordConfig(enable=True, backend="rtc", server_address=address, **options),
                robot,
                fps=30,
            )
        else:
            client = RTCRobotClient(
                RTCClientConfig(
                    robot=PiperFollowerConfig(port="unused"), server_address=address, fps=30, **options
                ),
                robot=robot,
            )
        try:
            with pytest.raises(RuntimeError, match="step intervals"):
                client.start()
            assert robot.connect_calls == 0 and not robot.connected
            assert service.session_id is None and service.rlt_head is None
        finally:
            client.stop()


def test_real_grpc_off_on_off_mixed_chunks_and_protected_bypass(setup_rlt):
    service, policy, robot, options, actor_targets = setup_rlt
    options["rlt"].start_step, options["rlt"].end_step = 50, 80
    with serve(service) as (address, _stub):
        client = RTCRobotClient(
            RTCClientConfig(
                robot=PiperFollowerConfig(port="unused"), server_address=address, fps=30, **options
            ),
            robot=robot,
        )
        try:
            client.start()
            reference = policy.actions[0].numpy() * 2
            # Before (even if a chunk crosses the future start), active with a
            # mixed tail, fully protected inside, active with five actor rows,
            # and after: one complete schedule without off-state actor calls.
            specifications = [(40, 0), (50, 0), (60, 30), (55, 30), (80, 0)]
            markers = []
            for request_id, (episode_step, prefix_count) in enumerate(specifications):
                prefix = None if not prefix_count else policy.actions[0, :prefix_count].numpy().copy()
                observation = dict(robot.get_observation(), task="test")
                request = RTCObservation(
                    request_id, episode_step, observation, prefix, 16, episode_step=episode_step
                )
                begin = time.monotonic_ns()
                result, _elapsed = client._exchange_chunk(request, begin, begin)
                expected = rlt_row_mask(options["rlt"], request, 50, options["rtc"].execution_horizon)
                np.testing.assert_array_equal(result.rlt_action_mask, expected)
                assert result.rlt_applied == bool(expected.any())
                markers.append(result.rlt_applied)
                np.testing.assert_array_equal(result.processed_actions[~expected], reference[~expected])
                if expected.any():
                    np.testing.assert_allclose(
                        result.processed_actions[expected], actor_targets[expected], atol=1e-6
                    )
                else:
                    assert result.rlt_reference_actions is None
            assert markers == [False, True, False, True, False]
            assert len(policy.calls) == 2 and len(policy.plain_calls) == 3
            assert policy.calls[1]["prepared_rtc_prefix"] is not None
        finally:
            client.stop()


@pytest.mark.parametrize("failure", ["missing_checkpoint", "contract_mismatch", "legacy_sg_checkpoint"])
def test_real_grpc_rejects_bad_rlt_before_robot_connect(setup_rlt, failure):
    service, policy, robot, options, _target = setup_rlt
    if failure == "missing_checkpoint":
        options["rlt"] = RTCRLTConfig(
            enabled=True, checkpoint=options["rlt"].checkpoint + ".missing", start_step=0, end_step=1000
        )
    elif failure == "legacy_sg_checkpoint":
        payload = torch.load(options["rlt"].checkpoint, map_location="cpu", weights_only=True)
        payload["contract"]["sg_window"] = 7
        torch.save(payload, options["rlt"].checkpoint)
    with serve(service) as (address, _stub):
        client = RTCRobotClient(
            RTCClientConfig(
                robot=PiperFollowerConfig(port="unused"),
                server_address=address,
                fps=20 if failure == "contract_mismatch" else 30,
                **options,
            ),
            robot=robot,
        )
        try:
            with pytest.raises(grpc.RpcError) as error:
                client.start()
            assert error.value.code() == grpc.StatusCode.INVALID_ARGUMENT
            assert "Could not prepare RLT" in error.value.details()
            if failure == "contract_mismatch":
                assert "fps" in error.value.details()
            elif failure == "legacy_sg_checkpoint":
                assert any(text in error.value.details() for text in ("sg_window", "S-G", "contract"))
            assert not robot.connected and robot.connect_calls == 0
            assert service.session_id is None and service.rlt_head is None
            assert not policy.calls
        finally:
            client.stop()
