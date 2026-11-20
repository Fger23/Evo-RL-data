"""Check connection warmup ordering without a network, model weights, or robot."""

import pickle
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import grpc
import pytest
import torch

from lerobot.configs.types import FeatureType, PolicyFeature, RTCAttentionSchedule
from lerobot.inference_time_rtc import policy_server
from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCServerConfig
from lerobot.inference_time_rtc.protocol import CHECKPOINT_PATH_HEADER, SESSION_HEADER, SessionInstructions
from lerobot.transport import services_pb2
from lerobot.utils.constants import ACTION, OBS_STATE


class SetupAbortError(Exception):
    def __init__(self, code, details):
        super().__init__(details)
        self.code = code


def setup_context(session_id):
    context = Mock()
    context.invocation_metadata.return_value = ((SESSION_HEADER, session_id),)
    context.is_active.return_value = True

    def abort(code, details):
        raise SetupAbortError(code, details)

    context.abort.side_effect = abort
    return context


@pytest.fixture
def cache_setup():
    config = RTCServerConfig(
        "unused",
        policy_dtype="bfloat16",
        compile_model=True,
        compile_mode="max-autotune",
        warmup_on_connect=True,
        warmup_task="test the RTC cache",
    )
    policy = Mock(
        config=SimpleNamespace(
            chunk_size=50,
            image_features={},
            input_features={OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(2,))},
            output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(2,))},
        ),
        rtc_processor=None,
    )
    policy.predict_action_chunk.return_value = torch.arange(100, dtype=torch.float32).view(1, 50, 2)
    identity = Mock(side_effect=lambda value: value)
    service = policy_server.RTCPolicyServer(config, components=(policy, identity, identity))
    rtc = InferenceTimeRTCConfig(
        execution_horizon=26,
        queue_threshold=35,
        max_guidance_weight=1.25,
        prefix_attention_schedule=RTCAttentionSchedule.LINEAR,
        auto_delay=False,
        inference_delay=18,
    )
    instructions = SessionInstructions({}, ("joint1", "joint2"), 25.0, rtc, 45, "cpu")
    request = services_pb2.PolicySetup(data=pickle.dumps(instructions))
    return service, policy, instructions, request


@pytest.mark.parametrize("policy_type", ["pi05", "pi0"])
def test_connection_warmup_requires_compiled_pi05(policy_type):
    with pytest.raises(ValueError, match="warmup_on_connect"):
        RTCServerConfig("unused", policy_type=policy_type, warmup_on_connect=True)


@pytest.mark.parametrize("enabled", [False, True])
def test_setup_runs_synthetic_first_and_guided_chunks_before_publish(cache_setup, monkeypatch, enabled):
    service, policy, instructions, request = cache_setup
    service.config.warmup_on_connect = enabled
    warmup = Mock(wraps=policy_server.warmup)
    monkeypatch.setattr(policy_server, "warmup", warmup)
    predict = policy.predict_action_chunk

    def check_setup(observation, **kwargs):
        assert service.session_id is None
        assert service.instructions is None
        assert service.pending is None
        assert service.device == "cpu"
        assert observation["task"] == service.config.warmup_task
        assert kwargs["inference_delay"] == instructions.rtc.inference_delay
        assert kwargs["execution_horizon"] == instructions.rtc.execution_horizon
        return predict.return_value

    predict.side_effect = check_setup
    context = setup_context("test-session")
    service.SendPolicyInstructions(request, context)

    assert service.session_id == "test-session"
    assert service.instructions == instructions
    assert service.last_request_id == -1
    assert service.pending is None
    context.send_initial_metadata.assert_called_once_with(
        (("rtc-chunk-size", "45"), ("rtc-policy-device", "cpu"))
    )
    if not enabled:
        warmup.assert_not_called()
        predict.assert_not_called()
        return

    warmup.assert_called_once()
    owner, settings = warmup.call_args.args
    assert owner is service
    assert settings.task == service.config.warmup_task
    assert settings.rtc == instructions.rtc
    assert settings.actions_per_chunk == 45
    assert settings.prefix_steps == [35] and settings.repeats == 3
    prefixes = [call.kwargs["prev_chunk_left_over"] for call in predict.call_args_list]
    assert prefixes[:3] == [None, None, None]
    assert [prefix.shape for prefix in prefixes[3:]] == [(35, 2)] * 3
    torch.testing.assert_close(prefixes[3][0], torch.tensor([20.0, 21.0]))


@pytest.mark.parametrize("status", [grpc.StatusCode.CANCELLED, grpc.StatusCode.INTERNAL])
def test_failed_warmup_keeps_session_unowned_and_accepts_retry(cache_setup, status):
    service, policy, _, request = cache_setup
    context = setup_context("failed-session")

    def fail_sample(*args, **kwargs):
        if status == grpc.StatusCode.INTERNAL:
            raise RuntimeError("synthetic warmup failure")
        context.is_active.return_value = False
        return policy.predict_action_chunk.return_value

    predict = policy.predict_action_chunk
    predict.side_effect = fail_sample
    with pytest.raises(SetupAbortError) as exc_info:
        service.SendPolicyInstructions(request, context)
    assert exc_info.value.code == status
    assert service.session_id is None and service.instructions is None and service.pending is None
    predict.assert_called_once()
    context.send_initial_metadata.assert_not_called()

    predict.reset_mock(side_effect=True)
    service.SendPolicyInstructions(request, setup_context("retry-session"))
    assert service.session_id == "retry-session"
    assert predict.call_count == 6


def test_client_checkpoint_is_used_for_loading_and_connection_warmup(cache_setup, monkeypatch, tmp_path):
    old, policy, instructions, _ = cache_setup
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    config = replace(old.config, pretrained_name_or_path=None, checkpoint_root=str(tmp_path))
    service = policy_server.RTCPolicyServer(config)
    load = Mock(return_value=(policy, old.preprocessor, old.postprocessor))
    monkeypatch.setattr(policy_server, "load_policy", load)
    warmup = Mock(wraps=policy_server.warmup)
    monkeypatch.setattr(policy_server, "warmup", warmup)
    client = replace(instructions, policy_type="pi05", pretrained_name_or_path=str(checkpoint))
    request = services_pb2.PolicySetup(data=pickle.dumps(client))
    context = setup_context("selected-checkpoint")

    service.SendPolicyInstructions(request, context)

    checkpoint = str(checkpoint.resolve())
    assert load.call_args.args[0].pretrained_name_or_path == checkpoint
    assert load.call_args.args[0].policy_device == "cpu"
    assert warmup.call_args.args[1].pretrained_name_or_path == checkpoint
    assert (
        dict(context.send_initial_metadata.call_args.args[0])[CHECKPOINT_PATH_HEADER].decode() == checkpoint
    )
    assert service.session_id == "selected-checkpoint"
