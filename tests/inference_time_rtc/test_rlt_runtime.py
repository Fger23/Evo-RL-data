"""Check the action-coordinate and timeline boundary, without robot hardware."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
from lerobot.inference_time_rtc.action_buffer import RTCActionBuffer
from lerobot.inference_time_rtc.action_filter import filter_joint_targets
from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCRLTConfig
from lerobot.inference_time_rtc.protocol import RTCActionChunk, RTCObservation, SessionInstructions
from lerobot.inference_time_rtc.rlt.networks import RLTHeadBundle, RLTHeadConfig
from lerobot.inference_time_rtc.rlt.runtime import build_contract, load_head, predict_chunk, prepare_reference
from lerobot.processor import NormalizerProcessorStep
from lerobot.utils.constants import ACTION


def fixture_service(tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "config.json").write_text('{"type":"pi05"}')
    (base / "policy_preprocessor.json").write_text('{"stats":"unchanged"}')
    normalizer = NormalizerProcessorStep(
        features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        norm_map={FeatureType.ACTION: NormalizationMode.MEAN_STD},
        stats={ACTION: {"mean": torch.tensor([10.0, 20.0]), "std": torch.tensor([2.0, 4.0])}},
    )
    physical = np.stack(
        (np.array([10, 11, 8, 14, 9, 13, 10, 12, 11, 10]), np.arange(10) + 20), axis=1
    ).astype(np.float32)
    raw = (physical - [10, 20]) / [2, 4]
    service = SimpleNamespace(
        device="cpu",
        chunk_size=10,
        action_dim=2,
        checkpoint_root=tmp_path,
        config=SimpleNamespace(policy_type="pi05"),
        model_config=SimpleNamespace(pretrained_name_or_path=str(base)),
        policy=SimpleNamespace(
            config=SimpleNamespace(robot_state_feature=SimpleNamespace(shape=(2,)), image_features={}),
            predict_action_chunk_with_features=lambda: None,
            model=SimpleNamespace(
                paligemma_with_expert=SimpleNamespace(
                    paligemma=SimpleNamespace(
                        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=8))
                    )
                )
            ),
        ),
        preprocessor=SimpleNamespace(steps=[normalizer]),
        postprocessor=lambda value: value * torch.tensor([2.0, 4.0]) + torch.tensor([10.0, 20.0]),
        _predict_vla=lambda *args, **kwargs: (
            RTCActionChunk(1, 0, raw.astype(np.float32), physical),
            torch.zeros(1, 6, 8),
            torch.ones(1, 6, dtype=torch.bool),
            torch.zeros(1, 2),
        ),
    )
    instructions = SessionInstructions(
        {},
        ("joint.pos", "gripper.pos"),
        25,
        InferenceTimeRTCConfig(execution_horizon=4, inference_delay=1, queue_threshold=6, auto_delay=False),
        actions_per_chunk=10,
        action_sg_window=0,
        rlt=RTCRLTConfig(enabled=True, checkpoint=str(tmp_path / "actor.pt"), start_step=0, end_step=1000),
    )
    service.predict_calls = []

    def predict_vla(*args, with_features=False, **kwargs):
        service.predict_calls.append(with_features)
        request = args[0] if args else RTCObservation(1, 0, {}, None, 1)
        result = RTCActionChunk(
            request.request_id, request.start_step, raw.astype(np.float32), physical.copy()
        )
        if not with_features:
            return result
        return result, torch.zeros(1, 6, 8), torch.ones(1, 6, dtype=torch.bool), torch.zeros(1, 2)

    service._predict_vla = predict_vla
    return service, instructions, physical


@pytest.mark.parametrize("recovery", [False, True])
def test_direct_vla_reference_and_actor_output_becomes_the_next_prefix(tmp_path, monkeypatch, recovery):
    service, instructions, physical = fixture_service(tmp_path)
    received = {}

    def forbidden(*args, **kwargs):
        raise AssertionError("RLT must not smooth or renormalize the RTC VLA reference")

    monkeypatch.setattr(
        "lerobot.inference_time_rtc.rlt.runtime.filter_joint_targets", forbidden, raising=False
    )
    monkeypatch.setattr("lerobot.inference_time_rtc.rlt.runtime.normalize_actions", forbidden)
    monkeypatch.setattr("lerobot.inference_time_rtc.action_buffer.filter_joint_targets", forbidden)

    def actor(tokens, proprio, reference, protected_steps, action_mask):
        received.update(reference=reference.clone(), protected=protected_steps, mask=action_mask.clone())
        result = reference.clone()
        result[:, :, 0] += action_mask * 0.1
        return result

    service.rlt_head = SimpleNamespace(config=SimpleNamespace(token_pool_size=4), predict_chunk=actor)
    request = RTCObservation(
        1,
        0,
        {},
        None if recovery else np.zeros((6, 2)),
        1,
        recovery_prefix_steps=2 if recovery else 0,
        episode_step=0,
    )
    result = predict_chunk(service, request, instructions)
    expected_normalized = ((physical - [10, 20]) / [2, 4]).astype(np.float32)
    np.testing.assert_allclose(received["reference"][0], expected_normalized, atol=1e-6)
    protected = 2 if recovery else 4
    assert received["protected"] == protected
    np.testing.assert_array_equal(result.rlt_action_mask, np.arange(10) >= protected)
    np.testing.assert_array_equal(result.processed_actions[:protected], physical[:protected])
    np.testing.assert_array_equal(result.processed_actions[:, 1], physical[:, 1])
    np.testing.assert_array_equal(result.rlt_reference_actions, physical)
    assert result.sg_reference_actions is None
    assert result.rlt_applied and result.applied_sg_window == 0

    # With output S-G disabled the actor targets and normalized prefix are exact.
    buffer = RTCActionBuffer(sg_window=0, joint_columns=(0,))
    if recovery:
        buffer.started = True
        buffer._motion_paused = True
        assert buffer.resume(result)
    else:
        assert buffer.merge(result) == 0
    np.testing.assert_array_equal(buffer.processed, result.processed_actions)
    _, prefix, _ = buffer.snapshot()
    np.testing.assert_array_equal(prefix, result.original_actions)
    assert buffer.reserve().request_id == result.request_id


def test_reference_retains_exact_vla_normalized_values_without_action_processor(tmp_path, monkeypatch):
    service, instructions, physical = fixture_service(tmp_path)
    vla_result, features, valid_mask, proprio = service._predict_vla(with_features=True)
    # Deliberately decouple normalized VLA values from its physical targets.
    # A decode/re-encode would lose this exact reference and fail this test.
    vla_result.original_actions += np.float32(0.03125)
    service._predict_vla = lambda *args, **kwargs: (vla_result, features, valid_mask, proprio)

    def forbidden(*args, **kwargs):
        raise AssertionError("RLT references must come directly from the VLA output")

    monkeypatch.setattr("lerobot.inference_time_rtc.rlt.runtime.normalize_actions", forbidden)
    monkeypatch.setattr(
        "lerobot.inference_time_rtc.rlt.runtime.filter_joint_targets", forbidden, raising=False
    )
    normalized, actual_physical, actual_features, actual_proprio = prepare_reference(
        service, RTCObservation(0, 0, {}, None, 1), instructions
    )
    np.testing.assert_array_equal(normalized[0], vla_result.original_actions)
    np.testing.assert_array_equal(actual_physical, physical)
    torch.testing.assert_close(actual_features, features)
    torch.testing.assert_close(actual_proprio, proprio)


@pytest.mark.parametrize("episode_step", [0, 20, 100])
def test_outside_interval_bypasses_features_and_actor(tmp_path, episode_step):
    service, instructions, physical = fixture_service(tmp_path)
    instructions.rlt.start_step, instructions.rlt.end_step = 10, 20

    def forbidden(*args, **kwargs):
        raise AssertionError("Unmarked action rows must use the ordinary RTC VLA path")

    service.rlt_head = SimpleNamespace(predict_chunk=forbidden)
    request = RTCObservation(1, 0, {}, None, 1, episode_step=episode_step)
    result = predict_chunk(service, request, instructions)
    assert service.predict_calls == [False]
    assert not result.rlt_applied and not result.rlt_action_mask.any()
    assert result.rlt_reference_actions is None
    np.testing.assert_array_equal(result.processed_actions, physical)


def test_mixed_interval_chunk_passes_exact_mask_and_keeps_other_rows(tmp_path):
    service, instructions, physical = fixture_service(tmp_path)
    instructions.rlt.start_step, instructions.rlt.end_step = 3, 7
    captured = {}

    def actor(tokens, proprio, reference, protected_steps, action_mask):
        captured["mask"] = action_mask.clone()
        output = reference.clone()
        output[:, :, 0] += action_mask * 0.1
        return output

    service.rlt_head = SimpleNamespace(config=SimpleNamespace(token_pool_size=4), predict_chunk=actor)
    result = predict_chunk(service, RTCObservation(1, 0, {}, None, 1, episode_step=3), instructions)
    expected = np.arange(10) < 4
    np.testing.assert_array_equal(captured["mask"][0], expected)
    np.testing.assert_array_equal(result.rlt_action_mask, expected)
    np.testing.assert_array_equal(result.processed_actions[~expected], physical[~expected])
    assert np.all(result.processed_actions[expected, 0] > physical[expected, 0])
    assert service.predict_calls == [True]


def test_warmup_can_exercise_actor_when_interval_is_shorter_than_prefix(tmp_path):
    service, instructions, _physical = fixture_service(tmp_path)
    instructions.rlt.start_step, instructions.rlt.end_step = 0, 2

    def actor(tokens, proprio, reference, protected_steps, action_mask):
        assert protected_steps == 4
        assert action_mask[0].tolist() == [False] * 4 + [True] * 6
        return reference

    service.rlt_head = SimpleNamespace(config=SimpleNamespace(token_pool_size=4), predict_chunk=actor)
    request = RTCObservation(1, 0, {}, np.zeros((6, 2)), 1, episode_step=0)
    ordinary = predict_chunk(service, request, instructions)
    assert not ordinary.rlt_applied
    warmed = predict_chunk(service, request, instructions, warmup=True)
    assert warmed.rlt_applied and service.predict_calls == [False, True]


def test_checkpoint_contract_rejects_new_stats_or_untrained_heads(tmp_path):
    service, instructions, _ = fixture_service(tmp_path)
    cfg = RLTHeadConfig(
        token_dim=8,
        token_heads=2,
        enc_layers=1,
        dec_layers=1,
        token_ff_dim=16,
        chunk_size=10,
        action_dim=2,
        proprio_dim=2,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
    )
    bundle = RLTHeadBundle(cfg, build_contract(service, instructions))
    bundle.save(instructions.rlt.checkpoint)
    with pytest.raises(ValueError, match="trained"):
        load_head(service, instructions)
    bundle.training_updates = {"token": 1, "critic": 1, "actor": 1}
    bundle.save(instructions.rlt.checkpoint)
    assert load_head(service, instructions).config.chunk_size == 10
    assert build_contract(service, instructions)["sg_window"] == 0
    (tmp_path / "base" / "policy_preprocessor.json").write_text('{"stats":"changed"}')
    with pytest.raises(ValueError, match="normalization"):
        load_head(service, instructions)


@pytest.mark.parametrize("legacy_sg_window", [5, 7])
def test_checkpoint_rejects_legacy_sg_contract(tmp_path, legacy_sg_window):
    service, instructions, _ = fixture_service(tmp_path)
    config = RLTHeadConfig(
        token_dim=8,
        token_heads=2,
        enc_layers=1,
        dec_layers=1,
        chunk_size=10,
        action_dim=2,
        proprio_dim=2,
        actor_hidden_dim=8,
        critic_hidden_dim=8,
    )
    bundle = RLTHeadBundle(config, build_contract(service, instructions))
    bundle.training_updates = {"token": 1, "critic": 1, "actor": 1}
    bundle.save(instructions.rlt.checkpoint)
    payload = torch.load(instructions.rlt.checkpoint, map_location="cpu", weights_only=True)
    payload["contract"]["sg_window"] = legacy_sg_window
    torch.save(payload, instructions.rlt.checkpoint)
    with pytest.raises(ValueError, match="sg_window|S-G|contract"):
        load_head(service, instructions)


def test_rlt_disabled_keeps_existing_client_filter_behavior():
    actions = np.arange(20, dtype=np.float32).reshape(10, 2)
    chunk = RTCActionChunk(0, 0, actions, actions)
    buffer = RTCActionBuffer(sg_window=7, joint_columns=(0,))
    buffer.merge(chunk)
    expected, _ = filter_joint_targets(actions, (0,), 7)
    np.testing.assert_array_equal(buffer.processed, expected)


@pytest.mark.parametrize("legacy_field", ["applied_sg_window", "sg_reference_actions"])
def test_filtered_legacy_rlt_payload_cannot_enter_action_buffer(legacy_field):
    actions = np.arange(20, dtype=np.float32).reshape(10, 2)
    chunk = RTCActionChunk(0, 0, actions, actions.copy(), rlt_applied=True)
    if legacy_field == "applied_sg_window":
        chunk.applied_sg_window = 7
    else:
        chunk.sg_reference_actions = actions.copy()
    buffer = RTCActionBuffer(sg_window=0, joint_columns=(0,))
    with pytest.raises(ValueError, match="server output must be unfiltered"):
        buffer.merge(chunk)
    assert buffer.qsize() == 0 and buffer.last_request_id == -1
