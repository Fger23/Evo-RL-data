"""Offline cache tests: temporal alignment and recorded-command critic targets."""

import json
from argparse import Namespace
from dataclasses import asdict

import numpy as np
import pytest
import torch

from lerobot.inference_time_rtc.rlt.cache import (
    build_cache,
    build_episode_transitions,
    dataset_frame_to_raw,
    load_dataset_specs,
    make_parser,
    phase_annotations,
    select_cache_features,
    success_label,
)


def test_cache_defaults_to_unfiltered_reference_and_rejects_sg_cli():
    arguments = ["--base-model", "/models/pi05", "--dataset-spec", "datasets.json", "--output", "cache.pt"]
    assert make_parser().parse_args(arguments).fps == 30.0
    assert make_parser().parse_args(arguments).sg_window == 0
    assert make_parser().parse_args([*arguments, "--sg-window", "0"]).sg_window == 0
    with pytest.raises(SystemExit):
        make_parser().parse_args([*arguments, "--sg-window", "7"])


def test_cache_api_rejects_filter_before_model_or_dataset_imports():
    # Deliberately omit all other fields: filter validation must precede any use
    # of model paths, datasets, camera backends, or the remaining arguments.
    with pytest.raises(ValueError, match="--sg-window=0.*rebuild old S-G caches and retrain"):
        build_cache(Namespace(sg_window=7))


@pytest.mark.parametrize("value,expected", [(True, True), (0, False), ("success", True), ("failure", False)])
def test_success_requires_explicit_outcomes(value, expected):
    assert success_label(value) is expected


@pytest.mark.parametrize("value", [None, "unknown", 2, float("nan"), torch.ones(2)])
def test_missing_outcome_is_not_assumed_success(value):
    with pytest.raises(ValueError, match="episode_success"):
        success_label(value)


def _episode(length=115, outcome=True):
    calls = []
    commands = torch.arange(length, dtype=torch.float32).unsqueeze(-1).repeat(1, 2) + 1000

    def reference_at(start, prefix):
        calls.append((start, None if prefix is None else prefix.clone()))
        reference = torch.arange(50, dtype=torch.float32).unsqueeze(-1).repeat(1, 2) + start * 100
        return reference, torch.full((3, 8), float(start)), torch.tensor([float(start), 1.0])

    data = build_episode_transitions(
        commands,
        reference_at,
        lambda value: value.unsqueeze(0) * 2,
        outcome=outcome,
        rlt_phase=torch.ones(length, dtype=torch.bool),
        rlt_phase_id=torch.zeros(length, dtype=torch.long),
        chunk_size=50,
        stride=20,
        execution_horizon=20,
    )
    return data, calls, commands


def test_critic_targets_are_recorded_commands_not_vla_references():
    data, _calls, commands = _episode()
    assert torch.equal(data["executed_actions"][0], commands[:50] * 2)
    assert not torch.equal(data["executed_actions"][0], data["reference_actions"][0])
    assert {0, 20, 40, 60, 80, 100} <= set(data["start_step"].tolist())
    assert torch.equal(data["n_steps"], (115 - data["start_step"]).clamp(max=50))


def test_continuation_uses_only_forward_leftovers_of_sampled_reference():
    data, calls, _commands = _episode()
    current_twenty = next(prefix for start, prefix in calls if start == 20)
    assert torch.equal(current_twenty[:, 0], torch.arange(20, 50, dtype=torch.float32))
    current_forty = next(prefix for start, prefix in calls if start == 40)
    assert torch.equal(current_forty[:, 0], torch.arange(20, 50, dtype=torch.float32) + 2000)
    assert data["protected_steps"][0] == 0
    assert torch.equal(data["protected_steps"][1:], data["n_steps"][1:].clamp(max=20))
    assert data["next_protected_steps"][0] == 20
    assert not data["next_protected_steps"][data["terminated"]].any()


def test_td_successor_is_full_horizon_later_not_reference_stride():
    data, _calls, _commands = _episode()
    assert data["vla_tokens"][0, 0, 0] == 0
    assert data["next_vla_tokens"][0, 0, 0] == 50
    assert data["vla_tokens"][1, 0, 0] == 20
    assert data["next_vla_tokens"][1, 0, 0] == 70
    assert torch.equal(data["terminated"], data["start_step"] + data["n_steps"] == 115)


def test_short_terminal_chunk_masks_padding_and_places_reward_at_last_valid_step():
    data, _calls, commands = _episode()
    row = int(torch.where(data["start_step"] == 100)[0].item())
    assert data["action_mask"][row].sum() == 15
    assert torch.equal(data["executed_actions"][row, :15], commands[100:] * 2)
    assert torch.equal(data["executed_actions"][row, 15:], (commands[-1:] * 2).expand(35, -1))
    assert data["rewards"][row, 14] == 1
    assert data["rewards"][row].sum() == 1
    assert not data["rewards"][~data["terminated"]].any()
    assert not data["next_action_mask"][row].any()


def test_failure_terminal_never_receives_success_reward():
    data, _calls, _commands = _episode(outcome=False)
    assert not data["rewards"].any()


def test_each_episode_starts_without_cross_episode_prefix():
    for _ in range(2):
        data, calls, _commands = _episode(length=12)
        assert calls == [(0, None)]
        assert data["protected_steps"].item() == 0
        assert data["n_steps"].item() == 12


def _marked_episode(length, intervals, outcome=True):
    commands = torch.arange(length, dtype=torch.float32).unsqueeze(-1).repeat(1, 2) + 1000
    active = torch.zeros(length, dtype=torch.bool)
    ids = torch.full((length,), -1, dtype=torch.long)
    for identifier, (start, stop) in enumerate(intervals):
        active[start:stop] = True
        ids[start:stop] = identifier

    def reference_at(start, prefix):
        reference = torch.full((50, 2), float(start))
        return reference, torch.full((3, 8), float(start)), torch.tensor([float(start), 1.0])

    data = build_episode_transitions(
        commands,
        reference_at,
        lambda value: value.unsqueeze(0) * 2,
        outcome=outcome,
        rlt_phase=active,
        rlt_phase_id=ids,
        chunk_size=50,
        stride=20,
        execution_horizon=20,
    )
    return data, commands


def test_each_short_marked_interval_is_anchored_independently_of_global_stride():
    data, _commands = _marked_episode(120, [(3, 5), (47, 50), (90, 91)])
    marked = data["rlt_action_mask"][:, 0]
    assert data["start_step"][marked].tolist() == [3, 47, 90]
    assert data["rlt_phase_id"][marked].tolist() == [0, 1, 2]
    assert data["rlt_action_mask"][marked].sum(dim=1).tolist() == [2, 3, 1]


def test_actor_mask_excludes_separate_interval_inside_the_same_full_action_chunk():
    data, commands = _marked_episode(120, [(3, 5), (47, 50), (90, 91)])
    row = int(torch.where(data["start_step"] == 3)[0].item())
    assert data["action_mask"][row].all()
    assert torch.equal(data["executed_actions"][row], commands[3:53] * 2)
    assert data["rlt_action_mask"][row, :2].all()
    assert not data["rlt_action_mask"][row, 2:].any()
    # The next TD state is t+50=53, outside any phase. Even though another
    # marked interval falls in that future H-window, this actor must stay off.
    assert data["next_vla_tokens"][row, 0, 0] == 53
    assert not data["next_rlt_action_mask"][row].any()


def test_phase_boundary_never_creates_fake_episode_termination_or_success_reward():
    data, _commands = _marked_episode(150, [(3, 70)], outcome=True)
    assert {3, 23, 43, 63} <= set(data["start_step"].tolist())
    marked = data["rlt_action_mask"][:, 0]
    assert (data["n_steps"][marked] == 50).all()
    assert not data["terminated"][marked].any()
    assert not data["rewards"][marked].any()
    row = int(torch.where(data["start_step"] == 3)[0].item())
    assert data["next_vla_tokens"][row, 0, 0] == 53
    assert data["next_rlt_action_mask"][row].sum() == 17
    row = int(torch.where(data["start_step"] == 23)[0].item())
    assert not data["next_rlt_action_mask"][row].any()


def test_td_successor_inside_another_interval_uses_that_successors_own_phase():
    data, _commands = _marked_episode(100, [(0, 5), (50, 54)])
    assert data["rlt_action_mask"][0].sum() == 5
    assert data["next_rlt_action_mask"][0].sum() == 4
    assert data["next_rlt_action_mask"][0, :4].all()
    assert not data["next_rlt_action_mask"][0, 4:].any()


def test_terminal_phase_actor_mask_is_clipped_to_valid_episode_rows():
    data, _commands = _marked_episode(115, [(110, 115)])
    row = int(torch.where(data["start_step"] == 110)[0].item())
    assert data["n_steps"][row] == 5
    assert data["terminated"][row]
    assert data["rlt_action_mask"][row].sum() == 5
    assert not data["rlt_action_mask"][row, 5:].any()
    assert data["rewards"][row, 4] == 1
    assert data["rewards"][row].sum() == 1
    assert not data["next_rlt_action_mask"][row].any()


def test_full_episode_critic_context_contains_every_bootstrap_successor():
    data, _commands = _marked_episode(150, [(3, 7)])
    positions = {int(start): row for row, start in enumerate(data["start_step"])}
    assert 0 in positions and 140 in positions
    assert not data["rlt_action_mask"][positions[0]].any()
    assert not data["rlt_action_mask"][positions[140]].any()
    assert data["rewards"][positions[140], 9] == 1
    for row, start in enumerate(data["start_step"].tolist()):
        if data["terminated"][row]:
            continue
        successor = positions[start + int(data["n_steps"][row])]
        assert torch.equal(data["next_vla_tokens"][row], data["vla_tokens"][successor])
        assert torch.equal(data["next_reference_actions"][row], data["reference_actions"][successor])
        assert torch.equal(data["next_action_mask"][row], data["action_mask"][successor])
        assert data["next_protected_steps"][row] == data["protected_steps"][successor]


def test_marked_cache_masks_satisfy_the_actual_training_interface():
    from lerobot.inference_time_rtc.rlt.networks import RLTHeadConfig
    from lerobot.inference_time_rtc.rlt.training import validate_cache

    tensors, _commands = _marked_episode(120, [(3, 5), (47, 50), (90, 91)])
    config = RLTHeadConfig(token_dim=8, token_heads=2, proprio_dim=2, action_dim=2, chunk_size=50)
    contract = {
        "base_model_path": "/models/pi05", "action_names": ["joint1.pos", "joint2.pos"],
        "sg_window": 0, "fps": 25.0, "normalization": {"processor_sha256": "a" * 64},
        "rtc": {"execution_horizon": 20, "inference_delay": 16},
    }
    validated, _contract, _tensors = validate_cache(
        {"metadata": {"head_config": asdict(config), "contract": contract}, "tensors": tensors},
        require_transitions=True,
    )
    assert validated == config


def test_missing_phase_columns_do_not_silently_train_the_whole_task():
    with pytest.raises(ValueError, match="requires complementary_info.rlt_phase"):
        build_episode_transitions(
            torch.ones(60, 2), lambda *_: pytest.fail("VLA must not run"), lambda value: value,
            outcome=True,
        )


@pytest.mark.parametrize(
    "active,ids,error",
    [
        ([False, False], [-1, -1], "at least one explicitly marked"),
        ([True, False], [-1, -1], "nonnegative phase IDs"),
        ([True, False], [0, 0], "outside frames require -1"),
        ([True, False, True], [0, -1, 0], "one contiguous interval"),
        ([True, True], [0.5, 0.5], "integer interval IDs"),
        ([1, 2], [0, 0], "boolean or explicit 0/1"),
    ],
)
def test_invalid_phase_annotations_are_rejected(active, ids, error):
    with pytest.raises(ValueError, match=error):
        phase_annotations(active, ids, len(active))


def test_scalar_and_singleton_phase_columns_produce_the_same_interval_schema():
    active, ids, intervals = phase_annotations(
        [torch.tensor([False]), torch.tensor([True]), torch.tensor([True])],
        torch.tensor([[-1], [4], [4]]),
        3,
    )
    assert active.tolist() == [False, True, True]
    assert ids.tolist() == [-1, 4, 4]
    assert intervals == [(1, 3, 4)]


def test_dataset_frame_restores_raw_state_order_and_exact_rgb_bytes():
    pixels = torch.tensor([[[0.0, 1.0]], [[128 / 255, 64 / 255]], [[1.0, 0.0]]])
    features = {
        "observation.state": {"names": ["right.joint.pos", "left.joint.pos"], "shape": (2,)},
        "observation.images.fixed": {"dtype": "image", "shape": (1, 2, 3)},
    }
    frame = {
        "observation.state": torch.tensor([12.0, 34.0]),
        "observation.images.fixed": pixels,
        "task": "pick",
    }
    raw = dataset_frame_to_raw(frame, features)
    assert raw["right.joint.pos"] == 12.0
    assert raw["left.joint.pos"] == 34.0
    assert raw["task"] == "pick"
    np.testing.assert_array_equal(raw["fixed"], np.array([[[0, 128, 255], [255, 64, 0]]], dtype=np.uint8))


def test_auxiliary_cameras_are_excluded_before_rtc_observation_conversion():
    features = {
        "observation.state": {"names": ["joint.pos"], "shape": (1,)},
        "action": {"names": ["joint.pos"], "shape": (1,)},
        "observation.images.fixed": {"dtype": "video", "shape": (4, 5, 3)},
        "observation.images.unused_camera": {"dtype": "video", "shape": (4, 5, 3)},
    }
    selected = select_cache_features(features, {"observation.images.fixed": object()})
    assert set(selected) == {"observation.state", "action", "observation.images.fixed"}
    assert "observation.images.unused_camera" in features
    frame = {
        "observation.state": torch.tensor([9.0]),
        "task": "pick",
        "observation.images.fixed": torch.zeros(3, 4, 5),
        # The ignored camera may have a different shape or even be unavailable.
        "observation.images.unused_camera": None,
    }
    raw = dataset_frame_to_raw(frame, selected)
    assert set(raw) == {"joint.pos", "fixed", "task"}


def test_missing_base_camera_is_rejected_before_cache_generation():
    features = {
        "observation.state": {"names": ["joint.pos"], "shape": (1,)},
        "action": {"names": ["joint.pos"], "shape": (1,)},
    }
    with pytest.raises(ValueError, match="base checkpoint camera"):
        select_cache_features(features, {"observation.images.fixed": object()})


def test_raw_pkl_is_rejected_with_conversion_instruction(tmp_path):
    root = tmp_path / "raw"
    root.mkdir()
    (root / "capture.pkl").write_bytes(b"raw")
    path = tmp_path / "datasets.json"
    path.write_text(json.dumps([{"repo_id": "piper/run", "root": "raw", "source": "rollout"}]))
    with pytest.raises(ValueError, match="convert_pkl_dataset.py"):
        load_dataset_specs(path)


def test_spec_paths_and_explicit_labels_resolve_relative_to_manifest(tmp_path):
    root = tmp_path / "converted"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text("{}")
    (tmp_path / "labels.json").write_text('{"0":"success"}')
    path = tmp_path / "datasets.json"
    path.write_text(
        json.dumps(
            [
                {
                    "repo_id": "piper/run",
                    "root": "converted",
                    "source": "rollout",
                    "success_labels": "labels.json",
                }
            ]
        )
    )
    spec = load_dataset_specs(path)[0]
    assert spec["root"] == str(root.resolve())
    assert spec["success_labels"] == {"0": "success"}
