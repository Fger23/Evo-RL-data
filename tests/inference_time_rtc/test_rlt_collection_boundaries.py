"""Record interval annotations without changing inference or moving hardware."""

import ast
import importlib.util
import json
import logging
import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from deepdiff import DeepDiff
from test_rollout_integration import (
    FakeDataset,
    FakeRobot,
    FakeTeleop,
    RolloutRobotConfig,
    RolloutTeleopConfig,
    config_module as config_module,
    load_definitions,
)

from lerobot.datasets.pkl_dataset import PklDataset
from lerobot.datasets.utils import DEFAULT_FEATURES, build_dataset_frame, validate_frame
from lerobot.utils.recording_annotations import resolve_collector_policy_id

ROOT = Path(__file__).resolve().parents[2]


def keyboard_namespace(clock):
    return {
        "Any": Any,
        "threading": threading,
        "sys": SimpleNamespace(stdin=SimpleNamespace(fileno=lambda: 0, isatty=lambda: False)),
        "time": SimpleNamespace(monotonic=lambda: clock.now),
        "logging": logging,
        "INTERVENTION_TOGGLE_COOLDOWN_S": 0.5,
        "EPISODE_SUCCESS": "success",
        "EPISODE_FAILURE": "failure",
    }


def test_tty_r_boundary_is_independent_of_i_and_ignored_outside_recording():
    clock = SimpleNamespace(now=10.0)
    namespace = keyboard_namespace(clock)
    load_definitions("utils/control_utils.py", {"TTYKeyboardListener"}, namespace)
    events = {"toggle_intervention": False, "toggle_rlt_phase": False, "rlt_phase_recording": False}
    listener = namespace["TTYKeyboardListener"](events, "i", "s", "f", "r")
    listener._handle_key("r")
    assert not events["toggle_rlt_phase"]
    events["rlt_phase_recording"] = True
    listener._handle_key("R")
    assert events["toggle_rlt_phase"] and not events["toggle_intervention"]
    listener._handle_key("i")
    assert events["toggle_intervention"]
    events["toggle_rlt_phase"] = False
    listener._handle_key("r")
    assert not events["toggle_rlt_phase"]  # Key-repeat debounce is separate from i.
    clock.now += 0.6
    listener._handle_key("r")
    assert events["toggle_rlt_phase"]
    events["toggle_rlt_phase"] = False
    events["rlt_phase_recording"] = False
    clock.now += 0.6
    listener._handle_key("r")
    assert not events["toggle_rlt_phase"]


def test_pynput_fallback_uses_the_same_r_boundary_and_reset_gate(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    namespace = keyboard_namespace(clock)
    namespace["is_headless"] = lambda: False
    captured = {}

    class Listener:
        def __init__(self, on_press):
            captured["on_press"] = on_press

        def start(self):
            pass

    module = types.ModuleType("pynput")
    module.keyboard = SimpleNamespace(
        Listener=Listener, Key=SimpleNamespace(right=object(), left=object(), esc=object())
    )
    monkeypatch.setitem(sys.modules, "pynput", module)
    load_definitions("utils/control_utils.py", {"init_keyboard_listener"}, namespace)
    _listener, events = namespace["init_keyboard_listener"](
        intervention_toggle_key="i",
        episode_success_key="s",
        episode_failure_key="f",
        rlt_phase_toggle_key="r",
    )
    captured["on_press"](SimpleNamespace(char="r"))
    assert not events["toggle_rlt_phase"]
    events["rlt_phase_recording"] = True
    captured["on_press"](SimpleNamespace(char="r"))
    assert events["toggle_rlt_phase"] and not events["toggle_intervention"]
    captured["on_press"](SimpleNamespace(char="i"))
    assert events["toggle_intervention"]
    captured["on_press"](SimpleNamespace(char="s"))
    assert events["episode_outcome"] == "success" and events["exit_early"]


@pytest.mark.parametrize("key", ["i", "I", "s", "F", "", "rr", " "])
def test_record_config_rejects_invalid_or_conflicting_r_boundary_keys(config_module, key):
    with pytest.raises(ValueError, match="rlt_phase_toggle_key"):
        config_module.RecordConfig(
            robot=RolloutRobotConfig(),
            dataset=config_module.DatasetRecordConfig("piper/test", "test"),
            teleop=RolloutTeleopConfig(),
            rlt_phase_toggle_key=key,
        )


def test_record_config_defaults_to_r(config_module):
    config = config_module.RecordConfig(
        robot=RolloutRobotConfig(),
        dataset=config_module.DatasetRecordConfig("piper/test", "test"),
        teleop=RolloutTeleopConfig(),
    )
    assert config.rlt_phase_toggle_key == "r"
    assert config.intervention_toggle_key == "i"


class BoundaryDataset(FakeDataset):
    features = {
        **FakeDataset.features,
        "complementary_info.rlt_phase": {"dtype": "bool", "shape": (1,), "names": ["rlt_phase"]},
        "complementary_info.rlt_phase_id": {"dtype": "int64", "shape": (1,), "names": ["rlt_phase_id"]},
        "complementary_info.rlt_applied": {"dtype": "bool", "shape": (1,), "names": ["rlt_applied"]},
    }

    def __init__(self, controls, events):
        self.frames = []
        self.controls = controls
        self.events = events
        self.phase_gate_states = []

    def add_frame(self, frame):
        self.phase_gate_states.append(self.events["rlt_phase_recording"])
        self.frames.append(frame)
        count = len(self.frames)
        if count == len(self.controls):
            self.events["exit_early"] = True
        else:
            self.events["toggle_rlt_phase"], self.events["toggle_intervention"] = self.controls[count]


def record_boundaries(
    config_module,
    monkeypatch,
    *,
    actual_rlt=True,
    events=None,
    controls=None,
    features=None,
    marking_enabled=True,
):
    events = {} if events is None else events
    events.update(exit_early=False, toggle_intervention=False)
    controls = controls or [
        (False, False),
        (True, False),
        (False, True),
        (True, False),
        (True, False),
        (False, True),
        (True, False),
    ]
    robot = FakeRobot()
    teleop = FakeTeleop()
    dataset = BoundaryDataset(controls, events)
    if features is not None:
        dataset.features = features
    clock = SimpleNamespace(now=0.0)

    class Remote:
        cfg = SimpleNamespace(backend="rtc", rlt=SimpleNamespace(enabled=actual_rlt))
        last_reservation = None

        def __init__(self):
            self.timesteps = []
            self.pauses = 0

        def reset(self):
            pass

        def pause(self):
            self.pauses += 1

        def get_action(self, *, timestep, **kwargs):
            self.timesteps.append(timestep)
            self.last_reservation = SimpleNamespace(
                request_id=1, step=timestep, chunk_index=timestep, rlt_applied=actual_rlt
            )
            return {"joint.pos": 7.0}

    remote = Remote()
    namespace = {
        "logging": logging,
        "np": np,
        "time": SimpleNamespace(perf_counter=lambda: clock.now, monotonic_ns=lambda: int(clock.now * 1e9)),
        "precise_sleep": lambda duration: setattr(clock, "now", clock.now + duration),
        "Teleoperator": FakeTeleop,
        "ACPInferenceConfig": config_module.ACPInferenceConfig,
        "INTERVENTION_STATE_POLICY": 0.0,
        "INTERVENTION_STATE_ACTIVE": 1.0,
        "INTERVENTION_STATE_RELEASE": 2.0,
        "build_dataset_frame": build_dataset_frame,
        "resolve_collector_policy_id": resolve_collector_policy_id,
        "ACTION": "action",
        "OBS_STR": "observation",
    }
    load_definitions("scripts/recording_loop.py", {"record_loop"}, namespace)
    from lerobot.inference_time_rtc import logging_utils

    monkeypatch.setattr(logging_utils, "log_action_event", lambda *args, **kwargs: None)
    namespace["record_loop"](
        robot=robot,
        events=events,
        fps=25,
        teleop_action_processor=lambda pair: pair[0],
        robot_action_processor=lambda pair: pair[0],
        robot_observation_processor=lambda observation: observation,
        dataset=dataset,
        teleop=teleop,
        remote_policy_client=remote,
        control_time_s=60,
        single_task="test",
        rlt_phase_annotations_enabled=marking_enabled,
    )
    return dataset, robot, teleop, remote, events


@pytest.mark.parametrize("missing_phase_schema", [False, True])
def test_legacy_or_disabled_annotation_schema_ignores_r_without_changing_control(
    config_module, monkeypatch, missing_phase_schema
):
    features = FakeDataset.features if missing_phase_schema else BoundaryDataset.features
    dataset, robot, teleop, _remote, _events = record_boundaries(
        config_module,
        monkeypatch,
        actual_rlt=False,
        features=features,
        marking_enabled=missing_phase_schema,
    )
    assert not any(dataset.phase_gate_states)
    if missing_phase_schema:
        assert all("complementary_info.rlt_phase" not in frame for frame in dataset.frames)
    else:
        assert all(not frame["complementary_info.rlt_phase"].item() for frame in dataset.frames)
        assert all(frame["complementary_info.rlt_phase_id"].item() == -1 for frame in dataset.frames)
    assert [command["joint.pos"] for command in robot.commands] == [7, 7, 99, 99, 99, 7, 7]
    assert teleop.manual_modes == [False, True, False]


def run_resume_schema_policy(*, rlt_enabled, legacy):
    """Run the real resume branch and compatibility check, with no device I/O."""
    features = {"action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]}}
    requested = dict(features)
    for name, dtype in (("rlt_phase", "bool"), ("rlt_phase_id", "int64"), ("rlt_applied", "bool")):
        requested[f"complementary_info.{name}"] = {"dtype": dtype, "shape": (1,), "names": [name]}
    original = {**(features if legacy else requested), **DEFAULT_FEATURES}
    dataset = SimpleNamespace(features=original, fps=25, meta=SimpleNamespace(robot_type="synthetic"))
    namespace = {
        "cfg": SimpleNamespace(
            dataset=SimpleNamespace(
                repo_id="piper/test", root="unused", video_encoding_batch_size=1, vcodec="libsvtav1", fps=25
            ),
            remote_policy=SimpleNamespace(rlt=SimpleNamespace(enabled=rlt_enabled)),
        ),
        "robot": SimpleNamespace(cameras={}, robot_type="synthetic"),
        "rtc_recording": rlt_enabled,
        "dataset_features": requested,
        "LeRobotDataset": lambda *args, **kwargs: dataset,
        "rlt_phase_annotations_enabled": True,
        "logging": logging,
        "DeepDiff": DeepDiff,
        "DEFAULT_FEATURES": DEFAULT_FEATURES,
    }
    load_definitions("utils/control_utils.py", {"sanity_check_dataset_robot_compatibility"}, namespace)
    parsed = ast.parse((ROOT / "src/lerobot/scripts/lerobot_record.py").read_text(encoding="utf-8"))
    record = next(node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "record")
    resume = next(
        node
        for node in ast.walk(record)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Attribute)
        and isinstance(node.test.value, ast.Name)
        and node.test.value.id == "cfg"
        and node.test.attr == "resume"
        and any(
            isinstance(child, ast.Assign)
            and isinstance(child.value, ast.Call)
            and isinstance(child.value.func, ast.Name)
            and child.value.func.id == "LeRobotDataset"
            for child in node.body
        )
    )
    exec(compile(ast.Module(body=resume.body, type_ignores=[]), "resume_schema_policy", "exec"), namespace)
    return namespace, original


def test_original_async_resume_keeps_legacy_schema_and_disables_marking(caplog):
    with caplog.at_level(logging.WARNING):
        namespace, original = run_resume_schema_policy(rlt_enabled=False, legacy=True)
    assert not namespace["rlt_phase_annotations_enabled"]
    assert namespace["dataset"].features is original
    assert not any(key.startswith("complementary_info.rlt_") for key in original)
    assert "interval marking is disabled" in caplog.text
    assert "existing rows and schema are preserved" in caplog.text


def test_rlt_resume_rejects_legacy_schema_instead_of_filling_historical_labels():
    with pytest.raises(ValueError, match="legacy dataset missing RLT metadata"):
        run_resume_schema_policy(rlt_enabled=True, legacy=True)


def test_annotated_dataset_resume_retains_the_complete_rlt_schema():
    namespace, original = run_resume_schema_policy(rlt_enabled=True, legacy=False)
    assert namespace["rlt_phase_annotations_enabled"]
    assert namespace["dataset"].features is original


@pytest.mark.parametrize("actual_rlt", [False, True])
def test_r_marks_intervals_while_i_alone_changes_action_source(config_module, monkeypatch, actual_rlt):
    dataset, robot, teleop, remote, events = record_boundaries(
        config_module, monkeypatch, actual_rlt=actual_rlt
    )
    assert [frame["complementary_info.rlt_phase"].item() for frame in dataset.frames] == [
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    ]
    assert [frame["complementary_info.rlt_phase_id"].item() for frame in dataset.frames] == [
        -1,
        0,
        0,
        -1,
        1,
        1,
        -1,
    ]
    assert [frame["complementary_info.rlt_applied"].item() for frame in dataset.frames] == [
        actual_rlt,
        actual_rlt,
        False,
        False,
        False,
        actual_rlt,
        actual_rlt,
    ]
    assert [command["joint.pos"] for command in robot.commands] == [7, 7, 99, 99, 99, 7, 7]
    assert teleop.manual_modes == [False, True, False] and remote.pauses == 1
    assert remote.timesteps == [0, 1, 5, 6]
    assert remote.cfg.rlt.enabled is actual_rlt
    assert not events["rlt_phase_recording"] and not events["toggle_rlt_phase"]
    for frame in dataset.frames:
        assert frame["complementary_info.rlt_phase"].dtype == np.bool_
        assert frame["complementary_info.rlt_phase_id"].dtype == np.int64
        assert frame["complementary_info.rlt_applied"].dtype == np.bool_
        validate_frame(dict(frame), dataset.features)


def test_episode_start_clears_stale_boundaries_and_restarts_phase_ids(config_module, monkeypatch, caplog):
    _first, _robot, _teleop, _remote, events = record_boundaries(config_module, monkeypatch)
    events["toggle_rlt_phase"] = True
    with caplog.at_level(logging.INFO):
        dataset, *_ = record_boundaries(
            config_module,
            monkeypatch,
            events=events,
            controls=[(False, False), (True, False), (False, False)],
        )
    assert [frame["complementary_info.rlt_phase_id"].item() for frame in dataset.frames] == [-1, 0, 0]
    assert not events["rlt_phase_recording"]
    assert "timestep=3 closed_phase_id=0 episode_end=True" in caplog.text


def test_pkl_and_conversion_keep_boolean_interval_and_integer_ids(config_module, monkeypatch, tmp_path):
    source, *_ = record_boundaries(config_module, monkeypatch)
    raw = tmp_path / "capture"
    raw.mkdir()
    metadata = SimpleNamespace(
        root=raw,
        repo_id="piper/rlt_phase_test",
        features=source.features,
        fps=25,
        info={"robot_type": "synthetic"},
        _close_writer=lambda: None,
    )
    writer = PklDataset(metadata, image_encoding="raw", validate_frame=validate_frame)
    try:
        for frame in source.frames:
            writer.add_frame(dict(frame))
        writer.save_episode(extra_episode_metadata={"episode_success": "success"})
    finally:
        writer.finalize()

    spec = importlib.util.spec_from_file_location("rlt_boundary_converter", ROOT / "convert_pkl_dataset.py")
    converter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(converter)
    _root, info, episodes = converter.capture_info(raw)
    assert converter.validate_capture(info, episodes) == len(source.frames)
    rows = list(converter.episode_records(*episodes[0]))
    for packet, frame in zip(rows, source.frames, strict=True):
        for name in ("rlt_phase", "rlt_phase_id", "rlt_applied"):
            np.testing.assert_array_equal(
                packet["frame"][f"complementary_info.{name}"], frame[f"complementary_info.{name}"]
            )
    output = tmp_path / "converted"
    converter.convert(raw, output, checkpoint_frames=3)
    features = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))["features"]
    table = pd.concat(pd.read_parquet(path) for path in sorted((output / "data").rglob("*.parquet")))
    for name, dtype in (("rlt_phase", "bool"), ("rlt_phase_id", "int64"), ("rlt_applied", "bool")):
        key = f"complementary_info.{name}"
        assert features[key]["dtype"] == dtype
        assert table[key].tolist() == [frame[key].item() for frame in source.frames]
