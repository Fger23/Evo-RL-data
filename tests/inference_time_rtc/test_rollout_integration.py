"""Exercise rollout recording without Linux keyboard or robot device imports.

The full production functions are compiled from their AST because importing the
record entry on Windows imports the Linux-only termios keyboard listener.
Dataset frame construction and the CLI/config parser remain real.
"""

import ast
import copy
import logging
import sys
import types
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from pprint import pformat
from types import SimpleNamespace

import draccus
import numpy as np
import pytest

from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.utils import build_dataset_frame
from lerobot.inference_time_rtc.configs import InferenceTimeRTCConfig, RTCRecoveryConfig, RTCRLTConfig
from lerobot.robots.config import RobotConfig
from lerobot.scripts.recording_remote_policy import RemotePolicyRecordConfig
from lerobot.teleoperators.config import TeleoperatorConfig
from lerobot.utils.recording_annotations import (
    infer_collector_policy_id,
    normalize_episode_success_label,
    resolve_collector_policy_id,
    resolve_episode_success_label,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "src/lerobot"


@RobotConfig.register_subclass("rtc_rollout_test_robot")
@dataclass
class RolloutRobotConfig(RobotConfig):
    pass


@TeleoperatorConfig.register_subclass("rtc_rollout_test_teleop")
@dataclass
class RolloutTeleopConfig(TeleoperatorConfig):
    pass


def load_definitions(relative, names, namespace, *, decorators=False, future_annotations=True):
    path = SOURCE / relative
    parsed = ast.parse(path.read_text(encoding="utf-8"))
    nodes = (
        [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
        if future_annotations
        else []
    )
    for node in parsed.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names:
            node = copy.deepcopy(node)
            if isinstance(node, ast.FunctionDef) and not decorators:
                node.decorator_list = []
            nodes.append(node)
    assert len(nodes) == len(names) + int(future_annotations)
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), "exec"), namespace)


@pytest.fixture
def config_module(monkeypatch):
    module = types.ModuleType("rtc_rollout_config_test")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.__dict__.update(
        dataclass=dataclass,
        field=field,
        fields=fields,
        Path=Path,
        RobotConfig=RobotConfig,
        TeleoperatorConfig=TeleoperatorConfig,
        PreTrainedConfig=PreTrainedConfig,
        RemotePolicyRecordConfig=RemotePolicyRecordConfig,
        InferenceTimeRTCConfig=InferenceTimeRTCConfig,
        RTCRecoveryConfig=RTCRecoveryConfig,
        RTCRLTConfig=RTCRLTConfig,
        parser=parser,
        sanity_check_bimanual_piper_pair=lambda *args: None,
        normalize_episode_success_label=normalize_episode_success_label,
    )
    load_definitions("scripts/recording_hil.py", {"ACPInferenceConfig"}, module.__dict__)
    load_definitions("scripts/lerobot_record.py", {"DatasetRecordConfig", "RecordConfig"}, module.__dict__)
    load_definitions(
        "inference_time_rtc/rollout.py",
        {"RTCRolloutConfig", "rollout"},
        module.__dict__,
        decorators=True,
        future_annotations=False,
    )
    return module


def cli_args():
    return [
        "--robot.type=rtc_rollout_test_robot",
        "--teleop.type=rtc_rollout_test_teleop",
        "--dataset.repo_id=piper/rollout_test",
        "--dataset.single_task=Move the card",
        "--dataset.storage=pkl",
        "--dataset.push_to_hub=false",
        "--dataset.fps=30",
        "--server_address=10.61.5.1:8110",
        "--pretrained_name_or_path=/nfs/policy/checkpoints/last/pretrained_model",
        "--policy_device=cuda:1",
        "--rtc.auto_delay=false",
        "--rtc.inference_delay=16",
        "--rtc.execution_horizon=20",
        "--rtc.queue_threshold=30",
        "--rtc.max_guidance_weight=10.0",
        "--rtc.observation_image_codec=lossless_png",
    ]


def test_rollout_cli_maps_to_exact_record_config(config_module, monkeypatch):
    cfg = draccus.parse(config_module.RTCRolloutConfig, args=cli_args())
    assert cfg.dataset.fps == 30
    assert cfg.remote_policy.backend == "rtc"
    assert cfg.remote_policy.server_address == "10.61.5.1:8110"
    assert cfg.remote_policy.policy_device == "cuda:1"
    assert cfg.remote_policy.rtc.inference_delay == 16
    assert cfg.remote_policy.rtc.observation_image_codec == "lossless_png"
    assert cfg.remote_policy.recovery.enabled
    assert cfg.remote_policy.recovery.prefix_steps == 6
    disabled = draccus.parse(config_module.RTCRolloutConfig, args=[*cli_args(), "--recovery.enabled=false"])
    assert not disabled.remote_policy.recovery.enabled
    calls = []
    monkeypatch.setattr(
        config_module, "human_inloop_record", lambda record_cfg: calls.append(record_cfg), raising=False
    )
    config_module.rollout(cfg)
    assert len(calls) == 1
    assert type(calls[0]) is config_module.RecordConfig
    assert calls[0].remote_policy is cfg.remote_policy
    assert calls[0].dataset.fps == 30
    assert calls[0].rlt_phase_toggle_key == "r"


@pytest.mark.parametrize("requested_sg_window", [0, 5, 7])
def test_rollout_cli_keeps_output_sg_for_rlt_and_keeps_png(config_module, requested_sg_window):
    cfg = draccus.parse(
        config_module.RTCRolloutConfig,
        args=[
            *cli_args(),
            f"--action_sg_window={requested_sg_window}",
            "--rlt.enabled=true",
            "--rlt.checkpoint=/nfs/policy/rlt/actor.pt",
            "--rlt.start_step=100",
            "--rlt.end_step=250",
        ],
    )
    assert cfg.remote_policy.rlt.enabled
    assert cfg.remote_policy.rlt.checkpoint == "/nfs/policy/rlt/actor.pt"
    assert cfg.remote_policy.rlt.start_step == 100
    assert cfg.remote_policy.rlt.end_step == 250
    assert cfg.remote_policy.action_sg_window == requested_sg_window
    assert cfg.remote_policy.rtc.observation_image_codec == "lossless_png"


def test_rollout_cli_keeps_requested_sg_when_rlt_disabled(config_module):
    cfg = draccus.parse(config_module.RTCRolloutConfig, args=[*cli_args(), "--action_sg_window=7"])
    assert not cfg.remote_policy.rlt.enabled
    assert cfg.remote_policy.action_sg_window == 7
    assert cfg.remote_policy.rtc.observation_image_codec == "lossless_png"


@pytest.mark.parametrize("option", ["--dataset.storage=lerobot", "--dataset.push_to_hub=true"])
def test_rollout_requires_local_pkl(config_module, option):
    key = option.split("=", 1)[0] + "="
    args = [arg for arg in cli_args() if not arg.startswith(key)] + [option]
    with pytest.raises(draccus.utils.ParsingError) as error:
        draccus.parse(config_module.RTCRolloutConfig, args=args)
    assert "dataset.storage=pkl" in str(error.value.__cause__)


class FakeTeleop:
    def __init__(self):
        self.manual_modes = []

    def set_manual_control(self, enabled):
        self.manual_modes.append(enabled)

    def get_action(self):
        return {"joint.pos": 99.0}


class FakeRobot:
    action_features = {"joint.pos": float}

    def __init__(self):
        self.commands = []

    def get_observation(self):
        return {"joint.pos": 5.0}

    def send_action(self, action):
        self.commands.append(dict(action))
        return {"joint.pos": min(float(action["joint.pos"]), 10.0)}


class FakeRemote:
    cfg = SimpleNamespace(backend="rtc")
    last_reservation = None

    def __init__(self, actions):
        self.actions = iter(actions)
        self.resets = 0
        self.pauses = 0
        self.get_calls = 0

    def reset(self):
        self.resets += 1

    def pause(self):
        self.pauses += 1

    def get_action(self, **kwargs):
        self.get_calls += 1
        return next(self.actions)


class FakeDataset:
    fps = 25
    num_episodes = 0
    features = {
        "observation.state": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "complementary_info.policy_action": {"dtype": "float32", "shape": (1,), "names": ["joint.pos"]},
        "complementary_info.policy_available": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["policy_available"],
        },
        "complementary_info.is_intervention": {
            "dtype": "float32",
            "shape": (1,),
            "names": ["is_intervention"],
        },
        "complementary_info.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "complementary_info.collector_policy_id": {
            "dtype": "string",
            "shape": (1,),
            "names": ["collector_policy_id"],
        },
    }

    def __init__(self, toggles, events):
        self.frames = []
        self.toggles = toggles
        self.events = events

    def add_frame(self, frame):
        self.frames.append(frame)
        count = len(self.frames)
        if count == len(self.toggles):
            self.events["exit_early"] = True
        else:
            self.events["toggle_intervention"] = self.toggles[count]


def run_loop(config_module, monkeypatch, actions, toggles):
    events = {"exit_early": False, "toggle_intervention": toggles[0]}
    robot = FakeRobot()
    teleop = FakeTeleop()
    remote = FakeRemote(actions)
    dataset = FakeDataset(toggles, events)
    clock = SimpleNamespace(now=0.0)
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

    logs = []
    monkeypatch.setattr(
        logging_utils, "log_action_event", lambda event, **values: logs.append((event, values))
    )
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
        single_task="Move the card",
    )
    return dataset, robot, teleop, remote, logs


def column(dataset, name):
    return [float(frame[name][0]) for frame in dataset.frames]


def test_waiting_policy_holds_observed_target_and_allows_takeover(config_module, monkeypatch):
    dataset, robot, teleop, remote, logs = run_loop(config_module, monkeypatch, [None], [False, True])
    assert robot.commands == [{"joint.pos": 5.0}, {"joint.pos": 99.0}]
    assert column(dataset, "action") == [5.0, 10.0]
    assert column(dataset, "complementary_info.policy_available") == [0.0, 0.0]
    assert column(dataset, "complementary_info.is_intervention") == [0.0, 1.0]
    assert remote.get_calls == 1 and remote.pauses == 1
    assert teleop.manual_modes == [False, True]
    assert logs[0][1]["holding"] is True
    assert logs[1][1]["sent_action"] == {"joint.pos": 10.0}


def test_release_waits_for_fresh_policy_and_records_executed_target(config_module, monkeypatch):
    actions = [None, None, {"joint.pos": 12.0}, {"joint.pos": 13.0}]
    dataset, robot, teleop, remote, _ = run_loop(
        config_module, monkeypatch, actions, [True, True, False, False, False]
    )
    assert column(dataset, "complementary_info.state") == [1.0, 2.0, 2.0, 2.0, 0.0]
    assert column(dataset, "complementary_info.policy_available") == [0.0, 0.0, 0.0, 1.0, 1.0]
    assert column(dataset, "action") == [10.0, 10.0, 10.0, 10.0, 10.0]
    assert column(dataset, "complementary_info.policy_action") == [0.0, 0.0, 0.0, 12.0, 13.0]
    assert robot.commands[1] == {"joint.pos": 10.0}
    assert remote.resets == 2 and remote.pauses == 1
    assert teleop.manual_modes == [False, True, False]


def test_takeover_key_remains_available_during_release_wait(config_module, monkeypatch):
    dataset, _, _, remote, _ = run_loop(config_module, monkeypatch, [None], [True, True, True])
    assert column(dataset, "complementary_info.state") == [1.0, 2.0, 1.0]
    assert column(dataset, "complementary_info.is_intervention") == [1.0, 0.0, 1.0]
    assert remote.get_calls == 1 and remote.pauses == 2


def test_partial_send_records_observed_value_for_omitted_channel(config_module, monkeypatch):
    monkeypatch.setattr(FakeRobot, "send_action", lambda self, action: {})
    dataset, _, _, _, logs = run_loop(config_module, monkeypatch, [{"joint.pos": 12.0}], [False])
    assert column(dataset, "action") == [5.0]
    assert column(dataset, "complementary_info.policy_action") == [12.0]
    assert logs[0][1]["sent_action"] == {}
    assert logs[0][1]["recorded_action"] == {"joint.pos": 5.0}


def test_cleanup_disconnects_partial_bimanual_connections_before_draining_storage():
    calls = []
    namespace = {"logging": logging, "log_say": lambda *args, **kwargs: None}
    load_definitions("scripts/lerobot_record.py", {"_finish_pkl_recording"}, namespace)
    robot = SimpleNamespace(
        is_connected=False,
        left_arm=SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("robot left")),
        right_arm=SimpleNamespace(is_connected=False),
    )
    teleop = SimpleNamespace(
        is_connected=False,
        left_arm=SimpleNamespace(is_connected=False),
        right_arm=SimpleNamespace(is_connected=True, disconnect=lambda: calls.append("teleop right")),
    )
    namespace["_finish_pkl_recording"](
        dataset=SimpleNamespace(root="raw", finalize=lambda: calls.append("finalize")),
        robot=robot,
        teleop=teleop,
        policy_sync_executor=None,
        remote_policy_client=SimpleNamespace(stop=lambda: calls.append("rtc stop")),
        listener=None,
        play_sounds=False,
    )
    assert calls == ["rtc stop", "robot left", "teleop right", "finalize"]


def test_record_prepares_rtc_before_hardware_and_pauses_before_episode_save(
    config_module, monkeypatch, tmp_path
):
    cfg = draccus.parse(config_module.RTCRolloutConfig, args=cli_args())
    cfg.dataset.num_episodes = 1
    cfg.log_file = str(tmp_path / "rollout.log")
    record_cfg = config_module.RecordConfig(
        **{item.name: getattr(cfg, item.name) for item in fields(config_module.RecordConfig)}
    )
    calls = []

    class Device:
        name = "fake_robot"
        action_features = FakeRobot.action_features
        observation_features = FakeRobot.action_features
        is_connected = False

        def __init__(self, name):
            self.label = name

        def connect(self):
            calls.append(f"connect {self.label}")
            self.is_connected = True

        def disconnect(self):
            calls.append(f"disconnect {self.label}")
            self.is_connected = False

    class Remote:
        policy_id = "fake_checkpoint"

        def __init__(self, **kwargs):
            pass

        def start(self):
            calls.append("rtc start")

        def pause(self):
            calls.append("rtc pause")

        def stop(self):
            calls.append("rtc stop")

    class Dataset:
        root = tmp_path
        num_episodes = 0

        def save_episode(self, **kwargs):
            calls.append("save episode")

        def finalize(self):
            calls.append("finalize")

    fake_module = types.ModuleType("lerobot.inference_time_rtc.recording_client")
    fake_module.RTCPolicyActionClient = Remote
    monkeypatch.setitem(sys.modules, fake_module.__name__, fake_module)
    from lerobot.inference_time_rtc import logging_utils

    monkeypatch.setattr(logging_utils, "log_actions_to_file", lambda path: nullcontext())
    features = {
        "action": FakeDataset.features["action"],
        "observation.state": FakeDataset.features["observation.state"],
    }
    namespace = {
        "logging": logging,
        "Path": Path,
        "asdict": asdict,
        "pformat": pformat,
        "nullcontext": nullcontext,
        "init_logging": lambda **kwargs: None,
        "make_robot_from_config": lambda config: Device("robot"),
        "make_teleoperator_from_config": lambda config: Device("teleop"),
        "make_default_processors": lambda: (None, None, None),
        "combine_feature_dicts": lambda *args: features.copy(),
        "aggregate_pipeline_dataset_features": lambda **kwargs: {},
        "create_initial_features": lambda **kwargs: {},
        "PklDataset": SimpleNamespace(create=lambda **kwargs: Dataset()),
        "sanity_check_dataset_name": lambda *args: None,
        "RemotePolicyActionClient": None,
        "infer_collector_policy_id": infer_collector_policy_id,
        "resolve_episode_success_label": resolve_episode_success_label,
        "ACTION": "action",
        "init_keyboard_listener": lambda **kwargs: (
            None,
            {"stop_recording": False, "rerecord_episode": False},
        ),
        "log_say": lambda *args, **kwargs: None,
        "record_loop": lambda **kwargs: calls.append("record loop"),
    }
    load_definitions(
        "scripts/lerobot_record.py",
        {"record", "_ensure_human_inloop_compatible_features", "_finish_pkl_recording"},
        namespace,
    )
    namespace["record"](record_cfg)
    assert calls.index("rtc start") < calls.index("connect robot")
    assert calls.index("rtc start") < calls.index("connect teleop")
    assert calls.index("record loop") < calls.index("rtc pause") < calls.index("save episode")
    assert calls.index("rtc stop") < calls.index("finalize")
