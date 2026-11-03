"""Exercise record's actual finally block without importing policies or hardware."""

import ast
import copy
import importlib.util
import logging
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]


def cleanup_functions(namespace):
    source = ROOT / "src/lerobot/scripts/lerobot_record.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == "_finish_pkl_recording")
    record = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "record")
    cleanup = next(node for node in record.body if isinstance(node, ast.Try)).finalbody
    exercise = ast.FunctionDef(
        name="exercise_finally",
        args=ast.arguments(posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
        body=copy.deepcopy(cleanup), decorator_list=[],
    )
    module = ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(helper), exercise], type_ignores=[]))
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["exercise_finally"]


class CleanupTests(unittest.TestCase):
    def environment(self, failures=None, storage="pkl"):
        failures = failures or {}
        events = []

        def operation(name):
            def invoke(*args, **kwargs):
                events.append(name)
                if name in failures:
                    raise failures[name]
            return invoke

        namespace = {
            "logging": logging,
            "cfg": SimpleNamespace(dataset=SimpleNamespace(storage=storage, push_to_hub=False), play_sounds=False),
            "dataset": SimpleNamespace(finalize=operation("finalize"), root="raw", repo_id="local/test"),
            "robot": SimpleNamespace(is_connected=True, disconnect=operation("robot")),
            "teleop": SimpleNamespace(is_connected=True, disconnect=operation("teleop")),
            "policy_sync_executor": SimpleNamespace(shutdown=operation("policy")),
            "remote_policy_client": SimpleNamespace(stop=operation("remote")),
            "listener": SimpleNamespace(stop=operation("listener")),
            "log_say": lambda message, *args, **kwargs: operation(message)(),
        }
        return namespace, events, cleanup_functions(namespace)

    def test_devices_disconnected_before_writer_drain(self):
        namespace, events, finish = self.environment()

        def verify_disconnected_before_drain():
            self.assertEqual(events, ["policy", "remote", "robot", "teleop", "listener"])
            events.append("finalize")

        namespace["dataset"].finalize = verify_disconnected_before_drain
        finish()
        self.assertEqual(events[-3:], ["finalize", "Stop recording", "Exiting"])

    def test_interrupt_and_disconnect_error_do_not_skip_other_cleanup(self):
        interruption = KeyboardInterrupt("stop during policy shutdown")
        _, events, finish = self.environment({"policy": interruption, "robot": OSError("disconnect failed")})
        with self.assertLogs(level=logging.ERROR), self.assertRaises(KeyboardInterrupt) as caught:
            finish()
        self.assertIs(caught.exception, interruption)
        self.assertEqual(events, ["policy", "remote", "robot", "teleop", "listener", "finalize", "Stop recording"])

    def test_interrupt_during_drain_leaves_hardware_already_released(self):
        interruption = KeyboardInterrupt("stop during disk drain")
        _, events, finish = self.environment({"finalize": interruption})
        with self.assertLogs(level=logging.ERROR), self.assertRaises(KeyboardInterrupt) as caught:
            finish()
        self.assertIs(caught.exception, interruption)
        self.assertEqual(events[:5], ["policy", "remote", "robot", "teleop", "listener"])
        self.assertIn("Stop recording", events)

    def test_startup_failure_without_dataset_still_releases_devices(self):
        namespace, events, finish = self.environment()
        namespace["dataset"] = None
        namespace["remote_policy_client"] = None
        finish()
        self.assertEqual(events, ["policy", "robot", "teleop", "listener", "Stop recording", "Exiting"])

    def test_existing_lerobot_mode_retains_finalize_before_disconnect(self):
        _, events, finish = self.environment(storage="lerobot")
        finish()
        self.assertEqual(events, ["Stop recording", "finalize", "policy", "remote", "robot", "teleop", "listener", "Exiting"])

    def test_invalid_codec_rejected_before_creating_capture(self):
        module_path = ROOT / "src/lerobot/datasets/pkl_dataset.py"
        spec = importlib.util.spec_from_file_location("pkl_storage_cleanup_test", module_path)
        storage = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(storage)
        dataset_module = types.ModuleType("lerobot.datasets.lerobot_dataset")
        dataset_module.VALID_VIDEO_CODECS = {"h264", "hevc", "libsvtav1"}
        metadata_create = Mock(side_effect=AssertionError("must validate before filesystem creation"))
        dataset_module.LeRobotDatasetMetadata = SimpleNamespace(create=metadata_create)
        utils_module = types.ModuleType("lerobot.datasets.utils")
        utils_module.validate_frame = Mock()
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "must-not-exist"
            with patch.dict(sys.modules, {
                "lerobot.datasets.lerobot_dataset": dataset_module,
                "lerobot.datasets.utils": utils_module,
            }), self.assertRaisesRegex(ValueError, "Invalid vcodec"):
                storage.PklDataset.create("local/test", 30, {}, root=target, vcodec="typo")
            metadata_create.assert_not_called()
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
