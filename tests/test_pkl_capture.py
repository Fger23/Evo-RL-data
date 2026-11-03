"""Synthetic-data checks; never connects to cameras, robots, SSH or SCP."""
import importlib.util
import json
import logging
import os
import subprocess
import sys
import tempfile
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


storage = load_module("lerobot.datasets.pkl_dataset", ROOT / "src/lerobot/datasets/pkl_dataset.py")
converter = load_module("pkl_converter_test", ROOT / "convert_pkl_dataset.py")


def features():
    return {
        **{f"observation.images.{name}": {"dtype": "video", "shape": (64, 64, 3),
                                        "names": ["height", "width", "channels"]}
           for name in ("left_fixed", "left_wrist", "right_wrist")},
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [f"j{i}" for i in range(14)]},
        "action": {"dtype": "float32", "shape": (14,), "names": [f"j{i}" for i in range(14)]},
        "complementary_info.policy_action": {"dtype": "float32", "shape": (14,), "names": [f"j{i}" for i in range(14)]},
        "complementary_info.is_intervention": {"dtype": "float32", "shape": (1,), "names": ["is_intervention"]},
        "complementary_info.state": {"dtype": "float32", "shape": (1,), "names": ["state"]},
        "complementary_info.collector_policy_id": {"dtype": "string", "shape": (1,), "names": ["collector_policy_id"]},
    }


def frame(value=0):
    return {
        **{f"observation.images.{name}": np.full((64, 64, 3), value, dtype=np.uint8)
           for name in ("left_fixed", "left_wrist", "right_wrist")},
        "observation.state": np.full(14, value, dtype=np.float32),
        "action": np.full(14, value + 1, dtype=np.float32),
        "complementary_info.policy_action": np.zeros(14, dtype=np.float32),
        "complementary_info.is_intervention": np.array([0], dtype=np.float32),
        "complementary_info.state": np.array([0], dtype=np.float32),
        "complementary_info.collector_policy_id": "human",
        "task": "synthetic task",
    }


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def writer(self, name="raw"):
        root = self.root / name
        root.mkdir()
        meta = SimpleNamespace(root=root, repo_id="local/synthetic", features=features(), fps=30,
                               info={"robot_type": "synthetic"}, _close_writer=lambda: None)
        # Tiny shards force rollover in a small, fast test.
        return storage.PklDataset(meta, queue_frames=2, shard_mb=0.002, sync_seconds=0.05)

    def completed(self):
        writer = self.writer()
        for index in range(3):
            writer.add_frame(frame(index))
        writer.save_episode(extra_episode_metadata={"episode_success": "success"})
        writer.finalize()
        return writer.root

    def test_snapshot_rollover_rerecord_and_metadata(self):
        writer = self.writer()
        sample = frame(7)
        writer.add_frame(sample)
        sample["observation.images.left_fixed"][:] = 99
        writer.add_frame(frame(8))
        writer.save_episode(extra_episode_metadata={"episode_success": "success"})
        writer.add_frame(frame(100))
        writer.clear_episode_buffer()
        writer.add_frame(frame(9))
        writer.save_episode(extra_episode_metadata={"episode_success": "failure"})
        writer.finalize()
        writer.finalize()  # idempotent
        _, info, episodes = converter.capture_info(writer.root)
        self.assertEqual(converter.validate_capture(info, episodes), 3)
        self.assertEqual(len(episodes), 2)
        first = list(converter.episode_records(*episodes[0]))
        self.assertEqual(int(first[0]["frame"]["observation.images.left_fixed"][0, 0, 0]), 7)
        self.assertGreater(len(episodes[0][1]["shards"]), 1)
        self.assertEqual(episodes[1][1]["metadata"]["episode_success"], "failure")
        self.assertTrue((writer.root / "episodes/attempt-000001/discarded.json").exists())

    def test_truncated_or_corrupt_shard_fails(self):
        root = self.completed()
        _, info, episodes = converter.capture_info(root)
        shard = episodes[0][0] / episodes[0][1]["shards"][0]
        content = bytearray(shard.read_bytes())
        content[-1] ^= 1
        shard.write_bytes(content)
        with self.assertRaisesRegex(ValueError, "CRC"):
            converter.validate_capture(info, episodes)
        shard.write_bytes(content[:-10])
        with self.assertRaisesRegex(ValueError, "Truncated"):
            converter.validate_capture(info, episodes)

    def test_uncommitted_episode_never_becomes_ready(self):
        writer = self.writer()
        writer.add_frame(frame())
        writer.finalize()
        self.assertTrue((writer.root / "episodes/attempt-000000/incomplete.json").exists())
        self.assertFalse(list(writer.root.rglob("ready.json")))
        with self.assertRaises(ValueError):
            converter.capture_info(writer.root)

    def test_converter_rejects_open_capture(self):
        writer = self.writer()
        try:
            with self.assertRaisesRegex(ValueError, "not closed"):
                converter.capture_info(writer.root)
        finally:
            writer.finalize()

    def test_writer_failure_is_propagated_without_hanging(self):
        writer = self.writer()
        with (
            patch.object(storage.pickle, "dumps", side_effect=OSError("simulated disk failure")),
            self.assertLogs(level=logging.ERROR),
        ):
            with suppress(RuntimeError):
                writer.add_frame(frame())
            writer._thread.join(timeout=5)
        self.assertFalse(writer._thread.is_alive())
        with self.assertRaisesRegex(RuntimeError, "writer failed"):
            writer.finalize()
        self.assertFalse((writer.root / "capture.closed.json").exists())

    def test_upload_failure_and_default_cli_configuration(self):
        output = self.root / "converted"
        output.mkdir()
        (output / "conversion.complete.json").write_text(json.dumps({"episodes": 1, "frames": 3}))
        with patch.object(converter, "verify_output"), patch.object(converter.subprocess, "run") as run:
            run.side_effect = [
                subprocess.CompletedProcess([], 0, stdout='{"status":"ready"}', stderr=""),
                subprocess.CalledProcessError(1, ["scp"]),
            ]
            with self.assertRaises(subprocess.CalledProcessError):
                converter.upload(output, "example.invalid", "test", "/data/datasets")
            self.assertEqual(run.call_count, 2)  # remote final verification/rename not invoked
        raw = self.completed()
        (self.root / "upload.json").write_text(json.dumps({
            "upload_host": "example.invalid", "upload_user": "test", "upload_dir": "/data/datasets",
        }))
        with patch.object(converter, "__file__", str(self.root / "convert_pkl_dataset.py")), \
                patch.object(converter, "convert", return_value=output), \
                patch.object(converter, "upload") as upload:
            with patch.object(sys, "argv", ["convert", "--input", str(raw)]):
                converter.main()
            upload.assert_called_once_with(output, "example.invalid", "test", "/data/datasets", 22)
            upload.reset_mock()
            with patch.object(sys, "argv", ["convert", "--input", str(raw), "--no-upload"]):
                converter.main()
            upload.assert_not_called()

    @unittest.skipUnless(os.environ.get("EVO_PKL_INTEGRATION") == "1", "requires original LeRobot environment")
    def test_original_writer_round_trip(self):
        # Slow test: uses the real PNG, video, Parquet and dataset metadata code.
        # No devices/network. Serial encoding makes the test portable on Windows.
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        writer = storage.PklDataset.create(
            repo_id="local/synthetic", fps=30, features=features(),
            root=self.root / "raw-integration", robot_type="synthetic",
            queue_frames=2, shard_mb=1,
        )
        for index in range(3):
            sample = frame(index)
            for key in sample:
                if key.startswith("observation.images."):
                    sample[key][:] = 30 + index * 35
            writer.add_frame(sample)
        writer.save_episode(extra_episode_metadata={"episode_success": "success"})
        writer.add_frame(frame(200))
        writer.clear_episode_buffer()
        for index in range(3, 6):
            sample = frame(index)
            for key in sample:
                if key.startswith("observation.images."):
                    sample[key][:] = 30 + index * 35
            writer.add_frame(sample)
        writer.save_episode(extra_episode_metadata={"episode_success": "failure"})
        writer.finalize()
        root = writer.root
        output = self.root / "converted"
        original_save = LeRobotDataset.save_episode

        def save_serial(dataset, *args, **kwargs):
            kwargs["parallel_encoding"] = False
            return original_save(dataset, *args, **kwargs)

        with patch.object(LeRobotDataset, "save_episode", save_serial):
            converter.convert(root, output)
        marker = json.loads((output / "conversion.complete.json").read_text())
        self.assertEqual(marker["frames"], 6)
        self.assertEqual(marker["episodes"], 2)
        self.assertEqual(len(list((output / "videos").rglob("*.mp4"))), 3)
        self.assertTrue(list((output / "data").rglob("*.parquet")))
        import pyarrow.parquet as pq
        table = pq.read_table(next((output / "data").rglob("*.parquet")))
        rows = table.to_pylist()
        self.assertEqual([row["observation.state"][0] for row in rows], list(range(6)))
        self.assertEqual([row["action"][0] for row in rows], list(range(1, 7)))
        self.assertEqual([row["complementary_info.collector_policy_id"] for row in rows], ["human"] * 6)
        np.testing.assert_allclose([row["timestamp"] for row in rows], [0, 1/30, 2/30] * 2)
        # Decode actual video PTS at the episode boundary using the original
        # metadata offsets. PyAV works without torchvision's removed VideoReader.
        import av

        from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

        restored = LeRobotDatasetMetadata("local/synthetic", root=output)
        for index in (2, 3):
            row = rows[index]
            self.assertEqual(row["frame_index"], index % 3)
            episode_index = row["episode_index"]
            episode = restored.episodes[episode_index]
            for key in restored.video_keys:
                timestamp = episode[f"videos/{key}/from_timestamp"] + row["timestamp"]
                path = output / restored.get_video_file_path(episode_index, key)
                with av.open(str(path)) as container:
                    decoded = list(container.decode(video=0))
                closest = min(decoded, key=lambda image: abs(float(image.pts * image.time_base) - timestamp))
                self.assertAlmostEqual(float(closest.pts * closest.time_base), timestamp, delta=1e-4)
                mean_pixel = closest.to_ndarray(format="rgb24").mean()
                self.assertAlmostEqual(mean_pixel, 30 + index * 35, delta=5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
