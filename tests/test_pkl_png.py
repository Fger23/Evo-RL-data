"""Lossless PNG capture and compatibility tests; no hardware or network."""
import json
import logging
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pkl_capture import converter, features, frame, storage


class PngCaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_writer(self, **kwargs):
        root = self.root / "capture"
        root.mkdir()
        meta = SimpleNamespace(root=root, repo_id="local/png-test", features=features(), fps=30,
                               info={"robot_type": "synthetic"}, _close_writer=lambda: None)
        return storage.PklDataset(meta, queue_frames=2, shard_mb=1, **kwargs)

    def sample(self, seed=0):
        result = frame(seed)
        rng = np.random.default_rng(seed)
        for key in result:
            if key.startswith("observation.images."):
                result[key] = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        return result

    def test_default_png_preserves_rgb_and_numeric_data(self):
        writer = self.make_writer()
        original = self.sample()
        writer.add_frame(original)
        expected = {key: value.copy() if isinstance(value, np.ndarray) else value for key, value in original.items()}
        original["observation.images.left_fixed"][:] = 0
        writer.save_episode(extra_episode_metadata={"episode_success": "success"})
        writer.finalize()
        _, info, episodes = converter.capture_info(writer.root)
        self.assertEqual(info["format"], "evo-pkl-v2")
        self.assertEqual(info["image_encoding"], "png")
        self.assertEqual(info["png_compression"], 1)
        self.assertEqual(converter.validate_capture(info, episodes), 1)
        packet = next(converter.episode_records(*episodes[0]))
        for key, value in expected.items():
            actual = packet["frame"][key]
            if key.startswith("observation.images."):
                self.assertIsInstance(actual, bytes)
                np.testing.assert_array_equal(storage.decode_png_rgb(actual, value.shape), value)
            elif isinstance(value, np.ndarray):
                np.testing.assert_array_equal(actual, value)
            else:
                self.assertEqual(actual, value)

    def test_invalid_compression_options_rejected(self):
        for args in [("jpeg", 1, 3), ("png", -1, 3), ("png", 10, 3), ("png", 1, 0)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                storage.PklDataset._validate_encoding(*args)

    def test_background_encoder_failure_reaches_caller(self):
        writer = self.make_writer()
        with (patch.object(storage, "encode_png_rgb", side_effect=RuntimeError("simulated encoder failure")),
              self.assertLogs(level=logging.ERROR)):
            writer.add_frame(self.sample())
            with self.assertRaisesRegex(RuntimeError, "writer failed"):
                writer.finalize()
        self.assertFalse(writer._thread.is_alive())
        self.assertFalse((writer.root / "capture.closed.json").exists())

    def test_bad_png_shape_rejected(self):
        image = self.sample()["observation.images.left_fixed"]
        encoded = storage.encode_png_rgb(image)
        with self.assertRaisesRegex(ValueError, "shape"):
            storage.decode_png_rgb(encoded, (480, 640, 3))
        with self.assertRaises(ValueError):
            storage.decode_png_rgb(encoded[:-30], image.shape)

    def test_legacy_v1_without_encoding_header_still_valid(self):
        writer = self.make_writer(image_encoding="raw")
        writer.add_frame(self.sample())
        writer.save_episode()
        writer.finalize()
        path = writer.root / "capture.pkl"
        with path.open("rb") as stream:
            info = pickle.load(stream)
        for key in ("image_encoding", "image_color_mode", "png_compression"):
            info.pop(key)
        path.write_bytes(pickle.dumps(info, protocol=5))
        _, info, episodes = converter.capture_info(writer.root)
        self.assertEqual(converter.validate_capture(info, episodes), 1)

    @unittest.skipUnless(os.environ.get("EVO_PKL_INTEGRATION") == "1", "requires original LeRobot environment")
    def test_png_converts_to_original_dataset_without_reencoding_png(self):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        writer = storage.PklDataset.create(
            repo_id="local/png-test", fps=30, features=features(), root=self.root / "raw",
            robot_type="synthetic", queue_frames=2,
        )
        for index in range(4):
            writer.add_frame(self.sample(index))
            if index in (1, 3):
                writer.save_episode(extra_episode_metadata={"episode_success": "success"})
        writer.finalize()
        original_save = LeRobotDataset.save_episode

        def serial_save(dataset, *args, **kwargs):
            return original_save(dataset, *args, parallel_encoding=False, **kwargs)

        output = self.root / "converted"
        with (patch.object(LeRobotDataset, "save_episode", serial_save),
              patch.object(LeRobotDataset, "_save_image", side_effect=AssertionError("PNG must not be re-encoded"))):
            converter.convert(writer.root, output)
        marker = json.loads((output / "conversion.complete.json").read_text())
        self.assertEqual(marker["frames"], 4)
        self.assertEqual(marker["episodes"], 2)
        self.assertEqual(len(list((output / "videos").rglob("*.mp4"))), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
