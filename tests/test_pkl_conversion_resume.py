"""Conversion crash recovery with real native datasets; no hardware/network.

Set EVO_PKL_INTEGRATION=1 in the original LeRobot environment. Failures occur
at persistent transaction boundaries, including one real SIGKILL of a writer.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))

from test_pkl_capture import features, frame, storage  # noqa: E402

import convert_pkl_dataset as converter  # noqa: E402
import resumable_conversion as engine  # noqa: E402
from resume_common import atomic_json  # noqa: E402


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    return {path.relative_to(root).as_posix():
            (path.stat().st_ino, path.stat().st_mtime_ns, digest(path))
            for path in root.rglob("*") if path.is_file()}


class DurableStateTests(unittest.TestCase):
    def test_stale_partial_file_does_not_block_atomic_state_updates(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            old = root / "state.json.partial"
            old.write_bytes(b"incomplete prior writer")
            atomic_json(root / "state.json", {"done": 1})
            atomic_json(root / "state.json", {"done": 2})
            self.assertEqual(json.loads((root / "state.json").read_text()), {"done": 2})
            self.assertEqual(old.read_bytes(), b"incomplete prior writer")


@unittest.skipUnless(os.environ.get("EVO_PKL_INTEGRATION") == "1",
                     "requires original LeRobot Python environment")
class ConversionResumeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "converted"
        self.progress = patch.object(converter, "PROGRESS_ENABLED", False)
        self.progress.start()

    def tearDown(self):
        self.progress.stop()
        self.temporary.cleanup()

    def capture(self, lengths=(3, 2, 4)):
        self.lengths = tuple(lengths)
        self.raw = self.root / "capture"
        self.expected = []
        writer = storage.PklDataset.create(
            repo_id="local/resume-test", fps=30, features=features(),
            root=self.raw, robot_type="synthetic", queue_frames=2,
        )
        try:
            for episode, length in enumerate(lengths):
                rows = []
                for index in range(length):
                    sample = frame(episode * 20 + index)
                    sample["task"] = "pick" if (index + episode) % 2 == 0 else "bind"
                    sample["complementary_info.collector_policy_id"] = f"operator-{episode}"
                    rows.append(copy.deepcopy(sample))
                    writer.add_frame(sample)
                self.expected.append(rows)
                writer.save_episode(extra_episode_metadata={
                    "episode_success": "failure" if episode == 1 else "success",
                    "operator_note": f"episode-{episode}",
                })
        finally:
            writer.finalize()
        return self.raw

    def convert(self):
        return converter.convert(self.raw, self.output, checkpoint_frames=2)

    @property
    def workspace(self):
        return self.output.with_name(self.output.name + ".resume")

    def assert_dataset(self):
        import av
        import pyarrow.parquet as pq

        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

        total = sum(self.lengths)
        converter.verify_output(self.output, len(self.lengths), total)
        meta = LeRobotDatasetMetadata("local/resume-test", root=self.output)
        tasks = {int(row["task_index"]): task for task, row in meta.tasks.iterrows()}
        global_index = 0
        for episode, length in enumerate(self.lengths):
            metadata = meta.episodes[episode]
            self.assertEqual(metadata["operator_note"], f"episode-{episode}")
            self.assertEqual(metadata["episode_success"], "failure" if episode == 1 else "success")
            self.assertEqual(metadata["dataset_from_index"], global_index)
            self.assertEqual(metadata["dataset_to_index"], global_index + length)
            rows = [row for row in pq.ParquetFile(self.output / meta.get_data_file_path(episode)).read().to_pylist()
                    if row["episode_index"] == episode]
            self.assertEqual(len(rows), length)
            for index, (row, expected) in enumerate(zip(rows, self.expected[episode], strict=True)):
                self.assertEqual(row["index"], global_index)
                self.assertEqual(row["frame_index"], index)
                self.assertAlmostEqual(row["timestamp"], index / 30, places=6)
                self.assertEqual(tasks[row["task_index"]], expected["task"])
                for key in ["observation.state", "action", "complementary_info.policy_action",
                            "complementary_info.is_intervention", "complementary_info.state"]:
                    np.testing.assert_array_equal(np.asarray(row[key]).reshape(-1), expected[key].reshape(-1))
                self.assertEqual(row["complementary_info.collector_policy_id"], f"operator-{episode}")
                global_index += 1
            for key in meta.video_keys:
                with av.open(str(self.output / meta.get_video_file_path(episode, key))) as container:
                    decoded = list(container.decode(video=0))
                start = float(metadata[f"videos/{key}/from_timestamp"])
                selected = [image for image in decoded
                            if start - 1e-5 <= float(image.pts * image.time_base)
                            < start + length / 30 - 1e-5]
                self.assertEqual(len(selected), length)
                for index, image in enumerate(selected):
                    self.assertAlmostEqual(float(image.pts * image.time_base) - start, index / 30, places=5)
        dataset = LeRobotDataset("local/resume-test", root=self.output, video_backend="pyav")
        self.assertEqual(len(dataset), total)
        self.assertEqual(dataset.hf_dataset.with_format(None)[:]["index"], list(range(total)))

    def _interrupt_after_prepared_chunk(self):
        original = engine._validate_packet

        def interrupt(packet, info, index, api):
            if index == 2:
                raise KeyboardInterrupt()
            return original(packet, info, index, api)

        with patch.object(engine, "_validate_packet", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.convert()

    def test_prepared_frames_resume_from_committed_byte_offset(self):
        self.capture((5,))
        self._interrupt_after_prepared_chunk()
        chunk = self.workspace / "episodes/episode-000000/prepared/chunk-000000"
        committed = snapshot(chunk)
        marker = json.loads((chunk / "complete.json").read_text())
        self.assertEqual(marker["frames"], 2)
        self.assertGreater(marker["next_cursor"]["offset"], 8)
        validated, cursors = [], []
        original_validate, original_packets = engine._validate_packet, engine.packets_from
        original_build = engine.build_episode_package

        def validate(packet, info, index, api):
            validated.append(index)
            return original_validate(packet, info, index, api)

        def packets(folder, descriptor, cursor=None):
            cursors.append(cursor)
            return original_packets(folder, descriptor, cursor)

        def build(*args, **kwargs):
            self.assertEqual(snapshot(chunk), committed)
            return original_build(*args, **kwargs)

        with (patch.object(engine, "_validate_packet", side_effect=validate),
              patch.object(engine, "packets_from", side_effect=packets),
              patch.object(engine, "build_episode_package", side_effect=build)):
            self.convert()
        self.assertEqual(validated, [2, 3, 4])
        self.assertEqual(cursors, [marker["next_cursor"]])
        self.assert_dataset()

    def test_committed_video_segment_is_not_encoded_again(self):
        self.capture((5,))
        original_publish = engine._publish_video
        first = None

        def interrupt(path, *args, **kwargs):
            nonlocal first
            result = original_publish(path, *args, **kwargs)
            if first is None and "encoded" in Path(path).parts:
                first = Path(path)
                raise KeyboardInterrupt()
            return result

        with patch.object(engine, "_publish_video", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.convert()
        self.assertIsNotNone(first)
        committed = (first.stat().st_mtime_ns, digest(first))
        from lerobot.datasets import video_utils
        original_encode = video_utils.encode_video_frames
        encoded = []
        original_build = engine.build_episode_package

        def encode(images, *args, **kwargs):
            encoded.append(Path(images))
            return original_encode(images, *args, **kwargs)

        def build(*args, **kwargs):
            self.assertEqual((first.stat().st_mtime_ns, digest(first)), committed)
            return original_build(*args, **kwargs)

        with (patch.object(video_utils, "encode_video_frames", side_effect=encode),
              patch.object(engine, "build_episode_package", side_effect=build)):
            self.convert()
        first_images = self.workspace / "episodes/episode-000000/prepared/chunk-000000" / first.stem
        self.assertNotIn(first_images, encoded)
        self.assert_dataset()

    @unittest.skipUnless(os.name == "posix", "requires SIGKILL and POSIX file locking")
    def test_sigkill_after_episode_commit_releases_lock_and_reuses_video(self):
        self.capture()
        signal = self.root / "committed.signal"
        log = self.root / "child.log"
        code = textwrap.dedent(f"""
            import sys,time
            from pathlib import Path
            sys.path.insert(0, {str(ROOT)!r})
            sys.path.insert(0, {str(ROOT / 'src')!r})
            import convert_pkl_dataset as converter
            import resumable_conversion as engine
            converter.PROGRESS_ENABLED=False
            original=engine.build_episode_package
            def block(*args, **kwargs):
                result=original(*args, **kwargs)
                if args[2]['episode_index']==0:
                    Path({str(signal)!r}).write_text('committed')
                    while True:time.sleep(0.1)
                return result
            engine.build_episode_package=block
            converter.convert({str(self.raw)!r},{str(self.output)!r},checkpoint_frames=2)
        """)
        process = None
        try:
            with log.open("wb") as stream:
                process = subprocess.Popen([sys.executable, "-c", code], cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT)
                deadline = time.monotonic() + 90
                while not signal.exists() and process.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.05)
                self.assertTrue(signal.exists(), log.read_text(errors="replace")[-6000:])
                with self.assertRaisesRegex(RuntimeError, "Another conversion/upload"):
                    self.convert()
                process.kill()
                process.wait(timeout=15)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=15)
        package = self.workspace / "packages/episode-000000"
        original_package = snapshot(package)
        original_build = engine.build_episode_package
        rebuilt = []

        def build(*args, **kwargs):
            rebuilt.append(args[2]["episode_index"])
            return original_build(*args, **kwargs)

        with patch.object(engine, "build_episode_package", side_effect=build):
            self.convert()
        self.assertEqual(rebuilt, [1, 2])
        self.assertEqual(snapshot(package), original_package)
        self.assert_dataset()

    def test_crash_after_completion_marker_before_directory_publish_recovers(self):
        self.capture((3,))
        original = engine.atomic_json

        def interrupt(path, value):
            original(path, value)
            if Path(path).name == "conversion.complete.json":
                raise KeyboardInterrupt()

        with patch.object(engine, "atomic_json", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.convert()
        marker = self.workspace / "dataset/conversion.complete.json"
        self.assertTrue(marker.is_file())
        self.assertFalse(self.output.exists())
        with patch.object(engine, "prepare_episode", side_effect=AssertionError("completed work must be reused")):
            self.convert()
        self.assert_dataset()

    def test_uncommitted_completion_marker_temporary_file_is_recovered(self):
        self.capture((3,))
        original = engine.atomic_json
        orphan = None

        def interrupt(path, value):
            nonlocal orphan
            if Path(path).name == "conversion.complete.json":
                # This is the on-disk state left when SIGKILL occurs during an
                # atomic write, before its finally block can unlink the file.
                orphan = Path(path).with_name(f".conversion.complete.json.{uuid.uuid4().hex}.partial")
                orphan.write_bytes(b'{"episodes":')
                raise KeyboardInterrupt()
            return original(path, value)

        with patch.object(engine, "atomic_json", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.convert()
        self.assertTrue(orphan.is_file())
        self.assertFalse(self.output.exists())
        with patch.object(engine, "prepare_episode", side_effect=AssertionError("completed packages must be reused")):
            self.convert()
        self.assertFalse(any(path.name.startswith(".conversion.complete.json.")
                             for path in self.output.rglob("*")))
        self.assert_dataset()

    def test_crash_after_output_rename_and_repeat_command_are_idempotent(self):
        self.capture((3,))
        original = engine.atomic_json

        def interrupt(path, value):
            if Path(path).name == "published.json":
                self.assertTrue(self.output.exists())
                raise KeyboardInterrupt()
            return original(path, value)

        with patch.object(engine, "atomic_json", side_effect=interrupt), self.assertRaises(KeyboardInterrupt):
            self.convert()
        before = snapshot(self.output)
        with patch.object(engine, "prepare_episode", side_effect=AssertionError("completed work must be reused")):
            self.convert()
            self.convert()
        self.assertEqual(snapshot(self.output), before)
        self.assert_dataset()

    def test_replaced_source_cannot_reuse_prepared_checkpoints(self):
        self.capture((5,))
        self._interrupt_after_prepared_chunk()
        original_state = snapshot(self.workspace)
        header = self.raw / "capture.pkl"
        info = pickle.loads(header.read_bytes())
        info["repo_id"] = "local/different-source"
        header.write_bytes(pickle.dumps(info, protocol=5))
        with self.assertRaisesRegex(ValueError, "another source"):
            self.convert()
        self.assertEqual(snapshot(self.workspace), original_state)
        self.assertFalse(self.output.exists())

    def test_completed_output_is_bound_to_source_and_content(self):
        self.capture((3,))
        self.convert()
        stats = self.output / "meta/stats.json"
        data = json.loads(stats.read_text())
        data["index"]["mean"] = [999]
        stats.write_text(json.dumps(data))
        with self.assertRaises((ValueError, RuntimeError)):
            self.convert()

    def _write_legacy_native_dataset(self, destination):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        _, info, episodes = converter.capture_info(self.raw)
        dataset = LeRobotDataset.create(
            repo_id=info["repo_id"], fps=30, features=copy.deepcopy(info["features"]),
            root=destination, robot_type="synthetic", image_writer_threads=0,
            image_writer_processes=0, batch_encoding_size=1, vcodec=info["vcodec"],
        )
        try:
            for folder, descriptor in episodes:
                for packet in converter.episode_records(folder, descriptor):
                    sample = dict(packet["frame"])
                    for key, feature in info["features"].items():
                        if feature["dtype"] in {"image", "video"}:
                            sample[key] = storage.decode_png_rgb(sample[key], feature["shape"])
                    dataset.add_frame(sample)
                for key, feature in dataset.features.items():
                    if (key not in converter.DEFAULT_FIELDS and tuple(feature["shape"]) == (1,)
                            and feature["dtype"] not in {"string", "image", "video"}):
                        dataset.episode_buffer[key] = [np.asarray(value).reshape(())[()]
                                                       for value in dataset.episode_buffer[key]]
                dataset.save_episode(parallel_encoding=False, extra_episode_metadata=descriptor["metadata"])
        finally:
            dataset.stop_image_writer()
            dataset.finalize()

    def test_legacy_broken_parquet_recovers_without_reencoding_complete_videos(self):
        self.capture()
        legacy = self.output.with_name(self.output.name + ".converting")
        self._write_legacy_native_dataset(legacy)
        for section in ["data", "meta/episodes"]:
            for path in (legacy / section).rglob("*.parquet"):
                path.write_bytes(path.read_bytes()[:-8])
        (legacy / ".convert-resume.json").write_text(json.dumps({
            "source": str(self.raw), "done": len(self.lengths),
            "frames": sum(self.lengths), "episodes_total": len(self.lengths),
        }))
        original = snapshot(legacy)
        from lerobot.datasets import video_utils
        with patch.object(video_utils, "encode_video_frames",
                          side_effect=AssertionError("Legacy complete MP4s must be remuxed, not re-encoded")):
            self.convert()
        self.assertEqual(snapshot(legacy), original)
        self.assert_dataset()

    def test_legacy_complete_dataset_is_validated_without_reconversion(self):
        self.capture()
        self._write_legacy_native_dataset(self.output)
        (self.output / "conversion.complete.json").write_text(json.dumps({
            "episodes": len(self.lengths), "frames": sum(self.lengths), "fps": 30,
        }))
        original = snapshot(self.output)
        with patch.object(engine, "prepare_episode", side_effect=AssertionError("legacy completed data must be reused")):
            self.convert()
        self.assertEqual(snapshot(self.output), original)
        self.assert_dataset()


if __name__ == "__main__":
    unittest.main()
