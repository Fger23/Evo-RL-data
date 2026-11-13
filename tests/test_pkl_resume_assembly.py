"""Native-format and crash-reentry checks for immutable package assembly."""
import errno
import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow.parquet as pq

import resume_assembly as assembly


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class SimulatedCrash(BaseException):
    pass


class AssemblyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        cls.temporary = tempfile.TemporaryDirectory(prefix="assembly-fixtures-")
        cls.fixture = Path(cls.temporary.name)
        cls.features = {
            "observation.state": {"dtype": "float32", "shape": (2,), "names": None},
            "action": {"dtype": "float32", "shape": (2,), "names": None},
            "observation.images.cam": {"dtype": "video", "shape": (64, 64, 3), "names": None},
        }
        cls.packages = []
        for package_index, episodes in enumerate((2, 1)):
            root = cls.fixture / f"package-{package_index}"
            dataset = LeRobotDataset.create(
                repo_id="local/resume-assembly", fps=30, features=cls.features,
                root=root, robot_type="test", vcodec="libsvtav1",
                image_writer_processes=0, image_writer_threads=0,
            )
            try:
                for episode in range(episodes):
                    for frame in range(3):
                        task = ["alpha", "beta"][(package_index + episode + frame) % 2]
                        dataset.add_frame({
                            "observation.state": np.array([package_index * 10 + episode, frame], np.float32),
                            "action": np.array([frame, frame + 1], np.float32),
                            "observation.images.cam": np.full((64, 64, 3),
                                                              frame * 40 + episode * 3, np.uint8),
                            "task": task,
                        })
                    dataset.save_episode(parallel_encoding=False,
                                         extra_episode_metadata={"episode_success": episode == 0,
                                                                 "custom_annotation": "preserved"})
            finally:
                dataset.stop_image_writer()
                dataset.finalize()
            # The checkpoint published by the main converter must not become output data.
            (root / ".episode.complete.json").write_text('{"private_checkpoint": true}')
            cls.packages.append(root)
        cls.info = {"fps": 30, "features": cls.features,
                    "robot_type": "test", "vcodec": "libsvtav1"}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix="assembly-test-")
        self.root = Path(self.workspace.name)
        self.output = self.root / "dataset"
        self.checkpoint = self.root / "checkpoint"

    def tearDown(self):
        self.workspace.cleanup()

    def assemble(self):
        return assembly.assemble_episode_packages(self.packages, self.output, self.info,
                                                  checkpoint_dir=self.checkpoint)

    def test_native_multiepisode_prefix_global_indices_tasks_stats_and_videos(self):
        from convert_pkl_dataset import verify_output
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.datasets.utils import load_stats, load_tasks

        self.assemble()
        verify_output(self.output, 3, 9)
        dataset = LeRobotDataset("local/resume-assembly", root=self.output, video_backend="pyav")
        self.assertEqual(len(dataset), 9)
        rows = dataset.hf_dataset.with_format(None)[:]
        self.assertEqual(rows["index"], list(range(9)))
        self.assertEqual(rows["episode_index"], [0] * 3 + [1] * 3 + [2] * 3)
        tasks = load_tasks(self.output)
        by_index = {int(row["task_index"]): task for task, row in tasks.iterrows()}
        self.assertEqual([by_index[value] for value in rows["task_index"]],
                         ["alpha", "beta", "alpha", "beta", "alpha", "beta",
                          "beta", "alpha", "beta"])
        stats = load_stats(self.output)
        np.testing.assert_allclose(stats["index"]["mean"], [4])
        np.testing.assert_allclose(stats["index"]["min"], [0])
        np.testing.assert_allclose(stats["index"]["max"], [8])
        np.testing.assert_allclose(stats["episode_index"]["mean"], [1])
        meta = LeRobotDatasetMetadata("local/resume-assembly", root=self.output)
        expected = [True, False, True]
        for index in range(3):
            row = meta.episodes[index]
            self.assertEqual(row["episode_success"], expected[index])
            self.assertEqual(row["custom_annotation"], "preserved")
            self.assertEqual(row["dataset_from_index"], index * 3)
            self.assertEqual(row["dataset_to_index"], index * 3 + 3)
            metadata = pq.ParquetFile(self.output / f"meta/episodes/chunk-000/file-{index:03d}.parquet").read().to_pylist()[0]
            self.assertEqual(metadata["stats/index/mean"], [index * 3 + 1.0])
        # Two prefix episodes share one original MP4. The third episode owns a second MP4.
        video_paths = list((self.output / "videos").rglob("*.mp4"))
        self.assertEqual(len(video_paths), 2)
        originals = [path for package in self.packages for path in (package / "videos").rglob("*.mp4")]
        self.assertCountEqual([digest(p) for p in video_paths], [digest(p) for p in originals])
        self.assertFalse((self.output / ".episode.complete.json").exists())
        source_schema = pq.ParquetFile(next((self.packages[0] / "data").rglob("*.parquet"))).schema_arrow
        result_schema = pq.ParquetFile(self.output / "data/chunk-000/file-000.parquet").schema_arrow
        self.assertEqual(source_schema.metadata, result_schema.metadata)

    def test_idempotent_reentry_reuses_all_artifacts(self):
        self.assemble()
        before = {path.relative_to(self.output).as_posix():
                  (path.stat().st_ino, path.stat().st_mtime_ns, digest(path))
                  for path in self.output.rglob("*") if path.is_file()}
        self.assemble()
        after = {path.relative_to(self.output).as_posix():
                 (path.stat().st_ino, path.stat().st_mtime_ns, digest(path))
                 for path in self.output.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_crash_between_artifact_publish_and_checkpoint_recovers(self):
        real_replace = os.replace
        crashed = False

        def interrupted_replace(source, target):
            nonlocal crashed
            real_replace(source, target)
            if not crashed and Path(target).suffix == ".parquet" and "data" in Path(target).parts:
                crashed = True
                raise SimulatedCrash()

        with patch.object(assembly.os, "replace", side_effect=interrupted_replace), self.assertRaises(SimulatedCrash):
            self.assemble()
        published = self.output / "data/chunk-000/file-000.parquet"
        inode = published.stat().st_ino
        self.assemble()
        self.assertEqual(published.stat().st_ino, inode)
        self.assertFalse(any(path.name.startswith(".assembly-") for path in self.output.rglob("*")))
        self.assertTrue(json.loads((self.checkpoint / assembly.STATE_NAME).read_text())["complete"])

    def test_missing_or_corrupt_owned_artifact_is_rebuilt(self):
        self.assemble()
        video = next((self.output / "videos").rglob("*.mp4"))
        wanted = digest(video)
        video.unlink()  # Break the hardlink before corrupting only the assembled copy.
        video.write_bytes(b"broken")
        (self.output / "data/chunk-000/file-001.parquet").unlink()
        self.assemble()
        self.assertEqual(digest(video), wanted)
        self.assertTrue((self.output / "data/chunk-000/file-001.parquet").is_file())

    def test_copy_fallback_preserves_video_bytes(self):
        with patch.object(assembly.os, "link", side_effect=OSError(errno.EXDEV, "different filesystem")):
            self.assemble()
        originals = [p for root in self.packages for p in (root / "videos").rglob("*.mp4")]
        copied = list((self.output / "videos").rglob("*.mp4"))
        self.assertCountEqual([digest(p) for p in originals], [digest(p) for p in copied])
        self.assertTrue(all(all(p.stat().st_ino != source.stat().st_ino for source in originals)
                            for p in copied))

    def test_unowned_files_and_symlinks_are_rejected(self):
        self.output.mkdir()
        existing = self.output / "personal.txt"
        existing.write_text("keep")
        with self.assertRaisesRegex(ValueError, "nonempty"):
            self.assemble()
        self.assertEqual(existing.read_text(), "keep")
        existing.unlink()
        if os.name != "posix":
            return
        linked = self.root / "package-link"
        linked.symlink_to(self.packages[0], target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "Symlink"):
            assembly.assemble_episode_packages([linked], self.output, self.info,
                                               checkpoint_dir=self.checkpoint)


if __name__ == "__main__":
    unittest.main()
