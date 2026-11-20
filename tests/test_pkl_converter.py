"""Converter backpressure and upload publication tests; no SSH/SCP execution."""
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
spec = importlib.util.spec_from_file_location("pkl_converter_focused", ROOT / "convert_pkl_dataset.py")
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


class ConverterTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def output(self):
        output = self.root / "local-dataset"
        (output / "data").mkdir(parents=True)
        (output / "data/example.bin").write_bytes(b"synthetic dataset content")
        (output / "conversion.complete.json").write_text(json.dumps({"episodes": 1, "frames": 61}))
        return output, converter.checksum_manifest(output)

    def remote_helper(self, script, manifest, *arguments, check=True):
        # Execute exactly the remote helper locally, with optimization enabled.
        # Assertions must not be relied on for any verification or publication.
        return subprocess.run(
            [sys.executable, "-O", "-c", script, *map(str, arguments)],
            input=json.dumps(manifest), text=True, capture_output=True, check=check,
        )

    def test_conversion_commits_bounded_frame_groups(self):
        import pickle
        import zlib

        import numpy as np

        from lerobot.datasets.pkl_dataset import MAGIC, RECORD_HEADER
        from resumable_conversion import prepare_episode
        folder = self.root / "raw-episode"
        folder.mkdir()
        with (folder / "part-000000.pkl").open("wb") as stream:
            stream.write(MAGIC)
            for index in range(61):
                packet = {"frame_index": index, "timestamp": index / 30,
                          "frame": {"state": np.array([index], dtype=np.float32), "task": "bounded"}}
                body = pickle.dumps(packet, protocol=5)
                stream.write(RECORD_HEADER.pack(len(body), zlib.crc32(body)))
                stream.write(body)
        descriptor = {"episode_index": 0, "frames": 61, "shards": ["part-000000.pkl"]}
        info = {"fps": 30, "features": {"state": {"dtype": "float32", "shape": (1,)}}}
        api = SimpleNamespace(DEFAULT_FIELDS=set())
        updates = []
        chunks = prepare_episode(folder, descriptor, info, self.root / "episode", "source-id", 30,
                                 api, SimpleNamespace(update=updates.append))
        self.assertEqual([marker["frames"] for _, marker in chunks], [30, 30, 1])
        self.assertEqual([marker["start"] for _, marker in chunks], [0, 30, 60])
        self.assertEqual(sum(updates), 61)
        for path, marker in chunks:
            with (path / "frames.pkl").open("rb") as stream:
                self.assertEqual(len(pickle.load(stream)), marker["frames"])

    def test_remote_publish_and_retry_verify_fresh_manifest_under_optimization(self):
        local, manifest = self.output()
        parent = self.root / "remote"
        temporary = parent / "dataset.uploading-first"
        final = parent / "dataset"
        prepared = self.remote_helper(converter.REMOTE_PREPARE_SCRIPT, manifest, parent, temporary, final)
        self.assertEqual(json.loads(prepared.stdout)["status"], "ready")
        shutil.copytree(local, temporary)
        published = self.remote_helper(converter.REMOTE_PUBLISH_SCRIPT, manifest, temporary, final)
        self.assertEqual(json.loads(published.stdout)["status"], "complete")
        self.assertTrue(final.is_dir())
        self.assertFalse(temporary.exists())
        retried = self.remote_helper(converter.REMOTE_PREPARE_SCRIPT, manifest, parent, parent / "retry", final)
        self.assertEqual(json.loads(retried.stdout)["status"], "already_complete")
        # A self-consistent old server manifest is insufficient: it must match
        # the fresh desired local manifest sent over stdin.
        (local / "data/example.bin").write_bytes(b"new desired content")
        new_manifest = converter.checksum_manifest(local)
        mismatch = self.remote_helper(converter.REMOTE_PREPARE_SCRIPT, new_manifest, parent, parent / "retry", final, check=False)
        self.assertNotEqual(mismatch.returncode, 0)
        self.assertIn("fresh local manifest", mismatch.stderr)

    def test_hash_mismatch_does_not_publish_under_optimization(self):
        local, manifest = self.output()
        parent = self.root / "remote"
        parent.mkdir()
        temporary, final = parent / "uploading", parent / "final"
        shutil.copytree(local, temporary)
        (temporary / "data/example.bin").write_bytes(b"corruption")
        failed = self.remote_helper(converter.REMOTE_PUBLISH_SCRIPT, manifest, temporary, final, check=False)
        self.assertNotEqual(failed.returncode, 0)
        self.assertIn("SHA256 mismatch", failed.stderr)
        self.assertFalse(final.exists())
        self.assertTrue(temporary.exists())

    def test_upload_backend_preserves_idempotent_completion_result(self):
        import resumable_upload
        local, _ = self.output()
        completed = {"status": "already_complete", "path": "/remote"}
        with (patch.object(converter, "verify_output") as verify,
              patch.object(resumable_upload, "resumable_upload", return_value=completed) as backend):
            result = converter.upload(local, "example.invalid", "test", "/remote")
        self.assertEqual(result, completed)
        verify.assert_called_once_with(local, 1, 61, expected_fps=30)
        self.assertEqual(backend.call_args.args[:5], (local, "example.invalid", "test", "/remote", 22))

    def test_upload_passes_exact_target_without_local_directory_name(self):
        import resumable_upload
        local, _ = self.output()
        with (patch.object(converter, "verify_output"),
              patch.object(resumable_upload, "resumable_upload") as backend):
            converter.upload(local, "example.invalid", "test", "/remote/test/", 2222)
        self.assertEqual(backend.call_args.args[:5],
                         (local, "example.invalid", "test", "/remote/test/", 2222))
        self.assertNotIn("local-dataset", backend.call_args.args[3])
        self.assertEqual(backend.call_args.args[5], ["-o", "BatchMode=yes"])

    def test_upload_validates_the_recorded_fps(self):
        import resumable_upload

        local, _ = self.output()
        (local / "conversion.complete.json").write_text(json.dumps({
            "episodes": 1, "frames": 61, "fps": 25,
        }))
        with (patch.object(converter, "verify_output") as verify,
              patch.object(resumable_upload, "resumable_upload")):
            converter.upload(local, "example.invalid", "test", "/remote/test")
        verify.assert_called_once_with(local, 1, 61, expected_fps=25)

    def test_existing_empty_target_is_published_in_place(self):
        local, manifest = self.output()
        parent = self.root / "remote"
        final = parent / "test"
        final.mkdir(parents=True)
        temporary = parent / "test.uploading"
        prepared = self.remote_helper(converter.REMOTE_PREPARE_SCRIPT, manifest, parent, temporary, final)
        self.assertEqual(json.loads(prepared.stdout)["status"], "ready")
        shutil.copytree(local, temporary)
        self.remote_helper(converter.REMOTE_PUBLISH_SCRIPT, manifest, temporary, final)
        self.assertTrue((final / "data/example.bin").is_file())
        self.assertFalse((final / local.name).exists())

    def test_existing_unrelated_target_is_preserved(self):
        local, manifest = self.output()
        parent = self.root / "remote"
        final = parent / "test"
        final.mkdir(parents=True)
        (final / "existing.txt").write_text("keep this data")
        temporary = parent / "test.uploading"
        prepared = self.remote_helper(converter.REMOTE_PREPARE_SCRIPT, manifest, parent, temporary, final, check=False)
        self.assertNotEqual(prepared.returncode, 0)
        shutil.copytree(local, temporary)
        published = self.remote_helper(converter.REMOTE_PUBLISH_SCRIPT, manifest, temporary, final, check=False)
        self.assertNotEqual(published.returncode, 0)
        self.assertEqual((final / "existing.txt").read_text(), "keep this data")
        self.assertTrue(temporary.exists())

    def test_upload_target_and_no_upload_placeholder_override(self):
        self.assertEqual(converter.parse_upload_target("test@example.invalid:/data/datasets"),
                         ("example.invalid", "test", "/data/datasets"))
        for invalid in ("test@example.invalid:relative", "example.invalid:/data", "test@host"):
            with self.assertRaises(ValueError):
                converter.parse_upload_target(invalid)
        config = self.root / "upload.json"
        config.write_text(json.dumps({"upload_target": "待填写目标"}), encoding="utf-8")
        raw, output = self.root / "raw", self.root / "converted"
        with patch.object(converter, "__file__", str(self.root / "convert_pkl_dataset.py")), \
                patch.object(converter, "capture_info", return_value=(raw, {}, [])), \
                patch.object(converter, "convert", return_value=output), \
                patch.object(converter, "upload") as upload:
            with patch.object(sys, "argv", ["convert", "--input", str(raw), "--config", str(config), "--no-upload"]):
                converter.main()
            upload.assert_not_called()
            config.write_text(json.dumps({"upload_target": "cfg@example.invalid:/configured"}), encoding="utf-8")
            with patch.object(sys, "argv", ["convert", "--input", str(raw), "--config", str(config)]):
                converter.main()
            upload.assert_called_once_with(output, "example.invalid", "cfg", "/configured", 22)
            upload.reset_mock()
            with patch.object(sys, "argv", ["convert", "--input", str(raw), "--upload-target", "test@example.invalid:/data"]):
                converter.main()
            upload.assert_called_once_with(output, "example.invalid", "test", "/data", 22)


if __name__ == "__main__":
    unittest.main(verbosity=2)
