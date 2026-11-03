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

    def test_conversion_drains_image_queue_every_thirty_samples(self):
        raw = self.root / "raw"
        output = self.root / "converted"
        pending = 0
        peak = 0
        drains = []
        count = 0

        class FakeDataset:
            features = {}

            def add_frame(self, frame):
                nonlocal pending, peak, count
                pending += 1
                count += 1
                peak = max(peak, pending)

            def _wait_image_writer(self):
                nonlocal pending
                drains.append(count)
                pending = 0

            def save_episode(self, **kwargs):
                self._wait_image_writer()

            def stop_image_writer(self):
                pass

            def finalize(self):
                pass

        dataset = FakeDataset()

        def create(**kwargs):
            kwargs["root"].mkdir()
            return dataset

        fake_module = SimpleNamespace(LeRobotDataset=SimpleNamespace(create=create))
        info = {"repo_id": "local/test", "fps": 30, "features": {}, "robot_type": "synthetic", "vcodec": "h264"}
        episodes = [(raw / "episode", {"episode_index": 0, "frames": 61, "metadata": {}})]
        with patch.dict(sys.modules, {"lerobot.datasets.lerobot_dataset": fake_module}), \
                patch.object(converter, "capture_info", return_value=(raw, info, episodes)), \
                patch.object(converter, "validate_capture", return_value=61), \
                patch.object(converter, "episode_records", return_value=iter({"frame": {}} for _ in range(61))), \
                patch.object(converter, "verify_output"):
            converter.convert(raw, output)
        self.assertEqual(drains, [30, 60, 61])
        self.assertEqual(peak, 30)

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

    def test_completed_upload_retry_skips_scp(self):
        local, _ = self.output()
        completed = subprocess.CompletedProcess([], 0, stdout=json.dumps({
            "status": "already_complete", "path": "/remote/local-dataset",
        }), stderr="")
        with patch.object(converter, "verify_output"), patch.object(converter.subprocess, "run", return_value=completed) as run:
            converter.upload(local, "example.invalid", "test", "/remote")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "ssh")
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), converter.checksum_manifest(local))

    def test_upload_target_and_no_upload_placeholder_override(self):
        self.assertEqual(converter.parse_upload_target("test@example.invalid:/data/datasets"),
                         ("example.invalid", "test", "/data/datasets"))
        for invalid in ("test@example.invalid:relative", "example.invalid:/data", "test@host"):
            with self.assertRaises(ValueError):
                converter.parse_upload_target(invalid)
        config = self.root / "upload.json"
        config.write_text(json.dumps({"upload_target": "待填写目标"}), encoding="utf-8")
        raw, output = self.root / "raw", self.root / "converted"
        with patch.object(converter, "capture_info", return_value=(raw, {}, [])), \
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
