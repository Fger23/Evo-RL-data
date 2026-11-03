"""Converter backpressure and upload publication tests; no SSH/SCP execution."""
import importlib.util
import json
import shlex
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

        FakeDataset.create = staticmethod(create)
        fake_module = SimpleNamespace(LeRobotDataset=FakeDataset)
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
            "status": "already_complete", "path": "/remote",
        }), stderr="")
        with patch.object(converter, "verify_output"), patch.object(converter.subprocess, "run", return_value=completed) as run:
            converter.upload(local, "example.invalid", "test", "/remote")
        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args.args[0][0], "ssh")
        self.assertEqual(json.loads(run.call_args.kwargs["input"]), converter.checksum_manifest(local))
        remote_arguments = shlex.split(run.call_args.args[0][-1])
        self.assertEqual(remote_arguments[-1], "/remote")
        self.assertEqual(remote_arguments[-3], "/")

    def test_upload_uses_exact_target_without_local_directory_name(self):
        local, _ = self.output()
        prepared = subprocess.CompletedProcess([], 0, stdout='{"status":"ready"}', stderr="")
        published = subprocess.CompletedProcess([], 0, stdout='{"status":"complete","path":"/remote/test"}', stderr="")
        with (patch.object(converter, "verify_output"),
              patch.object(converter.subprocess, "run", side_effect=[prepared, published]) as run,
              patch.object(converter, "send_directory") as send):
            converter.upload(local, "example.invalid", "test", "/remote/test/", 2222)
        prepare_arguments = shlex.split(run.call_args_list[0].args[0][-1])
        publish_arguments = shlex.split(run.call_args_list[1].args[0][-1])
        self.assertEqual(prepare_arguments[-3], "/remote")
        self.assertEqual(prepare_arguments[-1], "/remote/test")
        self.assertTrue(prepare_arguments[-2].startswith("/remote/test.uploading-"))
        self.assertEqual(publish_arguments[-1], "/remote/test")
        self.assertNotIn("local-dataset", prepare_arguments[-1])
        transfer_arguments = shlex.split(send.call_args.args[0][-1])
        self.assertEqual(transfer_arguments[-1], prepare_arguments[-2])

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
