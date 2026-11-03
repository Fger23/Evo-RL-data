"""Exercise real byte transfer and visible progress without an SSH server."""
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pkl_converter import converter


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_real_stream_transfer_matches_files_and_shows_byte_progress(self):
        source, target = self.root / "source", self.root / "target"
        (source / "videos").mkdir(parents=True)
        payload = bytes(range(256)) * 10000
        (source / "videos/example.mp4").write_bytes(payload)
        (source / "empty").write_bytes(b"")
        (source / "meta.json").write_text('{"frames": 299}')
        log = io.StringIO()
        with patch.object(converter, "PROGRESS_ENABLED", True), redirect_stderr(log):
            converter.send_directory([sys.executable, "-c", converter.REMOTE_RECEIVE_SCRIPT, str(target)], source)
        self.assertEqual((target / "videos/example.mp4").read_bytes(), payload)
        self.assertEqual((target / "empty").read_bytes(), b"")
        self.assertEqual((target / "meta.json").read_bytes(), (source / "meta.json").read_bytes())
        self.assertIn("Upload", log.getvalue())
        self.assertIn("100%", log.getvalue())

    def test_auth_failure_is_reported_and_sender_process_exits(self):
        source = self.root / "source"
        source.mkdir()
        (source / "file").write_bytes(b"x" * (3 * 1024 * 1024))
        command = [sys.executable, "-c", "import sys; sys.stderr.write('Permission denied (publickey)'); sys.exit(23)"]
        with patch.object(converter, "PROGRESS_ENABLED", False), self.assertRaisesRegex(RuntimeError, "Permission denied"):
            converter.send_directory(command, source)

    def test_receiver_rejects_truncated_file(self):
        target = self.root / "target"
        packet = (json.dumps({"path": "sample", "size": 10}) + "\n").encode() + b"short"
        result = subprocess.run([sys.executable, "-c", converter.REMOTE_RECEIVE_SCRIPT, str(target)],
                                input=packet, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b"before file was complete", result.stderr)

    def test_receiver_rejects_parent_path(self):
        target = self.root / "target"
        packet = (json.dumps({"path": "../escape", "size": 0}) + "\n").encode()
        result = subprocess.run([sys.executable, "-c", converter.REMOTE_RECEIVE_SCRIPT, str(target)],
                                input=packet, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "escape").exists())

    def test_no_progress_suppresses_meter(self):
        log = io.StringIO()
        with (patch.object(converter, "PROGRESS_ENABLED", False), redirect_stderr(log),
              converter.progress_bar(total=2, desc="quiet") as bar):
            bar.update(2)
        self.assertEqual(log.getvalue(), "")


if __name__ == "__main__":
    unittest.main(verbosity=2)
