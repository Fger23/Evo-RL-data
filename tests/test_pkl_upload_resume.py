"""Real subprocess fault tests for the SSH-independent resumable protocol."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import selectors
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pkl_resumable_upload_tested", ROOT / "resumable_upload.py")
upload = importlib.util.module_from_spec(spec)
spec.loader.exec_module(upload)


class CountingBar:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.completed = 0

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def set_postfix_str(self, *_args, **_kwargs):
        pass

    def update(self, count):
        self.completed += count


@unittest.skipUnless(os.name == "posix", "Receiver uses Linux flock and directory fsync")
class ResumeUploadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / "source"
        (self.source / "videos").mkdir(parents=True)
        self.payload = bytes(range(251)) * 13000  # Multiple durable transfer blocks.
        (self.source / "videos/example.mp4").write_bytes(self.payload)
        (self.source / "empty").write_bytes(b"")
        (self.source / "meta.json").write_text('{"episodes": 2, "frames": 299}')
        self.manifest = upload.build_manifest(self.source)
        self.target = self.root / "remote" / "dataset"
        self.processes = []

    def tearDown(self):
        for process in self.processes:
            if process.poll() is None:
                process.kill()
            if process.stdin is not None:
                process.stdin.close()
                process.stdin = None
            process.communicate(timeout=10)
        self.temp.cleanup()

    def command(self, target=None):
        # -O verifies that no safety guarantees depend on assert statements.
        return [sys.executable, "-u", "-O", "-c", upload.REMOTE_RESUME_SCRIPT,
                str(target or self.target)]

    def raw_server(self, idle_timeout=120):
        process = subprocess.Popen(self.command(), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.processes.append(process)
        self.send(process, {"version": 1, "files": self.manifest, "idle_timeout": idle_timeout})
        return process, self.receive(process)

    def send(self, process, value):
        process.stdin.write((json.dumps(value) + "\n").encode())
        process.stdin.flush()

    def receive(self, process):
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout=15):
                self.fail("Receiver did not respond")
        line = process.stdout.readline()
        if not line:
            self.fail("Receiver exited before its response: " + process.stderr.read().decode())
        return json.loads(line)

    def interrupt_partial_file(self, count):
        process, prepared = self.raw_server()
        self.assertEqual(prepared["status"], "ready")
        self.send(process, {"op": "file", "path": "videos/example.mp4"})
        self.assertEqual(self.receive(process)["offset"], 0)
        self.send(process, {"op": "start", "offset": 0})
        process.stdin.write(self.payload[:count])
        process.stdin.flush()
        process.stdin.close()
        process.stdin = None
        stdout, _ = process.communicate(timeout=15)
        self.assertNotEqual(process.returncode, 0)
        self.assertIn(b"partial bytes retained", stdout)
        staging = Path(prepared["temporary"])
        self.assertEqual((staging / "videos/example.mp4").read_bytes(), self.payload[:count])
        self.assertFalse(self.target.exists())
        return staging

    def perform(self):
        return upload.stream_upload(self.command(), self.source, self.manifest, io_timeout=15)

    def verify_final(self):
        for name, item in self.manifest.items():
            path = self.target / name
            self.assertEqual(path.stat().st_size, item["size"])
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), item["sha256"])
        self.assertFalse((self.target / upload.STATE_NAME).exists())
        self.assertFalse((self.target / upload.PROGRESS_NAME).exists())
        self.assertEqual(json.loads((self.target / upload.MANIFEST_NAME).read_text()),
                         {name: item["sha256"] for name, item in self.manifest.items()})

    def test_resumes_exact_durable_partial_bytes_and_progress(self):
        retained = upload.CHUNK_BYTES + 12345
        staging = self.interrupt_partial_file(retained)
        bars = []

        def progress(**kwargs):
            bar = CountingBar(**kwargs)
            bars.append(bar)
            return bar

        result = upload.stream_upload(self.command(), self.source, self.manifest,
                                      progress_factory=progress, io_timeout=15)
        total = sum(item["size"] for item in self.manifest.values())
        self.assertEqual(result["resumed_bytes"], retained)
        self.assertEqual(result["sent_bytes"], total - retained)
        self.assertEqual(bars[0].completed, total)
        self.assertFalse(staging.exists())
        self.verify_final()

    def test_corrupt_partial_and_corrupt_same_size_files_are_repaired(self):
        for count in [1234, len(self.payload), len(self.payload) + 100]:
            with self.subTest(size=count):
                if self.target.exists():
                    # Use a fresh destination rather than mutating a published one.
                    self.target = self.target.with_name(self.target.name + "_next")
                staging = self.interrupt_partial_file(100)
                (staging / "videos/example.mp4").write_bytes(b"X" * count)
                result = self.perform()
                self.assertEqual(result["resumed_bytes"], 0)
                self.assertEqual(result["sent_bytes"], sum(v["size"] for v in self.manifest.values()))
                self.verify_final()

    def test_existing_complete_target_is_verified_and_not_resent(self):
        self.perform()
        result = self.perform()
        self.assertEqual(result["status"], "already_complete")
        self.assertEqual(result["sent_bytes"], 0)
        self.verify_final()
        (self.target / "meta.json").write_bytes(b"corrupted")
        with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
            self.perform()

    def test_existing_empty_target_is_allowed(self):
        self.target.mkdir(parents=True)
        self.perform()
        self.verify_final()

    def test_nonempty_unrelated_destination_is_preserved(self):
        self.target.mkdir(parents=True)
        (self.target / "keep").write_text("do not overwrite")
        with self.assertRaisesRegex(RuntimeError, "file list"):
            self.perform()
        self.assertEqual((self.target / "keep").read_text(), "do not overwrite")

    def test_kernel_lock_released_after_sigkill_and_destination_lock_is_exclusive(self):
        first, prepared = self.raw_server()
        with self.assertRaisesRegex(RuntimeError, "Another upload"):
            self.perform()
        changed = dict(self.manifest)
        changed["other_file"] = {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()}
        with self.assertRaisesRegex(RuntimeError, "Another upload"):
            upload.stream_upload(self.command(), self.source, changed, io_timeout=15)
        first.kill()
        first.stdin.close()
        first.stdin = None
        first.communicate(timeout=10)
        self.perform()
        self.verify_final()

    def test_half_open_connection_expires_and_releases_lock(self):
        process, _ = self.raw_server(idle_timeout=0.2)
        # Keep our stdin open, as with a connection whose remote TCP peer has
        # vanished but whose sshd has not noticed. The receiver must time out.
        response = self.receive(process)
        self.assertEqual(response["status"], "error")
        self.assertIn("idle timeout", response["error"])
        process.wait(timeout=5)
        self.perform()
        self.verify_final()

    def test_keepalives_allow_long_prefix_verification(self):
        import time
        process, _ = self.raw_server(idle_timeout=0.3)
        for _ in range(8):
            time.sleep(0.1)
            self.send(process, {"op": "keepalive"})
        self.send(process, {"op": "file", "path": "empty"})
        self.assertEqual(self.receive(process)["status"], "offset")
        process.stdin.close()
        process.stdin = None
        process.communicate(timeout=5)
        self.perform()
        self.verify_final()

    def test_client_answers_receiver_liveness_probes_during_hashing(self):
        command = self.command()
        command[-2] = command[-2].replace("PROBE_SECONDS=10", "PROBE_SECONDS=0")
        result = upload.stream_upload(command, self.source, self.manifest, io_timeout=15)
        self.assertEqual(result["status"], "complete")
        self.verify_final()

    def test_unresponsive_client_during_remote_hashing_releases_lock(self):
        import shutil
        self.target.parent.mkdir(parents=True)
        shutil.copytree(self.source, self.target)
        command = self.command()
        command[-2] = command[-2].replace("PROBE_SECONDS=10", "PROBE_SECONDS=0")
        process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.processes.append(process)
        self.send(process, {"version": 1, "files": self.manifest, "idle_timeout": 0.2})
        self.assertEqual(self.receive(process)["status"], "ping")
        # No pong: a client that died while the remote side was hashing.
        response = self.receive(process)
        self.assertEqual(response["status"], "error")
        self.assertIn("idle timeout", response["error"])
        process.wait(timeout=5)
        self.assertEqual(self.perform()["status"], "already_complete")
        self.verify_final()

    def test_killing_client_writer_closes_remote_pipe_and_retains_progress(self):
        import time
        read_fd, write_fd = os.pipe()
        process = subprocess.Popen(self.command(), stdin=read_fd, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        self.processes.append(process)
        os.close(read_fd)
        ready = self.root / "client_sent"
        code = (
            "import json,os,sys,time; from pathlib import Path; "
            "f=os.fdopen(int(sys.argv[1]),'wb'); "
            "f.write((json.dumps({'version':1,'files':json.loads(sys.argv[2])})+'\\n').encode()); "
            "f.write(b'{\"op\":\"file\",\"path\":\"videos/example.mp4\"}\\n'); "
            "f.write(b'{\"op\":\"start\",\"offset\":0}\\n'); "
            "f.write(Path(sys.argv[3]).read_bytes()[:1048576]); f.flush(); "
            "Path(sys.argv[4]).write_text('ready'); time.sleep(60)"
        )
        client = subprocess.Popen([sys.executable, "-c", code, str(write_fd),
                                   json.dumps(self.manifest), str(self.source / "videos/example.mp4"),
                                   str(ready)], pass_fds=(write_fd,), stdout=subprocess.DEVNULL,
                                  stderr=subprocess.PIPE)
        os.close(write_fd)
        try:
            prepared = self.receive(process)
            self.assertEqual(prepared["status"], "ready")
            self.assertEqual(self.receive(process)["status"], "offset")
            until = time.monotonic() + 10
            while not ready.exists() and time.monotonic() < until:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            client.kill()
            client.communicate(timeout=5)
            process.communicate(timeout=10)
            self.assertNotEqual(process.returncode, 0)
            self.assertEqual((Path(prepared["temporary"]) / "videos/example.mp4").stat().st_size,
                             upload.CHUNK_BYTES)
            result = self.perform()
            self.assertEqual(result["resumed_bytes"], upload.CHUNK_BYTES)
            self.verify_final()
        finally:
            if client.poll() is None:
                client.kill()
            client.communicate(timeout=5)

    def test_missing_state_after_publish_window_and_leftover_json_temp_recover(self):
        staging = self.interrupt_partial_file(12345)
        (staging / upload.STATE_NAME).unlink()
        (staging / (upload.STATE_NAME + ".tmp")).write_bytes(b'{"incomplete')
        (staging / (upload.MANIFEST_NAME + ".tmp")).write_bytes(b'{"incomplete')
        self.perform()
        self.verify_final()

    def test_uncommitted_bad_tail_rolls_back_only_to_durable_cursor(self):
        retained = upload.CHUNK_BYTES + 123
        staging = self.interrupt_partial_file(retained)
        with (staging / "videos/example.mp4").open("ab") as stream:
            stream.write(b"uncommitted bad tail" * 5000)
        result = self.perform()
        self.assertEqual(result["resumed_bytes"], retained)
        self.assertEqual(result["sent_bytes"],
                         sum(item["size"] for item in self.manifest.values()) - retained)
        self.verify_final()

    def test_successful_publication_with_lost_ack_is_idempotent(self):
        process, _ = self.raw_server()
        for name, _item in sorted(self.manifest.items()):
            self.send(process, {"op": "file", "path": name})
            self.receive(process)
            self.send(process, {"op": "start", "offset": 0})
            process.stdin.write((self.source / name).read_bytes())
            process.stdin.flush()
            self.receive(process)
        self.send(process, {"op": "finish"})
        # Wait for the directory publication, without consuming the final ack.
        import time
        until = time.monotonic() + 15
        while not self.target.exists() and time.monotonic() < until:
            time.sleep(0.01)
        self.assertTrue(self.target.exists())
        if process.poll() is None:
            process.kill()
        process.stdin.close()
        process.stdin = None
        process.communicate(timeout=10)
        self.assertEqual(self.perform()["status"], "already_complete")
        self.verify_final()

    def test_rejects_symlinks_without_touching_their_targets(self):
        external = self.root / "external"
        external.mkdir()
        (external / "keep").write_text("preserved")
        self.target.parent.mkdir()
        self.target.symlink_to(external, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.perform()
        self.assertEqual((external / "keep").read_text(), "preserved")
        self.target.unlink()
        staging = self.interrupt_partial_file(100)
        (staging / "videos/example.mp4").unlink()
        (staging / "videos/example.mp4").symlink_to(external / "keep")
        with self.assertRaisesRegex(RuntimeError, "symlink"):
            self.perform()
        self.assertEqual((external / "keep").read_text(), "preserved")

    def test_rejects_payload_and_metadata_hardlinks_without_modifying_external_file(self):
        external = self.root / "keep_external"
        external.write_bytes(b"preserved")
        staging = self.interrupt_partial_file(100)
        payload = staging / "videos/example.mp4"
        payload.unlink()
        os.link(external, payload)
        with self.assertRaisesRegex(RuntimeError, "hardlink"):
            self.perform()
        self.assertEqual(external.read_bytes(), b"preserved")
        payload.unlink()
        metadata = staging / (upload.STATE_NAME + ".tmp")
        os.link(external, metadata)
        with self.assertRaisesRegex(RuntimeError, "hardlink"):
            self.perform()
        self.assertEqual(external.read_bytes(), b"preserved")

    def test_manifest_parent_escape_is_rejected(self):
        bad = {"../escape": {"size": 0, "sha256": hashlib.sha256(b"").hexdigest()}}
        with self.assertRaisesRegex(RuntimeError, "Invalid"):
            upload.stream_upload(self.command(), self.source, bad, io_timeout=15)
        self.assertFalse((self.root / "escape").exists())


class ResumeClientTests(unittest.TestCase):
    def test_authentication_error_is_preserved_and_stderr_is_drained(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "sample").write_bytes(b"abc")
            manifest = upload.build_manifest(source)
            command = [sys.executable, "-c",
                       "import sys; sys.stderr.write('x'*200000+' Permission denied (publickey)'); sys.exit(255)"]
            with self.assertRaisesRegex(RuntimeError, "Permission denied"):
                upload.stream_upload(command, source, manifest, io_timeout=15)

    def test_manifest_is_fresh_and_ignores_interrupted_manifest_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "sample").write_bytes(b"first")
            first = upload.build_manifest(source)
            (source / "sample").write_bytes(b"second")
            (source / (upload.MANIFEST_NAME + ".tmp")).write_bytes(b'{"unfinished')
            second = upload.build_manifest(source)
            self.assertNotEqual(first, second)
            self.assertNotIn(upload.MANIFEST_NAME, second)
            self.assertNotIn(upload.MANIFEST_NAME + ".tmp", second)
            self.assertFalse((source / (upload.MANIFEST_NAME + ".tmp")).exists())

    @unittest.skipUnless(hasattr(os, "link"), "Hardlinks unavailable")
    def test_read_only_local_payload_hardlinks_are_allowed_and_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            external = root / "native_episode_video.mp4"
            external.write_bytes(b"valid native video payload")
            os.link(external, source / "video.mp4")
            manifest = upload.build_manifest(source)
            self.assertEqual(manifest["video.mp4"]["sha256"],
                             hashlib.sha256(external.read_bytes()).hexdigest())
            self.assertEqual(external.read_bytes(), b"valid native video payload")
            self.assertEqual((source / "video.mp4").stat().st_nlink, 2)

    def test_invalid_endpoints_are_rejected_before_starting_ssh(self):
        with tempfile.TemporaryDirectory() as directory:
            for host, user, destination, port in [
                ("-host", "user", "/safe", 22), ("host", "-user", "/safe", 22),
                ("host", "user", "/", 22), ("host", "user", "/a/../b", 22),
                ("host", "user", "/safe", True),
            ]:
                with self.subTest(destination=destination, port=port), self.assertRaises(ValueError):
                    upload.resumable_upload(directory, host, user, destination, port, [], None)

    def test_public_upload_waits_for_busy_session_then_continues(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            (source / "file").write_bytes(b"abc")
            completed = {"status": "complete", "path": "/safe",
                         "resumed_bytes": 0, "sent_bytes": 3}
            with patch.object(upload, "stream_upload",
                              side_effect=[upload.UploadBusyError("Another upload"), completed]) as stream, \
                    patch.object(upload.time, "sleep") as sleep:
                self.assertEqual(upload.resumable_upload(source, "host", "user", "/safe",
                                                         22, [], None), completed)
            self.assertEqual(stream.call_count, 2)
            self.assertEqual(sleep.call_count, 1)
            with (patch.object(upload, "stream_upload", side_effect=upload.UploadBusyError("Another upload")),
                  self.assertRaisesRegex(upload.UploadBusyError, "still active")):
                upload.resumable_upload(source, "host", "user", "/safe", 22, [], None, busy_wait_s=0)

    @unittest.skipUnless(hasattr(os, "link"), "Hardlinks unavailable")
    def test_local_manifest_temp_hardlink_does_not_modify_external_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            (source / "sample").write_bytes(b"data")
            external = root / "keep"
            external.write_bytes(b"preserved")
            os.link(external, source / (upload.MANIFEST_NAME + ".tmp"))
            with self.assertRaisesRegex(ValueError, "hardlink"):
                upload.build_manifest(source)
            self.assertEqual(external.read_bytes(), b"preserved")


if __name__ == "__main__":
    unittest.main(verbosity=2)
