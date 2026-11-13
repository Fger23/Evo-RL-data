"""Bounded live preview without cameras, robot hardware or a real viewer."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import tracemalloc
import unittest
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lerobot.utils import live_preview as preview, live_preview_worker as worker  # noqa: E402


def packet(value=0):
    return {
        "observation": {
            "left_fixed": np.full((16, 20, 3), value, dtype=np.uint8),
            "state": np.array([value, value + 1], dtype=np.float32),
        },
        "action": {"j0": float(value)},
        "compress_images": True,
        "timestamp": 1700000000.0 + value,
    }


class MailboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.path = self.root / "frames.mmap"
        self.capacity = 64 * 1024
        self.producer = preview.LatestFrameMailbox(self.path, self.capacity, create=True)
        self.consumer = preview.LatestFrameMailbox(self.path, self.capacity)

    def tearDown(self):
        self.consumer.close()
        self.producer.close()
        self.temporary.cleanup()

    def test_thousand_frames_leave_only_latest_in_fixed_storage(self):
        size = self.path.stat().st_size
        for sequence in range(1, 1001):
            self.assertTrue(self.producer.publish(sequence, packet(sequence % 200)))
        self.assertEqual(self.path.stat().st_size, size)
        sequence, actual = self.consumer.read(0)
        self.assertEqual(sequence, 1000)
        self.assertEqual(actual["action"]["j0"], 0.0)
        self.assertIsNone(self.consumer.read(sequence))

    def test_snapshot_owns_pixels_and_numeric_values(self):
        original = packet(17)
        pixels = original["observation"]["left_fixed"].copy()
        state = original["observation"]["state"].copy()
        self.assertTrue(self.producer.publish(1, original))
        original["observation"]["left_fixed"][:] = 100
        original["observation"]["state"][:] = -999
        original["action"]["j0"] = -999
        _, actual = self.consumer.read(0)
        np.testing.assert_array_equal(actual["observation"]["left_fixed"], pixels)
        np.testing.assert_array_equal(actual["observation"]["state"], state)
        self.assertEqual(actual["action"]["j0"], 17)
        actual["observation"]["left_fixed"][:] = 2
        _, again = self.consumer.read(0)
        np.testing.assert_array_equal(again["observation"]["left_fixed"], pixels)

    def test_oversized_numpy_is_rejected_before_pickle_allocation(self):
        oversized = packet()
        oversized["observation"]["left_fixed"] = np.zeros((self.capacity + 1,), dtype=np.uint8)
        with patch.object(preview.pickle, "dumps", side_effect=AssertionError("oversized data must not serialize")):
            self.assertFalse(self.producer.publish(1, oversized))
        self.assertIsNone(self.consumer.read(0))

    @unittest.skipUnless(os.name == "posix", "requires independent POSIX flock holder")
    def test_busy_reader_drops_frame_without_waiting(self):
        code = (
            "import fcntl,sys,time; f=open(sys.argv[1],'r+b'); "
            "fcntl.flock(f,fcntl.LOCK_EX); print('LOCKED',flush=True); time.sleep(30)"
        )
        holder = subprocess.Popen([sys.executable, "-c", code, str(self.path)],
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            self.assertEqual(holder.stdout.readline().strip(), b"LOCKED")
            start = time.perf_counter()
            self.assertFalse(self.producer.publish(1, packet()))
            self.assertLess(time.perf_counter() - start, 0.1)
        finally:
            holder.terminate()
            holder.communicate(timeout=5)
        self.assertTrue(self.producer.publish(2, packet(2)))

    def test_close_is_idempotent_and_closed_publisher_drops_input(self):
        self.producer.close()
        self.producer.close()
        self.assertFalse(self.producer.publish(1, packet()))

    def test_close_race_after_initial_guard_never_uses_a_closed_descriptor(self):
        class CloseOnAcquire:
            def __init__(self, mailbox):
                self.mailbox = mailbox
                self.inner = mailbox._thread_lock
                self.triggered = False

            def acquire(self, blocking=False):
                if not self.triggered:
                    self.triggered = True
                    self.mailbox.close()
                return self.inner.acquire(blocking=blocking)

            def release(self):
                self.inner.release()

            def __enter__(self):
                self.inner.acquire()
                return self

            def __exit__(self, *_args):
                self.inner.release()

        self.producer._thread_lock = CloseOnAcquire(self.producer)
        self.assertFalse(self.producer.publish(1, packet()))
        self.consumer._thread_lock = CloseOnAcquire(self.consumer)
        self.assertIsNone(self.consumer.read(0))


class SessionTests(unittest.TestCase):
    @staticmethod
    def sleeping_worker():
        return [sys.executable, "-c", "import time; time.sleep(60)"]

    def session(self, **kwargs):
        return preview.LivePreviewSession("test-preview", ip="127.0.0.1", port=1,
                                          capacity=256 * 1024, max_fps=10000,
                                          worker_command=self.sleeping_worker(), monitor=False, **kwargs)

    def test_blocked_worker_does_not_build_an_unbounded_producer_queue(self):
        session = self.session()
        original = packet(12)
        image = np.full((64, 64, 3), 12, dtype=np.uint8)
        original["observation"].update(wrist=image, right_wrist=image)
        try:
            initial_size = session.mailbox.path.stat().st_size
            tracemalloc.start()
            start = time.perf_counter()
            accepted = 0
            for _ in range(1000):
                accepted += bool(session.submit(original["observation"], original["action"]))
            elapsed = time.perf_counter() - start
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.assertGreater(accepted, 0)
            self.assertLess(elapsed, 5.0)
            self.assertLess(peak, 4 * 1024 * 1024)
            self.assertEqual(session.mailbox.path.stat().st_size, initial_size)
            np.testing.assert_array_equal(image, np.full_like(image, 12))
            self.assertGreater(session.stats["submitted"], 0)
        finally:
            if tracemalloc.is_tracing():
                tracemalloc.stop()
            session.close()

    def test_slow_consumer_still_allows_paced_thirty_fps_producer(self):
        session = self.session()
        value = packet(10)
        durations = []
        try:
            deadline = time.perf_counter()
            for _ in range(15):
                start = time.perf_counter()
                self.assertTrue(session.submit(value["observation"], value["action"]))
                durations.append(time.perf_counter() - start)
                deadline += 1 / 30
                time.sleep(max(0, deadline - time.perf_counter()))
            self.assertLess(max(durations), 1 / 30)
            self.assertLess(sum(durations) / len(durations), 0.01)
        finally:
            session.close()

    def test_thirty_hz_scheduler_jitter_does_not_drop_every_second_preview_frame(self):
        session = preview.LivePreviewSession("jitter-tolerance", ip="127.0.0.1", port=1,
                                              capacity=64 * 1024, max_fps=30,
                                              worker_command=self.sleeping_worker(), monitor=False)
        try:
            size = session.mailbox.path.stat().st_size
            value = packet(5)
            start = time.monotonic()
            arrivals = [start + index / 30 + (0.0005 if index % 2 == 0 else -0.0005)
                        for index in range(90)]
            # Patch only after construction and restore before process waits.
            # The stand-in consumer remains a real, sleeping subprocess.
            with patch.object(preview.time, "monotonic", side_effect=arrivals):
                accepted = sum(bool(session.submit(value["observation"], value["action"]))
                               for _ in arrivals)
            self.assertGreaterEqual(accepted, len(arrivals) * 0.95)
            self.assertEqual(session.mailbox.path.stat().st_size, size)
            self.assertEqual(session.stats["submitted"], accepted)
        finally:
            session.close()

    def test_close_reaps_owned_worker_and_removes_resources(self):
        for _ in range(3):
            session = self.session()
            directory = session.directory
            pid = session.stats["worker_pid"]
            self.assertTrue(directory.is_dir())
            start = time.perf_counter()
            session.close()
            session.close()
            self.assertLess(time.perf_counter() - start, 3.0)
            self.assertFalse(directory.exists())
            if os.name == "posix":
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)

    def test_exited_worker_stops_accepting_frames(self):
        with tempfile.TemporaryDirectory() as folder:
            signal = Path(folder) / "done"
            command = [sys.executable, "-c",
                       "import pathlib,sys; pathlib.Path(sys.argv[1]).write_text('done')", str(signal)]
            session = preview.LivePreviewSession("dead-worker", ip="127.0.0.1", port=1,
                                                  worker_command=command, monitor=False, max_fps=10000)
            try:
                deadline = time.monotonic() + 5
                value = packet()
                while not signal.exists() and time.monotonic() < deadline:
                    session.submit(value["observation"], value["action"])
                    time.sleep(0.02)
                self.assertTrue(signal.exists())
                time.sleep(0.1)
                before = session.stats["submitted"]
                for _ in range(5):
                    self.assertFalse(session.submit(value["observation"], value["action"]))
                    time.sleep(0.01)
                self.assertEqual(session.stats["submitted"], before)
            finally:
                session.close()

    def test_close_only_reaps_the_worker_and_viewer_owned_by_session(self):
        unrelated = subprocess.Popen(self.sleeping_worker())
        session = None
        try:
            session = preview.LivePreviewSession("owned-processes", worker_command=self.sleeping_worker(),
                                                  viewer_command=self.sleeping_worker(), monitor=False)
            viewer_pid = session.stats["viewer_pid"]
            worker_pid = session.stats["worker_pid"]
            self.assertIsNotNone(viewer_pid)
            self.assertNotEqual(viewer_pid, worker_pid)
            session.close()
            self.assertIsNone(unrelated.poll())
            if os.name == "posix":
                for pid in (viewer_pid, worker_pid):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
        finally:
            if session is not None:
                session.close()
            unrelated.terminate()
            unrelated.wait(timeout=5)

    def test_watchdog_stops_owned_preview_when_sdk_memory_budget_is_exceeded(self):
        session = None
        with patch.object(preview, "_process_group_rss", return_value=16 * 1024 * 1024 * 1024):
            try:
                session = preview.LivePreviewSession("memory-guard", ip="127.0.0.1", port=1,
                                                      worker_command=self.sleeping_worker(), monitor=True)
                deadline = time.monotonic() + 3
                while not session.stats["disabled"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(session.stats["disabled"])
                self.assertFalse(session.submit(packet()["observation"], {}))
            finally:
                if session is not None:
                    session.close()

    def test_watchdog_stops_worker_stuck_in_native_flush(self):
        code = (
            "import json,pathlib,sys,time; c=json.loads(pathlib.Path(sys.argv[-1]).read_text()); "
            "pathlib.Path(c['status_file']).write_text(json.dumps({'phase':'flushing',"
            "'updated':time.monotonic()-100,'frames_sent':0,'seq':1})); time.sleep(60)"
        )
        session = preview.LivePreviewSession("stalled-flush", ip="127.0.0.1", port=1,
                                              worker_command=[sys.executable, "-c", code], monitor=True)
        try:
            deadline = time.monotonic() + 4
            while not session.stats["disabled"] and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(session.stats["disabled"])
            self.assertFalse(session.submit(packet()["observation"], {}))
        finally:
            session.close()

    def test_watchdog_stops_worker_stuck_initializing_sdk(self):
        code = (
            "import json,pathlib,sys,time; c=json.loads(pathlib.Path(sys.argv[-1]).read_text()); "
            "pathlib.Path(c['status_file']).write_text(json.dumps({'phase':'starting',"
            "'updated':time.monotonic()-100,'frames_sent':0,'seq':0})); time.sleep(60)"
        )
        session = preview.LivePreviewSession("stalled-startup", ip="127.0.0.1", port=1,
                                              worker_command=[sys.executable, "-c", code], monitor=True)
        try:
            deadline = time.monotonic() + 3
            while not session.stats["disabled"] and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(session.stats["disabled"])
            self.assertFalse(session.submit(packet()["observation"], {}))
        finally:
            session.close()


@unittest.skipUnless(sys.platform.startswith("linux"), "requires Linux process identity and SIGKILL")
class OrphanCleanupTests(unittest.TestCase):
    @staticmethod
    def active(pid):
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return fields[0] not in {"Z", "X"}
        except (OSError, IndexError):
            return False

    def test_creation_token_prevents_killing_an_unrelated_or_reused_viewer_pid(self):
        viewer = subprocess.Popen(SessionTests.sleeping_worker(), start_new_session=True)
        unrelated = subprocess.Popen(SessionTests.sleeping_worker(), start_new_session=True)
        try:
            token = preview._process_start_ticks(viewer.pid)
            self.assertIsNotNone(token)
            config = {"viewer_pid": viewer.pid, "viewer_start_ticks": token + 1,
                      "parent_pid": os.getpid()}
            self.assertFalse(worker._cleanup_orphan_viewer(config))
            self.assertIsNone(viewer.poll())
            config["viewer_start_ticks"] = token
            self.assertTrue(worker._cleanup_orphan_viewer(config))
            viewer.wait(timeout=3)
            self.assertIsNone(unrelated.poll())
            self.assertFalse(worker._cleanup_orphan_viewer({"viewer_pid": None, "parent_pid": os.getpid()}))
        finally:
            for process in (viewer, unrelated):
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=5)

    def test_parent_sigkill_cleans_owned_worker_and_viewer_even_while_main_worker_is_stuck(self):
        unrelated = subprocess.Popen(SessionTests.sleeping_worker(), start_new_session=True)
        parent = None
        pids = {}
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            worker_code = textwrap.dedent(f"""
                import json,pathlib,sys,threading,time
                sys.path.insert(0, {str(ROOT / 'src')!r})
                from lerobot.utils import live_preview_worker as worker
                config=json.loads(pathlib.Path(sys.argv[1]).read_text())
                stopped=threading.Event()
                watchdog=threading.Thread(target=worker._parent_watchdog,args=(config,stopped),daemon=True)
                watchdog.start()
                pathlib.Path(sys.argv[2]).write_text('ready')
                time.sleep(60)
            """)
            parent_code = textwrap.dedent(f"""
                import json,os,pathlib,subprocess,sys,time
                root=pathlib.Path(sys.argv[1])
                viewer=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)'],
                                        start_new_session=True,stdin=subprocess.DEVNULL,
                                        stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                def ticks(pid):
                    return int(pathlib.Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
                config={{'parent_pid':os.getpid(),'viewer_pid':viewer.pid,'viewer_start_ticks':ticks(viewer.pid)}}
                (root/'config.json').write_text(json.dumps(config))
                child=subprocess.Popen([sys.executable,'-c',{worker_code!r},str(root/'config.json'),str(root/'worker.ready')],
                                       start_new_session=True,stdin=subprocess.DEVNULL,
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
                config.update(worker_pid=child.pid,worker_start_ticks=ticks(child.pid))
                (root/'pids.json').write_text(json.dumps(config))
                time.sleep(60)
            """)
            try:
                parent = subprocess.Popen([sys.executable, "-c", parent_code, str(root)],
                                          start_new_session=True, stdin=subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                deadline = time.monotonic() + 10
                while not (root / "worker.ready").is_file() and parent.poll() is None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / "worker.ready").is_file(), "stand-in worker did not start")
                pids = json.loads((root / "pids.json").read_text())
                parent.kill()
                parent.wait(timeout=5)
                deadline = time.monotonic() + 5
                while (self.active(pids["viewer_pid"]) or self.active(pids["worker_pid"])) and time.monotonic() < deadline:
                    time.sleep(0.025)
                self.assertFalse(self.active(pids["viewer_pid"]))
                self.assertFalse(self.active(pids["worker_pid"]))
                self.assertIsNone(unrelated.poll())
            finally:
                if not pids and (root / "pids.json").is_file():
                    with suppress(ValueError):
                        pids = json.loads((root / "pids.json").read_text())
                if parent is not None:
                    if parent.poll() is None:
                        parent.kill()
                    parent.wait(timeout=5)
                for prefix in ("viewer", "worker"):
                    pid = pids.get(prefix + "_pid")
                    token = pids.get(prefix + "_start_ticks")
                    if pid and self.active(pid) and preview._process_start_ticks(pid) == token:
                        os.killpg(pid, signal.SIGKILL)
                if unrelated.poll() is None:
                    unrelated.terminate()
                unrelated.wait(timeout=5)


class FakeRR:
    def __init__(self):
        self.calls = []
        self.times = []

    class Scalars:
        def __init__(self, value):
            self.value = value

    class Image:
        def __init__(self, array):
            self.array = array
            self.quality = None

        def compress(self, *, jpeg_quality=80):
            self.quality = jpeg_quality
            return self

    def log(self, key, entity=None, *args, **kwargs):
        self.calls.append((key, entity, kwargs))

    def set_time(self, timeline, **kwargs):
        self.times.append((timeline, kwargs))


class WorkerLoggingTests(unittest.TestCase):
    def test_logs_compressed_images_and_numeric_values_without_changing_capture(self):
        rr = FakeRR()
        rec = object()
        value = packet(19)
        chw = np.arange(3 * 10 * 20, dtype=np.uint8).reshape(3, 10, 20)
        expected = chw.copy()
        value["observation"]["left_fixed"] = chw
        value["observation"]["none"] = None
        worker.log_snapshot(rr, rec, 73, value, jpeg_quality=77)
        calls = {key: (entity, options) for key, entity, options in rr.calls}
        self.assertEqual(set(calls), {"observation.left_fixed", "observation.state_0",
                                     "observation.state_1", "action.j0"})
        entity, options = calls["observation.left_fixed"]
        np.testing.assert_array_equal(entity.array, expected.transpose(1, 2, 0))
        self.assertEqual(entity.quality, 77)
        self.assertFalse(options.get("static", False))
        self.assertIs(options.get("recording"), rec)
        self.assertEqual(calls["observation.state_0"][0].value, 19)
        self.assertEqual(calls["action.j0"][0].value, 19)
        self.assertTrue(any(name == "frame" and options.get("sequence") == 73 for name, options in rr.times))
        self.assertTrue(any(options.get("timestamp") == value["timestamp"] for _, options in rr.times))
        np.testing.assert_array_equal(chw, expected)

    def test_gray_hwc_and_explicit_raw_display_are_supported(self):
        rr = FakeRR()
        value = packet()
        value["compress_images"] = False
        value["observation"] = {"observation.gray": np.zeros((8, 8, 1), dtype=np.uint8),
                                "rgb": np.zeros((9, 12, 3), dtype=np.uint8)}
        worker.log_snapshot(rr, object(), 1, value)
        calls = {key: entity for key, entity, _ in rr.calls}
        self.assertEqual(calls["observation.gray"].array.shape, (8, 8, 1))
        self.assertEqual(calls["observation.rgb"].array.shape, (9, 12, 3))
        self.assertIsNone(calls["observation.rgb"].quality)

    def test_budget_overflow_is_rejected_before_any_sdk_log(self):
        rr = FakeRR()
        value = packet()
        value["action"] = {"too_many": np.ones(1025, dtype=np.float32)}
        with self.assertRaises((ValueError, RuntimeError)):
            worker.log_snapshot(rr, object(), 1, value)
        self.assertEqual(rr.calls, [])

    def test_flush_failure_never_logs_the_next_snapshot(self):
        rr = FakeRR()
        recording = SimpleNamespace(flush=MagicMock(side_effect=TimeoutError("viewer is unavailable")))
        rr.RecordingStream = MagicMock(return_value=recording)
        rr.connect_grpc = MagicMock()
        mailbox = SimpleNamespace(read=MagicMock(side_effect=[(1, packet(1)), (2, packet(2))]), close=MagicMock())
        with tempfile.TemporaryDirectory() as folder:
            config = {"mailbox": str(Path(folder) / "frames.mmap"), "capacity": 65536,
                      "status_file": str(Path(folder) / "status.json"), "url": "rerun+http://127.0.0.1:9876/proxy",
                      "session_name": "test", "recording_id": "0bc818f0-2091-454d-b10d-c08c61db7eba",
                      "parent_pid": os.getpid(), "flush_timeout_s": 0.1,
                      "startup_timeout_s": 0.1, "jpeg_quality": 80}
            with (patch.object(worker, "rr", rr), patch.object(worker, "_open_mailbox", return_value=mailbox),
                  patch.object(worker, "_parent_alive", return_value=True),
                  patch.object(worker, "_write_status") as statuses,
                  patch.object(worker, "_exit_process", side_effect=SystemExit(2)),
                  patch.object(worker.socket, "create_connection", return_value=MagicMock()),
                  self.assertRaises(SystemExit) as result):
                worker.run_worker(config)
            self.assertEqual(result.exception.code, 2)
            self.assertEqual(mailbox.read.call_count, 1)
            recording.flush.assert_called_once()
            self.assertEqual(len([key for key, _, _ in rr.calls if key == "observation.left_fixed"]), 1)
            self.assertTrue(any((call.kwargs.get("phase") or (call.args[1] if len(call.args) > 1 else None)) == "failed"
                                for call in statuses.call_args_list))


if __name__ == "__main__":
    unittest.main(verbosity=2)
