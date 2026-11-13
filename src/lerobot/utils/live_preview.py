"""Bounded, best-effort preview transport isolated from robot control.

The SDK runs in a subprocess. A fixed-size mailbox contains only the newest
snapshot; a slow or disconnected viewer cannot create a Python/SDK frame queue
in the acquisition process.
"""
from __future__ import annotations

import json
import logging
import math
import mmap
import numbers
import os
import pickle
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path

import numpy as np

MAGIC = b"EVOPREV1"
HEADER = struct.Struct("<8sQI")
MAX_SCALARS = 1024
MAX_IMAGES = 16


def _estimate_bytes(value, depth=0):
    if depth > 8:
        raise ValueError("Preview data is nested too deeply")
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("Object arrays cannot be previewed")
        return value.nbytes + 512
    if isinstance(value, dict):
        if len(value) > 2048:
            raise ValueError("Too many preview fields")
        return 256 + sum(_estimate_bytes(key, depth + 1) + _estimate_bytes(item, depth + 1)
                         for key, item in value.items())
    if isinstance(value, (tuple, list)):
        if len(value) > 2048:
            raise ValueError("Too many preview values")
        return 256 + sum(_estimate_bytes(item, depth + 1) for item in value)
    if isinstance(value, (str, bytes)):
        return len(value) * 4 + 64
    if value is None or isinstance(value, (numbers.Number, np.generic)):
        return 64
    raise ValueError("Unsupported preview value")


def _try_file_lock(descriptor):
    if os.name == "posix":
        import fcntl
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
    else:
        import msvcrt
        os.lseek(descriptor, 0, os.SEEK_SET)
        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
    return True


def _unlock_file(descriptor):
    if os.name == "posix":
        import fcntl
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    else:
        import msvcrt
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)


class LatestFrameMailbox:
    """One fixed mmap slot; producer and consumer never wait for each other."""

    def __init__(self, path, capacity, create=False):
        if type(capacity) is not int or capacity < 1024 or capacity > 256 * 1024 * 1024:
            raise ValueError("Preview mailbox capacity must be between 1 KiB and 256 MiB")
        self.path = Path(path)
        self.capacity = capacity
        self._thread_lock = threading.Lock()
        self._closed = False
        flags = os.O_RDWR | (os.O_CREAT | os.O_EXCL if create else 0)
        self._descriptor = os.open(str(path), flags, 0o600)
        try:
            if create:
                os.ftruncate(self._descriptor, HEADER.size + capacity)
            if os.fstat(self._descriptor).st_size != HEADER.size + capacity:
                raise ValueError("Unexpected preview mailbox size")
            self._mapping = mmap.mmap(self._descriptor, HEADER.size + capacity)
            if create:
                self._mapping[:HEADER.size] = HEADER.pack(MAGIC, 0, 0)
        except BaseException:
            os.close(self._descriptor)
            raise

    def publish(self, sequence, packet):
        if self._closed or not self._thread_lock.acquire(blocking=False):
            return False
        locked = False
        try:
            if self._closed:
                return False
            if _estimate_bytes(packet) > self.capacity:
                return False
            if not _try_file_lock(self._descriptor):
                return False
            locked = True
            payload = pickle.dumps(packet, protocol=5)
            if len(payload) > self.capacity:
                return False
            self._mapping[HEADER.size:HEADER.size + len(payload)] = payload
            self._mapping[:HEADER.size] = HEADER.pack(MAGIC, sequence, len(payload))
            return True
        except (OSError, MemoryError, ValueError, TypeError, pickle.PickleError, struct.error):
            return False
        finally:
            if locked:
                _unlock_file(self._descriptor)
            self._thread_lock.release()

    def read(self, last_sequence):
        if self._closed or not self._thread_lock.acquire(blocking=False):
            return None
        locked = False
        try:
            if self._closed:
                return None
            if not _try_file_lock(self._descriptor):
                return None
            locked = True
            magic, sequence, length = HEADER.unpack(self._mapping[:HEADER.size])
            if magic != MAGIC or sequence <= last_sequence or not 0 < length <= self.capacity:
                return None
            payload = self._mapping[HEADER.size:HEADER.size + length]
        finally:
            if locked:
                _unlock_file(self._descriptor)
            self._thread_lock.release()
        # No file lock is held while unpickling or sending to the SDK.
        return sequence, pickle.loads(payload)

    def close(self):
        with self._thread_lock:
            if self._closed:
                return
            self._closed = True
            self._mapping.close()
            os.close(self._descriptor)


def _number_env(name, default, minimum, maximum):
    value = float(os.environ.get(name, default))
    if not math.isfinite(value) or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


def _process_group_rss(pid):
    if os.name != "posix" or not Path("/proc").is_dir():
        return 0
    pages = 0
    for path in Path("/proc").iterdir():
        if not path.name.isdecimal():
            continue
        try:
            fields = (path / "stat").read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == pid:
                pages += int(fields[21])
        except (OSError, ValueError, IndexError):
            continue
    return pages * os.sysconf("SC_PAGE_SIZE")


def _process_start_ticks(pid):
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19])
    except (OSError, ValueError, IndexError):
        return None


def _terminate_owned(process):
    if process is None:
        return
    # Every process here was started in its own session. Killing that session's
    # group also reaps the rerun CLI's native child, never an existing viewer.
    def send(signum, hard=False):
        try:
            if os.name == "posix":
                os.killpg(process.pid, signum)
            elif hard:
                process.kill()
            else:
                process.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if process.poll() is None or _process_group_rss(process.pid):
        send(signal.SIGTERM)
    try:
        process.wait(timeout=0.75)
    except subprocess.TimeoutExpired:
        send(getattr(signal, "SIGKILL", signal.SIGTERM), hard=True)
        with suppress(subprocess.TimeoutExpired):
            process.wait(timeout=0.75)
    if _process_group_rss(process.pid):
        send(getattr(signal, "SIGKILL", signal.SIGTERM), hard=True)


class LivePreviewSession:
    """Own the preview worker/viewer, with best-effort latest-frame submission."""

    def __init__(self, session_name, ip=None, port=None, *, capacity=None, max_fps=None,
                 worker_command=None, viewer_command=None, monitor=True):
        if (ip is None) != (port is None):
            raise ValueError("display_ip and display_port must be supplied together")
        self.capacity = int(capacity if capacity is not None else
                            _number_env("LEROBOT_DISPLAY_MAX_FRAME_MB", 16, 1, 256) * 1024 * 1024)
        self.max_fps = float(max_fps if max_fps is not None else
                             _number_env("LEROBOT_DISPLAY_FPS", 30, 1, 120))
        if not math.isfinite(self.max_fps) or not 0 < self.max_fps <= 100000:
            raise ValueError("Preview FPS must be positive")
        self._worker_limit = _number_env("LEROBOT_DISPLAY_WORKER_MEMORY_MB", 512, 128, 8192) * 1024 * 1024
        self._viewer_limit = _number_env("LEROBOT_DISPLAY_VIEWER_MEMORY_MB", 2048, 256, 16384) * 1024 * 1024
        self._flush_timeout = _number_env("LEROBOT_DISPLAY_FLUSH_TIMEOUT_S", 0.5, 0.05, 10)
        self._startup_timeout = _number_env("LEROBOT_DISPLAY_STARTUP_TIMEOUT_S", 15, 1, 120)
        self._closed = False
        self._disabled = False
        self._submitted = self._dropped = self._sequence = 0
        self._last_submit = float("-inf")
        # A half-frame of credit tolerates normal 30 Hz camera/scheduler
        # jitter. Comparing each arrival with the previous arrival would
        # otherwise drop roughly every second frame at the nominal FPS.
        self._rate_tokens = 1.5
        self._last_credit = time.monotonic()
        self._submit_lock = threading.Lock()
        self._close_lock = threading.Lock()
        self._stop = threading.Event()
        self._watchdog = None
        self._worker = self._viewer = None
        self._temporary = tempfile.TemporaryDirectory(prefix="lerobot-preview-")
        self.directory = Path(self._temporary.name)
        self._status_file = self.directory / "status.json"
        self.mailbox = LatestFrameMailbox(self.directory / "latest.mmap", self.capacity, create=True)
        try:
            environment = dict(os.environ)
            environment.setdefault("RERUN_FLUSH_NUM_BYTES", "1048576")
            if ip is None:
                with socket.socket() as listener:
                    listener.bind(("127.0.0.1", 0))
                    port = listener.getsockname()[1]
                ip = "127.0.0.1"
                cli = Path(sys.executable).parent / ("rerun.exe" if os.name == "nt" else "rerun")
                executable = str(cli) if cli.is_file() else shutil.which("rerun")
                if viewer_command is None and executable is None:
                    raise RuntimeError("Rerun viewer executable is unavailable")
                command = list(viewer_command) if viewer_command is not None else [
                    executable, "--bind", "127.0.0.1", "--port", str(port),
                    "--memory-limit", os.getenv("LEROBOT_RERUN_MEMORY_LIMIT", "512MiB"),
                    "--server-memory-limit", os.getenv("LEROBOT_RERUN_SERVER_MEMORY_LIMIT", "32MiB"),
                    "--expect-data-soon", "--hide-welcome-screen",
                ]
                self._viewer = subprocess.Popen(command, env=environment, stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=os.name == "posix")
            host = f"[{ip}]" if ":" in ip and not ip.startswith("[") else ip
            config = {"mailbox": str(self.mailbox.path), "capacity": self.capacity,
                      "status_file": str(self._status_file), "url": f"rerun+http://{host}:{port}/proxy",
                      "session_name": session_name, "recording_id": str(uuid.uuid4()),
                      "parent_pid": os.getpid(), "flush_timeout_s": self._flush_timeout,
                      "viewer_pid": self._viewer.pid if self._viewer else None,
                      "viewer_start_ticks": _process_start_ticks(self._viewer.pid) if self._viewer else None,
                      "startup_timeout_s": self._startup_timeout,
                      "jpeg_quality": int(_number_env("LEROBOT_DISPLAY_JPEG_QUALITY", 80, 1, 100))}
            self._config_path = self.directory / "config.json"
            self._config_path.write_text(json.dumps(config), encoding="utf-8")
            command = (list(worker_command) if worker_command is not None else
                       [sys.executable, "-m", "lerobot.utils.live_preview_worker"])
            self._worker = subprocess.Popen([*command, "--config", str(self._config_path)], env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                start_new_session=os.name == "posix")
            self._started = time.monotonic()
            if monitor:
                self._watchdog = threading.Thread(target=self._monitor, name="live-preview-watchdog", daemon=True)
                self._watchdog.start()
        except BaseException:
            self.close()
            raise

    @property
    def stats(self):
        return {"submitted": self._submitted, "dropped": self._dropped,
                "capacity": self.capacity, "disabled": self._disabled,
                "worker_pid": self._worker.pid if self._worker else None,
                "viewer_pid": self._viewer.pid if self._viewer else None}

    def submit(self, observation=None, action=None, compress_images=True):
        if self._closed or self._disabled or self._worker is None or self._worker.poll() is not None:
            self._dropped += 1
            return False
        now = time.monotonic()
        if not self._submit_lock.acquire(blocking=False):
            self._dropped += 1
            return False
        try:
            self._rate_tokens = min(1.5, self._rate_tokens + (now - self._last_credit) * self.max_fps)
            self._last_credit = now
            if self._rate_tokens < 1:
                self._dropped += 1
                return False
            scalars = images = 0
            values = []
            for source, is_action in ((observation, False), (action, True)):
                filtered = {}
                if source:
                    if len(source) > MAX_SCALARS:
                        self._dropped += 1
                        return False
                    for key, value in source.items():
                        if isinstance(value, np.ndarray):
                            if not is_action and value.ndim in (2, 3):
                                images += 1
                            else:
                                scalars += value.size
                            filtered[str(key)] = value
                        elif isinstance(value, (numbers.Real, np.integer, np.floating)):
                            scalars += 1
                            filtered[str(key)] = value
                values.append(filtered)
            if scalars > MAX_SCALARS or images > MAX_IMAGES:
                self._dropped += 1
                return False
            self._sequence += 1
            result = self.mailbox.publish(self._sequence, {"observation": values[0], "action": values[1],
                "compress_images": bool(compress_images), "timestamp": time.time()})
            if result:
                self._last_submit = now
                self._rate_tokens -= 1
                self._submitted += 1
            else:
                self._dropped += 1
            return result
        finally:
            self._submit_lock.release()

    def _status(self):
        try:
            return json.loads(self._status_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _disable(self, reason):
        if self._disabled or self._closed:
            return
        self._disabled = True
        logging.warning("Live preview stopped (%s). Robot control and data recording continue.", reason)
        _terminate_owned(self._worker)
        _terminate_owned(self._viewer)

    def _monitor(self):
        while not self._stop.wait(0.25):
            if self._closed or self._disabled:
                return
            if self._viewer is not None and self._viewer.poll() is not None:
                self._disable("viewer was closed or could not start")
                return
            status = self._status()
            if self._worker.poll() is not None:
                self._disable(str(status.get("error", "preview process exited"))[:500])
                return
            for process, limit, label in ((self._worker, self._worker_limit, "preview SDK"),
                                           (self._viewer, self._viewer_limit, "viewer")):
                if process is not None and _process_group_rss(process.pid) > limit:
                    self._disable(label + " exceeded the preview RAM budget")
                    return
            now = time.monotonic()
            if not status and now - self._started > self._startup_timeout + 5:
                self._disable("preview startup timed out")
                return
            if status and now - status.get("updated", self._started) > self._startup_timeout + 5:
                self._disable("preview process stopped responding")
                return
            if (status.get("phase") == "flushing"
                    and now - status.get("updated", now) > max(3, self._flush_timeout * 4)):
                self._disable("viewer stopped acknowledging preview frames")
                return
            if status.get("phase") == "failed":
                self._disable(str(status.get("error", "preview failed"))[:500])
                return

    def close(self):
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
            self._stop.set()
            _terminate_owned(self._worker)
            _terminate_owned(self._viewer)
            if self._watchdog is not None and self._watchdog is not threading.current_thread():
                self._watchdog.join(timeout=2)
            self.mailbox.close()
            self._temporary.cleanup()
