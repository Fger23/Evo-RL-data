"""A disposable Rerun client for live preview; never used to save robot data.

Only one bounded snapshot can enter the SDK before a successful flush. The
controller owns this process and can kill it if Rerun stalls inside native code.
No robot, processor, PyTorch, or camera modules are imported here.
"""
from __future__ import annotations

import argparse
import json
import math
import numbers
import os
import signal
import socket
import sys
import threading
import time
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

# Loaded only after the viewer's TCP endpoint is available. Tests may replace
# this binding with a fake SDK without loading the native Rerun extension.
rr = None
MAX_SCALARS = 1024
MAX_IMAGES = 16
MAX_ENTITY_PATH_BYTES = 1024


def _is_scalar(value):
    return isinstance(value, (numbers.Real, np.integer, np.floating, np.bool_)) or (
        isinstance(value, np.ndarray) and value.ndim == 0
    )


def _image_array(value):
    """Return HWC/gray input without modifying the caller's RGB channel order."""
    if value.ndim == 2:
        return value
    if value.ndim != 3:
        return None
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.transpose(value, (1, 2, 0))
    return value if value.shape[-1] in (1, 3, 4) else None


def _snapshot_entries(packet, capacity=None):
    """Validate an entire batch before its first SDK log call."""
    if not isinstance(packet, dict):
        raise ValueError("Preview packet must be a dictionary")
    entries = []
    scalar_count = image_count = array_bytes = 0
    for group, prefix in (("observation", "observation"), ("action", "action")):
        values = packet.get(group)
        if values is None:
            continue
        if not isinstance(values, dict):
            raise ValueError(f"Preview {group} must be a dictionary")
        for name, value in values.items():
            if value is None:
                continue
            name = str(name)
            key = name if name.startswith(prefix + ".") else prefix + "." + name
            if len(key.encode("utf-8")) > MAX_ENTITY_PATH_BYTES:
                raise ValueError("Preview entity path is too long")
            if isinstance(value, np.ndarray):
                if value.dtype.kind not in "buif":
                    raise ValueError(f"Unsupported preview array dtype: {value.dtype}")
                array_bytes += value.nbytes
                if capacity is not None and array_bytes > capacity:
                    raise ValueError("Preview array payload exceeds mailbox capacity")
            if _is_scalar(value):
                scalar_count += 1
                if scalar_count > MAX_SCALARS:
                    raise ValueError("Preview scalar limit exceeded")
                entries.append(("scalar", key, float(value)))
            elif isinstance(value, np.ndarray):
                image = _image_array(value) if group == "observation" else None
                if image is not None:
                    if image.shape[0] <= 0 or image.shape[1] <= 0:
                        raise ValueError("Preview image dimensions must be positive")
                    image_count += 1
                    if image_count > MAX_IMAGES:
                        raise ValueError("Preview image limit exceeded")
                    entries.append(("image", key, image))
                else:
                    scalar_count += value.size
                    if scalar_count > MAX_SCALARS:
                        raise ValueError("Preview scalar limit exceeded")
                    entries.extend(("scalar", f"{key}_{index}", float(item))
                                   for index, item in enumerate(value.flat))
            # Strings and other nonnumeric metadata do not produce plot data.
    return entries, scalar_count, image_count


def log_snapshot(rr, rec, sequence, packet, jpeg_quality=80):
    """Log one validated snapshot, without retaining it or mutating its arrays.

The caller must flush this recording before calling this function again.
Normal RGB8/grayscale preview images are JPEG-compressed. RGBA and non-U8
images retain their original representation because JPEG would lose semantics.
"""
    if not isinstance(sequence, numbers.Integral) or isinstance(sequence, bool) or sequence < 0:
        raise ValueError("Preview sequence must be a nonnegative integer")
    if type(jpeg_quality) is not int or not 1 <= jpeg_quality <= 100:
        raise ValueError("Preview JPEG quality must be between 1 and 100")
    if not isinstance(packet, dict):
        raise ValueError("Preview packet must be a dictionary")
    timestamp = float(packet.get("timestamp", time.time()))
    if not math.isfinite(timestamp):
        raise ValueError("Preview timestamp must be finite")
    entries, scalar_count, image_count = _snapshot_entries(packet)
    rr.set_time("frame", sequence=int(sequence), recording=rec)
    rr.set_time("time", timestamp=timestamp, recording=rec)
    compress = bool(packet.get("compress_images", True))
    for kind, key, value in entries:
        if kind == "scalar":
            rr.log(key, rr.Scalars(value), recording=rec)
        else:
            entity = rr.Image(value)
            channels = 1 if value.ndim == 2 else value.shape[-1]
            if compress and value.dtype == np.uint8 and channels in (1, 3):
                entity = entity.compress(jpeg_quality=jpeg_quality)
            # Rerun's default is temporal (static=False), with automatic
            # log_time as well as our explicit frame/time timelines.
            rr.log(key, entity, recording=rec)
    return {"scalars": scalar_count, "images": image_count}


def _write_status(path, phase, frames_sent, sequence, error=None):
    path = Path(path)
    state = {"phase": phase, "updated": time.monotonic(),
             "frames_sent": frames_sent, "seq": sequence}
    if error is not None:
        state["error"] = str(error)[:500]
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(state, ensure_ascii=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _parent_alive(pid):
    if os.name == "nt":
        # On Windows os.kill(pid, 0) can call TerminateProcess rather than
        # probe existence. Query a synchronization handle without signaling it.
        import ctypes
        from ctypes import wintypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.CloseHandle.restype = wintypes.BOOL
        handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE
        if not handle:
            return ctypes.get_last_error() == 5  # Access denied implies it exists.
        try:
            return kernel.WaitForSingleObject(handle, 0) == 258  # WAIT_TIMEOUT
        finally:
            kernel.CloseHandle(handle)
    if os.getppid() != pid:
        # A dead parent can remain a zombie (kill(pid, 0) still succeeds), or
        # its PID can be reused after this worker has been reparented.
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _viewer_identity_matches(config):
    if not sys.platform.startswith("linux"):
        return False
    pid = config.get("viewer_pid")
    token = config.get("viewer_start_ticks")
    if (type(pid) is not int or pid <= 1 or pid in (os.getpid(), config.get("parent_pid"))
            or token is None or isinstance(token, bool)):
        return False
    try:
        token = int(token)
        if token < 0:
            return False
        # comm can contain spaces and parentheses; fields after the final ')'
        # start at field 3, making starttime (field 22) offset 19.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return int(fields[19]) == token and os.getpgid(pid) == pid
    except (OSError, ValueError, IndexError, TypeError):
        return False


def _cleanup_orphan_viewer(config):
    """Stop only the Linux viewer process group whose creation token we own."""
    if not _viewer_identity_matches(config):
        return False
    pid = config["viewer_pid"]
    try:
        os.killpg(pid, signal.SIGTERM)
    except OSError:
        return False
    deadline = time.monotonic() + 0.5
    while time.monotonic() < deadline:
        if not _viewer_identity_matches(config):
            return True
        time.sleep(0.025)
    if _viewer_identity_matches(config):
        with suppress(OSError):
            os.killpg(pid, signal.SIGKILL)
    return True


def _parent_watchdog(config, stopped):
    """Keep orphan cleanup independent of a blocked native logging thread."""
    while not stopped.wait(0.25):
        try:
            alive = _parent_alive(config["parent_pid"])
        except Exception:
            alive = False
        if not alive:
            with suppress(BaseException):
                _cleanup_orphan_viewer(config)
            # Tests replace os._exit with a BaseException. Do not produce an
            # unhandled daemon-thread traceback or repeatedly invoke the mock.
            with suppress(BaseException):
                _exit_process(0)
            return


def _open_mailbox(config):
    from lerobot.utils.live_preview import LatestFrameMailbox

    return LatestFrameMailbox(config["mailbox"], config["capacity"], create=False)


def _exit_process(code):
    # In Rerun 0.26 a failed gRPC client's destructor may attempt an infinite
    # native flush. This process owns no capture files or robot devices; let
    # the OS release its mappings, buffers, and sockets without SDK shutdown.
    os._exit(code)


def _wait_for_endpoint(url, parent_pid, timeout_s, heartbeat):
    address = urlsplit(url)
    if not address.hostname:
        raise ValueError("Preview URL has no host")
    port = address.port or (443 if address.scheme.endswith("https") else 9876)
    deadline = time.monotonic() + timeout_s
    while _parent_alive(parent_pid):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Preview viewer did not become available before startup timeout")
        try:
            with socket.create_connection((address.hostname, port), timeout=min(0.2, remaining)):
                return True
        except OSError:
            heartbeat()
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
    return False


def run_worker(config):
    """Consume only the newest snapshot and allow at most one SDK batch in flight.

The process deliberately exits through os._exit on all terminal paths. Tests
can replace _exit_process with a function that raises SystemExit.
"""
    status_path = config.get("status_file")
    sequence = frames_sent = 0
    mailbox = None
    watchdog = None
    watchdog_stopped = threading.Event()
    try:
        if not status_path:
            raise ValueError("Preview status_file is required")
        capacity = config["capacity"]
        parent_pid = config["parent_pid"]
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("Invalid preview mailbox capacity")
        if type(parent_pid) is not int or parent_pid <= 0:
            raise ValueError("Invalid preview parent PID")
        flush_timeout = float(config.get("flush_timeout_s", 0.5))
        startup_timeout = float(config.get("startup_timeout_s", 15))
        quality = config.get("jpeg_quality", 80)
        if not math.isfinite(flush_timeout) or flush_timeout <= 0:
            raise ValueError("Invalid preview flush timeout")
        if not math.isfinite(startup_timeout) or startup_timeout <= 0:
            raise ValueError("Invalid preview startup timeout")
        watchdog = threading.Thread(target=_parent_watchdog, args=(config, watchdog_stopped),
                                    name="live-preview-parent-watchdog", daemon=True)
        watchdog.start()
        _write_status(status_path, "starting", frames_sent, sequence)
        last_status = time.monotonic()

        def starting_heartbeat():
            nonlocal last_status
            now = time.monotonic()
            if now - last_status >= 1:
                _write_status(status_path, "starting", frames_sent, sequence)
                last_status = now

        if not _wait_for_endpoint(config["url"], parent_pid, startup_timeout, starting_heartbeat):
            _cleanup_orphan_viewer(config)
            _exit_process(0)
        mailbox = _open_mailbox(config)
        sdk = rr
        if sdk is None:
            import rerun as sdk
        rec = sdk.RecordingStream(config.get("session_name", "lerobot_live_preview"),
                                  recording_id=config["recording_id"])
        sdk.connect_grpc(url=config["url"], recording=rec)
        _write_status(status_path, "ready", frames_sent, sequence)
        last_status = time.monotonic()
        while _parent_alive(parent_pid):
            newest = mailbox.read(sequence)
            if newest is None:
                if time.monotonic() - last_status >= 1:
                    _write_status(status_path, "idle", frames_sent, sequence)
                    last_status = time.monotonic()
                time.sleep(0.005)
                continue
            next_sequence, packet = newest
            if next_sequence <= sequence:
                time.sleep(0.005)
                continue
            _snapshot_entries(packet, capacity=capacity)
            sequence = int(next_sequence)
            _write_status(status_path, "flushing", frames_sent, sequence)
            log_snapshot(sdk, rec, sequence, packet, jpeg_quality=quality)
            rec.flush(timeout_sec=flush_timeout)
            frames_sent += 1
            _write_status(status_path, "idle", frames_sent, sequence)
            last_status = time.monotonic()
            # Do not keep a previous frame's large NumPy/Arrow owners alive
            # while waiting indefinitely for another camera update.
            del packet, newest
        _cleanup_orphan_viewer(config)
        _exit_process(0)
    except KeyboardInterrupt:
        with suppress(Exception):
            if not _parent_alive(config["parent_pid"]):
                _cleanup_orphan_viewer(config)
        _exit_process(0)
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"[:500]
        if status_path:
            with suppress(Exception):
                _write_status(status_path, "failed", frames_sent, sequence, error=detail)
        print("Live preview stopped: " + detail, file=sys.stderr, flush=True)
        with suppress(Exception):
            if not _parent_alive(config["parent_pid"]):
                _cleanup_orphan_viewer(config)
        _exit_process(2)
    finally:
        watchdog_stopped.set()
        if watchdog is not None and watchdog is not threading.current_thread():
            watchdog.join(timeout=1)
        if mailbox is not None:
            with suppress(Exception):
                mailbox.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
    except Exception as error:
        print(f"Invalid live preview configuration: {error}"[:500], file=sys.stderr, flush=True)
        _exit_process(2)
    run_worker(config)


if __name__ == "__main__":
    main()
