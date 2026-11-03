"""Append-only raw capture. Each .pkl contains length/CRC-framed pickle records.

Read only captures you trust: pickle loading can execute Python code.
Completed episode descriptors are published only after fsync of their shards.
"""
from __future__ import annotations

import copy
import json
import logging
import os
import pickle
import queue
import struct
import threading
import time
import zlib
from pathlib import Path

MAGIC = b"EVOPKL1\n"
RECORD_HEADER = struct.Struct("<QI")


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def iter_records(path: Path):
    """Reject corrupt/truncated records rather than silently shortening an episode."""
    with Path(path).open("rb") as stream:
        if stream.read(len(MAGIC)) != MAGIC:
            raise ValueError(f"Not an Evo PKL shard: {path}")
        while True:
            header = stream.read(RECORD_HEADER.size)
            if not header:
                return
            if len(header) != RECORD_HEADER.size:
                raise ValueError(f"Truncated record header: {path}")
            length, checksum = RECORD_HEADER.unpack(header)
            remaining = os.fstat(stream.fileno()).st_size - stream.tell()
            if length > remaining:
                raise ValueError(f"Truncated record body: {path}")
            payload = stream.read(length)
            if zlib.crc32(payload) != checksum:
                raise ValueError(f"CRC mismatch: {path}")
            yield pickle.loads(payload)


class PklDataset:
    """Small adapter for this repository's recording_loop and record entrypoint.

    add_frame snapshots arrays on the producer; serialization/fsync run on one
    background thread. Queue capacity counts complete three-camera samples.
    """

    @classmethod
    def create(cls, repo_id, fps, features, root=None, robot_type=None,
               use_videos=True, vcodec="libsvtav1", queue_frames=30,
               shard_mb=1024, sync_seconds=1.0, **unused):
        from lerobot.datasets.lerobot_dataset import VALID_VIDEO_CODECS, LeRobotDatasetMetadata
        from lerobot.datasets.utils import validate_frame

        if fps != 30 or not use_videos:
            raise ValueError("PKL capture currently requires fps=30 and video=true")
        if queue_frames < 1 or shard_mb < 1 or sync_seconds <= 0:
            raise ValueError("Invalid PKL queue/shard/sync configuration")
        if vcodec not in VALID_VIDEO_CODECS:
            raise ValueError(f"Invalid vcodec '{vcodec}'. Must be one of: {sorted(VALID_VIDEO_CODECS)}")
        meta = LeRobotDatasetMetadata.create(
            repo_id=repo_id, fps=fps, features=features, root=root,
            robot_type=robot_type, use_videos=True,
        )
        return cls(meta, vcodec, queue_frames, shard_mb, sync_seconds, validate_frame)

    def __init__(self, meta, vcodec="libsvtav1", queue_frames=30,
                 shard_mb=1024, sync_seconds=1.0, validate_frame=None):
        self.meta = meta
        self.root = Path(meta.root)
        self.repo_id = meta.repo_id
        self.features = meta.features
        self.fps = meta.fps
        self.num_episodes = 0
        self.image_writer = None
        self._validate = validate_frame
        self._queue = queue.Queue(maxsize=queue_frames)
        self._shard_bytes = int(shard_mb * 1024 * 1024)
        self._sync_seconds = sync_seconds
        self._error = None
        self._closed = False
        self._finish_requested = threading.Event()
        self._attempt = 0
        self._frame_count = 0
        self._queue_warning_t = 0.0
        (self.root / "episodes").mkdir()
        with (self.root / "capture.pkl").open("xb") as stream:
            pickle.dump({
                "format": "evo-pkl-v1", "fps": self.fps,
                "repo_id": self.repo_id, "robot_type": meta.info["robot_type"],
                "features": self.features, "vcodec": vcodec,
                "timestamp_mode": "frame_index/fps",
            }, stream, protocol=5)
            stream.flush()
            os.fsync(stream.fileno())
        self._thread = threading.Thread(target=self._run, name="pkl-writer", daemon=False)
        self._thread.start()

    def _check(self):
        if self._error is not None:
            raise RuntimeError("PKL writer failed; capture must stop") from self._error

    def _put(self, item):
        if self._closed:
            raise RuntimeError("PKL capture is already closed")
        while True:
            self._check()
            try:
                self._queue.put(item, timeout=0.1)
                self._check()
                return
            except queue.Full:
                now = time.monotonic()
                if now - self._queue_warning_t >= 5:
                    logging.warning("PKL queue full: disk writer cannot keep up; capture is waiting")
                    self._queue_warning_t = now

    def add_frame(self, frame):
        if self._validate is not None:
            self._validate(frame, self.features)
        snapshot = copy.deepcopy(frame)
        self._put(("frame", self._attempt, {
            "frame_index": self._frame_count,
            "timestamp": self._frame_count / self.fps,
            "enqueue_monotonic_ns": time.monotonic_ns(),
            "frame": snapshot,
        }))
        self._frame_count += 1

    def save_episode(self, extra_episode_metadata=None, **unused):
        if not self._frame_count:
            raise ValueError("Cannot commit an empty episode")
        self._put(("commit", self._attempt, {
            "episode_index": self.num_episodes, "frames": self._frame_count,
            "metadata": copy.deepcopy(extra_episode_metadata or {}),
        }))
        self.num_episodes += 1
        self._attempt += 1
        self._frame_count = 0

    def clear_episode_buffer(self, **unused):
        self._put(("abort", self._attempt, {}))
        self._attempt += 1
        self._frame_count = 0

    def finalize(self):
        if self._closed:
            self._check()
            return
        self._closed = True
        self._finish_requested.set()
        self._thread.join()
        self._check()
        atomic_json(self.root / "capture.closed.json", {
            "committed_episodes": self.num_episodes,
            "uncommitted_frames": self._frame_count,
        })
        self.meta._close_writer()

    def _run(self):
        stream = None
        folder = None
        shards = []
        written = 0
        active_attempt = None
        last_sync = time.monotonic()

        def finish_shard():
            nonlocal stream
            if stream is not None:
                stream.flush()
                os.fsync(stream.fileno())
                stream.close()
                stream = None

        try:
            while True:
                try:
                    kind, attempt, payload = self._queue.get(timeout=0.1)
                except queue.Empty:
                    if self._finish_requested.is_set():
                        finish_shard()
                        if folder is not None:
                            atomic_json(folder / "incomplete.json", {"frames": written})
                        return
                    if stream is not None and time.monotonic() - last_sync >= self._sync_seconds:
                        stream.flush()
                        os.fsync(stream.fileno())
                        last_sync = time.monotonic()
                    continue
                try:
                    if kind == "frame":
                        if folder is None:
                            active_attempt = attempt
                            folder = self.root / "episodes" / f"attempt-{attempt:06d}"
                            folder.mkdir()
                            shards = []
                            written = 0
                        if active_attempt != attempt or payload["frame_index"] != written:
                            raise ValueError("PKL writer frame order mismatch")
                        body = pickle.dumps(payload, protocol=5)
                        if stream is not None and stream.tell() + len(body) + RECORD_HEADER.size > self._shard_bytes:
                            finish_shard()
                        if stream is None:
                            name = f"part-{len(shards):06d}.pkl"
                            stream = (folder / name).open("xb", buffering=1024 * 1024)
                            stream.write(MAGIC)
                            shards.append(name)
                        stream.write(RECORD_HEADER.pack(len(body), zlib.crc32(body)))
                        stream.write(body)
                        written += 1
                        if time.monotonic() - last_sync >= self._sync_seconds:
                            stream.flush()
                            os.fsync(stream.fileno())
                            last_sync = time.monotonic()
                    elif kind in ("commit", "abort"):
                        if folder is not None and active_attempt != attempt:
                            raise ValueError("PKL writer episode order mismatch")
                        finish_shard()
                        if kind == "commit":
                            if folder is None or written != payload["frames"]:
                                raise ValueError("PKL episode frame count mismatch")
                            atomic_json(folder / "ready.json", {**payload, "shards": shards})
                        elif folder is not None:
                            atomic_json(folder / "discarded.json", {"frames": written})
                        folder = None
                        active_attempt = None
                        written = 0
                        shards = []
                    else:
                        raise ValueError(f"Unknown PKL command: {kind}")
                finally:
                    self._queue.task_done()
        except BaseException as error:
            self._error = error
            logging.exception("PKL background writer failed")
        finally:
            if stream is not None:
                stream.close()
