"""Crash-safe assembly of immutable native LeRobot episode packages.

Packages remain untouched. Numeric indices and their statistics are rewritten;
encoded videos are linked or copied, never encoded again. Checkpoint files live
outside the resulting training dataset.
"""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from resume_common import durable_mkdir

VERSION = 1
STATE_NAME = "assembly-state.json"


def _no_symlinks(path: Path) -> Path:
    path = Path(os.path.abspath(path))
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Symlink is not allowed: {candidate}")
    if path.is_dir():
        for candidate in path.rglob("*"):
            if candidate.is_symlink():
                raise ValueError(f"Symlink is not allowed: {candidate}")
    return path


def _digest(path: Path) -> dict:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while data := stream.read(4 * 1024 * 1024):
            digest.update(data)
            size += len(data)
    return {"size": size, "sha256": digest.hexdigest()}


def _sync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: dict) -> None:
    durable_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=False, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    _sync_directory(path.parent)


def _relative(root: Path, name: str) -> Path:
    relative = PurePosixPath(name)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid dataset path: {name}")
    result = root.joinpath(*relative.parts)
    _no_symlinks(result)
    if not result.is_relative_to(root):
        raise ValueError("Dataset path escapes its root")
    return result


@contextmanager
def _lock(checkpoint_dir: Path):
    durable_mkdir(checkpoint_dir)
    path = checkpoint_dir / "assembly.lock"
    _no_symlinks(path)
    with path.open("a+b") as stream:
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("Another process is assembling this dataset") from error
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(stream, fcntl.LOCK_UN)


def _read_packages(packages, info):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

    sources = []
    schemas = []
    signatures = []
    for root in packages:
        root = _no_symlinks(Path(root))
        native_info = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))
        if native_info.get("fps") != info.get("fps", 30):
            raise ValueError(f"Package FPS mismatch: {root}")
        if sources and native_info["features"] != sources[0]["info"]["features"]:
            raise ValueError("Package feature schemas differ")
        meta_paths = sorted((root / "meta/episodes").rglob("*.parquet"))
        rows = []
        for path in meta_paths:
            table = pq.ParquetFile(path).read()
            schemas.append(table.schema)
            rows.extend(table.to_pylist())
        rows.sort(key=lambda row: row["episode_index"])
        if not rows or len(rows) != native_info["total_episodes"]:
            raise ValueError(f"Package episode counts differ: {root}")
        if [row["episode_index"] for row in rows] != list(range(len(rows))):
            raise ValueError(f"Package episodes are not contiguous: {root}")
        if sum(row["length"] for row in rows) != native_info["total_frames"]:
            raise ValueError(f"Package frame counts differ: {root}")
        tasks = pd.read_parquet(root / "meta/tasks.parquet")
        task_lookup = {int(row["task_index"]): str(task) for task, row in tasks.iterrows()}
        if len(task_lookup) != len(tasks):
            raise ValueError("Package task indices are not unique")
        files = {}
        for top in ("data", "meta", "videos"):
            folder = root / top
            if folder.exists():
                for path in sorted(folder.rglob("*")):
                    if path.is_file():
                        files[path.relative_to(root).as_posix()] = _digest(path)
        signatures.append(files)
        sources.append({"root": root, "info": native_info, "episodes": rows,
                        "tasks": task_lookup, "files": files})
    if not sources:
        raise ValueError("No native episode packages to assemble")
    schema = pa.unify_schemas(schemas)
    fingerprint = hashlib.sha256(json.dumps(
        {"version": VERSION, "packages": signatures, "capture": {
            key: info.get(key) for key in ("fps", "features", "robot_type", "vcodec")}},
        sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    return sources, schema, fingerprint


class _Artifacts:
    def __init__(self, dataset_dir: Path, checkpoint_dir: Path, fingerprint: str):
        self.root = dataset_dir
        self.path = checkpoint_dir / STATE_NAME
        durable_mkdir(self.root)
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
            if (self.state.get("version") != VERSION
                    or self.state.get("fingerprint") != fingerprint
                    or self.state.get("dataset") != str(self.root)):
                raise ValueError("Assembly checkpoint belongs to different input or output")
        else:
            if any(path.is_file() for path in self.root.rglob("*")):
                raise ValueError("Refusing to adopt a nonempty dataset without its assembly checkpoint")
            self.state = {"version": VERSION, "fingerprint": fingerprint,
                          "dataset": str(self.root), "artifacts": {}, "pending": {}}
            self._save()
        for temporary in self.state.get("pending", {}).values():
            path = _relative(self.root, temporary)
            if path.exists():
                if not path.is_file() or not path.name.startswith(".assembly-"):
                    raise ValueError("Invalid assembly temporary file")
                path.unlink()
                _sync_directory(path.parent)
        self.state["pending"] = {}
        self.state["complete"] = False
        self._save()

    def _save(self):
        _atomic_json(self.path, self.state)

    def put(self, name: str, builder, expected=None):
        target = _relative(self.root, name)
        saved = self.state["artifacts"].get(name)
        if target.is_file():
            actual = _digest(target)
            if actual == (expected if expected is not None else saved):
                if actual != saved:
                    self.state["artifacts"][name] = actual
                    self._save()
                return
        elif target.exists():
            raise ValueError(f"Artifact target is not a file: {target}")
        durable_mkdir(target.parent)
        temporary = target.with_name(f".assembly-{uuid.uuid4().hex}.tmp")
        self.state["pending"][name] = temporary.relative_to(self.root).as_posix()
        self._save()
        builder(temporary)
        # Windows FlushFileBuffers requires a writable handle; POSIX fsync also
        # accepts a read-only descriptor, keeping read-only package links usable.
        with temporary.open("r+b" if os.name == "nt" else "rb") as stream:
            os.fsync(stream.fileno())
        actual = _digest(temporary)
        if expected is not None and actual != expected:
            raise ValueError(f"Source changed while assembling: {name}")
        if target.is_file() and _digest(target) == actual:
            temporary.unlink()
        else:
            os.replace(temporary, target)
        _sync_directory(target.parent)
        self.state["artifacts"][name] = actual
        self.state["pending"].pop(name, None)
        self._save()

    def link(self, name: str, source: Path, expected):
        def builder(temporary):
            try:
                os.link(source, temporary)
            except OSError as error:
                if error.errno not in (errno.EXDEV, errno.EPERM, errno.EACCES,
                                       errno.ENOTSUP, errno.EMLINK):
                    raise
                shutil.copyfile(source, temporary)
        self.put(name, builder, expected)

    def finish(self, expected_names):
        actual_names = {p.relative_to(self.root).as_posix()
                        for p in self.root.rglob("*") if p.is_file()}
        if actual_names != expected_names:
            raise ValueError("Unexpected or missing assembled files: "
                             + repr(sorted(actual_names ^ expected_names)))
        self.state["complete"] = True
        self._save()
        for directory in sorted((p for p in self.root.rglob("*") if p.is_dir()),
                                key=lambda p: len(p.parts), reverse=True):
            _sync_directory(directory)
        _sync_directory(self.root)
        _sync_directory(self.root.parent)


def assemble_episode_packages(packages: list[Path], dataset_dir: Path, info: dict,
                              *, checkpoint_dir: Path, progress_factory=None) -> Path:
    """Assemble immutable single- or multi-episode native packages, resumably.

The caller owns the workspace and final publication/verification. Repeating the
call verifies existing artifacts and preserves their inode/mtime when valid.
Any incomplete artifact is recreated from its immutable package. No video is
decoded or re-encoded by this function.
"""
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
    from lerobot.datasets.utils import serialize_dict, unflatten_dict

    dataset_dir = _no_symlinks(Path(dataset_dir))
    checkpoint_dir = _no_symlinks(Path(checkpoint_dir))
    if (dataset_dir == checkpoint_dir or dataset_dir in checkpoint_dir.parents
            or checkpoint_dir in dataset_dir.parents):
        raise ValueError("Assembly checkpoint and dataset directories must be separate")
    for package in packages:
        package = _no_symlinks(Path(package))
        for working in (dataset_dir, checkpoint_dir):
            if working == package or working in package.parents or package in working.parents:
                raise ValueError("Assembly workspace and immutable packages must not be nested")
    with _lock(checkpoint_dir):
        sources, metadata_schema, fingerprint = _read_packages(packages, info)
        artifacts = _Artifacts(dataset_dir, checkpoint_dir, fingerprint)
        native_info = copy.deepcopy(sources[0]["info"])
        chunks_size = int(native_info["chunks_size"])
        if chunks_size <= 0:
            raise ValueError("Invalid chunks_size")
        video_keys = [key for key, feature in native_info["features"].items()
                      if feature["dtype"] == "video"]
        total_episodes = sum(len(source["episodes"]) for source in sources)
        bar = (progress_factory(total=total_episodes, desc="Assemble native dataset", unit="episode")
               if progress_factory else None)
        tasks = {}
        all_stats = []
        physical_videos = {}
        expected_names = set()
        frame_offset = 0
        episode_index = 0
        try:
            for package_index, source in enumerate(sources):
                data_cache = {}
                package_info = source["info"]
                for original in source["episodes"]:
                    row = dict(original)
                    length = int(row["length"])
                    chunk_index, file_index = divmod(episode_index, chunks_size)
                    src_data_name = package_info["data_path"].format(
                        chunk_index=row["data/chunk_index"], file_index=row["data/file_index"])
                    src_data = _relative(source["root"], src_data_name)
                    if src_data_name not in data_cache:
                        data_cache[src_data_name] = pq.ParquetFile(src_data).read()
                    table = data_cache[src_data_name]
                    mask = pc.and_(pc.equal(table["episode_index"], row["episode_index"]),
                                   pc.and_(pc.greater_equal(table["index"], row["dataset_from_index"]),
                                           pc.less(table["index"], row["dataset_to_index"])))
                    table = table.filter(mask)
                    if table.num_rows != length or length <= 0:
                        raise ValueError("Episode rows do not match its package metadata")
                    old_task_ids = table["task_index"].to_numpy()
                    local_to_global = {}
                    for task_id in dict.fromkeys(int(value) for value in old_task_ids):
                        if task_id not in source["tasks"]:
                            raise ValueError("Episode references an unknown task")
                        task = source["tasks"][task_id]
                        if task not in tasks:
                            tasks[task] = len(tasks)
                        local_to_global[task_id] = tasks[task]
                    numeric = {
                        "episode_index": np.full(length, episode_index, dtype=np.int64),
                        "index": np.arange(frame_offset, frame_offset + length, dtype=np.int64),
                        "task_index": np.array([local_to_global[int(value)] for value in old_task_ids],
                                               dtype=np.int64),
                    }
                    for name, values in numeric.items():
                        column_index = table.schema.get_field_index(name)
                        field = table.schema.field(column_index)
                        table = table.set_column(column_index, field, pa.array(values, type=field.type))
                    stats = unflatten_dict({name[len("stats/"):]: np.asarray(value)
                                            for name, value in original.items()
                                            if name.startswith("stats/") and value is not None})
                    if not stats:
                        raise ValueError("Package episode statistics are missing")
                    for name, values in numeric.items():
                        stats[name] = get_feature_stats(values, axis=0, keepdims=True)
                        for stat_name, value in stats[name].items():
                            row[f"stats/{name}/{stat_name}"] = value.tolist()
                    all_stats.append(stats)
                    row.update({"episode_index": episode_index,
                                "dataset_from_index": frame_offset,
                                "dataset_to_index": frame_offset + length,
                                "data/chunk_index": chunk_index, "data/file_index": file_index,
                                "meta/episodes/chunk_index": chunk_index,
                                "meta/episodes/file_index": file_index})
                    data_name = native_info["data_path"].format(chunk_index=chunk_index,
                                                               file_index=file_index)
                    expected_names.add(data_name)
                    artifacts.put(data_name, lambda temporary, table=table:
                                  pq.write_table(table, temporary, compression="snappy", use_dictionary=True))
                    for video_key in video_keys:
                        src_name = package_info["video_path"].format(
                            video_key=video_key, chunk_index=original[f"videos/{video_key}/chunk_index"],
                            file_index=original[f"videos/{video_key}/file_index"])
                        video_identity = (package_index, video_key, src_name)
                        if video_identity not in physical_videos:
                            video_name = native_info["video_path"].format(
                                video_key=video_key, chunk_index=chunk_index, file_index=file_index)
                            src_path = _relative(source["root"], src_name)
                            expected = source["files"].get(src_name)
                            if expected is None:
                                raise ValueError("Episode references a missing video")
                            artifacts.link(video_name, src_path, expected)
                            physical_videos[video_identity] = (chunk_index, file_index, video_name)
                        video_chunk, video_file, video_name = physical_videos[video_identity]
                        expected_names.add(video_name)
                        row[f"videos/{video_key}/chunk_index"] = video_chunk
                        row[f"videos/{video_key}/file_index"] = video_file
                    meta_name = f"meta/episodes/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
                    expected_names.add(meta_name)
                    meta_table = pa.Table.from_pylist([row], schema=metadata_schema)
                    artifacts.put(meta_name, lambda temporary, table=meta_table:
                                  pq.write_table(table, temporary, compression="snappy", use_dictionary=True))
                    frame_offset += length
                    episode_index += 1
                    if bar is not None:
                        bar.update(1)
            native_info.update(total_episodes=total_episodes, total_frames=frame_offset,
                               total_tasks=len(tasks), splits={"train": f"0:{total_episodes}"})
            task_table = pd.DataFrame({"task_index": list(tasks.values())}, index=list(tasks))
            expected_names.update({"meta/info.json", "meta/stats.json", "meta/tasks.parquet"})
            artifacts.put("meta/tasks.parquet", lambda temporary: task_table.to_parquet(temporary))
            for name, value in (("meta/info.json", native_info),
                                ("meta/stats.json", serialize_dict(aggregate_stats(all_stats)))):
                artifacts.put(name, lambda temporary, value=value: temporary.write_text(
                    json.dumps(value, ensure_ascii=False, indent=4), encoding="utf-8"))
            artifacts.finish(expected_names)
        finally:
            if bar is not None:
                bar.close()
        return dataset_dir
