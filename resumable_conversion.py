"""Crash-safe offline conversion with immutable frame/video checkpoints.

Only a bounded, uncommitted chunk is repeated after a crash. Completed video
chunks and episode packages are never appended to or re-encoded on recovery.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import pickle
import re
import shutil
import uuid
import zlib
from fractions import Fraction
from pathlib import Path

from resume_common import (
    atomic_json,
    dataset_lock,
    durable_mkdir,
    file_manifest,
    reject_symlinks,
    sha256_file,
    sync_directory,
    sync_tree,
    verify_manifest,
)

FORMAT = "evo-conversion-resume-v1"
DEFAULT_CHECKPOINT_FRAMES = 300
EPISODE_MARKER = ".episode.complete.json"


def capture_identity(root, episodes):
    """Bind caches to closed source files, including inode/ctime change detection."""
    items = []
    for path in [root / "capture.pkl", root / "capture.closed.json"]:
        reject_symlinks(path)
        items.append({"path": path.relative_to(root).as_posix(), "sha256": sha256_file(path)})
    for folder, descriptor in episodes:
        ready = folder / "ready.json"
        items.append({"path": ready.relative_to(root).as_posix(), "sha256": sha256_file(ready)})
        for name in descriptor["shards"]:
            if not re.fullmatch(r"part-\d{6}\.pkl", name):
                raise ValueError("Invalid source shard name")
            path = reject_symlinks(folder / name)
            value = path.stat()
            items.append({"path": path.relative_to(root).as_posix(), "size": value.st_size,
                          "mtime_ns": value.st_mtime_ns, "ctime_ns": value.st_ctime_ns,
                          "inode": value.st_ino, "device": value.st_dev})
    value = {"root": str(root), "files": items}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest(), value


def _workspace(output, fingerprint, identity, chunk_frames):
    workspace = output.with_name(output.name + ".resume")
    reject_symlinks(workspace)
    if workspace.exists():
        state_path = workspace / "state.json"
        if not state_path.is_file():
            raise ValueError(f"Unrecognized resume directory; preserve it and choose another --output: {workspace}")
        state = json.loads(state_path.read_text())
        if state.get("format") != FORMAT or state.get("source_fingerprint") != fingerprint:
            raise ValueError("Resume checkpoints belong to another source or format; use another --output")
        if state.get("checkpoint_frames") != chunk_frames:
            raise ValueError(f"Resume requires --checkpoint-frames={state.get('checkpoint_frames')}")
    else:
        temporary = workspace.with_name(workspace.name + ".initializing-" + uuid.uuid4().hex)
        temporary.mkdir()
        atomic_json(temporary / "state.json", {
            "format": FORMAT, "source_fingerprint": fingerprint,
            "source_identity": identity, "checkpoint_frames": chunk_frames,
        })
        sync_tree(temporary)
        temporary.rename(workspace)
        sync_directory(workspace.parent)
    return workspace


def _quarantine(path):
    if path.exists():
        reject_symlinks(path)
        path.rename(path.with_name(path.name + ".invalid-" + uuid.uuid4().hex))
        sync_directory(path.parent)


def _discard_uncommitted(episode_dir, package_dir, index):
    """Reclaim only this engine's abandoned UUID transactions, under its lock."""
    for path in package_dir.glob(f"episode-{index:06d}.building-*"):
        if re.fullmatch(r"episode-\d{6}\.building-[0-9a-f]{32}", path.name):
            reject_symlinks(path)
            shutil.rmtree(path)
            sync_directory(path.parent)
    if not episode_dir.exists():
        return
    reject_symlinks(episode_dir)
    prepared = episode_dir / "prepared"
    for path in prepared.glob("chunk-*.preparing-*"):
        if re.fullmatch(r"chunk-\d{6}\.preparing-[0-9a-f]{32}", path.name):
            reject_symlinks(path)
            shutil.rmtree(path)
            sync_directory(path.parent)
    for top in (episode_dir / "encoded", episode_dir / "combined"):
        for path in top.rglob("*"):
            if (re.fullmatch(r".+\.encoding-[0-9a-f]{32}\.(mp4|ffconcat)", path.name)
                    or re.fullmatch(r"\..+\.json\.[0-9a-f]{32}\.partial", path.name)):
                reject_symlinks(path)
                if not path.is_file():
                    raise ValueError("Unexpected incomplete video transaction")
                path.unlink()
                sync_directory(path.parent)


def packets_from(folder, descriptor, cursor=None):
    """Read CRC-framed records from the last committed byte offset, not frame 0."""
    from lerobot.datasets.pkl_dataset import MAGIC, RECORD_HEADER
    cursor = cursor or {"shard": 0, "offset": len(MAGIC)}
    for shard_index in range(cursor["shard"], len(descriptor["shards"])):
        path = reject_symlinks(folder / descriptor["shards"][shard_index])
        with path.open("rb") as stream:
            if stream.read(len(MAGIC)) != MAGIC:
                raise ValueError(f"Not an Evo PKL shard: {path}")
            if shard_index == cursor["shard"]:
                offset = cursor["offset"]
                if type(offset) is not int or not len(MAGIC) <= offset <= path.stat().st_size:
                    raise ValueError("Invalid saved PKL byte offset")
                stream.seek(offset)
            while True:
                header = stream.read(RECORD_HEADER.size)
                if not header:
                    break
                if len(header) != RECORD_HEADER.size:
                    raise ValueError(f"Truncated record header: {path}")
                length, checksum = RECORD_HEADER.unpack(header)
                if length > os.fstat(stream.fileno()).st_size - stream.tell():
                    raise ValueError(f"Truncated record body: {path}")
                payload = stream.read(length)
                if zlib.crc32(payload) != checksum:
                    raise ValueError(f"CRC mismatch: {path}")
                packet = pickle.loads(payload)
                yield packet, {"shard": shard_index, "offset": stream.tell()}


def _validate_packet(packet, info, index, api):
    import numpy as np
    if packet["frame_index"] != index:
        raise ValueError("Frame order mismatch")
    if not math.isclose(packet["timestamp"], index / info["fps"], abs_tol=1e-12):
        raise ValueError("Timestamp mismatch")
    frame = packet["frame"]
    required = (set(info["features"]) - api.DEFAULT_FIELDS) | {"task"}
    if set(frame) != required or not isinstance(frame["task"], str):
        raise ValueError("Frame fields/task mismatch")
    encoded = {}
    for key, feature in info["features"].items():
        if key in api.DEFAULT_FIELDS:
            continue
        value = frame[key]
        if feature["dtype"] in {"image", "video"}:
            if info.get("image_encoding", "raw") == "png":
                api.decode_png_rgb(value, feature["shape"])
                encoded[key] = value
            else:
                from lerobot.datasets.pkl_dataset import encode_png_rgb
                if not isinstance(value, np.ndarray) or value.shape != tuple(feature["shape"]):
                    raise ValueError(f"Image shape mismatch: {key}")
                encoded[key] = encode_png_rgb(value, compression=1)
        elif feature["dtype"] == "string":
            if not isinstance(value, str):
                raise ValueError(f"String dtype mismatch: {key}")
        elif not isinstance(value, np.ndarray) or value.shape != tuple(feature["shape"]) or str(value.dtype) != feature["dtype"]:
            raise ValueError(f"Shape/dtype mismatch: {key}")
    return {key: value for key, value in frame.items() if key not in encoded}, encoded


def _prepared_chunks(episode_dir, fingerprint):
    prepared = episode_dir / "prepared"
    durable_mkdir(prepared)
    chunks = []
    frame_count = 0
    for index, path in enumerate(sorted(prepared.glob("chunk-[0-9][0-9][0-9][0-9][0-9][0-9]"))):
        if path.name != f"chunk-{index:06d}":
            raise ValueError("Gap in prepared frame checkpoints")
        marker_path = path / "complete.json"
        try:
            marker = json.loads(marker_path.read_text()) if marker_path.is_file() else {}
        except (OSError, ValueError):
            marker = {}
        if (marker.get("source_fingerprint") != fingerprint or marker.get("start") != frame_count
                or not verify_manifest(path, marker.get("files", {}), exclude=("complete.json",))):
            # Later chunks depend on this cursor. Preserve damaged artifacts;
            # regenerate this suffix without touching original capture files.
            for suffix in sorted(prepared.glob("chunk-[0-9][0-9][0-9][0-9][0-9][0-9]")):
                if suffix.name >= path.name:
                    _quarantine(suffix)
            break
        chunks.append((path, marker))
        frame_count += marker["frames"]
    return chunks, frame_count


def prepare_episode(folder, descriptor, info, episode_dir, fingerprint, chunk_frames, api, frame_bar):
    chunks, done = _prepared_chunks(episode_dir, fingerprint)
    if done > descriptor["frames"]:
        raise ValueError("Prepared checkpoints exceed episode length")
    if done:
        frame_bar.update(done)
        print(f"Resume episode {descriptor['episode_index'] + 1}: {done}/{descriptor['frames']} frames prepared", flush=True)
    cursor = chunks[-1][1]["next_cursor"] if chunks else None
    records = packets_from(folder, descriptor, cursor)
    while done < descriptor["frames"]:
        index = len(chunks)
        destination = episode_dir / "prepared" / f"chunk-{index:06d}"
        temporary = destination.with_name(destination.name + ".preparing-" + uuid.uuid4().hex)
        temporary.mkdir()
        rows = []
        for frame_index in range(done, min(done + chunk_frames, descriptor["frames"])):
            try:
                packet, cursor = next(records)
            except StopIteration as error:
                raise ValueError("Frame count mismatch: source episode is truncated") from error
            numeric, images = _validate_packet(packet, info, frame_index, api)
            rows.append(numeric)
            for key, encoded in images.items():
                path = temporary / key / f"frame-{frame_index:06d}.png"
                path.parent.mkdir(exist_ok=True)
                path.write_bytes(encoded)
            frame_bar.update(1)
        (temporary / "frames.pkl").write_bytes(pickle.dumps(rows, protocol=5))
        marker = {"source_fingerprint": fingerprint, "start": done, "frames": len(rows),
                  "next_cursor": cursor, "files": file_manifest(temporary)}
        sync_tree(temporary)
        atomic_json(temporary / "complete.json", marker)
        temporary.rename(destination)
        sync_directory(destination.parent)
        chunks.append((destination, marker))
        done += len(rows)
        print(f"Prepared checkpoint: episode {descriptor['episode_index'] + 1}, frames {done}/{descriptor['frames']}", flush=True)
    try:
        next(records)
    except StopIteration:
        pass
    else:
        raise ValueError("Frame count mismatch: source contains extra records")
    return chunks


def _verify_video(path, frames, shape, fps, *, decode=False):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if (stream.height, stream.width, 3) != tuple(shape):
            raise ValueError("Checkpoint video shape mismatch")
        if stream.average_rate is None or abs(float(stream.average_rate) - fps) > 1e-6:
            raise ValueError("Checkpoint video FPS mismatch")
        if stream.frames and stream.frames != frames:
            raise ValueError("Checkpoint video frame count mismatch")
        if decode or not stream.frames:
            count = 0
            for frame in container.decode(video=0):
                if frame.pts is None or abs(float(frame.pts * frame.time_base) - count / fps) > 1e-4:
                    raise ValueError("Checkpoint video presentation timestamp mismatch")
                count += 1
            if count != frames:
                raise ValueError("Decoded checkpoint video frame count mismatch")


def _cached_video(path, token):
    reject_symlinks(path)
    marker_path = path.with_suffix(".json")
    if not path.is_file() or not marker_path.is_file():
        return False
    try:
        marker = json.loads(marker_path.read_text())
    except (OSError, ValueError):
        return False
    return (marker.get("token") == token and marker.get("size") == path.stat().st_size
            and marker.get("sha256") == sha256_file(path))


def _publish_video(path, token, producer, frames, shape, fps):
    if _cached_video(path, token):
        return path
    durable_mkdir(path.parent)
    temporary = path.with_name(path.stem + ".encoding-" + uuid.uuid4().hex + ".mp4")
    producer(temporary)
    _verify_video(temporary, frames, shape, fps, decode=True)
    with temporary.open("r+b" if os.name == "nt" else "rb") as stream:
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    sync_directory(path.parent)
    atomic_json(path.with_suffix(".json"), {"token": token, "size": path.stat().st_size,
                                             "sha256": sha256_file(path), "frames": frames})
    print(f"Video checkpoint committed: {path.name}", flush=True)
    return path


def _concatenate(paths, destination):
    """Remux committed segments; all temporary files stay in our workspace."""
    import av
    if len(paths) == 1:
        try:
            os.link(paths[0], destination)
        except OSError:
            shutil.copyfile(paths[0], destination)
        return
    listing = destination.with_suffix(".ffconcat")
    # Paths can include quotes. Use concat-demuxer escaping, not shell quoting.
    listing.write_text("ffconcat version 1.0\n" + "".join(
        "file '" + str(path.resolve()).replace("'", "'\\''") + "'\n" for path in paths), encoding="utf-8")
    try:
        with (av.open(str(listing), format="concat", options={"safe": "0"}) as source,
              av.open(str(destination), "w", options={"movflags": "faststart"}) as output):
            stream = output.add_stream_from_template(source.streams.video[0], opaque=True)
            stream.time_base = source.streams.video[0].time_base
            for packet in source.demux(video=0):
                if packet.dts is not None:
                    packet.stream = stream
                    output.mux(packet)
    finally:
        listing.unlink(missing_ok=True)


def _slice_legacy_video(source_path, destination, start_frame, frames, fps):
    """Recover complete legacy episodes by packet copy, never pixel re-encoding."""
    import av
    with av.open(str(source_path)) as source:
        input_stream = source.streams.video[0]
        start = Fraction(start_frame, fps) / input_stream.time_base
        end = Fraction(start_frame + frames, fps) / input_stream.time_base
        if start.denominator != 1 or end.denominator != 1:
            raise ValueError("Legacy video boundaries are not exactly representable")
        start, end = int(start), int(end)
        source.seek(start, stream=input_stream, backward=True, any_frame=False)
        with av.open(str(destination), "w", options={"movflags": "faststart"}) as output:
            stream = output.add_stream_from_template(input_stream, opaque=True)
            stream.time_base = input_stream.time_base
            first = True
            for packet in source.demux(input_stream):
                if packet.dts is None or packet.pts is None:
                    continue
                if packet.dts >= end:
                    break
                if start <= packet.pts < end:
                    if first and not packet.is_keyframe:
                        raise ValueError("Legacy episode does not start at a key frame")
                    first = False
                    packet.pts -= start
                    packet.dts -= start
                    packet.stream = stream
                    output.mux(packet)
            if first:
                raise ValueError("No frames at legacy episode boundary")


def legacy_videos(root, info, episodes, output):
    """Read-only salvage of an older interrupted converter's complete MP4s."""
    legacy = output.with_name(output.name + ".converting")
    checkpoint = legacy / ".convert-resume.json"
    if not checkpoint.is_file():
        return None
    reject_symlinks(legacy)
    try:
        value = json.loads(checkpoint.read_text())
        meta = json.loads((legacy / "meta/info.json").read_text())
    except (OSError, ValueError) as error:
        print(f"Legacy metadata is incomplete ({error}); preserve it and rebuild from PKL", flush=True)
        return None
    count = value.get("done", 0)
    if (value.get("source") != str(root) or value.get("episodes_total") != len(episodes)
            or type(count) is not int or not 0 < count <= len(episodes)):
        print("Legacy checkpoint does not match this capture; preserving it without reuse", flush=True)
        return None
    frames = sum(data["frames"] for _, data in episodes[:count])
    if (frames != value.get("frames") or meta.get("total_frames") != frames
            or meta.get("total_episodes") != count or meta.get("fps") != info["fps"]):
        print("Legacy checkpoint counts differ; preserving it without reuse", flush=True)
        return None
    if (set(meta.get("features", {})) != set(info["features"])
            or meta.get("robot_type") != info["robot_type"]
            or any(meta["features"][key].get("dtype") != feature.get("dtype")
                   or tuple(meta["features"][key].get("shape", ())) != tuple(feature.get("shape", ()))
                   for key, feature in info["features"].items())):
        print("Legacy feature schemas differ; preserving videos without reuse", flush=True)
        return None
    checkpoint_time = checkpoint.stat().st_mtime_ns
    source_paths = [root / "capture.pkl", root / "capture.closed.json"]
    for folder, descriptor in episodes[:count]:
        source_paths.extend([folder / "ready.json", *(folder / name for name in descriptor["shards"])])
    if any(max(path.stat().st_mtime_ns, path.stat().st_ctime_ns) > checkpoint_time for path in source_paths):
        print("Legacy source changed after its checkpoint; preserve videos without reuse", flush=True)
        return None
    paths = {}
    try:
        for key, feature in info["features"].items():
            if feature["dtype"] == "video":
                candidates = list((legacy / "videos" / key).rglob("*.mp4"))
                if len(candidates) != 1:
                    raise ValueError("Legacy videos span multiple files")
                path = reject_symlinks(candidates[0])
                _verify_video(path, frames, feature["shape"], info["fps"])
                paths[key] = path
    except (OSError, ValueError) as error:
        print(f"Legacy video reuse unavailable ({error}); original files preserved", flush=True)
        return None
    print(f"Found {count} completed legacy episodes: recover videos without re-encoding; rebuild interrupted Parquet", flush=True)
    return {"count": count, "paths": paths}


def encode_episode(chunks, descriptor, info, episode_dir, fingerprint, api, legacy=None, frame_offset=0):
    from lerobot.datasets.video_utils import encode_video_frames
    result = {}
    video_keys = [key for key, feature in info["features"].items() if feature["dtype"] == "video"]
    for key in video_keys:
        combined = episode_dir / "combined" / (key + ".mp4")
        token = f"{fingerprint}:{descriptor['episode_index']}:{key}:{descriptor['frames']}:{info['vcodec']}:v1"
        if _cached_video(combined, token):
            result[key] = combined
            continue
        shape = info["features"][key]["shape"]
        if legacy is not None and descriptor["episode_index"] < legacy["count"]:
            try:
                result[key] = _publish_video(combined, token,
                    lambda path, legacy_path=legacy["paths"][key]: _slice_legacy_video(
                        legacy_path, path, frame_offset, descriptor["frames"], info["fps"]),
                    descriptor["frames"], shape, info["fps"])
                continue
            except (OSError, ValueError) as error:
                print(f"Legacy remux unavailable for episode {descriptor['episode_index'] + 1}/{key}: {error}; encode from source", flush=True)
        paths = []
        for index, (chunk_dir, marker) in enumerate(chunks):
            path = episode_dir / "encoded" / f"chunk-{index:06d}" / (key + ".mp4")
            chunk_token = f"{token}:{marker['start']}:{marker['frames']}"
            paths.append(_publish_video(path, chunk_token,
                lambda destination, images=chunk_dir / key: encode_video_frames(
                    images, destination, info["fps"], vcodec=info["vcodec"], overwrite=True),
                marker["frames"], shape, info["fps"]))
        result[key] = _publish_video(combined, token,
            lambda destination, paths=paths: _concatenate(paths, destination), descriptor["frames"], shape, info["fps"])
    return result


def build_episode_package(chunks, videos, descriptor, info, package, fingerprint, chunk_frames, api):
    import numpy as np

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    class CachedEpisodeDataset(LeRobotDataset):
        def _get_image_file_path(self, episode_index, image_key, frame_index):
            chunk = chunks[frame_index // chunk_frames][0]
            return chunk / image_key / f"frame-{frame_index:06d}.png"

        def _save_image(self, image, fpath, compress_level=1):
            # These exact RGB8 PNGs were decoded/validated before committing
            # their frame chunk. Numeric replay does not decode them again.
            if not Path(fpath).is_file():
                raise ValueError("Missing committed source image")

        def _encode_temporary_episode_video(self, video_key, episode_index):
            folder = self.root / ("video-copy-" + uuid.uuid4().hex)
            folder.mkdir()
            target = folder / "episode.mp4"
            try:
                os.link(videos[video_key], target)
            except OSError:
                shutil.copyfile(videos[video_key], target)
            return target

        def clear_episode_buffer(self, delete_images=True):
            # Frame caches are outside this native writer's transaction.
            self.episode_buffer = self.create_episode_buffer()

    temporary = package.with_name(package.name + ".building-" + uuid.uuid4().hex)
    dataset = CachedEpisodeDataset.create(repo_id=info["repo_id"], fps=info["fps"],
        features=copy.deepcopy(info["features"]), root=temporary, robot_type=info["robot_type"],
        use_videos=True, image_writer_threads=0, image_writer_processes=0,
        batch_encoding_size=1, vcodec=info["vcodec"])
    placeholders = {key: np.zeros(feature["shape"], dtype=np.uint8)
                    for key, feature in info["features"].items() if feature["dtype"] in {"video", "image"}}
    try:
        for chunk_dir, _ in chunks:
            with (chunk_dir / "frames.pkl").open("rb") as stream:
                rows = pickle.load(stream)
            for row in rows:
                dataset.add_frame({**row, **placeholders})
        for key, feature in dataset.features.items():
            if (key not in api.DEFAULT_FIELDS and tuple(feature["shape"]) == (1,)
                    and feature["dtype"] not in {"string", "image", "video"}):
                dataset.episode_buffer[key] = [np.asarray(value).reshape(())[()] for value in dataset.episode_buffer[key]]
        dataset.save_episode(parallel_encoding=False, extra_episode_metadata=descriptor["metadata"])
    finally:
        dataset.stop_image_writer()
        dataset.finalize()
    api.verify_output(temporary, 1, descriptor["frames"])
    marker = {"source_fingerprint": fingerprint, "episode_index": descriptor["episode_index"],
              "frames": descriptor["frames"], "files": file_manifest(temporary)}
    sync_tree(temporary)
    atomic_json(temporary / EPISODE_MARKER, marker)
    temporary.rename(package)
    sync_directory(package.parent)
    print(f"Episode package committed: {descriptor['episode_index'] + 1}", flush=True)
    return package


def _package_valid(package, descriptor, fingerprint):
    path = package / EPISODE_MARKER
    if not path.is_file():
        return False
    try:
        marker = json.loads(path.read_text())
    except (OSError, ValueError):
        return False
    return (marker.get("source_fingerprint") == fingerprint
            and marker.get("episode_index") == descriptor["episode_index"]
            and marker.get("frames") == descriptor["frames"]
            and verify_manifest(package, marker.get("files", {}), exclude=(EPISODE_MARKER,)))


def _complete_output(output, fingerprint, episodes, total, api, info):
    marker_path = output / "conversion.complete.json"
    if not marker_path.is_file():
        raise FileExistsError(f"Existing output has no completion marker; preserve it and choose another --output: {output}")
    marker = json.loads(marker_path.read_text())
    if marker.get("source_fingerprint") not in (None, fingerprint):
        raise ValueError("Completed output belongs to another capture")
    if marker.get("episodes") != len(episodes) or marker.get("frames") != total:
        raise ValueError("Completed output counts differ from the source capture")
    api.verify_output(output, len(episodes), total)
    if marker.get("files") is not None and not verify_manifest(
            output, marker["files"], exclude=("conversion.complete.json", "upload.sha256.json")):
        raise ValueError("Completed dataset SHA256 mismatch; original data and checkpoints are preserved")
    if marker.get("source_fingerprint") is None:
        # Legacy completed output has no source binding. Compare all numeric,
        # string and task values to the closed capture before accepting it.
        binding_path = output.with_name(output.name + ".resume") / "legacy-complete.json"
        binding = json.loads(binding_path.read_text()) if binding_path.is_file() else {}
        files = file_manifest(output, exclude=("upload.sha256.json",))
        if binding.get("source_fingerprint") != fingerprint or binding.get("files") != files:
            _verify_legacy_source(output, episodes, api, info)
            atomic_json(binding_path, {"source_fingerprint": fingerprint, "files": files})
    print(f"Conversion already complete and verified: {output}", flush=True)
    return output


def _verify_legacy_source(output, episodes, api, info):
    import numpy as np
    import pyarrow.parquet as pq

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata("local/converted", root=output)
    if (set(meta.features) != set(info["features"]) or meta.info.get("robot_type") != info["robot_type"]
            or any(meta.features[key]["dtype"] != feature["dtype"]
                   or tuple(meta.features[key]["shape"]) != tuple(feature["shape"])
                   for key, feature in info["features"].items())):
        raise ValueError("Existing dataset feature schema differs from the source capture")
    frames_checked = 0
    with api.progress_bar(total=sum(d["frames"] for _, d in episodes), desc="Verify existing dataset", unit="frame") as bar:
        current_path, rows = None, None
        for index, (folder, descriptor) in enumerate(episodes):
            path = output / meta.get_data_file_path(index)
            if path != current_path:
                rows = pq.read_table(path).to_pylist()
                current_path = path
            selected = [row for row in rows if row["episode_index"] == index]
            if len(selected) != descriptor["frames"]:
                raise ValueError("Existing episode length does not match source")
            for frame_index, (packet, row) in enumerate(zip(api.episode_records(folder, descriptor), selected, strict=True)):
                if (packet["frame_index"] != frame_index
                        or not math.isclose(packet["timestamp"], frame_index / meta.fps, abs_tol=1e-12)):
                    raise ValueError("Source frame order/timestamps are invalid")
                if row["frame_index"] != frame_index or row["index"] != frames_checked:
                    raise ValueError("Existing frame indexes differ from source")
                if abs(float(row["timestamp"]) - frame_index / meta.fps) > 1e-4:
                    raise ValueError("Existing frame timestamps differ from source")
                frame = packet["frame"]
                for key, value in frame.items():
                    if key == "task":
                        actual = meta.tasks.index[int(row["task_index"])]
                        if actual != value:
                            raise ValueError("Existing task differs from source")
                    elif meta.features[key]["dtype"] not in {"image", "video"}:
                        if key not in row:
                            raise ValueError(f"Existing dataset is missing source field: {key}")
                        left, right = np.asarray(row[key]).reshape(-1), np.asarray(value).reshape(-1)
                        equal = (np.array_equal(left, right, equal_nan=True) if right.dtype.kind in "fc"
                                 else np.array_equal(left, right))
                        if not equal:
                            raise ValueError(f"Existing dataset differs from source: {key}")
                frames_checked += 1
                bar.update(1)


def convert_resumable(source, output, *, api, chunk_frames=DEFAULT_CHECKPOINT_FRAMES):
    if type(chunk_frames) is not int or not 1 <= chunk_frames <= 10000:
        raise ValueError("checkpoint_frames must be an integer between 1 and 10000")
    root, info, episodes = api.capture_info(source)
    root = reject_symlinks(root)
    output = reject_symlinks(output)
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output and raw capture must be separate, non-nested directories")
    total = sum(value["frames"] for _, value in episodes)
    for key, feature in info["features"].items():
        if feature["dtype"] in {"image", "video"} and (Path(key).name != key or key in {".", ".."}):
            raise ValueError("Image feature names must be single safe path components")
    fingerprint, identity = capture_identity(root, episodes)
    with dataset_lock(output):
        if output.exists():
            return _complete_output(output, fingerprint, episodes, total, api, info)
        workspace = _workspace(output, fingerprint, identity, chunk_frames)
        dataset_dir = workspace / "dataset"
        if (dataset_dir / "conversion.complete.json").is_file():
            _complete_output(dataset_dir, fingerprint, episodes, total, api, info)
            dataset_dir.rename(output)
            sync_directory(output.parent)
            atomic_json(workspace / "published.json", {"path": str(output), "source_fingerprint": fingerprint})
            print(f"Resumed final dataset publication: {output}", flush=True)
            return output
        if dataset_dir.exists():
            for path in dataset_dir.iterdir():
                if re.fullmatch(r"\.conversion\.complete\.json\.[0-9a-f]{32}\.partial", path.name):
                    reject_symlinks(path)
                    if not path.is_file():
                        raise ValueError("Unexpected interrupted completion marker")
                    path.unlink()
            sync_directory(dataset_dir)
        legacy = legacy_videos(root, info, episodes, output)
        packages = []
        package_dir = workspace / "packages"
        durable_mkdir(package_dir)
        offset = 0
        with api.progress_bar(total=total, desc="Frames prepared", unit="frame") as frames_bar, \
                api.progress_bar(total=len(episodes), desc="Episodes encoded", unit="episode", position=1) as episodes_bar:
            for folder, descriptor in episodes:
                index = descriptor["episode_index"]
                package = package_dir / f"episode-{index:06d}"
                episode_dir = workspace / "episodes" / f"episode-{index:06d}"
                _discard_uncommitted(episode_dir, package_dir, index)
                if _package_valid(package, descriptor, fingerprint):
                    frames_bar.update(descriptor["frames"])
                    print(f"Resume: reuse completed episode {index + 1}/{len(episodes)}", flush=True)
                else:
                    _quarantine(package)
                    chunks = prepare_episode(folder, descriptor, info, episode_dir, fingerprint, chunk_frames, api, frames_bar)
                    videos = encode_episode(chunks, descriptor, info, episode_dir, fingerprint, api, legacy, offset)
                    build_episode_package(chunks, videos, descriptor, info, package, fingerprint, chunk_frames, api)
                # Only disposable caches belonging to a durably committed
                # episode are removed. Original PKLs and legacy work stay put.
                if episode_dir.exists():
                    reject_symlinks(episode_dir)
                    if episode_dir.parent != workspace / "episodes":
                        raise ValueError("Unexpected cache cleanup path")
                    shutil.rmtree(episode_dir)
                    sync_directory(episode_dir.parent)
                packages.append(package)
                offset += descriptor["frames"]
                episodes_bar.update(1)
        from resume_assembly import assemble_episode_packages
        assemble_episode_packages(packages, dataset_dir, info,
                                  checkpoint_dir=workspace / "assembly", progress_factory=api.progress_bar)
        print("Verifying complete native LeRobot dataset...", flush=True)
        api.verify_output(dataset_dir, len(episodes), total)
        # Recheck immutable source signatures before publishing, including any
        # replacement performed while this process was reading a source shard.
        if capture_identity(root, episodes)[0] != fingerprint:
            raise ValueError("Capture files changed during conversion; output was not published")
        atomic_json(dataset_dir / "conversion.complete.json", {
            "episodes": len(episodes), "frames": total, "fps": info["fps"],
            "source_fingerprint": fingerprint, "resume_format": FORMAT,
            "files": file_manifest(dataset_dir, exclude=("conversion.complete.json", "upload.sha256.json")),
        })
        sync_tree(dataset_dir)
        dataset_dir.rename(output)
        sync_directory(output.parent)
        atomic_json(workspace / "published.json", {"path": str(output), "source_fingerprint": fingerprint})
        print(f"Conversion verified and published: {output}", flush=True)
        return output
