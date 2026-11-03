"""Convert a trusted Evo PKL capture using this checkout's original dataset writer.

Run from the repository root, in the original LeRobot Python environment.
Upload is optional and starts only after conversion/output validation succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import re
import shlex
import subprocess
import sys
import uuid
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lerobot.datasets.pkl_dataset import atomic_json, iter_records

DEFAULT_FIELDS = {"index", "frame_index", "timestamp", "episode_index", "task_index"}
OFFLINE_WRITE_CHUNK_FRAMES = 30

# These helpers run on the upload server using only Python's standard library.
# Verification uses the fresh local manifest sent over SSH stdin, not solely a
# manifest already on the server. Explicit checks also work with python -O.
REMOTE_UPLOAD_HELPERS = '''import hashlib,json,pathlib,sys
def verify_directory(root, expected):
    if not root.is_dir() or root.is_symlink():
        raise ValueError('Upload destination is not a regular directory: '+str(root))
    paths=list(root.rglob('*'))
    if any(p.is_symlink() for p in paths):
        raise ValueError('Unexpected symlink in upload: '+str(root))
    actual={p.relative_to(root).as_posix() for p in paths if p.is_file()}
    if actual != set(expected)|{'upload.sha256.json'}:
        raise ValueError('Remote file list mismatch')
    uploaded=json.loads((root/'upload.sha256.json').read_text(encoding='utf-8'))
    if uploaded != expected:
        raise ValueError('Remote manifest differs from fresh local manifest')
    for name,wanted in expected.items():
        relative=pathlib.PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid manifest path: '+name)
        p=root/name
        if not p.resolve().is_relative_to(root.resolve()):
            raise ValueError('Invalid manifest path: '+name)
        digest=hashlib.sha256()
        with p.open('rb') as f:
            for chunk in iter(lambda:f.read(4194304),b''): digest.update(chunk)
        if digest.hexdigest()!=wanted:
            raise ValueError('Remote SHA256 mismatch: '+name)
'''

REMOTE_PREPARE_SCRIPT = REMOTE_UPLOAD_HELPERS + '''
parent,temporary,final=map(pathlib.Path,sys.argv[1:])
expected=json.load(sys.stdin)
parent.mkdir(parents=True,exist_ok=True)
if final.exists() or final.is_symlink():
    verify_directory(final,expected)
    print(json.dumps({'status':'already_complete','path':str(final)}))
else:
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError('Remote temporary destination already exists')
    print(json.dumps({'status':'ready'}))
'''

REMOTE_PUBLISH_SCRIPT = REMOTE_UPLOAD_HELPERS + '''
temporary,final=map(pathlib.Path,sys.argv[1:])
expected=json.load(sys.stdin)
verify_directory(temporary,expected)
if final.exists() or final.is_symlink():
    verify_directory(final,expected)
    status='already_complete'
else:
    try:
        temporary.rename(final)
        status='complete'
    except OSError:
        # Another publisher may have completed between our check and rename.
        if not final.exists():
            raise
        verify_directory(final,expected)
        status='already_complete'
print(json.dumps({'status':status,'path':str(final)}))
'''


def capture_info(source):
    source = Path(source).resolve()
    root = source.parent if source.is_file() else source
    if source.is_file() and source.name != "capture.pkl":
        raise ValueError("Provide capture.pkl or its capture directory, not an individual shard")
    if not (root / "capture.closed.json").is_file():
        raise ValueError("Capture is not closed. Finish recording successfully before converting")
    with (root / "capture.pkl").open("rb") as stream:
        info = pickle.load(stream)
    if info.get("format") != "evo-pkl-v1" or info.get("fps") != 30:
        raise ValueError("Expected an evo-pkl-v1 capture recorded at 30 FPS")
    episodes = []
    for path in (root / "episodes").glob("attempt-*/ready.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        episodes.append((path.parent, data))
    episodes.sort(key=lambda item: item[1]["episode_index"])
    closed = json.loads((root / "capture.closed.json").read_text(encoding="utf-8"))
    if not episodes or len(episodes) != closed["committed_episodes"]:
        raise ValueError("No committed episodes, or missing episode descriptors")
    for index, (_, data) in enumerate(episodes):
        if data["episode_index"] != index or data["frames"] <= 0 or not data["shards"]:
            raise ValueError("Episode indexes/counts are invalid")
    return root, info, episodes


def episode_records(folder, descriptor):
    for name in descriptor["shards"]:
        if not re.fullmatch(r"part-\d{6}\.pkl", name):
            raise ValueError("Invalid shard filename")
        yield from iter_records(folder / name)


def validate_capture(info, episodes):
    required = (set(info["features"]) - DEFAULT_FIELDS) | {"task"}
    total = 0
    for folder, descriptor in episodes:
        count = 0
        for packet in episode_records(folder, descriptor):
            if packet["frame_index"] != count:
                raise ValueError(f"Frame order mismatch in {folder}")
            if not math.isclose(packet["timestamp"], count / info["fps"], abs_tol=1e-12):
                raise ValueError(f"Timestamp mismatch in {folder}")
            frame = packet["frame"]
            if set(frame) != required:
                raise ValueError(f"Frame fields mismatch in {folder}")
            for key, feature in info["features"].items():
                if key in DEFAULT_FIELDS or feature["dtype"] == "string":
                    continue
                value = frame[key]
                if tuple(value.shape) != tuple(feature["shape"]):
                    raise ValueError(f"Shape mismatch: {folder}/{key}")
                dtype = "uint8" if feature["dtype"] in ("video", "image") else feature["dtype"]
                if str(value.dtype) != dtype:
                    raise ValueError(f"Dtype mismatch: {folder}/{key}")
            count += 1
        if count != descriptor["frames"]:
            raise ValueError(f"Frame count mismatch in {folder}")
        total += count
    return total


def verify_output(output, expected_episodes, expected_frames):
    import av
    import pyarrow.parquet as pq

    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata("local/converted", root=output)
    if meta.total_episodes != expected_episodes or meta.total_frames != expected_frames or meta.fps != 30:
        raise ValueError("Converted dataset metadata counts/FPS do not match capture")
    rows = sum(pq.ParquetFile(path).metadata.num_rows for path in (output / "data").rglob("*.parquet"))
    if rows != expected_frames:
        raise ValueError("Converted Parquet row count mismatch")
    expected = defaultdict(int)
    shapes = {}
    for index in range(meta.total_episodes):
        episode = meta.episodes[index]
        for key in meta.video_keys:
            path = output / meta.get_video_file_path(index, key)
            expected[path] += int(episode["length"])
            shapes[path] = tuple(meta.features[key]["shape"])
    for path, frames in expected.items():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            if stream.frames and stream.frames != frames:
                raise ValueError(f"Video frame count mismatch: {path}")
            if not stream.frames and sum(1 for _ in container.decode(video=0)) != frames:
                raise ValueError(f"Decoded video frame count mismatch: {path}")
            if (stream.height, stream.width, 3) != shapes[path]:
                raise ValueError(f"Video resolution mismatch: {path}")
            if stream.average_rate is None or abs(float(stream.average_rate) - 30) > 1e-6:
                raise ValueError(f"Video FPS mismatch: {path}")


def convert(source, output):
    import numpy as np
    root, info, episodes = capture_info(source)
    print("Checking PKL records/CRCs, frame order, shape, dtype and timestamps...", flush=True)
    total = validate_capture(info, episodes)
    output = Path(output).resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output and raw capture must be separate, non-nested directories")
    staging = output.with_name(output.name + ".converting")
    if output.exists() or staging.exists():
        raise FileExistsError(f"Use a new output path; refusing to overwrite {output} or {staging}")
    output.parent.mkdir(parents=True, exist_ok=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    dataset = LeRobotDataset.create(
        repo_id=info["repo_id"], fps=info["fps"], features=info["features"],
        root=staging, robot_type=info["robot_type"], use_videos=True,
        image_writer_threads=4, image_writer_processes=0,
        batch_encoding_size=1, vcodec=info["vcodec"],
    )
    try:
        for folder, descriptor in episodes:
            for frame_count, packet in enumerate(episode_records(folder, descriptor), start=1):
                # Original add_frame validates keys before generating timestamp.
                # Keep timestamp out of frame; original code regenerates n/fps.
                dataset.add_frame(packet["frame"])
                if frame_count % OFFLINE_WRITE_CHUNK_FRAMES == 0:
                    # Original AsyncImageWriter has an unbounded queue. Drain
                    # regularly so a fast raw reader cannot queue a whole
                    # multi-GB episode while PNG writes lag behind.
                    dataset._wait_image_writer()
            # This checkout maps shape=(1,) to an Arrow scalar. NumPy 2.4+
            # rejects float(array([x])); preserve dtype/value as numpy scalars.
            for key, feature in dataset.features.items():
                if (key not in DEFAULT_FIELDS and tuple(feature["shape"]) == (1,)
                        and feature["dtype"] not in ("string", "image", "video")):
                    dataset.episode_buffer[key] = [
                        np.asarray(value).reshape(())[()] for value in dataset.episode_buffer[key]
                    ]
            dataset.save_episode(extra_episode_metadata=descriptor["metadata"])
            print(f"Converted episode {descriptor['episode_index'] + 1}/{len(episodes)}", flush=True)
    finally:
        try:
            dataset.stop_image_writer()
        finally:
            dataset.finalize()
    verify_output(staging, len(episodes), total)
    atomic_json(staging / "conversion.complete.json", {
        "episodes": len(episodes), "frames": total, "fps": info["fps"],
    })
    staging.rename(output)
    print(f"Conversion verified: {output}", flush=True)
    return output


def checksum_manifest(output):
    manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected output symlink: {path}")
        if path.is_file() and path.relative_to(output).as_posix() != "upload.sha256.json":
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
            manifest[path.relative_to(output).as_posix()] = digest.hexdigest()
    (output / "upload.sha256.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    return manifest


def upload(output, host, user, remote_parent, port=22):
    output = Path(output).resolve()
    marker = json.loads((output / "conversion.complete.json").read_text(encoding="utf-8"))
    verify_output(output, marker["episodes"], marker["frames"])
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or host.startswith("-"):
        raise ValueError("Upload host must be an IPv4 address or DNS hostname")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user):
        raise ValueError("Invalid SSH username")
    # Restrict remote paths to a portable shell/SFTP-safe form for both old and new scp.
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_parent) or ".." in remote_parent.split("/"):
        raise ValueError("Use an absolute remote directory with ASCII letters/digits, /, _, ., -")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", output.name):
        raise ValueError("For SCP upload, use an ASCII output directory name without spaces")
    if not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    manifest = checksum_manifest(output)
    target = f"{user}@{host}"
    final = remote_parent.rstrip("/") + "/" + output.name
    temporary = final + ".uploading-" + uuid.uuid4().hex[:12]

    def ssh_python(code, *arguments):
        command = shlex.join(["python3", "-c", code, *arguments])
        try:
            result = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-p", str(port), target, command],
                input=json.dumps(manifest, ensure_ascii=True),
                text=True, capture_output=True, check=True,
            )
        except subprocess.CalledProcessError as error:
            detail = error.stderr.strip() if error.stderr else f"SSH exited with status {error.returncode}"
            raise RuntimeError(f"Remote upload step failed: {detail}") from error
        return json.loads(result.stdout)

    preparation = ssh_python(
        REMOTE_PREPARE_SCRIPT,
        remote_parent, temporary, final,
    )
    if preparation.get("status") == "already_complete":
        print(f"Upload already verified: {preparation['path']}", flush=True)
        return
    if preparation.get("status") != "ready":
        raise ValueError("Unexpected remote upload preparation response")
    subprocess.run([
        "scp", "-o", "BatchMode=yes", "-P", str(port), "-r",
        str(output), f"{target}:{temporary}",
    ], check=True)
    publication = ssh_python(REMOTE_PUBLISH_SCRIPT, temporary, final)
    if publication.get("status") not in {"complete", "already_complete"}:
        raise ValueError("Unexpected remote upload publication response")
    print(f"Upload verified: {publication['path']}", flush=True)


def parse_upload_target(target):
    """Split username@host:/absolute/path; upload() validates actual values."""
    if not isinstance(target, str):
        raise ValueError("upload_target must be username@host:/absolute/path")
    match = re.fullmatch(r"([^@:/\s]+)@([^@:/\s]+):(/.*)", target)
    if match is None:
        raise ValueError("upload_target must be username@host:/absolute/path")
    user, host, directory = match.groups()
    return host, user, directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Trusted capture.pkl or capture directory")
    parser.add_argument("--output", type=Path, help="New dataset directory; defaults to <capture>-dataset")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true", help="Retry upload of an already converted --output")
    default_config = Path(__file__).resolve().with_name("upload.json")
    parser.add_argument("--config", type=Path,
                        default=default_config if default_config.is_file() else None,
                        help="Upload defaults; automatically reads upload.json beside this script")
    parser.add_argument("--no-upload", action="store_true", help="Only convert, even if upload.json exists")
    parser.add_argument("--upload-target", help="SSH destination shorthand: username@host:/absolute/path")
    parser.add_argument("--upload-host")
    parser.add_argument("--upload-user")
    parser.add_argument("--upload-dir", help="Remote parent directory; dataset name is appended")
    parser.add_argument("--upload-port", type=int)
    args = parser.parse_args()
    settings = json.loads(args.config.read_text(encoding="utf-8")) if args.config else {}
    port = args.upload_port or settings.get("upload_port", 22)
    if args.no_upload:
        if args.upload_only:
            parser.error("no-upload cannot be combined with upload-only")
        host = user = remote = None
    else:
        shorthand = args.upload_target or settings.get("upload_target")
        try:
            shorthand_values = parse_upload_target(shorthand) if shorthand else (None, None, None)
        except ValueError as error:
            parser.error(str(error))
        host = args.upload_host or shorthand_values[0] or settings.get("upload_host")
        user = args.upload_user or shorthand_values[1] or settings.get("upload_user")
        remote = args.upload_dir or shorthand_values[2] or settings.get("upload_dir")
    if any((host, user, remote)) and not all((host, user, remote)):
        parser.error("Set upload_host, upload_user and upload_dir together")
    if args.validate_only and args.upload_only:
        parser.error("Choose validate-only or upload-only")
    if args.upload_only:
        if args.output is None or not host:
            parser.error("upload-only requires --output and upload configuration")
        output = args.output
    else:
        if args.input is None:
            parser.error("--input is required for validation/conversion")
        root, info, episodes = capture_info(args.input)
        if args.validate_only:
            frames = validate_capture(info, episodes)
            print(f"PKL verified: {len(episodes)} episodes, {frames} frames, {info['fps']} FPS")
            return
        output = args.output or root.with_name(root.name + "-dataset")
        output = convert(args.input, output)
    if host:
        upload(output, host, user, remote, port)


if __name__ == "__main__":
    main()
