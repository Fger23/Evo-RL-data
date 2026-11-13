"""Convert a trusted Evo PKL capture using this checkout's original dataset writer.

Run from the repository root, in the original LeRobot Python environment.
Upload is optional and starts only after conversion/output validation succeeds.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import re
import subprocess
import sys
import tempfile
import threading
from collections import defaultdict
from contextlib import contextmanager, suppress
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from lerobot.datasets.pkl_dataset import decode_png_rgb, iter_records

DEFAULT_FIELDS = {"index", "frame_index", "timestamp", "episode_index", "task_index"}
OFFLINE_WRITE_CHUNK_FRAMES = 30
PROGRESS_ENABLED = True


def progress_bar(*args, **kwargs):
    return tqdm(*args, disable=not PROGRESS_ENABLED, dynamic_ncols=True, mininterval=0.25, **kwargs)


@contextmanager
def refresh_progress(bar):
    """Refresh elapsed time while the original video encoder is blocking."""
    stopped = threading.Event()

    def refresh():
        while not stopped.wait(1):
            bar.refresh()

    thread = threading.Thread(target=refresh, daemon=True) if PROGRESS_ENABLED else None
    if thread is not None:
        thread.start()
    try:
        yield
    finally:
        stopped.set()
        if thread is not None:
            thread.join()


# A bounded-memory file stream over SSH. Progress counts payload bytes actually
# written into SSH's stdin; remote acknowledgement and SHA256 precede success.
REMOTE_RECEIVE_SCRIPT = '''import json,os,pathlib,sys
root=pathlib.Path(sys.argv[1])
root.mkdir(exist_ok=False)
source=sys.stdin.buffer
received=0
files=0
while True:
    line=source.readline(65537)
    if not line or len(line)>65536 or not line.endswith(b'\\n'):
        raise ValueError('Missing or oversized upload header')
    header=json.loads(line)
    if header.get('done') is True:
        break
    relative=pathlib.PurePosixPath(header['path'])
    size=header['size']
    if relative.is_absolute() or '..' in relative.parts or not relative.parts:
        raise ValueError('Invalid upload path')
    if type(size) is not int or size<0:
        raise ValueError('Invalid upload size')
    target=root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError('Upload path escapes destination')
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('xb') as output:
        remaining=size
        while remaining:
            chunk=source.read(min(1048576,remaining))
            if not chunk:
                raise ValueError('Upload ended before file was complete')
            output.write(chunk)
            remaining-=len(chunk)
            received+=len(chunk)
        output.flush()
        os.fsync(output.fileno())
    files+=1
print(json.dumps({'status':'received','bytes':received,'files':files}))
'''


def send_directory(command, output, env=None):
    """Stream files without buffering a dataset or requiring SCP's TTY meter."""
    files = sorted(path for path in output.rglob("*") if path.is_file())
    if any(path.is_symlink() for path in output.rglob("*")):
        raise ValueError("Upload source contains a symlink")
    total = sum(path.stat().st_size for path in files)
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env)
    try:
        with progress_bar(total=total, desc="Upload", unit="B", unit_scale=True, unit_divisor=1024) as bar:
            for path in files:
                relative = path.relative_to(output).as_posix()
                size = path.stat().st_size
                bar.set_postfix_str(relative[-55:], refresh=False)
                process.stdin.write((json.dumps({"path": relative, "size": size}) + "\n").encode())
                with path.open("rb") as source:
                    remaining = size
                    while remaining:
                        chunk = source.read(min(1024 * 1024, remaining))
                        if not chunk:
                            raise ValueError(f"Upload source changed while reading: {path}")
                        process.stdin.write(chunk)
                        remaining -= len(chunk)
                        bar.update(len(chunk))
            process.stdin.write(b'{"done":true}\n')
            process.stdin.close()
            process.stdin = None
            stdout, stderr = process.communicate()
            if process.returncode:
                detail = stderr.decode(errors="replace").strip() or f"exit status {process.returncode}"
                raise RuntimeError("SSH upload failed: " + detail)
            acknowledgement = json.loads(stdout)
            if acknowledgement != {"status": "received", "bytes": total, "files": len(files)}:
                raise ValueError("Remote upload acknowledgement does not match sent files")
    except BaseException as error:
        if process.stdin is not None:
            with suppress(OSError):
                process.stdin.close()
            process.stdin = None
        if process.poll() is None:
            process.terminate()
        try:
            _, stderr = process.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            _, stderr = process.communicate()
        if isinstance(error, BrokenPipeError):
            raise RuntimeError("SSH upload failed: " + stderr.decode(errors="replace")) from error
        raise

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
    if final.is_dir() and not final.is_symlink() and not any(final.iterdir()):
        print(json.dumps({'status':'ready'}))
    else:
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
if final.is_dir() and not final.is_symlink() and not any(final.iterdir()):
    # Remove only a confirmed empty placeholder directory. rmdir fails safely
    # if another process creates contents; never merge or recursively delete.
    final.rmdir()
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
    if info.get("format") not in {"evo-pkl-v1", "evo-pkl-v2"} or info.get("fps") != 30:
        raise ValueError("Expected an evo-pkl-v1/v2 capture recorded at 30 FPS")
    encoding = info.get("image_encoding", "raw")
    if encoding not in {"raw", "png"} or (info["format"] == "evo-pkl-v2" and encoding != "png"):
        raise ValueError("Unsupported PKL image encoding")
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
    with progress_bar(total=sum(data["frames"] for _, data in episodes), desc="Validate PKL", unit="frame") as bar:
        return _validate_capture(info, episodes, bar)


def _validate_capture(info, episodes, bar):
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
                if feature["dtype"] in {"image", "video"} and info.get("image_encoding", "raw") == "png":
                    value = decode_png_rgb(value, feature["shape"])
                if tuple(value.shape) != tuple(feature["shape"]):
                    raise ValueError(f"Shape mismatch: {folder}/{key}")
                dtype = "uint8" if feature["dtype"] in ("video", "image") else feature["dtype"]
                if str(value.dtype) != dtype:
                    raise ValueError(f"Dtype mismatch: {folder}/{key}")
            count += 1
            bar.update(1)
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


def convert(source, output, *, checkpoint_frames=300):
    """Resume committed frame/video transactions automatically."""
    from types import SimpleNamespace

    from resumable_conversion import convert_resumable
    api = SimpleNamespace(
        capture_info=capture_info, episode_records=episode_records,
        decode_png_rgb=decode_png_rgb, DEFAULT_FIELDS=DEFAULT_FIELDS,
        verify_output=verify_output, progress_bar=progress_bar,
    )
    return convert_resumable(source, output, api=api, chunk_frames=checkpoint_frames)


def checksum_manifest(output):
    manifest = {}
    paths = sorted(output.rglob("*"))
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Unexpected output symlink: {path}")
    files = [path for path in paths if path.is_file() and path.relative_to(output).as_posix() != "upload.sha256.json"]
    with progress_bar(total=sum(path.stat().st_size for path in files), desc="Local SHA256",
                      unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for path in files:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                    digest.update(chunk)
                    bar.update(len(chunk))
            manifest[path.relative_to(output).as_posix()] = digest.hexdigest()
    (output / "upload.sha256.json").write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")
    return manifest


@contextmanager
def ssh_authentication(password):
    """Use a private configuration password without exposing it in SSH argv."""
    if password is None:
        yield ["-o", "BatchMode=yes"], None
        return
    if not isinstance(password, str) or not password:
        raise ValueError("upload_password must be a nonempty string, or omitted for SSH keys")
    if os.name != "posix":
        raise RuntimeError("Password upload requires Linux/macOS; use SSH keys on Windows")
    with tempfile.TemporaryDirectory(prefix="evo-ssh-auth-") as directory:
        helper = Path(directory) / "askpass"
        helper.write_text(
            f"#!{sys.executable}\nimport os\nprint(os.environ['EVO_SSH_UPLOAD_PASSWORD'])\n",
            encoding="utf-8",
        )
        helper.chmod(0o700)
        environment = dict(os.environ)
        environment.update(SSH_ASKPASS=str(helper), SSH_ASKPASS_REQUIRE="force",
                           DISPLAY=environment.get("DISPLAY") or ":0",
                           EVO_SSH_UPLOAD_PASSWORD=password)
        try:
            yield ["-o", "BatchMode=no", "-o", "StrictHostKeyChecking=yes",
                   "-o", "PreferredAuthentications=password,keyboard-interactive",
                   "-o", "NumberOfPasswordPrompts=1"], environment
        finally:
            environment.pop("EVO_SSH_UPLOAD_PASSWORD", None)


def upload(output, host, user, remote_directory, port=22, password=None):
    with ssh_authentication(password) as (options, environment):
        return _upload_dataset(output, host, user, remote_directory, port, options, environment)


def _upload_dataset(output, host, user, remote_directory, port, ssh_options, environment):
    from resumable_upload import resumable_upload
    from resume_common import dataset_lock, reject_symlinks, verify_manifest
    output = reject_symlinks(output)
    with dataset_lock(output):
        marker = json.loads((output / "conversion.complete.json").read_text(encoding="utf-8"))
        verify_output(output, marker["episodes"], marker["frames"])
        if marker.get("files") is not None and not verify_manifest(
                output, marker["files"], exclude=("conversion.complete.json", "upload.sha256.json")):
            raise ValueError("Dataset SHA256 mismatch; refusing to upload modified conversion output")
        return resumable_upload(output, host, user, remote_directory, port,
                                ssh_options, environment, progress_factory=progress_bar)


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
    global PROGRESS_ENABLED
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Trusted capture.pkl or capture directory")
    parser.add_argument("--output", type=Path, help="Dataset directory; same command automatically resumes <output>.resume")
    parser.add_argument("--checkpoint-frames", type=int, default=300,
                        help="Frames per durable conversion checkpoint (default: 300 = 10 seconds at 30 FPS)")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--upload-only", action="store_true", help="Retry upload of an already converted --output")
    default_config = Path(__file__).resolve().with_name("upload.json")
    parser.add_argument("--config", type=Path,
                        default=default_config if default_config.is_file() else None,
                        help="Upload defaults; automatically reads upload.json beside this script")
    parser.add_argument("--no-upload", action="store_true", help="Only convert, even if upload.json exists")
    parser.add_argument("--upload-target", help="Exact dataset destination: username@host:/absolute/path")
    parser.add_argument("--upload-host")
    parser.add_argument("--upload-user")
    parser.add_argument("--upload-dir", help="Exact remote dataset directory; local directory name is not appended")
    parser.add_argument("--upload-port", type=int)
    parser.add_argument("--upload-password-env", help="Read SSH password from this environment variable")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars for log-only runs")
    args = parser.parse_args()
    PROGRESS_ENABLED = not args.no_progress
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
        output = convert(args.input, output, checkpoint_frames=args.checkpoint_frames)
    if host:
        password_env = args.upload_password_env or settings.get("upload_password_env")
        password = os.environ.get(password_env) if password_env else settings.get("upload_password")
        if password_env and not password:
            parser.error("The configured SSH password environment variable is empty or missing")
        if password:
            upload(output, host, user, remote, port, password=password)
        else:
            upload(output, host, user, remote, port)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Checkpoints are preserved; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
