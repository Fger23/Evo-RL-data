"""Small durable-file transactions shared by the offline converter."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import uuid
from contextlib import contextmanager
from pathlib import Path


def sync_directory(path):
    if os.name != "posix":
        return
    descriptor = os.open(str(path), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reject_symlinks(path):
    path = Path(path).absolute()
    for candidate in (path, *path.parents):
        if candidate.is_symlink():
            raise ValueError(f"Unexpected symbolic link: {candidate}")
    return path


def durable_mkdir(path):
    path = reject_symlinks(path)
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    if not current.is_dir():
        raise ValueError(f"Expected a directory: {current}")
    for current in reversed(missing):
        current.mkdir()
        sync_directory(current)
        sync_directory(current.parent)
    if not path.is_dir():
        raise ValueError(f"Expected a directory: {path}")
    return path


def atomic_bytes(path, data):
    path = reject_symlinks(path)
    durable_mkdir(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.partial")
    try:
        with temporary.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_json(path, value):
    atomic_bytes(path, json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2).encode())


def sha256_file(path):
    path = reject_symlinks(path)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_manifest(root, *, exclude=()):
    root = reject_symlinks(root)
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link: {path}")
        name = path.relative_to(root).as_posix()
        if path.is_file() and name not in exclude:
            result[name] = {"size": path.stat().st_size, "sha256": sha256_file(path)}
    return result


def verify_manifest(root, expected, *, exclude=()):
    return file_manifest(root, exclude=exclude) == expected


def sync_tree(root):
    root = reject_symlinks(root)
    directories = [root]
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Unexpected symbolic link: {path}")
        if path.is_dir():
            directories.append(path)
        elif path.is_file():
            with path.open("r+b" if os.name == "nt" else "rb") as stream:
                os.fsync(stream.fileno())
    for path in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        sync_directory(path)


@contextmanager
def dataset_lock(output):
    """The kernel releases this lock after SIGKILL; never unlink its inode."""
    output = reject_symlinks(output)
    durable_mkdir(output.parent)
    path = output.with_name(output.name + ".lock")
    reject_symlinks(path)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError("Dataset lock must be a regular file")
        if os.name == "posix":
            import fcntl
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another conversion/upload is using {output}") from error
        else:
            import msvcrt
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b" ")
            os.lseek(descriptor, 0, os.SEEK_SET)
            try:
                msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Another conversion/upload is using {output}") from error
        locked = True
        if os.name == "posix":
            os.ftruncate(descriptor, 0)
            os.write(descriptor, (str(os.getpid()) + "\n").encode())
            os.fsync(descriptor)
        yield
    finally:
        if locked:
            if os.name == "posix":
                import fcntl
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            else:
                import msvcrt
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        os.close(descriptor)
