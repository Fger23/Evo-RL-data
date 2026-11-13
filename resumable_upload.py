"""Crash-resumable, verified native-dataset upload over a single SSH session.

The remote side needs only Python 3.10+ and the Linux standard library. Payload
files stay in a deterministic staging directory until hashes have been checked
and publication is durable. Re-running the same operation resumes intact bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import queue
import re
import shlex
import stat
import subprocess
import threading
import time
from collections import deque
from contextlib import suppress
from pathlib import Path, PurePosixPath

CHUNK_BYTES = 1024 * 1024
MANIFEST_NAME = "upload.sha256.json"
STATE_NAME = ".evo-upload-state.json"
PROGRESS_NAME = ".evo-upload-progress.json"
RESERVED_NAMES = {MANIFEST_NAME, MANIFEST_NAME + ".tmp", STATE_NAME, STATE_NAME + ".tmp",
                  PROGRESS_NAME, PROGRESS_NAME + ".tmp"}

# Keep this executable independently: it is sent verbatim to remote python3.
# The lock is outside staging/final, keyed by destination rather than content.
# flock is released by the kernel on exit, including SIGKILL and broken SSH.
REMOTE_RESUME_SCRIPT = r'''import errno,fcntl,hashlib,json,os,pathlib,re,select,stat,sys,time
CHUNK=1048576
PROBE_SECONDS=10
MANIFEST='upload.sha256.json'
STATE='.evo-upload-state.json'
PROGRESS='.evo-upload-progress.json'
RESERVED={MANIFEST,MANIFEST+'.tmp',STATE,STATE+'.tmp',PROGRESS,PROGRESS+'.tmp'}
destination=pathlib.Path(sys.argv[1])

class UploadBusyError(RuntimeError):
    pass

class TimedInput:
    # Read the raw fd ourselves: selecting a buffered reader can time out even
    # while the reader already contains prefetched protocol/payload bytes.
    def __init__(self):
        self.fd=sys.stdin.buffer.fileno()
        self.buffer=bytearray()
        self.timeout=120

    def more(self,count):
        ready,_,_=select.select([self.fd],[],[],self.timeout)
        if not ready:
            raise TimeoutError('Upload input idle timeout; lock released, rerun to resume')
        return os.read(self.fd,count)

    def readline(self,limit):
        while True:
            end=self.buffer.find(b'\n')
            if end>=0:
                result=bytes(self.buffer[:end+1]);del self.buffer[:end+1]
                return result
            if len(self.buffer)>=limit:return bytes(self.buffer)
            data=self.more(min(65536,limit-len(self.buffer)))
            if not data:
                result=bytes(self.buffer);self.buffer.clear()
                return result
            self.buffer.extend(data)

    def read(self,count):
        result=bytearray()
        if self.buffer:
            available=min(count,len(self.buffer))
            result.extend(self.buffer[:available]);del self.buffer[:available]
        while len(result)<count:
            data=self.more(count-len(result))
            if not data:break
            result.extend(data)
        return bytes(result)

source=TimedInput()

def respond(value):
    sys.stdout.write(json.dumps(value,sort_keys=True,separators=(',',':'))+'\n')
    sys.stdout.flush()

def receive(skip_keepalive=True):
    while True:
        line=source.readline(67108865)
        if not line or len(line)>67108864 or not line.endswith(b'\n'):
            raise ValueError('Upload connection ended before the next protocol message')
        result=json.loads(line)
        if not isinstance(result,dict):
            raise ValueError('Invalid upload protocol message')
        if not skip_keepalive or result.get('op')!='keepalive':return result

def regular_exclusive_inode(descriptor,path):
    metadata=os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink!=1:
        raise ValueError('Unexpected special file or hardlink in upload: '+str(path))

def sync_directory(path):
    descriptor=os.open(str(path),os.O_RDONLY|os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)

def checked_directory(path,create=False):
    if not path.is_absolute():
        raise ValueError('Upload directory must be absolute')
    current=pathlib.Path(path.anchor)
    for name in path.parts[1:]:
        current=current/name
        if current.is_symlink():
            raise ValueError('Unexpected symlink in upload directory: '+str(current))
        if not current.exists():
            if not create: raise ValueError('Missing upload directory: '+str(current))
            current.mkdir()
            sync_directory(current.parent)
        if not current.is_dir():
            raise ValueError('Upload path is not a directory: '+str(current))
    return path

def relative_path(name):
    if not isinstance(name,str) or not name or chr(92) in name or chr(0) in name:
        raise ValueError('Invalid manifest path')
    relative=pathlib.PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or relative.as_posix()!=name or relative.parts[0] in RESERVED:
        raise ValueError('Invalid or reserved manifest path: '+name)
    return relative

def digest_file(path,limit=None):
    digest=hashlib.sha256()
    last_probe=time.monotonic()
    with path.open('rb') as stream:
        remaining=limit
        while remaining is None or remaining:
            data=stream.read(4194304 if remaining is None else min(4194304,remaining))
            if not data:
                if remaining: raise ValueError('Remote file changed during prefix verification')
                break
            digest.update(data)
            if remaining is not None: remaining-=len(data)
            # While hashing a large existing/prefix file, stdin is otherwise
            # unread for minutes. Probe the waiting client so a dead client or
            # half-open SSH connection cannot keep this destination locked.
            if time.monotonic()-last_probe>=PROBE_SECONDS:
                respond({'status':'ping'})
                if receive(skip_keepalive=False).get('op')!='keepalive':
                    raise ValueError('Missing upload liveness acknowledgement')
                last_probe=time.monotonic()
    return digest.hexdigest()

def atomic_json(path,value):
    temporary=path.with_name(path.name+'.tmp')
    if temporary.is_symlink() or (temporary.exists() and not temporary.is_file()):
        raise ValueError('Unexpected upload metadata path: '+str(temporary))
    flags=os.O_WRONLY|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK
    descriptor=os.open(str(temporary),flags,0o600)
    try:
        regular_exclusive_inode(descriptor,temporary)
        os.ftruncate(descriptor,0)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor,'w') as stream:
        json.dump(value,stream,sort_keys=True,separators=(',',':'))
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)
    sync_directory(path.parent)

def actual_files(root):
    found=set()
    for path in root.rglob('*'):
        if path.is_symlink(): raise ValueError('Unexpected symlink in upload: '+str(path))
        if path.is_file():
            if path.stat().st_nlink!=1:raise ValueError('Unexpected hardlink in upload: '+str(path))
            found.add(path.relative_to(root).as_posix())
        elif not path.is_dir(): raise ValueError('Unexpected special file in upload: '+str(path))
    return found

def verify(root,files,published=False):
    checked_directory(root)
    expected=set(files)|{MANIFEST}
    if not published: expected|={STATE,PROGRESS}
    if actual_files(root)!=expected:
        raise ValueError('Remote file list differs from the fresh local manifest')
    wanted={name:item['sha256'] for name,item in files.items()}
    if json.loads((root/MANIFEST).read_text())!=wanted:
        raise ValueError('Remote manifest differs from the fresh local manifest')
    for name,item in files.items():
        path=root/name
        if path.stat().st_size!=item['size'] or digest_file(path)!=item['sha256']:
            raise ValueError('Remote SHA256 mismatch: '+name)

def main():
    if not destination.is_absolute() or destination.name in ('','.') or '..' in destination.parts:
        raise ValueError('Invalid final upload directory')
    checked_directory(destination.parent,create=True)
    lock_path=destination.parent/('.'+destination.name+'.upload.lock')
    descriptor=os.open(str(lock_path),os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW,0o600)
    try:
        regular_exclusive_inode(descriptor,lock_path)
        try: fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise UploadBusyError('Another upload is using this destination; retry when it exits')
        request=receive()
        if request.get('version')!=1 or not isinstance(request.get('files'),dict):
            raise ValueError('Invalid upload manifest')
        idle=request.get('idle_timeout',120)
        if type(idle) not in (int,float) or not 0<idle<=3600:
            raise ValueError('Invalid upload input timeout')
        source.timeout=idle
        files=request['files']
        for name,item in files.items():
            relative_path(name)
            if not isinstance(item,dict) or set(item)!={'size','sha256'} or type(item['size']) is not int or item['size']<0 or not isinstance(item['sha256'],str) or not re.fullmatch('[0-9a-f]{64}',item['sha256']):
                raise ValueError('Invalid upload file metadata: '+name)
        # A file cannot also be an ancestor directory of another manifest file.
        for name in files:
            if any(parent.as_posix() in files for parent in pathlib.PurePosixPath(name).parents if parent.as_posix()!='.'):
                raise ValueError('Manifest contains a file/directory collision')
        identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        staging=destination.with_name(destination.name+'.uploading-'+identity[:24])
        if destination.is_symlink(): raise ValueError('Upload destination is a symlink')
        if destination.exists():
            checked_directory(destination)
            if any(destination.iterdir()):
                verify(destination,files,published=True)
                # A previous sender may have died just after rename and before
                # its parent-directory fsync. Make an idempotent ack durable.
                sync_directory(destination)
                sync_directory(destination.parent)
                respond({'status':'already_complete','path':str(destination),'resumed_bytes':sum(v['size'] for v in files.values()),'sent_bytes':0})
                return
        checked_directory(staging,create=True)
        if not actual_files(staging)<=set(files)|RESERVED:
            raise ValueError('Unexpected file in resumable upload staging directory')
        state={'format':'evo-upload-resume-v1','files':files}
        if (staging/STATE).exists() and json.loads((staging/STATE).read_text())!=state:
            raise ValueError('Upload staging identity differs from the fresh local manifest')
        atomic_json(staging/STATE,state)
        # This small cursor is updated only after a payload block is fsynced.
        # A power cut can leave a new file length with an uncommitted bad tail;
        # resume truncates that tail rather than retransmitting the whole file.
        progress={'identity':identity,'offsets':{}}
        if (staging/PROGRESS).exists():
            progress=json.loads((staging/PROGRESS).read_text())
            if progress.get('identity')!=identity:
                raise ValueError('Invalid durable upload cursor')
            # Read the initial single-file cursor format as well. Several
            # partial files can coexist when an interrupted older invocation
            # used a different ordering; finishing one must not erase another.
            if 'offsets' not in progress:
                name=progress.get('path');offset=progress.get('offset')
                progress={'identity':identity,'offsets':({} if name is None else {name:offset})}
            offsets=progress.get('offsets')
            if not isinstance(offsets,dict) or any(name not in files or type(offset) is not int or not 0<=offset<=files[name]['size'] for name,offset in offsets.items()):
                raise ValueError('Invalid durable upload cursor')
        atomic_json(staging/PROGRESS,progress)
        respond({'status':'ready','path':str(destination),'temporary':str(staging),'total':sum(v['size'] for v in files.values())})
        received=0
        resumed=0
        while True:
            command=receive()
            if command.get('op')=='finish':
                atomic_json(staging/MANIFEST,{name:item['sha256'] for name,item in files.items()})
                # Temp metadata may survive a kill between fsync and replace.
                for name in [STATE+'.tmp',MANIFEST+'.tmp',PROGRESS+'.tmp']:
                    p=staging/name
                    if p.exists():p.unlink()
                verify(staging,files)
                (staging/STATE).unlink()
                (staging/PROGRESS).unlink()
                sync_directory(staging)
                if destination.is_symlink():raise ValueError('Upload destination became a symlink')
                if destination.exists():
                    checked_directory(destination)
                    if any(destination.iterdir()):
                        verify(destination,files,published=True)
                        sync_directory(destination)
                        sync_directory(destination.parent)
                        respond({'status':'already_complete','path':str(destination),'resumed_bytes':resumed,'sent_bytes':received})
                        return
                    destination.rmdir()
                staging.rename(destination)
                sync_directory(destination.parent)
                respond({'status':'complete','path':str(destination),'resumed_bytes':resumed,'sent_bytes':received})
                return
            name=command.get('path')
            if command.get('op')!='file' or name not in files:
                raise ValueError('Invalid upload file request')
            item=files[name]
            path=staging.joinpath(*relative_path(name).parts)
            checked_directory(path.parent,create=True)
            if path.is_symlink() or (path.exists() and not path.is_file()):
                raise ValueError('Unexpected upload payload path: '+name)
            offered=path.stat().st_size if path.exists() else 0
            if name in progress['offsets']:offered=min(offered,progress['offsets'][name])
            elif offered>item['size']:offered=0
            prefix=digest_file(path,offered) if offered else hashlib.sha256(b'').hexdigest()
            respond({'status':'offset','path':name,'offset':offered,'prefix_sha256':prefix})
            start=receive()
            offset=start.get('offset')
            if start.get('op')!='start' or type(offset) is not int or offset not in (0,offered):
                raise ValueError('Invalid upload resume offset')
            flags=os.O_RDWR|os.O_CREAT|os.O_NOFOLLOW|os.O_NONBLOCK
            payload=os.open(str(path),flags,0o600)
            try:
                with os.fdopen(payload,'r+b',buffering=0) as stream:
                    regular_exclusive_inode(stream.fileno(),path)
                    stream.truncate(offset)
                    stream.seek(offset)
                    stream.flush();os.fsync(stream.fileno())
                    sync_directory(path.parent)
                    remaining=item['size']-offset
                    resumed+=offset
                    if remaining or name in progress['offsets']:
                        progress['offsets'][name]=offset
                        atomic_json(staging/PROGRESS,progress)
                    while remaining:
                        data=source.read(min(CHUNK,remaining))
                        if not data:raise ValueError('Upload interrupted before file was complete; partial bytes retained')
                        view=memoryview(data)
                        while view:
                            written=stream.write(view)
                            if not written:raise OSError('Failed to write upload payload')
                            view=view[written:]
                        os.fsync(stream.fileno())
                        remaining-=len(data)
                        received+=len(data)
                        progress['offsets'][name]=item['size']-remaining
                        atomic_json(staging/PROGRESS,progress)
            except BaseException:
                # fdopen owns the descriptor once entered; no deletion on error.
                raise
            if path.stat().st_size!=item['size'] or digest_file(path)!=item['sha256']:
                raise ValueError('Transferred file SHA256 mismatch: '+name)
            if name in progress['offsets']:
                del progress['offsets'][name]
                atomic_json(staging/PROGRESS,progress)
            respond({'status':'file_complete','path':name,'size':item['size']})
    finally:
        os.close(descriptor)

try:
    main()
except Exception as error:
    respond({'status':'error','error':str(error),'code':'busy' if isinstance(error,UploadBusyError) else 'failed'})
    sys.exit(1)
'''


class UploadBusyError(RuntimeError):
    """A previous or concurrent session still owns the remote kernel lock."""


class _QuietBar:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def update(self, _):
        pass

    def set_postfix_str(self, *_args, **_kwargs):
        pass


def _progress(factory, **kwargs):
    return factory(**kwargs) if factory else _QuietBar()


def _atomic_json(path, value):
    temporary = path.with_name(path.name + ".tmp")
    if path.is_symlink() or temporary.is_symlink():
        raise ValueError("Unexpected upload manifest symlink")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0), 0o600)
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Unexpected special file or hardlink in upload manifest")
        os.ftruncate(descriptor, 0)
    except BaseException:
        os.close(descriptor)
        raise
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    if os.name == "posix":
        descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _digest_prefix(path, count, heartbeat=None):
    digest = hashlib.sha256()
    last_heartbeat = time.monotonic()
    with path.open("rb") as stream:
        while count:
            data = stream.read(min(4 * CHUNK_BYTES, count))
            if not data:
                raise ValueError("Upload source changed during prefix verification: " + str(path))
            digest.update(data)
            count -= len(data)
            if heartbeat and time.monotonic() - last_heartbeat >= 10:
                heartbeat()
                last_heartbeat = time.monotonic()
    return digest.hexdigest()


def build_manifest(output, progress_factory=None):
    """Hash the current source instead of trusting an earlier upload manifest."""
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError("Upload source must be a regular directory")
    paths = sorted(output.rglob("*"))
    files = []
    for path in paths:
        if path.is_symlink():
            raise ValueError("Upload source contains a symlink: " + str(path))
        name = path.relative_to(output).as_posix()
        if path.is_file() and name not in {MANIFEST_NAME, MANIFEST_NAME + ".tmp"}:
            if PurePosixPath(name).parts[0] in RESERVED_NAMES:
                raise ValueError("Upload source contains reserved metadata: " + name)
            files.append(path)
        elif not path.is_file() and not path.is_dir():
            raise ValueError("Upload source contains a special file: " + str(path))
    result = {}
    with _progress(progress_factory, total=sum(path.stat().st_size for path in files),
                   desc="Local SHA256", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for path in files:
            initial = path.stat()
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for data in iter(lambda: stream.read(4 * CHUNK_BYTES), b""):
                    digest.update(data)
                    bar.update(len(data))
            final = path.stat()
            if initial.st_size != final.st_size or initial.st_mtime_ns != final.st_mtime_ns:
                raise ValueError("Upload source changed during hashing: " + str(path))
            result[path.relative_to(output).as_posix()] = {"size": final.st_size, "sha256": digest.hexdigest()}
    _atomic_json(output / MANIFEST_NAME, {name: item["sha256"] for name, item in result.items()})
    return result


class _Session:
    def __init__(self, command, environment, timeout):
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, env=environment)
        self.timeout = timeout
        self.messages = queue.Queue()
        self.errors = deque(maxlen=32)
        self.stdout_thread = threading.Thread(target=self._read_stdout, daemon=True)
        self.stderr_thread = threading.Thread(target=self._read_stderr, daemon=True)
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_stdout(self):
        try:
            while True:
                line = self.process.stdout.readline(65537)
                if not line:
                    break
                self.messages.put(line)
        finally:
            self.messages.put(None)

    def _read_stderr(self):
        while True:
            data = self.process.stderr.read(4096)
            if not data:
                break
            self.errors.append(data)

    def send(self, value):
        self.process.stdin.write((json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode())
        self.process.stdin.flush()

    def receive(self):
        while True:
            try:
                line = self.messages.get(timeout=self.timeout)
            except queue.Empty as error:
                raise TimeoutError("SSH upload stopped responding; rerun to resume") from error
            if line is None:
                self.stderr_thread.join(timeout=2)
                detail = b"".join(self.errors).decode(errors="replace").strip()
                raise RuntimeError("SSH upload connection closed: " + (detail or "rerun to resume retained progress"))
            if len(line) > 65536 or not line.endswith(b"\n"):
                raise RuntimeError("Invalid SSH upload response")
            try:
                result = json.loads(line)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("Invalid SSH upload response") from error
            if not isinstance(result, dict):
                raise RuntimeError("Invalid SSH upload response")
            if result.get("status") == "error":
                if result.get("code") == "busy":
                    raise UploadBusyError(str(result.get("error", "Upload destination is busy")))
                raise RuntimeError("Remote upload failed: " + str(result.get("error", "unknown error")))
            if result.get("status") == "ping":
                self.send({"op": "keepalive"})
                continue
            return result

    def close(self, success=False):
        if self.process.stdin is not None:
            with suppress(OSError):
                self.process.stdin.close()
            self.process.stdin = None
        if not success and self.process.poll() is None:
            self.process.terminate()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=15)
        self.stdout_thread.join(timeout=2)
        self.stderr_thread.join(timeout=2)
        self.process.stdout.close()
        self.process.stderr.close()


def stream_upload(command, output, manifest, *, environment=None, progress_factory=None,
                  io_timeout=3600, remote_idle_timeout=120):
    """Run the actual duplex protocol; command can be local Python in tests."""
    output = Path(output)
    session = _Session(command, environment, io_timeout)
    success = False
    try:
        session.send({"version": 1, "files": manifest, "idle_timeout": remote_idle_timeout})
        prepared = session.receive()
        if prepared.get("status") == "already_complete":
            success = True
            return prepared
        total = sum(item["size"] for item in manifest.values())
        if prepared.get("status") != "ready" or prepared.get("total") != total:
            raise RuntimeError("Unexpected upload preparation acknowledgement")
        sent = 0
        resumed = 0
        with _progress(progress_factory, total=total, desc="Upload", unit="B",
                       unit_scale=True, unit_divisor=1024) as bar:
            for name, item in sorted(manifest.items()):
                path = output / name
                if path.is_symlink() or not path.is_file() or path.stat().st_size != item["size"]:
                    raise ValueError("Upload source changed before transfer: " + str(path))
                bar.set_postfix_str(name[-55:], refresh=False)
                session.send({"op": "file", "path": name})
                offered = session.receive()
                offset = offered.get("offset")
                if offered.get("status") != "offset" or offered.get("path") != name or type(offset) is not int or not 0 <= offset <= item["size"]:
                    raise RuntimeError("Invalid upload resume acknowledgement")
                if _digest_prefix(path, offset, lambda: session.send({"op": "keepalive"})) != offered.get("prefix_sha256"):
                    offset = 0
                session.send({"op": "start", "offset": offset})
                resumed += offset
                bar.update(offset)
                with path.open("rb") as stream:
                    stream.seek(offset)
                    remaining = item["size"] - offset
                    while remaining:
                        data = stream.read(min(CHUNK_BYTES, remaining))
                        if not data:
                            raise ValueError("Upload source changed while reading: " + str(path))
                        session.process.stdin.write(data)
                        remaining -= len(data)
                        sent += len(data)
                        bar.update(len(data))
                session.process.stdin.flush()
                completed = session.receive()
                if completed != {"status": "file_complete", "path": name, "size": item["size"]}:
                    raise RuntimeError("Invalid uploaded-file acknowledgement")
            session.send({"op": "finish"})
            result = session.receive()
        if result.get("status") not in {"complete", "already_complete"}:
            raise RuntimeError("Invalid upload publication acknowledgement")
        if result.get("sent_bytes") != sent or result.get("resumed_bytes") != resumed:
            raise RuntimeError("Upload byte acknowledgement differs from transferred bytes")
        success = True
        return result
    except BrokenPipeError as error:
        session.stderr_thread.join(timeout=2)
        detail = b"".join(session.errors).decode(errors="replace").strip()
        # A remote protocol error can race our first stdin write. Preserve its
        # actual reason (for example a busy destination), rather than reporting
        # only BrokenPipeError when the response is already waiting on stdout.
        if not detail:
            with suppress(queue.Empty, ValueError, TypeError, AttributeError):
                pending = session.messages.get(timeout=2)
                response = json.loads(pending) if pending else {}
                if response.get("status") == "error":
                    if response.get("code") == "busy":
                        raise UploadBusyError(str(response.get("error", "Upload destination is busy"))) from error
                    detail = str(response.get("error", "remote upload failed"))
        raise RuntimeError("SSH upload failed: " + (detail or "connection interrupted; rerun to resume")) from error
    finally:
        session.close(success)


def resumable_upload(output, host, user, remote_directory, port, ssh_options, environment,
                     *, progress_factory=None, io_timeout=3600, remote_idle_timeout=120,
                     busy_wait_s=None):
    """Upload a directory, resuming durable matching bytes on repeated calls."""
    if not isinstance(host, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", host) or host.startswith("-"):
        raise ValueError("Upload host must be an IPv4 address or DNS hostname")
    if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", user):
        raise ValueError("Invalid SSH username")
    if not isinstance(remote_directory, str) or not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_directory) or ".." in remote_directory.split("/"):
        raise ValueError("Use an absolute remote directory with ASCII letters/digits, /, _, ., -")
    final = PurePosixPath(remote_directory).as_posix()
    if not PurePosixPath(final).name:
        raise ValueError("Upload destination must be a dataset directory, not filesystem root")
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("Invalid SSH port")
    output = Path(output)
    if output.is_symlink():
        raise ValueError("Upload source must not be a symlink")
    output = output.resolve()
    manifest = build_manifest(output, progress_factory)
    remote_command = shlex.join(["python3", "-u", "-c", REMOTE_RESUME_SCRIPT, final])
    command = ["ssh", *ssh_options, "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=15",
               "-o", "ServerAliveCountMax=3", "-o", "ConnectionAttempts=1", "-p", str(port),
               f"{user}@{host}", remote_command]
    if busy_wait_s is None:
        busy_wait_s = remote_idle_timeout + 30
    if type(busy_wait_s) not in (int, float) or not 0 <= busy_wait_s <= 7200:
        raise ValueError("Invalid busy upload wait duration")
    deadline = time.monotonic() + busy_wait_s
    announced = False
    while True:
        try:
            result = stream_upload(command, output, manifest, environment=environment,
                                   progress_factory=progress_factory, io_timeout=io_timeout,
                                   remote_idle_timeout=remote_idle_timeout)
            break
        except UploadBusyError as error:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise UploadBusyError(
                    "Upload destination is still active after waiting; retry when that session finishes"
                ) from error
            if not announced:
                print(f"Waiting up to {busy_wait_s:g}s for the previous upload connection to close...",
                      flush=True)
                announced = True
            time.sleep(min(2, remaining))
    print("Upload verified: " + str(result["path"]) +
          f" (resumed {result['resumed_bytes']:,} bytes, sent {result['sent_bytes']:,} bytes)", flush=True)
    return result
