"""Private Linux qualification-tool command owner; no Vulkan/runtime imports.

The host explicitly promises exclusive wait ownership of the bootstrap child.
Only the isolated worker changes process policy or performs all-child waits.
"""

from __future__ import annotations

import ctypes
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field, replace
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import resource
import secrets
import select
import selectors
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from itertools import islice
from typing import Any, Literal


WORKER_CLEANUP_SECONDS = 1.5
TERM_GRACE_SECONDS = 0.2
CALLER_REAP_SECONDS = 0.5
TOTAL_CLEANUP_SECONDS = 2.0
CHILD_BATCH = 64
REAP_BATCH = 64
CHILD_READ_BYTES = 64 * 1024
LOOP_SLICE_SECONDS = 0.01
MAX_START_BYTES = 8 * 1024 * 1024
MAX_REPLY_BYTES = 64 * 1024
MAX_CANCEL_BYTES = 4096
IO_CHUNK_BYTES = 64 * 1024
# Linux UAPI linux/wait.h, independently verified; use the existing waitid ABI.
_P_PIDFD = getattr(os, "P_PIDFD", 3)
_pidfd_api = None
PublicResult = dict[str, Any]
FinalState = Literal["complete-echild", "not-started", "unknown"]
_wait_scope = ContextVar("qualification_exclusive_wait", default=False)


@dataclass(frozen=True)
class CommandIdentity:
    argv: tuple[str, ...]
    cwd: str
    environment: tuple[tuple[str, str], ...]
    sha256: str


@dataclass(frozen=True)
class Certificate:
    invocation_id: str
    identity: CommandIdentity
    worker_pid: int | None
    worker_start_ticks: int | None
    worker_pidfd_authenticated: bool
    leader_pid: int | None
    leader_start_ticks: int | None
    leader_wait_status: int | None
    spawn_error: str | None
    reason: str
    execution_deadline: float
    cleanup_started: float | None
    cleanup_deadline: float | None
    method: Literal["isolated-linux-subreaper"]
    adopted_count: int
    reaped_count: int
    state: FinalState
    errors: tuple[str, ...]
    errors_dropped: int
    output_complete: bool
    terminal_received: bool
    worker_reaped: bool


class ContainmentError(RuntimeError):
    def __init__(self, message, *, partial_result, invocation_id, certificate,
                 authority=None, original_error=None):
        super().__init__(message)
        self.partial_result = partial_result
        self.invocation_id = invocation_id
        self.certificate = certificate
        self.authority = authority
        self.original_error = original_error


class _BootstrapProcess(subprocess.Popen):
    def __del__(self):
        # Never poll, reap, or register with subprocess._active, including when
        # __init__ raised after creating the child.
        pass


@dataclass
class AuthorityHandle:
    pidfd: int
    process: _BootstrapProcess
    reaped: bool = False
    _wait_lock: Any = field(default_factory=threading.Lock, repr=False)

    def observe_until(self, deadline: float) -> bool:
        if not self._wait_lock.acquire(blocking=False):
            raise RuntimeError("concurrent worker waiter")
        try:
            while not self.reaped:
                try:
                    pid, status = os.waitpid(self.process.pid, os.WNOHANG)
                except InterruptedError:
                    if time.monotonic() >= deadline:
                        return False
                    continue
                if pid:
                    self.process.returncode = os.waitstatus_to_exitcode(status)
                    self.reaped = True
                    break
                if time.monotonic() >= deadline:
                    return False
                select.select([], [], [], min(LOOP_SLICE_SECONDS,
                                              max(0, deadline - time.monotonic())))
            return True
        finally:
            self._wait_lock.release()

    def close(self) -> None:
        if self.pidfd >= 0:
            fd, self.pidfd = self.pidfd, -1
            os.close(fd)


@contextmanager
def caller_wait_scope(*, exclusive_wait: bool):
    """Hosting assertion, NOT a runtime proof about arbitrary host waiters."""
    if type(exclusive_wait) is not bool:
        raise TypeError("exclusive_wait must be bool")
    token = _wait_scope.set(exclusive_wait)
    try:
        yield
    finally:
        _wait_scope.reset(token)


_current_cancellation = ContextVar("qualification_cancellation", default=None)


@dataclass
class Cancellation:
    read_fd: int
    write_fd: int
    reason: str | None = None

    def request(self, reason: str) -> None:
        # The first reason is kept even when the wake-up pipe is full or closed.
        if self.reason is None:
            self.reason = reason
        try:
            os.write(self.write_fd, b"x")
        except OSError:
            pass


@contextmanager
def cli_cancellation_scope():
    """CLI-only INT/TERM cancellation; library callers never install handlers."""
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
    cancellation = Cancellation(read_fd, write_fd)
    names = (signal.SIGINT, signal.SIGTERM)
    previous = {}
    token = None
    try:
        for sig in names:
            previous[sig] = signal.signal(
                sig, lambda signum, frame: cancellation.request(signal.Signals(signum).name))
        token = _current_cancellation.set(cancellation)
        with caller_wait_scope(exclusive_wait=True):
            yield cancellation
    finally:
        if token is not None:
            _current_cancellation.reset(token)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        os.close(read_fd)
        os.close(write_fd)


def _earlier(current, candidate):
    return candidate if current is None else min(current, candidate)


def _coordination_error(message, caught, authority=None):
    """ContainmentError for a coordinating child that has no command certificate."""
    certificate = _empty_certificate(secrets.token_hex(16), CommandIdentity((), "", (), ""),
                                     time.monotonic(), state="unknown" if authority else "not-started",
                                     reason="coordination-error")
    return ContainmentError(message, partial_result=None, invocation_id=certificate.invocation_id,
                            certificate=certificate, authority=authority, original_error=caught)


def _cancellation_error(reason, result, certificate):
    if certificate is None:
        certificate = _empty_certificate(secrets.token_hex(16), CommandIdentity((), "", (), ""),
                                         time.monotonic(), reason="cancelled")
    return ContainmentError(f"command cancelled: {reason}", partial_result=result,
        invocation_id=certificate.invocation_id, certificate=certificate,
        original_error=RuntimeError(f"cancelled by {reason}"))


def check_cancellation(*, result=None, certificate=None):
    """Raise non-success when a scoped CLI cancellation is pending."""
    cancellation = _current_cancellation.get()
    if cancellation is not None and cancellation.reason is not None:
        raise _cancellation_error(cancellation.reason, result, certificate)


def _decode_output(value):
    return value.decode(errors="replace") if isinstance(value, bytes) else value or ""


def _json_bytes(message):
    return json.dumps(message, ensure_ascii=True, allow_nan=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def _encode_frame(message, frame_limit):
    body = _json_bytes(message)
    if not 0 < len(body) <= frame_limit:
        raise ValueError("control frame exceeds its message limit")
    return struct.pack("!I", len(body)) + body


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


class _FrameParser:
    def __init__(self, limit):
        self.limit = limit
        self.buffer = bytearray()

    def feed(self, data):
        # Accept at most one bounded read at a time. Check the advertised length
        # before accumulating its body, never allocate that advertised length.
        if len(data) > IO_CHUNK_BYTES:
            raise ValueError("unbounded control read")
        self.buffer.extend(data)
        messages = []
        while len(self.buffer) >= 4:
            length = struct.unpack("!I", self.buffer[:4])[0]
            if not 0 < length <= self.limit:
                raise ValueError("invalid control frame length")
            if len(self.buffer) < length + 4:
                break
            body = bytes(self.buffer[4:length + 4])
            del self.buffer[:length + 4]
            def invalid(value):
                raise ValueError(f"invalid JSON constant {value}")
            message = json.loads(body.decode("utf-8"), object_pairs_hook=_unique_object,
                                 parse_constant=invalid)
            if type(message) is not dict:
                raise ValueError("control frame must be an object")
            messages.append(message)
        if len(self.buffer) > self.limit + 4:
            raise ValueError("control receive buffer exceeds limit")
        return messages

    def eof(self):
        if self.buffer:
            raise ValueError("partial control frame at EOF")


_READY_FIELDS = {"v", "id", "type", "worker_pid", "worker_start_ticks", "identity_sha256"}
_START_FIELDS = {"v", "id", "type", "argv", "cwd", "environment", "identity_sha256",
                 "execution_deadline", "timeout_seconds"}
_CANCEL_FIELDS = {"v", "id", "type", "reason"}
_TERMINAL_FIELDS = (_READY_FIELDS | {"leader_pid", "leader_start_ticks", "leader_wait_status",
    "spawn_error", "reason", "execution_deadline", "cleanup_started", "cleanup_deadline",
    "method", "adopted_count", "reaped_count", "state", "errors", "errors_dropped", "output_complete"})


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def _text(value, bound=None):
    return (type(value) is str and "\0" not in value and
            (bound is None or len(value.encode("utf-8")) <= bound))


def _hex(value, size):
    return type(value) is str and len(value) == size and all(c in "0123456789abcdef" for c in value)


def _identity(argv, cwd, environment):
    if (type(argv) is not list or not argv or not all(_text(x) for x in argv)
            or not argv[0] or not _text(cwd) or not os.path.isabs(cwd)
            or type(environment) is not dict
            or not all(_text(k) and k and "=" not in k and _text(v)
                       for k, v in environment.items())):
        raise ValueError("invalid command identity")
    digest = hashlib.sha256(_json_bytes({"argv": argv, "cwd": cwd,
                                       "environment": environment})).hexdigest()
    return CommandIdentity(tuple(argv), cwd, tuple(sorted(environment.items())), digest)


def _validate_message(message, kind, invocation_id, digest):
    fields = {"READY": _READY_FIELDS, "START": _START_FIELDS,
              "CANCEL": _CANCEL_FIELDS, "TERMINAL": _TERMINAL_FIELDS}[kind]
    if (set(message) != fields or type(message["v"]) is not int or message["v"] != 1
            or message["type"] != kind or not _hex(message["id"], 32)
            or message["id"] != invocation_id):
        raise ValueError("invalid control schema/version/invocation/order")
    if kind != "CANCEL" and (not _hex(message["identity_sha256"], 64)
                             or message["identity_sha256"] != digest):
        raise ValueError("control identity digest mismatch")
    if kind in ("READY", "TERMINAL"):
        for key in ("worker_pid", "worker_start_ticks"):
            if type(message[key]) is not int or message[key] <= 0:
                raise ValueError("invalid worker diagnostic identity")
    if kind == "START":
        identity = _identity(message["argv"], message["cwd"], message["environment"])
        if (identity.sha256 != digest or type(message["execution_deadline"]) is not float
                or not _finite(message["execution_deadline"])
                or not _finite(message["timeout_seconds"]) or message["timeout_seconds"] < 0):
            raise ValueError("START identity or deadline invalid")
    if kind in ("CANCEL", "TERMINAL") and not _text(message["reason"], 512):
        raise ValueError("invalid reason")
    if kind == "TERMINAL":
        for key in ("leader_pid", "leader_start_ticks", "leader_wait_status"):
            value = message[key]
            if value is not None and (type(value) is not int or (key != "leader_wait_status" and value <= 0)):
                raise ValueError("invalid leader diagnostic")
        if message["spawn_error"] is not None and not _text(message["spawn_error"], 4096):
            raise ValueError("invalid spawn error")
        for key in ("execution_deadline", "cleanup_started", "cleanup_deadline"):
            if not (key != "execution_deadline" and message[key] is None) and not _finite(message[key]):
                raise ValueError("invalid terminal time")
        for key in ("adopted_count", "reaped_count", "errors_dropped"):
            if type(message[key]) is not int or message[key] < 0:
                raise ValueError("invalid terminal count")
        if (message["method"] != "isolated-linux-subreaper" or
                message["state"] not in ("complete-echild", "not-started", "unknown") or
                type(message["output_complete"]) is not bool or
                type(message["errors"]) is not list or len(message["errors"]) > 64 or
                not all(_text(e, 512) for e in message["errors"])):
            raise ValueError("invalid terminal state/errors")
        if message["state"] == "not-started" and any(message[k] is not None for k in
                ("leader_pid", "leader_start_ticks", "leader_wait_status")):
            raise ValueError("not-started terminal owns a leader")


class _Sigaction(ctypes.Structure):
    _fields_ = [("handler", ctypes.c_void_p), ("mask", ctypes.c_ubyte * 128),
                ("flags", ctypes.c_int), ("restorer", ctypes.c_void_p)]


def _sigaction_abi():
    # Only verified LP64 Linux libc layouts are supported. confstr is a positive
    # glibc identifier; do not infer musl from a failed glibc query.
    if platform.machine() not in ("x86_64", "aarch64") or ctypes.sizeof(ctypes.c_void_p) != 8:
        raise RuntimeError("unsupported sigaction architecture")
    libc_version = os.confstr("CS_GNU_LIBC_VERSION")
    if not libc_version or not libc_version.startswith("glibc "):
        raise RuntimeError("unrecognized sigaction libc ABI")
    return _Sigaction


def _check_sigchld():
    layout = _sigaction_abi()
    storage = ctypes.create_string_buffer(512)
    libc = ctypes.CDLL(None, use_errno=True)
    query = libc.sigaction
    query.argtypes = (ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    query.restype = ctypes.c_int
    if query(signal.SIGCHLD, None, ctypes.byref(storage)) != 0:
        raise OSError(ctypes.get_errno(), "sigaction query")
    action = layout.from_buffer(storage)
    if (action.handler not in (None, 0) or action.flags & 2 or
            signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL):
        raise RuntimeError("unsupported SIGCHLD handler/ignore/SA_NOCLDWAIT policy")


def _ticks(pid):
    return int(Path(f"/proc/{pid}/stat").read_bytes().rsplit(b") ", 1)[1].split()[19])


def _pidfd_pid(fd):
    with open(f"/proc/self/fdinfo/{fd}", "rb") as stream:
        info = stream.read(4097)
    values = [line.split()[1] for line in info.splitlines() if line.startswith(b"Pid:")]
    if len(info) > 4096 or len(values) != 1:
        raise ValueError("invalid pidfd identity")
    return int(values[0])


def certificate_record(certificate):
    record = asdict(certificate)
    record["identity"]["argv"] = list(certificate.identity.argv)
    record["identity"]["environment"] = dict(certificate.identity.environment)
    record["errors"] = list(certificate.errors)
    return record


def _empty_certificate(invocation_id, identity, deadline, *, state="not-started", reason="preflight"):
    return Certificate(invocation_id, identity, None, None, False, None, None, None,
                       None, reason, deadline, None, None, "isolated-linux-subreaper",
                       0, 0, state, (), 0, False, False, True)


def require_complete(result, certificate):
    def fail(message):
        raise ContainmentError(message, partial_result=result,
                               invocation_id=certificate.invocation_id, certificate=certificate)
    if (type(result) is not dict or set(result) != {"exit_code", "stdout", "stderr",
            "timeout_seconds", "elapsed_seconds"} or type(result["exit_code"]) is not int
            or type(result["stdout"]) is not str or type(result["stderr"]) is not str
            or not _finite(result["timeout_seconds"]) or result["timeout_seconds"] < 0
            or not _finite(result["elapsed_seconds"]) or result["elapsed_seconds"] < 0):
        fail("invalid public result")
    c = certificate
    if (not _hex(c.invocation_id, 32) or
            _identity(list(c.identity.argv), c.identity.cwd, dict(c.identity.environment)) != c.identity
            or c.errors or c.errors_dropped):
        fail("invalid or non-success certificate")
    if (c.state == "not-started" and c.reason == "worker-spawn-error"
            and c.spawn_error and result["exit_code"] == 127 and not result["stdout"]
            and result["stderr"] == c.spawn_error and c.leader_pid is None
            and c.leader_start_ticks is None and c.leader_wait_status is None
            and c.reaped_count == c.adopted_count == 0 and c.worker_reaped
            and not c.terminal_received):
        return
    if not (c.state == "complete-echild" and c.worker_pidfd_authenticated and c.worker_reaped
            and c.terminal_received and c.output_complete and c.leader_wait_status is not None
            and c.cleanup_started is not None and c.cleanup_deadline is not None):
        fail("missing complete authenticated containment barrier")


@dataclass
class ChildHandle:
    pid: int
    pidfd: int


@dataclass
class WorkerState:
    leader_pid: int | None = None
    leader_wait_status: int | None = None
    reaped_count: int = 0
    adopted_count: int = 0
    authenticated_pids: set[int] = field(default_factory=set)
    live_handles: dict[int, ChildHandle] = field(default_factory=dict)
    term_sent: set[int] = field(default_factory=set)
    cleanup_started: float | None = None
    cleanup_deadline: float | None = None
    reason: str | None = None
    errors: list[str] = field(default_factory=list)
    errors_dropped: int = 0
    launch_closed: bool = False


def _select_pidfd_bindings(native_open, native_signal, libc=None):
    if native_open is None or native_signal is None:
        libc = libc if libc is not None else ctypes.CDLL(None, use_errno=True)
    c_open = None if native_open is not None else getattr(libc, "pidfd_open")
    c_signal = None if native_signal is not None else getattr(libc, "pidfd_send_signal")
    if c_open is not None:
        c_open.argtypes = (ctypes.c_int, ctypes.c_uint)
        c_open.restype = ctypes.c_int
    if c_signal is not None:
        c_signal.argtypes = (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
        c_signal.restype = ctypes.c_int

    def opened(pid):
        fd = native_open(pid, 0) if native_open is not None else c_open(pid, 0)
        if fd < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        try:
            os.set_inheritable(fd, False)
        except BaseException:
            os.close(fd)
            raise
        return fd

    def signalled(fd, sig):
        if native_signal is not None:
            native_signal(fd, sig, None, 0)
        elif c_signal(fd, sig, None, 0) < 0:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    return opened, signalled


def _bindings():
    global _pidfd_api
    if _pidfd_api is None:
        _pidfd_api = _select_pidfd_bindings(getattr(os, "pidfd_open", None),
                                           getattr(signal, "pidfd_send_signal", None))
    return _pidfd_api


def _pidfd_open(pid: int) -> int:
    return _bindings()[0](pid)


def _pidfd_signal(fd: int, sig: int) -> None:
    _bindings()[1](fd, sig)


def _waitid_child(fd):
    return os.waitid(_P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)


def _wait_all():
    return os.waitpid(-1, os.WNOHANG)


def _capability_preflight():
    """Self-only, before any spawn. No policy mutation in a hosting caller."""
    if sys.platform != "linux" or type(_P_PIDFD) is not int:
        raise RuntimeError("Linux P_PIDFD required")
    for name in ("waitid", "waitpid"):
        if not callable(getattr(os, name, None)):
            raise RuntimeError(f"missing os.{name}")
    for name in ("WEXITED", "WNOHANG", "WNOWAIT"):
        if type(getattr(os, name, None)) is not int:
            raise RuntimeError(f"missing os.{name}")
    fd = _pidfd_open(os.getpid())
    try:
        with open(f"/proc/self/fdinfo/{fd}", "rb") as stream:
            info = stream.read(4097)
        pids = [line.split()[1] for line in info.splitlines() if line.startswith(b"Pid:")]
        if len(info) > 4096 or pids != [str(os.getpid()).encode()] or os.get_inheritable(fd):
            raise RuntimeError("self pidfd identity/CLOEXEC failure")
        _pidfd_signal(fd, 0)
        try:
            _waitid_child(fd)
        except ChildProcessError as error:
            if error.errno != errno.ECHILD:
                raise
        else:
            raise RuntimeError("self waitid must report ECHILD")
    finally:
        os.close(fd)


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if value is not None and libc.prctl(36, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER")
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_GET_CHILD_SUBREAPER")
    return current.value


def _verify_worker_invariants(state):
    with os.scandir("/proc/self/task") as tasks:
        if sum(1 for _ in tasks) != 1:
            raise RuntimeError("worker must have exactly one thread")
    if (_subreaper() != 1 or signal.getsignal(signal.SIGCHLD) != signal.SIG_DFL
            or not state.launch_closed):
        raise RuntimeError("worker subreaper/SIGCHLD/closed-launch invariant")


def _prepare_worker(state):
    _capability_preflight()
    signal.signal(signal.SIGCHLD, signal.SIG_DFL)
    if _subreaper(1) != 1:
        raise RuntimeError("subreaper verification failed")
    state.launch_closed = True
    _verify_worker_invariants(state)
    if _read_children() or not _reap_batch(state):
        raise RuntimeError("worker already owns children")
    state.launch_closed = False


def _parse_children(data):
    if len(data) > CHILD_READ_BYTES:
        raise ValueError("children read exceeds bound")
    if len(data) == CHILD_READ_BYTES and data and data[-1:] not in b" \t\r\n\v\f":
        # Only a cap-length read can end with a partial token.
        end = max((i for i, b in enumerate(data) if b in b" \t\r\n\v\f"), default=-1)
        data = data[:end + 1]
    ids = []
    seen = set()
    for word in data.split():
        if not word.isdigit() or not word.isascii() or int(word) <= 0:
            raise ValueError("invalid children token")
        pid = int(word)
        if pid in seen:
            raise ValueError("duplicate child identity")
        ids.append(pid)
        seen.add(pid)
    return ids


def _read_children() -> list[int]:
    with open(f"/proc/self/task/{threading.get_native_id()}/children", "rb") as stream:
        return _parse_children(stream.read(CHILD_READ_BYTES))


def _record_error(state, operation, error):
    if len(state.errors) < 64:
        text = f"{operation}: {type(error).__name__}: {error}"
        state.errors.append(text.encode("utf-8", errors="replace")[:512].decode("utf-8", errors="ignore"))
    else:
        state.errors_dropped += 1


def _acquire_batch(state: WorkerState) -> list[ChildHandle]:
    try:
        ids = _read_children()
    except (OSError, ValueError) as error:
        _record_error(state, "read children", error)
        return list(islice(state.live_handles.values(), CHILD_BATCH))
    term_phase = (state.cleanup_started is not None and
                  time.monotonic() < state.cleanup_started + TERM_GRACE_SECONDS)
    candidates = [pid for pid in ids if not (
        term_phase and pid in state.live_handles and
        state.live_handles[pid].pidfd in state.term_sent)]
    acquired = []
    try:
        for pid in candidates[:CHILD_BATCH]:
            if state.cleanup_deadline is not None and time.monotonic() >= state.cleanup_deadline:
                break
            current = state.live_handles.get(pid)
            if current is None:
                try:
                    fd = _pidfd_open(pid)
                    try:
                        _waitid_child(fd)
                    except BaseException:
                        os.close(fd)
                        raise
                except OSError as error:
                    _record_error(state, "acquire child", error)
                    continue
                current = ChildHandle(pid, fd)
                state.live_handles[pid] = current
                if pid not in state.authenticated_pids:
                    state.authenticated_pids.add(pid)
                    if not (pid == state.leader_pid and state.leader_wait_status is None):
                        state.adopted_count += 1
            acquired.append(current)
        return acquired
    finally:
        ids.clear()
        candidates.clear()  # Never carry unprocessed numeric IDs across a reap.


def _retire_handle(state, handle):
    state.live_handles.pop(handle.pid, None)
    state.term_sent.discard(handle.pidfd)
    os.close(handle.pidfd)


def _signal_handles(state: WorkerState, handles: list[ChildHandle], sig: int) -> None:
    for handle in handles:
        now = time.monotonic()
        if state.cleanup_deadline is not None and now >= state.cleanup_deadline:
            break
        if (sig == signal.SIGTERM and state.cleanup_started is not None and
                now >= state.cleanup_started + TERM_GRACE_SECONDS):
            sig = signal.SIGKILL
        if sig == signal.SIGTERM and handle.pidfd in state.term_sent:
            continue
        try:
            _pidfd_signal(handle.pidfd, sig)
        except ProcessLookupError:
            # Target readiness is not a tree-emptiness observation.
            pass
        except OSError as error:
            _record_error(state, "signal child", error)
        if sig == signal.SIGTERM:
            state.term_sent.add(handle.pidfd)
        elif not (handle.pid == state.leader_pid and state.leader_wait_status is None):
            _retire_handle(state, handle)


def _reap_batch(state: WorkerState) -> bool:
    for _ in range(REAP_BATCH):
        if state.cleanup_deadline is not None and time.monotonic() >= state.cleanup_deadline:
            return False
        try:
            pid, status = _wait_all()
        except InterruptedError:
            continue
        except ChildProcessError as error:
            if error.errno == errno.ECHILD:
                return True
            _record_error(state, "wait children", error)
            return False
        except OSError as error:
            _record_error(state, "wait children", error)
            return False
        if pid == 0:
            return False
        original_leader = pid == state.leader_pid and state.leader_wait_status is None
        if original_leader:
            state.leader_wait_status = status
        if not original_leader and pid not in state.authenticated_pids:
            state.adopted_count += 1
        state.authenticated_pids.discard(pid)
        handle = state.live_handles.get(pid)
        if handle is not None:
            _retire_handle(state, handle)
        state.reaped_count += 1
    return False


def _begin_cleanup(state: WorkerState, reason: str, now: float) -> None:
    if state.cleanup_started is None:
        state.cleanup_started = now
        state.cleanup_deadline = now + WORKER_CLEANUP_SECONDS
        state.reason = reason
    state.launch_closed = True


def _close_handles(state):
    for handle in list(state.live_handles.values()):
        _retire_handle(state, handle)


def _close_except(keep):
    # Close every descriptor >= 3 not kept. Enumerating procfs is exact and avoids
    # a close() syscall per slot of a very large RLIMIT_NOFILE (about 0.15 s each).
    keep = set(keep)
    with os.scandir("/proc/self/fd") as entries:
        open_fds = [int(entry.name) for entry in entries]
    for fd in open_fds:
        if fd >= 3 and fd not in keep:
            try:
                os.close(fd)
            except OSError as error:
                if error.errno != errno.EBADF:  # The listing's own descriptor.
                    raise


def _launch_workload(state, identity, stdout_fd, stderr_fd, exec_error_fd):
    """Fork only in the dedicated owner; return a one-byte parent-release gate.

    Caller records diagnostic start ticks and acquires the pinned leader pidfd
    before release. No launch transport or public core API is supplied here.
    """
    if state.launch_closed or state.leader_pid is not None:
        raise RuntimeError("workload launch gate closed")
    _capability_preflight()
    gate_read, gate_write = os.pipe2(os.O_CLOEXEC)
    try:
        pid = os.fork()
    except BaseException:
        os.close(gate_read)
        os.close(gate_write)
        raise
    if pid == 0:
        try:
            _close_except([gate_read, stdout_fd, stderr_fd, exec_error_fd])
            if os.read(gate_read, 1) != b"R":
                os._exit(127)
            os.close(gate_read)
            os.setsid()
            os.chdir(identity.cwd)
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            _close_except([exec_error_fd])
            os.execvpe(identity.argv[0], list(identity.argv), dict(identity.environment))
        except OSError as error:
            try:
                payload = json.dumps({"errno": error.errno, "message": str(error)},
                                     ensure_ascii=True, separators=(",", ":")).encode("ascii")
                if len(payload) > 4096:
                    payload = b'{"transport_error":"exec error exceeds private limit"}'
                os.write(exec_error_fd, payload)
            finally:
                os._exit(127)
        except BaseException:
            os._exit(127)
    state.leader_pid = pid  # Before any wait or opportunity to release identity.
    os.close(gate_read)
    return gate_write


def _worker_command(bootstrap_args):
    return [sys.executable, str(Path(__file__).resolve()), "--worker", *bootstrap_args]


def _terminal_frame(message):
    # Field limits are UTF-8 limits, but the wire encoder deliberately escapes
    # Unicode. Fit the actual encoded body, dropping only whole trailing errors.
    while len(_json_bytes(message)) > MAX_REPLY_BYTES and message["errors"]:
        message["errors"].pop()
        message["errors_dropped"] += 1
    return _encode_frame(message, MAX_REPLY_BYTES)


def _worker_terminal(invocation_id, digest, ticks, state, deadline, leader_ticks,
                     spawn_error, complete, output_complete):
    return dict(v=1, id=invocation_id, type="TERMINAL", identity_sha256=digest,
                worker_pid=os.getpid(), worker_start_ticks=ticks,
                leader_pid=state.leader_pid, leader_start_ticks=leader_ticks,
                leader_wait_status=state.leader_wait_status, spawn_error=spawn_error,
                reason=state.reason or "preflight", execution_deadline=float(deadline),
                cleanup_started=state.cleanup_started, cleanup_deadline=state.cleanup_deadline,
                method="isolated-linux-subreaper", adopted_count=state.adopted_count,
                reaped_count=state.reaped_count,
                state="complete-echild" if complete else "not-started" if state.leader_pid is None else "unknown",
                errors=list(state.errors), errors_dropped=state.errors_dropped,
                output_complete=output_complete)


def _worker_main(bootstrap_args):
    if len(bootstrap_args) != 8:
        raise ValueError("expected eight bootstrap arguments")
    invocation_id, life, control_fd, caller_fd, forward_out, forward_err, stdin_fd, digest = bootstrap_args
    if not _hex(invocation_id, 32) or not _hex(digest, 64):
        raise ValueError("invalid bootstrap identity")
    life, control_fd, caller_fd, forward_out, forward_err, stdin_fd = map(int,
        (life, control_fd, caller_fd, forward_out, forward_err, stdin_fd))
    private = [life, control_fd, caller_fd, forward_out, forward_err]
    if len(set(private)) != 5 or any(fd < 3 for fd in private) or stdin_fd != -1:
        raise ValueError("invalid bootstrap FD ledger")
    for fd in private:
        os.set_inheritable(fd, False)
        os.set_blocking(fd, False)
    _close_except(private)
    control = socket.socket(fileno=control_fd)
    state = WorkerState()
    parser = _FrameParser(MAX_START_BYTES)
    outgoing = bytearray()
    queues = [bytearray(), bytearray()]
    captures = {}
    exec_error = bytearray()
    ticks = _ticks(os.getpid())
    leader_ticks = None
    spawn_error = None
    execution_deadline = time.monotonic() + TOTAL_CLEANUP_SECONDS
    startup_deadline = execution_deadline
    complete = False
    start = None
    control_open = True
    forwarding = [forward_out, forward_err]
    pending_signal = [False]
    selector = None
    terminal_queued = False
    sent_terminal = False

    def signal_pending(signum, frame):
        pending_signal[0] = True

    signal.signal(signal.SIGINT, signal_pending)
    signal.signal(signal.SIGTERM, signal_pending)

    def caller_lost():
        if select.select([caller_fd], [], [], 0)[0]:
            return True
        try:
            return os.read(life, 1) == b""
        except BlockingIOError:
            return False

    def stop(reason):
        _begin_cleanup(state, reason, time.monotonic())

    try:
        try:
            if sys.version_info < (3, 10) or tuple(map(int, platform.release().split(".")[:2])) < (5, 4):
                raise RuntimeError("Linux 5.4 / CPython 3.10 required")
            _prepare_worker(state)
            _check_sigchld()
            _pidfd_signal(caller_fd, 0)
            if _pidfd_pid(caller_fd) != os.getppid() or caller_lost():
                raise RuntimeError("caller is not live authenticated parent")
            selector = selectors.DefaultSelector()
            for fd in (life, caller_fd, control_fd):
                selector.register(fd, selectors.EVENT_READ)
            selector.select(0)
            outgoing.extend(_encode_frame(dict(v=1, id=invocation_id, type="READY",
                worker_pid=os.getpid(), worker_start_ticks=ticks, identity_sha256=digest), MAX_REPLY_BYTES))
        except BaseException as error:
            _record_error(state, "worker capability", error)
            stop("capability-failure")

        while True:
            now = time.monotonic()
            # Lifetime checks precede receipt/validation of START and run again
            # immediately before fork. No channel is an output/protocol alias.
            if caller_lost() or pending_signal[0]:
                stop("caller-lost" if not pending_signal[0] else "worker-signal")
            if start is None and now >= startup_deadline:
                stop("startup-deadline")
            if control_open:
                try:
                    data = control.recv(IO_CHUNK_BYTES)
                except BlockingIOError:
                    data = None
                except OSError as error:
                    data = b""
                    _record_error(state, "control read", error)
                try:
                    if data == b"":
                        parser.eof()
                        control_open = False
                        stop("control-disconnect")
                    elif data:
                        messages = parser.feed(data)
                        for message in messages:
                            if message.get("type") == "CANCEL":
                                _validate_message(message, "CANCEL", invocation_id, digest)
                                if state.reason == "cancelled":
                                    raise ValueError("duplicate CANCEL")
                                stop("cancelled")
                            else:
                                if start is not None or state.launch_closed:
                                    raise ValueError("START after closed launch gate")
                                _validate_message(message, "START", invocation_id, digest)
                                if len(data) > MAX_START_BYTES + 4:
                                    raise ValueError("oversized START")
                                start = message
                                execution_deadline = message["execution_deadline"]
                                parser.limit = MAX_CANCEL_BYTES
                        # Reject a buffered/trailing START before any fork.
                        if start is not None and parser.buffer:
                            raise ValueError("trailing data after START")
                except (ValueError, UnicodeError, TypeError) as error:
                    _record_error(state, "control protocol", error)
                    stop("protocol-error")
                    control_open = False

            if start is not None and state.leader_pid is None and not state.launch_closed:
                if (caller_lost() or not control_open or pending_signal[0]
                        or time.monotonic() >= execution_deadline):
                    stop("deadline-before-start")
                else:
                    pairs = []
                    gate = None
                    try:
                        for _ in range(3):
                            pair = os.pipe2(os.O_CLOEXEC)
                            pairs.append(pair)
                            os.set_blocking(pair[0], False)
                        identity = _identity(start["argv"], start["cwd"], start["environment"])
                        gate = _launch_workload(state, identity, pairs[0][1], pairs[1][1], pairs[2][1])
                        # The original unreaped leader pins this number until its
                        # first wait. Failed acquisition never releases the gate.
                        fd = _pidfd_open(state.leader_pid)
                        try:
                            _waitid_child(fd)
                        except BaseException:
                            os.close(fd)
                            raise
                        state.live_handles[state.leader_pid] = ChildHandle(state.leader_pid, fd)
                        state.authenticated_pids.add(state.leader_pid)
                        leader_ticks = _ticks(state.leader_pid)
                        os.write(gate, b"R")
                    except BaseException as error:
                        _record_error(state, "workload launch", error)
                        stop("launch-error")
                    finally:
                        if gate is not None:
                            os.close(gate)
                        for index, (reader, writer) in enumerate(pairs):
                            os.close(writer)
                            captures[reader] = index

            now = time.monotonic()
            if state.leader_pid is not None and state.cleanup_started is None:
                try:
                    leader = state.live_handles[state.leader_pid]
                    exited = _waitid_child(leader.pidfd) is not None
                    if exited:
                        stop("leader-exit")
                    elif now >= execution_deadline:
                        stop("timeout")
                except BaseException as error:
                    _record_error(state, "leader observation", error)
                    stop("observation-error")

            if state.cleanup_started is not None and not complete:
                if now >= state.cleanup_deadline:
                    _record_error(state, "cleanup", TimeoutError("worker cleanup expired"))
                    break
                sig = signal.SIGTERM if now < state.cleanup_started + TERM_GRACE_SECONDS else signal.SIGKILL
                handles = _acquire_batch(state)
                leader = state.live_handles.get(state.leader_pid)
                if leader is not None and leader not in handles and len(handles) < CHILD_BATCH:
                    handles.append(leader)
                _signal_handles(state, handles, sig)
                handles.clear()
                if _reap_batch(state):
                    _verify_worker_invariants(state)
                    _check_sigchld()
                    complete = True

            # At most one fixed-size chunk per descriptor each round; neither
            # readable noisy producers nor a full forward pipe monopolize cleanup.
            for fd, index in list(captures.items()):
                try:
                    data = os.read(fd, IO_CHUNK_BYTES if index != 2 else 4097)
                    if not data:
                        os.close(fd)
                        del captures[fd]
                    elif index == 2:
                        exec_error.extend(data)
                        if len(exec_error) > 4096:
                            _record_error(state, "exec error", ValueError("oversized exec error transport"))
                            exec_error.clear()
                            stop("exec-error-transport")
                    else:
                        queues[index].extend(data)
                except BlockingIOError:
                    continue
                except (OSError, MemoryError) as error:
                    # Undeliverable output is discarded; cleanup keeps its original deadline.
                    _record_error(state, "output capture", error)
                    stop("output-error")
                    if fd in captures:
                        del captures[fd]
                        try:
                            os.close(fd)
                        except OSError:
                            pass
            for i, fd in enumerate(forwarding):
                if fd >= 0 and queues[i]:
                    try:
                        count = os.write(fd, queues[i][:IO_CHUNK_BYTES])
                        del queues[i][:count]
                    except BlockingIOError:
                        pass
                    except (OSError, MemoryError) as error:
                        _record_error(state, "output forward", error)
                        stop("output-error")
                        os.close(fd)
                        forwarding[i] = -1
                        queues[i].clear()
            if complete and not captures and not any(queues) and not terminal_queued:
                if exec_error:
                    try:
                        error = json.loads(exec_error, object_pairs_hook=_unique_object)
                        if (type(error) is not dict or set(error) != {"errno", "message"}
                                or type(error["errno"]) is not int or not _text(error["message"], 4096)):
                            raise ValueError("invalid/oversized exec-error transport marker")
                        spawn_error = error["message"]
                    except (ValueError, TypeError, UnicodeError) as error:
                        _record_error(state, "exec error transport", error)
                if state.leader_pid is None:
                    complete = False  # No-workload proof is not a tree certificate.
                outgoing.extend(_terminal_frame(_worker_terminal(invocation_id, digest, ticks,
                    state, execution_deadline, leader_ticks, spawn_error,
                    complete, all(fd >= 0 for fd in forwarding))))
                terminal_queued = True
                for i, fd in enumerate(forwarding):
                    if fd >= 0:
                        os.close(fd)
                        forwarding[i] = -1
            if outgoing and control_open:
                try:
                    count = control.send(outgoing[:IO_CHUNK_BYTES])
                    if not count:
                        raise BrokenPipeError("control send returned zero")
                    del outgoing[:count]
                except BlockingIOError:
                    pass
                except OSError:
                    control_open = False
                    stop("control-disconnect")
            if terminal_queued and (not outgoing or not control_open):
                sent_terminal = not outgoing
                break
            if complete and not control_open and not captures:
                break
            # Includes output/terminal drain in the original cleanup allowance.
            if state.cleanup_deadline is not None and time.monotonic() >= state.cleanup_deadline:
                _record_error(state, "delivery", TimeoutError("worker cleanup/delivery expired"))
                break
            select.select([], [], [], LOOP_SLICE_SECONDS)
        if not terminal_queued and control_open:
            # Best effort bounded UNKNOWN record; never blocks or claims success.
            try:
                control.send(_terminal_frame(_worker_terminal(invocation_id, digest, ticks,
                    state, execution_deadline, leader_ticks, spawn_error, False, False)))
            except (OSError, ValueError):
                pass
        return 0 if complete and sent_terminal and not state.errors and not state.errors_dropped else 1
    finally:
        _close_handles(state)
        for fd in captures:
            os.close(fd)
        for fd in forwarding:
            if fd >= 0:
                os.close(fd)
        if selector is not None:
            selector.close()
        control.close()
        os.close(life)
        os.close(caller_fd)


def run_command_core(command: list[str], cwd: Path, env: dict[str, str] | None = None,
                     *, timeout_seconds: float = 300):
    started = time.monotonic()
    invocation_id = secrets.token_hex(16)
    valid_timeout = _finite(timeout_seconds) and timeout_seconds >= 0
    deadline = started + timeout_seconds if valid_timeout else started
    # Even invalid arguments get an invocation-local diagnostic, not a shared
    # mutable last-result slot. This identity is never sent on validation failure.
    identity = CommandIdentity((), "", (), "")
    result = dict(exit_code=127, stdout="", stderr="", timeout_seconds=timeout_seconds,
                  elapsed_seconds=0.0)
    certificate = _empty_certificate(invocation_id, identity, deadline)
    ledger = []
    sockets = []
    authority = None
    process = None
    terminal = None
    ready = None
    start_sent = False
    control = None
    life_writer = None
    error = None
    shutdown_deadline = None  # Only ever reduced; never extended or reset.

    def close_fd(fd):
        if fd in ledger:
            ledger.remove(fd)
            os.close(fd)

    try:
        try:
            if not valid_timeout:
                raise ValueError("timeout must be finite and nonnegative")
            identity = _identity(command, str(Path(cwd).resolve()), dict(os.environ) if env is None else env.copy())
            certificate = _empty_certificate(invocation_id, identity, deadline)
            start_message = dict(v=1, id=invocation_id, type="START", argv=list(identity.argv),
                cwd=identity.cwd, environment=dict(identity.environment), identity_sha256=identity.sha256,
                execution_deadline=float(deadline), timeout_seconds=timeout_seconds)
            start_frame = _encode_frame(start_message, MAX_START_BYTES)
            if not _wait_scope.get():
                raise RuntimeError("explicit exclusive caller_wait_scope required")
            _check_sigchld()
            _capability_preflight()
            # Probe selector on a self-only fd, before launching anything.
            caller_fd = _pidfd_open(os.getpid())
            ledger.append(caller_fd)
            with selectors.DefaultSelector() as selector:
                selector.register(caller_fd, selectors.EVENT_READ)
                selector.select(0)
            if time.monotonic() >= deadline:
                raise TimeoutError("execution deadline before worker spawn")
            life_reader, life_writer = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
            ledger.extend((life_reader, life_writer))
            control, worker_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
            sockets.extend((control, worker_control))
            for channel in sockets:
                channel.setblocking(False)
            outputs = []
            for _ in range(2):
                pair = os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK)
                ledger.extend(pair)
                outputs.append(pair)
            worker_fds = (life_reader, worker_control.fileno(), caller_fd, outputs[0][1], outputs[1][1])
            args = [invocation_id, *map(str, worker_fds), "-1", identity.sha256]
        except BaseException as caught:
            raise ContainmentError("caller preflight blocked", partial_result=result,
                invocation_id=invocation_id, certificate=certificate, original_error=caught) from caught

        process = _BootstrapProcess.__new__(_BootstrapProcess)
        constructor_error = None
        try:
            _BootstrapProcess.__init__(process, _worker_command(args), stdin=None,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
                close_fds=True, pass_fds=worker_fds)
        except BaseException as caught:
            constructor_error = caught
        finally:
            close_fd(life_reader)
            worker_control.close()
            sockets.remove(worker_control)
            close_fd(caller_fd)
            for _, fd in outputs:
                close_fd(fd)

        if constructor_error is not None:
            close_fd(life_writer)
            life_writer = None
            child_created = getattr(process, "_child_created", False)
            child_reaped = not child_created or getattr(process, "returncode", None) is not None
            if not child_reaped:
                # No START exists. Keep this exact launch object even though
                # __init__ did not return; never reopen its numeric PID later.
                authority = AuthorityHandle(-1, process)
                # First lifeline closure fixes the invocation's cleanup bound for every later path.
                shutdown_deadline = _earlier(shutdown_deadline, time.monotonic() + TOTAL_CLEANUP_SECONDS)
                child_reaped = authority.observe_until(shutdown_deadline)
            if not child_reaped:
                raise ContainmentError("partial bootstrap not reaped", partial_result=result,
                    invocation_id=invocation_id,
                    certificate=replace(certificate, state="unknown", worker_reaped=False),
                    authority=authority, original_error=constructor_error)
            if not isinstance(constructor_error, OSError):
                raise ContainmentError("bootstrap construction failed", partial_result=result,
                    invocation_id=invocation_id, certificate=replace(certificate, worker_reaped=True),
                    original_error=constructor_error) from constructor_error
            text = str(constructor_error)
            certificate = replace(certificate, reason="worker-spawn-error", spawn_error=text,
                                  worker_reaped=True)
            result.update(stderr=text, elapsed_seconds=time.monotonic() - started)
            require_complete(result, certificate)
            return result, certificate

        # Exact direct-child wait ownership pins process.pid until this single
        # acquisition. Allocation/waitability failure never retries the open.
        authority = AuthorityHandle(-1, process)
        try:
            authority.pidfd = _pidfd_open(process.pid)
            _waitid_child(authority.pidfd)
            if _pidfd_pid(authority.pidfd) != process.pid:
                raise RuntimeError("worker pidfd identity mismatch")
            worker_ticks = _ticks(process.pid)
            certificate = replace(certificate, worker_pid=process.pid,
                worker_start_ticks=worker_ticks, worker_pidfd_authenticated=True, worker_reaped=False)
        except BaseException as caught:
            close_fd(life_writer)
            life_writer = None
            shutdown_deadline = _earlier(shutdown_deadline, time.monotonic() + TOTAL_CLEANUP_SECONDS)
            try:
                reaped = authority.observe_until(shutdown_deadline)
            except OSError:
                reaped = False
            certificate = replace(certificate, worker_reaped=reaped,
                                  state="not-started" if reaped else "unknown")
            raise ContainmentError("worker identity acquisition failed", partial_result=result,
                invocation_id=invocation_id, certificate=certificate,
                authority=None if reaped else authority, original_error=caught) from caught

        parser = _FrameParser(MAX_REPLY_BYTES)
        outgoing = bytearray()
        buffers = [bytearray(), bytearray()]
        readers = {pair[0]: i for i, pair in enumerate(outputs)}
        control_open = True
        while True:
            now = time.monotonic()
            cancellation = _current_cancellation.get()
            if cancellation is not None and cancellation.reason is not None and error is None:
                # Closing the lifeline starts worker cleanup; no deadline is reset.
                error = _cancellation_error(cancellation.reason, result, None).original_error
            if now >= deadline and shutdown_deadline is None:
                # The worker alone decides live-vs-already-exited leader status.
                shutdown_deadline = _earlier(shutdown_deadline, deadline + TOTAL_CLEANUP_SECONDS)
                if not start_sent:
                    error = error or TimeoutError("execution deadline before START")
                    close_fd(life_writer)
                    life_writer = None
                    outgoing.clear()
            if control_open:
                try:
                    data = control.recv(IO_CHUNK_BYTES)
                except BlockingIOError:
                    data = None
                except OSError as caught:
                    data = b""
                    error = error or caught
                try:
                    if data == b"":
                        parser.eof()
                        control_open = False
                        if terminal is None:
                            raise ValueError("missing TERMINAL")
                    elif data:
                        messages = parser.feed(data)
                        for message in messages:
                            if terminal is not None:
                                raise ValueError("trailing frame after TERMINAL")
                            kind = message.get("type")
                            if kind == "READY" and ready is None and not start_sent:
                                _validate_message(message, "READY", invocation_id, identity.sha256)
                                if (message["worker_pid"] != process.pid or
                                        message["worker_start_ticks"] != worker_ticks or
                                        _pidfd_pid(authority.pidfd) != message["worker_pid"]):
                                    raise ValueError("READY worker identity mismatch")
                                ready = message
                            elif kind == "TERMINAL":
                                _validate_message(message, "TERMINAL", invocation_id, identity.sha256)
                                if (message["worker_pid"] != process.pid or message["worker_start_ticks"] != worker_ticks):
                                    raise ValueError("TERMINAL worker identity mismatch")
                                if message["reason"] == "worker-spawn-error":
                                    # Reserved for the caller's own constructor-error path.
                                    raise ValueError("remote terminal claims local bootstrap error")
                                if not start_sent:
                                    if not (message["state"] == "not-started" and message["errors"]):
                                        raise ValueError("TERMINAL before START")
                                    error = error or RuntimeError("worker capability blocked")
                                elif message["execution_deadline"] != deadline:
                                    raise ValueError("TERMINAL deadline mismatch")
                                trigger = message["cleanup_started"]
                                if trigger is not None and (
                                        trigger > time.monotonic() + 0.05 or
                                        abs(message["cleanup_deadline"] - trigger - WORKER_CLEANUP_SECONDS) > 1e-6):
                                    raise ValueError("terminal cleanup timing inconsistent")
                                terminal = message
                                bound = time.monotonic() + CALLER_REAP_SECONDS
                                if trigger is not None:
                                    bound = min(bound, trigger + TOTAL_CLEANUP_SECONDS)
                                shutdown_deadline = _earlier(shutdown_deadline, bound)
                            else:
                                raise ValueError("duplicate/out-of-order control message")
                        if ready is not None and not start_sent and terminal is None and not outgoing and error is None:
                            if parser.buffer or time.monotonic() >= deadline:
                                raise ValueError("partial/trailing READY or expired START gate")
                            outgoing.extend(start_frame)
                except (ValueError, TypeError, UnicodeError) as caught:
                    error = error or caught
                    control_open = False
                    control.close()
            if outgoing and control_open and error is None:
                try:
                    count = control.send(outgoing[:IO_CHUNK_BYTES])
                    if not count:
                        raise BrokenPipeError("START send returned zero")
                    del outgoing[:count]
                    if not outgoing:
                        start_sent = True
                except BlockingIOError:
                    pass
                except OSError as caught:
                    error = error or caught
            if error is not None:
                outgoing.clear()
                if life_writer is not None:
                    close_fd(life_writer)
                    life_writer = None
                shutdown_deadline = _earlier(shutdown_deadline, time.monotonic() + TOTAL_CLEANUP_SECONDS)
            for fd, index in list(readers.items()):
                try:
                    data = os.read(fd, IO_CHUNK_BYTES)
                except BlockingIOError:
                    continue
                if data:
                    buffers[index].extend(data)
                else:
                    close_fd(fd)
                    del readers[fd]
            try:
                authority.observe_until(time.monotonic())
            except OSError as caught:
                error = error or caught
                break
            if authority.reaped and not control_open and not readers:
                break
            if shutdown_deadline is not None and time.monotonic() >= shutdown_deadline:
                error = error or TimeoutError("worker/terminal allowance expired")
                break
            select.select([], [], [], LOOP_SLICE_SECONDS)

        if terminal is not None:
            fields = {k: v for k, v in terminal.items() if k not in
                      ("v", "id", "type", "identity_sha256")}
            fields["errors"] = tuple(fields["errors"])
            certificate = Certificate(invocation_id=invocation_id, identity=identity,
                worker_pidfd_authenticated=True, terminal_received=True,
                worker_reaped=authority.reaped, **fields)
        else:
            certificate = replace(certificate, state="unknown" if start_sent else "not-started",
                                  worker_reaped=authority.reaped)
        result.update(stdout=_decode_output(bytes(buffers[0])), stderr=_decode_output(bytes(buffers[1])),
                      elapsed_seconds=time.monotonic() - started)
        if certificate.spawn_error:
            result.update(exit_code=127, stdout="", stderr=certificate.spawn_error)
        elif certificate.leader_wait_status is not None:
            result["exit_code"] = os.waitstatus_to_exitcode(certificate.leader_wait_status)
        if certificate.reason == "timeout":
            result["exit_code"] = 124
            result["stderr"] += f"\ncommand timed out after {timeout_seconds} seconds\n"
        if authority.reaped and authority.process.returncode != 0:
            error = error or RuntimeError(
                f"bootstrap worker exited abnormally: {authority.process.returncode}")
        cancellation = _current_cancellation.get()
        if cancellation is not None and cancellation.reason is not None:
            error = error or RuntimeError(f"cancelled by {cancellation.reason}")
        if error is not None or not authority.reaped:
            raise ContainmentError("command transport/ownership failed", partial_result=result,
                invocation_id=invocation_id, certificate=certificate,
                authority=None if authority.reaped else authority, original_error=error)
        require_complete(result, certificate)
        return result, certificate
    except ContainmentError:
        raise
    except BaseException as caught:
        # Independent worker lifetime still closes on any catchable caller error.
        # Task 4 extends this path with scoped cancellation and delivery faults.
        if life_writer is not None:
            close_fd(life_writer)
            life_writer = None
        if (authority is None and process is not None and getattr(process, "_child_created", False)
                and process.returncode is None):
            # Keep the exact launch object even if its handoff was interrupted.
            authority = AuthorityHandle(-1, process)
        if authority is not None and not authority.reaped:
            # The original absolute bound survives; a late failure never buys a fresh interval.
            unwind_deadline = _earlier(shutdown_deadline, time.monotonic() + TOTAL_CLEANUP_SECONDS)
            unwind_deadline = min(unwind_deadline, deadline + TOTAL_CLEANUP_SECONDS)
            while True:
                try:
                    authority.observe_until(unwind_deadline)
                    break
                except OSError:
                    break
                except BaseException:
                    # Repeated catchable interruptions never reset the absolute bound.
                    if time.monotonic() >= unwind_deadline:
                        break
        raise ContainmentError("command supervision failed", partial_result=result,
            invocation_id=invocation_id, certificate=replace(certificate,
                state="unknown", worker_reaped=bool(authority and authority.reaped)),
            authority=authority if authority and not authority.reaped else None,
            original_error=caught) from caught
    finally:
        for fd in ledger:
            os.close(fd)
        for channel in sockets:
            channel.close()
        if authority is not None and authority.reaped:
            authority.close()


if __name__ == "__main__":
    if len(sys.argv) != 10 or sys.argv[1] != "--worker":
        raise SystemExit("expected --worker and eight private bootstrap arguments")
    raise SystemExit(_worker_main(sys.argv[2:]))
