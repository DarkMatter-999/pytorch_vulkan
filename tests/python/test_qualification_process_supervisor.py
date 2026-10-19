"""CPU-only process actors and an independent, fixture-local safety owner.

The barrier receipt is immutable and precedes rescue.  The teardown receipt is a
different file: safety cleanup must never be mistaken for product containment.
Executable paths deliberately reach the dispatcher before importing pytest.
"""

from __future__ import annotations

import array
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import resource
import secrets
import select
import selectors
import signal
import socket
import struct
import subprocess
import sys
import time
import threading
import gc
from contextlib import nullcontext
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[2]
MODULE = Path(__file__).resolve()
SOURCE_PATHS = (
    "tools/qualification_process_supervisor.py",
    "tests/python/test_qualification_process_supervisor.py",
    "tools/run_vulkan_qualification.py",
    "tests/python/test_vulkan_qualification.py",
    "tools/run_mnist_training_qualification.py",
    "tests/python/test_mnist_training_qualification.py",
    "tests/python/mnist_training_evidence.py",
    "tests/python/test_mnist_training_evidence.py",
)
# Materialized only in tmp_path, with no dependency on pytest or a GPU wrapper.
ACTOR_ENTRY = "import runpy\nrunpy.run_path({module!r}, run_name='__main__')\n"
REGISTRATION_SECONDS = 1.0
SAFETY_SECONDS = 6.5
RESCUE_SECONDS = 2.0
ACTOR_SECONDS = 9.0
_teardown_unknown = False
# Linux UAPI /usr/include/linux/wait.h: #define P_PIDFD 3 (verified locally).
# No siginfo ABI or raw syscall-number fallback is used.
_P_PIDFD = getattr(os, "P_PIDFD", 3)
_pidfd_api = None


def _select_pidfd_bindings(native_open, native_signal, libc):
    """Thin fixture-local adapters; injected APIs below are unit-test seams."""
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


def _fixture_pidfd_bindings():
    global _pidfd_api
    if _pidfd_api is None:
        _pidfd_api = _select_pidfd_bindings(getattr(os, "pidfd_open", None),
                                           getattr(signal, "pidfd_send_signal", None), None)
    return _pidfd_api


def _pidfd_open(pid: int) -> int:
    return _fixture_pidfd_bindings()[0](pid)


def _pidfd_signal(fd: int, sig: int) -> None:
    _fixture_pidfd_bindings()[1](fd, sig)


def _require_fixture_bindings():
    """Presence-only selection; capability checks are separately self-only."""
    if sys.platform != "linux" or type(_P_PIDFD) is not int:
        raise RuntimeError("OS fixture requires Linux P_PIDFD")
    functions = ((os, "waitid"), (os, "waitpid"))
    constants = ((os, "WEXITED"), (os, "WNOHANG"), (os, "WNOWAIT"))
    missing = [f"{module.__name__}.{name}" for module, name in functions
               if not callable(getattr(module, name, None))]
    missing.extend(f"{module.__name__}.{name}" for module, name in constants
                   if type(getattr(module, name, None)) is not int)
    if missing:
        raise RuntimeError("OS fixture blocked before spawn; missing required Python bindings: "
                           + ", ".join(missing))
    _fixture_pidfd_bindings()


def _fixture_preflight():
    """No children: verify self identity, CLOEXEC, signal-zero and waitid ECHILD."""
    _require_fixture_bindings()
    fd = _pidfd_open(os.getpid())
    try:
        if _pidfd_pid(fd) != os.getpid() or os.get_inheritable(fd):
            raise RuntimeError("fixture self pidfd identity/CLOEXEC check failed")
        _pidfd_signal(fd, 0)
        try:
            os.waitid(_P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except ChildProcessError as error:
            if error.errno != errno.ECHILD:
                raise
        else:
            raise RuntimeError("fixture self waitid did not report ECHILD")
    finally:
        os.close(fd)


def _subreaper(value=None):
    libc = ctypes.CDLL(None, use_errno=True)
    if value is not None and libc.prctl(36, value, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "fixture PR_SET_CHILD_SUBREAPER")
    current = ctypes.c_int()
    if libc.prctl(37, ctypes.byref(current), 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "fixture PR_GET_CHILD_SUBREAPER")
    return current.value


def _frame(message):
    body = json.dumps(message, sort_keys=True, allow_nan=False).encode("ascii")
    if not 0 < len(body) <= 4096:
        raise ValueError("fixture frame size")
    return struct.pack("!I", len(body)) + body


def _pidfd_pid(fd):
    with open(f"/proc/self/fdinfo/{fd}", "rb") as stream:
        info = stream.read(4097)
    if len(info) > 4096:
        raise ValueError("oversized pidfd fdinfo")
    values = [line.split()[1] for line in info.splitlines() if line.startswith(b"Pid:")]
    if len(values) != 1:
        raise ValueError("received descriptor is not a pidfd")
    return int(values[0])


def _alive(fd):
    return not select.select([fd], [], [], 0)[0]


def _stat(pid):
    fields = Path(f"/proc/{pid}/stat").read_bytes().rsplit(b") ", 1)[1].split()
    return {"parent_pid": int(fields[1]), "start_ticks": int(fields[19]),
            "process_group": int(fields[2]), "session": int(fields[3])}


def _write_exclusive(path, record):
    with path.open("x", encoding="ascii") as stream:
        json.dump(record, stream, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _close_except(keep):
    limit = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
    if limit == resource.RLIM_INFINITY:
        limit = max(int(p.name) for p in Path("/proc/self/fd").iterdir()) + 1
    start = 3
    for fd in sorted(set(keep) | {int(limit)}):
        if fd >= start:
            os.closerange(start, fd)
            start = fd + 1


def _actor_command(config, mode, *args):
    return [sys.executable, config["actor_path"], mode, config["config_path"], *map(str, args)]


def _register(config, role):
    """A trusted actor transfers its *self* pidfd, remaining pinned until ACK."""
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + REGISTRATION_SECONDS

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("fixture identity registration deadline")
        return left

    fd = _pidfd_open(os.getpid())
    try:
        peer.settimeout(remaining())
        peer.connect(config["socket_path"])
        message = {"nonce": config["nonce"], "role": role, "pid": os.getpid(),
                   "parent_role": config["parents"][role]}
        frame = _frame(message)
        peer.settimeout(remaining())
        sent = peer.sendmsg([frame], [(socket.SOL_SOCKET, socket.SCM_RIGHTS,
                                      array.array("i", [fd]))])
        while sent < len(frame):
            peer.settimeout(remaining())
            count = peer.send(frame[sent:])
            if count == 0:
                raise BrokenPipeError("fixture registration disconnected")
            sent += count
        ack = b""
        while len(ack) < 3:
            peer.settimeout(remaining())
            chunk = peer.recv(3 - len(ack))
            if not chunk:
                raise BrokenPipeError("fixture ACK disconnected")
            ack += chunk
        if ack != b"ACK":
            raise ValueError("fixture registration not acknowledged")
        return peer
    except BaseException:
        peer.close()
        raise
    finally:
        os.close(fd)


def _event(peer, message):
    deadline = time.monotonic() + REGISTRATION_SECONDS
    data = _frame(message)
    while data:
        peer.settimeout(max(0.001, deadline - time.monotonic()))
        if time.monotonic() >= deadline:
            raise TimeoutError("fixture event deadline")
        sent = peer.send(data)
        if not sent:
            raise BrokenPipeError("fixture event disconnected")
        data = data[sent:]
    ack = b""
    while len(ack) < 3:
        if time.monotonic() >= deadline:
            raise TimeoutError("fixture event ACK deadline")
        peer.settimeout(deadline - time.monotonic())
        chunk = peer.recv(3 - len(ack))
        if not chunk:
            raise BrokenPipeError("fixture event ACK disconnected")
        ack += chunk
    if ack not in (b"ACK", b"GO!"):
        raise ValueError("fixture event not acknowledged")
    return ack


def _fork_actor(config, parent_peer, role, *, session=False):
    _fixture_preflight()
    _event(parent_peer, {"type": "expect", "role": role})
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:
        try:
            if session:
                os.setsid()
            os.set_inheritable(write_fd, True)
            _close_except([write_fd])
            os.execv(sys.executable, _actor_command(config, "--actor", role, write_fd))
        except BaseException:
            os._exit(91)
    os.close(write_fd)
    return pid, read_fd


def _await_notice(fd):
    try:
        if not select.select([fd], [], [], REGISTRATION_SECONDS)[0]:
            raise TimeoutError("fixture subtree readiness deadline")
        if os.read(fd, 1) != b"R":
            raise RuntimeError("fixture child exited before readiness")
    finally:
        os.close(fd)


def _notice(fd):
    if fd >= 0:
        try:
            os.write(fd, b"R")
        finally:
            os.close(fd)


def _hold():
    # A final, independent lifetime bound; normal safety rescue runs much sooner.
    deadline = time.monotonic() + ACTOR_SECONDS
    while time.monotonic() < deadline:
        select.select([], [], [], min(0.05, deadline - time.monotonic()))


def _wait_exact(pid):
    deadline = time.monotonic() + ACTOR_SECONDS
    while time.monotonic() < deadline:
        child, status = os.waitpid(pid, os.WNOHANG)
        if child:
            return status
        select.select([], [], [], 0.005)
    raise TimeoutError("fixture exact-child waiter deadline")


def _observe_workers(core, config, peer, role):
    """Hand safety the same pidfd of the shared wrapper's worker, observation only."""
    saved_open = core._pidfd_open
    def observed(pid):
        fd = saved_open(pid)
        if pid != os.getpid():
            _send_worker_identity(config, peer, fd, pid, role)
        return fd
    core._pidfd_open = observed


def _core_host(config, notify_fd):
    """Disposable caller that is SIGKILLed with no finally while an inherited writer survives."""
    from tools import qualification_process_supervisor as core
    peer = _register(config, "core-host")
    _event(peer, {"type": "ready"})
    saved_open, saved_command = core._pidfd_open, core._worker_command
    def observed(pid):
        fd = saved_open(pid)
        if pid != os.getpid():
            _send_worker_identity(config, peer, fd, pid)
        return fd
    core._pidfd_open = observed
    def command(args):
        # Fork after the lifeline exists: this incidental holder keeps the writer
        # (no exec, so CLOEXEC does not close it) after the caller is dead.
        if os.fork() == 0:
            try:
                time.sleep(5)  # Outlives the barrier; safety rescues it if still alive.
            finally:
                os._exit(0)
        return saved_command(args)
    core._worker_command = command
    _notice(notify_fd)
    threading.Timer(1.5, lambda: os.kill(os.getpid(), signal.SIGKILL)).start()
    with core.caller_wait_scope(exclusive_wait=True):
        core.run_command_core(_actor_command(config, "--actor", "leader", -1), ROOT, timeout_seconds=8)
    raise AssertionError("caller-death host survived")


def _core_death_caller(config, peer, concurrent):
    started = time.monotonic()
    host, notice = _fork_actor(config, peer, "core-host", session=True)
    _await_notice(notice)
    _wait_exact(host)
    # The worker alone cleans up (lifeline writer still held); safety observes it.
    time.sleep(1.0)
    result = {"exit_code": 127, "stdout": "", "stderr": "", "timeout_seconds": 8,
              "elapsed_seconds": time.monotonic() - started}
    record = {"evidence_kind": "task-3-real-core-caller-death", "result": result, "certificate": None,
        "exception": {"type": "FixtureCallerKilled", "message": "caller SIGKILLed; no summary exists",
                      "invocation_id": "none", "cleanup_state": "unknown"},
        "cleanup_elapsed_seconds": None, "term_first_at": None, "kill_first_at": None,
        "injected_fault": None, "elapsed_seconds": time.monotonic() - started}
    _write_exclusive(Path(config["directory"]) / "core-result.json", record)
    _event(peer, {"type": "summary", "result": result})
    _wait_exact(concurrent)
    return 0


def _actor(config, role, notify_fd):
    if role == "core-host":
        return _core_host(config, notify_fd)
    if role == "wire-host":
        peer = _register(config, role)
        _event(peer, {"type": "ready"})
        return _wire_driver(config, peer, die=True, notify_fd=notify_fd)
    if config["boundary"] == "core" and role not in ("sentinel", "concurrent"):
        if role == "engine-unit-actor":
            return _cleanup_engine_actor(config, notify_fd)
        if config.get("task3") and config["case"] not in _TOPOLOGY_CASES:
            return _task3_workload_actor(config, role, notify_fd)
        return _core_workload_actor(config, role, notify_fd)
    if role == "escaped-d":
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    peer = _register(config, role)
    try:
        if role in ("sentinel", "concurrent", "escaped-d"):
            _event(peer, {"type": "ready"})
            _notice(notify_fd)
            if role == "escaped-d" and not config["options"].get("retain_pipes", False):
                os.close(1)
                os.close(2)
            _hold()
            return 0
        if role == "reentry-r":
            _event(peer, {"type": "ready"})
            from tools.run_vulkan_qualification import run_command
            from tools import qualification_process_supervisor as core
            _observe_workers(core, config, peer, "inner-worker")
            with core.caller_wait_scope(exclusive_wait=True):
                result = run_command(_actor_command(config, "--actor", "inner-b", -1),
                                     ROOT, timeout_seconds=8)
            # Never invent an inner result when outer cancellation killed R.
            _write_exclusive(Path(config["directory"]) / "inner-result.json", result)
            return result["exit_code"]
        if role == "outer-a":
            pid, notice = _fork_actor(config, peer, "reentry-r")
            os.close(notice)  # Re-entry is intentionally in A's inherited group.
            deadline = time.monotonic() + REGISTRATION_SECONDS
            ready_file = Path(config["directory"]) / "inner-b-ready"
            while not ready_file.exists():
                if time.monotonic() >= deadline:
                    raise TimeoutError("nested tree did not become ready")
                select.select([], [], [], 0.002)
            _event(peer, {"type": "ready"})
            status = _wait_exact(pid)  # Exact child, as in profile re-entry.
            return os.waitstatus_to_exitcode(status)
        if role in ("leader", "inner-b", "intermediate"):
            child = "intermediate" if (role == "leader" and
                       config["options"].get("topology") == "double-fork") else "escaped-d"
            _, notice = _fork_actor(config, peer, child)
            _await_notice(notice)
            _event(peer, {"type": "ready"})
            _notice(notify_fd)
            if role == "inner-b":
                _write_exclusive(Path(config["directory"]) / "inner-b-ready", {"ready": True})
                _hold()
                return 0
            return config["options"].get("exit_code", 0) if role == "leader" else 0
        raise ValueError(f"unsupported Task 1 actor: {role}")
    finally:
        peer.close()


def _fixture_caller(config):
    peer = _register(config, "caller")
    try:
        concurrent, notice = _fork_actor(config, peer, "concurrent", session=True)
        _await_notice(notice)
        _event(peer, {"type": "ready"})
        leader = "outer-a" if config["case"] == "nested-timeout" else "leader"
        if config["boundary"] in ("core", "wire"):
            return _task3_caller(config, peer, concurrent)
        elif config["case"] == "safety-control":
            pid, notice = _fork_actor(config, peer, leader)
            _await_notice(notice)
            status = _wait_exact(pid)
            result = {"exit_code": os.waitstatus_to_exitcode(status), "stdout": "",
                      "stderr": "", "timeout_seconds": None, "elapsed_seconds": 0.0}
        else:
            # The worker observation, not this caller, releases the leader expectation.
            from tools.run_vulkan_qualification import run_command
            from tools import qualification_process_supervisor as core
            _observe_workers(core, config, peer, "cleanup-engine")
            with core.caller_wait_scope(exclusive_wait=True):
                result = run_command(_actor_command(config, "--actor", leader, -1), ROOT,
                                     timeout_seconds=1.5 if leader == "outer-a" else 1)
        response = _event(peer, {"type": "summary", "result": result})
        if response == b"GO!":
            # This launch is forbidden until the safety observer proves the tree
            # barrier; a mere returned helper result never authorizes it.
            subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=0.5)
            _event(peer, {"type": "observer", "started": True})
        deadline = time.monotonic() + ACTOR_SECONDS
        while time.monotonic() < deadline:
            child, _ = os.waitpid(concurrent, os.WNOHANG)
            if child:
                return 0
            select.select([], [], [], 0.01)
        raise TimeoutError("concurrent child-specific waiter deadline")
    finally:
        peer.close()


def _roles(case, options):
    parents = {"sentinel": "safety", "caller": "safety", "concurrent": "caller"}
    if case == "nested-timeout":
        # The shared wrapper starts an outer worker, and R's inner gate another.
        parents.update({"cleanup-engine": "caller", "outer-a": "cleanup-engine",
                        "reentry-r": "outer-a", "inner-worker": "reentry-r",
                        "inner-b": "inner-worker", "escaped-d": "inner-b"})
        jobs = ["escaped-d", "inner-b", "reentry-r", "outer-a"]
    else:
        if case == "safety-control":
            parents["leader"] = "caller"  # Safety-only control: no shared wrapper worker.
        else:
            parents.update({"cleanup-engine": "caller", "leader": "cleanup-engine"})
        if options.get("topology") == "double-fork":
            parents.update({"intermediate": "leader", "escaped-d": "intermediate"})
            jobs = ["escaped-d", "intermediate", "leader"]
        else:
            parents["escaped-d"] = "leader"
            jobs = ["escaped-d", "leader"]
    return parents, jobs


def _core_roles(case, options):
    parents = {"sentinel": "safety", "caller": "safety", "concurrent": "caller",
               "cleanup-engine": "caller", "leader": "cleanup-engine"}
    if case in ("bounded-fork-storm", "adoption-churn"):
        previous = "leader"
        for i in range(8):
            parents[f"generation-{i}"] = previous
            previous = f"generation-{i}"
    elif options.get("topology") in ("double-fork", "nested-subreaper"):
        parents.update({"intermediate": "leader", "escaped-d": "intermediate"})
    else:
        parents["escaped-d"] = "leader"
    if case == "fork-during-term":
        parents["term-child"] = "leader"
    return parents, [r for r in parents if r not in ("sentinel", "caller", "concurrent", "cleanup-engine")]


_TOPOLOGY_CASES = {"cleanup-engine", "leader-exit", "fork-during-term", "bounded-fork-storm",
                   "omitted-discovery", "adoption-churn", "cleanup-fault"}
_TASK3_CASES = {"plain-result", "caller-policy", "argv-cwd-env-stdin", "workload-private-fds",
    "nested-private-fds", "unchanged-parent-subreaper", "threaded-unrelated-child", "single-waiter",
    "unicode-error-frame", "oversize-exec-error", "preflight-fault", "workload-enoent",
    "workload-eacces", "worker-enoent", "bootstrap-allocation", "constructor-after-spawn", "constructor-after-spawn-runtime", "slow-ready",
    "output-fault", "delayed-worker-reap",
    "terminal-abnormal-exit", "terminal-abnormal-signal", "large-output", "timeout-output",
    "cancel-running", "caller-death-writer"}
_WIRE_CASES = {"startup-death", "startup-disconnect", "cancel-before-start", "deadline-before-start",
               "protocol-fault"}
_NO_WORKLOAD_CASES = {"caller-policy", "unicode-error-frame", "oversize-exec-error", "preflight-fault",
    "workload-enoent", "workload-eacces", "worker-enoent", "bootstrap-allocation",
    "constructor-after-spawn", "constructor-after-spawn-runtime", "slow-ready"} | _WIRE_CASES


def _task3_roles(case, options):
    if case in _TOPOLOGY_CASES:
        return _core_roles(case, options)
    parents = {"sentinel": "safety", "caller": "safety", "concurrent": "caller",
               "cleanup-engine": "caller", "leader": "cleanup-engine"}
    jobs = ["leader"]
    if case == "startup-death":
        parents["wire-host"] = "caller"
        parents["cleanup-engine"] = "wire-host"
    if case == "caller-death-writer":
        # The killed caller is a host below the fixture caller; safety observes the worker.
        parents["core-host"] = "caller"
        parents["cleanup-engine"] = "core-host"
    if case == "nested-private-fds":
        parents.update({"inner-worker": "leader", "inner-leader": "inner-worker"})
        jobs.append("inner-leader")
    if case == "single-waiter":
        parents.update({"intermediate": "leader", "escaped-d": "intermediate"})
        jobs += ["intermediate", "escaped-d"]
    return parents, jobs


def _send_worker_identity(config, peer, fd, pid, role="cleanup-engine"):
    # Observation-only transfer of the SAME already-authenticated child pidfd.
    # The caller is the Unix peer, so safety separately checks direct parenthood.
    os.waitid(_P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    payload = _frame({"type": "worker", "role": role, "pid": pid})
    count = peer.sendmsg([payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, array.array("i", [fd]))])
    if count < len(payload):
        peer.sendall(payload[count:])
    peer.settimeout(REGISTRATION_SECONDS)
    if peer.recv(3) != b"ACK":
        raise ValueError("worker observation ACK missing")


def _audit_private_fds(config):
    identities = set(json.loads((Path(config["directory"]) / "private-fds.json").read_text()))
    actual = []
    for entry in Path("/proc/self/fd").iterdir():
        try:
            actual.append(os.readlink(entry))
        except FileNotFoundError:
            pass
    assert not identities.intersection(actual), (identities, actual)


def _task3_workload_actor(config, role, notify_fd):
    case = config["case"]
    if case == "single-waiter":
        # Ordinary topology actor with an exited intermediate and adopted leaf.
        local = dict(config, case="leader-exit", options={"topology": "double-fork"})
        if role == "leader":
            sys.stdout.write("waiter-contract-ok\n")
            sys.stdout.flush()
        return _core_workload_actor(local, role, notify_fd)
    peer = _register(config, role)
    try:
        if case == "argv-cwd-env-stdin":
            assert Path.cwd() == Path(config["directory"]) / "cwd space-\u00e9"
            assert sys.argv[-3:] == ["a b", "", "\u00e9"]
            assert dict(os.environ) == config["exec_environment"]
            assert os.read(0, 64) == b"fixture-input\n"
        if case in ("workload-private-fds", "nested-private-fds"):
            _audit_private_fds(config)
        _event(peer, {"type": "ready"})
        _notice(notify_fd)
        if case in ("timeout-output", "cancel-running", "caller-death-writer", "output-fault"):
            print("partial-out", flush=True)
            _hold()
            return 0
        if case == "large-output":
            for fd, byte in ((1, b"o"), (2, b"e")):
                view = memoryview(byte * (2 * 1024 * 1024))
                while view:
                    view = view[os.write(fd, view[:65536]):]
            return 0
        if case == "plain-result":
            print("out", flush=True)
            print("err", file=sys.stderr, flush=True)
            code = config["options"]["exit_code"]
            if code < 0:
                # No TERM handler: preserve the actual negative signal status.
                os.kill(os.getpid(), -code)
            return code
        if case == "nested-private-fds" and role == "leader":
            from tools import qualification_process_supervisor as core
            saved_open = core._pidfd_open
            saved_command = core._worker_command
            def command(args):
                path = Path(config["directory"]) / "private-fds.json"
                identities = json.loads(path.read_text())
                identities += [os.readlink(f"/proc/self/fd/{fd}") for fd in map(int, args[1:6])]
                # This is a disposable fixture ledger, not a product receipt.
                path.write_text(json.dumps(identities))
                return saved_command(args)
            core._worker_command = command
            def observed(pid):
                fd = saved_open(pid)
                if pid != os.getpid():
                    _send_worker_identity(config, peer, fd, pid, "inner-worker")
                return fd
            core._pidfd_open = observed
            with core.caller_wait_scope(exclusive_wait=True):
                result, certificate = core.run_command_core(
                    _actor_command(config, "--actor", "inner-leader", -1), ROOT, timeout_seconds=2)
            core.require_complete(result, certificate)
            assert result["exit_code"] == 0
        print("contract-ok", flush=True)
        return 0
    finally:
        peer.close()


def _fault_worker(config, args):
    """Disposable instrumentation only; production has no fixture flags."""
    from tools import qualification_process_supervisor as core
    case = config["case"]
    real_begin = core._begin_cleanup
    real_signal = core._signal_handles
    directory = Path(config["directory"])
    omitted = [0]
    injected = [False]
    stats = {"term_first_at": None, "kill_first_at": None, "omitted_reads": 0}
    def begin(state, reason, now):
        real_begin(state, reason, now)
        if not injected[0]:
            injected[0] = True
            if case == "cleanup-fault":
                install_worker_fault(config["options"]["fault"])
            if case == "omitted-discovery":
                real_read = core._read_children
                def read():
                    omitted[0] += 1
                    stats["omitted_reads"] = omitted[0]
                    return [] if omitted[0] <= 3 else real_read()
                core._read_children = read
    def signalled(state, handles, sig):
        if handles:
            key = "term_first_at" if sig == signal.SIGTERM else "kill_first_at"
            if stats[key] is None:
                stats[key] = time.monotonic()
        return real_signal(state, handles, sig)
    core._begin_cleanup, core._signal_handles = begin, signalled
    if case == "preflight-fault":
        primitive = config["options"]["primitive"]
        def denied(*args, **kwargs):
            raise OSError(errno.EPERM, "injected " + primitive)
        if primitive in ("prctl-set", "prctl-get"):
            real_subreaper = core._subreaper
            def subreaper(value=None):
                if (primitive == "prctl-set" and value is not None) or (primitive == "prctl-get" and value is None):
                    return denied()
                return real_subreaper(value)
            core._subreaper = subreaper
        else:
            name = {"pidfd-open": "_pidfd_open", "pidfd-signal-zero": "_pidfd_signal",
                    "waitid-pidfd": "_waitid_child", "proc-children": "_read_children"}.get(primitive)
            if name:
                setattr(core, name, denied)
            else:
                core.selectors = SimpleNamespace(DefaultSelector=denied)
    if case in ("workload-private-fds", "oversize-exec-error"):
        saved_exec = os.execvpe
        def execute(*args):
            # This seam runs only AFTER the real fork child's FD closure.
            _audit_private_fds(config)
            _write_exclusive(directory / "fork-before-exec.json", {"private_absent": True})
            if case == "oversize-exec-error":
                raise OSError(2, "x" * 5000)
            return saved_exec(*args)
        os.execvpe = execute
    if case == "unicode-error-frame":
        saved_terminal = core._terminal_frame
        def terminal(message):
            # Real ECHILD was already observed. This is a transport branch, not
            # a claim that the OS generated these diagnostic errors.
            assert message["state"] == "complete-echild"
            message["errors"] = ["\u0080" * 256 for _ in range(64)]
            return saved_terminal(message)
        core._terminal_frame = terminal
    if case == "slow-ready":
        saved_encode = core._encode_frame
        def encode(message, limit):
            if message["type"] == "READY":
                time.sleep(0.5)
            return saved_encode(message, limit)
        core._encode_frame = encode
    if case == "protocol-fault":
        fault = config["options"]["fault"]
        saved_encode = core._encode_frame
        def encode_fault(message, limit):
            if message["type"] == "READY":
                if fault == "duplicate-ready":
                    data = saved_encode(message, limit)
                    return data + data
                if fault == "wrong-worker-pid":
                    message["worker_pid"] += 100000
                if fault == "oversize-reply":
                    return struct.pack("!I", core.MAX_REPLY_BYTES + 1)
                if fault == "terminal-before-start":
                    message = core._worker_terminal(args[0], args[-1], message["worker_start_ticks"],
                        core.WorkerState(), time.monotonic(), None, None, False, True)
            if message["type"] == "TERMINAL" and fault == "missing-terminal":
                return b""
            if message["type"] == "TERMINAL" and fault == "remote-spawn-error-terminal":
                # R2: a post-START remote claim of the caller-local bootstrap error.
                message.update(state="not-started", reason="worker-spawn-error",
                               spawn_error="remote claim", leader_pid=None, leader_start_ticks=None,
                               leader_wait_status=None, adopted_count=0, reaped_count=0,
                               errors=[], errors_dropped=0)
            return saved_encode(message, limit)
        core._encode_frame = encode_fault
    if case == "output-fault":
        real_os = core.os
        fired = [False]
        armed_at = time.monotonic() + 1.0  # After the leader has registered readiness.
        class LocalOS:
            def __getattr__(self, name):
                return getattr(real_os, name)
            def read(self, fd, count):
                # Post-launch capture buffering failure; lifeline/exec-error reads stay real.
                if count == core.IO_CHUNK_BYTES and not fired[0] and time.monotonic() >= armed_at:
                    fired[0] = True
                    raise MemoryError("injected output buffering failure")
                return real_os.read(fd, count)
        core.os = LocalOS()
    try:
        result = core._worker_main(args)
        if case == "delayed-worker-reap":
            time.sleep(0.8)  # Valid terminal, but the worker outlives the 0.5 s reap allowance.
        if case == "terminal-abnormal-exit":
            _write_exclusive(Path(config["directory"]) / "worker-exit-intent.json", {"status": 7})
            os._exit(7)
        if case == "terminal-abnormal-signal":
            _write_exclusive(Path(config["directory"]) / "worker-exit-intent.json",
                             {"signal": signal.SIGTERM})
            signal.signal(signal.SIGTERM, signal.SIG_DFL)
            os.kill(os.getpid(), signal.SIGTERM)
            os.pause()
        return result
    finally:
        _write_exclusive(directory / "worker-instrumentation.json", stats)


def _task3_caller(config, peer, concurrent):
    from tools import qualification_process_supervisor as core
    case = config["case"]
    if config["boundary"] == "wire":
        return _wire_caller(config, peer, concurrent)
    if case == "caller-death-writer":
        return _core_death_caller(config, peer, concurrent)
    directory = Path(config["directory"])
    options = config["options"]
    saved_open = core._pidfd_open
    private_identities = []
    def observed(pid):
        if pid != os.getpid() and case == "bootstrap-allocation" and options["fault"] == "worker-pidfd-emfile":
            raise OSError(errno.EMFILE, "injected worker identity allocation")
        fd = saved_open(pid)
        if pid != os.getpid():
            _send_worker_identity(config, peer, fd, pid)
        return fd
    core._pidfd_open = observed
    saved_command = core._worker_command
    instrumented = {"fork-during-term", "bounded-fork-storm", "omitted-discovery", "adoption-churn",
                     "cleanup-fault", "preflight-fault", "unicode-error-frame", "oversize-exec-error",
                     "terminal-abnormal-exit", "terminal-abnormal-signal",
                    "workload-private-fds", "slow-ready", "output-fault", "delayed-worker-reap"}
    def worker_command(args):
        for fd in map(int, args[1:6]):
            private_identities.append(os.readlink(f"/proc/self/fd/{fd}"))
        _write_exclusive(directory / "private-fds.json", private_identities)
        if case == "worker-enoent":
            return [str(directory / "missing-worker")]
        if case in instrumented:
            return _actor_command(config, "--fault-worker", *args)
        return saved_command(args)
    core._worker_command = worker_command
    if case in ("constructor-after-spawn", "constructor-after-spawn-runtime"):
        saved_init = core._BootstrapProcess.__init__
        def constructor(self, *args, **kwargs):
            saved_init(self, *args, **kwargs)
            if case.endswith("runtime"):
                raise RuntimeError("injected non-OSError after child creation")
            raise OSError(errno.EIO, "injected after child creation")
        core._BootstrapProcess.__init__ = constructor
    if case == "bootstrap-allocation" and options["fault"] == "channel-emfile":
        real_os = core.os
        class LocalOS:
            def __getattr__(self, name):
                return getattr(real_os, name)
            def pipe2(self, flags):
                raise OSError(errno.EMFILE, "injected first private channel allocation")
        core.os = LocalOS()
    scope = core.caller_wait_scope(exclusive_wait=True)
    saved_term = signal.getsignal(signal.SIGTERM)
    if case == "cancel-running":
        scope = core.cli_cancellation_scope()
        threading.Timer(1.0, lambda: os.kill(os.getpid(), signal.SIGTERM)).start()
    if case == "caller-policy":
        policy = options["policy"]
        if policy == "absent":
            scope = nullcontext()
        elif policy == "not-exclusive":
            scope = core.caller_wait_scope(exclusive_wait=False)
        elif policy == "sigchld-ignore":
            signal.signal(signal.SIGCHLD, signal.SIG_IGN)
        elif policy == "handler":
            signal.signal(signal.SIGCHLD, lambda *args: None)
        elif policy == "sa-nocldwait":
            action = core._Sigaction()
            action.flags = 2
            libc = ctypes.CDLL(None, use_errno=True)
            if libc.sigaction(signal.SIGCHLD, ctypes.byref(action), None) != 0:
                raise OSError(ctypes.get_errno(), "fixture set SA_NOCLDWAIT")
        elif policy == "unrecognized-abi":
            def unsupported():
                raise RuntimeError("injected unrecognized ABI")
            core._sigaction_abi = unsupported
    if case == "single-waiter":
        def forbidden(*args, **kwargs):
            raise AssertionError("competing Popen waiter")
        for name in ("poll", "wait", "communicate", "_internal_poll"):
            setattr(core._BootstrapProcess, name, forbidden)
        saved_observe = core.AuthorityHandle.observe_until
        def observe(self, deadline):
            gc.collect()
            if not self.reaped:
                with os.scandir(f"/proc/{self.process.pid}/task") as tasks:
                    assert sum(1 for _ in tasks) == 1
            return saved_observe(self, deadline)
        core.AuthorityHandle.observe_until = observe
    command = _actor_command(config, "--actor", "leader", -1)
    cwd, environment = ROOT, None
    saved_stdin = None
    if case == "argv-cwd-env-stdin":
        cwd = directory / "cwd space-\u00e9"
        cwd.mkdir()
        command += ["a b", "", "\u00e9"]
        environment = dict(config["exec_environment"])
        reader, writer = os.pipe2(os.O_CLOEXEC)
        os.write(writer, b"fixture-input\n")
        os.close(writer)
        saved_stdin = os.dup(0)
        os.dup2(reader, 0)
        os.close(reader)
    if case in ("unicode-error-frame", "workload-enoent"):
        command = [str(directory / "missing-workload")]
    if case == "workload-eacces":
        executable = directory / "not-executable"
        executable.write_text("no execute permissions\n")
        command = [str(executable)]
    waiter = None
    if case == "threaded-unrelated-child":
        # Existing independently authenticated concurrent child is waited only
        # by exact PID. It remains alive through the production barrier.
        waiter = threading.Thread(target=lambda: _wait_exact(concurrent))
        waiter.start()
    before = _subreaper()
    result = certificate = exception = None
    started = time.monotonic()
    try:
        with scope:
            result, certificate = core.run_command_core(command, cwd, environment,
                timeout_seconds=options.get("timeout_seconds", 1.6 if case in _TOPOLOGY_CASES else
                                            8 if case in ("large-output", "cancel-running") else 2))
            core.require_complete(result, certificate)
    except core.ContainmentError as error:
        result, certificate = error.partial_result, error.certificate
        exception = {"type": type(error).__name__, "message": str(error),
                     "invocation_id": error.invocation_id,
                     "cleanup_state": certificate.state if certificate else "unknown"}
    finally:
        if saved_stdin is not None:
            os.dup2(saved_stdin, 0)
            os.close(saved_stdin)
    assert _subreaper() == before
    assert signal.getsignal(signal.SIGTERM) == saved_term, "cancellation scope restored TERM policy"
    stats_path = directory / "worker-instrumentation.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    record = {"evidence_kind": "task-3-real-core", "result": result,
        "certificate": core.certificate_record(certificate) if certificate else None, "exception": exception,
        "cleanup_elapsed_seconds": time.monotonic() - certificate.cleanup_started if certificate and certificate.cleanup_started else None,
        "term_first_at": stats.get("term_first_at"), "kill_first_at": stats.get("kill_first_at"),
        "injected_fault": ({"name": options["fault"], "combined_observation_unavailable_through_expiry": options["fault"] != "cleanup-expiry"} if case == "cleanup-fault" else None),
        "elapsed_seconds": time.monotonic() - started}
    _write_exclusive(directory / "core-result.json", record)
    # The full result stays in core-result.json; the bounded fixture frame carries a prefix.
    response = _event(peer, {"type": "summary", "result": dict(result,
        stdout=result["stdout"][:1024], stderr=result["stderr"][:1024])})
    if response == b"GO!":
        subprocess.run([sys.executable, "-c", "pass"], check=True, timeout=0.5)
        _event(peer, {"type": "observer", "started": True})
    # Safety owns the teardown. No host policy repair or broad waiter here.
    if waiter is None and case != "caller-policy":
        _wait_exact(concurrent)
    elif waiter is not None:
        waiter.join(ACTOR_SECONDS)
    return 0


def _wire_driver(config, peer, *, die=False, notify_fd=-1):
    """Real bootstrap channels, explicit wire driver (not a proxy engine)."""
    from tools import qualification_process_supervisor as core
    directory = Path(config["directory"])
    invocation_id = secrets.token_hex(16)
    identity = core._identity(_actor_command(config, "--actor", "leader", -1), str(ROOT), dict(os.environ))
    execution_deadline = time.monotonic() + 0.2
    life_read, life_write = os.pipe2(os.O_NONBLOCK | os.O_CLOEXEC)
    control, child_control = socket.socketpair(socket.AF_UNIX, socket.SOCK_STREAM | socket.SOCK_CLOEXEC)
    control.setblocking(False)
    child_control.setblocking(False)
    caller_fd = _pidfd_open(os.getpid())
    outputs = [os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK) for _ in range(2)]
    worker_fds = (life_read, child_control.fileno(), caller_fd, outputs[0][1], outputs[1][1])
    args = [invocation_id, *map(str, worker_fds), "-1", identity.sha256]
    process = core._BootstrapProcess.__new__(core._BootstrapProcess)
    core._BootstrapProcess.__init__(process, core._worker_command(args), stdin=None,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
        close_fds=True, pass_fds=worker_fds)
    fd = _pidfd_open(process.pid)
    _send_worker_identity(config, peer, fd, process.pid)
    authority = core.AuthorityHandle(fd, process)
    os.close(life_read)
    os.close(caller_fd)
    child_control.close()
    for _, writer in outputs:
        os.close(writer)
    parser = core._FrameParser(core.MAX_REPLY_BYTES)
    terminal = None
    caught = None
    started = time.monotonic()
    try:
        ready = None
        while ready is None:
            if time.monotonic() - started > 1:
                raise TimeoutError("wire READY deadline")
            if select.select([control], [], [], 0.01)[0]:
                data = control.recv(core.IO_CHUNK_BYTES)
                if not data:
                    raise ValueError("wire missing READY")
                messages = parser.feed(data)
                if messages:
                    assert len(messages) == 1
                    ready = messages[0]
        core._validate_message(ready, "READY", invocation_id, identity.sha256)
        assert ready["worker_pid"] == _pidfd_pid(fd)
        if die:
            _notice(notify_fd)
            # Keep a writer duplicate outside the real host: caller pidfd, not
            # a convenient EOF, must close the startup gate.
            _write_exclusive(directory / "wire-host-ready.json", {"ready": True})
            os._exit(0)
        start = dict(v=1, id=invocation_id, type="START", argv=list(identity.argv), cwd=identity.cwd,
            environment=dict(identity.environment), identity_sha256=identity.sha256,
            execution_deadline=float(execution_deadline), timeout_seconds=0.2)
        case, fault = config["case"], config["options"].get("fault")
        if case == "startup-disconnect":
            control.close()
        elif case == "cancel-before-start":
            payload = core._encode_frame(dict(v=1, id=invocation_id, type="CANCEL", reason="fixture"), core.MAX_CANCEL_BYTES)
            control.sendall(payload + core._encode_frame(start, core.MAX_START_BYTES))
        elif case == "deadline-before-start":
            while time.monotonic() < execution_deadline:
                time.sleep(0.005)
            control.sendall(core._encode_frame(start, core.MAX_START_BYTES))
        elif case == "protocol-fault":
            if fault == "wrong-id":
                start["id"] = "f" * 32
            elif fault == "wrong-hash":
                start["identity_sha256"] = "f" * 64
            payload = core._encode_frame(start, core.MAX_START_BYTES)
            if fault == "duplicate-start":
                payload += payload
            elif fault == "truncated-frame":
                payload = payload[:-1]
            elif fault == "zero-frame":
                payload = struct.pack("!I", 0)
            elif fault == "oversize-start":
                payload = struct.pack("!I", core.MAX_START_BYTES + 1)
            elif fault == "duplicate-key":
                body = core._json_bytes(start)[:-1] + b',"v":1}'
                payload = struct.pack("!I", len(body)) + body
            elif fault == "nan-deadline":
                body = core._json_bytes(start).replace(str(execution_deadline).encode(), b"NaN")
                payload = struct.pack("!I", len(body)) + body
            control.sendall(payload)
            if fault == "truncated-frame":
                control.shutdown(socket.SHUT_WR)
        else:
            raise ValueError("unsupported wire case")
        limit = time.monotonic() + 2
        while time.monotonic() < limit:
            if control.fileno() >= 0 and select.select([control], [], [], 0)[0]:
                data = control.recv(core.IO_CHUNK_BYTES)
                if data:
                    for message in parser.feed(data):
                        core._validate_message(message, "TERMINAL", invocation_id, identity.sha256)
                        if terminal is not None:
                            raise ValueError("duplicate wire TERMINAL")
                        terminal = message
                else:
                    parser.eof()
                    control.close()
            if authority.observe_until(time.monotonic()):
                # Read any final frame before deciding terminal absence.
                if control.fileno() < 0:
                    break
            time.sleep(0.005)
        if not authority.reaped:
            raise TimeoutError("wire worker not reaped")
        if case == "protocol-fault":
            if terminal is None or terminal["errors"] or terminal["state"] != "complete-echild":
                raise ValueError("real worker rejected injected protocol frame")
            raise AssertionError("malformed protocol received clean completion")
    except (ValueError, OSError, TimeoutError) as error:
        caught = error
    finally:
        os.close(life_write)
        control.close()
        if not authority.reaped:
            authority.observe_until(time.monotonic() + 2)
        if not authority.reaped:
            raise RuntimeError("wire teardown unproven")
        authority.close()
        for reader, _ in outputs:
            os.close(reader)
    certificate = core._empty_certificate(invocation_id, identity, execution_deadline)
    if terminal:
        fields = {k: v for k, v in terminal.items() if k not in ("v", "id", "type", "identity_sha256")}
        fields["errors"] = tuple(fields["errors"])
        certificate = core.Certificate(invocation_id=invocation_id, identity=identity,
            worker_pidfd_authenticated=True, terminal_received=True, worker_reaped=True, **fields)
    return certificate, caught


def _wire_caller(config, peer, concurrent):
    from tools import qualification_process_supervisor as core
    directory = Path(config["directory"])
    fault = config["options"].get("fault")
    if config["case"] == "protocol-fault" and fault in (
            "duplicate-ready", "wrong-worker-pid", "oversize-reply", "terminal-before-start", "missing-terminal",
            "remote-spawn-error-terminal"):
        # Reply mutations still exercise the genuine core's parser over actual
        # channels; only the disposable fault worker changes bytes.
        local = dict(config, boundary="core")
        return _task3_reply_fault_caller(local, peer, concurrent)
    started = time.monotonic()
    if config["case"] == "startup-death":
        host, notice = _fork_actor(config, peer, "wire-host", session=True)
        _await_notice(notice)
        _wait_exact(host)
        # Worker lifetime is independently observed by safety, not fabricated as
        # a terminal from the dead host. Safety's bound is below ACTOR_SECONDS.
        time.sleep(0.15)
        certificate, error = None, RuntimeError("startup host exited before START")
    else:
        certificate, error = _wire_driver(config, peer)
        if error is None:
            # Intentional wire startup probes are not completed commands. Keep
            # the observed certificate unchanged, and never invent this outcome
            # for an unexpected successful launch or replace a real failure.
            assert certificate is not None and certificate.state == "not-started", "expected actual not-started wire outcome"
            error = RuntimeError("fixture withheld START or closed its launch gate; no command completion")
    result = {"exit_code": 127, "stdout": "", "stderr": "", "timeout_seconds": 0.2,
              "elapsed_seconds": time.monotonic() - started}
    record = {"evidence_kind": "task-3-real-wire", "result": result,
        "certificate": core.certificate_record(certificate) if certificate else None,
        "exception": ({"type": "ContainmentError", "message": str(error), "invocation_id": "wire",
                       "cleanup_state": certificate.state if certificate else "not-started"} if error else None),
        "cleanup_elapsed_seconds": None, "term_first_at": None, "kill_first_at": None,
        "injected_fault": None, "elapsed_seconds": time.monotonic() - started}
    _write_exclusive(directory / "core-result.json", record)
    _event(peer, {"type": "summary", "result": result})
    _wait_exact(concurrent)
    return 0


def _task3_reply_fault_caller(config, peer, concurrent):
    from tools import qualification_process_supervisor as core
    saved_command, saved_open = core._worker_command, core._pidfd_open
    core._worker_command = lambda args: _actor_command(config, "--fault-worker", *args)
    def observed(pid):
        fd = saved_open(pid)
        if pid != os.getpid():
            _send_worker_identity(config, peer, fd, pid)
        return fd
    core._pidfd_open = observed
    started = time.monotonic()
    try:
        with core.caller_wait_scope(exclusive_wait=True):
            core.run_command_core([str(Path(config["directory"]) / "absent")], ROOT, timeout_seconds=0.7)
        raise AssertionError("invalid reply returned normally")
    except core.ContainmentError as error:
        record = {"evidence_kind": "task-3-real-core-reply-injection", "result": error.partial_result,
            "certificate": core.certificate_record(error.certificate) if error.certificate else None,
            "exception": {"type": "ContainmentError", "message": str(error), "invocation_id": error.invocation_id,
                          "cleanup_state": error.certificate.state if error.certificate else "unknown"},
            "cleanup_elapsed_seconds": None, "term_first_at": None, "kill_first_at": None,
            "injected_fault": {"name": config["options"]["fault"]}, "elapsed_seconds": time.monotonic() - started}
    _write_exclusive(Path(config["directory"]) / "core-result.json", record)
    _event(peer, {"type": "summary", "result": record["result"]})
    _wait_exact(concurrent)
    return 0


def _require_cleanup_engine():
    from tools import qualification_process_supervisor as engine
    for name in ("ChildHandle", "WorkerState", "_read_children", "_acquire_batch",
                 "_signal_handles", "_reap_batch", "_begin_cleanup", "_launch_workload"):
        if not callable(getattr(engine, name, None)):
            raise RuntimeError(f"missing cleanup engine API: {name}")
    return engine


def _fork_core_actor(config, peer, role):
    """Bounded core-only fork; child registers before it can fork or exit."""
    _fixture_preflight()
    _event(peer, {"type": "expect", "role": role})
    read_fd, write_fd = os.pipe2(os.O_CLOEXEC)
    pid = os.fork()
    if pid == 0:
        try:
            # The fixture has only a tiny known ledger, not an RLIMIT-sized close
            # loop in the TERM handler. This is not a product FD-closure claim.
            fds = [int(p.name) for p in Path("/proc/self/fd").iterdir()]
            if len(fds) > 64:
                raise RuntimeError("core actor fixture FD bound")
            for fd in fds:
                if fd > 2 and fd != write_fd:
                    try:
                        os.close(fd)
                    except OSError as error:
                        if error.errno != errno.EBADF:
                            raise
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
            code = _core_workload_actor(config, role, write_fd)
            os._exit(code)
        except BaseException:
            os._exit(94)
    os.close(write_fd)
    return pid, read_fd


def _core_workload_actor(config, role, notify_fd):
    topology = config["options"].get("topology", "setsid")
    if role in ("escaped-d", "term-child") or role.startswith("generation-"):
        if topology in ("setsid", "double-fork", "nested-subreaper"):
            os.setsid()
        elif topology == "setpgid":
            os.setpgid(0, 0)
    if role == "intermediate" and topology == "nested-subreaper":
        if _subreaper(1) != 1:
            raise RuntimeError("nested fixture subreaper not verified")
        _write_exclusive(Path(config["directory"]) / "nested-subreaper.json",
                         {"verified": True, "pid": os.getpid()})
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    peer = _register(config, role)
    try:
        children = [r for r, parent in config["parents"].items()
                    if parent == role and r != "term-child"]
        storm_child = None
        if role == "leader" and config["case"] == "bounded-fork-storm":
            storm_child = children.pop()
        for child in children:
            _, notice = _fork_core_actor(config, peer, child)
            _await_notice(notice)
        if role == "leader" and config["case"] in ("fork-during-term", "bounded-fork-storm"):
            def fork_on_term(signum, frame):
                signal.signal(signal.SIGTERM, signal.SIG_IGN)
                _, notice = _fork_core_actor(config, peer, storm_child or "term-child")
                _await_notice(notice)
                _write_exclusive(Path(config["directory"]) / "term-fork.json",
                                 {"registered": True, "at": time.monotonic()})
            signal.signal(signal.SIGTERM, fork_on_term)
        _event(peer, {"type": "ready"})
        _notice(notify_fd)
        if role == "leader":
            _write_exclusive(Path(config["directory"]) / "engine-tree-ready", {"ready": True})
            if config["case"] == "leader-exit":
                return config["options"].get("exit_code", 0)
        if role == "intermediate" and topology == "double-fork":
            return 0
        if (role == "escaped-d" or role.startswith("generation-")) and not config["options"].get("retain_pipes", False):
            os.close(1)
            os.close(2)
        if role.startswith("generation-") and config["case"] == "adoption-churn":
            # Exits stagger while parent/descendant ownership is changing.
            select.select([], [], [], 0.015 * (8 - int(role.split("-")[1])))
            return 0
        _hold()
        return 0
    finally:
        peer.close()


def install_worker_fault(name: str) -> None:
    """Wrap only local engine operations in this isolated fixture worker.

    Invariant tests deliberately combine the named failure with unavailable
    all-child observation through expiry. No successful ECHILD is synthesized.
    """
    engine = _require_cleanup_engine()

    def denied(code, operation):
        def fail(*args, **kwargs):
            raise OSError(code, f"injected {operation}")
        return fail

    if name in ("acquire-echild", "acquire-esrch"):
        engine._waitid_child = denied(errno.ECHILD if name == "acquire-echild" else errno.ESRCH,
                                      name)
    elif name == "signal-eperm":
        engine._pidfd_signal = denied(errno.EPERM, name)
    elif name == "pidfd-emfile":
        engine._pidfd_open = denied(errno.EMFILE, name)
    elif name == "children-unreadable":
        engine._read_children = denied(errno.EACCES, name)
    elif name == "wait-error":
        pass  # The all-child operation below is the primary fault.
    elif name == "cleanup-expiry":
        engine._read_children = lambda: []
        engine._signal_handles = lambda *args: None
    else:
        raise ValueError(f"unsupported Task 2 fault: {name}")
    if name != "cleanup-expiry":
        engine._wait_all = denied(errno.EIO, "all-child observation unavailable through expiry")


def _cleanup_engine_actor(config, notify_fd):
    """Task 2 direct engine ONLY: no core/transport/certificate API integration."""
    engine = _require_cleanup_engine()
    state = engine.WorkerState()
    engine._prepare_worker(state)
    peer = _register(config, "cleanup-engine")
    _event(peer, {"type": "ready"})
    _notice(notify_fd)
    _event(peer, {"type": "expect", "role": "leader"})
    directory = Path(config["directory"])
    outputs = [os.pipe2(os.O_CLOEXEC | os.O_NONBLOCK) for _ in range(3)]
    identity = SimpleNamespace(argv=tuple(_actor_command(config, "--actor", "leader", -1)),
                               cwd=str(ROOT), environment=tuple(sorted(os.environ.items())))
    started = time.monotonic()
    gate = engine._launch_workload(state, identity, outputs[0][1], outputs[1][1], outputs[2][1])
    # No wait has released leader identity. Pin it, and record diagnostics BEFORE
    # releasing workload code. Safety independently registers its self pidfd.
    try:
        fd = engine._pidfd_open(state.leader_pid)
        try:
            engine._waitid_child(fd)
        except BaseException:
            os.close(fd)
            raise
        state.live_handles[state.leader_pid] = engine.ChildHandle(state.leader_pid, fd)
        state.authenticated_pids.add(state.leader_pid)
        _write_exclusive(directory / "leader-before-release.json",
                         {"pid": state.leader_pid, **_stat(state.leader_pid)})
        os.write(gate, b"R")
    finally:
        os.close(gate)
        for _, write_fd in outputs:
            os.close(write_fd)
    buffers = [bytearray(), bytearray(), bytearray()]
    readers = {pair[0]: i for i, pair in enumerate(outputs)}
    term_first = kill_first = None
    complete = False
    injected = False
    omitted = [0]
    ready_path = directory / "engine-tree-ready"
    execution_deadline = started + 3.5
    try:
        while True:
            now = time.monotonic()
            # Read leader readiness without reaping. No other waiter exists here.
            leader = state.live_handles.get(state.leader_pid)
            if state.cleanup_started is None and leader is not None:
                if engine._waitid_child(leader.pidfd) is not None:
                    engine._begin_cleanup(state, "leader-exit", now)
            if state.cleanup_started is None and ready_path.exists() and config["case"] != "leader-exit":
                engine._begin_cleanup(state, "fixture-cancel", now)
            if state.cleanup_started is None and now >= execution_deadline:
                raise RuntimeError("core fixture did not reach authenticated topology readiness")
            if state.cleanup_started is not None:
                if not injected:
                    fault = config["options"].get("fault")
                    if fault:
                        install_worker_fault(fault)
                    if config["case"] == "omitted-discovery":
                        real_read = engine._read_children
                        def omitted_read():
                            omitted[0] += 1
                            return [] if omitted[0] <= 3 else real_read()
                        engine._read_children = omitted_read
                    injected = True
                if now >= state.cleanup_deadline:
                    engine._record_error(state, "cleanup", TimeoutError("worker cleanup expired"))
                    break
                sig = signal.SIGTERM if now < state.cleanup_started + engine.TERM_GRACE_SECONDS else signal.SIGKILL
                handles = engine._acquire_batch(state)
                # The authenticated leader survives discovery failures. It never
                # authorizes opening or signalling a different raw numeric PID.
                leader = state.live_handles.get(state.leader_pid)
                if leader is not None and leader not in handles and len(handles) < engine.CHILD_BATCH:
                    handles.append(leader)
                if handles:
                    if sig == signal.SIGTERM and term_first is None:
                        term_first = time.monotonic()
                    if sig == signal.SIGKILL and kill_first is None:
                        kill_first = time.monotonic()
                    engine._signal_handles(state, handles, sig)
                handles.clear()
                if engine._reap_batch(state):
                    engine._verify_worker_invariants(state)
                    complete = True
            # One bounded I/O slice per round, even with readable noisy writers.
            for read_fd in list(readers):
                try:
                    data = os.read(read_fd, 65536)
                except BlockingIOError:
                    continue
                if data:
                    i = readers[read_fd]
                    if len(buffers[i]) + len(data) > 65536:
                        raise RuntimeError("Task 2 fixture output bound exceeded")
                    buffers[i].extend(data)
                else:
                    os.close(read_fd)
                    del readers[read_fd]
            if complete:
                break
            deadline = state.cleanup_deadline if state.cleanup_started is not None else execution_deadline
            select.select([], [], [], min(engine.LOOP_SLICE_SECONDS,
                                          max(0.0, deadline - time.monotonic())))
    finally:
        engine._close_handles(state)
        for read_fd in readers:
            os.close(read_fd)
        peer.close()
    # A labelled direct-engine diagnostic, NOT the shared Certificate type or a
    # manufactured run_command_core result. Task 3 must replace this seam.
    certificate = {"state": "complete-echild" if complete else "unknown",
                   "reaped_count": state.reaped_count, "adopted_count": state.adopted_count,
                   "leader_wait_status": state.leader_wait_status,
                   "cleanup_started": state.cleanup_started,
                   "cleanup_deadline": state.cleanup_deadline,
                   "reason": state.reason, "errors": state.errors,
                   "errors_dropped": state.errors_dropped}
    exception = None if complete and not state.errors else {
        "type": "DirectEngineFailure", "message": "cleanup fault or expiry",
        "invocation_id": "fixture-only-no-core", "cleanup_state": certificate["state"]}
    result = {"exit_code": os.waitstatus_to_exitcode(state.leader_wait_status)
              if state.leader_wait_status is not None else 127,
              "stdout": buffers[0].decode(errors="replace"),
              "stderr": buffers[1].decode(errors="replace"),
              "timeout_seconds": None, "elapsed_seconds": time.monotonic() - started}
    _write_exclusive(directory / "engine-result.json", {
        "evidence_kind": "task-2-direct-engine-not-real-core", "result": result,
        "certificate": certificate, "exception": exception,
        "cleanup_elapsed_seconds": time.monotonic() - state.cleanup_started,
        "term_first_at": term_first, "kill_first_at": kill_first,
        "injected_fault": ({"name": config["options"]["fault"],
                            "combined_observation_unavailable_through_expiry":
                            config["options"]["fault"] != "cleanup-expiry"}
                           if "fault" in config["options"] else None),
        "omitted_reads": omitted[0], "exec_error_bytes": buffers[2].decode(errors="replace")})
    return 0


class _SafetyOwner:
    """Single-threaded, safety-only registration, observation, and child waiter."""

    def __init__(self, config):
        self.config = config
        self.selector = selectors.DefaultSelector()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.setblocking(False)
        self.server.bind(config["socket_path"])
        self.server.listen(16)
        self.selector.register(self.server, selectors.EVENT_READ, None)
        self.peers = {}
        self.members = {}
        self.expected = {"sentinel", "caller"}
        self.spawned = {}
        self.owned = {}
        self.reaped = []
        self.summary = None
        self.summary_peer = None
        self.observation = None
        self.observer_started = False
        self.started = time.monotonic()
        self.deadline = self.started + SAFETY_SECONDS
        self.rescue_started = None
        self.first_kill = None
        self.echild = False
        self.receipt_written_at = None

    def spawn(self, role, mode):
        _fixture_preflight()
        pid = os.fork()
        if pid == 0:
            try:
                os.setsid()
                _close_except([])
                args = () if role == "caller" else (role, -1)
                os.execv(sys.executable, _actor_command(self.config, mode, *args))
            except BaseException:
                os._exit(92)
        # Direct, unreaped identity remains pinned. No JSON PID is authority.
        fd = _pidfd_open(pid)
        try:
            os.waitid(_P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        except BaseException:
            os.close(fd)
            raise
        self.spawned[role] = pid
        self.owned[pid] = fd

    def _drop_peer(self, state):
        for fd in state["fds"]:
            os.close(fd)
        state["fds"].clear()
        sock = state["socket"]
        self.selector.unregister(sock)
        self.peers.pop(sock.fileno())
        sock.close()

    def _queue(self, state, data):
        state["out"] += data
        self.selector.modify(state["socket"], selectors.EVENT_READ | selectors.EVENT_WRITE, state)

    def _registration(self, state, message):
        fds = state["fds"]
        if len(fds) != 1 or set(message) != {"nonce", "role", "pid", "parent_role"}:
            raise ValueError("fixture registration fields or FD count")
        role, pid = message["role"], message["pid"]
        if (type(role) is not str or role not in self.expected or role in self.members or
                type(pid) is not int or pid != state["credentials"][0] or
                state["credentials"][1:] != (os.getuid(), os.getgid()) or
                type(message["nonce"]) is not str or
                not secrets.compare_digest(message["nonce"], self.config["nonce"])):
            raise ValueError("fixture peer/nonce/expected-role authentication")
        parent_role = self.config["parents"][role]
        if message["parent_role"] != parent_role or _pidfd_pid(fds[0]) != pid or not _alive(fds[0]):
            raise ValueError("fixture pidfd/parent-role authentication")
        stat = _stat(pid)
        if parent_role == "safety":
            parent_pid = os.getpid()
            if self.spawned.get(role) != pid:
                raise ValueError("fixture bootstrap identity mismatch")
        else:
            parent = self.members[parent_role]
            if not _alive(parent["fd"]):
                raise ValueError("fixture parent exited before child registration")
            parent_pid = parent["pid"]
        if stat["parent_pid"] != parent_pid:
            raise ValueError("fixture invalid parent transition")
        fd = fds.pop()
        self.members[role] = {"pid": pid, "fd": fd, "was_ready": False,
                              "registered_at": time.monotonic(), **stat}
        state["role"] = role
        self.expected.remove(role)
        self._queue(state, b"ACK")

    def _message(self, state, message):
        if state["role"] is None:
            self._registration(state, message)
            return
        if message.get("type") == "worker" and self.config.get("worker_observed"):
            if set(message) != {"type", "role", "pid"} or len(state["fds"]) != 1:
                raise ValueError("worker observation schema/FD count")
            role, pid = message["role"], message["pid"]
            fd = state["fds"][0]
            if (role in self.members or self.config["parents"].get(role) != state["role"]
                    or type(pid) is not int or _pidfd_pid(fd) != pid or not _alive(fd)):
                raise ValueError("worker observation identity/role mismatch")
            stat = _stat(pid)
            if stat["parent_pid"] != self.members[state["role"]]["pid"]:
                raise ValueError("worker is not observer's direct child")
            state["fds"].pop()
            self.members[role] = {"pid": pid, "fd": fd, "was_ready": True,
                                  "registered_at": time.monotonic(), **stat}
            self.expected.discard(role)
            for child, parent in self.config["parents"].items():
                if parent == role:
                    self.expected.add(child)
            self._queue(state, b"ACK")
            return
        if state["fds"]:
            raise ValueError("ancillary FDs after fixture registration")
        kind, role = message.get("type"), state["role"]
        if kind == "expect" and set(message) == {"type", "role"}:
            child = message["role"]
            if (type(child) is not str or self.config["parents"].get(child) != role or
                    child in self.expected or child in self.members):
                raise ValueError("invalid fixture fork-release transition")
            self.expected.add(child)
        elif kind == "ready" and set(message) == {"type"}:
            if self.members[role]["was_ready"]:
                raise ValueError("duplicate fixture readiness")
            self.members[role]["was_ready"] = True
            self.members[role]["ready_at"] = time.monotonic()
        elif kind == "summary" and role == "caller" and set(message) == {"type", "result"}:
            if self.summary is not None:
                raise ValueError("duplicate caller summary")
            self.summary = message["result"]
            self.summary_peer = state
            self.observation = self.snapshot(None)
            o = self.observation
            permits = (o["ready"] and not any(m["alive_at_barrier"] for m in o["job_members"])
                       and all(m["reaped_at_barrier"] for m in o["job_members"])
                       and all(m["alive_at_barrier"] for m in o["unrelated_members"]))
            if self.config["boundary"] in ("core", "wire"):
                permits = (permits and o["exception"] is None and
                           o["certificate"]["state"] in ("complete-echild", "not-started") and
                           all(m["reaped_at_barrier"] for m in o["worker_members"]))
            elif self.config.get("worker_observed"):
                permits = permits and all(m["reaped_at_barrier"] for m in o["worker_members"])
            if permits:
                self._queue(state, b"GO!")
            return  # Otherwise ACK only after the immutable leak receipt.
        elif kind == "observer" and role == "caller" and message == {"type": "observer", "started": True}:
            if self.observation is None or any(m["alive_at_barrier"] for m in self.observation["job_members"]):
                raise ValueError("observer bypassed the tree barrier")
            self.observer_started = True
        else:
            raise ValueError("unsupported fixture event")
        self._queue(state, b"ACK")

    def step(self, timeout=0.01):
        for key, mask in self.selector.select(timeout):
            state = key.data
            if state is None:
                sock, _ = self.server.accept()
                sock.setblocking(False)
                state = {"socket": sock, "buffer": b"", "fds": [], "out": b"",
                         "role": None, "deadline": time.monotonic() + REGISTRATION_SECONDS,
                         "credentials": struct.unpack("3i", sock.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12))}
                self.peers[sock.fileno()] = state
                self.selector.register(sock, selectors.EVENT_READ, state)
                continue
            sock = state["socket"]
            if mask & selectors.EVENT_READ:
                data, anc, flags, _ = sock.recvmsg(8192, socket.CMSG_SPACE(8), socket.MSG_CMSG_CLOEXEC)
                for level, kind, payload in anc:
                    if level != socket.SOL_SOCKET or kind != socket.SCM_RIGHTS:
                        raise ValueError("unexpected fixture ancillary data")
                    fds = array.array("i")
                    fds.frombytes(payload[:len(payload) - len(payload) % fds.itemsize])
                    state["fds"].extend(fds)
                if flags & (socket.MSG_CTRUNC | socket.MSG_TRUNC):
                    raise ValueError("truncated fixture registration")
                if not data:
                    if state["buffer"] or state["role"] is None or state["fds"]:
                        raise ValueError("partial fixture EOF")
                    self._drop_peer(state)
                    continue
                state["buffer"] += data
                if len(state["buffer"]) > 8192:
                    raise ValueError("oversized fixture receive buffer")
                while len(state["buffer"]) >= 4:
                    length = struct.unpack("!I", state["buffer"][:4])[0]
                    if not 0 < length <= 4096:
                        raise ValueError("invalid fixture frame length")
                    if len(state["buffer"]) < 4 + length:
                        break
                    body, state["buffer"] = state["buffer"][4:4 + length], state["buffer"][4 + length:]
                    def unique(pairs):
                        result = {}
                        for k, v in pairs:
                            if k in result:
                                raise ValueError("duplicate fixture JSON key")
                            result[k] = v
                        return result
                    message = json.loads(body, object_pairs_hook=unique,
                                         parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
                    if type(message) is not dict:
                        raise ValueError("fixture message is not an object")
                    self._message(state, message)
            if mask & selectors.EVENT_WRITE and state["out"]:
                count = sock.send(state["out"])
                if not count:
                    raise BrokenPipeError("fixture ACK write disconnected")
                state["out"] = state["out"][count:]
                if not state["out"]:
                    self.selector.modify(sock, selectors.EVENT_READ, state)
        now = time.monotonic()
        for state in self.peers.values():
            if state["role"] is None and now >= state["deadline"]:
                raise TimeoutError("safety registration absolute deadline")

    def snapshot(self, error):
        def member(role):
            m = self.members.get(role)
            return {"role": role, "was_ready": bool(m and m["was_ready"]),
                    "alive_at_barrier": bool(m and _alive(m["fd"])),
                    "reaped_at_barrier": bool(m and _pidfd_pid(m["fd"]) == -1)}
        no_workload = self.config.get("task3") and self.config["case"] in _NO_WORKLOAD_CASES
        ready = ((no_workload or (not self.expected and set(self.members) == set(self.config["parents"])))
                 and all(m["was_ready"] for m in self.members.values())
                 and all(r in self.members for r in ("sentinel", "caller", "concurrent")))
        if not ready and error is None:
            error = "caller returned before authenticated full-tree readiness"
        observation = dict(case=self.config["case"], boundary=self.config["boundary"], ready=ready,
                    setup_error=error, result=self.summary, certificate=None, exception=None,
                     job_members=[member(r) for r in self.config["jobs"] if not no_workload or r in self.members],
                    unrelated_members=[member("sentinel"), member("concurrent")], worker_members=[],
                    caller_wait_status=None, caller_summary_present=self.summary is not None,
                    observer_started=self.observer_started, elapsed_seconds=time.monotonic() - self.started,
                    cleanup_elapsed_seconds=None, term_first_at=None, kill_first_at=None,
                    safety_echild=False, safety_reaped=False, handles_closed=False,
                    receipt_path=str(Path(self.config["directory"]) / "barrier-receipt.json"),
                     source_sha256=self.config["source_sha256"], injected_fault=None)
        if self.config.get("worker_observed"):
            observation["worker_members"] = [member(r) for r in ("cleanup-engine", "inner-worker") if r in self.members]
        if self.config["boundary"] in ("core", "wire"):
            if self.summary is not None:
                diagnostic = json.loads((Path(self.config["directory"]) / "core-result.json").read_text())
                for key in ("result", "certificate", "exception", "cleanup_elapsed_seconds", "term_first_at",
                            "kill_first_at", "injected_fault"):
                    observation[key] = diagnostic[key]
                observation["elapsed_seconds"] = diagnostic["elapsed_seconds"]
        return observation

    def receipt(self, observation, *, final=False):
        identities = {r: {k: v for k, v in m.items() if k != "fd"} for r, m in self.members.items()}
        record = {"observation": observation, "authenticated_identities": identities,
                  "commands": self.config["commands"], "cwd": str(ROOT),
                  "environment": self.config["environment"], "start_monotonic": self.started,
                  "record_monotonic": time.monotonic(), "observation_deadline": self.deadline,
                  "rescue_deadline": None if self.rescue_started is None else self.rescue_started + RESCUE_SECONDS,
                  "receipt_written_at": self.receipt_written_at, "first_safety_kill_at": self.first_kill,
                  "safety_pid": os.getpid(), "safety_subreaper": _subreaper(),
                  "reaped_children": self.reaped,
                  "returned_helper_would_permit_next_command": self.summary is not None,
                   "actor_entry_sha256": self.config["actor_entry_sha256"]}
        if self.config["boundary"] in ("core", "wire"):
            record["evidence_kind"] = "task-3-real-core" if self.config["boundary"] == "core" else "task-3-real-wire"
        path = Path(self.config["directory"]) / ("teardown-receipt.json" if final else "barrier-receipt.json")
        _write_exclusive(path, record)
        if not final:
            self.receipt_written_at = time.monotonic()

    def rescue(self):
        """Only this dedicated owner waits all children; repeated adoption ends at ECHILD."""
        if self.receipt_written_at is None:
            raise RuntimeError("refusing rescue before receipt fsync")
        self.rescue_started = time.monotonic()
        deadline = self.rescue_started + RESCUE_SECONDS
        while time.monotonic() < deadline:
            # Signals are exclusively retained, authenticated pidfds. This also
            # closes intermediary parents so nested orphans are adopted here.
            for fd in set(self.owned.values()) | {m["fd"] for m in self.members.values()}:
                try:
                    if self.first_kill is None:
                        self.first_kill = time.monotonic()
                    _pidfd_signal(fd, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            for _ in range(64):
                try:
                    pid, status = os.waitpid(-1, os.WNOHANG)
                except ChildProcessError:
                    self.echild = True
                    return
                if pid == 0:
                    break
                self.reaped.append({"pid": pid, "status": status, "at": time.monotonic()})
                if pid == self.spawned.get("caller"):
                    self.observation["caller_wait_status"] = status
            # A setup failure may leave a child before it registers. Discover
            # ONLY our own still-unreaped direct children, pin and authenticate
            # child waitability before any signal; never reopen a reaped PID.
            path = Path(f"/proc/self/task/{os.getpid()}/children")
            children = path.read_bytes()
            if len(children) > 65536:
                raise RuntimeError("safety children bound exceeded")
            for word in children.split():
                pid = int(word)
                if pid in self.owned:
                    continue
                fd = _pidfd_open(pid)
                try:
                    os.waitid(_P_PIDFD, fd, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                except BaseException:
                    os.close(fd)
                    raise
                self.owned[pid] = fd
            select.select([], [], [], 0.005)
        raise TimeoutError("safety teardown unknown: did not obtain ECHILD")

    def close(self):
        for state in list(self.peers.values()):
            self._drop_peer(state)
        self.selector.unregister(self.server)
        self.server.close()
        self.selector.close()
        fds = set(self.owned.values()) | {m["fd"] for m in self.members.values()}
        for fd in fds:
            os.close(fd)
        self.owned.clear()
        for member in self.members.values():
            member.pop("fd")
        os.unlink(self.config["socket_path"])

    def run(self):
        error = None
        try:
            self.spawn("sentinel", "--actor")
            self.spawn("caller", "--fixture-caller")
            while time.monotonic() < self.deadline:
                self.step()
                if self.observation is not None:
                    o = self.observation
                    blocked = any(m["alive_at_barrier"] or not m["reaped_at_barrier"] for m in o["job_members"])
                    if blocked or o["setup_error"] or self.observer_started or o["exception"] is not None:
                        break
            else:
                raise TimeoutError("fixture caller/observation watchdog")
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
        if self.observation is None or error is not None:
            self.observation = self.snapshot(error)
        self.observation["observer_started"] = self.observer_started
        self.receipt(self.observation)
        # Acknowledge a blocked caller only after the receipt has been fsynced.
        if self.summary_peer is not None and not self.observer_started:
            self._queue(self.summary_peer, b"ACK")
        self.rescue()
        self.observation["safety_echild"] = self.echild
        if self.config["boundary"] != "core":
            self.observation.update(cleanup_elapsed_seconds=time.monotonic() - self.rescue_started,
                                    kill_first_at=self.first_kill)
        self.close()
        self.observation["handles_closed"] = True
        # safety_reaped is set by the external, exact-child pytest waiter.
        self.receipt(self.observation, final=True)
        return 0 if error is None else 93


def fixture_main(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in ("--safety", "--fixture-caller", "--actor", "--fault-worker"):
        raise ValueError("expected a supported fixture executable mode")
    mode = argv[0]
    _fixture_preflight()
    config = json.loads(Path(argv[1]).read_text())
    expected = 10 if mode == "--fault-worker" else 7 if (
        mode == "--actor" and config["case"] == "argv-cwd-env-stdin" and argv[2] == "leader") else 4 if mode == "--actor" else 2
    if len(argv) != expected:
        raise ValueError("wrong fixture executable argument count")
    if mode == "--safety":
        if _subreaper(1) != 1:
            raise RuntimeError("safety subreaper verification failed")
        return _SafetyOwner(config).run()
    if mode == "--fixture-caller":
        return _fixture_caller(config)
    if mode == "--actor":
        return _actor(config, argv[2], int(argv[3]))
    return _fault_worker(config, argv[2:])


if __name__ == "__main__":
    raise SystemExit(fixture_main(sys.argv[1:]))


import pytest
from types import SimpleNamespace


def _validate_os_case(case: str, boundary: str, options: dict | None) -> dict:
    if ((boundary == "legacy" and case not in ("safety-control", "nested-timeout", "leader-exit")) or
            (boundary == "core" and case not in _TOPOLOGY_CASES | _TASK3_CASES) or
            (boundary == "wire" and case not in _WIRE_CASES) or boundary not in ("legacy", "core", "wire")):
        raise ValueError(f"unsupported Task 1 OS fixture: {boundary}/{case}")
    options = dict(options or {})
    allowed = ({"fault"} if case in ("cleanup-fault", "protocol-fault", "bootstrap-allocation") else
               {"policy"} if case == "caller-policy" else
               {"primitive"} if case == "preflight-fault" else
               {"timeout_seconds"} if case in ("slow-ready", "timeout-output") else
               {"retain_pipes"} if case == "nested-timeout" else {"retain_pipes", "topology", "exit_code"})
    topologies = (("same-group", "setpgid", "setsid", "double-fork", "nested-subreaper")
                  if boundary == "core" else ("setsid", "double-fork"))
    if (set(options) - allowed or options.get("topology", "setsid") not in topologies or
            options.get("exit_code", 0) not in (0, 7, -15) or
            type(options.get("retain_pipes", False)) is not bool):
        raise ValueError("unsupported Task 1 fixture options")
    if case == "cleanup-fault" and options.get("fault") not in (
            "acquire-echild", "acquire-esrch", "signal-eperm", "pidfd-emfile",
            "children-unreadable", "wait-error", "cleanup-expiry"):
        raise ValueError("unsupported Task 2 fault option")
    if case == "preflight-fault" and options.get("primitive") not in (
            "prctl-set", "prctl-get", "pidfd-open", "pidfd-signal-zero", "waitid-pidfd",
            "proc-children", "selector"):
        raise ValueError("unsupported Task 3 primitive option")
    if case == "caller-policy" and options.get("policy") not in (
            "absent", "not-exclusive", "sigchld-ignore", "sa-nocldwait", "handler",
            "unrecognized-abi"):
        raise ValueError("unsupported Task 3 caller policy option")
    return options


def run_os_case(tmp_path: Path, case: str, *, boundary: str = "core",
                options: dict | None = None) -> dict:
    global _teardown_unknown
    if _teardown_unknown:
        pytest.exit("Prior safety teardown unknown; no more OS fixtures authorized", returncode=2)
    options = _validate_os_case(case, boundary, options)
    if boundary == "core":
        engine = _require_cleanup_engine()
        engine._capability_preflight()
    _fixture_preflight()
    before = _subreaper()
    directory = tmp_path / "os-case"
    directory.mkdir()
    actor = directory / "actor.py"
    actor.write_text(ACTOR_ENTRY.format(module=str(MODULE)))
    config_path = directory / "config.json"
    parents, jobs = _task3_roles(case, options) if boundary in ("core", "wire") else _roles(case, options)
    config = dict(case=case, boundary=boundary, options=options, parents=parents, jobs=jobs,
                  task3=boundary in ("core", "wire"),
                  worker_observed=boundary in ("core", "wire") or (
                      boundary == "legacy" and case in ("nested-timeout", "leader-exit")),
                  directory=str(directory), actor_path=str(actor), config_path=str(config_path),
                  socket_path=str(directory / "r.sock"), nonce=secrets.token_hex(32),
                  environment=dict(os.environ),
                  source_sha256={p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in SOURCE_PATHS if (ROOT / p).is_file()},
                  actor_entry_sha256=hashlib.sha256(actor.read_bytes()).hexdigest())
    if case == "argv-cwd-env-stdin":
        config["exec_environment"] = dict(os.environ, TASK3_PRIVATE_VALUE="full environment \u00e9")
    if len(os.fsencode(config["socket_path"])) >= 108:
        raise ValueError("tmp_path is too long for a case-local Unix socket")
    command = [sys.executable, str(MODULE), "--safety", str(config_path)]
    config["commands"] = {"safety": command,
                          "caller": _actor_command(config, "--fixture-caller"),
                          "actor_templates": {r: _actor_command(config, "--actor", r, -1)
                                              for r in parents if r != "caller"}}
    _write_exclusive(config_path, config)
    with (directory / "safety.log").open("xb") as log:
        handle = None
        identity = {"pid": None, "pidfd_acquired": False, "waitid_validated": False}
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                                   start_new_session=True, close_fds=True)
        try:
            identity["pid"] = process.pid
            try:
                handle = _pidfd_open(process.pid)
                identity["pidfd_acquired"] = True
                os.waitid(_P_PIDFD, handle, os.WEXITED | os.WNOHANG | os.WNOWAIT)
                identity["waitid_validated"] = True
                process.wait(timeout=10)
                if process.returncode is None:
                    raise RuntimeError("safety exact wait did not certify reaping")
            finally:
                if handle is not None:
                    os.close(handle)
            final_path = directory / "teardown-receipt.json"
            if not final_path.is_file():
                raise RuntimeError("safety teardown receipt missing")
            record = json.loads(final_path.read_text())
            o = record["observation"]
            o["safety_reaped"] = True  # Exact wait above returned a terminal status.
            if (not o["safety_echild"] or not o["handles_closed"] or
                    record["first_safety_kill_at"] < record["receipt_written_at"]):
                raise RuntimeError("safety teardown/order unproven")
            assert _subreaper() == before, "OS fixture changed pytest's subreaper state"
            _write_exclusive(directory / "driver-receipt.json",
                             {"observation": o, "safety_exit_code": process.returncode,
                              "safety_pid": process.pid, "parent_subreaper_before": before,
                              "parent_subreaper_after": _subreaper(), "safety_pidfd_closed": True})
        except BaseException as error:
            _teardown_unknown = True
            try:
                pytest.exit(f"Safety ownership UNKNOWN; no further OS cases; {directory}: {error}", returncode=2)
            except pytest.exit.Exception as stopped:
                # Retain the exact launched object and acquired-identity facts;
                # closed FD numbers/diagnostic PIDs never become kill authority.
                stopped.safety_process = process
                stopped.safety_identity = identity
                raise stopped from error
    if process.returncode != 0 or o["setup_error"] is not None:
        _teardown_unknown = True
        pytest.exit(f"OS fixture setup failed; stopping further child trees. See {directory}", returncode=2)
    return o


def test_os_case_registration_is_narrow_and_has_no_spawn_side_effects():
    for case in ("terminal-abnormal-exit", "terminal-abnormal-signal"):
        assert _validate_os_case(case, "core", None) == {}
    for case, boundary, options in (
            ("terminal-abnormal-exit", "core", {"unexpected": True}),
            ("terminal-abnormal-signal", "core", {"exit_code": 9}),
            ("future-case", "core", {}),
            ("terminal-abnormal-exit", "wire", {})):
        with pytest.raises(ValueError):
            _validate_os_case(case, boundary, options)


@pytest.mark.parametrize("case,boundary", [("safety-control", "legacy"),
    ("terminal-abnormal-exit", "core"), ("terminal-abnormal-signal", "core")])
def test_os_case_reaches_driver_and_returns_observation_without_children(tmp_path, monkeypatch, case, boundary):
    state = _fake_safety_driver(tmp_path, monkeypatch, None)
    o = run_os_case(tmp_path, case, boundary=boundary)
    assert type(o) is dict
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]
    assert state.launches == 1  # The scoped fake Popen, never an actual process.
    assert state.config["case"] == case and state.config["boundary"] == boundary
    assert (tmp_path / "os-case" / "driver-receipt.json").is_file()


def assert_tree_reaped(o):
    assert o["setup_error"] is None, o
    assert o["ready"] is True, o
    assert o["job_members"], o
    assert all(m["was_ready"] for m in o["job_members"]), o
    assert not any(m["alive_at_barrier"] for m in o["job_members"]), o
    assert all(m["reaped_at_barrier"] for m in o["job_members"]), o
    assert all(m["alive_at_barrier"] for m in o["unrelated_members"]), o
    assert o["safety_echild"] is True, o
    assert o["safety_reaped"] is True, o
    assert o["handles_closed"] is True, o


def test_complete_success_output_is_not_truncated(tmp_path):
    o = run_os_case(tmp_path, "large-output")
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == 0
    assert o["result"]["stdout"] == "o" * (2 * 1024 * 1024)
    assert o["result"]["stderr"] == "e" * (2 * 1024 * 1024)


def test_timeout_keeps_partial_text_and_exact_suffix(tmp_path):
    o = run_os_case(tmp_path, "timeout-output", options={"timeout_seconds": 1.2})
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == 124
    assert "partial-out" in o["result"]["stdout"]
    assert o["result"]["stderr"].endswith("\ncommand timed out after 1.2 seconds\n")
    assert o["certificate"]["state"] == "complete-echild"


def test_scoped_cancellation_is_non_success_with_bounded_cleanup(tmp_path):
    o = run_os_case(tmp_path, "cancel-running")
    assert_tree_reaped(o)
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["certificate"]["state"] == "complete-echild"
    assert o["cleanup_elapsed_seconds"] <= 2.0


def test_caller_death_beats_inherited_writer(tmp_path):
    o = run_os_case(tmp_path, "caller-death-writer")
    assert_tree_reaped(o)
    assert o["certificate"] is None and o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_cleanup_engine_api_before_any_child(monkeypatch):
    def forbidden(event, args):
        if event in ("os.fork", "subprocess.Popen", "os.posix_spawn"):
            raise AssertionError("API check must not launch children")
    # The audit hook is permanent but only guards this short, no-child check.
    active = [True]
    sys.addaudithook(lambda event, args: forbidden(event, args) if active[0] else None)
    try:
        _require_cleanup_engine()
    finally:
        active[0] = False


def test_cancellation_scope_is_cli_only_and_restores_policy():
    from tools import qualification_process_supervisor as core
    saved = signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)
    core.check_cancellation()  # No scope: library callers are never cancelled.
    with core.cli_cancellation_scope() as cancellation:
        assert signal.getsignal(signal.SIGTERM) is not saved[1]
        core.check_cancellation()
        signal.raise_signal(signal.SIGTERM)
        signal.raise_signal(signal.SIGINT)  # The first reason survives repeats.
        assert cancellation.reason == "SIGTERM"
        result = dict(exit_code=0, stdout="", stderr="", timeout_seconds=1, elapsed_seconds=0.0)
        with pytest.raises(core.ContainmentError) as raised:
            core.check_cancellation(result=result)
        assert raised.value.partial_result is result
        assert raised.value.certificate.state == "not-started"
    assert (signal.getsignal(signal.SIGINT), signal.getsignal(signal.SIGTERM)) == saved
    core.check_cancellation()


@pytest.mark.parametrize("path", ["partial-constructor", "identity-acquisition"])
def test_startup_failure_paths_keep_one_cleanup_bound(monkeypatch, path):
    """No child: a handler interruption late in a startup wait must not buy a fresh interval."""
    from tools import qualification_process_supervisor as core
    deadlines = []

    def observe(self, deadline):
        deadlines.append(deadline)
        if len(deadlines) == 1:
            time.sleep(0.3)  # The startup wait nearly elapses before the handler raises.
            raise KeyboardInterrupt("handler raised near the end of the startup wait")
        return False  # The generic unwind observes with whatever bound it was given.

    def construct(self, *args, **kwargs):
        self._child_created, self.pid, self.returncode = True, 2 ** 22 + 1, None
        if path == "partial-constructor":
            raise RuntimeError("injected after child creation")

    monkeypatch.setattr(core.AuthorityHandle, "observe_until", observe)
    monkeypatch.setattr(core._BootstrapProcess, "__init__", construct)
    if path == "identity-acquisition":
        real_open = core._pidfd_open

        def failing_open(pid):
            if pid == os.getpid():
                return real_open(pid)  # The self-only selector probe stays real.
            raise OSError(errno.EMFILE, "injected identity acquisition")

        monkeypatch.setattr(core, "_pidfd_open", failing_open)
    with core.caller_wait_scope(exclusive_wait=True):
        with pytest.raises(core.ContainmentError) as raised:
            core.run_command_core([sys.executable, "-c", "pass"], ROOT, timeout_seconds=30)
    assert raised.value.authority is not None
    assert len(deadlines) == 2 and deadlines[1] <= deadlines[0] + 1e-9, deadlines


def test_core_api_before_any_child():
    from tools import qualification_process_supervisor as core
    for name in ("CommandIdentity", "Certificate", "AuthorityHandle", "ContainmentError",
                 "run_command_core", "require_complete", "certificate_record",
                 "caller_wait_scope", "_BootstrapProcess", "_worker_command", "_worker_main",
                 "_decode_output", "_encode_frame", "_FrameParser"):
        assert callable(getattr(core, name, None)), name


@pytest.mark.parametrize("exit_code", [0, 7, -15])
def test_real_core_result_and_separate_certificate(tmp_path, exit_code):
    o = run_os_case(tmp_path, "plain-result", options={"exit_code": exit_code})
    assert_tree_reaped(o)
    r, c = o["result"], o["certificate"]
    assert type(r) is dict
    assert set(r) == {"exit_code", "stdout", "stderr", "timeout_seconds", "elapsed_seconds"}
    assert r["exit_code"] == exit_code
    assert r["stdout"] == "out\n" and r["stderr"] == "err\n"
    assert c["state"] == "complete-echild"
    assert c["worker_pidfd_authenticated"] and c["worker_reaped"]
    assert c["terminal_received"] and c["output_complete"]


@pytest.mark.parametrize("case,worker_status", [("terminal-abnormal-exit", 7),
                                                  ("terminal-abnormal-signal", -signal.SIGTERM)])
def test_complete_terminal_does_not_hide_abnormal_worker_exit(tmp_path, case, worker_status):
    o = run_os_case(tmp_path, case)
    intent = json.loads((tmp_path / "os-case" / "worker-exit-intent.json").read_text())
    assert intent == ({"status": 7} if worker_status == 7 else {"signal": signal.SIGTERM})
    assert o["exception"]["type"] == "ContainmentError"
    assert o["certificate"]["state"] == "complete-echild"
    assert o["certificate"]["worker_reaped"] is True
    assert o["certificate"]["terminal_received"] is True
    assert o["result"]["exit_code"] == 0
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


@pytest.mark.parametrize("case", ["startup-death", "startup-disconnect",
                                  "cancel-before-start", "deadline-before-start"])
def test_no_start_after_closed_launch_gate(tmp_path, case):
    o = run_os_case(tmp_path, case, boundary="wire")
    assert o["setup_error"] is None
    assert not o["job_members"] and o["observer_started"] is False
    assert o["certificate"] is None or o["certificate"]["state"] == "not-started"
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


@pytest.mark.parametrize("policy", ["absent", "not-exclusive", "sigchld-ignore",
                                   "sa-nocldwait", "handler", "unrecognized-abi"])
def test_unsupported_caller_never_starts(tmp_path, policy):
    o = run_os_case(tmp_path, "caller-policy", options={"policy": policy})
    assert o["exception"]["type"] == "ContainmentError"
    assert o["exception"]["cleanup_state"] == "not-started"
    assert o["job_members"] == [] and o["worker_members"] == []
    assert o["observer_started"] is False
    assert o["safety_echild"] and o["handles_closed"]


@pytest.mark.parametrize("case", ["argv-cwd-env-stdin", "workload-private-fds",
                                  "nested-private-fds", "unchanged-parent-subreaper",
                                  "threaded-unrelated-child"])
def test_launch_contract(tmp_path, case):
    o = run_os_case(tmp_path, case)
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == 0
    assert o["result"]["stdout"].endswith("contract-ok\n")


def test_worker_single_waiter_and_reap_ownership(tmp_path):
    o = run_os_case(tmp_path, "single-waiter")
    assert_tree_reaped(o)
    assert o["certificate"]["reaped_count"] >= len(o["job_members"])
    assert o["result"]["stdout"] == "waiter-contract-ok\n"


@pytest.mark.parametrize("fault", ["duplicate-ready", "wrong-id", "wrong-hash",
    "duplicate-start", "terminal-before-start", "truncated-frame", "zero-frame",
    "oversize-start", "oversize-reply", "duplicate-key", "nan-deadline",
    "missing-terminal", "wrong-worker-pid", "remote-spawn-error-terminal"])
def test_protocol_rejects_invalid_messages(tmp_path, fault):
    o = run_os_case(tmp_path, "protocol-fault", boundary="wire", options={"fault": fault})
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_unicode_diagnostics_respect_actual_reply_frame_limit(tmp_path):
    o = run_os_case(tmp_path, "unicode-error-frame")
    assert o["exception"] is not None
    assert o["certificate"]["state"] == "complete-echild"
    assert o["certificate"]["errors_dropped"] > 0
    assert o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_oversized_exec_error_is_not_a_truncated_spawn_result(tmp_path):
    o = run_os_case(tmp_path, "oversize-exec-error")
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


@pytest.mark.parametrize("primitive", ["prctl-set", "prctl-get", "pidfd-open",
    "pidfd-signal-zero", "waitid-pidfd", "proc-children", "selector"])
def test_unavailable_primitive_blocks_before_workload(tmp_path, primitive):
    o = run_os_case(tmp_path, "preflight-fault", options={"primitive": primitive})
    assert o["exception"]["cleanup_state"] == "not-started"
    assert o["job_members"] == [] and o["observer_started"] is False
    assert o["safety_echild"] and o["handles_closed"]


@pytest.mark.parametrize("case", ["workload-enoent", "workload-eacces", "worker-enoent"])
def test_authenticated_spawn_error_is_127(tmp_path, case):
    o = run_os_case(tmp_path, case)
    assert o["result"]["exit_code"] == 127 and o["result"]["stdout"] == ""
    assert o["result"]["stderr"]
    assert o["certificate"]["state"] == ("not-started" if case == "worker-enoent" else "complete-echild")
    assert o["observer_started"] is True
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


@pytest.mark.parametrize("fault", ["channel-emfile", "worker-pidfd-emfile"])
def test_bootstrap_allocation_failure_never_starts_workload(tmp_path, fault):
    o = run_os_case(tmp_path, "bootstrap-allocation", options={"fault": fault})
    assert o["exception"] is not None and o["job_members"] == []
    assert o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_partial_bootstrap_constructor_is_retained_and_reaped(tmp_path):
    o = run_os_case(tmp_path, "constructor-after-spawn")
    assert o["result"]["exit_code"] == 127
    assert o["certificate"]["state"] == "not-started"
    assert o["certificate"]["worker_reaped"] is True and o["job_members"] == []
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_non_oserror_after_bootstrap_spawn_is_contained_not_dropped(tmp_path):
    o = run_os_case(tmp_path, "constructor-after-spawn-runtime")
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["certificate"]["worker_reaped"] is True and o["job_members"] == []
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_postlaunch_output_buffer_failure_still_cleans_the_tree(tmp_path):
    o = run_os_case(tmp_path, "output-fault")
    assert_tree_reaped(o)
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["cleanup_elapsed_seconds"] is None or o["cleanup_elapsed_seconds"] <= 2.0


def test_worker_reap_after_terminal_is_bounded_to_the_caller_allowance(tmp_path):
    o = run_os_case(tmp_path, "delayed-worker-reap")
    assert_tree_reaped(o)
    assert o["exception"] is not None and o["observer_started"] is False
    assert o["certificate"]["terminal_received"] is True


def test_startup_consumes_absolute_execution_budget(tmp_path):
    o = run_os_case(tmp_path, "slow-ready", options={"timeout_seconds": 0.4})
    assert o["elapsed_seconds"] <= 2.4
    assert o["observer_started"] is False and o["job_members"] == []
    assert o["exception"] is not None


@pytest.mark.parametrize("body", [b'{"x":1,"x":2}', b'{"x":NaN}', b'[]'])
def test_frame_parser_rejects_noncanonical_objects_without_children(body):
    from tools import qualification_process_supervisor as core
    parser = core._FrameParser(64)
    with pytest.raises(ValueError):
        parser.feed(struct.pack("!I", len(body)) + body)


def test_frame_encoding_counts_ascii_json_bytes_without_children():
    from tools import qualification_process_supervisor as core
    with pytest.raises(ValueError):
        core._encode_frame({"x": "\u0080" * 16}, 64)
    parser = core._FrameParser(64)
    with pytest.raises(ValueError):
        parser.feed(struct.pack("!I", 65))


@pytest.mark.parametrize("outcome", ["withheld", "observed-failure", "unexpected-complete"])
def test_wire_withheld_start_never_authorizes_go_without_children(tmp_path, monkeypatch, outcome):
    from tools import qualification_process_supervisor as core
    module = sys.modules[__name__]
    identity = core._identity(["no-child-test"], str(ROOT), {})
    certificate = core._empty_certificate("1" * 32, identity, 10.0)
    if outcome == "unexpected-complete":
        from dataclasses import replace
        certificate = replace(certificate, state="complete-echild")
    original_error = ValueError("actual production protocol failure") if outcome == "observed-failure" else None
    config = {"case": "startup-disconnect", "options": {}, "directory": str(tmp_path)}
    replies = []
    def event(peer, message):
        assert message["type"] == "summary"
        diagnostic = json.loads((tmp_path / "core-result.json").read_text())
        # Exact success predicate used by safety for an otherwise clean barrier.
        permits = (diagnostic["exception"] is None and
                   diagnostic["certificate"]["state"] in ("complete-echild", "not-started"))
        reply = b"GO!" if permits else b"ACK"
        replies.append(reply)
        return reply
    monkeypatch.setattr(module, "_wire_driver", lambda *args: (certificate, original_error))
    monkeypatch.setattr(module, "_event", event)
    monkeypatch.setattr(module, "_wait_exact", lambda *args: None)
    if outcome == "unexpected-complete":
        with pytest.raises(AssertionError, match="not-started"):
            _wire_caller(config, object(), 123)
        assert not replies
        assert not (tmp_path / "core-result.json").exists()
        return
    _wire_caller(config, object(), 123)
    record = json.loads((tmp_path / "core-result.json").read_text())
    assert replies == [b"ACK"]
    assert record["certificate"] == core.certificate_record(certificate)
    assert record["exception"] is not None
    if original_error is not None:
        assert record["exception"]["message"] == str(original_error)
    else:
        assert "withheld START" in record["exception"]["message"]


@pytest.mark.parametrize("topology", ["same-group", "setpgid", "setsid",
                                      "double-fork", "nested-subreaper"])
def test_adoption_topologies(tmp_path, topology):
    o = run_os_case(tmp_path, "cleanup-engine", options={"topology": topology})
    assert_tree_reaped(o)
    assert o["certificate"]["state"] == "complete-echild"
    assert o["certificate"]["reaped_count"] >= 2


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("retain_pipes", [True, False])
def test_leader_exit_is_not_tree_exit(tmp_path, exit_code, retain_pipes):
    o = run_os_case(tmp_path, "leader-exit", options={"exit_code": exit_code,
                    "topology": "double-fork", "retain_pipes": retain_pipes})
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == exit_code
    assert "timed out" not in o["result"]["stderr"]
    assert o["certificate"]["state"] == "complete-echild"


@pytest.mark.parametrize("case", ["fork-during-term", "bounded-fork-storm",
                                  "omitted-discovery", "adoption-churn"])
def test_cleanup_churn_and_omissions(tmp_path, case):
    o = run_os_case(tmp_path, case)
    assert_tree_reaped(o)
    assert o["certificate"]["state"] == "complete-echild"
    assert o["cleanup_elapsed_seconds"] <= 2.0
    assert o["kill_first_at"] - o["term_first_at"] <= 0.2 + 0.05
    if case in ("fork-during-term", "bounded-fork-storm"):
        proof = json.loads((Path(o["receipt_path"]).parent / "term-fork.json").read_text())
        assert proof["registered"] is True
        assert proof["at"] >= o["certificate"]["cleanup_started"]
    if case == "bounded-fork-storm":
        assert len(o["job_members"]) == 9  # leader plus at most eight generations
        assert o["certificate"]["reaped_count"] == 9


@pytest.mark.parametrize("fault", ["acquire-echild", "acquire-esrch", "signal-eperm",
                                   "pidfd-emfile", "children-unreadable",
                                   "wait-error", "cleanup-expiry"])
def test_cleanup_invariant_fault_is_not_complete(tmp_path, fault):
    o = run_os_case(tmp_path, "cleanup-fault", options={"fault": fault})
    assert o["exception"] is not None
    assert o["certificate"] is None or o["certificate"]["state"] == "unknown"
    assert o["observer_started"] is False
    assert o["safety_echild"] and o["safety_reaped"] and o["handles_closed"]


def test_cleanup_deadline_does_not_restart():
    engine = _require_cleanup_engine()
    state = engine.WorkerState()
    engine._begin_cleanup(state, "first", 10.0)
    engine._begin_cleanup(state, "later", 11.0)
    assert (state.cleanup_started, state.cleanup_deadline, state.reason,
            state.launch_closed) == (10.0, 11.5, "first", True)


@pytest.mark.parametrize("data", [b"0", b"-1", b"1 x", b"1 1", b"\xff"])
def test_children_parser_rejects_invalid_complete_tokens(data):
    with pytest.raises(ValueError):
        _require_cleanup_engine()._parse_children(data)


def test_children_parser_cap_only_discards_partial_token():
    engine = _require_cleanup_engine()
    assert engine._parse_children(b"1 23") == [1, 23]
    prefix = b" " * (engine.CHILD_READ_BYTES - 4)
    assert engine._parse_children(prefix + b"1 23") == [1]
    with pytest.raises(ValueError):
        engine._parse_children(b"1 x")


def test_discovery_failure_preserves_authenticated_signal_work(monkeypatch):
    engine = _require_cleanup_engine()
    state = engine.WorkerState()
    fd = os.open(os.devnull, os.O_RDONLY)
    handle = engine.ChildHandle(123, fd)
    state.live_handles[123] = handle
    def denied():
        raise PermissionError(errno.EACCES, "injected children read")
    monkeypatch.setattr(engine, "_read_children", denied)
    try:
        assert engine._acquire_batch(state) == [handle]
        assert state.errors
    finally:
        engine._close_handles(state)


def test_term_grace_cannot_extend_during_a_batch(monkeypatch):
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=123)
    engine._begin_cleanup(state, "test", time.monotonic() - 0.21)
    fd = os.open(os.devnull, os.O_RDONLY)
    handle = engine.ChildHandle(123, fd)
    state.live_handles[123] = handle
    signals = []
    monkeypatch.setattr(engine, "_pidfd_signal", lambda fd, sig: signals.append(sig))
    try:
        engine._signal_handles(state, [handle], signal.SIGTERM)
        assert signals == [signal.SIGKILL]
        assert not state.term_sent
    finally:
        engine._close_handles(state)


def test_acquisition_discards_unprocessed_ids_before_reap(monkeypatch):
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=1)
    ids = list(range(1, 66))
    opened = []
    def open_fd(pid):
        fd = os.open(os.devnull, os.O_RDONLY)
        opened.append(pid)
        return fd
    monkeypatch.setattr(engine, "_read_children", lambda: ids)
    monkeypatch.setattr(engine, "_pidfd_open", open_fd)
    monkeypatch.setattr(engine, "_waitid_child", lambda fd: None)
    try:
        handles = engine._acquire_batch(state)
        assert opened == list(range(1, 65))
        assert len(handles) == 64 and ids == []
        def empty():
            assert ids == []
            raise ChildProcessError(errno.ECHILD, "real-wait seam only")
        monkeypatch.setattr(engine, "_wait_all", empty)
        assert engine._reap_batch(state) is True
    finally:
        engine._close_handles(state)


def test_reap_preserves_leader_and_retires_identity_tracking(monkeypatch):
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=123)
    fd = os.open(os.devnull, os.O_RDONLY)
    state.live_handles[123] = engine.ChildHandle(123, fd)
    state.term_sent.add(fd)
    state.authenticated_pids.add(123)
    waits = iter([(123, 7 << 8), (124, 0), (0, 0)])
    monkeypatch.setattr(engine, "_wait_all", lambda: next(waits))
    try:
        assert engine._reap_batch(state) is False
        assert state.leader_wait_status == 7 << 8
        assert (state.reaped_count, state.adopted_count) == (2, 1)
        assert not state.live_handles and not state.authenticated_pids and not state.term_sent
        with pytest.raises(OSError):
            os.fstat(fd)
    finally:
        engine._close_handles(state)


@pytest.mark.parametrize("prior_status, adopted", [(None, 0), (7 << 8, 1)])
def test_leader_number_acquisition_counts_only_new_identity(monkeypatch, prior_status, adopted):
    """A saved diagnostic number is not the original leader after its reap."""
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=123, leader_wait_status=prior_status)
    fd = os.open(os.devnull, os.O_RDONLY)
    monkeypatch.setattr(engine, "_read_children", lambda: [123])
    monkeypatch.setattr(engine, "_pidfd_open", lambda pid: fd)
    monkeypatch.setattr(engine, "_waitid_child", lambda handle: None)
    waits = iter([(123, 9 << 8), (0, 0)])
    monkeypatch.setattr(engine, "_wait_all", lambda: next(waits))
    try:
        handles = engine._acquire_batch(state)
        assert len(handles) == 1
        assert state.adopted_count == adopted
        assert state.authenticated_pids == {123}
        assert engine._acquire_batch(state) == handles
        assert state.adopted_count == adopted  # Same unreaped child, not another adoption.
        assert engine._reap_batch(state) is False
        assert state.adopted_count == adopted  # Acquisition and reap cannot double-count.
        assert state.reaped_count == 1
        assert state.leader_pid == 123
        assert state.leader_wait_status == (9 << 8 if prior_status is None else 7 << 8)
        assert not state.authenticated_pids and not state.live_handles
        with pytest.raises(OSError) as closed:
            os.fstat(fd)
        assert closed.value.errno == errno.EBADF
    finally:
        engine._close_handles(state)


@pytest.mark.parametrize("prior_status, adopted", [(None, 0), (7 << 8, 1)])
def test_leader_number_reap_without_acquisition_counts_new_identity(monkeypatch, prior_status, adopted):
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=123, leader_wait_status=prior_status)
    waits = iter([(123, 9 << 8), (0, 0)])
    monkeypatch.setattr(engine, "_wait_all", lambda: next(waits))
    assert engine._reap_batch(state) is False
    assert (state.reaped_count, state.adopted_count) == (1, adopted)
    assert state.leader_pid == 123
    assert state.leader_wait_status == (9 << 8 if prior_status is None else 7 << 8)
    assert not state.authenticated_pids and not state.live_handles


@pytest.mark.parametrize("prior_status, retain", [(None, True), (7 << 8, False)])
def test_leader_number_kill_retains_only_original_handle(monkeypatch, prior_status, retain):
    engine = _require_cleanup_engine()
    state = engine.WorkerState(leader_pid=123, leader_wait_status=prior_status)
    fd = os.open(os.devnull, os.O_RDONLY)
    handle = engine.ChildHandle(123, fd)
    state.live_handles[123] = handle
    state.term_sent.add(fd)
    state.authenticated_pids.add(123)
    # Harmless FD and worker-local send seam: no signal or child is created.
    monkeypatch.setattr(engine, "_pidfd_signal", lambda pidfd, sig: None)
    try:
        engine._signal_handles(state, [handle], signal.SIGKILL)
        if retain:
            assert state.live_handles == {123: handle}
            assert state.term_sent == {fd}
            os.fstat(fd)
        else:
            assert not state.live_handles and not state.term_sent
            with pytest.raises(OSError) as closed:
                os.fstat(fd)
            assert closed.value.errno == errno.EBADF
        assert state.authenticated_pids == {123}  # Counting pin survives only until reap.
        waits = iter([(123, 9 << 8), (0, 0)])
        monkeypatch.setattr(engine, "_wait_all", lambda: next(waits))
        assert engine._reap_batch(state) is False
        assert not state.authenticated_pids and not state.live_handles and not state.term_sent
        assert state.leader_pid == 123
        assert state.leader_wait_status == (9 << 8 if prior_status is None else 7 << 8)
    finally:
        engine._close_handles(state)


def test_available_pidfd_interface_passes_self_only_preflight():
    # This specifically fails at the old Python-only missing-binding guard.
    # No child is needed to qualify self identity, signal-zero and ECHILD.
    _require_fixture_bindings()
    _fixture_preflight()


class _CFunction:
    def __init__(self, call):
        self.call = call

    def __call__(self, *args):
        return self.call(*args)


@pytest.mark.parametrize("native", [True, False])
def test_pidfd_binding_selection_and_cloexec(native):
    calls = []

    def opened(pid, flags):
        calls.append((pid, flags))
        fd = os.open(os.devnull, os.O_RDONLY)
        os.set_inheritable(fd, True)
        return fd

    def signalled(*args):
        calls.append(args)
        return 0

    libc = SimpleNamespace() if native else SimpleNamespace(
        pidfd_open=_CFunction(opened), pidfd_send_signal=_CFunction(signalled))
    open_fd, send_signal = _select_pidfd_bindings(opened if native else None,
                                                signalled if native else None, libc)
    fd = open_fd(123)
    try:
        assert os.get_inheritable(fd) is False
        send_signal(fd, 0)
        assert calls == [(123, 0), (fd, 0, None, 0)]
        if not native:
            assert libc.pidfd_open.argtypes == (ctypes.c_int, ctypes.c_uint)
            assert libc.pidfd_open.restype is ctypes.c_int
            assert libc.pidfd_send_signal.argtypes == (ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_uint)
            assert libc.pidfd_send_signal.restype is ctypes.c_int
    finally:
        os.close(fd)


@pytest.mark.parametrize("native", [True, False])
@pytest.mark.parametrize("operation", ["open", "signal"])
def test_pidfd_binding_propagates_errno(native, operation):
    def denied(*args):
        if native:
            raise OSError(errno.EPERM, "native denial")
        ctypes.set_errno(errno.EPERM)
        return -1

    libc = SimpleNamespace(pidfd_open=_CFunction(denied), pidfd_send_signal=_CFunction(denied))
    open_fd, send_signal = _select_pidfd_bindings(denied if native else None,
                                                denied if native else None, libc)
    with pytest.raises(OSError) as caught:
        open_fd(os.getpid()) if operation == "open" else send_signal(123, 0)
    assert caught.value.errno == errno.EPERM


def test_missing_libc_symbol_rejects_selection():
    with pytest.raises(AttributeError):
        _select_pidfd_bindings(None, None, SimpleNamespace())


def test_failed_self_preflight_closes_its_pidfd(monkeypatch):
    opened = []
    real_open = _pidfd_open

    def tracked(pid):
        fd = real_open(pid)
        opened.append(fd)
        return fd

    def denied(fd, sig):
        assert sig == 0
        raise OSError(errno.EPERM, "self signal denied")

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_pidfd_open", tracked)
    monkeypatch.setattr(module, "_pidfd_signal", denied)
    with pytest.raises(OSError) as caught:
        _fixture_preflight()
    assert caught.value.errno == errno.EPERM
    assert len(opened) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(opened[0])
    assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("error", [OSError(errno.EPERM, "denied self pidfd"),
                                  NotImplementedError("missing wait interface")])
def test_preflight_error_blocks_before_fixture_spawn(tmp_path, monkeypatch, error):
    def forbidden(*args, **kwargs):
        pytest.fail("failed preflight reached a process or subreaper operation")

    def rejected():
        raise error

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "_fixture_preflight", rejected)
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=forbidden))
    monkeypatch.setattr(module, "_subreaper", forbidden)
    with pytest.raises(type(error)) as caught:
        run_os_case(tmp_path, "safety-control", boundary="legacy")
    assert caught.value is error
    assert not (tmp_path / "os-case").exists()


def _fake_safety_driver(tmp_path, monkeypatch, failure):
    """Only local module boundaries are replaced; this never creates a child."""
    module = sys.modules[__name__]
    real_os = os
    state = SimpleNamespace(opened=[], closed=[], launches=0, config=None)
    process = SimpleNamespace(pid=123456, returncode=None)
    state.process = process
    real_write = _write_exclusive

    def write(path, record):
        if failure == "driver-write" and path.name == "driver-receipt.json":
            raise OSError(errno.EIO, "fake receipt write failure")
        real_write(path, record)

    def launch(*args, **kwargs):
        state.launches += 1
        directory = tmp_path / "os-case"
        state.config = json.loads((directory / "config.json").read_text())
        record = {"observation": {"safety_echild": failure != "invalid-teardown",
                                  "handles_closed": True, "setup_error": None},
                  "first_safety_kill_at": 2.0, "receipt_written_at": 1.0}
        path = directory / "teardown-receipt.json"
        if failure == "malformed-receipt":
            path.write_text("{")
        elif failure != "missing-receipt":
            real_write(path, record)
        return process

    def opened(pid):
        assert pid == process.pid
        if failure == "pidfd-open":
            raise OSError(errno.EMFILE, "fake pidfd allocation failure")
        fd = real_os.open(real_os.devnull, real_os.O_RDONLY)
        state.opened.append(fd)
        return fd

    def waitid(*args):
        if failure == "waitid":
            raise OSError(errno.ECHILD, "fake lost wait ownership")

    def waited(timeout):
        assert timeout == 10
        if failure == "wait-timeout":
            raise subprocess.TimeoutExpired("fake safety", timeout)
        if failure == "wait-error":
            raise OSError(errno.EIO, "fake exact wait failure")
        if failure != "wait-not-reaped":
            process.returncode = 0
        return 0

    def closed(fd):
        state.closed.append(fd)
        real_os.close(fd)

    class LocalOS:
        def __getattr__(self, name):
            return getattr(real_os, name)

    local_os = LocalOS()
    local_os.waitid, local_os.close = waitid, closed
    process.wait = waited
    monkeypatch.setattr(module, "os", local_os)
    monkeypatch.setattr(module, "subprocess", SimpleNamespace(Popen=launch, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(module, "_pidfd_open", opened)
    monkeypatch.setattr(module, "_fixture_preflight", lambda: None)
    monkeypatch.setattr(module, "_subreaper", lambda: 0)
    monkeypatch.setattr(module, "_teardown_unknown", False)
    monkeypatch.setattr(module, "_write_exclusive", write)
    return state


@pytest.mark.parametrize("failure", ["pidfd-open", "waitid", "wait-timeout", "wait-error",
                                     "wait-not-reaped", "missing-receipt", "malformed-receipt",
                                     "invalid-teardown", "driver-write"])
def test_postlaunch_failure_aborts_and_latches_unknown(tmp_path, monkeypatch, failure):
    state = _fake_safety_driver(tmp_path, monkeypatch, failure)
    try:
        with pytest.raises(pytest.exit.Exception) as stopped:
            run_os_case(tmp_path, "safety-control", boundary="legacy")
        assert _teardown_unknown is True
        assert stopped.value.returncode == 2
        assert stopped.value.safety_process is state.process
        assert stopped.value.safety_identity == {
            "pid": state.process.pid, "pidfd_acquired": failure != "pidfd-open",
            "waitid_validated": failure not in ("pidfd-open", "waitid")}
        assert state.closed == state.opened
        with pytest.raises(pytest.exit.Exception):
            run_os_case(tmp_path / "must-not-launch", "safety-control", boundary="legacy")
        assert state.launches == 1
    finally:
        # Keep even a failing RED unit run from leaking its harmless /dev/null FD.
        for fd in state.opened:
            if fd not in state.closed:
                os.close(fd)


def test_bootstrap_waitid_failure_closes_provisional_handle(monkeypatch):
    module = sys.modules[__name__]
    real_os = os
    owner = _SafetyOwner.__new__(_SafetyOwner)
    owner.owned, owner.spawned = {}, {}
    fd = os.open(os.devnull, os.O_RDONLY)
    closed = []

    def rejected(*args):
        raise OSError(errno.ECHILD, "fake provisional waitability failure")

    def close(handle):
        closed.append(handle)
        real_os.close(handle)

    # A local proxy makes the real method's fork return the parent branch only.
    monkeypatch.setattr(module, "os", SimpleNamespace(fork=lambda: 123457,
                        waitid=rejected, close=close, WEXITED=os.WEXITED,
                        WNOHANG=os.WNOHANG, WNOWAIT=os.WNOWAIT))
    monkeypatch.setattr(module, "_fixture_preflight", lambda: None)
    monkeypatch.setattr(module, "_pidfd_open", lambda pid: fd)
    try:
        with pytest.raises(OSError) as caught:
            owner.spawn("caller", "--fixture-caller")
        assert caught.value.errno == errno.ECHILD
        assert closed == [fd]
        assert owner.owned == {} and owner.spawned == {}
    finally:
        if not closed:
            os.close(fd)


def test_actor_receipt_argv_is_explicitly_a_template(tmp_path, monkeypatch):
    state = _fake_safety_driver(tmp_path, monkeypatch, None)
    o = run_os_case(tmp_path, "safety-control", boundary="legacy")
    assert o["safety_reaped"] is True
    assert _teardown_unknown is False
    commands = state.config["commands"]
    assert "actors" not in commands
    assert commands["actor_templates"]["escaped-d"][-1] == "-1"
    assert state.opened == state.closed


def test_safety_owner_is_local_and_reaps_its_fixture(tmp_path):
    o = run_os_case(tmp_path, "safety-control", boundary="legacy")
    assert o["setup_error"] is None
    assert o["ready"] is True
    assert o["job_members"][0]["alive_at_barrier"] is True
    assert o["safety_echild"] is True
    assert o["safety_reaped"] is True
    assert o["handles_closed"] is True
    assert all(m["alive_at_barrier"] for m in o["unrelated_members"])
