import json
import ctypes
import inspect
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from test_qualification_process_supervisor import run_os_case, assert_tree_reaped
from tools import run_vulkan_qualification as qualification
from tools.validate_vulkan_capabilities import load_manifest
from vulkan_conformance import (
    PROMOTED_SCALAR_OUT_SCHEMAS,
    SCALAR_OUT_CONTRACT_MATRIX,
    SCALAR_OUT_OUT_CASES,
    SCALAR_OUT_FUNCTIONAL_CASES,
    REQUIRED_SCALAR_OUT_REJECTION_BOUNDARIES,
    MANIFEST_CASE_NAMES,
)


SCALAR_OUT_QUALIFICATION_CASES = {
    schema: {
        "manifest_case": contract.case_name,
        "positive_parity": (
            "tests/python/test_vulkan_add.py::"
            "test_scalar_contract_matrix_cases_are_named_and_vulkan_resident"
            if contract.output_mode == "functional"
            else "tests/python/test_vulkan_out.py::"
            "test_scalar_out_contract_matrix_cases_return_named_output_and_do_vulkan_work"
        ),
        "negative_rejection": (
            "tests/python/test_vulkan_add.py::"
            "test_scalar_contract_matrix_rejects_invalid_scalar_without_work"
            if contract.output_mode == "functional"
            else (
                "tests/python/test_vulkan_out.py::"
                "test_scalar_out_contract_rejects_invalid_scalar_without_work"
                if contract.operand_order != "tensor-tensor"
                else "tests/python/test_vulkan_out.py::"
                "test_tensor_tensor_out_broadcast_rejected_before_work"
            )
        ),
        "counter_no_fallback": (
            "tests/python/test_vulkan_add.py::"
            "test_scalar_contract_matrix_cases_are_named_and_vulkan_resident"
            if contract.output_mode == "functional"
            else "tests/python/test_vulkan_out.py::"
            "test_scalar_out_contract_matrix_cases_return_named_output_and_do_vulkan_work"
        ),
        "alias": (
            "tests/python/test_vulkan_add.py::"
            "test_pointwise_python_scalar_matches_cpu_and_preserves_inputs"
            if contract.output_mode == "functional"
            else (
                "tests/python/test_vulkan_out.py::test_scalar_out_contract_exact_aliases"
                if contract.operand_order != "tensor-tensor"
                else (
                    "tests/python/test_vulkan_out.py::test_tensor_tensor_out_contract_exact_aliases",
                    "tests/python/test_vulkan_out.py::"
                    "test_tensor_tensor_out_contract_second_operand_exact_aliases",
                )
            )
        ),
    }
    for schema, contract in SCALAR_OUT_CONTRACT_MATRIX.items()
}


@pytest.fixture(autouse=True)
def explicit_qualification_wait_policy():
    from tools.qualification_process_supervisor import caller_wait_scope
    with caller_wait_scope(exclusive_wait=True):
        yield


@pytest.mark.parametrize("retain_pipes", [True, False])
def test_nested_timeout_barrier(tmp_path, retain_pipes):
    o = run_os_case(tmp_path, "nested-timeout", boundary="legacy",
                    options={"retain_pipes": retain_pipes})
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == 124
    assert o["observer_started"] is True


@pytest.mark.parametrize("exit_code", [0, 7])
@pytest.mark.parametrize("topology", ["setsid", "double-fork"])
def test_escaped_leader_exit_barrier(tmp_path, exit_code, topology):
    o = run_os_case(tmp_path, "leader-exit", boundary="legacy",
                    options={"exit_code": exit_code, "topology": topology,
                             "retain_pipes": False})
    assert_tree_reaped(o)
    assert o["result"]["exit_code"] == exit_code
    assert "timed out" not in o["result"]["stderr"]


@pytest.fixture
def owned_process_tree(tmp_path):
    # Adopt orphaned fixture grandchildren so even a failing RED run leaves no
    # zombies behind on hosts whose PID 1 does not promptly reap them.
    libc = ctypes.CDLL(None, use_errno=True)
    previous = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(previous), 0, 0, 0) == 0
    assert libc.prctl(36, 1, 0, 0, 0) == 0
    record = tmp_path / "owned-pids.json"
    try:
        yield record
    finally:
        try:
            if record.exists():
                for pid in json.loads(record.read_text())["pids"]:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                deadline = time.monotonic() + 2
                for pid in json.loads(record.read_text())["pids"]:
                    while time.monotonic() < deadline:
                        try:
                            reaped, _ = os.waitpid(pid, os.WNOHANG)
                        except ChildProcessError:
                            break
                        if reaped:
                            break
                        time.sleep(0.01)
                    assert not Path(f"/proc/{pid}").exists(), f"fixture process survived: {pid}"
        finally:
            assert libc.prctl(36, previous.value, 0, 0, 0) == 0


def _tree_command(record, *, ignore_term, parent_exits=False, exit_code=0):
    grandchild = (
        "import os, signal, time; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + "print('grandchild-ready', flush=True); time.sleep(30)"
    )
    parent = (
        "import json, os, pathlib, subprocess, sys, time; "
        f"child = subprocess.Popen([sys.executable, '-u', '-c', {grandchild!r}]); "
        f"pathlib.Path({str(record)!r}).write_text(json.dumps("
        "{'pids': [os.getpid(), child.pid], 'group': os.getpgrp()})); "
        "print('parent-partial', flush=True); "
        "print('stderr-partial', file=sys.stderr, flush=True); "
        + (f"time.sleep(0.1); sys.exit({exit_code})" if parent_exits else "time.sleep(30)")
    )
    return [sys.executable, "-u", "-c", parent]


def test_cleanup_handles_non_utf8_process_name(owned_process_tree):
    # Keep an unrelated process alive during the runner's /proc scan. Linux comm
    # may contain non-UTF-8 bytes, whitespace, and the stat delimiter itself.
    name = b"odd) \xff (\tname)"
    named = subprocess.Popen(
        [sys.executable, "-u", "-c",
         "import ctypes, time; "
         f"assert ctypes.CDLL(None).prctl(15, {name!r}, 0, 0, 0) == 0; "
         "print('named-ready', flush=True); time.sleep(30)"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True,
    )
    try:
        with pytest.raises(subprocess.TimeoutExpired) as ready:
            named.communicate(timeout=0.2)
        assert b"named-ready" in ready.value.stdout
        stat = Path(f"/proc/{named.pid}/stat").read_bytes()
        assert name in stat
        fields = stat.rsplit(b") ", 1)[1].split()
        assert int(fields[2]) == named.pid
        result = qualification.run_command(
            _tree_command(owned_process_tree, ignore_term=True),
            owned_process_tree.parent, timeout_seconds=0.4,
        )
        assert result["exit_code"] == 124
        assert "parent-partial" in result["stdout"]
        assert "stderr-partial" in result["stderr"]
        assert result["elapsed_seconds"] < 3
        assert named.poll() is None
        child, grandchild = json.loads(owned_process_tree.read_text())["pids"]
        assert not Path(f"/proc/{child}").exists()
        state = Path(f"/proc/{grandchild}/stat")
        assert not state.exists() or state.read_bytes().rsplit(b") ", 1)[1].split()[0] == b"Z"
    finally:
        named.kill()
        named.communicate(timeout=2)
        assert not Path(f"/proc/{named.pid}").exists()


@pytest.mark.parametrize("cleanup_error", [AssertionError, OSError])
def test_owned_fixture_restores_subreaper_after_cleanup_error(tmp_path, monkeypatch, cleanup_error):
    libc = ctypes.CDLL(None, use_errno=True)
    original = ctypes.c_int()
    assert libc.prctl(37, ctypes.byref(original), 0, 0, 0) == 0
    child = None
    fixture = None
    try:
        # Force a different scoped state so a skipped restoration cannot pass.
        assert libc.prctl(36, 0, 0, 0, 0) == 0
        fixture = owned_process_tree.__wrapped__(tmp_path)
        record = next(fixture)
        child = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
        )
        record.write_text(json.dumps({"pids": [child.pid], "group": child.pid}))
        original_exists = Path.exists

        def failing_probe(path):
            if path == Path(f"/proc/{child.pid}"):
                raise cleanup_error("injected cleanup check failure")
            return original_exists(path)

        with monkeypatch.context() as scoped:
            scoped.setattr(Path, "exists", failing_probe)
            with pytest.raises(cleanup_error, match="injected cleanup check failure"):
                fixture.close()
        # The fault is after bounded kill/wait, so it must leave no owned PID.
        assert not Path(f"/proc/{child.pid}").exists()
        restored = ctypes.c_int()
        assert libc.prctl(37, ctypes.byref(restored), 0, 0, 0) == 0
        assert restored.value == 0, "fixture did not restore its original subreaper state"
    finally:
        if fixture is not None:
            fixture.close()
        if child is not None:
            if child.poll() is None:
                child.kill()
            child.wait(timeout=2)
        assert libc.prctl(36, original.value, 0, 0, 0) == 0


@pytest.mark.parametrize("ignore_term", [False, True])
def test_timeout_reaps_direct_child_and_terminates_pipe_holding_grandchild(
    monkeypatch, owned_process_tree, ignore_term
):
    # Removing owned-group cleanup must leave a live grandchild and fail this.
    command = _tree_command(owned_process_tree, ignore_term=ignore_term)
    if "timeout_seconds" not in inspect.signature(qualification.run_command).parameters:
        # Exercise the old implementation's lifetime bug at a tiny bound too.
        original_run = subprocess.run

        def short_run(*args, **kwargs):
            kwargs["timeout"] = 0.4
            return original_run(*args, **kwargs)

        monkeypatch.setattr(qualification.subprocess, "run", short_run)
        result = qualification.run_command(command, owned_process_tree.parent)
    else:
        result = qualification.run_command(
            command, owned_process_tree.parent, timeout_seconds=0.4
        )
    assert result["exit_code"] == 124
    assert "parent-partial" in result["stdout"]
    assert "grandchild-ready" in result["stdout"]
    assert "stderr-partial" in result["stderr"]
    pids = json.loads(owned_process_tree.read_text())["pids"]
    assert not Path(f"/proc/{pids[0]}").exists(), "direct child was not reaped"
    state = Path(f"/proc/{pids[1]}/stat")
    assert not state.exists() or state.read_text().split(") ")[1].split()[0] == "Z", (
        "owned grandchild is still running after run_command returned"
    )
    assert result["timeout_seconds"] == 0.4
    assert 0.4 <= result["elapsed_seconds"] < 3
    assert json.loads(owned_process_tree.read_text())["group"] == pids[0]
    assert pids[0] != os.getpgrp()
    assert "after 0.4 seconds" in result["stderr"]
    assert qualification.classify_result(result, device_dependent=True) == "fail"


@pytest.mark.parametrize("exit_code", [0, 7])
def test_completed_parent_cleans_up_pipe_holding_descendant(owned_process_tree, exit_code):
    result = qualification.run_command(
        _tree_command(owned_process_tree, ignore_term=True, parent_exits=True, exit_code=exit_code),
        owned_process_tree.parent,
        timeout_seconds=0.4,
    )
    assert result["exit_code"] == exit_code
    assert "parent-partial" in result["stdout"]
    assert "timed out" not in result["stderr"]
    child, grandchild = json.loads(owned_process_tree.read_text())["pids"]
    assert not Path(f"/proc/{child}").exists()
    state = Path(f"/proc/{grandchild}/stat")
    assert not state.exists() or state.read_text().split(") ")[1].split()[0] == "Z"
    assert result["elapsed_seconds"] < 3


def test_next_command_observes_cleanup_and_unrelated_process_survives(owned_process_tree):
    unrelated = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True
    )
    try:
        observer = (
            "import json, pathlib; "
            f"pids = json.loads(pathlib.Path({str(owned_process_tree)!r}).read_text())['pids']; "
            "assert not pathlib.Path(f'/proc/{pids[0]}').exists(); "
            "stat = pathlib.Path(f'/proc/{pids[1]}/stat'); "
            "assert not stat.exists() or stat.read_text().split(') ')[1].split()[0] == 'Z'; "
            "print('cleanup-before-next-command')"
        )
        gate = qualification._gate(
            "lifecycle-fixture",
            [_tree_command(owned_process_tree, ignore_term=True),
             [sys.executable, "-c", observer]],
            timeout_seconds=0.4,
        )
        assert gate["status"] == "fail"
        assert gate["commands"][0]["exit_code"] == 124
        assert gate["commands"][1]["exit_code"] == 0
        assert gate["commands"][1]["stdout"] == "cleanup-before-next-command\n"
        assert unrelated.poll() is None
    finally:
        unrelated.kill()
        unrelated.wait(timeout=2)


def test_only_complete_ctest_retains_600_second_budget(monkeypatch, tmp_path):
    # A global budget raise or failure to forward the explicit gate budget fails.
    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        return {"exit_code": 0, "stdout": "", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(qualification, "device_available", lambda *args: {
        "status": "available", "reason": "fixture device"
    })
    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)
    retained = json.loads((tmp_path / "report.json").read_text())
    assert retained == summary
    for gate in retained["gates"]:
        for command in gate["commands"]:
            assert command["timeout_seconds"] == (
                600 if gate["name"] in {"ctest", "python_suite"} else 300)
    assert retained["repository"]["timeout_seconds"] == 300


@pytest.mark.parametrize("exit_code", [0, 7])
def test_real_command_retains_output_exit_and_budget(tmp_path, exit_code):
    result = qualification.run_command(
        [sys.executable, "-c", "import sys; print('out'); "
         f"print('err', file=sys.stderr); sys.exit({exit_code})"],
        tmp_path, timeout_seconds=2,
    )
    assert result["exit_code"] == exit_code
    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"
    assert result["timeout_seconds"] == 2
    assert 0 <= result["elapsed_seconds"] < 2


def test_scalar_out_qualification_inventory_covers_every_promoted_schema():
    assert set(SCALAR_OUT_QUALIFICATION_CASES) == PROMOTED_SCALAR_OUT_SCHEMAS
    assert len(SCALAR_OUT_QUALIFICATION_CASES) == 11
    assert len(SCALAR_OUT_FUNCTIONAL_CASES) == 4
    assert len(SCALAR_OUT_OUT_CASES) == 7
    manifest = load_manifest(
        Path(__file__).resolve().parents[2] / "docs/vulkan_capabilities.json"
    )
    manifest_case_schemas = {
        case["name"]: entry["schema"]
        for entry in manifest["entries"]
        for case in entry["test_cases"]
    }

    for schema, coverage in SCALAR_OUT_QUALIFICATION_CASES.items():
        contract = SCALAR_OUT_CONTRACT_MATRIX[schema]
        assert coverage["manifest_case"] in MANIFEST_CASE_NAMES
        assert manifest_case_schemas[coverage["manifest_case"]] == schema
        identifiers = [coverage["positive_parity"], coverage["negative_rejection"]]
        identifiers.append(coverage["counter_no_fallback"])
        identifiers.extend(
            coverage["alias"] if isinstance(coverage["alias"], tuple) else (coverage["alias"],)
        )
        for identifier in identifiers:
            relative_path, function = identifier.split("::", 1)
            function = function.split("[", 1)[0]
            source = (Path(__file__).resolve().parents[2] / relative_path).read_text()
            assert f"def {function}(" in source, identifier
        assert REQUIRED_SCALAR_OUT_REJECTION_BOUNDARIES <= contract.rejection_boundaries


def test_run_command_preserves_nonzero_exit_code(monkeypatch):
    monkeypatch.setattr(qualification, "run_command_core", lambda *a, **k: (
        {"exit_code": 7, "stdout": "out\n", "stderr": "err\n",
         "timeout_seconds": 300, "elapsed_seconds": 0.01}, None))
    monkeypatch.setattr(qualification, "require_complete", lambda result, certificate: None)

    result = qualification.run_command(["example", "--check"], Path("/tmp"))

    assert result["exit_code"] == 7
    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"


def test_wrapper_validates_certificate_and_returns_the_public_dict(monkeypatch):
    seen = []
    result = {"exit_code": 0, "stdout": "o", "stderr": "", "timeout_seconds": 5, "elapsed_seconds": 0.1}
    monkeypatch.setattr(qualification, "run_command_core", lambda *a, **k: (result, "certificate"))
    monkeypatch.setattr(qualification, "require_complete", lambda r, c: seen.append((r, c)))
    actual = qualification.run_command(["example"], Path("/tmp"), timeout_seconds=5)
    assert actual is result and seen == [(result, "certificate")]
    assert set(actual) == {"exit_code", "stdout", "stderr", "timeout_seconds", "elapsed_seconds"}


def test_unknown_cleanup_aborts_gate_comprehension(monkeypatch):
    from tools.qualification_process_supervisor import ContainmentError
    calls = []

    def failing(command, *args, **kwargs):
        calls.append(command)
        raise ContainmentError("unknown", partial_result={"exit_code": 127,
            "stdout": "partial", "stderr": "unknown", "timeout_seconds": 300,
            "elapsed_seconds": 0.1}, invocation_id="a" * 32, certificate=None)

    monkeypatch.setattr(qualification, "run_command", failing)
    with pytest.raises(ContainmentError):
        qualification._gate("fixture", [["first"], ["must-not-launch"]])
    assert calls == [["first"]]


def test_pending_cli_cancellation_wins_over_a_completed_wrapper_result(monkeypatch):
    from tools.qualification_process_supervisor import ContainmentError, cli_cancellation_scope
    result = {"exit_code": 0, "stdout": "", "stderr": "", "timeout_seconds": 5, "elapsed_seconds": 0.1}
    monkeypatch.setattr(qualification, "run_command_core", lambda *a, **k: (result, None))
    monkeypatch.setattr(qualification, "require_complete", lambda r, c: None)
    with cli_cancellation_scope() as cancellation:
        cancellation.request("SIGINT")
        with pytest.raises(ContainmentError):
            qualification.run_command(["example"], Path("/tmp"))


def test_profile_reentry_returns_status_and_cancellation_terminates_the_child():
    import threading
    from tools.qualification_process_supervisor import ContainmentError, cli_cancellation_scope
    environment = dict(os.environ)
    assert qualification._profile_reentry([sys.executable, "-c", "import sys; sys.exit(3)"], environment) == 3
    saved = signal.getsignal(signal.SIGTERM)
    timer = threading.Timer(0.3, lambda: signal.raise_signal(signal.SIGTERM))
    with cli_cancellation_scope():
        timer.start()
        with pytest.raises(ContainmentError):
            qualification._profile_reentry(
                [sys.executable, "-c", "import time; time.sleep(30)"], environment)
    timer.join()
    assert signal.getsignal(signal.SIGTERM) == saved


def test_profile_reentry_pidfd_failure_keeps_the_unreaped_child_handle(monkeypatch):
    import errno
    from tools.qualification_process_supervisor import ContainmentError
    monkeypatch.setattr(qualification, "_pidfd_open",
                        lambda pid: (_ for _ in ()).throw(OSError(errno.EMFILE, "injected pidfd")))
    with pytest.raises(ContainmentError) as raised:
        qualification._profile_reentry([sys.executable, "-c", "import time; time.sleep(30)"], dict(os.environ))
    authority = raised.value.authority
    assert authority is not None and not authority.reaped  # Never raw-PID signalled by product code.
    os.kill(authority.process.pid, signal.SIGKILL)  # Test-owned fixture child cleanup.
    assert authority.observe_until(time.monotonic() + 2) is True
    authority.close()


_SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]
_TERM_IGNORING = [sys.executable, "-c",
                  "import signal, time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)"]


def _kill_fixture_child(authority):
    os.kill(authority.process.pid, signal.SIGKILL)  # Test-owned fixture child cleanup.
    assert authority.observe_until(time.monotonic() + 2) is True
    authority.close()


def test_profile_reentry_never_signals_after_failed_authentication(monkeypatch):
    from tools.qualification_process_supervisor import ContainmentError
    signalled = []
    monkeypatch.setattr(qualification, "_pidfd_signal", lambda fd, sig: signalled.append(sig))
    monkeypatch.setattr(qualification, "_waitid_child",
                        lambda fd: (_ for _ in ()).throw(RuntimeError("injected waitability failure")))
    with pytest.raises(ContainmentError) as raised:
        qualification._profile_reentry(_SLEEPER, dict(os.environ))
    assert signalled == []  # An unauthenticated pidfd is never a signalling identity.
    authority = raised.value.authority  # The exact launch object is retained, uncertain.
    assert authority is not None and not authority.reaped
    assert isinstance(raised.value.original_error, RuntimeError)
    _kill_fixture_child(authority)


def test_profile_reentry_wait_failure_after_authentication_terminates_via_pidfd(monkeypatch):
    from tools.qualification_process_supervisor import ContainmentError
    signalled = []
    real_signal, real_waitpid = qualification._pidfd_signal, os.waitpid
    monkeypatch.setattr(qualification, "_pidfd_signal",
                        lambda fd, sig: (signalled.append(sig), real_signal(fd, sig))[1])
    state = {"failed": False}

    def waitpid(pid, flags):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("injected exact-child wait failure")
        return real_waitpid(pid, flags)

    monkeypatch.setattr(qualification.os, "waitpid", waitpid)
    with pytest.raises(ContainmentError) as raised:
        qualification._profile_reentry(_SLEEPER, dict(os.environ))
    assert signalled == [signal.SIGTERM]
    assert raised.value.authority is None  # Reaped, so nothing live remains.


def test_profile_reentry_error_unwind_keeps_the_cancellation_bound(monkeypatch):
    from tools.qualification_process_supervisor import ContainmentError, cli_cancellation_scope
    real_waitpid = os.waitpid
    started = time.monotonic()
    state = {"failed": False}

    def waitpid(pid, flags):
        if not state["failed"] and time.monotonic() >= started + 1.2:
            state["failed"] = True
            raise RuntimeError("injected wait failure during an active cancellation")
        return real_waitpid(pid, flags)

    monkeypatch.setattr(qualification.os, "waitpid", waitpid)
    import threading
    with cli_cancellation_scope() as cancellation:
        # Request after the child installed its TERM-ignoring handler; TERM then starts the bound.
        timer = threading.Timer(0.5, lambda: cancellation.request("SIGINT"))
        timer.start()
        with pytest.raises(ContainmentError) as raised:
            qualification._profile_reentry(_TERM_IGNORING, dict(os.environ))
        timer.join()
    elapsed = time.monotonic() - started
    assert elapsed < 2.9, elapsed  # A restarted bound would end near 3.4 s (1.2 s + 2.0 s).
    assert raised.value.authority is not None and not raised.value.authority.reaped
    _kill_fixture_child(raised.value.authority)


def test_classify_unavailable_is_skip_not_pass():
    result = qualification.classify_result(
        {"exit_code": 0, "stdout": "SKIPPED no suitable Vulkan device", "stderr": ""},
        device_dependent=True,
    )

    assert result == "skip"


def test_failed_output_with_unavailable_text_is_not_a_skip():
    result = qualification.classify_result(
        {"exit_code": 1, "stdout": "device unavailable\n", "stderr": ""},
        device_dependent=True,
    )

    assert result == "fail"


def test_qualification_artifact_schema_and_shape_consistency():
    artifact = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/vulkan-runtime-foundation-qualification.json")
        .read_text()
    )
    assert artifact["schema_version"] == 1
    assert artifact["status"] == "qualified"
    assert artifact["performance_claims"] == "timing evidence only; no faster-than-CPU claim"
    assert set(artifact["workloads"]) == {"small", "medium", "large", "skinny", "irregular"}
    required = {
        "shape", "dispatches", "submissions", "completions", "waits", "timing_status",
        "timing_reason", "gpu_timestamp_sample_count", "gpu_timestamp_intervals_ns",
        "host_latency_ns", "cache_metrics", "resource_bounds",
    }
    for name, workload in artifact["workloads"].items():
        assert required <= workload.keys(), name
        assert workload["dispatches"] == workload["submissions"] == workload["completions"]
        assert workload["waits"] == workload["completions"]
        assert workload["gpu_timestamp_sample_count"] == len(workload["gpu_timestamp_intervals_ns"])
        assert len(workload["shape"]) == 3
        assert workload["resource_bounds"]["pipeline_pending_destructions_after"] == 0


def test_run_command_normalizes_bytes_timeout_output(monkeypatch):
    from tools.qualification_process_supervisor import _decode_output
    monkeypatch.setattr(qualification, "run_command_core", lambda *a, **k: (
        {"exit_code": 124, "stdout": _decode_output(b"partial-out"),
         "stderr": _decode_output(b"partial-err") + "\ncommand timed out after 300 seconds\n",
         "timeout_seconds": 300, "elapsed_seconds": 0.01}, None))
    monkeypatch.setattr(qualification, "require_complete", lambda result, certificate: None)

    result = qualification.run_command(["example"], Path("/tmp"))

    assert {key: result[key] for key in ("exit_code", "stdout", "stderr")} == {
        "exit_code": 124,
        "stdout": "partial-out",
        "stderr": "partial-err\ncommand timed out after 300 seconds\n",
    }
    assert result["timeout_seconds"] == 300
    assert result["elapsed_seconds"] >= 0


def test_device_probe_validates_requested_device_and_preserves_evidence(monkeypatch):
    calls = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append((command, env))
        return {"exit_code": 77, "stdout": "", "stderr": "device unavailable"}

    monkeypatch.setattr(qualification, "run_command", fake_run)

    probe = qualification.device_available(Path("build"), "vk:3")

    assert probe["status"] == "unavailable"
    assert probe["reason"] == "device unavailable"
    assert probe["exit_code"] == 77
    assert probe["stdout"] == ""
    assert probe["stderr"] == "device unavailable"
    assert calls[0][0][-1] == "vk:3"


def test_invalid_device_is_explicitly_unavailable(monkeypatch):
    probe = qualification.device_available(Path("build"), "cuda:0")

    assert probe["status"] == "unavailable"
    assert "must match vk:<index>" in probe["reason"]
    assert probe["exit_code"] == 2


def test_gate_order_excludes_qualification_runner(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append(command)
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    names = [gate["name"] for gate in summary["gates"]]
    assert names == [
        "manifest",
        "build",
        "bootstrap_provenance",
        "ctest",
        "python_suite",
        "conformance",
        "shader_verification",
        "validation_layer",
        "stress",
        "timing",
        "benchmark",
        "environment",
        "additional_implementation",
    ]
    assert all("run_vulkan_qualification.py" not in part for command in calls for part in command)


def test_additional_device_runs_qualification_commands_with_device_environment(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append((command, env))
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda build, device, python=None: {
            "status": "available",
            "reason": f"{device} is available",
            "command": ["probe", device],
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        },
    )

    summary = qualification.run_qualification(
        tmp_path / "report.json", run_build=False, additional_device="vk:1"
    )

    gate = summary["gates"][-1]
    assert gate["status"] == "pass"
    assert gate["device_probe"]["reason"] == "vk:1 is available"
    assert any("test_vulkan_operator_capabilities.py" in part for command, _ in calls for part in command)
    additional_envs = [env for command, env in calls if env and env.get("VULKAN_DEVICE") == "vk:1"]
    assert additional_envs
    assert all(env["PYTORCH_VULKAN_DEVICE"] == "vk:1" for env in additional_envs)


def test_report_retains_device_probe_evidence(monkeypatch, tmp_path):
    probe = {
        "status": "unavailable",
        "reason": "requested device is unavailable",
        "command": ["probe", "vk:9"],
        "exit_code": 77,
        "stdout": "probe-out",
        "stderr": "probe-err",
    }
    monkeypatch.setattr(qualification, "device_available", lambda *args, **kwargs: probe)
    monkeypatch.setattr(qualification, "run_command", lambda *args, **kwargs: {
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
    })

    report = tmp_path / "report.json"
    summary = qualification.run_qualification(report, run_build=False)

    assert summary["device_probe"] == probe
    assert json.loads(report.read_text())["device_probe"]["stderr"] == "probe-err"


def test_python_and_artifact_paths_are_configurable(monkeypatch, tmp_path):
    commands = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        commands.append(command)
        return {"exit_code": 0, "stdout": "ok\n", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )
    artifacts = tmp_path / "evidence"
    qualification.run_qualification(
        tmp_path / "report.json",
        run_build=False,
        python_executable=Path("/custom/python"),
        artifacts=artifacts,
    )

    assert artifacts.is_dir()
    assert all(command[0] != str(qualification.DEFAULT_PYTHON) for command in commands)
    assert any(command[0] == "/custom/python" for command in commands)


def test_validation_ctest_registration_sets_validation_layer():
    cmake = Path(__file__).resolve().parents[2] / "CMakeLists.txt"
    text = cmake.read_text()

    assert "vulkan_supported_workload_validation" in text
    assert "VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation" in text


# Every CMake add_test registration whose COMMAND is a script invocation rather
# than a native C++ build target. The unsplit ctest gate used to run all of
# them; each must remain covered by an explicit qualification gate instead.
PYTHON_CTEST_COVERAGE = {
    "vulkan_gemm_shader_integrity": ("tools/verify_gemm_spv.py",),
    "vulkan_cat_gather_shader_integrity": ("tools/verify_cat_gather_spv.py",),
    "vulkan_rnn_sequence_shader_integrity": ("tools/verify_rnn_sequence_spv.py",),
    "vulkan_reduction_shader_integrity": ("tools/verify_reduction_indexing_spv.py",),
    "vulkan_model_shader_integrity": ("tools/verify_model_spv.py",),
    "vulkan_masked_select_shader_integrity": ("tools/verify_masked_select_spv.py",),
    "vulkan_convolution_shader_integrity": ("tools/verify_convolution_spv.py",),
    "vulkan_pooling_shader_integrity": ("tools/verify_pooling_spv.py",),
    "vulkan_interpolation_shader_integrity": ("tools/verify_interpolation_spv.py",),
    "vulkan_formatter_double_shader_integrity": ("tools/verify_formatter_double_spv.py",),
    "vulkan_pointwise_shader_integrity": ("tools/verify_pointwise_spv.py",),
    "vulkan_operator_shader_integrity": ("tools/verify_operator_spv.py",),
    "vulkan_linear_relu_backward_shader_integrity": ("tools/verify_linear_relu_backward_spv.py",),
    "vulkan_manifest_validation": ("tools/validate_vulkan_capabilities.py",),
    "vulkan_bootstrap_provenance_validation": ("tools/validate_vulkan_bootstrap_provenance.py",),
    "vulkan_workload_coverage_validation": ("tools/vulkan_workload_coverage.py",),
    "vulkan_supported_workload_validation": (
        "tools/vulkan_wrapper_pytest.py", "tests/python/test_vulkan_conformance.py",
    ),
    "vulkan_runtime_foundation_validation": ("tests/python/test_vulkan_reliability.py",),
    "vulkan_python_suite": ("tests/python",),
}

_CMAKE_ADD_TEST_NAME = re.compile(r"add_test\(\s*NAME\s+(\S+)\s+COMMAND\s")
_CMAKE_ADD_TEST_NATIVE = re.compile(r"add_test\(\s*NAME\s+(\S+)\s+COMMAND\s+([^\s${(]+)")


def test_native_ctest_selection_covers_all_native_cmake_registrations_and_zero_python():
    cmake = (Path(__file__).resolve().parents[2] / "CMakeLists.txt").read_text()
    all_names = _CMAKE_ADD_TEST_NAME.findall(cmake)
    native_names = {name for name, _ in _CMAKE_ADD_TEST_NATIVE.findall(cmake)}

    assert len(qualification.NATIVE_CTEST_TESTS) == 12
    assert native_names == set(qualification.NATIVE_CTEST_TESTS)
    assert set(all_names) - native_names == set(PYTHON_CTEST_COVERAGE)
    assert len(all_names) == 30

    # The -R selection is a single anchored regex over exactly those names.
    assert qualification.NATIVE_CTEST_REGEX == "^(%s)$" % "|".join(qualification.NATIVE_CTEST_TESTS)
    assert " " not in qualification.NATIVE_CTEST_REGEX
    assert "\n" not in qualification.NATIVE_CTEST_REGEX
    for name in qualification.NATIVE_CTEST_TESTS:
        assert re.fullmatch(qualification.NATIVE_CTEST_REGEX, name), name
    for name in all_names:
        if name not in native_names:
            assert re.fullmatch(qualification.NATIVE_CTEST_REGEX, name) is None, name
    assert "vulkan_python_suite" not in qualification.NATIVE_CTEST_REGEX


def _captured_gate_commands(monkeypatch, tmp_path, **kwargs):
    calls = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append((command, env, timeout_seconds))
        return {"exit_code": 0, "stdout": "ok\n", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )
    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False, **kwargs)
    return summary, [command for command, _, _ in calls]


def test_ctest_gate_selects_the_native_only_anchored_registration_set(monkeypatch, tmp_path):
    summary, commands = _captured_gate_commands(monkeypatch, tmp_path)

    ctest = next(gate for gate in summary["gates"] if gate["name"] == "ctest")
    assert [command["command"] for command in ctest["commands"]] == [
        qualification.native_ctest_command(qualification.ROOT / "build")]
    command = ctest["commands"][0]["command"]
    assert command[0] == "ctest"
    assert command[-2:] == ["-R", qualification.NATIVE_CTEST_REGEX]
    assert "--output-on-failure" in command
    # A substituted or unsplit selection is rejected by this contract.
    assert command != ["ctest", "--test-dir", str(qualification.ROOT / "build"),
                       "--output-on-failure"]


def test_removed_python_ctest_entries_are_covered_by_explicit_gate_commands(monkeypatch, tmp_path):
    summary, commands = _captured_gate_commands(monkeypatch, tmp_path)

    # Every non-native CMake registration maps to an executed gate argument.
    parts = [str(part) for command in commands for part in command]
    for registration, tokens in PYTHON_CTEST_COVERAGE.items():
        for token in tokens:
            assert any(token == part or token in part for part in parts), registration

    # Exactly one gate invokes the whole directory suite, and it is python_suite.
    python_suite = next(gate for gate in summary["gates"] if gate["name"] == "python_suite")
    full_suite_commands = [
        command for command in commands if "tests/python" in [str(part) for part in command]
    ]
    assert len(full_suite_commands) == 1
    assert full_suite_commands[0] == python_suite["commands"][0]["command"]
    assert not any(part == "vulkan_python_suite" for command in commands for part in command)


def test_python_suite_gate_is_wrapper_first_exactly_once_with_budget_and_no_concurrency(
    monkeypatch, tmp_path
):
    monkeypatch.delenv("PYTORCH_VULKAN_DEVICE", raising=False)
    monkeypatch.delenv("VULKAN_DEVICE", raising=False)
    summary, commands = _captured_gate_commands(
        monkeypatch, tmp_path, python_executable=Path("/custom/python"))

    names = [gate["name"] for gate in summary["gates"]]
    assert names.index("ctest") < names.index("python_suite") < names.index("conformance")
    python_suite = summary["gates"][names.index("python_suite")]

    assert python_suite["status"] == "pass"
    assert len(python_suite["commands"]) == 1
    command = python_suite["commands"][0]
    assert command["command"] == [
        "/custom/python", "tools/vulkan_wrapper_pytest.py", "-q", "-rs",
        "--durations=30", "tests/python",
    ]
    assert command["timeout_seconds"] == 600
    assert sum(1 for item in commands if item == command["command"]) == 1
    assert command["environment"]["PYTORCH_VULKAN_DEVICE"] == "vk:0"
    assert command["environment"]["VULKAN_DEVICE"] == "vk:0"
    assert not any(
        part in {"-n", "--numprocesses", "--dist"} or str(part).startswith("--numprocesses=")
        or str(part).startswith("--dist=")
        for part in command["command"]
    )


def test_python_suite_mixed_passed_and_skipped_results_pass_without_changing_conformance(
    monkeypatch, tmp_path
):
    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        output = "1 passed, 3 skipped in 0.12s\n"
        return {"exit_code": 0, "stdout": output, "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(qualification, "device_available", lambda *args, **kwargs: {
        "status": "available", "reason": "fixture device"})

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    gates = {gate["name"]: gate for gate in summary["gates"]}
    assert gates["python_suite"]["status"] == "pass"
    assert gates["conformance"]["status"] == "skip"


@pytest.mark.parametrize("exit_code", [7, 77])
def test_python_suite_nonzero_exit_fails_when_device_probe_is_available(
    monkeypatch, tmp_path, exit_code
):
    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        output = "1 passed, 3 skipped in 0.12s\n"
        return {"exit_code": exit_code, "stdout": output, "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(qualification, "device_available", lambda *args, **kwargs: {
        "status": "available", "reason": "fixture device"})

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    python_suite = next(gate for gate in summary["gates"] if gate["name"] == "python_suite")
    assert python_suite["status"] == "fail"


def test_python_suite_gate_skips_with_device_reason_when_device_is_unavailable(
    monkeypatch, tmp_path
):
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return {
            "exit_code": 0, "stdout": "ok\n", "stderr": ""
        }

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(
        qualification, "device_available", lambda *args, **kwargs: {
            "status": "unavailable", "reason": "requested device is unavailable"},
    )
    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    python_suite = next(gate for gate in summary["gates"] if gate["name"] == "python_suite")
    assert python_suite["status"] == "skip"
    assert python_suite["reason"] == "requested device is unavailable"
    assert python_suite["commands"] == []
    assert not any("tests/python" in command for command in calls)
    # The gate stays scheduled; it is never dropped from the gate inventory.
    assert "python_suite" in [gate["name"] for gate in summary["gates"]]


def test_python_suite_environment_imports_shared_helper_in_real_child(monkeypatch, tmp_path):
    calls = []

    def fake_run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append((command, env))
        return {"exit_code": 0, "stdout": "ok\n", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    monkeypatch.setattr(qualification, "run_command", fake_run)
    monkeypatch.setattr(qualification, "device_available", lambda *args, **kwargs: {
        "status": "available", "reason": "CPU fixture"})

    qualification.run_qualification(tmp_path / "report.json", run_build=False)
    _, lane_environment = next(
        (command, env) for command, env in calls if "tests/python" in command
    )
    child_environment = os.environ.copy()
    child_environment.update(lane_environment)
    child = subprocess.run(
        [sys.executable, "-c",
         "import composed_evidence_common; print(composed_evidence_common.__file__)"],
        cwd=qualification.ROOT, env=child_environment, capture_output=True, text=True,
        timeout=10,
    )
    assert child.returncode == 0, child.stderr
    assert str(Path("tests/python/composed_evidence_common.py")) in child.stdout


def test_command_records_effective_environment_deltas(monkeypatch, tmp_path):
    monkeypatch.delenv("VK_INSTANCE_LAYERS", raising=False)
    monkeypatch.setenv("VULKAN_DEVICE", "vk:0")
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )
    monkeypatch.setattr(
        qualification,
        "run_command",
        lambda *args, **kwargs: {"exit_code": 0, "stdout": "", "stderr": ""},
    )

    summary = qualification.run_qualification(tmp_path / "report.json", run_build=False)
    validation = next(gate for gate in summary["gates"] if gate["name"] == "validation_layer")
    environment = validation["commands"][0]["environment"]

    assert environment["VK_INSTANCE_LAYERS"] == "VK_LAYER_KHRONOS_validation"
    effective_environment = os.environ.copy()
    effective_environment.update(environment)
    assert effective_environment["VULKAN_DEVICE"] == "vk:0"
    assert effective_environment["PYTORCH_VULKAN_DEVICE"] == "vk:0"


def test_report_contains_machine_readable_summary(monkeypatch, tmp_path):
    monkeypatch.setattr(qualification, "run_command", lambda *args, **kwargs: {
        "exit_code": 0,
        "stdout": "ok\n",
        "stderr": "",
    })
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "available"},
    )

    report = tmp_path / "report.json"
    summary = qualification.run_qualification(report, run_build=False)

    assert json.loads(report.read_text()) == summary


def test_workload_commands_are_executed_and_failures_propagate(monkeypatch, tmp_path):
    calls = []

    def run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append(command)
        failed = any("vulkan_workload_coverage.py" in str(part) for part in command)
        return {
            "exit_code": int(failed),
            "stdout": "",
            "stderr": "invalid artifact" if failed else "",
        }

    monkeypatch.setattr(qualification, "run_command", run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda *args, **kwargs: {"status": "available", "reason": "mock device"},
    )

    report = qualification.run_qualification(tmp_path / "report.json", run_build=False)

    manifest = next(gate for gate in report["gates"] if gate["name"] == "manifest")
    assert manifest["status"] == "fail"
    assert any(
        any(
            str(part).startswith(
                "tests/python/test_vulkan_workload_conformance.py::"
            )
            for part in command
        )
        for command in calls
    )
    workload_commands = [
        [
            part
            for part in command
            if str(part).startswith(
                "tests/python/test_vulkan_workload_conformance.py::"
            )
        ]
        for command in calls
    ]
    workload_commands = [command for command in workload_commands if command]
    assert workload_commands == [
        [
            "tests/python/test_vulkan_workload_conformance.py::test_stock_sgd_executes_three_steps_in_both_reset_modes",
            "tests/python/test_vulkan_workload_conformance.py::test_parameter_hvp_executes_with_named_live_history",
            "tests/python/test_vulkan_workload_conformance.py::test_matrix_stock_sgd_executes_three_steps_in_both_reset_modes",
            "tests/python/test_vulkan_workload_conformance.py::test_matrix_parameter_hvp_executes_with_named_live_history",
        ]
    ]
    conformance_commands = [
        command
        for command in calls
        if "tests/python/test_vulkan_conformance.py" in command
    ]
    assert len(conformance_commands) == 1
    assert not any(
        str(part).startswith("tests/python/test_vulkan_workload_conformance.py::")
        for part in conformance_commands[0]
    )


def test_workload_nodes_are_limited_to_primary_vk_zero(monkeypatch, tmp_path):
    calls = []

    def run(command, cwd, env=None, *, timeout_seconds=300):
        calls.append(command)
        return {"exit_code": 0, "stdout": "ok", "stderr": ""}

    monkeypatch.setattr(qualification, "run_command", run)
    monkeypatch.setattr(
        qualification,
        "device_available",
        lambda build, device, python=None: {
            "status": "available",
            "reason": f"{device} is available",
        },
    )

    qualification.run_qualification(
        tmp_path / "report.json",
        device="vk:1",
        run_build=False,
        additional_device="vk:1",
    )

    assert not any(
        str(part).startswith("tests/python/test_vulkan_workload_conformance.py::")
        for command in calls
        for part in command
    )
