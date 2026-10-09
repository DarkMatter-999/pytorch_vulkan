import json
import ctypes
import inspect
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

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
            assert command["timeout_seconds"] == (600 if gate["name"] == "ctest" else 300)
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
    class Completed:
        returncode = 7

        def communicate(self, timeout):
            return "out\n", "err\n"

    monkeypatch.setattr(qualification.subprocess, "Popen", lambda *args, **kwargs: Completed())
    monkeypatch.setattr(qualification, "_finish_owned_group", lambda process: ("out\n", "err\n"))

    result = qualification.run_command(["example", "--check"], Path("/tmp"))

    assert result["exit_code"] == 7
    assert result["stdout"] == "out\n"
    assert result["stderr"] == "err\n"


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
    class TimedOut:
        returncode = -signal.SIGKILL

        def communicate(self, timeout):
            raise qualification.subprocess.TimeoutExpired(
                ["example"], timeout, output=b"partial-out", stderr=b"partial-err"
            )

        def poll(self):
            return None

    monkeypatch.setattr(qualification.subprocess, "Popen", lambda *args, **kwargs: TimedOut())
    monkeypatch.setattr(
        qualification, "_finish_owned_group", lambda process: (b"partial-out", b"partial-err")
    )

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


def test_command_records_effective_environment_deltas(monkeypatch, tmp_path):
    monkeypatch.delenv("VK_INSTANCE_LAYERS", raising=False)
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
    assert environment["VULKAN_DEVICE"] == "vk:0"
    assert environment["PYTORCH_VULKAN_DEVICE"] == "vk:0"


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
