"""CPU-only tests for the MNIST qualification controller.

All command-runner outputs in this module are labelled test fixtures, never GPU
evidence. The production entry point deliberately fails closed at the unresolved
legacy nested-process ownership boundary.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess
import uuid

import pytest

ROOT = Path(__file__).resolve().parents[2]
from tools import run_mnist_training_qualification as qualification
from tools import run_vulkan_qualification as legacy


_TEST_ARTIFACTS: set[Path] = set()


@pytest.fixture(autouse=True)
def cleanup_task6_test_artifacts():
    yield
    for path in tuple(_TEST_ARTIFACTS):
        _TEST_ARTIFACTS.discard(path)
        if path.exists():
            shutil.rmtree(path)


def _reference_stub():
    return {"result": {"label": "cpu-reference", "argv": [], "environment": {},
                       "timeout_seconds": 600, "elapsed_seconds": 0.01,
                       "exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "",
                       "owned_process_state": "test fixture"}, "validated": {}}


def _preflight_stub():
    return {"ok": True, "reference_sources": {"identity": "CPU FIXTURE"}}


def _args(tmp_path, *, root=None, artifacts=None, profile=None, reference_root=None):
    artifact_path = Path(artifacts) if artifacts is not None else (
        ROOT / ".superpowers" / f"task6-reviewfix-{uuid.uuid4().hex}"
    )
    if artifact_path.is_relative_to(ROOT / ".superpowers"):
        _TEST_ARTIFACTS.add(artifact_path)
    return argparse.Namespace(
        mnist_root=Path(root or tmp_path / "external-mnist"),
        artifacts=artifact_path,
        shader_toolchain_profile=Path(profile or tmp_path / "profile.json"),
        reference_source_root=Path(reference_root or tmp_path / "pytorch-source"),
        python=Path(sys.executable),
        build=ROOT / "build",
    )


@pytest.mark.parametrize("pythonpath", [None, str(ROOT)])
def test_fresh_package_api_process_resolves_real_cpu_validator_and_reaches_live_hold(
    tmp_path, pythonpath
):
    artifacts = ROOT / ".superpowers" / f"task6-fresh-import-{uuid.uuid4().hex}"
    script = r'''
import argparse, json, shutil, sys
from pathlib import Path
from tools import run_mnist_training_qualification as qualification
from tools import run_vulkan_qualification as legacy
root = qualification.ROOT
reference = root / ".superpowers/sdd/2026-10-12-vulkan-mnist-cross-entropy-training/task-1-cpu-reference-fix1.json"
qualification._preflight = lambda *a, **k: {"reference_sources": {"cpu_fixture": True}}
from tools import qualification_process_supervisor as containment
def cpu_child(command, cwd, env, *, timeout_seconds=300):
    assert command[1] == "tools/probe_mnist_training_reference.py"
    shutil.copyfile(reference, Path(command[command.index("--output") + 1]))
    identity = containment._identity(list(command), str(Path(cwd).resolve()), dict(env))
    certificate = containment.Certificate("a" * 32, identity, 1, 1, True, 2, 2, 0, None, "leader-exit",
        1.0, 1.0, 3.0, "isolated-linux-subreaper", 0, 1, "complete-echild", (), 0, True, True, True)
    return ({"exit_code": 0, "stdout": "CPU reference fixture", "stderr": "",
             "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}, certificate)
# CPU fixture of the single live launch seam: structurally valid, non-evidence.
qualification._contained_command = cpu_child
args = argparse.Namespace(mnist_root=root / "not-read-by-preflight", artifacts=Path(sys.argv[1]),
    shader_toolchain_profile=root / "not-read-by-preflight-profile", reference_source_root=root / "not-read-source",
    python=Path(sys.executable), build=root / "build")
summary = qualification.qualify(args)
assert summary["status"] == "blocked", summary
assert summary["evidence_scope"] == "live", summary
assert summary["owned_process_state"] != "not-started", summary
assert summary["reference_receipt_validation"] == "passed CPU validator; not GPU evidence", summary
assert "pytorch_vulkan" not in sys.modules
print(json.dumps({"status": summary["status"], "reason": summary["reason"]}))
'''
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    if pythonpath is not None:
        env["PYTHONPATH"] = pythonpath
    try:
        result = subprocess.run([sys.executable, "-c", script, str(artifacts)], cwd=ROOT,
                                env=env, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert '"status": "blocked"' in result.stdout
        assert "nested-session ownership" in result.stdout
    finally:
        if artifacts.exists():
            shutil.rmtree(artifacts)


def test_direct_cli_help_imports_without_inherited_pythonpath():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, str(ROOT / "tools/run_mnist_training_qualification.py"), "--help"],
                            cwd=ROOT, env=env, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    assert "--shader-toolchain-profile" in result.stdout
    assert "--capture-only" in result.stdout


def test_reference_source_root_is_required_without_argparse_filesystem_validation(tmp_path):
    arguments = [
        "--mnist-root", str(tmp_path / "mnist"),
        "--artifacts", str(tmp_path / "artifacts"),
        "--shader-toolchain-profile", str(tmp_path / "profile.json"),
    ]
    with pytest.raises(SystemExit) as error:
        qualification.parse_args(arguments)
    assert error.value.code == 2

    supplied_root = tmp_path / "arbitrary-nonexistent-reference-root"
    parsed = qualification.parse_args(arguments + ["--reference-source-root", str(supplied_root)])
    assert parsed.reference_source_root == supplied_root


def test_direct_cli_requires_reference_source_root_before_qualification_or_download(tmp_path):
    command = [
        sys.executable, str(ROOT / "tools/run_mnist_training_qualification.py"),
        "--mnist-root", str(tmp_path / "mnist"),
        "--artifacts", str(tmp_path / "artifacts"),
        "--shader-toolchain-profile", str(tmp_path / "profile.json"),
    ]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert "--reference-source-root" in result.stderr
    assert not (tmp_path / "artifacts").exists()

    help_result = subprocess.run(command[:2] + ["--help"], cwd=ROOT, capture_output=True,
                                 text=True, timeout=15)
    assert help_result.returncode == 0, help_result.stderr
    assert "--reference-source-root REFERENCE_SOURCE_ROOT" in help_result.stdout


def test_real_artifact_boundary_rejects_relative_unexcluded_alias_and_existing(tmp_path):
    with pytest.raises(qualification.PreflightBlocked, match="absolute"):
        qualification._safe_artifacts(Path("relative-artifact"))
    with pytest.raises(qualification.PreflightBlocked, match="excluded .superpowers"):
        qualification._safe_artifacts(tmp_path / "outside-artifacts")

    target = tmp_path / "alias-target"
    target.mkdir()
    alias = ROOT / ".superpowers" / f"task6-alias-{uuid.uuid4().hex}"
    alias.symlink_to(target, target_is_directory=True)
    try:
        with pytest.raises(qualification.PreflightBlocked, match="excluded .superpowers"):
            qualification._safe_artifacts(alias / "new-attempt")
    finally:
        alias.unlink()

    fresh = ROOT / ".superpowers" / f"task6-fresh-dir-{uuid.uuid4().hex}"
    resolved = qualification._safe_artifacts(fresh)
    assert resolved == fresh.resolve()
    assert not resolved.exists()

    fresh.mkdir()
    marker = fresh / "preserve"
    marker.write_text("existing")
    with pytest.raises(qualification.PreflightBlocked, match="existing"):
        qualification._safe_artifacts(fresh)
    assert marker.read_text() == "existing"
    shutil.rmtree(fresh)


def test_real_artifact_boundary_rejects_git_nonexcluded_path(tmp_path, monkeypatch):
    fresh = ROOT / ".superpowers" / f"task6-notignored-{uuid.uuid4().hex}"
    real_run = qualification.subprocess.run
    monkeypatch.setattr(qualification.subprocess, "run",
                        lambda *a, **k: type("Result", (), {"returncode": 1})())
    with pytest.raises(qualification.PreflightBlocked, match="not git-excluded"):
        qualification._safe_artifacts(fresh)
    monkeypatch.setattr(qualification.subprocess, "run", real_run)


def test_capture_environment_and_argv_match_task5_canonical_policy(tmp_path):
    from mnist_training_evidence import _pythonpath

    args = _args(tmp_path)
    env = qualification._child_environment({}, args, "async", real=False)
    assert env["PYTHONPATH"] == _pythonpath(ROOT)
    assert env["PYTHONPATH"] == os.pathsep.join((str(ROOT), str(ROOT / "build"), str(ROOT / "tests/python")))
    assert env["PYTORCH_VULKAN_ASYNC_EXECUTION"] == "1"
    assert env["VK_INSTANCE_LAYERS"] == "VK_LAYER_KHRONOS_validation"
    assert "PYTORCH_VULKAN_MNIST_ROOT" not in env
    mode_dir = ROOT / ".superpowers" / "fixture-mode"
    assert qualification._legacy_command(args, mode_dir, "async") == [
        str(args.python), "tools/run_vulkan_qualification.py", "--device", "vk:0",
        "--build", str((ROOT / "build").resolve()), "--python", str(args.python),
        "--output", str(mode_dir / "qualification.json"), "--artifacts", str(mode_dir / "artifacts"),
        "--shader-toolchain-profile", str(args.shader_toolchain_profile.resolve()),
    ]


def test_noncanonical_build_is_blocked_before_any_child(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.build = tmp_path / "alternate-build"
    args.build.mkdir()
    calls = []
    result = qualification.qualify(args, command_runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "blocked"
    assert "canonical ROOT/build" in result["reason"]
    assert calls == []


def test_unbounded_ctest_environment_blocks_before_reference_child(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setenv("CTEST_PARALLEL_LEVEL", "0")
    calls = []
    result = qualification.qualify(args, command_runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "blocked"
    assert "CTEST_PARALLEL_LEVEL" in result["reason"]
    assert calls == []


def test_missing_data_is_blocked_before_any_child(tmp_path):
    args = _args(tmp_path)
    calls = []
    result = qualification.qualify(args, command_runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "blocked"
    assert calls == []
    assert json.loads((args.artifacts / "summary.json").read_text())["status"] == "blocked"


def test_existing_artifacts_are_refused_without_modification(tmp_path):
    args = _args(tmp_path)
    args.artifacts.mkdir()
    marker = args.artifacts / "keep.txt"
    marker.write_text("user data")
    result = qualification.qualify(args, command_runner=lambda *a, **k: pytest.fail("spawn"))
    assert result["status"] == "blocked"
    assert marker.read_text() == "user data"
    assert not (args.artifacts / "summary.json").exists()


def test_missing_profile_blocks_before_any_child(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_validate_data", lambda *a: {"ok": True})
    calls = []
    result = qualification.qualify(args, command_runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "blocked"
    assert calls == []


def test_invalid_profile_fails_before_any_child(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_validate_data", lambda *a: {"ok": True})
    args.shader_toolchain_profile.parent.mkdir(parents=True, exist_ok=True)
    args.shader_toolchain_profile.write_text("{}\n")
    calls = []
    result = qualification.qualify(args, command_runner=lambda *a, **k: calls.append(a))
    assert result["status"] == "fail"
    assert calls == []


@pytest.mark.parametrize(("environment", "blocked"), [
    ({}, False), ({"CTEST_PARALLEL_LEVEL": "1"}, False),
    ({"CTEST_PARALLEL_LEVEL": "0"}, True), ({"CTEST_PARALLEL_LEVEL": "-1"}, True),
    ({"CTEST_PARALLEL_LEVEL": "2"}, True), ({"CTEST_PARALLEL_LEVEL": "01"}, True),
    ({"CTEST_PARALLEL_LEVEL": "bad"}, True),
    ({"PYTEST_ADDOPTS": "-n 2"}, True),
    ({"PYTEST_PLUGINS": "pytest_xdist.plugin"}, True),
])
def test_only_unset_or_canonical_sequential_ctest_setting_is_allowed(environment, blocked):
    if not blocked:
        assert qualification._validate_concurrency_environment(environment) is None
        return
    if environment.get("CTEST_PARALLEL_LEVEL") == "bad":
        with pytest.raises(qualification.QualificationFailure):
            qualification._validate_concurrency_environment(environment)
    else:
        with pytest.raises(qualification.PreflightBlocked):
            qualification._validate_concurrency_environment(environment)


def test_default_live_mode_stops_before_gpu_after_cpu_reference(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_run_reference", lambda *a, **k: _reference_stub())
    calls = []
    monkeypatch.setattr(qualification, "_contained_command", lambda *a, **k: calls.append(a))
    result = qualification.qualify(args)
    assert result["status"] == "blocked"
    assert "nested" in result["reason"].lower()
    assert not calls


def test_capture_only_live_owned_runner_stops_after_validated_16_record_matrix(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.capture_only = True
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    calls = []

    def contained(command, cwd, env, *, timeout_seconds):
        calls.append((command, env, timeout_seconds))
        if "probe_mnist_training_reference.py" in command[1]:
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
        else:
            pytest.fail("capture-only must stop before the full legacy runner")
        certificate = _fixture_certificate(command, env or {})
        return _ok_result(), certificate

    monkeypatch.setattr(qualification, "_contained_command", contained)
    result = qualification.qualify(args)
    assert result["status"] == "blocked"
    assert result["capture_only"] is True
    assert result["capture_matrix"] == {"records": 16, "current": True, "publishing": False}
    assert result["reason"] == "capture-only matrix validated; full qualification and publication remain held"
    assert [item["label"] for item in result["commands"]] == [
        "cpu-reference", "synthetic/async", "synthetic/sync", "real/async", "real/sync"]
    assert [item[2] for item in calls] == [600] * 5
    assert [item["timeout_seconds"] for item in result["commands"]] == [600] * 5
    expected_captures = []
    for stage, mode in (("synthetic", "async"), ("synthetic", "sync"),
                        ("real", "async"), ("real", "sync")):
        attempt = args.artifacts / f"{stage}-{mode}"
        command = [str(args.python), "tools/mnist_training_coverage.py", "capture",
                   "--stage", stage, "--mode", mode, "--output", str(attempt),
                   "--reference-receipt", str(args.artifacts / "cpu-reference.json")]
        if stage == "real":
            command += ["--mnist-root", str(args.mnist_root.resolve())]
        expected_captures.append(command)
    assert [item["argv"] for item in result["commands"][1:]] == expected_captures
    assert all(item["containment_certificate"]["state"] == "complete-echild"
               for item in result["commands"])
    assert result["legacy_reports"] == []
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["status"] == "blocked" and len(saved["commands"]) == 5


def test_capture_only_injected_runner_remains_non_evidence_and_stops_before_full(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.capture_only = True
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    calls = []

    def runner(command, cwd, env, *, timeout_seconds):
        calls.append((command, timeout_seconds))
        if "probe_mnist_training_reference.py" in command[1]:
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
        else:
            pytest.fail("capture-only must not launch legacy runner")
        return _ok_result()

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "blocked"
    assert result["evidence_scope"] == "cpu-test-injection-non-evidence"
    assert result["capture_matrix"]["records"] == 16
    assert len(calls) == 5 and [item[1] for item in calls] == [600] * 5
    assert all("containment_certificate" not in item for item in result["commands"])
    assert result["legacy_reports"] == []


def test_capture_only_current_validation_failure_stops_and_does_not_publish(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.capture_only = True
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k:
                        (_ for _ in ()).throw(ValueError("fixture current validation rejected")))
    calls = []

    def runner(command, cwd, env, *, timeout_seconds):
        calls.append(command)
        if "probe_mnist_training_reference.py" in command[1]:
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
        else:
            pytest.fail("current-validation failure must stop before full runner")
        return _ok_result()

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert "current validation rejected" in result["reason"]
    assert len(calls) == 5
    assert result["legacy_reports"] == []
    assert result.get("capture_matrix") is None
    assert not (args.artifacts / "publication.json").exists()


def test_injected_cpu_runner_uses_exact_order_and_is_non_evidence(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_read_legacy_report", lambda *a, **k: {"fixture": True})
    calls = []
    order = []
    monkeypatch.setattr(qualification, "_validate_reference_receipt",
                        lambda *a, **k: (order.append("reference-validated") or {"fixture": True}))
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: order.append("current-validated"))

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            order.append("cpu-reference")
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            order.append(command[command.index("--stage") + 1] + "/" + command[command.index("--mode") + 1])
            output = Path(command[command.index("--output") + 1])
            output.mkdir()
        else:
            order.append("full/" + env["PYTORCH_VULKAN_ASYNC_EXECUTION"])
        calls.append((command, cwd, env, timeout_seconds))
        return {"exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    labels = [item["label"] for item in result["commands"]]
    assert labels == ["cpu-reference", "synthetic/async", "synthetic/sync",
                      "real/async", "real/sync", "full/async", "full/sync"]
    assert [item[3] for item in calls] == [600] * 5 + [1800, 1800]
    assert result["status"] == "blocked"
    assert result["evidence_scope"] == "cpu-test-injection-non-evidence"
    assert result["commands"][-1]["environment"]["PYTORCH_VULKAN_ASYNC_EXECUTION"] == "0"
    assert order.index("reference-validated") < order.index("synthetic/async")
    assert order.index("current-validated") < order.index("full/1")
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["commands"][-1]["argv"] == result["commands"][-1]["argv"]
    expected_captures = []
    for stage, mode in (("synthetic", "async"), ("synthetic", "sync"),
                        ("real", "async"), ("real", "sync")):
        output = args.artifacts / f"{stage}-{mode}"
        command = [str(args.python), "tools/mnist_training_coverage.py", "capture",
                   "--stage", stage, "--mode", mode, "--output", str(output),
                   "--reference-receipt", str(args.artifacts / "cpu-reference.json")]
        if stage == "real":
            command += ["--mnist-root", str(args.mnist_root.resolve())]
        expected_captures.append(command)
    assert [item["argv"] for item in result["commands"][1:5]] == expected_captures
    from mnist_training_evidence import _pythonpath
    assert all(item["environment"]["PYTHONPATH"] == _pythonpath(ROOT)
               for item in result["commands"][1:5])


@pytest.mark.parametrize("failure_label", ["synthetic/async", "synthetic/sync", "real/async", "real/sync"])
def test_capture_failure_stops_dependent_schedule(tmp_path, monkeypatch, failure_label):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    calls = []

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            label = "cpu-reference"
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            stage = command[command.index("--stage") + 1]
            mode = command[command.index("--mode") + 1]
            label = f"{stage}/{mode}"
            Path(command[command.index("--output") + 1]).mkdir()
        else:
            label = "other"
        calls.append(label)
        status = 77 if label == failure_label else 0
        return {"exit_code": status, "stdout": "", "stderr": "fixture",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail", result
    assert result["owned_process_state"] == "injected CPU runner returned; no live ownership evidence"
    assert "full/async" not in [x["label"] for x in result["commands"]]
    assert calls[-1] == failure_label, (calls, result)
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["owned_process_state"] == result["owned_process_state"]
    assert saved["attempts"][-1]["validated"] is False


def test_success_exit_without_completed_capture_receipt_fails(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    calls = []

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            label = "cpu-reference"
            Path(command[command.index("--output") + 1]).write_text("{}")
        else:
            label = "synthetic/async"
            Path(command[command.index("--output") + 1]).mkdir()
        calls.append(label)
        return {"exit_code": 0, "stdout": "", "stderr": "", "timeout_seconds": timeout_seconds,
                "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert calls == ["cpu-reference", "synthetic/async"]
    assert result["attempts"][-1]["validated"] is False


def test_timeout_exception_is_retained_with_unknown_ownership(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())

    def runner(command, cwd, env, *, timeout_seconds):
        raise RuntimeError("cleanup ownership unresolved")

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert "cleanup ownership unresolved" in result["commands"][0]["exception"]["message"]
    assert result["commands"][0]["owned_process_state"] == "unknown-after-runner-exception"
    assert result["owned_process_state"] == "unknown-after-runner-exception"
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["status"] == "fail"
    assert saved["owned_process_state"] == "unknown-after-runner-exception"
    assert saved["commands"][0]["exception"] == result["commands"][0]["exception"]


def test_timeout_result_is_failure_and_is_written_to_summary(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    result = qualification.qualify(args, command_runner=lambda *a, **k: {
        "exit_code": 124, "stdout": "", "stderr": "command timed out", "timeout_seconds": 600,
        "elapsed_seconds": 600.0,
    })
    assert result["status"] == "fail"
    assert result["commands"][0]["exit_code"] == 124
    assert result["owned_process_state"] == "unknown-after-timeout"
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["commands"][0]["exit_code"] == 124
    assert saved["owned_process_state"] == "unknown-after-timeout"


def test_explicit_skip_output_is_not_required_capture_success(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            Path(command[command.index("--output") + 1]).write_text("{}")
            return {"exit_code": 0, "stdout": "", "stderr": "", "timeout_seconds": timeout_seconds,
                    "elapsed_seconds": 0.01}
        Path(command[command.index("--output") + 1]).mkdir()
        return {"exit_code": 0, "stdout": "SKIPPED [1] required capture", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert len(result["commands"]) == 2


def test_post_legacy_current_identity_change_fails_and_keeps_reports(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    calls_to_aggregate = []

    def aggregate(attempts):
        calls_to_aggregate.append(1)
        if len(calls_to_aggregate) > 1:
            raise qualification.QualificationFailure("source/binary identity changed after legacy build")
        return {"records": [None] * 16}

    monkeypatch.setattr(qualification, "_validate_capture_matrix", aggregate)
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_read_legacy_report", lambda *a, **k: {"fixture": True})

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            Path(command[command.index("--output") + 1]).write_text("{}")
        elif "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
        return {"exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "", "timeout_seconds": timeout_seconds,
                "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert "identity changed" in result["reason"]
    assert len(result["legacy_reports"]) == 2


def test_async_full_runner_failure_prevents_sync_full_runner(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    monkeypatch.setattr(qualification, "_validate_capture_matrix", lambda *a, **k: {"records": [None] * 16})
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    full_modes = []

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            Path(command[command.index("--output") + 1]).write_text("{}")
            exit_code = 0
        elif "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
            exit_code = 0
        else:
            mode = "async" if env["PYTORCH_VULKAN_ASYNC_EXECUTION"] == "1" else "sync"
            full_modes.append(mode)
            exit_code = 9
        return {"exit_code": exit_code, "stdout": "", "stderr": "fixture",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    assert result["status"] == "fail"
    assert full_modes == ["async"]
    assert result["legacy_reports"] == [{"mode": "async",
        "path": str(args.artifacts / "full-async" / "qualification.json"),
        "outer_exit_code": 9, "validated": False}]


@pytest.mark.parametrize("failure_mode", ["async", "sync"])
@pytest.mark.parametrize("failure_kind", ["timeout", "cleanup-exception"])
def test_full_child_ownership_failure_is_retained_and_stops_followup(
    tmp_path, monkeypatch, failure_mode, failure_kind
):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_validate_reference_receipt", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(qualification, "_validate_completed_receipt", lambda *a, **k: None)
    aggregate_calls = []

    def aggregate(attempts):
        aggregate_calls.append(tuple(attempts))
        return {"records": [None] * 16}

    monkeypatch.setattr(qualification, "_validate_capture_matrix", aggregate)
    monkeypatch.setattr(qualification, "_validate_current", lambda *a, **k: None)
    full_modes = []

    def runner(command, cwd, env, *, timeout_seconds):
        if any("probe_mnist_training_reference.py" in str(item) for item in command):
            Path(command[command.index("--output") + 1]).write_text("{}")
            return {"exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "",
                    "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}
        if "capture" in command:
            Path(command[command.index("--output") + 1]).mkdir()
            return {"exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "",
                    "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

        mode = "async" if env["PYTORCH_VULKAN_ASYNC_EXECUTION"] == "1" else "sync"
        full_modes.append(mode)
        if mode == failure_mode:
            if failure_kind == "cleanup-exception":
                raise RuntimeError("fixture cleanup ownership failure")
            return {"exit_code": 124, "stdout": "", "stderr": "command timed out",
                    "timeout_seconds": timeout_seconds, "elapsed_seconds": timeout_seconds}

        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_legacy_report_fixture(python=args.python)))
        return {"exit_code": 0, "stdout": "CPU FIXTURE", "stderr": "",
                "timeout_seconds": timeout_seconds, "elapsed_seconds": 0.01}

    result = qualification.qualify(args, command_runner=runner)
    state = "unknown-after-timeout" if failure_kind == "timeout" else "unknown-after-runner-exception"
    expected_exit = 124 if failure_kind == "timeout" else 127
    expected_modes = ["async"] if failure_mode == "async" else ["async", "sync"]
    assert result["status"] == "fail"
    assert result["owned_process_state"] == state
    assert full_modes == expected_modes
    assert len(aggregate_calls) == 1, "post-legacy aggregation must not run after child failure"
    assert result["commands"][-1]["label"] == f"full/{failure_mode}"
    assert result["commands"][-1]["exit_code"] == expected_exit
    report_entries = result["legacy_reports"]
    assert [entry["mode"] for entry in report_entries] == expected_modes
    last_report = report_entries[-1]
    assert last_report["path"] == str(args.artifacts / f"full-{failure_mode}" / "qualification.json")
    assert last_report["outer_exit_code"] == expected_exit
    assert last_report["validated"] is False
    saved = json.loads((args.artifacts / "summary.json").read_text())
    assert saved["owned_process_state"] == state
    assert saved["commands"][-1] == result["commands"][-1]
    if failure_kind == "cleanup-exception":
        assert saved["commands"][-1]["owned_process_state"] == state
    else:
        assert "no independent descendant-session certificate" in saved["commands"][-1]["owned_process_state"]
    assert saved["legacy_reports"] == report_entries


def _legacy_report_fixture(python="/fixture/python", *, skip_line=None, summary=None):
    # Minimal CPU-only projection of the retained actual runner report shape:
    # exact four wrapper-first conformance argv classes and pytest -rs summaries.
    # Counts and source line numbers are deliberately fixture variables.
    if skip_line is None:
        skip_line = "SKIPPED [7] tests/python/test_vulkan_conformance.py:44: positive case"
    if summary is None:
        summary = "12 passed, 7 skipped in 0.01s"
    gates = []
    for name in qualification.GATES:
        if name == "additional_implementation":
            gates.append({"name": name, "status": "skip", "reason":
                          "no additional Vulkan implementation was requested", "commands": []})
        elif name == "ctest":
            commands = [{"command": legacy.native_ctest_command((ROOT / "build").resolve()),
                         "exit_code": 0, "stdout": "100% tests passed", "stderr": "",
                         "timeout_seconds": 600}]
            gates.append({"name": name, "status": "pass", "commands": commands})
        elif name == "python_suite":
            commands = [{"command": legacy.python_suite_command(python),
                         "exit_code": 0, "stdout": "2331 passed in 478.88s", "stderr": "",
                         "timeout_seconds": 600}]
            gates.append({"name": name, "status": "pass", "commands": commands})
        elif name == "conformance":
            commands = []
            # Literal argv fixture independently locks the historical runner contract.
            prefix = [str(python), "tools/vulkan_wrapper_pytest.py", "-q", "-rs"]
            expected_commands = [
                prefix + ["tests/python/test_vulkan_conformance.py"],
                prefix + [
                    "tests/python/test_vulkan_workload_conformance.py::test_stock_sgd_executes_three_steps_in_both_reset_modes",
                    "tests/python/test_vulkan_workload_conformance.py::test_parameter_hvp_executes_with_named_live_history",
                    "tests/python/test_vulkan_workload_conformance.py::test_matrix_stock_sgd_executes_three_steps_in_both_reset_modes",
                    "tests/python/test_vulkan_workload_conformance.py::test_matrix_parameter_hvp_executes_with_named_live_history",
                ],
                prefix + ["tests/python/test_vulkan_linear.py::test_stock_linear_each_declared_rank"],
                prefix + ["tests/python/test_vulkan_composed_matmul.py::test_composed_model"],
            ]
            for index, argv in enumerate(expected_commands):
                output = f"{skip_line}\n{summary}\n" if index == 0 else "4 passed in 0.01s\n"
                commands.append({"command": argv, "exit_code": 0, "stdout": output, "stderr": ""})
            gates.append({"name": name, "status": "skip", "commands": commands})
        else:
            gates.append({"name": name, "status": "pass", "commands": [{
                "command": ["python", "gate.py"], "exit_code": 0, "stdout": "", "stderr": "",
            }]})
    return {"schema_version": 1, "device": "vk:0",
            "device_probe": {"status": "available", "exit_code": 0},
            "gates": gates, "counts": {"pass": 11, "skip": 2, "fail": 0}}


def test_legacy_report_requires_inventory_before_later_report_fields():
    report = _legacy_report_fixture()
    report["gates"] = []
    with pytest.raises(qualification.QualificationFailure, match="gate inventory/order"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


@pytest.mark.parametrize("mutation", ["missing-gate", "duplicate-gate"])
def test_legacy_report_rejects_independent_missing_and_duplicate_gates(mutation):
    report = _legacy_report_fixture()
    if mutation == "missing-gate":
        report["gates"].pop(2)
    else:
        report["gates"][2] = dict(report["gates"][1])
    with pytest.raises(qualification.QualificationFailure, match="gate inventory/order"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_report_rejects_malformed_gate_status_after_valid_probe():
    report = _legacy_report_fixture()
    report["gates"][2]["status"] = "unknown"
    with pytest.raises(qualification.QualificationFailure, match="malformed gate status"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_report_rejects_missing_conformance_command():
    report = _legacy_report_fixture()
    report["gates"][5]["commands"].pop()
    with pytest.raises(qualification.QualificationFailure, match="conformance command inventory"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_actual_legacy_report_contract_accepts_only_documented_skips():
    report = _legacy_report_fixture()
    accepted = qualification.validate_legacy_report(report, 0, python="/fixture/python")
    assert accepted["gates"] == {"pass": 11, "skip": 2, "fail": 0}
    report["gates"][0] = {"name": "manifest", "status": "skip", "reason": "not run", "commands": [{
        "command": ["python", "gate.py"], "exit_code": 0,
    }]}
    report["counts"] = {"pass": 10, "skip": 3, "fail": 0}
    with pytest.raises(qualification.QualificationFailure, match="unexpected skipped required gate"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_gate_inventory_schedules_python_suite_between_ctest_and_conformance():
    assert qualification.GATES == (
        "manifest", "build", "bootstrap_provenance", "ctest", "python_suite",
        "conformance", "shader_verification", "validation_layer", "stress",
        "timing", "benchmark", "environment", "additional_implementation",
    )


def test_legacy_report_rejects_missing_python_suite_gate():
    report = _legacy_report_fixture()
    report["gates"].pop(qualification.GATES.index("python_suite"))
    with pytest.raises(qualification.QualificationFailure, match="gate inventory/order"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


@pytest.mark.parametrize("unsplit", ["partial", "regex-subset"])
def test_legacy_report_rejects_old_unsplit_ctest_selection(unsplit):
    report = _legacy_report_fixture()
    index = qualification.GATES.index("ctest")
    argv = ["ctest", "--test-dir", str((ROOT / "build").resolve()), "--output-on-failure"]
    if unsplit == "regex-subset":
        # An unanchored or nonnative CTest filter is still not the split contract.
        argv = argv + ["-R", "vulkan"]
    report["gates"][index]["commands"][0]["command"] = argv
    with pytest.raises(qualification.QualificationFailure, match="native-only"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


@pytest.mark.parametrize("substitution", [
    "plain-pytest-module", "missing-durations", "unsplit-ctest-suite",
    "reordered-arguments", "wrong-suite-target",
])
def test_legacy_report_rejects_substituted_python_suite_invocation(substitution):
    report = _legacy_report_fixture()
    index = qualification.GATES.index("python_suite")
    command = report["gates"][index]["commands"][0]
    if substitution == "plain-pytest-module":
        command["command"] = ["/fixture/python", "-m", "pytest", "-q", "-rs",
                              "--durations=30", "tests/python"]
    elif substitution == "missing-durations":
        command["command"] = ["/fixture/python", "tools/vulkan_wrapper_pytest.py",
                              "-q", "-rs", "tests/python"]
    elif substitution == "unsplit-ctest-suite":
        command["command"] = ["ctest", "--test-dir", str((ROOT / "build").resolve()),
                              "--output-on-failure"]
    elif substitution == "reordered-arguments":
        command["command"] = ["/fixture/python", "tools/vulkan_wrapper_pytest.py",
                              "tests/python", "-q", "-rs", "--durations=30"]
    else:
        command["command"] = ["/fixture/python", "tools/vulkan_wrapper_pytest.py",
                              "-q", "-rs", "--durations=30", "tests/python/test_vulkan_conformance.py"]
    with pytest.raises(qualification.QualificationFailure, match="python_suite gate"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_report_rejects_python_suite_gate_with_two_invocations():
    report = _legacy_report_fixture()
    index = qualification.GATES.index("python_suite")
    surplus = dict(report["gates"][index]["commands"][0])
    report["gates"][index]["commands"].append(surplus)
    with pytest.raises(qualification.QualificationFailure, match="exactly one"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


@pytest.mark.parametrize("gate", ["ctest", "python_suite"])
def test_legacy_report_rejects_split_gates_without_the_600_second_budget(gate):
    report = _legacy_report_fixture()
    report["gates"][qualification.GATES.index(gate)]["commands"][0]["timeout_seconds"] = 300
    with pytest.raises(qualification.QualificationFailure, match="600"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_report_rejects_skipped_python_suite_gate():
    report = _legacy_report_fixture()
    index = qualification.GATES.index("python_suite")
    report["gates"][index] = {"name": "python_suite", "status": "skip",
                              "reason": "registered skip", "commands": [{
        "command": legacy.python_suite_command("/fixture/python"),
        "exit_code": 0, "stdout": "", "stderr": "", "timeout_seconds": 600}]}
    report["counts"] = {"pass": 10, "skip": 3, "fail": 0}
    with pytest.raises(qualification.QualificationFailure, match="unexpected skipped required gate"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_report_rejects_exit_77_even_if_outer_exit_is_zero():
    report = _legacy_report_fixture()
    report["gates"][0]["commands"][0]["exit_code"] = 77
    with pytest.raises(qualification.QualificationFailure, match="failed command"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_conformance_rejects_any_undocumented_skip_reason():
    report = _legacy_report_fixture()
    report["gates"][5]["commands"][0]["stdout"] += (
        "SKIPPED [1] tests/python/test_vulkan_conformance.py:20: environment unavailable\n"
    )
    with pytest.raises(qualification.QualificationFailure, match="undocumented skip"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


@pytest.mark.parametrize("skip_line", [
    "SKIPPED [7] tests/python/test_vulkan_attention.py:44: positive case",
    "SKIPPED [7] tests/python/test_vulkan_conformance.py:44: prefix positive case",
    "SKIPPED [7] tests/python/test_vulkan_conformance.py:44: positive case suffix",
])
def test_legacy_conformance_rejects_unknown_source_and_nonexact_reason(skip_line):
    report = _legacy_report_fixture(skip_line=skip_line)
    with pytest.raises(qualification.QualificationFailure, match="undocumented skip"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_legacy_conformance_requires_aggregate_skip_summary():
    report = _legacy_report_fixture(summary="12 passed in 0.01s")
    with pytest.raises(qualification.QualificationFailure, match="matching short and aggregate"):
        qualification.validate_legacy_report(report, 0, python="/fixture/python")


def test_cli_reports_blocked_as_nonzero(tmp_path, monkeypatch):
    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(qualification, "qualify", lambda *a, **k: {"status": "blocked"})
    assert qualification.main([]) != 0


def _fixture_certificate(command, env):
    from tools import qualification_process_supervisor as containment
    identity = containment._identity(list(command), str(ROOT), dict(env))
    return containment.Certificate("b" * 32, identity, 1, 1, True, 2, 2, 0, None, "leader-exit",
        1.0, 1.0, 3.0, "isolated-linux-subreaper", 0, 1, "complete-echild", (), 0, True, True, True)


def _ok_result():
    return {"exit_code": 0, "stdout": "o", "stderr": "", "timeout_seconds": 5, "elapsed_seconds": 0.1}


def test_contained_runner_keeps_the_certificate_beside_the_public_result(monkeypatch):
    environment = {"PATH": "/usr/bin", "SECRET_TOKEN": "must-not-be-duplicated"}
    monkeypatch.setattr(qualification, "_contained_command", lambda command, cwd, env, *, timeout_seconds: (
        _ok_result(), _fixture_certificate(command, env)))
    runner = qualification._ContainedRunner()
    result = qualification._run_result(runner, ["example"], environment, 5, "label", owned=runner)
    assert result["owned_process_state"] == "contained: certified empty tree and reaped worker"
    certificate = result["containment_certificate"]
    assert certificate["state"] == "complete-echild" and certificate["worker_reaped"] is True
    assert "environment" not in certificate["identity"]
    summary = {"commands": []}
    qualification._record_command(summary, result, injected=False)
    assert summary["owned_process_state"] == result["owned_process_state"]


@pytest.mark.parametrize("fake", [None, {"state": "complete-echild"}, "complete"])
def test_contained_runner_rejects_untrusted_certificates(monkeypatch, fake):
    monkeypatch.setattr(qualification, "_contained_command",
                        lambda *a, **k: (_ok_result(), fake))
    runner = qualification._ContainedRunner()
    result = qualification._run_result(runner, ["example"], {}, 5, "label", owned=runner)
    assert result["exit_code"] == 127 and result["exception"]
    assert result["owned_process_state"] == "unknown-after-containment-error"
    assert result["containment_certificate"] is None


def test_containment_error_retains_live_authority_and_never_reports_success(monkeypatch):
    from tools import qualification_process_supervisor as containment
    authority = object()
    certificate = _fixture_certificate(["example"], {})

    def failing(*args, **kwargs):
        raise containment.ContainmentError("unknown", partial_result=_ok_result(), invocation_id="b" * 32,
                                           certificate=certificate, authority=authority)

    monkeypatch.setattr(qualification, "_contained_command", failing)
    runner = qualification._ContainedRunner()
    result = qualification._run_result(runner, ["example"], {}, 5, "label", owned=runner)
    assert runner.retained_authorities == [authority]
    assert result["exit_code"] == 127
    assert result["owned_process_state"] == "unknown-after-containment-error"


def test_injected_runner_cannot_supply_a_containment_certificate():
    def runner(command, cwd, env=None, *, timeout_seconds=300):
        return dict(_ok_result(), containment_certificate={"state": "complete-echild"})

    result = qualification._run_result(runner, ["example"], {}, 5, "label")
    assert "containment_certificate" not in result
    assert result["owned_process_state"] == "helper-returned; no independent descendant-session certificate"
    summary = {"commands": []}
    qualification._record_command(summary, result, injected=True)
    assert summary["owned_process_state"] == "injected CPU runner returned; no live ownership evidence"


def test_real_cpu_adapter_returns_a_genuine_containment_certificate():
    runner = qualification._ContainedRunner()
    result = qualification._run_result(
        runner, [sys.executable, "-c", "print('adapter-ok')"], dict(os.environ), 20, "adapter", owned=runner)
    assert result["exit_code"] == 0 and "adapter-ok" in result["stdout"]
    certificate = result["containment_certificate"]
    assert certificate["state"] == "complete-echild" and certificate["terminal_received"] is True
    assert certificate["worker_reaped"] is True and certificate["method"] == "isolated-linux-subreaper"
    assert not runner.retained_authorities


def test_only_the_internally_bound_runner_instance_is_trusted(monkeypatch):
    class Laundering(qualification._ContainedRunner):
        def __call__(self, command, cwd, env=None, *, timeout_seconds=300):
            self._certificate = _fixture_certificate(command, env or {})
            return _ok_result()

    monkeypatch.setattr(qualification, "_contained_command", lambda command, cwd, env, *, timeout_seconds: (
        _ok_result(), _fixture_certificate(command, env)))
    internal = qualification._ContainedRunner()
    other_exact = qualification._ContainedRunner()
    cases = {
        "subclass bound as owned elsewhere": (Laundering(), internal),
        "different exact instance": (other_exact, internal),
        "no internal runner (injected path)": (qualification._ContainedRunner(), None),
    }
    for name, (runner, owned) in cases.items():
        result = qualification._run_result(runner, ["example"], {}, 5, "label", owned=owned)
        assert "containment_certificate" not in result, name
        assert result["owned_process_state"].startswith("helper-returned"), name
    ok = qualification._run_result(internal, ["example"], {}, 5, "label", owned=internal)
    assert ok["owned_process_state"].startswith("contained")


@pytest.mark.parametrize("kind", ["subclass", "exact-instance"])
def test_public_injection_of_adapter_types_stays_non_evidence(tmp_path, monkeypatch, kind):
    class Laundering(qualification._ContainedRunner):
        def __call__(self, command, cwd, env=None, *, timeout_seconds=300):
            self._certificate = _fixture_certificate(command, env or {})
            return _ok_result()

    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_contained_command", lambda command, cwd, env, *, timeout_seconds: (
        _ok_result(), _fixture_certificate(command, env)))
    runner = Laundering() if kind == "subclass" else qualification._ContainedRunner()
    result = qualification.qualify(_args(tmp_path), command_runner=runner)
    command = result["commands"][0]
    assert "containment_certificate" not in command
    assert result["evidence_scope"] == "cpu-test-injection-non-evidence"


def test_qualify_raises_the_original_error_when_live_authority_remains(tmp_path, monkeypatch):
    from tools import qualification_process_supervisor as containment

    class Pending:
        reaped = False

    partial = dict(_ok_result(), stdout="partial-out", stderr="partial-err", exit_code=0)

    def failing(*args, **kwargs):
        raise containment.ContainmentError("unknown", partial_result=partial, invocation_id="c" * 32,
                                           certificate=None, authority=Pending())

    args = _args(tmp_path)
    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_contained_command", failing)
    with pytest.raises(containment.ContainmentError) as raised:
        qualification.qualify(args)
    assert raised.value.authority is not None and raised.value.invocation_id == "c" * 32
    saved = json.loads((args.artifacts / "summary.json").read_text())
    command = saved["commands"][0]
    assert command["exit_code"] == 127 and "partial-out" in command["stdout"]
    assert "partial-err" in command["stderr"] and command["invocation_id"] == "c" * 32
    assert saved["owned_process_state"] == "unknown-after-containment-error"


class _AppInterrupt(BaseException):
    """A catchable non-Exception interruption, like an application signal handler raises."""


@pytest.mark.parametrize("interruption", [KeyboardInterrupt, _AppInterrupt])
def test_interrupted_summary_write_still_raises_the_same_authority_bearing_error(
        tmp_path, monkeypatch, interruption):
    from tools import qualification_process_supervisor as containment

    class Pending:
        reaped = False

    created = []

    def failing(*args, **kwargs):
        error = containment.ContainmentError("unknown", partial_result=_ok_result(),
            invocation_id="d" * 32, certificate=None, authority=Pending())
        created.append(error)
        raise error

    def interrupted(*args, **kwargs):
        raise interruption("injected during summary persistence")

    monkeypatch.setattr(qualification, "_preflight", lambda *a, **k: _preflight_stub())
    monkeypatch.setattr(qualification, "_contained_command", failing)
    monkeypatch.setattr(qualification, "_write_summary", interrupted)
    with pytest.raises(containment.ContainmentError) as raised:
        qualification.qualify(_args(tmp_path))
    assert raised.value is created[0] and raised.value.authority is not None
    assert isinstance(raised.value.__context__, interruption)
