#!/usr/bin/env python3
"""Run the mandatory ordered MNIST training qualification gates.

The live legacy-runner path is intentionally blocked until its nested-session
process ownership has a separately reviewed containment repair. Supplying a
``command_runner`` is a CPU test seam only: its output is always labelled
non-evidence and cannot produce a successful milestone.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
_IMPORT_PATHS = (ROOT, ROOT / "tools", ROOT / "tests/python")
for _import_path in reversed(_IMPORT_PATHS):
    _spelling = str(_import_path)
    if _spelling not in sys.path:
        sys.path.insert(0, _spelling)

DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
GATES = (
    "manifest", "build", "bootstrap_provenance", "ctest", "conformance",
    "shader_verification", "validation_layer", "stress", "timing", "benchmark",
    "environment", "additional_implementation",
)
CONFORMANCE_CASES = (
    ("tests/python/test_vulkan_conformance.py",),
    (
        "tests/python/test_vulkan_workload_conformance.py::test_stock_sgd_executes_three_steps_in_both_reset_modes",
        "tests/python/test_vulkan_workload_conformance.py::test_parameter_hvp_executes_with_named_live_history",
        "tests/python/test_vulkan_workload_conformance.py::test_matrix_stock_sgd_executes_three_steps_in_both_reset_modes",
        "tests/python/test_vulkan_workload_conformance.py::test_matrix_parameter_hvp_executes_with_named_live_history",
    ),
    ("tests/python/test_vulkan_linear.py::test_stock_linear_each_declared_rank",),
    ("tests/python/test_vulkan_composed_matmul.py::test_composed_model",),
)
OWNERSHIP_HOLD = (
    "unresolved nested-session ownership: the outer legacy-runner timeout cannot "
    "prove cleanup of separately sessioned inner GPU gate processes"
)


def _contained_command(command, cwd, env, *, timeout_seconds):
    """The only live launch seam: a real isolated worker and its local certificate."""
    from tools import qualification_process_supervisor as containment
    with containment.caller_wait_scope(exclusive_wait=True):
        return containment.run_command_core(command, cwd, env, timeout_seconds=timeout_seconds)


class _ContainedRunner:
    """Default live runner. The certificate stays beside the public dict, never inside it."""

    def __init__(self):
        self._certificate = None
        self.retained_authorities = []  # Unreaped worker authority is never dropped.
        self.retained_errors = []  # The original errors that carry that authority.

    def __call__(self, command, cwd, env=None, *, timeout_seconds=300):
        from tools import qualification_process_supervisor as containment
        self._certificate = None
        try:
            result, certificate = _contained_command(command, cwd, env, timeout_seconds=timeout_seconds)
            if type(certificate) is not containment.Certificate:
                raise QualificationFailure("untrusted containment certificate")
            containment.require_complete(result, certificate)
        except containment.ContainmentError as error:
            if error.authority is not None:
                self.retained_authorities.append(error.authority)
                self.retained_errors.append(error)
            self._certificate = error.certificate
            raise
        self._certificate = certificate
        return result

    def unreaped_error(self):
        for error in self.retained_errors:
            if not error.authority.reaped:
                return error
        return None

    def take_certificate(self):
        certificate, self._certificate = self._certificate, None
        return certificate


def _certificate_summary(certificate) -> dict | None:
    if certificate is None:
        return None
    from tools import qualification_process_supervisor as containment
    record = containment.certificate_record(certificate)
    # The full environment is already retained once on the command record.
    record["identity"] = {k: v for k, v in record["identity"].items() if k != "environment"}
    return record


class PreflightBlocked(RuntimeError):
    """A prerequisite or authorization boundary is unavailable."""


class QualificationFailure(RuntimeError):
    """Malformed identity, child execution, or evidence validation failure."""


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mnist-root", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--shader-toolchain-profile", type=Path, required=True)
    parser.add_argument("--reference-source-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--capture-only", action="store_true",
                        help="run and validate the four MNIST captures, then stop before full qualification")
    return parser.parse_args(argv)


def _canonical_build_path() -> Path:
    return (ROOT / "build").resolve()


def _canonical_pythonpath() -> str:
    return os.pathsep.join(str(ROOT / part) for part in (".", "build", "tests/python"))


def _require_canonical_build(path: Path) -> Path:
    supplied = Path(path).expanduser()
    canonical = _canonical_build_path()
    if not supplied.is_absolute() or supplied.resolve(strict=False) != canonical:
        raise PreflightBlocked(f"--build must select canonical ROOT/build ({canonical}); alternate builds are unsupported")
    if not canonical.is_dir():
        raise PreflightBlocked(f"canonical ROOT/build is missing: {canonical}")
    return canonical


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_artifacts(path: Path) -> Path:
    supplied = Path(path).expanduser()
    if not supplied.is_absolute():
        raise PreflightBlocked("--artifacts must be an absolute new directory")
    resolved = supplied.resolve(strict=False)
    if resolved.exists():
        raise PreflightBlocked(f"refusing existing artifact directory: {resolved}")
    if not resolved.is_relative_to((ROOT / ".superpowers").resolve()):
        raise PreflightBlocked("artifact directory must be under excluded .superpowers")
    relative = resolved.relative_to(ROOT)
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(relative)], cwd=ROOT,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=False,
    )
    if ignored.returncode != 0:
        raise PreflightBlocked("artifact directory is not git-excluded")
    return resolved


def _validate_concurrency_environment(environment: dict[str, str]) -> None:
    level = environment.get("CTEST_PARALLEL_LEVEL")
    if level is not None and level != "1":
        try:
            int(level)
        except ValueError as error:
            raise QualificationFailure("malformed CTEST_PARALLEL_LEVEL") from error
        raise PreflightBlocked("CTEST_PARALLEL_LEVEL must be unset or exactly '1' for sequential execution")
    options = environment.get("PYTEST_ADDOPTS", "")
    try:
        tokens = shlex.split(options)
    except ValueError as error:
        raise QualificationFailure("malformed PYTEST_ADDOPTS") from error
    for index, token in enumerate(tokens):
        if token == "-n" or token.startswith("-n") or token in {"--numprocesses", "--dist"}:
            raise PreflightBlocked("pytest worker/xdist settings are not permitted")
        if token.startswith("--numprocesses=") or token.startswith("--dist="):
            raise PreflightBlocked("pytest worker/xdist settings are not permitted")
    plugins = environment.get("PYTEST_PLUGINS", "").lower()
    if "xdist" in plugins:
        raise PreflightBlocked("pytest xdist plugin is not permitted")


def _validate_data(root: Path) -> dict:
    from mnist_data import external_data_root, load_mnist_subset

    try:
        resolved = external_data_root(root, ROOT)
    except ValueError as error:
        raise PreflightBlocked(str(error)) from error
    identities = {}
    # Each split load authenticates all four canonical compressed resources.
    for split in ("train", "test"):
        try:
            _, metadata = load_mnist_subset(resolved, split=split, repo_root=ROOT)
        except FileNotFoundError as error:
            raise PreflightBlocked(str(error)) from error
        identities[split] = metadata["data_identity"]
    return {"root": str(resolved), "identities": identities}


def _validate_profile(profile: Path) -> dict:
    from shader_toolchain import profile_provenance

    supplied = Path(profile).expanduser()
    if not supplied.is_absolute():
        raise PreflightBlocked("shader toolchain profile path must be absolute")
    if not supplied.is_file():
        raise PreflightBlocked(f"shader toolchain profile missing: {supplied}")
    try:
        provenance = profile_provenance(supplied)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise QualificationFailure(f"invalid shader toolchain profile: {error}") from error
    return provenance


def _validate_sources(root: Path) -> dict:
    supplied = Path(root).expanduser()
    if not supplied.is_absolute():
        raise PreflightBlocked("reference source root must be absolute")
    if not supplied.is_dir():
        raise PreflightBlocked(f"pinned PyTorch source snapshot missing: {supplied}")
    try:
        from probe_mnist_training_reference import source_inventory, validate_versions
        import torch

        inventory = source_inventory(supplied)
        validate_versions(str(torch.__version__), torch.version.git_version)
        return {"root": str(supplied.resolve()), "source_inventory": inventory,
                "torch_version": str(torch.__version__), "upstream_revision": torch.version.git_version}
    except FileNotFoundError as error:
        raise PreflightBlocked(str(error)) from error
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        raise QualificationFailure(f"pinned reference-source identity invalid: {error}") from error


def _validate_preflight(args, environment: dict[str, str]) -> dict:
    if not Path(args.mnist_root).is_absolute():
        raise PreflightBlocked("--mnist-root must be absolute")
    if not Path(args.python).is_absolute() or not Path(args.python).is_file() or not os.access(args.python, os.X_OK):
        raise PreflightBlocked("--python must name an absolute executable")
    build = _require_canonical_build(Path(args.build))
    _validate_concurrency_environment(environment)
    return {
        "data": _validate_data(Path(args.mnist_root)),
        "shader_toolchain": _validate_profile(Path(args.shader_toolchain_profile)),
        "reference_sources": _validate_sources(Path(args.reference_source_root)),
        "python": str(args.python),
        "python_realpath": str(Path(args.python).resolve()),
        "build": str(build),
    }


def _child_environment(base: dict[str, str], args, mode: str, *, real: bool) -> dict[str, str]:
    _require_canonical_build(Path(args.build))
    if mode not in {"async", "sync"}:
        raise ValueError(f"unsupported execution mode: {mode}")
    env = dict(base)
    env["PYTORCH_VULKAN_ASYNC_EXECUTION"] = "1" if mode == "async" else "0"
    env["PYTHONPATH"] = _canonical_pythonpath()
    env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    env["PYTORCH_VULKAN_DEVICE"] = "vk:0"
    env["VULKAN_DEVICE"] = "vk:0"
    if real:
        env["PYTORCH_VULKAN_MNIST_ROOT"] = str(Path(args.mnist_root).resolve())
    else:
        env.pop("PYTORCH_VULKAN_MNIST_ROOT", None)
    return env


def _run_result(command_runner, command, env, timeout, label, *, owned=None):
    # Authority belongs only to the exact adapter instance the live path created
    # (`owned`); injected instances, subclasses and lookalikes never match it.
    trusted = owned is not None and command_runner is owned
    began = time.monotonic()
    try:
        raw = command_runner(command, ROOT, env, timeout_seconds=timeout)
        if type(raw) is not dict or type(raw.get("exit_code")) is not int:
            raise QualificationFailure("command runner returned malformed result")
        result = dict(raw)
        # A runner-supplied certificate is never authority; only our own adapter has one.
        result.pop("containment_certificate", None)
        owned_state = "helper-returned; no independent descendant-session certificate"
        if trusted:
            certificate = command_runner.take_certificate()
            result["containment_certificate"] = _certificate_summary(certificate)
            owned_state = "contained: certified empty tree and reaped worker"
    except BaseException as error:
        # Keep cleanup/runtime errors, but do not claim the owned group was stopped.
        result = {"exit_code": 127, "stdout": "", "stderr": f"{type(error).__name__}: {error}",
                  "timeout_seconds": timeout, "elapsed_seconds": time.monotonic() - began,
                  "exception": {"type": type(error).__name__, "message": str(error)}}
        owned_state = "unknown-after-runner-exception"
        partial = getattr(error, "partial_result", None)
        if hasattr(error, "invocation_id") and type(partial) is dict:
            # Preserve original partial evidence; the command is still a failure.
            result["stdout"] = partial["stdout"] if type(partial.get("stdout")) is str else ""
            result["stderr"] = (partial["stderr"] if type(partial.get("stderr")) is str else "") \
                + f"\n{type(error).__name__}: {error}"
            result["invocation_id"] = error.invocation_id
        if trusted:
            # Diagnostic only: a failure certificate never proves success.
            result["containment_certificate"] = _certificate_summary(command_runner.take_certificate())
            owned_state = "unknown-after-containment-error"
    result.update(label=label, argv=list(command), environment=dict(env),
                  timeout_seconds=timeout, elapsed_seconds=result.get("elapsed_seconds", time.monotonic() - began),
                  owned_process_state=owned_state)
    return result


def _record_command(summary: dict, result: dict, *, injected: bool) -> None:
    summary["commands"].append(result)
    if result.get("exception"):
        summary["owned_process_state"] = result["owned_process_state"]
    elif result.get("containment_certificate"):
        summary["owned_process_state"] = result["owned_process_state"]
    elif result.get("exit_code") == 124:
        summary["owned_process_state"] = "unknown-after-timeout"
    elif result.get("exit_code") == 127:
        summary["owned_process_state"] = "unknown-after-spawn-failure"
    elif injected:
        summary["owned_process_state"] = "injected CPU runner returned; no live ownership evidence"
    else:
        summary["owned_process_state"] = "command returned; no orphan-free certificate"


def _failed_child(result: dict) -> bool:
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    return result["exit_code"] != 0 or result["exit_code"] == 77 or bool(
        re.search(r"(?im)^\s*(?:SKIPPED|\d+\s+skipped)\b", output)
    )


def _write_summary(path: Path | None, summary: dict) -> None:
    if path is None:
        return
    target = path / "summary.json"
    if target.exists():
        # Do not overwrite prior attempt/report data.
        raise PreflightBlocked(f"refusing existing summary: {target}")
    target.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _validate_reference_receipt(path: Path) -> dict:
    from mnist_training_evidence import validate_reference_receipt

    return validate_reference_receipt(path)


def _validate_completed_receipt(attempt: Path, stage: str, mode: str) -> None:
    from mnist_training_evidence import _validate_attempt

    fragment, receipt, _ = _validate_attempt(attempt, repo_root=ROOT)
    if receipt.get("stage") != stage or receipt.get("mode") != mode:
        raise QualificationFailure("completed receipt stage/mode mismatch")
    if fragment.get("stage") != stage or fragment.get("execution_mode") != mode:
        raise QualificationFailure("capture fragment stage/mode mismatch")


def _validate_capture_matrix(attempts: list[Path]) -> dict:
    from mnist_training_evidence import aggregate_attempts

    document = aggregate_attempts(attempts, repo_root=ROOT)
    if len(document.get("records", [])) != 16:
        raise QualificationFailure("current aggregate does not contain 16 records")
    return document


def _validate_current(document: dict, attempts: list[Path]) -> None:
    from mnist_training_evidence import validate_current

    validate_current(document, repo_root=ROOT, receipt_roots=attempts)


def _conformance_commands(python: str | Path) -> list[list[str]]:
    prefix = [str(python), "tools/vulkan_wrapper_pytest.py", "-q", "-rs"]
    return [prefix + list(targets) for targets in CONFORMANCE_CASES]


def _validate_conformance_skip(commands: list[dict], python: str | Path) -> None:
    expected = _conformance_commands(python)
    if len(commands) != len(expected) or [command.get("command") for command in commands] != expected:
        raise QualificationFailure("conformance command inventory/order differs from legacy runner")
    allowed = re.compile(
        r"SKIPPED \[(\d+)\] tests/python/test_vulkan_conformance\.py:(\d+): positive case"
    )
    short_counts = []
    aggregate_count = None
    for index, command in enumerate(commands):
        output = f"{command.get('stdout', '')}\n{command.get('stderr', '')}"
        for line in output.splitlines():
            if not re.search(r"\bskip(?:ped)?\b", line, re.I):
                continue
            short = allowed.fullmatch(line)
            if short is not None:
                if index != 0:
                    raise QualificationFailure("positive-case skip came from an unauthorized command class")
                count = int(short.group(1))
                if count < 1:
                    raise QualificationFailure("positive-case short summary must report a positive count")
                short_counts.append(count)
                continue
            summary = re.fullmatch(r"\s*\d+ passed, (\d+) skipped in \d+(?:\.\d+)?s\s*", line)
            if summary is not None and index == 0 and aggregate_count is None:
                aggregate_count = int(summary.group(1))
                continue
            raise QualificationFailure("conformance output contains an undocumented skip source/reason")
    if not short_counts or aggregate_count is None or sum(short_counts) != aggregate_count:
        raise QualificationFailure("conformance skip lacks matching short and aggregate summaries")


def validate_legacy_report(report: dict, exit_code: int, *, python: str | Path) -> dict:
    """Strictly authenticate report inventory/status/result/reason/count/device."""
    if exit_code != 0:
        raise QualificationFailure(f"legacy qualification child exited {exit_code}")
    if type(report) is not dict or report.get("schema_version") != 1 or report.get("device") != "vk:0":
        raise QualificationFailure("legacy qualification report identity mismatch")
    if type(report.get("schema_version")) is not int:
        raise QualificationFailure("legacy qualification report schema version type mismatch")
    probe = report.get("device_probe")
    if (type(probe) is not dict or probe.get("status") != "available"
            or type(probe.get("exit_code")) is not int or probe.get("exit_code") != 0):
        raise QualificationFailure("legacy qualification device probe is not available")
    gates = report.get("gates")
    if type(gates) is not list or [g.get("name") for g in gates if type(g) is dict] != list(GATES):
        raise QualificationFailure("legacy qualification gate inventory/order mismatch")
    statuses = {"pass": 0, "skip": 0, "fail": 0}
    for gate in gates:
        if type(gate) is not dict or gate.get("status") not in statuses:
            raise QualificationFailure("legacy report has malformed gate status")
        status = gate["status"]
        statuses[status] += 1
        commands = gate.get("commands")
        if type(commands) is not list:
            raise QualificationFailure("legacy gate command inventory is malformed")
        if gate["name"] != "additional_implementation" and not commands:
            raise QualificationFailure(f"legacy gate {gate['name']} has no command results")
        for command in commands:
            if (type(command) is not dict or type(command.get("exit_code")) is not int
                    or command.get("exit_code") != 0 or not command.get("command")):
                raise QualificationFailure(f"legacy gate {gate['name']} contains failed command")
        reason = gate.get("reason")
        if status == "pass":
            if reason is not None:
                raise QualificationFailure(f"passing gate {gate['name']} has a skip reason")
        elif gate["name"] == "additional_implementation":
            if reason != "no additional Vulkan implementation was requested" or commands:
                raise QualificationFailure("unapproved additional-implementation skip")
        elif gate["name"] == "conformance":
            if reason is not None or not commands:
                raise QualificationFailure("conformance skip has unexpected reason/commands")
            _validate_conformance_skip(commands, python)
        else:
            raise QualificationFailure(f"unexpected skipped required gate: {gate['name']}")
    if statuses["fail"] or statuses["skip"] != 2:
        raise QualificationFailure("legacy report has failed gates or wrong permitted skip count")
    counts = report.get("counts")
    if (type(counts) is not dict or counts != statuses
            or any(type(counts.get(name)) is not int for name in statuses)):
        raise QualificationFailure("legacy report counts do not match gate statuses")
    return {"gates": statuses, "allowed_skips": ["conformance: positive case",
                                                    "additional_implementation: no additional Vulkan implementation was requested"]}


def _read_legacy_report(path: Path, exit_code: int, *, python: str | Path) -> dict:
    if not path.is_file() or path.is_symlink():
        raise QualificationFailure(f"missing legacy report: {path}")
    try:
        report = json.loads(path.read_text(), object_pairs_hook=_unique_object)
    except (OSError, json.JSONDecodeError, ValueError) as error:
        raise QualificationFailure(f"malformed legacy report {path}: {error}") from error
    return validate_legacy_report(report, exit_code, python=python)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field {key}")
        result[key] = value
    return result


def _preflight(args, environment=None):
    return _validate_preflight(args, dict(os.environ if environment is None else environment))


def _run_reference(args, artifacts: Path, env: dict[str, str], command_runner, *, owned=None) -> dict:
    output = artifacts / "cpu-reference.json"
    command = [str(args.python), "tools/probe_mnist_training_reference.py",
               "--source-root", str(Path(args.reference_source_root).resolve()), "--output", str(output)]
    result = _run_result(command_runner, command, env, 600, "cpu-reference", owned=owned)
    if _failed_child(result):
        return {"result": result, "error": "CPU reference probe failed"}
    if not output.is_file() or output.is_symlink():
        return {"result": result, "error": "CPU reference child did not create its receipt"}
    try:
        receipt = _validate_reference_receipt(output)
    except Exception as error:
        return {"result": result, "error": f"CPU reference receipt invalid: {error}"}
    result["receipt"] = str(output)
    return {"result": result, "validated": receipt}


def _legacy_command(args, mode_dir: Path, mode: str) -> list[str]:
    build = _require_canonical_build(Path(args.build))
    return [str(args.python), "tools/run_vulkan_qualification.py", "--device", "vk:0",
            "--build", str(build), "--python", str(args.python),
            "--output", str(mode_dir / "qualification.json"), "--artifacts", str(mode_dir / "artifacts"),
            "--shader-toolchain-profile", str(Path(args.shader_toolchain_profile).resolve())]


def qualify(args, *, command_runner=None) -> dict:
    """Qualify in strict order; injected runners are CPU-only non-evidence.

    If a real command leaves unreaped worker authority, the summary is retained
    and the original ContainmentError is raised so the caller keeps that handle.
    """
    owned = _ContainedRunner() if command_runner is None else None
    try:
        summary = _qualify(args, command_runner=command_runner, owned=owned)
    except BaseException:
        # Even an interrupted summary write must not replace live recovery authority.
        error = owned.unreaped_error() if owned is not None else None
        if error is not None:
            raise error  # Raised inside the handler, so the interruption is chained context.
        raise
    if owned is not None:
        error = owned.unreaped_error()
        if error is not None:
            raise error
    return summary


def _qualify(args, *, command_runner, owned) -> dict:
    injected = command_runner is not None
    if command_runner is None:
        command_runner = owned
    artifact_root = None
    summary: dict[str, Any] = {
        "schema_version": 1, "status": "blocked", "commands": [], "attempts": [],
        "legacy_reports": [], "evidence_scope": "cpu-test-injection-non-evidence" if injected else "live",
        "owned_process_state": "not-started", "capture_only": bool(getattr(args, "capture_only", False)),
    }
    try:
        artifact_root = _safe_artifacts(Path(args.artifacts))
        artifact_root.parent.mkdir(parents=True, exist_ok=True)
        artifact_root.mkdir(exist_ok=False)
        # Retain the report before doing preflight that can fail.
        env_base = dict(os.environ)
        preflight = _preflight(args, env_base)
        summary["preflight"] = {k: v for k, v in preflight.items() if k != "reference_sources"}
        summary["reference_source_identity"] = preflight["reference_sources"]
        cpu_env = dict(env_base)
        cpu_env["PYTHONPATH"] = _canonical_pythonpath()
        cpu_env["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
        reference = _run_reference(args, artifact_root, cpu_env, command_runner, owned=owned)
        _record_command(summary, reference["result"], injected=injected)
        if reference.get("error"):
            raise QualificationFailure(reference["error"])
        summary["reference_receipt"] = str(artifact_root / "cpu-reference.json")
        summary["reference_receipt_validation"] = "passed CPU validator; not GPU evidence"

        # No authorization/bypass exists for unresolved outer/inner process ownership.
        if not injected and not summary["capture_only"]:
            summary.update(status="blocked", reason=OWNERSHIP_HOLD,
                           gpu_execution_state="not-started; live ownership hold before GPU/full child")
            _write_summary(artifact_root, summary)
            return summary

        attempts: list[Path] = []
        for stage, mode in (("synthetic", "async"), ("synthetic", "sync"),
                            ("real", "async"), ("real", "sync")):
            attempt = artifact_root / f"{stage}-{mode}"
            command = [str(args.python), "tools/mnist_training_coverage.py", "capture",
                       "--stage", stage, "--mode", mode, "--output", str(attempt),
                       "--reference-receipt", str(artifact_root / "cpu-reference.json")]
            is_real = stage == "real"
            if is_real:
                command += ["--mnist-root", str(Path(args.mnist_root).resolve())]
            env = _child_environment(env_base, args, mode, real=is_real)
            result = _run_result(command_runner, command, env, 600, f"{stage}/{mode}", owned=owned)
            _record_command(summary, result, injected=injected)
            attempt_state = {"stage": stage, "mode": mode, "path": str(attempt),
                             "receipt": str(attempt / "receipt.json"),
                             "fragment": str(attempt / "capture.json"),
                             "child_exit_code": result["exit_code"], "validated": False}
            summary["attempts"].append(attempt_state)
            if _failed_child(result):
                raise QualificationFailure(f"required {stage}/{mode} capture failed")
            if not attempt.is_dir():
                raise QualificationFailure(f"capture child did not create attempt: {attempt}")
            _validate_completed_receipt(attempt, stage, mode)
            attempts.append(attempt)
            attempt_state["validated"] = True

        document = _validate_capture_matrix(attempts)
        _validate_current(document, attempts)
        summary["capture_matrix"] = {"records": len(document["records"]), "current": True,
                                     "publishing": False}

        if summary["capture_only"]:
            reason = "capture-only matrix validated; full qualification and publication remain held"
            if injected:
                reason += "; injected command_runner output is CPU-test-only non-evidence"
            summary.update(status="blocked", reason=reason,
                           qualification_state="blocked; full qualification not run",
                           publication_state="held; no publication attempted")
            _write_summary(artifact_root, summary)
            return summary

        for mode in ("async", "sync"):
            mode_dir = artifact_root / f"full-{mode}"
            mode_dir.mkdir(exist_ok=False)
            env = _child_environment(env_base, args, mode, real=True)
            result = _run_result(command_runner, _legacy_command(args, mode_dir, mode), env, 1800,
                                 f"full/{mode}", owned=owned)
            _record_command(summary, result, injected=injected)
            report_path = mode_dir / "qualification.json"
            report_entry = {"mode": mode, "path": str(report_path),
                            "outer_exit_code": result["exit_code"], "validated": False}
            summary["legacy_reports"].append(report_entry)
            accepted = _read_legacy_report(report_path, result["exit_code"], python=args.python)
            report_entry.update(validation=accepted, validated=True)
            # Every child/full mode is only a fixture under injection.
        # Re-aggregate after the legacy build to recheck source/binary identity.
        final_document = _validate_capture_matrix(attempts)
        _validate_current(final_document, attempts)
        summary.update(status="blocked", reason="injected command_runner is CPU-test-only; outputs are non-evidence",
                       post_legacy_current_validation="passed CPU contract; not GPU evidence",
                       owned_process_state="CPU fixture runner only; no live ownership certificate")
        _write_summary(artifact_root, summary)
        return summary
    except PreflightBlocked as error:
        summary.update(status="blocked", reason=str(error))
    except (QualificationFailure, ValueError, OSError, RuntimeError, TypeError, KeyError) as error:
        summary.update(status="fail", reason=f"{type(error).__name__}: {error}")
    except BaseException as error:
        summary.update(status="fail", reason=f"unexpected {type(error).__name__}: {error}")
        summary["exception"] = {"type": type(error).__name__, "message": str(error)}
    if artifact_root is not None:
        try:
            _write_summary(artifact_root, summary)
        except Exception as error:
            summary["summary_write_error"] = f"{type(error).__name__}: {error}"
    return summary


def main(argv=None) -> int:
    args = parse_args(argv)
    summary = qualify(args)
    print(json.dumps({"status": summary["status"], "reason": summary.get("reason")}, sort_keys=True))
    return 0 if summary["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
