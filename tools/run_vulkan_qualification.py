#!/usr/bin/env python3
"""Run the ordered Vulkan release qualification gates and write evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from . import shader_toolchain
    from .qualification_process_supervisor import (
        AuthorityHandle, ContainmentError, TOTAL_CLEANUP_SECONDS, _BootstrapProcess,
        _cancellation_error, _check_sigchld, _coordination_error, _current_cancellation,
        _pidfd_open, _pidfd_signal, _waitid_child, caller_wait_scope,
        check_cancellation, cli_cancellation_scope, require_complete, run_command_core,
    )
except ImportError:  # Direct script invocation.
    import shader_toolchain
    from qualification_process_supervisor import (
        AuthorityHandle, ContainmentError, TOTAL_CLEANUP_SECONDS, _BootstrapProcess,
        _cancellation_error, _check_sigchld, _coordination_error, _current_cancellation,
        _pidfd_open, _pidfd_signal, _waitid_child, caller_wait_scope,
        check_cancellation, cli_cancellation_scope, require_complete, run_command_core,
    )


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"

# The CTest registrations whose COMMAND is a native C++ build target (a bare
# executable name, never a ${...} script invocation). The named ctest gate
# selects exactly these twelve; every other build/CMake registration is a
# Python or shell script covered by an explicit gate below
# (manifest/bootstrap_provenance/conformance/shader_verification/stress/
# python_suite), so the old unsplit CTest run no longer hides the Python lane.
NATIVE_CTEST_TESTS = (
    "vulkan_allocator_test",
    "vulkan_cat_test",
    "vulkan_descriptor_arena_test",
    "vulkan_device_probe",
    "vulkan_dtype_test",
    "vulkan_execution_test",
    "vulkan_multi_output_backward_test",
    "vulkan_pipeline_cache_test",
    "vulkan_serialization_probe",
    "vulkan_shader_registry_test",
    "vulkan_transfer_tensor_test",
    "vulkan_transfer_test",
)
NATIVE_CTEST_REGEX = "^(%s)$" % "|".join(NATIVE_CTEST_TESTS)
CTEST_NAMED_BUDGET_SECONDS = 600


def native_ctest_command(build: Path | str) -> list[str]:
    """Standalone native-only CTest invocation: exactly the 12 CMake C++ registrations."""
    return ["ctest", "--test-dir", str(build), "--output-on-failure",
            "-R", NATIVE_CTEST_REGEX]


def python_suite_command(python_executable: str | os.PathLike) -> list[str]:
    """Standalone wrapper-first full Python suite invocation (former unsplit CTest lane)."""
    return [str(python_executable), "tools/vulkan_wrapper_pytest.py", "-q", "-rs",
            "--durations=30", "tests/python"]

# Legacy process-group helpers, retained verbatim ONLY for the unrelated wrapper
# test helper (tests/python/wrapper_subprocess.py). They never supervise a
# qualification command and are not an ownership fallback for run_command.
def _signal_owned_group(process: subprocess.Popen, sig: int) -> None:
    # start_new_session makes this child's PID its private process-group ID.
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _finish_owned_group(process: subprocess.Popen) -> tuple[Any, Any]:
    """Stop descendants, drain pipes with bounds, and reap the direct child."""
    _signal_owned_group(process, signal.SIGTERM)
    try:
        output = process.communicate(timeout=0.2)
    except subprocess.TimeoutExpired as error:
        output = (error.stdout, error.stderr)
    # Do this even when the leader exited or all pipe holders accepted TERM:
    # a descendant can ignore TERM without retaining either output descriptor.
    _signal_owned_group(process, signal.SIGKILL)
    try:
        output = process.communicate(timeout=1)
    except subprocess.TimeoutExpired as error:
        output = (error.stdout, error.stderr)
        # Escaped pipe holders must not turn cleanup into an unbounded read.
        process.stdout.close()
        process.stderr.close()
    process.wait(timeout=1)
    # SIGKILL is asynchronous. On Linux, wait for owned group members to stop
    # executing; orphan zombies belong to their adopter, not to this runner.
    deadline = time.monotonic() + 1
    while True:
        running = False
        for stat in Path("/proc").glob("[0-9]*/stat"):
            try:
                # comm is arbitrary bytes and may itself contain ') ' or spaces.
                fields = stat.read_bytes().rsplit(b") ", 1)[1].split()
                if int(fields[2]) == process.pid and fields[0] not in {b"Z", b"X"}:
                    running = True
                    break
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                continue
        if not running:
            return output
        if time.monotonic() >= deadline:
            raise RuntimeError(f"owned process group {process.pid} survived cleanup")
        time.sleep(0.01)


def run_command(
    command: list[str], cwd: Path, env: dict[str, str] | None = None,
    *, timeout_seconds: float = 300,
) -> dict[str, Any]:
    """Run one command without a shell inside an isolated ownership worker.

    A normal return means the invocation's whole process tree was reaped.
    Unknown or cancelled ownership raises ContainmentError instead.
    """
    check_cancellation()
    result, certificate = run_command_core(command, cwd, env, timeout_seconds=timeout_seconds)
    require_complete(result, certificate)
    check_cancellation(result=result, certificate=certificate)
    return result


def classify_result(result: dict[str, Any], *, device_dependent: bool) -> str:
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}".lower()
    explicit_pytest_skip = result.get("exit_code") == 0 and (
        re.search(r"(^|\n)\s*skipped\b", output) is not None
        or re.search(r"\b\d+\s+skipped\b", output) is not None
    )
    if device_dependent and (result.get("exit_code") == 77 or explicit_pytest_skip):
        return "skip"
    return "pass" if result.get("exit_code") == 0 else "fail"


def _environment_delta(environment: dict[str, str] | None) -> dict[str, str]:
    if environment is None:
        return {}
    base = os.environ
    return {
        key: value
        for key, value in environment.items()
        if base.get(key) != value
    }


def _shader_environment(environment: dict[str, str] | None = None) -> dict[str, str] | None:
    base = environment if environment is not None else os.environ
    profile = base.get(shader_toolchain.PROFILE_ENV)
    if not profile:
        return environment
    expected = base.get(shader_toolchain.HASH_ENV)
    if not expected:
        raise ValueError("selected shader compiler profile requires its full profile hash")
    selected = shader_toolchain.selected_environment(profile, expected)
    if environment is not None:
        selected.update({k: v for k, v in environment.items() if k != "PATH"})
    return selected


def device_available(
    build: Path, device: str, python_executable: Path | None = None
) -> dict[str, Any]:
    python = str(python_executable or DEFAULT_PYTHON)
    command = [
        python,
        "-c",
        "import pytorch_vulkan, sys, torch\n"
        "device = sys.argv[1]\n"
        "if not pytorch_vulkan.is_available(): raise SystemExit(77)\n"
        "try: torch.ones(1).to(device)\n"
        "except Exception as error:\n"
        " print(f'{type(error).__name__}: {error}', file=sys.stderr)\n"
        " raise SystemExit(77)\n",
        device,
    ]
    if not re.fullmatch(r"vk:\d+", device):
        return {
            "command": command,
            "exit_code": 2,
            "stdout": "",
            "stderr": "",
            "status": "unavailable",
            "reason": "requested device must match vk:<index>",
            "environment": {},
        }
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(build), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment = _shader_environment(environment)
    probe = run_command(
        [
            *command,
        ],
        ROOT,
        environment,
    )
    detail = (probe["stderr"] or probe["stdout"] or "no suitable Vulkan device").strip()
    return {
        "command": command,
        **probe,
        "status": "available" if probe["exit_code"] == 0 else "unavailable",
        "reason": f"{device} is available" if probe["exit_code"] == 0 else detail,
        "environment": _environment_delta(environment),
    }


def _command_result(
    command: list[str], *, environment: dict[str, str] | None = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    environment = _shader_environment(environment)
    result = run_command(command, ROOT, environment, timeout_seconds=timeout_seconds)
    return {"command": command, "environment": _environment_delta(environment), **result}


def _gate(
    name: str,
    commands: list[list[str]],
    *,
    device_dependent: bool = False,
    environment: dict[str, str] | None = None,
    skip_reason: str | None = None,
    timeout_seconds: float = 300,
) -> dict[str, Any]:
    if skip_reason:
        return {
            "name": name,
            "status": "skip",
            "reason": skip_reason,
            "commands": [],
        }
    results = [
        _command_result(command, environment=environment, timeout_seconds=timeout_seconds)
        for command in commands
    ]
    statuses = [classify_result(result, device_dependent=device_dependent) for result in results]
    status = "fail" if "fail" in statuses else "skip" if "skip" in statuses else "pass"
    return {"name": name, "status": status, "commands": results}


def run_qualification(
    report: Path,
    *,
    device: str = "vk:0",
    build: Path = ROOT / "build",
    run_build: bool = True,
    additional_device: str | None = None,
    python_executable: Path | None = None,
    artifacts: Path | None = None,
) -> dict[str, Any]:
    # Validate before any device probe or gate; provenance is separate from
    # historical shader manifests and does not rebind captured source hashes.
    selected_environment = _shader_environment()
    shader_provenance = None
    if selected_environment is not None:
        shader_provenance = shader_toolchain.profile_provenance(
            selected_environment[shader_toolchain.PROFILE_ENV],
            selected_environment[shader_toolchain.HASH_ENV])
    python = str(python_executable or os.environ.get("VULKAN_PYTHON", DEFAULT_PYTHON))
    artifacts = artifacts or report.parent / ".vulkan-qualification-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    if shader_provenance:
        (artifacts / "shader-toolchain.json").write_text(
            json.dumps(shader_provenance, indent=2, sort_keys=True) + "\n")
    pythonpath = os.pathsep.join([str(build), str(ROOT)]).rstrip(os.pathsep)
    test_environment = os.environ.copy()
    test_environment["PYTHONPATH"] = pythonpath
    test_environment["PYTORCH_VULKAN_DEVICE"] = device
    test_environment["VULKAN_DEVICE"] = device
    device_probe = device_available(build, device, Path(python))
    available = device_probe["status"] == "available"
    device_reason = device_probe["reason"]

    verifiers = sorted(ROOT.glob("tools/verify_*_spv.py"))
    gates: list[dict[str, Any]] = []
    gates.append(
        _gate(
            "manifest",
            [
                [python, "tools/validate_vulkan_capabilities.py"],
                [python, "tools/validate_vulkan_composed_matmul.py"],
                [
                    python,
                    "tools/vulkan_workload_coverage.py",
                    "validate",
                    "docs/vulkan_workload_coverage.json",
                ],
            ],
        )
    )
    gates.append(
        _gate(
            "build",
            [["cmake", "--build", str(build), "-j10"]],
            skip_reason=None if run_build else "build execution disabled",
        )
    )
    gates.append(_gate(
        "bootstrap_provenance",
        [[python, "tools/validate_vulkan_bootstrap_provenance.py"]],
    ))
    device_skip = None if available else device_reason
    gates.append(
        _gate(
            "ctest",
            [native_ctest_command(build)],
            timeout_seconds=CTEST_NAMED_BUDGET_SECONDS,
        )
    )
    # Distinct Python lane, scheduled sequentially after the native lane. It
    # replaces the duplicate `vulkan_python_suite` CTest entry and runs the
    # wrapper-first whole-directory suite with per-test-phase duration
    # reporting and no xdist/concurrency. It is never dropped: without a
    # device it is an explicit recorded skip.
    gates.append(
        _gate(
            "python_suite",
            [python_suite_command(python)],
            device_dependent=True,
            environment=test_environment,
            skip_reason=device_skip,
            timeout_seconds=CTEST_NAMED_BUDGET_SECONDS,
        )
    )
    conformance_commands = [
        [python, "tools/vulkan_wrapper_pytest.py", "-q", "-rs", "tests/python/test_vulkan_conformance.py"]
    ]
    if device == "vk:0":
        conformance_commands.append(
            [
                python,
                "tools/vulkan_wrapper_pytest.py",
                "-q",
                "-rs",
                "tests/python/test_vulkan_workload_conformance.py::test_stock_sgd_executes_three_steps_in_both_reset_modes",
                "tests/python/test_vulkan_workload_conformance.py::test_parameter_hvp_executes_with_named_live_history",
                "tests/python/test_vulkan_workload_conformance.py::test_matrix_stock_sgd_executes_three_steps_in_both_reset_modes",
                "tests/python/test_vulkan_workload_conformance.py::test_matrix_parameter_hvp_executes_with_named_live_history",
            ]
        )
        conformance_commands.append([
            python, "tools/vulkan_wrapper_pytest.py", "-q", "-rs",
            "tests/python/test_vulkan_linear.py::test_stock_linear_each_declared_rank",
        ])
        conformance_commands.append([
            python, "tools/vulkan_wrapper_pytest.py", "-q", "-rs",
            "tests/python/test_vulkan_composed_matmul.py::test_composed_model",
        ])
    gates.append(
        _gate(
            "conformance",
            conformance_commands,
            device_dependent=True,
            environment=test_environment,
            skip_reason=device_skip,
        )
    )
    gates.append(
        _gate(
            "shader_verification",
            [[python, str(path.relative_to(ROOT))] for path in verifiers],
        )
    )
    validation_environment = test_environment | {"VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation"}
    gates.append(
        _gate(
            "validation_layer",
            [["ctest", "--test-dir", str(build), "--output-on-failure", "-R",
              "vulkan_supported_workload_validation"]],
            device_dependent=True,
            environment=validation_environment,
            skip_reason=device_skip,
        )
    )
    gates.append(
        _gate(
            "stress",
            [[python, "tools/vulkan_wrapper_pytest.py", "-q", "-rs", "tests/python/test_vulkan_reliability.py"]],
            device_dependent=True,
            environment=test_environment,
            skip_reason=device_skip,
        )
    )
    gates.append(
        _gate(
            "timing",
            [[python, "tools/vulkan_training_benchmark.py", "--device", device, "--workload", "both", "--mode", "both",
              "--warmups", "2", "--repetitions", "5", "--cpu-intraop-threads", "1",
              "--cpu-interop-threads", "1"]],
            device_dependent=True,
            environment=test_environment,
            skip_reason=device_skip,
        )
    )
    gates.append(
        _gate(
            "benchmark",
            [[python, "tools/vulkan_gemm_benchmark.py", "--device", device]],
            device_dependent=True,
            environment=test_environment,
            skip_reason=device_skip,
        )
    )
    environment_output = artifacts / "environment.json"
    gates.append(
        _gate(
            "environment",
            [[python, "tools/record_vulkan_environment.py", "--output", str(environment_output)]],
        )
    )
    if additional_device is None:
        additional_gate = _gate(
            "additional_implementation",
            [],
            skip_reason="no additional Vulkan implementation was requested",
        )
    else:
        additional_probe = device_available(build, additional_device, Path(python))
        additional_environment = test_environment | {
            "PYTORCH_VULKAN_DEVICE": additional_device,
            "VULKAN_DEVICE": additional_device,
        }
        additional_commands = [
            ["cmake", "--build", str(build), "-j10"],
            [python, "tools/validate_vulkan_capabilities.py"],
            [python, "tools/vulkan_wrapper_pytest.py", "-q", "tests/python/test_vulkan_operator_capabilities.py"],
            *[[python, str(path.relative_to(ROOT))] for path in verifiers],
            [
                python,
                "tools/vulkan_wrapper_pytest.py",
                "-q",
                "-rs",
                "tests/python/test_vulkan_conformance.py",
            ],
        ]
        additional_gate = _gate(
            "additional_implementation",
            additional_commands,
            device_dependent=True,
            environment=additional_environment,
            skip_reason=(
                None if additional_probe["status"] == "available" else additional_probe["reason"]
            ),
        )
        additional_gate["device_probe"] = additional_probe
    gates.append(additional_gate)

    summary = {
        "schema_version": 1,
        "entry_point": "wrapper-first",
        "device": device,
        "files": [
            "tools/run_vulkan_qualification.py",
            "CMakeLists.txt",
            "tests/python/test_vulkan_qualification.py",
            str(report.relative_to(ROOT)) if report.is_relative_to(ROOT) else str(report),
        ],
        "scope": {
            "platform": "Linux",
            "dtype": "float32",
            "supported_workload_shapes": ["small", "medium", "large", "skinny", "irregular"],
            "future_milestones": ["faster-than-CPU CNN", "faster-than-CPU transformer"],
        },
        "methodology": {
            "timing": "host wall-clock and Vulkan GPU timestamp records; warmups excluded",
            "device_loss": "reliability and timing gates exercise quarantine/rejection contracts",
            "exit_codes": "each subprocess result retains its individual exit code and output",
        },
        "device_probe": device_probe,
        "repository": _command_result(["git", "rev-parse", "HEAD"]),
        "gates": gates,
        "counts": {
            status: sum(gate["status"] == status for gate in gates)
            for status in ("pass", "skip", "fail")
        },
        "concerns": [
            "No additional Vulkan implementation was available or requested; portability is not qualified.",
            "Performance comparisons are timing evidence only and do not establish future CNN or transformer milestones.",
        ],
    }
    if shader_provenance:
        summary["shader_toolchain"] = shader_provenance
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _profile_reentry(command: list[str], environment: dict[str, str]) -> int:
    """Run the profile re-entry CLI as its sole exact-PID waiter.

    It stays in the caller's process group on purpose. It coordinates one known
    CLI child; the child's own scope and workers own any command trees. Every
    uncertain exit keeps the exact launch object in ContainmentError.authority.
    """
    _check_sigchld()  # Same audited wait policy as the core; blocks before any spawn.
    process = _BootstrapProcess.__new__(_BootstrapProcess)
    pidfd = -1
    authenticated = False  # Only an authenticated pidfd may ever signal the child.
    terminated_at = None
    try:
        _BootstrapProcess.__init__(process, command, env=environment, close_fds=True)
        pidfd = _pidfd_open(process.pid)
        _waitid_child(pidfd)  # WNOWAIT: the pinned identity is waitable before we rely on it.
        authenticated = True
        while True:
            pid, status = os.waitpid(process.pid, os.WNOHANG)
            if pid:
                process.returncode = os.waitstatus_to_exitcode(status)
                check_cancellation()
                return process.returncode
            cancellation = _current_cancellation.get()
            now = time.monotonic()
            if cancellation is not None and cancellation.reason is not None:
                if terminated_at is None:
                    _pidfd_signal(pidfd, signal.SIGTERM)
                    terminated_at = now
                elif now >= terminated_at + TOTAL_CLEANUP_SECONDS:
                    error = _cancellation_error(cancellation.reason, None, None)
                    error.authority = AuthorityHandle(pidfd, process)
                    pidfd = -1
                    raise error
            time.sleep(0.01)
    except ContainmentError:
        raise
    except BaseException as caught:
        if not getattr(process, "_child_created", False) or process.returncode is not None:
            raise _coordination_error("profile re-entry failed", caught) from caught
        authority = AuthorityHandle(pidfd, process)
        pidfd = -1
        now = time.monotonic()
        # An existing cancellation bound is retained, never restarted.
        deadline = (terminated_at if terminated_at is not None else now) + TOTAL_CLEANUP_SECONDS
        try:
            if authenticated and terminated_at is None:
                _pidfd_signal(authority.pidfd, signal.SIGTERM)  # Identity-safe, never a raw PID.
        except BaseException:
            pass
        while True:
            try:
                authority.observe_until(deadline)
                break
            except OSError:
                break
            except BaseException:
                if time.monotonic() >= deadline:
                    break
        if authority.reaped:
            authority.close()
            authority = None
        raise _coordination_error("profile re-entry failed", caught, authority) from caught
    finally:
        if pidfd >= 0:
            os.close(pidfd)


def main() -> int:
    try:
        with cli_cancellation_scope():
            return _main()
    except ContainmentError as error:
        print(f"qualification containment failure: {error}", file=sys.stderr)
        return 3


def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "vulkan-qualification-report.json"
    )
    parser.add_argument("--additional-device")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--artifacts", type=Path)
    parser.add_argument("--shader-toolchain-profile", type=Path)
    args = parser.parse_args()
    if args.shader_toolchain_profile:
        # Re-enter with compiler-only PATH selection, not a global Python loader
        # override. All existing API/gate environment deltas remain intact.
        command = [sys.executable, str(Path(__file__).resolve()), "--device", args.device,
                   "--build", str(args.build), "--output", str(args.output)]
        for flag, value in (("--additional-device", args.additional_device),
                            ("--python", args.python), ("--artifacts", args.artifacts)):
            if value is not None:
                command.extend([flag, str(value)])
        return _profile_reentry(command, shader_toolchain.selected_environment(args.shader_toolchain_profile))
    summary = run_qualification(
        args.output,
        device=args.device,
        build=args.build,
        additional_device=args.additional_device,
        python_executable=args.python,
        artifacts=args.artifacts,
    )
    check_cancellation()
    print(json.dumps(summary["counts"], sort_keys=True))
    return 1 if summary["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
