#!/usr/bin/env python3
"""Run the ordered Vulkan release qualification gates and write evidence."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
def run_command(command: list[str], cwd: Path, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Run one command without a shell and retain its exact result."""
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
        )
    except subprocess.TimeoutExpired as error:
        def as_text(value: Any) -> str:
            if isinstance(value, bytes):
                return value.decode(errors="replace")
            return value or ""

        return {
            "exit_code": 124,
            "stdout": as_text(error.stdout),
            "stderr": as_text(error.stderr) + "\ncommand timed out after 300 seconds\n",
        }
    except OSError as error:
        return {"exit_code": 127, "stdout": "", "stderr": str(error)}
    return {
        "exit_code": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


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


def device_available(
    build: Path, device: str, python_executable: Path | None = None
) -> dict[str, Any]:
    python = str(python_executable or DEFAULT_PYTHON)
    command = [
        python,
        "-c",
        "import sys, torch, pytorch_vulkan\n"
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


def _command_result(command: list[str], *, environment: dict[str, str] | None = None) -> dict[str, Any]:
    result = run_command(command, ROOT, environment)
    return {"command": command, "environment": _environment_delta(environment), **result}


def _gate(
    name: str,
    commands: list[list[str]],
    *,
    device_dependent: bool = False,
    environment: dict[str, str] | None = None,
    skip_reason: str | None = None,
) -> dict[str, Any]:
    if skip_reason:
        return {
            "name": name,
            "status": "skip",
            "reason": skip_reason,
            "commands": [],
        }
    results = [_command_result(command, environment=environment) for command in commands]
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
    python = str(python_executable or os.environ.get("VULKAN_PYTHON", DEFAULT_PYTHON))
    artifacts = artifacts or report.parent / ".vulkan-qualification-artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
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
    gates.append(_gate("manifest", [[python, "tools/validate_vulkan_capabilities.py"]]))
    gates.append(
        _gate(
            "build",
            [["cmake", "--build", str(build), "-j10"]],
            skip_reason=None if run_build else "build execution disabled",
        )
    )
    gates.append(
        _gate(
            "ctest",
            [["ctest", "--test-dir", str(build), "--output-on-failure"]],
        )
    )
    device_skip = None if available else device_reason
    gates.append(
        _gate(
            "conformance",
            [[python, "-m", "pytest", "-q", "-rs", "tests/python/test_vulkan_conformance.py"]],
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
            [[python, "-m", "pytest", "-q", "-rs", "tests/python/test_vulkan_reliability.py"]],
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
            [python, "-m", "pytest", "-q", "tests/python/test_vulkan_operator_capabilities.py"],
            *[[python, str(path.relative_to(ROOT))] for path in verifiers],
            [python, "-m", "pytest", "-q", "-rs", "tests/python/test_vulkan_conformance.py"],
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
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "vulkan-qualification-report.json"
    )
    parser.add_argument("--additional-device")
    parser.add_argument("--python", type=Path)
    parser.add_argument("--artifacts", type=Path)
    args = parser.parse_args()
    summary = run_qualification(
        args.output,
        device=args.device,
        build=args.build,
        additional_device=args.additional_device,
        python_executable=args.python,
        artifacts=args.artifacts,
    )
    print(json.dumps(summary["counts"], sort_keys=True))
    return 1 if summary["counts"]["fail"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
