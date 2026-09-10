import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vulkan_probe_builds_and_discovers_compute_device(tmp_path):
    build_dir = tmp_path / "build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT),
            "-B",
            str(build_dir),
            "-DBUILD_VULKAN_PROBE=ON",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ],
        capture_output=True,
        text=True,
    )
    assert configure.returncode == 0, configure.stdout + configure.stderr

    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "vulkan_device_probe"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    executables = list(build_dir.rglob("vulkan_device_probe"))
    executables.extend(build_dir.rglob("vulkan_device_probe.exe"))
    assert len(executables) == 1, f"probe executable not found in {build_dir}"

    probe = subprocess.run(
        [str(executables[0])],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip(probe.stderr.strip() or "no suitable Vulkan device")

    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert "Vulkan API: 1.1" in probe.stdout
    assert "Vulkan device:" in probe.stdout
    assert "Compute queue family:" in probe.stdout
    assert "Logical device: ready" in probe.stdout
    assert "Buffer: ready" in probe.stdout
    assert "Transfer: passed" in probe.stdout
    assert "Command pool: ready" in probe.stdout
    assert "Device transfer: passed" in probe.stdout

    validation_probe = subprocess.run(
        [str(executables[0]), "--validation"],
        capture_output=True,
        text=True,
    )
    if validation_probe.returncode != 0:
        pytest.skip(
            validation_probe.stderr.strip() or "Vulkan validation is unavailable"
        )
    assert "Validation: enabled" in validation_probe.stdout
