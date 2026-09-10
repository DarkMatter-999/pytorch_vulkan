import os
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vulkan_probe_reports_unavailable_device(tmp_path):
    build_dir = tmp_path / "build"
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT),
            "-B",
            str(build_dir),
            "-DBUILD_VULKAN_PROBE=ON",
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

    environment = os.environ.copy()
    environment["VK_ICD_FILENAMES"] = str(tmp_path / "missing-vulkan-driver.json")
    probe = subprocess.run(
        [str(executables[0])],
        env=environment,
        capture_output=True,
        text=True,
    )
    if probe.returncode == 0:
        pytest.fail("probe unexpectedly found a Vulkan device")
    assert "Could not create Vulkan instance" in probe.stderr
