import os
import subprocess
from pathlib import Path


def test_cpu_only_configuration_does_not_build_opencl_backend(tmp_path):
    build_dir = tmp_path / "build"
    environment = os.environ.copy()
    environment.pop("CMAKE_PREFIX_PATH", None)

    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(Path(__file__).resolve().parents[1]),
            "-B",
            str(build_dir),
            "-DBUILD_VULKAN_PROBE=OFF",
            "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )

    assert configure.returncode == 0, configure.stdout + configure.stderr
    assert "OpenCL" not in configure.stdout
    assert "OpenCL" not in configure.stderr
    cache = (build_dir / "CMakeCache.txt").read_text()
    assert "OpenCL" not in cache
    assert "OCL_" not in cache

    build = subprocess.run(
        ["cmake", "--build", str(build_dir)],
        capture_output=True,
        text=True,
    )

    assert build.returncode == 0, build.stdout + build.stderr
    assert "pt_ocl" not in build.stdout
    assert "CLTensor.cpp" not in build.stdout

    targets = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "help"],
        capture_output=True,
        text=True,
    )
    assert targets.returncode == 0, targets.stdout + targets.stderr
    assert "pt_ocl" not in targets.stdout
    assert "dlprim_core" not in targets.stdout
