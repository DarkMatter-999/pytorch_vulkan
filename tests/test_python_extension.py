import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_python_extension_imports_and_reports_vulkan(tmp_path):
    build_dir = tmp_path / "build"
    pybind11_dir = subprocess.check_output(
        [str(PYTHON), "-m", "pybind11", "--cmakedir"], text=True
    ).strip()
    configure = subprocess.run(
        [
            "cmake",
            "-S",
            str(PROJECT_ROOT),
            "-B",
            str(build_dir),
            "-DBUILD_PYTHON_EXTENSION=ON",
            "-DPython3_EXECUTABLE=" + str(PYTHON),
            "-Dpybind11_DIR=" + pybind11_dir,
        ],
        capture_output=True,
        text=True,
    )
    assert configure.returncode == 0, configure.stdout + configure.stderr

    build = subprocess.run(
        ["cmake", "--build", str(build_dir), "--target", "pytorch_vulkan_python"],
        capture_output=True,
        text=True,
    )
    assert build.returncode == 0, build.stdout + build.stderr

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(build_dir)
    imported = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import torch, pytorch_vulkan; "
            "assert torch.device('vulkan:0').type == 'vulkan'; "
            "assert isinstance(pytorch_vulkan.is_available(), bool); "
            "assert pytorch_vulkan.device_count() >= 0; "
            "assert torch.vulkan.is_available() == pytorch_vulkan.is_available()",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr
