import os
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def _configure_and_build_extension(build_dir):
    torch_prefix = subprocess.check_output(
        [str(PYTHON), "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
        text=True,
    ).strip()
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
            "-DCMAKE_PREFIX_PATH=" + torch_prefix,
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


@pytest.fixture
def built_extension(tmp_path):
    build_dir = tmp_path / "build"
    _configure_and_build_extension(build_dir)
    sys.path.insert(0, str(build_dir))
    try:
        yield build_dir
    finally:
        sys.path.remove(str(build_dir))


def test_python_extension_imports_and_reports_vulkan(tmp_path):
    build_dir = tmp_path / "build"
    probe_source = tmp_path / "torch_compile_probe.cpp"
    probe_source.write_text(
        '#include <ATen/ATen.h>\n'
        '#include <c10/core/Allocator.h>\n'
        '#include <c10/core/impl/DeviceGuardImplInterface.h>\n'
        'void pytorch_vulkan_torch_compile_probe() {}\n'
    )
    torch_prefix = subprocess.check_output(
        [str(PYTHON), "-c", "import torch; print(torch.utils.cmake_prefix_path)"],
        text=True,
    ).strip()
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
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DPython3_EXECUTABLE=" + str(PYTHON),
            "-Dpybind11_DIR=" + pybind11_dir,
            "-DCMAKE_PREFIX_PATH=" + torch_prefix,
            "-DPYTORCH_VULKAN_TORCH_COMPILE_PROBE=" + str(probe_source),
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
    compile_commands = (build_dir / "compile_commands.json").read_text()
    assert str(probe_source) in compile_commands

    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(build_dir)
    imported = subprocess.run(
        [
            str(PYTHON),
            "-c",
            "import torch, pytorch_vulkan; "
            "assert torch.device('vk:0').type == 'vk'; "
            "assert isinstance(pytorch_vulkan.is_available(), bool); "
            "assert pytorch_vulkan.device_count() >= 0; "
            "assert torch.vk.is_available() == pytorch_vulkan.is_available(); "
            "assert torch._C._dispatch_has_kernel_for_dispatch_key("
            "'aten::empty.memory_format', 'PrivateUse1')",
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert imported.returncode == 0, imported.stdout + imported.stderr


def test_privateuse1_device_guard_context_and_index_validation(built_extension):
    import torch

    sys.path.insert(0, str(built_extension))
    try:
        import pytorch_vulkan

        assert torch.vk.current_device() == 0
        assert torch.device("vk:0").type == "vk"
        with torch.device("vk:0"):
            assert torch.vk.current_device() == 0
            with torch.device("cpu"):
                assert torch.vk.current_device() == 0
            with torch.device("vk:0"):
                assert torch.vk.current_device() == 0
            assert torch.vk.current_device() == 0
        assert torch.vk.current_device() == 0
        with pytest.raises(RuntimeError, match="only device index 0"):
            torch.vk.set_device(1)
        assert pytorch_vulkan.current_device() == 0
    finally:
        sys.path.remove(str(built_extension))
