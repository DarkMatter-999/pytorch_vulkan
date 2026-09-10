# Development Build

Phase 1 targets Linux with a CPU-only project configuration and an optional
standalone Vulkan device probe. The legacy OpenCL sources and `dlprimitives`
submodule remain in the checkout as reference material, but are not entered by
the active root build.

## Environment

Install `uv`, Python 3.12, a C++17 compiler, CMake, Vulkan headers and loader,
and the Vulkan validation layer. Create the project environment with:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  --index-url https://download.pytorch.org/whl/cpu \
  torch==2.4.0
uv pip install --python .venv/bin/python numpy pytest pybind11==2.13.6
```

Activate it before running Python tests:

```bash
source .venv/bin/activate
```

## CPU-Only Build

The default build does not discover Vulkan or OpenCL and produces the CPU-only
project shell:

```bash
cmake -S . -B build/cpu \
  -DBUILD_VULKAN_PROBE=OFF
cmake --build build/cpu
```

## Vulkan Probe

Build the optional probe when Vulkan development files and a suitable device
are available:

```bash
cmake -S . -B build/vulkan \
  -DBUILD_VULKAN_PROBE=ON \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build/vulkan --target vulkan_device_probe
build/vulkan/vulkan_device_probe
build/vulkan/vulkan_device_probe --validation
```

The probe validates Vulkan instance creation, physical-device and compute queue
selection, logical-device creation, command-pool creation, buffer allocation,
host/device transfers, repeated buffer lifetimes, and optional validation.

## Tests

Run the active test suite with:

```bash
pytest -q tests/test_cpu_only_build.py \
  tests/test_vulkan_probe.py \
  tests/test_vulkan_unavailable.py \
  tests/test_python_extension.py
```

Run the CTest probe after configuring with `BUILD_VULKAN_PROBE=ON`:

```bash
ctest --test-dir build/vulkan --output-on-failure
```

The Vulkan hardware tests skip when no suitable device is available. The
unavailable-device test uses `VK_ICD_FILENAMES` to verify a clear failure path.

## Python Extension

The minimal Python status extension is opt-in and is not yet a tensor backend:

```bash
cmake -S . -B build/vulkan \
  -DBUILD_VULKAN_PROBE=ON \
  -DBUILD_PYTHON_EXTENSION=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$($PWD/.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build/vulkan --target pytorch_vulkan_python
PYTHONPATH=build/vulkan \
  .venv/bin/python -c 'import pytorch_vulkan; print(pytorch_vulkan.is_available())'
```

`PrivateUse1` tensor storage, factories, dispatch, and operators are deferred
until the low-level Vulkan lifecycle is complete.
