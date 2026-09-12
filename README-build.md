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

Build the opt-in Python extension and its PrivateUse1 allocator together:

```bash
cmake -S . -B build/vulkan \
  -DBUILD_VULKAN_PROBE=ON \
  -DBUILD_PYTHON_EXTENSION=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$($PWD/.venv/bin/python -m pybind11 --cmakedir)" \
  -DCMAKE_PREFIX_PATH="$($PWD/.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/vulkan --target pytorch_vulkan_python
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation \
PYTHONPATH=build/vulkan .venv/bin/python -m pytest \
  tests/python/test_vulkan_allocator.py tests/test_python_extension.py -q
```

The custom PrivateUse1 device name is `vk`; Vulkan API and runtime terminology
remains unchanged. The allocation contract covers `torch.empty` metadata and
tensor lifetimes. The public `Tensor.copy_` transfer contract is synchronous
and in-place for contiguous `float32` tensors between CPU and `vk:0`: it
mutates and returns the destination tensor while preserving its metadata.
Nonblocking copies, dtype conversion, strided tensors, Vulkan-to-Vulkan copies,
and other unsupported forms are rejected explicitly. The supported transfer
smoke test is a CPU→Vulkan→CPU round trip:

```bash
cmake -S . -B build/vulkan \
  -DBUILD_VULKAN_PROBE=ON \
  -DBUILD_PYTHON_EXTENSION=ON \
  -DPython3_EXECUTABLE="$PWD/.venv/bin/python" \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)" \
  -DCMAKE_PREFIX_PATH="$(.venv/bin/python -c 'import torch; print(torch.utils.cmake_prefix_path)')"
cmake --build build/vulkan --target pytorch_vulkan_python
PYTHONPATH=build/vulkan .venv/bin/python -m pytest \
  tests/python/test_vulkan_transfer.py -q
```

Multi-device support, views, autograd, and advanced operators are deferred.

## Vulkan pointwise scalar operations

The supported public functional scalar forms are `torch.add(tensor, scalar)` and
`torch.add(scalar, tensor)` (with `alpha=1`), `torch.sub(tensor, scalar)`,
`torch.sub(scalar, tensor)`, `torch.mul(tensor, scalar)`, and
`torch.mul(scalar, tensor)`. The corresponding `+`, `-`, and `*` operators use
the same dispatch where Python selects them. The tensor must be a contiguous,
strided `torch.float32` tensor on `vk:0`; scalar-tensor operands are not
accepted. Scalar values are converted once to finite, representable float32
values and placed in the Vulkan compute push constants—there is no scalar
buffer and no CPU staging or CPU fallback.

Scalar operations are synchronous: the returned tensor is ready when the call
returns, inputs are unchanged, and functional output storage is newly allocated
on the same Vulkan device with the input shape, dtype, and layout. Zero-element
tensors return an empty output with preserved metadata without a compute
dispatch. The supported tensor-tensor forms also require equal shapes and use
the same synchronous, newly allocated Vulkan path.

## Vulkan pointwise `out=` operations

The supported `.out` registrations are unary `neg.out`, `abs.out`, and
`relu.out`; binary `add.out`, `sub.out`, and `mul.out`; scalar `add.Scalar_out`,
`sub.Scalar_out`, `rsub.Scalar_out`, and `mul.Scalar_out`; and their corresponding
Python `torch.*(..., out=...)` forms. Binary `add` and `sub` accept only
`alpha=1`. Scalar forms accept the same Python-number operand positions and
`alpha=1` as the functional scalar operations above.

Every `.out` input and output must be a contiguous, strided `torch.float32`
tensor on `vk:0`, with zero storage offset. Binary inputs must have equal
shapes; broadcasting, dtype promotion, and scalar-tensor operands are not
supported. The operation returns the exact `out` object. A Vulkan output may
be resized to the requested shape while retaining Vulkan storage provenance;
zero-element outputs are supported without a compute dispatch. Inputs are
unchanged except for an exact full-tensor alias explicitly supplied as `out`.
Unary and tensor-tensor exact aliases are supported; tensor-scalar aliases are
supported for both operand positions for `add`, `mul`, and `rsub`, while
`sub.Scalar_out` remains tensor-left. Partial or uncertain overlap and
internally overlapping outputs are rejected.

`.out` operations involving Vulkan tensors never fall back to CPU: unsupported
devices, metadata, layouts, dtypes, shapes, overlap, scalar values, or
parameters are rejected explicitly. In-place variants and additional dtypes
remain unsupported.

CPU-only arithmetic remains normal PyTorch behavior because the PrivateUse1
registrations cannot intercept CPU dispatch. “No CPU fallback” means an
operation involving Vulkan tensors never silently stages through CPU. Mixed
CPU/Vulkan tensors, ordinary CPU scalar tensors (including zero-dimensional
user tensors) in either position, zero-dimensional Vulkan tensors, broadcasting
or unequal shapes, non-contiguous tensors, non-zero storage offsets,
non-`float32` dtypes, non-`vk:0` devices, non-unit `alpha`, and
in-place variants are rejected explicitly. Complex, non-finite, float32-range
overflowing, and float32-underflowing Python scalar values are also rejected.

Broadcasting, dtype promotion, scalar-tensor semantics, scalar buffers,
asynchronous execution, views, autograd, multi-device support, and advanced
operators remain deferred; only the documented Python-number forms are
supported.

## Vulkan unary operations

The functional unary operations `torch.neg`, `torch.abs`, and `torch.relu` are
supported for contiguous, strided `torch.float32` tensors on `vk:0`. The
operations allocate a fresh Vulkan output and execute synchronously. In-place
variants remain unsupported. No CPU fallback is provided for
calls involving Vulkan tensors; unsupported Vulkan inputs and overloads are
rejected explicitly rather than staged through CPU.

### Debug Printing And Future Views

PyTorch formatting currently reaches `aten::view` through `reshape(-1)`, so
natural `print(vulkan_tensor)` support is tracked as a separate narrow usability
slice. That slice will support only metadata-only views of contiguous zero-offset
`float32` tensors on `vk:0`, preserving storage aliasing without a copy. General
strided, offset, overlapping, arbitrary reshape, and autograd view semantics
remain deferred until a dedicated view/reshape design covers their storage,
aliasing, and replay requirements.
