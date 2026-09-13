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
  tests/test_vulkan_unavailable.py
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
  tests/python/test_vulkan_allocator.py -q

# Manual clean-extension build and compile-probe check:
PYTHONPATH=build/vulkan .venv/bin/python -m pytest \
  tests/manual_python_extension.py -q
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

Float16 is explicitly deferred. Vulkan allocation, CPU/Vulkan transfer, and
Vulkan operator requests with `torch.float16` fail at the capability boundary
with a `Vulkan float16 support is deferred` error, before Vulkan dispatch or
CPU materialization. CPU-only float16 behavior remains PyTorch behavior.

Before enabling float16, a separate design must specify all of the following:

- PyTorch-compatible storage byte layout and allocation sizing.
- Vulkan 16-bit storage and arithmetic feature requirements and device policy.
- Host/device transfer representation, staging, synchronization, and endianness.
- Scalar promotion and mixed-dtype operator rules.
- Float16 shader generation, arithmetic precision, and ABI/push-constant path.
- Operator registration and forward parity coverage.
- First-order reverse-mode autograd formulas, saved tensors, and gradient dtype/storage.

Multi-device support, general views, and advanced operators are deferred.

## Fixed model slices

The accepted Phase 6 model gate covers PyTorch 2.4's fixed F32 workloads:

- MLP: two `aten::linear` layers with ReLU, shapes `(2, 8) -> (2, 16) -> (2, 4)`.
- CNN: fixed `aten::convolution`, ReLU, and global
  `aten::_adaptive_avg_pool2d`, shapes `(2, 1, 8, 8) -> (2, 4, 1, 1)`.

Every intermediate, output, and first-order gradient is contiguous F32 on
`vk:0`. The implementation dispatches directly through Vulkan and does not
use CPU fallback, payload readback, or `dlprimitives`. CPU transfers in model
tests are explicit comparison boundaries only. Fixed variants use PyTorch 2.4
semantics and compare with the existing floating-point test tolerances.

Float16, general views, strided execution, unsupported model variants,
higher-order gradients, and implicit `fill_`/accumulated-leaf paths remain
explicitly rejected or deferred. Formatter-compatible Double is limited to the
separate formatter contract documented below; it is not general Double support.

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
asynchronous execution, general view execution and unsupported view layouts,
multi-device support, and advanced operators remain deferred; only the
documented Python-number forms are supported.

## Vulkan masked select

`torch.masked_select` supports contiguous, zero-offset F32 inputs and same-shaped
contiguous, zero-offset Vulkan bool masks on `vk:0`. It runs a Vulkan count pass,
reads only a host-visible coherent four-byte count, allocates exact-size F32
output, and runs an ordered Vulkan compaction pass. Tensor and mask payloads are
never read back and unsupported dtypes, devices, layouts, offsets, broadcasts,
and allocation forms are rejected without CPU fallback.

## Vulkan autograd capability

The initial Vulkan autograd boundary supports first-order reverse-mode backward
for six operator families: `neg`, `abs`, `relu`, `add`, `sub`, and `mul`. The
supported paths preserve the existing contiguous `float32`, `vk:0`, synchronous
forward contract and keep gradients on Vulkan:

| Operator | Backward paths |
| --- | --- |
| `neg`, `abs`, `relu` | tensor input |
| `add`, `sub`, `mul` | equal-shape tensor/tensor; documented tensor/Python-number forms |

Unsupported Vulkan operators, overloads, layouts, dtypes, and devices fail with
an explicit backend error before CPU materialization. CPU-only autograd remains
normal PyTorch behavior. Higher-order gradients, forward-mode AD, arbitrary view
replay and general reshape autograd semantics remain deferred; retained-graph
replay is supported within the documented in-place and gradient-accumulation
boundaries.

## Formatter-Compatible Double

The formatter Double tier is gated by the selected Vulkan physical device's
`shaderFloat64` feature. The feature is enabled only when reported as available;
devices without it reject Double allocation and F32-to-Double conversion with an
explicit capability error. Supported Double allocations use native 64-bit
storage (`sizeof(double)`), and the only conversion is a checked, contiguous,
zero-offset Vulkan F32 to Vulkan Double conversion on `vk:0`.

The exact formatter schemas are `abs.default`, `min.default`, `max.default`,
`ceil.default`, `ne.Tensor`, `div.Tensor`, `gt.Scalar`, `lt.Scalar`, and
`_local_scalar_dense.default`. Formatter statistics and intermediate values stay
Vulkan-resident. Only the final scalar value crosses the explicit presentation
transfer used by `_local_scalar_dense`; normal Double CPU payload readback,
hidden materialization, and normal CPU fallback remain unsupported. Double
`add`, `mul`, model operators, arbitrary views, and other unrelated operators
remain rejected. The dedicated shader artifacts are checked by
`vulkan_formatter_double_shader_integrity`.

## Vulkan unary operations

The functional unary operations `torch.neg`, `torch.abs`, and `torch.relu` are
supported for contiguous, strided `torch.float32` tensors on `vk:0`. The
operations allocate a fresh Vulkan output and execute synchronously. In-place
variants remain unsupported. No CPU fallback is provided for
calls involving Vulkan tensors; unsupported Vulkan inputs and overloads are
rejected explicitly rather than staged through CPU.

### Metadata Views And Debug Printing

PyTorch formatting currently reaches `aten::view` through `reshape(-1)`, so
metadata-only `as_strided` calls are supported for `float32` tensors on `vk:0`
when their requested non-negative-stride metadata references a valid
in-allocation range. `as_strided` may describe non-zero offsets, non-contiguous
layouts, overlap, and a different logical element count. Metadata-only `view` and
reshape-alias calls remain limited to compatible contiguous, zero-offset layouts
with equal logical element counts. All supported calls preserve storage metadata
without a copy or dispatch. Pointwise and copy execution of general views,
dtype-changing views, and complete formatting remain unsupported. Boolean
formatter support and later offset-aware and strided execution are deferred.

## Serialization And Multiprocessing

On PyTorch 2.4, generic `torch.save(vulkan_tensor)` is unsafe: the standard
storage path reaches an unsupported `aten::set_.source_Storage` operation for
the opaque Vulkan `DataPtr`. Use the explicit backend API instead:

```python
pytorch_vulkan.save(vulkan_tensor, path)
restored = pytorch_vulkan.load(path)
cpu_restored = pytorch_vulkan.load(path, map_location="cpu")
vulkan_restored = pytorch_vulkan.load(path, map_location="vk")
```

The explicit format materializes Vulkan storage into CPU-owned PyTorch tensors
and never serializes Vulkan handles or allocator state. A generic PyTorch
fallback is `torch.save(vulkan_tensor.cpu(), path)`. Explicit loading defaults
to `vk:0` for payloads containing Vulkan tensors and fails if Vulkan is
unavailable; `map_location="cpu"` forces CPU storage and `map_location="vk"`
requests `vk:0`. Invalid locations and unsupported metadata are rejected.

Multiprocessing transfers use CPU tensors as the process boundary. Materialize
with `vulkan_tensor.cpu()` before putting a tensor on a queue, pipe, or pool
input. `spawn` is the primary start method; each child initializes independent
Vulkan state. Forking after Vulkan initialization is rejected and callers
should use `spawn` or CPU materialization. Vulkan IPC and direct transfer of
Vulkan tensors between processes are not supported. Autograd is not part of
this serialization or multiprocessing contract.

### Manual Python Extension Verification

`tests/manual_python_extension.py` is intentionally excluded from the default
pytest discovery because its two checks each configure and build a fresh
Python extension in an isolated temporary directory. Run it explicitly when
validating a clean extension build or compile-probe integration:

```text
PYTHONPATH=build/vulkan .venv/bin/python -m pytest -q tests/manual_python_extension.py
```

The normal regression command uses the already-built extension and does not
include this manual build check.
