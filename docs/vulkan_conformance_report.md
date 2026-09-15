# Vulkan Conformance Report

## Gate Decision

**PASS.** The executable registry, matrix declarations, and PrivateUse1 source
inventory have exact supported/rejected/deferred declaration parity. Supported
cases remain Vulkan-resident during execution; unsupported cases reject before
dispatch or transfer. Training-safety work is listed separately below and is not
used to claim coverage beyond the declared conformance set.

## Supported Inventory

Every row is an executable `ConformanceCase`; `declaration_id` is the canonical
ATen schema used for matrix and source parity.

| Case | Family | Declaration ID | Mode | rtol | atol |
| --- | --- | --- | --- | ---: | ---: |
| `unary.neg.float32` | unary | `aten::neg.default` | compute | 1e-5 | 1e-8 |
| `unary.neg.float32.strided` | unary | `aten::neg.default` | compute | 1e-5 | 1e-8 |
| `unary.abs.float32` | unary | `aten::abs.default` | compute | 1e-5 | 1e-8 |
| `unary.relu.float32.empty` | unary | `aten::relu.default` | empty | 1e-5 | 1e-8 |
| `binary.add.float32` | binary | `aten::add.Tensor` | compute | 1e-5 | 1e-8 |
| `binary.sub.float32.strided` | binary | `aten::sub.Tensor` | compute | 1e-5 | 1e-8 |
| `binary.mul.float32.empty` | binary | `aten::mul.Tensor` | empty | 1e-5 | 1e-8 |
| `reduction.sum.dim` | reduction | `aten::sum.dim_IntList` | compute | 1e-5 | 1e-8 |
| `reduction.mean.dim` | reduction | `aten::mean.dim` | compute | 1e-5 | 1e-8 |
| `reduction.sum.keepdim.strided` | reduction | `aten::sum.dim_IntList` | compute | 1e-5 | 1e-8 |
| `reduction.mean.optional-dim` | reduction | `aten::mean.dim` | compute | 1e-5 | 1e-8 |
| `reduction.sum.empty-dim` | reduction | `aten::sum.dim_IntList` | empty | 1e-5 | 1e-8 |
| `reduction.mean.empty-dim` | reduction | `aten::mean.dim` | empty | 0 | 0 |
| `indexing.argmax.dim` | indexing | `aten::argmax.default` | compute | 1e-5 | 1e-8 |
| `indexing.argmax.optional-dim.keepdim` | indexing | `aten::argmax.default` | compute | 1e-5 | 1e-8 |
| `indexing.argmax.strided` | indexing | `aten::argmax.default` | compute | 1e-5 | 1e-8 |
| `view.reshape.float32` | view | `aten::reshape.default` | copy | 1e-5 | 1e-8 |
| `view.as-strided.metadata` | view | `aten::as_strided.default` | metadata | 1e-5 | 1e-8 |
| `view.view.metadata` | view | `aten::view.default` | metadata | 1e-5 | 1e-8 |
| `view.reshape-alias.metadata` | view | `aten::_reshape_alias.default` | metadata | 1e-5 | 1e-8 |
| `linear.forward` | linear | `aten::linear.default` | compute | 1e-5 | 1e-8 |
| `linear.forward.strided` | linear | `aten::linear.default` | compute | 1e-5 | 1e-8 |
| `convolution.forward` | convolution | `aten::convolution.default` | compute | 1e-5 | 1e-8 |
| `convolution.forward.strided` | convolution | `aten::convolution.default` | compute | 1e-5 | 1e-8 |
| `masked-select.bool-mask` | masked-select | `aten::masked_select.default` | compute | 1e-5 | 1e-8 |
| `masked-select.strided-value-view` | masked-select | `aten::masked_select.default` | copy | 1e-5 | 1e-8 |
| `aten._adaptive_avg_pool2d.global` | pooling | `aten::_adaptive_avg_pool2d.default` | compute | 1e-5 | 1e-8 |
| `aten._adaptive_avg_pool2d.global.strided` | pooling | `aten::_adaptive_avg_pool2d.default` | compute | 1e-5 | 1e-8 |

Masked-select accepts positive-stride and non-zero-offset F32 value views by
materializing them with a Vulkan-resident copy before compaction. The shared
lower-level Vulkan contiguous-copy path is the sole owner of that copy's
accounting, so one logical materialization records exactly one Vulkan copy. Its
mask remains same-shaped, contiguous, and zero-offset bool on `vk:0`; count and
compaction consume Vulkan payloads with no CPU fallback.

## Rejected Inventory

| Case | Family | Declaration ID |
| --- | --- | --- |
| `pooling.max.rejected` | pooling | `aten::max_pool2d_with_indices.default` |
| `unary.neg.bool.rejected` | unary | `aten::neg.default` |
| `unary.neg.float16.rejected` | unary | `aten::neg.default` |
| `binary.add.double.rejected` | binary | `aten::add.Tensor` |
| `binary.add.mixed-device.rejected` | binary | `aten::add.Tensor` |
| `binary.add.broadcast.rejected` | binary | `aten::add.Tensor` |
| `reduction.sum.non-contiguous-overlap.rejected` | reduction | `aten::sum.dim_IntList` |
| `comparison.ne.nonzero-offset-input.rejected` | comparison | `aten::ne.Tensor` |
| `unary.neg.invalid-out.rejected` | unary | `aten::neg.out` |
| `unary.neg.cpu-out.rejected` | unary | `aten::neg.out` |
| `pooling.parameters.rejected` | pooling | `aten::_adaptive_avg_pool2d.default` |
| `convolution.shape.rejected` | convolution | `aten::convolution.default` |
| `unary.neg_.unsupported-overload.rejected` | unary | `aten::neg_.default` |

The machine-readable matrix declares supported optimizer arithmetic separately
from the deferred set. The source audit separately covers all deferred registrations and explicit
`reject_*` registrations; the registry IDs are required to be contained in
the union of those matrix sets, while supported and conformance-rejected sets
must match in both directions.

## Measured Counter Policy

Counters are reset after input setup and immediately before each operation.
`compute` requires dispatches > 0 and Vulkan copies = 0. `copy` requires Vulkan
copies > 0; a case may also require compute dispatches when materialization is
followed by Vulkan kernels. `metadata` and `empty` require both counts = 0.
Every operation requires explicit transfers = 0. A final `.cpu()` comparison is
an explicit presentation transfer, not fallback, and occurs after these
assertions.

Measured supported counters (`dispatches, Vulkan copies, explicit transfers`):

```text
unary.neg.float32 (1,0,0); unary.neg.float32.strided (1,0,0); unary.abs.float32 (1,0,0); unary.relu.float32.empty (0,0,0)
binary.add.float32 (1,0,0); binary.sub.float32.strided (1,0,0); binary.mul.float32.empty (0,0,0)
reduction.sum.dim (1,0,0); reduction.mean.dim (1,0,0); reduction.sum.keepdim.strided (1,0,0); reduction.mean.optional-dim (1,0,0)
reduction.sum.empty-dim (0,0,0); reduction.mean.empty-dim (0,0,0)
indexing.argmax.dim (1,0,0); indexing.argmax.optional-dim.keepdim (1,0,0); indexing.argmax.strided (1,0,0)
view.reshape.float32 (0,1,0); view.as-strided.metadata (0,0,0); view.view.metadata (0,0,0); view.reshape-alias.metadata (0,0,0)
linear.forward (1,0,0); linear.forward.strided (1,0,0); convolution.forward (1,0,0); convolution.forward.strided (1,0,0)
masked-select.bool-mask (2,0,0); masked-select.strided-value-view (2,1,0)
aten._adaptive_avg_pool2d.global (1,0,0); aten._adaptive_avg_pool2d.global.strided (1,0,0)
```

## Boundaries and Follow-Up

Intentional unsupported boundaries are float16, bool arithmetic, general Double,
broadcasting, mixed devices, layouts/offsets outside each operator's declared
contract, unsupported in-place and `out=` overloads, unsupported pooling
parameters, variable model shapes, and higher-order/forward-mode AD. These are
explicit rejection contracts, not fallback; masked-select value views use the
declared Vulkan-resident materialization path.

Optimizer safety stress, including repeated queue updates and small-to-large-to-
small allocator transitions, is covered by the optimizer safety gate below. Remaining
follow-up gaps are accumulated-leaf and implicit `fill_` paths, broader training
graphs and MNIST coverage, and performance optimization; these are not implied
by the optimizer safety result.

## Optimizer Safety Gate

The Task 6 safety gate extends the basic optimizer tests with repeated queue and
allocator lifetime coverage. For both supported scalar optimizers, the test
performs 32 updates on a two-element contiguous F32 parameter, then updates a
separate 257-element allocation, and finally returns to another two-element
allocation. Each Vulkan update recorded compute dispatches greater than zero
and zero explicit transfers. CPU/Vulkan parameter and optimizer-state parity
comparisons occur only after those counter assertions and use explicit `.cpu()`
presentation boundaries.

The exact supported baseline remains PyTorch 2.4 scalar, non-capturable SGD and
Adam over contiguous F32 parameters and gradients on `vk:0`. SGD permits
momentum, dampening, and coupled weight decay with Nesterov, maximize, foreach,
and differentiable disabled. Adam permits coupled weight decay with AMSGrad,
maximize, foreach, capturable, differentiable, and fused disabled. Optimizer
parity uses `rtol=1e-5` and `atol=1e-8`; momentum and Adam moving-average state
remain Vulkan-resident, while Adam's scalar `step` metadata remains
host-resident. Nesterov, AMSGrad, maximize, decoupled weight decay, foreach,
fused, differentiable, float16, bfloat16, integer, mixed-dtype, and other
unsupported forms remain outside this branch.

Verification commands and results:

```text
cmake --build build -j2 --target pytorch_vulkan_python       PASS
PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_optimizer.py
                                                            105 passed, 1 skipped
PYTHONPATH=build .venv/bin/python -m pytest -q tests/python
                                                            689 passed, 38 skipped
ctest --test-dir build --output-on-failure                   13/13 passed
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ctest --test-dir build --output-on-failure
                                                            13/13 passed
for verifier in tools/verify_*_spv.py; do .venv/bin/python "$verifier"; done
                                                            PASS
git diff --check                                         PASS
```

The first native CTest invocation found the configured
`vulkan_device_probe` executable missing from `build`; building the existing
target with `cmake --build build -j2 --target vulkan_device_probe` restored the
expected test artifact. The mandated native and validation-layer invocations
then passed without a hang, device reset, allocator lifetime failure, or
synchronization error. Python emitted 73 existing warnings, including Vulkan
manual-seed and fork deprecation warnings; no test failed because of them.

This gate does not claim end-to-end MNIST readiness. Real MNIST shapes, labels,
loss, BatchNorm and broader training graphs are subsequent work. No performance 
or benchmark claim is made.
