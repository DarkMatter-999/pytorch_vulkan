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

Historical verification commands and results from the original optimizer gate
(not the current branch state):

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

For that original gate, the first native CTest invocation found the configured
`vulkan_device_probe` executable missing from `build`; building the existing
target with `cmake --build build -j2 --target vulkan_device_probe` restored the
expected test artifact. The mandated native and validation-layer invocations
then passed without a hang, device reset, allocator lifetime failure, or
synchronization error. Python emitted 73 existing warnings, including Vulkan
manual-seed and fork deprecation warnings. These results are historical and do
not claim that the current full Python suite is green; current Task 5 results
are recorded below.

This gate does not claim end-to-end MNIST readiness. Real MNIST shapes, labels,
loss, BatchNorm and broader training graphs are subsequent work. No performance 
or benchmark claim is made.

## Phase 6A Benchmark and Validation

The resident benchmark compares the existing synchronous path with the internal
step-scoped path for the fixed MLP (batch 3, 100 SGD steps) and synthetic
MNIST-shaped MLP (batch 512, 10 SGD steps). Inputs and one-hot targets are
uploaded before timing. Both modes recorded zero timed explicit transfers and
zero Vulkan copies; step scope recorded one submitted/completed/waited command
per step (100 and 10), while synchronous mode recorded one per dispatch (3292
and 322). The measured run showed lower wall time in step scope, and the
instrumented allocation, recording, submit/wait, and compute categories were
nonzero. The MNIST-shaped loss became `NaN`; this is a diagnostic observation,
not a performance, convergence, or real-MNIST claim.

The complete Phase 6A validation matrix was run. Native CTest and
validation-layer CTest passed, shader verifiers passed, and `git diff --check`
produced no output. The full Python suite is explicitly not green: 14 known
standalone in-place contract conflicts fail because those operations are
intentionally rejected outside the training scope. These tests were neither
masked nor rewritten.

## Training Validation

The verified fixed MLP training contract is `Sequential(Linear(8,16), ReLU,
Linear(16,4))`, with contiguous F32 `(batch,8)` inputs and `(batch,4)` targets.
The loss is the scalar summed squared error `((output - target) ** 2).sum()`.
The verified flattened MNIST-shaped contract is
`Sequential(Flatten(start_dim=1), Linear(784,32), ReLU, Linear(32,10))`, with
contiguous F32 `(batch,1,28,28)` inputs and same-shaped contiguous F32
`(batch,10)` targets. Runtime accepts arbitrary same-shaped F32 targets for the
summed squared-error loss; the synthetic fixture generates and asserts one-hot
targets, without runtime one-hot semantic validation. Both contracts use only
scalar SGD or Adam.

Training requires GPU forward and first-order backward execution. Parameters,
inputs, targets, outputs, loss, gradients, and optimizer tensor state remain
resident on `vk:0`; Adam's non-capturable scalar `step` metadata remains on the
host. Each Python training step uses the internal
`begin_training_step()`/`end_training_step()` scope, records all supported
forward, backward, and optimizer work into one command buffer, and completes
exactly one submit/wait boundary. Counters are reset per operation/step.
Training requires compute dispatches > 0, one completed submission, and
explicit transfers = 0. Retained descriptor pools and metadata allocations
are retired after completion; a failure closes/cancels the recording scope so
the context cannot remain recording. A final `.cpu()` is an explicit
presentation transfer after the counter assertion, not fallback. The
flattened MNIST-shaped parity check is exactly two optimizer steps at
`rtol=1e-4`, `atol=1e-4`; it is not a long-run convergence or performance claim.

Phase 6A does not expose a future, stream, or asynchronous tensor to Python.
Fusion, graph capture, and multi-queue execution remain deferred. Timing is
categorized as allocation, command recording, submit/wait, compute, and total
elapsed time; these measurements are diagnostic only and make no performance
claim.

Explicit exclusions are real-dataset MNIST readiness, cross-entropy, Long
labels, convolutional training, BatchNorm, arbitrary shapes or model graphs,
higher-order and forward-mode AD, unsupported optimizer options, and benchmark
or performance support.

Task 6 verification commands and results:

```text
PYTHONPATH=build .venv/bin/python -m pytest -q tests/python
                                                               714 passed, 14 failed, 38 skipped, 103 warnings
ctest --test-dir build --output-on-failure                    14/14 passed
VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation ctest --test-dir build --output-on-failure
                                                               14/14 passed
for verifier in tools/verify_*_spv.py; do .venv/bin/python "$verifier"; done
                                                               PASS (all shader verifiers)
git diff --check                                               PASS
```

The full Python suite is not green. The separate 14-test standalone in-place
contract conflict exercises operations intentionally rejected outside the
training scope; those failures are distinct from the passing Phase 6A training
module and parity coverage. The warnings are existing Vulkan manual-seed,
fork-deprecation, and related runtime warnings; none caused a test failure.
