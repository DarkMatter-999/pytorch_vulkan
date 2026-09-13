# Vulkan Phase 3 Roadmap Design

Date: 2026-09-11
Branch: `feat/vulkan-operator-support`

## Goal

Build the first usable data-movement and operator-support layer on top of the
completed Phase 2 Vulkan `PrivateUse1` allocator. Phase 3 will proceed as a
capability ladder: each subphase has a narrow contract, regression tests, and
a verification gate before the next capability is added.

## Existing Foundation

Phase 2 provides:

- Vulkan-backed, device-local allocations through PyTorch's native allocator.
- A registered `PrivateUse1` device guard exposed through the `vk` alias.
- Single-device support for `vk:0`.
- CPU-only configuration isolation and Vulkan availability skips.
- No usable tensor data movement or Vulkan operator dispatch yet.

Phase 3 must preserve these boundaries. It must not claim multi-device
semantics, asynchronous behavior, arbitrary dtype support, or operator support
until those capabilities have their own implementation and tests.

## Staged Capability Ladder

### 3A: Data Movement and Synchronization

3A establishes the transfer and command-completion foundation required by all
later operators. It is divided into independently tested steps.

#### 3A.1: Synchronous Contiguous Float32 Transfers

Support CPU-to-Vulkan and Vulkan-to-CPU transfers for contiguous `float32`
tensors. The public smoke-test paths are:

```python
vk_tensor = cpu_tensor.to("vk")
cpu_round_trip = vk_tensor.to("cpu")
```

Transfers complete before returning. The implementation may need the internal
`copy_` dispatch used by PyTorch's `.to()` path, but it does not claim the full
public `copy_` contract yet.

The memory strategy is hybrid: retain device-local memory for normal Vulkan
tensor storage, use host-visible direct mapping where valid, and fall back to
host-visible staging buffers when direct mapping is unavailable. The choice
must be hidden behind a transfer boundary rather than exposed to tensor users.

Out of scope for 3A.1: nonblocking transfers, strided tensors, dtype
conversion, overlap semantics, cross-device copies, and multi-device support.

Acceptance gate:

- CPU values round-trip through `vk:0` without corruption.
- Tensor shape, dtype, device, and contiguity metadata are correct.
- Transfer completion is observable by the time the API returns.
- Vulkan validation reports no synchronization or lifetime errors.
- Unavailable Vulkan remains an environmental skip; implementation errors fail.

#### 3A.2: Narrow Explicit `copy_`

Add explicit `copy_` support for the same synchronous, contiguous `float32`,
CPU/Vulkan contract. Test both directions and ensure the destination tensor is
updated in place without changing its metadata.

Out of scope remains nonblocking behavior, unsupported layouts and dtypes,
overlap handling, cross-device copies, and multi-device support.

Acceptance gate:

- Supported `copy_` calls update destination values correctly in both
  directions.
- Unsupported combinations fail explicitly rather than silently falling back.
- Existing `.to()` and allocator lifetime tests remain passing.
- Validation remains clean.

#### 3A.3: Broader Copy Capabilities

Expand transfer behavior only when a concrete consumer requires it. Candidate
increments are additional dtypes, strided layouts, nonblocking transfers, and
cross-device copies. Each candidate is separately specified, implemented, and
tested; none is implied by completing 3A.1 or 3A.2.

### 3B: Basic Pointwise Operators

After 3A is stable, add basic pointwise operators incrementally. The initial
candidate set is `add`, `sub`, `mul`, and `relu`, subject to the detailed 3B
design and the capabilities actually needed by the tests.

#### 3B.1a: Narrow Formatter-Compatible Views

Add the minimum metadata-only `aten::view` behavior needed for PyTorch tensor
formatting and `print(vulkan_tensor)`. Initially support only contiguous
`float32` Vulkan tensors on `vk:0` with zero storage offset and formatter-style
flattening/reshape to a one-dimensional view. The result must alias the existing
storage without copying data. General strided, offset, overlapping, arbitrary
reshape, and autograd view semantics remain deferred.

This is a debugging and usability slice, not a claim of general view support.
Its acceptance gate requires natural printing of supported Vulkan tensors,
unchanged source storage, correct alias/lifetime behavior, and explicit errors
for unsupported view forms.

Each operator increment must define:

- supported dtype, shape, layout, and broadcasting behavior;
- output allocation and device placement;
- command recording, submission, and completion behavior;
- unsupported-input error behavior;
- CPU reference comparisons and Vulkan validation coverage.

The first operator slice should prefer a small reusable dispatch/kernel
boundary over a broad registration table. Operators must not expand copy or
multi-device semantics implicitly.

Pointwise operators may proceed after the narrow formatter-compatible view
slice. Full view and reshape semantics receive a separate design and test
gate before operators depend on arbitrary aliasing or layouts.

#### 3B.3: Initial Autograd Boundary

The initial autograd gate covers first-order reverse-mode backward for the six
shipped differentiable operator families: `neg`, `abs`, `relu`, `add`, `sub`,
and `mul`. Tensor/tensor and documented tensor/scalar forms retain the same
metadata, device, synchronization, and no-CPU-fallback restrictions as forward
execution. Unsupported Vulkan operators fail explicitly. Higher-order gradients,
forward-mode AD, and general view replay remain later capability groups.

### 3C: Advanced Operators

Add advanced operators in capability groups rather than as one undifferentiated
milestone. Potential groups include reductions, view/reshape-related behavior,
normalization, convolution, and matrix operations. The actual order will be
chosen from concrete model or test requirements after 3B exposes the operator
boundary.

Every group receives its own design, supported-contract table, correctness
tests against CPU, error tests, and validation run. View and aliasing behavior
must be designed separately from mere metadata operations; no view support is
assumed by the completion of data movement.

### Practical Operator Coverage Target

The backend will not attempt to implement all of PyTorch's operator inventory.
A useful target is approximately 45--75 operator families, which could cover
roughly 80% of operators used by common CNN/MLP models, depending on the
workload. That is not 80% of PyTorch itself; operator usage is highly
concentrated. Coverage should be selected from concrete model requirements and
measured by workload rather than by the total number of Aten schemas.

#### Future View And Reshape Capability

Full `aten::view`/`reshape` support is a later capability group. It must define
legal shape/stride transformations, storage-offset bounds, aliasing and overlap
behavior, reshape copy-vs-view rules, in-place alias behavior, and autograd view
metadata/replay before it is implemented.

## Cross-Phase Constraints

- Keep PyTorch tensor and storage metadata as the source of truth; do not add a
  custom tensor or storage class.
- Keep Vulkan handles opaque to CPU-facing tensor data pointers.
- Preserve the `vk` alias and the single-device `vk:0` boundary until a
  dedicated multi-device design is approved.
- Use synchronous semantics until a nonblocking design specifies streams,
  events, ownership, and lifetime requirements.
- Preserve CPU-only builds without requiring Vulkan or PyTorch discovery.
- Maintain exit-code `77` only for genuine environmental Vulkan unavailability;
  implementation failures must remain visible.
- Keep OpenCL files as reference material and out of the active Vulkan target
  graph.

## Verification Pattern

Each subphase follows the same loop:

1. Define a narrow contract and write failing regression tests.
2. Implement only that contract.
3. Run focused Python and native tests.
4. Run Vulkan validation and CPU-only verification.
5. Review the diff and commit the completed subphase.

The Phase 3 roadmap is complete only when each shipped subphase has passed its
own gate. Deferred capabilities remain documented as unsupported rather than
being inferred from partial behavior.

## Immediate Next Design

The next document will design 3A.1 in detail: transfer entry points, the hybrid
direct/staging memory decision, command submission and synchronization,
allocation ownership, supported tensor checks, and the test matrix.
