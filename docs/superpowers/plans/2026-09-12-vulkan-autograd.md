# Vulkan Autograd Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add first-order reverse-mode autograd for the six shipped Vulkan pointwise operators while preserving the narrow `float32`/`vk:0` contract and strict unsupported-operation errors.

**Architecture:** Keep the existing Vulkan kernels as raw forward implementations and add a small C++ autograd boundary that saves only the forward tensors or metadata required by each gradient formula. The boundary must redispatch raw kernels below autograd to avoid recursion, return Vulkan gradients, and leave `out=` and in-place variants unsupported. Future differentiable operators must add forward and backward support together.

**Tech Stack:** C++17, PyTorch 2.4 C++ autograd API, PrivateUse1/AutogradPrivateUse1 dispatch, Vulkan compute kernels, pybind11 extension, pytest, CTest.

## Global Constraints

- Support only contiguous `float32` tensors on the single `vk:0` device and the documented synchronous overloads.
- Implement first-order reverse-mode differentiation only; defer higher-order gradients, forward-mode AD, and general view replay.
- Unsupported Vulkan operators and overloads must raise explicit errors; never materialize Vulkan inputs on CPU as a fallback.
- Do not add a custom Tensor or Storage implementation, generic CPU fallback, Vulkan IPC, multi-device behavior, or cross-process autograd.
- Preserve existing `out=` and in-place rejection behavior; these forms are not differentiable.
- Preserve CPU-only PyTorch dispatch and existing device, copy, view, serialization, multiprocessing, native, and validation tests.
- User creates commits; workers do not commit or push.

## File Map

- Modify `src/vulkan/operators/unary.cpp`: split raw unary dispatch from autograd-aware functional wrappers and register the three unary forward paths without changing `out=` or in-place paths.
- Modify `src/vulkan/operators/binary.h`: expose raw tensor/tensor and tensor/scalar entry points needed by autograd wrappers without exposing Vulkan internals.
- Modify `src/vulkan/operators/add.cpp`: add-aware functional wrapper and tensor/tensor plus tensor/scalar gradient plumbing.
- Modify `src/vulkan/operators/sub.cpp`: subtraction and reverse-subtraction functional wrappers and gradient plumbing.
- Modify `src/vulkan/operators/mul.cpp`: multiplication functional wrapper and gradient plumbing.
- Create `src/vulkan/operators/autograd.h`: small declarations for reusable unary and binary autograd wrapper helpers and their raw-kernel callback types.
- Create `src/vulkan/operators/autograd.cpp`: custom `torch::autograd::Function` implementations, saved-variable handling, and raw redispatch helpers; add to `pytorch_vulkan_python` in `CMakeLists.txt`.
- Modify `tests/python/test_vulkan_unary.py`: first-order unary gradient and graph behavior tests.
- Modify `tests/python/test_vulkan_add.py`: tensor/tensor and scalar add gradient tests.
- Modify `tests/python/test_vulkan_serialization.py` only if required to verify that autograd metadata remains compatible with the existing explicit serializer; do not expand serialization scope.
- Create `tests/python/test_vulkan_autograd.py`: shared cross-operator gradient, detach, unsupported-form, and retained-graph coverage.
- Modify `README-build.md`: replace the current autograd-deferred wording with the declared initial autograd capability and unsupported boundaries after verification passes.
- Modify `docs/superpowers/index.md`: link the implementation plan.

---

### Task 1: Establish The Autograd Boundary

**Files:**
- Create: `src/vulkan/operators/autograd.h`
- Create: `src/vulkan/operators/autograd.cpp`
- Modify: `CMakeLists.txt:96-102`
- Test: `tests/python/test_vulkan_autograd.py`

**Interfaces:**
- Consumes: existing raw Vulkan operator helpers and `at::Tensor` metadata validation.
- Produces: reusable `pytorch_vulkan::autograd_*` wrappers that execute a raw forward callback below autograd and return a Vulkan tensor with a recorded backward function.

- [ ] **Step 1: Write the failing boundary tests**

Add tests that construct a `requires_grad=True` Vulkan tensor, call
`torch.neg`, assert that the result has a grad function, and call `backward()`
with a Vulkan gradient. Add a test that CPU tensors still use normal CPU
autograd. The Vulkan tests skip only when
`pytorch_vulkan.is_available()` is false.

```python
def test_vulkan_operation_builds_autograd_graph(vulkan_backend):
    x = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    y = torch.neg(x)
    assert y.grad_fn is not None
    y.backward(torch.ones(y.shape, dtype=y.dtype).to(vulkan_backend))
    assert x.grad is not None
    assert x.grad.device == x.device
```

- [ ] **Step 2: Run the focused test to confirm the current failure**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_autograd.py`

Expected: the new Vulkan autograd assertion fails because the current PrivateUse1
functional kernels do not record a backward graph.

- [ ] **Step 3: Implement the minimal reusable boundary**

Define a raw-forward callback type and custom `torch::autograd::Function`
helpers. Each helper must use `at::AutoDispatchBelowAutograd` while calling the
raw callback. Save tensors with `ctx->save_for_backward` only when needed by
the backward formula. Return `variable_list` entries matching every forward
argument, including undefined gradients for scalar and non-differentiable
arguments. Add `autograd.cpp` to the Python extension target and route the
existing `neg` functional registration through the new boundary as the pilot
operator.

Do not register a CPU fallback. Do not wrap `out=` or in-place entry points.

- [ ] **Step 4: Run the focused boundary tests**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_autograd.py`

Expected: the boundary graph and CPU-preservation tests pass; operator-specific
gradient tests remain marked or failing until Tasks 2 and 3 add formulas.

### Task 2: Add Unary Autograd

**Files:**
- Modify: `src/vulkan/operators/unary.cpp:52-98,129-139`
- Modify: `src/vulkan/operators/autograd.cpp`
- Test: `tests/python/test_vulkan_unary.py`
- Test: `tests/python/test_vulkan_autograd.py`

**Interfaces:**
- Consumes: Task 1's unary autograd helper and the existing raw `dispatch_unary` implementation.
- Produces: differentiable `abs` and `relu` functional calls alongside Task 1's `neg` pilot; `out=` and in-place calls remain unchanged.

- [ ] **Step 1: Write failing unary gradient tests**

For each of `torch.abs` and `torch.relu`, compare Vulkan output
and `x.grad` against CPU for non-boundary float32 values. Include a zero and a
negative/positive ReLU boundary case using the repository's documented expected
convention. Assert gradient device, dtype, shape, and contiguity. Add a test
that `out=` with a requires-grad input still fails as currently documented.

```python
@pytest.mark.parametrize("operation", [torch.abs, torch.relu])
def test_unary_backward_matches_cpu(vulkan_backend, operation):
    cpu = torch.tensor([-2.0, -0.25, 0.5, 3.0], requires_grad=True)
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    cpu_result = operation(cpu)
    vk_result = operation(vk)
    cpu_result.sum().backward()
    vk_result.sum().backward()
    torch.testing.assert_close(vk_result.cpu(), cpu_result.detach())
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad)
```

- [ ] **Step 2: Run the unary tests to confirm failure**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_unary.py tests/python/test_vulkan_autograd.py -k backward`

Expected: forward tests pass but Vulkan backward tests fail before the formulas
are wired.

- [ ] **Step 3: Implement unary formulas**

Use the raw Vulkan forward callback from Task 1. Save the forward output for
`abs` and `relu` so the backward kernel can use the same activation semantics;
`neg` may return the negated incoming gradient. Allocate each gradient through
the existing Vulkan allocator and dispatch the existing compute operation or a
small dedicated gradient kernel. Return one gradient for the input and an
undefined gradient for every non-tensor argument.

The wrapper must execute only the functional schemas. Keep registrations for
`neg.out`, `abs.out`, `relu.out`, and all in-place rejection functions on their
existing non-autograd paths.

- [ ] **Step 4: Run the unary tests**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_unary.py tests/python/test_vulkan_autograd.py -k 'unary or backward'`

Expected: all selected unary forward and first-order backward tests pass.

### Task 3: Add Binary Autograd

**Files:**
- Modify: `src/vulkan/operators/binary.h:18-30`
- Modify: `src/vulkan/operators/add.cpp:141-242`
- Modify: `src/vulkan/operators/sub.cpp:7-55`
- Modify: `src/vulkan/operators/mul.cpp:7-26`
- Modify: `src/vulkan/operators/autograd.cpp`
- Test: `tests/python/test_vulkan_add.py`
- Create: `tests/python/test_vulkan_binary_autograd.py`

**Interfaces:**
- Consumes: Task 1's binary autograd helper and raw tensor/tensor and tensor/scalar callbacks.
- Produces: differentiable tensor/tensor and supported tensor/scalar forms for `add`, `sub`, `rsub`, and `mul`; scalar arguments never receive gradients.

- [ ] **Step 1: Write failing binary gradient tests**

Cover equal-shaped tensor/tensor `add`, `sub`, and `mul`; tensor-left and
scalar-left supported scalar forms; one differentiable input at a time; and
both differentiable tensor inputs for `mul`. Compare values and gradients to
CPU. Assert that unsupported broadcasting, non-unit `alpha`, and `out=` forms
still raise explicit errors.

```python
def test_binary_tensor_gradients_match_cpu(vulkan_backend):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], requires_grad=True)
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_()
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_()
    cpu_result = torch.mul(cpu_lhs, cpu_rhs)
    vk_result = torch.mul(vk_lhs, vk_rhs)
    cpu_result.sum().backward()
    vk_result.sum().backward()
    torch.testing.assert_close(vk_lhs.grad.cpu(), cpu_lhs.grad)
    torch.testing.assert_close(vk_rhs.grad.cpu(), cpu_rhs.grad)
```

- [ ] **Step 2: Run the binary tests to confirm failure**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_binary_autograd.py`

Expected: forward behavior passes where covered and Vulkan gradient assertions
fail until binary formulas are registered.

- [ ] **Step 3: Implement binary formulas**

For tensor/tensor operations, return gradients for both operands: `add` uses
the incoming gradient for both, `sub` uses positive and negative incoming
gradients, and `mul` multiplies the incoming gradient by the opposite operand.
For tensor/scalar operations, return only the tensor gradient; for scalar-left
subtraction, negate the incoming gradient. Use raw Vulkan pointwise dispatch
for these calculations and preserve the existing equal-shape, `alpha == 1`,
and scalar validation.

Save the opposite Vulkan operand for `mul`; save only required metadata or no
extra tensor for `add` and `sub`. Ensure saved tensors retain their Vulkan
allocation safely until backward completes.

- [ ] **Step 4: Run binary and existing operator tests**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_binary_autograd.py tests/python/test_vulkan_add.py`

Expected: binary first-order gradients and all existing add/sub/mul forward,
alias, validation, and rejection tests pass.

### Task 4: Edge Cases, Capability Documentation, And Full Verification

**Files:**
- Modify: `tests/python/test_vulkan_autograd.py`
- Modify: `README-build.md:117,174-177`
- Modify: `docs/superpowers/index.md`
- Modify: `docs/superpowers/specs/2026-09-11-vulkan-phase-3-roadmap-design.md`
- Modify: `docs/superpowers/specs/2026-09-12-vulkan-autograd-design.md`

**Interfaces:**
- Consumes: completed unary and binary autograd registrations.
- Produces: documented initial autograd capability table and a complete Phase 3 verification gate.

- [ ] **Step 1: Add edge-case tests**

Test detached inputs, inputs with `requires_grad=False`, repeated backward with
`retain_graph=True`, non-scalar outputs with explicit Vulkan `grad_output`, and
unsupported operators such as `torch.sin(vulkan_tensor)` producing an explicit
unsupported backend error rather than a CPU tensor. Verify no CPU fallback
occurs and that CPU-only autograd remains unchanged.

- [ ] **Step 2: Run focused edge-case tests**

Run: `PYTHONPATH=build .venv/bin/python -m pytest -q tests/python/test_vulkan_autograd.py tests/python/test_vulkan_unary.py tests/python/test_vulkan_binary_autograd.py`

Expected: all initial autograd tests pass, with only the existing environment
skip when no Vulkan device is available.

- [ ] **Step 3: Update capability documentation**

Document the six supported first-order backward paths, strict unsupported-op
errors, and deferred higher-order/view-replay behavior in `README-build.md`.
Keep the 45--75 operator-family and 80%-of-common-CNN/MLP usage target in the
roadmap and autograd design documents. Link this plan from the documentation
index.

- [ ] **Step 4: Run the complete verification matrix**

Run:

```text
cmake --build build/vulkan --target pytorch_vulkan_python vulkan_transfer_tensor_test vulkan_allocator_test vulkan_serialization_probe
ctest --test-dir build/vulkan --output-on-failure
PYTHONPATH=build .venv/bin/python -m pytest -q tests/python tests/test_cpu_only_build.py tests/test_vulkan_probe.py tests/test_vulkan_unavailable.py
git diff --check
```

Expected: native tests pass or skip only for genuine Vulkan unavailability;
the reduced Python suite passes; the manual extension-build test remains
excluded from this routine command and is run only explicitly when requested.

- [ ] **Step 5: Review the final diff and commit by task**

Inspect `git diff`, `git status`, and the complete test output. User creates the
commits after each independently reviewed task; suggested commit messages are
`feat: add Vulkan autograd boundary`, `feat: add unary Vulkan autograd`,
`feat: add binary Vulkan autograd`, and `test: close Vulkan autograd gate`.
