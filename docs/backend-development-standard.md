# PyTorch-reference-first backend development

## Purpose and standing decision

The destination is a functioning Vulkan backend comparable to CPU and CUDA in
functionality, usable through the ordinary PyTorch API. This is the standard
method for every operator family, tensor pattern, dispatch path and training
feature in this repository. It is not limited to convolution.

Use PyTorch as the semantic and architectural reference, then implement the
Vulkan primitives and contracts needed to make those paths work. A growing list
of narrow kernels or successful hand-adapted models is progress, not the final
compatibility target. Performance equivalence is a separate measurement goal.

The supported build currently targets PyTorch 2.4.0. Record the exact installed
version, corresponding upstream revision and relevant source locations for each
design. Review dependency maps when upgrading PyTorch; upstream internal routes
and schemas can change.

## Preferred implementation order

1. **Understand the reference contract.** Inspect dispatcher schemas, defaults,
   optional arguments, native/composite implementations, derivative definitions,
   generated autograd and relevant tests. PyTorch source is the primary design
   reference; test measured CPU behavior rather than guessing from a formula.
2. **Trace normal usage.** Start with stock modules/functions and an ordinary
   optimizer. Identify the actual dispatch path and recursively map required
   operators, layouts, copies, conversions and differentiation dependencies.
3. **Reuse PyTorch orchestration.** Prefer generated autograd and existing
   composite/view logic where their Vulkan leaves can satisfy the contract.
   Implement missing reusable primitives rather than duplicate PyTorch policies
   in a bespoke model helper or derivative wrapper.
4. **Implement device work.** Vulkan kernels can use different algorithms and
   execution strategies; they must preserve the reference's observable semantics
   within documented numerical tolerances and supported device constraints.
5. **Qualify the composition.** Compare Vulkan with the reference at direct
   operator and ordinary application levels, including requested derivatives,
   residency, aliasing/mutation behavior and repeat execution.

This is a preference for architectural reuse, not a requirement to copy CPU
kernels line-for-line. If generated/composite reuse cannot satisfy a device
constraint, first document the exact source/behavioral blocker and alternatives.
A custom implementation must preserve the reference contract and explicitly map
its differentiation/transform dependencies. Its narrower behavior must not become
the expected contract merely because it is easier to implement.

Do not remove an existing override without understanding the dispatcher route,
supplying required leaf kernels, and verifying the replacement. An Autograd
fallthrough is not a substitute for generated autograd.

## Contracts to map for each feature

- Schema, overload, defaults, optional inputs/outputs, shapes and broadcasting.
- Dtype storage/conversion/promotion and numerically justified tolerances.
- Strides, offsets, views, aliasing, internal overlap and mutation/version checks.
- Empty inputs, malformed operand rejection and error timing.
- Backend/device dispatch, allocation lifetime and synchronization requirements.
- First reverse derivative; create_graph and higher reverse derivatives when
  applicable; forward AD/JVP and torch.func/vmap as separately mapped contracts.
- Stock-module/optimizer composition, gradient reset/accumulation and state.

Document reference differences between CPU native, MKLDNN and CUDA when observed.
Do not emulate incidental extra work merely to match one algorithm's execution
pattern. Resolve observable semantic differences explicitly and test the chosen
contract against the relevant reference path.

## Autograd rules

Prefer the supported PyTorch version's generated derivative definitions.
`create_graph=True` must not silently return detached gradients where the
reference produces a differentiable dependence. Numerical first backward alone
does not establish autograd compatibility.

Raw device kernels executed below autograd are appropriate leaf implementations;
they do not create derivative history by themselves. Graph-preserving composition
must pass through dispatcher-visible differentiable operations or an explicitly
qualified derivative implementation.

Follow the derivative dependency closure: view backward, copy backward,
broadcast/reduction, accumulation and inverse operators can all be required.
Qualify undefined gradients, selective masks and saved-variable version checks.
Test mixed derivative directions with independent nonzero seeds. A second-order
pass does not prove arbitrary-order, forward-mode or whole-model support.

Existing first-order-only implementations are tracked compatibility gaps, not
the architectural template for new work. Expand them deliberately; do not change
all registrations in one unverified sweep.

## Evidence and qualification

Every design records three distinct evidence levels:

1. **Source-established:** installed schema/dispatch and version-matched upstream
   implementation establish the route or restriction.
2. **Reference-executed:** small CPU probes establish observed behavior and serve
   as reproducible oracles; use another reference implementation when needed.
3. **Vulkan-executed:** targeted tests establish actual values, shapes, layouts,
   gradients, graph behavior and residency on the named device/driver.

Neither declarations nor generated registrations count as runtime proof. Capability
evidence must preserve existing cases and bind dtype/rank claims to real inputs.
Do not widen declarations to make a validator pass. Update the evidence vocabulary
when it cannot accurately describe a newly qualified differentiation contract.

Acceptance workloads use ordinary model construction, module movement, loss and
optimizer APIs. CPU initialization followed by normal device transfer is valid;
private parameter replacement or a custom loss that avoids a missing API is not
evidence that the ordinary workload works.

No silent CPU fallback. Unsupported behavior remains explicitly documented and
rejected where possible until implemented. Avoid tests that canonize silent wrong
results or lost gradient history. Contract expansion replaces obsolete rejection
expectations with stronger positive parity and malformed-input coverage.

Use targeted checks during implementation and full required gates for stage
qualification. Verify the compiled extension and shader artifacts match source.
Run device gates sequentially; repeat when changes, failures or unresolved
intermittency justify it. Claims are scoped to executed hardware and versions.

## Planning and integration

When a workload hits a missing primitive, capture the failure and extend the
dependency map. Pause dependent work when necessary. Split broad foundational
changes into independently useful, testable prerequisite stages, each with its
own design and implementation plan. Do not quietly introduce backend-specific
application substitutions.

The human owns every commit and push. Keep specifications, evidence, verification
results and outstanding compatibility gaps discoverable. This document and root
AGENTS.md are durable repository policy; excluded planning ledgers do not replace
them.
