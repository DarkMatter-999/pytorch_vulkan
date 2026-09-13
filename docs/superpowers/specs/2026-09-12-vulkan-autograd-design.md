# Vulkan Autograd Integration Design

Date: 2026-09-12
Branch: `feat/vulkan-operator-support`

## Goal

Add PyTorch 2.4 autograd support for the currently shipped Vulkan operator
capability set, then make forward and backward support a paired requirement for
each future Vulkan operator. This closes the initial autograd portion of the
Phase 3 device-integration exit gate without claiming general PyTorch autograd
coverage.

## Current Scope

The initial implementation covers the existing differentiable Vulkan
operators:

- `neg`
- `abs`
- `relu`
- `add`
- `sub`
- `mul`

| Operator | Supported first-order backward paths |
| --- | --- |
| `neg`, `abs`, `relu` | Vulkan tensor input |
| `add`, `sub`, `mul` | equal-shape Vulkan tensor/tensor and documented Vulkan tensor/Python-number forms |

The existing operator restrictions remain in force: contiguous `float32`
tensors, the single `vk:0` device, synchronous execution, and only the
documented overloads and scalar forms. Unsupported Vulkan operators and
unsupported overloads fail immediately with an explicit backend error. They do
not materialize inputs on CPU, return a CPU tensor, or emit a warning and
continue.

## Design

Autograd must remain inside PyTorch's normal dispatcher and autograd machinery.
The backend will not add a custom Tensor or Storage implementation and will not
install a generic CPU fallback. Forward Vulkan operators provide the tensors
and metadata required by PyTorch autograd; their backward registrations use the
same Vulkan execution and validation boundaries as forward operations.

Each operator's implementation must define both:

- supported forward schemas, input metadata, output metadata, and alias rules;
- supported backward schemas, gradient formulas, saved forward state, and
  gradient metadata.

Backward outputs must remain Vulkan tensors when their corresponding
differentiable inputs are Vulkan tensors. Inputs that do not require gradients
must not receive gradients. Existing `out=` and in-place restrictions remain;
autograd support does not make those forms differentiable.

## Initial Autograd Contract

The first milestone supports first-order reverse-mode differentiation for the
six listed operators. Tests must cover tensor-tensor and supported tensor-scalar
forms where applicable, multiple differentiable inputs, detached inputs,
`requires_grad=False` inputs, repeated backward with retained graphs, and
CPU-reference gradient comparisons.

The following remain deferred:

- higher-order gradients;
- forward-mode AD;
- arbitrary view replay and general reshape autograd semantics;
- unsupported layouts, dtypes, devices, and overloads;
- sparse, quantized, and complex autograd;
- CPU fallback for unsupported Vulkan operators;
- multi-device and cross-process autograd behavior.

## Operator Rollout Policy

The backend will not target all PyTorch operators. PyTorch 2.4 exposes roughly
1,544 unique Aten operator families and 3,127 dispatch schemas, including
overloads and internal schemas. Coverage is instead driven by concrete model
and test requirements.

Every future differentiable operator must ship forward and backward support in
the same capability slice, with CPU reference tests, unsupported-input tests,
and Vulkan validation coverage. A capability table should record forward
schemas, backward support, metadata restrictions, and deferred behavior.

A practical long-term target is approximately 45--75 operator families for
common CNN and MLP workloads. This is expected to cover about 80% of operator
usage in those workloads, not 80% of PyTorch's total operator inventory. A
rough effort estimate is 7--15 months of full-time work for that target,
depending mainly on Vulkan kernel complexity, layout support, synchronization,
and model requirements. The initial six-operator autograd foundation is
estimated at 2--4 weeks.

| Operator category | Approximate families | Difficulty |
| --- | ---: | --- |
| Elementwise activations and arithmetic | 15--25 | Low to medium |
| Reductions, softmax, and losses | 8--15 | Medium |
| Shape, view, indexing, and concatenation | 8--15 | Medium to high |
| Matrix multiplication and linear layers | 3--6 | High |
| Pooling and normalization | 8--12 | High |
| Convolution and transpose convolution | 2--4 | Very high |
| Optimizer and training-support operations | 5--10 | Medium to high |

Likely later groups are elementwise operations, reductions and losses, shape
and indexing operations, matrix operations, pooling and normalization, and
convolution. Each group receives its own narrow contract and verification gate.

## Error Handling

An operation dispatched with a Vulkan tensor but without an implemented Vulkan
kernel must raise an explicit unsupported-operation error identifying the
operator and backend. The error must occur before any CPU materialization or
partial execution. CPU-only operations continue to use normal PyTorch CPU
dispatch because PrivateUse1 registrations do not intercept CPU tensors.

## Testing And Exit Gate

Initial autograd tests must prove:

- forward values and gradients match CPU references within the float32
  tolerance used by the existing test suite;
- gradients have correct device, dtype, shape, and layout metadata;
- graph construction, backward execution, and repeated retained-graph
  execution are stable;
- unsupported backward paths fail explicitly rather than falling back to CPU;
- existing device, copy, view, serialization, multiprocessing, CPU-only, native,
  and validation tests remain passing.

Retained-graph backward replay is covered within the existing rejection of
user-visible in-place Vulkan arithmetic; higher-order and general view-replay
semantics remain deferred.

Phase 3 remains open until the declared autograd capability set passes these
tests. Broader operator coverage and general view/reshape autograd semantics
are later capability groups, not prerequisites for this initial milestone.
