# Vulkan Operator Capability Matrix

This matrix is the declaration source for the Phase 4 reduction/indexing slice.
Only entries listed here may be registered by the Vulkan operator sources.

| Operator | Input dtype | Output dtype | Forms | Autograd | Empty input |
| --- | --- | --- | --- | --- | --- |
| `aten::sum` | F32 | F32 | all dimensions, `keepdim`, `out=` | first-order | explicitly rejected |
| `aten::mean` | F32 | F32 | all dimensions, `keepdim`, `out=` | first-order | explicitly rejected |
| `aten::argmax` | F32 | I64 | optional dimension, `keepdim`, metadata-safe `out=` only | not differentiable | explicitly rejected |

Bool and float16 are not declared for this slice. Vulkan I64 allocation is
permitted only while materializing a declared `argmax` result.
Wrong-shaped `argmax.out` tensors are outside this matrix and are explicitly
rejected before temporary allocation or Vulkan dispatch. Invalid device, dtype,
layout, contiguity, and offset metadata are rejected at the same boundary.
