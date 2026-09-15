# Vulkan Operator Capability Matrix

This matrix is the declaration source for the Vulkan operator slices. It
separates **metadata construction** from **consuming-operator execution**: a
successful view operation guarantees valid metadata and storage range, not
universal operator support.

| Operator/capability | Input dtype | Output dtype | Forms/layout contract | Autograd | Empty input |
| --- | --- | --- | --- | --- | --- |
| `aten::sum` | F32 | F32 | strided, non-overlapping input; rank within the Vulkan limit; selected dimensions, `keepdim`, contiguous F32 `out=` | first-order | empty input produces a Vulkan-resident identity result (0 for `sum`) |
| `aten::mean` | F32 | F32 | strided, non-overlapping input; rank within the Vulkan limit; selected dimensions, `keepdim`, contiguous F32 `out=` | first-order | empty input produces a Vulkan-resident NaN result for `mean` |
| `aten::argmax` | F32 | I64 | strided, non-overlapping input; optional dimension, `keepdim`, contiguous zero-offset I64 `out=` | not differentiable | explicitly rejected |
| `aten::linear` | F32 | F32 | 2-D strided input/weight with matching features; strided 1-D bias; non-overlapping output with no operand alias | first-order | explicit fixed-shape contract |
| `aten::convolution` | F32 | F32 | fixed F32 shapes, strided operands, bias required; stride/padding/dilation/groups fixed and non-transposed | first-order | explicit fixed-shape contract |
| `aten::_adaptive_avg_pool2d` | F32 | F32 | nonempty rank-4 NCHW strided, non-overlapping input; output size `(1, 1)` | first-order | explicitly rejected |
| unary `neg`/`abs`/`relu` | F32 (formatter Double exceptions documented below) | F32 | strided `vk:0`, rank <= 8, <= uint32 elements; fresh output | first-order | empty output supported |
| pointwise `add`/`sub`/`mul` | F32 | F32 | same-device strided tensors with equal shapes, or documented Python scalar forms; unsupported overlap/broadcasting rejected | first-order | empty output supported |
| `aten::as_strided` | F32 | F32 | Metadata-only on `vk:0`; requested sizes/strides must be non-negative and reference a valid in-allocation range. Non-zero offsets, non-contiguous layouts, overlap, and changed logical element counts are permitted. | chained first-order reverse mode | metadata contract |
| `aten::view` | F32 | F32 | Requires PyTorch-compatible `computeStride` metadata, then validates the resulting sizes/strides and storage range/device/dtype contract before creating the metadata-only alias. | chained first-order reverse mode | metadata contract |
| `aten::_reshape_alias` | F32 | F32 | Accepts the ATen-supplied size/stride alias metadata after shared storage-range, device, and dtype validation; creates no copy or dispatch. | chained first-order reverse mode | metadata contract |
| `aten::reshape` | F32 | F32 | Aliases with the computed strides when PyTorch `computeStride` succeeds; otherwise uses a Vulkan-resident contiguous copy followed by a metadata view. No CPU fallback. | first-order reverse mode | metadata contract |
| `torch.masked_select` | F32 values, bool mask | F32 | contiguous or positive-stride/non-zero-offset F32 value views; positive-stride and non-zero-offset value views are materialized with a Vulkan-resident copy before compaction. The mask must remain same-shaped, contiguous, and zero-offset bool on `vk:0`; count and compaction consume Vulkan payloads with no CPU fallback | forward-only | empty output supported |

Bool and float16 are not declared for the reduction/indexing slice. Vulkan I64
allocation is permitted only while materializing a declared `argmax` result.
Wrong-shaped `argmax.out` tensors are outside this matrix and are explicitly
rejected before temporary allocation or Vulkan dispatch. Invalid device, dtype,
layout, contiguity, and offset metadata are rejected at the same boundary.

The fixed model slices are the **fixed MLP** (`aten::linear`) and **fixed CNN**
(`aten::convolution` plus `aten::_adaptive_avg_pool2d`). Every consuming
operator must independently declare the layouts it accepts; view construction
does not widen those contracts.

## Formatter-Compatible Double

The formatter Double tier is gated by `shaderFloat64`, uses native
`sizeof(double)` storage, and supports only `aten::abs.default`,
`aten::min.default`, `aten::max.default`, `aten::ceil.default`,
`aten::ne.Tensor`, `aten::div.Tensor`, `aten::gt.Scalar`, `aten::lt.Scalar`,
and `aten::_local_scalar_dense.default`. It uses an explicit final-value presentation transfer,
with no normal CPU fallback. Vulkan formatter Double payload readback to CPU is unsupported.
Vulkan formatter Double support requires the shaderFloat64 device feature.
Vulkan formatter conversion supports only Vulkan float32 to Vulkan Double.
Double `add`, `mul`, and unrelated operators remain rejected.

## Serialization and multiprocessing

Explicit `pytorch_vulkan.save/load` preserves view sizes, strides, offsets,
shared storage, and `requires_grad`; autograd graphs are not serialized.
Generic `torch.save` of Vulkan tensors remains unsupported. Multiprocessing
uses CPU tensors at the process boundary; `spawn` children may independently
restore Vulkan tensors and rebuild views. Direct Vulkan IPC and post-
initialization fork remain unsupported.

All supported execution remains Linux `vk:0` and respects the documented dtype
limits. It avoids hidden CPU fallback; the masked-select value-view contract
permits intentional Vulkan-to-Vulkan value-view materialization and forbids hidden CPU payload materialization or readback. The final `.cpu()` comparison is an explicit presentation transfer, not fallback.

<!-- Vulkan conformance supported schemas:
aten::sum.dim_IntList,aten::mean.dim,aten::argmax.default,aten::linear.default,
aten::convolution.default,aten::_adaptive_avg_pool2d.default,aten::neg.default,
aten::abs.default,aten::relu.default,aten::add.Tensor,aten::sub.Tensor,
aten::mul.Tensor,aten::as_strided.default,aten::view.default,
aten::_reshape_alias.default,aten::reshape.default,aten::masked_select.default -->

<!-- Vulkan conformance rejected schemas:
aten::max_pool2d_with_indices.default,aten::neg.out,aten::ne.Tensor,
aten::neg_.default,aten::neg.default,aten::add.Tensor,aten::sum.dim_IntList,
aten::_adaptive_avg_pool2d.default,aten::convolution.default -->

<!-- Vulkan conformance deferred schemas:
aten::_adaptive_avg_pool2d_backward.default,aten::_cat.default,aten::_copy_from.default,aten::_copy_from_and_resize.default,
aten::_local_scalar_dense.default,aten::_log_softmax.out,aten::_log_softmax_backward_data.out,
aten::_native_multi_head_attention.default,aten::_native_multi_head_attention.out,aten::_softmax.out,
aten::_softmax_backward_data.out,aten::_to_copy.default,aten::_transform_bias_rescale_qkv.default,
aten::_upsample_nearest_exact2d.out,aten::_upsample_nearest_exact2d_backward.grad_input,aten::abs.out,
aten::add.Scalar,aten::add.Scalar_out,aten::add.out,aten::addcdiv.out,aten::addcmul.out,aten::addmm.default,
aten::addmm.out,aten::amax.out,aten::amin.out,aten::arange.start_out,aten::argmax.out,aten::atan.out,
aten::avg_pool2d.out,aten::avg_pool2d_backward.grad_input,aten::bernoulli_.float,aten::binary_cross_entropy.default,
aten::binary_cross_entropy_backward.default,aten::binary_cross_entropy_backward.grad_input,
aten::bitwise_and.Tensor_out,aten::bitwise_not.out,aten::bitwise_or.Tensor_out,aten::bitwise_xor.Tensor_out,
aten::bmm.out,aten::cat.out,aten::ceil.default,aten::ceil.out,aten::clamp.out,aten::clamp_min.out,
aten::convolution_backward_overrideable.default,aten::convolution_overrideable.default,aten::copy_.default,
aten::div.Tensor,aten::div.out,aten::dot.default,aten::empty.memory_format,aten::empty_strided.default,
aten::eq.Scalar_out,aten::eq.Tensor_out,aten::exp.out,aten::fill_.Scalar,aten::ge.Scalar_out,
aten::ge.Tensor_out,aten::gelu.out,aten::gelu_backward.grad_input,aten::gt.Scalar,aten::gt.Scalar_out,
aten::gt.Tensor_out,aten::hardsigmoid.out,aten::hardsigmoid_backward.grad_input,aten::hardswish_.default,
aten::hardswish_backward.default,aten::hardtanh.default,aten::hardtanh_.default,aten::hardtanh_backward.default,
aten::isfinite.out,aten::le.Scalar_out,aten::le.Tensor_out,aten::leaky_relu.out,
aten::leaky_relu_backward.grad_input,aten::lerp.Scalar_out,aten::log.out,aten::log_sigmoid_backward.default,
aten::log_sigmoid_backward.grad_input,aten::log_sigmoid_forward.default,aten::log_sigmoid_forward.output,
aten::logit.default,aten::logit.out,aten::lt.Scalar,aten::lt.Scalar_out,aten::lt.Tensor_out,aten::max.default,
aten::max_pool2d_with_indices.default,aten::maximum.out,aten::mean.default,aten::mean.out,aten::min.default,
aten::minimum.out,aten::mm.out,aten::mse_loss.default,aten::mse_loss_backward.default,aten::mul.Scalar,
aten::mul.Scalar_out,aten::mul.out,aten::native_batch_norm.default,aten::native_batch_norm_backward.default,
aten::native_dropout.default,aten::native_dropout_backward.default,aten::native_layer_norm.default,
aten::native_layer_norm_backward.default,aten::ne.Scalar_out,aten::ne.Tensor,aten::ne.Tensor_out,aten::neg.out,
aten::nll_loss_backward.grad_input,aten::nll_loss_forward.output,aten::normal_.default,
aten::pow.Tensor_Scalar_out,aten::prod.int_out,aten::reciprocal.out,aten::relu.out,aten::resize_.default,
aten::round.out,aten::rsub.Scalar,aten::rsub.Scalar_out,aten::set_.source_Storage,
aten::set_.source_Storage_storage_offset,aten::sgn.out,aten::sigmoid.default,aten::sigmoid.out,
aten::sigmoid_.default,aten::sigmoid_backward.grad_input,aten::silu.out,aten::silu_backward.grad_input,
aten::sqrt.out,aten::sub.Scalar,aten::sub.Scalar_out,aten::sub.out,aten::sum.IntList_out,aten::sum.default,
aten::tanh.default,aten::tanh.out,aten::tanh_.default,aten::tanh_backward.grad_input,
aten::threshold_backward.grad_input,aten::uniform_.default,aten::upsample_bilinear2d.out,
aten::upsample_bilinear2d_backward.grad_input,aten::upsample_nearest2d.out,
aten::upsample_nearest2d_backward.grad_input,aten::zero_.default -->
