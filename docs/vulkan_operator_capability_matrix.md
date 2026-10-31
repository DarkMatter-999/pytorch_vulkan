# Vulkan Operator Capability Matrix

This matrix is the declaration source for the Vulkan operator slices. It
separates **metadata construction** from **consuming-operator execution**: a
successful view operation guarantees valid metadata and storage range, not
universal operator support.

The machine-readable contract is `docs/vulkan_capabilities.json`. Its unique
schema entries mirror the supported and deferred inventories below; constrained
negative cases remain rejection boundaries of their supported schema rather
than duplicate schema entries.

| Operator/capability | Input dtype | Output dtype | Forms/layout contract | Autograd | Empty input |
| --- | --- | --- | --- | --- | --- |
| `aten::sum` | F32 | F32 | strided, non-overlapping input; rank within the Vulkan limit; selected dimensions, `keepdim`, contiguous F32 `out=` | first-order | empty input produces a Vulkan-resident identity result (0 for `sum`) |
| `aten::mean` | F32 | F32 | strided, non-overlapping input; rank within the Vulkan limit; selected dimensions, `keepdim`, contiguous F32 `out=` | first-order | empty input produces a Vulkan-resident NaN result for `mean` |
| `aten::argmax` | F32 | I64 | strided, non-overlapping input; optional dimension, `keepdim`, contiguous zero-offset I64 `out=` | not differentiable | explicitly rejected |
| `aten::amax.out` / `aten::amin.out` | F32 | F32 | strided, non-overlapping input; rank ≤ 8; selected dimensions, `keepdim`, contiguous F32 `out=` | first-order reverse mode | empty reduced dimensions rejected |
| `aten::prod.int_out` | F32 | F32 | strided, non-overlapping input; rank ≤ 8; selected dimension, `keepdim`, contiguous F32 `out=` | first-order reverse mode | empty reduced dimensions produce one |
| `aten::_softmax.out` / `aten::_log_softmax.out` | F32 | F32 | strided, non-overlapping input; rank ≤ 8; one non-empty dimension; `half_to_float=False`, contiguous F32 `out=` | first-order reverse mode | empty reduced dimensions rejected |
| `aten::_softmax_backward_data.out` / `aten::_log_softmax_backward_data.out` | F32 | F32 | contiguous F32 grad/output tensors; rank ≤ 8; one supported dimension | first-order backward kernel | unsupported higher-order forms rejected |
| public `F.mse_loss` / `nn.MSELoss`, numerical `aten::mse_loss` / `aten::mse_loss_backward` leaves | F32 | F32 | Finite scalar and rank-2 bases, reductions `none`, `sum`, `mean`; public unequal-shape broadcast and expanded upstream through stock composition. Numerical leaves require matching logical operand shapes and materialize supported readable layouts on device. | Generated input-only, target-only and both first reverses with connected history; selected second reverse at input, target and upstream with independent CPU FD | empty `sum` is zero, empty `mean` is named NaN, empty `none` is empty; no arbitrary higher/forward/transform/out claim |
| numerical `aten::div_.Scalar` | F32 | F32 | Real finite representable Scalar divisor, including zero; exact in-place alias; scalar/vector/matrix witnesses. Generated MSE mean double backward uses this leaf. | Stock generated AD/ADInplaceOrView ownership; no standalone arbitrary division derivative claim | empty/layout and write rejection contracts covered by focused Scalar tests; Python `Tensor.div_(number)` selects the broader Tensor overload and is separate |
| `aten::sigmoid` / `aten::tanh` / `aten::gelu` | F32 | F32 | strided `vk:0`, rank <= 8, <= uint32 elements; GELU requires `approximate="tanh"` and the standard `0.044715` polynomial; fresh Vulkan output | first-order reverse mode | empty output supported without dispatch |
| stock `aten::linear` composite (rank-2 weight) | F32 | F32 | Ordinary `nn.Linear`/`F.linear`, weight `(4,3)`, input ranks 1–8 with per-rank witnesses, bias present/absent, and trainable/frozen/no-grad states. Contiguous inputs at all ranks; for ranks 3–8, also nonzero-offset leading-axis-transposed slices. CPU-created tensors transferred to `vk:0`; observed stock routes include `addmm`, promoted `mm`/`squeeze_`, folded `mm`, folded `mm` with reshape-copy, and expanded `bmm` with optional broadcast bias `add_`. | First gradients for trainable and frozen-weight inputs on named witnesses; graph-preserving selected second direction for the rank-2 trainable+bias witness only | GEMM dispatch |
| allocating `aten::mul.Tensor` | F32 | F32 | Broadcasting allocation on `vk:0`; actual fixtures include rank-0/rank-1, `(2,1)`/`(1,3)`, and `(2,1,3)`/`(4,3)` operands. Captured readable views include offset operands; the executable suite covers rank-0, nonunit-stride, and expanded reads. Only allocating Tensor mul gains broadcasting; `mul_`/`mul.out` and existing aliases retain their separate contracts. Source-bound cases: `g1.mul.broadcast.rank0-rank1`, `g1.mul.broadcast.rank2`, and `g1.mul.broadcast.rank3-rank2`. | Stock generated first reverse; selected mixed reverse for the first two named broadcast cases with nonzero independent projections and centered CPU finite differences | Empty/broadcast result behavior remains covered by `tests/python/test_vulkan_mul_broadcast.py` |
| allocating `aten::dot` | F32 | F32 | Equal-length rank-1 vectors on `vk:0`; captured nonunit-stride vector input. Executable fixtures also cover offset, zero-stride, overlapping reads, and empty contraction. Source-bound cases: `g1.dot.rank1.forward`, `g1.dot.rank1.grad-a-wrt-b`, and `g1.dot.rank1.grad-b-wrt-a`. | Stock generated first reverse and both selected cross-operand mixed reverse directions; each mixed direction has an independent nonzero CPU centered-FD projection | Empty contraction returns rank-0 zero |
| allocating `aten::mv` | F32 | F32 | Rank-2 matrix and rank-1 vector with matching contraction size on `vk:0`; captured transposed matrix and nonunit-stride vector. Executable fixtures additionally cover offset, zero-stride, and read-overlapping inputs. Source-bound cases: `g1.mv.rank2-rank1.forward`, `g1.mv.rank2-rank1.grad-m-wrt-x`, and `g1.mv.rank2-rank1.grad-x-wrt-m`. | Stock generated first reverse and both selected cross-operand mixed reverse directions; each mixed direction has an independent nonzero CPU centered-FD projection | Empty rows produce an empty vector; empty contraction produces zeros |
| stock finite `aten::matmul` and `@` composite | F32 | F32 | Vector/vector-matrix routes plus the source-owned batched, broadcasted, layout and selection families below on `vk:0`. Public `torch.matmul` and `@` execute PyTorch 2.4 stock dot/mv/mm/bmm composition; rank bounds 1–8 describe witnessed ranks, not Cartesian rank-pair support. | Original-base first reverse, CPU-matched history, and both selected cross-operand mixed directions for the contiguous batch-matrix and cross-broadcast fixtures, each with independent CPU centered FD | Named empty M/N/K, zero-batch and cross-zero families; malformed empty geometry rejects |
| `aten::stack` | F32 | F32 | Bounded stack of matching contiguous 2-D Vulkan matrices, with insertion dimension `0` or `1`; output remains Vulkan-resident and uses Vulkan-to-Vulkan copies only | first-order | empty rejected |
| `aten::mm` | F32 | F32 | Rank-2 matrices on `vk:0` with matching inner dimensions; readable contiguous, transposed, positive-stride, offset, zero-stride, and internally-overlapping inputs are consumed or safely materialized; fresh dense output. Named layout witnesses: `test_mm_reads_valid_offset_positive_stride_and_transpose_views`, `test_mm_reads_zero_stride_and_overlapping_operands`, and `test_mm_materializes_singleton_stride_views_in_each_gemm_role`. | Generated first reverse and finite mixed second reverse on `matrix.graph.mm.generated`; forward-mode, transforms, and arbitrary derivative order are unclaimed | Empty M, N, and K witnesses in `test_mm_empty_dimensions_match_cpu` |
| `aten::addmm` | F32 | F32 | Allocating form: rank-2 matrices on `vk:0`, broadcastable `self`, independent alpha/beta handling, readable matrix layouts, and GEMM dispatch. Finite nonzero alpha uses the fixed shared GEMM policy: accumulate the F32 dot first, then multiply its accumulator by alpha once in the F32 epilogue; this does not emulate MKL layout/ISA routes or guarantee CPU overflow-class parity when finite intermediates overflow. Well-scaled finite cases retain CPU parity at the direct F32 tolerance. Extreme finite-alpha witnesses compare with the independent scalar policy oracle and separately record pinned-CPU classes/differences; they retain all original signed, transposed, offset, zero-stride, singleton, and dense-column inputs. NaN/Inf alpha keeps the qualified pre-product behavior. Named witnesses include `test_addmm_broadcast_self_and_independent_scalars`, `test_addmm_finite_alpha_uses_policy_oracle_and_records_cpu_class`, `test_addmm_finite_alpha_uses_policy_oracle_for_cpu_view_cases`, and `matrix.graph.addmm.generated`. `aten::addmm.out` retains its separate existing output/alias/autograd restrictions and is not included in the allocating contract. | Generated first reverse and finite mixed second reverse on `matrix.graph.addmm.generated`; forward-mode, transforms, and arbitrary derivative order are unclaimed | Empty M/N return empty outputs; K=0 follows CPU beta/self behavior and ignores alpha; witnessed by `test_addmm_empty_dimensions_match_cpu_and_stay_on_vulkan` and `test_addmm_zero_k_underflowed_nonzero_beta_still_multiplies_self` |
| `aten::bmm` | F32 | F32 | Rank-3 operands with equal batch counts and matching contraction dimensions on `vk:0`; no batch broadcasting or attention-only restriction. Readable contiguous, transposed, offset, positive-stride, zero-stride, and overlapping matrix reads are consumed or safely materialized; fresh dense output. Witnesses: `test_bmm_batches_and_readable_layouts_match_cpu` and `matrix.graph.bmm.generated`. | Generated first reverse and finite mixed second reverse on `matrix.graph.bmm.generated`; forward-mode, transforms, and arbitrary derivative order are unclaimed | Empty batch, rows, contraction, or columns; witnessed by `test_bmm_empty_dimensions_match_cpu` |
| `aten::convolution` / `aten::convolution_backward` / `aten::convolution_overrideable` / `aten::convolution_backward_overrideable` | F32 | F32 | Positive-size rank-4 NCHW input/weight; ordinary and transposed 2-D convolution with positive stride/dilation and nonnegative padding; optional bias; groups include grouped and depthwise-multiplier cases. Ordinary forward accepts nonnegative output-padding metadata and ignores it; ordinary backward requires `[0,0]`. Transposed weight is `[Cin,Cout/groups,Kh,Kw]`, with output channels `weight.size(1) * groups`; each transposed output-padding component is nonnegative and smaller than stride **or** dilation on that axis. Non-overlapping readable strided operands include qualified offset/transposed input or weight views; expanded/zero-stride overlap is qualified only for read-only `grad_output`. Backward supports all eight output masks and returns exact `None` slots (mask `000` has no result work); dBias is actual `(Cout,)`, independent of advisory `bias_sizes`. Overrideable backward has a separate all-true numerical witness against standard backward with absent advisory bias sizes. | Finite reverse-mode evidence: first reverse on the five named graph routes. Independent `ggI`, `ggW`, `ggb`, and combined second reverse are witnessed by the grouped-combined and transposed-second records; the grouped-selected-third and depthwise-selected-third records witness second `ggW` only. Selected third `d_g d_x d_w` is qualified only for the named grouped and depthwise routes. The transposed route is qualified through second reverse only. Overrideable forward is qualified through first reverse only. Forward AD/JVP, `torch.func`/vmap and other transforms, arbitrary reverse order, CUDA, HVP, and whole-model training are not claimed. For transposed convolution with nonzero output-padding, the selected next derivative reaches ordinary backward and remains CPU-reference rejected. | channels-last-only layouts and empty operands/results unsupported; bounded positive F32 rank-4 strided contract only |
| `aten::_adaptive_avg_pool2d` / `_adaptive_avg_pool2d_backward` | F32 | F32 | nonempty rank-4 NCHW strided, non-overlapping input; output size `(1, 1)`; backward grad shape `(N,C,1,1)` | first-order | empty and non-global forms rejected |
| `aten::native_batch_norm` / `native_batch_norm_backward` | F32 | F32 | fixed contiguous training inputs `(2,4)` or `(2,4,2,2)`; affine F32 `(4,)`; Vulkan running mean/variance; momentum `0.1`, eps `1e-5`; backward output mask `[true,true,true]` | first-order | eval, partial affine, missing stats, other shapes, and masks rejected |
| `aten::native_layer_norm` / `native_layer_norm_backward` | F32 | F32 | `vk:0`, strided input (non-contiguous, offset and stride-0 operands are densified with a Vulkan copy), rank 2-4 witnessed and up to 8 accepted; `normalized_shape` of any length >= 1 matching the trailing input dimensions; weight and bias independently absent or F32 `vk:0` with shape `normalized_shape`; any `eps` (converted to float32 as on CPU, including `0`, negative, overflow and non-finite values, with CPU's NaN/Inf results); `mean`/`rstd` are F32 with shape `[*leading, 1 x len(normalized_shape)]`; empty batches (M = 0, backward returns zero `dW`/`dB`) and zero-width normalization (N = 0: `mean` 0, `rstd` NaN, empty gradients) follow CPU; backward accepts every `output_mask` that autograd produces (`mask[1]` needs weight, `mask[2]` needs bias) and returns undefined tensors for masked-off outputs; results are fresh contiguous Vulkan tensors. Rejected before Vulkan work: non-F32 or CPU operands, rank above 8, `M*N` above uint32, PyTorch's shape/`normalized_shape` errors, and mismatched `grad_out`/`mean`/`rstd` shapes (stricter than CPU, which does not check them). Two dispatches forward and three backward; no `.out` variants. | first-order reverse through PyTorch's generated `NativeLayerNormBackward0` for input, weight and bias (`nn.LayerNorm`, `F.layer_norm`); `create_graph`, higher-order, forward AD/JVP, `torch.func` and `vmap` are not claimed | empty input supported |
| `aten::avg_pool2d` / `avg_pool2d_backward` / `max_pool2d_with_indices` / `max_pool2d_with_indices_backward` | F32 (`indices`: int64) | F32 (`indices`: int64) | `vk:0`; rank 3 `(C,H,W)` or 4 `(N,C,H,W)` (`N = 0` allowed; zero channels/height/width rejected); the full PyTorch geometry: any kernel/stride/padding/dilation (max only)/`ceil_mode`, single-element lists and an empty stride as in ATen, avg `count_include_pad` and `divisor_override`, each parameter and extent up to 65535 and element counts up to uint32; PyTorch's own geometry errors (via `at::native::pool2d_shape_check` and friends). Strided, offset, channels-last and stride-0 (expanded) operands are read in place through the metadata buffer; outputs, `indices` and `grad_input` use CPU's memory formats (channels-last input gives channels-last output and indices; `grad_input` follows `self`). Average windows divide by the clipped window size exactly as ATen (ceil-mode hanging windows included). Max pooling reproduces ATen's update rule (`val > max || isnan(val)`: ties keep the first element, the last NaN wins, an all `-inf` window reports its first in-image element); `indices` are plane-relative `h*W + w` int64 tensors. Max backward equals ATen's ascending-order scatter-add bit for bit for every in-range index, including hand-supplied indices that point outside their own window: a per-plane device-side validation pass selects a fast gather over candidate windows (forward-produced indices) or an exact ascending scan of the plane. Index entries outside `[0, H*W)` are undefined behaviour on CPU (memory corruption, plane leakage or abort, differing between NCHW and channels-last); Vulkan defines them: they contribute nothing and nothing faults, and this is a recorded deviation, not CPU parity (int64 data reaches Vulkan only through the label upload, so negative values other than -100 cannot be constructed). Rejected before Vulkan work: non-F32 or CPU operands, non-int64/non-`vk:0` `indices`, rank other than 3/4, mismatched `grad_output`/`indices` shapes, `divisor_override == 0`, parameters beyond the shader limits. `.out`/`grad_input` forms, `max_pool2d_backward`, `max_unpool2d`, fractional and general adaptive pooling remain unregistered. | first-order reverse through PyTorch's generated `AvgPool2DBackward0` / `MaxPool2DWithIndicesBackward0` for the input (`nn.AvgPool2d`, `nn.MaxPool2d`, `F.avg_pool2d`, `F.max_pool2d`); `create_graph`, higher-order (including average pooling's structurally composable second derivative), forward AD/JVP, `torch.func` and `vmap` are not claimed | empty batch supported |
| `aten::upsample_nearest2d` / `_upsample_nearest_exact2d` / `upsample_bilinear2d` and their `_backward` forms (`F.interpolate`, `nn.Upsample`, `UpsamplingNearest2d`, `UpsamplingBilinear2d`) | F32 | F32 | `vk:0`; rank 4 only (`N = 0` allowed; zero channels, height or width rejected with ATen's messages via `upsample_2d_common_check`); `size`, `scale_factor` and `recompute_scale_factor` are resolved by PyTorch, and the leaf's `scales_h`/`scales_w` hints are honoured exactly as ATen: `static_cast<float>(1.0 / hint)` when present and positive, otherwise `float(in) / out`, with per-dimension identity (`out == in`, all modes) and `d >> 1` (`out == 2 * in`, `nearest` only) shortcuts that ignore the hint; `align_corners` for `bilinear`. The three modes are separate operations of one shader (`interpolation.comp`); scales are computed on the host with ATen's own helpers and the shader runs only correctly rounded add/sub/mul, floor, min and max (no float division). Source indices and weights are float32 and exact: `nearest` and `nearest-exact` outputs and gradients are bit-exact against CPU (index maps verified against CPU on 57,600 size/hint combinations); `bilinear` is bitwise equal to the unfused float32 model of its own expression and within about 1e-5 of CPU (CPU's own paths contract differently). FMA contraction is controlled with `precise`/`NoContraction`, proved structurally on the compiled and embedded SPIR-V by `tools/verify_interpolation_spv.py` (all 75 float add/sub/mul decorated; `Fma`, `OpFDiv`, `RelaxedPrecision` rejected) and on the device by contraction canaries (device equals the unfused model and differs from the fused form at 395/395 positions). Strided, offset, channels-last and stride-0 operands are read in place; outputs follow `input.suggest_memory_format()` and `grad_input` follows `grad_output.suggest_memory_format()` as on CPU. Backward is a deterministic gather (one thread per input element, exact monotone binary-search ranges, no atomics). Narrower than CPU and rejected before Vulkan work: non-F32 or non-`vk:0` operands, spatial sizes above 2^23, element counts above uint32 or the device launch ceiling, hints whose scale is non-finite or whose source coordinate reaches 2^62 (CPU undefined behaviour). `.out`/`.grad_input` forms, the `.vec` overloads (lowered by PyTorch), 1d/3d, bicubic, antialias, `area`, trilinear and non-F32 dtypes remain unregistered. | first-order reverse through PyTorch's generated `UpsampleNearest2DBackward0` / `UpsampleNearestExact2DBackward0` / `UpsampleBilinear2DBackward0`; `create_graph`, higher-order, forward AD/JVP, `torch.func` and `vmap` are not claimed | empty batch supported |
| `aten::_log_softmax` plus `nll_loss_forward` / `nll_loss_backward` | F32 logits, I64 labels | F32 | contiguous Vulkan logits `(2,3)`, class dimension `1`, contiguous Vulkan I64 labels `(2,)`, no weight, reduction `mean`, `ignore_index=-100` | first-order | other reductions, weights, labels, shapes, and class dimensions rejected |
| unary `neg`/`abs`/`relu` | F32 (formatter Double exceptions documented below) | F32 | strided `vk:0`, rank <= 8, <= uint32 elements; fresh output | first-order | empty output supported |
| `aten::ceil` (allocating `torch.ceil`) | F32 | F32 | strided `vk:0`, rank <= 8 by the shared unary guard, <= uint32 elements; witnessed ranks 1-2 (strided, empty `(0,3)`; rank-0 and offset views in `test_vulkan_unary.py`); NaN, ±inf and signed zero match CPU. The output is a fresh contiguous tensor: unlike CPU, a transposed input does not keep its strides. Non-F32 inputs (bool, int64, float16), rank above 8 and `vk:1` are rejected; Double takes the separate formatter-Double route documented below. `ceil.out` is covered by the unary `out=` row below; in-place `ceil_` is explicitly rejected. | Stock generated `CeilBackward0` (constant zero gradient, no history) on the witnessed first reverse only | empty output supported |
| unary `out=`: `aten::ceil.out` / `aten::sigmoid.out` / `aten::tanh.out` / `aten::gelu.out` | F32 | F32 | Shared unary out contract on `vk:0`: F32 input and F32 `out`; a same-shaped, mismatched or empty `out` is resized as in PyTorch; exact-alias `out=input` computes in place; partial overlap and internal overlap are rejected; a non-contiguous `out` keeps its strides. Witnessed rank 2 `(3,4)` and empty `(0,3)`; rank-0, offset and non-contiguous cases are in `test_vulkan_out.py`. Rejected with zero Vulkan work: non-F32 or CPU `out`, non-F32 input, rank above 8, and `requires_grad` input (PyTorch's own out= autograd rejection). Recorded deviations from CPU: `sigmoid.out`/`tanh.out` also accept a float64 `out` and integer inputs, and `ceil.out` only an equal-dtype `out`; Vulkan accepts only F32. `gelu.out` accepts only `approximate="tanh"` (the functional GELU contract); `"none"`, any other value and the `"none"` default are rejected before Vulkan work, and PyTorch itself rejects non-matching dtypes for GELU, so it has no dtype deviation. In-place `ceil_`, `sigmoid_`, `tanh_` and `gelu_` are explicitly rejected (their `.out` fallback is not used). These four entries declare `out: strided_out_supported` (a strided output of any non-overlapping layout is accepted and its strides preserved); other `out=` entries still carry the older `contiguous_out_required` value, which has not been re-examined. | not applicable (out= does not support autograd) | empty input and output supported |
| pointwise `add`/`sub` and non-broadcast `mul` scalar/`out=` forms | F32 | F32 | `aten::add.Scalar`, `aten::sub.Scalar`, `aten::rsub.Scalar`, `aten::mul.Scalar` plus existing scalar-out and tensor-out forms; equal-shape tensor forms retain their prior contracts. Allocating `mul.Tensor` broadcasting is specified separately above. No promotion; `add`/`sub` require finite representable `alpha == 1`. | first-order for functional forms; `out=` follows PyTorch autograd restrictions | empty output supported without dispatch |
| `aten::as_strided` | F32 | F32 | Metadata-only on `vk:0`; requested sizes/strides must be non-negative and reference a valid in-allocation range. Non-zero offsets, non-contiguous layouts, overlap, and changed logical element counts are permitted. The returned alias preserves the input autograd relationship. | chained first-order reverse mode | metadata contract |
| `aten::view` | F32 | F32 | Requires PyTorch-compatible `computeStride` metadata, then validates the resulting sizes/strides and storage range/device/dtype contract before creating the metadata-only alias. Named evidence: `view.view.trainable-seed` (float32 primary rank 2, shape `(3,4)`). | reverse second-order on the named seed witness only | metadata contract |
| `aten::_reshape_alias` | F32 | F32 | Accepts the ATen-supplied size/stride alias metadata after shared storage-range, device, and dtype validation; creates no copy or dispatch. The returned alias preserves the input autograd relationship. | chained first-order reverse mode | metadata contract |
| `aten::reshape` | F32 | F32 | PyTorch 2.4 stock `reshape_symint` aliases when compatible, otherwise uses clone(contiguous)+`_unsafe_view`; tested Vulkan leaves are `_reshape_alias`, clone/copy, and `_unsafe_view`. Named float32 rank-2 witnesses: `view.reshape.copy.trainable-seed` and `view.reshape.offset-copy.second-order`. | reverse second-order on the named witnesses only | metadata route or Vulkan copy, as selected by stock geometry |
| `aten::cat.default` | F32 | F32 | Functional `torch.cat(Tensor[], dim)` on `vk:0`; ranks 1–4, contiguous and qualified positive-stride/nonzero-offset inputs, rank-four channels-last selection, CPU-compatible skipped rank-one empty tensors; named actual TensorList cases in the manifest | reverse second-order on `cat.rank4.native-seed` and `cat.offset.trainable-seed` only | empty segments supported; empty list remains ordinary PyTorch rejection |
| tensor plumbing (`copy_`, `empty`, `empty_strided`, `_to_copy`) | F32 | F32 | Explicitly declared CPU↔`vk:0` copies (blocking unless an F32 `non_blocking` request qualifies under the async-transfer row below), same-dtype Vulkan-to-Vulkan copies, contiguous and non-negative-stride storage, valid storage ranges, and zero-sized tensors; `_to_copy` dtype conversion is limited to the formatter-compatible F32→Double path documented below. Unsupported devices, dtypes, overlap, non-F32 `non_blocking=True` requests (Bool/Long/Double), and invalid ranges are rejected before Vulkan work or fallback; F32 `non_blocking=True` is accepted as described in the async-transfer row. | not applicable | empty and zero-sized copies are no-ops |
| `CPU.pin_memory(device='vk')` / `CPU.is_pinned(device='vk')` (targeted pinned-host acceptance) | F32 | CPU F32 | On PyTorch 2.4.0 / RADV Renoir, explicit CPU pinning creates persistent mapped, host-visible Vulkan-backed storage (coherent-preferred); exact input sizes/strides and offset 0 are retained, including stride holes, while storage extent follows the stride extent. Storage is non-resizable. Recognition is allocator/DataPtr-provenance-based: direct pinned allocation is true, views (including empty views) retain provenance, foreign wrappers are false, clone/fresh pageable storage is false, and zero-byte allocation is false. This is not OS page-locking or a direct-DMA guarantee; F32 `non_blocking` transfer acceptance is the separate async-transfer row. The manifest entries `aten::_pin_memory.default` (float32 ranks 1-2: rank-1, rank-2 and stride-holes inputs) and `aten::is_pinned.default` (rank 1: pinned, pageable and pinned-view inputs) use the execution contract `host_pinned_allocation`: CPU-resident results over Vulkan-owned pinned storage with zero dispatches, Vulkan copies, explicit transfers and fallbacks; the identity first reverse is witnessed in `test_vulkan_copy_contracts.py`, not by conformance gradients; zero-size behavior is not claimed. | Generated `PinMemoryBackward` / identity first reverse only | Zero-byte pinned allocation is unrecognized; empty views over nonempty pinned storage remain recognized |
| F32 `non_blocking` transfers (`copy_`, `.to`, `_to_copy`; targeted acceptance) | F32 | F32 | On PyTorch 2.4.0 / RADV Renoir (extension `ec0bb464…2a63`), F32 `non_blocking=True` is accepted. In async execution mode, outside a training scope, on eligible layouts (both operands contiguous, or identical sizes/strides and dense non-overlapping, including preserve-format transposed, channels-last and channels-last-3d), the call returns before backend-observed completion (hold-seam evidence on Renoir; unsignaled-fence proof only in native fakes) for: Vulkan-pinned host⇄Vulkan copies (no staging), pageable host→Vulkan with a call-time staged snapshot (≤ 64 MiB; larger completes before return), and Vulkan→Vulkan copies with either flag and no host wait. Completes before return: pageable device→host, ineligible layouts (stride-0 broadcast, offset non-contiguous, gapped, `copy_` of a transposed source into a contiguous destination) and sync execution mode. In an active training scope host transfers complete before return; Vulkan→Vulkan copies are ordered in the training recording and complete at the end of the step (pre-existing blocking-D2D behavior); training-scope evidence is native engine + CPU classifier only, with no public-API witness. A single copy command was observed for pageable `.to(vk, non_blocking=True)` in async mode on the preserve-format dense layouts. `vk.to('cpu', non_blocking=True)` of a strided Vulkan tensor returns a Vulkan-pinned CPU tensor; `_to_copy(pin_memory=True)` for CPU output is accepted and ignored (unpinned), matching PyTorch 2.4. Bool/Long/Double `non_blocking=True` stay rejected; H2D and Vulkan-output `pin_memory=True` rejections are recorded deviations (PyTorch ignores them). | Generated `ToCopyBackward0` / `torch::autograd::CopyBackwards`: first reverse and one connected second-reverse witness (Task 0 seeds `[2,3,5]`/`[7,11,13]` → `[4,12,30]`/`[28,66,130]`, rtol/atol 0.003) through both routes; saved-version rejection after an in-place non-blocking `copy_`. Backward blocking by PyTorch reference (Task 0 source citation); Renoir observes a backward host wait that does not discriminate the flag because pageable D2H is blocking. | Higher-order beyond the one connected witness, forward-mode AD and transforms are unproven. Layout-degraded host transfers and pageable D2H are blocking; the degrade matrix covers H2D only with a `query()`-based (not hold-discriminated) completion check. A blocking dense non-contiguous readback issues per-element copy commands (about 48 for the 48-element channels-last-3d case, inferred as 49 total minus the single upload command, not measured). `set_` lease is not discriminated (an alias view keeps the old storage alive). D2D WAR/WAW, a public-API training-scope witness, degraded D2H layouts and >64 MiB pageable degrade are not covered on device. No OS page-lock/DMA claim; noncoherent flush/invalidate maintenance is synthetic-only evidence. No boxed PrivateUse1 fallback is registered; `fallback_count` (a seam-only counter) was unchanged. One coherent Renoir iGPU; independent streams/events are not provided; the earlier full qualification is historical and does not qualify these bytes. Receipts are retained locally and are not part of the repository. |
| `torch.masked_select` | F32 values, bool mask | F32 | contiguous or positive-stride/non-zero-offset F32 value views; positive-stride and non-zero-offset value views are materialized with a Vulkan-resident copy before compaction. The mask must remain same-shaped, contiguous, and zero-offset bool on `vk:0`; count and compaction consume Vulkan payloads with no CPU fallback | forward-only | empty output supported |
| scalar `torch.optim.SGD` | F32 parameters, gradients, `momentum_buffer` | F32 | contiguous, non-overlapping tensors on `vk:0`; scalar `lr`, momentum, dampening, and weight decay; `nesterov=False`, `maximize=False`, `foreach=False`, `differentiable=False` | first-order gradients supplied by supported autograd | empty updates follow PyTorch optimizer semantics |
| scalar `torch.optim.Adam` | F32 parameters, gradients, `exp_avg`, `exp_avg_sq` | F32 | contiguous, non-overlapping tensors on `vk:0`; scalar `lr`, betas, eps, and weight decay; `amsgrad=False`, `maximize=False`, `foreach=False`, `fused=False`, `capturable=False`, `differentiable=False`; tensor state stays Vulkan-resident and non-capturable `step` metadata stays host-resident | first-order gradients supplied by supported autograd | empty updates follow PyTorch optimizer semantics |
| fixed MLP training | F32 | F32 | `Sequential(Linear(8,16), ReLU, Linear(16,4))`; contiguous `(batch,8)` inputs and `(batch,4)` targets on `vk:0`; scalar summed squared-error loss; SGD or Adam only | forward and first-order backward on Vulkan | not part of the contract |
| flattened MNIST-shaped training | F32 | F32 | `Sequential(Flatten(1), Linear(784,32), ReLU, Linear(32,10))`; contiguous `(batch,1,28,28)` inputs and same-shaped contiguous F32 `(batch,10)` targets on `vk:0`; scalar summed squared-error loss; SGD or Adam only. The synthetic fixture generates one-hot targets; runtime does not validate one-hot semantics. | forward and first-order backward on Vulkan | not part of the contract |
| bounded vanilla RNN training | F32 | F32 | Explicit sequence loop with `batch ∈ [1,16]`, `sequence_length ∈ [1,64]`, input feature `128`, hidden feature `256`, contiguous `vk:0` state/intermediates, and fixed `Linear(128,256) + Linear(256,256,bias=False) + tanh` cell; no GRU/LSTM/general recurrent compiler | first-order backward through all timesteps and SGD | not part of the contract |

Pinned-host acceptance is deliberately narrower than CUDA recognition. In the
CPU-only PyTorch 2.4 build, `torch.empty(..., pin_memory=True)` and
`empty_strided(..., pin_memory=True)` select the Vulkan pinned allocator;
CUDA > XPU > PrivateUse1 priority remains unchanged in builds where those
backends are available. `cpu.pin_memory()` and `cpu.is_pinned()` retain their
existing default-device behavior. Stock BackendSelect rejects every
non-CPU-source `pin_memory(device=...)` request before Vulkan dispatch and
reports `is_pinned` false for Vulkan tensors; use the qualified composition
`vk.to('cpu').pin_memory(device='vk')` to pin Vulkan data. F32
`non_blocking=True` copy and `_to_copy(pin_memory=True)` for CPU output are
accepted as described in the async-transfer row; Bool/Long/Double
`non_blocking=True` rejections remain in force. Pinned-host acceptance is targeted
to the tested PyTorch 2.4.0/RADV Renoir configuration and first reverse only; it
establishes neither OS page locking, direct DMA, broader dtype coverage, nor
portability.

Bool and float16 are not declared for the reduction/indexing slice. Vulkan I64
allocation is permitted only while materializing a declared `argmax` result.
Wrong-shaped `argmax.out` tensors are outside this matrix and are explicitly
rejected before temporary allocation or Vulkan dispatch. Invalid device, dtype,
layout, contiguity, and offset metadata are rejected at the same boundary.

The fixed model slices are the **fixed MLP** (`aten::linear`) and **fixed CNN**
(`aten::convolution` plus `aten::relu`, `aten::flatten`, and `aten::linear`). Every consuming
operator must independently declare the layouts it accepts; view construction
does not widen those contracts.

The bounded attention fixture stores one deterministic F32 causal mask buffer with
shape `(batch,128,128)`, containing `0` on and below the diagonal and `-10000`
above it. Masked forward accepts only that module-registered buffer object on the
same Vulkan device; clones, altered values, wrong metadata, and other devices are
rejected before allocation or Vulkan work. No arbitrary Vulkan mask payload is
read back for validation.

Allocating `mm`, `addmm`, and `bmm`, plus the finite stock rank-2-weight Linear
routes above, use the GEMM dispatch by default and do not require an environment
variable. Allocating `addmm` broadcast preparation, including a 1-D bias, uses
the shared GEMM path; retained `addmm.out` behavior is separate. Named direct
matrix witnesses include `matrix.graph.mm.generated`,
`matrix.graph.addmm.generated`, and `matrix.graph.bmm.generated`, with layout,
empty, and scalar execution coverage in `tests/python/test_vulkan_matrix.py`.
`aten::mm.out`, `aten::bmm.out`, and other unlisted GEMM forms remain deferred
and are listed in the deferred schema inventory below. Forward AD/JVP,
`torch.func` transforms, vector-weight Linear, arbitrary matmul geometry, other dtypes,
arbitrary derivative order, and portability are not claimed.

The stock Linear contract is not a general arbitrary-shape matmul or vector-weight
`F.linear` contract. Standalone `addmv`/`addmv_`/`addmv.out`, dot/mv/matmul `out`
forms, and batched/broadcasted shapes outside the finite general-matmul families remain unqualified. The finite
matmul and selected mul/dot/mv reverse directions above do not qualify forward AD/JVP,
`torch.func`/vmap or other transforms, arbitrary higher reverse order, or other
numerical dtypes. Shared in-place `add_` supports
the qualified broadcast required by these Linear routes, but rejects an RHS
broadcast whose expanded storage intersects the destination: the pinned CPU
implementation admits `TooHard` overlap with shape/vector-chunk-dependent results,
so no stable parity contract is established. Exact aliases and proven-disjoint
broadcast storage remain supported; this narrow restriction does not widen the
contract of other in-place operators.

For the finite vector/matmul captures, well-scaled CPU/Vulkan values and
derivatives use `rtol=0.003`, `atol=0.003`. Dot/mv and broadcast-mul mixed
reverse projections use independent nonzero seeds and directions; their CPU
centered differences use `eps=0.001`, `rtol=0.008`, `atol=0.002`. These are
finite fixture comparisons, not bitwise guarantees or broad extreme-value
claims. The existing fixed F32 GEMM policy and its documented CPU overflow-class
distinction remain unchanged.

Durable vector/matmul evidence validation is CPU-only: it checks current fixture/leaf source
hashes and re-executes the deterministic ordinary PyTorch CPU output, first
gradients, selected mixed tensor and independent centered finite difference.
The persisted upstream seed, mixed probe, direction, shapes and epsilon are
source-bound; stored CPU values must reproduce this pinned reference recipe.
The capture/oracle factory's complete local semantic code is included in its
source hash, and its external ATen/autograd dependency is identified by the
PyTorch version and upstream revision. Vulkan componentwise parity keeps the
tolerances above.

Each case retains its extension artifact SHA256, absolute capture path, checkout
HEAD, execution mode and named device/driver as historical provenance. A normal
commit, rebuild or relocation does not invalidate that history. Validation checks
metadata formats and cross-record consistency, rather than equating that history
with the live checkout/build/mode. Live qualification uses the separate explicit
`qualify_vector_matmul_current_runtime` gate and fresh execution; validating an
async record in a sync process is not sync execution evidence. Vector/matmul captures
here remain async Renoir observations; live qualification owns mode-specific gates.
Neither CPU recipe reproducibility nor source hashes establish hardware portability
or registration/dispatch behavior beyond the executed witnesses.

### Public MSE and connected-model evidence

`mse_autograd_cases.py` owns 45 scalar/matrix recipes: scalar, ordinary, empty,
public broadcast, expanded upstream × none/mean/sum × input/target/both selection.
Both `F.mse_loss` and `nn.MSELoss` execute each recipe: 90 source-bound
`mse-ad.*` records per actual mode, with 135 unique logical CPU FD projections
(270 per public-API inventory). All three original bases require grad even for a
selected first reverse; selected second reverse targets input, target and upstream.
Rank-1 forward/first reverse has a separate ordinary conformance witness; its
existence does not extend the selected-second inventory. The autograd vocabulary
`reverse_mse_finite_second_order_witnessed` refers exactly to those named links.
Readable transpose/expanded layouts and direct backward addressing, no_grad,
original-base mutation and malformed preflight are additionally qualified by
`test_vulkan_mse_autograd.py`; broad rank/layout Cartesian coverage is not inferred.

`docs/vulkan_coverage.json` holds async public MSE, numerical Scalar division and
rank-1 MSE first-reverse witnesses alongside historical operator evidence.
`docs/vulkan_mse_sync_coverage.json` holds separately executed sync witnesses.
Historical MSE summary records retain their first-order contract; source/graph
claims come from their dedicated IDs, without rebinding old payloads or binary
identities. General-matmul, Linear and workload evidence remain independent;
model evidence does not extend the schema-v1 workload inventory.

`docs/vulkan_composed_matmul_coverage.json` is a separate schema-v1 model document:
`{schema_version:1, records:{scenario_id+"."+mode:record}}`, exactly 18 records.
The nine source-owned scenarios cover three default-policy momentum-SGD steps in
both reset modes, connected output first reverse, two mixed directions, joint
stock-MSE parameter HVP and three CPU-reference mutation outcomes. The normal
51-parameter module connects folded mm, cross-broadcast bmm, transposed singleton
bmm and mv through public `@`. All original-base tensor trees, optimizer state,
history nodes, independent FD endpoints and phase leaf shapes are CPU-replayed.
Vulkan tensor entries admit only componentwise .003/.003 numerical parity; metadata
and CPU values are strict. FD uses epsilon .001 and .008/.002. Only empty-mean
`loss` may use the named `empty-mean-loss-NaN` exception; arbitrary nonfinite numeric
fields, strings and bool-as-number are rejected.

Phase validation independently replays CPU routes, binds generated matrix/MSE
nodes and manual mean Scalar division, and requires source-established fold,
cross-broadcast and scalar-upstream materialization. Counters precede readback,
with zero fallback and explicit transfer. Semantic source maps cover all local
factories/serializers/capture validators, production loss/Scalar/matrix/layout/
pointwise/reduction/optimizer leaves, runtime orchestration and shader dependencies.
Recorded execution is scoped to RADV Renoir, Mesa 26.2.3-arch1.1; exact extension
identities remain in the evidence payloads and do not imply current-build qualification.

`tools/validate_vulkan_composed_matmul.py --coverage PATH` is CPU-only historical
replay: recorded HEAD/path/mode do not need to match the host. `--live` explicitly
checks the current extension, mode, device, driver and pinned PyTorch reference.
The qualification runner includes offline connected-model validation and the direct ordinary
model node, retaining all previous gates, status policies and the 300s timeout.
Fresh generation compares actual captures in both modes. General-matmul fresh
generation separately live-qualifies its current binary and compares unchanged
semantic arrays/routes/mode/source maps to history; unrelated loss changes do not
require rewriting historical general-matmul binary SHA values. The general-matmul comparison
validates unmodified historical evidence and strictly live-qualifies unmodified
fresh evidence before comparing disposable semantic projections. Only capture
HEAD/path, extension SHA, driver and instance version are omitted from that
cross-generation comparison. Device, hardware, device API, PyTorch reference and
mode, all source maps, numerical arrays, gradients, fixtures, routes and counters
remain exact. The strict live probe still checks the fresh extension, driver and
instance version; old-driver history cannot qualify a new-driver execution.
This bounded comparison policy neither rewrites history nor proves portability.
The final regenerated capability manifest must still match committed bytes.
Capture and guarded
publication procedures are documented in [bootstrap evidence policy](vulkan-bootstrap-evidence.md).

### Finite batched/broadcast general-matmul evidence

`tests/python/general_matmul_cases.py` owns 218 ordinary cases across 29 families,
each through both public spellings. The nine starting families are batch/vector,
vector/batch, batch/matrix, matrix/batch, equal-batch, left/right singleton,
unequal-rank and cross-broadcast. Further witnesses cover equal ranks 4–8,
rank8/vector and rank8/rank3 in both operand orders, matrix-transposed,
nonfoldable leading strides (matrix and vector), offset-fold, expanded zero-stride
reads, cross-transposed matrices, and five empty families. Exact seeds, dimensions,
strides, offsets, selection and output geometry are source-bound in each record.
Inference, both-gradient and true `torch.no_grad` cases execute for every family;
the source-declared selective subset also executes left-only and right-only.

102 cases retain original-base first gradients. Four mixed cases (two fixtures,
two spellings) each capture both opposite-operand directions, with separate
cotangent/probe/perturbation seeds and independently recomputed CPU centered FD.
Tensor tolerances are .003/.003; FD epsilon .001 and tolerances .008/.002.
Profiler routes and synchronized device counters precede result readback;
measured scopes contain zero explicit transfers and zero fallback. Stock
materialization copies are recorded rather than treated as implicit fallback.
The combined CPU profiler set must reproduce the ordinary CPU oracle exactly.
The distinct forward CPU set must independently reproduce a fresh source-owned
CPU forward profile. Forward Vulkan evidence preserves the source/CPU-established
matrix leaf, inverse/shape views and required clone/copy composition. Each fixture
also records a geometry-derived `materialization_minimum`: stock fold and expanded
batch reshapes are tested for alias-preserving viewability, and right-large matrix
folds account for the prescribed output transpose/contiguous step. Cross-broadcast,
cross-transposed and unequal-rank witnesses require two planned copies. Offsets,
zero strides and empty shapes are not alone treated as copy requirements. The
minimum bounds these finite compositions rather than imposing a universal clone
count, an upper bound on backend copies or a profiler event-order contract.
First and each selected mixed phase also retain separately measured CPU profiler
sets; offline replay binds them to the generated derivative closure and requires
the corresponding Vulkan matrix leaves, inverse views and broadcast reductions.
CPU BLAS internals are not device requirements; bounded Vulkan leaf metadata
composition is allowed without inferring universal clone counts.
Acceptance additionally tests CPU-matched saved-version rejection for alias/offset
folds and saved-materialization independence for cross-broadcast, in both original
operand mutation directions. Mutation evidence resides in ordinary acceptance
tests; the capability numerical captures do not claim to replay mutation.

`docs/vulkan_coverage.json` contains the source-owned **async** general-matmul
records alongside historical operator evidence. `docs/vulkan_general_matmul_sync_coverage.json`
contains the separately executed **sync** inventory with identical IDs and source
hashes. Both captures identify the Renoir device/driver, extension and pinned
PyTorch revision. `validate_general_matmul_evidence` is CPU-only historical replay;
`qualify_general_matmul_current_runtime` explicitly checks current artifact,
device/reference and mode. HEAD and extension path remain provenance across human
commits/relocation. The generator deduplicates shared conformance/evidence IDs,
retains both test sources and rejects conflicting fixtures. These observations
do not establish arbitrary higher derivatives, forward AD, transforms, new dtype,
`out=`, composed-model/training coverage or other hardware.

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

## Vulkan Basic Optimizers

The declared optimizer baseline is scalar, non-capturable SGD and Adam over
zero-offset, contiguous, non-overlapping F32 parameters and gradients on
`vk:0`. SGD keeps `momentum_buffer`
resident on Vulkan; Adam keeps `exp_avg` and `exp_avg_sq` resident on Vulkan
while its non-capturable scalar `step` metadata remains on the host. The
baseline rejects Nesterov, AMSGrad, maximize, fused, foreach, differentiable,
capturable, non-F32, non-`vk:0`, unsupported layouts, and mixed-device or
non-contiguous optimizer tensors before dispatch. Optimizer updates do not
implicitly transfer payloads to or from the CPU.

The optimizer execution schemas are `div.Tensor`, `lerp.Scalar_out`,
`lerp_.Scalar`, `sqrt.out`, `add_.Tensor`, `mul_.Scalar`, `addcmul_`,
`addcdiv_`, and `zero_`.

The generic public pointwise in-place forms (`add_`, `sub_`, and `mul_`) are
supported whenever the destination passes alias validation: exact full-tensor
aliasing, and contiguous slice views. Partial overlap, non-exact full overlap,
strided views, and any destination with internal overlap are rejected before
dispatch, and rejection leaves the tensor unmodified with no recorded work. An
active Vulkan optimizer/training step is not required. The optimizer execution
schemas above remain a distinct internal update path.

## Deferred operator-expansion inventory

The following roadmap families are explicitly inventoried for future expansion.
Every schema in this table remains deferred; these entries do not add runtime
registrations or declare support. Schemas promoted to the supported manifest
are removed from this inventory.

| Roadmap family | Deferred schemas |
| --- | --- |
| transfer/creation | `_copy_from_and_resize`, `_local_scalar_dense`, `resize_`, `set_` |
| reductions/indexing | `argmax.out`, `max`, `mean`, `mean.out`, `min`, `prod.out`, `prod.Dimname_out`, `sum.IntList_out`, `sum.default` |
| sigmoid/tanh/GELU | `gelu_backward.grad_input` |
| convolution/pooling backward | `avg_pool2d_backward.grad_input`, `convolution_backward_overrideable`, `upsample_bilinear2d_backward.grad_input`, `upsample_nearest2d_backward.grad_input`, `_upsample_nearest_exact2d_backward.grad_input` |
| cross-entropy/NLL | `nll_loss_forward.output`, `nll_loss_backward.grad_input` |

The cross-entropy/NLL row inventories the deferred log-softmax and NLL pieces
of the composed loss; it does not claim a `cross_entropy` runtime registration.

The promoted scalar and `out=` pointwise F32 forms are limited to `vk:0` and
the 11 schemas listed in the supported-schema comment above. Functional scalar
forms accept documented finite Python scalars; `add` and `sub` accept only the
verified representable `alpha == 1` contract. Tensor operands and `out` tensors
must remain F32 Vulkan tensors with equal shapes; promotion, mixed devices,
unsupported broadcasting, non-finite scalars, and invalid device/dtype contracts
reject before Vulkan work. Exact input/output aliases are permitted where
PyTorch permits them; partial and internally overlapping outputs are rejected.
No hidden CPU fallback or payload readback is part of this declaration.

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
 aten::sum.dim_IntList,aten::mean.dim,aten::argmax.default,aten::amax.default,aten::amax.out,
 aten::amin.default,aten::amin.out,aten::prod.dim_int,aten::prod.int_out,
 aten::_softmax.default,aten::_softmax.out,aten::_log_softmax.default,aten::_log_softmax.out,
 aten::_softmax_backward_data.out,aten::_log_softmax_backward_data.out,
   aten::linear.default,aten::mm.default,aten::addmm.default,aten::addmm.out,
  aten::convolution.default,aten::convolution_backward.default,aten::_adaptive_avg_pool2d.default,
    aten::_adaptive_avg_pool2d_backward.default,aten::mse_loss.default,
   aten::native_batch_norm.default,aten::native_batch_norm_backward.default,
   aten::native_layer_norm.default,aten::native_layer_norm_backward.default,
  aten::nll_loss_forward.default,aten::nll_loss_backward.default,
 aten::mse_loss_backward.default,aten::neg.default,
 aten::abs.default,aten::relu.default,aten::sigmoid.default,aten::tanh.default,
 aten::gelu.default,aten::sigmoid_backward.grad_input,aten::tanh_backward.grad_input,
 aten::gelu_backward.grad_input,aten::add.Tensor,aten::sub.Tensor,
aten::mul.Tensor,aten::as_strided.default,aten::view.default,
aten::_reshape_alias.default,aten::reshape.default,aten::masked_select.default,
aten::add.Scalar,aten::add.Scalar_out,aten::add.out,aten::sub.Scalar,
aten::sub.Scalar_out,aten::sub.out,aten::mul.Scalar,aten::mul.Scalar_out,
aten::mul.out,aten::rsub.Scalar,aten::rsub.Scalar_out,
aten::div.Tensor,aten::lerp.Scalar_out,aten::lerp_.Scalar,aten::sqrt.out,
 aten::add_.Tensor,aten::mul_.Scalar,aten::addcmul_.default,aten::addcdiv_.default,
 aten::zero_.default,aten::_copy_from.default,aten::_to_copy.default,
 aten::copy_.default,aten::empty.memory_format,aten::empty_strided.default,
 aten::ceil.default,aten::_pin_memory.default,aten::is_pinned.default,
 aten::ceil.out,aten::sigmoid.out,aten::tanh.out,aten::gelu.out,
 aten::avg_pool2d.default,aten::avg_pool2d_backward.default,
 aten::max_pool2d_with_indices.default,aten::max_pool2d_with_indices_backward.default,
 aten::upsample_nearest2d.default,aten::upsample_nearest2d_backward.default,
 aten::_upsample_nearest_exact2d.default,aten::_upsample_nearest_exact2d_backward.default,
 aten::upsample_bilinear2d.default,aten::upsample_bilinear2d_backward.default -->

<!-- Vulkan conformance rejected schemas:
aten::neg.out,aten::ne.Tensor,
aten::neg_.default,aten::neg.default,aten::add.Tensor,aten::sum.dim_IntList,
aten::_adaptive_avg_pool2d.default,aten::convolution.default -->

<!-- Vulkan conformance whole-schema rejected schemas:
aten::abs_.default,aten::neg_.default,aten::relu_.default,
aten::ceil_.default,aten::sigmoid_.default,aten::tanh_.default,
aten::gelu_.default -->

<!-- Vulkan conformance deferred schemas:
 aten::concat.default,aten::_copy_from_and_resize.default,
 aten::_local_scalar_dense.default,
 aten::_native_multi_head_attention.default,aten::_native_multi_head_attention.out,
 aten::_transform_bias_rescale_qkv.default,
aten::_upsample_nearest_exact2d.out,aten::_upsample_nearest_exact2d_backward.grad_input,aten::abs.out,
  aten::addcdiv.out,aten::addcmul.out,aten::arange.start_out,aten::argmax.out,aten::atan.out,
aten::avg_pool2d.out,aten::avg_pool2d_backward.grad_input,aten::bernoulli_.float,aten::binary_cross_entropy.default,
aten::binary_cross_entropy_backward.default,aten::binary_cross_entropy_backward.grad_input,
aten::bitwise_and.Tensor_out,aten::bitwise_not.out,aten::bitwise_or.Tensor_out,aten::bitwise_xor.Tensor_out,
aten::bmm.out,aten::cat.out,aten::clamp.out,aten::clamp_min.out,
 aten::convolution_backward_overrideable.default,aten::convolution_overrideable.default,
 aten::div.out,aten::dot.default,
aten::add_.Scalar,aten::mul_.Tensor,aten::sub_.Scalar,aten::sub_.Tensor,
aten::eq.Scalar_out,aten::eq.Tensor_out,aten::exp.out,aten::fill_.Scalar,aten::ge.Scalar_out,
 aten::ge.Tensor_out,aten::gt.Scalar,aten::gt.Scalar_out,
aten::gt.Tensor_out,aten::hardsigmoid.out,aten::hardsigmoid_backward.grad_input,aten::hardswish_.default,
aten::hardswish_backward.default,aten::hardtanh.default,aten::hardtanh_.default,aten::hardtanh_backward.default,
aten::isfinite.out,aten::le.Scalar_out,aten::le.Tensor_out,aten::leaky_relu.out,
aten::leaky_relu_backward.grad_input,aten::log.out,aten::log_sigmoid_backward.default,
aten::log_sigmoid_backward.grad_input,aten::log_sigmoid_forward.default,aten::log_sigmoid_forward.output,
aten::logit.default,aten::logit.out,aten::lt.Scalar,aten::lt.Scalar_out,aten::lt.Tensor_out,aten::max.default,
aten::maximum.out,aten::mean.default,aten::mean.out,aten::min.default,
  aten::minimum.out,aten::mm.out,
aten::native_dropout.default,aten::native_dropout_backward.default,
aten::ne.Scalar_out,aten::ne.Tensor,aten::ne.Tensor_out,aten::neg.out,
  aten::normal_.default,
 aten::pow.Tensor_Scalar_out,aten::reciprocal.out,aten::relu.out,aten::resize_.default,
aten::round.out,aten::set_.source_Storage,
  aten::set_.source_Storage_storage_offset,aten::sgn.out,
  aten::nll_loss_forward.output,aten::nll_loss_backward.grad_input,
 aten::silu.out,aten::silu_backward.grad_input,
aten::sum.IntList_out,aten::sum.default,
aten::threshold_backward.grad_input,aten::uniform_.default,aten::upsample_bilinear2d.out,
aten::upsample_bilinear2d_backward.grad_input,aten::upsample_nearest2d.out,
aten::upsample_nearest2d_backward.grad_input -->
# Autograd evidence vocabulary

`reverse_second_order_witnessed` describes generated reverse-autograd behavior
only on the named `witnesses.reverse_second_order_cases`. Each linked case must
have parity execution evidence in `docs/vulkan_coverage.json` with
`reverse_autograd: {"order": 2, "graph_preserved": true}` recorded only after
CPU/Vulkan first- and seeded second-derivative comparisons. Convolution graph
labels further scope the finite evidence: first reverse is case-linked; second
reverse names the exercised `ggI`, `ggW`, and `ggb` directions; and the selected
third reverse is only `d_g d_x d_w` for the grouped and depthwise witnesses. These
labels do not imply arbitrary higher-order differentiation, forward AD/JVP,
`torch.func`/vmap or other transforms, CUDA parity, HVP, or whole-model training.
The transposed witness stops at second reverse; its nonzero-output-padding
selected-next ordinary-backward route remains CPU-reference rejected. The
vocabulary does not require a constant derivative to itself have history.
Existing primary-input dtype/rank witnesses remain independent of this
case-scoped derivative evidence.

The stock-composite exception is limited to `aten::reshape.default` on the
versioned PyTorch 2.4 `reshape_symint` route. Its required closure names the
direct `_reshape_alias` leaf and the stock-generated clone/`_unsafe_view`
routes through Vulkan `copy_`/`view` leaves; those stock-generated schemas are
not reported as direct C++ registrations. The exception is accepted only with
the named reshape-copy runtime records, primary-input metadata, parity, reverse
witnesses, and actual Vulkan-copy/no-transfer/no-fallback execution evidence.
