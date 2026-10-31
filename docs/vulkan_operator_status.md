# Vulkan operator status inventory

This file lists **every dispatcher schema and overload** in the project's supported stock PyTorch build, one row each, with the Vulkan status that existing repository evidence supports. It is an audit and planning roadmap, not a qualification: it adds no support, widens no contract and runs nothing. Detailed contracts remain in [`docs/vulkan_operator_capability_matrix.md`](vulkan_operator_capability_matrix.md) and [`docs/vulkan_capabilities.json`](vulkan_capabilities.json); this inventory only links to them. It does **not** claim CPU/CUDA parity.

## Baseline

| Field | Value |
|---|---|
| PyTorch package | `2.4.0+cpu` |
| Upstream revision (`torch.version.git_version`) | `e4ee3be4063b7c430974252fdf7db42273388d86` |
| Extraction date | 2026-10-31 |
| Schema count (rows) | 3563 |
| Distinct operator names (`namespace::op`) | 1938 |
| Namespaces | 21 |
| Schema-inventory SHA-256 | `ffc40f8b592adb3d174fccedc14f8a703bf41eda05d79d8194fce49cd6b67349` |

**Extraction method.** In a fresh `python -I` process importing only stock `torch` (the Vulkan extension is never imported; `sys.modules` is asserted free of it), take every name from `torch._C._dispatch_get_all_op_names()` and resolve its full `FunctionSchema` from `torch._C._jit_get_all_schemas()`. The row key is `namespace::operator.overload` with the empty overload written `default`. The digest is SHA-256 over the UTF-8 rows `key<TAB>str(FunctionSchema)` sorted bytewise-by-string, joined with LF plus a trailing LF. Checks: no duplicate keys, every dispatcher key has a schema, no operator name without schemas, and two independent extractions are byte-identical.

**Scope note.** The TorchScript (JIT) registry also contains 781 schemas that are not dispatcher operators (builtins such as `aten::Bool.Tensor`, `prim::*`, `onnx::*`). They are outside the dispatcher universe and are not rows here (digest of that excluded set: `35a337b755884aa2aa406d0397b3e71066fe71d3598333c1b400a00735b8211f`).

## Status summary

| Status | Rows |
|---|---:|
| Supported (scoped) | 63 |
| Partial | 44 |
| Deferred | 86 |
| Unsupported (verified) | 4 |
| Unreviewed | 3366 |
| **Total** | **3563** |


**`Unreviewed` is not a support claim and not a rejection claim.** It means no adequate source/dispatch/capability/test evidence has been recorded yet; it marks an open work item. Missing direct Vulkan registration is never, by itself, a reason for `Unsupported (verified)`.

## Status definitions and rules used

- **Supported (scoped):** a documented Vulkan contract with direct or traced-composite evidence. The *scope* is always narrower than the schema: every such row is bound to F32 (unless the note says otherwise), the stated rank/layout limits, device `vk:0`, and the autograd level in its note. Higher-order, forward-mode and transform behavior is claimed only where a note names a witness.
- **Partial:** the same, but the repository documents a restriction on the schema's own arguments, modes or fixed shape set (listed under *Subset* in the note), or the schema is covered only by a targeted acceptance row.
- **Deferred:** the capability manifest declares the schema `deferred_contract` / `deferred_before_vulkan`. It is not a synonym for "not found".
- **Unsupported (verified):** positive rejection evidence (a registered `reject_*` handler plus a rejection test). Only the three in-place overloads below qualify.
- **Unreviewed:** everything else.

Evidence sources: `docs/vulkan_capabilities.json` (192 entries, read directly; `tools/validate_vulkan_capabilities.py` imports the built Vulkan extension, so it was deliberately not run for this CPU-only inventory), direct `PrivateUse1` registration inventory from `tools/validate_vulkan_capabilities.py`, the capability matrix, focused tests named in each manifest entry, and `git log` pickaxe history.

**Commit(s) are introduction points**, not a full change history. `registration` is the earliest commit adding the `m.impl("…")` line; `declaration` is the earliest commit adding the schema to the capability manifest. Later widening of a contract is not enumerated. Composite routes with no direct registration (`matmul`, `linear`, `reshape`) show the declaration commit only. A row with no implemented support shows `—`. Nothing here substitutes the current HEAD for an implementation commit.

**Known source inconsistencies (recorded, not resolved here).** The matrix's "Deferred operator-expansion inventory" table still lists some schemas that the manifest marks supported (for example `sum.default`, `mean`, `gelu_backward.grad_input`, `argmax.out`); this inventory follows the machine-readable manifest. The manifest entry `aten::isfinite.out` (`schema_absent_from_pytorch_dispatcher`) is an audited absence from the 2.4.0 dispatcher; it is therefore not a row. The matrix documents/qualifies `add` and `sub` with `alpha == 1`, but source inspection shows that `add.Scalar`, `add.Tensor`, `sub.Tensor`, `add_.Tensor`, and `sub_.Tensor` forward non-unit alpha values to the Vulkan operation (`src/vulkan/operators/add.cpp:254-258,267-275,306-311`; `src/vulkan/operators/sub.cpp:15-18`; `src/vulkan/operators/out.cpp:434-475`). Those non-unit cases are **not qualified** here; the inventory does not claim they are rejected or supported. Scalar/out handlers that explicitly guard alpha remain limited to `alpha == 1`.

Each namespace section ends with a signature appendix keyed by the exact same schema key as the table.

## Contents

- [`_c10d_functional`](#c10dfunctional-namespace) — 13 schemas
- [`_c10d_functional_autograd`](#c10dfunctionalautograd-namespace) — 3 schemas
- [`_dtensor`](#dtensor-namespace) — 1 schemas
- [`_quantized`](#quantized-namespace) — 20 schemas
- [`_test`](#test-namespace) — 3 schemas
- [`aten`](#aten-namespace) — 3127 schemas
- [`c10d`](#c10d-namespace) — 20 schemas
- [`debugprims`](#debugprims-namespace) — 1 schemas
- [`inductor`](#inductor-namespace) — 5 schemas
- [`mkl`](#mkl-namespace) — 2 schemas
- [`mkldnn`](#mkldnn-namespace) — 16 schemas
- [`mkldnn_prepacked`](#mkldnnprepacked-namespace) — 2 schemas
- [`onednn`](#onednn-namespace) — 10 schemas
- [`prepacked`](#prepacked-namespace) — 8 schemas
- [`prims`](#prims-namespace) — 128 schemas
- [`profiler`](#profiler-namespace) — 4 schemas
- [`quantization`](#quantization-namespace) — 2 schemas
- [`quantized`](#quantized-namespace) — 170 schemas
- [`rngprims`](#rngprims-namespace) — 1 schemas
- [`sparse`](#sparse-namespace) — 6 schemas
- [`static_runtime`](#staticruntime-namespace) — 21 schemas

<a id="c10dfunctional-namespace"></a>

## `_c10d_functional` namespace (13 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `_c10d_functional::all_gather_into_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_gather_into_tensor_coalesced.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_gather_into_tensor_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_reduce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_reduce_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_reduce_coalesced.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_reduce_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::all_to_all_single.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::broadcast.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::broadcast_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::reduce_scatter_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::reduce_scatter_tensor_coalesced.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional::wait_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>_c10d_functional</code> (13)</summary>

```text
_c10d_functional::all_gather_into_tensor.default  _c10d_functional::all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor
_c10d_functional::all_gather_into_tensor_coalesced.default  _c10d_functional::all_gather_into_tensor_coalesced(Tensor[] inputs, int group_size, str group_name) -> Tensor[]
_c10d_functional::all_gather_into_tensor_out.default  _c10d_functional::all_gather_into_tensor_out(Tensor input, int group_size, str group_name, *, Tensor(a!) out) -> Tensor(a!)
_c10d_functional::all_reduce.default  _c10d_functional::all_reduce(Tensor input, str reduce_op, str group_name) -> Tensor
_c10d_functional::all_reduce_.default  _c10d_functional::all_reduce_(Tensor(a!) input, str reduce_op, str group_name) -> Tensor(a!)
_c10d_functional::all_reduce_coalesced.default  _c10d_functional::all_reduce_coalesced(Tensor[] inputs, str reduce_op, str group_name) -> Tensor[]
_c10d_functional::all_reduce_coalesced_.default  _c10d_functional::all_reduce_coalesced_(Tensor[](a!) inputs, str reduce_op, str group_name) -> Tensor[](a!)
_c10d_functional::all_to_all_single.default  _c10d_functional::all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor
_c10d_functional::broadcast.default  _c10d_functional::broadcast(Tensor input, int src, str group_name) -> Tensor
_c10d_functional::broadcast_.default  _c10d_functional::broadcast_(Tensor(a!) input, int src, str group_name) -> Tensor(a!)
_c10d_functional::reduce_scatter_tensor.default  _c10d_functional::reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor
_c10d_functional::reduce_scatter_tensor_coalesced.default  _c10d_functional::reduce_scatter_tensor_coalesced(Tensor[] inputs, str reduce_op, int group_size, str group_name) -> Tensor[]
_c10d_functional::wait_tensor.default  _c10d_functional::wait_tensor(Tensor tensor) -> Tensor
```

</details>

<a id="c10dfunctionalautograd-namespace"></a>

## `_c10d_functional_autograd` namespace (3 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `_c10d_functional_autograd::all_gather_into_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional_autograd::all_to_all_single.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_c10d_functional_autograd::reduce_scatter_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>_c10d_functional_autograd</code> (3)</summary>

```text
_c10d_functional_autograd::all_gather_into_tensor.default  _c10d_functional_autograd::all_gather_into_tensor(Tensor input, int group_size, str group_name) -> Tensor
_c10d_functional_autograd::all_to_all_single.default  _c10d_functional_autograd::all_to_all_single(Tensor input, SymInt[] output_split_sizes, SymInt[] input_split_sizes, str group_name) -> Tensor
_c10d_functional_autograd::reduce_scatter_tensor.default  _c10d_functional_autograd::reduce_scatter_tensor(Tensor input, str reduce_op, int group_size, str group_name) -> Tensor
```

</details>

<a id="dtensor-namespace"></a>

## `_dtensor` namespace (1 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `_dtensor::shard_dim_alltoall.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>_dtensor</code> (1)</summary>

```text
_dtensor::shard_dim_alltoall.default  _dtensor::shard_dim_alltoall(Tensor input, int gather_dim, int shard_dim, str group_name) -> Tensor
```

</details>

<a id="quantized-namespace"></a>

## `_quantized` namespace (20 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `_quantized::add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv2d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv2d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv3d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv3d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv_transpose1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv_transpose1d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv_transpose2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv_transpose2d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::conv_transpose3d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear_prepack_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear_prepack_fp16_legacy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::linear_prepack_legacy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::wrapped_fbgemm_linear_fp16_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>_quantized</code> (20)</summary>

```text
_quantized::add.default  _quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
_quantized::conv2d.default  _quantized::conv2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv2d_prepack.default  _quantized::conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
_quantized::conv2d_relu.default  _quantized::conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv3d.default  _quantized::conv3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv3d_prepack.default  _quantized::conv3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase
_quantized::conv3d_relu.default  _quantized::conv3d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv_transpose1d.default  _quantized::conv_transpose1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv_transpose1d_prepack.default  _quantized::conv_transpose1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
_quantized::conv_transpose2d.default  _quantized::conv_transpose2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
_quantized::conv_transpose2d_prepack.default  _quantized::conv_transpose2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
_quantized::conv_transpose3d_prepack.default  _quantized::conv_transpose3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase
_quantized::linear.default  _quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
_quantized::linear_dynamic.default  _quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y
_quantized::linear_prepack.default  _quantized::linear_prepack(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack
_quantized::linear_prepack_fp16.default  _quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack
_quantized::linear_prepack_fp16_legacy.default  _quantized::linear_prepack_fp16_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack
_quantized::linear_prepack_legacy.default  _quantized::linear_prepack_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack
_quantized::wrapped_fbgemm_linear_fp16_weight.default  _quantized::wrapped_fbgemm_linear_fp16_weight(Tensor X, Tensor W, Tensor B, int out_channel) -> Tensor
_quantized::wrapped_fbgemm_pack_gemm_matrix_fp16.default  _quantized::wrapped_fbgemm_pack_gemm_matrix_fp16(Tensor W) -> Tensor
```

</details>

<a id="test-namespace"></a>

## `_test` namespace (3 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `_test::cat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_test::get_first.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `_test::leaky_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>_test</code> (3)</summary>

```text
_test::cat.default  _test::cat(Tensor[] inputs) -> Tensor
_test::get_first.default  _test::get_first(str[][] _0) -> str _0
_test::leaky_relu.default  _test::leaky_relu(Tensor self, float v=0.01) -> Tensor
```

</details>

<a id="aten-namespace"></a>

## `aten` namespace (3127 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `aten::__and__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__and__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__iand__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__iand__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ilshift__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ilshift__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ior__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ior__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__irshift__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__irshift__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ixor__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__ixor__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__lshift__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__lshift__.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__lshift__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__lshift__.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__or__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__or__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__rshift__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__rshift__.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__rshift__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__rshift__.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__xor__.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::__xor__.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool2d.default` | Partial | `43a213e` (registration), `852df95` (declaration) | Subset: output size (1,1) only; non-global and empty forms rejected (matrix L35). Scope: float32; rank 4; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::_adaptive_avg_pool2d.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `aten._adaptive_avg_pool2d.global`, `aten._adaptive_avg_pool2d.global.strided`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_adaptive_avg_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool2d_backward.default` | Partial | `43a213e` (registration), `852df95` (declaration) | Subset: grad shape (N,C,1,1) only; non-global forms rejected (matrix L35). Scope: float32; rank 4; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::_adaptive_avg_pool2d_backward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `aten._adaptive_avg_pool2d.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_adaptive_avg_pool2d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_adaptive_avg_pool3d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_batch_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_add_relu_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_addmm_activation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_addmm_activation.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_aminmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_aminmax.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_aminmax.dim_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_aminmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_foreach_non_finite_check_and_unscale.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_foreach_non_finite_check_and_unscale.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_foreach_non_finite_check_and_unscale_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_update_scale.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_update_scale.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_amp_update_scale_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_assert_async.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_assert_async.msg` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_assert_scalar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_assert_tensor_metadata.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_autocast_to_full_precision.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_autocast_to_reduced_precision.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_impl_index.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_impl_index_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_no_update.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_no_update.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_with_update.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_with_update.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_batch_norm_with_update_functional.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Byte.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Char.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Double.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Float.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Half.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Int.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Long.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cast_Short.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cdist_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cdist_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cdist_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cdist_forward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cholesky_solve_helper.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cholesky_solve_helper.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_choose_qparams_per_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_chunk_cat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_chunk_cat.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_coalesce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_coalesce.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_coalesced.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_coalesced.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_compute_linear_combination.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_compute_linear_combination.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conj.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conj_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conj_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conj_physical.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conj_physical.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conv_depthwise2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_conv_depthwise2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convert_indices_from_coo_to_csr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convert_indices_from_coo_to_csr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convert_indices_from_csr_to_coo.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convert_indices_from_csr_to_coo.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convert_weight_to_int4pack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convolution.deprecated` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convolution_double_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_convolution_mode.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_copy_from.default` | Partial | `0308432` (registration), `852df95` (declaration) | Subset: tensor-plumbing copy primitive; same F32/declared-device limits as copy_ (matrix L45). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::_copy_from.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `transfer.copy_from.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_copy_from.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_copy_from_and_resize.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::_copy_from_and_resize.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_copy_from_and_resize.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cslt_compress.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cslt_sparse_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cslt_sparse_mm_search.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss_backward.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_ctc_loss_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_ctc_loss.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_ctc_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_ctc_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_init_dropout_state.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_init_dropout_state.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn_flatten_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cudnn_rnn_flatten_weight.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cufft_clear_plan_cache.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cufft_get_plan_cache_max_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cufft_get_plan_cache_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cufft_set_plan_cache_max_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cummax_helper.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_cummin_helper.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_debug_has_internal_overlap.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_dimI.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_dimV.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_dim_arange.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_dirichlet_grad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_dirichlet_grad.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_efficient_attention_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_efficient_attention_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_efficientzerotensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_efficientzerotensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_dense_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_dense_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_forward_only.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_forward_only.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_per_sample_weights_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_per_sample_weights_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_embedding_bag_sparse_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_empty_affine_quantized.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_empty_affine_quantized.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_empty_per_channel_affine_quantized.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_empty_per_channel_affine_quantized.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_euclidean_dist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_euclidean_dist.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_channel_affine.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_channel_affine.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_channel_affine_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_tensor_affine.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_tensor_affine.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_learnable_per_tensor_affine_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_c2c.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_c2c.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_c2r.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_c2r.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_r2c.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fft_r2c.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fill_mem_eff_dropout_mask_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_flash_attention_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_flash_attention_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foobar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foobar.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_abs.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_abs.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_abs_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_acos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_acos.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_acos_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_add_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcdiv_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_addcmul_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_asin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_asin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_asin_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_atan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_atan.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_atan_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_ceil.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_ceil.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_ceil_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_max_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_clamp_min_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_copy_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cos.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cos_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cosh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_cosh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_div_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erf_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erfc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erfc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_erfc_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_exp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_exp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_exp_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_expm1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_expm1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_expm1_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_floor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_floor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_floor_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_frac.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_frac.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_frac_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lerp_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lgamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lgamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_lgamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log10.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log10.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log10_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log1p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log1p.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log1p_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log2_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_log_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_max.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_max.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_maximum_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_minimum_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_mul_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_neg.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_neg.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_neg_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_norm.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_norm.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.ScalarAndTensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_pow_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_reciprocal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_reciprocal.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_reciprocal_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_round.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_round.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_round_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sigmoid.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sigmoid.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sigmoid_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sign.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sign.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sign_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sin_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sinh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sinh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sqrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sqrt.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sqrt_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.List_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.ScalarList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub_.List` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_sub_.ScalarList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tan.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tan_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tanh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_tanh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_trunc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_trunc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_trunc_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_zero.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_zero.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_foreach_zero_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_functional_assert_async.msg` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_functional_assert_scalar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_functional_sym_constrain_range.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_functional_sym_constrain_range_for_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adagrad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adagrad.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adagrad_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam.tensor_lr_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adam_.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw.tensor_lr_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_adamw_.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_dropout.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_moving_avg_obs_fq_helper.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_moving_avg_obs_fq_helper.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_moving_avg_obs_fq_helper_functional.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sdp_choice.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd.tensor_lr_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fused_sgd_.tensor_lr` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fw_primal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fw_primal_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_fw_primal_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_gather_sparse_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_grid_sampler_2d_cpu_fallback.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_grid_sampler_2d_cpu_fallback.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_grid_sampler_2d_cpu_fallback_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_has_compatible_shallow_copy_type.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_has_same_storage_numel.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_bin_edges.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_bin_edges.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_from_bin_cts.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_from_bin_cts.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_from_bin_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_histogramdd_from_bin_tensors.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_index_put_impl.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_index_put_impl.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_index_put_impl_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_int_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_int_mm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_is_all_true.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_is_any_true.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_is_zerotensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_jagged_to_padded_dense_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_lazy_clone.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_check_errors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_det.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_det.result` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_eigh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_eigh.eigenvalues` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_eigvals.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_slogdet.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_slogdet.sign` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_solve_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_solve_ex.result` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_svd.U` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_linalg_svd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_local_scalar_dense.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::_local_scalar_dense.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_log_softmax.default` | Partial | `c8e8e39` (registration), `852df95` (declaration) | Subset: witnessed as part of the fixed logits (2,3), class dim 1 loss slice (matrix L37). Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::_log_softmax.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.log-softmax.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_log_softmax.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::_log_softmax.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.log-softmax.dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_log_softmax_backward_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_log_softmax_backward_data.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `backward_kernel`; out `contiguous_out_required`. Evidence: [`aten::_log_softmax_backward_data.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.log-softmax.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_logcumsumexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_logcumsumexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_lstm_mps.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_lstm_mps.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_lu_with_info.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_dep_token.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_dual.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_dual_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_dual_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_per_channel_quantized_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_per_channel_quantized_tensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_per_tensor_quantized_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_make_per_tensor_quantized_tensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_scale.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_scale.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_softmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_softmax_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_masked_softmax_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mixed_dtypes_linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mkldnn_reshape.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mkldnn_reshape.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mkldnn_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mkldnn_transpose.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mkldnn_transpose_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mps_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mps_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mps_convolution_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_mps_convolution_transpose.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit.no_stats` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit.no_stats_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit_functional.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit_no_training.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_batch_norm_legit_no_training.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_native_multi_head_attention.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::_native_multi_head_attention.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_native_multi_head_attention.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::_native_multi_head_attention.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_neg_view.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_neg_view_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_neg_view_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_compute_contiguous_strides_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_from_padded.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_from_padded.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_from_padded_and_nested_example.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_from_padded_and_nested_example.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_jagged_dummy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_lengths.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_ragged_idx.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_values.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_values_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_get_values_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_select_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_sum_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_from_mask.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_from_mask.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_from_mask_left_aligned.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_from_tensor_list.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_from_tensor_list.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_size.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_softmax_with_shape.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_storage_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_storage_offsets.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_strides.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_tensor_strides.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_buffer.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_buffer_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_buffer_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_jagged.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_jagged_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nested_view_from_jagged_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_new_zeros_with_same_feature_meta.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_new_zeros_with_same_feature_meta.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nnpack_available.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nnpack_spatial_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nnpack_spatial_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_nnz.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pack_padded_sequence.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pack_padded_sequence.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pack_padded_sequence_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pad_circular.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pad_enum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pad_packed_sequence.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_padded_dense_to_jagged_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pdist_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pdist_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pdist_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pdist_forward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_pin_memory.default` | Partial | `1c088f2` (registration) — no declaration commit (not in capability manifest) | Subset: targeted pinned-host acceptance on PyTorch 2.4.0/RADV Renoir: persistent mapped host-visible Vulkan-backed storage; non-resizable; not OS page-locking or a DMA guarantee (matrix L46). No capability-manifest entry. Scope: CPU F32 tensors pinned for `vk:0` on PyTorch 2.4.0 / RADV Renoir only. Evidence: registration src/vulkan_pinned_memory.cpp:67-68; contract in [matrix](vulkan_operator_capability_matrix.md) (targeted pinned-host acceptance row); tests tests/python/test_vulkan_copy_contracts.py, test_vulkan_async_transfer.py (pin_memory/is_pinned cases). |
| `aten::_pin_memory.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_prelu_kernel.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_prelu_kernel_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_print.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_propagate_xla_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_remove_batch_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_reshape_alias.default` | Supported (scoped) | `d8e8177` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_view_alias`. Evidence: [`aten::_reshape_alias.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `view.reshape-alias.metadata`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_reshape_alias_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_reshape_alias_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_reshape_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_reshape_from_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_resize_output.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_resize_output.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_resize_output_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_rowwise_prune.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sample_dirichlet.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sample_dirichlet.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_saturate_weight_to_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_attention_math.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_cudnn_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_cudnn_attention_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_efficient_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_efficient_attention_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_flash_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_flash_attention_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_flash_attention_for_cpu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_dot_product_flash_attention_for_cpu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_scaled_mm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_segment_reduce_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_segment_reduce_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_shape_as_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_slow_conv2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_slow_conv2d_backward.output_mask` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_slow_conv2d_backward.output_mask_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_slow_conv2d_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_slow_conv2d_forward.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sobol_engine_draw.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sobol_engine_ff_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sobol_engine_initialize_state_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sobol_engine_scramble_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_softmax.default` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::_softmax.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.softmax.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_softmax.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::_softmax.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.softmax.dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_softmax_backward_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_softmax_backward_data.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `backward_kernel`; out `contiguous_out_required`. Evidence: [`aten::_softmax_backward_data.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.softmax.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_sparse_addmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_addmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_broadcast_to.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_broadcast_to_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_broadcast_to_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_bsc_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_bsr_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_compressed_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_compressed_tensor_with_dims.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_coo_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_coo_tensor_with_dims.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_coo_tensor_with_dims.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_coo_tensor_with_dims_and_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_coo_tensor_with_dims_and_tensors.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csc_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csr_prod.dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csr_prod.dim_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csr_sum.dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csr_sum.dim_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_csr_tensor_unsafe.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax_backward_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_log_softmax_backward_data.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mask_projection.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mask_projection.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mm.reduce` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mm_reduce_impl.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_mm_reduce_impl_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_addmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_apply.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_apply_dense.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_semi_structured_tile.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax_backward_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_softmax_backward_data.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sparse_matmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sparse_matmul.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum.dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum.dim_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_sparse_sum_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_spdiags.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_spdiags.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_stack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_stack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_standard_gamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_standard_gamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_standard_gamma_grad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_standard_gamma_grad.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_ambiguous_defaults.a` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_ambiguous_defaults.b` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch.fullcoverage` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch.fullcoverage_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch.ntonly` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch_view.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch_view_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_autograd_multiple_dispatch_view_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_check_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_functorch_fallback.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_functorch_fallback.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_filled_intlist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_filled_intlist.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_floatlist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_floatlist.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_intlist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_optional_intlist.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_parallel_materialize.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_serialization_subcmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_string_default.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_warn_in_autograd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_test_warn_in_autograd.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_differentiable_gru_cell_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_differentiable_lstm_cell_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_gru_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_gru_cell.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_gru_cell_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_gru_cell_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_lstm_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_lstm_cell.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_lstm_cell_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_lstm_cell_backward_impl.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_thnn_fused_lstm_cell_backward_impl.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_copy.default` | Partial | `c3fbf3a` (registration), `852df95` (declaration) | Subset: dtype conversion limited to the formatter-compatible F32->Double path; H2D/Vulkan-output pin_memory=True rejected, a recorded deviation (matrix L45,L47). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::_to_copy.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `transfer.to_copy.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::_to_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_cpu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_dense.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_dense.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse.sparse_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse.sparse_dim_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_bsc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_bsc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_bsr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_bsr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_csc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_csc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_csr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_csr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_to_sparse_semi_structured.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_transform_bias_rescale_qkv.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::_transform_bias_rescale_qkv.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_transform_bias_rescale_qkv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_transformer_encoder_layer_fwd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_transformer_encoder_layer_fwd.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_trilinear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_trilinear.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_triton_multi_head_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_triton_multi_head_attention.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_triton_scaled_dot_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_triton_scaled_dot_attention.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unique.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unique.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unique2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unique2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unpack_dual.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unsafe_index.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unsafe_index_put.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_unsafe_view.default` | Unreviewed | — | named as a tested leaf of the reshape copy route (matrix L43); no capability entry; not individually audited. |
| `aten::_unsafe_view.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bicubic2d_aa.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bicubic2d_aa.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bicubic2d_aa.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bicubic2d_aa_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bicubic2d_aa_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bilinear2d_aa.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bilinear2d_aa.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bilinear2d_aa.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bilinear2d_aa_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_bilinear2d_aa_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact1d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact1d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact1d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact2d.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::_upsample_nearest_exact2d.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_upsample_nearest_exact2d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact2d_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::_upsample_nearest_exact2d_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::_upsample_nearest_exact3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact3d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_upsample_nearest_exact3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_use_cudnn_ctc_loss.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_use_cudnn_ctc_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_use_cudnn_rnn_flatten_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_compressed_sparse_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_bsc_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_bsr_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_compressed_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_coo_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_csc_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_validate_sparse_csr_tensor_args.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_values.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_values_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_values_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_version.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_int4pack_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_int8pack_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm_differentiable_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm_interface.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm_interface.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm_interface_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::_weight_norm_interface_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::abs.default` | Supported (scoped) | `044819c` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::abs.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.abs.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::abs.out` | Supported (scoped) | `6c926fc` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::abs.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.abs-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::abs_.default` | Unsupported (verified) | — | Positive rejection evidence: `reject_abs_inplace` registered for the in-place overload (src/vulkan/operators/unary.cpp:249-278) and asserted rejected by tests/python/test_vulkan_unary.py (in-place rejection tests); declared `explicit_source_rejection` in [`aten::abs_.default`](vulkan_capabilities.json). |
| `aten::absolute.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::absolute.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::absolute_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acos.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acos_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acosh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::acosh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_avg_pool3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adaptive_max_pool3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::add.Scalar` | Partial | `6dec4d3` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::add.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `scalar.add.float32`, `arithmetic.autograd.add-scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::add.Scalar_out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::add.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.add.scalar.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::add.Tensor` | Partial | `508df47` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 1; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::add.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `binary.add.float32`, `arithmetic.autograd.add-tensor`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::add.out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::add.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.add.tensor.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::add_.Scalar` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::add_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `inplace.add-scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::add_.Tensor` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::add_.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.add.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::addbmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addbmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addbmm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addcdiv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addcdiv.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::addcdiv.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::addcdiv_.default` | Supported (scoped) | `d19ce73` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::addcdiv_.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.addcdiv.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::addcmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addcmul.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::addcmul.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::addcmul_.default` | Supported (scoped) | `d19ce73` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::addcmul_.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.addcmul.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::addmm.default` | Supported (scoped) | `43a213e` (registration), `852df95` (declaration) | Scope: float32; rank 2; contiguous/non-overlapping/zero-offset; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::addmm.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `addmm.forward`, `matrix.graph.addmm.generated`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::addmm.out` | Supported (scoped) | `43a213e` (registration), `852df95` (declaration) | Scope: float32; rank 2; contiguous/non-overlapping/zero-offset; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::addmm.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `addmm.out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::addmm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addmv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addmv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addmv_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::addr_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::adjoint.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::affine_grid_generator.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::affine_grid_generator.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::affine_grid_generator_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::alias.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::alias_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::alias_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::align_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::align_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::align_to.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::align_to.ellipsis_idx` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.all_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.dims` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.dims_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::all.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::allclose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::alpha_dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::alpha_dropout_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::amax.default` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::amax.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.amax.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::amax.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::amax.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.amax.dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::amin.default` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::amin.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.amin.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::amin.out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::amin.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.amin.dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::aminmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::aminmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::angle.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::angle.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.all_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.dims` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.dims_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::any.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arange.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arange.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arange.start` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arange.start_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::arange.start_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::arange.start_step` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccos.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccos_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccosh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arccosh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsin_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsinh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arcsinh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan2_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctan_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctanh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::arctanh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argmax.default` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_differentiable`. Evidence: [`aten::argmax.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `indexing.argmax.dim`, `indexing.argmax.optional-dim.keepdim`, `indexing.argmax.strided`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::argmax.out` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::argmax.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.argmax-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::argmin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argmin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argsort.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argsort.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argsort.stable` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argsort.stable_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::argwhere.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::as_strided.default` | Supported (scoped) | `d8e8177` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_view_alias`. Evidence: [`aten::as_strided.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `view.as-strided.metadata`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::as_strided_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::as_strided_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::as_strided_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::as_strided_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::as_strided_scatter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asin_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asinh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::asinh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atan.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::atan.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::atan2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atan2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atan2_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atan_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atanh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atanh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_1d.Sequence` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_2d.Sequence` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_3d.Sequence` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::atleast_3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::avg_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::avg_pool2d.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; rank 3 or 4 (empty batch allowed); full kernel/stride/padding/ceil_mode geometry; strided, channels-last and expanded operands read in place; outputs follow CPU memory formats; first-order autograd only via generated autograd (`create_graph`, forward AD, `torch.func`, `vmap` unclaimed). `count_include_pad` and `divisor_override` supported. Evidence: [`aten::avg_pool2d.default`](vulkan_capabilities.json), tests test_vulkan_pooling_general.py, test_vulkan_conformance.py; cases `pooling.avg-pool2d*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::avg_pool2d.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::avg_pool2d.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::avg_pool2d_backward.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; rank 3 or 4 (empty batch allowed); full kernel/stride/padding/ceil_mode geometry; strided, channels-last and expanded operands read in place; outputs follow CPU memory formats; first-order autograd only via generated autograd (`create_graph`, forward AD, `torch.func`, `vmap` unclaimed). Gather formulation sums in ATen's ascending output order. Evidence: [`aten::avg_pool2d_backward.default`](vulkan_capabilities.json), tests test_vulkan_pooling_general.py, test_vulkan_conformance.py; cases `pooling.avg-pool2d*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::avg_pool2d_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::avg_pool2d_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::avg_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::avg_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::avg_pool3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::avg_pool3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::baddbmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::baddbmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::baddbmm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bartlett_window.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bartlett_window.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bartlett_window.periodic` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bartlett_window.periodic_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_backward_elemt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_backward_elemt.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_backward_reduce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_backward_reduce.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_elemt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_elemt.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_gather_stats.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_gather_stats.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_gather_stats_with_counts.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_gather_stats_with_counts.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_stats.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_stats.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_update_stats.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::batch_norm_update_stats.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.float_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli.p` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bernoulli_.float` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::bernoulli_.float`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::bilinear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::binary_cross_entropy.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::binary_cross_entropy.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::binary_cross_entropy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::binary_cross_entropy_backward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::binary_cross_entropy_backward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::binary_cross_entropy_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::binary_cross_entropy_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::binary_cross_entropy_with_logits.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::binary_cross_entropy_with_logits.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bincount.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bincount.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::binomial.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::binomial.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and.Tensor_out` | Supported (scoped) | `43a213e` (registration), `852df95` (declaration) | Scope: bool; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::bitwise_and.Tensor_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.bitwise-and-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::bitwise_and_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_and_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_left_shift_.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_not.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_not.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::bitwise_not.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::bitwise_not_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::bitwise_or.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::bitwise_or_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_or_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_right_shift_.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::bitwise_xor.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::bitwise_xor_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bitwise_xor_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::blackman_window.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::blackman_window.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::blackman_window.periodic` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::blackman_window.periodic_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::block_diag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::block_diag.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bmm.default` | Supported (scoped) | `a7acdc5` (registration, declaration) | Scope: float32; rank 3; contiguous/transposed-contiguous/non-overlapping; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::bmm.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `attention.bmm.forward_backward`, `matrix.graph.bmm.generated`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::bmm.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::bmm.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::broadcast_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::broadcast_to.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bucketize.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bucketize.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bucketize.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::bucketize.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::can_cast.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cartesian_prod.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cat.default` | Supported (scoped) | `a44dce8` (registration, declaration) | Scope: float32; rank 1-4; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::cat.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `cat.rank1.forward`, `cat.rank2.forward`, `cat.rank3.forward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::cat.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cat.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cat.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::cat.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::cauchy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cauchy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cauchy_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ccol_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ccol_indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ccol_indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cdist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ceil.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::ceil.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::ceil.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::ceil.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::ceil_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::celu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::celu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::celu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::chain_matmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::chain_matmul.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::chalf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::channel_shuffle.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::channel_shuffle.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky_inverse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky_inverse.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cholesky_solve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::choose_qparams_optimized.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::chunk.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::clamp.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::clamp_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_max_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_min.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_min.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_min.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_min.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::clamp_min.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::clamp_min_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clamp_min_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clip_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::clone.default` | Unreviewed | — | named as a tested leaf of the reshape copy route (matrix L43); no capability entry; not individually audited. |
| `aten::clone.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::coalesce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::col2im.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::col2im.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::col_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::col_indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::col_indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::column_stack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::column_stack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::combinations.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::complex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::complex.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concat.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `be13aea`; direct PrivateUse1 registration absent. Evidence: [`aten::concat.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::concat.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concat.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concat.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concatenate.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concatenate.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concatenate.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::concatenate.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conj.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conj_physical.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conj_physical.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conj_physical_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::constant_pad_nd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::constant_pad_nd.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::contiguous.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv1d.padding` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv2d.padding` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv3d.padding` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_depthwise3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_depthwise3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_tbc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_tbc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_tbc_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_transpose1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_transpose2d.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::conv_transpose3d.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::convolution.default` | Partial | `43a213e` (registration), `852df95` (declaration) | Subset: 2-D rank-4 NCHW only; positive stride/dilation, nonnegative padding; ordinary-forward output_padding ignored (matrix L34). Scope: float32; rank 4; strided/non-overlapping; autograd `reverse_selected_third_order_witnessed`. Evidence: [`aten::convolution.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `convolution.forward`, `convolution.forward.strided`, `convolution.cnn.forward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::convolution_backward.default` | Partial | `43b7d4d` (registration), `852df95` (declaration) | Subset: 2-D rank-4 NCHW only; ordinary backward requires output_padding [0,0] (matrix L34). Scope: float32; rank 4; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::convolution_backward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `convolution.groups.2.grad-input`, `convolution.groups.2.grad-weight`, `convolution.groups.2.grad-bias`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::convolution_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::convolution_backward_overrideable.default` | Partial | `d7d6d21` (registration), `852df95` (declaration) | Subset: 2-D rank-4 NCHW only; separate all-true numerical witness (matrix L34). Scope: float32; rank 4; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::convolution_backward_overrideable.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `convolution.backward-overrideable.bias-present.mask-111`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::convolution_backward_overrideable.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::convolution_overrideable.default` | Partial | `1ba0bb8` (registration), `852df95` (declaration) | Subset: 2-D rank-4 NCHW only; first-reverse graph witness (matrix L34). Scope: float32; rank 4; strided/non-overlapping; autograd `reverse_first_order_graph_witnessed`. Evidence: [`aten::convolution_overrideable.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `convolution.graph.overrideable.first`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::convolution_overrideable.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copy_.default` | Partial | `f5a18a5` (registration), `852df95` (declaration) | Subset: declared CPU<->vk:0 and Vulkan-to-Vulkan F32 copies; non-F32 non_blocking rejected; F32 non_blocking per matrix async-transfer row (matrix L45,L47). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::copy_.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `transfer.copy.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::copy_sparse_to_sparse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copy_sparse_to_sparse.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copy_sparse_to_sparse_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::copysign_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::corrcoef.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cos.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cos_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cosh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cosh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cosine_embedding_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cosine_similarity.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::count_nonzero.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::count_nonzero.dim_IntList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::count_nonzero.dim_IntList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::count_nonzero.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cov.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cross.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cross.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cross_entropy_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::crow_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::crow_indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::crow_indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ctc_loss.IntList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ctc_loss.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_affine_grid_generator.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_affine_grid_generator.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_affine_grid_generator_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_affine_grid_generator_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_batch_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_batch_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_batch_norm_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_batch_norm_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_add_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_add_relu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_relu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_convolution_transpose.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_grid_sampler.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_grid_sampler.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_grid_sampler_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_grid_sampler_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cudnn_is_acceptable.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummax.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummax.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummaxmin_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummin.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummin.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cummin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod_.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumprod_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumsum_.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumulative_trapezoid.dx` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::cumulative_trapezoid.x` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::deg2rad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::deg2rad.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::deg2rad_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dense_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dequantize.self` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dequantize.self_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dequantize.tensors` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dequantize.tensors_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::det.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::detach.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::detach_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::detach_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::detach_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diag.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diag_embed.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diag_embed.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagflat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diagonal_scatter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diff.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::diff.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::digamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::digamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::digamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dist.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.Scalar_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.Scalar_mode_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.Tensor` | Supported (scoped) | `c3fbf3a` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::div.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.div.scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::div.Tensor_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::div.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::div.out_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div_.Scalar` | Partial | `509827c` (registration, declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 0-2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::div_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `scalar-division.rank0`, `scalar-division.rank1`, `scalar-division.rank2`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::div_.Scalar_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::div_.Tensor_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.Scalar_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.Tensor_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide.out_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide_.Scalar_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::divide_.Tensor_mode` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dot.default` | Supported (scoped) | `9467be4` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::dot.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py, test_vulkan_vector_matmul_capability_evidence.py; cases `g1.dot.rank1.forward`, `g1.dot.rank1.grad-a-wrt-b`, `g1.dot.rank1.grad-b-wrt-a`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::dot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dropout_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dsplit.array` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dsplit.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dstack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::dstack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::einsum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::elu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::elu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::elu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::elu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::elu_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_bag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_bag.padding_idx` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_dense_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_dense_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_renorm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_renorm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_renorm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::embedding_sparse_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty.memory_format` | Supported (scoped) | `a0aeba6` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::empty.memory_format`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `transfer.empty.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::empty.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_permuted.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_permuted.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_quantized.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_quantized.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::empty_strided.default` | Supported (scoped) | `0308432` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::empty_strided.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `transfer.empty_strided.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::empty_strided.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eq.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eq.Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::eq.Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::eq.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eq.Tensor_out` | Supported (scoped) | `43a213e` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::eq.Tensor_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.eq-tensor-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::eq_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eq_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::equal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erf_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfc_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfinv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfinv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::erfinv_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exp.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::exp.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::exp2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exp2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exp2_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exp_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expand.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expand_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expand_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expand_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expm1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expm1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::expm1_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exponential.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exponential.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::exponential_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eye.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eye.m` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eye.m_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::eye.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_channel_affine.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_channel_affine_cachemask.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_channel_affine_cachemask.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_channel_affine_cachemask_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_tensor_affine.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_tensor_affine.tensor_qparams` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_tensor_affine_cachemask.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_tensor_affine_cachemask.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fake_quantize_per_tensor_affine_cachemask_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_linear_fp16_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_linear_fp16_weight_fp32_activation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_linear_int8_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_linear_int8_weight_fp32_activation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_linear_quantize_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_pack_gemm_matrix_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_pack_quantized_matrix.KN` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fbgemm_pack_quantized_matrix.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::feature_alpha_dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::feature_alpha_dropout_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::feature_dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::feature_dropout_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fftfreq.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fftfreq.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_fftshift.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_hfftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ifftshift.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_ihfftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_irfftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfft.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfft2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfft2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfftfreq.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfftfreq.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfftn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fft_rfftn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill_.Scalar` | Partial | `d19ce73` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::fill_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `inplace.fill-scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::fill_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fill_diagonal_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fix.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fix.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fix_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flatten.DimnameList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flatten.named_out_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flatten.using_ints` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flatten.using_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flatten_dense_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flip.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flip.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fliplr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::flipud.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::float_power_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::floor_divide_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmax.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fmod_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frac.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frac.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frac_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool2d.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool3d.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fractional_max_pool3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frexp.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frexp.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frobenius_norm.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::frobenius_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::from_file.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::from_file.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::full_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::fused_moving_avg_obs_fake_quant.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gather.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gather.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gather.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gather.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gather_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gcd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gcd.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gcd_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ge.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ge.Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::ge.Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::ge.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ge.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::ge.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::ge_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ge_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gelu.default` | Partial | `e6ca761` (registration), `852df95` (declaration) | Subset: approximate="tanh" with the 0.044715 polynomial only (matrix L24). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::gelu.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.gelu.tanh.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::gelu.out` | Partial | (this change; pending commit) | Subset: `approximate="tanh"` only (matrix L24, L40); other values rejected before Vulkan work. Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `strided_out_supported`. Evidence: [`aten::gelu.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_out.py; cases `unary.gelu-out`, `unary.gelu-out.empty`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::gelu_.default` | Unsupported (verified) | — | Positive rejection evidence: `reject_gelu_inplace` registered for the in-place overload (src/vulkan/operators/unary.cpp) and asserted rejected by tests/python/test_vulkan_out.py (`test_gelu_inplace_form_stays_rejected_after_gelu_out_registration`); declared `explicit_source_rejection` in [`aten::gelu_.default`](vulkan_capabilities.json). |
| `aten::gelu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gelu_backward.grad_input` | Partial | `e6ca761` (registration), `852df95` (declaration) | Subset: GELU slice requires approximate="tanh" (matrix L24); grad_input form only. Scope: float32; rank 1; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::gelu_backward.grad_input`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.gelu.tanh.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::geometric.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::geometric.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::geometric_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::geqrf.a` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::geqrf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ger.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ger.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::get_gradients.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_backward_jvp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_backward_jvp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_jvp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::glu_jvp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.array` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.scalararray` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.scalarint` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.scalarrayarray` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.scalarrayint` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.tensorarray` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gradient.tensorarrayint` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::greater_equal_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_2d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::grid_sampler_3d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::group_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gru.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gru.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gru_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gt.Scalar` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::gt.Scalar`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::gt.Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::gt.Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::gt.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gt.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::gt.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::gt_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::gt_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic_alpha` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic_alpha_beta` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic_alpha_beta_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic_alpha_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hamming_window.periodic_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hann_window.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hann_window.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hann_window.periodic` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hann_window.periodic_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardshrink.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardshrink.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardshrink_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardshrink_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardsigmoid.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardsigmoid.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardsigmoid.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardsigmoid_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardsigmoid_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardsigmoid_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardsigmoid_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardswish.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardswish.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardswish_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardswish_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardswish_backward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardswish_backward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardswish_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardtanh.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardtanh.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardtanh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hardtanh_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardtanh_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardtanh_backward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::hardtanh_backward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::hardtanh_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::heaviside.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::heaviside.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::heaviside_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hinge_embedding_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogram.bin_ct` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogram.bin_ct_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogram.bins_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogram.bins_tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogramdd.TensorList_bins` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogramdd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::histogramdd.int_bins` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hsplit.array` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hsplit.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hspmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hspmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hstack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hstack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::huber_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::huber_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::huber_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::huber_loss_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hypot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hypot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::hypot_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::i0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::i0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::i0_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igammac.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igammac.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::igammac_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::im2col.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::im2col.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::imag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_add.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_add.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_add_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_copy.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_copy_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_copy_.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.Dimname_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.Dimname_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.int_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.int_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.int_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill.int_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill_.Dimname_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill_.Dimname_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill_.int_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_fill_.int_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_put.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_put.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_put_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_reduce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_reduce.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_reduce_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_select.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_select.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_select.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_select.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::index_select_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::infinitely_differentiable_gelu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::inner.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::inner.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::instance_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::int_repr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::int_repr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::inverse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::inverse.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_coalesced.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_complex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_conj.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_distributed.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_floating_point.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_inference.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_leaf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_neg.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_nonzero.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_pinned.default` | Partial | `1c088f2` (registration) — no declaration commit (not in capability manifest) | Subset: allocator/DataPtr-provenance recognition; zero-byte allocation unrecognized; foreign wrappers false (matrix L46). No capability-manifest entry. Scope: CPU F32 tensors pinned for `vk:0` on PyTorch 2.4.0 / RADV Renoir only. Evidence: registration src/vulkan_pinned_memory.cpp:67-68; contract in [matrix](vulkan_operator_capability_matrix.md) (targeted pinned-host acceptance row); tests tests/python/test_vulkan_copy_contracts.py, test_vulkan_async_transfer.py (pin_memory/is_pinned cases). |
| `aten::is_same_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_set_to.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_signed.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::is_vulkan_available.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isclose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isfinite.default` | Supported (scoped) | `b271038` (registration, declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`. Evidence: [`aten::isfinite.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.isfinite`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::isin.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isin.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isin.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isin.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isin.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isin.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isinf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isinf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isnan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isnan.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isneginf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isneginf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isposinf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isposinf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::isreal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::istft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::item.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.beta` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.beta_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.periodic` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kaiser_window.periodic_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kl_div.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kron.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kron.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kthvalue.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kthvalue.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kthvalue.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::kthvalue.values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::l1_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::layer_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lcm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lcm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lcm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ldexp.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ldexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ldexp_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::le.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::le.Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::le.Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::le.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::le.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::le.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::le_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::le_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::leaky_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::leaky_relu.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::leaky_relu.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::leaky_relu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::leaky_relu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::leaky_relu_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::leaky_relu_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::lerp.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lerp.Scalar_out` | Partial | `d19ce73` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::lerp.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.lerp.out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::lerp.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lerp.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lerp_.Scalar` | Partial | `d19ce73` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::lerp_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.lerp.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::lerp_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::less_equal_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lgamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lgamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lgamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lift.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lift.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lift_fresh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lift_fresh_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lift_fresh_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cholesky.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cholesky.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cholesky_ex.L` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cholesky_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cond.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cond.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cond.p_str` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cond.p_str_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cross.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_cross.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_det.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_det.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_diagonal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eig.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eig.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigh.eigvals` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigvals.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigvals.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigvalsh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_eigvalsh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_householder_product.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_householder_product.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_inv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_inv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_inv_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_inv_ex.inverse` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_factor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_factor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_factor_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_factor_ex.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_ldl_solve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lstsq.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lstsq.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_factor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_factor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_factor_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_factor_ex.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_lu_solve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matmul.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_exp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_exp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_norm.str_ord` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_norm.str_ord_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_power.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_power.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.atol_rtol_float` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.atol_rtol_float_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.atol_rtol_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.atol_rtol_tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.out_tol_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_matrix_rank.tol_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_multi_dot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_multi_dot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_norm.ord_str` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_norm.ord_str_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.atol_rtol_float` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.atol_rtol_float_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.atol_rtol_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.atol_rtol_tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.out_rcond_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_pinv.rcond_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_qr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_qr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_slogdet.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_slogdet.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve_ex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve_ex.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve_triangular.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_solve_triangular.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_svd.U` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_svd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_svdvals.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_svdvals.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_tensorinv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_tensorinv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_tensorsolve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_tensorsolve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_vander.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_vecdot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_vecdot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_vector_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linalg_vector_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linear.default` | Partial | `1ba0bb8` (registration), `852df95` (declaration) — composite route, no direct registration; declaration commit only | Subset: stock composite; rank-2 weight (4,3) witnesses, input ranks 1-8, named routes only (matrix L25; STOCK_LINEAR_ROUTE_CONTRACT). Scope: float32; rank 1-8; contiguous/non-overlapping/zero-offset; autograd `first_order_or_none`. Evidence: [`aten::linear.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_linear.py, test_vulkan_operator_capabilities.py; cases `linear.forward`, `linear.forward.strided`, `linear.attention.3d`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::linear.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linear_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linear_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::linspace.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::log.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::log10.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log10.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log10_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log1p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log1p.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log1p_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log2_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_normal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_normal.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_normal_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_sigmoid.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_sigmoid.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_sigmoid_backward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::log_sigmoid_backward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::log_sigmoid_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::log_sigmoid_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::log_sigmoid_forward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::log_sigmoid_forward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::log_sigmoid_forward.output` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::log_sigmoid_forward.output`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::log_softmax.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_softmax.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::log_softmax.int_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logaddexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logaddexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logaddexp2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logaddexp2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logcumsumexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logcumsumexp.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logcumsumexp.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logcumsumexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logdet.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_and.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_and.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_and_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_not.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_not.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_not_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_or.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_or.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_or_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_xor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_xor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logical_xor_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logit.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::logit.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::logit.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::logit.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::logit_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logit_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logit_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Tensor_Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logspace.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logsumexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logsumexp.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logsumexp.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::logsumexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lstm.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lstm.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lstm_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lstm_mps_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lstm_mps_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lt.Scalar` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::lt.Scalar`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::lt.Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::lt.Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::lt.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lt.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::lt.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::lt_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lt_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lu_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lu_solve.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lu_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::lu_unpack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mH.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mT.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::margin_ranking_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_fill_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_scatter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_scatter_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_scatter_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_select.default` | Partial | `43a213e` (registration), `852df95` (declaration) | Subset: F32 value views; mask must be same-shaped contiguous zero-offset bool on vk:0 (matrix L48). Scope: float32; rank 1-2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::masked_select.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `masked-select.bool-mask`, `masked-select.strided-value-view`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::masked_select.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::masked_select_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matmul.default` | Supported (scoped) | `9467be4` (declaration) — composite route, no direct registration; declaration commit only | Scope: float32; rank 1-8; strided/non-overlapping; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::matmul.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_general_matmul.py, test_vulkan_general_matmul_capability_evidence.py, test_vulkan_operator_capabilities.py, test_vulkan_vector_matmul_capability_evidence.py; cases `g1.matmul.vv.torch-matmul`, `g1.matmul.mv.torch-matmul`, `g1.matmul.vm.torch-matmul`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::matmul.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matmul_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matmul_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matrix_H.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matrix_exp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matrix_exp_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matrix_power.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::matrix_power.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::max.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::max.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.dim_max` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.names_dim_max` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max.unary_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool1d_with_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool2d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool2d_with_indices.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; rank 3 or 4 (empty batch allowed); full kernel/stride/padding/ceil_mode geometry; strided, channels-last and expanded operands read in place; outputs follow CPU memory formats; first-order autograd only via generated autograd (`create_graph`, forward AD, `torch.func`, `vmap` unclaimed). Dilation supported; int64 plane-relative `indices`; ATen's tie/NaN/`-inf` rules reproduced. Evidence: [`aten::max_pool2d_with_indices.default`](vulkan_capabilities.json), tests test_vulkan_pooling_general.py, test_vulkan_conformance.py; cases `pooling.max-pool2d*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::max_pool2d_with_indices.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool2d_with_indices_backward.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; rank 3 or 4 (empty batch allowed); full kernel/stride/padding/ceil_mode geometry; strided, channels-last and expanded operands read in place; outputs follow CPU memory formats; first-order autograd only via generated autograd (`create_graph`, forward AD, `torch.func`, `vmap` unclaimed). Equal to ATen's scatter-add bit for bit for every in-range index (hand-supplied included); out-of-range indices (undefined on CPU) contribute nothing by Vulkan contract. Evidence: [`aten::max_pool2d_with_indices_backward.default`](vulkan_capabilities.json), tests test_vulkan_pooling_general.py, test_vulkan_conformance.py; cases `pooling.max-pool2d*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::max_pool2d_with_indices_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool3d_with_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool3d_with_indices.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool3d_with_indices_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_pool3d_with_indices_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_unpool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_unpool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_unpool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::max_unpool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::maximum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::maximum.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::maximum.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::mean.default` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_backward`. Evidence: [`aten::mean.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.mean.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mean.dim` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::mean.dim`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.mean.dim`, `reduction.mean.optional-dim`, `reduction.mean.empty-dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mean.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mean.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mean.out` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_differentiable`; out `contiguous_out_required`. Evidence: [`aten::mean.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.mean.out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::median.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::median.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::median.dim_values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::median.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::median.names_dim_values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::median.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::meshgrid.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::meshgrid.indexing` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::min.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::min.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.dim_min` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.names_dim_min` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::min.unary_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::minimum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::minimum.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::minimum.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::miopen_batch_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_batch_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_batch_norm_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_batch_norm_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution_add_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_convolution_transpose.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_depthwise_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_depthwise_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_rnn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_rnn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_rnn_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::miopen_rnn_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mish.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mish.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mish_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mish_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_adaptive_avg_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_adaptive_avg_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_adaptive_avg_pool2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_adaptive_avg_pool2d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_convolution.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_convolution.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward_input.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward_input.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward_weights.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_linear_backward_weights.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool2d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_max_pool3d_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_reorder_conv2d_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_reorder_conv2d_weight.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_reorder_conv3d_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_reorder_conv3d_weight.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_rnn_layer.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_rnn_layer.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_rnn_layer_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mkldnn_rnn_layer_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mm.default` | Supported (scoped) | `082e5d6` (registration), `852df95` (declaration) | Scope: float32; rank 2; contiguous/non-overlapping/zero-offset; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::mm.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `mm.forward`, `matrix.graph.mm.generated`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mm.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::mm.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::mode.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mode.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mode.dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mode.values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::moveaxis.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::moveaxis.intlist` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::movedim.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::movedim.intlist` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mps_convolution_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mps_convolution_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mps_convolution_transpose_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mps_convolution_transpose_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mse_loss.default` | Supported (scoped) | `3e13200` (registration), `852df95` (declaration) | Scope: float32; rank 0-2; strided/non-overlapping; autograd `reverse_mse_finite_second_order_witnessed`. Evidence: [`aten::mse_loss.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_mse_autograd.py, test_vulkan_mse_capability_evidence.py, test_vulkan_operator_capabilities.py; cases `loss.mse.none`, `loss.mse.sum`, `loss.mse.mean`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mse_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mse_loss_backward.default` | Supported (scoped) | `3e13200` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::mse_loss_backward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `loss.mse.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mse_loss_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::msort.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::msort.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mul.Scalar` | Partial | `6dec4d3` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::mul.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `scalar.mul.float32`, `arithmetic.autograd.mul-scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mul.Scalar_out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::mul.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.mul.scalar.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mul.Tensor` | Supported (scoped) | `6dec4d3` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::mul.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py, test_vulkan_vector_matmul_capability_evidence.py; cases `g1.mul.broadcast.rank0-rank1`, `g1.mul.broadcast.rank2`, `g1.mul.broadcast.rank3-rank2`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mul.out` | Supported (scoped) | `6c926fc` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::mul.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.mul.tensor.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mul_.Scalar` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::mul_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.mul.scalar.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mul_.Tensor` | Supported (scoped) | `6c926fc` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::mul_.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `inplace.mul-tensor`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::multi_margin_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multi_margin_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multi_margin_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multi_margin_loss_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multilabel_margin_loss_forward.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multinomial.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multinomial.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multiply.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multiply.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multiply.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multiply_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::multiply_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mv.default` | Supported (scoped) | `9467be4` (registration, declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `reverse_finite_second_order_witnessed`. Evidence: [`aten::mv.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py, test_vulkan_vector_matmul_capability_evidence.py; cases `g1.mv.rank2-rank1.forward`, `g1.mv.rank2-rank1.grad-m-wrt-x`, `g1.mv.rank2-rank1.grad-x-wrt-m`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::mv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mvlgamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mvlgamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::mvlgamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nan_to_num.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nan_to_num.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nan_to_num_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmean.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmean.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.dim_values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.names_dim_values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanmedian.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanquantile.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanquantile.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanquantile.scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nanquantile.scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nansum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nansum.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::narrow.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::narrow.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::narrow_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::narrow_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_batch_norm.default` | Partial | `2c8d712` (registration), `852df95` (declaration) | Subset: fixed contiguous training inputs (2,4)/(2,4,2,2), affine (4,), momentum 0.1, eps 1e-5; eval, partial affine, other shapes rejected (matrix L36). Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::native_batch_norm.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `normalization.native-batch-norm`, `normalization.native-batch-norm.2d`, `normalization.native-batch-norm.4d`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::native_batch_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_batch_norm_backward.default` | Partial | `2c8d712` (registration), `852df95` (declaration) | Subset: same fixed training slice; output mask [true,true,true] only (matrix L36). Scope: float32; rank 2; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::native_batch_norm_backward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `normalization.native-batch-norm-backward`, `normalization.native-batch-norm-backward.4d`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::native_batch_norm_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_channel_shuffle.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_dropout.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::native_dropout.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::native_dropout.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_dropout_backward.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::native_dropout_backward.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::native_dropout_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_group_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_group_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_group_norm_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_group_norm_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_layer_norm.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; strided input (densified by a Vulkan copy); rank 2-4 witnessed, up to 8; any `normalized_shape` length; optional weight/bias; any `eps` incl. 0/negative/non-finite (CPU NaN/Inf results); empty batches and zero-width normalization follow CPU; autograd first-order only via generated `NativeLayerNormBackward0` (`create_graph`, forward AD, `torch.func`, `vmap` unclaimed). Evidence: [`aten::native_layer_norm.default`](vulkan_capabilities.json), tests test_vulkan_layer_norm.py, test_vulkan_conformance.py; cases `normalization.native-layer-norm*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::native_layer_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_layer_norm_backward.default` | Supported (scoped) | (this change; pending commit) | Scope: float32 `vk:0`; all autograd-produced `output_mask` values; strict `grad_out`/`mean`/`rstd` shape checks; expanded or non-contiguous `grad_out`/input densified by a Vulkan copy; M = 0 gives zero `dW`/`dB`; first-order only. Evidence: [`aten::native_layer_norm_backward.default`](vulkan_capabilities.json), tests test_vulkan_layer_norm.py, test_vulkan_conformance.py; cases `normalization.native-layer-norm-backward*`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::native_layer_norm_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_norm.ScalarOpt_dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_norm.ScalarOpt_dim_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::native_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ne.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ne.Scalar_out` | Partial | `43a213e` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::ne.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.ne-scalar-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::ne.Tensor` | Supported (scoped) | `c3fbf3a` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`. Evidence: [`aten::ne.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `comparison.ne-tensor`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::ne.Tensor_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::ne.Tensor_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::ne_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ne_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::neg.default` | Supported (scoped) | `044819c` (registration), `852df95` (declaration) | Scope: float32; rank 1-2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::neg.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.neg.float32`, `unary.neg.float32.strided`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::neg.out` | Supported (scoped) | `6c926fc` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::neg.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.neg-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::neg_.default` | Unsupported (verified) | — | Positive rejection evidence: `reject_neg_inplace` registered for the in-place overload (src/vulkan/operators/unary.cpp:249-278) and asserted rejected by tests/python/test_vulkan_unary.py (in-place rejection tests); declared `explicit_source_rejection` in [`aten::neg_.default`](vulkan_capabilities.json). |
| `aten::negative.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::negative.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::negative_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nested_to_padded_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_empty.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_empty.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_empty_strided.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_empty_strided.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_full.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_full.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_ones.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_ones.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_zeros.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::new_zeros.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nextafter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nextafter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nextafter_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss2d_forward.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nll_loss_backward.default` | Partial | `2c8d712` (registration), `852df95` (declaration) | Subset: same fixed slice: reduction mean, no weight, ignore_index=-100 (matrix L37). Scope: float32; rank 1; contiguous/non-overlapping/zero-offset; autograd `backward_kernel`. Evidence: [`aten::nll_loss_backward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `classification.nll-backward`, `classification.nll-backward.wide`, `classification.nll-backward.sum`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::nll_loss_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::nll_loss_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::nll_loss_forward.default` | Partial | `2c8d712` (registration), `852df95` (declaration) | Subset: logits (2,3), I64 labels (2,), no weight, reduction mean, ignore_index=-100 only (matrix L37). Scope: float32; rank 2; contiguous/non-overlapping/zero-offset; autograd `first_order_or_none`. Evidence: [`aten::nll_loss_forward.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `classification.nll-forward`, `classification.nll-forward.wide`, `classification.nll-forward.none.wide`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::nll_loss_forward.output` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::nll_loss_forward.output`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::nll_loss_nd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nonzero.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nonzero.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nonzero_numpy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nonzero_static.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nonzero_static.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.ScalarOpt_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.ScalarOpt_dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.ScalarOpt_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.ScalarOpt_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.names_ScalarOpt_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.names_ScalarOpt_dim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.names_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::norm_except_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.Tensor_float` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.Tensor_float_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.float_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.float_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.float_float` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.float_float_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::normal_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::normal_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::normal_functional.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::not_equal_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nuclear_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nuclear_norm.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nuclear_norm.dim_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::nuclear_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::numpy_T.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::one_hot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ones_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::orgqr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::orgqr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ormqr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ormqr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::outer.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::outer.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::output_nr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pad_sequence.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pairwise_distance.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pdist.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::permute.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::permute_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::permute_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pin_memory.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pinverse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pixel_shuffle.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pixel_shuffle.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pixel_unshuffle.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pixel_unshuffle.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::poisson.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::poisson.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::poisson_nll_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::polar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::polar.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::polygamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::polygamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::polygamma_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::positive.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow.Tensor_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow.Tensor_Scalar_out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::pow.Tensor_Scalar_out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::pow.Tensor_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow.Tensor_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::pow_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::prelu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::prod.Dimname_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::prod.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::prod.dim_Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::prod.dim_int` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::prod.dim_int`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.prod.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::prod.int_out` | Supported (scoped) | `c8e8e39` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::prod.int_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.prod.dim`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::prod.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::promote_types.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::put.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::put.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::put_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_per_channel_axis.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_per_channel_scales.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_per_channel_scales.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_per_channel_zero_points.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_per_channel_zero_points.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_scale.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::q_zero_point.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::qr.Q` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::qr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::qscheme.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantile.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantile.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantile.scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantile.scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_channel.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_channel.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.tensor_qparams` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.tensor_qparams_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.tensors` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor.tensors_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantize_per_tensor_dynamic.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_batch_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_batch_norm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_gru.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_gru.data_legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_gru.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_gru.input_legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_gru_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_lstm.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_lstm.data_legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_lstm.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_lstm.input_legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_lstm_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_max_pool3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_rnn_relu_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::quantized_rnn_tanh_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rad2deg.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rad2deg.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rad2deg_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.generator` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.generator_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.generator_with_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.generator_with_names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rand_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.generator` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.generator_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.low` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.low_generator` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.low_generator_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.low_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint_like.low_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint_like.low_dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randint_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.generator` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.generator_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.generator_with_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.generator_with_names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randn_like.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.from` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.from_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.to` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random.to_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random_.from` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::random_.to` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randperm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randperm.generator` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randperm.generator_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::randperm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::range.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::range.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::range.out_` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::range.step` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::ravel.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::real.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reciprocal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reciprocal.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::reciprocal.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::reciprocal_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::record_stream.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::refine_names.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad1d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad1d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reflection_pad3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::relu.default` | Supported (scoped) | `044819c` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::relu.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.relu.float32.empty`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::relu.out` | Supported (scoped) | `6c926fc` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; out `contiguous_out_required`. Evidence: [`aten::relu.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.relu-out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::relu6.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::relu6_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::relu_.default` | Unsupported (verified) | — | Positive rejection evidence: `reject_relu_inplace` registered for the in-place overload (src/vulkan/operators/unary.cpp:249-278) and asserted rejected by tests/python/test_vulkan_unary.py (in-place rejection tests); declared `explicit_source_rejection` in [`aten::relu_.default`](vulkan_capabilities.json). |
| `aten::remainder.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder.Scalar_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::remainder_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rename.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rename_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::renorm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::renorm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::renorm_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat_interleave.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat_interleave.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat_interleave.self_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::repeat_interleave.self_int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad1d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad1d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::replication_pad3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::requires_grad_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::reshape.default` | Supported (scoped) | `6d44f9c` (registration), `852df95` (declaration) — composite route, no direct registration; declaration commit only | Scope: float32; rank 2; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::reshape.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `view.reshape.float32`, `view.reshape.copy.trainable-seed`, `view.reshape.offset-copy.second-order`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::reshape_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::resize_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::resize_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_as.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_as_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_as_sparse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_as_sparse.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resize_as_sparse_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resolve_conj.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::resolve_neg.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::result_type.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::result_type.Scalar_Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::result_type.Scalar_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::result_type.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::retain_grad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::retains_grad.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rms_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_relu.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_relu.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_relu_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_tanh.data` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_tanh.input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rnn_tanh_cell.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::roll.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::roll.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rot90.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rot90.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::round.decimals` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::round.decimals_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::round.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::round.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::round.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::round_.decimals` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::round_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::row_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::row_indices_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::row_indices_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::row_stack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::row_stack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_with_noise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_with_noise.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_with_noise_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_with_noise_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rrelu_with_noise_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rsqrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rsqrt.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rsqrt_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rsub.Scalar` | Partial | `6dec4d3` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::rsub.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `scalar.rsub.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::rsub.Scalar_out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::rsub.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.rsub.scalar.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::rsub.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::rsub.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scalar_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scalar_tensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scaled_dot_product_attention.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.dimname_src` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.dimname_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.reduce` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.reduce_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.src` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.src_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.value_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.value_reduce` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter.value_reduce_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_.reduce` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_.src` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_.value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_.value_reduce` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_add.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_add.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_add_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_reduce.two` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_reduce.two_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::scatter_reduce_.two` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::searchsorted.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::searchsorted.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::searchsorted.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::searchsorted.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::segment_reduce.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::segment_reduce.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_copy.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_copy.int_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::select_scatter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::selu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::selu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Storage` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Storage_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Storage_storage_offset` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Storage_storage_offset_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set.source_Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set_.default` | Unreviewed | — | direct PrivateUse1 registration exists (src/vulkan_allocator.cpp); no capability entry. The matrix async row only records that the `set_` lease is not discriminated; not audited as an operator contract. |
| `aten::set_.source_Storage` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::set_.source_Storage`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::set_.source_Storage_storage_offset` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration present. Evidence: [`aten::set_.source_Storage_storage_offset`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::set_.source_Tensor` | Unreviewed | — | direct PrivateUse1 registration exists (src/vulkan_allocator.cpp); no capability entry. The matrix async row only records that the `set_` lease is not discriminated; not audited as an operator contract. |
| `aten::set_.source_Tensor_storage_offset` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::set_data.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sgn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sgn.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::sgn.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::sgn_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sigmoid.default` | Supported (scoped) | `e6ca761` (registration), `852df95` (declaration) | Scope: float32; rank 1-2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::sigmoid.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.sigmoid.float32`, `unary.sigmoid.float32.empty`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sigmoid.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::sigmoid.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::sigmoid_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::sigmoid_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::sigmoid_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sigmoid_backward.grad_input` | Supported (scoped) | `e6ca761` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::sigmoid_backward.grad_input`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.sigmoid.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sign.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sign.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sign_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::signbit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::signbit.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::silu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::silu.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::silu.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::silu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::silu_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::silu_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::silu_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::sin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sin.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sin_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinc_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinh.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sinh_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::size.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::size.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_copy.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_copy.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_inverse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slice_scatter.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slogdet.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slogdet.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv3d_forward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv3d_forward.output` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_dilated2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_dilated2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_dilated3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_dilated3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_transpose2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_transpose2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_transpose3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::slow_conv_transpose3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::smm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::smooth_l1_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::smooth_l1_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::smooth_l1_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::smooth_l1_loss_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::soft_margin_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::soft_margin_loss.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::soft_margin_loss_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::soft_margin_loss_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softmax.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softmax.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softmax.int_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softplus.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softplus.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softplus_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softplus_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softshrink.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softshrink.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softshrink_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::softshrink_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.dimname_stable` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.dimname_values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.dimname_values_stable` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.stable` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sort.values_stable` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_bsc_tensor.ccol_row_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_bsc_tensor.ccol_row_value_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_bsr_tensor.crow_col_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_bsr_tensor.crow_col_value_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_compressed_tensor.comp_plain_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_compressed_tensor.comp_plain_value_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_coo_tensor.indices` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_coo_tensor.indices_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_coo_tensor.size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_coo_tensor.size_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_csc_tensor.ccol_row_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_csc_tensor.ccol_row_value_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_csr_tensor.crow_col_value` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_csr_tensor.crow_col_value_size` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_mask.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_mask.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize_and_clear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize_and_clear.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_resize_and_clear_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_sampled_addmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sparse_sampled_addmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_airy_ai.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_airy_ai.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_j0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_j0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_j1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_j1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_y0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_y0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_y1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_bessel_y1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_t.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_u.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_v.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_chebyshev_polynomial_w.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_digamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_digamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_entr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_entr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erf.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfcx.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfcx.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfinv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_erfinv.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_exp2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_exp2.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_expit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_expit.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_expm1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_expm1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammainc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammainc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammaincc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammaincc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammaln.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_gammaln.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_h.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_hermite_polynomial_he.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i0e.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i0e.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i1e.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_i1e.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_laguerre_polynomial_l.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_legendre_polynomial_p.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_log1p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_log1p.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_log_ndtr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_log_ndtr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_log_softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_logit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_logit.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_logsumexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_logsumexp.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_i0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_i0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_i1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_i1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_k0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_k0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_k1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_modified_bessel_k1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_multigammaln.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_multigammaln.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_ndtr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_ndtr.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_ndtri.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_ndtri.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_polygamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_polygamma.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_psi.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_psi.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_round.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_round.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_scaled_modified_bessel_k0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_scaled_modified_bessel_k0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_scaled_modified_bessel_k1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_scaled_modified_bessel_k1.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_t.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_u.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_v.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.n_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.n_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.x_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_shifted_chebyshev_polynomial_w.x_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_sinc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_sinc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_spherical_bessel_j0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_spherical_bessel_j0.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.other_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.other_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.self_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlog1py.self_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.other_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.other_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.self_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_xlogy.self_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.other_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.other_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.self_scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::special_zeta.self_scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split.sizes` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split_copy.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split_copy.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split_with_sizes.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split_with_sizes_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::split_with_sizes_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sqrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sqrt.out` | Supported (scoped) | `d19ce73` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::sqrt.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.sqrt.out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sqrt_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::square.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::square.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::square_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze.dims` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_.dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_.dims` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.dim_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.dims` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.dims_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::squeeze_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sspaddmm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sspaddmm.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::stack.default` | Partial | `a7acdc5` (registration, declaration) | Subset: matching contiguous 2-D matrices, insertion dim 0 or 1, Vulkan-to-Vulkan copies only (matrix L30). Scope: float32; rank 2; contiguous/non-overlapping; autograd `first_order_backward`. Evidence: [`aten::stack.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `sequence.stack.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::stack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.correction` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.correction_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.correction_names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.correction_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.correction` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.correction_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.correction_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::std_mean.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::stft.center` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::stft.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::stride.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::stride.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sub.Scalar` | Partial | `6dec4d3` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::sub.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `scalar.sub.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sub.Scalar_out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::sub.Scalar_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.sub.scalar.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sub.Tensor` | Partial | `6dec4d3` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 2; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::sub.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `binary.sub.float32.strided`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sub.out` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`; out `contiguous_out_required`. Evidence: [`aten::sub.out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `out.sub.tensor.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sub_.Scalar` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: finite representable Python scalars only; non-finite scalars rejected; add/sub/rsub documented at alpha == 1 (matrix L317-325). Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::sub_.Scalar`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `inplace.sub-scalar`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sub_.Tensor` | Partial | `6c926fc` (registration), `852df95` (declaration) | Subset: documented pointwise contract accepts only the verified representable alpha == 1 (matrix L317-325); equal-shape F32 operands. Scope: float32; rank 2; strided/non-overlapping; autograd `not_applicable`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::sub_.Tensor`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `inplace.sub-tensor`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::subtract.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::subtract.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::subtract.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::subtract_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::subtract_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sum.DimnameList_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sum.IntList_out` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `not_differentiable`; out `contiguous_out_required`. Evidence: [`aten::sum.IntList_out`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `reduction.sum.intlist.out`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sum.default` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::sum.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `arithmetic.autograd.sum-default`, `reduction.sum.default`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sum.dim_DimnameList` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sum.dim_IntList` | Supported (scoped) | `1ba0bb8` (registration), `852df95` (declaration) | Scope: float32; rank 2-3; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::sum.dim_IntList`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `arithmetic.autograd.sum-dim`, `reduction.sum.dim`, `reduction.sum.keepdim.strided`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::sum.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sum_to_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::svd.U` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::svd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::swapaxes.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::swapaxes_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::swapdims.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::swapdims_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_constrain_range.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_constrain_range_for_size.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_numel.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_size.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_storage_offset.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::sym_stride.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::t.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::t_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::t_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::t_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::take.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::take.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::take_along_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::take_along_dim.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tan.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tan_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tanh.default` | Supported (scoped) | `e6ca761` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `first_order_or_none`. Evidence: [`aten::tanh.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.tanh.float32`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::tanh.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::tanh.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::tanh_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::tanh_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::tanh_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tanh_backward.grad_input` | Supported (scoped) | `e6ca761` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `backward_kernel`. Evidence: [`aten::tanh_backward.grad_input`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `unary.tanh.backward`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::tensor_split.indices` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tensor_split.sections` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tensor_split.tensor_indices_or_sections` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tensordot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tensordot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::thnn_conv2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::thnn_conv2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::threshold.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::threshold.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::threshold_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::threshold_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::threshold_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::threshold_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::tile.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to.device` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to.dtype_layout` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to.other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_dense.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_dense_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_mkldnn.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_mkldnn.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_mkldnn_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_padded_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_padded_tensor.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse.sparse_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse_bsc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse_bsr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse_csc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::to_sparse_csr.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::topk.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::topk.values` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trace.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trace.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trace_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::transpose.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::transpose.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::transpose_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::transpose_copy.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::transpose_copy.int_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trapezoid.dx` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trapezoid.x` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trapz.dx` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trapz.x` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triangular_solve.X` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triangular_solve.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tril.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tril.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tril_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tril_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::tril_indices.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triplet_margin_loss.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triu_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triu_indices.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::triu_indices.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::true_divide.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::true_divide.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::true_divide.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::true_divide_.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::true_divide_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trunc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trunc.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::trunc_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::type_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unbind.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unbind.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unbind_copy.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unbind_copy.int_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unflatten.Dimname` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unflatten.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unflatten_dense_tensors.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unfold.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unfold_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unfold_backward.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unfold_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unfold_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::uniform.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::uniform.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::uniform_.default` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::uniform_.default`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::unique_consecutive.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unique_consecutive.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unique_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unique_dim.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unique_dim_consecutive.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unique_dim_consecutive.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsafe_chunk.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsafe_split.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsafe_split.Tensor_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsafe_split_with_sizes.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsafe_split_with_sizes.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsqueeze.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsqueeze_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsqueeze_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::unsqueeze_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bicubic2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bicubic2d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bicubic2d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bicubic2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bicubic2d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bilinear2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bilinear2d.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::upsample_bilinear2d.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::upsample_bilinear2d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bilinear2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_bilinear2d_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::upsample_bilinear2d_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::upsample_linear1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_linear1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_linear1d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_linear1d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_linear1d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest1d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest1d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest1d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest1d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest2d.out` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::upsample_nearest2d.out`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::upsample_nearest2d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest2d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest2d_backward.grad_input` | Deferred | — | Deferred by capability declaration (`reason: deferred_contract`, `execution_contract: deferred_before_vulkan`), declared in `852df95`; direct PrivateUse1 registration absent. Evidence: [`aten::upsample_nearest2d_backward.grad_input`](vulkan_capabilities.json); see [Deferred operator-expansion inventory](vulkan_operator_capability_matrix.md). A registration is source evidence only, not a support claim; it may serve a different documented slice. No runtime support claimed. |
| `aten::upsample_nearest3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest3d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_nearest3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_trilinear3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_trilinear3d.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_trilinear3d.vec` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_trilinear3d_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::upsample_trilinear3d_backward.grad_input` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::value_selecting_reduction_backward.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::values.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::values_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::values_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vander.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.correction` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.correction_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.correction_names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.correction_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.correction` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.correction_names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.correction_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::var_mean.names_dim` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vdot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vdot.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view.default` | Supported (scoped) | `d8e8177` (registration), `852df95` (declaration) | Scope: float32; rank 2; strided/non-overlapping; autograd `reverse_second_order_witnessed`. Evidence: [`aten::view.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `view.view.metadata`, `view.view.trainable-seed`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::view.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_complex.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_complex_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_complex_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_real.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_real_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_as_real_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_copy.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_copy.dtype_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::view_copy.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vsplit.array` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vsplit.int` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vstack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::vstack.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.ScalarOther` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.ScalarSelf` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.self` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::where.self_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.OutScalar_Other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.OutScalar_Self` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.OutTensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.Scalar_Other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.Scalar_Self` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy_.Scalar_Other` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::xlogy_.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zero.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zero.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zero_.default` | Supported (scoped) | `d19ce73` (registration), `852df95` (declaration) | Scope: float32; rank 1; strided/non-overlapping; autograd `optimizer_update`; in-place `validated_exact_alias_inplace`. Evidence: [`aten::zero_.default`](vulkan_capabilities.json), tests test_vulkan_conformance.py, test_vulkan_operator_capabilities.py; cases `optimizer.zero.inplace`; contract in [matrix](vulkan_operator_capability_matrix.md). |
| `aten::zeros.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zeros.names` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zeros.names_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zeros.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zeros_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `aten::zeros_like.out` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>aten</code> (3127)</summary>

```text
aten::__and__.Scalar  aten::__and__.Scalar(Tensor self, Scalar other) -> Tensor
aten::__and__.Tensor  aten::__and__.Tensor(Tensor self, Tensor other) -> Tensor
aten::__iand__.Scalar  aten::__iand__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::__iand__.Tensor  aten::__iand__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::__ilshift__.Scalar  aten::__ilshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::__ilshift__.Tensor  aten::__ilshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::__ior__.Scalar  aten::__ior__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::__ior__.Tensor  aten::__ior__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::__irshift__.Scalar  aten::__irshift__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::__irshift__.Tensor  aten::__irshift__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::__ixor__.Scalar  aten::__ixor__.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::__ixor__.Tensor  aten::__ixor__.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::__lshift__.Scalar  aten::__lshift__.Scalar(Tensor self, Scalar other) -> Tensor
aten::__lshift__.Scalar_out  aten::__lshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::__lshift__.Tensor  aten::__lshift__.Tensor(Tensor self, Tensor other) -> Tensor
aten::__lshift__.Tensor_out  aten::__lshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::__or__.Scalar  aten::__or__.Scalar(Tensor self, Scalar other) -> Tensor
aten::__or__.Tensor  aten::__or__.Tensor(Tensor self, Tensor other) -> Tensor
aten::__rshift__.Scalar  aten::__rshift__.Scalar(Tensor self, Scalar other) -> Tensor
aten::__rshift__.Scalar_out  aten::__rshift__.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::__rshift__.Tensor  aten::__rshift__.Tensor(Tensor self, Tensor other) -> Tensor
aten::__rshift__.Tensor_out  aten::__rshift__.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::__xor__.Scalar  aten::__xor__.Scalar(Tensor self, Scalar other) -> Tensor
aten::__xor__.Tensor  aten::__xor__.Tensor(Tensor self, Tensor other) -> Tensor
aten::_adaptive_avg_pool2d.default  aten::_adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
aten::_adaptive_avg_pool2d.out  aten::_adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::_adaptive_avg_pool2d_backward.default  aten::_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
aten::_adaptive_avg_pool2d_backward.out  aten::_adaptive_avg_pool2d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_adaptive_avg_pool3d.default  aten::_adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
aten::_adaptive_avg_pool3d.out  aten::_adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::_adaptive_avg_pool3d_backward.default  aten::_adaptive_avg_pool3d_backward(Tensor grad_output, Tensor self) -> Tensor
aten::_adaptive_avg_pool3d_backward.out  aten::_adaptive_avg_pool3d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_add_batch_dim.default  aten::_add_batch_dim(Tensor self, int batch_dim, int level) -> Tensor
aten::_add_relu.Scalar  aten::_add_relu.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::_add_relu.Scalar_out  aten::_add_relu.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
aten::_add_relu.Tensor  aten::_add_relu.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
aten::_add_relu.out  aten::_add_relu.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::_add_relu_.Scalar  aten::_add_relu_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
aten::_add_relu_.Tensor  aten::_add_relu_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
aten::_addmm_activation.default  aten::_addmm_activation(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False) -> Tensor
aten::_addmm_activation.out  aten::_addmm_activation.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, bool use_gelu=False, Tensor(a!) out) -> Tensor(a!)
aten::_aminmax.default  aten::_aminmax(Tensor self) -> (Tensor, Tensor)
aten::_aminmax.dim  aten::_aminmax.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor, Tensor)
aten::_aminmax.dim_out  aten::_aminmax.dim_out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_aminmax.out  aten::_aminmax.out(Tensor self, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_amp_foreach_non_finite_check_and_unscale.default  aten::_amp_foreach_non_finite_check_and_unscale(Tensor[] self, Tensor found_inf, Tensor inv_scale) -> (Tensor[] self_out, Tensor found_inf_out)
aten::_amp_foreach_non_finite_check_and_unscale.out  aten::_amp_foreach_non_finite_check_and_unscale.out(Tensor[] self, Tensor(b!) found_inf, Tensor inv_scale, *, Tensor(a!)[] out) -> ()
aten::_amp_foreach_non_finite_check_and_unscale_.default  aten::_amp_foreach_non_finite_check_and_unscale_(Tensor(a!)[] self, Tensor(b!) found_inf, Tensor inv_scale) -> ()
aten::_amp_update_scale.default  aten::_amp_update_scale(Tensor self, Tensor growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> (Tensor, Tensor growth_tracker_out)
aten::_amp_update_scale.out  aten::_amp_update_scale.out(Tensor self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval, *, Tensor(a!) out) -> Tensor(a!)
aten::_amp_update_scale_.default  aten::_amp_update_scale_(Tensor(a!) self, Tensor(b!) growth_tracker, Tensor found_inf, float scale_growth_factor, float scale_backoff_factor, int growth_interval) -> Tensor(a!)
aten::_assert_async.default  aten::_assert_async(Tensor self) -> ()
aten::_assert_async.msg  aten::_assert_async.msg(Tensor self, str assert_msg) -> ()
aten::_assert_scalar.default  aten::_assert_scalar(Scalar self, str assert_msg) -> ()
aten::_assert_tensor_metadata.default  aten::_assert_tensor_metadata(Tensor a, SymInt[]? size=None, SymInt[]? stride=None, ScalarType? dtype=None) -> ()
aten::_autocast_to_full_precision.default  aten::_autocast_to_full_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled) -> Tensor(a)
aten::_autocast_to_reduced_precision.default  aten::_autocast_to_reduced_precision(Tensor(a) self, bool cuda_enabled, bool cpu_enabled, ScalarType cuda_dtype, ScalarType cpu_dtype) -> Tensor(a)
aten::_backward.default  aten::_backward(Tensor self, Tensor[] inputs, Tensor? gradient=None, bool? retain_graph=None, bool create_graph=False) -> ()
aten::_batch_norm_impl_index.default  aten::_batch_norm_impl_index(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> (Tensor, Tensor, Tensor, Tensor, int)
aten::_batch_norm_impl_index_backward.default  aten::_batch_norm_impl_index_backward(int impl_index, Tensor input, Tensor grad_output, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var_transform, bool train, float eps, bool[3] output_mask, Tensor reservedSpace) -> (Tensor, Tensor, Tensor)
aten::_batch_norm_no_update.default  aten::_batch_norm_no_update(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)
aten::_batch_norm_no_update.out  aten::_batch_norm_no_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::_batch_norm_with_update.default  aten::_batch_norm_with_update(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor)
aten::_batch_norm_with_update.out  aten::_batch_norm_with_update.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd, Tensor(g!) reserve) -> (Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!))
aten::_batch_norm_with_update_functional.default  aten::_batch_norm_with_update_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)
aten::_cast_Byte.default  aten::_cast_Byte(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Char.default  aten::_cast_Char(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Double.default  aten::_cast_Double(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Float.default  aten::_cast_Float(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Half.default  aten::_cast_Half(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Int.default  aten::_cast_Int(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Long.default  aten::_cast_Long(Tensor self, bool non_blocking=False) -> Tensor
aten::_cast_Short.default  aten::_cast_Short(Tensor self, bool non_blocking=False) -> Tensor
aten::_cdist_backward.default  aten::_cdist_backward(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist) -> Tensor
aten::_cdist_backward.out  aten::_cdist_backward.out(Tensor grad, Tensor x1, Tensor x2, float p, Tensor cdist, *, Tensor(a!) out) -> Tensor(a!)
aten::_cdist_forward.default  aten::_cdist_forward(Tensor x1, Tensor x2, float p, int? compute_mode) -> Tensor
aten::_cdist_forward.out  aten::_cdist_forward.out(Tensor x1, Tensor x2, float p, int? compute_mode, *, Tensor(a!) out) -> Tensor(a!)
aten::_cholesky_solve_helper.default  aten::_cholesky_solve_helper(Tensor self, Tensor A, bool upper) -> Tensor
aten::_cholesky_solve_helper.out  aten::_cholesky_solve_helper.out(Tensor self, Tensor A, bool upper, *, Tensor(a!) out) -> Tensor(a!)
aten::_choose_qparams_per_tensor.default  aten::_choose_qparams_per_tensor(Tensor self, bool reduce_range=False) -> (float, int)
aten::_chunk_cat.default  aten::_chunk_cat(Tensor[] tensors, int dim, int num_chunks) -> Tensor
aten::_chunk_cat.out  aten::_chunk_cat.out(Tensor[] tensors, int dim, int num_chunks, *, Tensor(a!) out) -> Tensor(a!)
aten::_coalesce.default  aten::_coalesce(Tensor self) -> Tensor
aten::_coalesce.out  aten::_coalesce.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_coalesced.default  aten::_coalesced(Tensor self, bool coalesced) -> Tensor
aten::_coalesced.out  aten::_coalesced.out(Tensor self, bool coalesced, *, Tensor(a!) out) -> Tensor(a!)
aten::_coalesced_.default  aten::_coalesced_(Tensor(a!) self, bool coalesced) -> Tensor(a!)
aten::_compute_linear_combination.default  aten::_compute_linear_combination(Tensor input, Tensor coefficients) -> Tensor
aten::_compute_linear_combination.out  aten::_compute_linear_combination.out(Tensor input, Tensor coefficients, *, Tensor(a!) out) -> Tensor(a!)
aten::_conj.default  aten::_conj(Tensor(a) self) -> Tensor(a)
aten::_conj_copy.default  aten::_conj_copy(Tensor self) -> Tensor
aten::_conj_copy.out  aten::_conj_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_conj_physical.default  aten::_conj_physical(Tensor self) -> Tensor
aten::_conj_physical.out  aten::_conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_conv_depthwise2d.default  aten::_conv_depthwise2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation) -> Tensor
aten::_conv_depthwise2d.out  aten::_conv_depthwise2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, SymInt[2] dilation, *, Tensor(a!) out) -> Tensor(a!)
aten::_convert_indices_from_coo_to_csr.default  aten::_convert_indices_from_coo_to_csr(Tensor self, int size, *, bool out_int32=False) -> Tensor
aten::_convert_indices_from_coo_to_csr.out  aten::_convert_indices_from_coo_to_csr.out(Tensor self, int size, *, bool out_int32=False, Tensor(a!) out) -> Tensor(a!)
aten::_convert_indices_from_csr_to_coo.default  aten::_convert_indices_from_csr_to_coo(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False) -> Tensor
aten::_convert_indices_from_csr_to_coo.out  aten::_convert_indices_from_csr_to_coo.out(Tensor crow_indices, Tensor col_indices, *, bool out_int32=False, bool transpose=False, Tensor(a!) out) -> Tensor(a!)
aten::_convert_weight_to_int4pack.default  aten::_convert_weight_to_int4pack(Tensor self, int innerKTiles) -> Tensor
aten::_convolution.default  aten::_convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32) -> Tensor
aten::_convolution.deprecated  aten::_convolution.deprecated(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, int[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled) -> Tensor
aten::_convolution.out  aten::_convolution.out(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool benchmark, bool deterministic, bool cudnn_enabled, bool allow_tf32, *, Tensor(a!) out) -> Tensor(a!)
aten::_convolution_double_backward.default  aten::_convolution_double_backward(Tensor? ggI, Tensor? ggW, Tensor? ggb, Tensor gO, Tensor weight, Tensor self, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::_convolution_mode.default  aten::_convolution_mode(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, str padding, SymInt[] dilation, SymInt groups) -> Tensor
aten::_copy_from.default  aten::_copy_from(Tensor self, Tensor dst, bool non_blocking=False) -> Tensor
aten::_copy_from.out  aten::_copy_from.out(Tensor self, Tensor dst, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
aten::_copy_from_and_resize.default  aten::_copy_from_and_resize(Tensor self, Tensor dst) -> Tensor
aten::_copy_from_and_resize.out  aten::_copy_from_and_resize.out(Tensor self, Tensor dst, *, Tensor(a!) out) -> Tensor(a!)
aten::_cslt_compress.default  aten::_cslt_compress(Tensor input) -> Tensor
aten::_cslt_sparse_mm.default  aten::_cslt_sparse_mm(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False, int alg_id=0) -> Tensor
aten::_cslt_sparse_mm_search.default  aten::_cslt_sparse_mm_search(Tensor compressed_A, Tensor dense_B, Tensor? bias=None, Tensor? alpha=None, ScalarType? out_dtype=None, bool transpose_result=False) -> int
aten::_ctc_loss.Tensor  aten::_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
aten::_ctc_loss.Tensor_out  aten::_ctc_loss.Tensor_out(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, bool zero_infinity=False, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_ctc_loss.default  aten::_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False) -> (Tensor, Tensor)
aten::_ctc_loss.out  aten::_ctc_loss.out(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, bool zero_infinity=False, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_ctc_loss_backward.Tensor  aten::_ctc_loss_backward.Tensor(Tensor grad, Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
aten::_ctc_loss_backward.default  aten::_ctc_loss_backward(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False) -> Tensor
aten::_ctc_loss_backward.out  aten::_ctc_loss_backward.out(Tensor grad, Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, Tensor neg_log_likelihood, Tensor log_alpha, int blank, bool zero_infinity=False, *, Tensor(a!) out) -> Tensor(a!)
aten::_cudnn_ctc_loss.Tensor  aten::_cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
aten::_cudnn_ctc_loss.default  aten::_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity) -> (Tensor, Tensor)
aten::_cudnn_ctc_loss.out  aten::_cudnn_ctc_loss.out(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank, bool deterministic, bool zero_infinity, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_cudnn_init_dropout_state.default  aten::_cudnn_init_dropout_state(float dropout, bool train, int dropout_seed, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::_cudnn_init_dropout_state.out  aten::_cudnn_init_dropout_state.out(float dropout, bool train, int dropout_seed, *, Tensor(a!) out) -> Tensor(a!)
aten::_cudnn_rnn.default  aten::_cudnn_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_cudnn_rnn.out  aten::_cudnn_rnn.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor? weight_buf, Tensor hx, Tensor? cx, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))
aten::_cudnn_rnn_backward.default  aten::_cudnn_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
aten::_cudnn_rnn_backward.out  aten::_cudnn_rnn_backward.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, SymInt[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!)[] out3) -> ()
aten::_cudnn_rnn_flatten_weight.default  aten::_cudnn_rnn_flatten_weight(Tensor[] weight_arr, int weight_stride0, SymInt input_size, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, bool bidirectional) -> Tensor
aten::_cudnn_rnn_flatten_weight.out  aten::_cudnn_rnn_flatten_weight.out(Tensor[] weight_arr, int weight_stride0, SymInt input_size, int mode, SymInt hidden_size, SymInt proj_size, int num_layers, bool batch_first, bool bidirectional, *, Tensor(a!) out) -> Tensor(a!)
aten::_cufft_clear_plan_cache.default  aten::_cufft_clear_plan_cache(int device_index) -> ()
aten::_cufft_get_plan_cache_max_size.default  aten::_cufft_get_plan_cache_max_size(int device_index) -> int
aten::_cufft_get_plan_cache_size.default  aten::_cufft_get_plan_cache_size(int device_index) -> int
aten::_cufft_set_plan_cache_max_size.default  aten::_cufft_set_plan_cache_max_size(int device_index, int max_size) -> ()
aten::_cummax_helper.default  aten::_cummax_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
aten::_cummin_helper.default  aten::_cummin_helper(Tensor self, Tensor(a!) values, Tensor(b!) indices, int dim) -> ()
aten::_debug_has_internal_overlap.default  aten::_debug_has_internal_overlap(Tensor self) -> int
aten::_dimI.default  aten::_dimI(Tensor self) -> int
aten::_dimV.default  aten::_dimV(Tensor self) -> int
aten::_dim_arange.default  aten::_dim_arange(Tensor like, int dim) -> Tensor
aten::_dirichlet_grad.default  aten::_dirichlet_grad(Tensor x, Tensor alpha, Tensor total) -> Tensor
aten::_dirichlet_grad.out  aten::_dirichlet_grad.out(Tensor x, Tensor alpha, Tensor total, *, Tensor(a!) out) -> Tensor(a!)
aten::_efficient_attention_backward.default  aten::_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor out, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt max_seqlen_q, SymInt max_seqlen_k, Tensor logsumexp, float dropout_p, Tensor philox_seed, Tensor philox_offset, int custom_mask_type, bool bias_requires_grad, *, float? scale=None, int? num_splits_key=None, int? window_size=None, bool shared_storage_dqdkdv=False) -> (Tensor, Tensor, Tensor, Tensor)
aten::_efficient_attention_forward.default  aten::_efficient_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? bias, Tensor? cu_seqlens_q, Tensor? cu_seqlens_k, SymInt? max_seqlen_q, SymInt? max_seqlen_k, float dropout_p, int custom_mask_type, bool compute_log_sumexp=False, *, float? scale=None, Tensor? seqlen_k=None, int? window_size=None) -> (Tensor output, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, SymInt max_seqlen_batch_q, SymInt max_seqlen_batch_k)
aten::_efficientzerotensor.default  aten::_efficientzerotensor(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_efficientzerotensor.out  aten::_efficientzerotensor.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::_embedding_bag.default  aten::_embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
aten::_embedding_bag.out  aten::_embedding_bag.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::_embedding_bag_backward.default  aten::_embedding_bag_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
aten::_embedding_bag_dense_backward.default  aten::_embedding_bag_dense_backward(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
aten::_embedding_bag_dense_backward.out  aten::_embedding_bag_dense_backward.out(Tensor grad, Tensor indices, Tensor offset2bag, Tensor bag_size, Tensor maximum_indices, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)
aten::_embedding_bag_forward_only.default  aten::_embedding_bag_forward_only(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1) -> (Tensor, Tensor, Tensor, Tensor)
aten::_embedding_bag_forward_only.out  aten::_embedding_bag_forward_only.out(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False, int padding_idx=-1, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::_embedding_bag_per_sample_weights_backward.default  aten::_embedding_bag_per_sample_weights_backward(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1) -> Tensor
aten::_embedding_bag_per_sample_weights_backward.out  aten::_embedding_bag_per_sample_weights_backward.out(Tensor grad, Tensor weight, Tensor indices, Tensor offsets, Tensor offset2bag, int mode, int padding_idx=-1, *, Tensor(a!) out) -> Tensor(a!)
aten::_embedding_bag_sparse_backward.default  aten::_embedding_bag_sparse_backward(Tensor grad, Tensor indices, Tensor offsets, Tensor offset2bag, Tensor bag_size, SymInt num_weights, bool scale_grad_by_freq, int mode, Tensor? per_sample_weights, int padding_idx=-1) -> Tensor
aten::_empty_affine_quantized.default  aten::_empty_affine_quantized(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, float scale=1., int zero_point=0, MemoryFormat? memory_format=0) -> Tensor
aten::_empty_affine_quantized.out  aten::_empty_affine_quantized.out(SymInt[] size, *, float scale=1., int zero_point=0, MemoryFormat? memory_format=0, Tensor(a!) out) -> Tensor(a!)
aten::_empty_per_channel_affine_quantized.default  aten::_empty_per_channel_affine_quantized(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=0) -> Tensor
aten::_empty_per_channel_affine_quantized.out  aten::_empty_per_channel_affine_quantized.out(SymInt[] size, *, Tensor scales, Tensor zero_points, int axis, MemoryFormat? memory_format=0, Tensor(a!) out) -> Tensor(a!)
aten::_euclidean_dist.default  aten::_euclidean_dist(Tensor x1, Tensor x2) -> Tensor
aten::_euclidean_dist.out  aten::_euclidean_dist.out(Tensor x1, Tensor x2, *, Tensor(a!) out) -> Tensor(a!)
aten::_fake_quantize_learnable_per_channel_affine.default  aten::_fake_quantize_learnable_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.) -> Tensor
aten::_fake_quantize_learnable_per_channel_affine.out  aten::_fake_quantize_learnable_per_channel_affine.out(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1., *, Tensor(a!) out) -> Tensor(a!)
aten::_fake_quantize_learnable_per_channel_affine_backward.default  aten::_fake_quantize_learnable_per_channel_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, float grad_factor=1.) -> (Tensor, Tensor, Tensor)
aten::_fake_quantize_learnable_per_tensor_affine.default  aten::_fake_quantize_learnable_per_tensor_affine(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.) -> Tensor
aten::_fake_quantize_learnable_per_tensor_affine.out  aten::_fake_quantize_learnable_per_tensor_affine.out(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1., *, Tensor(a!) out) -> Tensor(a!)
aten::_fake_quantize_learnable_per_tensor_affine_backward.default  aten::_fake_quantize_learnable_per_tensor_affine_backward(Tensor grad, Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max, float grad_factor=1.) -> (Tensor, Tensor, Tensor)
aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out  aten::_fake_quantize_per_tensor_affine_cachemask_tensor_qparams.out(Tensor self, Tensor scale, Tensor zero_point, Tensor fake_quant_enabled, int quant_min, int quant_max, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_fft_c2c.default  aten::_fft_c2c(Tensor self, SymInt[] dim, int normalization, bool forward) -> Tensor
aten::_fft_c2c.out  aten::_fft_c2c.out(Tensor self, SymInt[] dim, int normalization, bool forward, *, Tensor(a!) out) -> Tensor(a!)
aten::_fft_c2r.default  aten::_fft_c2r(Tensor self, int[] dim, int normalization, SymInt last_dim_size) -> Tensor
aten::_fft_c2r.out  aten::_fft_c2r.out(Tensor self, int[] dim, int normalization, SymInt last_dim_size, *, Tensor(a!) out) -> Tensor(a!)
aten::_fft_r2c.default  aten::_fft_r2c(Tensor self, int[] dim, int normalization, bool onesided) -> Tensor
aten::_fft_r2c.out  aten::_fft_r2c.out(Tensor self, int[] dim, int normalization, bool onesided, *, Tensor(a!) out) -> Tensor(a!)
aten::_fill_mem_eff_dropout_mask_.default  aten::_fill_mem_eff_dropout_mask_(Tensor(a!) self, float dropout_p, int seed, int offset) -> Tensor(a!)
aten::_flash_attention_backward.default  aten::_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None, SymInt? window_size_left=None, SymInt? window_size_right=None) -> (Tensor, Tensor, Tensor)
aten::_flash_attention_forward.default  aten::_flash_attention_forward(Tensor query, Tensor key, Tensor value, Tensor? cum_seq_q, Tensor? cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, bool return_debug_mask, *, float? scale=None, SymInt? window_size_left=None, SymInt? window_size_right=None, Tensor? seqused_k=None, Tensor? alibi_slopes=None) -> (Tensor output, Tensor softmax_logsumexp, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
aten::_foobar.default  aten::_foobar(Tensor self, bool arg1=True, bool arg2=True, *, bool arg3=True) -> Tensor
aten::_foobar.out  aten::_foobar.out(Tensor self, bool arg1=True, bool arg2=True, *, bool arg3=True, Tensor(a!) out) -> Tensor(a!)
aten::_foreach_abs.default  aten::_foreach_abs(Tensor[] self) -> Tensor[]
aten::_foreach_abs.out  aten::_foreach_abs.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_abs_.default  aten::_foreach_abs_(Tensor(a!)[] self) -> ()
aten::_foreach_acos.default  aten::_foreach_acos(Tensor[] self) -> Tensor[]
aten::_foreach_acos.out  aten::_foreach_acos.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_acos_.default  aten::_foreach_acos_(Tensor(a!)[] self) -> ()
aten::_foreach_add.List  aten::_foreach_add.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
aten::_foreach_add.List_out  aten::_foreach_add.List_out(Tensor[] self, Tensor[] other, *, Scalar alpha=1, Tensor(a!)[] out) -> ()
aten::_foreach_add.Scalar  aten::_foreach_add.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_add.ScalarList  aten::_foreach_add.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_add.ScalarList_out  aten::_foreach_add.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_add.Scalar_out  aten::_foreach_add.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_add.Tensor  aten::_foreach_add.Tensor(Tensor[] self, Tensor other, *, Scalar alpha=1) -> Tensor[]
aten::_foreach_add.Tensor_out  aten::_foreach_add.Tensor_out(Tensor[] self, Tensor other, *, Scalar alpha=1, Tensor(a!)[] out) -> ()
aten::_foreach_add_.List  aten::_foreach_add_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
aten::_foreach_add_.Scalar  aten::_foreach_add_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_add_.ScalarList  aten::_foreach_add_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_add_.Tensor  aten::_foreach_add_.Tensor(Tensor(a!)[] self, Tensor other, *, Scalar alpha=1) -> ()
aten::_foreach_addcdiv.Scalar  aten::_foreach_addcdiv.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
aten::_foreach_addcdiv.ScalarList  aten::_foreach_addcdiv.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
aten::_foreach_addcdiv.ScalarList_out  aten::_foreach_addcdiv.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcdiv.Scalar_out  aten::_foreach_addcdiv.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcdiv.Tensor  aten::_foreach_addcdiv.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]
aten::_foreach_addcdiv.Tensor_out  aten::_foreach_addcdiv.Tensor_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcdiv_.Scalar  aten::_foreach_addcdiv_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
aten::_foreach_addcdiv_.ScalarList  aten::_foreach_addcdiv_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
aten::_foreach_addcdiv_.Tensor  aten::_foreach_addcdiv_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()
aten::_foreach_addcmul.Scalar  aten::_foreach_addcmul.Scalar(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> Tensor[]
aten::_foreach_addcmul.ScalarList  aten::_foreach_addcmul.ScalarList(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> Tensor[]
aten::_foreach_addcmul.ScalarList_out  aten::_foreach_addcmul.ScalarList_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcmul.Scalar_out  aten::_foreach_addcmul.Scalar_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcmul.Tensor  aten::_foreach_addcmul.Tensor(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> Tensor[]
aten::_foreach_addcmul.Tensor_out  aten::_foreach_addcmul.Tensor_out(Tensor[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_addcmul_.Scalar  aten::_foreach_addcmul_.Scalar(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar value=1) -> ()
aten::_foreach_addcmul_.ScalarList  aten::_foreach_addcmul_.ScalarList(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Scalar[] scalars) -> ()
aten::_foreach_addcmul_.Tensor  aten::_foreach_addcmul_.Tensor(Tensor(a!)[] self, Tensor[] tensor1, Tensor[] tensor2, Tensor scalars) -> ()
aten::_foreach_asin.default  aten::_foreach_asin(Tensor[] self) -> Tensor[]
aten::_foreach_asin.out  aten::_foreach_asin.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_asin_.default  aten::_foreach_asin_(Tensor(a!)[] self) -> ()
aten::_foreach_atan.default  aten::_foreach_atan(Tensor[] self) -> Tensor[]
aten::_foreach_atan.out  aten::_foreach_atan.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_atan_.default  aten::_foreach_atan_(Tensor(a!)[] self) -> ()
aten::_foreach_ceil.default  aten::_foreach_ceil(Tensor[] self) -> Tensor[]
aten::_foreach_ceil.out  aten::_foreach_ceil.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_ceil_.default  aten::_foreach_ceil_(Tensor(a!)[] self) -> ()
aten::_foreach_clamp_max.List  aten::_foreach_clamp_max.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_clamp_max.List_out  aten::_foreach_clamp_max.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_max.Scalar  aten::_foreach_clamp_max.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_clamp_max.ScalarList  aten::_foreach_clamp_max.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_clamp_max.ScalarList_out  aten::_foreach_clamp_max.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_max.Scalar_out  aten::_foreach_clamp_max.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_max_.List  aten::_foreach_clamp_max_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_clamp_max_.Scalar  aten::_foreach_clamp_max_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_clamp_max_.ScalarList  aten::_foreach_clamp_max_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_clamp_min.List  aten::_foreach_clamp_min.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_clamp_min.List_out  aten::_foreach_clamp_min.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_min.Scalar  aten::_foreach_clamp_min.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_clamp_min.ScalarList  aten::_foreach_clamp_min.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_clamp_min.ScalarList_out  aten::_foreach_clamp_min.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_min.Scalar_out  aten::_foreach_clamp_min.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_clamp_min_.List  aten::_foreach_clamp_min_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_clamp_min_.Scalar  aten::_foreach_clamp_min_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_clamp_min_.ScalarList  aten::_foreach_clamp_min_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_copy.default  aten::_foreach_copy(Tensor[] self, Tensor[] src, bool non_blocking=False) -> Tensor[] self_out
aten::_foreach_copy.out  aten::_foreach_copy.out(Tensor[] self, Tensor[] src, bool non_blocking=False, *, Tensor(a!)[] out) -> ()
aten::_foreach_copy_.default  aten::_foreach_copy_(Tensor(a!)[] self, Tensor[] src, bool non_blocking=False) -> ()
aten::_foreach_cos.default  aten::_foreach_cos(Tensor[] self) -> Tensor[]
aten::_foreach_cos.out  aten::_foreach_cos.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_cos_.default  aten::_foreach_cos_(Tensor(a!)[] self) -> ()
aten::_foreach_cosh.default  aten::_foreach_cosh(Tensor[] self) -> Tensor[]
aten::_foreach_cosh.out  aten::_foreach_cosh.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_cosh_.default  aten::_foreach_cosh_(Tensor(a!)[] self) -> ()
aten::_foreach_div.List  aten::_foreach_div.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_div.List_out  aten::_foreach_div.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_div.Scalar  aten::_foreach_div.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_div.ScalarList  aten::_foreach_div.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_div.ScalarList_out  aten::_foreach_div.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_div.Scalar_out  aten::_foreach_div.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_div.Tensor  aten::_foreach_div.Tensor(Tensor[] self, Tensor other) -> Tensor[]
aten::_foreach_div.Tensor_out  aten::_foreach_div.Tensor_out(Tensor[] self, Tensor other, *, Tensor(a!)[] out) -> ()
aten::_foreach_div_.List  aten::_foreach_div_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_div_.Scalar  aten::_foreach_div_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_div_.ScalarList  aten::_foreach_div_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_div_.Tensor  aten::_foreach_div_.Tensor(Tensor(a!)[] self, Tensor other) -> ()
aten::_foreach_erf.default  aten::_foreach_erf(Tensor[] self) -> Tensor[]
aten::_foreach_erf.out  aten::_foreach_erf.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_erf_.default  aten::_foreach_erf_(Tensor(a!)[] self) -> ()
aten::_foreach_erfc.default  aten::_foreach_erfc(Tensor[] self) -> Tensor[]
aten::_foreach_erfc.out  aten::_foreach_erfc.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_erfc_.default  aten::_foreach_erfc_(Tensor(a!)[] self) -> ()
aten::_foreach_exp.default  aten::_foreach_exp(Tensor[] self) -> Tensor[]
aten::_foreach_exp.out  aten::_foreach_exp.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_exp_.default  aten::_foreach_exp_(Tensor(a!)[] self) -> ()
aten::_foreach_expm1.default  aten::_foreach_expm1(Tensor[] self) -> Tensor[]
aten::_foreach_expm1.out  aten::_foreach_expm1.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_expm1_.default  aten::_foreach_expm1_(Tensor(a!)[] self) -> ()
aten::_foreach_floor.default  aten::_foreach_floor(Tensor[] self) -> Tensor[]
aten::_foreach_floor.out  aten::_foreach_floor.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_floor_.default  aten::_foreach_floor_(Tensor(a!)[] self) -> ()
aten::_foreach_frac.default  aten::_foreach_frac(Tensor[] self) -> Tensor[]
aten::_foreach_frac.out  aten::_foreach_frac.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_frac_.default  aten::_foreach_frac_(Tensor(a!)[] self) -> ()
aten::_foreach_lerp.List  aten::_foreach_lerp.List(Tensor[] self, Tensor[] tensors1, Tensor[] weights) -> Tensor[]
aten::_foreach_lerp.List_out  aten::_foreach_lerp.List_out(Tensor[] self, Tensor[] tensors1, Tensor[] weights, *, Tensor(a!)[] out) -> ()
aten::_foreach_lerp.Scalar  aten::_foreach_lerp.Scalar(Tensor[] self, Tensor[] tensors1, Scalar weight) -> Tensor[]
aten::_foreach_lerp.Scalar_out  aten::_foreach_lerp.Scalar_out(Tensor[] self, Tensor[] tensors1, Scalar weight, *, Tensor(a!)[] out) -> ()
aten::_foreach_lerp_.List  aten::_foreach_lerp_.List(Tensor(a!)[] self, Tensor[] tensors1, Tensor[] weights) -> ()
aten::_foreach_lerp_.Scalar  aten::_foreach_lerp_.Scalar(Tensor(a!)[] self, Tensor[] tensors1, Scalar weight) -> ()
aten::_foreach_lgamma.default  aten::_foreach_lgamma(Tensor[] self) -> Tensor[]
aten::_foreach_lgamma.out  aten::_foreach_lgamma.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_lgamma_.default  aten::_foreach_lgamma_(Tensor(a!)[] self) -> ()
aten::_foreach_log.default  aten::_foreach_log(Tensor[] self) -> Tensor[]
aten::_foreach_log.out  aten::_foreach_log.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_log10.default  aten::_foreach_log10(Tensor[] self) -> Tensor[]
aten::_foreach_log10.out  aten::_foreach_log10.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_log10_.default  aten::_foreach_log10_(Tensor(a!)[] self) -> ()
aten::_foreach_log1p.default  aten::_foreach_log1p(Tensor[] self) -> Tensor[]
aten::_foreach_log1p.out  aten::_foreach_log1p.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_log1p_.default  aten::_foreach_log1p_(Tensor(a!)[] self) -> ()
aten::_foreach_log2.default  aten::_foreach_log2(Tensor[] self) -> Tensor[]
aten::_foreach_log2.out  aten::_foreach_log2.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_log2_.default  aten::_foreach_log2_(Tensor(a!)[] self) -> ()
aten::_foreach_log_.default  aten::_foreach_log_(Tensor(a!)[] self) -> ()
aten::_foreach_max.default  aten::_foreach_max(Tensor[] self) -> Tensor[]
aten::_foreach_max.out  aten::_foreach_max.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_maximum.List  aten::_foreach_maximum.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_maximum.List_out  aten::_foreach_maximum.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_maximum.Scalar  aten::_foreach_maximum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_maximum.ScalarList  aten::_foreach_maximum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_maximum.ScalarList_out  aten::_foreach_maximum.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_maximum.Scalar_out  aten::_foreach_maximum.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_maximum_.List  aten::_foreach_maximum_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_maximum_.Scalar  aten::_foreach_maximum_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_maximum_.ScalarList  aten::_foreach_maximum_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_minimum.List  aten::_foreach_minimum.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_minimum.List_out  aten::_foreach_minimum.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_minimum.Scalar  aten::_foreach_minimum.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_minimum.ScalarList  aten::_foreach_minimum.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_minimum.ScalarList_out  aten::_foreach_minimum.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_minimum.Scalar_out  aten::_foreach_minimum.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_minimum_.List  aten::_foreach_minimum_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_minimum_.Scalar  aten::_foreach_minimum_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_minimum_.ScalarList  aten::_foreach_minimum_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_mul.List  aten::_foreach_mul.List(Tensor[] self, Tensor[] other) -> Tensor[]
aten::_foreach_mul.List_out  aten::_foreach_mul.List_out(Tensor[] self, Tensor[] other, *, Tensor(a!)[] out) -> ()
aten::_foreach_mul.Scalar  aten::_foreach_mul.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_mul.ScalarList  aten::_foreach_mul.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_mul.ScalarList_out  aten::_foreach_mul.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_mul.Scalar_out  aten::_foreach_mul.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_mul.Tensor  aten::_foreach_mul.Tensor(Tensor[] self, Tensor other) -> Tensor[]
aten::_foreach_mul.Tensor_out  aten::_foreach_mul.Tensor_out(Tensor[] self, Tensor other, *, Tensor(a!)[] out) -> ()
aten::_foreach_mul_.List  aten::_foreach_mul_.List(Tensor(a!)[] self, Tensor[] other) -> ()
aten::_foreach_mul_.Scalar  aten::_foreach_mul_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_mul_.ScalarList  aten::_foreach_mul_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_mul_.Tensor  aten::_foreach_mul_.Tensor(Tensor(a!)[] self, Tensor other) -> ()
aten::_foreach_neg.default  aten::_foreach_neg(Tensor[] self) -> Tensor[]
aten::_foreach_neg.out  aten::_foreach_neg.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_neg_.default  aten::_foreach_neg_(Tensor(a!)[] self) -> ()
aten::_foreach_norm.Scalar  aten::_foreach_norm.Scalar(Tensor[] self, Scalar ord=2, ScalarType? dtype=None) -> Tensor[]
aten::_foreach_norm.Scalar_out  aten::_foreach_norm.Scalar_out(Tensor[] self, Scalar ord=2, ScalarType? dtype=None, *, Tensor(a!)[] out) -> ()
aten::_foreach_pow.List  aten::_foreach_pow.List(Tensor[] self, Tensor[] exponent) -> Tensor[]
aten::_foreach_pow.List_out  aten::_foreach_pow.List_out(Tensor[] self, Tensor[] exponent, *, Tensor(a!)[] out) -> ()
aten::_foreach_pow.Scalar  aten::_foreach_pow.Scalar(Tensor[] self, Scalar exponent) -> Tensor[]
aten::_foreach_pow.ScalarAndTensor  aten::_foreach_pow.ScalarAndTensor(Scalar self, Tensor[] exponent) -> Tensor[]
aten::_foreach_pow.ScalarList  aten::_foreach_pow.ScalarList(Tensor[] self, Scalar[] exponent) -> Tensor[]
aten::_foreach_pow.ScalarList_out  aten::_foreach_pow.ScalarList_out(Tensor[] self, Scalar[] exponent, *, Tensor(a!)[] out) -> ()
aten::_foreach_pow.Scalar_out  aten::_foreach_pow.Scalar_out(Tensor[] self, Scalar exponent, *, Tensor(a!)[] out) -> ()
aten::_foreach_pow_.List  aten::_foreach_pow_.List(Tensor(a!)[] self, Tensor[] exponent) -> ()
aten::_foreach_pow_.Scalar  aten::_foreach_pow_.Scalar(Tensor(a!)[] self, Scalar exponent) -> ()
aten::_foreach_pow_.ScalarList  aten::_foreach_pow_.ScalarList(Tensor(a!)[] self, Scalar[] exponent) -> ()
aten::_foreach_reciprocal.default  aten::_foreach_reciprocal(Tensor[] self) -> Tensor[]
aten::_foreach_reciprocal.out  aten::_foreach_reciprocal.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_reciprocal_.default  aten::_foreach_reciprocal_(Tensor(a!)[] self) -> ()
aten::_foreach_round.default  aten::_foreach_round(Tensor[] self) -> Tensor[]
aten::_foreach_round.out  aten::_foreach_round.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_round_.default  aten::_foreach_round_(Tensor(a!)[] self) -> ()
aten::_foreach_sigmoid.default  aten::_foreach_sigmoid(Tensor[] self) -> Tensor[]
aten::_foreach_sigmoid.out  aten::_foreach_sigmoid.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_sigmoid_.default  aten::_foreach_sigmoid_(Tensor(a!)[] self) -> ()
aten::_foreach_sign.default  aten::_foreach_sign(Tensor[] self) -> Tensor[]
aten::_foreach_sign.out  aten::_foreach_sign.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_sign_.default  aten::_foreach_sign_(Tensor(a!)[] self) -> ()
aten::_foreach_sin.default  aten::_foreach_sin(Tensor[] self) -> Tensor[]
aten::_foreach_sin.out  aten::_foreach_sin.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_sin_.default  aten::_foreach_sin_(Tensor(a!)[] self) -> ()
aten::_foreach_sinh.default  aten::_foreach_sinh(Tensor[] self) -> Tensor[]
aten::_foreach_sinh.out  aten::_foreach_sinh.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_sinh_.default  aten::_foreach_sinh_(Tensor(a!)[] self) -> ()
aten::_foreach_sqrt.default  aten::_foreach_sqrt(Tensor[] self) -> Tensor[]
aten::_foreach_sqrt.out  aten::_foreach_sqrt.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_sqrt_.default  aten::_foreach_sqrt_(Tensor(a!)[] self) -> ()
aten::_foreach_sub.List  aten::_foreach_sub.List(Tensor[] self, Tensor[] other, *, Scalar alpha=1) -> Tensor[]
aten::_foreach_sub.List_out  aten::_foreach_sub.List_out(Tensor[] self, Tensor[] other, *, Scalar alpha=1, Tensor(a!)[] out) -> ()
aten::_foreach_sub.Scalar  aten::_foreach_sub.Scalar(Tensor[] self, Scalar scalar) -> Tensor[]
aten::_foreach_sub.ScalarList  aten::_foreach_sub.ScalarList(Tensor[] self, Scalar[] scalars) -> Tensor[]
aten::_foreach_sub.ScalarList_out  aten::_foreach_sub.ScalarList_out(Tensor[] self, Scalar[] scalars, *, Tensor(a!)[] out) -> ()
aten::_foreach_sub.Scalar_out  aten::_foreach_sub.Scalar_out(Tensor[] self, Scalar scalar, *, Tensor(a!)[] out) -> ()
aten::_foreach_sub_.List  aten::_foreach_sub_.List(Tensor(a!)[] self, Tensor[] other, *, Scalar alpha=1) -> ()
aten::_foreach_sub_.Scalar  aten::_foreach_sub_.Scalar(Tensor(a!)[] self, Scalar scalar) -> ()
aten::_foreach_sub_.ScalarList  aten::_foreach_sub_.ScalarList(Tensor(a!)[] self, Scalar[] scalars) -> ()
aten::_foreach_tan.default  aten::_foreach_tan(Tensor[] self) -> Tensor[]
aten::_foreach_tan.out  aten::_foreach_tan.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_tan_.default  aten::_foreach_tan_(Tensor(a!)[] self) -> ()
aten::_foreach_tanh.default  aten::_foreach_tanh(Tensor[] self) -> Tensor[]
aten::_foreach_tanh.out  aten::_foreach_tanh.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_tanh_.default  aten::_foreach_tanh_(Tensor(a!)[] self) -> ()
aten::_foreach_trunc.default  aten::_foreach_trunc(Tensor[] self) -> Tensor[]
aten::_foreach_trunc.out  aten::_foreach_trunc.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_trunc_.default  aten::_foreach_trunc_(Tensor(a!)[] self) -> ()
aten::_foreach_zero.default  aten::_foreach_zero(Tensor[] self) -> Tensor[] self_out
aten::_foreach_zero.out  aten::_foreach_zero.out(Tensor[] self, *, Tensor(a!)[] out) -> ()
aten::_foreach_zero_.default  aten::_foreach_zero_(Tensor(a!)[] self) -> ()
aten::_functional_assert_async.msg  aten::_functional_assert_async.msg(Tensor self, str assert_msg, Tensor dep_token) -> Tensor
aten::_functional_assert_scalar.default  aten::_functional_assert_scalar(Scalar self, str assert_msg, Tensor dep_token) -> Tensor
aten::_functional_sym_constrain_range.default  aten::_functional_sym_constrain_range(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
aten::_functional_sym_constrain_range_for_size.default  aten::_functional_sym_constrain_range_for_size(Scalar size, int? min, int? max, Tensor dep_token) -> Tensor
aten::_fused_adagrad.default  aten::_fused_adagrad(Tensor[] self, Tensor[] grads, Tensor[] state_sums, Tensor[] state_steps, *, float lr, float lr_decay, float weight_decay, float eps, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] state_sums_out, Tensor[] state_steps_out)
aten::_fused_adagrad.out  aten::_fused_adagrad.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] state_sums, Tensor(d!)[] state_steps, *, float lr, float lr_decay, float weight_decay, float eps, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_adagrad_.default  aten::_fused_adagrad_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] state_sums, Tensor(d!)[] state_steps, *, float lr, float lr_decay, float weight_decay, float eps, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_adam.default  aten::_fused_adam(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)
aten::_fused_adam.out  aten::_fused_adam.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_adam.tensor_lr  aten::_fused_adam.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)
aten::_fused_adam.tensor_lr_out  aten::_fused_adam.tensor_lr_out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_adam_.default  aten::_fused_adam_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_adam_.tensor_lr  aten::_fused_adam_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_adamw.default  aten::_fused_adamw(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)
aten::_fused_adamw.out  aten::_fused_adamw.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_adamw.tensor_lr  aten::_fused_adamw.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] exp_avgs, Tensor[] exp_avg_sqs, Tensor[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] exp_avgs_out, Tensor[] exp_avg_sqs_out, Tensor[] max_exp_avg_sqs_out)
aten::_fused_adamw.tensor_lr_out  aten::_fused_adamw.tensor_lr_out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_adamw_.default  aten::_fused_adamw_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, float lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_adamw_.tensor_lr  aten::_fused_adamw_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] exp_avgs, Tensor(d!)[] exp_avg_sqs, Tensor(e!)[] max_exp_avg_sqs, Tensor[] state_steps, *, Tensor lr, float beta1, float beta2, float weight_decay, float eps, bool amsgrad, bool maximize, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_dropout.default  aten::_fused_dropout(Tensor self, float p, Generator? generator=None) -> (Tensor, Tensor)
aten::_fused_dropout.out  aten::_fused_dropout.out(Tensor self, float p, Generator? generator=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_fused_moving_avg_obs_fq_helper.default  aten::_fused_moving_avg_obs_fq_helper(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask)
aten::_fused_moving_avg_obs_fq_helper.out  aten::_fused_moving_avg_obs_fq_helper.out(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False, *, Tensor(e!) out0, Tensor(f!) out1) -> (Tensor(e!), Tensor(f!))
aten::_fused_moving_avg_obs_fq_helper_functional.default  aten::_fused_moving_avg_obs_fq_helper_functional(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor running_min, Tensor running_max, Tensor scale, Tensor zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> (Tensor output, Tensor mask, Tensor running_min_out, Tensor running_max_out, Tensor scale_out, Tensor zero_point_out)
aten::_fused_sdp_choice.default  aten::_fused_sdp_choice(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> int
aten::_fused_sgd.default  aten::_fused_sgd(Tensor[] self, Tensor[] grads, Tensor[] momentum_buffer_list, *, float weight_decay, float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] momentum_buffer_list_out)
aten::_fused_sgd.out  aten::_fused_sgd.out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_sgd.tensor_lr  aten::_fused_sgd.tensor_lr(Tensor[] self, Tensor[] grads, Tensor[] momentum_buffer_list, *, float weight_decay, float momentum, Tensor lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> (Tensor[] self_out, Tensor[] grads_out, Tensor[] momentum_buffer_list_out)
aten::_fused_sgd.tensor_lr_out  aten::_fused_sgd.tensor_lr_out(Tensor[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, Tensor lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None, Tensor(a!)[] out) -> ()
aten::_fused_sgd_.default  aten::_fused_sgd_(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, float lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fused_sgd_.tensor_lr  aten::_fused_sgd_.tensor_lr(Tensor(a!)[] self, Tensor(b!)[] grads, Tensor(c!)[] momentum_buffer_list, *, float weight_decay, float momentum, Tensor lr, float dampening, bool nesterov, bool maximize, bool is_first_step, Tensor? grad_scale=None, Tensor? found_inf=None) -> ()
aten::_fw_primal.default  aten::_fw_primal(Tensor(a) self, int level) -> Tensor(a)
aten::_fw_primal_copy.default  aten::_fw_primal_copy(Tensor self, int level) -> Tensor
aten::_fw_primal_copy.out  aten::_fw_primal_copy.out(Tensor self, int level, *, Tensor(a!) out) -> Tensor(a!)
aten::_gather_sparse_backward.default  aten::_gather_sparse_backward(Tensor self, int dim, Tensor index, Tensor grad) -> Tensor
aten::_grid_sampler_2d_cpu_fallback.default  aten::_grid_sampler_2d_cpu_fallback(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
aten::_grid_sampler_2d_cpu_fallback.out  aten::_grid_sampler_2d_cpu_fallback.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
aten::_grid_sampler_2d_cpu_fallback_backward.default  aten::_grid_sampler_2d_cpu_fallback_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> (Tensor, Tensor)
aten::_has_compatible_shallow_copy_type.default  aten::_has_compatible_shallow_copy_type(Tensor self, Tensor from) -> bool
aten::_has_same_storage_numel.default  aten::_has_same_storage_numel(Tensor self, Tensor other) -> bool
aten::_histogramdd_bin_edges.default  aten::_histogramdd_bin_edges(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor[]
aten::_histogramdd_bin_edges.out  aten::_histogramdd_bin_edges.out(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!)[] out) -> ()
aten::_histogramdd_from_bin_cts.default  aten::_histogramdd_from_bin_cts(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False) -> Tensor
aten::_histogramdd_from_bin_cts.out  aten::_histogramdd_from_bin_cts.out(Tensor self, int[] bins, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) out) -> Tensor(a!)
aten::_histogramdd_from_bin_tensors.default  aten::_histogramdd_from_bin_tensors(Tensor self, Tensor[] bins, *, Tensor? weight=None, bool density=False) -> Tensor
aten::_histogramdd_from_bin_tensors.out  aten::_histogramdd_from_bin_tensors.out(Tensor self, Tensor[] bins, *, Tensor? weight=None, bool density=False, Tensor(a!) out) -> Tensor(a!)
aten::_index_put_impl.default  aten::_index_put_impl(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor
aten::_index_put_impl.out  aten::_index_put_impl.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False, *, Tensor(a!) out) -> Tensor(a!)
aten::_index_put_impl_.default  aten::_index_put_impl_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False, bool unsafe=False) -> Tensor(a!)
aten::_indices.default  aten::_indices(Tensor(a) self) -> Tensor(a)
aten::_indices_copy.default  aten::_indices_copy(Tensor self) -> Tensor
aten::_indices_copy.out  aten::_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_int_mm.default  aten::_int_mm(Tensor self, Tensor mat2) -> Tensor
aten::_int_mm.out  aten::_int_mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
aten::_is_all_true.default  aten::_is_all_true(Tensor self) -> Tensor
aten::_is_any_true.default  aten::_is_any_true(Tensor self) -> Tensor
aten::_is_zerotensor.default  aten::_is_zerotensor(Tensor self) -> bool
aten::_jagged_to_padded_dense_forward.default  aten::_jagged_to_padded_dense_forward(Tensor values, Tensor[] offsets, SymInt[] max_lengths, float padding_value=0.) -> Tensor
aten::_lazy_clone.default  aten::_lazy_clone(Tensor self) -> Tensor
aten::_linalg_check_errors.default  aten::_linalg_check_errors(Tensor info, str api_name, *, bool is_matrix) -> ()
aten::_linalg_det.default  aten::_linalg_det(Tensor A) -> (Tensor result, Tensor LU, Tensor pivots)
aten::_linalg_det.result  aten::_linalg_det.result(Tensor A, *, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots)
aten::_linalg_eigh.default  aten::_linalg_eigh(Tensor A, str UPLO="L", bool compute_v=True) -> (Tensor eigenvalues, Tensor eigenvectors)
aten::_linalg_eigh.eigenvalues  aten::_linalg_eigh.eigenvalues(Tensor A, str UPLO="L", bool compute_v=True, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
aten::_linalg_eigvals.default  aten::_linalg_eigvals(Tensor self) -> Tensor
aten::_linalg_slogdet.default  aten::_linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet, Tensor LU, Tensor pivots)
aten::_linalg_slogdet.sign  aten::_linalg_slogdet.sign(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots) -> (Tensor(a!) sign, Tensor(b!) logabsdet, Tensor(c!) LU, Tensor(d!) pivots)
aten::_linalg_solve_ex.default  aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)
aten::_linalg_solve_ex.result  aten::_linalg_solve_ex.result(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info) -> (Tensor(a!) result, Tensor(b!) LU, Tensor(c!) pivots, Tensor(d!) info)
aten::_linalg_svd.U  aten::_linalg_svd.U(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
aten::_linalg_svd.default  aten::_linalg_svd(Tensor A, bool full_matrices=False, bool compute_uv=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
aten::_local_scalar_dense.default  aten::_local_scalar_dense(Tensor self) -> Scalar
aten::_log_softmax.default  aten::_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::_log_softmax.out  aten::_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
aten::_log_softmax_backward_data.default  aten::_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
aten::_log_softmax_backward_data.out  aten::_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::_logcumsumexp.default  aten::_logcumsumexp(Tensor self, int dim) -> Tensor
aten::_logcumsumexp.out  aten::_logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::_lstm_mps.default  aten::_lstm_mps(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_lstm_mps.out  aten::_lstm_mps.out(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4, Tensor(f!) out5) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!))
aten::_lu_with_info.default  aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) -> (Tensor LU, Tensor pivots, Tensor info)
aten::_make_dep_token.default  aten::_make_dep_token(*, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::_make_dual.default  aten::_make_dual(Tensor(a) primal, Tensor tangent, int level) -> Tensor(a)
aten::_make_dual_copy.default  aten::_make_dual_copy(Tensor primal, Tensor tangent, int level) -> Tensor
aten::_make_dual_copy.out  aten::_make_dual_copy.out(Tensor primal, Tensor tangent, int level, *, Tensor(a!) out) -> Tensor(a!)
aten::_make_per_channel_quantized_tensor.default  aten::_make_per_channel_quantized_tensor(Tensor self, Tensor scale, Tensor zero_point, int axis) -> Tensor
aten::_make_per_channel_quantized_tensor.out  aten::_make_per_channel_quantized_tensor.out(Tensor self, Tensor scale, Tensor zero_point, int axis, *, Tensor(a!) out) -> Tensor(a!)
aten::_make_per_tensor_quantized_tensor.default  aten::_make_per_tensor_quantized_tensor(Tensor self, float scale, int zero_point) -> Tensor
aten::_make_per_tensor_quantized_tensor.out  aten::_make_per_tensor_quantized_tensor.out(Tensor self, float scale, int zero_point, *, Tensor(a!) out) -> Tensor(a!)
aten::_masked_scale.default  aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor
aten::_masked_scale.out  aten::_masked_scale.out(Tensor self, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)
aten::_masked_softmax.default  aten::_masked_softmax(Tensor self, Tensor mask, int? dim=None, int? mask_type=None) -> Tensor
aten::_masked_softmax.out  aten::_masked_softmax.out(Tensor self, Tensor mask, int? dim=None, int? mask_type=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_masked_softmax_backward.default  aten::_masked_softmax_backward(Tensor grad_output, Tensor output, Tensor mask, int? dim=None) -> Tensor
aten::_masked_softmax_backward.out  aten::_masked_softmax_backward.out(Tensor grad_output, Tensor output, Tensor mask, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_mixed_dtypes_linear.default  aten::_mixed_dtypes_linear(Tensor input, Tensor weight, Tensor scale, *, Tensor? bias=None, str? activation=None) -> Tensor
aten::_mkldnn_reshape.default  aten::_mkldnn_reshape(Tensor self, int[] shape) -> Tensor
aten::_mkldnn_reshape.out  aten::_mkldnn_reshape.out(Tensor self, int[] shape, *, Tensor(a!) out) -> Tensor(a!)
aten::_mkldnn_transpose.default  aten::_mkldnn_transpose(Tensor self, int dim0, int dim1) -> Tensor
aten::_mkldnn_transpose.out  aten::_mkldnn_transpose.out(Tensor self, int dim0, int dim1, *, Tensor(a!) out) -> Tensor(a!)
aten::_mkldnn_transpose_.default  aten::_mkldnn_transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
aten::_mps_convolution.default  aten::_mps_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
aten::_mps_convolution.out  aten::_mps_convolution.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::_mps_convolution_transpose.default  aten::_mps_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
aten::_mps_convolution_transpose.out  aten::_mps_convolution_transpose.out(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::_native_batch_norm_legit.default  aten::_native_batch_norm_legit(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
aten::_native_batch_norm_legit.no_stats  aten::_native_batch_norm_legit.no_stats(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
aten::_native_batch_norm_legit.no_stats_out  aten::_native_batch_norm_legit.no_stats_out(Tensor input, Tensor? weight, Tensor? bias, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_native_batch_norm_legit.out  aten::_native_batch_norm_legit.out(Tensor input, Tensor? weight, Tensor? bias, Tensor(a!) running_mean, Tensor(b!) running_var, bool training, float momentum, float eps, *, Tensor(d!) out, Tensor(e!) save_mean, Tensor(f!) save_invstd) -> (Tensor(d!), Tensor(e!), Tensor(f!))
aten::_native_batch_norm_legit_functional.default  aten::_native_batch_norm_legit_functional(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor, Tensor running_mean_out, Tensor running_var_out)
aten::_native_batch_norm_legit_no_training.default  aten::_native_batch_norm_legit_no_training(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps) -> (Tensor, Tensor, Tensor)
aten::_native_batch_norm_legit_no_training.out  aten::_native_batch_norm_legit_no_training.out(Tensor input, Tensor? weight, Tensor? bias, Tensor running_mean, Tensor running_var, float momentum, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_native_multi_head_attention.default  aten::_native_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None) -> (Tensor, Tensor)
aten::_native_multi_head_attention.out  aten::_native_multi_head_attention.out(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, bool need_weights=True, bool average_attn_weights=True, int? mask_type=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_neg_view.default  aten::_neg_view(Tensor(a) self) -> Tensor(a)
aten::_neg_view_copy.default  aten::_neg_view_copy(Tensor self) -> Tensor
aten::_neg_view_copy.out  aten::_neg_view_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_compute_contiguous_strides_offsets.default  aten::_nested_compute_contiguous_strides_offsets(Tensor nested_size) -> (Tensor, Tensor)
aten::_nested_from_padded.default  aten::_nested_from_padded(Tensor padded, Tensor cpu_nested_shape_example, bool fuse_transform_0213=False) -> Tensor
aten::_nested_from_padded.out  aten::_nested_from_padded.out(Tensor padded, Tensor cpu_nested_shape_example, bool fuse_transform_0213=False, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_from_padded_and_nested_example.default  aten::_nested_from_padded_and_nested_example(Tensor padded, Tensor nt_example) -> Tensor
aten::_nested_from_padded_and_nested_example.out  aten::_nested_from_padded_and_nested_example.out(Tensor padded, Tensor nt_example, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_get_jagged_dummy.default  aten::_nested_get_jagged_dummy(Tensor any) -> Tensor
aten::_nested_get_lengths.default  aten::_nested_get_lengths(Tensor self) -> Tensor
aten::_nested_get_offsets.default  aten::_nested_get_offsets(Tensor self) -> Tensor
aten::_nested_get_ragged_idx.default  aten::_nested_get_ragged_idx(Tensor self) -> int
aten::_nested_get_values.default  aten::_nested_get_values(Tensor(a) self) -> Tensor(a)
aten::_nested_get_values_copy.default  aten::_nested_get_values_copy(Tensor self) -> Tensor
aten::_nested_get_values_copy.out  aten::_nested_get_values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_select_backward.default  aten::_nested_select_backward(Tensor grad_output, Tensor self, int dim, SymInt index) -> Tensor
aten::_nested_sum_backward.default  aten::_nested_sum_backward(Tensor grad, Tensor self, int[1]? dim, bool keepdim=False) -> Tensor
aten::_nested_tensor_from_mask.default  aten::_nested_tensor_from_mask(Tensor t, Tensor mask, bool mask_check=True) -> Tensor
aten::_nested_tensor_from_mask.out  aten::_nested_tensor_from_mask.out(Tensor t, Tensor mask, bool mask_check=True, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_tensor_from_mask_left_aligned.default  aten::_nested_tensor_from_mask_left_aligned(Tensor t, Tensor mask) -> bool
aten::_nested_tensor_from_tensor_list.default  aten::_nested_tensor_from_tensor_list(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_nested_tensor_from_tensor_list.out  aten::_nested_tensor_from_tensor_list.out(Tensor[] list, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_tensor_size.default  aten::_nested_tensor_size(Tensor self) -> Tensor
aten::_nested_tensor_size.out  aten::_nested_tensor_size.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_tensor_softmax_with_shape.default  aten::_nested_tensor_softmax_with_shape(Tensor self, Tensor query) -> Tensor
aten::_nested_tensor_storage_offsets.default  aten::_nested_tensor_storage_offsets(Tensor self) -> Tensor
aten::_nested_tensor_storage_offsets.out  aten::_nested_tensor_storage_offsets.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_tensor_strides.default  aten::_nested_tensor_strides(Tensor self) -> Tensor
aten::_nested_tensor_strides.out  aten::_nested_tensor_strides.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_view_from_buffer.default  aten::_nested_view_from_buffer(Tensor(a) self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor(a)
aten::_nested_view_from_buffer_copy.default  aten::_nested_view_from_buffer_copy(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets) -> Tensor
aten::_nested_view_from_buffer_copy.out  aten::_nested_view_from_buffer_copy.out(Tensor self, Tensor nested_size, Tensor nested_strides, Tensor offsets, *, Tensor(a!) out) -> Tensor(a!)
aten::_nested_view_from_jagged.default  aten::_nested_view_from_jagged(Tensor(a) self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1) -> Tensor(a)
aten::_nested_view_from_jagged_copy.default  aten::_nested_view_from_jagged_copy(Tensor self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1) -> Tensor
aten::_nested_view_from_jagged_copy.out  aten::_nested_view_from_jagged_copy.out(Tensor self, Tensor offsets, Tensor dummy, Tensor? lengths=None, int ragged_idx=1, *, Tensor(a!) out) -> Tensor(a!)
aten::_new_zeros_with_same_feature_meta.default  aten::_new_zeros_with_same_feature_meta(Tensor self, Tensor other, *, int self_num_batch_dims=0) -> Tensor
aten::_new_zeros_with_same_feature_meta.out  aten::_new_zeros_with_same_feature_meta.out(Tensor self, Tensor other, *, int self_num_batch_dims=0, Tensor(a!) out) -> Tensor(a!)
aten::_nnpack_available.default  aten::_nnpack_available() -> bool
aten::_nnpack_spatial_convolution.default  aten::_nnpack_spatial_convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[2] padding, SymInt[2] stride=[1, 1]) -> Tensor
aten::_nnpack_spatial_convolution.out  aten::_nnpack_spatial_convolution.out(Tensor input, Tensor weight, Tensor? bias, SymInt[2] padding, SymInt[2] stride=[1, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::_nnz.default  aten::_nnz(Tensor self) -> int
aten::_pack_padded_sequence.default  aten::_pack_padded_sequence(Tensor input, Tensor lengths, bool batch_first) -> (Tensor, Tensor)
aten::_pack_padded_sequence.out  aten::_pack_padded_sequence.out(Tensor input, Tensor lengths, bool batch_first, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_pack_padded_sequence_backward.default  aten::_pack_padded_sequence_backward(Tensor grad, SymInt[] input_size, Tensor batch_sizes, bool batch_first) -> Tensor
aten::_pad_circular.default  aten::_pad_circular(Tensor self, SymInt[] pad) -> Tensor
aten::_pad_enum.default  aten::_pad_enum(Tensor self, SymInt[] pad, int mode, float? value=None) -> Tensor
aten::_pad_packed_sequence.default  aten::_pad_packed_sequence(Tensor data, Tensor batch_sizes, bool batch_first, Scalar padding_value, int total_length) -> (Tensor, Tensor)
aten::_padded_dense_to_jagged_forward.default  aten::_padded_dense_to_jagged_forward(Tensor dense, Tensor[] offsets, SymInt? total_L=None) -> Tensor
aten::_pdist_backward.default  aten::_pdist_backward(Tensor grad, Tensor self, float p, Tensor pdist) -> Tensor
aten::_pdist_backward.out  aten::_pdist_backward.out(Tensor grad, Tensor self, float p, Tensor pdist, *, Tensor(a!) out) -> Tensor(a!)
aten::_pdist_forward.default  aten::_pdist_forward(Tensor self, float p=2.) -> Tensor
aten::_pdist_forward.out  aten::_pdist_forward.out(Tensor self, float p=2., *, Tensor(a!) out) -> Tensor(a!)
aten::_pin_memory.default  aten::_pin_memory(Tensor self, Device? device=None) -> Tensor
aten::_pin_memory.out  aten::_pin_memory.out(Tensor self, Device? device=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_prelu_kernel.default  aten::_prelu_kernel(Tensor self, Tensor weight) -> Tensor
aten::_prelu_kernel_backward.default  aten::_prelu_kernel_backward(Tensor grad_output, Tensor self, Tensor weight) -> (Tensor, Tensor)
aten::_print.default  aten::_print(str s) -> ()
aten::_propagate_xla_data.default  aten::_propagate_xla_data(Tensor input, Tensor output) -> ()
aten::_remove_batch_dim.default  aten::_remove_batch_dim(Tensor self, int level, int batch_size, int out_dim) -> Tensor
aten::_reshape_alias.default  aten::_reshape_alias(Tensor(a) self, SymInt[] size, SymInt[] stride) -> Tensor(a)
aten::_reshape_alias_copy.default  aten::_reshape_alias_copy(Tensor self, SymInt[] size, SymInt[] stride) -> Tensor
aten::_reshape_alias_copy.out  aten::_reshape_alias_copy.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)
aten::_reshape_copy.default  aten::_reshape_copy(Tensor self, SymInt[] size) -> Tensor
aten::_reshape_from_tensor.default  aten::_reshape_from_tensor(Tensor self, Tensor shape) -> Tensor
aten::_resize_output.default  aten::_resize_output(Tensor self, SymInt[] size, Device device) -> Tensor
aten::_resize_output.out  aten::_resize_output.out(Tensor self, SymInt[] size, Device device, *, Tensor(a!) out) -> Tensor(a!)
aten::_resize_output_.default  aten::_resize_output_(Tensor(a!) self, SymInt[] size, Device device) -> Tensor(a!)
aten::_rowwise_prune.default  aten::_rowwise_prune(Tensor weight, Tensor mask, ScalarType compressed_indices_dtype) -> (Tensor, Tensor)
aten::_sample_dirichlet.default  aten::_sample_dirichlet(Tensor self, Generator? generator=None) -> Tensor
aten::_sample_dirichlet.out  aten::_sample_dirichlet.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_saturate_weight_to_fp16.default  aten::_saturate_weight_to_fp16(Tensor weight) -> Tensor
aten::_scaled_dot_product_attention_math.default  aten::_scaled_dot_product_attention_math(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, Tensor? dropout_mask=None, *, float? scale=None) -> (Tensor, Tensor)
aten::_scaled_dot_product_cudnn_attention.default  aten::_scaled_dot_product_cudnn_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
aten::_scaled_dot_product_cudnn_attention_backward.default  aten::_scaled_dot_product_cudnn_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor, Tensor, Tensor)
aten::_scaled_dot_product_efficient_attention.default  aten::_scaled_dot_product_efficient_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_bias, bool compute_log_sumexp, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> (Tensor output, Tensor log_sumexp, Tensor philox_seed, Tensor philox_offset)
aten::_scaled_dot_product_efficient_attention_backward.default  aten::_scaled_dot_product_efficient_attention_backward(Tensor grad_out_, Tensor query, Tensor key, Tensor value, Tensor attn_bias, Tensor out, Tensor logsumexp, Tensor philox_seed, Tensor philox_offset, float dropout_p, bool[4] grad_input_mask, bool is_causal=False, *, float? scale=None) -> (Tensor, Tensor, Tensor, Tensor)
aten::_scaled_dot_product_flash_attention.default  aten::_scaled_dot_product_flash_attention(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, bool return_debug_mask=False, *, float? scale=None) -> (Tensor output, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, Tensor philox_seed, Tensor philox_offset, Tensor debug_attn_mask)
aten::_scaled_dot_product_flash_attention_backward.default  aten::_scaled_dot_product_flash_attention_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, Tensor cum_seq_q, Tensor cum_seq_k, SymInt max_q, SymInt max_k, float dropout_p, bool is_causal, Tensor philox_seed, Tensor philox_offset, *, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
aten::_scaled_dot_product_flash_attention_for_cpu.default  aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key, Tensor value, float dropout_p=0., bool is_causal=False, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor output, Tensor logsumexp)
aten::_scaled_dot_product_flash_attention_for_cpu_backward.default  aten::_scaled_dot_product_flash_attention_for_cpu_backward(Tensor grad_out, Tensor query, Tensor key, Tensor value, Tensor out, Tensor logsumexp, float dropout_p, bool is_causal, *, Tensor? attn_mask=None, float? scale=None) -> (Tensor grad_query, Tensor grad_key, Tensor grad_value)
aten::_scaled_mm.default  aten::_scaled_mm(Tensor self, Tensor mat2, *, Tensor? bias=None, ScalarType? out_dtype=None, Tensor? scale_a=None, Tensor? scale_b=None, Tensor? scale_result=None, bool use_fast_accum=False) -> (Tensor, Tensor)
aten::_scaled_mm.out  aten::_scaled_mm.out(Tensor self, Tensor mat2, *, Tensor? bias=None, ScalarType? out_dtype=None, Tensor? scale_a=None, Tensor? scale_b=None, Tensor? scale_result=None, bool use_fast_accum=False, Tensor(a!) out, Tensor(b!) out_amax) -> (Tensor(a!), Tensor(b!))
aten::_segment_reduce_backward.default  aten::_segment_reduce_backward(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, Tensor? offsets=None, int axis=0, Scalar? initial=None) -> Tensor
aten::_segment_reduce_backward.out  aten::_segment_reduce_backward.out(Tensor grad, Tensor output, Tensor data, str reduce, *, Tensor? lengths=None, Tensor? offsets=None, int axis=0, Scalar? initial=None, Tensor(a!) out) -> Tensor(a!)
aten::_shape_as_tensor.default  aten::_shape_as_tensor(Tensor self) -> Tensor
aten::_slow_conv2d_backward.grad_input  aten::_slow_conv2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, *, Tensor(a!) grad_input, Tensor(b!) grad_weight, Tensor(c!) grad_bias) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_slow_conv2d_backward.output_mask  aten::_slow_conv2d_backward.output_mask(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
aten::_slow_conv2d_backward.output_mask_out  aten::_slow_conv2d_backward.output_mask_out(Tensor grad_output, Tensor self, Tensor weight, SymInt[2] kernel_size, SymInt[2] stride, SymInt[2] padding, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_slow_conv2d_forward.default  aten::_slow_conv2d_forward(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding) -> Tensor
aten::_slow_conv2d_forward.output  aten::_slow_conv2d_forward.output(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias, SymInt[2] stride, SymInt[2] padding, *, Tensor(a!) output) -> Tensor(a!)
aten::_sobol_engine_draw.default  aten::_sobol_engine_draw(Tensor quasi, int n, Tensor sobolstate, int dimension, int num_generated, ScalarType? dtype) -> (Tensor, Tensor)
aten::_sobol_engine_ff_.default  aten::_sobol_engine_ff_(Tensor(a!) self, int n, Tensor sobolstate, int dimension, int num_generated) -> Tensor(a!)
aten::_sobol_engine_initialize_state_.default  aten::_sobol_engine_initialize_state_(Tensor(a!) self, int dimension) -> Tensor(a!)
aten::_sobol_engine_scramble_.default  aten::_sobol_engine_scramble_(Tensor(a!) self, Tensor ltm, int dimension) -> Tensor(a!)
aten::_softmax.default  aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::_softmax.out  aten::_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
aten::_softmax_backward_data.default  aten::_softmax_backward_data(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype) -> Tensor
aten::_softmax_backward_data.out  aten::_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, ScalarType input_dtype, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_sparse_addmm.default  aten::_sparse_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::_sparse_addmm.out  aten::_sparse_addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_broadcast_to.default  aten::_sparse_broadcast_to(Tensor(a) self, int[] size) -> Tensor(a)
aten::_sparse_broadcast_to_copy.default  aten::_sparse_broadcast_to_copy(Tensor self, int[] size) -> Tensor
aten::_sparse_broadcast_to_copy.out  aten::_sparse_broadcast_to_copy.out(Tensor self, int[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_bsc_tensor_unsafe.default  aten::_sparse_bsc_tensor_unsafe(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_sparse_bsr_tensor_unsafe.default  aten::_sparse_bsr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_sparse_compressed_tensor_unsafe.default  aten::_sparse_compressed_tensor_unsafe(Tensor compressed_indices, Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_sparse_compressed_tensor_with_dims.default  aten::_sparse_compressed_tensor_with_dims(int nnz, int dense_dim, int[] size, int[] blocksize, ScalarType index_dtype, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::_sparse_coo_tensor_unsafe.default  aten::_sparse_coo_tensor_unsafe(Tensor indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool? is_coalesced=None) -> Tensor
aten::_sparse_coo_tensor_with_dims.default  aten::_sparse_coo_tensor_with_dims(int sparse_dim, int dense_dim, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::_sparse_coo_tensor_with_dims.out  aten::_sparse_coo_tensor_with_dims.out(int sparse_dim, int dense_dim, int[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_coo_tensor_with_dims_and_tensors.default  aten::_sparse_coo_tensor_with_dims_and_tensors(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False, bool? is_coalesced=None) -> Tensor
aten::_sparse_coo_tensor_with_dims_and_tensors.out  aten::_sparse_coo_tensor_with_dims_and_tensors.out(int sparse_dim, int dense_dim, SymInt[] size, Tensor indices, Tensor values, *, bool? is_coalesced=None, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_csc_tensor_unsafe.default  aten::_sparse_csc_tensor_unsafe(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_sparse_csr_prod.dim_dtype  aten::_sparse_csr_prod.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::_sparse_csr_prod.dim_dtype_out  aten::_sparse_csr_prod.dim_dtype_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_csr_sum.dim_dtype  aten::_sparse_csr_sum.dim_dtype(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::_sparse_csr_sum.dim_dtype_out  aten::_sparse_csr_sum.dim_dtype_out(Tensor self, int[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_csr_tensor_unsafe.default  aten::_sparse_csr_tensor_unsafe(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::_sparse_log_softmax.Dimname  aten::_sparse_log_softmax.Dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::_sparse_log_softmax.default  aten::_sparse_log_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::_sparse_log_softmax.int  aten::_sparse_log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
aten::_sparse_log_softmax.out  aten::_sparse_log_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_log_softmax_backward_data.default  aten::_sparse_log_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
aten::_sparse_log_softmax_backward_data.out  aten::_sparse_log_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_mask_projection.default  aten::_sparse_mask_projection(Tensor self, Tensor mask, bool accumulate_matches=False) -> Tensor
aten::_sparse_mask_projection.out  aten::_sparse_mask_projection.out(Tensor self, Tensor mask, bool accumulate_matches=False, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_mm.default  aten::_sparse_mm(Tensor sparse, Tensor dense) -> Tensor
aten::_sparse_mm.reduce  aten::_sparse_mm.reduce(Tensor sparse, Tensor dense, str reduce) -> Tensor
aten::_sparse_mm_reduce_impl.default  aten::_sparse_mm_reduce_impl(Tensor self, Tensor other, str reduce) -> (Tensor, Tensor)
aten::_sparse_mm_reduce_impl_backward.default  aten::_sparse_mm_reduce_impl_backward(Tensor self, Tensor grad_out, Tensor weight, str reduce, Tensor arg_out, bool[2] output_mask) -> (Tensor, Tensor)
aten::_sparse_semi_structured_addmm.default  aten::_sparse_semi_structured_addmm(Tensor input, Tensor mat1, Tensor mat1_meta, Tensor mat2, *, Scalar alpha=1, Scalar beta=1, ScalarType? out_dtype=None) -> Tensor
aten::_sparse_semi_structured_apply.default  aten::_sparse_semi_structured_apply(Tensor input, Tensor thread_masks) -> (Tensor, Tensor)
aten::_sparse_semi_structured_apply_dense.default  aten::_sparse_semi_structured_apply_dense(Tensor input, Tensor thread_masks) -> Tensor
aten::_sparse_semi_structured_linear.default  aten::_sparse_semi_structured_linear(Tensor input, Tensor weight, Tensor meta, *, Tensor? bias=None, str? activation=None, ScalarType? out_dtype=None) -> Tensor
aten::_sparse_semi_structured_mm.default  aten::_sparse_semi_structured_mm(Tensor mat1, Tensor mat1_meta, Tensor mat2, *, ScalarType? out_dtype=None) -> Tensor
aten::_sparse_semi_structured_tile.default  aten::_sparse_semi_structured_tile(Tensor input, str algorithm="", bool use_cutlass=True) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_sparse_softmax.Dimname  aten::_sparse_softmax.Dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::_sparse_softmax.default  aten::_sparse_softmax(Tensor self, int dim, bool half_to_float) -> Tensor
aten::_sparse_softmax.int  aten::_sparse_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
aten::_sparse_softmax.out  aten::_sparse_softmax.out(Tensor self, int dim, bool half_to_float, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_softmax_backward_data.default  aten::_sparse_softmax_backward_data(Tensor grad_output, Tensor output, int dim, Tensor self) -> Tensor
aten::_sparse_softmax_backward_data.out  aten::_sparse_softmax_backward_data.out(Tensor grad_output, Tensor output, int dim, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_sparse_matmul.default  aten::_sparse_sparse_matmul(Tensor self, Tensor other) -> Tensor
aten::_sparse_sparse_matmul.out  aten::_sparse_sparse_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_sum.default  aten::_sparse_sum(Tensor self) -> Tensor
aten::_sparse_sum.dim  aten::_sparse_sum.dim(Tensor self, int[1] dim) -> Tensor
aten::_sparse_sum.dim_dtype  aten::_sparse_sum.dim_dtype(Tensor self, int[1] dim, *, ScalarType dtype) -> Tensor
aten::_sparse_sum.dim_out  aten::_sparse_sum.dim_out(Tensor self, int[1] dim, *, Tensor(a!) out) -> Tensor(a!)
aten::_sparse_sum.dtype  aten::_sparse_sum.dtype(Tensor self, *, ScalarType dtype) -> Tensor
aten::_sparse_sum_backward.default  aten::_sparse_sum_backward(Tensor grad, Tensor self, int[] dim) -> Tensor
aten::_sparse_sum_backward.out  aten::_sparse_sum_backward.out(Tensor grad, Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)
aten::_spdiags.default  aten::_spdiags(Tensor diagonals, Tensor offsets, int[] shape, Layout? layout=None) -> Tensor
aten::_spdiags.out  aten::_spdiags.out(Tensor diagonals, Tensor offsets, int[] shape, Layout? layout=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_stack.default  aten::_stack(Tensor[] tensors, int dim=0) -> Tensor
aten::_stack.out  aten::_stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
aten::_standard_gamma.default  aten::_standard_gamma(Tensor self, Generator? generator=None) -> Tensor
aten::_standard_gamma.out  aten::_standard_gamma.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_standard_gamma_grad.default  aten::_standard_gamma_grad(Tensor self, Tensor output) -> Tensor
aten::_standard_gamma_grad.out  aten::_standard_gamma_grad.out(Tensor self, Tensor output, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_ambiguous_defaults.a  aten::_test_ambiguous_defaults.a(Tensor dummy, int a=1, int b=1) -> Tensor
aten::_test_ambiguous_defaults.b  aten::_test_ambiguous_defaults.b(Tensor dummy, int a=2, str b="2") -> Tensor
aten::_test_autograd_multiple_dispatch.fullcoverage  aten::_test_autograd_multiple_dispatch.fullcoverage(Tensor self) -> Tensor
aten::_test_autograd_multiple_dispatch.fullcoverage_out  aten::_test_autograd_multiple_dispatch.fullcoverage_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_autograd_multiple_dispatch.ntonly  aten::_test_autograd_multiple_dispatch.ntonly(Tensor self, bool b) -> Tensor
aten::_test_autograd_multiple_dispatch_view.default  aten::_test_autograd_multiple_dispatch_view(Tensor(a) self) -> Tensor(a)
aten::_test_autograd_multiple_dispatch_view_copy.default  aten::_test_autograd_multiple_dispatch_view_copy(Tensor self) -> Tensor
aten::_test_autograd_multiple_dispatch_view_copy.out  aten::_test_autograd_multiple_dispatch_view_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_check_tensor.default  aten::_test_check_tensor(Tensor self) -> Tensor
aten::_test_functorch_fallback.default  aten::_test_functorch_fallback(Tensor self, Tensor other) -> Tensor
aten::_test_functorch_fallback.out  aten::_test_functorch_fallback.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_optional_filled_intlist.default  aten::_test_optional_filled_intlist(Tensor values, int[2]? addends) -> Tensor
aten::_test_optional_filled_intlist.out  aten::_test_optional_filled_intlist.out(Tensor values, int[2]? addends, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_optional_floatlist.default  aten::_test_optional_floatlist(Tensor values, float[]? addends) -> Tensor
aten::_test_optional_floatlist.out  aten::_test_optional_floatlist.out(Tensor values, float[]? addends, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_optional_intlist.default  aten::_test_optional_intlist(Tensor values, int[]? addends) -> Tensor
aten::_test_optional_intlist.out  aten::_test_optional_intlist.out(Tensor values, int[]? addends, *, Tensor(a!) out) -> Tensor(a!)
aten::_test_parallel_materialize.default  aten::_test_parallel_materialize(Tensor self, int num_parallel, bool skip_first=False) -> Tensor
aten::_test_serialization_subcmul.default  aten::_test_serialization_subcmul(Tensor self, Tensor other, Scalar alpha=1) -> Tensor
aten::_test_string_default.default  aten::_test_string_default(Tensor dummy, str a="\"\'\\", str b="\"\'\\") -> Tensor
aten::_test_warn_in_autograd.default  aten::_test_warn_in_autograd(Tensor self) -> Tensor
aten::_test_warn_in_autograd.out  aten::_test_warn_in_autograd.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_thnn_differentiable_gru_cell_backward.default  aten::_thnn_differentiable_gru_cell_backward(Tensor grad_hy, Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias, Tensor? hidden_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_thnn_differentiable_lstm_cell_backward.default  aten::_thnn_differentiable_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor input_gates, Tensor hidden_gates, Tensor? input_bias, Tensor? hidden_bias, Tensor cx, Tensor cy) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_thnn_fused_gru_cell.default  aten::_thnn_fused_gru_cell(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor)
aten::_thnn_fused_gru_cell.out  aten::_thnn_fused_gru_cell.out(Tensor input_gates, Tensor hidden_gates, Tensor hx, Tensor? input_bias=None, Tensor? hidden_bias=None, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_thnn_fused_gru_cell_backward.default  aten::_thnn_fused_gru_cell_backward(Tensor grad_hy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_thnn_fused_gru_cell_backward.out  aten::_thnn_fused_gru_cell_backward.out(Tensor grad_hy, Tensor workspace, bool has_bias, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))
aten::_thnn_fused_lstm_cell.default  aten::_thnn_fused_lstm_cell(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None) -> (Tensor, Tensor, Tensor)
aten::_thnn_fused_lstm_cell.out  aten::_thnn_fused_lstm_cell.out(Tensor input_gates, Tensor hidden_gates, Tensor cx, Tensor? input_bias=None, Tensor? hidden_bias=None, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_thnn_fused_lstm_cell_backward.default  aten::_thnn_fused_lstm_cell_backward(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::_thnn_fused_lstm_cell_backward_impl.default  aten::_thnn_fused_lstm_cell_backward_impl(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias) -> (Tensor, Tensor, Tensor)
aten::_thnn_fused_lstm_cell_backward_impl.out  aten::_thnn_fused_lstm_cell_backward_impl.out(Tensor? grad_hy, Tensor? grad_cy, Tensor cx, Tensor cy, Tensor workspace, bool has_bias, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_to_copy.default  aten::_to_copy(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, MemoryFormat? memory_format=None) -> Tensor
aten::_to_copy.out  aten::_to_copy.out(Tensor self, *, bool non_blocking=False, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::_to_cpu.default  aten::_to_cpu(Tensor[] tensors) -> Tensor[]
aten::_to_dense.default  aten::_to_dense(Tensor self, ScalarType? dtype=None, bool? masked_grad=None) -> Tensor
aten::_to_dense.out  aten::_to_dense.out(Tensor self, ScalarType? dtype=None, bool? masked_grad=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse.default  aten::_to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
aten::_to_sparse.out  aten::_to_sparse.out(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse.sparse_dim  aten::_to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
aten::_to_sparse.sparse_dim_out  aten::_to_sparse.sparse_dim_out(Tensor self, int sparse_dim, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse_bsc.default  aten::_to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
aten::_to_sparse_bsc.out  aten::_to_sparse_bsc.out(Tensor self, int[2] blocksize, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse_bsr.default  aten::_to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
aten::_to_sparse_bsr.out  aten::_to_sparse_bsr.out(Tensor self, int[2] blocksize, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse_csc.default  aten::_to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
aten::_to_sparse_csc.out  aten::_to_sparse_csc.out(Tensor self, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse_csr.default  aten::_to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
aten::_to_sparse_csr.out  aten::_to_sparse_csr.out(Tensor self, int? dense_dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_to_sparse_semi_structured.default  aten::_to_sparse_semi_structured(Tensor dense) -> (Tensor, Tensor)
aten::_transform_bias_rescale_qkv.default  aten::_transform_bias_rescale_qkv(Tensor qkv, Tensor qkv_bias, int num_heads) -> (Tensor, Tensor, Tensor)
aten::_transform_bias_rescale_qkv.out  aten::_transform_bias_rescale_qkv.out(Tensor qkv, Tensor qkv_bias, int num_heads, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_transformer_encoder_layer_fwd.default  aten::_transformer_encoder_layer_fwd(Tensor src, int embed_dim, int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None, int? mask_type=None) -> Tensor
aten::_transformer_encoder_layer_fwd.out  aten::_transformer_encoder_layer_fwd.out(Tensor src, int embed_dim, int num_heads, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, bool use_gelu, bool norm_first, float eps, Tensor norm_weight_1, Tensor norm_bias_1, Tensor norm_weight_2, Tensor norm_bias_2, Tensor ffn_weight_1, Tensor ffn_bias_1, Tensor ffn_weight_2, Tensor ffn_bias_2, Tensor? mask=None, int? mask_type=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_trilinear.default  aten::_trilinear(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1) -> Tensor
aten::_trilinear.out  aten::_trilinear.out(Tensor i1, Tensor i2, Tensor i3, int[] expand1, int[] expand2, int[] expand3, int[] sumdim, int unroll_dim=1, *, Tensor(a!) out) -> Tensor(a!)
aten::_triton_multi_head_attention.default  aten::_triton_multi_head_attention(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None) -> Tensor
aten::_triton_multi_head_attention.out  aten::_triton_multi_head_attention.out(Tensor query, Tensor key, Tensor value, int embed_dim, int num_head, Tensor qkv_weight, Tensor qkv_bias, Tensor proj_weight, Tensor proj_bias, Tensor? mask=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_triton_scaled_dot_attention.default  aten::_triton_scaled_dot_attention(Tensor q, Tensor k, Tensor v, float dropout_p=0.) -> Tensor
aten::_triton_scaled_dot_attention.out  aten::_triton_scaled_dot_attention.out(Tensor q, Tensor k, Tensor v, float dropout_p=0., *, Tensor(a!) out) -> Tensor(a!)
aten::_unique.default  aten::_unique(Tensor self, bool sorted=True, bool return_inverse=False) -> (Tensor, Tensor)
aten::_unique.out  aten::_unique.out(Tensor self, bool sorted=True, bool return_inverse=False, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_unique2.default  aten::_unique2(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
aten::_unique2.out  aten::_unique2.out(Tensor self, bool sorted=True, bool return_inverse=False, bool return_counts=False, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::_unpack_dual.default  aten::_unpack_dual(Tensor(a) dual, int level) -> (Tensor(a) primal, Tensor tangent)
aten::_unsafe_index.Tensor  aten::_unsafe_index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
aten::_unsafe_index_put.default  aten::_unsafe_index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
aten::_unsafe_view.default  aten::_unsafe_view(Tensor self, SymInt[] size) -> Tensor
aten::_unsafe_view.out  aten::_unsafe_view.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_bicubic2d_aa.default  aten::_upsample_bicubic2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_bicubic2d_aa.out  aten::_upsample_bicubic2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_bicubic2d_aa.vec  aten::_upsample_bicubic2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::_upsample_bicubic2d_aa_backward.default  aten::_upsample_bicubic2d_aa_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_bicubic2d_aa_backward.grad_input  aten::_upsample_bicubic2d_aa_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_upsample_bilinear2d_aa.default  aten::_upsample_bilinear2d_aa(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_bilinear2d_aa.out  aten::_upsample_bilinear2d_aa.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_bilinear2d_aa.vec  aten::_upsample_bilinear2d_aa.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::_upsample_bilinear2d_aa_backward.default  aten::_upsample_bilinear2d_aa_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_bilinear2d_aa_backward.grad_input  aten::_upsample_bilinear2d_aa_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_upsample_nearest_exact1d.default  aten::_upsample_nearest_exact1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor
aten::_upsample_nearest_exact1d.out  aten::_upsample_nearest_exact1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_nearest_exact1d.vec  aten::_upsample_nearest_exact1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::_upsample_nearest_exact1d_backward.default  aten::_upsample_nearest_exact1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None) -> Tensor
aten::_upsample_nearest_exact1d_backward.grad_input  aten::_upsample_nearest_exact1d_backward.grad_input(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_upsample_nearest_exact2d.default  aten::_upsample_nearest_exact2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_nearest_exact2d.out  aten::_upsample_nearest_exact2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_nearest_exact2d.vec  aten::_upsample_nearest_exact2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::_upsample_nearest_exact2d_backward.default  aten::_upsample_nearest_exact2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_nearest_exact2d_backward.grad_input  aten::_upsample_nearest_exact2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_upsample_nearest_exact3d.default  aten::_upsample_nearest_exact3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_nearest_exact3d.out  aten::_upsample_nearest_exact3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::_upsample_nearest_exact3d.vec  aten::_upsample_nearest_exact3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::_upsample_nearest_exact3d_backward.default  aten::_upsample_nearest_exact3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::_upsample_nearest_exact3d_backward.grad_input  aten::_upsample_nearest_exact3d_backward.grad_input(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::_use_cudnn_ctc_loss.Tensor  aten::_use_cudnn_ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank) -> bool
aten::_use_cudnn_ctc_loss.default  aten::_use_cudnn_ctc_loss(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank) -> bool
aten::_use_cudnn_rnn_flatten_weight.default  aten::_use_cudnn_rnn_flatten_weight() -> bool
aten::_validate_compressed_sparse_indices.default  aten::_validate_compressed_sparse_indices(bool is_crow, Tensor compressed_idx, Tensor plain_idx, int cdim, int dim, int nnz) -> ()
aten::_validate_sparse_bsc_tensor_args.default  aten::_validate_sparse_bsc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size) -> ()
aten::_validate_sparse_bsr_tensor_args.default  aten::_validate_sparse_bsr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
aten::_validate_sparse_compressed_tensor_args.default  aten::_validate_sparse_compressed_tensor_args(Tensor compressed_indices, Tensor plain_indices, Tensor values, int[] size, Layout layout) -> ()
aten::_validate_sparse_coo_tensor_args.default  aten::_validate_sparse_coo_tensor_args(Tensor indices, Tensor values, int[] size, bool? is_coalesced=None) -> ()
aten::_validate_sparse_csc_tensor_args.default  aten::_validate_sparse_csc_tensor_args(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size) -> ()
aten::_validate_sparse_csr_tensor_args.default  aten::_validate_sparse_csr_tensor_args(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size) -> ()
aten::_values.default  aten::_values(Tensor(a) self) -> Tensor(a)
aten::_values_copy.default  aten::_values_copy(Tensor self) -> Tensor
aten::_values_copy.out  aten::_values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::_version.default  aten::_version(Tensor self) -> int
aten::_weight_int4pack_mm.default  aten::_weight_int4pack_mm(Tensor self, Tensor mat2, int qGroupSize, Tensor qScaleAndZeros) -> Tensor
aten::_weight_int8pack_mm.default  aten::_weight_int8pack_mm(Tensor self, Tensor mat2, Tensor scales) -> Tensor
aten::_weight_norm.default  aten::_weight_norm(Tensor v, Tensor g, int dim=0) -> Tensor
aten::_weight_norm_differentiable_backward.default  aten::_weight_norm_differentiable_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
aten::_weight_norm_interface.default  aten::_weight_norm_interface(Tensor v, Tensor g, int dim=0) -> (Tensor, Tensor)
aten::_weight_norm_interface.out  aten::_weight_norm_interface.out(Tensor v, Tensor g, int dim=0, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::_weight_norm_interface_backward.default  aten::_weight_norm_interface_backward(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim) -> (Tensor, Tensor)
aten::_weight_norm_interface_backward.out  aten::_weight_norm_interface_backward.out(Tensor grad_w, Tensor saved_v, Tensor saved_g, Tensor saved_norms, int dim, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::abs.default  aten::abs(Tensor self) -> Tensor
aten::abs.out  aten::abs.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::abs_.default  aten::abs_(Tensor(a!) self) -> Tensor(a!)
aten::absolute.default  aten::absolute(Tensor self) -> Tensor
aten::absolute.out  aten::absolute.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::absolute_.default  aten::absolute_(Tensor(a!) self) -> Tensor(a!)
aten::acos.default  aten::acos(Tensor self) -> Tensor
aten::acos.out  aten::acos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::acos_.default  aten::acos_(Tensor(a!) self) -> Tensor(a!)
aten::acosh.default  aten::acosh(Tensor self) -> Tensor
aten::acosh.out  aten::acosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::acosh_.default  aten::acosh_(Tensor(a!) self) -> Tensor(a!)
aten::adaptive_avg_pool1d.default  aten::adaptive_avg_pool1d(Tensor self, int[1] output_size) -> Tensor
aten::adaptive_avg_pool2d.default  aten::adaptive_avg_pool2d(Tensor self, SymInt[2] output_size) -> Tensor
aten::adaptive_avg_pool2d.out  aten::adaptive_avg_pool2d.out(Tensor self, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::adaptive_avg_pool3d.default  aten::adaptive_avg_pool3d(Tensor self, SymInt[3] output_size) -> Tensor
aten::adaptive_avg_pool3d.out  aten::adaptive_avg_pool3d.out(Tensor self, SymInt[3] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::adaptive_avg_pool3d_backward.grad_input  aten::adaptive_avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::adaptive_max_pool1d.default  aten::adaptive_max_pool1d(Tensor self, int[1] output_size) -> (Tensor, Tensor)
aten::adaptive_max_pool2d.default  aten::adaptive_max_pool2d(Tensor self, int[2] output_size) -> (Tensor, Tensor)
aten::adaptive_max_pool2d.out  aten::adaptive_max_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::adaptive_max_pool2d_backward.default  aten::adaptive_max_pool2d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
aten::adaptive_max_pool2d_backward.grad_input  aten::adaptive_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::adaptive_max_pool3d.default  aten::adaptive_max_pool3d(Tensor self, int[3] output_size) -> (Tensor, Tensor)
aten::adaptive_max_pool3d.out  aten::adaptive_max_pool3d.out(Tensor self, int[3] output_size, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::adaptive_max_pool3d_backward.default  aten::adaptive_max_pool3d_backward(Tensor grad_output, Tensor self, Tensor indices) -> Tensor
aten::adaptive_max_pool3d_backward.grad_input  aten::adaptive_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::add.Scalar  aten::add.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::add.Scalar_out  aten::add.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
aten::add.Tensor  aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
aten::add.out  aten::add.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::add_.Scalar  aten::add_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
aten::add_.Tensor  aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
aten::addbmm.default  aten::addbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::addbmm.out  aten::addbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::addbmm_.default  aten::addbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
aten::addcdiv.default  aten::addcdiv(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
aten::addcdiv.out  aten::addcdiv.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
aten::addcdiv_.default  aten::addcdiv_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
aten::addcmul.default  aten::addcmul(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor
aten::addcmul.out  aten::addcmul.out(Tensor self, Tensor tensor1, Tensor tensor2, *, Scalar value=1, Tensor(a!) out) -> Tensor(a!)
aten::addcmul_.default  aten::addcmul_(Tensor(a!) self, Tensor tensor1, Tensor tensor2, *, Scalar value=1) -> Tensor(a!)
aten::addmm.default  aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::addmm.out  aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::addmm_.default  aten::addmm_(Tensor(a!) self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
aten::addmv.default  aten::addmv(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::addmv.out  aten::addmv.out(Tensor self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::addmv_.default  aten::addmv_(Tensor(a!) self, Tensor mat, Tensor vec, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
aten::addr.default  aten::addr(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::addr.out  aten::addr.out(Tensor self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::addr_.default  aten::addr_(Tensor(a!) self, Tensor vec1, Tensor vec2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
aten::adjoint.default  aten::adjoint(Tensor(a) self) -> Tensor(a)
aten::affine_grid_generator.default  aten::affine_grid_generator(Tensor theta, SymInt[] size, bool align_corners) -> Tensor
aten::affine_grid_generator.out  aten::affine_grid_generator.out(Tensor theta, SymInt[] size, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
aten::affine_grid_generator_backward.default  aten::affine_grid_generator_backward(Tensor grad, SymInt[] size, bool align_corners) -> Tensor
aten::alias.default  aten::alias(Tensor(a) self) -> Tensor(a)
aten::alias_copy.default  aten::alias_copy(Tensor self) -> Tensor
aten::alias_copy.out  aten::alias_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::align_as.default  aten::align_as(Tensor self, Tensor other) -> Tensor
aten::align_tensors.default  aten::align_tensors(Tensor[] tensors) -> Tensor[]
aten::align_to.default  aten::align_to(Tensor(a) self, str[] names) -> Tensor(a)
aten::align_to.ellipsis_idx  aten::align_to.ellipsis_idx(Tensor(a) self, str[] order, int ellipsis_idx) -> Tensor(a)
aten::all.all_out  aten::all.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::all.default  aten::all(Tensor self) -> Tensor
aten::all.dim  aten::all.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
aten::all.dimname  aten::all.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor
aten::all.dimname_out  aten::all.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::all.dims  aten::all.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
aten::all.dims_out  aten::all.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::all.out  aten::all.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::allclose.default  aten::allclose(Tensor self, Tensor other, float rtol=1.0000000000000001e-05, float atol=1e-08, bool equal_nan=False) -> bool
aten::alpha_dropout.default  aten::alpha_dropout(Tensor input, float p, bool train) -> Tensor
aten::alpha_dropout_.default  aten::alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
aten::amax.default  aten::amax(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
aten::amax.out  aten::amax.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::amin.default  aten::amin(Tensor self, int[1] dim=[], bool keepdim=False) -> Tensor
aten::amin.out  aten::amin.out(Tensor self, int[1] dim=[], bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::aminmax.default  aten::aminmax(Tensor self, *, int? dim=None, bool keepdim=False) -> (Tensor min, Tensor max)
aten::aminmax.out  aten::aminmax.out(Tensor self, *, int? dim=None, bool keepdim=False, Tensor(a!) min, Tensor(b!) max) -> (Tensor(a!) min, Tensor(b!) max)
aten::angle.default  aten::angle(Tensor self) -> Tensor
aten::angle.out  aten::angle.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::any.all_out  aten::any.all_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::any.default  aten::any(Tensor self) -> Tensor
aten::any.dim  aten::any.dim(Tensor self, int dim, bool keepdim=False) -> Tensor
aten::any.dimname  aten::any.dimname(Tensor self, str dim, bool keepdim=False) -> Tensor
aten::any.dimname_out  aten::any.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::any.dims  aten::any.dims(Tensor self, int[]? dim=None, bool keepdim=False) -> Tensor
aten::any.dims_out  aten::any.dims_out(Tensor self, int[]? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::any.out  aten::any.out(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::arange.default  aten::arange(Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::arange.out  aten::arange.out(Scalar end, *, Tensor(a!) out) -> Tensor(a!)
aten::arange.start  aten::arange.start(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::arange.start_out  aten::arange.start_out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
aten::arange.start_step  aten::arange.start_step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::arccos.default  aten::arccos(Tensor self) -> Tensor
aten::arccos.out  aten::arccos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arccos_.default  aten::arccos_(Tensor(a!) self) -> Tensor(a!)
aten::arccosh.default  aten::arccosh(Tensor self) -> Tensor
aten::arccosh.out  aten::arccosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arccosh_.default  aten::arccosh_(Tensor(a!) self) -> Tensor(a!)
aten::arcsin.default  aten::arcsin(Tensor self) -> Tensor
aten::arcsin.out  aten::arcsin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arcsin_.default  aten::arcsin_(Tensor(a!) self) -> Tensor(a!)
aten::arcsinh.default  aten::arcsinh(Tensor self) -> Tensor
aten::arcsinh.out  aten::arcsinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arcsinh_.default  aten::arcsinh_(Tensor(a!) self) -> Tensor(a!)
aten::arctan.default  aten::arctan(Tensor self) -> Tensor
aten::arctan.out  aten::arctan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arctan2.default  aten::arctan2(Tensor self, Tensor other) -> Tensor
aten::arctan2.out  aten::arctan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::arctan2_.default  aten::arctan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::arctan_.default  aten::arctan_(Tensor(a!) self) -> Tensor(a!)
aten::arctanh.default  aten::arctanh(Tensor self) -> Tensor
aten::arctanh.out  aten::arctanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::arctanh_.default  aten::arctanh_(Tensor(a!) self) -> Tensor(a!)
aten::argmax.default  aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
aten::argmax.out  aten::argmax.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::argmin.default  aten::argmin(Tensor self, int? dim=None, bool keepdim=False) -> Tensor
aten::argmin.out  aten::argmin.out(Tensor self, int? dim=None, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::argsort.default  aten::argsort(Tensor self, int dim=-1, bool descending=False) -> Tensor
aten::argsort.dimname  aten::argsort.dimname(Tensor self, str dim, bool descending=False) -> Tensor
aten::argsort.stable  aten::argsort.stable(Tensor self, *, bool stable, int dim=-1, bool descending=False) -> Tensor
aten::argsort.stable_out  aten::argsort.stable_out(Tensor self, *, bool stable, int dim=-1, bool descending=False, Tensor(a!) out) -> Tensor(a!)
aten::argwhere.default  aten::argwhere(Tensor self) -> Tensor
aten::as_strided.default  aten::as_strided(Tensor(a) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a)
aten::as_strided_.default  aten::as_strided_(Tensor(a!) self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor(a!)
aten::as_strided_copy.default  aten::as_strided_copy(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
aten::as_strided_copy.out  aten::as_strided_copy.out(Tensor self, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)
aten::as_strided_scatter.default  aten::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None) -> Tensor
aten::as_strided_scatter.out  aten::as_strided_scatter.out(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt? storage_offset=None, *, Tensor(a!) out) -> Tensor(a!)
aten::asin.default  aten::asin(Tensor self) -> Tensor
aten::asin.out  aten::asin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::asin_.default  aten::asin_(Tensor(a!) self) -> Tensor(a!)
aten::asinh.default  aten::asinh(Tensor self) -> Tensor
aten::asinh.out  aten::asinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::asinh_.default  aten::asinh_(Tensor(a!) self) -> Tensor(a!)
aten::atan.default  aten::atan(Tensor self) -> Tensor
aten::atan.out  aten::atan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::atan2.default  aten::atan2(Tensor self, Tensor other) -> Tensor
aten::atan2.out  aten::atan2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::atan2_.default  aten::atan2_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::atan_.default  aten::atan_(Tensor(a!) self) -> Tensor(a!)
aten::atanh.default  aten::atanh(Tensor self) -> Tensor
aten::atanh.out  aten::atanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::atanh_.default  aten::atanh_(Tensor(a!) self) -> Tensor(a!)
aten::atleast_1d.Sequence  aten::atleast_1d.Sequence(Tensor[] tensors) -> Tensor[]
aten::atleast_1d.default  aten::atleast_1d(Tensor self) -> Tensor
aten::atleast_2d.Sequence  aten::atleast_2d.Sequence(Tensor[] tensors) -> Tensor[]
aten::atleast_2d.default  aten::atleast_2d(Tensor self) -> Tensor
aten::atleast_3d.Sequence  aten::atleast_3d.Sequence(Tensor[] tensors) -> Tensor[]
aten::atleast_3d.default  aten::atleast_3d(Tensor self) -> Tensor
aten::avg_pool1d.default  aten::avg_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], bool ceil_mode=False, bool count_include_pad=True) -> Tensor
aten::avg_pool2d.default  aten::avg_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
aten::avg_pool2d.out  aten::avg_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
aten::avg_pool2d_backward.default  aten::avg_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
aten::avg_pool2d_backward.grad_input  aten::avg_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::avg_pool3d.default  aten::avg_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None) -> Tensor
aten::avg_pool3d.out  aten::avg_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, bool ceil_mode=False, bool count_include_pad=True, int? divisor_override=None, *, Tensor(a!) out) -> Tensor(a!)
aten::avg_pool3d_backward.default  aten::avg_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override) -> Tensor
aten::avg_pool3d_backward.grad_input  aten::avg_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, bool ceil_mode, bool count_include_pad, int? divisor_override, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::baddbmm.default  aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::baddbmm.out  aten::baddbmm.out(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::baddbmm_.default  aten::baddbmm_(Tensor(a!) self, Tensor batch1, Tensor batch2, *, Scalar beta=1, Scalar alpha=1) -> Tensor(a!)
aten::bartlett_window.default  aten::bartlett_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::bartlett_window.out  aten::bartlett_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
aten::bartlett_window.periodic  aten::bartlett_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::bartlett_window.periodic_out  aten::bartlett_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
aten::batch_norm.default  aten::batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, bool cudnn_enabled) -> Tensor
aten::batch_norm_backward.default  aten::batch_norm_backward(Tensor grad_out, Tensor input, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, bool update, float eps, bool[3] output_mask, Tensor reserve) -> (Tensor, Tensor, Tensor)
aten::batch_norm_backward_elemt.default  aten::batch_norm_backward_elemt(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count) -> Tensor
aten::batch_norm_backward_elemt.out  aten::batch_norm_backward_elemt.out(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, Tensor sum_dy, Tensor sum_dy_xmu, Tensor count, *, Tensor(a!) out) -> Tensor(a!)
aten::batch_norm_backward_reduce.default  aten::batch_norm_backward_reduce(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g) -> (Tensor, Tensor, Tensor, Tensor)
aten::batch_norm_backward_reduce.out  aten::batch_norm_backward_reduce.out(Tensor grad_out, Tensor input, Tensor mean, Tensor invstd, Tensor? weight, bool input_g, bool weight_g, bool bias_g, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::batch_norm_elemt.default  aten::batch_norm_elemt(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps) -> Tensor
aten::batch_norm_elemt.out  aten::batch_norm_elemt.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor invstd, float eps, *, Tensor(a!) out) -> Tensor(a!)
aten::batch_norm_gather_stats.default  aten::batch_norm_gather_stats(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count) -> (Tensor, Tensor)
aten::batch_norm_gather_stats.out  aten::batch_norm_gather_stats.out(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, int count, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::batch_norm_gather_stats_with_counts.default  aten::batch_norm_gather_stats_with_counts(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts) -> (Tensor, Tensor)
aten::batch_norm_gather_stats_with_counts.out  aten::batch_norm_gather_stats_with_counts.out(Tensor input, Tensor mean, Tensor invstd, Tensor? running_mean, Tensor? running_var, float momentum, float eps, Tensor counts, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::batch_norm_stats.default  aten::batch_norm_stats(Tensor input, float eps) -> (Tensor, Tensor)
aten::batch_norm_stats.out  aten::batch_norm_stats.out(Tensor input, float eps, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::batch_norm_update_stats.default  aten::batch_norm_update_stats(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum) -> (Tensor, Tensor)
aten::batch_norm_update_stats.out  aten::batch_norm_update_stats.out(Tensor input, Tensor? running_mean, Tensor? running_var, float momentum, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::bernoulli.Tensor  aten::bernoulli.Tensor(Tensor self, Tensor p, *, Generator? generator=None) -> Tensor
aten::bernoulli.Tensor_out  aten::bernoulli.Tensor_out(Tensor self, Tensor p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::bernoulli.default  aten::bernoulli(Tensor self, *, Generator? generator=None) -> Tensor
aten::bernoulli.float_out  aten::bernoulli.float_out(Tensor self, float p=0.5, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::bernoulli.out  aten::bernoulli.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::bernoulli.p  aten::bernoulli.p(Tensor self, float p, *, Generator? generator=None) -> Tensor
aten::bernoulli_.Tensor  aten::bernoulli_.Tensor(Tensor(a!) self, Tensor p, *, Generator? generator=None) -> Tensor(a!)
aten::bernoulli_.float  aten::bernoulli_.float(Tensor(a!) self, float p=0.5, *, Generator? generator=None) -> Tensor(a!)
aten::bilinear.default  aten::bilinear(Tensor input1, Tensor input2, Tensor weight, Tensor? bias=None) -> Tensor
aten::binary_cross_entropy.default  aten::binary_cross_entropy(Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor
aten::binary_cross_entropy.out  aten::binary_cross_entropy.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::binary_cross_entropy_backward.default  aten::binary_cross_entropy_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1) -> Tensor
aten::binary_cross_entropy_backward.grad_input  aten::binary_cross_entropy_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight=None, int reduction=1, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::binary_cross_entropy_with_logits.default  aten::binary_cross_entropy_with_logits(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=1) -> Tensor
aten::binary_cross_entropy_with_logits.out  aten::binary_cross_entropy_with_logits.out(Tensor self, Tensor target, Tensor? weight=None, Tensor? pos_weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::bincount.default  aten::bincount(Tensor self, Tensor? weights=None, int minlength=0) -> Tensor
aten::bincount.out  aten::bincount.out(Tensor self, Tensor? weights=None, int minlength=0, *, Tensor(a!) out) -> Tensor(a!)
aten::binomial.default  aten::binomial(Tensor count, Tensor prob, Generator? generator=None) -> Tensor
aten::binomial.out  aten::binomial.out(Tensor count, Tensor prob, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_and.Scalar  aten::bitwise_and.Scalar(Tensor self, Scalar other) -> Tensor
aten::bitwise_and.Scalar_Tensor  aten::bitwise_and.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::bitwise_and.Scalar_Tensor_out  aten::bitwise_and.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_and.Scalar_out  aten::bitwise_and.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_and.Tensor  aten::bitwise_and.Tensor(Tensor self, Tensor other) -> Tensor
aten::bitwise_and.Tensor_out  aten::bitwise_and.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_and_.Scalar  aten::bitwise_and_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::bitwise_and_.Tensor  aten::bitwise_and_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::bitwise_left_shift.Scalar_Tensor  aten::bitwise_left_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::bitwise_left_shift.Scalar_Tensor_out  aten::bitwise_left_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_left_shift.Tensor  aten::bitwise_left_shift.Tensor(Tensor self, Tensor other) -> Tensor
aten::bitwise_left_shift.Tensor_Scalar  aten::bitwise_left_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
aten::bitwise_left_shift.Tensor_Scalar_out  aten::bitwise_left_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_left_shift.Tensor_out  aten::bitwise_left_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_left_shift_.Tensor  aten::bitwise_left_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::bitwise_left_shift_.Tensor_Scalar  aten::bitwise_left_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::bitwise_not.default  aten::bitwise_not(Tensor self) -> Tensor
aten::bitwise_not.out  aten::bitwise_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_not_.default  aten::bitwise_not_(Tensor(a!) self) -> Tensor(a!)
aten::bitwise_or.Scalar  aten::bitwise_or.Scalar(Tensor self, Scalar other) -> Tensor
aten::bitwise_or.Scalar_Tensor  aten::bitwise_or.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::bitwise_or.Scalar_Tensor_out  aten::bitwise_or.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_or.Scalar_out  aten::bitwise_or.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_or.Tensor  aten::bitwise_or.Tensor(Tensor self, Tensor other) -> Tensor
aten::bitwise_or.Tensor_out  aten::bitwise_or.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_or_.Scalar  aten::bitwise_or_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::bitwise_or_.Tensor  aten::bitwise_or_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::bitwise_right_shift.Scalar_Tensor  aten::bitwise_right_shift.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::bitwise_right_shift.Scalar_Tensor_out  aten::bitwise_right_shift.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_right_shift.Tensor  aten::bitwise_right_shift.Tensor(Tensor self, Tensor other) -> Tensor
aten::bitwise_right_shift.Tensor_Scalar  aten::bitwise_right_shift.Tensor_Scalar(Tensor self, Scalar other) -> Tensor
aten::bitwise_right_shift.Tensor_Scalar_out  aten::bitwise_right_shift.Tensor_Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_right_shift.Tensor_out  aten::bitwise_right_shift.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_right_shift_.Tensor  aten::bitwise_right_shift_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::bitwise_right_shift_.Tensor_Scalar  aten::bitwise_right_shift_.Tensor_Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::bitwise_xor.Scalar  aten::bitwise_xor.Scalar(Tensor self, Scalar other) -> Tensor
aten::bitwise_xor.Scalar_Tensor  aten::bitwise_xor.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::bitwise_xor.Scalar_Tensor_out  aten::bitwise_xor.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_xor.Scalar_out  aten::bitwise_xor.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_xor.Tensor  aten::bitwise_xor.Tensor(Tensor self, Tensor other) -> Tensor
aten::bitwise_xor.Tensor_out  aten::bitwise_xor.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::bitwise_xor_.Scalar  aten::bitwise_xor_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::bitwise_xor_.Tensor  aten::bitwise_xor_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::blackman_window.default  aten::blackman_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::blackman_window.out  aten::blackman_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
aten::blackman_window.periodic  aten::blackman_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::blackman_window.periodic_out  aten::blackman_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
aten::block_diag.default  aten::block_diag(Tensor[] tensors) -> Tensor
aten::block_diag.out  aten::block_diag.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::bmm.default  aten::bmm(Tensor self, Tensor mat2) -> Tensor
aten::bmm.out  aten::bmm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
aten::broadcast_tensors.default  aten::broadcast_tensors(Tensor[] tensors) -> Tensor[]
aten::broadcast_to.default  aten::broadcast_to(Tensor(a) self, SymInt[] size) -> Tensor(a)
aten::bucketize.Scalar  aten::bucketize.Scalar(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
aten::bucketize.Scalar_out  aten::bucketize.Scalar_out(Scalar self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
aten::bucketize.Tensor  aten::bucketize.Tensor(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False) -> Tensor
aten::bucketize.Tensor_out  aten::bucketize.Tensor_out(Tensor self, Tensor boundaries, *, bool out_int32=False, bool right=False, Tensor(a!) out) -> Tensor(a!)
aten::can_cast.default  aten::can_cast(ScalarType from_, ScalarType to) -> bool
aten::cartesian_prod.default  aten::cartesian_prod(Tensor[] tensors) -> Tensor
aten::cat.default  aten::cat(Tensor[] tensors, int dim=0) -> Tensor
aten::cat.names  aten::cat.names(Tensor[] tensors, str dim) -> Tensor
aten::cat.names_out  aten::cat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!)
aten::cat.out  aten::cat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
aten::cauchy.default  aten::cauchy(Tensor self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor
aten::cauchy.out  aten::cauchy.out(Tensor self, float median=0., float sigma=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::cauchy_.default  aten::cauchy_(Tensor(a!) self, float median=0., float sigma=1., *, Generator? generator=None) -> Tensor(a!)
aten::ccol_indices.default  aten::ccol_indices(Tensor(a) self) -> Tensor(a)
aten::ccol_indices_copy.default  aten::ccol_indices_copy(Tensor self) -> Tensor
aten::ccol_indices_copy.out  aten::ccol_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::cdist.default  aten::cdist(Tensor x1, Tensor x2, float p=2., int? compute_mode=None) -> Tensor
aten::ceil.default  aten::ceil(Tensor self) -> Tensor
aten::ceil.out  aten::ceil.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::ceil_.default  aten::ceil_(Tensor(a!) self) -> Tensor(a!)
aten::celu.default  aten::celu(Tensor self, Scalar alpha=1.) -> Tensor
aten::celu.out  aten::celu.out(Tensor self, Scalar alpha=1., *, Tensor(a!) out) -> Tensor(a!)
aten::celu_.default  aten::celu_(Tensor(a!) self, Scalar alpha=1.) -> Tensor(a!)
aten::chain_matmul.default  aten::chain_matmul(Tensor[] matrices) -> Tensor
aten::chain_matmul.out  aten::chain_matmul.out(Tensor[] matrices, *, Tensor(a!) out) -> Tensor(a!)
aten::chalf.default  aten::chalf(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
aten::channel_shuffle.default  aten::channel_shuffle(Tensor self, SymInt groups) -> Tensor
aten::channel_shuffle.out  aten::channel_shuffle.out(Tensor self, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::cholesky.default  aten::cholesky(Tensor self, bool upper=False) -> Tensor
aten::cholesky.out  aten::cholesky.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
aten::cholesky_inverse.default  aten::cholesky_inverse(Tensor self, bool upper=False) -> Tensor
aten::cholesky_inverse.out  aten::cholesky_inverse.out(Tensor self, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
aten::cholesky_solve.default  aten::cholesky_solve(Tensor self, Tensor input2, bool upper=False) -> Tensor
aten::cholesky_solve.out  aten::cholesky_solve.out(Tensor self, Tensor input2, bool upper=False, *, Tensor(a!) out) -> Tensor(a!)
aten::choose_qparams_optimized.default  aten::choose_qparams_optimized(Tensor input, int numel, int n_bins, float ratio, int bit_width) -> (Tensor, Tensor)
aten::chunk.default  aten::chunk(Tensor(a -> *) self, int chunks, int dim=0) -> Tensor(a)[]
aten::clamp.Tensor  aten::clamp.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
aten::clamp.Tensor_out  aten::clamp.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp.default  aten::clamp(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
aten::clamp.out  aten::clamp.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp_.Tensor  aten::clamp_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
aten::clamp_.default  aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
aten::clamp_max.Tensor  aten::clamp_max.Tensor(Tensor self, Tensor max) -> Tensor
aten::clamp_max.Tensor_out  aten::clamp_max.Tensor_out(Tensor self, Tensor max, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp_max.default  aten::clamp_max(Tensor self, Scalar max) -> Tensor
aten::clamp_max.out  aten::clamp_max.out(Tensor self, Scalar max, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp_max_.Tensor  aten::clamp_max_.Tensor(Tensor(a!) self, Tensor max) -> Tensor(a!)
aten::clamp_max_.default  aten::clamp_max_(Tensor(a!) self, Scalar max) -> Tensor(a!)
aten::clamp_min.Tensor  aten::clamp_min.Tensor(Tensor self, Tensor min) -> Tensor
aten::clamp_min.Tensor_out  aten::clamp_min.Tensor_out(Tensor self, Tensor min, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp_min.default  aten::clamp_min(Tensor self, Scalar min) -> Tensor
aten::clamp_min.out  aten::clamp_min.out(Tensor self, Scalar min, *, Tensor(a!) out) -> Tensor(a!)
aten::clamp_min_.Tensor  aten::clamp_min_.Tensor(Tensor(a!) self, Tensor min) -> Tensor(a!)
aten::clamp_min_.default  aten::clamp_min_(Tensor(a!) self, Scalar min) -> Tensor(a!)
aten::clip.Tensor  aten::clip.Tensor(Tensor self, Tensor? min=None, Tensor? max=None) -> Tensor
aten::clip.Tensor_out  aten::clip.Tensor_out(Tensor self, Tensor? min=None, Tensor? max=None, *, Tensor(a!) out) -> Tensor(a!)
aten::clip.default  aten::clip(Tensor self, Scalar? min=None, Scalar? max=None) -> Tensor
aten::clip.out  aten::clip.out(Tensor self, Scalar? min=None, Scalar? max=None, *, Tensor(a!) out) -> Tensor(a!)
aten::clip_.Tensor  aten::clip_.Tensor(Tensor(a!) self, Tensor? min=None, Tensor? max=None) -> Tensor(a!)
aten::clip_.default  aten::clip_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)
aten::clone.default  aten::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
aten::clone.out  aten::clone.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::coalesce.default  aten::coalesce(Tensor(a) self) -> Tensor(a)
aten::col2im.default  aten::col2im(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
aten::col2im.out  aten::col2im.out(Tensor self, SymInt[2] output_size, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
aten::col_indices.default  aten::col_indices(Tensor(a) self) -> Tensor(a)
aten::col_indices_copy.default  aten::col_indices_copy(Tensor self) -> Tensor
aten::col_indices_copy.out  aten::col_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::column_stack.default  aten::column_stack(Tensor[] tensors) -> Tensor
aten::column_stack.out  aten::column_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::combinations.default  aten::combinations(Tensor self, int r=2, bool with_replacement=False) -> Tensor
aten::complex.default  aten::complex(Tensor real, Tensor imag) -> Tensor
aten::complex.out  aten::complex.out(Tensor real, Tensor imag, *, Tensor(a!) out) -> Tensor(a!)
aten::concat.default  aten::concat(Tensor[] tensors, int dim=0) -> Tensor
aten::concat.names  aten::concat.names(Tensor[] tensors, str dim) -> Tensor
aten::concat.names_out  aten::concat.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!)
aten::concat.out  aten::concat.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
aten::concatenate.default  aten::concatenate(Tensor[] tensors, int dim=0) -> Tensor
aten::concatenate.names  aten::concatenate.names(Tensor[] tensors, str dim) -> Tensor
aten::concatenate.names_out  aten::concatenate.names_out(Tensor[] tensors, str dim, *, Tensor(a!) out) -> Tensor(a!)
aten::concatenate.out  aten::concatenate.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
aten::conj.default  aten::conj(Tensor(a) self) -> Tensor(a)
aten::conj_physical.default  aten::conj_physical(Tensor self) -> Tensor
aten::conj_physical.out  aten::conj_physical.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::conj_physical_.default  aten::conj_physical_(Tensor(a!) self) -> Tensor(a!)
aten::constant_pad_nd.default  aten::constant_pad_nd(Tensor self, SymInt[] pad, Scalar value=0) -> Tensor
aten::constant_pad_nd.out  aten::constant_pad_nd.out(Tensor self, SymInt[] pad, Scalar value=0, *, Tensor(a!) out) -> Tensor(a!)
aten::contiguous.default  aten::contiguous(Tensor(a) self, *, MemoryFormat memory_format=0) -> Tensor(a)
aten::conv1d.default  aten::conv1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=[1], SymInt[1] padding=[0], SymInt[1] dilation=[1], SymInt groups=1) -> Tensor
aten::conv1d.padding  aten::conv1d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=[1], str padding="valid", SymInt[1] dilation=[1], SymInt groups=1) -> Tensor
aten::conv2d.default  aten::conv2d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] dilation=[1, 1], SymInt groups=1) -> Tensor
aten::conv2d.padding  aten::conv2d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1], str padding="valid", SymInt[2] dilation=[1, 1], SymInt groups=1) -> Tensor
aten::conv3d.default  aten::conv3d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1], SymInt groups=1) -> Tensor
aten::conv3d.padding  aten::conv3d.padding(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], str padding="valid", SymInt[3] dilation=[1, 1, 1], SymInt groups=1) -> Tensor
aten::conv_depthwise3d.default  aten::conv_depthwise3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, SymInt[3] dilation) -> Tensor
aten::conv_depthwise3d.out  aten::conv_depthwise3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, SymInt[3] dilation, *, Tensor(a!) out) -> Tensor(a!)
aten::conv_tbc.default  aten::conv_tbc(Tensor self, Tensor weight, Tensor bias, int pad=0) -> Tensor
aten::conv_tbc.out  aten::conv_tbc.out(Tensor self, Tensor weight, Tensor bias, int pad=0, *, Tensor(a!) out) -> Tensor(a!)
aten::conv_tbc_backward.default  aten::conv_tbc_backward(Tensor self, Tensor input, Tensor weight, Tensor bias, int pad) -> (Tensor, Tensor, Tensor)
aten::conv_transpose1d.default  aten::conv_transpose1d(Tensor input, Tensor weight, Tensor? bias=None, SymInt[1] stride=[1], SymInt[1] padding=[0], SymInt[1] output_padding=[0], SymInt groups=1, SymInt[1] dilation=[1]) -> Tensor
aten::conv_transpose2d.input  aten::conv_transpose2d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] output_padding=[0, 0], SymInt groups=1, SymInt[2] dilation=[1, 1]) -> Tensor
aten::conv_transpose3d.input  aten::conv_transpose3d.input(Tensor input, Tensor weight, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] output_padding=[0, 0, 0], SymInt groups=1, SymInt[3] dilation=[1, 1, 1]) -> Tensor
aten::convolution.default  aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
aten::convolution.out  aten::convolution.out(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::convolution_backward.default  aten::convolution_backward(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::convolution_backward.out  aten::convolution_backward.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[]? bias_sizes, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::convolution_backward_overrideable.default  aten::convolution_backward_overrideable(Tensor grad_output, Tensor input, Tensor weight, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask) -> (Tensor grad_input, Tensor grad_weight, Tensor grad_bias)
aten::convolution_backward_overrideable.out  aten::convolution_backward_overrideable.out(Tensor grad_output, Tensor input, Tensor weight, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::convolution_overrideable.default  aten::convolution_overrideable(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups) -> Tensor
aten::convolution_overrideable.out  aten::convolution_overrideable.out(Tensor input, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, bool transposed, SymInt[] output_padding, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::copy.default  aten::copy(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
aten::copy.out  aten::copy.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
aten::copy_.default  aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
aten::copy_sparse_to_sparse.default  aten::copy_sparse_to_sparse(Tensor self, Tensor src, bool non_blocking=False) -> Tensor
aten::copy_sparse_to_sparse.out  aten::copy_sparse_to_sparse.out(Tensor self, Tensor src, bool non_blocking=False, *, Tensor(a!) out) -> Tensor(a!)
aten::copy_sparse_to_sparse_.default  aten::copy_sparse_to_sparse_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
aten::copysign.Scalar  aten::copysign.Scalar(Tensor self, Scalar other) -> Tensor
aten::copysign.Scalar_out  aten::copysign.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::copysign.Tensor  aten::copysign.Tensor(Tensor self, Tensor other) -> Tensor
aten::copysign.out  aten::copysign.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::copysign_.Scalar  aten::copysign_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::copysign_.Tensor  aten::copysign_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::corrcoef.default  aten::corrcoef(Tensor self) -> Tensor
aten::cos.default  aten::cos(Tensor self) -> Tensor
aten::cos.out  aten::cos.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::cos_.default  aten::cos_(Tensor(a!) self) -> Tensor(a!)
aten::cosh.default  aten::cosh(Tensor self) -> Tensor
aten::cosh.out  aten::cosh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::cosh_.default  aten::cosh_(Tensor(a!) self) -> Tensor(a!)
aten::cosine_embedding_loss.default  aten::cosine_embedding_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> Tensor
aten::cosine_similarity.default  aten::cosine_similarity(Tensor x1, Tensor x2, int dim=1, float eps=1e-08) -> Tensor
aten::count_nonzero.default  aten::count_nonzero(Tensor self, int? dim=None) -> Tensor
aten::count_nonzero.dim_IntList  aten::count_nonzero.dim_IntList(Tensor self, int[] dim) -> Tensor
aten::count_nonzero.dim_IntList_out  aten::count_nonzero.dim_IntList_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)
aten::count_nonzero.out  aten::count_nonzero.out(Tensor self, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::cov.default  aten::cov(Tensor self, *, int correction=1, Tensor? fweights=None, Tensor? aweights=None) -> Tensor
aten::cross.default  aten::cross(Tensor self, Tensor other, int? dim=None) -> Tensor
aten::cross.out  aten::cross.out(Tensor self, Tensor other, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::cross_entropy_loss.default  aten::cross_entropy_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, float label_smoothing=0.) -> Tensor
aten::crow_indices.default  aten::crow_indices(Tensor(a) self) -> Tensor(a)
aten::crow_indices_copy.default  aten::crow_indices_copy(Tensor self) -> Tensor
aten::crow_indices_copy.out  aten::crow_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::ctc_loss.IntList  aten::ctc_loss.IntList(Tensor log_probs, Tensor targets, int[] input_lengths, int[] target_lengths, int blank=0, int reduction=1, bool zero_infinity=False) -> Tensor
aten::ctc_loss.Tensor  aten::ctc_loss.Tensor(Tensor log_probs, Tensor targets, Tensor input_lengths, Tensor target_lengths, int blank=0, int reduction=1, bool zero_infinity=False) -> Tensor
aten::cudnn_affine_grid_generator.default  aten::cudnn_affine_grid_generator(Tensor theta, int N, int C, int H, int W) -> Tensor grid
aten::cudnn_affine_grid_generator.out  aten::cudnn_affine_grid_generator.out(Tensor theta, int N, int C, int H, int W, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_affine_grid_generator_backward.default  aten::cudnn_affine_grid_generator_backward(Tensor grad, int N, int C, int H, int W) -> Tensor grad_theta
aten::cudnn_affine_grid_generator_backward.out  aten::cudnn_affine_grid_generator_backward.out(Tensor grad, int N, int C, int H, int W, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_batch_norm.default  aten::cudnn_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor, Tensor)
aten::cudnn_batch_norm.out  aten::cudnn_batch_norm.out(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::cudnn_batch_norm_backward.default  aten::cudnn_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace) -> (Tensor, Tensor, Tensor)
aten::cudnn_batch_norm_backward.out  aten::cudnn_batch_norm_backward.out(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, Tensor reserveSpace, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::cudnn_convolution.default  aten::cudnn_convolution(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
aten::cudnn_convolution.out  aten::cudnn_convolution.out(Tensor self, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_convolution_add_relu.default  aten::cudnn_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
aten::cudnn_convolution_add_relu.out  aten::cudnn_convolution_add_relu.out(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_convolution_relu.default  aten::cudnn_convolution_relu(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
aten::cudnn_convolution_relu.out  aten::cudnn_convolution_relu.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_convolution_transpose.default  aten::cudnn_convolution_transpose(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32) -> Tensor
aten::cudnn_convolution_transpose.out  aten::cudnn_convolution_transpose.out(Tensor self, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, bool allow_tf32, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_grid_sampler.default  aten::cudnn_grid_sampler(Tensor self, Tensor grid) -> Tensor output
aten::cudnn_grid_sampler.out  aten::cudnn_grid_sampler.out(Tensor self, Tensor grid, *, Tensor(a!) out) -> Tensor(a!)
aten::cudnn_grid_sampler_backward.default  aten::cudnn_grid_sampler_backward(Tensor self, Tensor grid, Tensor grad_output) -> (Tensor grad_self, Tensor grad_grid)
aten::cudnn_grid_sampler_backward.out  aten::cudnn_grid_sampler_backward.out(Tensor self, Tensor grid, Tensor grad_output, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::cudnn_is_acceptable.default  aten::cudnn_is_acceptable(Tensor self) -> bool
aten::cummax.default  aten::cummax(Tensor self, int dim) -> (Tensor values, Tensor indices)
aten::cummax.dimname  aten::cummax.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)
aten::cummax.dimname_out  aten::cummax.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::cummax.out  aten::cummax.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::cummaxmin_backward.default  aten::cummaxmin_backward(Tensor grad, Tensor input, Tensor indices, int dim) -> Tensor
aten::cummin.default  aten::cummin(Tensor self, int dim) -> (Tensor values, Tensor indices)
aten::cummin.dimname  aten::cummin.dimname(Tensor self, str dim) -> (Tensor values, Tensor indices)
aten::cummin.dimname_out  aten::cummin.dimname_out(Tensor self, str dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::cummin.out  aten::cummin.out(Tensor self, int dim, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::cumprod.default  aten::cumprod(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
aten::cumprod.dimname  aten::cumprod.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::cumprod.dimname_out  aten::cumprod.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::cumprod.out  aten::cumprod.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::cumprod_.default  aten::cumprod_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
aten::cumprod_.dimname  aten::cumprod_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!)
aten::cumprod_backward.default  aten::cumprod_backward(Tensor grad, Tensor input, int dim, Tensor output) -> Tensor
aten::cumsum.default  aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
aten::cumsum.dimname  aten::cumsum.dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::cumsum.dimname_out  aten::cumsum.dimname_out(Tensor self, str dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::cumsum.out  aten::cumsum.out(Tensor self, int dim, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::cumsum_.default  aten::cumsum_(Tensor(a!) self, int dim, *, ScalarType? dtype=None) -> Tensor(a!)
aten::cumsum_.dimname  aten::cumsum_.dimname(Tensor(a!) self, str dim, *, ScalarType? dtype=None) -> Tensor(a!)
aten::cumulative_trapezoid.dx  aten::cumulative_trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
aten::cumulative_trapezoid.x  aten::cumulative_trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
aten::data.default  aten::data(Tensor self) -> Tensor
aten::deg2rad.default  aten::deg2rad(Tensor self) -> Tensor
aten::deg2rad.out  aten::deg2rad.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::deg2rad_.default  aten::deg2rad_(Tensor(a!) self) -> Tensor(a!)
aten::dense_dim.default  aten::dense_dim(Tensor self) -> int
aten::dequantize.self  aten::dequantize.self(Tensor self) -> Tensor
aten::dequantize.self_out  aten::dequantize.self_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::dequantize.tensors  aten::dequantize.tensors(Tensor[] tensors) -> Tensor[]
aten::dequantize.tensors_out  aten::dequantize.tensors_out(Tensor[] tensors, *, Tensor(a!)[] out) -> ()
aten::det.default  aten::det(Tensor self) -> Tensor
aten::detach.default  aten::detach(Tensor(a) self) -> Tensor(a)
aten::detach_.default  aten::detach_(Tensor(a!) self) -> Tensor(a!)
aten::detach_copy.default  aten::detach_copy(Tensor self) -> Tensor
aten::detach_copy.out  aten::detach_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::diag.default  aten::diag(Tensor self, int diagonal=0) -> Tensor
aten::diag.out  aten::diag.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
aten::diag_embed.default  aten::diag_embed(Tensor self, int offset=0, int dim1=-2, int dim2=-1) -> Tensor
aten::diag_embed.out  aten::diag_embed.out(Tensor self, int offset=0, int dim1=-2, int dim2=-1, *, Tensor(a!) out) -> Tensor(a!)
aten::diagflat.default  aten::diagflat(Tensor self, int offset=0) -> Tensor
aten::diagonal.Dimname  aten::diagonal.Dimname(Tensor(a) self, *, str outdim, str dim1, str dim2, int offset=0) -> Tensor(a)
aten::diagonal.default  aten::diagonal(Tensor(a) self, int offset=0, int dim1=0, int dim2=1) -> Tensor(a)
aten::diagonal_backward.default  aten::diagonal_backward(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2) -> Tensor
aten::diagonal_backward.out  aten::diagonal_backward.out(Tensor grad_output, SymInt[] input_sizes, int offset, int dim1, int dim2, *, Tensor(a!) out) -> Tensor(a!)
aten::diagonal_copy.default  aten::diagonal_copy(Tensor self, int offset=0, int dim1=0, int dim2=1) -> Tensor
aten::diagonal_copy.out  aten::diagonal_copy.out(Tensor self, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)
aten::diagonal_scatter.default  aten::diagonal_scatter(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1) -> Tensor
aten::diagonal_scatter.out  aten::diagonal_scatter.out(Tensor self, Tensor src, int offset=0, int dim1=0, int dim2=1, *, Tensor(a!) out) -> Tensor(a!)
aten::diff.default  aten::diff(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None) -> Tensor
aten::diff.out  aten::diff.out(Tensor self, int n=1, int dim=-1, Tensor? prepend=None, Tensor? append=None, *, Tensor(a!) out) -> Tensor(a!)
aten::digamma.default  aten::digamma(Tensor self) -> Tensor
aten::digamma.out  aten::digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::digamma_.default  aten::digamma_(Tensor(a!) self) -> Tensor(a!)
aten::dist.default  aten::dist(Tensor self, Tensor other, Scalar p=2) -> Tensor
aten::dist.out  aten::dist.out(Tensor self, Tensor other, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)
aten::div.Scalar  aten::div.Scalar(Tensor self, Scalar other) -> Tensor
aten::div.Scalar_mode  aten::div.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
aten::div.Scalar_mode_out  aten::div.Scalar_mode_out(Tensor self, Scalar other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
aten::div.Scalar_out  aten::div.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::div.Tensor  aten::div.Tensor(Tensor self, Tensor other) -> Tensor
aten::div.Tensor_mode  aten::div.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
aten::div.out  aten::div.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::div.out_mode  aten::div.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
aten::div_.Scalar  aten::div_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::div_.Scalar_mode  aten::div_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
aten::div_.Tensor  aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::div_.Tensor_mode  aten::div_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
aten::divide.Scalar  aten::divide.Scalar(Tensor self, Scalar other) -> Tensor
aten::divide.Scalar_mode  aten::divide.Scalar_mode(Tensor self, Scalar other, *, str? rounding_mode) -> Tensor
aten::divide.Tensor  aten::divide.Tensor(Tensor self, Tensor other) -> Tensor
aten::divide.Tensor_mode  aten::divide.Tensor_mode(Tensor self, Tensor other, *, str? rounding_mode) -> Tensor
aten::divide.out  aten::divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::divide.out_mode  aten::divide.out_mode(Tensor self, Tensor other, *, str? rounding_mode, Tensor(a!) out) -> Tensor(a!)
aten::divide_.Scalar  aten::divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::divide_.Scalar_mode  aten::divide_.Scalar_mode(Tensor(a!) self, Scalar other, *, str? rounding_mode) -> Tensor(a!)
aten::divide_.Tensor  aten::divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::divide_.Tensor_mode  aten::divide_.Tensor_mode(Tensor(a!) self, Tensor other, *, str? rounding_mode) -> Tensor(a!)
aten::dot.default  aten::dot(Tensor self, Tensor tensor) -> Tensor
aten::dot.out  aten::dot.out(Tensor self, Tensor tensor, *, Tensor(a!) out) -> Tensor(a!)
aten::dropout.default  aten::dropout(Tensor input, float p, bool train) -> Tensor
aten::dropout_.default  aten::dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
aten::dsplit.array  aten::dsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
aten::dsplit.int  aten::dsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
aten::dstack.default  aten::dstack(Tensor[] tensors) -> Tensor
aten::dstack.out  aten::dstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::einsum.default  aten::einsum(str equation, Tensor[] tensors, *, int[]? path=None) -> Tensor
aten::elu.default  aten::elu(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
aten::elu.out  aten::elu.out(Tensor self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1, *, Tensor(a!) out) -> Tensor(a!)
aten::elu_.default  aten::elu_(Tensor(a!) self, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor(a!)
aten::elu_backward.default  aten::elu_backward(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result) -> Tensor
aten::elu_backward.grad_input  aten::elu_backward.grad_input(Tensor grad_output, Scalar alpha, Scalar scale, Scalar input_scale, bool is_result, Tensor self_or_result, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::embedding.default  aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False) -> Tensor
aten::embedding.out  aten::embedding.out(Tensor weight, Tensor indices, SymInt padding_idx=-1, bool scale_grad_by_freq=False, bool sparse=False, *, Tensor(a!) out) -> Tensor(a!)
aten::embedding_backward.default  aten::embedding_backward(Tensor grad, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, bool sparse) -> Tensor
aten::embedding_bag.default  aten::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor, Tensor)
aten::embedding_bag.padding_idx  aten::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor, Tensor)
aten::embedding_dense_backward.default  aten::embedding_dense_backward(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq) -> Tensor
aten::embedding_dense_backward.out  aten::embedding_dense_backward.out(Tensor grad_output, Tensor indices, SymInt num_weights, SymInt padding_idx, bool scale_grad_by_freq, *, Tensor(a!) out) -> Tensor(a!)
aten::embedding_renorm.default  aten::embedding_renorm(Tensor self, Tensor indices, float max_norm, float norm_type) -> Tensor
aten::embedding_renorm.out  aten::embedding_renorm.out(Tensor self, Tensor indices, float max_norm, float norm_type, *, Tensor(a!) out) -> Tensor(a!)
aten::embedding_renorm_.default  aten::embedding_renorm_(Tensor(a!) self, Tensor indices, float max_norm, float norm_type) -> Tensor(a!)
aten::embedding_sparse_backward.default  aten::embedding_sparse_backward(Tensor grad, Tensor indices, int num_weights, int padding_idx, bool scale_grad_by_freq) -> Tensor
aten::empty.memory_format  aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::empty.names  aten::empty.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::empty.names_out  aten::empty.names_out(int[] size, *, str[]? names, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::empty.out  aten::empty.out(SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::empty_like.default  aten::empty_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::empty_like.out  aten::empty_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::empty_permuted.default  aten::empty_permuted(SymInt[] size, int[] physical_layout, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::empty_permuted.out  aten::empty_permuted.out(SymInt[] size, int[] physical_layout, *, Tensor(a!) out) -> Tensor(a!)
aten::empty_quantized.default  aten::empty_quantized(int[] size, Tensor qtensor, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::empty_quantized.out  aten::empty_quantized.out(int[] size, Tensor qtensor, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::empty_strided.default  aten::empty_strided(SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::empty_strided.out  aten::empty_strided.out(SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)
aten::eq.Scalar  aten::eq.Scalar(Tensor self, Scalar other) -> Tensor
aten::eq.Scalar_out  aten::eq.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::eq.Tensor  aten::eq.Tensor(Tensor self, Tensor other) -> Tensor
aten::eq.Tensor_out  aten::eq.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::eq_.Scalar  aten::eq_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::eq_.Tensor  aten::eq_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::equal.default  aten::equal(Tensor self, Tensor other) -> bool
aten::erf.default  aten::erf(Tensor self) -> Tensor
aten::erf.out  aten::erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::erf_.default  aten::erf_(Tensor(a!) self) -> Tensor(a!)
aten::erfc.default  aten::erfc(Tensor self) -> Tensor
aten::erfc.out  aten::erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::erfc_.default  aten::erfc_(Tensor(a!) self) -> Tensor(a!)
aten::erfinv.default  aten::erfinv(Tensor self) -> Tensor
aten::erfinv.out  aten::erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::erfinv_.default  aten::erfinv_(Tensor(a!) self) -> Tensor(a!)
aten::exp.default  aten::exp(Tensor self) -> Tensor
aten::exp.out  aten::exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::exp2.default  aten::exp2(Tensor self) -> Tensor
aten::exp2.out  aten::exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::exp2_.default  aten::exp2_(Tensor(a!) self) -> Tensor(a!)
aten::exp_.default  aten::exp_(Tensor(a!) self) -> Tensor(a!)
aten::expand.default  aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False) -> Tensor(a)
aten::expand_as.default  aten::expand_as(Tensor(a) self, Tensor other) -> Tensor(a)
aten::expand_copy.default  aten::expand_copy(Tensor self, SymInt[] size, *, bool implicit=False) -> Tensor
aten::expand_copy.out  aten::expand_copy.out(Tensor self, SymInt[] size, *, bool implicit=False, Tensor(a!) out) -> Tensor(a!)
aten::expm1.default  aten::expm1(Tensor self) -> Tensor
aten::expm1.out  aten::expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::expm1_.default  aten::expm1_(Tensor(a!) self) -> Tensor(a!)
aten::exponential.default  aten::exponential(Tensor self, float lambd=1., *, Generator? generator=None) -> Tensor
aten::exponential.out  aten::exponential.out(Tensor self, float lambd=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::exponential_.default  aten::exponential_(Tensor(a!) self, float lambd=1., *, Generator? generator=None) -> Tensor(a!)
aten::eye.default  aten::eye(SymInt n, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::eye.m  aten::eye.m(SymInt n, SymInt m, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::eye.m_out  aten::eye.m_out(SymInt n, SymInt m, *, Tensor(a!) out) -> Tensor(a!)
aten::eye.out  aten::eye.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
aten::fake_quantize_per_channel_affine.default  aten::fake_quantize_per_channel_affine(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> Tensor
aten::fake_quantize_per_channel_affine_cachemask.default  aten::fake_quantize_per_channel_affine_cachemask(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
aten::fake_quantize_per_channel_affine_cachemask.out  aten::fake_quantize_per_channel_affine_cachemask.out(Tensor self, Tensor scale, Tensor zero_point, int axis, int quant_min, int quant_max, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::fake_quantize_per_channel_affine_cachemask_backward.default  aten::fake_quantize_per_channel_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
aten::fake_quantize_per_tensor_affine.default  aten::fake_quantize_per_tensor_affine(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> Tensor
aten::fake_quantize_per_tensor_affine.tensor_qparams  aten::fake_quantize_per_tensor_affine.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor
aten::fake_quantize_per_tensor_affine_cachemask.default  aten::fake_quantize_per_tensor_affine_cachemask(Tensor self, float scale, int zero_point, int quant_min, int quant_max) -> (Tensor output, Tensor mask)
aten::fake_quantize_per_tensor_affine_cachemask.out  aten::fake_quantize_per_tensor_affine_cachemask.out(Tensor self, float scale, int zero_point, int quant_min, int quant_max, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::fake_quantize_per_tensor_affine_cachemask_backward.default  aten::fake_quantize_per_tensor_affine_cachemask_backward(Tensor grad, Tensor mask) -> Tensor
aten::fbgemm_linear_fp16_weight.default  aten::fbgemm_linear_fp16_weight(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
aten::fbgemm_linear_fp16_weight_fp32_activation.default  aten::fbgemm_linear_fp16_weight_fp32_activation(Tensor input, Tensor packed_weight, Tensor bias) -> Tensor
aten::fbgemm_linear_int8_weight.default  aten::fbgemm_linear_int8_weight(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
aten::fbgemm_linear_int8_weight_fp32_activation.default  aten::fbgemm_linear_int8_weight_fp32_activation(Tensor input, Tensor weight, Tensor packed, Tensor col_offsets, Scalar weight_scale, Scalar weight_zero_point, Tensor bias) -> Tensor
aten::fbgemm_linear_quantize_weight.default  aten::fbgemm_linear_quantize_weight(Tensor input) -> (Tensor, Tensor, float, int)
aten::fbgemm_pack_gemm_matrix_fp16.default  aten::fbgemm_pack_gemm_matrix_fp16(Tensor input) -> Tensor
aten::fbgemm_pack_quantized_matrix.KN  aten::fbgemm_pack_quantized_matrix.KN(Tensor input, int K, int N) -> Tensor
aten::fbgemm_pack_quantized_matrix.default  aten::fbgemm_pack_quantized_matrix(Tensor input) -> Tensor
aten::feature_alpha_dropout.default  aten::feature_alpha_dropout(Tensor input, float p, bool train) -> Tensor
aten::feature_alpha_dropout_.default  aten::feature_alpha_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
aten::feature_dropout.default  aten::feature_dropout(Tensor input, float p, bool train) -> Tensor
aten::feature_dropout_.default  aten::feature_dropout_(Tensor(a!) self, float p, bool train) -> Tensor(a!)
aten::fft_fft.default  aten::fft_fft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_fft.out  aten::fft_fft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_fft2.default  aten::fft_fft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_fft2.out  aten::fft_fft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_fftfreq.default  aten::fft_fftfreq(int n, float d=1., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::fft_fftfreq.out  aten::fft_fftfreq.out(int n, float d=1., *, Tensor(a!) out) -> Tensor(a!)
aten::fft_fftn.default  aten::fft_fftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_fftn.out  aten::fft_fftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_fftshift.default  aten::fft_fftshift(Tensor self, int[1]? dim=None) -> Tensor
aten::fft_hfft.default  aten::fft_hfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_hfft.out  aten::fft_hfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_hfft2.default  aten::fft_hfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_hfft2.out  aten::fft_hfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_hfftn.default  aten::fft_hfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_hfftn.out  aten::fft_hfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ifft.default  aten::fft_ifft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_ifft.out  aten::fft_ifft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ifft2.default  aten::fft_ifft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_ifft2.out  aten::fft_ifft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ifftn.default  aten::fft_ifftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_ifftn.out  aten::fft_ifftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ifftshift.default  aten::fft_ifftshift(Tensor self, int[1]? dim=None) -> Tensor
aten::fft_ihfft.default  aten::fft_ihfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_ihfft.out  aten::fft_ihfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ihfft2.default  aten::fft_ihfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_ihfft2.out  aten::fft_ihfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_ihfftn.default  aten::fft_ihfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_ihfftn.out  aten::fft_ihfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_irfft.default  aten::fft_irfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_irfft.out  aten::fft_irfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_irfft2.default  aten::fft_irfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_irfft2.out  aten::fft_irfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_irfftn.default  aten::fft_irfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_irfftn.out  aten::fft_irfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_rfft.default  aten::fft_rfft(Tensor self, SymInt? n=None, int dim=-1, str? norm=None) -> Tensor
aten::fft_rfft.out  aten::fft_rfft.out(Tensor self, SymInt? n=None, int dim=-1, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_rfft2.default  aten::fft_rfft2(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None) -> Tensor
aten::fft_rfft2.out  aten::fft_rfft2.out(Tensor self, SymInt[1]? s=None, int[1] dim=[-2, -1], str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fft_rfftfreq.default  aten::fft_rfftfreq(int n, float d=1., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::fft_rfftfreq.out  aten::fft_rfftfreq.out(int n, float d=1., *, Tensor(a!) out) -> Tensor(a!)
aten::fft_rfftn.default  aten::fft_rfftn(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None) -> Tensor
aten::fft_rfftn.out  aten::fft_rfftn.out(Tensor self, SymInt[1]? s=None, int[1]? dim=None, str? norm=None, *, Tensor(a!) out) -> Tensor(a!)
aten::fill.Scalar  aten::fill.Scalar(Tensor self, Scalar value) -> Tensor
aten::fill.Scalar_out  aten::fill.Scalar_out(Tensor self, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
aten::fill.Tensor  aten::fill.Tensor(Tensor self, Tensor value) -> Tensor
aten::fill.Tensor_out  aten::fill.Tensor_out(Tensor self, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
aten::fill_.Scalar  aten::fill_.Scalar(Tensor(a!) self, Scalar value) -> Tensor(a!)
aten::fill_.Tensor  aten::fill_.Tensor(Tensor(a!) self, Tensor value) -> Tensor(a!)
aten::fill_diagonal_.default  aten::fill_diagonal_(Tensor(a!) self, Scalar fill_value, bool wrap=False) -> Tensor(a!)
aten::fix.default  aten::fix(Tensor self) -> Tensor
aten::fix.out  aten::fix.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::fix_.default  aten::fix_(Tensor(a!) self) -> Tensor(a!)
aten::flatten.DimnameList  aten::flatten.DimnameList(Tensor(a) self, str[] dims, str out_dim) -> Tensor(a)
aten::flatten.named_out_dim  aten::flatten.named_out_dim(Tensor(a) self, int start_dim, int end_dim, str out_dim) -> Tensor(a)
aten::flatten.using_ints  aten::flatten.using_ints(Tensor(a) self, int start_dim=0, int end_dim=-1) -> Tensor(a)
aten::flatten.using_names  aten::flatten.using_names(Tensor(a) self, str start_dim, str end_dim, str out_dim) -> Tensor(a)
aten::flatten_dense_tensors.default  aten::flatten_dense_tensors(Tensor[] tensors) -> Tensor
aten::flip.default  aten::flip(Tensor self, int[] dims) -> Tensor
aten::flip.out  aten::flip.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)
aten::fliplr.default  aten::fliplr(Tensor self) -> Tensor
aten::flipud.default  aten::flipud(Tensor self) -> Tensor
aten::float_power.Scalar  aten::float_power.Scalar(Scalar self, Tensor exponent) -> Tensor
aten::float_power.Scalar_out  aten::float_power.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::float_power.Tensor_Scalar  aten::float_power.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
aten::float_power.Tensor_Scalar_out  aten::float_power.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::float_power.Tensor_Tensor  aten::float_power.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
aten::float_power.Tensor_Tensor_out  aten::float_power.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::float_power_.Scalar  aten::float_power_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
aten::float_power_.Tensor  aten::float_power_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
aten::floor.default  aten::floor(Tensor self) -> Tensor
aten::floor.out  aten::floor.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::floor_.default  aten::floor_(Tensor(a!) self) -> Tensor(a!)
aten::floor_divide.Scalar  aten::floor_divide.Scalar(Tensor self, Scalar other) -> Tensor
aten::floor_divide.Scalar_out  aten::floor_divide.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::floor_divide.default  aten::floor_divide(Tensor self, Tensor other) -> Tensor
aten::floor_divide.out  aten::floor_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::floor_divide_.Scalar  aten::floor_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::floor_divide_.Tensor  aten::floor_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::fmax.default  aten::fmax(Tensor self, Tensor other) -> Tensor
aten::fmax.out  aten::fmax.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::fmin.default  aten::fmin(Tensor self, Tensor other) -> Tensor
aten::fmin.out  aten::fmin.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::fmod.Scalar  aten::fmod.Scalar(Tensor self, Scalar other) -> Tensor
aten::fmod.Scalar_out  aten::fmod.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::fmod.Tensor  aten::fmod.Tensor(Tensor self, Tensor other) -> Tensor
aten::fmod.Tensor_out  aten::fmod.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::fmod_.Scalar  aten::fmod_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::fmod_.Tensor  aten::fmod_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::frac.default  aten::frac(Tensor self) -> Tensor
aten::frac.out  aten::frac.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::frac_.default  aten::frac_(Tensor(a!) self) -> Tensor(a!)
aten::fractional_max_pool2d.default  aten::fractional_max_pool2d(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples) -> (Tensor, Tensor)
aten::fractional_max_pool2d.output  aten::fractional_max_pool2d.output(Tensor self, int[2] kernel_size, int[2] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::fractional_max_pool2d_backward.default  aten::fractional_max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices) -> Tensor
aten::fractional_max_pool2d_backward.grad_input  aten::fractional_max_pool2d_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::fractional_max_pool3d.default  aten::fractional_max_pool3d(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples) -> (Tensor, Tensor)
aten::fractional_max_pool3d.output  aten::fractional_max_pool3d.output(Tensor self, int[3] kernel_size, int[3] output_size, Tensor random_samples, *, Tensor(a!) output, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::fractional_max_pool3d_backward.default  aten::fractional_max_pool3d_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices) -> Tensor
aten::fractional_max_pool3d_backward.grad_input  aten::fractional_max_pool3d_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] output_size, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::frexp.Tensor  aten::frexp.Tensor(Tensor self) -> (Tensor mantissa, Tensor exponent)
aten::frexp.Tensor_out  aten::frexp.Tensor_out(Tensor self, *, Tensor(a!) mantissa, Tensor(b!) exponent) -> (Tensor(a!) mantissa, Tensor(b!) exponent)
aten::frobenius_norm.dim  aten::frobenius_norm.dim(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
aten::frobenius_norm.out  aten::frobenius_norm.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::from_file.default  aten::from_file(str filename, bool? shared=None, int? size=0, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::from_file.out  aten::from_file.out(str filename, bool? shared=None, int? size=0, *, Tensor(a!) out) -> Tensor(a!)
aten::full.default  aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::full.names  aten::full.names(int[] size, Scalar fill_value, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::full.names_out  aten::full.names_out(int[] size, Scalar fill_value, *, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::full.out  aten::full.out(SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
aten::full_like.default  aten::full_like(Tensor self, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::full_like.out  aten::full_like.out(Tensor self, Scalar fill_value, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::fused_moving_avg_obs_fake_quant.default  aten::fused_moving_avg_obs_fake_quant(Tensor self, Tensor observer_on, Tensor fake_quant_on, Tensor(a!) running_min, Tensor(b!) running_max, Tensor(c!) scale, Tensor(d!) zero_point, float averaging_const, int quant_min, int quant_max, int ch_axis, bool per_row_fake_quant=False, bool symmetric_quant=False) -> Tensor
aten::gather.default  aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor
aten::gather.dimname  aten::gather.dimname(Tensor self, str dim, Tensor index, *, bool sparse_grad=False) -> Tensor
aten::gather.dimname_out  aten::gather.dimname_out(Tensor self, str dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
aten::gather.out  aten::gather.out(Tensor self, int dim, Tensor index, *, bool sparse_grad=False, Tensor(a!) out) -> Tensor(a!)
aten::gather_backward.default  aten::gather_backward(Tensor grad, Tensor self, int dim, Tensor index, bool sparse_grad) -> Tensor
aten::gcd.default  aten::gcd(Tensor self, Tensor other) -> Tensor
aten::gcd.out  aten::gcd.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::gcd_.default  aten::gcd_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::ge.Scalar  aten::ge.Scalar(Tensor self, Scalar other) -> Tensor
aten::ge.Scalar_out  aten::ge.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::ge.Tensor  aten::ge.Tensor(Tensor self, Tensor other) -> Tensor
aten::ge.Tensor_out  aten::ge.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::ge_.Scalar  aten::ge_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::ge_.Tensor  aten::ge_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::gelu.default  aten::gelu(Tensor self, *, str approximate="none") -> Tensor
aten::gelu.out  aten::gelu.out(Tensor self, *, str approximate="none", Tensor(a!) out) -> Tensor(a!)
aten::gelu_.default  aten::gelu_(Tensor(a!) self, *, str approximate="none") -> Tensor(a!)
aten::gelu_backward.default  aten::gelu_backward(Tensor grad_output, Tensor self, *, str approximate="none") -> Tensor
aten::gelu_backward.grad_input  aten::gelu_backward.grad_input(Tensor grad_output, Tensor self, *, str approximate="none", Tensor(a!) grad_input) -> Tensor(a!)
aten::geometric.default  aten::geometric(Tensor self, float p, *, Generator? generator=None) -> Tensor
aten::geometric.out  aten::geometric.out(Tensor self, float p, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::geometric_.default  aten::geometric_(Tensor(a!) self, float p, *, Generator? generator=None) -> Tensor(a!)
aten::geqrf.a  aten::geqrf.a(Tensor self, *, Tensor(a!) a, Tensor(b!) tau) -> (Tensor(a!) a, Tensor(b!) tau)
aten::geqrf.default  aten::geqrf(Tensor self) -> (Tensor a, Tensor tau)
aten::ger.default  aten::ger(Tensor self, Tensor vec2) -> Tensor
aten::ger.out  aten::ger.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
aten::get_gradients.default  aten::get_gradients(int context_id) -> Dict(Tensor, Tensor)
aten::glu.default  aten::glu(Tensor self, int dim=-1) -> Tensor
aten::glu.out  aten::glu.out(Tensor self, int dim=-1, *, Tensor(a!) out) -> Tensor(a!)
aten::glu_backward.default  aten::glu_backward(Tensor grad_output, Tensor self, int dim) -> Tensor
aten::glu_backward.grad_input  aten::glu_backward.grad_input(Tensor grad_output, Tensor self, int dim, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::glu_backward_jvp.default  aten::glu_backward_jvp(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim) -> Tensor
aten::glu_backward_jvp.out  aten::glu_backward_jvp.out(Tensor grad_x, Tensor grad_glu, Tensor x, Tensor dgrad_glu, Tensor dx, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::glu_jvp.default  aten::glu_jvp(Tensor glu, Tensor x, Tensor dx, int dim) -> Tensor
aten::glu_jvp.out  aten::glu_jvp.out(Tensor glu, Tensor x, Tensor dx, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::gradient.array  aten::gradient.array(Tensor self, *, int[] dim, int edge_order=1) -> Tensor[]
aten::gradient.scalararray  aten::gradient.scalararray(Tensor self, *, Scalar spacing, int[] dim, int edge_order=1) -> Tensor[]
aten::gradient.scalarint  aten::gradient.scalarint(Tensor self, *, Scalar? spacing=None, int? dim=None, int edge_order=1) -> Tensor[]
aten::gradient.scalarrayarray  aten::gradient.scalarrayarray(Tensor self, *, Scalar[] spacing, int[] dim, int edge_order=1) -> Tensor[]
aten::gradient.scalarrayint  aten::gradient.scalarrayint(Tensor self, *, Scalar[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
aten::gradient.tensorarray  aten::gradient.tensorarray(Tensor self, *, Tensor[] spacing, int[] dim, int edge_order=1) -> Tensor[]
aten::gradient.tensorarrayint  aten::gradient.tensorarrayint(Tensor self, *, Tensor[] spacing, int? dim=None, int edge_order=1) -> Tensor[]
aten::greater.Scalar  aten::greater.Scalar(Tensor self, Scalar other) -> Tensor
aten::greater.Scalar_out  aten::greater.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::greater.Tensor  aten::greater.Tensor(Tensor self, Tensor other) -> Tensor
aten::greater.Tensor_out  aten::greater.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::greater_.Scalar  aten::greater_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::greater_.Tensor  aten::greater_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::greater_equal.Scalar  aten::greater_equal.Scalar(Tensor self, Scalar other) -> Tensor
aten::greater_equal.Scalar_out  aten::greater_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::greater_equal.Tensor  aten::greater_equal.Tensor(Tensor self, Tensor other) -> Tensor
aten::greater_equal.Tensor_out  aten::greater_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::greater_equal_.Scalar  aten::greater_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::greater_equal_.Tensor  aten::greater_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::grid_sampler.default  aten::grid_sampler(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
aten::grid_sampler_2d.default  aten::grid_sampler_2d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
aten::grid_sampler_2d.out  aten::grid_sampler_2d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
aten::grid_sampler_2d_backward.default  aten::grid_sampler_2d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)
aten::grid_sampler_2d_backward.out  aten::grid_sampler_2d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::grid_sampler_3d.default  aten::grid_sampler_3d(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners) -> Tensor
aten::grid_sampler_3d.out  aten::grid_sampler_3d.out(Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, *, Tensor(a!) out) -> Tensor(a!)
aten::grid_sampler_3d_backward.default  aten::grid_sampler_3d_backward(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask) -> (Tensor, Tensor)
aten::grid_sampler_3d_backward.out  aten::grid_sampler_3d_backward.out(Tensor grad_output, Tensor input, Tensor grid, int interpolation_mode, int padding_mode, bool align_corners, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::group_norm.default  aten::group_norm(Tensor input, int num_groups, Tensor? weight=None, Tensor? bias=None, float eps=1.0000000000000001e-05, bool cudnn_enabled=True) -> Tensor
aten::gru.data  aten::gru.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
aten::gru.input  aten::gru.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
aten::gru_cell.default  aten::gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
aten::gt.Scalar  aten::gt.Scalar(Tensor self, Scalar other) -> Tensor
aten::gt.Scalar_out  aten::gt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::gt.Tensor  aten::gt.Tensor(Tensor self, Tensor other) -> Tensor
aten::gt.Tensor_out  aten::gt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::gt_.Scalar  aten::gt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::gt_.Tensor  aten::gt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::hamming_window.default  aten::hamming_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hamming_window.out  aten::hamming_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
aten::hamming_window.periodic  aten::hamming_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hamming_window.periodic_alpha  aten::hamming_window.periodic_alpha(int window_length, bool periodic, float alpha, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hamming_window.periodic_alpha_beta  aten::hamming_window.periodic_alpha_beta(int window_length, bool periodic, float alpha, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hamming_window.periodic_alpha_beta_out  aten::hamming_window.periodic_alpha_beta_out(int window_length, bool periodic, float alpha, float beta, *, Tensor(a!) out) -> Tensor(a!)
aten::hamming_window.periodic_alpha_out  aten::hamming_window.periodic_alpha_out(int window_length, bool periodic, float alpha, *, Tensor(a!) out) -> Tensor(a!)
aten::hamming_window.periodic_out  aten::hamming_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
aten::hann_window.default  aten::hann_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hann_window.out  aten::hann_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
aten::hann_window.periodic  aten::hann_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::hann_window.periodic_out  aten::hann_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
aten::hardshrink.default  aten::hardshrink(Tensor self, Scalar lambd=0.5) -> Tensor
aten::hardshrink.out  aten::hardshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
aten::hardshrink_backward.default  aten::hardshrink_backward(Tensor grad_out, Tensor self, Scalar lambd) -> Tensor
aten::hardshrink_backward.grad_input  aten::hardshrink_backward.grad_input(Tensor grad_out, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::hardsigmoid.default  aten::hardsigmoid(Tensor self) -> Tensor
aten::hardsigmoid.out  aten::hardsigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::hardsigmoid_.default  aten::hardsigmoid_(Tensor(a!) self) -> Tensor(a!)
aten::hardsigmoid_backward.default  aten::hardsigmoid_backward(Tensor grad_output, Tensor self) -> Tensor
aten::hardsigmoid_backward.grad_input  aten::hardsigmoid_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::hardswish.default  aten::hardswish(Tensor self) -> Tensor
aten::hardswish.out  aten::hardswish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::hardswish_.default  aten::hardswish_(Tensor(a!) self) -> Tensor(a!)
aten::hardswish_backward.default  aten::hardswish_backward(Tensor grad_output, Tensor self) -> Tensor
aten::hardswish_backward.out  aten::hardswish_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::hardtanh.default  aten::hardtanh(Tensor self, Scalar min_val=-1, Scalar max_val=1) -> Tensor
aten::hardtanh.out  aten::hardtanh.out(Tensor self, Scalar min_val=-1, Scalar max_val=1, *, Tensor(a!) out) -> Tensor(a!)
aten::hardtanh_.default  aten::hardtanh_(Tensor(a!) self, Scalar min_val=-1, Scalar max_val=1) -> Tensor(a!)
aten::hardtanh_backward.default  aten::hardtanh_backward(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val) -> Tensor
aten::hardtanh_backward.grad_input  aten::hardtanh_backward.grad_input(Tensor grad_output, Tensor self, Scalar min_val, Scalar max_val, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::heaviside.default  aten::heaviside(Tensor self, Tensor values) -> Tensor
aten::heaviside.out  aten::heaviside.out(Tensor self, Tensor values, *, Tensor(a!) out) -> Tensor(a!)
aten::heaviside_.default  aten::heaviside_(Tensor(a!) self, Tensor values) -> Tensor(a!)
aten::hinge_embedding_loss.default  aten::hinge_embedding_loss(Tensor self, Tensor target, float margin=1., int reduction=1) -> Tensor
aten::histc.default  aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor
aten::histc.out  aten::histc.out(Tensor self, int bins=100, Scalar min=0, Scalar max=0, *, Tensor(a!) out) -> Tensor(a!)
aten::histogram.bin_ct  aten::histogram.bin_ct(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
aten::histogram.bin_ct_out  aten::histogram.bin_ct_out(Tensor self, int bins=100, *, float[]? range=None, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
aten::histogram.bins_tensor  aten::histogram.bins_tensor(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor bin_edges)
aten::histogram.bins_tensor_out  aten::histogram.bins_tensor_out(Tensor self, Tensor bins, *, Tensor? weight=None, bool density=False, Tensor(a!) hist, Tensor(b!) bin_edges) -> (Tensor(a!) hist, Tensor(b!) bin_edges)
aten::histogramdd.TensorList_bins  aten::histogramdd.TensorList_bins(Tensor self, Tensor[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
aten::histogramdd.default  aten::histogramdd(Tensor self, int[] bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
aten::histogramdd.int_bins  aten::histogramdd.int_bins(Tensor self, int bins, float[]? range=None, Tensor? weight=None, bool density=False) -> (Tensor hist, Tensor[] bin_edges)
aten::hsplit.array  aten::hsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
aten::hsplit.int  aten::hsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
aten::hspmm.default  aten::hspmm(Tensor mat1, Tensor mat2) -> Tensor
aten::hspmm.out  aten::hspmm.out(Tensor mat1, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
aten::hstack.default  aten::hstack(Tensor[] tensors) -> Tensor
aten::hstack.out  aten::hstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::huber_loss.default  aten::huber_loss(Tensor self, Tensor target, int reduction=1, float delta=1.) -> Tensor
aten::huber_loss.out  aten::huber_loss.out(Tensor self, Tensor target, int reduction=1, float delta=1., *, Tensor(a!) out) -> Tensor(a!)
aten::huber_loss_backward.default  aten::huber_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta) -> Tensor
aten::huber_loss_backward.out  aten::huber_loss_backward.out(Tensor grad_output, Tensor self, Tensor target, int reduction, float delta, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::hypot.default  aten::hypot(Tensor self, Tensor other) -> Tensor
aten::hypot.out  aten::hypot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::hypot_.default  aten::hypot_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::i0.default  aten::i0(Tensor self) -> Tensor
aten::i0.out  aten::i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::i0_.default  aten::i0_(Tensor(a!) self) -> Tensor(a!)
aten::igamma.default  aten::igamma(Tensor self, Tensor other) -> Tensor
aten::igamma.out  aten::igamma.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::igamma_.default  aten::igamma_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::igammac.default  aten::igammac(Tensor self, Tensor other) -> Tensor
aten::igammac.out  aten::igammac.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::igammac_.default  aten::igammac_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::im2col.default  aten::im2col(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride) -> Tensor
aten::im2col.out  aten::im2col.out(Tensor self, int[2] kernel_size, int[2] dilation, int[2] padding, int[2] stride, *, Tensor(a!) out) -> Tensor(a!)
aten::imag.default  aten::imag(Tensor(a) self) -> Tensor(a)
aten::index.Tensor  aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor
aten::index.Tensor_out  aten::index.Tensor_out(Tensor self, Tensor?[] indices, *, Tensor(a!) out) -> Tensor(a!)
aten::index_add.default  aten::index_add(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
aten::index_add.dimname  aten::index_add.dimname(Tensor self, str dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor
aten::index_add.out  aten::index_add.out(Tensor self, int dim, Tensor index, Tensor source, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::index_add_.default  aten::index_add_(Tensor(a!) self, int dim, Tensor index, Tensor source, *, Scalar alpha=1) -> Tensor(a!)
aten::index_copy.default  aten::index_copy(Tensor self, int dim, Tensor index, Tensor source) -> Tensor
aten::index_copy.dimname  aten::index_copy.dimname(Tensor self, str dim, Tensor index, Tensor source) -> Tensor
aten::index_copy.out  aten::index_copy.out(Tensor self, int dim, Tensor index, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
aten::index_copy_.default  aten::index_copy_(Tensor(a!) self, int dim, Tensor index, Tensor source) -> Tensor(a!)
aten::index_copy_.dimname  aten::index_copy_.dimname(Tensor(a!) self, str dim, Tensor index, Tensor source) -> Tensor(a!)
aten::index_fill.Dimname_Scalar  aten::index_fill.Dimname_Scalar(Tensor self, str dim, Tensor index, Scalar value) -> Tensor
aten::index_fill.Dimname_Tensor  aten::index_fill.Dimname_Tensor(Tensor self, str dim, Tensor index, Tensor value) -> Tensor
aten::index_fill.int_Scalar  aten::index_fill.int_Scalar(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
aten::index_fill.int_Scalar_out  aten::index_fill.int_Scalar_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
aten::index_fill.int_Tensor  aten::index_fill.int_Tensor(Tensor self, int dim, Tensor index, Tensor value) -> Tensor
aten::index_fill.int_Tensor_out  aten::index_fill.int_Tensor_out(Tensor self, int dim, Tensor index, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
aten::index_fill_.Dimname_Scalar  aten::index_fill_.Dimname_Scalar(Tensor(a!) self, str dim, Tensor index, Scalar value) -> Tensor(a!)
aten::index_fill_.Dimname_Tensor  aten::index_fill_.Dimname_Tensor(Tensor(a!) self, str dim, Tensor index, Tensor value) -> Tensor(a!)
aten::index_fill_.int_Scalar  aten::index_fill_.int_Scalar(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
aten::index_fill_.int_Tensor  aten::index_fill_.int_Tensor(Tensor(a!) self, int dim, Tensor index, Tensor value) -> Tensor(a!)
aten::index_put.default  aten::index_put(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor
aten::index_put.out  aten::index_put.out(Tensor self, Tensor?[] indices, Tensor values, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)
aten::index_put_.default  aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool accumulate=False) -> Tensor(a!)
aten::index_reduce.default  aten::index_reduce(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor
aten::index_reduce.out  aten::index_reduce.out(Tensor self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
aten::index_reduce_.default  aten::index_reduce_(Tensor(a!) self, int dim, Tensor index, Tensor source, str reduce, *, bool include_self=True) -> Tensor(a!)
aten::index_select.default  aten::index_select(Tensor self, int dim, Tensor index) -> Tensor
aten::index_select.dimname  aten::index_select.dimname(Tensor self, str dim, Tensor index) -> Tensor
aten::index_select.dimname_out  aten::index_select.dimname_out(Tensor self, str dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
aten::index_select.out  aten::index_select.out(Tensor self, int dim, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
aten::index_select_backward.default  aten::index_select_backward(Tensor grad, SymInt[] self_sizes, int dim, Tensor index) -> Tensor
aten::indices.default  aten::indices(Tensor(a) self) -> Tensor(a)
aten::indices_copy.default  aten::indices_copy(Tensor self) -> Tensor
aten::indices_copy.out  aten::indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::infinitely_differentiable_gelu_backward.default  aten::infinitely_differentiable_gelu_backward(Tensor grad, Tensor self) -> Tensor
aten::inner.default  aten::inner(Tensor self, Tensor other) -> Tensor
aten::inner.out  aten::inner.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::instance_norm.default  aten::instance_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool use_input_stats, float momentum, float eps, bool cudnn_enabled) -> Tensor
aten::int_repr.default  aten::int_repr(Tensor self) -> Tensor
aten::int_repr.out  aten::int_repr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::inverse.default  aten::inverse(Tensor self) -> Tensor
aten::inverse.out  aten::inverse.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::is_coalesced.default  aten::is_coalesced(Tensor self) -> bool
aten::is_complex.default  aten::is_complex(Tensor self) -> bool
aten::is_conj.default  aten::is_conj(Tensor self) -> bool
aten::is_distributed.default  aten::is_distributed(Tensor self) -> bool
aten::is_floating_point.default  aten::is_floating_point(Tensor self) -> bool
aten::is_inference.default  aten::is_inference(Tensor self) -> bool
aten::is_leaf.default  aten::is_leaf(Tensor self) -> bool
aten::is_neg.default  aten::is_neg(Tensor self) -> bool
aten::is_nonzero.default  aten::is_nonzero(Tensor self) -> bool
aten::is_pinned.default  aten::is_pinned(Tensor self, Device? device=None) -> bool
aten::is_same_size.default  aten::is_same_size(Tensor self, Tensor other) -> bool
aten::is_set_to.default  aten::is_set_to(Tensor self, Tensor tensor) -> bool
aten::is_signed.default  aten::is_signed(Tensor self) -> bool
aten::is_vulkan_available.default  aten::is_vulkan_available() -> bool
aten::isclose.default  aten::isclose(Tensor self, Tensor other, float rtol=1.0000000000000001e-05, float atol=1e-08, bool equal_nan=False) -> Tensor
aten::isfinite.default  aten::isfinite(Tensor self) -> Tensor
aten::isin.Scalar_Tensor  aten::isin.Scalar_Tensor(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
aten::isin.Scalar_Tensor_out  aten::isin.Scalar_Tensor_out(Scalar element, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
aten::isin.Tensor_Scalar  aten::isin.Tensor_Scalar(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False) -> Tensor
aten::isin.Tensor_Scalar_out  aten::isin.Tensor_Scalar_out(Tensor elements, Scalar test_element, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
aten::isin.Tensor_Tensor  aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False) -> Tensor
aten::isin.Tensor_Tensor_out  aten::isin.Tensor_Tensor_out(Tensor elements, Tensor test_elements, *, bool assume_unique=False, bool invert=False, Tensor(a!) out) -> Tensor(a!)
aten::isinf.default  aten::isinf(Tensor self) -> Tensor
aten::isinf.out  aten::isinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::isnan.default  aten::isnan(Tensor self) -> Tensor
aten::isnan.out  aten::isnan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::isneginf.default  aten::isneginf(Tensor self) -> Tensor
aten::isneginf.out  aten::isneginf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::isposinf.default  aten::isposinf(Tensor self) -> Tensor
aten::isposinf.out  aten::isposinf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::isreal.default  aten::isreal(Tensor self) -> Tensor
aten::istft.default  aten::istft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, bool normalized=False, bool? onesided=None, int? length=None, bool return_complex=False) -> Tensor
aten::item.default  aten::item(Tensor self) -> Scalar
aten::kaiser_window.beta  aten::kaiser_window.beta(int window_length, bool periodic, float beta, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::kaiser_window.beta_out  aten::kaiser_window.beta_out(int window_length, bool periodic, float beta, *, Tensor(a!) out) -> Tensor(a!)
aten::kaiser_window.default  aten::kaiser_window(int window_length, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::kaiser_window.out  aten::kaiser_window.out(int window_length, *, Tensor(a!) out) -> Tensor(a!)
aten::kaiser_window.periodic  aten::kaiser_window.periodic(int window_length, bool periodic, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::kaiser_window.periodic_out  aten::kaiser_window.periodic_out(int window_length, bool periodic, *, Tensor(a!) out) -> Tensor(a!)
aten::kl_div.default  aten::kl_div(Tensor self, Tensor target, int reduction=1, *, bool log_target=False) -> Tensor
aten::kron.default  aten::kron(Tensor self, Tensor other) -> Tensor
aten::kron.out  aten::kron.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::kthvalue.default  aten::kthvalue(Tensor self, int k, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::kthvalue.dimname  aten::kthvalue.dimname(Tensor self, int k, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::kthvalue.dimname_out  aten::kthvalue.dimname_out(Tensor self, int k, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::kthvalue.values  aten::kthvalue.values(Tensor self, int k, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::l1_loss.default  aten::l1_loss(Tensor self, Tensor target, int reduction=1) -> Tensor
aten::layer_norm.default  aten::layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1.0000000000000001e-05, bool cudnn_enable=True) -> Tensor
aten::lcm.default  aten::lcm(Tensor self, Tensor other) -> Tensor
aten::lcm.out  aten::lcm.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::lcm_.default  aten::lcm_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::ldexp.Tensor  aten::ldexp.Tensor(Tensor self, Tensor other) -> Tensor
aten::ldexp.out  aten::ldexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::ldexp_.default  aten::ldexp_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::le.Scalar  aten::le.Scalar(Tensor self, Scalar other) -> Tensor
aten::le.Scalar_out  aten::le.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::le.Tensor  aten::le.Tensor(Tensor self, Tensor other) -> Tensor
aten::le.Tensor_out  aten::le.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::le_.Scalar  aten::le_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::le_.Tensor  aten::le_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::leaky_relu.default  aten::leaky_relu(Tensor self, Scalar negative_slope=0.01) -> Tensor
aten::leaky_relu.out  aten::leaky_relu.out(Tensor self, Scalar negative_slope=0.01, *, Tensor(a!) out) -> Tensor(a!)
aten::leaky_relu_.default  aten::leaky_relu_(Tensor(a!) self, Scalar negative_slope=0.01) -> Tensor(a!)
aten::leaky_relu_backward.default  aten::leaky_relu_backward(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result) -> Tensor
aten::leaky_relu_backward.grad_input  aten::leaky_relu_backward.grad_input(Tensor grad_output, Tensor self, Scalar negative_slope, bool self_is_result, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::lerp.Scalar  aten::lerp.Scalar(Tensor self, Tensor end, Scalar weight) -> Tensor
aten::lerp.Scalar_out  aten::lerp.Scalar_out(Tensor self, Tensor end, Scalar weight, *, Tensor(a!) out) -> Tensor(a!)
aten::lerp.Tensor  aten::lerp.Tensor(Tensor self, Tensor end, Tensor weight) -> Tensor
aten::lerp.Tensor_out  aten::lerp.Tensor_out(Tensor self, Tensor end, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
aten::lerp_.Scalar  aten::lerp_.Scalar(Tensor(a!) self, Tensor end, Scalar weight) -> Tensor(a!)
aten::lerp_.Tensor  aten::lerp_.Tensor(Tensor(a!) self, Tensor end, Tensor weight) -> Tensor(a!)
aten::less.Scalar  aten::less.Scalar(Tensor self, Scalar other) -> Tensor
aten::less.Scalar_out  aten::less.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::less.Tensor  aten::less.Tensor(Tensor self, Tensor other) -> Tensor
aten::less.Tensor_out  aten::less.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::less_.Scalar  aten::less_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::less_.Tensor  aten::less_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::less_equal.Scalar  aten::less_equal.Scalar(Tensor self, Scalar other) -> Tensor
aten::less_equal.Scalar_out  aten::less_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::less_equal.Tensor  aten::less_equal.Tensor(Tensor self, Tensor other) -> Tensor
aten::less_equal.Tensor_out  aten::less_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::less_equal_.Scalar  aten::less_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::less_equal_.Tensor  aten::less_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::lgamma.default  aten::lgamma(Tensor self) -> Tensor
aten::lgamma.out  aten::lgamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::lgamma_.default  aten::lgamma_(Tensor(a!) self) -> Tensor(a!)
aten::lift.default  aten::lift(Tensor self) -> Tensor
aten::lift.out  aten::lift.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::lift_fresh.default  aten::lift_fresh(Tensor(a) self) -> Tensor(a)
aten::lift_fresh_copy.default  aten::lift_fresh_copy(Tensor self) -> Tensor
aten::lift_fresh_copy.out  aten::lift_fresh_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_cholesky.default  aten::linalg_cholesky(Tensor self, *, bool upper=False) -> Tensor
aten::linalg_cholesky.out  aten::linalg_cholesky.out(Tensor self, *, bool upper=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_cholesky_ex.L  aten::linalg_cholesky_ex.L(Tensor self, *, bool upper=False, bool check_errors=False, Tensor(a!) L, Tensor(b!) info) -> (Tensor(a!) L, Tensor(b!) info)
aten::linalg_cholesky_ex.default  aten::linalg_cholesky_ex(Tensor self, *, bool upper=False, bool check_errors=False) -> (Tensor L, Tensor info)
aten::linalg_cond.default  aten::linalg_cond(Tensor self, Scalar? p=None) -> Tensor
aten::linalg_cond.out  aten::linalg_cond.out(Tensor self, Scalar? p=None, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_cond.p_str  aten::linalg_cond.p_str(Tensor self, str p) -> Tensor
aten::linalg_cond.p_str_out  aten::linalg_cond.p_str_out(Tensor self, str p, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_cross.default  aten::linalg_cross(Tensor self, Tensor other, *, int dim=-1) -> Tensor
aten::linalg_cross.out  aten::linalg_cross.out(Tensor self, Tensor other, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)
aten::linalg_det.default  aten::linalg_det(Tensor A) -> Tensor
aten::linalg_det.out  aten::linalg_det.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_diagonal.default  aten::linalg_diagonal(Tensor(a) A, *, int offset=0, int dim1=-2, int dim2=-1) -> Tensor(a)
aten::linalg_eig.default  aten::linalg_eig(Tensor self) -> (Tensor eigenvalues, Tensor eigenvectors)
aten::linalg_eig.out  aten::linalg_eig.out(Tensor self, *, Tensor(a!) eigenvalues, Tensor(b!) eigenvectors) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
aten::linalg_eigh.default  aten::linalg_eigh(Tensor self, str UPLO="L") -> (Tensor eigenvalues, Tensor eigenvectors)
aten::linalg_eigh.eigvals  aten::linalg_eigh.eigvals(Tensor self, str UPLO="L", *, Tensor(a!) eigvals, Tensor(b!) eigvecs) -> (Tensor(a!) eigenvalues, Tensor(b!) eigenvectors)
aten::linalg_eigvals.default  aten::linalg_eigvals(Tensor self) -> Tensor
aten::linalg_eigvals.out  aten::linalg_eigvals.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_eigvalsh.default  aten::linalg_eigvalsh(Tensor self, str UPLO="L") -> Tensor
aten::linalg_eigvalsh.out  aten::linalg_eigvalsh.out(Tensor self, str UPLO="L", *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_householder_product.default  aten::linalg_householder_product(Tensor input, Tensor tau) -> Tensor
aten::linalg_householder_product.out  aten::linalg_householder_product.out(Tensor input, Tensor tau, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_inv.default  aten::linalg_inv(Tensor A) -> Tensor
aten::linalg_inv.out  aten::linalg_inv.out(Tensor A, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_inv_ex.default  aten::linalg_inv_ex(Tensor A, *, bool check_errors=False) -> (Tensor inverse, Tensor info)
aten::linalg_inv_ex.inverse  aten::linalg_inv_ex.inverse(Tensor A, *, bool check_errors=False, Tensor(a!) inverse, Tensor(b!) info) -> (Tensor(a!) inverse, Tensor(b!) info)
aten::linalg_ldl_factor.default  aten::linalg_ldl_factor(Tensor self, *, bool hermitian=False) -> (Tensor LD, Tensor pivots)
aten::linalg_ldl_factor.out  aten::linalg_ldl_factor.out(Tensor self, *, bool hermitian=False, Tensor(a!) LD, Tensor(b!) pivots) -> (Tensor(a!) LD, Tensor(b!) pivots)
aten::linalg_ldl_factor_ex.default  aten::linalg_ldl_factor_ex(Tensor self, *, bool hermitian=False, bool check_errors=False) -> (Tensor LD, Tensor pivots, Tensor info)
aten::linalg_ldl_factor_ex.out  aten::linalg_ldl_factor_ex.out(Tensor self, *, bool hermitian=False, bool check_errors=False, Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LD, Tensor(b!) pivots, Tensor(c!) info)
aten::linalg_ldl_solve.default  aten::linalg_ldl_solve(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False) -> Tensor
aten::linalg_ldl_solve.out  aten::linalg_ldl_solve.out(Tensor LD, Tensor pivots, Tensor B, *, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_lstsq.default  aten::linalg_lstsq(Tensor self, Tensor b, float? rcond=None, *, str? driver=None) -> (Tensor solution, Tensor residuals, Tensor rank, Tensor singular_values)
aten::linalg_lstsq.out  aten::linalg_lstsq.out(Tensor self, Tensor b, float? rcond=None, *, str? driver=None, Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values) -> (Tensor(a!) solution, Tensor(b!) residuals, Tensor(c!) rank, Tensor(d!) singular_values)
aten::linalg_lu.default  aten::linalg_lu(Tensor A, *, bool pivot=True) -> (Tensor P, Tensor L, Tensor U)
aten::linalg_lu.out  aten::linalg_lu.out(Tensor A, *, bool pivot=True, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
aten::linalg_lu_factor.default  aten::linalg_lu_factor(Tensor A, *, bool pivot=True) -> (Tensor LU, Tensor pivots)
aten::linalg_lu_factor.out  aten::linalg_lu_factor.out(Tensor A, *, bool pivot=True, Tensor(a!) LU, Tensor(b!) pivots) -> (Tensor(a!) LU, Tensor(b!) pivots)
aten::linalg_lu_factor_ex.default  aten::linalg_lu_factor_ex(Tensor A, *, bool pivot=True, bool check_errors=False) -> (Tensor LU, Tensor pivots, Tensor info)
aten::linalg_lu_factor_ex.out  aten::linalg_lu_factor_ex.out(Tensor A, *, bool pivot=True, bool check_errors=False, Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info) -> (Tensor(a!) LU, Tensor(b!) pivots, Tensor(c!) info)
aten::linalg_lu_solve.default  aten::linalg_lu_solve(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False) -> Tensor
aten::linalg_lu_solve.out  aten::linalg_lu_solve.out(Tensor LU, Tensor pivots, Tensor B, *, bool left=True, bool adjoint=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matmul.default  aten::linalg_matmul(Tensor self, Tensor other) -> Tensor
aten::linalg_matmul.out  aten::linalg_matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_exp.default  aten::linalg_matrix_exp(Tensor self) -> Tensor
aten::linalg_matrix_exp.out  aten::linalg_matrix_exp.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_norm.default  aten::linalg_matrix_norm(Tensor self, Scalar ord, int[] dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::linalg_matrix_norm.out  aten::linalg_matrix_norm.out(Tensor self, Scalar ord, int[] dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_norm.str_ord  aten::linalg_matrix_norm.str_ord(Tensor self, str ord="fro", int[] dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::linalg_matrix_norm.str_ord_out  aten::linalg_matrix_norm.str_ord_out(Tensor self, str ord="fro", int[] dim=[-2, -1], bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_power.default  aten::linalg_matrix_power(Tensor self, int n) -> Tensor
aten::linalg_matrix_power.out  aten::linalg_matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_rank.atol_rtol_float  aten::linalg_matrix_rank.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
aten::linalg_matrix_rank.atol_rtol_float_out  aten::linalg_matrix_rank.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_rank.atol_rtol_tensor  aten::linalg_matrix_rank.atol_rtol_tensor(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
aten::linalg_matrix_rank.atol_rtol_tensor_out  aten::linalg_matrix_rank.atol_rtol_tensor_out(Tensor input, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_rank.default  aten::linalg_matrix_rank(Tensor self, float tol, bool hermitian=False) -> Tensor
aten::linalg_matrix_rank.out  aten::linalg_matrix_rank.out(Tensor self, float tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_rank.out_tol_tensor  aten::linalg_matrix_rank.out_tol_tensor(Tensor input, Tensor tol, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_matrix_rank.tol_tensor  aten::linalg_matrix_rank.tol_tensor(Tensor input, Tensor tol, bool hermitian=False) -> Tensor
aten::linalg_multi_dot.default  aten::linalg_multi_dot(Tensor[] tensors) -> Tensor
aten::linalg_multi_dot.out  aten::linalg_multi_dot.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_norm.default  aten::linalg_norm(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::linalg_norm.ord_str  aten::linalg_norm.ord_str(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::linalg_norm.ord_str_out  aten::linalg_norm.ord_str_out(Tensor self, str ord, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::linalg_norm.out  aten::linalg_norm.out(Tensor self, Scalar? ord=None, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::linalg_pinv.atol_rtol_float  aten::linalg_pinv.atol_rtol_float(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False) -> Tensor
aten::linalg_pinv.atol_rtol_float_out  aten::linalg_pinv.atol_rtol_float_out(Tensor self, *, float? atol=None, float? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_pinv.atol_rtol_tensor  aten::linalg_pinv.atol_rtol_tensor(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False) -> Tensor
aten::linalg_pinv.atol_rtol_tensor_out  aten::linalg_pinv.atol_rtol_tensor_out(Tensor self, *, Tensor? atol=None, Tensor? rtol=None, bool hermitian=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_pinv.default  aten::linalg_pinv(Tensor self, float rcond, bool hermitian=False) -> Tensor
aten::linalg_pinv.out  aten::linalg_pinv.out(Tensor self, float rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_pinv.out_rcond_tensor  aten::linalg_pinv.out_rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_pinv.rcond_tensor  aten::linalg_pinv.rcond_tensor(Tensor self, Tensor rcond, bool hermitian=False) -> Tensor
aten::linalg_qr.default  aten::linalg_qr(Tensor A, str mode="reduced") -> (Tensor Q, Tensor R)
aten::linalg_qr.out  aten::linalg_qr.out(Tensor A, str mode="reduced", *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
aten::linalg_slogdet.default  aten::linalg_slogdet(Tensor A) -> (Tensor sign, Tensor logabsdet)
aten::linalg_slogdet.out  aten::linalg_slogdet.out(Tensor A, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
aten::linalg_solve.default  aten::linalg_solve(Tensor A, Tensor B, *, bool left=True) -> Tensor
aten::linalg_solve.out  aten::linalg_solve.out(Tensor A, Tensor B, *, bool left=True, Tensor(a!) out) -> Tensor(a!)
aten::linalg_solve_ex.default  aten::linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, bool check_errors=False) -> (Tensor result, Tensor info)
aten::linalg_solve_ex.out  aten::linalg_solve_ex.out(Tensor A, Tensor B, *, bool left=True, bool check_errors=False, Tensor(a!) result, Tensor(b!) info) -> (Tensor(a!) result, Tensor(b!) info)
aten::linalg_solve_triangular.default  aten::linalg_solve_triangular(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False) -> Tensor
aten::linalg_solve_triangular.out  aten::linalg_solve_triangular.out(Tensor self, Tensor B, *, bool upper, bool left=True, bool unitriangular=False, Tensor(a!) out) -> Tensor(a!)
aten::linalg_svd.U  aten::linalg_svd.U(Tensor A, bool full_matrices=True, *, str? driver=None, Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) Vh)
aten::linalg_svd.default  aten::linalg_svd(Tensor A, bool full_matrices=True, *, str? driver=None) -> (Tensor U, Tensor S, Tensor Vh)
aten::linalg_svdvals.default  aten::linalg_svdvals(Tensor A, *, str? driver=None) -> Tensor
aten::linalg_svdvals.out  aten::linalg_svdvals.out(Tensor A, *, str? driver=None, Tensor(a!) out) -> Tensor(a!)
aten::linalg_tensorinv.default  aten::linalg_tensorinv(Tensor self, int ind=2) -> Tensor
aten::linalg_tensorinv.out  aten::linalg_tensorinv.out(Tensor self, int ind=2, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_tensorsolve.default  aten::linalg_tensorsolve(Tensor self, Tensor other, int[]? dims=None) -> Tensor
aten::linalg_tensorsolve.out  aten::linalg_tensorsolve.out(Tensor self, Tensor other, int[]? dims=None, *, Tensor(a!) out) -> Tensor(a!)
aten::linalg_vander.default  aten::linalg_vander(Tensor x, *, SymInt? N=None) -> Tensor
aten::linalg_vecdot.default  aten::linalg_vecdot(Tensor x, Tensor y, *, int dim=-1) -> Tensor
aten::linalg_vecdot.out  aten::linalg_vecdot.out(Tensor x, Tensor y, *, int dim=-1, Tensor(a!) out) -> Tensor(a!)
aten::linalg_vector_norm.default  aten::linalg_vector_norm(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::linalg_vector_norm.out  aten::linalg_vector_norm.out(Tensor self, Scalar ord=2, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::linear.default  aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor
aten::linear.out  aten::linear.out(Tensor input, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
aten::linear_backward.default  aten::linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::linear_backward.out  aten::linear_backward.out(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::linspace.Scalar_Tensor  aten::linspace.Scalar_Tensor(Scalar start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::linspace.Scalar_Tensor_out  aten::linspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
aten::linspace.Tensor_Scalar  aten::linspace.Tensor_Scalar(Tensor start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::linspace.Tensor_Scalar_out  aten::linspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
aten::linspace.Tensor_Tensor  aten::linspace.Tensor_Tensor(Tensor start, Tensor end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::linspace.Tensor_Tensor_out  aten::linspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, *, Tensor(a!) out) -> Tensor(a!)
aten::linspace.default  aten::linspace(Scalar start, Scalar end, int steps, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::linspace.out  aten::linspace.out(Scalar start, Scalar end, int steps, *, Tensor(a!) out) -> Tensor(a!)
aten::log.default  aten::log(Tensor self) -> Tensor
aten::log.out  aten::log.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::log10.default  aten::log10(Tensor self) -> Tensor
aten::log10.out  aten::log10.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::log10_.default  aten::log10_(Tensor(a!) self) -> Tensor(a!)
aten::log1p.default  aten::log1p(Tensor self) -> Tensor
aten::log1p.out  aten::log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::log1p_.default  aten::log1p_(Tensor(a!) self) -> Tensor(a!)
aten::log2.default  aten::log2(Tensor self) -> Tensor
aten::log2.out  aten::log2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::log2_.default  aten::log2_(Tensor(a!) self) -> Tensor(a!)
aten::log_.default  aten::log_(Tensor(a!) self) -> Tensor(a!)
aten::log_normal.default  aten::log_normal(Tensor self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor
aten::log_normal.out  aten::log_normal.out(Tensor self, float mean=1., float std=2., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::log_normal_.default  aten::log_normal_(Tensor(a!) self, float mean=1., float std=2., *, Generator? generator=None) -> Tensor(a!)
aten::log_sigmoid.default  aten::log_sigmoid(Tensor self) -> Tensor
aten::log_sigmoid.out  aten::log_sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::log_sigmoid_backward.default  aten::log_sigmoid_backward(Tensor grad_output, Tensor self, Tensor buffer) -> Tensor
aten::log_sigmoid_backward.grad_input  aten::log_sigmoid_backward.grad_input(Tensor grad_output, Tensor self, Tensor buffer, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::log_sigmoid_forward.default  aten::log_sigmoid_forward(Tensor self) -> (Tensor output, Tensor buffer)
aten::log_sigmoid_forward.output  aten::log_sigmoid_forward.output(Tensor self, *, Tensor(a!) output, Tensor(b!) buffer) -> (Tensor(a!), Tensor(b!))
aten::log_softmax.Dimname  aten::log_softmax.Dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::log_softmax.int  aten::log_softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
aten::log_softmax.int_out  aten::log_softmax.int_out(Tensor self, int dim, ScalarType? dtype=None, *, Tensor(a!) out) -> Tensor(a!)
aten::logaddexp.default  aten::logaddexp(Tensor self, Tensor other) -> Tensor
aten::logaddexp.out  aten::logaddexp.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::logaddexp2.default  aten::logaddexp2(Tensor self, Tensor other) -> Tensor
aten::logaddexp2.out  aten::logaddexp2.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::logcumsumexp.default  aten::logcumsumexp(Tensor self, int dim) -> Tensor
aten::logcumsumexp.dimname  aten::logcumsumexp.dimname(Tensor self, str dim) -> Tensor
aten::logcumsumexp.dimname_out  aten::logcumsumexp.dimname_out(Tensor self, str dim, *, Tensor(a!) out) -> Tensor(a!)
aten::logcumsumexp.out  aten::logcumsumexp.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::logdet.default  aten::logdet(Tensor self) -> Tensor
aten::logical_and.default  aten::logical_and(Tensor self, Tensor other) -> Tensor
aten::logical_and.out  aten::logical_and.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::logical_and_.default  aten::logical_and_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::logical_not.default  aten::logical_not(Tensor self) -> Tensor
aten::logical_not.out  aten::logical_not.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::logical_not_.default  aten::logical_not_(Tensor(a!) self) -> Tensor(a!)
aten::logical_or.default  aten::logical_or(Tensor self, Tensor other) -> Tensor
aten::logical_or.out  aten::logical_or.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::logical_or_.default  aten::logical_or_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::logical_xor.default  aten::logical_xor(Tensor self, Tensor other) -> Tensor
aten::logical_xor.out  aten::logical_xor.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::logical_xor_.default  aten::logical_xor_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::logit.default  aten::logit(Tensor self, float? eps=None) -> Tensor
aten::logit.out  aten::logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
aten::logit_.default  aten::logit_(Tensor(a!) self, float? eps=None) -> Tensor(a!)
aten::logit_backward.default  aten::logit_backward(Tensor grad_output, Tensor self, float? eps=None) -> Tensor
aten::logit_backward.grad_input  aten::logit_backward.grad_input(Tensor grad_output, Tensor self, float? eps=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::logspace.Scalar_Tensor  aten::logspace.Scalar_Tensor(Scalar start, Tensor end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::logspace.Scalar_Tensor_out  aten::logspace.Scalar_Tensor_out(Scalar start, Tensor end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)
aten::logspace.Tensor_Scalar  aten::logspace.Tensor_Scalar(Tensor start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::logspace.Tensor_Scalar_out  aten::logspace.Tensor_Scalar_out(Tensor start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)
aten::logspace.Tensor_Tensor  aten::logspace.Tensor_Tensor(Tensor start, Tensor end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::logspace.Tensor_Tensor_out  aten::logspace.Tensor_Tensor_out(Tensor start, Tensor end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)
aten::logspace.default  aten::logspace(Scalar start, Scalar end, int steps, float base=10., *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::logspace.out  aten::logspace.out(Scalar start, Scalar end, int steps, float base=10., *, Tensor(a!) out) -> Tensor(a!)
aten::logsumexp.default  aten::logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
aten::logsumexp.names  aten::logsumexp.names(Tensor self, str[1] dim, bool keepdim=False) -> Tensor
aten::logsumexp.names_out  aten::logsumexp.names_out(Tensor self, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::logsumexp.out  aten::logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::lstm.data  aten::lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor, Tensor)
aten::lstm.input  aten::lstm.input(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor, Tensor)
aten::lstm_cell.default  aten::lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> (Tensor, Tensor)
aten::lstm_mps_backward.default  aten::lstm_mps_backward(Tensor? grad_y, Tensor? grad_hy, Tensor? grad_cy, Tensor z_state, Tensor cell_state_fwd, Tensor input, Tensor layersOutputs, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor[], Tensor[])
aten::lstm_mps_backward.out  aten::lstm_mps_backward.out(Tensor? grad_y, Tensor? grad_hy, Tensor? grad_cy, Tensor z_state, Tensor cell_state_fwd, Tensor input, Tensor layersOutputs, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, Tensor(a!) out0, Tensor(b!)[] out1, Tensor(c!)[] out2) -> ()
aten::lt.Scalar  aten::lt.Scalar(Tensor self, Scalar other) -> Tensor
aten::lt.Scalar_out  aten::lt.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::lt.Tensor  aten::lt.Tensor(Tensor self, Tensor other) -> Tensor
aten::lt.Tensor_out  aten::lt.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::lt_.Scalar  aten::lt_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::lt_.Tensor  aten::lt_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::lu_solve.default  aten::lu_solve(Tensor self, Tensor LU_data, Tensor LU_pivots) -> Tensor
aten::lu_solve.out  aten::lu_solve.out(Tensor self, Tensor LU_data, Tensor LU_pivots, *, Tensor(a!) out) -> Tensor(a!)
aten::lu_unpack.default  aten::lu_unpack(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True) -> (Tensor P, Tensor L, Tensor U)
aten::lu_unpack.out  aten::lu_unpack.out(Tensor LU_data, Tensor LU_pivots, bool unpack_data=True, bool unpack_pivots=True, *, Tensor(a!) P, Tensor(b!) L, Tensor(c!) U) -> (Tensor(a!) P, Tensor(b!) L, Tensor(c!) U)
aten::mH.default  aten::mH(Tensor(a) self) -> Tensor(a)
aten::mT.default  aten::mT(Tensor(a) self) -> Tensor(a)
aten::margin_ranking_loss.default  aten::margin_ranking_loss(Tensor input1, Tensor input2, Tensor target, float margin=0., int reduction=1) -> Tensor
aten::masked_fill.Scalar  aten::masked_fill.Scalar(Tensor self, Tensor mask, Scalar value) -> Tensor
aten::masked_fill.Scalar_out  aten::masked_fill.Scalar_out(Tensor self, Tensor mask, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
aten::masked_fill.Tensor  aten::masked_fill.Tensor(Tensor self, Tensor mask, Tensor value) -> Tensor
aten::masked_fill.Tensor_out  aten::masked_fill.Tensor_out(Tensor self, Tensor mask, Tensor value, *, Tensor(a!) out) -> Tensor(a!)
aten::masked_fill_.Scalar  aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)
aten::masked_fill_.Tensor  aten::masked_fill_.Tensor(Tensor(a!) self, Tensor mask, Tensor value) -> Tensor(a!)
aten::masked_scatter.default  aten::masked_scatter(Tensor self, Tensor mask, Tensor source) -> Tensor
aten::masked_scatter.out  aten::masked_scatter.out(Tensor self, Tensor mask, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
aten::masked_scatter_.default  aten::masked_scatter_(Tensor(a!) self, Tensor mask, Tensor source) -> Tensor(a!)
aten::masked_scatter_backward.default  aten::masked_scatter_backward(Tensor grad_output, Tensor mask, SymInt[] sizes) -> Tensor
aten::masked_select.default  aten::masked_select(Tensor self, Tensor mask) -> Tensor
aten::masked_select.out  aten::masked_select.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
aten::masked_select_backward.default  aten::masked_select_backward(Tensor grad, Tensor input, Tensor mask) -> Tensor
aten::matmul.default  aten::matmul(Tensor self, Tensor other) -> Tensor
aten::matmul.out  aten::matmul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::matmul_backward.default  aten::matmul_backward(Tensor grad, Tensor self, Tensor other, bool[2] mask) -> (Tensor, Tensor)
aten::matmul_backward.out  aten::matmul_backward.out(Tensor grad, Tensor self, Tensor other, bool[2] mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::matrix_H.default  aten::matrix_H(Tensor(a) self) -> Tensor(a)
aten::matrix_exp.default  aten::matrix_exp(Tensor self) -> Tensor
aten::matrix_exp_backward.default  aten::matrix_exp_backward(Tensor self, Tensor grad) -> Tensor
aten::matrix_power.default  aten::matrix_power(Tensor self, int n) -> Tensor
aten::matrix_power.out  aten::matrix_power.out(Tensor self, int n, *, Tensor(a!) out) -> Tensor(a!)
aten::max.default  aten::max(Tensor self) -> Tensor
aten::max.dim  aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::max.dim_max  aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
aten::max.names_dim  aten::max.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::max.names_dim_max  aten::max.names_dim_max(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)
aten::max.other  aten::max.other(Tensor self, Tensor other) -> Tensor
aten::max.out  aten::max.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::max.unary_out  aten::max.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::max_pool1d.default  aten::max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> Tensor
aten::max_pool1d_with_indices.default  aten::max_pool1d_with_indices(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> (Tensor, Tensor)
aten::max_pool2d.default  aten::max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
aten::max_pool2d_backward.default  aten::max_pool2d_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
aten::max_pool2d_backward.out  aten::max_pool2d_backward.out(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::max_pool2d_with_indices.default  aten::max_pool2d_with_indices(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
aten::max_pool2d_with_indices.out  aten::max_pool2d_with_indices.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::max_pool2d_with_indices_backward.default  aten::max_pool2d_with_indices_backward(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices) -> Tensor
aten::max_pool2d_with_indices_backward.grad_input  aten::max_pool2d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[2] kernel_size, int[2] stride, int[2] padding, int[2] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::max_pool3d.default  aten::max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
aten::max_pool3d_with_indices.default  aten::max_pool3d_with_indices(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> (Tensor, Tensor)
aten::max_pool3d_with_indices.out  aten::max_pool3d_with_indices.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out, Tensor(b!) indices) -> (Tensor(a!), Tensor(b!))
aten::max_pool3d_with_indices_backward.default  aten::max_pool3d_with_indices_backward(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices) -> Tensor
aten::max_pool3d_with_indices_backward.grad_input  aten::max_pool3d_with_indices_backward.grad_input(Tensor grad_output, Tensor self, int[3] kernel_size, int[3] stride, int[3] padding, int[3] dilation, bool ceil_mode, Tensor indices, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::max_unpool2d.default  aten::max_unpool2d(Tensor self, Tensor indices, SymInt[2] output_size) -> Tensor
aten::max_unpool2d.out  aten::max_unpool2d.out(Tensor self, Tensor indices, SymInt[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::max_unpool3d.default  aten::max_unpool3d(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding) -> Tensor
aten::max_unpool3d.out  aten::max_unpool3d.out(Tensor self, Tensor indices, SymInt[3] output_size, int[3] stride, int[3] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::maximum.default  aten::maximum(Tensor self, Tensor other) -> Tensor
aten::maximum.out  aten::maximum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::mean.default  aten::mean(Tensor self, *, ScalarType? dtype=None) -> Tensor
aten::mean.dim  aten::mean.dim(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::mean.names_dim  aten::mean.names_dim(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::mean.names_out  aten::mean.names_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::mean.out  aten::mean.out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::median.default  aten::median(Tensor self) -> Tensor
aten::median.dim  aten::median.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::median.dim_values  aten::median.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::median.names_dim  aten::median.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::median.names_dim_values  aten::median.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::median.out  aten::median.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::meshgrid.default  aten::meshgrid(Tensor[] tensors) -> Tensor[]
aten::meshgrid.indexing  aten::meshgrid.indexing(Tensor[] tensors, *, str indexing) -> Tensor[]
aten::min.default  aten::min(Tensor self) -> Tensor
aten::min.dim  aten::min.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::min.dim_min  aten::min.dim_min(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::min.names_dim  aten::min.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::min.names_dim_min  aten::min.names_dim_min(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) min, Tensor(b!) min_indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::min.other  aten::min.other(Tensor self, Tensor other) -> Tensor
aten::min.out  aten::min.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::min.unary_out  aten::min.unary_out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::minimum.default  aten::minimum(Tensor self, Tensor other) -> Tensor
aten::minimum.out  aten::minimum.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::miopen_batch_norm.default  aten::miopen_batch_norm(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon) -> (Tensor, Tensor, Tensor)
aten::miopen_batch_norm.out  aten::miopen_batch_norm.out(Tensor input, Tensor weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float exponential_average_factor, float epsilon, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::miopen_batch_norm_backward.default  aten::miopen_batch_norm_backward(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon) -> (Tensor, Tensor, Tensor)
aten::miopen_batch_norm_backward.out  aten::miopen_batch_norm_backward.out(Tensor input, Tensor grad_output, Tensor weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_var, float epsilon, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::miopen_convolution.default  aten::miopen_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
aten::miopen_convolution.out  aten::miopen_convolution.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, *, Tensor(a!) out) -> Tensor(a!)
aten::miopen_convolution_add_relu.default  aten::miopen_convolution_add_relu(Tensor self, Tensor weight, Tensor z, Scalar? alpha, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
aten::miopen_convolution_relu.default  aten::miopen_convolution_relu(Tensor self, Tensor weight, Tensor? bias, SymInt[] stride, SymInt[] padding, SymInt[] dilation, SymInt groups) -> Tensor
aten::miopen_convolution_transpose.default  aten::miopen_convolution_transpose(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
aten::miopen_convolution_transpose.out  aten::miopen_convolution_transpose.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, *, Tensor(a!) out) -> Tensor(a!)
aten::miopen_depthwise_convolution.default  aten::miopen_depthwise_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic) -> Tensor
aten::miopen_depthwise_convolution.out  aten::miopen_depthwise_convolution.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool benchmark, bool deterministic, *, Tensor(a!) out) -> Tensor(a!)
aten::miopen_rnn.default  aten::miopen_rnn(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state) -> (Tensor, Tensor, Tensor, Tensor, Tensor)
aten::miopen_rnn.out  aten::miopen_rnn.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor hx, Tensor? cx, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!))
aten::miopen_rnn_backward.default  aten::miopen_rnn_backward(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask) -> (Tensor, Tensor, Tensor, Tensor[])
aten::miopen_rnn_backward.out  aten::miopen_rnn_backward.out(Tensor input, Tensor[] weight, int weight_stride0, Tensor weight_buf, Tensor hx, Tensor? cx, Tensor output, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, int mode, int hidden_size, int num_layers, bool batch_first, float dropout, bool train, bool bidirectional, int[] batch_sizes, Tensor? dropout_state, Tensor reserve, bool[4] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!)[] out3) -> ()
aten::mish.default  aten::mish(Tensor self) -> Tensor
aten::mish.out  aten::mish.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::mish_.default  aten::mish_(Tensor(a!) self) -> Tensor(a!)
aten::mish_backward.default  aten::mish_backward(Tensor grad_output, Tensor self) -> Tensor
aten::mkldnn_adaptive_avg_pool2d.default  aten::mkldnn_adaptive_avg_pool2d(Tensor self, int[2] output_size) -> Tensor
aten::mkldnn_adaptive_avg_pool2d.out  aten::mkldnn_adaptive_avg_pool2d.out(Tensor self, int[2] output_size, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_adaptive_avg_pool2d_backward.default  aten::mkldnn_adaptive_avg_pool2d_backward(Tensor grad_output, Tensor self) -> Tensor
aten::mkldnn_adaptive_avg_pool2d_backward.out  aten::mkldnn_adaptive_avg_pool2d_backward.out(Tensor grad_output, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_convolution.default  aten::mkldnn_convolution(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups) -> Tensor
aten::mkldnn_convolution.out  aten::mkldnn_convolution.out(Tensor self, Tensor weight, Tensor? bias, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_linear.default  aten::mkldnn_linear(Tensor self, Tensor weight, Tensor? bias=None) -> Tensor
aten::mkldnn_linear.out  aten::mkldnn_linear.out(Tensor self, Tensor weight, Tensor? bias=None, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_linear_backward.default  aten::mkldnn_linear_backward(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::mkldnn_linear_backward.out  aten::mkldnn_linear_backward.out(Tensor self, Tensor grad_output, Tensor weight, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::mkldnn_linear_backward_input.default  aten::mkldnn_linear_backward_input(int[] input_size, Tensor grad_output, Tensor weight) -> Tensor
aten::mkldnn_linear_backward_input.out  aten::mkldnn_linear_backward_input.out(int[] input_size, Tensor grad_output, Tensor weight, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_linear_backward_weights.default  aten::mkldnn_linear_backward_weights(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined) -> (Tensor, Tensor)
aten::mkldnn_linear_backward_weights.out  aten::mkldnn_linear_backward_weights.out(Tensor grad_output, Tensor input, Tensor weight, bool bias_defined, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::mkldnn_max_pool2d.default  aten::mkldnn_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
aten::mkldnn_max_pool2d.out  aten::mkldnn_max_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_max_pool2d_backward.default  aten::mkldnn_max_pool2d_backward(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
aten::mkldnn_max_pool2d_backward.out  aten::mkldnn_max_pool2d_backward.out(Tensor grad_output, Tensor output, Tensor input, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_max_pool3d.default  aten::mkldnn_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
aten::mkldnn_max_pool3d.out  aten::mkldnn_max_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_max_pool3d_backward.default  aten::mkldnn_max_pool3d_backward(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
aten::mkldnn_max_pool3d_backward.out  aten::mkldnn_max_pool3d_backward.out(Tensor grad_output, Tensor output, Tensor input, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_reorder_conv2d_weight.default  aten::mkldnn_reorder_conv2d_weight(Tensor self, SymInt[2] padding=[0, 0], SymInt[2] stride=[1, 1], SymInt[2] dilation=[1, 1], SymInt groups=1, SymInt[]? input_size=None) -> Tensor
aten::mkldnn_reorder_conv2d_weight.out  aten::mkldnn_reorder_conv2d_weight.out(Tensor self, SymInt[2] padding=[0, 0], SymInt[2] stride=[1, 1], SymInt[2] dilation=[1, 1], SymInt groups=1, SymInt[]? input_size=None, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_reorder_conv3d_weight.default  aten::mkldnn_reorder_conv3d_weight(Tensor self, SymInt[3] padding=[0, 0, 0], SymInt[3] stride=[1, 1, 1], SymInt[3] dilation=[1, 1, 1], SymInt groups=1, SymInt[]? input_size=None) -> Tensor
aten::mkldnn_reorder_conv3d_weight.out  aten::mkldnn_reorder_conv3d_weight.out(Tensor self, SymInt[3] padding=[0, 0, 0], SymInt[3] stride=[1, 1, 1], SymInt[3] dilation=[1, 1, 1], SymInt groups=1, SymInt[]? input_size=None, *, Tensor(a!) out) -> Tensor(a!)
aten::mkldnn_rnn_layer.default  aten::mkldnn_rnn_layer(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train) -> (Tensor, Tensor, Tensor, Tensor)
aten::mkldnn_rnn_layer.out  aten::mkldnn_rnn_layer.out(Tensor input, Tensor weight0, Tensor weight1, Tensor weight2, Tensor weight3, Tensor hx_, Tensor cx_, bool reverse, int[] batch_sizes, int mode, int hidden_size, int num_layers, bool has_biases, bool bidirectional, bool batch_first, bool train, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!))
aten::mkldnn_rnn_layer_backward.default  aten::mkldnn_rnn_layer_backward(Tensor input, Tensor weight1, Tensor weight2, Tensor weight3, Tensor weight4, Tensor hx_, Tensor cx_tmp, Tensor output, Tensor hy_, Tensor cy_, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, bool reverse, int mode, int hidden_size, int num_layers, bool has_biases, bool train, bool bidirectional, int[] batch_sizes, bool batch_first, Tensor workspace) -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)
aten::mkldnn_rnn_layer_backward.out  aten::mkldnn_rnn_layer_backward.out(Tensor input, Tensor weight1, Tensor weight2, Tensor weight3, Tensor weight4, Tensor hx_, Tensor cx_tmp, Tensor output, Tensor hy_, Tensor cy_, Tensor? grad_output, Tensor? grad_hy, Tensor? grad_cy, bool reverse, int mode, int hidden_size, int num_layers, bool has_biases, bool train, bool bidirectional, int[] batch_sizes, bool batch_first, Tensor workspace, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2, Tensor(d!) out3, Tensor(e!) out4, Tensor(f!) out5, Tensor(g!) out6) -> (Tensor(a!), Tensor(b!), Tensor(c!), Tensor(d!), Tensor(e!), Tensor(f!), Tensor(g!))
aten::mm.default  aten::mm(Tensor self, Tensor mat2) -> Tensor
aten::mm.out  aten::mm.out(Tensor self, Tensor mat2, *, Tensor(a!) out) -> Tensor(a!)
aten::mode.default  aten::mode(Tensor self, int dim=-1, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::mode.dimname  aten::mode.dimname(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::mode.dimname_out  aten::mode.dimname_out(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::mode.values  aten::mode.values(Tensor self, int dim=-1, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::moveaxis.int  aten::moveaxis.int(Tensor(a) self, int source, int destination) -> Tensor(a)
aten::moveaxis.intlist  aten::moveaxis.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
aten::movedim.int  aten::movedim.int(Tensor(a) self, int source, int destination) -> Tensor(a)
aten::movedim.intlist  aten::movedim.intlist(Tensor(a) self, int[] source, int[] destination) -> Tensor(a)
aten::mps_convolution_backward.default  aten::mps_convolution_backward(Tensor self, Tensor grad_output, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::mps_convolution_backward.out  aten::mps_convolution_backward.out(Tensor self, Tensor grad_output, Tensor weight, SymInt[] padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::mps_convolution_transpose_backward.default  aten::mps_convolution_transpose_backward(Tensor self, Tensor grad_output, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool[2] output_mask) -> (Tensor, Tensor)
aten::mps_convolution_transpose_backward.out  aten::mps_convolution_transpose_backward.out(Tensor self, Tensor grad_output, Tensor weight, SymInt[] padding, SymInt[] output_padding, SymInt[] stride, SymInt[] dilation, SymInt groups, bool[2] output_mask, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::mse_loss.default  aten::mse_loss(Tensor self, Tensor target, int reduction=1) -> Tensor
aten::mse_loss.out  aten::mse_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::mse_loss_backward.default  aten::mse_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
aten::mse_loss_backward.grad_input  aten::mse_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::msort.default  aten::msort(Tensor self) -> Tensor
aten::msort.out  aten::msort.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::mul.Scalar  aten::mul.Scalar(Tensor self, Scalar other) -> Tensor
aten::mul.Scalar_out  aten::mul.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::mul.Tensor  aten::mul.Tensor(Tensor self, Tensor other) -> Tensor
aten::mul.out  aten::mul.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::mul_.Scalar  aten::mul_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::mul_.Tensor  aten::mul_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::multi_margin_loss.default  aten::multi_margin_loss(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=1) -> Tensor
aten::multi_margin_loss.out  aten::multi_margin_loss.out(Tensor self, Tensor target, Scalar p=1, Scalar margin=1, Tensor? weight=None, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::multi_margin_loss_backward.default  aten::multi_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=1) -> Tensor
aten::multi_margin_loss_backward.grad_input  aten::multi_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Scalar p, Scalar margin, Tensor? weight=None, int reduction=1, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::multilabel_margin_loss.default  aten::multilabel_margin_loss(Tensor self, Tensor target, int reduction=1) -> Tensor
aten::multilabel_margin_loss.out  aten::multilabel_margin_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::multilabel_margin_loss_backward.default  aten::multilabel_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target) -> Tensor
aten::multilabel_margin_loss_backward.grad_input  aten::multilabel_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, Tensor is_target, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::multilabel_margin_loss_forward.default  aten::multilabel_margin_loss_forward(Tensor self, Tensor target, int reduction) -> (Tensor output, Tensor is_target)
aten::multilabel_margin_loss_forward.output  aten::multilabel_margin_loss_forward.output(Tensor self, Tensor target, int reduction, *, Tensor(a!) output, Tensor(b!) is_target) -> (Tensor(a!), Tensor(b!))
aten::multinomial.default  aten::multinomial(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None) -> Tensor
aten::multinomial.out  aten::multinomial.out(Tensor self, int num_samples, bool replacement=False, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::multiply.Scalar  aten::multiply.Scalar(Tensor self, Scalar other) -> Tensor
aten::multiply.Tensor  aten::multiply.Tensor(Tensor self, Tensor other) -> Tensor
aten::multiply.out  aten::multiply.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::multiply_.Scalar  aten::multiply_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::multiply_.Tensor  aten::multiply_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::mv.default  aten::mv(Tensor self, Tensor vec) -> Tensor
aten::mv.out  aten::mv.out(Tensor self, Tensor vec, *, Tensor(a!) out) -> Tensor(a!)
aten::mvlgamma.default  aten::mvlgamma(Tensor self, int p) -> Tensor
aten::mvlgamma.out  aten::mvlgamma.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
aten::mvlgamma_.default  aten::mvlgamma_(Tensor(a!) self, int p) -> Tensor(a!)
aten::nan_to_num.default  aten::nan_to_num(Tensor self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor
aten::nan_to_num.out  aten::nan_to_num.out(Tensor self, float? nan=None, float? posinf=None, float? neginf=None, *, Tensor(a!) out) -> Tensor(a!)
aten::nan_to_num_.default  aten::nan_to_num_(Tensor(a!) self, float? nan=None, float? posinf=None, float? neginf=None) -> Tensor(a!)
aten::nanmean.default  aten::nanmean(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::nanmean.out  aten::nanmean.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::nanmedian.default  aten::nanmedian(Tensor self) -> Tensor
aten::nanmedian.dim  aten::nanmedian.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::nanmedian.dim_values  aten::nanmedian.dim_values(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::nanmedian.names_dim  aten::nanmedian.names_dim(Tensor self, str dim, bool keepdim=False) -> (Tensor values, Tensor indices)
aten::nanmedian.names_dim_values  aten::nanmedian.names_dim_values(Tensor self, str dim, bool keepdim=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::nanmedian.out  aten::nanmedian.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::nanquantile.default  aten::nanquantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> Tensor
aten::nanquantile.out  aten::nanquantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear", Tensor(a!) out) -> Tensor(a!)
aten::nanquantile.scalar  aten::nanquantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> Tensor
aten::nanquantile.scalar_out  aten::nanquantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear", Tensor(a!) out) -> Tensor(a!)
aten::nansum.default  aten::nansum(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::nansum.out  aten::nansum.out(Tensor self, int[1]? dim=None, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::narrow.Tensor  aten::narrow.Tensor(Tensor(a) self, int dim, Tensor start, SymInt length) -> Tensor(a)
aten::narrow.default  aten::narrow(Tensor(a) self, int dim, SymInt start, SymInt length) -> Tensor(a)
aten::narrow_copy.default  aten::narrow_copy(Tensor self, int dim, SymInt start, SymInt length) -> Tensor
aten::narrow_copy.out  aten::narrow_copy.out(Tensor self, int dim, SymInt start, SymInt length, *, Tensor(a!) out) -> Tensor(a!)
aten::native_batch_norm.default  aten::native_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps) -> (Tensor, Tensor, Tensor)
aten::native_batch_norm.out  aten::native_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor? running_mean, Tensor? running_var, bool training, float momentum, float eps, *, Tensor(a!) out, Tensor(b!) save_mean, Tensor(c!) save_invstd) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_batch_norm_backward.default  aten::native_batch_norm_backward(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::native_batch_norm_backward.out  aten::native_batch_norm_backward.out(Tensor grad_out, Tensor input, Tensor? weight, Tensor? running_mean, Tensor? running_var, Tensor? save_mean, Tensor? save_invstd, bool train, float eps, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_channel_shuffle.default  aten::native_channel_shuffle(Tensor self, SymInt groups) -> Tensor
aten::native_dropout.default  aten::native_dropout(Tensor input, float p, bool? train) -> (Tensor, Tensor)
aten::native_dropout.out  aten::native_dropout.out(Tensor input, float p, bool? train, *, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::native_dropout_backward.default  aten::native_dropout_backward(Tensor grad_output, Tensor mask, float scale) -> Tensor
aten::native_dropout_backward.out  aten::native_dropout_backward.out(Tensor grad_output, Tensor mask, float scale, *, Tensor(a!) out) -> Tensor(a!)
aten::native_group_norm.default  aten::native_group_norm(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps) -> (Tensor, Tensor, Tensor)
aten::native_group_norm.out  aten::native_group_norm.out(Tensor input, Tensor? weight, Tensor? bias, SymInt N, SymInt C, SymInt HxW, int group, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_group_norm_backward.default  aten::native_group_norm_backward(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::native_group_norm_backward.out  aten::native_group_norm_backward.out(Tensor grad_out, Tensor input, Tensor mean, Tensor rstd, Tensor? weight, SymInt N, SymInt C, SymInt HxW, int group, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_layer_norm.default  aten::native_layer_norm(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)
aten::native_layer_norm.out  aten::native_layer_norm.out(Tensor input, SymInt[] normalized_shape, Tensor? weight, Tensor? bias, float eps, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_layer_norm_backward.default  aten::native_layer_norm_backward(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask) -> (Tensor, Tensor, Tensor)
aten::native_layer_norm_backward.out  aten::native_layer_norm_backward.out(Tensor grad_out, Tensor input, SymInt[] normalized_shape, Tensor mean, Tensor rstd, Tensor? weight, Tensor? bias, bool[3] output_mask, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::native_norm.ScalarOpt_dim_dtype  aten::native_norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype) -> Tensor
aten::native_norm.ScalarOpt_dim_dtype_out  aten::native_norm.ScalarOpt_dim_dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, ScalarType? dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::native_norm.default  aten::native_norm(Tensor self, Scalar p=2) -> Tensor
aten::native_norm.out  aten::native_norm.out(Tensor self, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)
aten::ne.Scalar  aten::ne.Scalar(Tensor self, Scalar other) -> Tensor
aten::ne.Scalar_out  aten::ne.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::ne.Tensor  aten::ne.Tensor(Tensor self, Tensor other) -> Tensor
aten::ne.Tensor_out  aten::ne.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::ne_.Scalar  aten::ne_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::ne_.Tensor  aten::ne_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::neg.default  aten::neg(Tensor self) -> Tensor
aten::neg.out  aten::neg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::neg_.default  aten::neg_(Tensor(a!) self) -> Tensor(a!)
aten::negative.default  aten::negative(Tensor self) -> Tensor
aten::negative.out  aten::negative.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::negative_.default  aten::negative_(Tensor(a!) self) -> Tensor(a!)
aten::nested_to_padded_tensor.default  aten::nested_to_padded_tensor(Tensor self, float padding, int[]? output_size=None) -> Tensor
aten::new_empty.default  aten::new_empty(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::new_empty.out  aten::new_empty.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::new_empty_strided.default  aten::new_empty_strided(Tensor self, SymInt[] size, SymInt[] stride, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::new_empty_strided.out  aten::new_empty_strided.out(Tensor self, SymInt[] size, SymInt[] stride, *, Tensor(a!) out) -> Tensor(a!)
aten::new_full.default  aten::new_full(Tensor self, SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::new_full.out  aten::new_full.out(Tensor self, SymInt[] size, Scalar fill_value, *, Tensor(a!) out) -> Tensor(a!)
aten::new_ones.default  aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::new_ones.out  aten::new_ones.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::new_zeros.default  aten::new_zeros(Tensor self, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::new_zeros.out  aten::new_zeros.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::nextafter.default  aten::nextafter(Tensor self, Tensor other) -> Tensor
aten::nextafter.out  aten::nextafter.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::nextafter_.default  aten::nextafter_(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::nll_loss.default  aten::nll_loss(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100) -> Tensor
aten::nll_loss.out  aten::nll_loss.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
aten::nll_loss2d.default  aten::nll_loss2d(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100) -> Tensor
aten::nll_loss2d.out  aten::nll_loss2d.out(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100, *, Tensor(a!) out) -> Tensor(a!)
aten::nll_loss2d_backward.default  aten::nll_loss2d_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor
aten::nll_loss2d_backward.grad_input  aten::nll_loss2d_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::nll_loss2d_forward.default  aten::nll_loss2d_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)
aten::nll_loss2d_forward.output  aten::nll_loss2d_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
aten::nll_loss_backward.default  aten::nll_loss_backward(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight) -> Tensor
aten::nll_loss_backward.grad_input  aten::nll_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, Tensor total_weight, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::nll_loss_forward.default  aten::nll_loss_forward(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index) -> (Tensor output, Tensor total_weight)
aten::nll_loss_forward.output  aten::nll_loss_forward.output(Tensor self, Tensor target, Tensor? weight, int reduction, SymInt ignore_index, *, Tensor(a!) output, Tensor(b!) total_weight) -> (Tensor(a!), Tensor(b!))
aten::nll_loss_nd.default  aten::nll_loss_nd(Tensor self, Tensor target, Tensor? weight=None, int reduction=1, SymInt ignore_index=-100) -> Tensor
aten::nonzero.default  aten::nonzero(Tensor self) -> Tensor
aten::nonzero.out  aten::nonzero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::nonzero_numpy.default  aten::nonzero_numpy(Tensor self) -> Tensor[]
aten::nonzero_static.default  aten::nonzero_static(Tensor self, *, int size, int fill_value=-1) -> Tensor
aten::nonzero_static.out  aten::nonzero_static.out(Tensor self, *, int size, int fill_value=-1, Tensor(a!) out) -> Tensor(a!)
aten::norm.Scalar  aten::norm.Scalar(Tensor self, Scalar p=2) -> Tensor
aten::norm.ScalarOpt_dim  aten::norm.ScalarOpt_dim(Tensor self, Scalar? p, int[1] dim, bool keepdim=False) -> Tensor
aten::norm.ScalarOpt_dim_dtype  aten::norm.ScalarOpt_dim_dtype(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
aten::norm.ScalarOpt_dtype  aten::norm.ScalarOpt_dtype(Tensor self, Scalar? p, *, ScalarType dtype) -> Tensor
aten::norm.ScalarOpt_dtype_out  aten::norm.ScalarOpt_dtype_out(Tensor self, Scalar? p, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
aten::norm.Scalar_out  aten::norm.Scalar_out(Tensor self, Scalar p=2, *, Tensor(a!) out) -> Tensor(a!)
aten::norm.dtype_out  aten::norm.dtype_out(Tensor self, Scalar? p, int[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
aten::norm.names_ScalarOpt_dim  aten::norm.names_ScalarOpt_dim(Tensor self, Scalar? p, str[1] dim, bool keepdim=False) -> Tensor
aten::norm.names_ScalarOpt_dim_dtype  aten::norm.names_ScalarOpt_dim_dtype(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype) -> Tensor
aten::norm.names_dtype_out  aten::norm.names_dtype_out(Tensor self, Scalar? p, str[1] dim, bool keepdim, *, ScalarType dtype, Tensor(a!) out) -> Tensor(a!)
aten::norm.names_out  aten::norm.names_out(Tensor self, Scalar? p, str[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::norm.out  aten::norm.out(Tensor self, Scalar? p, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::norm_except_dim.default  aten::norm_except_dim(Tensor v, int pow=2, int dim=0) -> Tensor
aten::normal.Tensor_Tensor  aten::normal.Tensor_Tensor(Tensor mean, Tensor std, *, Generator? generator=None) -> Tensor
aten::normal.Tensor_Tensor_out  aten::normal.Tensor_Tensor_out(Tensor mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::normal.Tensor_float  aten::normal.Tensor_float(Tensor mean, float std=1., *, Generator? generator=None) -> Tensor
aten::normal.Tensor_float_out  aten::normal.Tensor_float_out(Tensor mean, float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::normal.float_Tensor  aten::normal.float_Tensor(float mean, Tensor std, *, Generator? generator=None) -> Tensor
aten::normal.float_Tensor_out  aten::normal.float_Tensor_out(float mean, Tensor std, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::normal.float_float  aten::normal.float_float(float mean, float std, SymInt[] size, *, Generator? generator=None, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::normal.float_float_out  aten::normal.float_float_out(float mean, float std, SymInt[] size, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::normal.out  aten::normal.out(Tensor self, float mean=0., float std=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::normal_.default  aten::normal_(Tensor(a!) self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor(a!)
aten::normal_functional.default  aten::normal_functional(Tensor self, float mean=0., float std=1., *, Generator? generator=None) -> Tensor
aten::not_equal.Scalar  aten::not_equal.Scalar(Tensor self, Scalar other) -> Tensor
aten::not_equal.Scalar_out  aten::not_equal.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::not_equal.Tensor  aten::not_equal.Tensor(Tensor self, Tensor other) -> Tensor
aten::not_equal.Tensor_out  aten::not_equal.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::not_equal_.Scalar  aten::not_equal_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::not_equal_.Tensor  aten::not_equal_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::nuclear_norm.default  aten::nuclear_norm(Tensor self, bool keepdim=False) -> Tensor
aten::nuclear_norm.dim  aten::nuclear_norm.dim(Tensor self, int[2] dim, bool keepdim=False) -> Tensor
aten::nuclear_norm.dim_out  aten::nuclear_norm.dim_out(Tensor self, int[2] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::nuclear_norm.out  aten::nuclear_norm.out(Tensor self, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::numpy_T.default  aten::numpy_T(Tensor(a) self) -> Tensor(a)
aten::one_hot.default  aten::one_hot(Tensor self, int num_classes=-1) -> Tensor
aten::ones.default  aten::ones(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::ones.names  aten::ones.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::ones.names_out  aten::ones.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::ones.out  aten::ones.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::ones_like.default  aten::ones_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::ones_like.out  aten::ones_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::orgqr.default  aten::orgqr(Tensor self, Tensor input2) -> Tensor
aten::orgqr.out  aten::orgqr.out(Tensor self, Tensor input2, *, Tensor(a!) out) -> Tensor(a!)
aten::ormqr.default  aten::ormqr(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False) -> Tensor
aten::ormqr.out  aten::ormqr.out(Tensor self, Tensor input2, Tensor input3, bool left=True, bool transpose=False, *, Tensor(a!) out) -> Tensor(a!)
aten::outer.default  aten::outer(Tensor self, Tensor vec2) -> Tensor
aten::outer.out  aten::outer.out(Tensor self, Tensor vec2, *, Tensor(a!) out) -> Tensor(a!)
aten::output_nr.default  aten::output_nr(Tensor self) -> int
aten::pad.default  aten::pad(Tensor self, SymInt[] pad, str mode="constant", float? value=None) -> Tensor
aten::pad_sequence.default  aten::pad_sequence(Tensor[] sequences, bool batch_first=False, float padding_value=0.) -> Tensor
aten::pairwise_distance.default  aten::pairwise_distance(Tensor x1, Tensor x2, float p=2., float eps=9.9999999999999995e-07, bool keepdim=False) -> Tensor
aten::pdist.default  aten::pdist(Tensor self, float p=2.) -> Tensor
aten::permute.default  aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)
aten::permute_copy.default  aten::permute_copy(Tensor self, int[] dims) -> Tensor
aten::permute_copy.out  aten::permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)
aten::pin_memory.default  aten::pin_memory(Tensor(a) self, Device? device=None) -> Tensor(a)
aten::pinverse.default  aten::pinverse(Tensor self, float rcond=1.0000000000000001e-15) -> Tensor
aten::pixel_shuffle.default  aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
aten::pixel_shuffle.out  aten::pixel_shuffle.out(Tensor self, int upscale_factor, *, Tensor(a!) out) -> Tensor(a!)
aten::pixel_unshuffle.default  aten::pixel_unshuffle(Tensor self, int downscale_factor) -> Tensor
aten::pixel_unshuffle.out  aten::pixel_unshuffle.out(Tensor self, int downscale_factor, *, Tensor(a!) out) -> Tensor(a!)
aten::poisson.default  aten::poisson(Tensor self, Generator? generator=None) -> Tensor
aten::poisson.out  aten::poisson.out(Tensor self, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
aten::poisson_nll_loss.default  aten::poisson_nll_loss(Tensor input, Tensor target, bool log_input, bool full, float eps, int reduction) -> Tensor
aten::polar.default  aten::polar(Tensor abs, Tensor angle) -> Tensor
aten::polar.out  aten::polar.out(Tensor abs, Tensor angle, *, Tensor(a!) out) -> Tensor(a!)
aten::polygamma.default  aten::polygamma(int n, Tensor self) -> Tensor
aten::polygamma.out  aten::polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::polygamma_.default  aten::polygamma_(Tensor(a!) self, int n) -> Tensor(a!)
aten::positive.default  aten::positive(Tensor(a) self) -> Tensor(a)
aten::pow.Scalar  aten::pow.Scalar(Scalar self, Tensor exponent) -> Tensor
aten::pow.Scalar_out  aten::pow.Scalar_out(Scalar self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::pow.Tensor_Scalar  aten::pow.Tensor_Scalar(Tensor self, Scalar exponent) -> Tensor
aten::pow.Tensor_Scalar_out  aten::pow.Tensor_Scalar_out(Tensor self, Scalar exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::pow.Tensor_Tensor  aten::pow.Tensor_Tensor(Tensor self, Tensor exponent) -> Tensor
aten::pow.Tensor_Tensor_out  aten::pow.Tensor_Tensor_out(Tensor self, Tensor exponent, *, Tensor(a!) out) -> Tensor(a!)
aten::pow_.Scalar  aten::pow_.Scalar(Tensor(a!) self, Scalar exponent) -> Tensor(a!)
aten::pow_.Tensor  aten::pow_.Tensor(Tensor(a!) self, Tensor exponent) -> Tensor(a!)
aten::prelu.default  aten::prelu(Tensor self, Tensor weight) -> Tensor
aten::prod.Dimname_out  aten::prod.Dimname_out(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::prod.default  aten::prod(Tensor self, *, ScalarType? dtype=None) -> Tensor
aten::prod.dim_Dimname  aten::prod.dim_Dimname(Tensor self, str dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::prod.dim_int  aten::prod.dim_int(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::prod.int_out  aten::prod.int_out(Tensor self, int dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::prod.out  aten::prod.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::promote_types.default  aten::promote_types(ScalarType type1, ScalarType type2) -> ScalarType
aten::put.default  aten::put(Tensor self, Tensor index, Tensor source, bool accumulate=False) -> Tensor
aten::put.out  aten::put.out(Tensor self, Tensor index, Tensor source, bool accumulate=False, *, Tensor(a!) out) -> Tensor(a!)
aten::put_.default  aten::put_(Tensor(a!) self, Tensor index, Tensor source, bool accumulate=False) -> Tensor(a!)
aten::q_per_channel_axis.default  aten::q_per_channel_axis(Tensor self) -> int
aten::q_per_channel_scales.default  aten::q_per_channel_scales(Tensor self) -> Tensor
aten::q_per_channel_scales.out  aten::q_per_channel_scales.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::q_per_channel_zero_points.default  aten::q_per_channel_zero_points(Tensor self) -> Tensor
aten::q_per_channel_zero_points.out  aten::q_per_channel_zero_points.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::q_scale.default  aten::q_scale(Tensor self) -> float
aten::q_zero_point.default  aten::q_zero_point(Tensor self) -> int
aten::qr.Q  aten::qr.Q(Tensor self, bool some=True, *, Tensor(a!) Q, Tensor(b!) R) -> (Tensor(a!) Q, Tensor(b!) R)
aten::qr.default  aten::qr(Tensor self, bool some=True) -> (Tensor Q, Tensor R)
aten::qscheme.default  aten::qscheme(Tensor self) -> QScheme
aten::quantile.default  aten::quantile(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> Tensor
aten::quantile.out  aten::quantile.out(Tensor self, Tensor q, int? dim=None, bool keepdim=False, *, str interpolation="linear", Tensor(a!) out) -> Tensor(a!)
aten::quantile.scalar  aten::quantile.scalar(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear") -> Tensor
aten::quantile.scalar_out  aten::quantile.scalar_out(Tensor self, float q, int? dim=None, bool keepdim=False, *, str interpolation="linear", Tensor(a!) out) -> Tensor(a!)
aten::quantize_per_channel.default  aten::quantize_per_channel(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype) -> Tensor
aten::quantize_per_channel.out  aten::quantize_per_channel.out(Tensor self, Tensor scales, Tensor zero_points, int axis, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::quantize_per_tensor.default  aten::quantize_per_tensor(Tensor self, float scale, int zero_point, ScalarType dtype) -> Tensor
aten::quantize_per_tensor.out  aten::quantize_per_tensor.out(Tensor self, float scale, int zero_point, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::quantize_per_tensor.tensor_qparams  aten::quantize_per_tensor.tensor_qparams(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype) -> Tensor
aten::quantize_per_tensor.tensor_qparams_out  aten::quantize_per_tensor.tensor_qparams_out(Tensor self, Tensor scale, Tensor zero_point, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::quantize_per_tensor.tensors  aten::quantize_per_tensor.tensors(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype) -> Tensor[]
aten::quantize_per_tensor.tensors_out  aten::quantize_per_tensor.tensors_out(Tensor[] tensors, Tensor scales, Tensor zero_points, ScalarType dtype, *, Tensor(a!)[] out) -> ()
aten::quantize_per_tensor_dynamic.default  aten::quantize_per_tensor_dynamic(Tensor self, ScalarType dtype, bool reduce_range) -> Tensor
aten::quantize_per_tensor_dynamic.out  aten::quantize_per_tensor_dynamic.out(Tensor self, ScalarType dtype, bool reduce_range, *, Tensor(a!) out) -> Tensor(a!)
aten::quantized_batch_norm.default  aten::quantized_batch_norm(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
aten::quantized_batch_norm.out  aten::quantized_batch_norm.out(Tensor input, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point, *, Tensor(a!) out) -> Tensor(a!)
aten::quantized_gru.data  aten::quantized_gru.data(Tensor data, Tensor batch_sizes, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
aten::quantized_gru.data_legacy  aten::quantized_gru.data_legacy(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
aten::quantized_gru.input  aten::quantized_gru.input(Tensor input, Tensor hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
aten::quantized_gru.input_legacy  aten::quantized_gru.input_legacy(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
aten::quantized_gru_cell.default  aten::quantized_gru_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
aten::quantized_lstm.data  aten::quantized_lstm.data(Tensor data, Tensor batch_sizes, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
aten::quantized_lstm.data_legacy  aten::quantized_lstm.data_legacy(Tensor data, Tensor batch_sizes, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
aten::quantized_lstm.input  aten::quantized_lstm.input(Tensor input, Tensor[] hx, __torch__.torch.classes.rnn.CellParamsBase[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
aten::quantized_lstm.input_legacy  aten::quantized_lstm.input_legacy(Tensor input, Tensor[] hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first, *, ScalarType? dtype=None, bool use_dynamic=False) -> (Tensor, Tensor, Tensor)
aten::quantized_lstm_cell.default  aten::quantized_lstm_cell(Tensor input, Tensor[] hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> (Tensor, Tensor)
aten::quantized_max_pool1d.default  aten::quantized_max_pool1d(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False) -> Tensor
aten::quantized_max_pool1d.out  aten::quantized_max_pool1d.out(Tensor self, int[1] kernel_size, int[1] stride=[], int[1] padding=[0], int[1] dilation=[1], bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::quantized_max_pool2d.default  aten::quantized_max_pool2d(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False) -> Tensor
aten::quantized_max_pool2d.out  aten::quantized_max_pool2d.out(Tensor self, int[2] kernel_size, int[2] stride=[], int[2] padding=0, int[2] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::quantized_max_pool3d.default  aten::quantized_max_pool3d(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False) -> Tensor
aten::quantized_max_pool3d.out  aten::quantized_max_pool3d.out(Tensor self, int[3] kernel_size, int[3] stride=[], int[3] padding=0, int[3] dilation=1, bool ceil_mode=False, *, Tensor(a!) out) -> Tensor(a!)
aten::quantized_rnn_relu_cell.default  aten::quantized_rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
aten::quantized_rnn_tanh_cell.default  aten::quantized_rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh, Tensor packed_ih, Tensor packed_hh, Tensor col_offsets_ih, Tensor col_offsets_hh, Scalar scale_ih, Scalar scale_hh, Scalar zero_point_ih, Scalar zero_point_hh) -> Tensor
aten::rad2deg.default  aten::rad2deg(Tensor self) -> Tensor
aten::rad2deg.out  aten::rad2deg.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::rad2deg_.default  aten::rad2deg_(Tensor(a!) self) -> Tensor(a!)
aten::rand.default  aten::rand(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::rand.generator  aten::rand.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::rand.generator_out  aten::rand.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
aten::rand.generator_with_names  aten::rand.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::rand.generator_with_names_out  aten::rand.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::rand.names  aten::rand.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::rand.names_out  aten::rand.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::rand.out  aten::rand.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::rand_like.default  aten::rand_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::rand_like.out  aten::rand_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::randint.default  aten::randint(SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randint.generator  aten::randint.generator(SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randint.generator_out  aten::randint.generator_out(SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
aten::randint.low  aten::randint.low(SymInt low, SymInt high, SymInt[] size, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randint.low_generator  aten::randint.low_generator(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randint.low_generator_out  aten::randint.low_generator_out(SymInt low, SymInt high, SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
aten::randint.low_out  aten::randint.low_out(SymInt low, SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::randint.out  aten::randint.out(SymInt high, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::randint_like.default  aten::randint_like(Tensor self, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::randint_like.low_dtype  aten::randint_like.low_dtype(Tensor self, SymInt low, SymInt high, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::randint_like.low_dtype_out  aten::randint_like.low_dtype_out(Tensor self, SymInt low, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::randint_like.out  aten::randint_like.out(Tensor self, SymInt high, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::randn.default  aten::randn(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randn.generator  aten::randn.generator(SymInt[] size, *, Generator? generator, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randn.generator_out  aten::randn.generator_out(SymInt[] size, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
aten::randn.generator_with_names  aten::randn.generator_with_names(SymInt[] size, *, Generator? generator, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randn.generator_with_names_out  aten::randn.generator_with_names_out(SymInt[] size, *, Generator? generator, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::randn.names  aten::randn.names(SymInt[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randn.names_out  aten::randn.names_out(SymInt[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::randn.out  aten::randn.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::randn_like.default  aten::randn_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::randn_like.out  aten::randn_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::random.default  aten::random(Tensor self, *, Generator? generator=None) -> Tensor
aten::random.from  aten::random.from(Tensor self, int from, int? to, *, Generator? generator=None) -> Tensor
aten::random.from_out  aten::random.from_out(Tensor self, int from, int? to, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::random.out  aten::random.out(Tensor self, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::random.to  aten::random.to(Tensor self, int to, *, Generator? generator=None) -> Tensor
aten::random.to_out  aten::random.to_out(Tensor self, int to, *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::random_.default  aten::random_(Tensor(a!) self, *, Generator? generator=None) -> Tensor(a!)
aten::random_.from  aten::random_.from(Tensor(a!) self, int from, int? to, *, Generator? generator=None) -> Tensor(a!)
aten::random_.to  aten::random_.to(Tensor(a!) self, int to, *, Generator? generator=None) -> Tensor(a!)
aten::randperm.default  aten::randperm(SymInt n, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randperm.generator  aten::randperm.generator(SymInt n, *, Generator? generator, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::randperm.generator_out  aten::randperm.generator_out(SymInt n, *, Generator? generator, Tensor(a!) out) -> Tensor(a!)
aten::randperm.out  aten::randperm.out(SymInt n, *, Tensor(a!) out) -> Tensor(a!)
aten::range.default  aten::range(Scalar start, Scalar end, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::range.out  aten::range.out(Scalar start, Scalar end, Scalar step=1, *, Tensor(a!) out) -> Tensor(a!)
aten::range.out_  aten::range.out_(Scalar start, Scalar end, *, Tensor(a!) out) -> Tensor(a!)
aten::range.step  aten::range.step(Scalar start, Scalar end, Scalar step=1, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::ravel.default  aten::ravel(Tensor(a) self) -> Tensor(a)
aten::real.default  aten::real(Tensor(a) self) -> Tensor(a)
aten::reciprocal.default  aten::reciprocal(Tensor self) -> Tensor
aten::reciprocal.out  aten::reciprocal.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::reciprocal_.default  aten::reciprocal_(Tensor(a!) self) -> Tensor(a!)
aten::record_stream.default  aten::record_stream(Tensor(a!) self, Stream s) -> ()
aten::refine_names.default  aten::refine_names(Tensor(a) self, str[] names) -> Tensor(a)
aten::reflection_pad1d.default  aten::reflection_pad1d(Tensor self, SymInt[2] padding) -> Tensor
aten::reflection_pad1d.out  aten::reflection_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::reflection_pad1d_backward.default  aten::reflection_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor
aten::reflection_pad1d_backward.grad_input  aten::reflection_pad1d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::reflection_pad2d.default  aten::reflection_pad2d(Tensor self, SymInt[4] padding) -> Tensor
aten::reflection_pad2d.out  aten::reflection_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::reflection_pad2d_backward.default  aten::reflection_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor
aten::reflection_pad2d_backward.grad_input  aten::reflection_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::reflection_pad3d.default  aten::reflection_pad3d(Tensor self, SymInt[6] padding) -> Tensor
aten::reflection_pad3d.out  aten::reflection_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::reflection_pad3d_backward.default  aten::reflection_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor
aten::reflection_pad3d_backward.grad_input  aten::reflection_pad3d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::relu.default  aten::relu(Tensor self) -> Tensor
aten::relu.out  aten::relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::relu6.default  aten::relu6(Tensor self) -> Tensor
aten::relu6_.default  aten::relu6_(Tensor(a!) self) -> Tensor(a!)
aten::relu_.default  aten::relu_(Tensor(a!) self) -> Tensor(a!)
aten::remainder.Scalar  aten::remainder.Scalar(Tensor self, Scalar other) -> Tensor
aten::remainder.Scalar_Tensor  aten::remainder.Scalar_Tensor(Scalar self, Tensor other) -> Tensor
aten::remainder.Scalar_Tensor_out  aten::remainder.Scalar_Tensor_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::remainder.Scalar_out  aten::remainder.Scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::remainder.Tensor  aten::remainder.Tensor(Tensor self, Tensor other) -> Tensor
aten::remainder.Tensor_out  aten::remainder.Tensor_out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::remainder_.Scalar  aten::remainder_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::remainder_.Tensor  aten::remainder_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::rename.default  aten::rename(Tensor(a) self, str[]? names) -> Tensor(a)
aten::rename_.default  aten::rename_(Tensor(a!) self, str[]? names) -> Tensor(a!)
aten::renorm.default  aten::renorm(Tensor self, Scalar p, int dim, Scalar maxnorm) -> Tensor
aten::renorm.out  aten::renorm.out(Tensor self, Scalar p, int dim, Scalar maxnorm, *, Tensor(a!) out) -> Tensor(a!)
aten::renorm_.default  aten::renorm_(Tensor(a!) self, Scalar p, int dim, Scalar maxnorm) -> Tensor(a!)
aten::repeat.default  aten::repeat(Tensor self, SymInt[] repeats) -> Tensor
aten::repeat.out  aten::repeat.out(Tensor self, SymInt[] repeats, *, Tensor(a!) out) -> Tensor(a!)
aten::repeat_interleave.Tensor  aten::repeat_interleave.Tensor(Tensor repeats, *, SymInt? output_size=None) -> Tensor
aten::repeat_interleave.Tensor_out  aten::repeat_interleave.Tensor_out(Tensor repeats, *, SymInt? output_size=None, Tensor(a!) out) -> Tensor(a!)
aten::repeat_interleave.self_Tensor  aten::repeat_interleave.self_Tensor(Tensor self, Tensor repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
aten::repeat_interleave.self_int  aten::repeat_interleave.self_int(Tensor self, SymInt repeats, int? dim=None, *, SymInt? output_size=None) -> Tensor
aten::replication_pad1d.default  aten::replication_pad1d(Tensor self, SymInt[2] padding) -> Tensor
aten::replication_pad1d.out  aten::replication_pad1d.out(Tensor self, SymInt[2] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::replication_pad1d_backward.default  aten::replication_pad1d_backward(Tensor grad_output, Tensor self, SymInt[2] padding) -> Tensor
aten::replication_pad1d_backward.grad_input  aten::replication_pad1d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[2] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::replication_pad2d.default  aten::replication_pad2d(Tensor self, SymInt[4] padding) -> Tensor
aten::replication_pad2d.out  aten::replication_pad2d.out(Tensor self, SymInt[4] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::replication_pad2d_backward.default  aten::replication_pad2d_backward(Tensor grad_output, Tensor self, SymInt[4] padding) -> Tensor
aten::replication_pad2d_backward.grad_input  aten::replication_pad2d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[4] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::replication_pad3d.default  aten::replication_pad3d(Tensor self, SymInt[6] padding) -> Tensor
aten::replication_pad3d.out  aten::replication_pad3d.out(Tensor self, SymInt[6] padding, *, Tensor(a!) out) -> Tensor(a!)
aten::replication_pad3d_backward.default  aten::replication_pad3d_backward(Tensor grad_output, Tensor self, SymInt[6] padding) -> Tensor
aten::replication_pad3d_backward.grad_input  aten::replication_pad3d_backward.grad_input(Tensor grad_output, Tensor self, SymInt[6] padding, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::requires_grad_.default  aten::requires_grad_(Tensor(a!) self, bool requires_grad=True) -> Tensor(a!)
aten::reshape.default  aten::reshape(Tensor(a) self, SymInt[] shape) -> Tensor(a)
aten::reshape_as.default  aten::reshape_as(Tensor(a) self, Tensor other) -> Tensor(a)
aten::resize.default  aten::resize(Tensor self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor
aten::resize.out  aten::resize.out(Tensor self, SymInt[] size, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::resize_.default  aten::resize_(Tensor(a!) self, SymInt[] size, *, MemoryFormat? memory_format=None) -> Tensor(a!)
aten::resize_as.default  aten::resize_as(Tensor self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor
aten::resize_as.out  aten::resize_as.out(Tensor self, Tensor the_template, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
aten::resize_as_.default  aten::resize_as_(Tensor(a!) self, Tensor the_template, *, MemoryFormat? memory_format=None) -> Tensor(a!)
aten::resize_as_sparse.default  aten::resize_as_sparse(Tensor self, Tensor the_template) -> Tensor
aten::resize_as_sparse.out  aten::resize_as_sparse.out(Tensor self, Tensor the_template, *, Tensor(a!) out) -> Tensor(a!)
aten::resize_as_sparse_.default  aten::resize_as_sparse_(Tensor(a!) self, Tensor the_template) -> Tensor(a!)
aten::resolve_conj.default  aten::resolve_conj(Tensor(a) self) -> Tensor(a)
aten::resolve_neg.default  aten::resolve_neg(Tensor(a) self) -> Tensor(a)
aten::result_type.Scalar  aten::result_type.Scalar(Tensor tensor, Scalar other) -> ScalarType
aten::result_type.Scalar_Scalar  aten::result_type.Scalar_Scalar(Scalar scalar1, Scalar scalar2) -> ScalarType
aten::result_type.Scalar_Tensor  aten::result_type.Scalar_Tensor(Scalar scalar, Tensor tensor) -> ScalarType
aten::result_type.Tensor  aten::result_type.Tensor(Tensor tensor, Tensor other) -> ScalarType
aten::retain_grad.default  aten::retain_grad(Tensor(a!) self) -> ()
aten::retains_grad.default  aten::retains_grad(Tensor self) -> bool
aten::rms_norm.default  aten::rms_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, float? eps=None) -> Tensor
aten::rnn_relu.data  aten::rnn_relu.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
aten::rnn_relu.input  aten::rnn_relu.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
aten::rnn_relu_cell.default  aten::rnn_relu_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
aten::rnn_tanh.data  aten::rnn_tanh.data(Tensor data, Tensor batch_sizes, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional) -> (Tensor, Tensor)
aten::rnn_tanh.input  aten::rnn_tanh.input(Tensor input, Tensor hx, Tensor[] params, bool has_biases, int num_layers, float dropout, bool train, bool bidirectional, bool batch_first) -> (Tensor, Tensor)
aten::rnn_tanh_cell.default  aten::rnn_tanh_cell(Tensor input, Tensor hx, Tensor w_ih, Tensor w_hh, Tensor? b_ih=None, Tensor? b_hh=None) -> Tensor
aten::roll.default  aten::roll(Tensor self, SymInt[1] shifts, int[1] dims=[]) -> Tensor
aten::roll.out  aten::roll.out(Tensor self, SymInt[1] shifts, int[1] dims=[], *, Tensor(a!) out) -> Tensor(a!)
aten::rot90.default  aten::rot90(Tensor self, int k=1, int[] dims=[0, 1]) -> Tensor
aten::rot90.out  aten::rot90.out(Tensor self, int k=1, int[] dims=[0, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::round.decimals  aten::round.decimals(Tensor self, *, int decimals) -> Tensor
aten::round.decimals_out  aten::round.decimals_out(Tensor self, *, int decimals, Tensor(a!) out) -> Tensor(a!)
aten::round.default  aten::round(Tensor self) -> Tensor
aten::round.out  aten::round.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::round_.decimals  aten::round_.decimals(Tensor(a!) self, *, int decimals) -> Tensor(a!)
aten::round_.default  aten::round_(Tensor(a!) self) -> Tensor(a!)
aten::row_indices.default  aten::row_indices(Tensor(a) self) -> Tensor(a)
aten::row_indices_copy.default  aten::row_indices_copy(Tensor self) -> Tensor
aten::row_indices_copy.out  aten::row_indices_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::row_stack.default  aten::row_stack(Tensor[] tensors) -> Tensor
aten::row_stack.out  aten::row_stack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::rrelu.default  aten::rrelu(Tensor self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor
aten::rrelu_.default  aten::rrelu_(Tensor(a!) self, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor(a!)
aten::rrelu_with_noise.default  aten::rrelu_with_noise(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor
aten::rrelu_with_noise.out  aten::rrelu_with_noise.out(Tensor self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None, *, Tensor(a!) out) -> Tensor(a!)
aten::rrelu_with_noise_.default  aten::rrelu_with_noise_(Tensor(a!) self, Tensor noise, Scalar lower=0.125, Scalar upper=0.33333333333333331, bool training=False, Generator? generator=None) -> Tensor(a!)
aten::rrelu_with_noise_backward.default  aten::rrelu_with_noise_backward(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result) -> Tensor
aten::rrelu_with_noise_backward.out  aten::rrelu_with_noise_backward.out(Tensor grad_output, Tensor self, Tensor noise, Scalar lower, Scalar upper, bool training, bool self_is_result, *, Tensor(a!) out) -> Tensor(a!)
aten::rsqrt.default  aten::rsqrt(Tensor self) -> Tensor
aten::rsqrt.out  aten::rsqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::rsqrt_.default  aten::rsqrt_(Tensor(a!) self) -> Tensor(a!)
aten::rsub.Scalar  aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::rsub.Scalar_out  aten::rsub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
aten::rsub.Tensor  aten::rsub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
aten::rsub.Tensor_out  aten::rsub.Tensor_out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::scalar_tensor.default  aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::scalar_tensor.out  aten::scalar_tensor.out(Scalar s, *, Tensor(a!) out) -> Tensor(a!)
aten::scaled_dot_product_attention.default  aten::scaled_dot_product_attention(Tensor query, Tensor key, Tensor value, Tensor? attn_mask=None, float dropout_p=0., bool is_causal=False, *, float? scale=None) -> Tensor
aten::scatter.dimname_src  aten::scatter.dimname_src(Tensor self, str dim, Tensor index, Tensor src) -> Tensor
aten::scatter.dimname_value  aten::scatter.dimname_value(Tensor self, str dim, Tensor index, Scalar value) -> Tensor
aten::scatter.reduce  aten::scatter.reduce(Tensor self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor
aten::scatter.reduce_out  aten::scatter.reduce_out(Tensor self, int dim, Tensor index, Tensor src, *, str reduce, Tensor(a!) out) -> Tensor(a!)
aten::scatter.src  aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
aten::scatter.src_out  aten::scatter.src_out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
aten::scatter.value  aten::scatter.value(Tensor self, int dim, Tensor index, Scalar value) -> Tensor
aten::scatter.value_out  aten::scatter.value_out(Tensor self, int dim, Tensor index, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
aten::scatter.value_reduce  aten::scatter.value_reduce(Tensor self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor
aten::scatter.value_reduce_out  aten::scatter.value_reduce_out(Tensor self, int dim, Tensor index, Scalar value, *, str reduce, Tensor(a!) out) -> Tensor(a!)
aten::scatter_.reduce  aten::scatter_.reduce(Tensor(a!) self, int dim, Tensor index, Tensor src, *, str reduce) -> Tensor(a!)
aten::scatter_.src  aten::scatter_.src(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
aten::scatter_.value  aten::scatter_.value(Tensor(a!) self, int dim, Tensor index, Scalar value) -> Tensor(a!)
aten::scatter_.value_reduce  aten::scatter_.value_reduce(Tensor(a!) self, int dim, Tensor index, Scalar value, *, str reduce) -> Tensor(a!)
aten::scatter_add.default  aten::scatter_add(Tensor self, int dim, Tensor index, Tensor src) -> Tensor
aten::scatter_add.dimname  aten::scatter_add.dimname(Tensor self, str dim, Tensor index, Tensor src) -> Tensor
aten::scatter_add.out  aten::scatter_add.out(Tensor self, int dim, Tensor index, Tensor src, *, Tensor(a!) out) -> Tensor(a!)
aten::scatter_add_.default  aten::scatter_add_(Tensor(a!) self, int dim, Tensor index, Tensor src) -> Tensor(a!)
aten::scatter_reduce.two  aten::scatter_reduce.two(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor
aten::scatter_reduce.two_out  aten::scatter_reduce.two_out(Tensor self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True, Tensor(a!) out) -> Tensor(a!)
aten::scatter_reduce_.two  aten::scatter_reduce_.two(Tensor(a!) self, int dim, Tensor index, Tensor src, str reduce, *, bool include_self=True) -> Tensor(a!)
aten::searchsorted.Scalar  aten::searchsorted.Scalar(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
aten::searchsorted.Scalar_out  aten::searchsorted.Scalar_out(Tensor sorted_sequence, Scalar self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
aten::searchsorted.Tensor  aten::searchsorted.Tensor(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None) -> Tensor
aten::searchsorted.Tensor_out  aten::searchsorted.Tensor_out(Tensor sorted_sequence, Tensor self, *, bool out_int32=False, bool right=False, str? side=None, Tensor? sorter=None, Tensor(a!) out) -> Tensor(a!)
aten::segment_reduce.default  aten::segment_reduce(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None) -> Tensor
aten::segment_reduce.out  aten::segment_reduce.out(Tensor data, str reduce, *, Tensor? lengths=None, Tensor? indices=None, Tensor? offsets=None, int axis=0, bool unsafe=False, Scalar? initial=None, Tensor(a!) out) -> Tensor(a!)
aten::select.Dimname  aten::select.Dimname(Tensor(a) self, str dim, int index) -> Tensor(a)
aten::select.int  aten::select.int(Tensor(a) self, int dim, SymInt index) -> Tensor(a)
aten::select_backward.default  aten::select_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index) -> Tensor
aten::select_backward.out  aten::select_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)
aten::select_copy.int  aten::select_copy.int(Tensor self, int dim, SymInt index) -> Tensor
aten::select_copy.int_out  aten::select_copy.int_out(Tensor self, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)
aten::select_scatter.default  aten::select_scatter(Tensor self, Tensor src, int dim, SymInt index) -> Tensor
aten::select_scatter.out  aten::select_scatter.out(Tensor self, Tensor src, int dim, SymInt index, *, Tensor(a!) out) -> Tensor(a!)
aten::selu.default  aten::selu(Tensor self) -> Tensor
aten::selu_.default  aten::selu_(Tensor(a!) self) -> Tensor(a!)
aten::set.default  aten::set(Tensor self) -> Tensor
aten::set.out  aten::set.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::set.source_Storage  aten::set.source_Storage(Tensor self, Storage source) -> Tensor
aten::set.source_Storage_out  aten::set.source_Storage_out(Tensor self, Storage source, *, Tensor(a!) out) -> Tensor(a!)
aten::set.source_Storage_storage_offset  aten::set.source_Storage_storage_offset(Tensor self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor
aten::set.source_Storage_storage_offset_out  aten::set.source_Storage_storage_offset_out(Tensor self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[], *, Tensor(a!) out) -> Tensor(a!)
aten::set.source_Tensor  aten::set.source_Tensor(Tensor self, Tensor source) -> Tensor
aten::set.source_Tensor_out  aten::set.source_Tensor_out(Tensor self, Tensor source, *, Tensor(a!) out) -> Tensor(a!)
aten::set_.default  aten::set_(Tensor(a!) self) -> Tensor(a!)
aten::set_.source_Storage  aten::set_.source_Storage(Tensor(a!) self, Storage source) -> Tensor(a!)
aten::set_.source_Storage_storage_offset  aten::set_.source_Storage_storage_offset(Tensor(a!) self, Storage source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)
aten::set_.source_Tensor  aten::set_.source_Tensor(Tensor(a!) self, Tensor source) -> Tensor(a!)
aten::set_.source_Tensor_storage_offset  aten::set_.source_Tensor_storage_offset(Tensor(a!) self, Tensor source, SymInt storage_offset, SymInt[] size, SymInt[] stride=[]) -> Tensor(a!)
aten::set_data.default  aten::set_data(Tensor(a!) self, Tensor new_data) -> ()
aten::sgn.default  aten::sgn(Tensor self) -> Tensor
aten::sgn.out  aten::sgn.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sgn_.default  aten::sgn_(Tensor(a!) self) -> Tensor(a!)
aten::sigmoid.default  aten::sigmoid(Tensor self) -> Tensor
aten::sigmoid.out  aten::sigmoid.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sigmoid_.default  aten::sigmoid_(Tensor(a!) self) -> Tensor(a!)
aten::sigmoid_backward.default  aten::sigmoid_backward(Tensor grad_output, Tensor output) -> Tensor
aten::sigmoid_backward.grad_input  aten::sigmoid_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::sign.default  aten::sign(Tensor self) -> Tensor
aten::sign.out  aten::sign.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sign_.default  aten::sign_(Tensor(a!) self) -> Tensor(a!)
aten::signbit.default  aten::signbit(Tensor self) -> Tensor
aten::signbit.out  aten::signbit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::silu.default  aten::silu(Tensor self) -> Tensor
aten::silu.out  aten::silu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::silu_.default  aten::silu_(Tensor(a!) self) -> Tensor(a!)
aten::silu_backward.default  aten::silu_backward(Tensor grad_output, Tensor self) -> Tensor
aten::silu_backward.grad_input  aten::silu_backward.grad_input(Tensor grad_output, Tensor self, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::sin.default  aten::sin(Tensor self) -> Tensor
aten::sin.out  aten::sin.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sin_.default  aten::sin_(Tensor(a!) self) -> Tensor(a!)
aten::sinc.default  aten::sinc(Tensor self) -> Tensor
aten::sinc.out  aten::sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sinc_.default  aten::sinc_(Tensor(a!) self) -> Tensor(a!)
aten::sinh.default  aten::sinh(Tensor self) -> Tensor
aten::sinh.out  aten::sinh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sinh_.default  aten::sinh_(Tensor(a!) self) -> Tensor(a!)
aten::size.Dimname  aten::size.Dimname(Tensor self, str dim) -> int
aten::size.int  aten::size.int(Tensor self, int dim) -> int
aten::slice.Tensor  aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
aten::slice_backward.default  aten::slice_backward(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step) -> Tensor
aten::slice_backward.out  aten::slice_backward.out(Tensor grad_output, SymInt[] input_sizes, int dim, SymInt start, SymInt end, SymInt step, *, Tensor(a!) out) -> Tensor(a!)
aten::slice_copy.Tensor  aten::slice_copy.Tensor(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
aten::slice_copy.Tensor_out  aten::slice_copy.Tensor_out(Tensor self, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)
aten::slice_inverse.default  aten::slice_inverse(Tensor(a) self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor(a)
aten::slice_scatter.default  aten::slice_scatter(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1) -> Tensor
aten::slice_scatter.out  aten::slice_scatter.out(Tensor self, Tensor src, int dim=0, SymInt? start=None, SymInt? end=None, SymInt step=1, *, Tensor(a!) out) -> Tensor(a!)
aten::slogdet.default  aten::slogdet(Tensor self) -> (Tensor sign, Tensor logabsdet)
aten::slogdet.out  aten::slogdet.out(Tensor self, *, Tensor(a!) sign, Tensor(b!) logabsdet) -> (Tensor(a!) sign, Tensor(b!) logabsdet)
aten::slow_conv3d.default  aten::slow_conv3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0]) -> Tensor
aten::slow_conv3d.out  aten::slow_conv3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], *, Tensor(a!) out) -> Tensor(a!)
aten::slow_conv3d_forward.default  aten::slow_conv3d_forward(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding) -> Tensor
aten::slow_conv3d_forward.output  aten::slow_conv3d_forward.output(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias, SymInt[3] stride, SymInt[3] padding, *, Tensor(a!) output) -> Tensor(a!)
aten::slow_conv_dilated2d.default  aten::slow_conv_dilated2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] dilation=[1, 1]) -> Tensor
aten::slow_conv_dilated2d.out  aten::slow_conv_dilated2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] dilation=[1, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::slow_conv_dilated3d.default  aten::slow_conv_dilated3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1]) -> Tensor
aten::slow_conv_dilated3d.out  aten::slow_conv_dilated3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::slow_conv_transpose2d.default  aten::slow_conv_transpose2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] output_padding=[0, 0], SymInt[2] dilation=[1, 1]) -> Tensor
aten::slow_conv_transpose2d.out  aten::slow_conv_transpose2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], SymInt[2] output_padding=[0, 0], SymInt[2] dilation=[1, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::slow_conv_transpose3d.default  aten::slow_conv_transpose3d(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] output_padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1]) -> Tensor
aten::slow_conv_transpose3d.out  aten::slow_conv_transpose3d.out(Tensor self, Tensor weight, SymInt[3] kernel_size, Tensor? bias=None, SymInt[3] stride=[1, 1, 1], SymInt[3] padding=[0, 0, 0], SymInt[3] output_padding=[0, 0, 0], SymInt[3] dilation=[1, 1, 1], *, Tensor(a!) out) -> Tensor(a!)
aten::smm.default  aten::smm(Tensor self, Tensor mat2) -> Tensor
aten::smooth_l1_loss.default  aten::smooth_l1_loss(Tensor self, Tensor target, int reduction=1, float beta=1.) -> Tensor
aten::smooth_l1_loss.out  aten::smooth_l1_loss.out(Tensor self, Tensor target, int reduction=1, float beta=1., *, Tensor(a!) out) -> Tensor(a!)
aten::smooth_l1_loss_backward.default  aten::smooth_l1_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta) -> Tensor
aten::smooth_l1_loss_backward.grad_input  aten::smooth_l1_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, float beta, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::soft_margin_loss.default  aten::soft_margin_loss(Tensor self, Tensor target, int reduction=1) -> Tensor
aten::soft_margin_loss.out  aten::soft_margin_loss.out(Tensor self, Tensor target, int reduction=1, *, Tensor(a!) out) -> Tensor(a!)
aten::soft_margin_loss_backward.default  aten::soft_margin_loss_backward(Tensor grad_output, Tensor self, Tensor target, int reduction) -> Tensor
aten::soft_margin_loss_backward.grad_input  aten::soft_margin_loss_backward.grad_input(Tensor grad_output, Tensor self, Tensor target, int reduction, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::softmax.Dimname  aten::softmax.Dimname(Tensor self, str dim, *, ScalarType? dtype=None) -> Tensor
aten::softmax.int  aten::softmax.int(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
aten::softmax.int_out  aten::softmax.int_out(Tensor self, int dim, ScalarType? dtype=None, *, Tensor(a!) out) -> Tensor(a!)
aten::softplus.default  aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor
aten::softplus.out  aten::softplus.out(Tensor self, Scalar beta=1, Scalar threshold=20, *, Tensor(a!) out) -> Tensor(a!)
aten::softplus_backward.default  aten::softplus_backward(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold) -> Tensor
aten::softplus_backward.grad_input  aten::softplus_backward.grad_input(Tensor grad_output, Tensor self, Scalar beta, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::softshrink.default  aten::softshrink(Tensor self, Scalar lambd=0.5) -> Tensor
aten::softshrink.out  aten::softshrink.out(Tensor self, Scalar lambd=0.5, *, Tensor(a!) out) -> Tensor(a!)
aten::softshrink_backward.default  aten::softshrink_backward(Tensor grad_output, Tensor self, Scalar lambd) -> Tensor
aten::softshrink_backward.grad_input  aten::softshrink_backward.grad_input(Tensor grad_output, Tensor self, Scalar lambd, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::sort.default  aten::sort(Tensor self, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
aten::sort.dimname  aten::sort.dimname(Tensor self, str dim, bool descending=False) -> (Tensor values, Tensor indices)
aten::sort.dimname_stable  aten::sort.dimname_stable(Tensor self, *, bool? stable, str dim, bool descending=False) -> (Tensor values, Tensor indices)
aten::sort.dimname_values  aten::sort.dimname_values(Tensor self, str dim, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::sort.dimname_values_stable  aten::sort.dimname_values_stable(Tensor self, *, bool? stable, str dim, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::sort.stable  aten::sort.stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False) -> (Tensor values, Tensor indices)
aten::sort.values  aten::sort.values(Tensor self, int dim=-1, bool descending=False, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::sort.values_stable  aten::sort.values_stable(Tensor self, *, bool? stable, int dim=-1, bool descending=False, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::sparse_bsc_tensor.ccol_row_value  aten::sparse_bsc_tensor.ccol_row_value(Tensor ccol_indices, Tensor row_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_bsc_tensor.ccol_row_value_size  aten::sparse_bsc_tensor.ccol_row_value_size(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_bsr_tensor.crow_col_value  aten::sparse_bsr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_bsr_tensor.crow_col_value_size  aten::sparse_bsr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_compressed_tensor.comp_plain_value  aten::sparse_compressed_tensor.comp_plain_value(Tensor compressed_indices, Tensor plain_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_compressed_tensor.comp_plain_value_size  aten::sparse_compressed_tensor.comp_plain_value_size(Tensor compressed_indices, Tensor plain_indices, Tensor values, SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_coo_tensor.indices  aten::sparse_coo_tensor.indices(Tensor indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool? is_coalesced=None) -> Tensor
aten::sparse_coo_tensor.indices_size  aten::sparse_coo_tensor.indices_size(Tensor indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool? is_coalesced=None) -> Tensor
aten::sparse_coo_tensor.size  aten::sparse_coo_tensor.size(int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_coo_tensor.size_out  aten::sparse_coo_tensor.size_out(int[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::sparse_csc_tensor.ccol_row_value  aten::sparse_csc_tensor.ccol_row_value(Tensor ccol_indices, Tensor row_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_csc_tensor.ccol_row_value_size  aten::sparse_csc_tensor.ccol_row_value_size(Tensor ccol_indices, Tensor row_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_csr_tensor.crow_col_value  aten::sparse_csr_tensor.crow_col_value(Tensor crow_indices, Tensor col_indices, Tensor values, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_csr_tensor.crow_col_value_size  aten::sparse_csr_tensor.crow_col_value_size(Tensor crow_indices, Tensor col_indices, Tensor values, int[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=False) -> Tensor
aten::sparse_dim.default  aten::sparse_dim(Tensor self) -> int
aten::sparse_mask.default  aten::sparse_mask(Tensor self, Tensor mask) -> Tensor
aten::sparse_mask.out  aten::sparse_mask.out(Tensor self, Tensor mask, *, Tensor(a!) out) -> Tensor(a!)
aten::sparse_resize.default  aten::sparse_resize(Tensor self, int[] size, int sparse_dim, int dense_dim) -> Tensor
aten::sparse_resize.out  aten::sparse_resize.out(Tensor self, int[] size, int sparse_dim, int dense_dim, *, Tensor(a!) out) -> Tensor(a!)
aten::sparse_resize_.default  aten::sparse_resize_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
aten::sparse_resize_and_clear.default  aten::sparse_resize_and_clear(Tensor self, int[] size, int sparse_dim, int dense_dim) -> Tensor
aten::sparse_resize_and_clear.out  aten::sparse_resize_and_clear.out(Tensor self, int[] size, int sparse_dim, int dense_dim, *, Tensor(a!) out) -> Tensor(a!)
aten::sparse_resize_and_clear_.default  aten::sparse_resize_and_clear_(Tensor(a!) self, int[] size, int sparse_dim, int dense_dim) -> Tensor(a!)
aten::sparse_sampled_addmm.default  aten::sparse_sampled_addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::sparse_sampled_addmm.out  aten::sparse_sampled_addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::special_airy_ai.default  aten::special_airy_ai(Tensor x) -> Tensor
aten::special_airy_ai.out  aten::special_airy_ai.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
aten::special_bessel_j0.default  aten::special_bessel_j0(Tensor self) -> Tensor
aten::special_bessel_j0.out  aten::special_bessel_j0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_bessel_j1.default  aten::special_bessel_j1(Tensor self) -> Tensor
aten::special_bessel_j1.out  aten::special_bessel_j1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_bessel_y0.default  aten::special_bessel_y0(Tensor self) -> Tensor
aten::special_bessel_y0.out  aten::special_bessel_y0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_bessel_y1.default  aten::special_bessel_y1(Tensor self) -> Tensor
aten::special_bessel_y1.out  aten::special_bessel_y1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_t.default  aten::special_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_t.n_scalar  aten::special_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_chebyshev_polynomial_t.n_scalar_out  aten::special_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_t.out  aten::special_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_t.x_scalar  aten::special_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_t.x_scalar_out  aten::special_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_u.default  aten::special_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_u.n_scalar  aten::special_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_chebyshev_polynomial_u.n_scalar_out  aten::special_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_u.out  aten::special_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_u.x_scalar  aten::special_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_u.x_scalar_out  aten::special_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_v.default  aten::special_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_v.n_scalar  aten::special_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_chebyshev_polynomial_v.n_scalar_out  aten::special_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_v.out  aten::special_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_v.x_scalar  aten::special_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_v.x_scalar_out  aten::special_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_w.default  aten::special_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_w.n_scalar  aten::special_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_chebyshev_polynomial_w.n_scalar_out  aten::special_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_w.out  aten::special_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_chebyshev_polynomial_w.x_scalar  aten::special_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_chebyshev_polynomial_w.x_scalar_out  aten::special_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_digamma.default  aten::special_digamma(Tensor self) -> Tensor
aten::special_digamma.out  aten::special_digamma.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_entr.default  aten::special_entr(Tensor self) -> Tensor
aten::special_entr.out  aten::special_entr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_erf.default  aten::special_erf(Tensor self) -> Tensor
aten::special_erf.out  aten::special_erf.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_erfc.default  aten::special_erfc(Tensor self) -> Tensor
aten::special_erfc.out  aten::special_erfc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_erfcx.default  aten::special_erfcx(Tensor self) -> Tensor
aten::special_erfcx.out  aten::special_erfcx.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_erfinv.default  aten::special_erfinv(Tensor self) -> Tensor
aten::special_erfinv.out  aten::special_erfinv.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_exp2.default  aten::special_exp2(Tensor self) -> Tensor
aten::special_exp2.out  aten::special_exp2.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_expit.default  aten::special_expit(Tensor self) -> Tensor
aten::special_expit.out  aten::special_expit.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_expm1.default  aten::special_expm1(Tensor self) -> Tensor
aten::special_expm1.out  aten::special_expm1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_gammainc.default  aten::special_gammainc(Tensor self, Tensor other) -> Tensor
aten::special_gammainc.out  aten::special_gammainc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_gammaincc.default  aten::special_gammaincc(Tensor self, Tensor other) -> Tensor
aten::special_gammaincc.out  aten::special_gammaincc.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_gammaln.default  aten::special_gammaln(Tensor self) -> Tensor
aten::special_gammaln.out  aten::special_gammaln.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_h.default  aten::special_hermite_polynomial_h(Tensor x, Tensor n) -> Tensor
aten::special_hermite_polynomial_h.n_scalar  aten::special_hermite_polynomial_h.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_hermite_polynomial_h.n_scalar_out  aten::special_hermite_polynomial_h.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_h.out  aten::special_hermite_polynomial_h.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_h.x_scalar  aten::special_hermite_polynomial_h.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_hermite_polynomial_h.x_scalar_out  aten::special_hermite_polynomial_h.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_he.default  aten::special_hermite_polynomial_he(Tensor x, Tensor n) -> Tensor
aten::special_hermite_polynomial_he.n_scalar  aten::special_hermite_polynomial_he.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_hermite_polynomial_he.n_scalar_out  aten::special_hermite_polynomial_he.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_he.out  aten::special_hermite_polynomial_he.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_hermite_polynomial_he.x_scalar  aten::special_hermite_polynomial_he.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_hermite_polynomial_he.x_scalar_out  aten::special_hermite_polynomial_he.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_i0.default  aten::special_i0(Tensor self) -> Tensor
aten::special_i0.out  aten::special_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_i0e.default  aten::special_i0e(Tensor self) -> Tensor
aten::special_i0e.out  aten::special_i0e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_i1.default  aten::special_i1(Tensor self) -> Tensor
aten::special_i1.out  aten::special_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_i1e.default  aten::special_i1e(Tensor self) -> Tensor
aten::special_i1e.out  aten::special_i1e.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_laguerre_polynomial_l.default  aten::special_laguerre_polynomial_l(Tensor x, Tensor n) -> Tensor
aten::special_laguerre_polynomial_l.n_scalar  aten::special_laguerre_polynomial_l.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_laguerre_polynomial_l.n_scalar_out  aten::special_laguerre_polynomial_l.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_laguerre_polynomial_l.out  aten::special_laguerre_polynomial_l.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_laguerre_polynomial_l.x_scalar  aten::special_laguerre_polynomial_l.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_laguerre_polynomial_l.x_scalar_out  aten::special_laguerre_polynomial_l.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_legendre_polynomial_p.default  aten::special_legendre_polynomial_p(Tensor x, Tensor n) -> Tensor
aten::special_legendre_polynomial_p.n_scalar  aten::special_legendre_polynomial_p.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_legendre_polynomial_p.n_scalar_out  aten::special_legendre_polynomial_p.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_legendre_polynomial_p.out  aten::special_legendre_polynomial_p.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_legendre_polynomial_p.x_scalar  aten::special_legendre_polynomial_p.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_legendre_polynomial_p.x_scalar_out  aten::special_legendre_polynomial_p.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_log1p.default  aten::special_log1p(Tensor self) -> Tensor
aten::special_log1p.out  aten::special_log1p.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_log_ndtr.default  aten::special_log_ndtr(Tensor self) -> Tensor
aten::special_log_ndtr.out  aten::special_log_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_log_softmax.default  aten::special_log_softmax(Tensor self, int dim, *, ScalarType? dtype=None) -> Tensor
aten::special_logit.default  aten::special_logit(Tensor self, float? eps=None) -> Tensor
aten::special_logit.out  aten::special_logit.out(Tensor self, float? eps=None, *, Tensor(a!) out) -> Tensor(a!)
aten::special_logsumexp.default  aten::special_logsumexp(Tensor self, int[1] dim, bool keepdim=False) -> Tensor
aten::special_logsumexp.out  aten::special_logsumexp.out(Tensor self, int[1] dim, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::special_modified_bessel_i0.default  aten::special_modified_bessel_i0(Tensor self) -> Tensor
aten::special_modified_bessel_i0.out  aten::special_modified_bessel_i0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_modified_bessel_i1.default  aten::special_modified_bessel_i1(Tensor self) -> Tensor
aten::special_modified_bessel_i1.out  aten::special_modified_bessel_i1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_modified_bessel_k0.default  aten::special_modified_bessel_k0(Tensor self) -> Tensor
aten::special_modified_bessel_k0.out  aten::special_modified_bessel_k0.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_modified_bessel_k1.default  aten::special_modified_bessel_k1(Tensor self) -> Tensor
aten::special_modified_bessel_k1.out  aten::special_modified_bessel_k1.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_multigammaln.default  aten::special_multigammaln(Tensor self, int p) -> Tensor
aten::special_multigammaln.out  aten::special_multigammaln.out(Tensor self, int p, *, Tensor(a!) out) -> Tensor(a!)
aten::special_ndtr.default  aten::special_ndtr(Tensor self) -> Tensor
aten::special_ndtr.out  aten::special_ndtr.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_ndtri.default  aten::special_ndtri(Tensor self) -> Tensor
aten::special_ndtri.out  aten::special_ndtri.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_polygamma.default  aten::special_polygamma(int n, Tensor self) -> Tensor
aten::special_polygamma.out  aten::special_polygamma.out(int n, Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_psi.default  aten::special_psi(Tensor self) -> Tensor
aten::special_psi.out  aten::special_psi.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_round.default  aten::special_round(Tensor self, *, int decimals=0) -> Tensor
aten::special_round.out  aten::special_round.out(Tensor self, *, int decimals=0, Tensor(a!) out) -> Tensor(a!)
aten::special_scaled_modified_bessel_k0.default  aten::special_scaled_modified_bessel_k0(Tensor x) -> Tensor
aten::special_scaled_modified_bessel_k0.out  aten::special_scaled_modified_bessel_k0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
aten::special_scaled_modified_bessel_k1.default  aten::special_scaled_modified_bessel_k1(Tensor x) -> Tensor
aten::special_scaled_modified_bessel_k1.out  aten::special_scaled_modified_bessel_k1.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_t.default  aten::special_shifted_chebyshev_polynomial_t(Tensor x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_t.n_scalar  aten::special_shifted_chebyshev_polynomial_t.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_shifted_chebyshev_polynomial_t.n_scalar_out  aten::special_shifted_chebyshev_polynomial_t.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_t.out  aten::special_shifted_chebyshev_polynomial_t.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_t.x_scalar  aten::special_shifted_chebyshev_polynomial_t.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_t.x_scalar_out  aten::special_shifted_chebyshev_polynomial_t.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_u.default  aten::special_shifted_chebyshev_polynomial_u(Tensor x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_u.n_scalar  aten::special_shifted_chebyshev_polynomial_u.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_shifted_chebyshev_polynomial_u.n_scalar_out  aten::special_shifted_chebyshev_polynomial_u.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_u.out  aten::special_shifted_chebyshev_polynomial_u.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_u.x_scalar  aten::special_shifted_chebyshev_polynomial_u.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_u.x_scalar_out  aten::special_shifted_chebyshev_polynomial_u.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_v.default  aten::special_shifted_chebyshev_polynomial_v(Tensor x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_v.n_scalar  aten::special_shifted_chebyshev_polynomial_v.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_shifted_chebyshev_polynomial_v.n_scalar_out  aten::special_shifted_chebyshev_polynomial_v.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_v.out  aten::special_shifted_chebyshev_polynomial_v.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_v.x_scalar  aten::special_shifted_chebyshev_polynomial_v.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_v.x_scalar_out  aten::special_shifted_chebyshev_polynomial_v.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_w.default  aten::special_shifted_chebyshev_polynomial_w(Tensor x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_w.n_scalar  aten::special_shifted_chebyshev_polynomial_w.n_scalar(Tensor x, Scalar n) -> Tensor
aten::special_shifted_chebyshev_polynomial_w.n_scalar_out  aten::special_shifted_chebyshev_polynomial_w.n_scalar_out(Tensor x, Scalar n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_w.out  aten::special_shifted_chebyshev_polynomial_w.out(Tensor x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_shifted_chebyshev_polynomial_w.x_scalar  aten::special_shifted_chebyshev_polynomial_w.x_scalar(Scalar x, Tensor n) -> Tensor
aten::special_shifted_chebyshev_polynomial_w.x_scalar_out  aten::special_shifted_chebyshev_polynomial_w.x_scalar_out(Scalar x, Tensor n, *, Tensor(a!) out) -> Tensor(a!)
aten::special_sinc.default  aten::special_sinc(Tensor self) -> Tensor
aten::special_sinc.out  aten::special_sinc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::special_softmax.default  aten::special_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor
aten::special_spherical_bessel_j0.default  aten::special_spherical_bessel_j0(Tensor x) -> Tensor
aten::special_spherical_bessel_j0.out  aten::special_spherical_bessel_j0.out(Tensor x, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlog1py.default  aten::special_xlog1py(Tensor self, Tensor other) -> Tensor
aten::special_xlog1py.other_scalar  aten::special_xlog1py.other_scalar(Tensor self, Scalar other) -> Tensor
aten::special_xlog1py.other_scalar_out  aten::special_xlog1py.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlog1py.out  aten::special_xlog1py.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlog1py.self_scalar  aten::special_xlog1py.self_scalar(Scalar self, Tensor other) -> Tensor
aten::special_xlog1py.self_scalar_out  aten::special_xlog1py.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlogy.default  aten::special_xlogy(Tensor self, Tensor other) -> Tensor
aten::special_xlogy.other_scalar  aten::special_xlogy.other_scalar(Tensor self, Scalar other) -> Tensor
aten::special_xlogy.other_scalar_out  aten::special_xlogy.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlogy.out  aten::special_xlogy.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_xlogy.self_scalar  aten::special_xlogy.self_scalar(Scalar self, Tensor other) -> Tensor
aten::special_xlogy.self_scalar_out  aten::special_xlogy.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_zeta.default  aten::special_zeta(Tensor self, Tensor other) -> Tensor
aten::special_zeta.other_scalar  aten::special_zeta.other_scalar(Tensor self, Scalar other) -> Tensor
aten::special_zeta.other_scalar_out  aten::special_zeta.other_scalar_out(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_zeta.out  aten::special_zeta.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::special_zeta.self_scalar  aten::special_zeta.self_scalar(Scalar self, Tensor other) -> Tensor
aten::special_zeta.self_scalar_out  aten::special_zeta.self_scalar_out(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::split.Tensor  aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0) -> Tensor(a)[]
aten::split.sizes  aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]
aten::split_copy.Tensor  aten::split_copy.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
aten::split_copy.Tensor_out  aten::split_copy.Tensor_out(Tensor self, SymInt split_size, int dim=0, *, Tensor(a!)[] out) -> ()
aten::split_with_sizes.default  aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int dim=0) -> Tensor(a)[]
aten::split_with_sizes_copy.default  aten::split_with_sizes_copy(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
aten::split_with_sizes_copy.out  aten::split_with_sizes_copy.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()
aten::sqrt.default  aten::sqrt(Tensor self) -> Tensor
aten::sqrt.out  aten::sqrt.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sqrt_.default  aten::sqrt_(Tensor(a!) self) -> Tensor(a!)
aten::square.default  aten::square(Tensor self) -> Tensor
aten::square.out  aten::square.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::square_.default  aten::square_(Tensor(a!) self) -> Tensor(a!)
aten::squeeze.default  aten::squeeze(Tensor(a) self) -> Tensor(a)
aten::squeeze.dim  aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)
aten::squeeze.dimname  aten::squeeze.dimname(Tensor(a) self, str dim) -> Tensor(a)
aten::squeeze.dims  aten::squeeze.dims(Tensor(a) self, int[] dim) -> Tensor(a)
aten::squeeze_.default  aten::squeeze_(Tensor(a!) self) -> Tensor(a!)
aten::squeeze_.dim  aten::squeeze_.dim(Tensor(a!) self, int dim) -> Tensor(a!)
aten::squeeze_.dimname  aten::squeeze_.dimname(Tensor(a!) self, str dim) -> Tensor(a!)
aten::squeeze_.dims  aten::squeeze_.dims(Tensor(a!) self, int[] dim) -> Tensor(a!)
aten::squeeze_copy.default  aten::squeeze_copy(Tensor self) -> Tensor
aten::squeeze_copy.dim  aten::squeeze_copy.dim(Tensor self, int dim) -> Tensor
aten::squeeze_copy.dim_out  aten::squeeze_copy.dim_out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::squeeze_copy.dims  aten::squeeze_copy.dims(Tensor self, int[] dim) -> Tensor
aten::squeeze_copy.dims_out  aten::squeeze_copy.dims_out(Tensor self, int[] dim, *, Tensor(a!) out) -> Tensor(a!)
aten::squeeze_copy.out  aten::squeeze_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::sspaddmm.default  aten::sspaddmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1) -> Tensor
aten::sspaddmm.out  aten::sspaddmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::stack.default  aten::stack(Tensor[] tensors, int dim=0) -> Tensor
aten::stack.out  aten::stack.out(Tensor[] tensors, int dim=0, *, Tensor(a!) out) -> Tensor(a!)
aten::std.correction  aten::std.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
aten::std.correction_names  aten::std.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
aten::std.correction_names_out  aten::std.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
aten::std.correction_out  aten::std.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
aten::std.default  aten::std(Tensor self, bool unbiased=True) -> Tensor
aten::std.dim  aten::std.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
aten::std.names_dim  aten::std.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
aten::std.names_out  aten::std.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::std.out  aten::std.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::std_mean.correction  aten::std_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
aten::std_mean.correction_names  aten::std_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
aten::std_mean.correction_out  aten::std_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::std_mean.default  aten::std_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
aten::std_mean.dim  aten::std_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
aten::std_mean.names_dim  aten::std_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
aten::stft.center  aten::stft.center(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool center=True, str pad_mode="reflect", bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
aten::stft.default  aten::stft(Tensor self, int n_fft, int? hop_length=None, int? win_length=None, Tensor? window=None, bool normalized=False, bool? onesided=None, bool? return_complex=None) -> Tensor
aten::stride.Dimname  aten::stride.Dimname(Tensor self, str dim) -> int
aten::stride.int  aten::stride.int(Tensor self, int dim) -> int
aten::sub.Scalar  aten::sub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::sub.Scalar_out  aten::sub.Scalar_out(Tensor self, Scalar other, Scalar alpha=1, *, Tensor(a!) out) -> Tensor(a!)
aten::sub.Tensor  aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
aten::sub.out  aten::sub.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::sub_.Scalar  aten::sub_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
aten::sub_.Tensor  aten::sub_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
aten::subtract.Scalar  aten::subtract.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor
aten::subtract.Tensor  aten::subtract.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
aten::subtract.out  aten::subtract.out(Tensor self, Tensor other, *, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)
aten::subtract_.Scalar  aten::subtract_.Scalar(Tensor(a!) self, Scalar other, Scalar alpha=1) -> Tensor(a!)
aten::subtract_.Tensor  aten::subtract_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)
aten::sum.DimnameList_out  aten::sum.DimnameList_out(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::sum.IntList_out  aten::sum.IntList_out(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::sum.default  aten::sum(Tensor self, *, ScalarType? dtype=None) -> Tensor
aten::sum.dim_DimnameList  aten::sum.dim_DimnameList(Tensor self, str[1] dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::sum.dim_IntList  aten::sum.dim_IntList(Tensor self, int[1]? dim, bool keepdim=False, *, ScalarType? dtype=None) -> Tensor
aten::sum.out  aten::sum.out(Tensor self, *, ScalarType? dtype=None, Tensor(a!) out) -> Tensor(a!)
aten::sum_to_size.default  aten::sum_to_size(Tensor self, SymInt[] size) -> Tensor
aten::svd.U  aten::svd.U(Tensor self, bool some=True, bool compute_uv=True, *, Tensor(a!) U, Tensor(b!) S, Tensor(c!) V) -> (Tensor(a!) U, Tensor(b!) S, Tensor(c!) V)
aten::svd.default  aten::svd(Tensor self, bool some=True, bool compute_uv=True) -> (Tensor U, Tensor S, Tensor V)
aten::swapaxes.default  aten::swapaxes(Tensor(a) self, int axis0, int axis1) -> Tensor(a)
aten::swapaxes_.default  aten::swapaxes_(Tensor(a!) self, int axis0, int axis1) -> Tensor(a!)
aten::swapdims.default  aten::swapdims(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
aten::swapdims_.default  aten::swapdims_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
aten::sym_constrain_range.default  aten::sym_constrain_range(Scalar size, *, int? min=None, int? max=None) -> ()
aten::sym_constrain_range_for_size.default  aten::sym_constrain_range_for_size(Scalar size, *, int? min=None, int? max=None) -> ()
aten::sym_numel.default  aten::sym_numel(Tensor self) -> SymInt
aten::sym_size.int  aten::sym_size.int(Tensor self, int dim) -> SymInt
aten::sym_storage_offset.default  aten::sym_storage_offset(Tensor self) -> SymInt
aten::sym_stride.int  aten::sym_stride.int(Tensor self, int dim) -> SymInt
aten::t.default  aten::t(Tensor(a) self) -> Tensor(a)
aten::t_.default  aten::t_(Tensor(a!) self) -> Tensor(a!)
aten::t_copy.default  aten::t_copy(Tensor self) -> Tensor
aten::t_copy.out  aten::t_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::take.default  aten::take(Tensor self, Tensor index) -> Tensor
aten::take.out  aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
aten::take_along_dim.default  aten::take_along_dim(Tensor self, Tensor indices, int? dim=None) -> Tensor
aten::take_along_dim.out  aten::take_along_dim.out(Tensor self, Tensor indices, int? dim=None, *, Tensor(a!) out) -> Tensor(a!)
aten::tan.default  aten::tan(Tensor self) -> Tensor
aten::tan.out  aten::tan.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::tan_.default  aten::tan_(Tensor(a!) self) -> Tensor(a!)
aten::tanh.default  aten::tanh(Tensor self) -> Tensor
aten::tanh.out  aten::tanh.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::tanh_.default  aten::tanh_(Tensor(a!) self) -> Tensor(a!)
aten::tanh_backward.default  aten::tanh_backward(Tensor grad_output, Tensor output) -> Tensor
aten::tanh_backward.grad_input  aten::tanh_backward.grad_input(Tensor grad_output, Tensor output, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::tensor_split.indices  aten::tensor_split.indices(Tensor(a -> *) self, SymInt[] indices, int dim=0) -> Tensor(a)[]
aten::tensor_split.sections  aten::tensor_split.sections(Tensor(a -> *) self, SymInt sections, int dim=0) -> Tensor(a)[]
aten::tensor_split.tensor_indices_or_sections  aten::tensor_split.tensor_indices_or_sections(Tensor(a -> *) self, Tensor tensor_indices_or_sections, int dim=0) -> Tensor(a)[]
aten::tensordot.default  aten::tensordot(Tensor self, Tensor other, int[] dims_self, int[] dims_other) -> Tensor
aten::tensordot.out  aten::tensordot.out(Tensor self, Tensor other, int[] dims_self, int[] dims_other, *, Tensor(a!) out) -> Tensor(a!)
aten::thnn_conv2d.default  aten::thnn_conv2d(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0]) -> Tensor
aten::thnn_conv2d.out  aten::thnn_conv2d.out(Tensor self, Tensor weight, SymInt[2] kernel_size, Tensor? bias=None, SymInt[2] stride=[1, 1], SymInt[2] padding=[0, 0], *, Tensor(a!) out) -> Tensor(a!)
aten::threshold.default  aten::threshold(Tensor self, Scalar threshold, Scalar value) -> Tensor
aten::threshold.out  aten::threshold.out(Tensor self, Scalar threshold, Scalar value, *, Tensor(a!) out) -> Tensor(a!)
aten::threshold_.default  aten::threshold_(Tensor(a!) self, Scalar threshold, Scalar value) -> Tensor(a!)
aten::threshold_backward.default  aten::threshold_backward(Tensor grad_output, Tensor self, Scalar threshold) -> Tensor
aten::threshold_backward.grad_input  aten::threshold_backward.grad_input(Tensor grad_output, Tensor self, Scalar threshold, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::tile.default  aten::tile(Tensor self, SymInt[] dims) -> Tensor
aten::to.device  aten::to.device(Tensor(a) self, Device device, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
aten::to.dtype  aten::to.dtype(Tensor(a) self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
aten::to.dtype_layout  aten::to.dtype_layout(Tensor(a) self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
aten::to.other  aten::to.other(Tensor(a) self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor(a)
aten::to_dense.default  aten::to_dense(Tensor self, ScalarType? dtype=None, *, bool? masked_grad=None) -> Tensor
aten::to_dense_backward.default  aten::to_dense_backward(Tensor grad, Tensor input, bool? masked_grad=None) -> Tensor
aten::to_mkldnn.default  aten::to_mkldnn(Tensor self, ScalarType? dtype=None) -> Tensor
aten::to_mkldnn.out  aten::to_mkldnn.out(Tensor self, ScalarType? dtype=None, *, Tensor(a!) out) -> Tensor(a!)
aten::to_mkldnn_backward.default  aten::to_mkldnn_backward(Tensor grad, Tensor input) -> Tensor
aten::to_padded_tensor.default  aten::to_padded_tensor(Tensor self, float padding, SymInt[]? output_size=None) -> Tensor
aten::to_padded_tensor.out  aten::to_padded_tensor.out(Tensor self, float padding, SymInt[]? output_size=None, *, Tensor(a!) out) -> Tensor(a!)
aten::to_sparse.default  aten::to_sparse(Tensor self, *, Layout? layout=None, int[2]? blocksize=None, int? dense_dim=None) -> Tensor
aten::to_sparse.sparse_dim  aten::to_sparse.sparse_dim(Tensor self, int sparse_dim) -> Tensor
aten::to_sparse_bsc.default  aten::to_sparse_bsc(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
aten::to_sparse_bsr.default  aten::to_sparse_bsr(Tensor self, int[2] blocksize, int? dense_dim=None) -> Tensor
aten::to_sparse_csc.default  aten::to_sparse_csc(Tensor self, int? dense_dim=None) -> Tensor
aten::to_sparse_csr.default  aten::to_sparse_csr(Tensor self, int? dense_dim=None) -> Tensor
aten::topk.default  aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True) -> (Tensor values, Tensor indices)
aten::topk.values  aten::topk.values(Tensor self, SymInt k, int dim=-1, bool largest=True, bool sorted=True, *, Tensor(a!) values, Tensor(b!) indices) -> (Tensor(a!) values, Tensor(b!) indices)
aten::trace.default  aten::trace(Tensor self) -> Tensor
aten::trace.out  aten::trace.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::trace_backward.default  aten::trace_backward(Tensor grad, SymInt[] sizes) -> Tensor
aten::transpose.Dimname  aten::transpose.Dimname(Tensor(a) self, str dim0, str dim1) -> Tensor(a)
aten::transpose.int  aten::transpose.int(Tensor(a) self, int dim0, int dim1) -> Tensor(a)
aten::transpose_.default  aten::transpose_(Tensor(a!) self, int dim0, int dim1) -> Tensor(a!)
aten::transpose_copy.int  aten::transpose_copy.int(Tensor self, int dim0, int dim1) -> Tensor
aten::transpose_copy.int_out  aten::transpose_copy.int_out(Tensor self, int dim0, int dim1, *, Tensor(a!) out) -> Tensor(a!)
aten::trapezoid.dx  aten::trapezoid.dx(Tensor y, *, Scalar dx=1, int dim=-1) -> Tensor
aten::trapezoid.x  aten::trapezoid.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
aten::trapz.dx  aten::trapz.dx(Tensor y, *, float dx=1., int dim=-1) -> Tensor
aten::trapz.x  aten::trapz.x(Tensor y, Tensor x, *, int dim=-1) -> Tensor
aten::triangular_solve.X  aten::triangular_solve.X(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False, *, Tensor(a!) X, Tensor(b!) M) -> (Tensor(a!) solution, Tensor(b!) cloned_coefficient)
aten::triangular_solve.default  aten::triangular_solve(Tensor self, Tensor A, bool upper=True, bool transpose=False, bool unitriangular=False) -> (Tensor solution, Tensor cloned_coefficient)
aten::tril.default  aten::tril(Tensor self, int diagonal=0) -> Tensor
aten::tril.out  aten::tril.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
aten::tril_.default  aten::tril_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
aten::tril_indices.default  aten::tril_indices(int row, int col, int offset=0, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::tril_indices.out  aten::tril_indices.out(int row, int col, int offset=0, *, Tensor(a!) out) -> Tensor(a!)
aten::triplet_margin_loss.default  aten::triplet_margin_loss(Tensor anchor, Tensor positive, Tensor negative, float margin=1., float p=2., float eps=9.9999999999999995e-07, bool swap=False, int reduction=1) -> Tensor
aten::triu.default  aten::triu(Tensor self, int diagonal=0) -> Tensor
aten::triu.out  aten::triu.out(Tensor self, int diagonal=0, *, Tensor(a!) out) -> Tensor(a!)
aten::triu_.default  aten::triu_(Tensor(a!) self, int diagonal=0) -> Tensor(a!)
aten::triu_indices.default  aten::triu_indices(int row, int col, int offset=0, *, ScalarType? dtype=4, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::triu_indices.out  aten::triu_indices.out(int row, int col, int offset=0, *, Tensor(a!) out) -> Tensor(a!)
aten::true_divide.Scalar  aten::true_divide.Scalar(Tensor self, Scalar other) -> Tensor
aten::true_divide.Tensor  aten::true_divide.Tensor(Tensor self, Tensor other) -> Tensor
aten::true_divide.out  aten::true_divide.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::true_divide_.Scalar  aten::true_divide_.Scalar(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::true_divide_.Tensor  aten::true_divide_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::trunc.default  aten::trunc(Tensor self) -> Tensor
aten::trunc.out  aten::trunc.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::trunc_.default  aten::trunc_(Tensor(a!) self) -> Tensor(a!)
aten::type_as.default  aten::type_as(Tensor self, Tensor other) -> Tensor
aten::unbind.Dimname  aten::unbind.Dimname(Tensor(a -> *) self, str dim) -> Tensor(a)[]
aten::unbind.int  aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]
aten::unbind_copy.int  aten::unbind_copy.int(Tensor self, int dim=0) -> Tensor[]
aten::unbind_copy.int_out  aten::unbind_copy.int_out(Tensor self, int dim=0, *, Tensor(a!)[] out) -> ()
aten::unflatten.Dimname  aten::unflatten.Dimname(Tensor(a) self, str dim, SymInt[] sizes, str[] names) -> Tensor(a)
aten::unflatten.int  aten::unflatten.int(Tensor(a) self, int dim, SymInt[] sizes) -> Tensor(a)
aten::unflatten_dense_tensors.default  aten::unflatten_dense_tensors(Tensor flat, Tensor[] tensors) -> Tensor[]
aten::unfold.default  aten::unfold(Tensor(a) self, int dimension, int size, int step) -> Tensor(a)
aten::unfold_backward.default  aten::unfold_backward(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step) -> Tensor
aten::unfold_backward.out  aten::unfold_backward.out(Tensor grad_in, SymInt[] input_sizes, int dim, int size, int step, *, Tensor(a!) out) -> Tensor(a!)
aten::unfold_copy.default  aten::unfold_copy(Tensor self, int dimension, int size, int step) -> Tensor
aten::unfold_copy.out  aten::unfold_copy.out(Tensor self, int dimension, int size, int step, *, Tensor(a!) out) -> Tensor(a!)
aten::uniform.default  aten::uniform(Tensor self, float from=0., float to=1., *, Generator? generator=None) -> Tensor
aten::uniform.out  aten::uniform.out(Tensor self, float from=0., float to=1., *, Generator? generator=None, Tensor(a!) out) -> Tensor(a!)
aten::uniform_.default  aten::uniform_(Tensor(a!) self, float from=0., float to=1., *, Generator? generator=None) -> Tensor(a!)
aten::unique_consecutive.default  aten::unique_consecutive(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None) -> (Tensor, Tensor, Tensor)
aten::unique_consecutive.out  aten::unique_consecutive.out(Tensor self, bool return_inverse=False, bool return_counts=False, int? dim=None, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::unique_dim.default  aten::unique_dim(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
aten::unique_dim.out  aten::unique_dim.out(Tensor self, int dim, bool sorted=True, bool return_inverse=False, bool return_counts=False, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::unique_dim_consecutive.default  aten::unique_dim_consecutive(Tensor self, int dim, bool return_inverse=False, bool return_counts=False) -> (Tensor, Tensor, Tensor)
aten::unique_dim_consecutive.out  aten::unique_dim_consecutive.out(Tensor self, int dim, bool return_inverse=False, bool return_counts=False, *, Tensor(a!) out0, Tensor(b!) out1, Tensor(c!) out2) -> (Tensor(a!), Tensor(b!), Tensor(c!))
aten::unsafe_chunk.default  aten::unsafe_chunk(Tensor self, int chunks, int dim=0) -> Tensor[]
aten::unsafe_split.Tensor  aten::unsafe_split.Tensor(Tensor self, SymInt split_size, int dim=0) -> Tensor[]
aten::unsafe_split.Tensor_out  aten::unsafe_split.Tensor_out(Tensor self, SymInt split_size, int dim=0, *, Tensor(a!)[] out) -> ()
aten::unsafe_split_with_sizes.default  aten::unsafe_split_with_sizes(Tensor self, SymInt[] split_sizes, int dim=0) -> Tensor[]
aten::unsafe_split_with_sizes.out  aten::unsafe_split_with_sizes.out(Tensor self, SymInt[] split_sizes, int dim=0, *, Tensor(a!)[] out) -> ()
aten::unsqueeze.default  aten::unsqueeze(Tensor(a) self, int dim) -> Tensor(a)
aten::unsqueeze_.default  aten::unsqueeze_(Tensor(a!) self, int dim) -> Tensor(a!)
aten::unsqueeze_copy.default  aten::unsqueeze_copy(Tensor self, int dim) -> Tensor
aten::unsqueeze_copy.out  aten::unsqueeze_copy.out(Tensor self, int dim, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_bicubic2d.default  aten::upsample_bicubic2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_bicubic2d.out  aten::upsample_bicubic2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_bicubic2d.vec  aten::upsample_bicubic2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::upsample_bicubic2d_backward.default  aten::upsample_bicubic2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_bicubic2d_backward.grad_input  aten::upsample_bicubic2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_bilinear2d.default  aten::upsample_bilinear2d(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_bilinear2d.out  aten::upsample_bilinear2d.out(Tensor self, SymInt[2] output_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_bilinear2d.vec  aten::upsample_bilinear2d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::upsample_bilinear2d_backward.default  aten::upsample_bilinear2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_bilinear2d_backward.grad_input  aten::upsample_bilinear2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, bool align_corners, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_linear1d.default  aten::upsample_linear1d(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None) -> Tensor
aten::upsample_linear1d.out  aten::upsample_linear1d.out(Tensor self, SymInt[1] output_size, bool align_corners, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_linear1d.vec  aten::upsample_linear1d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::upsample_linear1d_backward.default  aten::upsample_linear1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, bool align_corners, float? scales=None) -> Tensor
aten::upsample_linear1d_backward.grad_input  aten::upsample_linear1d_backward.grad_input(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, bool align_corners, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_nearest1d.default  aten::upsample_nearest1d(Tensor self, SymInt[1] output_size, float? scales=None) -> Tensor
aten::upsample_nearest1d.out  aten::upsample_nearest1d.out(Tensor self, SymInt[1] output_size, float? scales=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_nearest1d.vec  aten::upsample_nearest1d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::upsample_nearest1d_backward.default  aten::upsample_nearest1d_backward(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None) -> Tensor
aten::upsample_nearest1d_backward.grad_input  aten::upsample_nearest1d_backward.grad_input(Tensor grad_output, SymInt[1] output_size, SymInt[3] input_size, float? scales=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_nearest2d.default  aten::upsample_nearest2d(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_nearest2d.out  aten::upsample_nearest2d.out(Tensor self, SymInt[2] output_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_nearest2d.vec  aten::upsample_nearest2d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::upsample_nearest2d_backward.default  aten::upsample_nearest2d_backward(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_nearest2d_backward.grad_input  aten::upsample_nearest2d_backward.grad_input(Tensor grad_output, SymInt[2] output_size, SymInt[4] input_size, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_nearest3d.default  aten::upsample_nearest3d(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_nearest3d.out  aten::upsample_nearest3d.out(Tensor self, SymInt[3] output_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_nearest3d.vec  aten::upsample_nearest3d.vec(Tensor input, SymInt[]? output_size, float[]? scale_factors) -> Tensor
aten::upsample_nearest3d_backward.default  aten::upsample_nearest3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_nearest3d_backward.grad_input  aten::upsample_nearest3d_backward.grad_input(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::upsample_trilinear3d.default  aten::upsample_trilinear3d(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_trilinear3d.out  aten::upsample_trilinear3d.out(Tensor self, SymInt[3] output_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) out) -> Tensor(a!)
aten::upsample_trilinear3d.vec  aten::upsample_trilinear3d.vec(Tensor input, SymInt[]? output_size, bool align_corners, float[]? scale_factors) -> Tensor
aten::upsample_trilinear3d_backward.default  aten::upsample_trilinear3d_backward(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None) -> Tensor
aten::upsample_trilinear3d_backward.grad_input  aten::upsample_trilinear3d_backward.grad_input(Tensor grad_output, SymInt[3] output_size, SymInt[5] input_size, bool align_corners, float? scales_d=None, float? scales_h=None, float? scales_w=None, *, Tensor(a!) grad_input) -> Tensor(a!)
aten::value_selecting_reduction_backward.default  aten::value_selecting_reduction_backward(Tensor grad, int dim, Tensor indices, SymInt[] sizes, bool keepdim) -> Tensor
aten::values.default  aten::values(Tensor(a) self) -> Tensor(a)
aten::values_copy.default  aten::values_copy(Tensor self) -> Tensor
aten::values_copy.out  aten::values_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::vander.default  aten::vander(Tensor x, int? N=None, bool increasing=False) -> Tensor
aten::var.correction  aten::var.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> Tensor
aten::var.correction_names  aten::var.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> Tensor
aten::var.correction_names_out  aten::var.correction_names_out(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
aten::var.correction_out  aten::var.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out) -> Tensor(a!)
aten::var.default  aten::var(Tensor self, bool unbiased=True) -> Tensor
aten::var.dim  aten::var.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> Tensor
aten::var.names_dim  aten::var.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> Tensor
aten::var.names_out  aten::var.names_out(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::var.out  aten::var.out(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False, *, Tensor(a!) out) -> Tensor(a!)
aten::var_mean.correction  aten::var_mean.correction(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
aten::var_mean.correction_names  aten::var_mean.correction_names(Tensor self, str[1] dim, *, Scalar? correction=None, bool keepdim=False) -> (Tensor, Tensor)
aten::var_mean.correction_out  aten::var_mean.correction_out(Tensor self, int[1]? dim=None, *, Scalar? correction=None, bool keepdim=False, Tensor(a!) out0, Tensor(b!) out1) -> (Tensor(a!), Tensor(b!))
aten::var_mean.default  aten::var_mean(Tensor self, bool unbiased=True) -> (Tensor, Tensor)
aten::var_mean.dim  aten::var_mean.dim(Tensor self, int[1]? dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
aten::var_mean.names_dim  aten::var_mean.names_dim(Tensor self, str[1] dim, bool unbiased=True, bool keepdim=False) -> (Tensor, Tensor)
aten::vdot.default  aten::vdot(Tensor self, Tensor other) -> Tensor
aten::vdot.out  aten::vdot.out(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::view.default  aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)
aten::view.dtype  aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)
aten::view_as.default  aten::view_as(Tensor(a) self, Tensor other) -> Tensor(a)
aten::view_as_complex.default  aten::view_as_complex(Tensor(a) self) -> Tensor(a)
aten::view_as_complex_copy.default  aten::view_as_complex_copy(Tensor self) -> Tensor
aten::view_as_complex_copy.out  aten::view_as_complex_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::view_as_real.default  aten::view_as_real(Tensor(a) self) -> Tensor(a)
aten::view_as_real_copy.default  aten::view_as_real_copy(Tensor self) -> Tensor
aten::view_as_real_copy.out  aten::view_as_real_copy.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::view_copy.default  aten::view_copy(Tensor self, SymInt[] size) -> Tensor
aten::view_copy.dtype  aten::view_copy.dtype(Tensor self, ScalarType dtype) -> Tensor
aten::view_copy.dtype_out  aten::view_copy.dtype_out(Tensor self, ScalarType dtype, *, Tensor(a!) out) -> Tensor(a!)
aten::view_copy.out  aten::view_copy.out(Tensor self, SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::vsplit.array  aten::vsplit.array(Tensor(a -> *) self, int[] indices) -> Tensor(a)[]
aten::vsplit.int  aten::vsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
aten::vstack.default  aten::vstack(Tensor[] tensors) -> Tensor
aten::vstack.out  aten::vstack.out(Tensor[] tensors, *, Tensor(a!) out) -> Tensor(a!)
aten::where.Scalar  aten::where.Scalar(Tensor condition, Scalar self, Scalar other) -> Tensor
aten::where.ScalarOther  aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor
aten::where.ScalarSelf  aten::where.ScalarSelf(Tensor condition, Scalar self, Tensor other) -> Tensor
aten::where.default  aten::where(Tensor condition) -> Tensor[]
aten::where.self  aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor
aten::where.self_out  aten::where.self_out(Tensor condition, Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::xlogy.OutScalar_Other  aten::xlogy.OutScalar_Other(Tensor self, Scalar other, *, Tensor(a!) out) -> Tensor(a!)
aten::xlogy.OutScalar_Self  aten::xlogy.OutScalar_Self(Scalar self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::xlogy.OutTensor  aten::xlogy.OutTensor(Tensor self, Tensor other, *, Tensor(a!) out) -> Tensor(a!)
aten::xlogy.Scalar_Other  aten::xlogy.Scalar_Other(Tensor self, Scalar other) -> Tensor
aten::xlogy.Scalar_Self  aten::xlogy.Scalar_Self(Scalar self, Tensor other) -> Tensor
aten::xlogy.Tensor  aten::xlogy.Tensor(Tensor self, Tensor other) -> Tensor
aten::xlogy_.Scalar_Other  aten::xlogy_.Scalar_Other(Tensor(a!) self, Scalar other) -> Tensor(a!)
aten::xlogy_.Tensor  aten::xlogy_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)
aten::zero.default  aten::zero(Tensor self) -> Tensor
aten::zero.out  aten::zero.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)
aten::zero_.default  aten::zero_(Tensor(a!) self) -> Tensor(a!)
aten::zeros.default  aten::zeros(SymInt[] size, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::zeros.names  aten::zeros.names(int[] size, *, str[]? names, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None) -> Tensor
aten::zeros.names_out  aten::zeros.names_out(int[] size, *, str[]? names, Tensor(a!) out) -> Tensor(a!)
aten::zeros.out  aten::zeros.out(SymInt[] size, *, Tensor(a!) out) -> Tensor(a!)
aten::zeros_like.default  aten::zeros_like(Tensor self, *, ScalarType? dtype=None, Layout? layout=None, Device? device=None, bool? pin_memory=None, MemoryFormat? memory_format=None) -> Tensor
aten::zeros_like.out  aten::zeros_like.out(Tensor self, *, MemoryFormat? memory_format=None, Tensor(a!) out) -> Tensor(a!)
```

</details>

<a id="c10d-namespace"></a>

## `c10d` namespace (20 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `c10d::_allgather_base_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::_reduce_scatter_base_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::allgather_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::allgather_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::allgather_into_tensor_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::allreduce_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::allreduce_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::alltoall_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::alltoall_base_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::barrier.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::broadcast_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::gather_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::monitored_barrier_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::recv_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::recv_any_source_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::reduce_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::reduce_scatter_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::reduce_scatter_tensor_coalesced_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::scatter_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `c10d::send.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>c10d</code> (20)</summary>

```text
c10d::_allgather_base_.default  c10d::_allgather_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, bool asyncOp, int timeout) -> (Tensor, __torch__.torch.classes.c10d.Work)
c10d::_reduce_scatter_base_.default  c10d::_reduce_scatter_base_(Tensor output_tensor, Tensor input_tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, bool asyncOp, int timeout) -> (Tensor, __torch__.torch.classes.c10d.Work)
c10d::allgather_.default  c10d::allgather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> (Tensor[][], __torch__.torch.classes.c10d.Work)
c10d::allgather_coalesced_.default  c10d::allgather_coalesced_(Tensor[][] output_lists, Tensor[] input_list, __torch__.torch.classes.c10d.ProcessGroup process_group) -> __torch__.torch.classes.c10d.Work
c10d::allgather_into_tensor_coalesced_.default  c10d::allgather_into_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, __torch__.torch.classes.c10d.ProcessGroup process_group) -> __torch__.torch.classes.c10d.Work
c10d::allreduce_.default  c10d::allreduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, Tensor? sparse_indices, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)
c10d::allreduce_coalesced_.default  c10d::allreduce_coalesced_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::alltoall_.default  c10d::alltoall_(Tensor[] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)
c10d::alltoall_base_.default  c10d::alltoall_base_(Tensor output, Tensor input, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] output_split_sizes, int[] input_split_sizes, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::barrier.default  c10d::barrier(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::broadcast_.default  c10d::broadcast_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int root_tensor, bool asyncOp, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)
c10d::gather_.default  c10d::gather_(Tensor[][] output_tensors, Tensor[] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::monitored_barrier_.default  c10d::monitored_barrier_(Tensor tensor, __torch__.torch.classes.c10d.ProcessGroup process_group, int[] device_ids, int timeout, bool wait_all_ranks) -> ()
c10d::recv_.default  c10d::recv_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int src, int tag) -> __torch__.torch.classes.c10d.Work
c10d::recv_any_source_.default  c10d::recv_any_source_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int tag) -> __torch__.torch.classes.c10d.Work
c10d::reduce_.default  c10d::reduce_(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int root_rank, int root_tensor, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::reduce_scatter_.default  c10d::reduce_scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)
c10d::reduce_scatter_tensor_coalesced_.default  c10d::reduce_scatter_tensor_coalesced_(Tensor[] outputs, Tensor[] inputs, __torch__.torch.classes.c10d.ProcessGroup process_group, __torch__.torch.classes.c10d.ReduceOp reduce_op, int timeout) -> __torch__.torch.classes.c10d.Work
c10d::scatter_.default  c10d::scatter_(Tensor[] output_tensors, Tensor[][] input_tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int root_rank, bool asyncOp, int timeout) -> (Tensor[], __torch__.torch.classes.c10d.Work)
c10d::send.default  c10d::send(Tensor[] tensors, __torch__.torch.classes.c10d.ProcessGroup process_group, int dst, int tag) -> __torch__.torch.classes.c10d.Work
```

</details>

<a id="debugprims-namespace"></a>

## `debugprims` namespace (1 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `debugprims::load_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>debugprims</code> (1)</summary>

```text
debugprims::load_tensor.default  debugprims::load_tensor(str name, int[] size, int[] stride, *, ScalarType dtype, Device device) -> Tensor
```

</details>

<a id="inductor-namespace"></a>

## `inductor` namespace (5 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `inductor::_alloc_from_pool.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `inductor::_mm_plus_mm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `inductor::_reinterpret_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `inductor::accumulate_grad_.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `inductor::resize_storage_bytes_.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>inductor</code> (5)</summary>

```text
inductor::_alloc_from_pool.default  inductor::_alloc_from_pool(Tensor self, int offset_bytes, ScalarType dtype, int[] size, int[] stride) -> Tensor
inductor::_mm_plus_mm.default  inductor::_mm_plus_mm(Tensor a, Tensor b, Tensor c, Tensor d, Tensor(t!) out) -> Tensor(t!)
inductor::_reinterpret_tensor.default  inductor::_reinterpret_tensor(Tensor self, int[] size, int[] stride, int offset_increment=0) -> Tensor
inductor::accumulate_grad_.default  inductor::accumulate_grad_(Tensor variable, Tensor new_grad) -> ()
inductor::resize_storage_bytes_.default  inductor::resize_storage_bytes_(Tensor variable, SymInt new_size) -> ()
```

</details>

<a id="mkl-namespace"></a>

## `mkl` namespace (2 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `mkl::_mkl_linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkl::_mkl_reorder_linear_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>mkl</code> (2)</summary>

```text
mkl::_mkl_linear.default  mkl::_mkl_linear(Tensor X, Tensor MKL_W, Tensor ORI_W, Tensor? B, int batch_size) -> Tensor
mkl::_mkl_reorder_linear_weight.default  mkl::_mkl_reorder_linear_weight(Tensor X, int batch_size) -> Tensor
```

</details>

<a id="mkldnn-namespace"></a>

## `mkldnn` namespace (16 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `mkldnn::_convolution_pointwise.binary` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_convolution_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_convolution_pointwise_.binary` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_convolution_transpose_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_get_mkldnn_serialized_md.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_is_mkldnn_acl_supported.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_is_mkldnn_bf16_supported.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_is_mkldnn_fp16_supported.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_linear_pointwise.binary` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_linear_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_nbytes.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_reorder_convolution_transpose_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_reorder_convolution_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_reorder_linear_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::_reorder_mkldnn_rnn_layer_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn::data_ptr.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>mkldnn</code> (16)</summary>

```text
mkldnn::_convolution_pointwise.binary  mkldnn::_convolution_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor Y
mkldnn::_convolution_pointwise.default  mkldnn::_convolution_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y
mkldnn::_convolution_pointwise_.binary  mkldnn::_convolution_pointwise_.binary(Tensor(a!) other, Tensor X, Tensor W, Tensor? B, int[] padding, int[] stride, int[] dilation, int groups, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor(a!) Y
mkldnn::_convolution_transpose_pointwise.default  mkldnn::_convolution_transpose_pointwise(Tensor X, Tensor W, Tensor? B, int[] padding, int[] output_padding, int[] stride, int[] dilation, int groups, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y
mkldnn::_get_mkldnn_serialized_md.default  mkldnn::_get_mkldnn_serialized_md(Tensor mkldnn_tensor) -> Tensor
mkldnn::_is_mkldnn_acl_supported.default  mkldnn::_is_mkldnn_acl_supported() -> bool _0
mkldnn::_is_mkldnn_bf16_supported.default  mkldnn::_is_mkldnn_bf16_supported() -> bool _0
mkldnn::_is_mkldnn_fp16_supported.default  mkldnn::_is_mkldnn_fp16_supported() -> bool _0
mkldnn::_linear_pointwise.binary  mkldnn::_linear_pointwise.binary(Tensor X, Tensor other, Tensor W, Tensor? B, str attr) -> Tensor Y
mkldnn::_linear_pointwise.default  mkldnn::_linear_pointwise(Tensor X, Tensor W, Tensor? B, str attr, Scalar?[] scalars, str? algorithm) -> Tensor Y
mkldnn::_nbytes.default  mkldnn::_nbytes(Tensor mkldnn_tensor) -> int
mkldnn::_reorder_convolution_transpose_weight.default  mkldnn::_reorder_convolution_transpose_weight(Tensor self, int[2] padding=0, int[2] output_padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None) -> Tensor Y
mkldnn::_reorder_convolution_weight.default  mkldnn::_reorder_convolution_weight(Tensor self, int[2] padding=0, int[2] stride=1, int[2] dilation=1, int groups=1, int[]? input_size=None) -> Tensor Y
mkldnn::_reorder_linear_weight.default  mkldnn::_reorder_linear_weight(Tensor self, int? batch_size=None) -> Tensor Y
mkldnn::_reorder_mkldnn_rnn_layer_weight.default  mkldnn::_reorder_mkldnn_rnn_layer_weight(Tensor weight0, Tensor weight1, int hidden_size, bool reverse, bool has_biases, bool batch_first, int[]? input_size=None) -> Tensor[] Y
mkldnn::data_ptr.default  mkldnn::data_ptr(Tensor mkldnn_tensor) -> int
```

</details>

<a id="mkldnnprepacked-namespace"></a>

## `mkldnn_prepacked` namespace (2 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `mkldnn_prepacked::conv2d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `mkldnn_prepacked::conv2d_run.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>mkldnn_prepacked</code> (2)</summary>

```text
mkldnn_prepacked::conv2d_prepack.default  mkldnn_prepacked::conv2d_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, int[4] input_size, str attr) -> __torch__.torch.classes.mkldnn.ConvOpContext
mkldnn_prepacked::conv2d_run.default  mkldnn_prepacked::conv2d_run(Tensor X, __torch__.torch.classes.mkldnn.ConvOpContext W_prepack) -> Tensor Y
```

</details>

<a id="onednn-namespace"></a>

## `onednn` namespace (10 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `onednn::qconv1d_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qconv2d_pointwise.binary` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qconv2d_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qconv3d_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qconv_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qlinear_pointwise.binary` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qlinear_pointwise.binary_tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qlinear_pointwise.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qlinear_pointwise.tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `onednn::qlinear_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>onednn</code> (10)</summary>

```text
onednn::qconv1d_pointwise.default  onednn::qconv1d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor
onednn::qconv2d_pointwise.binary  onednn::qconv2d_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qaccum, float accum_scale, int accum_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str binary_attr, Scalar? alpha, str? unary_attr, Scalar?[] unary_scalars, str? unary_algorithm) -> Tensor
onednn::qconv2d_pointwise.default  onednn::qconv2d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor
onednn::qconv3d_pointwise.default  onednn::qconv3d_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point, ScalarType? output_dtype, str attr, Scalar?[] scalars, str? algorithm) -> Tensor
onednn::qconv_prepack.default  onednn::qconv_prepack(Tensor weight, Tensor w_scales, float x_scale, int x_zp, int[] stride, int[] padding, int[] dilation, int groups, int[]? x_shape=None) -> Tensor
onednn::qlinear_pointwise.binary  onednn::qlinear_pointwise.binary(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, Tensor? other, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor
onednn::qlinear_pointwise.binary_tensor  onednn::qlinear_pointwise.binary_tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, Tensor? other, float other_scale, int other_zp, str binary_post_op, float binary_alpha, str unary_post_op, Scalar?[] unary_post_op_args, str unary_post_op_algorithm) -> Tensor
onednn::qlinear_pointwise.default  onednn::qlinear_pointwise(Tensor qx, float x_scale, int x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor
onednn::qlinear_pointwise.tensor  onednn::qlinear_pointwise.tensor(Tensor qx, Tensor x_scale, Tensor x_zero_point, Tensor qw, Tensor w_scale, Tensor w_zero_point, Tensor? bias, float output_scale, int output_zero_point, ScalarType? output_dtype, str post_op_name, Scalar?[] post_op_args, str post_op_algorithm) -> Tensor
onednn::qlinear_prepack.default  onednn::qlinear_prepack(Tensor weight, int[]? x_shape) -> Tensor
```

</details>

<a id="prepacked-namespace"></a>

## `prepacked` namespace (8 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `prepacked::conv2d_clamp_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::conv2d_clamp_run.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::conv2d_transpose_clamp_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::conv2d_transpose_clamp_run.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::linear_clamp_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::linear_clamp_run.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::unpack_prepacked_sizes_conv2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prepacked::unpack_prepacked_sizes_linear.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>prepacked</code> (8)</summary>

```text
prepacked::conv2d_clamp_prepack.default  prepacked::conv2d_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.Conv2dOpContext
prepacked::conv2d_clamp_run.default  prepacked::conv2d_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.Conv2dOpContext W_prepack) -> Tensor Y
prepacked::conv2d_transpose_clamp_prepack.default  prepacked::conv2d_transpose_clamp_prepack(Tensor W, Tensor? B, int[2] stride, int[2] padding, int[2] output_padding, int[2] dilation, int groups, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.TransposeConv2dOpContext
prepacked::conv2d_transpose_clamp_run.default  prepacked::conv2d_transpose_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.TransposeConv2dOpContext W_prepack) -> Tensor Y
prepacked::linear_clamp_prepack.default  prepacked::linear_clamp_prepack(Tensor W, Tensor? B=None, Scalar? output_min=None, Scalar? output_max=None) -> __torch__.torch.classes.xnnpack.LinearOpContext
prepacked::linear_clamp_run.default  prepacked::linear_clamp_run(Tensor X, __torch__.torch.classes.xnnpack.LinearOpContext W_prepack) -> Tensor Y
prepacked::unpack_prepacked_sizes_conv2d.default  prepacked::unpack_prepacked_sizes_conv2d(Any W_prepack) -> Any
prepacked::unpack_prepacked_sizes_linear.default  prepacked::unpack_prepacked_sizes_linear(Any W_prepack) -> Any
```

</details>

<a id="prims-namespace"></a>

## `prims` namespace (128 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `prims::_make_token.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::_sink_tokens.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::abs.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::acos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::acosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::amax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::amin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::as_strided.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::as_strided_scatter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::asin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::asinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::atan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::atan2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::atanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_i0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_i0e.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_i1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_i1e.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_j0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bessel_j1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bitwise_and.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bitwise_not.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bitwise_or.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::bitwise_xor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::broadcast_in_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::cat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::cbrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::ceil.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::clone.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::collapse.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::collapse_view.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::conj.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::conj_physical.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::convert_element_type.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::copy_strided.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::copy_to.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::cos.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::cosh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::device_put.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::digamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::div.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::empty.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::empty_permuted.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::empty_strided.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::eq.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::erf.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::erf_inv.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::erfc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::erfcx.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::exp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::exp2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::expm1.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fft_c2c.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fft_c2r.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fft_r2c.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fill.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::floor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fmin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::fmod.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::frexp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::full.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::full_like.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::gcd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::ge.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::gt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::hypot.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::igamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::igammac.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::imag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::iota.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::isfinite.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::item.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::le.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::lgamma.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::log.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::log10.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::log1p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::log2.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::lt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::maximum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::maximum_value.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::minimum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::minimum_value.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::mul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::ndtri.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::ne.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::neg.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::nextafter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::normal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::pow.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::prod.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::real.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::reciprocal.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::remainder.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::reshape.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::resize.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::rev.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::round.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::rsqrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::scalar_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::shift_left.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::shift_right_arithmetic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sign.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::signbit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sin.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sinh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::slice.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::slice_in_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::spherical_bessel_j0.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::split_dim.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sqrt.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::squeeze.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sub.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::sum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::svd.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::tan.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::tanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::trunc.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::uniform.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::var.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::view_of.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::view_of_dtype.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::where.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::xor_sum.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `prims::zeta.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>prims</code> (128)</summary>

```text
prims::_make_token.default  prims::_make_token() -> Tensor
prims::_sink_tokens.default  prims::_sink_tokens(Tensor[] tokens) -> ()
prims::abs.default  prims::abs(Tensor self) -> Tensor
prims::acos.default  prims::acos(Tensor self) -> Tensor
prims::acosh.default  prims::acosh(Tensor self) -> Tensor
prims::add.default  prims::add(Tensor self, Tensor other) -> Tensor
prims::amax.default  prims::amax(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor
prims::amin.default  prims::amin(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor
prims::as_strided.default  prims::as_strided(Tensor(a!) a, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor(a!)
prims::as_strided_scatter.default  prims::as_strided_scatter(Tensor self, Tensor src, SymInt[] size, SymInt[] stride, SymInt storage_offset) -> Tensor
prims::asin.default  prims::asin(Tensor self) -> Tensor
prims::asinh.default  prims::asinh(Tensor self) -> Tensor
prims::atan.default  prims::atan(Tensor self) -> Tensor
prims::atan2.default  prims::atan2(Tensor self, Tensor other) -> Tensor
prims::atanh.default  prims::atanh(Tensor self) -> Tensor
prims::bessel_i0.default  prims::bessel_i0(Tensor self) -> Tensor
prims::bessel_i0e.default  prims::bessel_i0e(Tensor self) -> Tensor
prims::bessel_i1.default  prims::bessel_i1(Tensor self) -> Tensor
prims::bessel_i1e.default  prims::bessel_i1e(Tensor self) -> Tensor
prims::bessel_j0.default  prims::bessel_j0(Tensor self) -> Tensor
prims::bessel_j1.default  prims::bessel_j1(Tensor self) -> Tensor
prims::bitwise_and.default  prims::bitwise_and(Tensor self, Tensor other) -> Tensor
prims::bitwise_not.default  prims::bitwise_not(Tensor self) -> Tensor
prims::bitwise_or.default  prims::bitwise_or(Tensor self, Tensor other) -> Tensor
prims::bitwise_xor.default  prims::bitwise_xor(Tensor self, Tensor other) -> Tensor
prims::broadcast_in_dim.default  prims::broadcast_in_dim(Tensor(a) a, SymInt[] shape, int[] broadcast_dimensions) -> Tensor(a)
prims::cat.default  prims::cat(Tensor[] tensors, int dim) -> Tensor
prims::cbrt.default  prims::cbrt(Tensor self) -> Tensor
prims::ceil.default  prims::ceil(Tensor self) -> Tensor
prims::clone.default  prims::clone(Tensor self, *, MemoryFormat? memory_format=None) -> Tensor
prims::collapse.default  prims::collapse(Tensor a, int start, int end) -> Tensor
prims::collapse_view.default  prims::collapse_view(Tensor(a) a, int start, int end) -> Tensor(a)
prims::conj.default  prims::conj(Tensor(a) a) -> Tensor(a)
prims::conj_physical.default  prims::conj_physical(Tensor self) -> Tensor
prims::convert_element_type.default  prims::convert_element_type(Tensor a, ScalarType dtype) -> Tensor
prims::copy_strided.default  prims::copy_strided(Tensor a, SymInt[] stride) -> Tensor
prims::copy_to.default  prims::copy_to(Tensor(a!) a, Tensor b) -> Tensor(a!)
prims::cos.default  prims::cos(Tensor self) -> Tensor
prims::cosh.default  prims::cosh(Tensor self) -> Tensor
prims::device_put.default  prims::device_put(Tensor a, Device device) -> Tensor
prims::digamma.default  prims::digamma(Tensor self) -> Tensor
prims::div.default  prims::div(Tensor self, Tensor other) -> Tensor
prims::empty.default  prims::empty(SymInt[] shape, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::empty_permuted.default  prims::empty_permuted(SymInt[] shape, int[] physical_layout, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::empty_strided.default  prims::empty_strided(SymInt[] shape, SymInt[] strides, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::eq.default  prims::eq(Tensor self, Tensor other) -> Tensor
prims::erf.default  prims::erf(Tensor self) -> Tensor
prims::erf_inv.default  prims::erf_inv(Tensor self) -> Tensor
prims::erfc.default  prims::erfc(Tensor self) -> Tensor
prims::erfcx.default  prims::erfcx(Tensor self) -> Tensor
prims::exp.default  prims::exp(Tensor self) -> Tensor
prims::exp2.default  prims::exp2(Tensor self) -> Tensor
prims::expm1.default  prims::expm1(Tensor self) -> Tensor
prims::fft_c2c.default  prims::fft_c2c(Tensor self, *, int[] dim, bool forward) -> Tensor
prims::fft_c2r.default  prims::fft_c2r(Tensor self, *, int[] dim, SymInt last_dim_size) -> Tensor
prims::fft_r2c.default  prims::fft_r2c(Tensor self, *, int[] dim, bool onesided) -> Tensor
prims::fill.default  prims::fill(Tensor self, Scalar value) -> Tensor
prims::floor.default  prims::floor(Tensor self) -> Tensor
prims::fmax.default  prims::fmax(Tensor self, Tensor other) -> Tensor
prims::fmin.default  prims::fmin(Tensor self, Tensor other) -> Tensor
prims::fmod.default  prims::fmod(Tensor self, Tensor other) -> Tensor
prims::frexp.default  prims::frexp(Tensor self) -> (Tensor mantissa, Tensor exponent)
prims::full.default  prims::full(SymInt[] shape, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::full_like.default  prims::full_like(Tensor a, Scalar fill_value, *, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::gcd.default  prims::gcd(Tensor self, Tensor other) -> Tensor
prims::ge.default  prims::ge(Tensor self, Tensor other) -> Tensor
prims::gt.default  prims::gt(Tensor self, Tensor other) -> Tensor
prims::hypot.default  prims::hypot(Tensor self, Tensor other) -> Tensor
prims::igamma.default  prims::igamma(Tensor self, Tensor other) -> Tensor
prims::igammac.default  prims::igammac(Tensor self, Tensor other) -> Tensor
prims::imag.default  prims::imag(Tensor(a) self) -> Tensor(a)
prims::iota.default  prims::iota(SymInt length, *, SymInt start, SymInt step, ScalarType dtype, Device device, bool requires_grad) -> Tensor
prims::isfinite.default  prims::isfinite(Tensor self) -> Tensor
prims::item.default  prims::item(Tensor a) -> Scalar
prims::le.default  prims::le(Tensor self, Tensor other) -> Tensor
prims::lgamma.default  prims::lgamma(Tensor self) -> Tensor
prims::log.default  prims::log(Tensor self) -> Tensor
prims::log10.default  prims::log10(Tensor self) -> Tensor
prims::log1p.default  prims::log1p(Tensor self) -> Tensor
prims::log2.default  prims::log2(Tensor self) -> Tensor
prims::lt.default  prims::lt(Tensor self, Tensor other) -> Tensor
prims::maximum.default  prims::maximum(Tensor self, Tensor other) -> Tensor
prims::maximum_value.default  prims::maximum_value(ScalarType dtype) -> Scalar
prims::minimum.default  prims::minimum(Tensor self, Tensor other) -> Tensor
prims::minimum_value.default  prims::minimum_value(ScalarType dtype) -> Scalar
prims::mul.default  prims::mul(Tensor self, Tensor other) -> Tensor
prims::ndtri.default  prims::ndtri(Tensor self) -> Tensor
prims::ne.default  prims::ne(Tensor self, Tensor other) -> Tensor
prims::neg.default  prims::neg(Tensor self) -> Tensor
prims::nextafter.default  prims::nextafter(Tensor self, Tensor other) -> Tensor
prims::normal.default  prims::normal(SymInt[] shape, *, Scalar mean, Scalar std, ScalarType dtype, Device device, bool requires_grad, Generator? generator=None) -> Tensor
prims::pow.default  prims::pow(Tensor self, Tensor other) -> Tensor
prims::prod.default  prims::prod(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor
prims::real.default  prims::real(Tensor(a) self) -> Tensor(a)
prims::reciprocal.default  prims::reciprocal(Tensor self) -> Tensor
prims::remainder.default  prims::remainder(Tensor self, Tensor other) -> Tensor
prims::reshape.default  prims::reshape(Tensor a, SymInt[] shape) -> Tensor
prims::resize.default  prims::resize(Tensor(a!) a, SymInt[] shape) -> Tensor(a!)
prims::rev.default  prims::rev(Tensor a, int[] dims) -> Tensor
prims::round.default  prims::round(Tensor self) -> Tensor
prims::rsqrt.default  prims::rsqrt(Tensor self) -> Tensor
prims::scalar_tensor.default  prims::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Device? device=None) -> Tensor
prims::shift_left.default  prims::shift_left(Tensor self, Tensor other) -> Tensor
prims::shift_right_arithmetic.default  prims::shift_right_arithmetic(Tensor self, Tensor other) -> Tensor
prims::sign.default  prims::sign(Tensor self) -> Tensor
prims::signbit.default  prims::signbit(Tensor self) -> Tensor
prims::sin.default  prims::sin(Tensor self) -> Tensor
prims::sinh.default  prims::sinh(Tensor self) -> Tensor
prims::slice.default  prims::slice(Tensor(a) a, SymInt[] start_indices, SymInt[] limit_indices, SymInt[]? strides=None) -> Tensor(a)
prims::slice_in_dim.default  prims::slice_in_dim(Tensor(a) a, SymInt start_index, SymInt limit_index, int stride=1, int axis=0) -> Tensor(a)
prims::spherical_bessel_j0.default  prims::spherical_bessel_j0(Tensor self) -> Tensor
prims::split_dim.default  prims::split_dim(Tensor(a) a, int dim, SymInt outer_length) -> Tensor(a)
prims::sqrt.default  prims::sqrt(Tensor self) -> Tensor
prims::squeeze.default  prims::squeeze(Tensor(a) a, int[] dimensions) -> Tensor(a)
prims::sub.default  prims::sub(Tensor self, Tensor other) -> Tensor
prims::sum.default  prims::sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor
prims::svd.default  prims::svd(Tensor A, *, bool full_matrices) -> (Tensor U, Tensor S, Tensor Vh)
prims::tan.default  prims::tan(Tensor self) -> Tensor
prims::tanh.default  prims::tanh(Tensor self) -> Tensor
prims::transpose.default  prims::transpose(Tensor(a) a, int[] permutation) -> Tensor(a)
prims::trunc.default  prims::trunc(Tensor self) -> Tensor
prims::uniform.default  prims::uniform(SymInt[] shape, *, Scalar low, Scalar high, ScalarType dtype, Device device, Generator? generator=None) -> Tensor
prims::var.default  prims::var(Tensor inp, int[]? dims, float? correction=1, *, ScalarType? output_dtype=None) -> Tensor
prims::view_of.default  prims::view_of(Tensor(a) a) -> Tensor(a)
prims::view_of_dtype.default  prims::view_of_dtype(Tensor(a) a, ScalarType dtype) -> Tensor(a)
prims::where.default  prims::where(Tensor pred, Tensor a, Tensor b) -> Tensor
prims::xor_sum.default  prims::xor_sum(Tensor inp, int[]? dims, *, ScalarType? output_dtype=None) -> Tensor
prims::zeta.default  prims::zeta(Tensor self, Tensor other) -> Tensor
```

</details>

<a id="profiler-namespace"></a>

## `profiler` namespace (4 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `profiler::_record_function_enter.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `profiler::_record_function_enter_new.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `profiler::_record_function_exit._RecordFunction` | Unreviewed | — | Not yet audited for Vulkan. |
| `profiler::_record_function_exit.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>profiler</code> (4)</summary>

```text
profiler::_record_function_enter.default  profiler::_record_function_enter(str name, str? args=None) -> Tensor
profiler::_record_function_enter_new.default  profiler::_record_function_enter_new(str name, str? args=None) -> __torch__.torch.classes.profiler._RecordFunction
profiler::_record_function_exit._RecordFunction  profiler::_record_function_exit._RecordFunction(__torch__.torch.classes.profiler._RecordFunction _0) -> ()
profiler::_record_function_exit.default  profiler::_record_function_exit(Tensor _0) -> ()
```

</details>

<a id="quantization-namespace"></a>

## `quantization` namespace (2 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `quantization::_Bfloat16QuantizedToFloat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantization::_FloatToBfloat16Quantized.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>quantization</code> (2)</summary>

```text
quantization::_Bfloat16QuantizedToFloat.default  quantization::_Bfloat16QuantizedToFloat(Tensor input) -> Tensor
quantization::_FloatToBfloat16Quantized.default  quantization::_FloatToBfloat16Quantized(Tensor input) -> Tensor
```

</details>

<a id="quantized-namespace"></a>

## `quantized` namespace (170 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `quantized::add.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add.Scalar2` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu.Scalar2` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_relu_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_out.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_relu.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_relu_out.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::add_scalar_relu_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm1d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm2d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm3d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::batch_norm_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::cat.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::cat_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::cat_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::cat_relu_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::celu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::clamp.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv1d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv1d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv1d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv1d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d.new` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_add.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_add_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_dilation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_groups.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_output_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_relu.new` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_stride.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv2d_unpack_sizes.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d.new` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_dilation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_groups.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_output_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_relu.new` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_stride.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv3d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose1d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose1d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose1d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_dilation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_groups.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_output_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_stride.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose2d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_dilation.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_groups.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_output_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_padding.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_stride.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_transpose.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_transpose3d_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::conv_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::dropout.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::elu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_4bit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_2bit_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_2bit_rowwise_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_2bit_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_4bit.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_4bit_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_4bit_rowwise_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_4bit_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_byte.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_byte_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_byte_rowwise_offsets.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_byte_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_bag_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::embedding_byte.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::group_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::hardswish.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::instance_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::layer_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::leaky_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_dynamic_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_dynamic_fp16_unpacked_weight.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_leaky_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_prepack_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_prepack_fp16_legacy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_prepack_legacy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_relu_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_relu_dynamic_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_tanh.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_unpack.legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_unpack_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_unpack_fp16.legacy` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_with_input_q_dq_qweight_dq_output_fp32.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::make_quantized_cell_params.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::make_quantized_cell_params_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::make_quantized_cell_params_fp16.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::matmul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::max_pool1d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::max_pool2d.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul.Scalar2` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu.Scalar` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu.Scalar2` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu.Scalar_out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu.out` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_relu_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_out.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_relu.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_relu_out.Tensor` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::mul_scalar_relu_out.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::prelu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::quantized_gru_cell_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::quantized_lstm_cell_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::quantized_rnn_relu_cell_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::quantized_rnn_tanh_cell_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::relu6.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::sigmoid.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::softmax.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `quantized::threshold.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>quantized</code> (170)</summary>

```text
quantized::add.Scalar  quantized::add.Scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::add.Scalar2  quantized::add.Scalar2(Scalar b, Tensor qa) -> Tensor qc
quantized::add.Scalar_out  quantized::add.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::add.default  quantized::add(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
quantized::add.out  quantized::add.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::add_out.default  quantized::add_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::add_relu.Scalar  quantized::add_relu.Scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::add_relu.Scalar2  quantized::add_relu.Scalar2(Scalar b, Tensor qa) -> Tensor qc
quantized::add_relu.Scalar_out  quantized::add_relu.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::add_relu.default  quantized::add_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
quantized::add_relu.out  quantized::add_relu.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::add_relu_out.default  quantized::add_relu_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::add_scalar.Tensor  quantized::add_scalar.Tensor(Tensor qa, Tensor b) -> Tensor qc
quantized::add_scalar.default  quantized::add_scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::add_scalar_out.Tensor  quantized::add_scalar_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out
quantized::add_scalar_out.default  quantized::add_scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::add_scalar_relu.Tensor  quantized::add_scalar_relu.Tensor(Tensor qa, Tensor b) -> Tensor qc
quantized::add_scalar_relu.default  quantized::add_scalar_relu(Tensor qa, Scalar b) -> Tensor qc
quantized::add_scalar_relu_out.Tensor  quantized::add_scalar_relu_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out
quantized::add_scalar_relu_out.default  quantized::add_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::batch_norm.default  quantized::batch_norm(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm1d.default  quantized::batch_norm1d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm1d_relu.default  quantized::batch_norm1d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm2d.default  quantized::batch_norm2d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm2d_relu.default  quantized::batch_norm2d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm3d.default  quantized::batch_norm3d(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm3d_relu.default  quantized::batch_norm3d_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::batch_norm_relu.default  quantized::batch_norm_relu(Tensor qx, Tensor? weight, Tensor? bias, Tensor mean, Tensor var, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::cat.default  quantized::cat(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor
quantized::cat_out.default  quantized::cat_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)
quantized::cat_relu.default  quantized::cat_relu(Tensor[] qx, int dim, float? scale, int? zero_point) -> Tensor
quantized::cat_relu_out.default  quantized::cat_relu_out(Tensor[] qx, int dim, Tensor(a!) out) -> Tensor(a!)
quantized::celu.default  quantized::celu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1) -> Tensor
quantized::clamp.default  quantized::clamp(Tensor qx, Scalar? min=None, Scalar? max=None) -> Tensor qy
quantized::conv1d.default  quantized::conv1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv1d_dynamic.default  quantized::conv1d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv1d_prepack.default  quantized::conv1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
quantized::conv1d_relu.default  quantized::conv1d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv1d_unpack.default  quantized::conv1d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv2d.default  quantized::conv2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d.new  quantized::conv2d.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d_add.default  quantized::conv2d_add(Tensor qx, Tensor qaccum, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d_add_relu.default  quantized::conv2d_add_relu(Tensor qx, Tensor qaccum, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d_dilation.default  quantized::conv2d_dilation(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv2d_dynamic.default  quantized::conv2d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv2d_groups.default  quantized::conv2d_groups(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int
quantized::conv2d_output_padding.default  quantized::conv2d_output_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv2d_padding.default  quantized::conv2d_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv2d_prepack.default  quantized::conv2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
quantized::conv2d_relu.default  quantized::conv2d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d_relu.new  quantized::conv2d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv2d_stride.default  quantized::conv2d_stride(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv2d_transpose.default  quantized::conv2d_transpose(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int
quantized::conv2d_unpack.default  quantized::conv2d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv2d_unpack_sizes.default  quantized::conv2d_unpack_sizes(Any packed_weights) -> Any
quantized::conv3d.default  quantized::conv3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor
quantized::conv3d.new  quantized::conv3d.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv3d_dilation.default  quantized::conv3d_dilation(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv3d_dynamic.default  quantized::conv3d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv3d_groups.default  quantized::conv3d_groups(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int
quantized::conv3d_output_padding.default  quantized::conv3d_output_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv3d_padding.default  quantized::conv3d_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv3d_prepack.default  quantized::conv3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase
quantized::conv3d_relu.default  quantized::conv3d_relu(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase weight, int[] stride, int[] padding, int[] dilation, int groups, float output_scale, int output_zero_point) -> Tensor
quantized::conv3d_relu.new  quantized::conv3d_relu.new(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv3d_stride.default  quantized::conv3d_stride(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv3d_transpose.default  quantized::conv3d_transpose(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int
quantized::conv3d_unpack.default  quantized::conv3d_unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv_prepack.default  quantized::conv_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
quantized::conv_transpose1d.default  quantized::conv_transpose1d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv_transpose1d_dynamic.default  quantized::conv_transpose1d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv_transpose1d_prepack.default  quantized::conv_transpose1d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
quantized::conv_transpose1d_unpack.default  quantized::conv_transpose1d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv_transpose2d.default  quantized::conv_transpose2d(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv_transpose2d_dilation.default  quantized::conv_transpose2d_dilation(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose2d_dynamic.default  quantized::conv_transpose2d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv_transpose2d_groups.default  quantized::conv_transpose2d_groups(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int
quantized::conv_transpose2d_output_padding.default  quantized::conv_transpose2d_output_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose2d_padding.default  quantized::conv_transpose2d_padding(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose2d_prepack.default  quantized::conv_transpose2d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv2dPackedParamsBase
quantized::conv_transpose2d_stride.default  quantized::conv_transpose2d_stride(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose2d_transpose.default  quantized::conv_transpose2d_transpose(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> int
quantized::conv_transpose2d_unpack.default  quantized::conv_transpose2d_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv_transpose3d.default  quantized::conv_transpose3d(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, float output_scale, int output_zero_point) -> Tensor
quantized::conv_transpose3d_dilation.default  quantized::conv_transpose3d_dilation(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose3d_dynamic.default  quantized::conv_transpose3d_dynamic(Tensor qx, __torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weight, bool reduce_range=False) -> Tensor
quantized::conv_transpose3d_groups.default  quantized::conv_transpose3d_groups(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int
quantized::conv_transpose3d_output_padding.default  quantized::conv_transpose3d_output_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose3d_padding.default  quantized::conv_transpose3d_padding(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose3d_prepack.default  quantized::conv_transpose3d_prepack(Tensor weight, Tensor? bias, int[] stride, int[] padding, int[] output_padding, int[] dilation, int groups) -> __torch__.torch.classes.quantized.Conv3dPackedParamsBase
quantized::conv_transpose3d_stride.default  quantized::conv_transpose3d_stride(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int[]
quantized::conv_transpose3d_transpose.default  quantized::conv_transpose3d_transpose(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> int
quantized::conv_transpose3d_unpack.default  quantized::conv_transpose3d_unpack(__torch__.torch.classes.quantized.Conv3dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::conv_unpack.default  quantized::conv_unpack(__torch__.torch.classes.quantized.Conv2dPackedParamsBase packed_weights) -> (Tensor unpacked_weights, Tensor? B_origin)
quantized::dropout.default  quantized::dropout(Tensor self, float output_scale, int output_zero_point, Scalar p=0.5, bool training=False) -> Tensor
quantized::elu.default  quantized::elu(Tensor self, float output_scale, int output_zero_point, Scalar alpha=1, Scalar scale=1, Scalar input_scale=1) -> Tensor
quantized::embedding_4bit.default  quantized::embedding_4bit(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, bool pruned_weights=False) -> Tensor
quantized::embedding_bag_2bit_prepack.default  quantized::embedding_bag_2bit_prepack(Tensor weight, bool optimized_qparams=False, int nbins=200, float ratio=0.16) -> Tensor
quantized::embedding_bag_2bit_rowwise_offsets.default  quantized::embedding_bag_2bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor
quantized::embedding_bag_2bit_unpack.default  quantized::embedding_bag_2bit_unpack(Tensor weight) -> Tensor
quantized::embedding_bag_4bit.default  quantized::embedding_bag_4bit(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor
quantized::embedding_bag_4bit_prepack.default  quantized::embedding_bag_4bit_prepack(Tensor weight, bool optimized_qparams=False, int nbins=200, float ratio=0.16) -> Tensor
quantized::embedding_bag_4bit_rowwise_offsets.default  quantized::embedding_bag_4bit_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor
quantized::embedding_bag_4bit_unpack.default  quantized::embedding_bag_4bit_unpack(Tensor weight) -> Tensor
quantized::embedding_bag_byte.default  quantized::embedding_bag_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor
quantized::embedding_bag_byte_prepack.default  quantized::embedding_bag_byte_prepack(Tensor weight) -> Tensor
quantized::embedding_bag_byte_rowwise_offsets.default  quantized::embedding_bag_byte_rowwise_offsets(Tensor weight, Tensor indices, Tensor? offsets=None, bool scale_grad_by_freq=False, int mode=0, bool pruned_weights=False, Tensor? per_sample_weights=None, Tensor? compressed_indices_mapping=None, bool include_last_offset=False) -> Tensor
quantized::embedding_bag_byte_unpack.default  quantized::embedding_bag_byte_unpack(Tensor weight) -> Tensor
quantized::embedding_bag_prepack.default  quantized::embedding_bag_prepack(Tensor weight) -> __torch__.torch.classes.quantized.EmbeddingPackedParamsBase W_prepack
quantized::embedding_bag_unpack.default  quantized::embedding_bag_unpack(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase W_prepack) -> Tensor W_origin
quantized::embedding_byte.default  quantized::embedding_byte(__torch__.torch.classes.quantized.EmbeddingPackedParamsBase weight, Tensor indices, bool pruned_weights=False) -> Tensor
quantized::group_norm.default  quantized::group_norm(Tensor input, int num_groups, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::hardswish.default  quantized::hardswish(Tensor input, float output_scale, int output_zero_point) -> Tensor
quantized::instance_norm.default  quantized::instance_norm(Tensor input, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::layer_norm.default  quantized::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight, Tensor? bias, float eps, float output_scale, int output_zero_point) -> Tensor
quantized::leaky_relu.default  quantized::leaky_relu(Tensor qx, Scalar negative_slope, bool inplace, float output_scale, int output_zero_point) -> Tensor
quantized::linear.default  quantized::linear(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
quantized::linear_dynamic.default  quantized::linear_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y
quantized::linear_dynamic_fp16.default  quantized::linear_dynamic_fp16(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y
quantized::linear_dynamic_fp16_unpacked_weight.default  quantized::linear_dynamic_fp16_unpacked_weight(Tensor X, Tensor weight, Tensor bias) -> Tensor Y
quantized::linear_leaky_relu.default  quantized::linear_leaky_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i, float negative_slope) -> Tensor Y
quantized::linear_prepack.default  quantized::linear_prepack(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack
quantized::linear_prepack_fp16.default  quantized::linear_prepack_fp16(Tensor W, Tensor? B=None) -> __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack
quantized::linear_prepack_fp16_legacy.default  quantized::linear_prepack_fp16_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack
quantized::linear_prepack_legacy.default  quantized::linear_prepack_legacy(Tensor W, Tensor? B=None) -> Tensor W_prepack
quantized::linear_relu.default  quantized::linear_relu(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
quantized::linear_relu_dynamic.default  quantized::linear_relu_dynamic(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, bool reduce_range=False) -> Tensor Y
quantized::linear_relu_dynamic_fp16.default  quantized::linear_relu_dynamic_fp16(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y
quantized::linear_tanh.default  quantized::linear_tanh(Tensor X, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
quantized::linear_unpack.default  quantized::linear_unpack(__torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin)
quantized::linear_unpack.legacy  quantized::linear_unpack.legacy(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)
quantized::linear_unpack_fp16.default  quantized::linear_unpack_fp16(__torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin)
quantized::linear_unpack_fp16.legacy  quantized::linear_unpack_fp16.legacy(Tensor W_prepack) -> (Tensor W_origin, Tensor? B_origin)
quantized::linear_with_input_q_dq_qweight_dq_output_fp32.default  quantized::linear_with_input_q_dq_qweight_dq_output_fp32(Tensor X, float X_scale, int X_zero_point, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y
quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32.default  quantized::linear_with_input_q_dq_qweight_dq_relu_output_fp32(Tensor X, float X_scale, int X_zero_point, __torch__.torch.classes.quantized.LinearPackedParamsBase W_prepack) -> Tensor Y
quantized::make_quantized_cell_params.default  quantized::make_quantized_cell_params(Tensor w_ih, Tensor w_hh, Tensor b_ih, Tensor b_hh) -> __torch__.torch.classes.rnn.CellParamsBase
quantized::make_quantized_cell_params_dynamic.default  quantized::make_quantized_cell_params_dynamic(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh, bool reduce_range=False) -> __torch__.torch.classes.rnn.CellParamsBase
quantized::make_quantized_cell_params_fp16.default  quantized::make_quantized_cell_params_fp16(__torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh) -> __torch__.torch.classes.rnn.CellParamsBase
quantized::matmul.default  quantized::matmul(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
quantized::max_pool1d.default  quantized::max_pool1d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor
quantized::max_pool2d.default  quantized::max_pool2d(Tensor qx, int[] kernel_size, int[] stride, int[] padding, int[] dilation, bool ceil_mode) -> Tensor
quantized::mul.Scalar  quantized::mul.Scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::mul.Scalar2  quantized::mul.Scalar2(Scalar b, Tensor qa) -> Tensor qc
quantized::mul.Scalar_out  quantized::mul.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::mul.default  quantized::mul(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
quantized::mul.out  quantized::mul.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_out.default  quantized::mul_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_relu.Scalar  quantized::mul_relu.Scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::mul_relu.Scalar2  quantized::mul_relu.Scalar2(Scalar b, Tensor qa) -> Tensor qc
quantized::mul_relu.Scalar_out  quantized::mul_relu.Scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_relu.default  quantized::mul_relu(Tensor qa, Tensor qb, float scale, int zero_point) -> Tensor qc
quantized::mul_relu.out  quantized::mul_relu.out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_relu_out.default  quantized::mul_relu_out(Tensor qa, Tensor qb, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_scalar.Tensor  quantized::mul_scalar.Tensor(Tensor qa, Tensor b) -> Tensor qc
quantized::mul_scalar.default  quantized::mul_scalar(Tensor qa, Scalar b) -> Tensor qc
quantized::mul_scalar_out.Tensor  quantized::mul_scalar_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_scalar_out.default  quantized::mul_scalar_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_scalar_relu.Tensor  quantized::mul_scalar_relu.Tensor(Tensor qa, Tensor b) -> Tensor qc
quantized::mul_scalar_relu.default  quantized::mul_scalar_relu(Tensor qa, Scalar b) -> Tensor qc
quantized::mul_scalar_relu_out.Tensor  quantized::mul_scalar_relu_out.Tensor(Tensor qa, Tensor b, Tensor(a!) out) -> Tensor(a!) out
quantized::mul_scalar_relu_out.default  quantized::mul_scalar_relu_out(Tensor qa, Scalar b, Tensor(a!) out) -> Tensor(a!) out
quantized::prelu.default  quantized::prelu(Tensor qx, Tensor weight, float output_scale, int output_zero_point) -> Tensor
quantized::quantized_gru_cell_dynamic.default  quantized::quantized_gru_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor
quantized::quantized_lstm_cell_dynamic.default  quantized::quantized_lstm_cell_dynamic(Tensor input, Tensor[] hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor bias_ih, Tensor bias_hh) -> (Tensor, Tensor)
quantized::quantized_rnn_relu_cell_dynamic.default  quantized::quantized_rnn_relu_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor
quantized::quantized_rnn_tanh_cell_dynamic.default  quantized::quantized_rnn_tanh_cell_dynamic(Tensor input, Tensor hx, __torch__.torch.classes.quantized.LinearPackedParamsBase w_ih, __torch__.torch.classes.quantized.LinearPackedParamsBase w_hh, Tensor b_ih, Tensor b_hh) -> Tensor
quantized::relu6.default  quantized::relu6(Tensor qx, bool inplace=False) -> Tensor
quantized::sigmoid.default  quantized::sigmoid(Tensor qx, float output_scale, int output_zero_point) -> Tensor
quantized::softmax.default  quantized::softmax(Tensor qx, int dim, float output_scale, int output_zero_point) -> Tensor
quantized::threshold.default  quantized::threshold(Tensor qx, Scalar threshold, Scalar value) -> Tensor qy
```

</details>

<a id="rngprims-namespace"></a>

## `rngprims` namespace (1 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `rngprims::philox_rand.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>rngprims</code> (1)</summary>

```text
rngprims::philox_rand.default  rngprims::philox_rand(SymInt[] size, Tensor seed, Tensor offset, int[]? stride, Device? device=None, ScalarType? dtype=None) -> (Tensor, Tensor)
```

</details>

<a id="sparse-namespace"></a>

## `sparse` namespace (6 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `sparse::qlinear.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `sparse::qlinear_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `sparse::qlinear_prepack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `sparse::qlinear_relu.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `sparse::qlinear_relu_dynamic.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `sparse::qlinear_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>sparse</code> (6)</summary>

```text
sparse::qlinear.default  sparse::qlinear(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
sparse::qlinear_dynamic.default  sparse::qlinear_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y
sparse::qlinear_prepack.default  sparse::qlinear_prepack(Tensor W, Tensor? B, int out_features_block_size, int in_features_block_size) -> __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack
sparse::qlinear_relu.default  sparse::qlinear_relu(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack, float Y_scale_i, int Y_zero_point_i) -> Tensor Y
sparse::qlinear_relu_dynamic.default  sparse::qlinear_relu_dynamic(Tensor X, __torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> Tensor Y
sparse::qlinear_unpack.default  sparse::qlinear_unpack(__torch__.torch.classes.sparse.LinearPackedParamsBase W_prepack) -> (Tensor W_origin, Tensor? B_origin, int[] block_pattern)
```

</details>

<a id="staticruntime-namespace"></a>

## `static_runtime` namespace (21 schemas)

| Operator schema | Status | Commit(s) | Notes |
|---|---|---|---|
| `static_runtime::VarTupleUnpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::clamp_nan_to_num.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::create_owned_ref.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::dequantize_copy.self` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::dict_unpack.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::embedding_bag.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::embedding_bag.padding_idx` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::expand_dims_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::flatten_copy.using_ints` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::fused_equally_split.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::layer_norm.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::permute_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::reshape_copy.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::select_tensor.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::signed_log1p.default` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_copy.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_copy.other` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_copy.prim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_maybe_copy_out.dtype` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_maybe_copy_out.other` | Unreviewed | — | Not yet audited for Vulkan. |
| `static_runtime::to_maybe_copy_out.prim_dtype` | Unreviewed | — | Not yet audited for Vulkan. |

<details><summary>Signature appendix: <code>static_runtime</code> (21)</summary>

```text
static_runtime::VarTupleUnpack.default  static_runtime::VarTupleUnpack(...) -> ...
static_runtime::clamp_nan_to_num.default  static_runtime::clamp_nan_to_num(Tensor input, Scalar? min, Scalar? max, float? nan, float? posinf, float? posinf) -> Tensor
static_runtime::create_owned_ref.default  static_runtime::create_owned_ref(...) -> ...
static_runtime::dequantize_copy.self  static_runtime::dequantize_copy.self(Tensor self) -> Tensor
static_runtime::dict_unpack.default  static_runtime::dict_unpack(...) -> ...
static_runtime::embedding_bag.default  static_runtime::embedding_bag(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq=False, int mode=0, bool sparse=False, Tensor? per_sample_weights=None, bool include_last_offset=False) -> (Tensor, Tensor, Tensor)
static_runtime::embedding_bag.padding_idx  static_runtime::embedding_bag.padding_idx(Tensor weight, Tensor indices, Tensor offsets, bool scale_grad_by_freq, int mode, bool sparse, Tensor? per_sample_weights, bool include_last_offset, int? padding_idx) -> (Tensor, Tensor, Tensor)
static_runtime::expand_dims_copy.default  static_runtime::expand_dims_copy(Tensor input, int[] dims) -> Tensor
static_runtime::flatten_copy.using_ints  static_runtime::flatten_copy.using_ints(Tensor self, int start_dim=0, int end_dim=-1) -> Tensor
static_runtime::fused_equally_split.default  static_runtime::fused_equally_split(Tensor input, int num_split, int dim) -> ...
static_runtime::layer_norm.default  static_runtime::layer_norm(Tensor input, int[] normalized_shape, Tensor? weight=None, Tensor? bias=None, float eps=1.0000000000000001e-05, bool cudnn_enable=True) -> (Tensor, Tensor, Tensor)
static_runtime::permute_copy.default  static_runtime::permute_copy(Tensor self, int[] dims) -> Tensor
static_runtime::reshape_copy.default  static_runtime::reshape_copy(Tensor self, int[] shape) -> Tensor
static_runtime::select_tensor.default  static_runtime::select_tensor(Tensor(a) a, Tensor(b) b, bool use_b) -> Tensor(b|a)
static_runtime::signed_log1p.default  static_runtime::signed_log1p(Tensor input) -> Tensor
static_runtime::to_copy.dtype  static_runtime::to_copy.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
static_runtime::to_copy.other  static_runtime::to_copy.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> Tensor
static_runtime::to_copy.prim_dtype  static_runtime::to_copy.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> Tensor
static_runtime::to_maybe_copy_out.dtype  static_runtime::to_maybe_copy_out.dtype(Tensor self, ScalarType dtype, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)
static_runtime::to_maybe_copy_out.other  static_runtime::to_maybe_copy_out.other(Tensor self, Tensor other, bool non_blocking=False, bool copy=False, MemoryFormat? memory_format=None) -> (Tensor, bool)
static_runtime::to_maybe_copy_out.prim_dtype  static_runtime::to_maybe_copy_out.prim_dtype(Tensor self, int? dtype=None, bool non_blocking=False, bool copy=False) -> (Tensor, bool)
```

</details>

## Maintenance

To refresh after a supported PyTorch version change (CPU-only, no Vulkan extension):

1. In a clean interpreter running `python -I`, import only `torch`. Build the row set exactly as in *Extraction method*: dispatcher names from `torch._C._dispatch_get_all_op_names()`, schemas from `torch._C._jit_get_all_schemas()`, key `ns::op.overload` (empty overload = `default`), one `key<TAB>str(schema)` row each; assert no duplicate key, no missing schema, and `pytorch_vulkan` not in `sys.modules`. Run it twice and require byte-identical sorted output.
2. Record `torch.__version__`, `torch.version.git_version`, row count, distinct-name count and the SHA-256 of the sorted LF-joined rows.
3. Diff the new key set against this file's keys: **added** keys become `Unreviewed`; **removed** keys are deleted; for keys whose signature (see the appendices) changed, re-review the status against the new native/composite/derivative reference before keeping it.
4. Re-check every non-`Unreviewed` row against `docs/vulkan_capabilities.json`, the registrations and the named tests; update commit references only from `git log` evidence.
5. Update the header, status-count table and digest, and confirm every key appears exactly once.
