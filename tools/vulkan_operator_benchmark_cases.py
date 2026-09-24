"""Operator registration inventory and case catalog for Vulkan benchmarks."""

from __future__ import annotations

import re
from pathlib import Path

from tools.validate_vulkan_capabilities import (
    _source_registration_inventory,
    load_manifest,
)


CASE_SCHEMA_VERSION = 1

_CASE_SCHEMAS = {
    "aten::abs.default", "aten::neg.default", "aten::relu.default",
    "aten::sigmoid.default", "aten::tanh.default", "aten::gelu.default",
    "aten::add.Tensor", "aten::add.Scalar", "aten::sub.Tensor", "aten::sub.Scalar",
    "aten::rsub.Scalar", "aten::mul.Tensor", "aten::mul.Scalar", "aten::div.Tensor",
    "aten::mm.default", "aten::bmm.default", "aten::linear.default", "aten::addmm.default",
    "aten::convolution.default", "aten::_adaptive_avg_pool2d.default",
    "aten::mean.dim", "aten::sum.dim_IntList", "aten::amax.default", "aten::amin.default",
    "aten::prod.dim_int", "aten::argmax.default", "aten::_softmax.default",
    "aten::_log_softmax.default", "aten::native_batch_norm.default", "aten::mse_loss.default",
    "aten::nll_loss_forward.default", "aten::zero_.default", "aten::addcmul_.default",
    "aten::addcdiv_.default", "aten::view.default", "aten::reshape.default",
    "aten::_reshape_alias.default", "aten::as_strided.default", "aten::stack.default",
    "aten::gelu_backward.grad_input", "aten::sigmoid_backward.grad_input",
    "aten::tanh_backward.grad_input", "aten::_adaptive_avg_pool2d_backward.default",
    "aten::_softmax_backward_data.out", "aten::_log_softmax_backward_data.out",
    "aten::mse_loss_backward.default", "aten::convolution_backward.default",
    "aten::native_batch_norm_backward.default", "aten::nll_loss_backward.default",
    "aten::add.Scalar_out", "aten::add.out", "aten::add_.Tensor",
    "aten::sub.Scalar_out", "aten::sub.out", "aten::rsub.Scalar_out",
    "aten::mul.Scalar_out", "aten::mul.out", "aten::mul_.Scalar",
    "aten::lerp.Scalar_out", "aten::lerp_.Scalar", "aten::sqrt.out",
    "aten::amax.out", "aten::amin.out", "aten::prod.int_out",
    "aten::_softmax.out", "aten::_log_softmax.out", "aten::addmm.out",
    "aten::copy_.default", "aten::masked_select.default",
    "pytorch_vulkan::rnn_sequence", "pytorch_vulkan::linear_relu",
}
_METADATA_SCHEMAS = {
    "aten::view.default", "aten::reshape.default", "aten::_reshape_alias.default",
    "aten::as_strided.default",
}
_DIRECT_BACKWARD_SCHEMAS = {
    schema for schema in _CASE_SCHEMAS
    if "_backward" in schema
}
_OUTPUT_SCHEMAS = {
    "aten::add.Scalar_out", "aten::add.out", "aten::sub.Scalar_out", "aten::sub.out",
    "aten::rsub.Scalar_out", "aten::mul.Scalar_out", "aten::mul.out",
    "aten::lerp.Scalar_out", "aten::sqrt.out", "aten::amax.out", "aten::amin.out",
    "aten::prod.int_out", "aten::_softmax.out", "aten::_log_softmax.out", "aten::addmm.out",
}
_INPLACE_SCHEMAS = {
    "aten::add_.Tensor", "aten::mul_.Scalar",
    "aten::lerp_.Scalar", "aten::zero_.default", "aten::addcmul_.default",
    "aten::addcdiv_.default", "aten::copy_.default",
}
_AUTOGRAD_SCHEMAS = {
    schema for schema in _CASE_SCHEMAS
    if schema not in {
        "aten::argmax.default", "aten::zero_.default", "aten::addcmul_.default",
        "aten::addcdiv_.default", "aten::nll_loss_forward.default",
    } and schema not in _DIRECT_BACKWARD_SCHEMAS
    and schema not in _OUTPUT_SCHEMAS and schema not in _INPLACE_SCHEMAS
    and schema != "aten::masked_select.default"
}

_MASKED_SELECT_PROFILES = (
    "zero", "medium_none", "medium_sparse", "medium_half", "medium_dense",
    "wide_none", "wide_sparse", "wide_half", "wide_dense",
)


def _family(schema: str) -> str:
    name = schema.split("::", 1)[-1].split(".", 1)[0]
    if name in {"linear", "mm", "bmm", "addmm"}:
        return "linear_gemm"
    if name in {"convolution", "convolution_overrideable", "convolution_backward"}:
        return "convolution_pooling"
    if "pool" in name:
        return "convolution_pooling"
    if "batch_norm" in name or name in {"layer_norm", "native_layer_norm"}:
        return "normalization"
    if name in {"sum", "mean", "amax", "amin", "prod", "argmax", "softmax", "_softmax", "_log_softmax"}:
        return "reduction_indexing"
    if "loss" in name or name.startswith("nll_"):
        return "loss_classification"
    if name in {"addcmul_", "addcdiv_", "sgd", "adam"}:
        return "optimizer"
    if name in {"view", "reshape", "_reshape_alias", "as_strided", "stack", "flatten"}:
        return "view_metadata"
    if name in {"_to_copy", "copy_", "to"}:
        return "copy"
    if schema.startswith("pytorch_vulkan::"):
        return "custom"
    if name.startswith(("conv", "max_pool", "adaptive_avg_pool")):
        return "convolution_pooling"
    return "pointwise"


def _custom_registration_inventory(root: Path) -> set[str]:
    registrations: set[str] = set()
    macro = re.compile(
        r"TORCH_LIBRARY(?:_FRAGMENT)?\s*\(\s*pytorch_vulkan\s*,\s*[^)]*\)"
    )
    definition = re.compile(r'\bm\s*\.\s*m?def\s*\(\s*"([^"(]+)')
    for path in (root / "src" / "vulkan" / "operators").glob("*.cpp"):
        source = re.sub(r"//[^\n]*|/\*.*?\*/", " ", path.read_text(), flags=re.DOTALL)
        for match in macro.finditer(source):
            opening = source.find("{", match.end())
            if opening < 0:
                continue
            depth = 0
            for index in range(opening, len(source)):
                if source[index] == "{":
                    depth += 1
                elif source[index] == "}":
                    depth -= 1
                    if depth == 0:
                        block = source[opening + 1 : index]
                        registrations.update(
                            f"pytorch_vulkan::{name}" for name in definition.findall(block)
                        )
                        break
    return registrations


def load_operator_cases(root: Path) -> dict:
    """Load the explicit benchmark inventory, retaining non-measurable entries."""
    manifest = load_manifest(root / "docs" / "vulkan_capabilities.json")
    entries = []
    for capability in manifest["entries"]:
        schema = capability["schema"]
        execution = capability["execution_contract"]
        if schema in _CASE_SCHEMAS:
            status, reason = "measurable", "case factory available"
        elif execution in {"rejected_before_vulkan", "deferred_before_vulkan"} or capability["status"] != "supported":
            status, reason = "non_comparable", f"registration contract is {execution}"
        else:
            status, reason = "non_comparable", "no standalone benchmark adapter for this registered overload"
        entries.append(
            {"schema": schema, "family": _family(schema), "status": status, "reason": reason}
        )
    for schema in sorted(_custom_registration_inventory(root)):
        if schema not in {entry["schema"] for entry in entries}:
            measurable = schema in _CASE_SCHEMAS
            entries.append(
                {
                    "schema": schema,
                    "family": "custom",
                    "status": "measurable" if measurable else "unbenchmarked",
                    "reason": (
                        "case factory available"
                        if measurable
                        else "benchmark case adapter has not been added"
                    ),
                }
            )
    entries.sort(key=lambda entry: entry["schema"])
    catalog = {"schema_version": CASE_SCHEMA_VERSION, "entries": entries}
    aten_registrations, _ = _source_registration_inventory(root)
    validate_operator_coverage(
        catalog, aten_registrations, _custom_registration_inventory(root)
    )
    return catalog


def _runtime():
    import torch
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        raise RuntimeError("no suitable Vulkan device is available")
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    return torch, pytorch_vulkan, device


def _shape(profile: str, small: tuple[int, ...], large: tuple[int, ...]) -> tuple[int, ...]:
    if profile == "small":
        return small
    if profile == "large":
        return large
    raise ValueError("shape_profile must be 'small' or 'large'")


def _operator_call(torch, schema: str, args):
    if schema.startswith("pytorch_vulkan::"):
        return getattr(torch.ops.pytorch_vulkan, schema.split("::", 1)[1])(*args)
    name, overload = schema.removeprefix("aten::").split(".", 1)
    if schema in _OUTPUT_SCHEMAS:
        op = getattr(getattr(torch.ops.aten, name), overload)
        if schema == "aten::add.Scalar_out":
            return op(args[0], args[1], args[2], out=args[3])
        if schema in {"aten::add.out", "aten::sub.out", "aten::mul.out"}:
            return op(args[0], args[1], out=args[2])
        if schema in {"aten::sub.Scalar_out", "aten::rsub.Scalar_out"}:
            return op(args[0], args[1], args[2], out=args[3])
        if schema == "aten::mul.Scalar_out":
            return op(args[0], args[1], out=args[2])
        if schema == "aten::lerp.Scalar_out":
            return op(args[0], args[1], args[2], out=args[3])
        if schema == "aten::sqrt.out":
            return op(args[0], out=args[1])
        if schema in {"aten::amax.out", "aten::amin.out"}:
            return op(args[0], args[1], args[2], out=args[3])
        if schema == "aten::prod.int_out":
            return op(args[0], args[1], args[2], out=args[3])
        if schema in {"aten::_softmax.out", "aten::_log_softmax.out"}:
            return op(args[0], args[1], args[2], out=args[3])
        if schema == "aten::addmm.out":
            return op(args[0], args[1], args[2], out=args[3])
    if name in {"gelu_backward", "sigmoid_backward", "tanh_backward"}:
        grad_output, saved, output = args
        kwargs = {"grad_input": output}
        if name == "gelu_backward":
            kwargs["approximate"] = "tanh"
        return getattr(getattr(torch.ops.aten, name), overload)(grad_output, saved, **kwargs)
    if name == "gelu":
        return getattr(getattr(torch.ops.aten, name), overload)(args[0], approximate="tanh")
    if name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
        grad_output, output, dim, dtype, result = args
        keyword = "grad_input" if name == "_softmax_backward_data" else "out"
        return getattr(getattr(torch.ops.aten, name), overload)(grad_output, output, dim, dtype, **{keyword: result})
    return getattr(getattr(torch.ops.aten, name), overload)(*args)


def _arguments(torch, schema: str, profile: str, seed: int):
    torch.random.default_generator.manual_seed(seed)
    name = schema.split("::", 1)[1].split(".", 1)[0]
    if schema == "aten::masked_select.default" and profile in _MASKED_SELECT_PROFILES:
        size = 0 if profile == "zero" else 4096 if profile.startswith("medium_") else 65536
        values = torch.randn(size)
        indices = torch.arange(size)
        if profile.endswith("_sparse"):
            mask = indices % 64 == 0
        elif profile.endswith("_half"):
            mask = indices % 2 == 0
        elif profile.endswith("_dense"):
            mask = indices % 64 != 0
        else:
            mask = torch.zeros(size, dtype=torch.bool)
        return (values, mask)
    element_shape = _shape(profile, (32,), (1 << 18,))
    if name in {"mm", "addmm", "linear"}:
        m, k, n = (16, 8, 12) if profile == "small" else (256, 128, 192)
        x, weight = torch.randn(m, k), torch.randn(n, k)
        bias = torch.randn(n)
        if name == "addmm" and schema == "aten::addmm.out":
            return (torch.randn(m, n), x, weight.t().contiguous(), torch.empty(m, n))
        if name == "mm":
            return (x, weight.t().contiguous())
        if name == "linear":
            return (x, weight, bias)
        return (torch.randn(m, n), x, weight.t().contiguous())
    if name == "bmm":
        b, m, k, n = (2, 8, 8, 6) if profile == "small" else (8, 64, 64, 48)
        return (torch.randn(b, m, k), torch.randn(b, k, n))
    if name == "convolution":
        batch = 2 if profile == "small" else 16
        x, weight, bias = torch.randn(batch, 3, 32, 32), torch.randn(8, 3, 3, 3), torch.randn(8)
        return (x, weight, bias, [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    if name == "_adaptive_avg_pool2d":
        return (torch.randn(2, 4, 8, 8), [1, 1])
    if name == "native_batch_norm":
        x = torch.randn((2, 4) if profile == "small" else (2, 4, 2, 2))
        return (x, torch.randn(4), torch.randn(4), torch.randn(4), torch.rand(4) + 1, True, 0.1, 1e-5)
    if name in {"gelu_backward", "sigmoid_backward", "tanh_backward"}:
        x = torch.randn(element_shape)
        grad_output = torch.randn_like(x)
        saved = torch.sigmoid(x) if name == "sigmoid_backward" else torch.tanh(x) if name == "tanh_backward" else x
        return (grad_output, saved, torch.empty_like(x))
    if name == "_adaptive_avg_pool2d_backward":
        return (torch.randn(2, 4, 1, 1), torch.randn(2, 4, 8, 8))
    if name in {"_softmax_backward_data", "_log_softmax_backward_data"}:
        x = torch.randn(2, 2)
        output = torch.softmax(x, dim=1) if name == "_softmax_backward_data" else torch.log_softmax(x, dim=1)
        return (torch.randn_like(output), output, 1, torch.float32, torch.empty_like(x))
    if name in {"mean", "sum", "amax", "amin", "prod", "argmax", "_softmax", "_log_softmax"}:
        x = torch.randn(_shape(profile, (16, 32), (1024, 512)))
        if schema in {"aten::amax.out", "aten::amin.out"}:
            return (x, [1], False, torch.empty(x.shape[0]))
        if schema == "aten::prod.int_out":
            return (x.abs() + 0.2, 1, False, torch.empty(x.shape[0]))
        if schema in {"aten::_softmax.out", "aten::_log_softmax.out"}:
            return (x, 1, False, torch.empty_like(x))
        if name in {"mean", "sum"}:
            return (x, [1], False)
        if name in {"amax", "amin", "prod"}:
            return (x, [1]) if name != "prod" else (x, 1, False)
        if name == "argmax":
            return (x, 1, False)
        return (x, -1, False)
    if name == "mse_loss":
        x = torch.randn(_shape(profile, (16, 8), (1024, 64)))
        return (x, torch.randn_like(x), 1)
    if name == "mse_loss_backward":
        x = torch.randn(_shape(profile, (16, 8), (1024, 64)))
        return (torch.ones(()), x, torch.randn_like(x), 1)
    if name == "nll_loss_forward":
        x = torch.log_softmax(torch.randn(2, 3), dim=1)
        return (x, torch.randint(3, (2,), dtype=torch.int64), None, 1, -100)
    if name == "nll_loss_backward":
        x = torch.log_softmax(torch.randn(2, 3), dim=1)
        target = torch.tensor([0, 2], dtype=torch.int64)
        loss, total_weight = torch.ops.aten.nll_loss_forward.default(x, target, None, 1, -100)
        return (torch.ones_like(loss), x, target, None, 1, -100, total_weight)
    if schema in _OUTPUT_SCHEMAS:
        x = torch.rand(element_shape) + 0.5 if name in {"sqrt", "div"} else torch.randn(element_shape)
        if schema in {"aten::add.Scalar_out", "aten::sub.Scalar_out", "aten::rsub.Scalar_out"}:
            return (x, 2.0, 1.0, torch.empty_like(x))
        if schema in {"aten::add.out", "aten::sub.out", "aten::mul.out"}:
            return (x, torch.randn_like(x), torch.empty_like(x))
        if schema == "aten::mul.Scalar_out":
            return (x, 2.0, torch.empty_like(x))
        if schema == "aten::lerp.Scalar_out":
            return (x, torch.randn_like(x), 0.25, torch.empty_like(x))
        if schema == "aten::sqrt.out":
            return (x, torch.empty_like(x))
        if schema in {"aten::amax.out", "aten::amin.out", "aten::prod.int_out"}:
            value = torch.rand(8, 16) + 0.2 if schema == "aten::prod.int_out" else torch.randn(8, 16)
            return (value, 1, False, torch.empty(8))
        if schema in {"aten::_softmax.out", "aten::_log_softmax.out"}:
            value = torch.randn(2, 2)
            return (value, 1, False, torch.empty_like(value))
        if schema == "aten::addmm.out":
            return (torch.randn(16, 12), torch.randn(16, 8), torch.randn(8, 12), torch.empty(16, 12))
    if schema in _INPLACE_SCHEMAS:
        x = torch.randn(element_shape)
        if name == "zero_":
            return (x,)
        if name in {"addcmul_", "addcdiv_"}:
            return (x, torch.randn_like(x), torch.randn_like(x))
        if schema in {"aten::add_.Tensor", "aten::lerp_.Scalar"}:
            return (x, torch.randn_like(x), 0.25) if name == "lerp_" else (x, torch.randn_like(x))
        if schema == "aten::copy_.default":
            return (torch.empty_like(x), torch.randn_like(x))
        return (x, 2.0)
    if schema == "aten::masked_select.default":
        value = torch.randn(element_shape)
        return (value, value > 0)
    if name == "convolution_backward":
        x, weight, bias = torch.randn(2, 3, 32, 32), torch.randn(8, 3, 3, 3), torch.randn(8)
        output = torch.nn.functional.conv2d(x, weight, bias, padding=1)
        return (torch.randn_like(output), x, weight, [8], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, True])
    if name == "native_batch_norm_backward":
        x = torch.randn(2, 4)
        weight, bias, running_mean, running_var = torch.randn(4), torch.randn(4), torch.randn(4), torch.rand(4) + 1
        _, save_mean, save_invstd = torch.ops.aten.native_batch_norm.default(x, weight, bias, running_mean, running_var, True, 0.1, 1e-5)
        return (torch.randn_like(x), x, weight, running_mean, running_var, save_mean, save_invstd, True, 1e-5, [True, True, True])
    if name in {"zero_", "addcmul_", "addcdiv_"}:
        x = torch.randn(element_shape)
        if name == "zero_":
            return (x,)
        return (x, torch.randn_like(x), torch.randn_like(x))
    if name in {"view", "reshape", "_reshape_alias", "as_strided"}:
        x = torch.randn(4, 8)
        if name == "as_strided":
            return (x, [2, 4], [8, 1], 0)
        if name == "_reshape_alias":
            return (x, [2, 16], [16, 1])
        return (x, [2, 16])
    if name == "stack":
        return ([torch.randn(4, 8), torch.randn(4, 8)], 0)
    if schema == "pytorch_vulkan::rnn_sequence":
        batch, sequence, input_size, hidden = (2, 4, 8, 16) if profile == "small" else (16, 64, 128, 256)
        inputs = torch.randn(batch, sequence, input_size)
        if profile == "large":
            weight = torch.randn(hidden, input_size) / input_size**0.5
            recurrent = torch.randn(hidden, hidden) / hidden**0.5
            bias = torch.randn(hidden) * 0.1
        else:
            weight, recurrent, bias = torch.randn(hidden, input_size), torch.randn(hidden, hidden), torch.randn(hidden)
        return (inputs, weight, recurrent, bias)
    if schema == "pytorch_vulkan::linear_relu":
        return (torch.randn(16, 8), torch.randn(12, 8), torch.randn(12))
    if name == "add":
        x = torch.randn(element_shape)
        return (x, torch.randn_like(x)) if schema.endswith("Tensor") else (x, 2.0)
    if name in {"sub", "mul", "div", "rsub"}:
        x = torch.rand(element_shape) + 0.2 if name == "div" else torch.randn(element_shape)
        if schema == "aten::div.Tensor":
            return (x, torch.tensor(2.0))
        if schema.endswith("Tensor"):
            y = torch.rand_like(x) + 0.2 if name == "div" else torch.randn_like(x)
            return (x, y)
        return (x, 2.0)
    if name in {"abs", "neg", "relu", "sigmoid", "tanh", "gelu"}:
        return (torch.randn(element_shape),)
    raise KeyError(schema)


def _clone_args(args, device, require_grad=False):
    cloned = []
    for value in args:
        if hasattr(value, "to"):
            result = value.detach().clone().to(device)
            if require_grad and result.is_floating_point():
                result.requires_grad_()
            cloned.append(result)
        elif isinstance(value, list) and value and hasattr(value[0], "to"):
            cloned.append([item.detach().clone().to(device) for item in value])
        else:
            cloned.append(value)
    return tuple(cloned)


def _tensor_leaves(value):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensor_leaves(item)
    elif hasattr(value, "detach"):
        yield value


def _compare_results(torch, cpu_result, vulkan_result, rtol, atol):
    cpu_leaves = list(_tensor_leaves(cpu_result))
    vk_leaves = list(_tensor_leaves(vulkan_result))
    if len(cpu_leaves) != len(vk_leaves):
        return {"passed": False, "reason": "result structures differ", "max_abs_difference": None}
    maximum = 0.0
    try:
        for cpu, vk in zip(cpu_leaves, vk_leaves):
            actual = vk.detach().to("cpu")
            expected = cpu.detach().to("cpu")
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
            if actual.numel():
                maximum = max(maximum, float((actual - expected).abs().max()))
    except AssertionError as error:
        return {"passed": False, "reason": str(error), "max_abs_difference": maximum}
    return {"passed": True, "reason": None, "max_abs_difference": maximum}


def build_case(schema: str, shape_profile: str, seed: int) -> dict:
    if schema not in _CASE_SCHEMAS:
        raise KeyError(f"no standalone case factory for {schema}")
    torch, pytorch_vulkan, vk_device = _runtime()
    template = _arguments(torch, schema, shape_profile, seed)
    base = {
        "schema": schema,
        "family": _family(schema),
        "phase": "backward" if "_backward" in schema else "forward",
        "seed": seed,
        "status": "measurable",
        "input_metadata": {
            "shape": [list(value.shape) for value in _tensor_leaves(template)],
            "dtype": "float32",
            "stride": [list(value.stride()) for value in _tensor_leaves(template)],
            "shape_profile": shape_profile,
        },
        "vulkan_api": pytorch_vulkan._C,
        "gpu_timing_not_applicable": schema in _METADATA_SCHEMAS,
    }

    cached_mask = {}

    def prepare_pair(run_seed):
        source = _arguments(torch, schema, shape_profile, run_seed)
        if schema == "aten::div.Tensor":
            return source, (source[0].to(vk_device), source[1].clone())
        if schema == "aten::masked_select.default":
            if shape_profile == "zero":
                # Preserve an actual Vulkan allocation for metadata validation
                # while exposing the deterministic zero-length logical inputs.
                vk_values = torch.empty((1,), dtype=source[0].dtype, device=vk_device)[:0]
                vk_mask = torch.empty((1,), dtype=torch.bool, device=vk_device)[:0]
                return source, (vk_values, vk_mask)
            vk_values = source[0].to(vk_device)
            if run_seed not in cached_mask:
                cached_mask[run_seed] = source[1].to(vk_device)
            return source, (vk_values, cached_mask[run_seed])
        return source, _clone_args(source, vk_device)

    def cpu_call(args):
        if schema == "pytorch_vulkan::rnn_sequence":
            inputs, weight, recurrent, bias = args
            state = torch.zeros((inputs.shape[0], weight.shape[0]), dtype=inputs.dtype)
            states = []
            for step in inputs.transpose(0, 1).unbind(0):
                state = torch.tanh(torch.nn.functional.linear(step, weight, bias) + torch.nn.functional.linear(state, recurrent))
                states.append(state)
            return torch.stack(states, dim=1)
        if schema == "pytorch_vulkan::linear_relu":
            inputs, weight, bias = args
            return torch.nn.functional.relu(torch.nn.functional.linear(inputs, weight, bias))
        return _operator_call(torch, schema, args)

    def vulkan_call(args):
        if schema in _INPLACE_SCHEMAS:
            pytorch_vulkan._C.begin_training_step()
            try:
                result = _operator_call(torch, schema, args)
                pytorch_vulkan._C.end_training_step()
                return result
            except BaseException:
                pytorch_vulkan._C.cancel_training_step()
                raise
        return _operator_call(torch, schema, args)

    base.update({
        "prepare_pair": prepare_pair,
        "cpu_call": cpu_call,
        "vulkan_call": vulkan_call,
        "compare": lambda cpu, vk, rtol, atol: _compare_results(torch, cpu, vk, rtol, atol),
        "expected_scopes": {"operator", "gemm", "training"},
    })

    if schema in _AUTOGRAD_SCHEMAS and schema not in _METADATA_SCHEMAS:
        base["backward_case"] = _make_backward_case(base, torch, vk_device)
    return base


def _make_backward_case(forward_case, torch, vk_device):
    backward = dict(forward_case)
    backward["phase"] = "backward"

    def prepare_pair(seed):
        source = _arguments(torch, forward_case["schema"], forward_case["input_metadata"]["shape_profile"], seed)

        def clone_leaf(value, device, requires_grad):
            if isinstance(value, list):
                return [clone_leaf(item, device, requires_grad) for item in value]
            if hasattr(value, "detach"):
                result = value.detach().clone().to(device)
                if requires_grad and result.is_floating_point():
                    result.requires_grad_()
                return result
            return value

        schema = forward_case["schema"]

        def differentiable_argument(index):
            if schema == "aten::native_batch_norm.default":
                return index in {0, 1, 2}
            if schema == "aten::div.Tensor":
                return index == 0
            return True

        cpu_args = tuple(
            clone_leaf(value, "cpu", differentiable_argument(index))
            for index, value in enumerate(source)
        )
        vk_args = tuple(
            clone_leaf(
                value,
                "cpu" if schema == "aten::div.Tensor" and index == 1 else vk_device,
                differentiable_argument(index),
            )
            for index, value in enumerate(source)
        )
        cpu_output = forward_case["cpu_call"](cpu_args)
        vk_output = forward_case["vulkan_call"](vk_args)
        cpu_primary = next(_tensor_leaves(cpu_output))
        vk_primary = next(_tensor_leaves(vk_output))
        return (
            {"output": cpu_output, "args": cpu_args, "loss": cpu_primary.sum() if cpu_primary.numel() != 1 else cpu_primary},
            {"output": vk_output, "args": vk_args, "loss": vk_primary.sum() if vk_primary.numel() != 1 else vk_primary},
        )

    def backward_result(payload):
        payload["loss"].backward()
        output, args = payload["output"], payload["args"]
        return {"output": output, "gradients": [arg.grad for arg in args if getattr(arg, "requires_grad", False)]}

    backward.update({
        "prepare_pair": prepare_pair,
        "cpu_call": backward_result,
        "vulkan_call": backward_result,
        "compare": lambda cpu, vk, rtol, atol: _compare_results(torch, cpu, vk, rtol, atol),
        "backward_case": None,
    })
    return backward


def iter_cases(catalog: dict, profiles=("small", "large")):
    """Yield measurable profile/phase cases and explicit records for all others."""
    for entry in catalog["entries"]:
        if entry["status"] != "measurable":
            yield {
                "schema": entry["schema"],
                "family": entry["family"],
                "phase": "forward",
                "status": "non_comparable",
                "reason": entry["reason"],
            }
            continue
        case_profiles = (*profiles, *_MASKED_SELECT_PROFILES) if (
            entry["schema"] == "aten::masked_select.default" and "large" in profiles
        ) else profiles
        for profile in case_profiles:
            case = build_case(entry["schema"], profile, seed=1729)
            case["catalog_status"] = entry["status"]
            yield case
            if case.get("backward_case") is not None:
                yield case["backward_case"]


def validate_operator_coverage(
    catalog: dict, aten_registrations: set[str], custom_registrations: set[str]
) -> None:
    entries = catalog.get("entries")
    if not isinstance(entries, list):
        raise ValueError("catalog entries must be a list")
    schemas = [entry.get("schema") for entry in entries]
    if len(schemas) != len(set(schemas)):
        raise ValueError("catalog contains duplicate schemas")
    for index, entry in enumerate(entries):
        if entry.get("status") == "non_comparable" and not entry.get("reason", "").strip():
            raise ValueError(f"catalog entry {index} non_comparable status requires a reason")
    expected = set(aten_registrations) | set(custom_registrations)
    actual = set(schemas)
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise ValueError(f"operator catalog coverage mismatch: missing={missing}, extra={extra}")
