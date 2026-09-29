"""Generate docs/vulkan_capabilities.json.

Derived fields come from the C++ registrations and the conformance registry.
Declared fields come from tools.vulkan_capability_declarations. Output is
deterministic so a drift test can byte-compare it against the committed file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/python"))

from tools.validate_vulkan_capabilities import (  # noqa: E402
    REQUIRED_ENTRY_KEYS,
    _source_registration_inventory,
)  # noqa: E402
from tools.vulkan_capability_declarations import DECLARATIONS  # noqa: E402

MANIFEST = ROOT / "docs/vulkan_capabilities.json"

_COMMITTED: dict[str, dict] = {
    entry["schema"]: entry
    for entry in json.loads(MANIFEST.read_text())["entries"]
}

ENTRY_ORDER = (
    'aten::bmm.default',
    'aten::_adaptive_avg_pool2d.default',
    'aten::_adaptive_avg_pool2d_backward.default',
    'aten::_copy_from.default',
    'aten::_log_softmax.default',
    'aten::_log_softmax.out',
    'aten::_log_softmax_backward_data.out',
    'aten::_reshape_alias.default',
    'aten::_softmax.default',
    'aten::_softmax.out',
    'aten::_softmax_backward_data.out',
    'aten::_to_copy.default',
    'aten::abs.default',
    'aten::add.Scalar',
    'aten::add.Scalar_out',
    'aten::add.Tensor',
    'aten::add.out',
    'aten::add_.Tensor',
    'aten::addcdiv_.default',
    'aten::addcmul_.default',
    'aten::addmm.default',
    'aten::addmm.out',
    'aten::amax.default',
    'aten::amax.out',
    'aten::amin.default',
    'aten::amin.out',
    'aten::argmax.default',
    'aten::as_strided.default',
    'aten::convolution.default',
    'aten::convolution_backward.default',
    'aten::copy_.default',
    'aten::div.Tensor',
    'aten::empty.memory_format',
    'aten::empty_strided.default',
    'aten::gelu.default',
    'aten::gelu_backward.grad_input',
    'aten::lerp.Scalar_out',
    'aten::lerp_.Scalar',
    'aten::linear.default',
    'aten::masked_select.default',
    'aten::mean.dim',
    'aten::mm.default',
    'aten::mse_loss.default',
    'aten::mse_loss_backward.default',
    'aten::mul.Scalar',
    'aten::mul.Scalar_out',
    'aten::mul.Tensor',
    'aten::mul.out',
    'aten::mul_.Scalar',
    'aten::native_batch_norm.default',
    'aten::native_batch_norm_backward.default',
    'aten::neg.default',
    'aten::nll_loss_backward.default',
    'aten::nll_loss_forward.default',
    'aten::prod.dim_int',
    'aten::prod.int_out',
    'aten::relu.default',
    'aten::reshape.default',
    'aten::rsub.Scalar',
    'aten::rsub.Scalar_out',
    'aten::sigmoid.default',
    'aten::sigmoid_backward.grad_input',
    'aten::sqrt.out',
    'aten::sub.Scalar',
    'aten::sub.Scalar_out',
    'aten::sub.Tensor',
    'aten::sub.out',
    'aten::sum.dim_IntList',
    'aten::tanh.default',
    'aten::tanh_backward.grad_input',
    'aten::view.default',
    'aten::zero_.default',
    'aten::_cat.default',
    'aten::_copy_from_and_resize.default',
    'aten::_local_scalar_dense.default',
    'aten::_native_multi_head_attention.default',
    'aten::_native_multi_head_attention.out',
    'aten::_transform_bias_rescale_qkv.default',
    'aten::_upsample_nearest_exact2d.out',
    'aten::_upsample_nearest_exact2d_backward.grad_input',
    'aten::abs.out',
    'aten::add_.Scalar',
    'aten::addcdiv.out',
    'aten::addcmul.out',
    'aten::arange.start_out',
    'aten::argmax.out',
    'aten::atan.out',
    'aten::avg_pool2d.out',
    'aten::avg_pool2d_backward.grad_input',
    'aten::bernoulli_.float',
    'aten::binary_cross_entropy.default',
    'aten::binary_cross_entropy_backward.default',
    'aten::binary_cross_entropy_backward.grad_input',
    'aten::bitwise_and.Tensor_out',
    'aten::bitwise_not.out',
    'aten::bitwise_or.Tensor_out',
    'aten::bitwise_xor.Tensor_out',
    'aten::bmm.out',
    'aten::cat.out',
    'aten::ceil.default',
    'aten::ceil.out',
    'aten::clamp.out',
    'aten::clamp_min.out',
    'aten::convolution_backward_overrideable.default',
    'aten::convolution_overrideable.default',
    'aten::div.out',
    'aten::dot.default',
    'aten::eq.Scalar_out',
    'aten::eq.Tensor_out',
    'aten::exp.out',
    'aten::fill_.Scalar',
    'aten::ge.Scalar_out',
    'aten::ge.Tensor_out',
    'aten::gelu.out',
    'aten::gt.Scalar',
    'aten::gt.Scalar_out',
    'aten::gt.Tensor_out',
    'aten::hardsigmoid.out',
    'aten::hardsigmoid_backward.grad_input',
    'aten::hardswish_.default',
    'aten::hardswish_backward.default',
    'aten::hardtanh.default',
    'aten::hardtanh_.default',
    'aten::hardtanh_backward.default',
    'aten::isfinite.out',
    'aten::le.Scalar_out',
    'aten::le.Tensor_out',
    'aten::leaky_relu.out',
    'aten::leaky_relu_backward.grad_input',
    'aten::log.out',
    'aten::log_sigmoid_backward.default',
    'aten::log_sigmoid_backward.grad_input',
    'aten::log_sigmoid_forward.default',
    'aten::log_sigmoid_forward.output',
    'aten::logit.default',
    'aten::logit.out',
    'aten::lt.Scalar',
    'aten::lt.Scalar_out',
    'aten::lt.Tensor_out',
    'aten::max.default',
    'aten::max_pool2d_with_indices.default',
    'aten::maximum.out',
    'aten::mean.default',
    'aten::mean.out',
    'aten::min.default',
    'aten::minimum.out',
    'aten::mm.out',
    'aten::mul_.Tensor',
    'aten::native_dropout.default',
    'aten::native_dropout_backward.default',
    'aten::native_layer_norm.default',
    'aten::native_layer_norm_backward.default',
    'aten::ne.Scalar_out',
    'aten::ne.Tensor',
    'aten::ne.Tensor_out',
    'aten::neg.out',
    'aten::nll_loss_backward.grad_input',
    'aten::nll_loss_forward.output',
    'aten::normal_.default',
    'aten::pow.Tensor_Scalar_out',
    'aten::reciprocal.out',
    'aten::relu.out',
    'aten::resize_.default',
    'aten::round.out',
    'aten::set_.source_Storage',
    'aten::set_.source_Storage_storage_offset',
    'aten::sgn.out',
    'aten::sigmoid.out',
    'aten::sigmoid_.default',
    'aten::silu.out',
    'aten::silu_backward.grad_input',
    'aten::sub_.Scalar',
    'aten::sub_.Tensor',
    'aten::sum.IntList_out',
    'aten::sum.default',
    'aten::tanh.out',
    'aten::tanh_.default',
    'aten::threshold_backward.grad_input',
    'aten::uniform_.default',
    'aten::upsample_bilinear2d.out',
    'aten::upsample_bilinear2d_backward.grad_input',
    'aten::upsample_nearest2d.out',
    'aten::upsample_nearest2d_backward.grad_input',
    'aten::abs_.default',
    'aten::neg_.default',
    'aten::relu_.default',
    'aten::stack.default',
)

def _case_index() -> dict[str, dict[str, object]]:
    """Schema -> the declaration fields its supported cases contribute."""
    from vulkan_conformance import ALL_CASES

    index: dict[str, dict[str, object]] = {}
    for case in ALL_CASES:
        entry = index.setdefault(
            case.declaration_id, {"test_cases": [], "tests": set(), "shapes": set()}
        )
        entry["test_cases"].append(
            {"name": case.name, "supported": case.supported}
        )
        if not case.supported:
            continue
        entry["tests"].add("tests/python/test_vulkan_conformance.py")
        entry["tests"].add("tests/python/test_vulkan_operator_capabilities.py")
        for shape in case.declared_shapes:
            entry["shapes"].add(shape)
    return index


def build_manifest() -> dict:
    schemas = set(DECLARATIONS)
    cases = _case_index()
    entries = []
    for schema in ENTRY_ORDER:
        if schema not in schemas:
            continue
        declared = DECLARATIONS[schema]
        entry = dict(declared)
        entry["schema"] = schema
        entry["test_cases"] = list(_COMMITTED[schema]["test_cases"])
        entry["tests"] = list(_COMMITTED[schema]["tests"])
        case = cases.get(schema, {"shapes": set()})
        entry["shape_constraints"] = sorted(case["shapes"]) or ["unwitnessed"]
        entries.append(entry)
    return {"entries": entries, "version": 1}


def render(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def manifest_matches(committed_text: str, generated_manifest: dict) -> bool:
    """Whether committed bytes exactly match deterministic generated output."""
    return committed_text == render(generated_manifest)


def write_manifest() -> None:
    """Write atomically, so a failure never leaves a half-written manifest."""
    text = render(build_manifest())
    temporary = MANIFEST.with_suffix(".json.tmp")
    temporary.write_text(text)
    os.replace(temporary, MANIFEST)


def _validate_declarations() -> None:
    """Reject registrations that are missing declarations."""
    registrations, _ = _source_registration_inventory(ROOT / "src")
    missing = sorted(registrations - set(DECLARATIONS))
    if missing:
        raise SystemExit(
            "vulkan_capability_declarations.py is missing declarations for: "
            + ", ".join(missing)
        )
    required = set(REQUIRED_ENTRY_KEYS) - {
        "schema",
        "test_cases",
        "tests",
        "shape_constraints",
    }
    for schema, declaration in sorted(DECLARATIONS.items()):
        keys = set(declaration)
        if keys != required:
            raise SystemExit(
                f"{schema}: declaration keys must be exactly {sorted(required)}, "
                f"got {sorted(keys)}"
            )


def _roadmap_schemas() -> set[str]:
    """Return declared schemas that are not implemented registrations yet."""
    registrations, _ = _source_registration_inventory(ROOT / "src")
    return set(DECLARATIONS) - registrations


if __name__ == "__main__":
    _validate_declarations()
    write_manifest()
