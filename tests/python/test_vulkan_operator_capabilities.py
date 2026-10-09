import os
import re
from pathlib import Path

import pytest
import pytorch_vulkan
import torch
from vulkan_conformance import (
    ALL_CASES,
    DECLARED_OPERATION_MANIFEST,
    PROMOTED_SCALAR_OUT_SCHEMAS,
    REQUIRED_SCALAR_OUT_REJECTION_BOUNDARIES,
    ROADMAP_DEFERRED_REASON_BY_FAMILY,
    ROADMAP_DEFERRED_SCHEMAS,
    ROADMAP_OPERATION_FAMILIES,
    SCALAR_OUT_CONTRACT_MATRIX,
)

from tools.validate_vulkan_capabilities import (
    _extract_privateuse1_blocks,
    _parse_m_impl_registrations,
    load_manifest,
)
from tools.validate_vulkan_capabilities import (
    _source_registration_inventory as _shared_source_registration_inventory,
)
from tools.vulkan_capability_declarations import (
    STOCK_COMPOSITE_ROUTE_CONTRACT,
    STOCK_COMPOSITE_ROUTES,
    STOCK_LINEAR_ROUTE_CONTRACT,
    STOCK_LINEAR_ROUTE_CASES,
)


def _parse_m_impl_schemas(block):
    return {schema for schema, _ in _parse_m_impl_registrations(block)}


def _source_registration_inventory(root=Path("src")):
    return _shared_source_registration_inventory(root)[0]


def _source_registration_classifications(root=Path("src")):
    return _shared_source_registration_inventory(root)


_MANIFEST = load_manifest(Path("docs/vulkan_capabilities.json"))
EXPLICIT_REJECTED_SOURCE_SCHEMAS = frozenset(
    entry["schema"]
    for entry in _MANIFEST["entries"]
    if entry["status"] == "rejected"
    # Rejected for a *source-side* decision: the schema exists and is
    # registered, and we decline it. Entries rejected as
    # schema_absent_from_pytorch_dispatcher are a different category -- they
    # have no registration to be explicit about, so they must not appear in
    # the set compared against the source inventory below.
    and entry["reason"] != "schema_absent_from_pytorch_dispatcher"
)
DEFERRED_SOURCE_SCHEMAS = frozenset(
    entry["schema"] for entry in _MANIFEST["entries"] if entry["status"] == "deferred"
)


def _undeclared_source_registrations():
    source_inventory, explicit_rejected = _source_registration_classifications()
    return frozenset(
        source_inventory
        - DECLARED_OPERATION_MANIFEST
        - ROADMAP_DEFERRED_SCHEMAS
        - explicit_rejected
    )


def test_declared_manifest_matches_source_registrations():
    manifest = load_manifest(Path("docs/vulkan_capabilities.json"))
    assert {
        entry["schema"]
        for entry in manifest["entries"]
        if entry["status"] == "supported"
    } == DECLARED_OPERATION_MANIFEST
    classifications = (
        DECLARED_OPERATION_MANIFEST,
        DEFERRED_SOURCE_SCHEMAS,
        EXPLICIT_REJECTED_SOURCE_SCHEMAS,
    )
    assert classifications[0].isdisjoint(classifications[1])
    assert classifications[0].isdisjoint(classifications[2])
    assert classifications[1].isdisjoint(classifications[2])
    source_inventory, source_explicit_rejected = _source_registration_classifications()
    assert ROADMAP_DEFERRED_SCHEMAS == DEFERRED_SOURCE_SCHEMAS
    assert source_explicit_rejected == EXPLICIT_REJECTED_SOURCE_SCHEMAS
    assert EXPLICIT_REJECTED_SOURCE_SCHEMAS
    direct_supported = DECLARED_OPERATION_MANIFEST - set(STOCK_COMPOSITE_ROUTES)
    assert direct_supported == source_inventory - DEFERRED_SOURCE_SCHEMAS - source_explicit_rejected
    assert "aten::reshape.default" in STOCK_COMPOSITE_ROUTES
    assert "aten::reshape.default" not in source_inventory


def test_stock_composite_routes_are_an_explicit_source_owned_allowlist():
    assert set(STOCK_COMPOSITE_ROUTES) == {
        "aten::reshape.default",
        "aten::linear.default",
        "aten::matmul.default",
    }
    reshape_route = STOCK_COMPOSITE_ROUTES["aten::reshape.default"]
    linear_route = STOCK_COMPOSITE_ROUTES["aten::linear.default"]
    matmul_route = STOCK_COMPOSITE_ROUTES["aten::matmul.default"]
    assert reshape_route == STOCK_COMPOSITE_ROUTE_CONTRACT
    assert linear_route == STOCK_LINEAR_ROUTE_CONTRACT
    assert matmul_route["reference"]["source"] == (
        "aten/src/ATen/native/LinearAlgebra.cpp::_matmul_impl"
    )
    assert {item["schema"] for item in matmul_route["dependencies"]} == {
        "aten::dot.default", "aten::mv.default", "aten::mm.default", "aten::bmm.default",
        "aten::as_strided.default", "aten::_copy_from.default", "aten::sum.dim_IntList",
    }
    assert reshape_route["reference"] == {
        "pytorch_version": "2.4.0",
        "source": "aten/src/ATen/native/TensorShape.cpp::reshape_symint",
        "routes": [
            "view-compatible geometry -> _reshape_alias",
            "otherwise -> clone(MemoryFormat::Contiguous) -> _unsafe_view",
        ],
    }
    dependencies = {item["schema"]: item for item in reshape_route["dependencies"]}
    assert dependencies["aten::clone.default"]["dispatch"] == "stock_generated_privateuse1"
    assert dependencies["aten::clone.default"]["vulkan_leaf"] == "aten::copy_.default"
    assert dependencies["aten::_unsafe_view.default"]["dispatch"] == "stock_generated_privateuse1"
    assert dependencies["aten::_unsafe_view.default"]["vulkan_leaf"] == "aten::view.default"
    source_inventory, _ = _source_registration_classifications()
    assert "aten::clone.default" not in source_inventory
    assert "aten::_unsafe_view.default" not in source_inventory
    linear_dependencies = {
        item["schema"] for item in linear_route["dependencies"]
    }
    assert linear_dependencies == {
        "aten::matmul.default",
        "aten::mm.default",
        "aten::addmm.default",
        "aten::bmm.default",
        "aten::add_.Tensor",
        "aten::clone.default",
        "aten::reshape.default",
        "aten::view.default",
        "aten::as_strided.default",
        "aten::expand.default",
        "aten::_unsafe_view.default",
        "aten::t.default",
        "aten::sum_to_size.default",
    }
    assert linear_route["evidence_cases"] == [case["name"] for case in STOCK_LINEAR_ROUTE_CASES]
def test_gemm_declarations_cover_supported_frontends_and_deferred_forms():
    assert {
        "aten::mm.default",
        "aten::addmm.default",
        "aten::addmm.out",
        "aten::linear.default",
    } <= DECLARED_OPERATION_MANIFEST
    assert {
        "aten::bmm.out",
        "aten::mm.out",
    } <= ROADMAP_DEFERRED_SCHEMAS


def test_scalar_out_contract_matrix_matches_selected_manifest_schemas():
    selected = PROMOTED_SCALAR_OUT_SCHEMAS
    manifest = load_manifest(Path("docs/vulkan_capabilities.json"))
    manifest_supported = frozenset(
        entry["schema"]
        for entry in manifest["entries"]
        if entry["status"] == "supported"
    )
    assert selected == {
        "aten::add.Scalar",
        "aten::add.Scalar_out",
        "aten::add.out",
        "aten::sub.Scalar",
        "aten::rsub.Scalar",
        "aten::sub.Scalar_out",
        "aten::rsub.Scalar_out",
        "aten::sub.out",
        "aten::mul.Scalar",
        "aten::mul.Scalar_out",
        "aten::mul.out",
    }
    assert selected <= manifest_supported
    source_inventory, _ = _source_registration_classifications()
    registered_scalar_out = frozenset(
        schema
        for schema in source_inventory
        if schema.split("::", 1)[-1].split(".", 1)[0]
        in {"add", "sub", "rsub", "mul"}
        and schema.rsplit(".", 1)[-1] in {"Scalar", "Scalar_out", "out"}
    )
    assert selected == registered_scalar_out
    assert selected <= DECLARED_OPERATION_MANIFEST
    assert not selected & ROADMAP_DEFERRED_SCHEMAS
    assert all(case.status == "supported" for case in SCALAR_OUT_CONTRACT_MATRIX.values())


def test_scalar_out_contract_matrix_has_named_case_and_counter_contracts():
    assert len({case.case_name for case in SCALAR_OUT_CONTRACT_MATRIX.values()}) == len(
        SCALAR_OUT_CONTRACT_MATRIX
    )
    for schema, case in SCALAR_OUT_CONTRACT_MATRIX.items():
        assert case.schema == schema
        assert case.operand_order in {
            "tensor-scalar",
            "scalar-tensor",
            "tensor-tensor",
        }
        assert case.output_mode in {"functional", "out"}
        assert isinstance(case.cpu_expression, str)
        assert callable(case.cpu_expression)
        assert callable(case.cpu_reference)
        assert case.check_gradients == (case.output_mode == "functional")
        assert case.scalar in {2.0, -2.0, 0.0}
        assert case.alpha in {1.0, 0.5}
        assert case.expected_counters == {
            "compute": "positive",
            "vulkan_copy": 0,
            "explicit_transfer": 0,
            "fallback": 0,
        }
        assert case.empty_supported
        assert case.empty_expected_counters == {
            "compute": 0,
            "vulkan_copy": 0,
            "explicit_transfer": 0,
            "fallback": 0,
        }
        assert REQUIRED_SCALAR_OUT_REJECTION_BOUNDARIES <= case.rejection_boundaries
        assert case.cpu_reference is not None
        assert case.expected_counters["compute"] == "positive"


def test_scalar_out_promotions_have_positive_and_negative_conformance_coverage():
    cases_by_name = {case.name: case for case in ALL_CASES if case.supported}
    for schema in PROMOTED_SCALAR_OUT_SCHEMAS:
        contract = SCALAR_OUT_CONTRACT_MATRIX[schema]
        case = cases_by_name[contract.case_name]
        assert case.declaration_id == schema
        assert callable(case.cpu_reference)
        assert case.execution_mode == "compute"
    assert all(
        contract.rejection_boundaries
        for contract in SCALAR_OUT_CONTRACT_MATRIX.values()
    )
    assert {
        boundary
        for contract in SCALAR_OUT_CONTRACT_MATRIX.values()
        for boundary in contract.rejection_boundaries
    } >= {"alpha", "broadcast", "dtype", "device", "non_finite", "overlap"}


def test_registration_parser_accepts_formatting_variants():
    source = """
    TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m ) {
      m . impl ( "aten::view", &view );
      if (enabled) { m.impl("linear", &linear); }
    }
    """
    blocks = _extract_privateuse1_blocks(source)
    assert len(blocks) == 1
    assert _parse_m_impl_schemas(blocks[0]) == {
        "aten::view.default",
        "aten::linear.default",
    }


def test_registration_parser_ignores_comments_and_allows_comments_between_tokens():
    source = """
    // TORCH_LIBRARY_IMPL(aten, PrivateUse1, ignored) {
    //   m.impl("commented_out", &reject_commented_out);
    // }
    TORCH_LIBRARY_IMPL(/* before */ aten, PrivateUse1, /* before m */ m) {
      m /* between */ . /* tokens */ impl(
        "view" /* schema comment */, &view); // m.impl("fake", &fake);
      /* m.impl("block_commented", &reject_block_commented); */
      m.impl("linear", /* handler */ &linear);
    }
    """
    blocks = _extract_privateuse1_blocks(source)
    assert len(blocks) == 1
    assert _parse_m_impl_schemas(blocks[0]) == {
        "aten::view.default",
        "aten::linear.default",
    }


def test_registration_discovery_includes_non_cpp_source_files(tmp_path):
    source = tmp_path / "registration.hpp"
    source.write_text(
        "TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {\n"
        '  m.impl("header_only", &header_only);\n}\n'
    )
    assert _source_registration_inventory(tmp_path) == {"aten::header_only.default"}


def test_source_classifications_are_pairwise_disjoint():
    assert DECLARED_OPERATION_MANIFEST.isdisjoint(DEFERRED_SOURCE_SCHEMAS)
    assert DECLARED_OPERATION_MANIFEST.isdisjoint(EXPLICIT_REJECTED_SOURCE_SCHEMAS)
    assert DEFERRED_SOURCE_SCHEMAS.isdisjoint(EXPLICIT_REJECTED_SOURCE_SCHEMAS)


def test_source_registrations_cannot_be_supported_without_a_declaration():
    """Every source registration must be supported, rejected, or roadmap-deferred."""
    assert not _undeclared_source_registrations()
    source_inventory, explicit_rejected = _source_registration_classifications()
    source_supported = source_inventory - ROADMAP_DEFERRED_SCHEMAS - explicit_rejected
    assert source_supported | set(STOCK_COMPOSITE_ROUTES) == DECLARED_OPERATION_MANIFEST
    from vector_matmul_capability_evidence import VECTOR_MATMUL_CASES

    source_owned_cases = {case["schema"] for case in VECTOR_MATMUL_CASES}
    assert (
        {case.declaration_id for case in ALL_CASES if case.supported}
        | source_owned_cases
    ) - set(STOCK_COMPOSITE_ROUTES) == source_supported


def test_every_deferred_roadmap_schema_has_an_explicit_reason():
    assert set(ROADMAP_OPERATION_FAMILIES) == set(ROADMAP_DEFERRED_REASON_BY_FAMILY)
    assert ROADMAP_DEFERRED_SCHEMAS
    assert sum(map(len, ROADMAP_OPERATION_FAMILIES.values())) == len(
        ROADMAP_DEFERRED_SCHEMAS
    )
    for family, schemas in ROADMAP_OPERATION_FAMILIES.items():
        reason = ROADMAP_DEFERRED_REASON_BY_FAMILY[family]
        assert schemas
        assert isinstance(reason, str) and reason.strip()
        assert "deferred" in reason or "not implemented" in reason


def test_deferred_operations_are_explicitly_separate_from_declared_manifest():
    deferred = frozenset(
        {
            "aten::exp.default",
            "aten::double_add",
            "aten::double_mul",
        }
    )
    assert not DECLARED_OPERATION_MANIFEST & deferred


def test_conformance_registry_covers_declared_operator_families():
    families = {case.family for case in ALL_CASES}
    assert families >= {
        "unary",
        "binary",
        "reduction",
        "indexing",
        "view",
        "linear",
        "convolution",
        "pooling",
        "masked-select",
    }


def test_conformance_registry_covers_each_declared_rejection_boundary():
    names = {case.name for case in ALL_CASES if not case.supported}
    assert any("float16" in name for name in names)
    assert any("bool" in name for name in names)
    assert any("double" in name for name in names)
    assert any("mixed-device" in name for name in names)
    assert any("broadcast" in name for name in names)
    assert any("shape" in name for name in names)
    assert any("offset" in name for name in names)
    assert any("overlap" in name for name in names)
    assert any("invalid-out" in name for name in names)
    assert any("parameters" in name for name in names)
    assert any("unsupported-overload" in name for name in names)


def test_manifest_separates_public_inplace_forms_from_optimizer_updates():
    assert "aten::add_.Tensor" in DECLARED_OPERATION_MANIFEST
    assert "aten::mul_.Scalar" in DECLARED_OPERATION_MANIFEST


@pytest.mark.parametrize("mode", [torch.no_grad, torch.inference_mode])
@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_"])
def test_public_inplace_pointwise_is_supported_under_no_grad_and_inference_mode(
    vulkan_backend, mode, operation
):
    tensor = torch.ones(2, dtype=torch.float32, device=vulkan_backend)
    other = torch.full_like(tensor, 2.0)
    pytorch_vulkan._C.reset_execution_counters()

    with mode():
        if operation == "mul_":
            getattr(tensor, operation)(2.0)
        else:
            getattr(tensor, operation)(other)

    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = os.environ.get(
        "VULKAN_DEVICE", f"{torch._C._get_privateuse1_backend_name()}:0"
    )
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


def assert_vulkan_capability(operation, inputs, *, supported, error_pattern=None):
    """Run an operation and assert explicit support or rejection."""
    if supported:
        result = operation(*inputs)
        assert result.device.type == torch._C._get_privateuse1_backend_name()
        assert result.device.index == inputs[0].device.index
        return result

    pattern = error_pattern or r"Vulkan"
    with pytest.raises(RuntimeError, match=pattern) as error:
        operation(*inputs)
    assert type(error.value) is RuntimeError


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_baseline_remains_supported(vulkan_backend, operation):
    tensor = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float32).to(vulkan_backend)

    result = assert_vulkan_capability(operation, (tensor,), supported=True)

    assert result.dtype is torch.float32
    torch.testing.assert_close(result.cpu(), operation(tensor.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_baseline_remains_supported(vulkan_backend, operation):
    lhs = torch.tensor([1.0, -2.0], dtype=torch.float32).to(vulkan_backend)
    rhs = torch.tensor([3.0, 4.0], dtype=torch.float32).to(vulkan_backend)

    result = assert_vulkan_capability(operation, (lhs, rhs), supported=True)

    assert result.dtype is torch.float32
    torch.testing.assert_close(
        result.cpu(), operation(lhs.cpu(), rhs.cpu()), rtol=0, atol=0
    )


def test_float16_operator_request_is_rejected_without_cpu_fallback(vulkan_backend):
    expected = (
        "Vulkan float16 support is deferred; allocation cannot use float16 until its "
        "storage, transfer, shader, promotion, and autograd contracts are implemented"
    )
    with pytest.raises(RuntimeError, match=re.escape(expected)):
        torch.neg(torch.empty((2,), dtype=torch.float16, device=vulkan_backend))


def test_fallback_policy_seam_counts_attempt_and_strict_mode_rejects(vulkan_backend):
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.set_strict_mode(False)

    pytorch_vulkan._C.test_inject_fallback()
    assert pytorch_vulkan._C.fallback_count() == 1

    pytorch_vulkan._C.set_strict_mode(True)
    with pytest.raises(RuntimeError, match="strict fallback mode"):
        pytorch_vulkan._C.test_inject_fallback()
    assert pytorch_vulkan._C.fallback_count() == 2
    pytorch_vulkan._C.set_strict_mode(False)


def test_supported_and_rejected_operations_do_not_count_as_fallback(vulkan_backend):
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.set_strict_mode(False)
    tensor = torch.ones(2, dtype=torch.float32, device=vulkan_backend)

    torch.neg(tensor)
    assert pytorch_vulkan._C.fallback_count() == 0

    with pytest.raises(RuntimeError, match="Vulkan"):
        torch.neg(torch.empty((2,), dtype=torch.float16, device=vulkan_backend))
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu, torch.sub])
def test_unsupported_bool_operator_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    inputs = (
        (tensor,)
        if operation in [torch.neg, torch.abs, torch.relu]
        else (tensor, tensor)
    )

    assert_vulkan_capability(
        operation,
        inputs,
        supported=False,
        error_pattern=r"Vulkan.*float32|Vulkan.*dtype|support.*bool",
    )


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_gradient_contract(vulkan_backend, operation):
    cpu_input = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=True)
    vk_input = cpu_input.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu_input)
    vk_result = operation(vk_input)
    grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cpu_result.backward(grad)
    vk_result.backward(grad.to(vulkan_backend))

    assert vk_result.device == vk_input.device
    assert vk_result.dtype is torch.float32
    assert vk_result.shape == vk_input.shape
    assert vk_input.grad is not None
    assert vk_input.grad.device == vk_input.device
    assert vk_input.grad.dtype is torch.float32
    assert vk_input.grad.shape == vk_input.shape
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_gradient_contract(vulkan_backend, operation):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], dtype=torch.float32, requires_grad=True)
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], dtype=torch.float32, requires_grad=True)
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_()
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_result = operation(vk_lhs, vk_rhs)
    grad = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    cpu_result.backward(grad)
    vk_result.backward(grad.to(vulkan_backend))

    assert vk_result.device == vk_lhs.device
    assert vk_result.dtype is torch.float32
    assert vk_result.shape == vk_lhs.shape
    for vk_input, cpu_input in ((vk_lhs, cpu_lhs), (vk_rhs, cpu_rhs)):
        assert vk_input.grad is not None
        assert vk_input.grad.device == vk_input.device
        assert vk_input.grad.dtype is torch.float32
        assert vk_input.grad.shape == vk_input.shape
        torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)


def test_sum_default_float32_reduction_contract(vulkan_backend):
    cpu_input = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    vk_input = cpu_input.to(vulkan_backend)

    cpu_result = torch.sum(cpu_input)
    vk_result = torch.sum(vk_input)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_result.cpu(), cpu_result, rtol=0, atol=0)

    cpu_strided = cpu_input.t()
    vk_strided = vk_input.t()
    assert not vk_strided.is_contiguous()
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(
        torch.sum(vk_strided).cpu(), torch.sum(cpu_strided), rtol=0, atol=0
    )

    cpu_empty = torch.empty((0, 3), dtype=torch.float32)
    vk_empty_result = torch.sum(cpu_empty.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_empty_result.cpu(), torch.sum(cpu_empty), rtol=0, atol=0)
    assert vk_empty_result.item() == 0

    cpu_grad_input = cpu_input.clone().requires_grad_()
    vk_grad_input = cpu_input.to(vulkan_backend).requires_grad_()
    cpu_sum = torch.sum(cpu_grad_input)
    vk_sum = torch.sum(vk_grad_input)
    cpu_sum.backward()
    vk_sum.backward()
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_grad_input.grad.cpu(), cpu_grad_input.grad, rtol=0, atol=0)


def test_mean_default_float32_reduction_contract(vulkan_backend):
    cpu_input = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    vk_input = cpu_input.to(vulkan_backend)

    cpu_result = torch.mean(cpu_input)
    vk_result = torch.mean(vk_input)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_result.cpu(), cpu_result, rtol=1e-6, atol=1e-6)

    cpu_strided = cpu_input.t()
    vk_strided = vk_input.t()
    assert not vk_strided.is_contiguous()
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(
        torch.mean(vk_strided).cpu(), torch.mean(cpu_strided), rtol=1e-6, atol=1e-6
    )

    cpu_empty = torch.empty((0, 3), dtype=torch.float32)
    vk_empty_result = torch.mean(cpu_empty.to(vulkan_backend))
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(
        vk_empty_result.cpu(), torch.mean(cpu_empty), equal_nan=True
    )
    assert vk_empty_result.item() != vk_empty_result.item()


def test_sum_intlist_out_float32_reduction_contract(vulkan_backend):
    cpu_input = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    vk_input = cpu_input.to(vulkan_backend)
    cpu_out = torch.empty((), dtype=torch.float32)
    vk_out = torch.empty((), dtype=torch.float32).to(vulkan_backend)
    assert vk_out.is_contiguous()

    torch.sum(cpu_input, dim=[0, 1], out=cpu_out)
    torch.sum(vk_input, dim=[0, 1], out=vk_out)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_out.cpu(), cpu_out, rtol=0, atol=0)


def test_mean_out_float32_reduction_contract(vulkan_backend):
    cpu_input = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    vk_input = cpu_input.to(vulkan_backend)
    cpu_out = torch.empty((), dtype=torch.float32)
    vk_out = torch.empty((), dtype=torch.float32).to(vulkan_backend)
    assert vk_out.is_contiguous()

    torch.mean(cpu_input, dim=[0, 1], out=cpu_out)
    torch.mean(vk_input, dim=[0, 1], out=vk_out)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(vk_out.cpu(), cpu_out, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu])
def test_unary_float32_layout_shape_and_explicit_transfer_contract(
    vulkan_backend, operation
):
    non_contiguous = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    assert not non_contiguous.is_contiguous()
    result = assert_vulkan_capability(operation, (non_contiguous,), supported=True)
    assert result.shape == non_contiguous.shape

    zero_dimensional = torch.empty((), dtype=torch.float32, device=vulkan_backend)
    zero_result = assert_vulkan_capability(
        operation, (zero_dimensional,), supported=True
    )
    assert zero_result.dim() == 0

    source = torch.tensor([1.0, -2.0], dtype=torch.float32, device=vulkan_backend)
    result = assert_vulkan_capability(operation, (source,), supported=True)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    transferred = result.to("cpu")
    assert transferred.device.type == "cpu"
    torch.testing.assert_close(transferred, operation(source.cpu()), rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_float32_shape_and_explicit_transfer_contract(vulkan_backend, operation):
    lhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    mismatched_rhs = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    assert_vulkan_capability(
        operation, (lhs, mismatched_rhs), supported=False, error_pattern=r"size|shape"
    )

    rhs = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    result = assert_vulkan_capability(operation, (lhs, rhs), supported=True)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    assert result.to("cpu").device.type == "cpu"
