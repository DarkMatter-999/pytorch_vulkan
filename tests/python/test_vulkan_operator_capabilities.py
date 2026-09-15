import re
from pathlib import Path

import pytest
import torch

import pytorch_vulkan
from vulkan_conformance import ALL_CASES, DECLARED_OPERATION_MANIFEST


SOURCE_SCHEMA_ALIASES = {
    "argmax": "aten::argmax.default", "linear": "aten::linear.default",
    "convolution": "aten::convolution.default",
    "_adaptive_avg_pool2d": "aten::_adaptive_avg_pool2d.default",
    "neg": "aten::neg.default", "abs": "aten::abs.default",
    "relu": "aten::relu.default", "add.Tensor": "aten::add.Tensor",
    "sub.Tensor": "aten::sub.Tensor", "mul.Tensor": "aten::mul.Tensor",
    "as_strided": "aten::as_strided.default", "view": "aten::view.default",
    "_reshape_alias": "aten::_reshape_alias.default",
    "reshape": "aten::reshape.default",
    "masked_select": "aten::masked_select.default",
}

EXPLICIT_REJECTED_SOURCE_SCHEMAS = frozenset({
    "aten::abs_.default", "aten::neg_.default", "aten::relu_.default",
})


def _strip_cpp_comments(source):
    """Replace C++ comments with whitespace without changing string literals."""
    result = []
    index = 0
    quote = None
    escaped = False
    while index < len(source):
        char = source[index]
        next_char = source[index + 1] if index + 1 < len(source) else ""
        if quote:
            result.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == quote:
                quote = None
            index += 1
        elif char in ('"', "'"):
            quote = char
            result.append(char)
            index += 1
        elif char == "/" and next_char == "/":
            result.extend("  ")
            index += 2
            while index < len(source) and source[index] != "\n":
                result.append(" ")
                index += 1
        elif char == "/" and next_char == "*":
            result.extend("  ")
            index += 2
            while index < len(source):
                if source[index] == "*" and index + 1 < len(source) and source[index + 1] == "/":
                    result.extend("  ")
                    index += 2
                    break
                result.append("\n" if source[index] == "\n" else " ")
                index += 1
        else:
            result.append(char)
            index += 1
    return "".join(result)


def _extract_privateuse1_blocks(source):
    """Extract blocks while tolerating whitespace and nested braces."""
    source = _strip_cpp_comments(source)
    blocks = []
    macro = re.compile(
        r"TORCH_LIBRARY_IMPL\s*\(\s*aten\s*,\s*PrivateUse1\s*,\s*[^)]*\)"
    )
    for match in macro.finditer(source):
        opening = source.find("{", match.end())
        if opening < 0:
            continue
        depth = 0
        quote = None
        escaped = False
        for index in range(opening, len(source)):
            char = source[index]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in ('"', "'"):
                quote = char
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(source[opening + 1:index])
                    break
    return blocks


def _canonical_source_schema(identifier):
    identifier = identifier.removeprefix("aten::")
    return SOURCE_SCHEMA_ALIASES.get(
        identifier,
        f"aten::{identifier}" if "." in identifier else f"aten::{identifier}.default",
    )


def _parse_m_impl_schemas(block):
    return {
        schema for schema, _ in _parse_m_impl_registrations(block)
    }


def _parse_m_impl_registrations(block):
    return {
        (_canonical_source_schema(identifier), handler)
        for identifier, handler in re.findall(
            r'\bm\s*\.\s*impl\s*\(\s*"([^"]+)"\s*,\s*&?\s*([A-Za-z_]\w*(?:::\w+)*)',
            _strip_cpp_comments(block),
        )
    }


def _source_registration_inventory(root=Path("src")):
    registrations = set()
    for path in root.rglob("*"):
        if path.is_file():
            for block in _extract_privateuse1_blocks(path.read_text()):
                registrations.update(_parse_m_impl_schemas(block))
    return registrations


def _source_registration_classifications(root=Path("src")):
    registrations = set()
    explicit_rejected = set()
    for path in root.rglob("*"):
        if path.is_file():
            for block in _extract_privateuse1_blocks(path.read_text()):
                for schema, handler in _parse_m_impl_registrations(block):
                    registrations.add(schema)
                    if handler.split("::")[-1].startswith("reject_"):
                        explicit_rejected.add(schema)
    return registrations, explicit_rejected


def _source_deferred_schemas(root=Path("src")):
    registrations, explicit_rejected = _source_registration_classifications(root)
    return frozenset(registrations - DECLARED_OPERATION_MANIFEST - explicit_rejected)


DEFERRED_SOURCE_SCHEMAS = _source_deferred_schemas()


def test_declared_manifest_matches_matrix_and_source_registrations():
    matrix = Path("docs/vulkan_operator_capability_matrix.md").read_text()
    marker = re.search(
        r"Vulkan conformance supported schemas:\s*(.*?)\s*-->", matrix, re.DOTALL
    )
    assert marker is not None
    matrix_rows = {item.strip() for item in marker.group(1).replace("\n", "").split(",") if item.strip()}
    assert DECLARED_OPERATION_MANIFEST == matrix_rows
    matrix_declarations = {
        "aten::sum.dim_IntList": "`aten::sum`", "aten::mean.dim": "`aten::mean`",
        "aten::argmax.default": "`aten::argmax`", "aten::linear.default": "`aten::linear`",
        "aten::convolution.default": "`aten::convolution`",
        "aten::_adaptive_avg_pool2d.default": "`aten::_adaptive_avg_pool2d`",
        "aten::neg.default": "unary `neg`/`abs`/`relu`",
        "aten::abs.default": "unary `neg`/`abs`/`relu`",
        "aten::relu.default": "unary `neg`/`abs`/`relu`",
        "aten::add.Tensor": "pointwise `add`/`sub`/`mul`",
        "aten::sub.Tensor": "pointwise `add`/`sub`/`mul`",
        "aten::mul.Tensor": "pointwise `add`/`sub`/`mul`",
        "aten::as_strided.default": "`aten::as_strided`",
        "aten::view.default": "`aten::view`",
        "aten::_reshape_alias.default": "`aten::_reshape_alias`",
        "aten::reshape.default": "`aten::reshape`",
        "aten::masked_select.default": "`torch.masked_select`",
        "aten::div.Tensor": "`div.Tensor`",
        "aten::lerp.Scalar_out": "`lerp.Scalar_out`",
        "aten::lerp_.Scalar": "`lerp_.Scalar`",
        "aten::sqrt.out": "`sqrt.out`",
        "aten::add_.Tensor": "`add_.Tensor`",
        "aten::mul_.Scalar": "`mul_.Scalar`",
        "aten::addcmul_.default": "`addcmul_`",
        "aten::addcdiv_.default": "`addcdiv_`",
        "aten::zero_.default": "`zero_`",
    }
    assert set(matrix_declarations) == matrix_rows
    assert all(declaration in matrix for declaration in matrix_declarations.values())
    classifications = (
        DECLARED_OPERATION_MANIFEST,
        DEFERRED_SOURCE_SCHEMAS,
        EXPLICIT_REJECTED_SOURCE_SCHEMAS,
    )
    assert classifications[0].isdisjoint(classifications[1])
    assert classifications[0].isdisjoint(classifications[2])
    assert classifications[1].isdisjoint(classifications[2])
    source_inventory, source_explicit_rejected = _source_registration_classifications()
    assert _matrix_schema_set(matrix, "deferred") == DEFERRED_SOURCE_SCHEMAS
    assert source_explicit_rejected == EXPLICIT_REJECTED_SOURCE_SCHEMAS
    assert EXPLICIT_REJECTED_SOURCE_SCHEMAS
    assert source_inventory == (
        matrix_rows | DEFERRED_SOURCE_SCHEMAS | EXPLICIT_REJECTED_SOURCE_SCHEMAS
    )


def _matrix_schema_set(matrix, label):
    marker = re.search(
        rf"Vulkan conformance {label} schemas:\s*(.*?)\s*-->", matrix, re.DOTALL
    )
    assert marker is not None
    return frozenset(item.strip() for item in marker.group(1).replace("\n", "").split(",")
                     if item.strip())


def test_conformance_declaration_ids_match_matrix_in_both_directions():
    matrix = Path("docs/vulkan_operator_capability_matrix.md").read_text()
    supported = _matrix_schema_set(matrix, "supported")
    rejected = _matrix_schema_set(matrix, "rejected")
    deferred = _matrix_schema_set(matrix, "deferred")
    supported_cases = {case.declaration_id for case in ALL_CASES if case.supported}
    rejected_cases = {case.declaration_id for case in ALL_CASES if not case.supported}
    assert supported_cases == supported
    assert rejected_cases == rejected
    # A schema can have both a supported overload and a rejected overload in
    # the executable case registry; deferred schemas remain separate from the
    # supported source manifest and are checked independently below.
    assert {case.declaration_id for case in ALL_CASES} <= supported | rejected | deferred
    assert all(case.declaration_id.startswith("aten::") for case in ALL_CASES)


def test_registration_parser_accepts_formatting_variants():
    source = '''
    TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m ) {
      m . impl ( "aten::view", &view );
      if (enabled) { m.impl("linear", &linear); }
    }
    '''
    blocks = _extract_privateuse1_blocks(source)
    assert len(blocks) == 1
    assert _parse_m_impl_schemas(blocks[0]) == {
        "aten::view.default", "aten::linear.default"
    }


def test_registration_parser_ignores_comments_and_allows_comments_between_tokens():
    source = '''
    // TORCH_LIBRARY_IMPL(aten, PrivateUse1, ignored) {
    //   m.impl("commented_out", &reject_commented_out);
    // }
    TORCH_LIBRARY_IMPL(/* before */ aten, PrivateUse1, /* before m */ m) {
      m /* between */ . /* tokens */ impl(
        "view" /* schema comment */, &view); // m.impl("fake", &fake);
      /* m.impl("block_commented", &reject_block_commented); */
      m.impl("linear", /* handler */ &linear);
    }
    '''
    blocks = _extract_privateuse1_blocks(source)
    assert len(blocks) == 1
    assert _parse_m_impl_schemas(blocks[0]) == {
        "aten::view.default", "aten::linear.default"
    }


def test_registration_discovery_includes_non_cpp_source_files(tmp_path):
    source = tmp_path / "registration.hpp"
    source.write_text(
        'TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {\n'
        '  m.impl("header_only", &header_only);\n}\n'
    )
    assert _source_registration_inventory(tmp_path) == {"aten::header_only.default"}


def test_source_classifications_are_pairwise_disjoint():
    assert DECLARED_OPERATION_MANIFEST.isdisjoint(DEFERRED_SOURCE_SCHEMAS)
    assert DECLARED_OPERATION_MANIFEST.isdisjoint(EXPLICIT_REJECTED_SOURCE_SCHEMAS)
    assert DEFERRED_SOURCE_SCHEMAS.isdisjoint(EXPLICIT_REJECTED_SOURCE_SCHEMAS)


def test_deferred_operations_are_explicitly_separate_from_declared_manifest():
    deferred = frozenset({
        "aten::exp.default", "aten::double_add", "aten::double_mul",
    })
    assert not DECLARED_OPERATION_MANIFEST & deferred


def test_reduction_indexing_matrix_is_declared():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    for operation in ("`aten::sum`", "`aten::mean`", "`aten::argmax`"):
        assert operation in matrix


def test_documentation_declares_empty_reduction_and_presentation_contracts():
    matrix = Path("docs/vulkan_operator_capability_matrix.md").read_text()
    readme = Path("README-build.md").read_text()
    for entry in (
        "empty input produces a Vulkan-resident identity result (0 for `sum`)",
        "empty input produces a Vulkan-resident NaN result for `mean`",
        "final `.cpu()` comparison is an explicit presentation transfer, not fallback",
        "positive-stride and non-zero-offset value views are materialized with a Vulkan-resident copy before compaction",
        "same-shaped, contiguous, and zero-offset bool on `vk:0`",
        "no CPU fallback",
    ):
        assert entry in matrix
    for entry in (
        "PYTHONPATH=build .venv/bin/python -m pytest -q tests/python",
        "compute dispatch count",
        "Vulkan copy count",
        "explicit transfer count",
        "final `.cpu()` comparison is an explicit presentation transfer, not fallback",
    ):
        assert entry in readme
    assert (
        "`torch.masked_select` supports contiguous or positive-stride/non-zero-offset "
        "F32 value views."
    ) in readme
    assert (
        "These value views are materialized via a Vulkan-resident copy before "
        "compaction;"
    ) in readme
    assert (
        "the operation requires same-shaped contiguous, zero-offset Vulkan bool "
        "masks on `vk:0`."
    ) in readme
    assert "Tensor and mask payloads are never read back" in readme
    assert "rejected without CPU fallback." in readme
    assert "avoids hidden CPU fallback or payload materialization" not in matrix
    assert "permits intentional Vulkan-to-Vulkan value-view materialization" in matrix
    assert "forbids hidden CPU payload materialization or readback" in matrix


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


def test_model_matrix_declares_fixed_phase_6_slices():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    for operation in (
        "`aten::linear`",
        "`aten::convolution`",
        "`aten::_adaptive_avg_pool2d`",
        "fixed MLP",
        "fixed CNN",
    ):
        assert operation in matrix


def test_formatter_double_matrix_declares_exact_surface_and_boundaries():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    for entry in (
        "`aten::abs.default`",
        "`aten::min.default`",
        "`aten::max.default`",
        "`aten::ceil.default`",
        "`aten::ne.Tensor`",
        "`aten::div.Tensor`",
        "`aten::gt.Scalar`",
        "`aten::lt.Scalar`",
        "`aten::_local_scalar_dense.default`",
        "shaderFloat64",
        "sizeof(double)",
        "explicit final-value presentation transfer",
        "no normal CPU fallback",
        "Vulkan formatter Double payload readback to CPU is unsupported",
        "Vulkan formatter Double support requires the shaderFloat64 device feature",
        "Vulkan formatter conversion supports only Vulkan float32 to Vulkan Double",
        "Double `add`, `mul`, and unrelated operators remain rejected",
    ):
        assert entry in matrix


def test_formatter_double_matrix_rejects_unrelated_schema_claims():
    matrix = open("docs/vulkan_operator_capability_matrix.md").read()
    assert "`aten::add.Tensor`" not in matrix
    assert "`aten::mul.Tensor`" not in matrix
    assert "general Double readback" not in matrix


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def assert_vulkan_capability(operation, inputs, *, supported, error_pattern=None):
    """Run an operation and assert explicit support or rejection."""
    if supported:
        result = operation(*inputs)
        assert result.device.type == "vk"
        assert result.device.index == 0
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


@pytest.mark.parametrize("operation", [torch.neg, torch.abs, torch.relu, torch.sub])
def test_unsupported_bool_operator_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    inputs = (tensor,) if operation in [torch.neg, torch.abs, torch.relu] else (tensor, tensor)

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
    zero_result = assert_vulkan_capability(operation, (zero_dimensional,), supported=True)
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
