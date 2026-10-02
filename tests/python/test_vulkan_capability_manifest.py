import copy
import json
from pathlib import Path

import pytest

import tools.validate_vulkan_capabilities as capability_validator
from tools.validate_vulkan_capabilities import (
    REQUIRED_ENTRY_KEYS,
    _source_registration_inventory,
    load_manifest,
    validate_manifest_data,
    validate_stock_composite_routes,
)
from tools.vulkan_capability_declarations import STOCK_COMPOSITE_ROUTES

ROOT = Path(__file__).parents[2]


def _entry(schema="aten::abs.default", *, status="supported"):
    # A real PyTorch schema, because the validator resolves every non-rejected
    # entry against torch.ops.aten. A synthetic name here makes every assertion
    # below fail on "not a dispatcher overload" before reaching its own check.
    return {
        "schema": schema,
        "status": status,
        "device": {"type": "PrivateUse1", "index": 0},
        "dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "ranks": {"min": 0, "max": 8},
        "layouts": ["strided", "non-overlapping"],
        "shape_constraints": ["2x4"],
        "empty": "empty_output_supported",
        "aliasing": "no_overlap",
        "out": "not_applicable",
        "inplace": "not_applicable",
        "autograd": "first_order_or_none",
        "scalar_constraints": ["none"],
        "required_vulkan_features": [],
        "execution_contract": "vulkan_compute",
        "tests": ["tests/python/test_vulkan_conformance.py"],
        "test_cases": [{"name": "example.case", "supported": True}],
        "reason": "supported_contract",
        # Stage 4 made `witnesses` a required, derived field: a declared rank
        # or dtype must be exercised by a case's primary input. A fixture
        # without it fails every assertion below on "missing keys" before
        # reaching the check it is meant to exercise. It must also be
        # self-consistent -- witness case names have to appear in test_cases,
        # and the declared 0-8 rank range has to be covered.
        "witnesses": {
            "dtypes": ["float32"],
            "ranks": [0, 1, 2, 3, 4, 5, 6, 7, 8],
            "pairs": [["float32", r] for r in range(9)],
            "cases": ["example.case"],
        },
    }


def test_manifest_has_unique_schema_entries_and_required_contract_keys():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")

    schemas = [entry["schema"] for entry in manifest["entries"]]
    assert len(schemas) == len(set(schemas))
    for entry in manifest["entries"]:
        assert REQUIRED_ENTRY_KEYS <= entry.keys()


def _transposed_manifest_fixture(monkeypatch):
    from test_vulkan_capability_generation import _complete_convolution_fixture

    from tools import generate_vulkan_capabilities as generator

    coverage = _complete_convolution_fixture()
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    data = generator.build_coverage_manifest(coverage)
    return data, coverage


def _change_transposed_weight_and_aggregate_dtype(record):
    record["convolution_operands"]["weight"]["dtype"] = "float64"
    record["input_dtypes"] = ["float32", "float64"]


def _rewrite_transposed_mask(record):
    mask = [False, False, False]
    record["convolution_context"]["output_mask"] = mask
    record["schema_args"]["output_mask"] = mask


def test_manifest_accepts_complete_transposed_convolution_runtime_fixture(monkeypatch):
    data, _ = _transposed_manifest_fixture(monkeypatch)
    validate_manifest_data(data, ROOT)


@pytest.mark.parametrize("mutation,match", [
    (lambda record: record["convolution_context"]["warm_forward_bias"]["vulkan"].update(dtype="float64"), "warm bias"),
    (lambda record: record["convolution_context"]["warm_forward_bias"]["vulkan"].update(rank=2), "warm bias"),
    (lambda record: record["convolution_context"]["warm_forward_bias"]["vulkan"].update(defined=False), "warm bias"),
    (lambda record: record["convolution_context"].update(forward_bias_present=False), "forward_bias_present"),
    (lambda record: record["convolution_context"]["warm_forward_bias"]["vulkan"].update(device="cpu"), "warm bias"),
    (lambda record: record["convolution_context"].pop("warm_forward_bias"), "warm bias"),
    (_change_transposed_weight_and_aggregate_dtype, "operand metadata"),
    (lambda record: record["output_slots"][0].update(dtype="float64"), "output slot"),
    (lambda record: record["convolution_context"].update(direction="forward"), "direction"),
    (lambda record: record["convolution_context"].update(expected_numerical_operations=[1]), "operation"),
    (lambda record: record["convolution_context"].update(expected_result_slots=[1]), "slot"),
    (_rewrite_transposed_mask, "output_mask"),
    (lambda record: record["execution"].update(compute_dispatches=0), "dispatch"),
])
def test_manifest_rejects_transposed_convolution_coverage_tampering(monkeypatch, mutation, match):
    data, coverage = _transposed_manifest_fixture(monkeypatch)
    target = "convolution.transposed.backward.bias-present.mask-100"
    mutation(coverage[target])
    with pytest.raises(ValueError, match=match):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_transposed_mask000_allocation(monkeypatch):
    data, coverage = _transposed_manifest_fixture(monkeypatch)
    record = coverage["convolution.transposed.backward.bias-present.mask-000"]
    record["execution"]["live_allocations_delta"] = 1
    with pytest.raises(ValueError, match="mask 000"):
        validate_manifest_data(data, ROOT)


def test_manifest_convolution_completeness_survives_joint_transposed_record_and_link_removal(monkeypatch):
    data, coverage = _transposed_manifest_fixture(monkeypatch)
    for name in list(coverage):
        if name.startswith("convolution.transposed."):
            del coverage[name]
    for entry in data["entries"]:
        if entry["schema"] not in {"aten::convolution.default", "aten::convolution_backward.default"}:
            continue
        entry["test_cases"] = [case for case in entry["test_cases"]
                               if not case["name"].startswith("convolution.transposed.")]
        entry["witnesses"]["cases"] = [name for name in entry["witnesses"]["cases"]
                                        if not name.startswith("convolution.transposed.")]
    with pytest.raises(ValueError, match="required executed convolution witness"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_joint_transposed_direction_operation_slot_schema_rewrite(monkeypatch):
    data, coverage = _transposed_manifest_fixture(monkeypatch)
    record = coverage["convolution.transposed.backward.bias-present.mask-100"]
    record["schema"] = "aten::convolution.default"
    record["convolution_context"].update(
        direction="forward", expected_numerical_operations=[1], expected_result_slots=[0]
    )
    with pytest.raises(ValueError, match="schema|direction|identity"):
        validate_manifest_data(data, ROOT)


@pytest.mark.parametrize("dtype_group,witness_field", [
    ("inputs", "primary"), ("outputs", "output"),
])
def test_manifest_rejects_record_and_manifest_joint_dtype_claim_rewrite(
    monkeypatch, dtype_group, witness_field
):
    data, coverage = _transposed_manifest_fixture(monkeypatch)
    entry = next(item for item in data["entries"]
                 if item["schema"] == "aten::convolution_backward.default")
    record = coverage["convolution.transposed.backward.bias-present.mask-100"]
    if witness_field == "primary":
        record["convolution_operands"]["grad_output"]["dtype"] = "float64"
        record["primary_input"]["dtype"] = "float64"
        record["input_dtypes"] = ["float64"]
        entry["dtypes"][dtype_group] = ["float64"]
        entry["witnesses"]["dtypes"] = ["float64"]
        entry["witnesses"]["pairs"] = [["float64", 4]]
    else:
        record["output_slots"][0]["dtype"] = "float64"
        entry["dtypes"][dtype_group] = ["float64"]
    with pytest.raises(ValueError, match="operand metadata|output slot|declares .* dtypes"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_unknown_status_enum():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["status"] = "experimental"

    with pytest.raises(ValueError, match=r"entries\[0\]\.status"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_duplicate_schema_entries():
    first = _entry()
    data = {"version": 1, "entries": [first, copy.deepcopy(first)]}

    with pytest.raises(ValueError, match="duplicate schema"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_missing_test_reference(tmp_path):
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["tests"] = ["tests/python/does_not_exist.py"]

    with pytest.raises(ValueError, match=r"does_not_exist\.py"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_non_object_json(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps([]))

    with pytest.raises(ValueError, match="top-level object"):
        load_manifest(manifest_path)


def test_manifest_rejects_unknown_entry_keys():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["unexpected"] = "drift"

    with pytest.raises(ValueError, match=r"entries\[0\]: unknown keys"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_invalid_contract_field_types():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["ranks"] = "rank-anywhere"

    with pytest.raises(ValueError, match=r"entries\[0\]\.ranks"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_generic_supported_contract_placeholders():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["empty"] = "documented_behavior"

    with pytest.raises(ValueError, match=r"entries\[0\]\.empty"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_unknown_closed_contract_values():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["empty"] = "invented_behavior"

    with pytest.raises(ValueError, match=r"entries\[0\]\.empty"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_reverse_evidence_without_promoted_enum():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["witnesses"]["reverse_second_order_cases"] = ["example.case"]
    with pytest.raises(ValueError, match="reverse_second_order_cases"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_promoted_enum_without_reverse_cases():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["autograd"] = "reverse_second_order_witnessed"
    with pytest.raises(ValueError, match="reverse_second_order_cases"):
        validate_manifest_data(data, ROOT)


@pytest.mark.parametrize("metadata", [
    {"order": 1, "graph_preserved": True},
    {"order": 2, "graph_preserved": False},
    {"order": 2, "graph_preserved": 1},
    {"order": 2, "unknown": True},
])
def test_manifest_rejects_invalid_reverse_autograd_metadata(metadata):
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["reverse_autograd"] = metadata
    with pytest.raises(ValueError, match="reverse_autograd"):
        validate_manifest_data(data, ROOT)


@pytest.mark.parametrize("metadata", [
    {"order": 1, "graph_preserved": True},
    {"order": 2, "graph_preserved": False},
    {"order": 2, "graph_preserved": 1},
    {"order": 2, "graph_preserved": True, "source": "invented"},
])
def test_manifest_rejects_unproven_reverse_case_links(tmp_path, monkeypatch, metadata):
    data = {"version": 1, "entries": [_entry()]}
    entry = data["entries"][0]
    entry["autograd"] = "reverse_second_order_witnessed"
    entry["witnesses"]["reverse_second_order_cases"] = ["example.case"]
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "vulkan_coverage.json").write_text(json.dumps({"example.case": {
        "schema": "aten::abs.default", "parity": True, "reverse_autograd": metadata
    }}))
    test_path = tmp_path / "tests/python/test_vulkan_conformance.py"
    test_path.parent.mkdir(parents=True)
    test_path.touch()
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(
        capability_validator,
        "_source_registration_inventory",
        lambda _source: ({"aten::abs.default"}, set()),
    )
    with pytest.raises(ValueError, match="lacks matching executed reverse evidence"):
        validate_manifest_data(data, tmp_path)


def test_manifest_rejects_shape_token_naming_a_phantom_constraint():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["shape_constraints"] = ["schema_example_default"]

    with pytest.raises(ValueError, match=r"entries\[0\]\.shape_constraints"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_status_reason_contradiction():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["reason"] = "deferred_contract"

    with pytest.raises(ValueError, match=r"entries\[0\]\.reason"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_status_execution_contradiction():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["execution_contract"] = "deferred_before_vulkan"

    with pytest.raises(ValueError, match=r"entries\[0\]\.execution_contract"):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_aliasing_inplace_with_no_overlap():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["inplace"] = "validated_exact_alias_inplace"

    with pytest.raises(ValueError, match=r"entries\[0\]\.aliasing.*same_storage_alias"):
        validate_manifest_data(data, ROOT)


def test_manifest_accepts_aliasing_inplace_with_same_storage_alias(tmp_path, monkeypatch):
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["inplace"] = "validated_exact_alias_inplace"
    data["entries"][0]["aliasing"] = "same_storage_alias"
    test_path = tmp_path / "tests/python/test_vulkan_conformance.py"
    test_path.parent.mkdir(parents=True)
    test_path.touch()
    (tmp_path / "src").mkdir()
    monkeypatch.setattr(
        capability_validator,
        "_source_registration_inventory",
        lambda _source: ({"aten::abs.default"}, set()),
    )
    monkeypatch.setattr(capability_validator, "STOCK_COMPOSITE_ROUTES", {})

    validate_manifest_data(data, tmp_path)


def test_manifest_validation_does_not_require_capability_matrix(tmp_path, monkeypatch):
    monkeypatch.setattr(capability_validator, "STOCK_COMPOSITE_ROUTES", {})
    validate_manifest_data({"version": 1, "entries": []}, tmp_path)


def test_manifest_case_references_are_typed_and_unique():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    names = []
    for entry in manifest["entries"]:
        for case in entry["test_cases"]:
            assert set(case) == {"name", "supported"}
            assert isinstance(case["name"], str) and case["name"]
            assert isinstance(case["supported"], bool)
            names.append(case["name"])
    assert len(names) == len(set(names))


def test_manifest_rejected_inventory_is_checked_against_source_parser(tmp_path):
    source = tmp_path / "registration.cpp"
    source.write_text(
        'TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {\n'
        '  m.impl("reject_me", &reject_reject_me);\n'
        '}\n'
    )

    registrations, rejected = _source_registration_inventory(tmp_path)
    assert registrations == {"aten::reject_me.default"}
    assert rejected == {"aten::reject_me.default"}


def test_checked_in_manifest_covers_registered_source():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    validate_manifest_data(manifest, ROOT)


def test_arbitrary_supported_schema_cannot_be_authorized_as_stock_composite(monkeypatch):
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    monkeypatch.setitem(
        capability_validator.STOCK_COMPOSITE_ROUTES,
        "aten::abs.default",
        {"reference": "unjustified", "required_leaves": ()},
    )
    with pytest.raises(ValueError, match="unknown stock-composite route"):
        validate_manifest_data(manifest, ROOT)


def test_stock_composite_allowlist_route_must_be_present_in_manifest(monkeypatch):
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    monkeypatch.setitem(
        capability_validator.STOCK_COMPOSITE_ROUTES,
        "aten::view_as_real.default",
        {"reference": "unjustified", "required_leaves": ()},
    )
    with pytest.raises(ValueError, match="unknown stock-composite route"):
        validate_manifest_data(manifest, ROOT)


def _stock_route_inputs():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    source, rejected = _source_registration_inventory(ROOT / "src")
    return manifest["entries"], coverage, source, rejected


def test_stock_route_rejects_unresolved_or_unqualified_required_dependency():
    entries, coverage, source, rejected = _stock_route_inputs()
    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    routes["aten::reshape.default"]["dependencies"][0]["vulkan_leaf"] = (
        "aten::definitely_missing.default"
    )
    with pytest.raises(ValueError, match="dependency closure"):
        validate_stock_composite_routes(routes, entries, coverage, source, rejected)

    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    source_with_fake_stock_kernel = source | {"aten::clone.default"}
    with pytest.raises(ValueError, match="misclassified as direct source"):
        validate_stock_composite_routes(
            routes, entries, coverage, source_with_fake_stock_kernel, rejected
        )


def test_stock_route_rejects_unexecuted_or_wrongly_linked_evidence():
    entries, coverage, source, rejected = _stock_route_inputs()
    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    routes["aten::reshape.default"]["evidence_cases"] = ["review.unexecuted"]
    with pytest.raises(ValueError, match="evidence case links"):
        validate_stock_composite_routes(routes, entries, coverage, source, rejected)

    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    altered = copy.deepcopy(coverage)
    del altered["view.reshape.copy.trainable-seed"]
    with pytest.raises(ValueError, match="executed route evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)

    altered = copy.deepcopy(coverage)
    altered["view.reshape.copy.trainable-seed"]["schema"] = "aten::view.default"
    with pytest.raises(ValueError, match="executed route evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)

    altered = copy.deepcopy(coverage)
    altered["view.reshape.copy.trainable-seed"]["parity"] = False
    with pytest.raises(ValueError, match="executed route evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)

    altered = copy.deepcopy(coverage)
    altered["view.reshape.copy.trainable-seed"]["execution"] = {
        "mode": "metadata",
        "compute_dispatches": 0,
        "vulkan_copies": 0,
        "explicit_transfers": 0,
        "fallbacks": 0,
    }
    with pytest.raises(ValueError, match="execution evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)

    altered = copy.deepcopy(coverage)
    altered["view.reshape.copy.trainable-seed"]["execution"] = {
        "mode": "copy",
        "compute_dispatches": 1,
        "vulkan_copies": 1,
        "explicit_transfers": 0,
        "fallbacks": 0,
    }
    with pytest.raises(ValueError, match="execution evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)


def test_stock_route_rejects_unknown_supported_route_with_fabricated_witness(monkeypatch):
    entries, coverage, source, rejected = _stock_route_inputs()
    view = next(entry for entry in entries if entry["schema"] == "aten::view.default")
    fabricated = copy.deepcopy(view)
    fabricated["schema"] = "aten::view_as_real.default"
    fabricated["test_cases"] = [{"name": "review.unexecuted", "supported": True}]
    fabricated["witnesses"]["cases"] = ["review.unexecuted"]
    fabricated["witnesses"].pop("reverse_second_order_cases", None)
    fabricated["autograd"] = "first_order_view_alias"
    entries.append(fabricated)
    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    routes["aten::view_as_real.default"] = {
        "reference": "unjustified",
        "dependencies": [],
        "evidence_cases": ["review.unexecuted"],
    }
    with pytest.raises(ValueError, match="unknown stock-composite route"):
        validate_stock_composite_routes(routes, entries, coverage, source, rejected)
    monkeypatch.setattr(capability_validator, "STOCK_COMPOSITE_ROUTES", routes)
    with pytest.raises(ValueError, match="unknown stock-composite route"):
        validate_manifest_data({"version": 1, "entries": entries}, ROOT)


def test_checked_in_gemm_manifest_matches_bounded_operator_contracts():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    entries = {entry["schema"]: entry for entry in manifest["entries"]}

    assert entries["aten::bmm.default"]["layouts"] == [
        "contiguous",
        "transposed-contiguous",
        "non-overlapping",
    ]
    assert entries["aten::bmm.default"]["ranks"] == {"min": 3, "max": 3}
    for schema in ("aten::addmm.default", "aten::addmm.out"):
        assert entries[schema]["ranks"] == {"min": 2, "max": 2}
