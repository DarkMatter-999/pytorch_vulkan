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
)


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

    validate_manifest_data(data, tmp_path)


def test_manifest_validation_does_not_require_capability_matrix(tmp_path):
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
