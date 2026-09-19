import copy
import json
from pathlib import Path

import pytest

from tools.validate_vulkan_capabilities import (
    REQUIRED_ENTRY_KEYS,
    _source_registration_inventory,
    load_manifest,
    validate_manifest_data,
)


ROOT = Path(__file__).parents[2]


def _entry(schema="aten::example.default", *, status="supported"):
    return {
        "schema": schema,
        "status": status,
        "device": {"type": "PrivateUse1", "index": 0},
        "dtypes": {"inputs": ["float32"], "outputs": ["float32"]},
        "ranks": {"min": 0, "max": 8},
        "layouts": ["strided", "non-overlapping"],
        "shape_constraints": ["schema_example_default"],
        "empty": "empty_output_supported",
        "aliasing": "no_overlap",
        "out": "not_applicable",
        "inplace": "not_applicable",
        "autograd": "first_order_or_none",
        "scalar_constraints": ["none"],
        "required_vulkan_features": [],
        "execution_contract": "vulkan_compute",
        "tests": ["tests/python/test_vulkan_conformance.py"],
        "test_cases": [],
        "reason": "supported_contract",
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


def test_manifest_rejects_unknown_shape_constraint():
    data = {"version": 1, "entries": [_entry()]}
    data["entries"][0]["shape_constraints"] = ["invented_constraint"]

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
