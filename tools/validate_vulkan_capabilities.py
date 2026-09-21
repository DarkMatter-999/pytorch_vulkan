#!/usr/bin/env python3
"""Validate the checked-in Vulkan capability contract and its drift boundaries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


REQUIRED_ENTRY_KEYS = frozenset(
    {
        "schema",
        "status",
        "device",
        "dtypes",
        "ranks",
        "layouts",
        "shape_constraints",
        "empty",
        "aliasing",
        "out",
        "inplace",
        "autograd",
        "scalar_constraints",
        "required_vulkan_features",
        "execution_contract",
        "tests",
        "test_cases",
        "reason",
    }
)
STATUSES = frozenset({"supported", "rejected", "deferred"})
ENTRY_KEYS = REQUIRED_ENTRY_KEYS
KNOWN_DTYPES = frozenset({"float16", "float32", "float64", "int64", "bool"})
KNOWN_LAYOUTS = frozenset(
    {"strided", "contiguous", "transposed-contiguous", "non-overlapping", "zero-offset"}
)
EMPTY_VALUES = frozenset({"empty_output_supported", "empty_rejected", "empty_deferred", "reduction_identity_or_nan", "zero_size_noop"})
ALIASING_VALUES = frozenset({"no_overlap", "same_storage_alias", "no_aliasing"})
OUT_VALUES = frozenset({"not_applicable", "contiguous_out_required"})
INPLACE_VALUES = frozenset({"not_applicable", "optimizer_scoped_inplace"})
AUTOGRAD_VALUES = frozenset({"first_order_or_none", "first_order_backward", "backward_kernel", "not_differentiable", "optimizer_update", "first_order_view_alias", "not_applicable"})
EXECUTION_VALUES = frozenset({"vulkan_compute", "vulkan_copy", "metadata_only", "vulkan_copy_then_compute", "rejected_before_vulkan", "deferred_before_vulkan"})
REASON_VALUES = frozenset({"supported_contract", "explicit_source_rejection", "deferred_contract"})
SCALAR_VALUES = frozenset({"none", "scalar_supported"})
FEATURE_VALUES = frozenset({"vulkan_1_2_8bit_storage_int8"})
TEST_VALUES = frozenset(
    {
        "tests/python/test_vulkan_capability_manifest.py",
        "tests/python/test_vulkan_conformance.py",
        "tests/python/test_vulkan_operator_capabilities.py",
    }
)
def load_manifest(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        raise ValueError(f"{path}: invalid JSON: {error}") from error
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top-level object required")
    defaults = data.get("defaults", {})
    expanded = []
    for group in data.get("entries", []):
        if not isinstance(group, dict):
            expanded.append(group)
            continue
        schemas = group.get("schemas", [group.get("schema")])
        for schema in schemas:
            entry = dict(defaults)
            entry.update({key: value for key, value in group.items() if key != "schemas"})
            entry["schema"] = schema
            expanded.append(entry)
    data["entries"] = expanded
    return data


def _canonical_source_schema(identifier: str) -> str:
    identifier = identifier.removeprefix("aten::")
    aliases = {
        "argmax": "aten::argmax.default",
        "linear": "aten::linear.default",
        "convolution": "aten::convolution.default",
        "_adaptive_avg_pool2d": "aten::_adaptive_avg_pool2d.default",
    }
    return aliases.get(
        identifier,
        f"aten::{identifier}" if "." in identifier else f"aten::{identifier}.default",
    )


def _extract_privateuse1_blocks(source: str) -> list[str]:
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
        for index in range(opening, len(source)):
            if source[index] == "{":
                depth += 1
            elif source[index] == "}":
                depth -= 1
                if depth == 0:
                    blocks.append(source[opening + 1 : index])
                    break
    return blocks


def _parse_m_impl_registrations(block: str) -> set[tuple[str, str]]:
    pattern = re.compile(
        r'\bm\s*\.\s*impl\s*\(\s*"([^"]+)"\s*,\s*&?\s*([A-Za-z_]\w*(?:::\w+)*)'
    )
    return {
        (_canonical_source_schema(identifier), handler)
        for identifier, handler in pattern.findall(_strip_cpp_comments(block))
    }


def _strip_cpp_comments(source: str) -> str:
    return re.sub(r"//[^\n]*|/\*.*?\*/", " ", source, flags=re.DOTALL)


def _source_registration_inventory(root: Path) -> tuple[set[str], set[str]]:
    registrations = set()
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp"}:
            continue
        for block in _extract_privateuse1_blocks(path.read_text()):
            for schema, handler in _parse_m_impl_registrations(block):
                registrations.add((schema, handler.split("::")[-1].startswith("reject_")))
    return {schema for schema, _ in registrations}, {
        schema for schema, rejected in registrations if rejected
    }


def validate_manifest_data(data: dict[str, Any], root: Path) -> None:
    if data.get("version") != 1:
        raise ValueError("version: expected 1")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("entries: expected array")

    schemas: set[str] = set()
    case_names: set[str] = set()
    for index, entry in enumerate(entries):
        path = f"entries[{index}]"
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: expected object")
        unknown = set(entry) - ENTRY_KEYS
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        missing = REQUIRED_ENTRY_KEYS - entry.keys()
        if missing:
            raise ValueError(f"{path}: missing keys {sorted(missing)}")
        schema = entry["schema"]
        if not isinstance(schema, str) or not schema.startswith("aten::"):
            raise ValueError(f"{path}.schema: expected aten schema")
        if schema in schemas:
            raise ValueError(f"duplicate schema: {schema}")
        schemas.add(schema)
        if entry["status"] not in STATUSES:
            raise ValueError(f"{path}.status: unknown value {entry['status']!r}")
        device = entry["device"]
        if not isinstance(device, dict) or device.keys() != {"type", "index"}:
            raise ValueError(f"{path}.device: expected type/index object")
        if device["type"] != "PrivateUse1" or device["index"] != 0:
            raise ValueError(f"{path}.device: unsupported device contract")
        dtypes = entry["dtypes"]
        if not isinstance(dtypes, dict) or set(dtypes) != {"inputs", "outputs"}:
            raise ValueError(f"{path}.dtypes: expected inputs/outputs object")
        for dtype_group in dtypes.values():
            if not isinstance(dtype_group, list) or not dtype_group or not set(dtype_group) <= KNOWN_DTYPES:
                raise ValueError(f"{path}.dtypes: invalid dtype list")
        ranks = entry["ranks"]
        if not isinstance(ranks, dict) or set(ranks) != {"min", "max"}:
            raise ValueError(f"{path}.ranks: expected min/max object")
        if not all(isinstance(ranks[key], int) and ranks[key] >= 0 for key in ("min", "max")):
            raise ValueError(f"{path}.ranks: bounds must be non-negative integers")
        if ranks["min"] > ranks["max"]:
            raise ValueError(f"{path}.ranks: min exceeds max")
        if not isinstance(entry["layouts"], list) or not entry["layouts"] or not set(entry["layouts"]) <= KNOWN_LAYOUTS:
            raise ValueError(f"{path}.layouts: invalid layout list")
        if not isinstance(entry["shape_constraints"], list) or not entry["shape_constraints"]:
            raise ValueError(f"{path}.shape_constraints: expected non-empty list")
        shape_token = re.sub(r"[^a-z0-9]+", "_", schema.removeprefix("aten::").lower()).strip("_")
        allowed_shapes = {"any_supported", f"schema_{shape_token}"}
        if not set(entry["shape_constraints"]) <= allowed_shapes:
            raise ValueError(f"{path}.shape_constraints: unknown constraint")
        closed_fields = {
            "empty": EMPTY_VALUES,
            "aliasing": ALIASING_VALUES,
            "out": OUT_VALUES,
            "inplace": INPLACE_VALUES,
            "autograd": AUTOGRAD_VALUES,
            "execution_contract": EXECUTION_VALUES,
            "reason": REASON_VALUES,
        }
        for field, allowed in closed_fields.items():
            value = entry[field]
            if value not in allowed:
                raise ValueError(f"{path}.{field}: unknown contract value {value!r}")
        scalar_constraints = entry["scalar_constraints"]
        if not isinstance(scalar_constraints, list) or not scalar_constraints or not set(scalar_constraints) <= SCALAR_VALUES:
            raise ValueError(f"{path}.scalar_constraints: unknown contract value")
        features = entry["required_vulkan_features"]
        if not isinstance(features, list) or not set(features) <= FEATURE_VALUES:
            raise ValueError(f"{path}.required_vulkan_features: unknown feature")
        test_cases = entry["test_cases"]
        if not isinstance(test_cases, list):
            raise ValueError(f"{path}.test_cases: expected list")
        for case in test_cases:
            if not isinstance(case, dict) or set(case) != {"name", "supported"}:
                raise ValueError(f"{path}.test_cases: expected name/supported objects")
            if not isinstance(case["name"], str) or not case["name"] or not isinstance(case["supported"], bool):
                raise ValueError(f"{path}.test_cases: invalid case contract")
            if case["name"] in case_names:
                raise ValueError(f"duplicate test case: {case['name']}")
            case_names.add(case["name"])
            if entry["status"] != "supported" and case["supported"]:
                raise ValueError(f"{path}.test_cases: non-supported schema cannot have supported case")
        if not isinstance(entry["tests"], list) or not entry["tests"]:
            raise ValueError(f"{path}.tests: expected non-empty array")
        for test_path in entry["tests"]:
            if test_path not in TEST_VALUES or not (root / test_path).is_file():
                raise ValueError(f"{path}.tests: missing reference {test_path}")
        if entry["status"] == "supported" and entry["execution_contract"] in {"rejected_before_vulkan", "deferred_before_vulkan"}:
            raise ValueError(f"{path}.execution_contract: invalid supported status contract")
        expected_reason = {
            "supported": "supported_contract",
            "rejected": "explicit_source_rejection",
            "deferred": "deferred_contract",
        }[entry["status"]]
        if entry["reason"] != expected_reason:
            raise ValueError(f"{path}.reason: inconsistent with status")
        if entry["status"] == "rejected" and entry["execution_contract"] != "rejected_before_vulkan":
            raise ValueError(f"{path}.execution_contract: rejected entry must reject before Vulkan")
        if entry["status"] == "deferred" and entry["execution_contract"] != "deferred_before_vulkan":
            raise ValueError(f"{path}.execution_contract: deferred entry must be deferred before Vulkan")

    supported = {
        entry["schema"] for entry in entries if entry["status"] == "supported"
    }
    deferred = {
        entry["schema"] for entry in entries if entry["status"] == "deferred"
    }
    rejected = {
        entry["schema"] for entry in entries if entry["status"] == "rejected"
    }

    source, explicit_rejected = _source_registration_inventory(root / "src")
    if rejected != explicit_rejected:
        raise ValueError("manifest rejected schemas drift from source registrations")
    if source - deferred - rejected != supported:
        raise ValueError("manifest supported schemas drift from source registrations")


def validate(root: Path) -> None:
    path = root / "docs/vulkan_capabilities.json"
    data = load_manifest(path)
    validate_manifest_data(data, root)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    try:
        validate(args.root.resolve())
    except (OSError, ValueError) as error:
        print(f"capability validation failed: {error}")
        return 1
    print("Vulkan capability manifest is valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
