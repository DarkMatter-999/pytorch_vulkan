#!/usr/bin/env python3
"""Validate the checked-in Vulkan capability contract and its drift boundaries."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.vulkan_capability_declarations import (
    STOCK_COMPOSITE_ROUTE_CONTRACT,
    STOCK_COMPOSITE_ROUTES,
)

REQUIRED_ENTRY_KEYS = frozenset(
    {
        "schema",
        "status",
        "device",
        "dtypes",
        "ranks",
        "layouts",
        "shape_constraints",
        "witnesses",
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
SHAPE_PATTERN = re.compile(r"(?:unwitnessed|[1-9][0-9]*x[1-9][0-9]*(?:x[1-9][0-9]*)*)")
ALIASING_VALUES = frozenset({"no_overlap", "same_storage_alias", "no_aliasing"})
OUT_VALUES = frozenset({"not_applicable", "contiguous_out_required"})
INPLACE_VALUES = frozenset({"not_applicable", "optimizer_scoped_inplace", "validated_exact_alias_inplace"})
AUTOGRAD_VALUES = frozenset({"first_order_or_none", "first_order_backward", "backward_kernel", "not_differentiable", "optimizer_update", "first_order_view_alias", "reverse_second_order_witnessed", "not_applicable"})
EXECUTION_VALUES = frozenset({"vulkan_compute", "vulkan_copy", "metadata_only", "vulkan_copy_then_compute", "rejected_before_vulkan", "deferred_before_vulkan"})
REASON_VALUES = frozenset({"supported_contract", "explicit_source_rejection", "deferred_contract", "schema_absent_from_pytorch_dispatcher"})
SCALAR_VALUES = frozenset({"none", "scalar_supported"})
FEATURE_VALUES = frozenset({"vulkan_1_2_8bit_storage_int8"})
TEST_VALUES = frozenset(
    {
        "tests/python/test_vulkan_capability_manifest.py",
        "tests/python/test_vulkan_conformance.py",
        "tests/python/test_vulkan_operator_capabilities.py",
    }
)


def schema_exists(schema: str) -> bool:
    """True when `schema` names a real PyTorch dispatcher overload.

    Manifest keys are the canonical `aten::foo.overload` form. The `aten::`
    prefix must be stripped BEFORE splitting on the dot, or the op lookup
    silently misses.
    """
    if not schema.startswith("aten::"):
        return False
    op_name, _, overload = schema.removeprefix("aten::").partition(".")
    op = getattr(torch.ops.aten, op_name, None)
    if op is None:
        return False
    try:
        return overload in op.overloads()
    except (AttributeError, RuntimeError):
        return False


def load_coverage_evidence(root: Path) -> dict[str, Any]:
    try:
        coverage = json.loads((root / "docs/vulkan_coverage.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("coverage evidence unavailable") from error
    if not isinstance(coverage, dict):
        raise ValueError("coverage evidence must be an object")
    return coverage


def validate_tensor_list_evidence(coverage: dict[str, Any], *, require_complete: bool = False) -> None:
    """Check cat Tensor[] metadata is internally bound to its executed case record."""
    expected = {
        "cat.rank1.forward": [[2], [3]],
        "cat.rank2.forward": [[2, 3], [2, 4]],
        "cat.rank3.forward": [[2, 2, 3], [2, 2, 4]],
        "cat.rank4.native-seed": [[2, 3, 4, 2], [2, 3, 4, 2]],
        "cat.offset.trainable-seed": [[2, 3, 4, 3], [2, 2, 4, 3]],
        "cat.channels-last.forward": [[2, 2, 3, 4], [2, 2, 3, 4]],
        "cat.empty.reference": [[0], [2, 3]],
    }
    for name in expected if require_complete else coverage.keys() & expected.keys():
        record = coverage.get(name)
        if (not isinstance(record, dict) or "tensor_lists" not in record
                or record.get("schema") != "aten::cat.default"
                or record.get("parity") is not True):
            raise ValueError(f"{name}: required executed cat.default TensorList witness is missing")
    for name, record in coverage.items():
        if name.startswith("cat.") and record.get("schema") == "aten::cat.default" and "tensor_lists" not in record:
            raise ValueError(f"{name}: executed cat witness is missing mandatory TensorList evidence")
        if "tensor_lists" not in record:
            continue
        if name not in {
            "cat.rank1.forward", "cat.rank2.forward", "cat.rank3.forward",
            "cat.rank4.native-seed", "cat.offset.trainable-seed",
            "cat.channels-last.forward", "cat.empty.reference",
        } or record.get("schema") != "aten::cat.default" or record.get("parity") is not True:
            raise ValueError(f"{name}: TensorList evidence is not attached to an executed cat.default case")
        lists = record["tensor_lists"]
        if not isinstance(lists, list) or len(lists) != 1 or lists[0].get("role") != "primary_input" or lists[0].get("path") != "inputs[0]":
            raise ValueError(f"{name}: malformed primary TensorList path")
        tensors = lists[0].get("tensors")
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"{name}: empty or malformed actual TensorList")
        for index, tensor in enumerate(tensors):
            if (not isinstance(tensor, dict) or set(tensor) != {"position", "dtype", "rank", "shape"}
                    or tensor["position"] != index or tensor["dtype"] not in KNOWN_DTYPES
                    or type(tensor["rank"]) is not int or tensor["rank"] < 0
                    or not isinstance(tensor["shape"], list)
                    or len(tensor["shape"]) != tensor["rank"]
                    or any(type(size) is not int or size < 0 for size in tensor["shape"])):
                raise ValueError(f"{name}: malformed tensor-list member {index}")
        if any(tensor.get("dtype") != "float32" for tensor in tensors):
            raise ValueError(f"{name}: named cat TensorList members must all be float32")
        if record.get("output_dtype") != "float32":
            raise ValueError(f"{name}: named cat witness output dtype must be float32")
        if name in expected and [tensor["shape"] for tensor in tensors] != expected[name]:
            raise ValueError(f"{name}: actual TensorList geometry does not match its named witness")
        if name == "cat.rank4.native-seed" and record.get("primary_input") != {"dtype": "float32", "rank": 4}:
            raise ValueError(f"{name}: native reverse witness must retain rank-four primary metadata")
        dtypes = sorted({tensor["dtype"] for tensor in tensors})
        ranks = sorted({tensor["rank"] for tensor in tensors})
        shapes = sorted({"x".join(map(str, tensor["shape"])) for tensor in tensors if tensor["rank"] > 0})
        primary = next((tensor for tensor in tensors if not (tensor["rank"] == 1 and tensor["shape"] == [0])), tensors[0])
        if (record.get("input_dtypes") != dtypes or record.get("input_ranks") != ranks
                or record.get("input_shapes") != shapes
                or record.get("primary_input") != {"dtype": primary["dtype"], "rank": primary["rank"]}):
            raise ValueError(f"{name}: aggregate coverage metadata differs from its actual TensorList")
        if record.get("operands") != [
            {"role": "primary_input" if tensor is primary else "operand", "dtype": tensor["dtype"], "rank": tensor["rank"]}
            for tensor in tensors
        ]:
            raise ValueError(f"{name}: operands differ from actual TensorList members")


def validate_stock_composite_routes(
    routes: dict[str, Any],
    entries: list[dict[str, Any]],
    coverage: dict[str, Any],
    source: set[str],
    explicit_rejected: set[str],
) -> None:
    """Validate the sole versioned composite route and its executed witness links."""
    expected_schema = STOCK_COMPOSITE_ROUTE_CONTRACT["schema"]
    route_entries = [entry for entry in entries if entry.get("schema") == expected_schema]
    if not routes and not route_entries:
        return
    if set(routes) != {expected_schema}:
        raise ValueError(
            f"unknown stock-composite route(s): {sorted(set(routes) - {expected_schema})}"
        )
    route = routes[expected_schema]
    if (
        not isinstance(route, dict)
        or set(route) != set(STOCK_COMPOSITE_ROUTE_CONTRACT)
        or route.get("schema") != expected_schema
        or route.get("reference") != STOCK_COMPOSITE_ROUTE_CONTRACT["reference"]
        or route.get("dependencies") != STOCK_COMPOSITE_ROUTE_CONTRACT["dependencies"]
    ):
        raise ValueError("stock-composite route descriptor/reference/dependency closure is unqualified")
    if torch.__version__.split("+", 1)[0] != route["reference"]["pytorch_version"]:
        raise ValueError("stock-composite route PyTorch reference version mismatch")

    # Check dispatcher schemas and direct Vulkan leaf registrations without
    # representing stock generated clone/_unsafe_view routes as source kernels.
    for dependency in route["dependencies"]:
        if not schema_exists(dependency["schema"]):
            raise ValueError(f"stock-composite dependency schema is unresolved: {dependency['schema']}")
        if (
            dependency["dispatch"] == "stock_generated_privateuse1"
            and dependency["schema"] in source
        ):
            raise ValueError(
                f"stock-generated dependency is misclassified as direct source: {dependency['schema']}"
            )
        leaf = dependency["vulkan_leaf"]
        if not schema_exists(leaf) or leaf not in source or leaf in explicit_rejected:
            raise ValueError(f"stock-composite dependency closure is unresolved at {leaf}")
    if expected_schema in source or expected_schema in explicit_rejected:
        raise ValueError("stock-composite route duplicates a direct or rejected source registration")

    entries_by_schema = {entry["schema"]: entry for entry in entries}
    entry = entries_by_schema.get(expected_schema)
    if entry is None or entry.get("status") != "supported":
        raise ValueError("stock-composite route has no supported manifest entry")
    cases = {
        case.get("name")
        for case in entry.get("test_cases", [])
        if isinstance(case, dict) and case.get("supported") is True
    }
    evidence_names = route["evidence_cases"]
    if (
        evidence_names != STOCK_COMPOSITE_ROUTE_CONTRACT["evidence_cases"]
        or not evidence_names
        or len(set(evidence_names)) != len(evidence_names)
        or not set(evidence_names) <= cases
    ):
        raise ValueError("stock-composite route evidence case links are missing or unsupported")
    expected_shapes = {
        "view.reshape.copy.trainable-seed": "2x3",
        "view.reshape.offset-copy.second-order": "4x6",
    }
    for name in evidence_names:
        record = coverage.get(name)
        if (
            name not in expected_shapes
            or not isinstance(record, dict)
            or record.get("schema") != expected_schema
            or record.get("parity") is not True
            or record.get("gradients") is not True
            or record.get("primary_input") != {"dtype": "float32", "rank": 2}
            or record.get("input_shapes") != [expected_shapes[name]]
            or record.get("reverse_autograd") != {"order": 2, "graph_preserved": True}
        ):
            raise ValueError(f"stock-composite route lacks matching executed route evidence: {name}")
        execution = record.get("execution")
        if (
            not isinstance(execution, dict)
            or set(execution) != {
                "mode", "compute_dispatches", "vulkan_copies",
                "explicit_transfers", "fallbacks",
            }
            or execution.get("mode") != "copy"
            or any(type(execution.get(key)) is not int for key in (
                "compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks"
            ))
            or execution["vulkan_copies"] <= 0
            or execution["compute_dispatches"] != 0
            or execution["explicit_transfers"] != 0
            or execution["fallbacks"] != 0
        ):
            raise ValueError(f"stock-composite route execution evidence is invalid: {name}")
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
        for shape in entry["shape_constraints"]:
            if not isinstance(shape, str) or not SHAPE_PATTERN.fullmatch(shape):
                raise ValueError(f"{path}.shape_constraints: malformed shape token {shape!r}")
        if entry["status"] != "rejected" and not schema_exists(entry["schema"]):
            raise ValueError(
                f"{path}.schema: {entry['schema']!r} (status={entry['status']!r}) is not a PyTorch dispatcher overload"
            )
        witnesses = entry["witnesses"]
        if not isinstance(witnesses, dict) or set(witnesses) not in (
            {"dtypes", "ranks", "pairs", "cases"},
            {"dtypes", "ranks", "pairs", "cases", "reverse_second_order_cases"},
        ):
            raise ValueError(f"{path}.witnesses: expected primary-input dtypes/ranks/pairs/cases")
        for field in ("dtypes", "ranks", "cases"):
            values = witnesses[field]
            expected_type = int if field == "ranks" else str
            if not isinstance(values, list) or any(type(value) is not expected_type for value in values):
                raise ValueError(f"{path}.witnesses.{field}: expected list of {expected_type.__name__}")
            if values != sorted(set(values)):
                raise ValueError(f"{path}.witnesses.{field}: expected sorted unique values")
        pairs = witnesses["pairs"]
        if not isinstance(pairs, list) or any(
            not isinstance(pair, list) or len(pair) != 2
            or pair[0] not in KNOWN_DTYPES or type(pair[1]) is not int or pair[1] < 0
            for pair in pairs
        ) or pairs != [list(pair) for pair in sorted({tuple(pair) for pair in pairs})]:
            raise ValueError(f"{path}.witnesses.pairs: expected sorted unique primary-input dtype/rank pairs")
        if witnesses["dtypes"] != sorted({pair[0] for pair in pairs}) or witnesses["ranks"] != sorted({pair[1] for pair in pairs}):
            raise ValueError(f"{path}.witnesses: dtype/rank unions differ from primary-input pairs")
        if not set(witnesses["cases"]) <= {case.get("name") for case in entry["test_cases"]}:
            raise ValueError(f"{path}.witnesses.cases: case is not present in test_cases")
        promoted = entry["autograd"] == "reverse_second_order_witnessed"
        if promoted != ("reverse_second_order_cases" in witnesses):
            raise ValueError(f"{path}.witnesses.reverse_second_order_cases: required only for reverse_second_order_witnessed")
        if "reverse_second_order_cases" in witnesses:
            reverse_cases = witnesses["reverse_second_order_cases"]
            supported_cases = {case["name"] for case in entry["test_cases"] if isinstance(case, dict) and case.get("supported") is True}
            if (
                not isinstance(reverse_cases, list)
                or not reverse_cases
                or any(type(name) is not str for name in reverse_cases)
                or reverse_cases != sorted(set(reverse_cases))
                or not set(reverse_cases) <= set(witnesses["cases"])
                or not set(reverse_cases) <= supported_cases
            ):
                raise ValueError(f"{path}.witnesses.reverse_second_order_cases: expected sorted supported witness case links")
        if promoted:
            coverage = load_coverage_evidence(root)
            for name in witnesses["reverse_second_order_cases"]:
                record = coverage.get(name)
                if (
                    not isinstance(record, dict)
                    or record.get("schema") != schema
                    or record.get("parity") is not True
                    or record.get("reverse_autograd") != {"order": 2, "graph_preserved": True}
                    or type(record.get("reverse_autograd", {}).get("order")) is not int
                    or type(record.get("reverse_autograd", {}).get("graph_preserved")) is not bool
                ):
                    raise ValueError(f"{path}.witnesses.reverse_second_order_cases: {name!r} lacks matching executed reverse evidence")
        if entry["status"] == "supported":
            if not pairs or not witnesses["cases"]:
                raise ValueError(f"{path}.witnesses: supported entry has no witnesses")
            missing_dtypes = set(dtypes["inputs"]) - {pair[0] for pair in pairs}
            if missing_dtypes:
                raise ValueError(f"{schema}: declares input dtypes {sorted(missing_dtypes)} that no case exercised")
            for rank in range(ranks["min"], ranks["max"] + 1):
                if rank not in witnesses["ranks"]:
                    raise ValueError(f"{schema}: declares ranks {ranks['min']}-{ranks['max']} but rank {rank} was never exercised")
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
        if entry["inplace"] == "validated_exact_alias_inplace" and entry["aliasing"] != "same_storage_alias":
            raise ValueError(
                f"{path}.aliasing: validated exact-alias in-place requires same_storage_alias"
            )
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
            "rejected": entry["reason"],
            "deferred": "deferred_contract",
        }[entry["status"]]
        if entry["reason"] != expected_reason:
            raise ValueError(f"{path}.reason: inconsistent with status")
        if entry["status"] == "rejected" and entry["reason"] not in {
            "explicit_source_rejection",
            "schema_absent_from_pytorch_dispatcher",
        }:
            raise ValueError(f"{path}.reason: invalid rejected-entry reason")
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
    coverage = load_coverage_evidence(root) if STOCK_COMPOSITE_ROUTES else {}
    validate_tensor_list_evidence(
        coverage,
        require_complete=any(entry.get("schema") == "aten::cat.default" for entry in entries),
    )
    validate_stock_composite_routes(
        STOCK_COMPOSITE_ROUTES, entries, coverage, source, explicit_rejected
    )
    for entry in entries:
        if entry.get("schema") != "aten::cat.default":
            continue
        actual = {
            name: record for name, record in coverage.items()
            if record.get("schema") == "aten::cat.default" and record.get("parity") is True
        }
        dtypes = sorted({record["primary_input"]["dtype"] for record in actual.values()})
        ranks = sorted({record["primary_input"]["rank"] for record in actual.values()})
        pairs = sorted({(record["primary_input"]["dtype"], record["primary_input"]["rank"])
                        for record in actual.values()})
        reverse = sorted(name for name, record in actual.items()
                         if record.get("reverse_autograd") == {"order": 2, "graph_preserved": True})
        expected_witnesses = {"dtypes": dtypes, "ranks": ranks,
                              "pairs": [list(pair) for pair in pairs],
                              "cases": sorted(actual)}
        if reverse:
            expected_witnesses["reverse_second_order_cases"] = reverse
        if entry.get("witnesses") != expected_witnesses:
            raise ValueError("aten::cat.default manifest witnesses differ from validated TensorList records")
    composite = set(STOCK_COMPOSITE_ROUTES)
    if composite & source:
        raise ValueError(f"stock-composite schemas have duplicate direct registrations: {sorted(composite & source)}")
    if composite & explicit_rejected:
        raise ValueError(f"stock-composite schemas are explicitly rejected: {sorted(composite & explicit_rejected)}")
    schema_absent_rejected = {
        entry["schema"]
        for entry in entries
        if entry["status"] == "rejected"
        and entry["reason"] == "schema_absent_from_pytorch_dispatcher"
    }
    if rejected - schema_absent_rejected != explicit_rejected:
        raise ValueError("manifest rejected schemas drift from source registrations")
    direct_supported = source - deferred - rejected
    if supported - composite != direct_supported:
        raise ValueError("manifest supported schemas drift from source registrations")
    if not composite <= supported:
        raise ValueError(f"manifest omits declared stock-composite routes: {sorted(composite - supported)}")


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
