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
    STOCK_MATMUL_ROUTE_CONTRACT,
    STOCK_COMPOSITE_ROUTES,
    STOCK_LINEAR_ROUTE_CASES,
)
KNOWN_STOCK_COMPOSITE_SCHEMAS = frozenset({
    STOCK_COMPOSITE_ROUTE_CONTRACT["schema"],
    STOCK_MATMUL_ROUTE_CONTRACT["schema"],
    "aten::linear.default",
})

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
SHAPE_PATTERN = re.compile(r"(?:unwitnessed|[1-9][0-9]*(?:x[1-9][0-9]*)*)")
ALIASING_VALUES = frozenset({"no_overlap", "same_storage_alias", "no_aliasing"})
OUT_VALUES = frozenset({"not_applicable", "contiguous_out_required"})
INPLACE_VALUES = frozenset({"not_applicable", "optimizer_scoped_inplace", "validated_exact_alias_inplace"})
AUTOGRAD_VALUES = frozenset({"first_order_or_none", "first_order_backward", "backward_kernel", "not_differentiable", "optimizer_update", "first_order_view_alias", "reverse_second_order_witnessed", "reverse_first_order_graph_witnessed", "reverse_finite_second_order_witnessed", "reverse_selected_third_order_witnessed", "not_applicable"})
EXECUTION_VALUES = frozenset({"vulkan_compute", "vulkan_copy", "metadata_only", "vulkan_copy_then_compute", "rejected_before_vulkan", "deferred_before_vulkan"})
REASON_VALUES = frozenset({"supported_contract", "explicit_source_rejection", "deferred_contract", "schema_absent_from_pytorch_dispatcher"})
SCALAR_VALUES = frozenset({"none", "scalar_supported"})
FEATURE_VALUES = frozenset({"vulkan_1_2_8bit_storage_int8"})
ORDINARY_CONVOLUTION_REQUIRED_CASES = frozenset({
    "convolution.forward.bias-present", "convolution.forward.bias-absent",
    *(f"convolution.backward.bias-{state}.mask-{mask:03b}"
      for state in ("present", "absent") for mask in range(8)),
})
TRANSPOSED_CONVOLUTION_REQUIRED_CASES = frozenset({
    "convolution.transposed.forward.bias-present",
    "convolution.transposed.forward.bias-absent",
    *(f"convolution.transposed.backward.bias-{state}.mask-{mask:03b}"
      for state in ("present", "absent") for mask in range(8)),
})
CONVOLUTION_REQUIRED_CASES = ORDINARY_CONVOLUTION_REQUIRED_CASES | TRANSPOSED_CONVOLUTION_REQUIRED_CASES
TRANSPOSED_MAPPING_ID = "transposed-convolution-stage-e-v1"
# python/pytorch_vulkan/__init__.py renames PrivateUse1 to the public ``vk``
# spelling before any evidence Tensor is allocated.
TRANSPOSED_DEVICE = "vk:0"
OVERRIDEABLE_BACKWARD_SCHEMA = "aten::convolution_backward_overrideable.default"
OVERRIDEABLE_BACKWARD_CASE = "convolution.backward-overrideable.bias-present.mask-111"
OVERRIDEABLE_BACKWARD_INPUT_SHAPES = [
    "1x2x4x5", "1x3x3x3", "3x2x2x3",
]
TEST_VALUES = frozenset(
    {
        "tests/python/test_vulkan_capability_manifest.py",
        "tests/python/test_vulkan_conformance.py",
        "tests/python/test_vulkan_operator_capabilities.py",
        "tests/python/test_vulkan_linear.py",
        "tests/python/test_vulkan_vector_matmul_capability_evidence.py",
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


def _validate_vector_capture_identity(runtime: Any, name: str) -> None:
    """Recorded provenance is historical, not an implicit live GPU qualification."""
    fields = {"pytorch_version", "pytorch_git_revision", "checkout_head",
              "extension_sha256", "extension_path", "device", "hardware", "driver",
              "vulkan_api_version", "vulkan_instance_version", "execution_mode"}
    if (not isinstance(runtime, dict) or set(runtime) != fields
            or any(not isinstance(value, str) or not value.strip()
                   for value in runtime.values())
            or not re.fullmatch(r"[0-9a-f]{40}", runtime["checkout_head"])
            or not re.fullmatch(r"[0-9a-f]{64}", runtime["extension_sha256"])
            or not re.fullmatch(r"[0-9a-f]{40}", runtime["pytorch_git_revision"])
            or not Path(runtime["extension_path"]).is_absolute()
            or runtime["device"] != "vk:0"
            or runtime["execution_mode"] not in {"async", "sync"}
            or any(not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime[key])
                   for key in ("vulkan_api_version", "vulkan_instance_version"))):
        raise ValueError(f"{name}: runtime/build identity metadata is malformed")
    # The CPU oracle executes this revision; HEAD/path/mode/artifact remain capture
    # provenance. Other CPU versions need a separately qualified reference recipe.
    if (runtime["pytorch_version"] != torch.__version__
            or runtime["pytorch_git_revision"] != str(torch.version.git_version)):
        raise ValueError(f"{name}: runtime/build identity uses a different CPU reference revision")


def qualify_vector_matmul_current_runtime(coverage: dict[str, Any], root: Path) -> None:
    """Explicit live build/device/mode gate, separate from offline validation.

    A different recorded mode cannot establish current-mode execution. Relocation
    and ordinary commits are allowed when the artifact and source content agree.
    """
    validate_vector_matmul_evidence(coverage, root, require_complete=True)
    from vector_matmul_capability_evidence import runtime_identity

    current = runtime_identity(root)
    compared = set(current) - {"checkout_head", "extension_path"}
    for name, record in coverage.items():
        if name.startswith("g1."):
            recorded = record["vector_matmul_evidence"]["runtime_identity"]
            if any(recorded[key] != current[key] for key in compared):
                raise ValueError(f"{name}: capture does not qualify the current build/device/mode")


def validate_vector_matmul_evidence(
    coverage: dict[str, Any], root: Path, *, require_complete: bool = False
) -> None:
    """Bind finite G1 vector/matmul claims to source-owned executed fixtures."""
    import math
    import sys

    tests_path = root / "tests/python"
    if str(tests_path) not in sys.path:
        sys.path.insert(0, str(tests_path))
    try:
        from vector_matmul_capability_evidence import (
            REQUIRED_CASES,
            cpu_fixture_reference,
            source_identity,
        )
    except ImportError as error:
        raise ValueError("source-owned vector/matmul evidence factory is unavailable") from error

    named = {name for name in coverage if name.startswith("g1.")}
    unexpected = named - set(REQUIRED_CASES)
    if unexpected:
        raise ValueError(f"unexpected source-owned vector/matmul case IDs: {sorted(unexpected)}")
    missing = set(REQUIRED_CASES) - set(coverage)
    if require_complete and missing:
        raise ValueError(
            "required executed vector/matmul witness is missing: " + ", ".join(sorted(missing))
        )
    if not named:
        return

    expected_source = source_identity(root)
    observed_runtime: dict[str, Any] | None = None

    def validate_value_fact(value: Any, *, case_name: str, expected_shape: list[int],
                            label: str) -> tuple[float, list[float], list[float]]:
        import math

        message = f"{case_name}: {label} values violate componentwise CPU/Vulkan parity"
        if (not isinstance(value, dict) or set(value) != {
                "value_class", "shape", "cpu_values", "vulkan_values"}
                or value.get("value_class") != "finite"
                or not isinstance(value.get("shape"), list)
                or any(type(extent) is not int for extent in value["shape"])
                or value.get("shape") != expected_shape):
            raise ValueError(f"{case_name}: {label} value evidence has invalid shape or value-class metadata")
        count = 1
        for extent in expected_shape:
            if type(extent) is not int or extent < 0:
                raise ValueError(f"{case_name}: {label} value evidence has invalid shape metadata")
            count *= extent
        cpu_values, vk_values = value.get("cpu_values"), value.get("vulkan_values")
        if (not isinstance(cpu_values, list) or not isinstance(vk_values, list)
                or len(cpu_values) != count or len(vk_values) != count):
            raise ValueError(f"{case_name}: {label} value evidence has invalid flattened length")
        for item in cpu_values + vk_values:
            if type(item) not in (int, float) or not math.isfinite(item):
                raise ValueError(
                    f"{case_name}: numerical value evidence must contain finite real numbers"
                )
        errors = [abs(float(vk) - float(cpu)) for cpu, vk in zip(cpu_values, vk_values)]
        if any(error > 0.003 + 0.003 * abs(float(cpu))
               for error, cpu in zip(errors, cpu_values)):
            raise ValueError(message)
        return (max(errors, default=0.0), cpu_values, vk_values)

    def bind_cpu_values(actual, expected, *, case_name, label):
        # Exact deterministic CPU replay of this pinned tiny fixture, independent
        # of the Vulkan parity tolerance and of mutable persisted summaries.
        if actual != expected:
            raise ValueError(f"{case_name}: {label} values differ from recomputed CPU fixture")

    def bind_recipe(actual, expected, *, case_name):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError(f"{case_name}: CPU reference recipe metadata is missing or malformed")
        for key in ("upstream_seed", "mixed_probe"):
            fact = actual[key]
            if expected[key] is None:
                if fact is not None:
                    raise ValueError(f"{case_name}: CPU reference recipe differs from recomputed CPU fixture")
            elif (not isinstance(fact, dict) or set(fact) != {"shape", "values"}
                  or not isinstance(fact.get("shape"), list)
                  or any(type(x) is not int for x in fact["shape"])
                  or not isinstance(fact.get("values"), list)
                  or any(type(x) not in (int, float) or not math.isfinite(x) for x in fact["values"])
                  or fact != expected[key]):
                raise ValueError(f"{case_name}: CPU reference recipe differs from recomputed CPU fixture")
        eps = actual["fd_epsilon"]
        if ((expected["fd_epsilon"] is None and eps is not None)
                or (expected["fd_epsilon"] is not None
                    and (type(eps) not in (int, float) or eps != expected["fd_epsilon"]))):
            raise ValueError(f"{case_name}: CPU reference recipe differs from recomputed CPU fixture")

    def validate_error_summary(actual: Any, measured: float, *, case_name: str,
                               label: str) -> None:
        import math

        if (type(actual) not in (int, float) or not math.isfinite(actual)
                or not math.isclose(float(actual), measured, rel_tol=1e-7, abs_tol=1e-12)):
            raise ValueError(f"{case_name}: {label} summary does not match stored CPU/Vulkan values")

    def valid_counter(value: Any) -> bool:
        return (isinstance(value, dict)
                and set(value) == {"compute_dispatches", "vulkan_copies",
                                   "explicit_transfers", "fallbacks"}
                and all(type(count) is int and count >= 0 for count in value.values())
                and value["compute_dispatches"] > 0
                and value["explicit_transfers"] == 0
                and value["fallbacks"] == 0)

    for name in sorted(named):
        case = REQUIRED_CASES[name]
        record = coverage[name]
        if not isinstance(record, dict) or record.get("parity") is not True:
            raise ValueError(f"{name}: vector/matmul record is not an executed parity witness")
        evidence = record.get("vector_matmul_evidence")
        if not isinstance(evidence, dict) or evidence.get("case_id") != name:
            raise ValueError(f"{name}: missing actual source-owned fixture evidence")
        if evidence.get("source_identity") != expected_source:
            raise ValueError(f"{name}: source identity differs from current fixture/source")
        runtime = evidence.get("runtime_identity")
        _validate_vector_capture_identity(runtime, name)
        if observed_runtime is None:
            observed_runtime = runtime
        elif runtime != observed_runtime:
            raise ValueError(f"{name}: runtime/build identity differs across captured records")

        expected_shapes = case["shapes"]
        reference = cpu_fixture_reference(case)
        expected_input_values = reference["inputs"]
        bind_recipe(evidence.get("cpu_reference_recipe"), reference["recipe"], case_name=name)
        operands = evidence.get("inputs")
        if (not isinstance(operands, list) or len(operands) != len(expected_shapes)
                or any(not isinstance(item, dict) or set(item) != {"cpu", "vulkan", "values"}
                       for item in operands)):
            raise ValueError(f"{name}: actual input metadata differs from its source-owned fixture")
        for index, (operand, shape) in enumerate(zip(operands, expected_shapes)):
            layout = case["layouts"][index]
            if layout == "nonunit":
                expected_strides = [2]
                expected_offset = 0
            elif layout == "transpose":
                expected_strides = [1, shape[0]]
                expected_offset = 0
            else:
                expected_strides = []
                stride = 1
                for extent in reversed(shape):
                    expected_strides.append(stride)
                    stride *= extent
                expected_strides.reverse()
                expected_offset = 1 if layout == "offset" else 0
            for side, device in (("cpu", "cpu"), ("vulkan", "vk:0")):
                meta = operand[side]
                if (not isinstance(meta, dict)
                        or meta.get("dtype") != "float32"
                        or type(meta.get("rank")) is not int
                        or meta.get("rank") != len(shape)
                        or not isinstance(meta.get("shape"), list)
                        or any(type(extent) is not int for extent in meta["shape"])
                        or meta.get("shape") != shape
                        or meta.get("device") != device
                        or not isinstance(meta.get("strides"), list)
                        or any(type(stride) is not int for stride in meta["strides"])
                        or meta.get("strides") != expected_strides
                        or type(meta.get("storage_offset")) is not int
                        or meta.get("storage_offset") != expected_offset):
                    raise ValueError(
                        f"{name}: actual input metadata differs from its source-owned fixture"
                    )
            if (operands[index]["cpu"]["strides"] != operands[index]["vulkan"]["strides"]
                    or operands[index]["cpu"]["storage_offset"]
                    != operands[index]["vulkan"]["storage_offset"]):
                raise ValueError(f"{name}: CPU/Vulkan input layout witness differs")
            _, fixture_cpu_values, _ = validate_value_fact(
                operand["values"], case_name=name, expected_shape=shape,
                label=f"input{index}",
            )
            if fixture_cpu_values != expected_input_values[index]:
                raise ValueError(f"{name}: CPU input values differ from the deterministic source fixture")

        if (record.get("schema") != case["schema"]
                or record.get("primary_input") != {"dtype": "float32", "rank": len(expected_shapes[0])}
                or record.get("input_dtypes") != ["float32"]
                or record.get("input_ranks") != sorted({len(shape) for shape in expected_shapes})
                or record.get("input_shapes") != sorted({"x".join(map(str, shape))
                                                          for shape in expected_shapes if shape})
                or record.get("gradients") is not True):
            raise ValueError(f"{name}: aggregate capability metadata differs from actual inputs")

        output = evidence.get("output")
        if not isinstance(output, dict) or set(output) != {"cpu", "vulkan"}:
            raise ValueError(f"{name}: actual output metadata is missing")
        cpu_output, vk_output = output["cpu"], output["vulkan"]
        if (not isinstance(cpu_output, dict) or not isinstance(vk_output, dict)
                or cpu_output.get("dtype") != "float32" or vk_output.get("dtype") != "float32"
                or cpu_output.get("device") != "cpu" or vk_output.get("device") != "vk:0"
                or any(type(meta.get("rank")) is not int
                       or meta["rank"] != len(case["output_shape"])
                       or not isinstance(meta.get("shape"), list)
                       or any(type(extent) is not int for extent in meta["shape"])
                       for meta in (cpu_output, vk_output))
                or cpu_output.get("shape") != case["output_shape"]
                or vk_output.get("shape") != case["output_shape"]
                or cpu_output.get("rank") != vk_output.get("rank")
                or record.get("output_dtype") != "float32"
                or record.get("output_rank") != cpu_output.get("rank")):
            raise ValueError(f"{name}: actual output residency/metadata is inconsistent")

        if (evidence.get("api") != case["api"]
                or evidence.get("route_leaf") != case["route"]
                or evidence.get("seed") != case["seed"]
                or evidence.get("layouts") != case["layouts"]
                or evidence.get("capture_command") != (
                    "vector_matmul_capability_evidence.capture_all_vector_matmul_cases(device='vk:0')"
                )):
            raise ValueError(f"{name}: API, route, or seed differs from source-owned fixture")
        for side in ("cpu_forward_ops", "vulkan_forward_ops"):
            ops = evidence.get(side)
            if (not isinstance(ops, list) or any(not isinstance(op, str) for op in ops)
                    or ops != sorted(set(ops))):
                raise ValueError(f"{name}: profiler route observations are malformed")
            if case["route"] and f"aten::{case['route']}" not in ops:
                raise ValueError(f"{name}: profiler route does not contain the expected stock leaf")

        for key in ("runtime", "first_backward_runtime"):
            if not valid_counter(evidence.get(key)):
                raise ValueError(f"{name}: runtime execution evidence is malformed")
        output_error, cpu_values, _ = validate_value_fact(
            evidence.get("output_values"), case_name=name,
            expected_shape=cpu_output["shape"], label="output",
        )
        bind_cpu_values(cpu_values, reference["output"], case_name=name, label="output")
        validate_error_summary(
            evidence.get("output_max_abs_error"), output_error,
            case_name=name, label="output_max_abs_error",
        )

        graph = evidence.get("graph")
        expected_directions = ["first_reverse"]
        if case["mixed"] is not None:
            expected_directions.append("selected_mixed_reverse")
        if not isinstance(graph, dict) or graph.get("directions") != expected_directions:
            raise ValueError(f"{name}: graph history is incomplete or differs from its fixture")
        first = graph.get("first_reverse")
        if (not isinstance(first, dict)
                or first.get("targets") != [f"input{i}" for i in range(len(expected_shapes))]
                or not isinstance(first.get("history"), list)
                or len(first["history"]) != len(expected_shapes)
                or len(first.get("max_abs_errors", [])) != len(expected_shapes)
                or not isinstance(first.get("values"), list)
                or len(first["values"]) != len(expected_shapes)):
            raise ValueError(f"{name}: graph history is incomplete")
        for index, history in enumerate(first["history"]):
            if (not isinstance(history, dict) or history.get("target") != f"input{index}"
                    or history.get("cpu_requires_grad") is not True
                    or history.get("vulkan_requires_grad") is not True
                    or not history.get("cpu_grad_fn") or not history.get("vulkan_grad_fn")
                    or history.get("parity") is not True):
                raise ValueError(f"{name}: graph history is incomplete")
        for index, (error, value, shape) in enumerate(zip(
                first["max_abs_errors"], first["values"], expected_shapes)):
            grad_shape = operands[index]["cpu"]["shape"]
            measured_error, cpu_values, _ = validate_value_fact(
                value, case_name=name, expected_shape=grad_shape,
                label=f"first-reverse input{index}",
            )
            bind_cpu_values(cpu_values, reference["first"][index],
                            case_name=name, label=f"first-reverse input{index}")
            validate_error_summary(
                error, measured_error, case_name=name,
                label=f"first_reverse.max_abs_errors[{index}]",
            )

        if case["mixed"] is not None:
            mixed = graph.get("selected_mixed_reverse")
            if (not isinstance(mixed, dict)
                    or mixed.get("direction") != f"grad_input{case['mixed'][0]}_wrt_input{case['mixed'][1]}"
                    or mixed.get("target") != f"input{case['mixed'][1]}"
                    or mixed.get("cpu_nonzero") is not True
                    or mixed.get("vulkan_nonzero") is not True):
                raise ValueError(f"{name}: selected mixed graph history/signal is incomplete")
            mixed_shape = operands[case["mixed"][1]]["cpu"]["shape"]
            mixed_error, cpu_mixed_values, vk_mixed_values = validate_value_fact(
                mixed.get("values"), case_name=name,
                expected_shape=mixed_shape, label="selected mixed-reverse",
            )
            bind_cpu_values(cpu_mixed_values, reference["mixed"],
                            case_name=name, label="selected mixed-reverse")
            validate_error_summary(
                mixed.get("max_abs_error"), mixed_error, case_name=name,
                label="selected_mixed_reverse.max_abs_error",
            )
            direction_values = mixed.get("direction_values")
            if (not isinstance(direction_values, list)
                    or len(direction_values) != len(cpu_mixed_values)
                    or any(type(item) not in (int, float) or not math.isfinite(item)
                           for item in direction_values)
                    or direction_values != reference["direction"]):
                raise ValueError(f"{name}: selected mixed direction values are malformed")
            cpu_projection = sum(float(value) * float(direction)
                                 for value, direction in zip(cpu_mixed_values, direction_values))
            vk_projection = sum(float(value) * float(direction)
                                for value, direction in zip(vk_mixed_values, direction_values))
            numeric = ("cpu_analytic_projection", "cpu_centered_fd_projection",
                       "cpu_fd_abs_error", "vulkan_projection", "max_abs_error")
            if any(type(mixed.get(field)) not in (int, float)
                   or not math.isfinite(mixed[field]) for field in numeric):
                raise ValueError(f"{name}: selected mixed numerical evidence is malformed")
            analytic = mixed["cpu_analytic_projection"]
            finite_difference = mixed["cpu_centered_fd_projection"]
            fd_error = abs(analytic - finite_difference)
            if (not math.isclose(fd_error, mixed["cpu_fd_abs_error"], rel_tol=1e-7, abs_tol=1e-9)
                    or fd_error > max(0.002, 0.008 * abs(analytic))):
                raise ValueError(f"{name}: CPU finite-difference projection does not match its analytic signal")
            if (not math.isclose(analytic, cpu_projection, rel_tol=1e-7, abs_tol=1e-9)
                    or not math.isclose(mixed["vulkan_projection"], vk_projection,
                                        rel_tol=1e-7, abs_tol=1e-9)):
                raise ValueError(f"{name}: stored mixed projections do not match stored tensor values")
            if abs(vk_projection - cpu_projection) > 0.003 + 0.003 * abs(cpu_projection):
                raise ValueError(f"{name}: Vulkan mixed projection differs from its CPU analytic oracle")
            if finite_difference != reference["fd"]:
                raise ValueError(f"{name}: centered FD differs from recomputed CPU fixture")
            if not valid_counter(mixed.get("runtime")):
                raise ValueError(f"{name}: runtime execution evidence is malformed")


def validate_vector_matmul_manifest_bindings(
    entries: list[dict[str, Any]], coverage: dict[str, Any], root: Path
) -> None:
    """Require source fixture IDs and graph directions in their schema entries."""
    import sys

    tests_path = str(root / "tests/python")
    if tests_path not in sys.path:
        sys.path.insert(0, tests_path)
    from vector_matmul_capability_evidence import REQUIRED_CASES

    by_schema = {entry.get("schema"): entry for entry in entries}
    schemas = sorted({case["schema"] for case in REQUIRED_CASES.values()})
    for schema in schemas:
        entry = by_schema.get(schema)
        if entry is None:
            raise ValueError(f"{schema}: source-owned vector/matmul evidence has no manifest entry")
        expected_names = {name for name, case in REQUIRED_CASES.items()
                          if case["schema"] == schema}
        supported = {case.get("name") for case in entry.get("test_cases", [])
                     if isinstance(case, dict) and case.get("supported") is True}
        witness_cases = set(entry.get("witnesses", {}).get("cases", []))
        if not expected_names <= supported or not expected_names <= witness_cases:
            raise ValueError(f"{schema}: manifest is missing source-owned vector/matmul test-case bindings")
        first_names = {name for name in expected_names
                       if "first_reverse" in coverage[name]["vector_matmul_evidence"]["graph"]["directions"]}
        mixed_names = {name for name in expected_names
                       if "selected_mixed_reverse" in coverage[name]["vector_matmul_evidence"]["graph"]["directions"]}
        first_links = set(entry["witnesses"].get("reverse_first_order_graph_cases", []))
        mixed_links = set(entry["witnesses"].get("reverse_second_order_cases", []))
        if (entry.get("autograd") in {
                "reverse_first_order_graph_witnessed",
                "reverse_finite_second_order_witnessed",
                "reverse_selected_third_order_witnessed",
        } and not first_names <= first_links):
            raise ValueError(f"{schema}: source-owned first-reverse graph links are missing")
        if mixed_names and not mixed_names <= mixed_links:
            raise ValueError(f"{schema}: source-owned finite mixed-reverse links are missing")


def validate_convolution_evidence(
    coverage: dict[str, Any], *, require_complete: bool = False
) -> None:
    """Bind the new convolution witnesses to their names, real tensor roles and schemas."""
    required = ORDINARY_CONVOLUTION_REQUIRED_CASES
    named = {name for name in coverage if name.startswith("convolution.forward.bias-")
             or name.startswith("convolution.backward.bias-")}
    if named - required:
        raise ValueError(f"unexpected named convolution evidence cases: {sorted(named - required)}")
    if require_complete and not required <= set(coverage):
        missing = sorted(required - set(coverage))
        raise ValueError(f"required executed convolution witness is missing: {missing}")

    input_shape = [1, 4, 5, 6]
    weight_shape = [4, 2, 2, 2]
    bias_shape = [4]
    grad_shape = [1, 4, 4, 5]
    def expected_operand(shape):
        strides = (
            [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1]
            if len(shape) == 4 else [1]
        )
        return {"defined": True, "dtype": "float32", "rank": len(shape),
                "shape": shape, "strides": strides, "storage_offset": 0,
                "format": "contiguous"}

    for name in sorted(named):
        record = coverage[name]
        if not isinstance(record, dict) or record.get("parity") is not True:
            raise ValueError(f"{name}: convolution evidence must be a parity-checked execution")
        is_forward = name.startswith("convolution.forward.")
        bias_present = ".bias-present" in name
        expected_schema = "aten::convolution.default" if is_forward else "aten::convolution_backward.default"
        if record.get("schema") != expected_schema:
            raise ValueError(f"{name}: convolution schema does not match case identity")
        context = record.get("convolution_context")
        expected_oracle = "torch.nn.functional.conv2d" if is_forward else "aten::convolution_backward.default"
        expected_context_keys = {"device", "cpu_oracle", "forward_bias_present"}
        if not is_forward:
            expected_context_keys.add("output_mask")
        if (not isinstance(context, dict) or set(context) != expected_context_keys
                or context.get("device") != "vk:0" or context.get("cpu_oracle") != expected_oracle):
            raise ValueError(f"{name}: convolution execution device or CPU oracle is inconsistent")
        if context.get("forward_bias_present") is not bias_present:
            raise ValueError(f"{name}: forward bias state does not match case identity")
        if is_forward:
            mask = None
            expected_roles = {"input": expected_operand(input_shape),
                              "weight": expected_operand(weight_shape),
                              "bias": expected_operand(bias_shape) if bias_present else {"defined": False}}
            expected_args = {"stride": [1, 1], "padding": [0, 0],
                             "dilation": [1, 1], "groups": 2}
            if "output_slots" in record:
                raise ValueError(f"{name}: forward single-Tensor result cannot contain tuple output slots")
            if record.get("output_dtype") != "float32" or record.get("output_rank") != 4:
                raise ValueError(f"{name}: forward output witness differs from actual result")
            expected_shapes = [input_shape, weight_shape] + ([bias_shape] if bias_present else [])
        else:
            try:
                mask_text = name.rsplit("mask-", 1)[1]
                if len(mask_text) != 3 or set(mask_text) - {"0", "1"}:
                    raise ValueError
                mask = [digit == "1" for digit in mask_text]
            except (IndexError, ValueError) as error:
                raise ValueError(f"{name}: invalid output mask case identity") from error
            if context.get("output_mask") != mask:
                raise ValueError(f"{name}: output_mask does not match case identity")
            expected_roles = {"grad_output": expected_operand(grad_shape),
                              "input": expected_operand(input_shape),
                              "weight": expected_operand(weight_shape),
                              "bias": expected_operand(bias_shape) if bias_present else {"defined": False}}
            expected_args = {"bias_sizes": [4], "stride": [1, 1], "padding": [0, 0],
                             "dilation": [1, 1], "transposed": False,
                             "output_padding": [0, 0], "groups": 2, "output_mask": mask}
            expected_shapes = [grad_shape, input_shape, weight_shape] + ([bias_shape] if bias_present else [])
            slots = record.get("output_slots")
            expected_slots = []
            shapes_by_slot = (input_shape, weight_shape, bias_shape)
            for index, requested in enumerate(mask):
                slot = {"index": index, "defined": requested}
                if requested:
                    shape = shapes_by_slot[index]
                    slot.update(dtype="float32", rank=len(shape), shape=shape)
                expected_slots.append(slot)
            if slots != expected_slots:
                raise ValueError(f"{name}: output slots do not match schema mask/geometry")
            if "output_dtype" in record or "output_rank" in record:
                raise ValueError(f"{name}: tuple results cannot use legacy single-Tensor output witnesses")
        operands = record.get("convolution_operands")
        if operands != expected_roles:
            raise ValueError(f"{name}: actual convolution operand metadata does not match named geometry/roles")
        if record.get("schema_args") != expected_args:
            raise ValueError(f"{name}: actual convolution schema arguments do not match case identity")
        primary = record.get("primary_input")
        if primary != {"dtype": "float32", "rank": 4}:
            raise ValueError(f"{name}: primary schema input dtype/rank witness is inconsistent")
        expected_ranks = sorted({operand["rank"] for operand in expected_roles.values()
                                 if operand.get("defined")})
        if record.get("input_dtypes") != ["float32"] or record.get("input_ranks") != expected_ranks:
            raise ValueError(f"{name}: aggregate convolution input dtype/rank witnesses are inconsistent")
        shape_tokens = sorted({"x".join(map(str, shape)) for shape in expected_shapes})
        if record.get("input_shapes") != shape_tokens:
            raise ValueError(f"{name}: aggregate convolution shapes differ from actual operand roles")
        execution = record.get("execution")
        execution_keys = {"compute_dispatches", "vulkan_copies", "explicit_transfers",
                          "fallbacks", "buffer_creations_delta", "live_allocations_delta"}
        if not isinstance(execution, dict) or set(execution) != execution_keys or any(
            type(value) is not int or value < 0 for value in execution.values()
        ):
            raise ValueError(f"{name}: execution deltas are missing, malformed, or negative")
        expected_dispatches = 1 if is_forward else sum(mask)
        if execution["compute_dispatches"] != expected_dispatches:
            raise ValueError(f"{name}: dispatch count does not match requested outputs")
        if any(execution[key] for key in ("vulkan_copies", "explicit_transfers", "fallbacks")):
            raise ValueError(f"{name}: convolution witness recorded copy, transfer, or fallback work")
        if not is_forward and not any(mask) and any(
            execution[key] for key in ("buffer_creations_delta", "live_allocations_delta")
        ):
            raise ValueError(f"{name}: mask 000 recorded result allocation")
    _validate_transposed_convolution_evidence(coverage, require_complete=require_complete)


def _validate_transposed_convolution_evidence(coverage: dict[str, Any], *, require_complete: bool) -> None:
    input_shape, weight_shape, grad_shape, bias_shape = (
        [2, 4, 3, 4], [4, 3, 2, 3], [2, 6, 8, 6], [6]
    )

    def operand(shape):
        strides = [1] * len(shape)
        for index in range(len(shape) - 2, -1, -1):
            strides[index] = strides[index + 1] * shape[index + 1]
        return {"defined": True, "dtype": "float32", "rank": len(shape),
                "shape": shape, "strides": strides, "storage_offset": 0,
                "format": "contiguous"}

    def warm_bias(present, device):
        if not present:
            return {"defined": False, "dtype": None, "rank": None, "shape": None,
                    "strides": None, "storage_offset": None, "device": device}
        return {"defined": True, "dtype": "float32", "rank": 1, "shape": bias_shape,
                "strides": [1], "storage_offset": 0, "device": device}

    named = {name for name in coverage if name.startswith("convolution.transposed.")}
    unexpected = named - TRANSPOSED_CONVOLUTION_REQUIRED_CASES
    if unexpected:
        raise ValueError(f"unexpected named transposed convolution cases: {sorted(unexpected)}")
    missing = TRANSPOSED_CONVOLUTION_REQUIRED_CASES - set(coverage)
    if require_complete and missing:
        raise ValueError(f"required executed convolution witness is missing: {sorted(missing)}")
    for name in sorted(named):
        record = coverage[name]
        forward = name.startswith("convolution.transposed.forward.")
        bias_present = ".bias-present" in name
        direction = "forward" if forward else "backward"
        schema = "aten::convolution.default" if forward else "aten::convolution_backward.default"
        if not isinstance(record, dict) or record.get("schema") != schema or record.get("parity") is not True:
            raise ValueError(f"{name}: case identity/schema is not a parity-checked execution")
        context = record.get("convolution_context")
        oracle = "torch.nn.functional.conv_transpose2d" if forward else "aten::convolution_backward.default"
        expected_context_keys = {
            "device", "cpu_oracle", "forward_bias_present", "direction",
            "warm_forward_bias", "expected_numerical_operations",
            "expected_result_slots", "mapping_id",
        }
        if not forward:
            expected_context_keys.add("output_mask")
        if not isinstance(context, dict) or "warm_forward_bias" not in context:
            raise ValueError(f"{name}: warm bias context is missing or malformed")
        if (set(context) != expected_context_keys
                or context.get("device") != TRANSPOSED_DEVICE
                or context.get("direction") != direction or context.get("cpu_oracle") != oracle
                or context.get("mapping_id") != TRANSPOSED_MAPPING_ID):
            raise ValueError(f"{name}: direction, device, oracle, or case identity mismatch")
        warm = context.get("warm_forward_bias")
        if not isinstance(warm, dict) or set(warm) != {"cpu", "vulkan"}:
            raise ValueError(f"{name}: warm bias context is missing or malformed")
        if warm.get("cpu") != warm_bias(bias_present, "cpu") or warm.get("vulkan") != warm_bias(bias_present, TRANSPOSED_DEVICE):
            raise ValueError(f"{name}: warm bias context metadata/parity mismatch")
        if context.get("forward_bias_present") is not bias_present:
            raise ValueError(f"{name}: forward_bias_present does not match actual warm bias")
        if forward:
            expected_ops, expected_slots = [1], [0]
            roles = {"input": operand(input_shape), "weight": operand(weight_shape),
                     "bias": operand(bias_shape) if bias_present else {"defined": False}}
            args = {"stride": [2, 1], "padding": [1, 1], "dilation": [3, 2],
                    "transposed": True, "output_padding": [2, 0], "groups": 2}
            shapes = [input_shape, weight_shape] + ([bias_shape] if bias_present else [])
            if (record.get("output_dtype") != "float32" or record.get("output_rank") != 4
                    or record.get("output_shape") != grad_shape or "output_slots" in record):
                raise ValueError(f"{name}: forward output shape/dtype witness mismatch")
            mask = None
        else:
            try:
                bits = name.rsplit("mask-", 1)[1]
                if len(bits) != 3 or set(bits) - {"0", "1"}:
                    raise ValueError
                mask = [bit == "1" for bit in bits]
            except (IndexError, ValueError) as error:
                raise ValueError(f"{name}: invalid output mask case identity") from error
            expected_ops = [op for op, bit in zip((0, 2, 3), mask) if bit]
            expected_slots = [slot for slot, bit in enumerate(mask) if bit]
            roles = {"grad_output": operand(grad_shape), "input": operand(input_shape),
                     "weight": operand(weight_shape)}
            bias_sizes = record.get("schema_args", {}).get("bias_sizes")
            if type(bias_sizes) not in (list, type(None)) or bias_sizes not in (None, [], [6], [999]):
                raise ValueError(f"{name}: advisory bias_sizes metadata is invalid")
            args = {"bias_sizes": bias_sizes, "stride": [2, 1], "padding": [1, 1],
                    "dilation": [3, 2], "transposed": True, "output_padding": [2, 0],
                    "groups": 2, "output_mask": mask}
            slots = []
            for index, requested in enumerate(mask):
                slot = {"index": index, "defined": requested}
                if requested:
                    shape = (input_shape, weight_shape, bias_shape)[index]
                    slot.update(dtype="float32", rank=len(shape), shape=shape)
                slots.append(slot)
            if (record.get("output_slots") != slots or "output_dtype" in record
                    or "output_rank" in record or "output_shape" in record):
                raise ValueError(f"{name}: output slot definitions/shapes mismatch")
            if context.get("output_mask") != mask:
                raise ValueError(f"{name}: output_mask does not match case identity")
            shapes = [grad_shape, input_shape, weight_shape]
        if context.get("expected_numerical_operations") != expected_ops:
            raise ValueError(f"{name}: expected numerical operation mapping mismatch")
        if context.get("expected_result_slots") != expected_slots:
            raise ValueError(f"{name}: expected semantic result slot mapping mismatch")
        if record.get("convolution_operands") != roles or record.get("schema_args") != args:
            raise ValueError(f"{name}: actual operand metadata or schema args mismatch")
        if record.get("primary_input") != {"dtype": "float32", "rank": 4}:
            raise ValueError(f"{name}: primary schema input dtype/rank mismatch")
        expected_ranks = sorted({len(shape) for shape in shapes})
        if (record.get("input_dtypes") != ["float32"]
                or record.get("input_ranks") != expected_ranks
                or record.get("input_shapes") != sorted({"x".join(map(str, shape)) for shape in shapes})):
            raise ValueError(f"{name}: aggregate operand metadata mismatch")
        execution = record.get("execution")
        keys = {"compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks",
                "buffer_creations_delta", "live_allocations_delta"}
        if not isinstance(execution, dict) or set(execution) != keys or any(
            type(value) is not int or value < 0 for value in execution.values()
        ):
            raise ValueError(f"{name}: malformed execution counters")
        if execution["compute_dispatches"] != (1 if forward else sum(mask)):
            raise ValueError(f"{name}: dispatch count mismatch")
        if any(execution[key] for key in ("vulkan_copies", "explicit_transfers", "fallbacks")):
            raise ValueError(f"{name}: copy, transfer, or fallback counters are nonzero")
        if not forward and not any(mask) and any(execution[key] for key in ("buffer_creations_delta", "live_allocations_delta")):
            raise ValueError(f"{name}: mask 000 recorded buffer creation or live allocation")


def validate_convolution_manifest_bindings(
    entries: list[dict[str, Any]], coverage: dict[str, Any]
) -> None:
    """Require ordinary convolution declarations to match executed coverage."""
    schemas = ("aten::convolution.default", "aten::convolution_backward.default")
    for schema in schemas:
        entry = next((item for item in entries if item.get("schema") == schema), None)
        if entry is None:
            continue
        actual = {
            name: record for name, record in coverage.items()
            if isinstance(record, dict)
            and record.get("schema") == schema
            and record.get("parity") is True
        }
        if not actual:
            raise ValueError(f"{schema}: no validated convolution coverage records")
        try:
            pairs = sorted({
                (record["primary_input"]["dtype"], record["primary_input"]["rank"])
                for record in actual.values()
            })
            output_dtypes = set()
            for record in actual.values():
                if "output_dtype" in record:
                    output_dtypes.add(record["output_dtype"])
                else:
                    output_dtypes.update(
                        slot["dtype"] for slot in record["output_slots"]
                        if slot["defined"]
                    )
            expected = {
                "dtypes": sorted({dtype for dtype, _ in pairs}),
                "ranks": sorted({rank for _, rank in pairs}),
                "pairs": [list(pair) for pair in pairs],
                "cases": sorted(actual),
            }
            graph_first = sorted(name for name, record in actual.items()
                                 if record.get("graph_autograd") and "first_reverse" in record["graph_autograd"]["directions"])
            graph_second = sorted(name for name, record in actual.items()
                                  if record.get("graph_autograd") and any(direction.startswith("second_reverse") for direction in record["graph_autograd"]["directions"]))
            graph_third = sorted(name for name, record in actual.items()
                                 if record.get("graph_autograd") and "selected_third_d_g_d_x_d_w" in record["graph_autograd"]["directions"])
            if graph_first: expected["reverse_first_order_graph_cases"] = graph_first
            if graph_second: expected["reverse_second_order_cases"] = graph_second
            if graph_third: expected["reverse_selected_third_order_cases"] = graph_third
        except (KeyError, TypeError) as error:
            raise ValueError(f"{schema}: malformed validated convolution coverage metadata") from error
        if entry.get("witnesses") != expected:
            raise ValueError(f"{schema}: manifest witnesses differ from validated convolution records")
        if (
            entry["dtypes"]["inputs"] != expected["dtypes"]
            or entry["dtypes"]["outputs"] != sorted(output_dtypes)
        ):
            raise ValueError(f"{schema}: manifest dtypes differ from validated convolution records")
        if entry["ranks"] != {"min": expected["ranks"][0], "max": expected["ranks"][-1]}:
            raise ValueError(f"{schema}: manifest ranks differ from validated convolution records")
        supported_cases = {
            case["name"] for case in entry.get("test_cases", [])
            if isinstance(case, dict) and case.get("supported") is True
        }
        required_named = (
            {"convolution.forward.bias-present", "convolution.forward.bias-absent"}
            if schema == "aten::convolution.default"
            else {
                f"convolution.backward.bias-{state}.mask-{mask:03b}"
                for state in ("present", "absent") for mask in range(8)
            }
        )
        if not required_named <= supported_cases:
            raise ValueError(
                f"{schema}: required named convolution coverage cases are not supported test_cases"
            )
        transposed_required = {
            name for name in TRANSPOSED_CONVOLUTION_REQUIRED_CASES
            if (schema == "aten::convolution.default")
            == name.startswith("convolution.transposed.forward.")
        }
        if not transposed_required <= supported_cases:
            raise ValueError(
                f"{schema}: required named transposed convolution cases are not supported test_cases"
            )


def validate_overrideable_backward_evidence(
    entries: list[dict[str, Any]], coverage: dict[str, Any]
) -> None:
    """Bind the promoted ten-argument adapter witness to its fixed numerical leaf."""
    entry = next(
        (item for item in entries if item.get("schema") == OVERRIDEABLE_BACKWARD_SCHEMA),
        None,
    )
    record = coverage.get(OVERRIDEABLE_BACKWARD_CASE)
    if entry is None or not isinstance(record, dict):
        raise ValueError("overrideable backward requires its source-owned numerical witness and manifest entry")
    if (record.get("schema") != OVERRIDEABLE_BACKWARD_SCHEMA
            or type(record.get("parity")) is not bool or record["parity"] is not True):
        raise ValueError("overrideable backward witness must be a parity-checked exact-schema execution")
    expected_operands = [
        {"role": "primary_input", "dtype": "float32", "rank": 4},
        {"role": "operand", "dtype": "float32", "rank": 4},
        {"role": "operand", "dtype": "float32", "rank": 4},
    ]
    if (record.get("operands") != expected_operands
            or record.get("primary_input") != {"dtype": "float32", "rank": 4}
            or record.get("input_dtypes") != ["float32"]
            or record.get("input_ranks") != [4]
            or record.get("input_shapes") != OVERRIDEABLE_BACKWARD_INPUT_SHAPES
            or record.get("gradients") is not False):
        raise ValueError("overrideable backward witness differs from its fixed float32 rank-four leaf inputs")

    expected_witnesses = {
        "dtypes": ["float32"], "ranks": [4],
        "pairs": [["float32", 4]], "cases": [OVERRIDEABLE_BACKWARD_CASE],
    }
    supported_cases = {
        case.get("name") for case in entry.get("test_cases", [])
        if isinstance(case, dict) and case.get("supported") is True
    }
    if (entry.get("status") != "supported"
            or entry.get("dtypes") != {"inputs": ["float32"], "outputs": ["float32"]}
            or entry.get("ranks") != {"min": 4, "max": 4}
            or entry.get("witnesses") != expected_witnesses
            or supported_cases != {OVERRIDEABLE_BACKWARD_CASE}):
        raise ValueError("overrideable backward manifest contract differs from its fixed source-bound witness")


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
    """Validate versioned stock composite routes and their executed witness links."""
    expected_schemas = KNOWN_STOCK_COMPOSITE_SCHEMAS
    route_entries = [entry for entry in entries if entry.get("schema") in expected_schemas]
    if not routes and not route_entries:
        return
    if set(routes) != expected_schemas:
        raise ValueError(
            f"unknown stock-composite route(s): {sorted(set(routes) - expected_schemas)}"
        )
    entries_by_schema = {entry["schema"]: entry for entry in entries}
    for expected_schema, route in routes.items():
        contract = STOCK_COMPOSITE_ROUTES[expected_schema]
        if (not isinstance(route, dict) or set(route) != set(contract)
                or route.get("schema") != contract["schema"]
                or route.get("reference") != contract["reference"]
                or route.get("dependencies") != contract["dependencies"]):
            raise ValueError("stock-composite route descriptor/reference/dependency closure is unqualified")
        if torch.__version__.split("+", 1)[0] != route["reference"]["pytorch_version"]:
            raise ValueError("stock-composite route PyTorch reference version mismatch")
        for dependency in route["dependencies"]:
            if not schema_exists(dependency["schema"]):
                raise ValueError(f"stock-composite dependency schema is unresolved: {dependency['schema']}")
            if dependency["dispatch"] == "stock_generated_privateuse1" and dependency["schema"] in source:
                raise ValueError(f"stock-generated dependency is misclassified as direct source: {dependency['schema']}")
            for leaf_key in ("vulkan_leaf", "alternate_vulkan_leaf"):
                leaf = dependency.get(leaf_key)
                if leaf is not None and (not schema_exists(leaf) or leaf not in source or leaf in explicit_rejected):
                    raise ValueError(f"stock-composite dependency closure is unresolved at {leaf}")
        if expected_schema in source or expected_schema in explicit_rejected:
            raise ValueError("stock-composite route duplicates a direct or rejected source registration")
        entry = entries_by_schema.get(expected_schema)
        if entry is None or entry.get("status") != "supported":
            raise ValueError("stock-composite route has no supported manifest entry")
        cases = {case.get("name") for case in entry.get("test_cases", [])
                 if isinstance(case, dict) and case.get("supported") is True}
        evidence_names = route["evidence_cases"]
        if (evidence_names != contract["evidence_cases"]
                or not evidence_names or len(set(evidence_names)) != len(evidence_names)
                or not set(evidence_names) <= cases):
            raise ValueError("stock-composite route evidence case links are missing or unsupported")
        if expected_schema == STOCK_COMPOSITE_ROUTE_CONTRACT["schema"]:
            expected_shapes = {"view.reshape.copy.trainable-seed": "2x3",
                               "view.reshape.offset-copy.second-order": "4x6"}
            for name in evidence_names:
                record = coverage.get(name)
                if (name not in expected_shapes or not isinstance(record, dict)
                        or record.get("schema") != expected_schema or record.get("parity") is not True
                        or record.get("gradients") is not True
                        or record.get("primary_input") != {"dtype": "float32", "rank": 2}
                        or record.get("input_shapes") != [expected_shapes[name]]
                        or record.get("reverse_autograd") != {"order": 2, "graph_preserved": True}):
                    raise ValueError(f"stock-composite route lacks matching executed route evidence: {name}")
                execution = record.get("execution")
                if (not isinstance(execution, dict) or set(execution) != {
                        "mode", "compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks"}
                        or execution.get("mode") != "copy"
                        or any(type(execution.get(key)) is not int for key in (
                            "compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks"))
                        or execution["vulkan_copies"] <= 0 or execution["compute_dispatches"] != 0
                        or execution["explicit_transfers"] != 0 or execution["fallbacks"] != 0):
                    raise ValueError(f"stock-composite route execution evidence is invalid: {name}")
        elif expected_schema == STOCK_MATMUL_ROUTE_CONTRACT["schema"]:
            validate_vector_matmul_evidence(
                coverage, Path(__file__).resolve().parents[1], require_complete=True
            )
            from vector_matmul_capability_evidence import REQUIRED_CASES
            expected_route_cases = set(STOCK_MATMUL_ROUTE_CONTRACT["evidence_cases"])
            for name in expected_route_cases:
                record = coverage.get(name)
                evidence = record.get("vector_matmul_evidence") if isinstance(record, dict) else None
                if (name not in REQUIRED_CASES or not isinstance(evidence, dict)
                        or evidence.get("case_id") != name
                        or evidence.get("route_leaf") != REQUIRED_CASES[name]["route"]
                        or evidence.get("api") != REQUIRED_CASES[name]["api"]):
                    raise ValueError(f"stock matmul route lacks matching executed route evidence: {name}")
        else:
            validate_stock_linear_route_evidence(coverage)


def validate_stock_linear_route_evidence(coverage: dict[str, Any]) -> None:
    """Validate all 84 persisted CPU/Vulkan-executed stock Linear route records."""
    expected = {case["name"]: case for case in STOCK_LINEAR_ROUTE_CASES}
    names = {name for name in coverage if name.startswith("linear.rank")}
    if names != set(expected):
        missing = sorted(set(expected) - names)
        extra = sorted(names - set(expected))
        raise ValueError(
            f"executed Linear route evidence IDs are incomplete or unexpected: missing={missing}, extra={extra}"
        )

    def check_counter(record: Any, case_name: str, *, required: bool) -> None:
        if not isinstance(record, dict) or set(record) != {
            "compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks"
        }:
            raise ValueError(f"{case_name}: executed Linear route counter record is malformed")
        if any(type(value) is not int or value < 0 for value in record.values()):
            raise ValueError(f"{case_name}: executed Linear route counters must be nonnegative integers")
        if ((required and record["compute_dispatches"] < 1)
                or record["explicit_transfers"] != 0 or record["fallbacks"] != 0):
            raise ValueError(f"{case_name}: executed Linear route counters show no compute or fallback/transfer")

    for name, case in expected.items():
        record = coverage.get(name)
        route = record.get("stock_linear_route") if isinstance(record, dict) else None
        if not isinstance(record, dict) or not isinstance(route, dict):
            raise ValueError(f"{name}: missing executed Linear route evidence record")
        expected_primary = {"dtype": "float32", "rank": case["rank"]}
        expected_shape = case["input_shape"]
        expected_input_shapes = sorted({
            "x".join(map(str, expected_shape)), "4x3",
            *( ["4"] if case["bias_present"] else [] ),
        })
        expected_input_ranks = sorted({case["rank"], 2, *( [1] if case["bias_present"] else [] )})
        expected_operands = [
            {"role": "primary_input", "dtype": "float32", "rank": case["rank"]},
            {"role": "operand", "dtype": "float32", "rank": 2},
        ]
        if case["bias_present"]:
            expected_operands.append({"role": "operand", "dtype": "float32", "rank": 1})
        if (record.get("schema") != "aten::linear.default"
                or record.get("primary_input") != expected_primary
                or record.get("operands") != expected_operands
                or record.get("input_dtypes") != ["float32"]
                or record.get("input_ranks") != expected_input_ranks
                or record.get("input_shapes") != expected_input_shapes
                or record.get("parity") is not True
                or record.get("gradients") is not (case["state"] != "no-grad")
                or record.get("output_dtype") != "float32"
                or record.get("output_rank") != case["rank"]):
            raise ValueError(f"{name}: executed Linear route aggregate metadata disagrees with source case")

        required_route_keys = {
            "case_id", "schema", "rank", "input_shape", "layout", "state",
            "bias_present", "input_metadata", "cpu_forward_ops", "vulkan_forward_ops",
            "matrix_leaf", "route", "required_forward_ops", "forward_counters",
            "first_gradient_facts", "first_backward_counters",
            "first_gradient_graph_preserved", "selected_second_direction",
            "selected_second_counters", "output_device", "output_requires_grad",
            "grad_mode",
        }
        if set(route) != required_route_keys:
            raise ValueError(f"{name}: executed Linear route fields are malformed")
        for key, expected_value in (
            ("case_id", name), ("schema", "aten::linear.default"),
            ("rank", case["rank"]), ("input_shape", expected_shape),
            ("layout", case["layout"]), ("state", case["state"]),
            ("bias_present", case["bias_present"]),
            ("matrix_leaf", case["matrix_leaf"]), ("route", case["route"]),
            ("required_forward_ops", case["required_forward_ops"]),
        ):
            if route.get(key) != expected_value:
                raise ValueError(f"{name}: executed Linear route {key} differs from its source contract")
        metadata = route["input_metadata"]
        if not isinstance(metadata, dict) or set(metadata) != {"cpu", "vulkan"}:
            raise ValueError(f"{name}: executed Linear route input layout metadata is malformed")
        cpu_meta, vk_meta = metadata["cpu"], metadata["vulkan"]
        if not isinstance(cpu_meta, dict) or set(cpu_meta) != {"shape", "strides", "storage_offset"}:
            raise ValueError(f"{name}: CPU input layout metadata is malformed")
        if not isinstance(vk_meta, dict) or set(vk_meta) != {"shape", "strides", "storage_offset"}:
            raise ValueError(f"{name}: Vulkan input layout metadata is malformed")
        if cpu_meta != vk_meta or cpu_meta["shape"] != expected_shape:
            raise ValueError(f"{name}: CPU/Vulkan input layout metadata differs")
        if case["layout"] == "contiguous":
            expected_strides = []
            stride = 1
            for size in reversed(expected_shape):
                expected_strides.append(stride)
                stride *= size
            expected_strides.reverse()
            expected_offset = 0
        else:
            base_shape = [2] * (case["rank"] - 1) + [5]
            expected_strides = []
            stride = 1
            for size in reversed(base_shape):
                expected_strides.append(stride)
                stride *= size
            expected_strides.reverse()
            expected_strides[0], expected_strides[1] = expected_strides[1], expected_strides[0]
            expected_offset = 1
        if cpu_meta["strides"] != expected_strides or cpu_meta["storage_offset"] != expected_offset:
            raise ValueError(f"{name}: executed Linear input strides/offset differ from source layout")

        for ops_key in ("cpu_forward_ops", "vulkan_forward_ops"):
            ops = route[ops_key]
            if not isinstance(ops, list) or ops != sorted(set(ops)) or any(not isinstance(op, str) for op in ops):
                raise ValueError(f"{name}: profiler operator record is malformed")
            if case["matrix_leaf"].removesuffix(".default") not in ops:
                raise ValueError(f"{name}: actual profiler trace lacks its declared matrix leaf")
            required_ops = {op.removesuffix(".default") for op in case["required_forward_ops"]}
            if not required_ops <= set(ops):
                raise ValueError(f"{name}: actual profiler trace lacks required stock-route operators")

        check_counter(route["forward_counters"], name, required=True)
        for key, expected_value in case["runtime_counters"].items():
            counter_key = key.removesuffix("_min")
            observed_value = route["forward_counters"].get(counter_key)
            if (key.endswith("_min") and observed_value < expected_value):
                raise ValueError(f"{name}: runtime dispatch count is below source contract")
            if (not key.endswith("_min") and observed_value != expected_value):
                raise ValueError(f"{name}: runtime counter differs from source contract for {key}")
        if route["output_device"] != "vk:0" or route["output_requires_grad"] != (case["state"] != "no-grad"):
            raise ValueError(f"{name}: Linear output residency/grad-mode evidence is inconsistent")
        expected_grad_mode = "no_grad" if case["state"] == "no-grad" else "enabled"
        if route["grad_mode"] != expected_grad_mode:
            raise ValueError(f"{name}: actual Linear forward grad mode is incorrect")

        expected_targets = [] if case["state"] == "no-grad" else ["input"]
        if case["state"] == "trainable":
            expected_targets.append("weight")
            if case["bias_present"]:
                expected_targets.append("bias")
        facts = route["first_gradient_facts"]
        if not isinstance(facts, list) or [item.get("name") for item in facts if isinstance(item, dict)] != expected_targets:
            raise ValueError(f"{name}: first-gradient target records differ from trainability contract")
        if case["state"] == "no-grad":
            if route["first_backward_counters"] is not None or facts:
                raise ValueError(f"{name}: no-grad route contains backward evidence")
        else:
            check_counter(route["first_backward_counters"], name, required=True)
            expected_gradient_shapes = {"input": expected_shape, "weight": [4, 3], "bias": [4]}
            for fact in facts:
                history_fields = ("cpu_requires_grad", "vulkan_requires_grad")
                grad_fn_fields = ("cpu_grad_fn", "vulkan_grad_fn")
                if (set(fact) != {"name", "cpu_shape", "vulkan_shape", "cpu_requires_grad",
                                  "vulkan_requires_grad", "cpu_grad_fn", "vulkan_grad_fn", "parity"}
                        or fact["name"] not in expected_gradient_shapes
                        or fact["cpu_shape"] != expected_gradient_shapes.get(fact["name"])
                        or fact["cpu_shape"] != fact["vulkan_shape"]
                        or fact["parity"] is not True
                        or any(type(fact[key]) is not bool for key in history_fields)
                        or fact["cpu_requires_grad"] != fact["vulkan_requires_grad"]
                        or (fact["cpu_grad_fn"] is None) != (fact["vulkan_grad_fn"] is None)
                        or fact["cpu_requires_grad"] != (case["state"] == "trainable")):
                    raise ValueError(f"{name}: CPU/Vulkan first-gradient history record is inconsistent")
                for history_key, grad_fn_key in zip(history_fields, grad_fn_fields):
                    history = fact[history_key]
                    grad_fn = fact[grad_fn_key]
                    if (history and (type(grad_fn) is not str or not grad_fn)
                            or not history and grad_fn is not None):
                        raise ValueError(f"{name}: CPU/Vulkan first-gradient history record is malformed")

        second_expected = bool(case["second_direction"])
        if route["first_gradient_graph_preserved"] is not (case["state"] == "trainable"):
            raise ValueError(f"{name}: first-gradient graph evidence is missing or overclaimed")
        second = route["selected_second_direction"]
        if second_expected:
            if (not isinstance(second, dict)
                    or set(second) != {
                        "direction", "target_names", "cpu_vulkan_parity",
                        "cpu_requires_grad", "vulkan_requires_grad",
                    }
                    or second.get("direction") != ["0.25", "0.50", "0.75"]
                    or second.get("cpu_vulkan_parity") is not True
                    or second.get("target_names") != expected_targets
                    or second.get("cpu_requires_grad") != [True] * len(expected_targets)
                    or second.get("vulkan_requires_grad") != [True] * len(expected_targets)):
                raise ValueError(f"{name}: selected second-direction graph facts are invalid")
            check_counter(route["selected_second_counters"], name, required=True)
        elif second is not None or route["selected_second_counters"] is not None:
            raise ValueError(f"{name}: unqualified Linear case claims a selected second derivative")
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

    coverage_cache: dict[str, Any] | None = None

    def coverage_evidence() -> dict[str, Any]:
        nonlocal coverage_cache
        if coverage_cache is None:
            coverage_cache = load_coverage_evidence(root)
        return coverage_cache

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
        witness_base = {"dtypes", "ranks", "pairs", "cases"}
        witness_optional = {"reverse_second_order_cases", "reverse_first_order_graph_cases", "reverse_selected_third_order_cases"}
        if not isinstance(witnesses, dict) or not witness_base <= set(witnesses) or set(witnesses) - witness_base - witness_optional:
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
        if promoted and "reverse_second_order_cases" not in witnesses:
            raise ValueError(
                f"{path}.witnesses.reverse_second_order_cases: required for reverse_second_order_witnessed"
            )
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
            coverage = coverage_evidence()
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
        autograd_contract = entry["autograd"] if isinstance(entry["autograd"], str) else None
        allowed_link_keys = {
            "reverse_second_order_witnessed": {"reverse_second_order_cases"},
            "reverse_first_order_graph_witnessed": {"reverse_first_order_graph_cases"},
            "reverse_finite_second_order_witnessed": {
                "reverse_first_order_graph_cases", "reverse_second_order_cases",
            },
            "reverse_selected_third_order_witnessed": {
                "reverse_first_order_graph_cases", "reverse_second_order_cases",
                "reverse_selected_third_order_cases",
            },
        }.get(autograd_contract, set())
        graph_link_keys = {
            "reverse_first_order_graph_cases", "reverse_second_order_cases",
            "reverse_selected_third_order_cases",
        }
        invalid_link_keys = (set(witnesses) & graph_link_keys) - allowed_link_keys
        if invalid_link_keys:
            raise ValueError(
                f"{path}.witnesses.{sorted(invalid_link_keys)[0]}: graph witness link does not match autograd contract"
            )
        coverage = None
        if set(witnesses) & {
            "reverse_first_order_graph_cases", "reverse_second_order_cases",
            "reverse_selected_third_order_cases",
        }:
            coverage = coverage_evidence()
        for link_key in ("reverse_first_order_graph_cases", "reverse_second_order_cases", "reverse_selected_third_order_cases"):
            if link_key not in witnesses:
                continue
            links = witnesses[link_key]
            supported_cases = {case["name"] for case in entry["test_cases"] if isinstance(case, dict) and case.get("supported") is True}
            if not isinstance(links, list) or not links or links != sorted(set(links)) or not set(links) <= supported_cases:
                raise ValueError(f"{path}.witnesses.{link_key}: expected sorted supported graph witness links")
            import sys
            tests_path = str(root / "tests/python")
            if tests_path not in sys.path:
                sys.path.insert(0, tests_path)
            import vulkan_conformance as vc
            for name in links:
                record = coverage.get(name)
                if not isinstance(record, dict) or record.get("schema") != schema or record.get("parity") is not True:
                    raise ValueError(f"{path}.witnesses.{link_key}: {name!r} lacks matching parity evidence")
                if record.get("graph_autograd") is None:
                    vector_graph = record.get("vector_matmul_evidence", {}).get("graph")
                    vector_directions = vector_graph.get("directions", []) if isinstance(vector_graph, dict) else []
                    if (link_key == "reverse_first_order_graph_cases"
                            and "first_reverse" in vector_directions):
                        continue
                    if (link_key != "reverse_second_order_cases"
                            or record.get("reverse_autograd") != {"order": 2, "graph_preserved": True}):
                        raise ValueError(f"{path}.witnesses.{link_key}: {name!r} lacks matching nested graph evidence")
                    continue
                vc.validate_graph_autograd_record(name, record)
                directions = record["graph_autograd"]["directions"]
                required_direction = {"reverse_first_order_graph_cases": "first_reverse", "reverse_second_order_cases": "second_reverse", "reverse_selected_third_order_cases": "selected_third_d_g_d_x_d_w"}[link_key]
                if not any(direction == required_direction or (required_direction == "second_reverse" and direction.startswith("second_reverse")) for direction in directions):
                    raise ValueError(f"{path}.witnesses.{link_key}: {name!r} has no nested {required_direction} evidence")
        if entry["status"] == "supported":
            if not pairs or not witnesses["cases"]:
                raise ValueError(f"{path}.witnesses: supported entry has no witnesses")
            missing_dtypes = set(dtypes["inputs"]) - {pair[0] for pair in pairs}
            if missing_dtypes:
                raise ValueError(f"{schema}: declares input dtypes {sorted(missing_dtypes)} that no case exercised")
            for rank in range(ranks["min"], ranks["max"] + 1):
                if rank not in witnesses["ranks"]:
                    raise ValueError(f"{schema}: declares ranks {ranks['min']}-{ranks['max']} but rank {rank} was never exercised")
        graph_autograd = entry["autograd"]
        if graph_autograd == "reverse_first_order_graph_witnessed" and not witnesses.get("reverse_first_order_graph_cases"):
            raise ValueError(f"{path}.witnesses.reverse_first_order_graph_cases: required for first-order graph witness")
        if graph_autograd == "reverse_finite_second_order_witnessed":
            for link_key in ("reverse_first_order_graph_cases", "reverse_second_order_cases"):
                if not witnesses.get(link_key):
                    raise ValueError(f"{path}.witnesses.{link_key}: required for finite second-order witness")
        if graph_autograd == "reverse_selected_third_order_witnessed":
            for link_key in ("reverse_first_order_graph_cases", "reverse_second_order_cases", "reverse_selected_third_order_cases"):
                if not witnesses.get(link_key):
                    raise ValueError(f"{path}.witnesses.{link_key}: required selected-third witness link is missing")
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
    has_convolution_source = bool(
        {"aten::convolution.default", "aten::convolution_backward.default"} & source
    )
    coverage = coverage_evidence() if STOCK_COMPOSITE_ROUTES or has_convolution_source else {}
    validate_convolution_evidence(
        coverage,
        require_complete=any(
            entry.get("schema") in {
                "aten::convolution.default", "aten::convolution_backward.default"
            }
            for entry in entries
        ),
    )
    validate_convolution_manifest_bindings(entries, coverage)
    if OVERRIDEABLE_BACKWARD_SCHEMA in source:
        validate_overrideable_backward_evidence(entries, coverage)
    validate_tensor_list_evidence(
        coverage,
        require_complete=any(entry.get("schema") == "aten::cat.default" for entry in entries),
    )
    has_vector_matmul_entry = any(
        entry.get("schema") in {"aten::mul.Tensor", "aten::dot.default",
                                 "aten::mv.default", "aten::matmul.default"}
        for entry in entries
    )
    if has_vector_matmul_entry:
        validate_vector_matmul_evidence(coverage, root, require_complete=True)
        validate_vector_matmul_manifest_bindings(entries, coverage, root)
    validate_stock_composite_routes(
        STOCK_COMPOSITE_ROUTES, entries, coverage, source, explicit_rejected
    )
    if has_convolution_source:
        import sys
        tests_path = str(root / "tests/python")
        if tests_path not in sys.path:
            sys.path.insert(0, tests_path)
        import vulkan_conformance as vc
        vc.validate_graph_autograd_evidence(coverage, entries)
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
