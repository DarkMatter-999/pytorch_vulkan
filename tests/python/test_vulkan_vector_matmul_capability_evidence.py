"""Source binding for the finite G1 vector/matmul capability evidence."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from tools.validate_vulkan_capabilities import validate_vector_matmul_evidence

ROOT = Path(__file__).parents[2]
COVERAGE = ROOT / "docs/vulkan_coverage.json"


def _coverage():
    return json.loads(COVERAGE.read_text())


def test_vector_matmul_source_owned_evidence_is_complete_and_current():
    validate_vector_matmul_evidence(_coverage(), ROOT, require_complete=True)


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("missing_case", "required executed vector/matmul witness is missing"),
        ("rank", "actual input metadata differs from its source-owned fixture"),
        ("history", "graph history is incomplete"),
        ("source", "source identity differs from current fixture/source"),
        ("execution", "runtime execution evidence is malformed"),
        ("fd", "CPU finite-difference projection does not match its analytic signal"),
        ("output_summary", "output_max_abs_error summary does not match stored CPU/Vulkan values"),
        ("first_summary", r"first_reverse.max_abs_errors\[0\] summary does not match stored CPU/Vulkan values"),
        ("output_forgery", "output values violate componentwise CPU/Vulkan parity"),
        ("output_forgery_plausible_summary", "output values violate componentwise CPU/Vulkan parity"),
        ("first_forgery", r"first-reverse .* values violate componentwise CPU/Vulkan parity"),
        ("runtime_identity", "runtime/build identity differs"),
        ("nonfinite_value", "numerical value evidence must contain finite real numbers"),
        ("bool_value", "numerical value evidence must contain finite real numbers"),
        ("string_value", "numerical value evidence must contain finite real numbers"),
        ("value_length", "output value evidence has invalid flattened length"),
        ("value_shape", "output value evidence has invalid shape or value-class metadata"),
        ("capture_command", "API, route, or seed differs from source-owned fixture"),
        ("value_class", "output value evidence has invalid shape or value-class metadata"),
        ("mixed_projection", "stored mixed projections do not match stored tensor values"),
        ("mixed_forgery", "selected mixed-reverse values violate componentwise CPU/Vulkan parity"),
        ("input_fixture", "CPU input values differ from the deterministic source fixture"),
        ("output_shape_metadata", "actual output residency/metadata is inconsistent"),
    ],
)
def test_vector_matmul_evidence_cannot_be_widened_by_forged_or_stale_records(
    mutation, message
):
    from vector_matmul_capability_evidence import REQUIRED_CASES

    coverage = _coverage()
    name = sorted(REQUIRED_CASES)[0]
    if mutation in {"fd", "mixed_projection", "mixed_forgery"}:
        name = next(case for case, contract in REQUIRED_CASES.items()
                    if contract["mixed"] is not None)
    if mutation == "missing_case":
        del coverage[name]
    else:
        coverage[name] = copy.deepcopy(coverage[name])
        evidence = coverage[name]["vector_matmul_evidence"]
        output_shape = evidence["output"]["cpu"]["shape"]
        output_count = math.prod(output_shape)
        fake_output = {
            "value_class": "finite",
            "shape": output_shape,
            "cpu_values": [0.0] * output_count,
            "vulkan_values": [0.0] * output_count,
        }
        if mutation == "rank":
            evidence["inputs"][0]["cpu"]["rank"] += 1
        elif mutation == "history":
            evidence["graph"]["directions"] = []
        elif mutation == "source":
            evidence["source_identity"][next(iter(evidence["source_identity"]))] = "0" * 64
        elif mutation == "fd":
            evidence["graph"]["selected_mixed_reverse"]["cpu_fd_abs_error"] = 1.0
        elif mutation == "output_summary":
            evidence["output_max_abs_error"] = 1e9
        elif mutation == "first_summary":
            evidence["graph"]["first_reverse"]["max_abs_errors"][0] = 1e9
        elif mutation == "output_forgery":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"][0] += 0.1
        elif mutation == "output_forgery_plausible_summary":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"][0] += 0.1
            evidence["output_max_abs_error"] = abs(
                values["vulkan_values"][0] - values["cpu_values"][0]
            )
        elif mutation == "first_forgery":
            values = evidence["graph"]["first_reverse"].setdefault("values", [{
                "shape": [1], "cpu_values": [0.0], "vulkan_values": [0.0]}])
            values[0]["vulkan_values"][0] += 0.1
        elif mutation == "runtime_identity":
            evidence.setdefault("runtime_identity", {})["extension_sha256"] = "0" * 64
        elif mutation == "nonfinite_value":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"][0] = float("nan")
        elif mutation == "bool_value":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"][0] = True
        elif mutation == "string_value":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"][0] = "0.0"
        elif mutation == "value_length":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["vulkan_values"].append(0.0)
        elif mutation == "value_shape":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["shape"] = [2]
        elif mutation == "capture_command":
            evidence["capture_command"] = "invented capture command"
        elif mutation == "value_class":
            values = evidence.setdefault("output_values", copy.deepcopy(fake_output))
            values["value_class"] = "nonfinite"
        elif mutation == "mixed_projection":
            evidence["graph"]["selected_mixed_reverse"]["vulkan_projection"] = 1e9
        elif mutation == "mixed_forgery":
            mixed_values = evidence["graph"]["selected_mixed_reverse"]["values"]
            mixed_values["vulkan_values"][0] += 0.1
        elif mutation == "input_fixture":
            evidence["inputs"][0]["values"]["cpu_values"][0] += 0.1
            evidence["inputs"][0]["values"]["vulkan_values"][0] += 0.1
        elif mutation == "output_shape_metadata":
            evidence["output"]["cpu"]["shape"] = [1]
            evidence["output"]["vulkan"]["shape"] = [1]
        else:
            evidence["runtime"]["compute_dispatches"] = True

    with pytest.raises(ValueError, match=message):
        validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)


@pytest.mark.parametrize("phase", ["output", "first", "mixed", "fd"])
def test_coordinated_cpu_vulkan_forgery_is_rejected_by_cpu_execution(phase):
    coverage = _coverage()
    name = "g1.dot.rank1.grad-a-wrt-b"
    evidence = coverage[name]["vector_matmul_evidence"]
    if phase == "output":
        fact = evidence["output_values"]
    elif phase == "first":
        fact = evidence["graph"]["first_reverse"]["values"][0]
    else:
        mixed = evidence["graph"]["selected_mixed_reverse"]
        fact = mixed["values"]
    if phase != "fd":
        for side in ("cpu_values", "vulkan_values"):
            fact[side][0] += 0.1
    if phase in {"mixed", "fd"}:
        projection = sum(v * d for v, d in zip(fact["cpu_values"], mixed["direction_values"]))
        if phase == "mixed":
            mixed["cpu_analytic_projection"] = projection
            mixed["vulkan_projection"] = projection
            mixed["cpu_centered_fd_projection"] = projection
        else:
            # Still passes the analytic/FD tolerance, but is not the measured FD.
            mixed["cpu_centered_fd_projection"] += 0.0001
        mixed["cpu_fd_abs_error"] = abs(
            mixed["cpu_analytic_projection"] - mixed["cpu_centered_fd_projection"])
    with pytest.raises(ValueError, match="recomputed CPU"):
        validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)


@pytest.mark.parametrize("field,value", [
    ("checkout_head", "a" * 40),
    ("extension_path", "/relocated/build/pytorch_vulkan/_C.so"),
    ("extension_sha256", "b" * 64),
    ("execution_mode", "sync"),
])
def test_consistent_historical_provenance_survives_commit_rebuild_and_mode(field, value):
    coverage = _coverage()
    for name, record in coverage.items():
        if name.startswith("g1."):
            record["vector_matmul_evidence"]["runtime_identity"][field] = value
    validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)


def test_historical_validator_never_probes_hardware_or_current_build(monkeypatch):
    import vector_matmul_capability_evidence as factory

    def forbidden(*args, **kwargs):
        raise AssertionError("historical validation must remain CPU-only")

    monkeypatch.setattr(factory, "runtime_identity", forbidden)
    validate_vector_matmul_evidence(_coverage(), ROOT, require_complete=True)


def test_identical_source_checkout_relocation_needs_no_git_or_extension(tmp_path):
    import shutil
    from vector_matmul_capability_evidence import SOURCE_IDENTITY_PATHS

    for relative in SOURCE_IDENTITY_PATHS:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    validate_vector_matmul_evidence(_coverage(), tmp_path, require_complete=True)
    helper = tmp_path / "tests/python/vector_matmul_capability_evidence.py"
    helper.write_text(helper.read_text() + "\n# stale fixture source\n")
    with pytest.raises(ValueError, match="source identity differs from current fixture/source"):
        validate_vector_matmul_evidence(_coverage(), tmp_path, require_complete=True)


@pytest.mark.parametrize("field,value", [
    ("checkout_head", True), ("checkout_head", "wrong"),
    ("extension_sha256", "nan"), ("extension_path", "relative/_C.so"),
    ("execution_mode", "invented"), ("hardware", ""),
    ("vulkan_api_version", "NaN"), ("pytorch_git_revision", "wrong"),
])
def test_historical_identity_formats_are_checked_even_when_consistent(field, value):
    coverage = _coverage()
    for name, record in coverage.items():
        if name.startswith("g1."):
            record["vector_matmul_evidence"]["runtime_identity"][field] = value
    with pytest.raises(ValueError, match="runtime/build identity metadata is malformed"):
        validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)


@pytest.mark.parametrize("mutation", [
    "missing", "seed_value", "seed_bool", "seed_string", "probe_value",
    "probe_nan", "shape_bool", "epsilon", "epsilon_bool", "direction",
])
def test_cpu_seed_probe_direction_and_fd_recipe_are_source_bound(mutation):
    coverage = _coverage()
    evidence = coverage["g1.mul.broadcast.rank0-rank1"]["vector_matmul_evidence"]
    recipe = evidence["cpu_reference_recipe"]
    if mutation == "missing":
        del evidence["cpu_reference_recipe"]
    elif mutation == "seed_value":
        recipe["upstream_seed"]["values"][0] += 0.1
    elif mutation == "seed_bool":
        recipe["upstream_seed"]["values"][0] = True
    elif mutation == "seed_string":
        recipe["upstream_seed"]["values"][0] = "0.0"
    elif mutation == "probe_value":
        recipe["mixed_probe"]["values"][0] += 0.1
    elif mutation == "probe_nan":
        recipe["mixed_probe"]["values"][0] = float("nan")
    elif mutation == "shape_bool":
        recipe["mixed_probe"]["shape"] = [True]
    elif mutation == "epsilon":
        recipe["fd_epsilon"] = 0.01
    elif mutation == "epsilon_bool":
        recipe["fd_epsilon"] = True
    else:
        evidence["graph"]["selected_mixed_reverse"]["direction_values"][0] += 0.1
    with pytest.raises(ValueError, match="CPU reference recipe|selected mixed direction"):
        validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)


def test_async_history_does_not_qualify_live_sync_mode(monkeypatch):
    import vector_matmul_capability_evidence as factory
    from tools.validate_vulkan_capabilities import qualify_vector_matmul_current_runtime

    coverage = _coverage()
    captured = copy.deepcopy(coverage["g1.dot.rank1.forward"]["vector_matmul_evidence"]["runtime_identity"])
    current = dict(captured, execution_mode="sync" if captured["execution_mode"] == "async" else "async")
    # Only the hardware measurement is stubbed; CPU semantics/source/parity are real.
    monkeypatch.setattr(factory, "runtime_identity", lambda root: current)
    validate_vector_matmul_evidence(coverage, ROOT, require_complete=True)
    with pytest.raises(ValueError, match="does not qualify the current build/device/mode"):
        qualify_vector_matmul_current_runtime(coverage, ROOT)
    current.update(captured, checkout_head="c" * 40, extension_path="/relocated/_C.so")
    qualify_vector_matmul_current_runtime(coverage, ROOT)
    assert coverage["g1.dot.rank1.forward"]["vector_matmul_evidence"]["runtime_identity"] == captured
