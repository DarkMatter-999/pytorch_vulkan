"""CPU-only historical validation: mutable paired values are not an oracle."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from general_matmul_capability_evidence import (
    REQUIRED_CASES, validate_general_matmul_evidence,
    qualify_general_matmul_current_runtime,
)

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def records():
    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    return {name: coverage[name] for name in REQUIRED_CASES}


def test_complete_offline(records, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("offline probe"))
    validate_general_matmul_evidence(records, ROOT, require_complete=True)


@pytest.mark.parametrize("field", ["output", "first", "mixed", "projection", "fd"])
def test_coordinated_numerical_forgery(records, field):
    bad = copy.deepcopy(records)
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    evidence = bad[name]["general_matmul_evidence"]
    if field in ("projection", "fd"):
        evidence["mixed"][0][field] += .1
    else:
        value = evidence["output"] if field == "output" else evidence[field][0]
        value["cpu"][0] += .1
        value["vulkan"][0] += .1
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


@pytest.mark.parametrize("field,value", [
    ("rank", True), ("selection", "a"), ("history", False),
    ("route", "dot"), ("source", "0" * 64), ("counter", True),
    ("value", True), ("value", "0"), ("value", float("nan")),
    ("shape", [1]), ("length", []), ("runtime", "bad"),
])
def test_malformed_or_false_metadata(records, field, value):
    bad = copy.deepcopy(records)
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    e = bad[name]["general_matmul_evidence"]
    if field == "rank":
        e["inputs"][0]["cpu_fact"]["rank"] = value
    elif field == "selection":
        e["fixture"]["selection"] = value
    elif field == "history":
        e["first"][0]["vulkan_fact"]["history"] = value
    elif field == "route":
        e["forward"]["vulkan_ops"] = ["aten::" + value]
    elif field == "source":
        e["source_identity"][next(iter(e["source_identity"]))] = value
    elif field == "counter":
        e["forward"]["counters"][0] = value
    elif field == "value":
        e["output"]["cpu"][0] = e["output"]["vulkan"][0] = value
    elif field == "shape":
        e["output"]["cpu_fact"]["shape"] = value
    elif field == "length":
        e["output"]["cpu"] = e["output"]["vulkan"] = value
    else:
        e["runtime_identity"]["extension_sha256"] = value
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


def test_missing_required_case(records):
    bad = copy.deepcopy(records)
    bad.pop(next(iter(bad)))
    with pytest.raises(ValueError, match="missing"):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


@pytest.mark.parametrize("field", ["fixture", "source_identity", "runtime_identity", "forward", "first_execution", "first", "mixed", "seed", "targets"])
def test_missing_required_payload(records, field):
    bad = copy.deepcopy(records)
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    del bad[name]["general_matmul_evidence"][field]
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


@pytest.mark.parametrize("field,value", [("projection", True), ("fd", "0"), ("epsilon", float("nan"))])
def test_malformed_mixed_scalar(records, field, value):
    bad = copy.deepcopy(records)
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    bad[name]["general_matmul_evidence"]["mixed"][0][field] = value
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


@pytest.mark.parametrize("counter", [2, 3])
def test_device_transfer_or_fallback_is_not_execution_proof(records, counter):
    bad = copy.deepcopy(records)
    name = next(iter(bad))
    bad[name]["general_matmul_evidence"]["forward"]["counters"][counter] = 1
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


@pytest.mark.parametrize("mutation", ["extra_leaf", "missing_copy"])
def test_source_route_excludes_false_leaf_and_missing_materialization(records, mutation):
    bad = copy.deepcopy(records)
    name = next(n for n, c in REQUIRED_CASES.items() if c["copy_route"])
    phase = bad[name]["general_matmul_evidence"]["forward"]
    if mutation == "extra_leaf":
        phase["vulkan_ops"] = sorted(set(phase["vulkan_ops"]) | {"aten::dot"})
    else:
        phase["counters"][1] = 0
    with pytest.raises(ValueError):
        validate_general_matmul_evidence(bad, ROOT, require_complete=True)


def test_source_relocation_is_offline(records, tmp_path, monkeypatch):
    from general_matmul_capability_evidence import SOURCE_IDENTITY_PATHS
    for path in SOURCE_IDENTITY_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("offline probe"))
    validate_general_matmul_evidence(records, tmp_path, require_complete=True)


def test_historical_provenance_is_not_current_runtime(records, monkeypatch):
    bad = copy.deepcopy(records)
    for record in bad.values():
        runtime = record["general_matmul_evidence"]["runtime_identity"]
        runtime.update(checkout_head="a" * 40, extension_path="/relocated/extension.so",
                       execution_mode="sync")
    validate_general_matmul_evidence(bad, ROOT, require_complete=True)
    import general_matmul_capability_evidence as helper
    current = dict(next(iter(bad.values()))["general_matmul_evidence"]["runtime_identity"])
    current["execution_mode"] = "async"
    monkeypatch.setattr(helper, "runtime_identity", lambda root: current)
    with pytest.raises(ValueError, match="current build/device/mode"):
        qualify_general_matmul_current_runtime(bad, ROOT)


def test_manifest_requires_all_unique_case_and_graph_bindings(records):
    from tools import generate_vulkan_capabilities as generator
    from tools.validate_vulkan_capabilities import validate_general_matmul_manifest_bindings
    manifest = generator.build_manifest()
    entry = next(e for e in manifest["entries"] if e["schema"] == "aten::matmul.default")
    validate_general_matmul_manifest_bindings(manifest["entries"], records)
    entry["witnesses"]["reverse_second_order_cases"].pop()
    with pytest.raises(ValueError, match="graph-direction"):
        validate_general_matmul_manifest_bindings(manifest["entries"], records)


def test_generator_rejects_conflicting_coincident_fixture(monkeypatch):
    from dataclasses import replace
    import vulkan_conformance as vc
    from tools import generate_vulkan_capabilities as generator
    name = next(iter(REQUIRED_CASES))
    monkeypatch.setattr(vc, "ALL_CASES", tuple(
        replace(c, expected_shape=(999,)) if c.name == name else c for c in vc.ALL_CASES))
    with pytest.raises(ValueError, match="conflicting"):
        generator._case_index()


@pytest.mark.parametrize("field", ["first", "mixed", "cpu"])
@pytest.mark.parametrize("ops", [["aten::dot"], ["aten::nonexistent_g2_route"]])
def test_derivative_and_combined_cpu_routes_are_reference_bound(records, field, ops):
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    bad = {name: copy.deepcopy(records[name])}
    evidence = bad[name]["general_matmul_evidence"]
    if field == "first":
        evidence["first_execution"]["vulkan_ops"] = ops
    elif field == "mixed":
        evidence["mixed"][0]["execution"]["vulkan_ops"] = ops
    else:
        evidence["cpu_execution_ops"] = ops
    with pytest.raises(ValueError, match="route"):
        validate_general_matmul_evidence(bad, ROOT)


@pytest.mark.parametrize("mutation", ["values", "seed", "selection", "operation", "api", "no_grad", "layout", "setup"])
def test_generator_rejects_same_shape_semantic_fixture_conflict(monkeypatch, mutation):
    from dataclasses import replace
    import torch
    import vulkan_conformance as vc
    from tools import generate_vulkan_capabilities as generator
    from general_matmul_cases import make_cpu_bases, make_operands
    name = next(n for n, c in REQUIRED_CASES.items() if c["mixed"])
    fixture = REQUIRED_CASES[name]
    original = next(c for c in vc.ALL_CASES if c.name == name)
    if mutation == "operation":
        changed = replace(original, operation=lambda *inputs: original.operation(*inputs) + 1)
    elif mutation == "api":
        changed = replace(original, operation=lambda *inputs:
                          make_operands(fixture, inputs)[0] @ make_operands(fixture, inputs)[1])
    elif mutation == "no_grad":
        def operation(*inputs):
            with torch.no_grad():
                return torch.matmul(*make_operands(fixture, inputs))
        changed = replace(original, operation=operation)
    elif mutation == "setup":
        changed = replace(original, setup_inputs=lambda inputs, device: tuple(x.clone() for x in inputs))
    else:
        def factory(*, requires_grad=False):
            recipe = dict(fixture)
            if mutation == "seed":
                recipe["seed"] += 1
            if mutation == "selection":
                recipe["selection"] = "none"
            bases = make_cpu_bases(recipe)
            if mutation == "values":
                bases = tuple((x.detach() + 1).requires_grad_(x.requires_grad) for x in bases)
            if mutation == "layout":
                value = bases[0]
                stride = tuple(s * 2 for s in value.stride())
                altered = torch.empty_strided(value.shape, stride).copy_(value.detach()).requires_grad_(value.requires_grad)
                bases = (altered, bases[1])
            return bases
        changed = replace(original, input_factory=factory)
    monkeypatch.setattr(vc, "ALL_CASES", tuple(changed if c.name == name else c for c in vc.ALL_CASES))
    with pytest.raises(ValueError, match="conflicting"):
        generator._case_index()


@pytest.mark.parametrize("field,mutation", [("first", "view"), ("first", "reduction"),
                                           ("mixed", "view"), ("mixed", "reduction"),
                                           ("cpu", "extra")])
def test_missing_generated_inverse_or_extra_combined_cpu_route(records, field, mutation):
    name = next(n for n, c in REQUIRED_CASES.items()
                if c["family"] == "cross-broadcast" and c["mixed"])
    bad = {name: copy.deepcopy(records[name])}
    e = bad[name]["general_matmul_evidence"]
    if field == "cpu":
        e["cpu_execution_ops"] = sorted(set(e["cpu_execution_ops"]) | {"aten::sin"})
    else:
        phase = e["first_execution"] if field == "first" else e["mixed"][0]["execution"]
        removed = "aten::reshape" if mutation == "view" else "aten::sum"
        assert removed in phase["vulkan_ops"]
        phase["vulkan_ops"].remove(removed)
    with pytest.raises(ValueError, match="route"):
        validate_general_matmul_evidence(bad, ROOT)


@pytest.mark.parametrize("family", ["cross-broadcast", "cross-transposed", "unequal-rank"])
@pytest.mark.parametrize("mutation", ["erase-vulkan-materialization", "replace-cpu-forward",
                                      "erase-copy-event", "one-missing-copy", "erase-expand"])
def test_forward_route_and_broadcast_materialization_are_source_bound(records, family, mutation):
    name = f"g2.{family}.both.grad-mode.matmul"
    bad = {name: copy.deepcopy(records[name])}
    phase = bad[name]["general_matmul_evidence"]["forward"]
    assert "aten::clone" in phase["cpu_ops"] and phase["counters"][1] > 0
    if mutation == "erase-vulkan-materialization":
        phase["vulkan_ops"] = [op for op in phase["vulkan_ops"]
                               if op not in {"aten::clone", "aten::copy_"}]
        phase["counters"][1] = 0
    elif mutation == "replace-cpu-forward":
        phase["cpu_ops"] = ["aten::bmm"]
    elif mutation == "erase-copy-event":
        phase["vulkan_ops"].remove("aten::copy_")
    elif mutation == "one-missing-copy":
        assert phase["counters"][1] >= 2
        phase["counters"][1] = 1
    else:
        phase["vulkan_ops"].remove("aten::expand")
    with pytest.raises(ValueError, match="route|materialization"):
        validate_general_matmul_evidence(bad, ROOT)


@pytest.mark.parametrize("family", ["cross-broadcast", "cross-transposed", "unequal-rank"])
def test_source_predicts_required_broadcast_materialization(family):
    case = REQUIRED_CASES[f"g2.{family}.both.grad-mode.matmul"]
    assert case["copy_route"] is True
    assert case["materialization_minimum"] == 2


def test_forward_source_geometry_and_cpu_routes_agree_for_all_finite_cases(records):
    from general_matmul_capability_evidence import _cpu_forward_route
    for name, case in REQUIRED_CASES.items():
        ops = set(_cpu_forward_route(case))
        assert case["copy_route"] == ("aten::clone" in ops) == ("aten::copy_" in ops)
        assert case["materialization_minimum"] >= 0
        assert records[name]["general_matmul_evidence"]["forward"]["counters"][1] >= case["materialization_minimum"]
        if case["family"].startswith("empty-") or case["family"] in ("zero-batch", "cross-zero", "equal-batch"):
            assert case["materialization_minimum"] == 0
