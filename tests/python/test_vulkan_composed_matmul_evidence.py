"""CPU replay rejects coordinated numerical and execution-record forgeries."""
import copy
import json
import subprocess
from pathlib import Path

import pytest

from composed_matmul_evidence import validate_composed_evidence, SOURCE_IDENTITY_PATHS

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def document():
    return json.loads((ROOT / "docs/vulkan_composed_matmul_coverage.json").read_text())


def document_with(document, record):
    result = copy.deepcopy(document)
    result["records"][record["scenario"]["id"] + "." + record["runtime_identity"]["execution_mode"]] = record
    return result


def test_complete_offline(document, monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("offline probe"))
    validate_composed_evidence(document, ROOT)


@pytest.mark.parametrize("field", ["first", "second", "fd", "fixture", "history", "forward-cpu", "first-cpu", "second-cpu", "first-vk", "second-vk", "cross-copy", "counter", "reset", "state"])
def test_forgery(document, field):
    record = copy.deepcopy(document["records"]["g3.mse-parameter-hvp.async"])
    if field in ("first", "second"):
        for side in ("cpu", "vulkan"):
            record[side][field]["shared"]["values"][0] += .25
    elif field == "fd":
        record["fd"][0]["fd"] += .25
    elif field == "fixture":
        record["fixture"]["bases"]["shared"]["values"][0] += .25
    elif field == "history":
        record["vulkan"]["first"]["shared"]["history"] = None
    elif field.endswith("-cpu") or field.endswith("-vk"):
        phase, side = field.split("-")
        record["phases"][phase]["cpu_ops" if side == "cpu" else "vulkan_ops"] = []
    elif field == "cross-copy":
        record["phases"]["forward"]["vulkan_ops"] = [x for x in record["phases"]["forward"]["vulkan_ops"] if x not in ("aten::clone", "aten::copy_")]
        record["counters"][1] = 0
    elif field == "counter":
        record["counters"][0] = True
    else:
        record = copy.deepcopy(document["records"]["g3.sgd-zero.async"])
        for side in ("cpu", "vulkan"):
            tensor = record[side]["steps"][1]["reset"]["shared"] if field == "reset" else record[side]["steps"][1]["state"]["shared"]["momentum_buffer"]
            tensor["values"][0] += .25
    with pytest.raises(ValueError):
        validate_composed_evidence(document_with(document, record), ROOT)


@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf")])
def test_invalid_numbers(document, value):
    record = copy.deepcopy(document["records"]["g3.output-first.async"])
    for side in ("cpu", "vulkan"):
        record[side]["first"]["shared"]["values"][0] = value
    with pytest.raises(ValueError):
        validate_composed_evidence(document_with(document, record), ROOT)


def test_relocation(document, tmp_path, monkeypatch):
    for path in SOURCE_IDENTITY_PATHS:
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / path).read_bytes())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("offline probe"))
    validate_composed_evidence(document, tmp_path)


def test_missing_and_conflicting_ids(document):
    for mutate in (lambda d: d["records"].pop("g3.output-first.sync"),
                   lambda d: d["records"]["g3.output-first.async"]["scenario"].update(kind="hvp")):
        bad = copy.deepcopy(document)
        mutate(bad)
        with pytest.raises(ValueError):
            validate_composed_evidence(bad, ROOT)


@pytest.mark.parametrize("field", ["stride", "offset", "history_nodes", "shape", "dtype", "device", "requires_grad", "leaves", "second-copy", "options"])
def test_strict_metadata(document, field):
    record = copy.deepcopy(document["records"]["g3.mse-parameter-hvp.async"])
    tensor = record["vulkan"]["first"]["shared"]
    if field == "leaves":
        record["phases"]["second"]["vulkan_leaves"][0]["inputs"][0][0] = True
    elif field == "second-copy":
        # Scalar first upstream now legitimately has no materialization. The
        # nontrivial second-gradient layout copies remain a required closure.
        record["phases"]["second"]["vulkan_ops"] = [op for op in record["phases"]["second"]["vulkan_ops"] if op not in ("aten::clone", "aten::copy_")]
    elif field == "options":
        record = copy.deepcopy(document["records"]["g3.sgd-zero.async"])
        record["vulkan"]["optimizer_options"]["lr"] += .0001
    else:
        tensor[field] = {"stride": [1, 1], "offset": 1, "history_nodes": [], "shape": [True, 3],
                         "dtype": "torch.float64", "device": "cpu", "requires_grad": False}[field]
    with pytest.raises(ValueError):
        validate_composed_evidence(document_with(document, record), ROOT)


@pytest.mark.parametrize("name,count", [
    ("g3.output-first.async", 1), ("g3.output-first.async", 3),
    ("g3.mse-parameter-hvp.async", 3),
    ("g3.sgd-none.async", 11), ("g3.sgd-zero.sync", 11),
    ("g3.mutation-shared.async", 3), ("g3.mutation-input.sync", 3),
])
def test_counter_only_model_materialization_forgery(document, name, count):
    record = copy.deepcopy(document["records"][name])
    phases = copy.deepcopy(record["phases"])
    record["counters"][1] = count
    assert record["phases"] == phases
    with pytest.raises(ValueError, match="materialization minimum"):
        validate_composed_evidence(document_with(document, record), ROOT)


def test_source_geometry_model_minima_include_each_executed_forward():
    from composed_matmul_evidence import REQUIRED_SCENARIOS, model_materialization_minima
    for scenario in REQUIRED_SCENARIOS.values():
        minima = model_materialization_minima(scenario)
        forwards = {k: v for k, v in minima.items() if k.endswith("forward")}
        assert list(forwards.values()) == ([4, 4, 4] if scenario["kind"] == "training" else [4])
        expected = 12 if scenario["kind"] == "training" else 4
        assert sum(minima.values()) == expected
        if scenario["kind"] == "mutation":
            assert minima["backward"] == 0 and "second" not in minima
