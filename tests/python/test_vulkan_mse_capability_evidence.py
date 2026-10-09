import copy
import json
from pathlib import Path

import pytest

from mse_capability_evidence import validate_mse_evidence, REQUIRED_CASES

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def coverage():
    return json.loads((ROOT / "docs/vulkan_coverage.json").read_text())


def test_complete(coverage):
    validate_mse_evidence(coverage, ROOT, require_complete=True)


@pytest.mark.parametrize("field", ["first", "second", "fd", "fixture", "cpu-forward", "cpu-first", "cpu-second", "vk-first", "vk-second", "history"])
def test_coordinated_forgery(coverage, field):
    name = "mse-ad.ordinary-mean-both.functional"
    bad = {name: copy.deepcopy(coverage[name])}
    e = bad[name]["mse_autograd_evidence"]
    if field in ("first", "second"):
        for side in ("cpu", "vulkan"):
            e[side][field][0]["values"][0] += .25
    elif field == "fd":
        e["fd"][0]["fd"] += .25
    elif field == "fixture":
        e["fixture"]["bases"][0]["values"][0] += .25
    elif field == "history":
        e["vulkan"]["first"][0]["history"] = None
    else:
        side, phase = field.split("-")
        e["phases"][phase]["cpu_ops" if side == "cpu" else "vulkan_ops"] = []
    with pytest.raises(ValueError):
        validate_mse_evidence(bad, ROOT)


@pytest.mark.parametrize("value", [True, "0", float("nan"), float("inf")])
def test_numeric_types(coverage, value):
    name = "mse-ad.ordinary-mean-both.functional"
    bad = {name: copy.deepcopy(coverage[name])}
    for side in ("cpu", "vulkan"):
        bad[name]["mse_autograd_evidence"][side]["loss"]["values"][0] = value
    with pytest.raises(ValueError):
        validate_mse_evidence(bad, ROOT)


def test_missing(coverage):
    bad = {k: v for k, v in coverage.items() if k != next(iter(REQUIRED_CASES))}
    with pytest.raises(ValueError, match="missing"):
        validate_mse_evidence(bad, ROOT, require_complete=True)


def test_only_named_empty_mean_exception(coverage):
    for name, change in (("mse-ad.empty-mean-both.functional", "numeric"),
                         ("mse-ad.ordinary-mean-both.functional", "exception")):
        bad = {name: copy.deepcopy(coverage[name])}
        for side in ("cpu", "vulkan"):
            loss = bad[name]["mse_autograd_evidence"][side]["loss"]
            if change == "numeric":
                loss["values"] = [float("nan")]
            else:
                loss["values"], loss["exception"] = [], "empty-mean-loss-NaN"
        with pytest.raises(ValueError):
            validate_mse_evidence(bad, ROOT)


@pytest.mark.parametrize("mutation", ["seed", "layout", "api", "targets", "duplicate", "drop", "append", "reorder", "container"])
def test_coincident_registry_factory_conflicts(mutation):
    from dataclasses import replace
    import torch
    import vulkan_conformance as vc
    from mse_capability_evidence import validate_mse_conformance_case
    name = "mse-ad.ordinary-mean-both.functional"
    case = next(c for c in vc.ALL_CASES if c.name == name)
    original = case.input_factory
    if mutation in ("seed", "layout"):
        def inputs(**kwargs):
            values = list(original(**kwargs))
            if mutation == "seed":
                values[0] = (values[0].detach() + .01).requires_grad_()
            else:
                values[0] = values[0].detach().t().contiguous().t().requires_grad_()
            return tuple(values)
        case = replace(case, input_factory=inputs)
    elif mutation == "api":
        case = replace(case, operation=lambda a, b, g: torch.nn.MSELoss()(a, b))
    elif mutation == "targets":
        case = replace(case, check_gradients=False)
    elif mutation == "duplicate":
        case = replace(case, setup_inputs=lambda bases, device: tuple(t.clone() for t in bases))
    else:
        setups = {"drop": lambda b: b[:2], "append": lambda b: b + (b[2],),
                  "reorder": lambda b: (b[0], b[2], b[1]), "container": list}
        case = replace(case, setup_inputs=lambda bases, device: setups[mutation](bases))
    with pytest.raises(ValueError, match="conflicting"):
        validate_mse_conformance_case(case, REQUIRED_CASES[name])


@pytest.mark.parametrize("name,forged_count", [
    ("mse-ad.broadcast-mean-both.functional", 0),
    ("mse-ad.broadcast-mean-both.functional", 5),
    ("mse-ad.broadcast-none-both.functional", 0),
    ("mse-ad.broadcast-none-both.functional", 5),
    ("mse-ad.expanded-upstream-none-both.functional", 1),
])
def test_counter_only_materialization_forgery(coverage, name, forged_count):
    bad = {name: copy.deepcopy(coverage[name])}
    phases = copy.deepcopy(bad[name]["mse_autograd_evidence"]["phases"])
    bad[name]["mse_autograd_evidence"]["counters"][1] = forged_count
    assert bad[name]["mse_autograd_evidence"]["phases"] == phases
    with pytest.raises(ValueError, match="materialization minimum"):
        validate_mse_evidence(bad, ROOT)


@pytest.mark.parametrize("name", ["mse-ad.empty-mean-both.functional",
                                    "mse-ad.empty-none-both.functional",
                                    "mse-ad.ordinary-none-both.functional",
                                    "mse-ad.ordinary-mean-both.functional",
                                    "mse-ad.ordinary-sum-both.functional"])
def test_zero_copy_source_contract_remains_valid(coverage, name):
    assert coverage[name]["mse_autograd_evidence"]["counters"][1] == 0
    validate_mse_evidence({name: coverage[name]}, ROOT)


@pytest.mark.parametrize("kind,reduction,totals", [
    ("scalar", "mean", (0, 0, 0)), ("empty", "mean", (0, 0, 0)),
    ("ordinary", "none", (0, 0, 0)), ("ordinary", "mean", (0, 0, 0)),
    ("broadcast", "none", (3, 4, 6)), ("broadcast", "mean", (3, 4, 6)),
    ("expanded-upstream", "none", (1, 1, 2)),
])
def test_source_geometry_materialization_bounds(kind, reduction, totals):
    from mse_capability_evidence import mse_materialization_minima
    for selection, expected in zip(("input", "target", "both"), totals):
        case = REQUIRED_CASES[f"mse-ad.{kind}-{reduction}-{selection}.functional"]
        minima = mse_materialization_minima(case)
        assert sum(minima.values()) == expected
        assert set(minima) == {"forward", "first", "second"}


@pytest.mark.parametrize("bad_setup_call", [1, 2])
@pytest.mark.parametrize("mutation", ["drop", "append", "reorder", "container"])
def test_setup_inventory_checked_before_each_operation_or_reference(bad_setup_call, mutation):
    from dataclasses import replace
    import vulkan_conformance as vc
    from mse_capability_evidence import validate_mse_conformance_case
    name = "mse-ad.ordinary-mean-both.functional"
    case = next(c for c in vc.ALL_CASES if c.name == name)
    operation = case.operation
    setup_calls, operation_calls = [], []
    def setup(bases, device):
        setup_calls.append(device)
        if len(setup_calls) != bad_setup_call:
            return bases
        return {"drop": lambda: bases[:2], "append": lambda: bases + (bases[2],),
                "reorder": lambda: (bases[1], bases[0], bases[2]),
                "container": lambda: list(bases)}[mutation]()
    def observe(*bases):
        operation_calls.append(bases)
        return operation(*bases)
    forged = replace(case, setup_inputs=setup, operation=observe, cpu_reference=observe)
    with pytest.raises(ValueError, match="changed base gradient targets"):
        validate_mse_conformance_case(forged, REQUIRED_CASES[name])
    assert len(operation_calls) == bad_setup_call - 1
