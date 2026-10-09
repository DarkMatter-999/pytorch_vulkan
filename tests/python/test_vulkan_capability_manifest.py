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
    validate_stock_linear_route_evidence,
)
from tools.vulkan_capability_declarations import STOCK_COMPOSITE_ROUTES
from tools.vulkan_capability_declarations import DECLARATIONS

ROOT = Path(__file__).parents[2]


def test_convolution_autograd_manifest_claims_only_source_required_cases():
    import vulkan_conformance as vc

    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    entries = {entry["schema"]: entry for entry in manifest["entries"]}
    graph_cases = vc.GRAPH_AUTOGRAD_CASES

    convolution_graph_cases = {name for name in graph_cases if name.startswith("convolution.graph.")}
    expected_first = convolution_graph_cases
    expected_second = {
        name for name, contract in graph_cases.items() if name.startswith("convolution.graph.")
        if any(direction.startswith("second_reverse_")
               for direction in contract["directions"])
    }
    expected_selected_third = {
        name for name, contract in graph_cases.items() if name.startswith("convolution.graph.")
        if "selected_third_d_g_d_x_d_w" in contract["directions"]
    }
    assert len(expected_first) == 5
    assert len(expected_second) == 4
    assert len(expected_selected_third) == 2

    standard = entries["aten::convolution.default"]["witnesses"]
    overrideable = entries["aten::convolution_overrideable.default"]["witnesses"]
    assert set(standard["reverse_first_order_graph_cases"]) == expected_first - {
        "convolution.graph.overrideable.first"
    }
    assert set(overrideable["reverse_first_order_graph_cases"]) == {
        "convolution.graph.overrideable.first"
    }
    assert set(standard["reverse_second_order_cases"]) == expected_second
    assert set(standard["reverse_selected_third_order_cases"]) == expected_selected_third

    convolution_schemas = (
        "aten::convolution.default",
        "aten::convolution_backward.default",
        "aten::convolution_overrideable.default",
        "aten::convolution_backward_overrideable.default",
    )
    unsupported_claims = {
        "reverse_arbitrary_order_witnessed", "forward_ad_witnessed",
        "jvp_witnessed", "transforms_witnessed", "cuda_witnessed",
        "hvp_witnessed", "training_witnessed",
    }
    for schema in convolution_schemas:
        declaration = DECLARATIONS[schema]
        entry = entries[schema]
        assert declaration["autograd"] not in unsupported_claims
        assert entry["autograd"] == declaration["autograd"]
        assert not (set(entry["witnesses"]) & unsupported_claims)


def test_matrix_autograd_manifest_claims_only_finite_generated_routes():
    import vulkan_conformance as vc

    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    entries = {entry["schema"]: entry for entry in manifest["entries"]}
    expected = {
        "aten::mm.default": "matrix.graph.mm.generated",
        "aten::addmm.default": "matrix.graph.addmm.generated",
        "aten::bmm.default": "matrix.graph.bmm.generated",
    }
    for schema, case_name in expected.items():
        declaration = DECLARATIONS[schema]
        entry = entries[schema]
        assert declaration["autograd"] == "reverse_finite_second_order_witnessed"
        assert entry["autograd"] == declaration["autograd"]
        assert case_name in entry["witnesses"]["reverse_first_order_graph_cases"]
        assert case_name in entry["witnesses"]["reverse_second_order_cases"]
        contract = vc.GRAPH_AUTOGRAD_CASES[case_name]
        assert contract["directions"] == ["first_reverse", "second_reverse_mixed"]
        assert contract["graph_levels"] == [1, 2]
        assert contract["schema"] == schema


@pytest.mark.parametrize(
    "case_name",
    ["matrix.graph.mm.generated", "matrix.graph.addmm.generated",
     "matrix.graph.bmm.generated"],
)
@pytest.mark.parametrize(
    "mutation,error",
    [
        ("first_detached_both", "first_reverse.*history"),
        ("first_detached_cpu", "first_reverse.*history"),
        ("first_detached_vulkan", "first_reverse.*history"),
        ("duplicate_first", "duplicate derivative result"),
        ("wrong_slot", "derivative result identities"),
        ("wrong_direction_type", "malformed derivative result identity"),
        ("malformed_result", "malformed derivative result identity"),
        ("non_boolean_history", "malformed derivative result field"),
        ("second_history_added", "second_reverse_mixed.*graphless"),
        ("frozen_seed", "grad_output.*requires_grad"),
        ("missing_counter_field", "compute-only"),
        ("boolean_counter", "compute-only"),
        ("negative_copy_counter", "compute-only"),
        ("extra_counter_field", "compute-only"),
    ],
)
def test_matrix_graph_autograd_rejects_reference_contract_mutations(
    case_name, mutation, error
):
    import vulkan_conformance as vc

    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    record = copy.deepcopy(coverage[case_name])
    results = record["graph_autograd"]["derivative_results"]
    if mutation.startswith("first_detached"):
        row = next(item for item in results if item["direction"] == "first_reverse")
        sides = {"first_detached_both": ("cpu", "vulkan"),
                 "first_detached_cpu": ("cpu",),
                 "first_detached_vulkan": ("vulkan",)}[mutation]
        for side in sides:
            row[f"{side}_requires_grad"] = False
            row[f"{side}_grad_fn"] = None
    elif mutation == "duplicate_first":
        row = next(item for item in results if item["direction"] == "first_reverse")
        results.append(copy.deepcopy(row))
    elif mutation == "wrong_slot":
        row = next(item for item in results if item["direction"] == "first_reverse")
        row["slot"] = 99
    elif mutation == "wrong_direction_type":
        results[0]["direction"] = []
    elif mutation == "malformed_result":
        results[0] = None
    elif mutation == "non_boolean_history":
        results[0]["cpu_requires_grad"] = 1
    elif mutation == "second_history_added":
        row = next(item for item in results if item["direction"] == "second_reverse_mixed")
        row["cpu_requires_grad"] = row["vulkan_requires_grad"] = True
        row["cpu_grad_fn"] = row["vulkan_grad_fn"] = "InventedBackward0"
    elif mutation == "frozen_seed":
        for side in ("cpu", "vulkan"):
            record["graph_autograd"]["operands"]["grad_output"][side]["requires_grad"] = False
    elif mutation in {"missing_counter_field", "boolean_counter", "negative_copy_counter", "extra_counter_field"}:
        counters = record["graph_autograd"]["execution"]["first_backward"]
        if mutation == "missing_counter_field":
            del counters["buffer_creations_delta"]
        elif mutation == "boolean_counter":
            counters["compute_dispatches"] = True
        elif mutation == "negative_copy_counter":
            counters["vulkan_copies"] = -1
        else:
            counters["unexpected"] = 0

    with pytest.raises(ValueError, match=error):
        vc.validate_graph_autograd_record(case_name, record)


@pytest.mark.parametrize("phase", ["first_backward", "higher_order"])
@pytest.mark.parametrize("mutation", [
    "missing_field", "boolean_dispatch", "boolean_transfer", "negative_copy", "extra_field",
])
def test_addmm_graph_validator_rejects_malformed_counter_fields_in_both_phases(
    phase, mutation
):
    import vulkan_conformance as vc

    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    record = copy.deepcopy(coverage["matrix.graph.addmm.generated"])
    counters = record["graph_autograd"]["execution"][phase]
    if mutation == "missing_field":
        del counters["buffer_creations_delta"]
    elif mutation == "boolean_dispatch":
        counters["compute_dispatches"] = True
    elif mutation == "boolean_transfer":
        counters["explicit_transfers"] = False
    elif mutation == "negative_copy":
        counters["vulkan_copies"] = -1
    else:
        counters["unexpected"] = 0
    with pytest.raises(ValueError, match="compute-only Vulkan evidence"):
        vc.validate_graph_autograd_record("matrix.graph.addmm.generated", record)


def test_overrideable_backward_source_owned_case_cannot_be_jointly_deleted(monkeypatch):
    from tools import generate_vulkan_capabilities as generator

    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    name = "convolution.backward-overrideable.bias-present.mask-111"
    del coverage[name]
    manifest = copy.deepcopy(load_manifest(ROOT / "docs/vulkan_capabilities.json"))
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="overrideable backward requires"):
        validate_manifest_data(manifest, ROOT)

    entry = next(item for item in manifest["entries"]
                 if item["schema"] == "aten::convolution_backward_overrideable.default")
    entry["test_cases"] = []
    entry["witnesses"] = {"dtypes": [], "ranks": [], "pairs": [], "cases": []}

    with pytest.raises(ValueError, match="overrideable backward requires"):
        generator.build_coverage_manifest(coverage)
    with pytest.raises(ValueError, match="overrideable backward requires"):
        capability_validator.validate_overrideable_backward_evidence(manifest["entries"], coverage)


def _valid_grouped_combined_coverage_fixture():
    """Validator-structure fixture only; this is never runtime evidence."""
    name = "convolution.graph.grouped.ggI-ggW-ggb"
    geometry = {"stride": [2, 1], "padding": [1, 0], "dilation": [1, 2],
                "transposed": False, "output_padding": [0, 0], "groups": 2}
    specs = {
        "input": ([1, 4, 6, 7], [168, 42, 7, 1]),
        "weight": ([6, 2, 3, 2], [12, 6, 2, 1]),
        "bias": ([6], [1]),
        "grad_output": ([1, 6, 3, 5], [90, 15, 5, 1]),
    }
    operands = {}
    for role, (shape, strides) in specs.items():
        operands[role] = {}
        for side, device in (("cpu", "cpu"), ("vulkan", "vk:0")):
            operands[role][side] = {"defined": True, "dtype": "float32", "rank": len(shape),
                                    "shape": shape, "strides": strides, "storage_offset": 0,
                                    "device": device, "requires_grad": True}
    def result(direction, target, slot, shape, defined=True, history=None):
        history = direction == "first_reverse" if history is None else history
        return {"direction": direction, "target": target, "slot": slot,
                "cpu_defined": defined, "vulkan_defined": defined,
                "cpu_shape": shape if defined else None, "vulkan_shape": shape if defined else None,
                "cpu_requires_grad": history if defined else False,
                "vulkan_requires_grad": history if defined else False,
                "cpu_grad_fn": "ConvolutionBackwardBackward0" if history and defined else None,
                "vulkan_grad_fn": "ConvolutionBackwardBackward0" if history and defined else None,
                "cpu_nonzero": defined, "vulkan_nonzero": defined, "oracle": "cpu"}
    results = [result("first_reverse", role, slot, specs[role][0])
               for slot, role in enumerate(("input", "weight", "bias"))]
    results += [result("second_reverse_ggI", "weight", 0, specs["weight"][0]),
                result("second_reverse_ggI", "grad_output", 1, specs["grad_output"][0]),
                result("second_reverse_ggW", "input", 0, specs["input"][0]),
                result("second_reverse_ggW", "grad_output", 1, specs["grad_output"][0]),
                result("second_reverse_ggb", "input", 0, None, False),
                result("second_reverse_ggb", "weight", 1, None, False),
                result("second_reverse_ggb", "grad_output", 2, specs["grad_output"][0]),
                result("second_reverse_combined", "input", 0, specs["input"][0]),
                result("second_reverse_combined", "weight", 1, specs["weight"][0]),
                result("second_reverse_combined", "bias", 2, None, False),
                result("second_reverse_combined", "grad_output", 3, specs["grad_output"][0])]
    def event(phase, op, arguments=(), output=()):
        return {"phase": phase, "operator": op, "arguments": list(arguments), "output": list(output)}
    view = {"shape": [2,1,6,7], "strides": [42,168,7,1], "storage_offset": 0, "device": "cpu"}
    traces = {}
    for side, device in (("cpu", "cpu"), ("vulkan", "vk:0")):
        views = [dict(shape=[2,1,6,7],strides=[42,168,7,1],storage_offset=offset,device=device)
                 for offset in (0,84)] + [dict(shape=[3,1,3,5],strides=[15,90,5,1],storage_offset=offset,device=device)
                                          for offset in (0,45)]
        traces[side] = [
            event("first_reverse", "aten.convolution_backward.default", views[:2]),
            event("second_reverse_ggI", "aten.view.default", [views[0]], [views[0]]),
            event("second_reverse_ggI", "aten.convolution.default", [views[0], views[2]]),
            event("second_reverse_ggI", "aten.convolution.default", [views[1], views[3]]),
            event("second_reverse_ggI", "aten.cat.default", output=[{"shape":[2,6,4,2],"strides":[48,8,2,1],"storage_offset":0,"device":device}]),
            event("second_reverse_ggW", "aten.convolution.default", [views[0], views[2]]),
            event("second_reverse_ggb", "aten.view.default"),
            event("second_reverse_ggb", "aten.expand.default"),
            event("second_reverse_combined", "aten.add.Tensor"),
            event("second_reverse_combined", "aten.cat.default"),
            event("second_reverse_combined", "aten.convolution.default"),
        ]
    zero = {"compute_dispatches": 31, "vulkan_copies": 0, "explicit_transfers": 0,
            "fallbacks": 0, "buffer_creations_delta": 0, "live_allocations_delta": 0}
    return {"schema": "aten::convolution.default",
            "operands": [{"role": "primary_input", "dtype": "float32", "rank": 4},
                         {"role": "operand", "dtype": "float32", "rank": 4}],
            "primary_input": {"dtype": "float32", "rank": 4},
            "input_dtypes": ["float32"], "input_ranks": [1,4],
            "input_shapes": ["1x4x6x7", "1x6x3x5", "6", "6x2x3x2"],
            "gradients": True, "parity": True, "output_dtype": "float32", "output_rank": 4,
            "graph_autograd": {"route": {"forward_schema":"aten::convolution.default",
                "generated_backward_schema":"aten::convolution_backward.default",
                "native_double_backward_schema":"aten::_convolution_double_backward.default"},
                "geometry": geometry, "operands": operands,
                "directions": ["first_reverse", "second_reverse_ggI", "second_reverse_ggW", "second_reverse_ggb", "second_reverse_combined"],
                "graph_levels": [1,2], "derivative_results": results,
                "route_observations": traces,
            "execution": {"first_backward": dict(zero, compute_dispatches=3), "higher_order": dict(zero)}}}


def test_graph_autograd_valid_grouped_combined_structure_fixture_passes():
    import vulkan_conformance as vc
    fixture = _valid_grouped_combined_coverage_fixture()
    assert vc.validate_graph_autograd_record(
        "convolution.graph.grouped.ggI-ggW-ggb", fixture
    )


@pytest.mark.parametrize("mutation", ["schema", "role_shape", "role_stride", "role_offset",
    "all_float64", "history", "history_requires_grad", "direction", "level", "defined", "nonzero",
    "trace_operator", "trace_layout", "transfer", "fallback"])
def test_graph_autograd_grouped_fixture_rejects_owning_mutation(mutation):
    import vulkan_conformance as vc
    fixture = _valid_grouped_combined_coverage_fixture()
    payload = fixture["graph_autograd"]
    if mutation == "schema":
        fixture["schema"] = "aten::convolution_backward.default"
    elif mutation in {"role_shape", "role_stride", "role_offset"}:
        meta = payload["operands"]["input"]["cpu"]
        key = {"role_shape": "shape", "role_stride": "strides", "role_offset": "storage_offset"}[mutation]
        meta[key] = [9, 9] if mutation != "role_offset" else 9
    elif mutation == "all_float64":
        fixture["input_dtypes"] = ["float64"]
        for pair in payload["operands"].values():
            for meta in pair.values():
                if meta["defined"]: meta["dtype"] = "float64"
    elif mutation == "history":
        payload["derivative_results"][0]["vulkan_grad_fn"] = None
    elif mutation == "history_requires_grad":
        payload["derivative_results"][0]["vulkan_requires_grad"] = False
    elif mutation == "direction":
        payload["directions"].remove("second_reverse_ggb")
    elif mutation == "level":
        payload["graph_levels"] = [1]
    elif mutation == "defined":
        payload["derivative_results"][0]["vulkan_defined"] = False
    elif mutation == "nonzero":
        payload["derivative_results"][0]["vulkan_nonzero"] = False
    elif mutation == "trace_operator":
        payload["route_observations"]["vulkan"].append({"phase":"second_reverse_ggI", "operator":"aten::mul.Tensor", "arguments":[], "output":[]})
        payload["route_observations"]["vulkan"] = [e for e in payload["route_observations"]["vulkan"] if e["operator"] != "aten::cat.default"]
    elif mutation == "trace_layout":
        payload["route_observations"]["cpu"][1]["arguments"][0]["strides"] = [1,1,1,1]
    elif mutation == "transfer":
        payload["execution"]["higher_order"]["explicit_transfers"] = 1
    elif mutation == "fallback":
        payload["execution"]["higher_order"]["fallbacks"] = 1
    owning_errors = {
        "schema": "schema does not match", "role_shape": "invalid cpu input metadata",
        "role_stride": "invalid cpu input metadata", "role_offset": "invalid cpu input metadata",
            "all_float64": "must be float32", "history": "native grad_fn",
            "history_requires_grad": "first_reverse.*requires_grad", "direction": "directions does not match",
        "level": "graph_levels does not match", "defined": "defined slot differs",
            "nonzero": "nonzero on CPU and Vulkan", "trace_operator": "unrecognized canonical",
            "trace_layout": "ggI per-group view/cat", "transfer": "transfers/fallbacks",
        "fallback": "transfers/fallbacks",
    }
    with pytest.raises(ValueError, match=owning_errors[mutation]):
        vc.validate_graph_autograd_record("convolution.graph.grouped.ggI-ggW-ggb", fixture)


@pytest.mark.parametrize("mutation", ["first_detached", "invented_first_node",
    "underreported_first_dispatches", "unknown_result_direction", "duplicate_result",
    "undefined_bias_trainable", "undefined_result_history", "unrecognized_oracle",
    "wrong_first_phase", "malformed_counter"])
def test_graph_autograd_rejects_currently_accepted_source_contract_violations(mutation):
    import vulkan_conformance as vc
    fixture_name = "convolution.graph.grouped.ggI-ggW-ggb"
    if mutation == "undefined_bias_trainable":
        fixture_name = "convolution.graph.depthwise.selected-third"
        fixture = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())[fixture_name]
    else:
        fixture = _valid_grouped_combined_coverage_fixture()
    graph = fixture["graph_autograd"]
    first = [row for row in graph["derivative_results"] if row["direction"] == "first_reverse"]
    if mutation == "first_detached":
        for row in first:
            row["cpu_requires_grad"] = row["vulkan_requires_grad"] = False
    elif mutation == "invented_first_node":
        for row in first:
            row["cpu_grad_fn"] = row["vulkan_grad_fn"] = "InventedNode"
    elif mutation == "underreported_first_dispatches":
        graph["execution"]["first_backward"]["compute_dispatches"] = 1
    elif mutation == "unknown_result_direction":
        row = copy.deepcopy(first[0])
        row["direction"] = "unqualified_fourth_reverse"
        graph["derivative_results"].append(row)
    elif mutation == "duplicate_result":
        graph["derivative_results"].append(copy.deepcopy(first[0]))
    elif mutation == "undefined_bias_trainable":
        graph["operands"]["bias"]["cpu"]["requires_grad"] = True
        graph["operands"]["bias"]["vulkan"]["requires_grad"] = True
    elif mutation == "undefined_result_history":
        row = next(row for row in graph["derivative_results"] if not row["cpu_defined"])
        row["cpu_requires_grad"] = row["vulkan_requires_grad"] = True
    elif mutation == "unrecognized_oracle":
        next(row for row in graph["derivative_results"]
             if row["direction"] == "second_reverse_ggI")["oracle"] = "fabricated"
    elif mutation == "wrong_first_phase":
        next(event for event in graph["route_observations"]["vulkan"]
             if event["phase"] == "first_reverse")["phase"] = "first_reverse_typo"
    elif mutation == "malformed_counter":
        graph["execution"]["higher_order"]["vulkan_copies"] = None
    expected_errors = {
        "first_detached": "first_reverse.*requires_grad",
        "invented_first_node": "native grad_fn",
        "underreported_first_dispatches": "first-backward dispatch count",
        "unknown_result_direction": "unrecognized derivative direction",
        "duplicate_result": "duplicate derivative result",
        "undefined_bias_trainable": "undefined bias.*requires_grad",
        "undefined_result_history": "undefined derivative.*history",
        "unrecognized_oracle": "unrecognized oracle",
        "wrong_first_phase": "unrecognized route-observation phase",
        "malformed_counter": "non-runtime execution counter value",
    }
    with pytest.raises(ValueError, match=expected_errors[mutation]):
        vc.validate_graph_autograd_record(fixture_name, fixture)


def test_graph_autograd_source_forbids_cpu_rejected_t_selected_next_direction():
    import vulkan_conformance as vc
    assert not any("selected_third_d_g_d_x_d_t" in directions
                   for directions in (spec["directions"] for spec in vc.GRAPH_AUTOGRAD_CASES.values()))
    fixture = _valid_grouped_combined_coverage_fixture()
    fixture["graph_autograd"]["directions"].append("selected_third_d_g_d_x_d_t")
    with pytest.raises(ValueError, match="source-owned"):
        vc.validate_graph_autograd_record("convolution.graph.grouped.ggI-ggW-ggb", fixture)


def test_graph_autograd_manifest_validation_rejects_joint_record_and_link_deletion(monkeypatch):
    import vulkan_conformance as vc
    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    for name in vc.GRAPH_AUTOGRAD_REQUIRED_CASES:
        coverage.pop(name)
    manifest = copy.deepcopy(load_manifest(ROOT / "docs/vulkan_capabilities.json"))
    for entry in manifest["entries"]:
        entry["test_cases"] = [case for case in entry["test_cases"]
                               if case["name"] not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
        entry["witnesses"]["cases"] = [name for name in entry["witnesses"]["cases"]
                                        if name not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
        for field in ("reverse_first_order_graph_cases", "reverse_second_order_cases",
                      "reverse_selected_third_order_cases"):
            if field in entry["witnesses"]:
                entry["witnesses"][field] = [name for name in entry["witnesses"][field]
                                              if name not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
                if not entry["witnesses"][field]:
                    del entry["witnesses"][field]
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="reverse_first_order_graph_cases"):
        validate_manifest_data(manifest, ROOT)


def test_generator_and_validator_require_graph_witnesses_from_source_inventory(monkeypatch):
    import vulkan_conformance as vc
    from tools import generate_vulkan_capabilities as generator

    coverage = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    manifest = copy.deepcopy(load_manifest(ROOT / "docs/vulkan_capabilities.json"))
    for name in vc.GRAPH_AUTOGRAD_REQUIRED_CASES:
        coverage.pop(name)
    for entry in manifest["entries"]:
        entry["test_cases"] = [case for case in entry["test_cases"]
                               if case["name"] not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
        entry["witnesses"]["cases"] = [name for name in entry["witnesses"]["cases"]
                                         if name not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
        for field in ("reverse_first_order_graph_cases", "reverse_second_order_cases",
                      "reverse_selected_third_order_cases"):
            if field in entry["witnesses"]:
                entry["witnesses"][field] = [name for name in entry["witnesses"][field]
                                              if name not in vc.GRAPH_AUTOGRAD_REQUIRED_CASES]
                if not entry["witnesses"][field]:
                    del entry["witnesses"][field]

    with pytest.raises(ValueError, match="required graph autograd witness"):
        generator.build_coverage_manifest(coverage)
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="reverse_first_order_graph_cases"):
        validate_manifest_data(manifest, ROOT)


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


def _make_transposed_manifest_factory():
    from test_vulkan_capability_generation import _complete_convolution_fixture
    from tools import generate_vulkan_capabilities as generator

    baseline = None

    def fresh_copy():
        nonlocal baseline
        if baseline is None:
            coverage = _complete_convolution_fixture()
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
                data = generator.build_coverage_manifest(coverage)
            baseline = copy.deepcopy((data, coverage))
        data, coverage = baseline
        return copy.deepcopy(data), copy.deepcopy(coverage)

    return fresh_copy


@pytest.fixture(scope="module")
def transposed_manifest_factory():
    return _make_transposed_manifest_factory()


def _transposed_manifest_fixture(monkeypatch, factory):
    data, coverage = factory()
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    return data, coverage


def test_transposed_manifest_factory_builds_once_and_isolates_both_payloads(monkeypatch):
    from tools import generate_vulkan_capabilities as generator

    original = generator.build_coverage_manifest
    calls = []

    def counted(coverage):
        calls.append(coverage)
        return original(coverage)

    monkeypatch.setattr(generator, "build_coverage_manifest", counted)
    factory = _make_transposed_manifest_factory()
    first_data, first_coverage = factory()
    expected_data, expected_coverage = copy.deepcopy((first_data, first_coverage))
    first_data["entries"].clear()
    first_coverage["convolution.transposed.backward.bias-present.mask-100"]["execution"]["compute_dispatches"] = 0
    next_data, next_coverage = factory()
    assert next_data == expected_data
    assert next_coverage == expected_coverage
    assert len(calls) == 1


def test_transposed_manifest_factory_does_not_share_source_monkeypatches(monkeypatch):
    from tools import generate_vulkan_capabilities as generator

    with monkeypatch.context() as patch:
        patch.setattr(generator, "build_coverage_manifest", lambda coverage: {"source": "patched"})
        patched = _make_transposed_manifest_factory()
        assert patched()[0] == {"source": "patched"}
    fresh = _make_transposed_manifest_factory()
    data, coverage = _transposed_manifest_fixture(monkeypatch, fresh)
    assert "entries" in data
    validate_manifest_data(data, ROOT)


def _change_transposed_weight_and_aggregate_dtype(record):
    record["convolution_operands"]["weight"]["dtype"] = "float64"
    record["input_dtypes"] = ["float32", "float64"]


def _rewrite_transposed_mask(record):
    mask = [False, False, False]
    record["convolution_context"]["output_mask"] = mask
    record["schema_args"]["output_mask"] = mask


def test_manifest_accepts_complete_transposed_convolution_runtime_fixture(monkeypatch, transposed_manifest_factory):
    data, _ = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
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
def test_manifest_rejects_transposed_convolution_coverage_tampering(monkeypatch, mutation, match, transposed_manifest_factory):
    data, coverage = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
    target = "convolution.transposed.backward.bias-present.mask-100"
    mutation(coverage[target])
    with pytest.raises(ValueError, match=match):
        validate_manifest_data(data, ROOT)


def test_manifest_rejects_transposed_mask000_allocation(monkeypatch, transposed_manifest_factory):
    data, coverage = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
    record = coverage["convolution.transposed.backward.bias-present.mask-000"]
    record["execution"]["live_allocations_delta"] = 1
    with pytest.raises(ValueError, match="mask 000"):
        validate_manifest_data(data, ROOT)


def test_manifest_convolution_completeness_survives_joint_transposed_record_and_link_removal(monkeypatch, transposed_manifest_factory):
    data, coverage = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
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


def test_manifest_rejects_joint_transposed_direction_operation_slot_schema_rewrite(monkeypatch, transposed_manifest_factory):
    data, coverage = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
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
    monkeypatch, dtype_group, witness_field, transposed_manifest_factory
):
    data, coverage = _transposed_manifest_fixture(monkeypatch, transposed_manifest_factory)
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


def test_stock_linear_composite_route_is_source_owned_and_finite():
    route = STOCK_COMPOSITE_ROUTES["aten::linear.default"]
    assert route["schema"] == "aten::linear.default"
    assert route["reference"]["pytorch_version"] == "2.4.0"
    assert route["reference"]["source"].endswith("Linear.cpp::linear")
    assert len(route["evidence_cases"]) == 84
    assert len(set(route["evidence_cases"])) == 84


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
    routes["aten::reshape.default"]["evidence_cases"] = [
        "view.reshape.copy.trainable-seed"
    ]
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


def test_stock_linear_route_requires_durable_measured_case_records():
    entries, coverage, source, rejected = _stock_route_inputs()
    routes = copy.deepcopy(STOCK_COMPOSITE_ROUTES)
    validate_stock_composite_routes(routes, entries, coverage, source, rejected)
    altered = copy.deepcopy(coverage)
    del altered["linear.rank8.offset-transposed.bias-present.no-grad"]
    with pytest.raises(ValueError, match="executed Linear route evidence"):
        validate_stock_composite_routes(routes, entries, altered, source, rejected)


def _linear_route_coverage():
    return json.loads((ROOT / "docs/vulkan_coverage.json").read_text())


def test_stock_linear_records_are_durable_and_self_contained():
    coverage = _linear_route_coverage()
    validate_stock_linear_route_evidence(coverage)
    records = {name: record["stock_linear_route"] for name, record in coverage.items()
               if name.startswith("linear.rank")}
    assert len(records) == 84
    assert records["linear.rank1.contiguous.bias-absent.no-grad"]["grad_mode"] == "no_grad"
    assert records["linear.rank2.contiguous.bias-present.trainable"]["selected_second_direction"]


def test_linear_manifest_rank_range_requires_measured_rank_witnesses():
    manifest = load_manifest(ROOT / "docs/vulkan_capabilities.json")
    linear = next(entry for entry in manifest["entries"] if entry["schema"] == "aten::linear.default")
    linear["witnesses"]["ranks"].remove(8)
    linear["witnesses"]["pairs"] = [
        pair for pair in linear["witnesses"]["pairs"] if pair[1] != 8
    ]
    with pytest.raises(ValueError, match="rank 8 was never exercised"):
        validate_manifest_data(manifest, ROOT)


@pytest.mark.parametrize("mutation", [
    "missing", "wrong-rank", "wrong-state", "wrong-layout", "wrong-stride",
    "missing-first-graph", "missing-second-graph", "zero-dispatch", "no-grad-mode",
    "extra-case-id", "malformed-grad-history", "boolean-grad-history",
])
def test_stock_linear_route_validator_rejects_mutated_measured_records(mutation):
    coverage = _linear_route_coverage()
    target = "linear.rank2.contiguous.bias-present.trainable"
    record = coverage[target]
    route = record["stock_linear_route"]
    if mutation == "missing":
        del coverage[target]
    elif mutation == "wrong-rank":
        route["rank"] = 3
    elif mutation == "wrong-state":
        route["state"] = "frozen-weight"
    elif mutation == "wrong-layout":
        route["layout"] = "offset-transposed"
    elif mutation == "wrong-stride":
        route["input_metadata"]["vulkan"]["strides"] = [1, 2]
    elif mutation == "missing-first-graph":
        route["first_gradient_graph_preserved"] = False
    elif mutation == "missing-second-graph":
        route["selected_second_direction"] = None
    elif mutation == "zero-dispatch":
        route["forward_counters"]["compute_dispatches"] = 0
    elif mutation == "no-grad-mode":
        coverage["linear.rank1.contiguous.bias-absent.no-grad"]["stock_linear_route"]["grad_mode"] = "enabled"
    elif mutation == "extra-case-id":
        coverage["linear.rank9.contiguous.bias-absent.trainable"] = copy.deepcopy(record)
        coverage["linear.rank9.contiguous.bias-absent.trainable"]["stock_linear_route"]["case_id"] = (
            "linear.rank9.contiguous.bias-absent.trainable"
        )
    elif mutation == "malformed-grad-history":
        route["first_gradient_facts"][0]["cpu_grad_fn"] = []
        route["first_gradient_facts"][0]["vulkan_grad_fn"] = []
    elif mutation == "boolean-grad-history":
        route["first_gradient_facts"][0]["cpu_requires_grad"] = 1
    with pytest.raises(ValueError, match="executed Linear route|input layout|profiler|first-gradient|second-direction|counter|grad mode"):
        validate_stock_linear_route_evidence(coverage)


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
