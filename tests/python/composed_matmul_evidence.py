"""Versioned connected-model capture and strict CPU-only historical validation."""
import math
from unittest.mock import patch

import torch

from composed_matmul_cases import (
    COMPOSED_MATMUL_SCENARIOS, make_model_case, model_graph, model_metadata,
    model_reference, mutation_case, training_steps,
)
from composed_evidence_common import (
    ROOT, DEPENDENCIES, tree, equal, phases, combine_phases, source_identity,
    validate_phases, validate_counters, validate_runtime, validate_fd, qualify_runtime,
)

SOURCE_IDENTITY_PATHS = DEPENDENCIES + (
    "tests/python/composed_matmul_cases.py", "tests/python/composed_matmul_evidence.py",
)
REQUIRED_SCENARIOS = {s["id"]: s for s in COMPOSED_MATMUL_SCENARIOS}


def fixture(context):
    return tree({k: v for k, v in context.items() if k != "model"})


def model_materialization_minima(scenario, reference=None):
    """Count obligatory stock-matmul reshape copies from actual CPU geometry.

    Pinned LinearAlgebra.cpp:2007-2183 folds rank4 x rank2/rank1 operands and
    expands+flattens batch operands for bmm. A failing .view of that exact shape
    establishes reshape materialization independently of CPU BLAS's internal work.
    Observe the source-owned public model itself, rather than duplicate its math.
    Snapshots, optimizer clones, matrix-kernel gather compute, and unestablished
    derivative copies are deliberately excluded from this conservative lower bound.
    """
    context = make_model_case()
    forward_copies = 0
    original = torch.Tensor.__matmul__
    def reshape_copy(tensor, shape):
        if tensor.numel() == 0:
            return 0
        try:
            tensor.view(shape)
        except RuntimeError:
            return 1
        return 0
    def observe(a, b):
        nonlocal forward_copies
        if a.ndim > 2 and b.ndim in (1, 2):
            forward_copies += reshape_copy(a, (-1, a.shape[-1]))
        else:
            batch = torch.broadcast_shapes(a.shape[:-2], b.shape[:-2])
            for operand in (a, b):
                expanded = operand.expand(*batch, *operand.shape[-2:])
                forward_copies += reshape_copy(expanded, (math.prod(batch), *operand.shape[-2:]))
        return original(a, b)
    with patch.object(torch.Tensor, "__matmul__", observe):
        output = context["model"](context["bases"]["input"])
    # Stock mean loss receives a numel1 seed. loss.cpp reads that allocation
    # element directly; only multi-element noncontiguous seeds materialize.
    seed = torch.ones(())
    dense_seed_copies = int(seed.numel() != 1 and output.numel() != 0 and
                             not seed.expand(output.shape).is_contiguous())
    if scenario["kind"] == "training":
        if reference is None:
            reference = model_reference(scenario)
        return {f"step-{i}-{phase}": count
                for i in range(len(reference["values"]["steps"]))
                for phase, count in (("forward", forward_copies),
                                     ("backward", dense_seed_copies), ("update", 0))}
    result = {"forward": forward_copies}
    if scenario["kind"] == "mutation":
        # Saved-version rejection is an attempted backward, not another forward
        # or an obligation to materialize gradients after its rejection point.
        result["backward"] = 0
    else:
        result["first"] = dense_seed_copies if scenario["kind"] == "hvp" else 0
        if scenario["kind"] in ("mixed", "hvp"):
            result["second"] = 0
    return result


def capture_composed_case(scenario, device="vk:0", root=ROOT, measured_runtime=None):
    import pytorch_vulkan
    from vector_matmul_capability_evidence import runtime_identity
    if scenario != REQUIRED_SCENARIOS.get(scenario.get("id")) or device != "vk:0":
        raise ValueError("wrong source-owned scenario/device")
    cpu_routes, vk_routes = {}, {}
    reference = model_reference(scenario, phase=phases(cpu_routes))
    context = make_model_case(device)
    metadata = {**model_metadata(context), "scenario": dict(scenario)}
    # Fixture snapshots are read before counter reset and before mutations/SGD.
    pytorch_vulkan._C.synchronize()
    vk_fixture = fixture(context)
    cpu_fixture = fixture(make_model_case())
    equal(vk_fixture, cpu_fixture, "fixture", parity=True)
    pytorch_vulkan._C.reset_execution_counters()
    if scenario["kind"] == "training":
        actual = training_steps(context, scenario["set_to_none"], phases(vk_routes))
    elif scenario["kind"] == "mutation":
        actual = mutation_case(context, scenario["mutated"], phases(vk_routes))
    else:
        actual = model_graph(context, scenario, phases(vk_routes))
    pytorch_vulkan._C.synchronize()
    counters = list(pytorch_vulkan._C.execution_counter_snapshot())
    validate_counters(counters, minimum_copies=sum(model_materialization_minima(scenario, reference).values()))
    record = {"scenario": dict(scenario), "fixture": cpu_fixture, "metadata": metadata,
              "cpu": tree(reference["values"]), "vulkan": tree(actual),
              "fd": reference["fd"], "phases": combine_phases(cpu_routes, vk_routes),
              "counters": counters, "source_identity": source_identity(SOURCE_IDENTITY_PATHS, root),
              "runtime_identity": {k: str(v) for k, v in (measured_runtime or runtime_identity(root)).items()}}
    validate_record(record, root)
    return record


def capture_composed_evidence(device="vk:0", root=ROOT):
    from vector_matmul_capability_evidence import runtime_identity
    runtime = runtime_identity(root)
    return {"schema_version": 1, "records": {
        s["id"] + "." + runtime["execution_mode"]: capture_composed_case(s, device, root, runtime)
        for s in COMPOSED_MATMUL_SCENARIOS}}


def validate_record(record, root=ROOT):
    if not isinstance(record, dict) or set(record) != {"scenario", "fixture", "metadata", "cpu", "vulkan", "fd", "phases", "counters", "source_identity", "runtime_identity"}:
        raise ValueError("invalid model record fields")
    scenario = record["scenario"]
    if not isinstance(scenario, dict) or scenario.get("id") not in REQUIRED_SCENARIOS:
        raise ValueError("unknown scenario")
    equal(scenario, REQUIRED_SCENARIOS[scenario["id"]], "scenario")
    equal(record["source_identity"], source_identity(SOURCE_IDENTITY_PATHS, root), "source identity")
    validate_runtime(record["runtime_identity"])
    routes = {}
    reference = model_reference(scenario, phase=phases(routes))
    equal(record["fixture"], fixture(make_model_case()), "fixture")
    equal(record["metadata"], reference["metadata"], "metadata")
    oracle = tree(reference["values"])
    equal(record["cpu"], oracle, "cpu")
    equal(record["vulkan"], oracle, "vulkan", parity=True)
    equal(record["fd"], reference["fd"], "fd")
    validate_fd(reference["fd"])
    validate_phases(record["phases"], routes, model=True)
    validate_counters(record["counters"], minimum_copies=sum(model_materialization_minima(scenario, reference).values()))


def validate_composed_evidence(document, root=ROOT):
    if not isinstance(document, dict) or document.keys() != {"schema_version", "records"} or type(document["schema_version"]) is not int or document["schema_version"] != 1 or not isinstance(document["records"], dict):
        raise ValueError("invalid connected-model schema")
    required = {s + "." + mode for s in REQUIRED_SCENARIOS for mode in ("async", "sync")}
    if document["records"].keys() != required:
        raise ValueError("missing/conflicting connected-model record IDs")
    for key, record in document["records"].items():
        validate_record(record, root)
        equal(key, record["scenario"]["id"] + "." + record["runtime_identity"]["execution_mode"], "record key")


def qualify_composed_current_runtime(document, root=ROOT):
    import pytorch_vulkan
    validate_composed_evidence(document, root)
    mode = pytorch_vulkan._C.execution_mode()
    for key, record in document["records"].items():
        if key.endswith("." + mode):
            qualify_runtime(record["runtime_identity"], root)
