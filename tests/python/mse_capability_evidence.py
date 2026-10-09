"""Public F/nn MSE finite first/selected second reverse execution evidence."""
import torch

from mse_autograd_cases import MSE_AUTOGRAD_CASES, make_mse_case, mse_forward, mse_reference
from composed_evidence_common import (
    ROOT, DEPENDENCIES, tree, equal, phases, combine_phases, source_identity,
    validate_phases, validate_counters, validate_runtime, validate_fd, qualify_runtime,
)

SOURCE_IDENTITY_PATHS = DEPENDENCIES + (
    "tests/python/mse_autograd_cases.py", "tests/python/mse_capability_evidence.py",
)
REQUIRED_CASES = {"mse-ad." + c["id"] + "." + api: {**c, "api": api}
                  for c in MSE_AUTOGRAD_CASES for api in ("functional", "module")}


def execute(case, context, phase):
    bases = context["bases"]
    with phase("forward"):
        loss = mse_forward(case, bases, case["api"])
    with phase("first"):
        upstream = bases[2]
        if case["kind"] == "expanded-upstream" and case["reduction"] == "none":
            upstream = upstream.expand_as(loss)
        selected = {"input": (0,), "target": (1,), "both": (0, 1)}[case["selection"]]
        first = torch.autograd.grad(loss, tuple(bases[i] for i in selected), upstream, create_graph=True)
    with phase("second"):
        contraction = sum((g * context["probes"][i]).sum() for i, g in zip(selected, first))
        second = torch.autograd.grad(contraction, bases)
    return {"loss": loss, "upstream": upstream, "selected": selected,
            "first": first, "second": second, "contraction": contraction}


def replay(case):
    routes = {}
    context = make_mse_case(case)
    values = execute(case, context, phases(routes))
    base_case = {k: v for k, v in case.items() if k != "api"}
    reference = mse_reference(base_case, api=case["api"])
    return context, values, reference["fd"], routes


def exceptional(case):
    return case["kind"] == "empty" and case["reduction"] == "mean"


def mse_materialization_minima(case):
    """Geometry-derived lower bounds, independent of recorded ops/counters.

    Public F/nn broadcast the original operands. loss.cpp:44-45,102-104
    materializes each nonempty noncontiguous operand on every native call.
    derivatives.yaml:2006-2008 invokes one native first backward per selected
    operand; :2486-2489 invokes native backward again for each requested upstream
    second derivative. Broadcast inverse SumBackward expands the original-base
    contraction probe before that second native call. No CPU BLAS event counts or
    stored evidence field is used; dense/scalar/empty .contiguous() is not a copy.
    """
    context = make_mse_case(case)
    operands = torch.broadcast_tensors(*context["bases"][:2])
    shape = operands[0].shape
    def copy_needed(tensor):
        return int(tensor.numel() != 0 and not tensor.is_contiguous())
    operand_copies = sum(copy_needed(t) for t in operands)
    def upstream_copy(tensor):
        # loss.cpp reads numel1 directly; every multi-element broadcast still
        # expands/materializes before the per-logical-element shader read.
        return 0 if tensor.numel() == 1 else copy_needed(tensor.expand(shape))
    upstream_copies = upstream_copy(context["bases"][2])
    selected = {"input": (0,), "target": (1,), "both": (0, 1)}[case["selection"]]
    return {"forward": operand_copies,
            "first": len(selected) * (operand_copies + upstream_copies),
            "second": sum(operand_copies + upstream_copy(context["probes"][i])
                          for i in selected)}


def summary(case):
    context = make_mse_case(case)
    inputs = context["bases"][:2]
    output = mse_forward(case, context["bases"], case["api"])
    return {"schema": "aten::mse_loss.default",
            "operands": [{"role": "primary_input" if i == 0 else "operand", "dtype": "float32", "rank": t.ndim} for i, t in enumerate(inputs)],
            "primary_input": {"dtype": "float32", "rank": inputs[0].ndim},
            "input_dtypes": ["float32"], "input_ranks": sorted({t.ndim for t in inputs}),
            "input_shapes": sorted({"x".join(map(str, t.shape)) for t in inputs if t.ndim}),
            "output_dtype": "float32", "output_rank": output.ndim,
            "gradients": True, "parity": True,
            "reverse_autograd": {"order": 2, "graph_preserved": True}}


def capture_mse_case(name, device="vk:0", root=ROOT, measured_runtime=None):
    import pytorch_vulkan
    from vector_matmul_capability_evidence import runtime_identity
    if name not in REQUIRED_CASES or device != "vk:0":
        raise ValueError("wrong MSE case/device")
    case = REQUIRED_CASES[name]
    context, cpu, fd, cpu_routes = replay(case)
    vk_context = make_mse_case(case, device)
    vk_routes = {}
    pytorch_vulkan._C.synchronize()
    equal(tree(vk_context), tree(context), "fixture", parity=True)
    pytorch_vulkan._C.reset_execution_counters()
    actual = execute(case, vk_context, phases(vk_routes))
    pytorch_vulkan._C.synchronize()
    counters = list(pytorch_vulkan._C.execution_counter_snapshot())
    validate_counters(counters, minimum_copies=sum(mse_materialization_minima(case).values()))
    record = summary(case)
    record["mse_autograd_evidence"] = {
        "case": dict(case), "fixture": tree(context),
        "cpu": tree(cpu, empty_mean=exceptional(case)),
        "vulkan": tree(actual, empty_mean=exceptional(case)), "fd": fd,
        "phases": combine_phases(cpu_routes, vk_routes), "counters": counters,
        "source_identity": source_identity(SOURCE_IDENTITY_PATHS, root),
        "runtime_identity": {k: str(v) for k, v in (measured_runtime or runtime_identity(root)).items()}}
    validate_mse_evidence({name: record}, root)
    return record


def capture_all_mse_cases(device="vk:0", root=ROOT):
    from vector_matmul_capability_evidence import runtime_identity
    runtime = runtime_identity(root)
    return {name: capture_mse_case(name, device, root, runtime) for name in REQUIRED_CASES}


def validate_mse_evidence(coverage, root=ROOT, *, require_complete=False):
    names = {n for n in coverage if n.startswith("mse-ad.")}
    if names - REQUIRED_CASES.keys():
        raise ValueError("unknown MSE case IDs")
    if require_complete and names != REQUIRED_CASES.keys():
        raise ValueError("missing MSE cases")
    for name in names:
        record, case = coverage[name], REQUIRED_CASES[name]
        expected = summary(case)
        if not isinstance(record, dict) or record.keys() != expected.keys() | {"mse_autograd_evidence"}:
            raise ValueError(name + ": invalid record fields")
        equal({k: record[k] for k in expected}, expected, name + ".summary")
        e = record["mse_autograd_evidence"]
        if not isinstance(e, dict) or e.keys() != {"case", "fixture", "cpu", "vulkan", "fd", "phases", "counters", "source_identity", "runtime_identity"}:
            raise ValueError(name + ": invalid evidence fields")
        equal(e["case"], case, name + ".case")
        equal(e["source_identity"], source_identity(SOURCE_IDENTITY_PATHS, root), name + ".source")
        validate_runtime(e["runtime_identity"])
        context, values, fd, routes = replay(case)
        equal(e["fixture"], tree(context), name + ".fixture")
        oracle = tree(values, empty_mean=exceptional(case))
        equal(e["cpu"], oracle, name + ".cpu")
        equal(e["vulkan"], oracle, name + ".vulkan", parity=True)
        equal(e["fd"], fd, name + ".fd")
        validate_fd(fd)
        validate_phases(e["phases"], routes, mse_case=case)
        validate_counters(e["counters"], minimum_copies=sum(mse_materialization_minima(case).values()))


def qualify_mse_current_runtime(coverage, root=ROOT):
    validate_mse_evidence(coverage, root, require_complete=True)
    for name in REQUIRED_CASES:
        qualify_runtime(coverage[name]["mse_autograd_evidence"]["runtime_identity"], root)


def validate_mse_conformance_case(conformance, case):
    """Execute coincident factories, bind full values/layout/API and base targets."""
    from unittest.mock import patch
    import torch.nn.functional as F
    expected = make_mse_case(case)["bases"]
    label = "conflicting MSE conformance/evidence fixture"
    shape = tuple(mse_forward(case, expected, case["api"]).shape)
    if (conformance.declaration_id != "aten::mse_loss.default" or not conformance.supported
            or not conformance.check_gradients or conformance.expected_shape != shape
            or conformance.setup_inputs is None
            or conformance.declared_shapes != tuple("x".join(map(str, t.shape)) for t in expected[:2] if t.ndim)):
        raise ValueError(label)
    for operation in (conformance.operation, conformance.cpu_reference):
        bases = conformance.inputs()
        if type(bases) is not tuple or len(bases) != 3:
            raise ValueError(label + ": wrong original-base container/inventory")
        equal(tree(bases), tree(expected), label + ".bases")
        setup = conformance.setup_inputs(bases, "cpu")
        if type(setup) is not tuple or len(setup) != 3 or any(setup[i] is not bases[i] for i in range(3)):
            raise ValueError(label + ": changed base gradient targets")
        calls = []
        original_f, original_module = F.mse_loss, torch.nn.MSELoss.forward
        def observe_f(a, b, *args, **kwargs):
            calls.append("functional")
            if a is not bases[0] or b is not bases[1] or args or kwargs != {"reduction": case["reduction"]}:
                raise ValueError(label + ": operands/reduction")
            return original_f(a, b, **kwargs)
        def observe_module(module, a, b):
            calls.append("module")
            if a is not bases[0] or b is not bases[1] or module.reduction != case["reduction"]:
                raise ValueError(label + ": module operands/reduction")
            return original_module(module, a, b)
        with patch.object(F, "mse_loss", observe_f), patch.object(torch.nn.MSELoss, "forward", observe_module):
            output = operation(*setup)
        equal(calls, ["functional"] if case["api"] == "functional" else ["module", "functional"], label + ".API")
        equal(tree(output, empty_mean=exceptional(case), path="loss"),
              tree(mse_forward(case, expected, case["api"]), empty_mean=exceptional(case), path="loss"), label + ".output")


def validate_mse_manifest_bindings(entries, coverage):
    entry = next(e for e in entries if e["schema"] == "aten::mse_loss.default")
    expected = sorted(REQUIRED_CASES)
    if entry["autograd"] != "reverse_mse_finite_second_order_witnessed":
        raise ValueError("MSE requires finite generated reverse declaration")
    if entry["ranks"] != {"min": 0, "max": 2} or entry["witnesses"]["ranks"] != [0, 1, 2]:
        raise ValueError("MSE ranks must match scalar/matrix AD and vector first-reverse witnesses")
    for key in ("reverse_first_order_graph_cases", "reverse_second_order_cases"):
        if entry["witnesses"].get(key) != expected:
            raise ValueError("MSE graph links differ from source-owned executed cases")
    if not set(expected) <= {c["name"] for c in entry["test_cases"] if c["supported"]}:
        raise ValueError("MSE missing supported source-owned cases")
