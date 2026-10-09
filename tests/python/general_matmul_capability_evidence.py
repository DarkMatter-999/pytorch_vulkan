"""Finite G2 execution records, CPU replay and explicit live qualification.

Historical validation consumes only source bytes and ordinary CPU PyTorch. HEAD,
extension path and execution mode describe the capture, not the validator host.
"""
import hashlib
import json
import math
import re
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from general_matmul_cases import (
    GENERAL_MATMUL_CASES, make_cpu_bases, make_operands, call_matmul,
    cpu_reference, tensor_fact,
)
from vector_matmul_capability_evidence import runtime_identity

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_CASES = {case["id"]: case for case in GENERAL_MATMUL_CASES}
# Entire modules bind every local semantic dependency, including runtime capture.
SOURCE_IDENTITY_PATHS = (
    "tests/python/general_matmul_cases.py",
    "tests/python/general_matmul_capability_evidence.py",
    "tests/python/vector_matmul_capability_evidence.py",
    "src/vulkan/operators/add.cpp",
    "src/vulkan/operators/linear.cpp",
    "src/vulkan/operators/linear.h",
)


def source_identity(root=ROOT):
    return {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
            for path in SOURCE_IDENTITY_PATHS}


def _json(value):
    return json.loads(json.dumps(value))


def _values(tensor):
    return tensor.detach().cpu().reshape(-1).tolist()


def _fact(tensor):
    return _json(tensor_fact(tensor))


def _pair(cpu, vulkan):
    cpu_values, vk_values = _values(cpu), _values(vulkan)
    return {"cpu": cpu_values, "vulkan": vk_values,
            "cpu_fact": _fact(cpu), "vulkan_fact": _fact(vulkan),
            "device": str(vulkan.device), "value_class": "finite",
            "max_abs_error": max((abs(a-b) for a, b in zip(cpu_values, vk_values)), default=0.)}


def _profile(call):
    with profile(activities=[ProfilerActivity.CPU]) as trace:
        result = call()
    return result, sorted({e.key for e in trace.key_averages() if e.key.startswith("aten::")})


def _forward(case, inputs):
    with torch.set_grad_enabled(not case["no_grad"]):
        return call_matmul(case, *inputs)


def _cpu_forward_route(case):
    # Reconstruct from source independently of every persisted profiler field.
    inputs = make_operands(case, make_cpu_bases(case))
    return _profile(lambda: _forward(case, inputs))[1]


def _phase(call):
    import pytorch_vulkan
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result, ops = _profile(call)
    pytorch_vulkan._C.synchronize()
    # Must precede all CPU readbacks of this phase's results.
    counters = list(pytorch_vulkan._C.execution_counter_snapshot())
    return result, {"vulkan_ops": ops, "counters": counters}


def _cpu_derivative_routes(case, reference):
    """Observe the stock generated derivative closure at the original CPU bases.

    Pinned derivatives.yaml: bmm (377), mm (1172), mv (1190), expand
    (656), and view/transpose/slice inverses. Each phase runs ordinary autograd;
    the mixed directions reuse the independently seeded CPU oracle's probes.
    No persisted profiler list, GPU observation or hand-coded derivative is used.
    """
    if not reference["gradients"]:
        return None, []
    bases = make_cpu_bases(case)
    output = call_matmul(case, *make_operands(case, bases))
    first, first_ops = _profile(lambda: torch.autograd.grad(
        output, tuple(bases[i] for i in reference["targets"]), reference["seed"],
        create_graph=True, retain_graph=True))
    mixed_ops = []
    for witness in reference["mixed"]:
        i, j = witness["direction"]
        _, ops = _profile(lambda: torch.autograd.grad(
            first[i], bases[j], witness["probe"], retain_graph=True)[0])
        mixed_ops.append(ops)
    return first_ops, mixed_ops


def _summary(case, reference):
    inputs = reference["inputs"]
    return {"schema": "aten::matmul.default",
            "operands": [{"role": "primary_input" if i == 0 else "operand",
                          "dtype": "float32", "rank": x.ndim} for i, x in enumerate(inputs)],
            "primary_input": {"dtype": "float32", "rank": inputs[0].ndim},
            "input_dtypes": ["float32"], "input_ranks": sorted({x.ndim for x in inputs}),
            "input_shapes": sorted({"x".join(map(str, x.shape)) for x in (*inputs, *reference["bases"])}),
            "output_dtype": "float32", "output_rank": reference["output"].ndim,
            "gradients": bool(reference["gradients"]), "parity": True}


def capture_general_matmul_case(case, device="vk:0", root=ROOT, measured_runtime=None):
    """Execute source-owned ordinary API, original-base first and selected mixed AD."""
    import pytorch_vulkan
    if case != REQUIRED_CASES.get(case.get("id")) or device != "vk:0":
        raise ValueError("case/device does not match source-owned G2 fixture")
    reference, cpu_all_ops = _profile(lambda: cpu_reference(case))
    cpu_first_ops, cpu_mixed_ops = _cpu_derivative_routes(case, reference)
    cpu_bases = make_cpu_bases(case)
    bases = tuple(x.detach().to(device).requires_grad_(x.requires_grad) for x in cpu_bases)
    operands = make_operands(case, bases)
    cpu_ops = _cpu_forward_route(case)
    output, forward_phase = _phase(lambda: _forward(case, operands))
    forward_phase["cpu_ops"] = cpu_ops
    seed = reference["seed"].to(device)
    targets = reference["targets"]
    first, first_phase = ((), None)
    if reference["gradients"]:
        first, first_phase = _phase(lambda: torch.autograd.grad(
            output, tuple(bases[i] for i in targets), seed, create_graph=True, retain_graph=True))
        first_phase["cpu_ops"] = cpu_first_ops
    mixed = []
    for witness, cpu_ops in zip(reference["mixed"], cpu_mixed_ops):
        i, j = witness["direction"]
        probe = witness["probe"].to(device)
        derivative, phase = _phase(lambda: torch.autograd.grad(
            first[i], bases[j], probe, retain_graph=True)[0])
        phase["cpu_ops"] = cpu_ops
        value = _pair(witness["tensor"], derivative)
        direction_values = _values(witness["perturbation"])
        value.update(direction=list(witness["direction"]), probe=_values(witness["probe"]),
                     perturbation=direction_values, epsilon=witness["epsilon"],
                     projection=witness["projection"], fd=witness["fd"],
                     fd_error=witness["fd_error"],
                     vulkan_projection=sum(a*b for a, b in zip(value["vulkan"], direction_values)),
                     execution=phase)
        mixed.append(value)
    record = _summary(case, reference)
    record["general_matmul_evidence"] = {
        "fixture": _json(case), "source_identity": source_identity(root),
        "runtime_identity": measured_runtime or runtime_identity(root),
        "bases": [_pair(c, v) for c, v in zip(reference["bases"], bases)],
        "inputs": [_pair(c, v) for c, v in zip(reference["inputs"], operands)],
        "output": _pair(reference["output"], output),
        "targets": list(targets), "seed": _values(reference["seed"]),
        "first": [_pair(c, v) for c, v in zip(reference["gradients"], first)],
        "mixed": mixed, "forward": forward_phase, "first_execution": first_phase,
        "cpu_execution_ops": cpu_all_ops,
        "capture_command": "general_matmul_capability_evidence.capture_all_general_matmul_cases(device='vk:0')",
    }
    if mixed:
        record["reverse_autograd"] = {"order": 2, "graph_preserved": True}
    validate_general_matmul_evidence({case["id"]: record}, root)
    return record


def capture_all_general_matmul_cases(device="vk:0", root=ROOT):
    measured = runtime_identity(root)
    return {case["id"]: capture_general_matmul_case(case, device, root, measured)
            for case in GENERAL_MATMUL_CASES}


def _equal(actual, expected, label):
    """Strict recursive equality: Python's bool/int equality is insufficient."""
    if type(actual) is not type(expected):
        raise ValueError(f"{label}: wrong field type")
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{label}: wrong fields")
        for key in expected:
            _equal(actual[key], expected[key], f"{label}.{key}")
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{label}: wrong length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            _equal(a, b, f"{label}[{i}]")
    elif actual != expected:
        raise ValueError(f"{label}: does not match CPU/source reference")


def validate_general_matmul_conformance_case(conformance, case):
    """Bind a coincident registry to its executed CPU factory/API/view contract.

    Equal shape metadata alone cannot establish equal fixtures. Intercept the two
    public Python spellings only to observe which is called, the actual operands,
    their original-base storage and grad mode; execute their original methods.
    Output and original-base derivatives still use ordinary pinned CPU PyTorch.
    No registry-supplied recipe metadata is trusted as a substitute for execution.
    """
    from unittest.mock import patch

    label = case["id"] + ": conflicting conformance/evidence fixture"
    reference = cpu_reference(case)
    expected_gradient = bool(reference["gradients"])
    if (conformance.declaration_id != "aten::matmul.default" or not conformance.supported
            or conformance.expected_shape != case["output_shape"]
            or conformance.declared_shapes != tuple("x".join(map(str, s)) for s in case["base_shapes"])
            or conformance.check_gradients != expected_gradient
            or conformance.setup_inputs is None):
        raise ValueError(label)

    def check_bases(bases):
        if not isinstance(bases, tuple) or len(bases) != 2:
            raise ValueError(label + ": wrong original-base inputs")
        for base, expected in zip(bases, reference["bases"]):
            if not isinstance(base, torch.Tensor) or base.device.type != "cpu":
                raise ValueError(label + ": factory must produce CPU tensors")
            _equal(_fact(base), _fact(expected), label + ".base_fact")
            _equal(_values(base), _values(expected), label + ".base_values")

    matmul, at = torch.matmul, torch.Tensor.__matmul__
    for operation in (conformance.operation, conformance.cpu_reference):
        bases = conformance.inputs()
        check_bases(bases)
        original_bases = bases
        bases = conformance.setup_inputs(bases, "cpu")
        check_bases(bases)
        if any(a is not b for a, b in zip(bases, original_bases)):
            raise ValueError(label + ": setup changed original gradient targets")
        calls = []
        def observe(api, original, a, b, *args, **kwargs):
            calls.append(api)
            if (api != case["api"] or args or kwargs
                    or torch.is_grad_enabled() != (not case["no_grad"])):
                raise ValueError(label + ": public API/grad mode mismatch")
            for operand, expected, base in zip((a, b), reference["inputs"], bases):
                if not isinstance(operand, torch.Tensor):
                    raise ValueError(label + ": non-tensor operand")
                # A view constructed under true no_grad can legitimately have
                # different history than the oracle's preconstructed view.
                actual_fact, expected_fact = _fact(operand), _fact(expected)
                actual_fact.pop("history"); expected_fact.pop("history")
                _equal(actual_fact, expected_fact, label + ".operand_layout_selection")
                _equal(_values(operand), _values(expected), label + ".operand_values")
                if operand.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
                    raise ValueError(label + ": view recipe lost original-base storage")
            return original(a, b, *args, **kwargs)
        with patch.object(torch, "matmul", lambda a, b, *args, **kwargs:
                          observe("matmul", matmul, a, b, *args, **kwargs)), patch.object(
                torch.Tensor, "__matmul__", lambda a, b: observe("@", at, a, b)):
            try:
                output = operation(*bases, *conformance.args, **(conformance.kwargs or {}))
            except (AssertionError, RuntimeError, TypeError) as error:
                raise ValueError(label + ": CPU operation violates source contract") from error
        if calls != [case["api"]] or not isinstance(output, torch.Tensor):
            raise ValueError(label + ": missing/extra allocating public call")
        _equal(_fact(output), _fact(reference["output"]), label + ".output_fact")
        _equal(_values(output), _values(reference["output"]), label + ".output_values")
        if expected_gradient:
            gradients = torch.autograd.grad(output, tuple(bases[i] for i in reference["targets"]),
                                            reference["seed"], create_graph=True)
            for gradient, expected in zip(gradients, reference["gradients"]):
                _equal(_fact(gradient), _fact(expected), label + ".original_base_gradient_fact")
                _equal(_values(gradient), _values(expected), label + ".original_base_gradient_values")


def _numeric(value, label):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{label}: expected finite number")


def _validate_pair(value, cpu, label, extra=()):
    keys = {"cpu", "vulkan", "cpu_fact", "vulkan_fact", "device", "value_class", "max_abs_error"} | set(extra)
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{label}: malformed numerical evidence")
    _equal(value["cpu"], _values(cpu), label + ".cpu")
    _equal(value["cpu_fact"], _fact(cpu), label + ".cpu_fact")
    _equal(value["vulkan_fact"], _fact(cpu), label + ".vulkan_fact")
    if value["device"] != "vk:0" or value["value_class"] != "finite":
        raise ValueError(f"{label}: wrong residency/value class")
    vk = value["vulkan"]
    if not isinstance(vk, list) or len(vk) != cpu.numel():
        raise ValueError(f"{label}: wrong Vulkan length")
    for a, b in zip(vk, value["cpu"]):
        _numeric(a, label)
        if abs(a-b) > .003 + .003*abs(b):
            raise ValueError(f"{label}: componentwise parity failed")
    error = max((abs(a-b) for a, b in zip(vk, value["cpu"])), default=0.)
    _numeric(value["max_abs_error"], label)
    if value["max_abs_error"] != error:
        raise ValueError(f"{label}: false numerical error")


def _validate_phase(phase, label):
    keys = {"vulkan_ops", "cpu_ops", "counters"}
    if not isinstance(phase, dict) or set(phase) != keys:
        raise ValueError(f"{label}: missing execution route/counters")
    for key in keys - {"counters"}:
        ops = phase[key]
        if (not isinstance(ops, list) or not ops or any(type(x) is not str or not x.startswith("aten::") for x in ops)
                or ops != sorted(set(ops))):
            raise ValueError(f"{label}: malformed profiler route")
        for op in ops:
            name = op.removeprefix("aten::")
            if not hasattr(torch.ops.aten, name):
                raise ValueError(f"{label}: nonexistent ATen profiler route {op}")
    counters = phase["counters"]
    if (not isinstance(counters, list) or len(counters) != 4
            or any(type(x) is not int or x < 0 for x in counters) or counters[2:] != [0, 0]):
        raise ValueError(f"{label}: invalid measured device counters")


# Generated matrix derivatives preserve these semantic leaves and inverse views /
# broadcast reductions across CPU and Vulkan. CPU BLAS implementation details
# (addmv_, resolve_conj, reduction fill_) are not Vulkan requirements. Existing
# Vulkan mm/mv/bmm leaves add allocation/as_strided/expand metadata; the stock mv
# composition can additionally insert squeeze/unsqueeze. These are allowed bounds,
# not universal clone counts or an exact backend profiler event-order requirement.
_MATRIX_DERIVATIVE_OPS = {"aten::dot", "aten::mv", "aten::mm", "aten::bmm",
                          "aten::ger", "aten::outer"}
_DERIVATIVE_SEMANTIC_OPS = _MATRIX_DERIVATIVE_OPS | {
    "aten::view", "aten::reshape", "aten::_unsafe_view", "aten::_reshape_alias",
    "aten::as_strided", "aten::transpose", "aten::t", "aten::squeeze",
    "aten::unsqueeze", "aten::expand", "aten::sum", "aten::slice_backward",
}
_VULKAN_LEAF_METADATA_OPS = {
    "aten::empty", "aten::as_strided", "aten::expand", "aten::squeeze", "aten::unsqueeze",
}


def _validate_derivative_route(phase, cpu_ops, label):
    _equal(phase["cpu_ops"], cpu_ops, label + ".cpu_route")
    actual, reference = set(phase["vulkan_ops"]), set(cpu_ops)
    required = reference & _DERIVATIVE_SEMANTIC_OPS
    if (not required <= actual
            or actual & _MATRIX_DERIVATIVE_OPS != reference & _MATRIX_DERIVATIVE_OPS
            or not actual <= reference | _VULKAN_LEAF_METADATA_OPS):
        raise ValueError(f"{label}: Vulkan route does not match generated CPU derivative closure")


def _validate_forward_route(case, phase, cpu_ops, label):
    _equal(phase["cpu_ops"], cpu_ops, label + ".cpu_route")
    actual, reference = set(phase["vulkan_ops"]), set(cpu_ops)
    # Stock _matmul_impl's fold, expand/reshape/bmm, and right-large output
    # contiguous composition is shared. Native CPU BLAS conjugation/addmv/zero
    # implementation details are excluded; device leaf metadata remains bounded.
    semantic = _DERIVATIVE_SEMANTIC_OPS | {
        "aten::matmul", "aten::mT", "aten::contiguous", "aten::clone", "aten::copy_",
    }
    required = reference & semantic
    leaves = {"aten::dot", "aten::mv", "aten::mm", "aten::bmm"}
    if (not required <= actual or actual & leaves != {"aten::" + case["route"]}
            or not actual <= reference | _VULKAN_LEAF_METADATA_OPS):
        raise ValueError(f"{label}: Vulkan route does not match stock CPU forward closure")
    cpu_materializes = "aten::clone" in reference or "aten::copy_" in reference
    if cpu_materializes != case["copy_route"]:
        raise ValueError(f"{label}: source geometry and CPU materialization route disagree")
    if case["copy_route"] and (not {"aten::clone", "aten::copy_"} <= actual
                               or phase["counters"][1] < case["materialization_minimum"]):
        raise ValueError(f"{label}: necessary stock materialization has no device copy")


def validate_general_matmul_evidence(coverage, root, require_complete=False):
    names = {n for n in coverage if n.startswith("g2.")}
    if names - REQUIRED_CASES.keys():
        raise ValueError("unexpected G2 case IDs")
    if require_complete and REQUIRED_CASES.keys() - names:
        raise ValueError("required executed general matmul witness missing")
    sources = source_identity(root)
    observed = None
    for name in sorted(names):
        case = REQUIRED_CASES[name]
        reference, cpu_all_ops = _profile(lambda: cpu_reference(case))
        record = coverage[name]
        if not isinstance(record, dict):
            raise ValueError(f"{name}: malformed coverage record")
        e = record.get("general_matmul_evidence")
        if not isinstance(e, dict) or set(e) != {
            "fixture", "source_identity", "runtime_identity", "bases", "inputs", "output",
            "targets", "seed", "first", "mixed", "forward", "first_execution",
            "cpu_execution_ops", "capture_command",
        }:
            raise ValueError(f"{name}: missing/malformed general matmul evidence")
        expected_record = _summary(case, reference)
        if case["mixed"]:
            expected_record["reverse_autograd"] = {"order": 2, "graph_preserved": True}
        _equal({k: v for k, v in record.items() if k != "general_matmul_evidence"}, expected_record, name)
        _equal(e["fixture"], _json(case), name + ".fixture")
        _equal(e["source_identity"], sources, name + ".source_identity")
        _validate_runtime(e["runtime_identity"], name)
        if observed is not None and observed != e["runtime_identity"]:
            raise ValueError(f"{name}: inconsistent runtime provenance")
        observed = e["runtime_identity"]
        _equal(e["targets"], list(reference["targets"]), name + ".targets")
        _equal(e["seed"], _values(reference["seed"]), name + ".seed")
        for key, cpu_key in (("bases", "bases"), ("inputs", "inputs"), ("first", "gradients")):
            if not isinstance(e[key], list) or len(e[key]) != len(reference[cpu_key]):
                raise ValueError(f"{name}.{key}: missing values/history")
            for i, (value, cpu) in enumerate(zip(e[key], reference[cpu_key])):
                _validate_pair(value, cpu, f"{name}.{key}[{i}]")
        _validate_pair(e["output"], reference["output"], name + ".output")
        _validate_phase(e["forward"], name + ".forward")
        if reference["output"].numel() and e["forward"]["counters"][0] == 0:
            raise ValueError(f"{name}: nonempty forward has no device compute")
        _validate_forward_route(case, e["forward"], _cpu_forward_route(case), name + ".forward")
        if reference["gradients"]:
            cpu_first_ops, cpu_mixed_ops = _cpu_derivative_routes(case, reference)
            _validate_phase(e["first_execution"], name + ".first")
            _validate_derivative_route(e["first_execution"], cpu_first_ops, name + ".first")
            if all(x.numel() for x in reference["inputs"]) and sum(e["first_execution"]["counters"][:2]) == 0:
                raise ValueError(f"{name}: first reverse has no device work")
        elif e["first_execution"] is not None:
            raise ValueError(f"{name}: false first-gradient execution")
        if not isinstance(e["mixed"], list) or len(e["mixed"]) != len(reference["mixed"]):
            raise ValueError(f"{name}: missing mixed direction")
        for index, (value, witness) in enumerate(zip(e["mixed"], reference["mixed"])):
            extra = ("direction", "probe", "perturbation", "epsilon", "projection", "fd", "fd_error", "vulkan_projection", "execution")
            _validate_pair(value, witness["tensor"], name + ".mixed", extra)
            for key in ("direction", "probe", "perturbation", "epsilon", "projection", "fd", "fd_error"):
                expected = (_values(witness[key]) if key in ("probe", "perturbation")
                            else _json(witness[key]))
                _equal(value[key], expected, name + ".mixed." + key)
            projection = sum(a*b for a, b in zip(value["vulkan"], value["perturbation"]))
            _numeric(value["vulkan_projection"], name)
            if value["vulkan_projection"] != projection or not any(value["vulkan"]) or not any(value["cpu"]):
                raise ValueError(f"{name}: false/zero mixed projection")
            if abs(witness["projection"]-witness["fd"]) > .002 + .008*abs(witness["fd"]):
                raise ValueError(f"{name}: independent CPU FD failed")
            _validate_phase(value["execution"], name + ".mixed.execution")
            _validate_derivative_route(value["execution"], cpu_mixed_ops[index], name + ".mixed.execution")
            if sum(value["execution"]["counters"][:2]) == 0 or witness["projection"] == 0:
                raise ValueError(f"{name}: mixed direction has no work/signal")
        ops = e["cpu_execution_ops"]
        if not isinstance(ops, list) or not ops or any(type(x) is not str or not x.startswith("aten::") for x in ops) or ops != sorted(set(ops)):
            raise ValueError(f"{name}: malformed CPU execution profiler")
        _equal(ops, cpu_all_ops, name + ".combined_cpu_route")
        if e["capture_command"] != "general_matmul_capability_evidence.capture_all_general_matmul_cases(device='vk:0')":
            raise ValueError(f"{name}: wrong capture recipe")


def _validate_runtime(runtime, name):
    fields = {"pytorch_version", "pytorch_git_revision", "checkout_head", "extension_sha256",
              "extension_path", "device", "hardware", "driver", "vulkan_api_version",
              "vulkan_instance_version", "execution_mode"}
    if (not isinstance(runtime, dict) or set(runtime) != fields
            or any(not isinstance(v, str) or not v.strip() for v in runtime.values())
            or not re.fullmatch(r"[0-9a-f]{40}", runtime["checkout_head"])
            or not re.fullmatch(r"[0-9a-f]{40}", runtime["pytorch_git_revision"])
            or not re.fullmatch(r"[0-9a-f]{64}", runtime["extension_sha256"])
            or not Path(runtime["extension_path"]).is_absolute()
            or runtime["device"] != "vk:0" or runtime["execution_mode"] not in {"async", "sync"}
            or any(not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", runtime[k])
                   for k in ("vulkan_api_version", "vulkan_instance_version"))
            or runtime["pytorch_version"] != torch.__version__
            or runtime["pytorch_git_revision"] != str(torch.version.git_version)):
        raise ValueError(f"{name}: malformed/wrong CPU reference runtime identity")


def qualify_general_matmul_current_runtime(coverage, root):
    validate_general_matmul_evidence(coverage, root, require_complete=True)
    current = runtime_identity(root)
    compared = current.keys() - {"checkout_head", "extension_path"}
    for name in REQUIRED_CASES:
        runtime = coverage[name]["general_matmul_evidence"]["runtime_identity"]
        if any(runtime[k] != current[k] for k in compared):
            raise ValueError(f"{name}: capture does not qualify the current build/device/mode")
