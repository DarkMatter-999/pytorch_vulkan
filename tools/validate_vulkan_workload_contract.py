"""CPU-only validation for the source-bound Stage G workload evidence."""

from __future__ import annotations

import copy
import math
import numbers
import re

import torch


# Deliberately independent of tests.python.vulkan_workload_conformance.SCENARIOS.
REQUIRED_WORKLOADS = (
    ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none", "none"),
    ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero", "zero"),
    ("hvp.grouped-depthwise.output-energy.parameters", None),
    ("dense.two-linear.output-energy.sgd-momentum.zero-grad-none", "none"),
    ("dense.two-linear.output-energy.sgd-momentum.zero-grad-zero", "zero"),
    ("hvp.dense.two-linear.output-energy.parameters", None),
)
REQUIRED_MODES = ("async", "sync")
SCHEMA_VERSION = 1
TORCH_VERSION = "2.4.0+cpu"
TORCH_GIT_REVISION = "e4ee3be4063b7c430974252fdf7db42273388d86"
RTOL = 3e-3
ATOL = 3e-3
_CLASSIFIER_NAMES = ("0.weight", "2.weight", "2.bias", "5.weight", "5.bias")
_HVP_NAMES = ("0.weight", "1.weight", "1.bias")
_SHAPES = {
    "0.weight": (6, 2, 3, 3),
    "2.weight": (6, 1, 3, 3),
    "2.bias": (6,),
    "5.weight": (3, 150),
    "5.bias": (3,),
    "1.weight": (6, 1, 3, 3),
    "1.bias": (6,),
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _record_identity_key(record):
    """Validate identity scalars before any set/dict membership or tuple key use."""
    if not isinstance(record, dict):
        _fail("record must be an object")
    for field in ("workload_id", "execution_mode"):
        if type(record.get(field)) is not str:
            _fail(f"record field {field} must be a string")
    reset_mode = record.get("reset_mode")
    if reset_mode is not None and type(reset_mode) is not str:
        _fail("record field reset_mode must be a string or null")
    return record["workload_id"], reset_mode, record["execution_mode"]


def installed_torch_metadata() -> tuple[str, str]:
    """Require and return the installed source-reference PyTorch identity."""
    version = torch.__version__
    revision = torch.version.git_version
    if version != TORCH_VERSION or revision != TORCH_GIT_REVISION:
        _fail(
            "installed PyTorch version/revision differs from source contract: "
            f"{version!r}/{revision!r}"
        )
    return version, revision


def validate_required_matrix(scenarios) -> None:
    declared = {(item.workload_id, item.reset_mode) for item in scenarios}
    if declared != set(REQUIRED_WORKLOADS):
        _fail("source-owned required workload matrix was changed or is incomplete")


def _is_number(value) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _walk_tensor_pairs(cpu, vk, path=""):
    if isinstance(cpu, dict) and isinstance(vk, dict):
        tensor_keys = {"dtype", "shape", "device", "values", "contiguous", "nonzero_count"}
        if tensor_keys <= cpu.keys() and tensor_keys <= vk.keys():
            yield path, cpu, vk
            return
        if cpu.keys() != vk.keys():
            _fail(f"{path}: CPU/Vulkan payload keys differ")
        for key in cpu:
            yield from _walk_tensor_pairs(cpu[key], vk[key], f"{path}.{key}" if path else key)
    elif isinstance(cpu, list) and isinstance(vk, list):
        if len(cpu) != len(vk):
            _fail(f"{path}: CPU/Vulkan list lengths differ")
        for index, (left, right) in enumerate(zip(cpu, vk)):
            yield from _walk_tensor_pairs(left, right, f"{path}[{index}]")


def _validate_tensor(tensor, path, expected_device, *, require_contiguous=True):
    if not isinstance(tensor, dict):
        _fail(f"{path}: expected tensor payload")
    required = {"dtype", "shape", "device", "contiguous", "values", "nonzero_count"}
    if tensor.keys() != required:
        _fail(f"{path}: malformed tensor fields")
    if tensor["dtype"] != "torch.float32" or tensor["device"] != expected_device:
        _fail(f"{path}: wrong dtype or device")
    shape = tensor["shape"]
    if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
        _fail(f"{path}: malformed shape")
    if type(tensor["contiguous"]) is not bool or (require_contiguous and tensor["contiguous"] is not True):
        _fail(f"{path}: tensor contiguity metadata is invalid")
    values = tensor["values"]
    count = math.prod(shape)
    if not isinstance(values, list) or len(values) != count:
        _fail(f"{path}: tensor element count does not match shape")
    if any(not _is_number(value) or not math.isfinite(float(value)) for value in values):
        _fail(f"{path}: values must be finite non-boolean numbers")
    nonzero = sum(float(value) != 0.0 for value in values)
    if type(tensor["nonzero_count"]) is not int or tensor["nonzero_count"] != nonzero:
        _fail(f"{path}: incorrect nonzero_count")


def _validate_tensor_tree(node, path, device, *, require_contiguous=True):
    if isinstance(node, dict):
        if {"dtype", "shape", "device", "values", "contiguous", "nonzero_count"} <= node.keys():
            _validate_tensor(node, path, device, require_contiguous=require_contiguous)
            return
        for key, child in node.items():
            _validate_tensor_tree(child, f"{path}.{key}" if path else key, device, require_contiguous=require_contiguous)
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _validate_tensor_tree(child, f"{path}[{index}]", device, require_contiguous=require_contiguous)


def _same_json(left, right, path=""):
    if type(left) is not type(right):
        _fail(f"{path}: CPU oracle payload type differs")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            _fail(f"{path}: CPU oracle payload fields differ")
        for key in left:
            _same_json(left[key], right[key], f"{path}.{key}" if path else key)
    elif isinstance(left, list):
        if len(left) != len(right):
            _fail(f"{path}: CPU oracle payload list length differs")
        for index, (a, b) in enumerate(zip(left, right)):
            _same_json(a, b, f"{path}[{index}]")
    elif isinstance(left, float):
        if not math.isfinite(left) or left != right:
            _fail(f"{path}: CPU oracle payload value differs")
    elif left != right:
        _fail(f"{path}: CPU oracle payload value differs")


def _set_expected_vk_devices(cpu, vk):
    if isinstance(cpu, dict) and isinstance(vk, dict):
        tensor = {"dtype", "shape", "device", "values", "contiguous", "nonzero_count"}
        if tensor <= cpu.keys() and tensor <= vk.keys():
            if vk["device"] != "vk:0":
                _fail("Vulkan tensor must retain original vk:0 residency")
            return
        if cpu.keys() != vk.keys():
            _fail("CPU/Vulkan payload nested keysets differ")
        for key in cpu:
            _set_expected_vk_devices(cpu[key], vk[key])
    elif isinstance(cpu, list) and isinstance(vk, list):
        if len(cpu) != len(vk):
            _fail("CPU/Vulkan payload list lengths differ")
        for left, right in zip(cpu, vk):
            _set_expected_vk_devices(left, right)


def _check_vulkan_parity(cpu, vk, path=""):
    if path == "history.grad_fns":
        return {}
    if path == "finite_difference" and vk is None:
        return {}
    if isinstance(cpu, dict) and isinstance(vk, dict):
        tensor = {"dtype", "shape", "device", "values", "contiguous", "nonzero_count"}
        if tensor <= cpu.keys() and tensor <= vk.keys():
            if cpu["shape"] != vk["shape"] or cpu["dtype"] != vk["dtype"]:
                _fail(f"{path}: CPU/Vulkan tensor metadata differs")
            actual = torch.tensor(vk["values"], dtype=torch.float32).reshape(cpu["shape"])
            reference = torch.tensor(cpu["values"], dtype=torch.float32).reshape(cpu["shape"])
            try:
                torch.testing.assert_close(actual, reference, rtol=RTOL, atol=ATOL, msg=path)
            except AssertionError as error:
                _fail(f"{path}: Vulkan tensor exceeds source parity tolerance")
            errors = [abs(float(a) - float(b)) for a, b in zip(cpu["values"], vk["values"])]
            return {path: max(errors, default=0.0)}
        if cpu.keys() != vk.keys():
            _fail(f"{path}: CPU/Vulkan nested keysets differ")
        result = {}
        for key in cpu:
            result.update(_check_vulkan_parity(cpu[key], vk[key], f"{path}.{key}" if path else key))
        return result
    if isinstance(cpu, list) and isinstance(vk, list):
        if len(cpu) != len(vk):
            _fail(f"{path}: CPU/Vulkan list lengths differ")
        result = {}
        for index, (left, right) in enumerate(zip(cpu, vk)):
            result.update(_check_vulkan_parity(left, right, f"{path}[{index}]"))
        return result
    if type(cpu) is not type(vk) or cpu != vk:
        _fail(f"{path}: CPU/Vulkan values differ")
    return {}


def _expected_execution(is_hvp):
    phases = [("first_reverse", None), ("second_reverse", None)] if is_hvp else [
        (phase, step)
        for step in (1, 2, 3)
        for phase in ("reset", "forward_backward", "optimizer")
    ]
    return phases


def _validate_execution(record, is_hvp):
    execution = record.get("execution")
    expected = _expected_execution(is_hvp)
    if not isinstance(execution, list) or len(execution) != len(expected):
        _fail(f"{record['workload_id']}: incomplete measured execution windows")
    for item, (phase, step) in zip(execution, expected):
        keys = {"phase", "step", "dispatches", "fallbacks", "explicit_transfers",
                "observation_order", "synchronized_before_snapshot", "snapshot_before_readback"}
        if not isinstance(item, dict) or item.keys() != keys:
            _fail(f"{record['workload_id']}: malformed execution window")
        if type(item["phase"]) is not str or item["phase"] != phase:
            _fail(f"{record['workload_id']}: wrong execution phase/step")
        if step is None:
            if item["step"] is not None:
                _fail(f"{record['workload_id']}: wrong execution phase/step")
        elif type(item["step"]) is not int or item["step"] != step:
            _fail(f"{record['workload_id']}: wrong execution phase/step")
        for key in ("dispatches", "fallbacks", "explicit_transfers"):
            if type(item[key]) is not int or item[key] < 0:
                _fail(f"{record['workload_id']}: invalid {key}")
        if item["fallbacks"] != 0 or item["explicit_transfers"] != 0:
            _fail(f"{record['workload_id']}: fallback/transfer observed")
        if item["observation_order"] != ["sync", "snapshot", "readback"]:
            _fail(f"{record['workload_id']}: invalid observation order")
        if item["synchronized_before_snapshot"] is not True or item["snapshot_before_readback"] is not True:
            _fail(f"{record['workload_id']}: invalid observation ordering flags")
        if phase in ("forward_backward", "first_reverse", "second_reverse"):
            if item["dispatches"] <= 0:
                _fail(f"{record['workload_id']}: required {phase} phase has no measured Vulkan dispatch")


def _validate_record(record, oracle, expected_key):
    workload_id, reset_mode, mode = expected_key
    if not isinstance(record, dict):
        _fail("record must be an object")
    if (record.get("workload_id"), record.get("reset_mode"), record.get("execution_mode")) != expected_key:
        _fail(f"source-owned required workload identity/mode mismatch: {expected_key}")
    expected_record_keys = {"workload_id", "reset_mode", "execution_mode", "device", "hardware", "driver",
                            "extension_sha256", "fixture", "cpu", "vulkan", "comparison", "execution"}
    if record.keys() != expected_record_keys:
        _fail(f"{workload_id}: missing or extra record fields")
    if type(record["reset_mode"]) is not type(reset_mode):
        _fail(f"{workload_id}: reset identity type mismatch")
    if record["device"] != "vk:0":
        _fail(f"{workload_id}: expected original device vk:0")
    for field in ("hardware", "driver"):
        if not isinstance(record[field], str) or not record[field].strip():
            _fail(f"{workload_id}: missing {field}")
    if not isinstance(record["extension_sha256"], str) or re.fullmatch(r"[0-9a-f]{64}", record["extension_sha256"]) is None:
        _fail(f"{workload_id}: invalid extension build SHA-256")
    try:
        _same_json(_workload_module().fixture_metadata(workload_id), record["fixture"], f"{workload_id}: fixture")
    except ValueError as error:
        raise ValueError(f"{workload_id}: source fixture metadata mismatch ({error})") from error
    if not isinstance(record["cpu"], dict) or not isinstance(record["vulkan"], dict):
        _fail(f"{workload_id}: CPU and Vulkan payloads are required")
    _same_json(oracle, record["cpu"], f"{workload_id}: CPU oracle payload")
    matrix_workload = workload_id.startswith("dense.two-linear.") or workload_id.startswith("hvp.dense.two-linear.")
    _validate_tensor_tree(record["cpu"], "cpu", "cpu", require_contiguous=not matrix_workload)
    _validate_tensor_tree(record["vulkan"], "vulkan", "vk:0", require_contiguous=not matrix_workload)
    _set_expected_vk_devices(record["cpu"], record["vulkan"])
    errors = _check_vulkan_parity(record["cpu"], record["vulkan"])
    is_hvp = reset_mode is None
    if is_hvp:
        if record["vulkan"].get("finite_difference", "missing") is not None:
            _fail(f"{workload_id}: Vulkan finite_difference must be null")
        history = record["vulkan"].get("history")
        if not isinstance(history, dict) or history.keys() != {"grad_fns", "first_gradients_require_grad"}:
            _fail(f"{workload_id}: missing HVP history")
        requires_grad = history["first_gradients_require_grad"]
        hvp_names = ("0.weight", "0.bias", "1.weight", "1.bias") if workload_id == "hvp.dense.two-linear.output-energy.parameters" else _HVP_NAMES
        if not isinstance(requires_grad, dict) or requires_grad.keys() != set(hvp_names) or any(
            type(requires_grad[name]) is not bool or requires_grad[name] is not True
            for name in _HVP_NAMES
        ):
            _fail(f"{workload_id}: first-gradient history is detached")
        if not isinstance(history["grad_fns"], dict) or history["grad_fns"].keys() != set(hvp_names) or any(
            not isinstance(value, str) or not value or value == "NoneType"
            for value in history["grad_fns"].values()
        ):
            _fail(f"{workload_id}: invalid first-gradient grad_fn history")
        hvp = record["vulkan"].get("hvp", {})
        if hvp.keys() != set(hvp_names) or any(tensor["nonzero_count"] <= 0 for tensor in hvp.values()):
            _fail(f"{workload_id}: every HVP role must be nonzero")
    else:
        if workload_id.startswith("dense.two-linear."):
            _validate_matrix_sgd_state(record, workload_id, reset_mode)
        else:
            _validate_classifier_state(record, workload_id, reset_mode)
    comparison = record["comparison"]
    if not isinstance(comparison, dict) or comparison.keys() != {"rtol", "atol", "passed", "max_abs_errors"}:
        _fail(f"{workload_id}: malformed comparison report")
    if comparison["rtol"] != RTOL or comparison["atol"] != ATOL or type(comparison["passed"]) is not bool or not comparison["passed"]:
        _fail(f"{workload_id}: comparison tolerances/result do not match source contract")
    declared_errors = comparison["max_abs_errors"]
    if not isinstance(declared_errors, dict) or declared_errors.keys() != errors.keys():
        _fail(f"{workload_id}: comparison.max_abs_errors must cover every paired tensor path")
    for path, expected_error in errors.items():
        actual_error = declared_errors[path]
        if not _is_number(actual_error) or not math.isfinite(float(actual_error)) or float(actual_error) != expected_error:
            _fail(f"{workload_id}: incorrect max_abs_errors for {path}")
    _validate_execution(record, is_hvp)


def _validate_classifier_state(record, workload_id, reset_mode):
    vk = record["vulkan"]
    if vk.get("initial_state_keys") != {}:
        _fail(f"{workload_id}: initial optimizer state must be empty")
    steps = vk.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        _fail(f"{workload_id}: exactly three update steps required")
    names = set(_CLASSIFIER_NAMES)
    previous_momentum = None
    for index, step in enumerate(steps, 1):
        if step.get("index") != index:
            _fail(f"{workload_id}: truncated or misnumbered update")
        if step.get("gradients", {}).keys() != names or step.get("parameters", {}).keys() != names or step.get("momentum", {}).keys() != names:
            _fail(f"{workload_id}: every named parameter needs gradient, parameter and momentum")
        if step.get("state_keys") != {name: ["momentum_buffer"] for name in _CLASSIFIER_NAMES}:
            _fail(f"{workload_id}: SGD state keys differ from source contract")
        if previous_momentum is not None:
            for name in _CLASSIFIER_NAMES:
                grad = torch.tensor(step["gradients"][name]["values"], dtype=torch.float32)
                prev = torch.tensor(previous_momentum[name]["values"], dtype=torch.float32)
                momentum = torch.tensor(step["momentum"][name]["values"], dtype=torch.float32)
                if not torch.allclose(momentum, 0.9 * prev + grad, rtol=RTOL, atol=ATOL):
                    _fail(f"{workload_id}: momentum recurrence failed at step {index} {name}")
        previous_momentum = step["momentum"]
    reset = vk.get("reset_observations")
    if not isinstance(reset, list) or len(reset) != 3:
        _fail(f"{workload_id}: reset observations missing")
    for index, observation in enumerate(reset, 1):
        if observation.get("index") != index:
            _fail(f"{workload_id}: reset identity/index missing")
        if reset_mode == "none" or index == 1:
            if observation.get("all_grad_none") is not True or observation.get("same_grad_objects") is not None:
                _fail(f"{workload_id}: zero_grad(set_to_none=True) reset mismatch")
        else:
            if observation.get("all_grad_none") is not False or observation.get("same_grad_objects") is not True or observation.get("all_grad_zero") is not True:
                _fail(f"{workload_id}: zero_grad(set_to_none=False) identity/reset mismatch")
            gradients = observation.get("gradients")
            if not isinstance(gradients, dict) or gradients.keys() != names:
                _fail(f"{workload_id}: reset gradients missing named parameters")
            if any(value is None or any(float(v) != 0.0 for v in value["values"]) for value in gradients.values()):
                _fail(f"{workload_id}: reset gradients must be present and zero")


def _validate_matrix_sgd_state(record, workload_id, reset_mode):
    vk = record["vulkan"]
    names = ("0.weight", "0.bias", "1.weight", "1.bias")
    steps = vk.get("steps")
    if not isinstance(steps, list) or len(steps) != 3:
        _fail(f"{workload_id}: exactly three update steps required")
    if vk.get("initial_state_keys") != {}:
        _fail(f"{workload_id}: initial optimizer state must be empty")
    previous_momentum = None
    for index, step in enumerate(steps, 1):
        if step.get("index") != index or any(step.get(group, {}).keys() != set(names)
                for group in ("gradients", "parameters", "momentum")):
            _fail(f"{workload_id}: named two-Linear step evidence is incomplete")
        if step.get("state_keys") != {name: ["momentum_buffer"] for name in names}:
            _fail(f"{workload_id}: SGD momentum state mismatch")
        if previous_momentum is not None:
            for name in names:
                grad = torch.tensor(step["gradients"][name]["values"], dtype=torch.float32)
                previous = torch.tensor(previous_momentum[name]["values"], dtype=torch.float32)
                momentum = torch.tensor(step["momentum"][name]["values"], dtype=torch.float32)
                if not torch.allclose(momentum, 0.9 * previous + grad, rtol=RTOL, atol=ATOL):
                    _fail(f"{workload_id}: momentum recurrence failed at step {index} {name}")
        previous_momentum = step["momentum"]
    resets = vk.get("reset_observations")
    if not isinstance(resets, list) or len(resets) != 3:
        _fail(f"{workload_id}: reset observations missing")
    for index, reset in enumerate(resets, 1):
        if reset.get("index") != index:
            _fail(f"{workload_id}: reset index mismatch")
        if reset_mode == "none" or index == 1:
            if reset.get("all_grad_none") is not True:
                _fail(f"{workload_id}: zero_grad(set_to_none=True) mismatch")
        elif reset.get("all_grad_none") is not False or reset.get("same_grad_objects") is not True or reset.get("all_grad_zero") is not True:
            _fail(f"{workload_id}: zero_grad(set_to_none=False) reset mismatch")
        else:
            gradients = reset.get("gradients")
            if not isinstance(gradients, dict) or gradients.keys() != set(names) or any(
                any(float(value) != 0.0 for value in tensor["values"])
                for tensor in gradients.values() if isinstance(tensor, dict)
            ):
                _fail(f"{workload_id}: reset gradients must remain present and zero")


def _workload_module():
    import vulkan_workload_conformance as workload
    return workload


def _oracle_cache():
    installed_torch_metadata()
    workload = _workload_module()
    validate_required_matrix(workload.SCENARIOS)
    return {
        (workload_id, reset_mode): (
            workload.run_cpu_matrix_hvp() if workload_id == "hvp.dense.two-linear.output-energy.parameters" else
            workload.run_cpu_hvp() if reset_mode is None else
            workload.run_cpu_matrix_sgd(reset_mode) if workload_id.startswith("dense.two-linear.") else workload.run_cpu_classifier(reset_mode)
        )
        for workload_id, reset_mode in REQUIRED_WORKLOADS
    }


def _validate_collection(document, expected_mode, oracles):
    if not isinstance(document, dict) or document.keys() != {"schema_version", "torch_version", "torch_git_revision", "records"}:
        _fail("malformed workload collection document")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported workload schema_version")
    if document["torch_version"] != TORCH_VERSION or document["torch_git_revision"] != TORCH_GIT_REVISION:
        _fail("PyTorch version/revision differs from source contract")
    records = document["records"]
    if not isinstance(records, list) or len(records) != len(REQUIRED_WORKLOADS):
        _fail("source-owned required workload collection must contain exactly six records")
    expected = {(workload_id, reset_mode, expected_mode) for workload_id, reset_mode in REQUIRED_WORKLOADS}
    seen = set()
    for record in records:
        key = _record_identity_key(record)
        if key in seen or key not in expected:
            _fail(f"source-owned required workload identity/mode missing or duplicated: {key}")
        seen.add(key)
        _validate_record(record, oracles[(key[0], key[1])], key)
    if seen != expected:
        _fail("source-owned required workload collection is incomplete")
    physical = {(r["hardware"], r["driver"], r["device"]) for r in records}
    builds = {(r["extension_sha256"], r["workload_id"].startswith("dense.two-linear.") or r["workload_id"].startswith("hvp.dense.two-linear.")) for r in records}
    if len(physical) != 1 or len({(digest, family) for digest, family in builds}) > 2 or any(
        len({digest for digest, is_matrix in builds if is_matrix == family}) > 1 for family in (False, True)
    ):
        _fail("workload records mix hardware/device or incoherent within-stage extension builds")
    return document


def validate_collection(document, expected_mode):
    if expected_mode not in REQUIRED_MODES:
        _fail("expected_mode must be async or sync")
    return _validate_collection(document, expected_mode, _oracle_cache())


def validate_fragment(document, expected_mode, selected_workload_ids):
    """Validate a deliberately selected, nonempty per-mode source fragment."""
    if expected_mode not in REQUIRED_MODES or not selected_workload_ids:
        _fail("fragment requires a mode and at least one selected workload")
    if not isinstance(document, dict) or document.keys() != {"schema_version", "torch_version", "torch_git_revision", "records"}:
        _fail("malformed workload fragment")
    if document["schema_version"] != SCHEMA_VERSION or (document["torch_version"], document["torch_git_revision"]) != (TORCH_VERSION, TORCH_GIT_REVISION):
        _fail("fragment PyTorch version/revision differs from source contract")
    workload = _workload_module()
    source_ids = {item.workload_id for item in workload.SCENARIOS}
    if len(set(selected_workload_ids)) != len(selected_workload_ids) or not set(selected_workload_ids) <= source_ids:
        _fail("selected workload IDs must be unique source-owned IDs")
    records = document["records"]
    if not isinstance(records, list):
        _fail("fragment records must be a list")
    fragment_keys = [_record_identity_key(record) for record in records]
    if len(set(fragment_keys)) != len(fragment_keys):
        _fail("fragment contains duplicate identity/mode records")
    if len(records) != len(selected_workload_ids):
        _fail("fragment must contain exactly one record for each selected workload")
    seen = set()
    for record in records:
        key = _record_identity_key(record)
        if key[2] != expected_mode or key[0] not in selected_workload_ids or key in seen:
            _fail("fragment has missing, duplicate, or unselected identity/mode")
        seen.add(key)
        validate_record(record)
    if {key[0] for key in seen} != set(selected_workload_ids):
        _fail("fragment is missing a selected workload identity")
    return document


def validate_historical_stage_g_base(document):
    """Validate the immutable pre-matrix six-record Stage G base."""
    legacy = {
        ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none", "none"),
        ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero", "zero"),
        ("hvp.grouped-depthwise.output-energy.parameters", None),
    }
    if not isinstance(document, dict) or document.keys() != {"schema_version", "torch_version", "torch_git_revision", "records"}:
        _fail("malformed historical Stage G base")
    if document["schema_version"] != SCHEMA_VERSION or (document["torch_version"], document["torch_git_revision"]) != (TORCH_VERSION, TORCH_GIT_REVISION):
        _fail("historical base PyTorch identity mismatch")
    expected = {(wid, reset, mode) for wid, reset in legacy for mode in REQUIRED_MODES}
    records = document["records"]
    if not isinstance(records, list) or len(records) != len(expected):
        _fail("historical Stage G base must contain its exact six records")
    oracles = _oracle_cache(); seen = set()
    for record in records:
        key = _record_identity_key(record)
        if key not in expected or key in seen:
            _fail("historical Stage G base has missing or duplicate identity")
        seen.add(key)
        _validate_record(record, oracles[(key[0], key[1])], key)
    if seen != expected:
        _fail("historical Stage G base is incomplete")
    return document


def validate_record(record):
    """Validate one live runner result against its source-owned CPU oracle."""
    key = _record_identity_key(record)
    installed_torch_metadata()
    workload = _workload_module()
    validate_required_matrix(workload.SCENARIOS)
    literal = {(workload_id, reset_mode, mode) for workload_id, reset_mode in REQUIRED_WORKLOADS for mode in REQUIRED_MODES}
    if key not in literal or key not in workload.REQUIRED_RECORD_KEYS:
        _fail(f"source-owned required workload identity/mode mismatch: {key}")
    if key[0] == "hvp.dense.two-linear.output-energy.parameters": oracle = workload.run_cpu_matrix_hvp()
    elif key[1] is None: oracle = workload.run_cpu_hvp()
    elif key[0].startswith("dense.two-linear."): oracle = workload.run_cpu_matrix_sgd(key[1])
    else: oracle = workload.run_cpu_classifier(key[1])
    _validate_record(record, oracle, key)
    return record


def validate_document(document):
    if not isinstance(document, dict):
        _fail("workload artifact must be a JSON object")
    oracles = _oracle_cache()
    if document.keys() != {"schema_version", "torch_version", "torch_git_revision", "records"}:
        _fail("malformed workload document")
    if type(document["schema_version"]) is not int or document["schema_version"] != SCHEMA_VERSION:
        _fail("unsupported workload schema_version")
    if document["torch_version"] != TORCH_VERSION or document["torch_git_revision"] != TORCH_GIT_REVISION:
        _fail("PyTorch version/revision differs from source contract")
    records = document["records"]
    expected = set(_workload_module().REQUIRED_RECORD_KEYS)
    literal_expected = {(workload_id, reset_mode, mode) for workload_id, reset_mode in REQUIRED_WORKLOADS for mode in REQUIRED_MODES}
    if expected != literal_expected:
        _fail("source-owned required workload matrix differs from literal contract")
    if not isinstance(records, list) or len(records) != len(literal_expected):
        _fail("source-owned required workload document must contain exactly twelve records")
    seen = set()
    for record in records:
        key = _record_identity_key(record)
        if key in seen or key not in literal_expected:
            _fail(f"source-owned required workload identity/mode missing or duplicated: {key}")
        seen.add(key)
        _validate_record(record, oracles[(key[0], key[1])], key)
    if seen != literal_expected:
        _fail("source-owned required workload document is incomplete")
    physical = {(r["hardware"], r["driver"], r["device"]) for r in records}
    by_family = {}
    for record in records:
        matrix = record["workload_id"].startswith("dense.two-linear.") or record["workload_id"].startswith("hvp.dense.two-linear.")
        by_family.setdefault(matrix, set()).add(record["extension_sha256"])
    if len(physical) != 1 or any(len(hashes) != 1 for hashes in by_family.values()):
        _fail("workload records mix hardware/device or incoherent within-stage extension builds")
    return document
