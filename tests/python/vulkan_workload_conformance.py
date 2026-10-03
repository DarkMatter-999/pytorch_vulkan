"""Source-owned CPU oracles and lazily imported live Vulkan workload runners.

Import and CPU fixture/oracle paths do not depend on the Vulkan extension; only
the runtime entry points import it.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
from typing import Iterator, Literal

import torch
import torch.nn as nn


@dataclass(frozen=True)
class ScenarioContract:
    workload_id: str
    reset_mode: str | None
    required_modes: tuple[str, ...]


CLASSIFIER_NONE_ID = "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none"
CLASSIFIER_ZERO_ID = "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero"
HVP_ID = "hvp.grouped-depthwise.output-energy.parameters"
_MODES = ("async", "sync")

SCENARIOS: tuple[ScenarioContract, ...] = (
    ScenarioContract(CLASSIFIER_NONE_ID, "none", _MODES),
    ScenarioContract(CLASSIFIER_ZERO_ID, "zero", _MODES),
    ScenarioContract(HVP_ID, None, _MODES),
)

# Independent immutable authority for required records; do not derive this from
# scenario declarations or a mutable evidence document.
REQUIRED_RECORD_KEYS: frozenset[tuple[str, str | None, str]] = frozenset(
    {
        (CLASSIFIER_NONE_ID, "none", "async"),
        (CLASSIFIER_NONE_ID, "none", "sync"),
        (CLASSIFIER_ZERO_ID, "zero", "async"),
        (CLASSIFIER_ZERO_ID, "zero", "sync"),
        (HVP_ID, None, "async"),
        (HVP_ID, None, "sync"),
    }
)

_CLASSIFIER_NAMES = ("0.weight", "2.weight", "2.bias", "5.weight", "5.bias")
_HVP_NAMES = ("0.weight", "1.weight", "1.bias")
_CLASSIFIER_SHAPES = ((6, 2, 3, 3), (6, 1, 3, 3), (6,), (3, 150), (3,))
_EXPECTED_LOSSES = (1.0897274017333984, 1.0827380418777466, 1.0696792602539062)
_FD_EPSILON = 1e-4
_FD_RTOL = 3e-2
_FD_ATOL = 3e-3
_PARITY_RTOL = 3e-3
_PARITY_ATOL = 3e-3


def _synchronized_observation(synchronize, snapshot, readback):
    """Synchronize and snapshot measured work before observing tensor values."""
    events: list[str] = []
    synchronize()
    events.append("sync")
    counters = snapshot()
    events.append("snapshot")
    payload = readback(counters)
    events.append("readback")
    return counters, payload, events


@contextmanager
def _cpu_oracle_scope() -> Iterator[None]:
    """Isolate oracle work from caller CPU thread and RNG state."""
    prior_threads = torch.get_num_threads()
    prior_rng = torch.random.get_rng_state()
    try:
        torch.set_num_threads(1)
        yield
    finally:
        torch.random.set_rng_state(prior_rng)
        torch.set_num_threads(prior_threads)


def _classifier_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=False),
        nn.ReLU(),
        nn.Conv2d(6, 6, 3, padding=1, groups=6, bias=True),
        nn.ReLU(),
        nn.Flatten(1),
        nn.Linear(150, 3),
    )


def _hvp_model() -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(4, 6, 3, padding=1, groups=2, bias=False),
        nn.Conv2d(6, 6, 3, padding=1, groups=6, bias=True),
    )


def classifier_fixture() -> tuple[nn.Sequential, torch.Tensor, torch.Tensor]:
    """Construct the fixed stock classifier and CPU input from seed 811."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(811)
        model = _classifier_model()
        inputs = torch.randn(2, 4, 5, 5, dtype=torch.float32, device="cpu")
    labels = torch.tensor([0, 2], dtype=torch.int64, device="cpu")
    return model, inputs, labels


def hvp_fixture() -> tuple[nn.Sequential, torch.Tensor, tuple[torch.Tensor, ...]]:
    """Construct the fixed smooth HVP workload and independent CPU directions."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1701)
        model = _hvp_model()
        inputs = torch.randn(2, 4, 5, 5, dtype=torch.float32, device="cpu")
    directions = tuple(
        torch.randn(
            parameter.shape,
            dtype=torch.float32,
            device="cpu",
            generator=torch.Generator(device="cpu").manual_seed(200 + index),
        )
        for index, parameter in enumerate(model.parameters())
    )
    return model, inputs, directions


def fixture_metadata(workload_id: str) -> dict[str, object]:
    """Return the immutable, source-owned recipe metadata for a workload."""
    if workload_id in (CLASSIFIER_NONE_ID, CLASSIFIER_ZERO_ID):
        return {
            "seed": 811,
            "reset_mode": "none" if workload_id == CLASSIFIER_NONE_ID else "zero",
            "model": [
                {"type": "Conv2d", "in_channels": 4, "out_channels": 6, "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 2, "bias": False, "padding_mode": "zeros"},
                {"type": "ReLU", "inplace": False},
                {"type": "Conv2d", "in_channels": 6, "out_channels": 6, "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 6, "bias": True, "padding_mode": "zeros"},
                {"type": "ReLU", "inplace": False},
                {"type": "Flatten", "start_dim": 1, "end_dim": -1},
                {"type": "Linear", "in_features": 150, "out_features": 3, "bias": True},
            ],
            "input": {"shape": [2, 4, 5, 5], "dtype": "torch.float32"},
            "input_requires_grad": False,
            "labels": {"shape": [2], "values": [0, 2], "dtype": "torch.int64"},
            "loss": "nn.CrossEntropyLoss()",
            "loss_options": {"weight": None, "size_average": None, "ignore_index": -100, "reduce": None, "reduction": "mean", "label_smoothing": 0.0},
            "optimizer": {"type": "SGD", "lr": 0.01, "momentum": 0.9, "dampening": 0.0, "weight_decay": 0.0, "nesterov": False, "maximize": False, "foreach": None, "differentiable": False, "fused": None},
            "updates": 3,
            "parameter_names": list(_CLASSIFIER_NAMES),
            "parameter_roles": {
                "0.weight": {"role": "grouped_convolution_weight", "shape": [6, 2, 3, 3]},
                "2.weight": {"role": "depthwise_convolution_weight", "shape": [6, 1, 3, 3]},
                "2.bias": {"role": "depthwise_convolution_bias", "shape": [6]},
                "5.weight": {"role": "classifier_weight", "shape": [3, 150]},
                "5.bias": {"role": "classifier_bias", "shape": [3]},
            },
        }
    if workload_id == HVP_ID:
        return {
            "seed": 1701,
            "model": [
                {"type": "Conv2d", "in_channels": 4, "out_channels": 6, "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 2, "bias": False, "padding_mode": "zeros"},
                {"type": "Conv2d", "in_channels": 6, "out_channels": 6, "kernel_size": [3, 3], "stride": [1, 1], "padding": [1, 1], "dilation": [1, 1], "groups": 6, "bias": True, "padding_mode": "zeros"},
            ],
            "input": {"shape": [2, 4, 5, 5], "dtype": "torch.float32"},
            "input_requires_grad": False,
            "direction_seeds": [200, 201, 202],
            "objective": "(y * y).sum()",
            "parameter_names": list(_HVP_NAMES),
            "parameter_roles": {
                "0.weight": {"role": "grouped_convolution_weight", "shape": [6, 2, 3, 3]},
                "1.weight": {"role": "depthwise_convolution_weight", "shape": [6, 1, 3, 3]},
                "1.bias": {"role": "depthwise_convolution_bias", "shape": [6]},
            },
        }
    raise ValueError(f"unknown workload_id: {workload_id}")


def _tensor_payload(value: torch.Tensor) -> dict[str, object]:
    snapshot = value.detach().to(device="cpu").contiguous().clone()
    assert snapshot.dtype == torch.float32
    assert bool(torch.isfinite(snapshot).all())
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "device": str(value.device),
        "contiguous": value.is_contiguous(),
        "values": snapshot.reshape(-1).tolist(),
        "nonzero_count": int(torch.count_nonzero(snapshot).item()),
    }


def _named_payloads(names: tuple[str, ...], values: tuple[torch.Tensor, ...] | list[torch.Tensor]) -> dict[str, object]:
    return {name: _tensor_payload(value) for name, value in zip(names, values)}


def _assert_tensor_contract(value: torch.Tensor, shape: tuple[int, ...]) -> None:
    assert tuple(value.shape) == shape
    assert value.dtype == torch.float32
    assert value.device.type == "cpu"
    assert value.is_contiguous()
    assert bool(torch.isfinite(value).all())


def run_cpu_classifier(reset_mode: Literal["none", "zero"]) -> dict[str, object]:
    """Execute the three-update stock CE/default-SGD CPU reference."""
    if reset_mode not in ("none", "zero"):
        raise ValueError("reset_mode must be 'none' or 'zero'")
    with _cpu_oracle_scope():
        model, inputs, labels = classifier_fixture()
        names = tuple(name for name, _ in model.named_parameters())
        assert names == _CLASSIFIER_NAMES
        parameters = dict(model.named_parameters())
        shapes = tuple(tuple(parameters[name].shape) for name in names)
        assert shapes == _CLASSIFIER_SHAPES
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        loss_fn = nn.CrossEntropyLoss()
        initial_parameters = _named_payloads(names, [parameters[name] for name in names])
        initial_state_keys: dict[str, object] = {}
        assert len(optimizer.state) == 0
        steps: list[dict[str, object]] = []
        reset_observations: list[dict[str, object]] = []

        for index in range(1, 4):
            before = tuple(parameter.grad for parameter in parameters.values())
            before_present = tuple(grad is not None for grad in before)
            optimizer.zero_grad(set_to_none=(reset_mode == "none"))
            after = tuple(parameter.grad for parameter in parameters.values())
            all_none = all(grad is None for grad in after)
            if reset_mode == "none" or index == 1:
                same_objects: bool | None = None
                all_zero: bool | None = None if all_none else False
            else:
                assert all(before_present)
                same_objects = all(old is new for old, new in zip(before, after))
                assert same_objects
                assert all(grad is not None for grad in after)
                all_zero = all(not bool(torch.count_nonzero(grad).item()) for grad in after)
                assert all_zero
            if reset_mode == "none":
                assert all_none
            elif index == 1:
                assert all_none
            reset_observations.append(
                {
                    "index": index,
                    "all_grad_none": all_none,
                    "same_grad_objects": same_objects,
                    "all_grad_zero": all_zero,
                    "gradients": {
                        name: None if parameter.grad is None else _tensor_payload(parameter.grad)
                        for name, parameter in parameters.items()
                    },
                }
            )

            logits = model(inputs)
            loss = loss_fn(logits, labels)
            loss.backward()
            gradients = tuple(parameter.grad for parameter in parameters.values())
            assert all(gradient is not None for gradient in gradients)
            gradient_values = tuple(gradient.detach().clone() for gradient in gradients)
            for gradient, shape in zip(gradient_values, shapes):
                _assert_tensor_contract(gradient, shape)

            optimizer.step()
            parameter_values = tuple(parameter.detach().clone() for parameter in parameters.values())
            momentum_values = tuple(
                optimizer.state[parameter]["momentum_buffer"].detach().clone()
                for parameter in parameters.values()
            )
            for value, shape in zip(parameter_values + momentum_values, shapes + shapes):
                _assert_tensor_contract(value, shape)
            state_keys = {
                name: sorted(optimizer.state[parameter].keys())
                for name, parameter in parameters.items()
            }
            assert all(keys == ["momentum_buffer"] for keys in state_keys.values())
            steps.append(
                {
                    "index": index,
                    "output": _tensor_payload(logits),
                    "loss": _tensor_payload(loss.reshape(())),
                    "gradients": _named_payloads(names, gradient_values),
                    "parameters": _named_payloads(names, parameter_values),
                    "momentum": _named_payloads(names, momentum_values),
                    "state_keys": state_keys,
                }
            )
            if index == 1:
                for gradient, momentum in zip(gradient_values, momentum_values):
                    torch.testing.assert_close(momentum, gradient, rtol=0, atol=0)
            else:
                previous = steps[index - 2]["momentum"]
                for name, momentum in zip(names, momentum_values):
                    previous_values = torch.tensor(previous[name]["values"], dtype=torch.float32).reshape(previous[name]["shape"])
                    torch.testing.assert_close(momentum, 0.9 * previous_values + gradient_values[names.index(name)])
        assert len(steps) == 3
        for step, expected_loss in zip(steps, _EXPECTED_LOSSES):
            assert abs(step["loss"]["values"][0] - expected_loss) <= 1e-7
        return {
            "steps": steps,
            "reset_observations": reset_observations,
            "initial_parameters": initial_parameters,
            "initial_state_keys": initial_state_keys,
        }


def run_cpu_hvp() -> dict[str, object]:
    """Execute the smooth output-energy parameter HVP and CPU FD oracle."""
    with _cpu_oracle_scope():
        model, inputs, directions = hvp_fixture()
        named_parameters = tuple(model.named_parameters())
        names = tuple(name for name, _ in named_parameters)
        parameters = tuple(parameter for _, parameter in named_parameters)
        assert names == _HVP_NAMES
        shapes = tuple(tuple(parameter.shape) for parameter in parameters)
        assert shapes == ((6, 2, 3, 3), (6, 1, 3, 3), (6,))

        def output_energy() -> tuple[torch.Tensor, torch.Tensor]:
            output_value = model(inputs)
            return output_value, (output_value * output_value).sum()

        output, loss = output_energy()
        first_gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        for gradient, shape in zip(first_gradients, shapes):
            _assert_tensor_contract(gradient, shape)
            assert gradient.requires_grad and gradient.grad_fn is not None
        contraction = sum((gradient * direction).sum() for gradient, direction in zip(first_gradients, directions))
        hvp_values = torch.autograd.grad(contraction, parameters)
        for value, shape in zip(hvp_values, shapes):
            _assert_tensor_contract(value, shape)
            assert int(torch.count_nonzero(value).item()) > 0

        originals = tuple(parameter.detach().clone() for parameter in parameters)
        estimates: tuple[torch.Tensor, ...]
        try:
            with torch.no_grad():
                for parameter, original, direction in zip(parameters, originals, directions):
                    parameter.copy_(original + _FD_EPSILON * direction)
            _, plus_loss = output_energy()
            plus_gradients = torch.autograd.grad(plus_loss, parameters)
            with torch.no_grad():
                for parameter, original, direction in zip(parameters, originals, directions):
                    parameter.copy_(original - _FD_EPSILON * direction)
            _, minus_loss = output_energy()
            minus_gradients = torch.autograd.grad(minus_loss, parameters)
            estimates = tuple((plus - minus) / (2 * _FD_EPSILON) for plus, minus in zip(plus_gradients, minus_gradients))
            for hvp_value, estimate, shape in zip(hvp_values, estimates, shapes):
                _assert_tensor_contract(estimate, shape)
                torch.testing.assert_close(hvp_value, estimate, rtol=_FD_RTOL, atol=_FD_ATOL)
        finally:
            with torch.no_grad():
                for parameter, original in zip(parameters, originals):
                    parameter.copy_(original)

        history = {
            "grad_fns": {name: type(gradient.grad_fn).__name__ for name, gradient in zip(names, first_gradients)},
            "first_gradients_require_grad": {name: bool(gradient.requires_grad) for name, gradient in zip(names, first_gradients)},
        }
        return {
            "output": _tensor_payload(output),
            "loss": _tensor_payload(loss.reshape(())),
            "first_gradients": _named_payloads(names, first_gradients),
            "hvp": _named_payloads(names, hvp_values),
            "history": history,
            "finite_difference": {
                "checked": True,
                "epsilon": _FD_EPSILON,
                "rtol": _FD_RTOL,
                "atol": _FD_ATOL,
                "values": _named_payloads(names, estimates),
                "max_abs_errors": {
                    name: float((value - estimate).abs().max().item())
                    for name, value, estimate in zip(names, hvp_values, estimates)
                },
            },
        }


def _parse_vulkan_device_summary(info: str) -> dict[str, str]:
    """Parse hardware identity only when the summary has one complete GPU block."""
    lines = info.splitlines()
    device_headers = [index for index, line in enumerate(lines) if line.strip() == "Devices:"]
    if len(device_headers) != 1:
        raise RuntimeError("vulkaninfo summary must contain exactly one Devices section")

    gpu_header = re.compile(r"^\s*GPU\d+:\s*$")
    field_line = re.compile(r"^\s*(deviceName|driverName|driverInfo)\s*=\s*(.*?)\s*$")
    start = device_headers[0] + 1
    gpu_starts = [index for index in range(start, len(lines)) if gpu_header.fullmatch(lines[index])]
    if len(gpu_starts) != 1:
        raise RuntimeError("vulkaninfo summary must contain exactly one GPU device block")

    fields: dict[str, list[str]] = {name: [] for name in ("deviceName", "driverName", "driverInfo")}
    for line in lines[gpu_starts[0] + 1 :]:
        match = field_line.fullmatch(line)
        if match is not None:
            fields[match.group(1)].append(match.group(2).strip())
    for name, values in fields.items():
        if len(values) != 1 or not values[0]:
            raise RuntimeError(f"vulkaninfo GPU device block must contain exactly one nonempty {name}")
    return {
        "hardware": fields["deviceName"][0],
        "driver": f"{fields['driverName'][0]} {fields['driverInfo'][0]}",
    }


def _runtime_context():
    # Import only on the live runtime path; CPU oracle/test selection stays usable
    # without importing the Vulkan extension.
    import pytorch_vulkan

    extension = Path(pytorch_vulkan._C.__file__).resolve()
    binary_hash = hashlib.sha256(extension.read_bytes()).hexdigest()
    info = subprocess.run(
        ["vulkaninfo", "--summary"], check=True, capture_output=True, text=True
    ).stdout

    metadata = _parse_vulkan_device_summary(info)

    return pytorch_vulkan, {
        "execution_mode": pytorch_vulkan._C.execution_mode(),
        "device": "vk:0",
        "hardware": metadata["hardware"],
        "driver": metadata["driver"],
        "extension_sha256": binary_hash,
    }


def _capture_window(api, phase: str, step: int | None, run_operation, observe):
    api._C.reset_execution_counters()
    operation_result = run_operation()
    counters, observed, observation_order = _synchronized_observation(
        api._C.synchronize,
        api._C.execution_counter_snapshot,
        lambda snapshot: observe(operation_result),
    )
    dispatches, _, transfers, fallbacks = counters
    if transfers != 0 or fallbacks != 0:
        raise AssertionError(
            f"{phase} step {step}: transfers={transfers}, fallbacks={fallbacks}"
        )
    return observed, {
        "phase": phase,
        "step": step,
        "dispatches": dispatches,
        "fallbacks": fallbacks,
        "explicit_transfers": transfers,
        "observation_order": observation_order,
        "synchronized_before_snapshot": observation_order.index("sync") < observation_order.index("snapshot"),
        "snapshot_before_readback": observation_order.index("snapshot") < observation_order.index("readback"),
    }


def _compare_payloads(cpu_value, vk_value, path: str, errors: dict[str, float]):
    if isinstance(cpu_value, dict) and isinstance(vk_value, dict):
        if "values" in cpu_value and "values" in vk_value:
            cpu = torch.tensor(cpu_value["values"], dtype=torch.float32).reshape(cpu_value["shape"])
            vk = torch.tensor(vk_value["values"], dtype=torch.float32).reshape(vk_value["shape"])
            torch.testing.assert_close(vk, cpu, rtol=_PARITY_RTOL, atol=_PARITY_ATOL, msg=path)
            errors[path] = float((vk - cpu).abs().max().item())
            return
        for key in cpu_value.keys() & vk_value.keys():
            _compare_payloads(cpu_value[key], vk_value[key], f"{path}.{key}" if path else key, errors)
    elif isinstance(cpu_value, list) and isinstance(vk_value, list):
        for index, (cpu_item, vk_item) in enumerate(zip(cpu_value, vk_value)):
            _compare_payloads(cpu_item, vk_item, f"{path}[{index}]", errors)


def _comparison(cpu_payload, vk_payload):
    errors: dict[str, float] = {}
    _compare_payloads(cpu_payload, vk_payload, "", errors)
    return {
        "rtol": _PARITY_RTOL,
        "atol": _PARITY_ATOL,
        "passed": True,
        "max_abs_errors": errors,
    }


def run_vulkan_classifier(reset_mode: Literal["none", "zero"]) -> dict[str, object]:
    """Execute the fixed classifier through public Module/loss/optimizer APIs."""
    if reset_mode not in ("none", "zero"):
        raise ValueError("reset_mode must be 'none' or 'zero'")
    api, runtime = _runtime_context()
    workload_id = CLASSIFIER_NONE_ID if reset_mode == "none" else CLASSIFIER_ZERO_ID
    cpu = run_cpu_classifier(reset_mode)
    model, inputs, labels = classifier_fixture()
    model = model.to("vk:0")
    vk_inputs = inputs.to("vk:0")
    vk_labels = labels.to("vk:0")
    names = tuple(name for name, _ in model.named_parameters())
    assert names == _CLASSIFIER_NAMES
    parameters = dict(model.named_parameters())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()
    assert len(optimizer.state) == 0
    initial_state_keys = {}
    api._C.synchronize()
    initial = _named_payloads(names, [parameters[name] for name in names])

    steps: list[dict[str, object]] = []
    resets: list[dict[str, object]] = []
    execution: list[dict[str, object]] = []
    for index in range(1, 4):
        before = tuple(parameter.grad for parameter in parameters.values())
        reset_payload, window = _capture_window(
            api, "reset", index,
            lambda: optimizer.zero_grad(set_to_none=(reset_mode == "none")),
            lambda _: {
                "all_grad_none": all(parameter.grad is None for parameter in parameters.values()),
                "same_grad_objects": (
                    None if reset_mode == "none" or index == 1 else
                    all(old is new for old, new in zip(before, (parameter.grad for parameter in parameters.values())))
                ),
                "all_grad_zero": (
                    None if all(parameter.grad is None for parameter in parameters.values()) else
                    all(_tensor_payload(parameter.grad)["nonzero_count"] == 0 for parameter in parameters.values())
                ),
                "gradients": {
                    name: None if parameter.grad is None else _tensor_payload(parameter.grad)
                    for name, parameter in parameters.items()
                },
            },
        )
        execution.append(window)
        assert reset_payload["all_grad_none"] == (reset_mode == "none" or index == 1)
        if reset_mode == "zero" and index > 1:
            assert reset_payload["same_grad_objects"] and reset_payload["all_grad_zero"]
        resets.append({"index": index, **reset_payload})

        def forward_backward():
            logits_value = model(vk_inputs)
            loss_value = loss_fn(logits_value, vk_labels)
            loss_value.backward()
            return logits_value, loss_value

        measured_payload, observed = _capture_window(
            api, "forward_backward", index, forward_backward,
            lambda result: {
                "output": _tensor_payload(result[0]),
                "loss": _tensor_payload(result[1].reshape(())),
                "gradients": _named_payloads(names, [parameters[name].grad for name in names]),
            },
        )
        execution.append(observed)
        assert observed["dispatches"] > 0

        def optimizer_step():
            optimizer.step()
            return None

        parameter_state, observed = _capture_window(
            api, "optimizer", index, optimizer_step,
            lambda _: {
                "parameters": _named_payloads(names, [parameters[name] for name in names]),
                "momentum": _named_payloads(names, [optimizer.state[parameters[name]]["momentum_buffer"] for name in names]),
                "state_keys": {name: sorted(optimizer.state[parameters[name]]) for name in names},
            },
        )
        execution.append(observed)
        steps.append({
            "index": index,
            "output": measured_payload["output"],
            "loss": measured_payload["loss"],
            "gradients": measured_payload["gradients"],
            "parameters": parameter_state["parameters"],
            "momentum": parameter_state["momentum"],
            "state_keys": parameter_state["state_keys"],
        })
    vk_payload = {
        "steps": steps,
        "reset_observations": resets,
        "initial_parameters": initial,
        "initial_state_keys": initial_state_keys,
    }
    comparison = _comparison(cpu, vk_payload)
    for vk_step in steps:
        assert all(keys == ["momentum_buffer"] for keys in vk_step["state_keys"].values())
        for group in ("gradients", "parameters", "momentum"):
            for name in names:
                payload = vk_step[group][name]
                assert payload["device"] == "vk:0" and payload["dtype"] == "torch.float32"
                assert payload["shape"] == list(parameters[name].shape) and payload["contiguous"]
        assert vk_step["output"]["device"] == "vk:0"
        assert vk_step["loss"]["device"] == "vk:0"
        assert vk_step["output"]["dtype"] == "torch.float32"
        assert vk_step["output"]["shape"] == [2, 3] and vk_step["output"]["contiguous"]
        assert vk_step["loss"]["dtype"] == "torch.float32"
        assert vk_step["loss"]["shape"] == [] and vk_step["loss"]["contiguous"]
    return {
        "workload_id": workload_id,
        "reset_mode": reset_mode,
        **runtime,
        "fixture": fixture_metadata(workload_id),
        "cpu": cpu,
        "vulkan": vk_payload,
        "comparison": comparison,
        "execution": execution,
    }


def run_vulkan_hvp() -> dict[str, object]:
    """Execute the smooth parameter HVP on Vulkan while preserving grad history."""
    api, runtime = _runtime_context()
    cpu = run_cpu_hvp()
    model, inputs, directions = hvp_fixture()
    model = model.to("vk:0")
    vk_inputs = inputs.to("vk:0")
    vk_directions = tuple(direction.to("vk:0") for direction in directions)
    named_parameters = tuple(model.named_parameters())
    names = tuple(name for name, _ in named_parameters)
    parameters = tuple(parameter for _, parameter in named_parameters)
    assert names == _HVP_NAMES
    api._C.synchronize()

    retained: dict[str, object] = {}

    def forward_first_reverse():
        output = model(vk_inputs)
        loss = (output * output).sum()
        first_gradients = torch.autograd.grad(loss, parameters, create_graph=True)
        retained.update(output=output, loss=loss, first_gradients=first_gradients)
        return output, loss, first_gradients

    first_observation, first_window = _capture_window(
        api, "first_reverse", None, forward_first_reverse,
        lambda result: {
            "output": _tensor_payload(result[0]),
            "loss": _tensor_payload(result[1].reshape(())),
            "first_gradients": _named_payloads(names, result[2]),
            "history": {
                "grad_fns": {name: type(value.grad_fn).__name__ for name, value in zip(names, result[2])},
                "first_gradients_require_grad": {name: bool(value.requires_grad) for name, value in zip(names, result[2])},
            },
        },
    )
    assert first_window["dispatches"] > 0
    output = retained["output"]
    loss = retained["loss"]
    first_gradients = retained["first_gradients"]
    assert isinstance(output, torch.Tensor) and isinstance(loss, torch.Tensor)
    assert isinstance(first_gradients, tuple)
    def second_reverse():
        contraction = sum(
            (gradient * direction).sum()
            for gradient, direction in zip(first_gradients, vk_directions)
        )
        values = torch.autograd.grad(contraction, parameters)
        retained["hvp"] = values
        return values

    hvp_payload, second_window = _capture_window(
        api, "second_reverse", None,
        second_reverse,
        lambda values: _named_payloads(names, values),
    )
    assert second_window["dispatches"] > 0
    hvp_values = retained["hvp"]
    assert isinstance(hvp_values, tuple)
    for gradient, parameter in zip(first_gradients, parameters):
        assert gradient.requires_grad and gradient.grad_fn is not None
        assert tuple(gradient.shape) == tuple(parameter.shape)
        assert gradient.device == torch.device("vk:0") and gradient.dtype == torch.float32
        assert gradient.is_contiguous()
    assert first_observation["output"]["device"] == "vk:0"
    assert first_observation["output"]["dtype"] == "torch.float32"
    assert first_observation["output"]["shape"] == [2, 6, 5, 5]
    assert first_observation["output"]["contiguous"]
    assert first_observation["loss"]["device"] == "vk:0"
    assert first_observation["loss"]["dtype"] == "torch.float32"
    assert first_observation["loss"]["shape"] == []
    assert first_observation["loss"]["contiguous"]
    for name, parameter in zip(names, parameters):
        gradient_payload = first_observation["first_gradients"][name]
        assert gradient_payload["device"] == "vk:0" and gradient_payload["dtype"] == "torch.float32"
        assert gradient_payload["shape"] == list(parameter.shape) and gradient_payload["contiguous"]
    for value, parameter in zip(hvp_values, parameters):
        assert tuple(value.shape) == tuple(parameter.shape)
        assert value.device == torch.device("vk:0") and value.dtype == torch.float32
        assert value.is_contiguous()
    for name, parameter in zip(names, parameters):
        value_payload = hvp_payload[name]
        assert value_payload["device"] == "vk:0" and value_payload["dtype"] == "torch.float32"
        assert value_payload["shape"] == list(parameter.shape) and value_payload["contiguous"]
    vk_payload = {
        "output": first_observation["output"],
        "loss": first_observation["loss"],
        "first_gradients": first_observation["first_gradients"],
        "hvp": hvp_payload,
        "history": first_observation["history"],
        "finite_difference": None,
    }
    comparison = _comparison(cpu, vk_payload)
    assert all(payload["nonzero_count"] > 0 for payload in hvp_payload.values())
    execution = [first_window, second_window]
    return {
        "workload_id": HVP_ID,
        "reset_mode": None,
        **runtime,
        "fixture": fixture_metadata(HVP_ID),
        "cpu": cpu,
        "vulkan": vk_payload,
        "comparison": comparison,
        "execution": execution,
    }


def validate_workload_record(record):
    """Lazily validate one executed record without changing CPU-only imports."""
    try:
        from validate_vulkan_workload_contract import validate_record
    except ImportError:
        from tools.validate_vulkan_workload_contract import validate_record
    return validate_record(record)
