#!/usr/bin/env python3
"""Deterministic bounded model fixtures and CPU/Vulkan benchmark artifacts."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import torch


SEED = 1729
SCHEMA_VERSION = 1
# Means may differ from a serialized sample average by one nanosecond or one
# part per million, whichever is larger, to allow integer rounding.
TIMING_MEAN_REL_TOLERANCE = 1e-6
TIMING_MEAN_ABS_TOLERANCE_NS = 1.0
MODEL_PARITY_ABS_TOLERANCE = 3e-3


class _Attention(torch.nn.Module):
    def __init__(self, embed_dim: int, batch: int):
        super().__init__()
        self.query = torch.nn.Linear(embed_dim, embed_dim)
        self.key = torch.nn.Linear(embed_dim, embed_dim)
        self.value = torch.nn.Linear(embed_dim, embed_dim)
        self.output = torch.nn.Linear(embed_dim, embed_dim)
        self.scale = math.sqrt(embed_dim)
        expected = torch.zeros((batch, 128, 128), dtype=torch.float32)
        expected[:, torch.triu(torch.ones((128, 128), dtype=torch.bool), diagonal=1)] = -10000.0
        self.register_buffer("causal_mask", expected, persistent=False)

    def forward(self, inputs, mask=None):
        if inputs.dtype is not torch.float32:
            raise RuntimeError("attention requires float32 inputs")
        if inputs.dim() != 3 or inputs.shape[1:] != (128, 256):
            raise RuntimeError("attention input has unsupported shape")
        if not inputs.is_contiguous():
            raise RuntimeError("attention requires contiguous input layout")
        batch, sequence, embed = inputs.shape
        if mask is not None:
            if mask.device != inputs.device or mask.dtype is not torch.float32:
                raise RuntimeError("attention mask requires matching float32 device")
            if mask.shape != (batch, sequence, sequence):
                raise RuntimeError("attention mask has unsupported shape")
            if not mask.is_contiguous():
                raise RuntimeError("attention mask requires contiguous layout")
            if mask is not self.causal_mask:
                raise RuntimeError("attention mask must be the registered causal mask identity")
        query = self.query(inputs)
        key = self.key(inputs)
        value = self.value(inputs)
        scores = torch.bmm(query, key.transpose(1, 2)) / self.scale
        if mask is not None:
            scores = scores + mask
        weights = torch.softmax(scores, dim=-1)
        attended = torch.bmm(weights, value)
        output = self.output(attended)
        self.last_intermediates = {
            "query": query,
            "key": key,
            "value": value,
            "scores": scores,
            "weights": weights,
            "attended": attended,
            "output_projection": output,
        }
        return output


class _VanillaRNN(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.input = torch.nn.Linear(input_dim, hidden_dim)
        self.hidden = torch.nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, inputs):
        if (inputs.device.type == "privateuseone" and inputs.device.index != 0) or inputs.dim() != 3:
            raise RuntimeError("RNN input has unsupported device or shape")
        if inputs.dtype is not torch.float32:
            raise RuntimeError("RNN requires float32 inputs")
        if inputs.shape[0] < 1 or inputs.shape[0] > 16:
            raise RuntimeError("RNN batch must be between 1 and 16")
        if inputs.shape[2] != self.input.in_features or inputs.shape[1] < 1 or inputs.shape[1] > 64:
            raise RuntimeError("RNN input has unsupported sequence or feature shape")
        if not inputs.is_contiguous():
            raise RuntimeError("RNN requires contiguous input layout")
        state = torch.zeros(
            inputs.shape[0], self.hidden.out_features, dtype=inputs.dtype, device=inputs.device
        )
        outputs = []
        self.last_states = []
        for step in inputs.transpose(0, 1).contiguous().unbind(dim=0):
            step = step.reshape(inputs.shape[0], self.input.in_features)
            state = torch.tanh(self.input(step) + self.hidden(state))
            if state.requires_grad:
                state.retain_grad()
            outputs.append(state)
            self.last_states.append(state)
        return torch.stack(outputs, dim=1)


@dataclass
class _Fixture:
    batch: int
    dtype: torch.dtype = torch.float32
    seed: int = SEED

    def __post_init__(self):
        if self.dtype is not torch.float32:
            raise ValueError("only float32 fixtures are supported")
        if self.batch <= 0:
            raise ValueError("batch must be greater than zero")

    @property
    def shape(self):
        return (self.batch, *self._input_shape)

    def make_inputs(self):
        torch.manual_seed(self.seed + 1)
        return (
            torch.randn(self.shape, dtype=self.dtype),
            torch.randn(self.target_shape, dtype=self.dtype),
        )

    def trace(self):
        model = self.make_cpu()
        inputs, target = self.make_inputs()
        with torch.autograd.profiler.profile(use_cpu=True) as profiler:
            mask = model.causal_mask if hasattr(model, "causal_mask") else None
            output = model(inputs, mask) if mask is not None else model(inputs)
            self.loss(output, target).backward()
        names = {event.key for event in profiler.key_averages()}
        expected = self.expected_schema_set()
        return {
            "forward": sorted(names & expected["forward"]),
            "backward": sorted(names & expected["backward"]),
            "operators": sorted(names),
        }


class CNNFixture(_Fixture):
    channels: int = 3
    height: int = 32
    width: int = 32
    classes: int = 10

    def __init__(self, batch=8, channels=3, height=32, width=32, dtype=torch.float32, seed=SEED):
        super().__init__(batch, dtype, seed)
        if (channels, height, width) != (3, 32, 32):
            raise ValueError("CNN dimensions must be channels=3, height=32, width=32")
        self.channels, self.height, self.width = channels, height, width
        self._input_shape = (channels, height, width)
        self.target_shape = (batch, self.classes)

    def make_cpu(self):
        torch.manual_seed(self.seed)
        return torch.nn.Sequential(
            torch.nn.Conv2d(self.channels, 8, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(8 * self.height * self.width, self.classes),
        ).to(dtype=self.dtype)

    def loss(self, output, target):
        return torch.nn.functional.mse_loss(output, target)

    def expected_schema_set(self):
        return {
            "forward": {"aten::conv2d", "aten::relu", "aten::flatten", "aten::linear"},
            "backward": {"aten::convolution_backward", "aten::threshold_backward", "aten::mm"},
        }

    def expected_parameter_shapes(self):
        return {
            "0.weight": (8, 3, 3, 3),
            "0.bias": (8,),
            "3.weight": (10, 8 * 32 * 32),
            "3.bias": (10,),
        }

    def arithmetic_operations(self):
        return (2 * self.batch * self.height * self.width * self.channels * 9 * 8
                + 2 * self.batch * (8 * self.height * self.width) * self.classes)


class AttentionFixture(_Fixture):
    def __init__(self, batch=4, sequence_length=128, embed_dim=256, dtype=torch.float32, seed=SEED):
        super().__init__(batch, dtype, seed)
        if sequence_length != 128:
            raise ValueError("sequence_length must be 128")
        if embed_dim != 256:
            raise ValueError("embed_dim must be 256")
        self.sequence_length, self.embed_dim = sequence_length, embed_dim
        self._input_shape = (sequence_length, embed_dim)
        self.target_shape = (batch, sequence_length, embed_dim)
        self.mask_shape = (batch, sequence_length, sequence_length)

    def make_mask(self, device=None):
        mask = torch.zeros(self.mask_shape, dtype=self.dtype, device=device)
        return mask.masked_fill(
            torch.triu(torch.ones(self.mask_shape, dtype=torch.bool, device=device), diagonal=1),
            -10000.0,
        )

    def make_cpu(self):
        torch.manual_seed(self.seed)
        return _Attention(self.embed_dim, self.batch).to(dtype=self.dtype)

    def loss(self, output, target):
        return torch.nn.functional.mse_loss(output, target)

    def expected_schema_set(self):
        return {
            "forward": {"aten::linear", "aten::bmm", "aten::transpose", "aten::softmax", "aten::add"},
            "backward": {"aten::mm", "aten::_softmax_backward_data"},
        }

    def expected_parameter_shapes(self):
        return {
            "query.weight": (256, 256),
            "query.bias": (256,),
            "key.weight": (256, 256),
            "key.bias": (256,),
            "value.weight": (256, 256),
            "value.bias": (256,),
            "output.weight": (256, 256),
            "output.bias": (256,),
        }

    def arithmetic_operations(self):
        linear = 2 * self.batch * self.sequence_length * self.embed_dim * self.embed_dim
        bmm = 2 * self.batch * self.sequence_length * self.sequence_length * self.embed_dim
        return 4 * linear + 2 * bmm


class RNNFixture(_Fixture):
    def __init__(self, batch=16, sequence_length=64, input_dim=128, hidden_dim=256, dtype=torch.float32, seed=SEED):
        super().__init__(batch, dtype, seed)
        if batch > 16:
            raise ValueError("batch must be between 1 and 16")
        if sequence_length < 1 or sequence_length > 64:
            raise ValueError("sequence_length must be between 1 and 64")
        if input_dim != 128:
            raise ValueError("input_dim must be 128")
        if hidden_dim != 256:
            raise ValueError("hidden_dim must be 256")
        self.sequence_length, self.input_dim, self.hidden_dim = sequence_length, input_dim, hidden_dim
        self._input_shape = (sequence_length, input_dim)
        self.target_shape = (batch, sequence_length, hidden_dim)

    def make_cpu(self):
        torch.manual_seed(self.seed)
        return _VanillaRNN(self.input_dim, self.hidden_dim).to(dtype=self.dtype)

    def loss(self, output, target):
        return torch.nn.functional.mse_loss(output, target)

    def expected_schema_set(self):
        return {
            "forward": {"aten::linear", "aten::tanh", "aten::stack"},
            "backward": {"aten::mm", "aten::tanh_backward"},
        }

    def expected_parameter_shapes(self):
        return {
            "input.weight": (256, 128),
            "input.bias": (256,),
            "hidden.weight": (256, 256),
        }

    def arithmetic_operations(self):
        return self.sequence_length * (
            2 * self.batch * self.input_dim * self.hidden_dim
            + 2 * self.batch * self.hidden_dim * self.hidden_dim
        )


FIXTURES = {"cnn": CNNFixture, "attention": AttentionFixture, "rnn": RNNFixture}


def _counter_fields(device):
    if device == "cpu":
        return {
            "dispatches": 0,
            "vulkan_copies": 0,
            "explicit_transfers": 0,
            "fallbacks": 0,
            "submissions": 0,
            "completions": 0,
            "waits": 0,
            "transfer_operations": 0,
            "transfer_submissions": 0,
            "transfer_completions": 0,
            "transfer_waits": 0,
            "compute_submissions": 0,
            "fallback_status": False,
        }
    import pytorch_vulkan

    counters = pytorch_vulkan._C.execution_counter_snapshot()
    return {
        "dispatches": counters[0],
        "vulkan_copies": counters[1],
        "explicit_transfers": counters[2],
        "fallbacks": counters[3],
        "submissions": pytorch_vulkan._C.compute_submitted_count(),
        "completions": pytorch_vulkan._C.compute_completed_count(),
        "waits": pytorch_vulkan._C.compute_wait_count(),
        "transfer_operations": pytorch_vulkan._C.transfer_operation_count(),
        "transfer_submissions": pytorch_vulkan._C.transfer_submission_count(),
        "transfer_completions": pytorch_vulkan._C.transfer_completion_count(),
        "transfer_waits": pytorch_vulkan._C.transfer_wait_count(),
        "compute_submissions": pytorch_vulkan._C.compute_submitted_count(),
        "fallback_status": counters[3] != 0,
    }


def _move_model(model, device):
    if device == "cpu":
        return model
    runtime_device = "privateuseone:0"
    # Module.to() asks for torch.privateuseone when moving Parameters.  The
    # extension exposes the backend through PrivateUse1 without that Python
    # module, so replace Parameters explicitly while preserving autograd leafs.
    for name, parameter in list(model.named_parameters()):
        parent_name, _, parameter_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, parameter_name, torch.nn.Parameter(parameter.detach().to(runtime_device)))
    for name, buffer in list(model.named_buffers()):
        parent_name, _, buffer_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, buffer_name, buffer.detach().to(runtime_device))
    return model


def _run_row(name, fixture, device, warmups, repetitions, initial_state=None, return_state=False):
    if device not in {"cpu", "vk:0"}:
        raise ValueError("device must be exactly vk:0")
    runtime_device = "privateuseone:0" if device == "vk:0" else device
    if device != "cpu":
        import pytorch_vulkan  # noqa: F401 - registers the PrivateUse1 backend
    model = fixture.make_cpu()
    if initial_state is not None and device == "cpu":
        model.load_state_dict(initial_state)
    model = _move_model(model, device)
    if initial_state is not None and device == "vk:0":
        model = _move_model(fixture.make_cpu(), device)
        for parameter_name_full, parameter in list(model.named_parameters()):
            parent_name, _, parameter_name = parameter_name_full.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            value = initial_state[parameter_name_full].detach().to(runtime_device)
            setattr(parent, parameter_name, torch.nn.Parameter(value))
    inputs, target = (value.to(runtime_device) for value in fixture.make_inputs())
    mask = model.causal_mask if name == "attention" else None
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def train_once():
        scoped = device != "cpu"
        scope_started = False
        try:
            if scoped:
                import pytorch_vulkan
                pytorch_vulkan._C.begin_training_step()
                scope_started = True
            optimizer.zero_grad(set_to_none=False)
            loss = fixture.loss(model(inputs, mask) if mask is not None else model(inputs), target)
            loss.backward()
            optimizer.step()
            measured_loss = loss
            if scope_started:
                pytorch_vulkan._C.end_training_step()
                scope_started = False
        except BaseException:
            if scope_started:
                pytorch_vulkan._C.cancel_training_step()
            raise
        return measured_loss

    for _ in range(warmups):
        train_once()
    samples = []
    losses = []
    gpu_samples = []
    timing_supported = False
    if device != "cpu":
        import pytorch_vulkan
        pytorch_vulkan._C.reset_execution_counters()
        timing_supported = pytorch_vulkan._C.timestamp_queries_supported()
    for _ in range(repetitions):
        if device != "cpu" and timing_supported:
            pytorch_vulkan._C.reset_gpu_timing()
        start = time.perf_counter_ns()
        loss = train_once()
        samples.append(time.perf_counter_ns() - start)
        losses.append(loss.detach())
        if device != "cpu" and timing_supported:
            gpu_samples.append(sum(
                sample["gpu_time_ns"]
                for sample in pytorch_vulkan._C.gpu_timing_snapshot()
                if sample.get("available")
            ))
    counters = _counter_fields(device)
    # Readback is presentation-only and intentionally happens after the
    # execution counter snapshot so it cannot be reported as model work.
    losses = [float(loss.detach().cpu()) for loss in losses]
    row = {
        "model": name,
        "mode": "cpu" if device == "cpu" else "vulkan",
        "device": device,
        "dtype": "float32",
        "shape": list(fixture.shape),
        "batch": fixture.batch,
        "sequence_length": getattr(fixture, "sequence_length", None),
        "warmups": warmups,
        "repetitions": repetitions,
        "seed": fixture.seed,
        "host_time": {"samples_ns": samples, "mean_ns": statistics.mean(samples)},
        "gpu_time": (
            {"status": "not_applicable", "samples_ns": None, "mean_ns": None}
            if device == "cpu" else
            {"status": "available", "samples_ns": gpu_samples, "mean_ns": statistics.mean(gpu_samples)}
            if timing_supported and len(gpu_samples) == repetitions and all(gpu_samples)
            else {"status": "unavailable", "samples_ns": None, "mean_ns": None}
        ),
        "timing_source": (
            "host_wall" if device == "cpu" else
            "gpu_timestamp" if timing_supported and len(gpu_samples) == repetitions and all(gpu_samples)
            else "unavailable"
        ),
        "arithmetic_operations": fixture.arithmetic_operations(),
        "effective_tflops": (
            fixture.arithmetic_operations() / (
                (statistics.mean(gpu_samples) if timing_supported and len(gpu_samples) == repetitions and all(gpu_samples) else statistics.mean(samples))
                * 1_000.0
            )
            if device == "cpu" or timing_supported and len(gpu_samples) == repetitions and all(gpu_samples)
            else None
        ),
        "validation": "passed" if device == "cpu" else "blocked",
        "cpu_parity": {"status": "reference" if device == "cpu" else "pending"},
        "transfer_count": counters["explicit_transfers"],
        "loss": losses[-1],
        "parity": {
            "status": "reference" if device == "cpu" else "pending",
            "loss_abs_difference": None,
            "parameter_max_abs_difference": None,
        },
    }
    row.update(counters)
    final_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }
    return (row, final_state) if return_state else row


def _apply_parity(vulkan_row, cpu_row, cpu_state, vulkan_state):
    vulkan_row["parity"] = {
        "status": "measured",
        "loss_abs_difference": abs(vulkan_row["loss"] - cpu_row["loss"]),
        "parameter_max_abs_difference": max(
            float((vulkan_state[name] - cpu_state[name]).abs().max())
            for name in cpu_state
        ),
    }
    vulkan_row["cpu_parity"] = {"status": "measured"}
    vulkan_row["validation"] = "passed"


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"artifact schema_version must be {SCHEMA_VERSION}")
    rows = artifact.get("rows") if isinstance(artifact, dict) else None
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("artifact must contain exactly one CPU and one Vulkan row")
    required = {"model", "mode", "device", "dtype", "shape", "batch", "sequence_length", "warmups", "repetitions", "seed", "host_time", "gpu_time", "timing_source", "loss", "parity", "vulkan_copies", "explicit_transfers", "fallbacks", "dispatches", "submissions", "completions", "waits", "arithmetic_operations", "effective_tflops", "validation", "cpu_parity", "transfer_count", "transfer_operations", "transfer_submissions", "transfer_completions", "transfer_waits", "compute_submissions", "fallback_status"}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        missing = required - row.keys()
        if missing:
            raise ValueError(f"rows[{index}] missing fields: {sorted(missing)}")
        if row["mode"] not in {"cpu", "vulkan"} or row["device"] != ("cpu" if row["mode"] == "cpu" else "vk:0"):
            raise ValueError(f"rows[{index}] has an invalid device identity")
        if row["model"] not in FIXTURES:
            raise ValueError(f"rows[{index}] has an invalid model")
        fixture = FIXTURES[row["model"]]()
        if row["shape"] != list(fixture.shape) or row["batch"] != fixture.batch or row["sequence_length"] != getattr(fixture, "sequence_length", None):
            raise ValueError(f"rows[{index}] has invalid fixture shape")
        if isinstance(row["seed"], bool) or not isinstance(row["seed"], int) or row["seed"] < 0:
            raise ValueError(f"rows[{index}] has an invalid seed")
        for field in ("warmups", "repetitions", "batch"):
            if isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < (1 if field == "repetitions" else 0):
                raise ValueError(f"rows[{index}] has invalid {field}")
        if not isinstance(row["loss"], (int, float)) or not math.isfinite(row["loss"]):
            raise ValueError(f"rows[{index}] has an invalid loss")
        if not isinstance(row["arithmetic_operations"], int) or row["arithmetic_operations"] <= 0:
            raise ValueError(f"rows[{index}] has invalid arithmetic operation count")
        if row["timing_source"] not in {"gpu_timestamp", "host_wall", "unavailable"}:
            raise ValueError(f"rows[{index}] has invalid timing source")
        if row["timing_source"] == "unavailable":
            if row["effective_tflops"] is not None:
                raise ValueError(f"rows[{index}] reports TFLOP/s without timing")
        elif not isinstance(row["effective_tflops"], (int, float)) or not math.isfinite(row["effective_tflops"]) or row["effective_tflops"] < 0:
            raise ValueError(f"rows[{index}] has invalid effective TFLOP/s")
        if row["validation"] != "passed" or not isinstance(row["cpu_parity"], dict) or row["cpu_parity"].get("status") not in {"reference", "measured"}:
            raise ValueError(f"rows[{index}] lacks saturation validation evidence")
        if row["mode"] == "vulkan" and row["cpu_parity"]["status"] != "measured":
            raise ValueError(f"rows[{index}] lacks measured CPU parity")
        if not isinstance(row["transfer_count"], int) or row["transfer_count"] < 0 or row["transfer_count"] != row["explicit_transfers"]:
            raise ValueError(f"rows[{index}] has invalid transfer count")
        counter_fields = ("dispatches", "vulkan_copies", "explicit_transfers", "fallbacks", "submissions", "completions", "waits", "transfer_operations", "transfer_submissions", "transfer_completions", "transfer_waits", "compute_submissions")
        if any(isinstance(row[field], bool) or not isinstance(row[field], int) or row[field] < 0 for field in counter_fields):
            raise ValueError(f"rows[{index}] violates counter contract")
        if not isinstance(row["fallback_status"], bool) or row["fallback_status"] != (row["fallbacks"] != 0):
            raise ValueError(f"rows[{index}] violates counter contract")
        if row["mode"] == "cpu" and any(row[field] != 0 for field in counter_fields):
            raise ValueError(f"rows[{index}] violates CPU counter contract")
        if row["transfer_operations"] != row["vulkan_copies"]:
            raise ValueError(f"rows[{index}] has inconsistent transfer operation counters")
        if row["transfer_submissions"] != row["transfer_completions"]:
            raise ValueError(f"rows[{index}] has incomplete transfer lifecycle")
        if row["transfer_waits"] < row["transfer_submissions"]:
            raise ValueError(f"rows[{index}] has incomplete transfer waits")
        if row["transfer_operations"] == 0 and any(
            row[field] != 0
            for field in ("transfer_submissions", "transfer_completions", "transfer_waits")
        ):
            raise ValueError(f"rows[{index}] has transfer lifecycle without operations")
        if row["transfer_operations"] > 0 and row["transfer_submissions"] == 0 and row["compute_submissions"] == 0:
            raise ValueError(f"rows[{index}] has operations without transfer submission")
        if row["transfer_submissions"] > row["transfer_operations"]:
            raise ValueError(f"rows[{index}] has transfer submissions exceed transfer operations")
        # Attention's bounded transposed-bmm path and RNN's explicit sequence
        # materialization use measured Vulkan-side copies; neither is a host
        # transfer or fallback.
        allowed_copy = (
            row["model"] == "attention" and row["vulkan_copies"] <= row["repetitions"]
        ) or (
            row["model"] == "rnn"
            and row["vulkan_copies"] <= 3 * row["sequence_length"] * row["repetitions"]
        )
        if row["dtype"] != "float32" or (row["vulkan_copies"] != 0 and not allowed_copy) or row["fallbacks"] != 0:
            raise ValueError(f"rows[{index}] violates counter contract")
        parity = row["parity"]
        if row["mode"] == "cpu":
            if parity != {"status": "reference", "loss_abs_difference": None, "parameter_max_abs_difference": None}:
                raise ValueError(f"rows[{index}] has invalid reference parity")
        elif parity.get("status") != "measured" or any(
            not isinstance(parity.get(key), (int, float))
            or not math.isfinite(parity[key])
            or parity[key] < 0
            for key in ("loss_abs_difference", "parameter_max_abs_difference")
        ):
            raise ValueError(f"rows[{index}] lacks measured parity")
        elif any(
            parity[key] > MODEL_PARITY_ABS_TOLERANCE
            for key in ("loss_abs_difference", "parameter_max_abs_difference")
        ):
            raise ValueError(f"rows[{index}] exceeds model parity tolerance")
        host_samples = row["host_time"].get("samples_ns")
        if not isinstance(host_samples, list) or len(host_samples) != row["repetitions"] or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in host_samples):
            raise ValueError(f"rows[{index}] has invalid host timing samples")
        host_mean = row["host_time"].get("mean_ns")
        if not isinstance(host_mean, (int, float)) or not math.isfinite(host_mean) or host_mean < 0:
            raise ValueError(f"rows[{index}] has invalid host timing mean")
        expected_host_mean = statistics.mean(host_samples)
        if abs(host_mean - expected_host_mean) > max(TIMING_MEAN_ABS_TOLERANCE_NS, abs(expected_host_mean) * TIMING_MEAN_REL_TOLERANCE):
            raise ValueError(f"rows[{index}] has an inconsistent timing mean")
        if row["mode"] == "vulkan" and any(row.get(key, 0) <= 0 for key in ("dispatches", "submissions", "completions", "waits")):
            raise ValueError(f"rows[{index}] lacks Vulkan execution counters")
        if row["mode"] == "vulkan" and not (row["submissions"] == row["completions"] == row["waits"] <= row["dispatches"]):
            raise ValueError(f"rows[{index}] has inconsistent Vulkan counters")
        if row["compute_submissions"] != row["submissions"]:
            raise ValueError(f"rows[{index}] has inconsistent compute submission counters")
        gpu = row["gpu_time"]
        if row["mode"] == "cpu":
            if gpu != {"status": "not_applicable", "samples_ns": None, "mean_ns": None}:
                raise ValueError(f"rows[{index}] has invalid CPU timing status")
        elif gpu.get("status") == "available":
            samples = gpu.get("samples_ns")
            if not isinstance(samples, list) or len(samples) != row["repetitions"] or any(not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in samples):
                raise ValueError(f"rows[{index}] has invalid GPU timing samples")
            mean = gpu.get("mean_ns")
            expected_mean = statistics.mean(samples)
            if not isinstance(mean, (int, float)) or not math.isfinite(mean) or mean < 0 or abs(mean - expected_mean) > max(TIMING_MEAN_ABS_TOLERANCE_NS, abs(expected_mean) * TIMING_MEAN_REL_TOLERANCE):
                raise ValueError(f"rows[{index}] has an inconsistent GPU timing mean")
        elif gpu != {"status": "unavailable", "samples_ns": None, "mean_ns": None}:
            raise ValueError(f"rows[{index}] has invalid GPU timing status")
        if row["mode"] == "cpu" and row["timing_source"] != "host_wall":
            raise ValueError(f"rows[{index}] CPU timing source must be host_wall")
        if row["mode"] == "vulkan" and gpu["status"] == "available" and row["timing_source"] != "gpu_timestamp":
            raise ValueError(f"rows[{index}] GPU timing source must be gpu_timestamp")
        if row["mode"] == "vulkan" and gpu["status"] == "unavailable" and row["timing_source"] != "unavailable":
            raise ValueError(f"rows[{index}] unavailable GPU timing source mismatch")
    models = {row["model"] for row in rows}
    modes = [row["mode"] for row in rows]
    if len(models) != 1 or sorted(modes) != ["cpu", "vulkan"]:
        raise ValueError("artifact must contain exactly one CPU and one Vulkan row")
    cpu, vulkan = (next(row for row in rows if row["mode"] == mode) for mode in ("cpu", "vulkan"))
    for field in ("seed", "dtype", "shape", "batch", "sequence_length", "warmups", "repetitions"):
        if cpu[field] != vulkan[field]:
            raise ValueError(f"CPU/Vulkan {field} does not match")
    if artifact.get("seed") != cpu["seed"] or artifact["seed"] != vulkan["seed"]:
        raise ValueError("artifact and row seeds do not match")
    return artifact


def run(model_name, device, warmups, repetitions):
    if model_name not in FIXTURES:
        raise ValueError(f"unsupported model: {model_name}")
    fixture = FIXTURES[model_name]()
    if device != "vk:0":
        raise ValueError("device must be exactly vk:0")
    # Register PrivateUse1 before the CPU reference initializes autograd's
    # device queues; the Vulkan row runs in the same process.
    import pytorch_vulkan  # noqa: F401
    initial_state = {
        name: value.detach().clone()
        for name, value in fixture.make_cpu().state_dict().items()
    }
    cpu_row, cpu_state = _run_row(model_name, fixture, "cpu", warmups, repetitions, initial_state, True)
    vulkan_row, vulkan_state = _run_row(model_name, fixture, device, warmups, repetitions, initial_state, True)
    _apply_parity(vulkan_row, cpu_row, cpu_state, vulkan_state)
    rows = [cpu_row, vulkan_row]
    artifact = {"schema_version": SCHEMA_VERSION, "seed": SEED, "rows": rows}
    return validate_artifact(artifact)


def run_all(device, warmups, repetitions):
    artifact = {
        "schema_version": SCHEMA_VERSION,
        "seed": SEED,
        "models": {
            name: run(name, device, warmups, repetitions) for name in FIXTURES
        },
    }
    return validate_aggregate_artifact(artifact)


def validate_aggregate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"aggregate schema_version must be {SCHEMA_VERSION}")
    models = artifact.get("models")
    if not isinstance(models, dict) or set(models) != set(FIXTURES):
        raise ValueError("aggregate models must contain cnn, attention, and rnn")
    if artifact.get("seed") != SEED:
        raise ValueError("aggregate seed does not match benchmark seed")
    for name in FIXTURES:
        nested = validate_artifact(models[name])
        if nested["rows"][0]["model"] != name:
            raise ValueError(f"aggregate {name} artifact has the wrong model")
    return artifact


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=tuple(FIXTURES) + ("all",), required=True)
    parser.add_argument("--device", default="vk:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be non-negative and repetitions must be positive")
    try:
        artifact = run_all(args.device, args.warmups, args.repetitions) if args.model == "all" else run(args.model, args.device, args.warmups, args.repetitions)
    except (RuntimeError, NotImplementedError) as error:
        args.output.write_text(json.dumps({
            "schema_version": 1,
            "status": "blocked",
            "reason": str(error),
            "rows": [],
        }, indent=2, sort_keys=True) + "\n")
        print(f"Vulkan benchmark unavailable: {error}", file=sys.stderr)
        return 77
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "ok", "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
