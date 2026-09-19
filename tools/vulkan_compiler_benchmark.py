#!/usr/bin/env python3
"""Compare eager, compiler-unfused, and compiler-fused Vulkan training."""

import argparse
import json
import os
import subprocess
import time

import torch

import pytorch_vulkan
from pytorch_vulkan import _C


MODES = ("eager", "compiler-unfused", "compiler-fused")


def squared_error(output, target):
    error = output - target
    return error.mul(error).sum()


def make_model(kind, device, seed):
    torch.manual_seed(seed)
    if kind == "mlp":
        layers = (torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    else:
        layers = (
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(784, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )
    return torch.nn.Sequential(*layers).to(device=device, dtype=torch.float32)


def make_data(kind, seed, batch_size):
    torch.manual_seed(seed)
    if kind == "mlp":
        return torch.randn(batch_size, 8), torch.randn(batch_size, 4)
    inputs = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(10, (batch_size,))
    return inputs, torch.nn.functional.one_hot(labels, 10).float()


def compile_model(model, mode):
    if mode == "eager":
        return model
    backend = "eager" if mode == "compiler-unfused" else pytorch_vulkan.vulkan_backend
    torch._dynamo.reset()
    return torch.compile(model, backend=backend, fullgraph=True)


def step(model, optimizer, inputs, targets):
    optimizer.zero_grad(set_to_none=False)
    output = model(inputs)
    loss = squared_error(output, targets)
    loss.backward(torch.ones_like(loss))
    optimizer.step()
    return loss


def counters():
    snapshot = _C.execution_counter_snapshot()
    return {
        "dispatches": snapshot[0],
        "vulkan_copies": snapshot[1],
        "explicit_transfers": snapshot[2],
        "fallbacks": snapshot[3],
        "submissions": _C.compute_submitted_count(),
        "completions": _C.compute_completed_count(),
        "waits": _C.compute_wait_count(),
    }


def state_cpu(model):
    return {name: value.detach().cpu() for name, value in model.state_dict().items()}


def snapshot_training_state(model, optimizer):
    return {
        "parameters": [
            parameter.detach().cpu().clone() for parameter in model.parameters()
        ],
        "gradients": [
            None if parameter.grad is None else parameter.grad.detach().cpu().clone()
            for parameter in model.parameters()
        ],
        "optimizer": [
            {
                name: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for name, value in optimizer.state[parameter].items()
            }
            for parameter in model.parameters()
        ],
    }


def restore_training_state(model, optimizer, snapshot):
    with torch.no_grad():
        for parameter, value, gradient in zip(
            model.parameters(), snapshot["parameters"], snapshot["gradients"]
        ):
            parameter.copy_(value.to("vk:0"))
            parameter.grad = None if gradient is None else gradient.to("vk:0")
    optimizer.state.clear()
    for parameter, state in zip(model.parameters(), snapshot["optimizer"]):
        optimizer.state[parameter] = {
            name: value.to("vk:0")
            if isinstance(value, torch.Tensor) and name != "step"
            else value
            for name, value in state.items()
        }


def cpu_reference(kind, steps, batch_size, seed, learning_rate):
    model = make_model(kind, "cpu", seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    inputs, targets = make_data(kind, seed + 1, batch_size)
    loss = None
    for _ in range(steps):
        loss = step(model, optimizer, inputs, targets)
    return model, float(loss)


def run_workload(kind, mode, backward_mode, steps, batch_size, seed, learning_rate):
    if backward_mode == "unfused":
        os.environ["PYTORCH_VULKAN_DISABLE_MULTI_OUTPUT_BACKWARD"] = "1"
    else:
        os.environ.pop("PYTORCH_VULKAN_DISABLE_MULTI_OUTPUT_BACKWARD", None)
    initial_model = make_model(kind, "cpu", seed)
    cpu_model, cpu_loss = cpu_reference(kind, steps, batch_size, seed, learning_rate)
    model = make_model(kind, "vk:0", seed)
    model.load_state_dict(
        {name: value.to("vk:0") for name, value in initial_model.state_dict().items()}
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    cpu_inputs, cpu_targets = make_data(kind, seed + 1, batch_size)
    upload_start = time.monotonic()
    inputs = cpu_inputs.to("vk:0")
    targets = cpu_targets.to("vk:0")
    upload_seconds = time.monotonic() - upload_start
    compiled = compile_model(model, mode)
    initial_state = snapshot_training_state(model, optimizer)

    # Compile and allocate the first execution, then restore every mutable state.
    pytorch_vulkan._C.begin_training_step()
    try:
        step(compiled, optimizer, inputs, targets)
        pytorch_vulkan._C.end_training_step()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    restore_training_state(model, optimizer, initial_state)

    _C.reset_execution_counters()
    _C.reset_timing()
    start = time.monotonic()
    last_loss = None
    for _ in range(steps):
        _C.begin_training_step()
        try:
            last_loss = step(compiled, optimizer, inputs, targets)
            _C.end_training_step()
        except BaseException:
            _C.cancel_training_step()
            raise
    elapsed = time.monotonic() - start
    measured_counters = counters()
    measured_timing = _C.timing_snapshot()
    if mode == "compiler-fused" and kind == "mlp":
        # The first measured step initializes optimizer state. The combined
        # backward path uses 28 steady-state dispatches; the separate path
        # retains the historical 32-dispatch count.
        expected_dispatches = (
            28 * steps - 8 if backward_mode == "fused" else 32 * steps - 8
        )
        if measured_counters["dispatches"] != expected_dispatches:
            raise RuntimeError(
                f"MLP fusion dispatch gate failed: expected {expected_dispatches}, "
                f"got {measured_counters['dispatches']}"
            )
    result_state = state_cpu(model)
    expected_state = state_cpu(cpu_model)
    final_state_parity = all(
        torch.allclose(result_state[name], expected_state[name], rtol=1e-4, atol=1e-4)
        for name in expected_state
    )
    final_loss = float(last_loss.cpu())
    stats = pytorch_vulkan.compiler_stats() if mode == "compiler-fused" else None
    if stats is not None:
        stats = {
            "node_count": stats["node_count"],
            "calls": stats["calls"],
            "setup_time": stats["setup_time"],
            "replay_time": stats["replay_time"],
            "fusion_applied": stats["fusion_applied"],
            "first_call_metrics": stats["call_metrics"][0]
            if stats["call_metrics"]
            else None,
            "last_call_metrics": stats["call_metrics"][-1]
            if stats["call_metrics"]
            else None,
        }
    return {
        "workload": kind,
        "mode": mode,
        "backward_mode": backward_mode,
        "steps": steps,
        "batch_size": batch_size,
        "seed": seed,
        "optimizer": "SGD",
        "learning_rate": learning_rate,
        "input_target_upload_seconds": upload_seconds,
        "allocation_seconds": measured_timing[0],
        "recording_seconds": measured_timing[1],
        "submit_seconds": measured_timing[2],
        "host_fence_wait_seconds": measured_timing[3],
        "total_seconds": elapsed,
        "seconds_per_step": elapsed / steps,
        "counters": measured_counters,
        "finite_loss": bool(torch.isfinite(torch.tensor(final_loss))),
        "final_loss": final_loss,
        "cpu_final_loss": cpu_loss,
        "final_state_parity": final_state_parity,
        "optimizer_state_parity": "covered_by_tests/python/test_vulkan_training.py",
        "compiler_stats": stats,
        "fusion_applied": bool(stats and stats["fusion_applied"]),
    }


def telemetry():
    try:
        completed = subprocess.run(
            ["rocm-smi"], capture_output=True, text=True, timeout=30
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": "rocm-smi", "available": False, "error": str(error)}
    return {
        "command": "rocm-smi",
        "available": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("mlp", "mnist", "both"), default="both")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--mlp-batch-size", type=int, default=3)
    parser.add_argument("--mnist-batch-size", type=int, default=512)
    parser.add_argument("--mlp-lr", type=float, default=1e-5)
    parser.add_argument("--mnist-lr", type=float, default=1e-7)
    parser.add_argument(
        "--backward-mode", choices=("fused", "unfused"), default="fused"
    )
    parser.add_argument("--json-out", type=str)
    args = parser.parse_args()
    if not pytorch_vulkan.is_available():
        print("Vulkan unavailable")
        return 77
    workloads = ("mlp", "mnist") if args.workload == "both" else (args.workload,)
    rows = []
    for kind in workloads:
        batch_size = args.mlp_batch_size if kind == "mlp" else args.mnist_batch_size
        learning_rate = args.mlp_lr if kind == "mlp" else args.mnist_lr
        for mode in MODES:
            row = run_workload(
                kind,
                mode,
                args.backward_mode,
                args.steps,
                batch_size,
                17,
                learning_rate,
            )
            rows.append(row)
    report = {"schema": 1, "device": "vk:0", "rows": rows, "rocm_smi": telemetry()}
    encoded = json.dumps(report, allow_nan=False, indent=2, sort_keys=True)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.write("\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
