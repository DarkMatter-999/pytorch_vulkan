#!/usr/bin/env python3
"""Resident Vulkan training benchmark for the Phase 6A execution gate."""

import argparse
import json
import time

import torch

import pytorch_vulkan
from pytorch_vulkan import _C


def squared_error(output, target):
    error = output - target
    return error.mul(error).sum()


def make_mlp(device, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device=device, dtype=torch.float32)


def make_mnist(device, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(784, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ).to(device=device, dtype=torch.float32)


def make_data(kind, seed, batch_size):
    torch.manual_seed(seed)
    if kind == "mlp":
        return torch.randn(batch_size, 8), torch.randn(batch_size, 4)
    inputs = torch.randn(batch_size, 1, 28, 28)
    labels = torch.randint(10, (batch_size,))
    return inputs, torch.nn.functional.one_hot(labels, 10).float()


def train_step(model, optimizer, inputs, targets, scoped):
    if scoped:
        _C.begin_training_step()
    try:
        optimizer.zero_grad(set_to_none=False)
        output = model(inputs)
        loss = squared_error(output, targets)
        loss.backward(torch.ones_like(loss))
        optimizer.step()
        if scoped:
            _C.end_training_step()
    except BaseException:
        if scoped:
            _C.cancel_training_step()
        raise
    return loss


def run(kind, mode, steps, batch_size, seed):
    device = "vk:0"
    model = make_mlp(device, seed) if kind == "mlp" else make_mnist(device, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    cpu_inputs, cpu_targets = make_data(kind, seed + 1, batch_size)

    # Upload and setup are intentionally outside the measured training interval.
    upload_start = time.monotonic()
    inputs = cpu_inputs.to(device)
    targets = cpu_targets.to(device)
    upload = time.monotonic() - upload_start
    _C.reset_execution_counters()
    _C.reset_timing()

    start = time.monotonic()
    for _ in range(steps):
        loss = train_step(model, optimizer, inputs, targets, mode == "step")
    total = time.monotonic() - start
    counters = _C.execution_counter_snapshot()
    timing = _C.timing_snapshot()
    final_loss = float(loss.cpu())
    finite_loss = bool(torch.isfinite(torch.tensor(final_loss)))
    result = {
        "workload": kind,
        "mode": mode,
        "batch_size": batch_size,
        "steps": steps,
        "seed": seed,
        "optimizer": "SGD(lr=0.01)",
        "upload_seconds": upload,
        "allocation_seconds": timing[0],
        "recording_seconds": timing[1],
        "submit_wait_seconds": timing[2],
        "compute_seconds": timing[3],
        "total_seconds": total,
        "seconds_per_step": total / steps,
        "dispatches": counters[0],
        "vulkan_copies": counters[1],
        "explicit_transfers": counters[2],
        "submitted": _C.compute_submitted_count(),
        "completed": _C.compute_completed_count(),
        "waits": _C.compute_wait_count(),
        "finite_loss": finite_loss,
        "final_loss": final_loss if finite_loss else None,
        "diverged": not finite_loss,
    }
    print(json.dumps(result, allow_nan=False, sort_keys=True))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("mlp", "mnist", "both"), default="both")
    parser.add_argument("--mode", choices=("sync", "step", "both"), default="both")
    parser.add_argument("--mlp-steps", type=int, default=100)
    parser.add_argument("--mnist-steps", type=int, default=10)
    parser.add_argument("--mnist-batch-size", type=int, default=512)
    args = parser.parse_args()
    if not pytorch_vulkan.is_available():
        print("Vulkan unavailable")
        return 77
    workloads = ("mlp", "mnist") if args.workload == "both" else (args.workload,)
    modes = ("sync", "step") if args.mode == "both" else (args.mode,)
    for kind in workloads:
        for mode in modes:
            run(kind, mode, args.mlp_steps if kind == "mlp" else args.mnist_steps,
                3 if kind == "mlp" else args.mnist_batch_size, 17)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
