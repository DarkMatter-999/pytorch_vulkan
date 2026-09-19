#!/usr/bin/env python3
"""Deterministic resident CPU/Vulkan training baseline benchmark."""

import argparse
import json
import os
import statistics
import time

import torch

import pytorch_vulkan
from pytorch_vulkan import _C


def squared_error(output, target):
    error = output - target
    return error.mul(error).sum()


def make_model(kind, device, seed):
    torch.manual_seed(seed)
    if kind == "mlp":
        modules = (torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    else:
        modules = (
            torch.nn.Flatten(start_dim=1),
            torch.nn.Linear(784, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10),
        )
    return torch.nn.Sequential(*modules).to(device=device, dtype=torch.float32)


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
        loss.backward()
        optimizer.step()
        if scoped:
            _C.end_training_step()
    except BaseException:
        if scoped:
            _C.cancel_training_step()
        raise
    return loss


def forward_step(model, inputs, targets):
    return squared_error(model(inputs), targets)


def _summary(samples):
    mean = statistics.mean(samples)
    median = statistics.median(samples)
    variance = statistics.pvariance(samples) if len(samples) > 1 else 0.0
    deviation = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    outliers = [
        value for value in samples if deviation and abs(value - mean) > 2.0 * deviation
    ]
    return {
        "mean_seconds": mean,
        "median_seconds": median,
        "variance_seconds2": variance,
        "outlier_seconds": outliers,
        "samples_seconds": samples,
    }


def _cpu_threads():
    return {
        "intraop": torch.get_num_threads(),
        "interop": torch.get_num_interop_threads(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def run(
    kind,
    mode,
    backward_mode,
    device,
    warmups,
    repetitions,
    steps,
    batch_size,
    seed,
    initial_state=None,
):
    if device.startswith("vk"):
        if backward_mode == "unfused":
            os.environ["PYTORCH_VULKAN_DISABLE_MULTI_OUTPUT_BACKWARD"] = "1"
        else:
            os.environ.pop("PYTORCH_VULKAN_DISABLE_MULTI_OUTPUT_BACKWARD", None)
    model = make_model(kind, device, seed)
    if initial_state is not None:
        model.load_state_dict(
            {
                name: value.detach().clone().to(device)
                for name, value in initial_state.items()
            }
        )
    learning_rate = 0.01 if kind == "mlp" else 0.0001
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    cpu_inputs, cpu_targets = make_data(kind, seed + 1, batch_size)
    inputs = cpu_inputs.to(device)
    targets = cpu_targets.to(device)
    training = mode == "step"
    scoped = device.startswith("vk") and training

    for _ in range(warmups):
        if training:
            train_step(model, optimizer, inputs, targets, scoped)
        else:
            forward_step(model, inputs, targets)

    samples = []
    dispatches = []
    copies = []
    transfers = []
    fallbacks = []
    submitted = []
    completed = []
    waits = []
    descriptor_pools = []
    descriptor_sets = []
    descriptor_reuses = []
    component_samples = {
        "allocation": [],
        "recording": [],
        "submit": [],
        "host_fence_wait": [],
        "total": [],
    }
    last_loss = None
    for _ in range(repetitions):
        if device.startswith("vk"):
            _C.reset_execution_counters()
            _C.reset_timing()
            _C.reset_descriptor_resource_counters()
        start = time.monotonic()
        for _ in range(steps):
            if training:
                last_loss = train_step(model, optimizer, inputs, targets, scoped)
            else:
                last_loss = forward_step(model, inputs, targets)
        samples.append(time.monotonic() - start)
        if device.startswith("vk"):
            counters = _C.execution_counter_snapshot()
            timing = _C.timing_snapshot()
            dispatches.append(counters[0])
            copies.append(counters[1])
            transfers.append(counters[2])
            fallbacks.append(counters[3])
            submitted.append(_C.compute_submitted_count())
            completed.append(_C.compute_completed_count())
            waits.append(_C.compute_wait_count())
            descriptor_pools.append(_C.descriptor_pool_creation_count())
            descriptor_sets.append(_C.descriptor_set_allocation_count())
            descriptor_reuses.append(_C.descriptor_set_reuse_count())
            for name, value in zip(component_samples, timing):
                component_samples[name].append(value)
        else:
            dispatches.append(0)
            copies.append(0)
            transfers.append(0)
            fallbacks.append(0)
            submitted.append(0)
            completed.append(0)
            waits.append(0)
            descriptor_pools.append(0)
            descriptor_sets.append(0)
            descriptor_reuses.append(0)
            for values in component_samples.values():
                values.append(0.0)

    # This is deliberately after timing: presentation/readback is not steady state.
    final_loss = float(last_loss.detach().cpu())
    finite_loss = bool(torch.isfinite(torch.tensor(final_loss)))
    result = {
        "device": "vulkan" if device.startswith("vk") else "cpu",
        "workload": kind,
        "dtype": "float32",
        "shape": list(inputs.shape),
        "batch": batch_size,
        "mode": mode if device.startswith("vk") else "cpu",
        "backward_mode": backward_mode if device.startswith("vk") else "native",
        "warmups": warmups,
        "repetitions": repetitions,
        "steps_per_repetition": steps,
        "seed": seed,
        "optimizer": f"SGD(lr={learning_rate})",
        "cpu_threads": _cpu_threads(),
        "synchronization_boundaries": (
            "one training-scope submit/wait per step"
            if scoped
            else "per-operation synchronous submit/wait"
            if device.startswith("vk")
            else "CPU operation completion"
        ),
        "wall_time": _summary(samples),
        "component_timings": {
            name: _summary(values) for name, values in component_samples.items()
        },
        "dispatches": dispatches,
        "vulkan_copies": copies,
        "explicit_transfers": transfers,
        "fallbacks": fallbacks,
        "submitted": submitted,
        "completed": completed,
        "waits": waits,
        "descriptor_pool_creations": descriptor_pools,
        "descriptor_set_allocations": descriptor_sets,
        "descriptor_set_reuses": descriptor_reuses,
        "finite_loss": finite_loss,
        "final_loss": final_loss if finite_loss else None,
        "diverged": not finite_loss,
    }
    return result, model, cpu_inputs, cpu_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=("mlp", "mnist", "both"), default="both")
    parser.add_argument("--mode", choices=("sync", "step", "both"), default="both")
    parser.add_argument(
        "--backward-mode", choices=("fused", "unfused"), default="fused"
    )
    parser.add_argument("--mlp-steps", type=int, default=10)
    parser.add_argument("--mnist-steps", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--mnist-batch-size", type=int, default=32)
    parser.add_argument("--cpu-intraop-threads", type=int, default=1)
    parser.add_argument("--cpu-interop-threads", type=int, default=1)
    args = parser.parse_args()
    if (
        args.warmups < 0
        or args.repetitions < 1
        or args.cpu_intraop_threads < 1
        or args.cpu_interop_threads < 1
    ):
        parser.error("warmups/repetitions/thread counts are out of range")
    workloads = ("mlp", "mnist") if args.workload == "both" else (args.workload,)
    modes = ("sync", "step") if args.mode == "both" else (args.mode,)
    torch.set_num_threads(args.cpu_intraop_threads)
    torch.set_num_interop_threads(args.cpu_interop_threads)
    for kind in workloads:
        steps = args.mlp_steps if kind == "mlp" else args.mnist_steps
        batch = 3 if kind == "mlp" else args.mnist_batch_size
        baseline = make_model(kind, "cpu", 17).state_dict()
        for mode in modes:
            cpu_result, cpu_model, _, _ = run(
                kind,
                mode,
                args.backward_mode,
                "cpu",
                args.warmups,
                args.repetitions,
                steps,
                batch,
                17,
                baseline,
            )
            print(json.dumps(cpu_result, allow_nan=False, sort_keys=True))
            if not pytorch_vulkan.is_available():
                continue
            vk_result, vk_model, _, _ = run(
                kind,
                mode,
                args.backward_mode,
                "vk:0",
                args.warmups,
                args.repetitions,
                steps,
                batch,
                17,
                baseline,
            )
            vk_result["cpu_comparison"] = {
                "final_loss": cpu_result["final_loss"],
                "loss_abs_difference": abs(
                    vk_result["final_loss"] - cpu_result["final_loss"]
                ),
                "parameter_max_abs_difference": max(
                    float((a.detach().cpu() - b.detach().cpu()).abs().max())
                    for a, b in zip(vk_model.parameters(), cpu_model.parameters())
                ),
            }
            print(json.dumps(vk_result, allow_nan=False, sort_keys=True))
    return 0 if pytorch_vulkan.is_available() else 77


if __name__ == "__main__":
    raise SystemExit(main())
