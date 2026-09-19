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


def train_step(model, optimizer, inputs, targets, scoped, phase="optimizer"):
    if scoped:
        _C.begin_training_step()
    try:
        if phase != "forward":
            optimizer.zero_grad(set_to_none=False)
        output = model(inputs)
        loss = squared_error(output, targets)
        if phase != "forward":
            loss.backward()
        if phase == "optimizer":
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


def timing_status(supported, reason):
    if supported is None:
        return {"status": "not_applicable", "reason": reason}
    return {
        "status": "available" if supported else "unsupported",
        "reason": reason if not supported else "timestamp queries completed",
    }


def timing_fields(supported, reason, unavailable_reason=None):
    if supported and unavailable_reason is not None:
        return {
            "gpu_time_ns": None,
            "timing_status": "unavailable",
            "timing_reason": unavailable_reason,
        }
    status = timing_status(supported, reason)
    return {
        "gpu_time_ns": [] if supported else None,
        "timing_status": status["status"],
        "timing_reason": status["reason"],
    }


def validate_timing_samples(samples, expected_scopes, submission_floor, completed_submissions):
    if completed_submissions <= 0:
        return False, "no completed submission is available for timing"
    candidates = [
        sample
        for sample in samples
        if sample.get("available", False)
        and sample.get("submission_id", 0) > submission_floor
    ]
    if not candidates:
        return False, "no completed timestamp sample is available"
    for sample in candidates:
        if sample.get("scope") not in expected_scopes:
            return False, "timestamp sample scope is unexpected"
        if sample.get("gpu_time_ns", -1) < 0:
            return False, "timestamp sample GPU duration is invalid"
    return True, "available"


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
    phase=None,
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
    phase = phase or ("optimizer" if mode == "step" else "forward")
    training = phase != "forward"
    scoped = device.startswith("vk") and training
    timing_supported = None
    timing_reason = "timing is not applicable to CPU execution"
    if device.startswith("vk"):
        timing_supported = _C.timestamp_queries_supported()
        timing_reason = _C.timestamp_query_support_reason()

    for _ in range(warmups):
        if training:
            train_step(model, optimizer, inputs, targets, scoped, phase)
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
    gpu_time_ns = []
    timing_valid = True
    timing_failure_reason = None
    host_total_ns = []
    transfer_time_ns = []
    scope_labels = set()
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
            previous_samples = _C.gpu_timing_snapshot()
            submission_floor = max(
                (sample["submission_id"] for sample in previous_samples), default=0
            )
            _C.reset_execution_counters()
            _C.reset_timing()
            _C.reset_gpu_timing()
            _C.reset_descriptor_resource_counters()
        start = time.monotonic()
        for _ in range(steps):
            if training:
                last_loss = train_step(model, optimizer, inputs, targets, scoped, phase)
            else:
                last_loss = forward_step(model, inputs, targets)
        elapsed = time.monotonic() - start
        samples.append(elapsed)
        host_total_ns.append(int(elapsed * 1_000_000_000))
        if device.startswith("vk"):
            counters = _C.execution_counter_snapshot()
            timing = _C.timing_snapshot()
            gpu_samples = _C.gpu_timing_snapshot()
            expected_scopes = {"training"} if scoped else {"gemm", "operator"}
            valid, validation_reason = validate_timing_samples(
                gpu_samples, expected_scopes, submission_floor, _C.compute_completed_count()
            )
            if timing_supported and not valid:
                timing_valid = False
                timing_failure_reason = validation_reason
            scope_labels.update(
                sample["scope"]
                for sample in gpu_samples
                if sample.get("available", False) and sample["submission_id"] > submission_floor
            )
            gpu_time_ns.append(
                sum(
                    sample["gpu_time_ns"]
                    for sample in gpu_samples
                    if sample.get("available", False)
                    and sample["submission_id"] > submission_floor
                )
                if timing_supported and valid
                else None
            )
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
            gpu_time_ns.append(None)
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
        "phase": phase,
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
        "host_total_ns": host_total_ns,
        "gpu_time_ns": gpu_time_ns,
        "transfer_time_ns": transfer_time_ns or [0] * repetitions,
        "transfer_time_semantics": (
            "zero means transfer activity is excluded from steady-state timing; "
            "inspect explicit_transfers for observed activity"
        ),
        "warmup": {"count": warmups, "excluded_from_steady_state": True},
        "scope": phase,
        "scope_labels": sorted(scope_labels) or [phase],
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
    result.update(
        timing_fields(
            timing_supported,
            timing_reason,
            timing_failure_reason if timing_supported and not timing_valid else None,
        )
    )
    result["gpu_time_ns"] = gpu_time_ns if timing_supported and timing_valid else None
    return result, model, cpu_inputs, cpu_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="vk:0")
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
    if args.mode == "both":
        phases = ("forward", "backward", "optimizer")
    elif args.mode == "sync":
        phases = ("forward",)
    else:
        phases = ("optimizer",)
    torch.set_num_threads(args.cpu_intraop_threads)
    torch.set_num_interop_threads(args.cpu_interop_threads)
    for kind in workloads:
        steps = args.mlp_steps if kind == "mlp" else args.mnist_steps
        batch = 3 if kind == "mlp" else args.mnist_batch_size
        baseline = make_model(kind, "cpu", 17).state_dict()
        for phase in phases:
            mode = "sync" if phase == "forward" else "step"
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
                phase,
            )
            print(json.dumps(cpu_result, allow_nan=False, sort_keys=True))
            if not pytorch_vulkan.is_available():
                continue
            vk_result, vk_model, _, _ = run(
                kind,
                mode,
                args.backward_mode,
                args.device,
                args.warmups,
                args.repetitions,
                steps,
                batch,
                17,
                baseline,
                phase,
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
