#!/usr/bin/env python3
"""Run the production GEMM large-shape workload in an isolated process."""

import argparse
import json
import multiprocessing
import time
from pathlib import Path


LEFT_SHAPE = (2048, 4096)
RIGHT_SHAPE = (4096, 2048)


def validate_timing_samples(samples, submission_id):
    if not samples or not any(sample.get("available", False) for sample in samples):
        return False, "no completed timestamp sample is available"
    for sample in samples:
        if not sample.get("available", False):
            continue
        if sample.get("scope") != "gemm":
            return False, "timestamp sample scope is not gemm"
        if sample.get("submission_id") != submission_id:
            return False, "timestamp sample submission does not match GEMM submission"
    return True, "available"


def _probe_device(torch, pytorch_vulkan, requested_device=None):
    if not pytorch_vulkan.is_available():
        return None, "no suitable Vulkan device is available"
    device = requested_device or f"{torch._C._get_privateuse1_backend_name()}:0"
    torch.ones(1).to(device)
    return device, None


def _child(result_path, requested_device=None):
    import torch
    import pytorch_vulkan

    device, skip_reason = _probe_device(torch, pytorch_vulkan, requested_device)
    if skip_reason:
        result_path.write_text(json.dumps({"status": "skipped", "reason": skip_reason}))
        return
    timing_supported = pytorch_vulkan._C.timestamp_queries_supported()
    timing_reason = pytorch_vulkan._C.timestamp_query_support_reason()

    torch.manual_seed(47)
    left = torch.randn(*LEFT_SHAPE).to(device)
    right = torch.randn(*RIGHT_SHAPE).to(device)
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.reset_gpu_timing()
    start = time.monotonic()
    output = torch.mm(left, right)
    elapsed = time.monotonic() - start
    expected_shape = (LEFT_SHAPE[0], RIGHT_SHAPE[1])
    if tuple(output.shape) != expected_shape:
        raise RuntimeError(
            f"large GEMM produced shape {tuple(output.shape)}, expected {expected_shape}"
        )
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    submissions = pytorch_vulkan._C.compute_submitted_count()
    completions = pytorch_vulkan._C.compute_completed_count()
    waits = pytorch_vulkan._C.compute_wait_count()
    timing = pytorch_vulkan._C.timing_snapshot()
    gpu_samples = pytorch_vulkan._C.gpu_timing_snapshot()
    if counters != (1, 0, 0, 0) or (submissions, completions, waits) != (1, 1, 1):
        raise RuntimeError(
            "large GEMM counter contract failed: "
            f"counters={counters}, submissions={submissions}, "
            f"completions={completions}, waits={waits}"
        )
    timing_valid, timing_validation_reason = (
        validate_timing_samples(gpu_samples, submissions)
        if timing_supported
        else (False, "Vulkan timestamp queries unsupported: " + timing_reason)
    )
    timed = timing_supported and timing_valid
    result_path.write_text(
        json.dumps(
            {
                "status": "completed" if timed else "completed_without_timing",
                "timing_status": "available" if timed else "unavailable",
                "timing_reason": "available" if timed else timing_validation_reason,
                "shape": list(expected_shape),
                "seconds": elapsed,
                "host_total_ns": int(elapsed * 1_000_000_000),
                "gpu_time_ns": sum(
                    sample["gpu_time_ns"] for sample in gpu_samples if sample["available"]
                ) if timed else None,
                "warmup": {"count": 0, "excluded_from_steady_state": True},
                "transfer_time_ns": 0,
                "transfer_time_semantics": (
                    "zero means transfer activity is excluded from steady-state timing; "
                    "inspect counters for observed activity"
                ),
                "scope": "gemm",
                "scope_labels": ["gemm"],
                "host_fence_wait_seconds": timing[3],
                "dispatches": counters[0],
                "submissions": submissions,
                "completions": completions,
                "waits": waits,
                "fallbacks": counters[3],
            }
        )
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--device", default=None)
    arguments = parser.parse_args()
    output = arguments.output or Path("vulkan_gemm_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if multiprocessing.current_process().name != "MainProcess":
        _child(output)
        return
    context = multiprocessing.get_context("spawn")
    process = context.Process(
        target=_child, args=(output, arguments.device), name="vulkan-gemm"
    )
    process.start()
    process.join()
    if process.exitcode != 0:
        raise SystemExit(process.exitcode)
    print(output.read_text())


if __name__ == "__main__":
    main()
