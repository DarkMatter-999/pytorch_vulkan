#!/usr/bin/env python3
"""Run the production GEMM large-shape workload in an isolated process."""

import argparse
import json
import multiprocessing
import time
import math
import statistics
from pathlib import Path


LEFT_SHAPE = (2048, 4096)
RIGHT_SHAPE = (4096, 2048)
SCHEMA_VERSION = 1
EXPECTED_OUTPUT_SHAPE = [LEFT_SHAPE[0], RIGHT_SHAPE[1]]
EXPECTED_ARITHMETIC_OPERATIONS = 2 * LEFT_SHAPE[0] * RIGHT_SHAPE[1] * LEFT_SHAPE[1]


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


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"GEMM schema_version must be exactly {SCHEMA_VERSION}")
    required = {
        "status", "timing_status", "timing_reason", "timing_source", "seed",
        "repetitions", "shape", "seconds", "host_total_ns", "gpu_time_ns",
        "host_time", "gpu_time", "arithmetic_operations", "effective_tflops",
        "dispatches", "submissions", "completions", "waits", "fallbacks",
        "transfer_count", "explicit_transfers", "vulkan_copies", "validation",
        "cpu_parity",
    }
    missing = required - artifact.keys()
    if missing:
        raise ValueError(f"GEMM artifact missing fields: {sorted(missing)}")
    if artifact["status"] != "blocked":
        raise ValueError("GEMM artifact status must remain blocked without parity")
    if artifact["seed"] != 47 or isinstance(artifact["seed"], bool):
        raise ValueError("GEMM seed must be exactly 47")
    if artifact["repetitions"] != 1 or isinstance(artifact["repetitions"], bool):
        raise ValueError("GEMM repetitions must be exactly 1")
    if artifact["shape"] != EXPECTED_OUTPUT_SHAPE:
        raise ValueError(f"GEMM shape must be exactly {EXPECTED_OUTPUT_SHAPE}")
    if artifact["arithmetic_operations"] != EXPECTED_ARITHMETIC_OPERATIONS:
        raise ValueError("GEMM arithmetic operation count is inconsistent with shape")
    for field in ("seconds", "host_total_ns"):
        if not isinstance(artifact[field], (int, float)) or isinstance(artifact[field], bool) or not math.isfinite(artifact[field]) or artifact[field] < 0:
            raise ValueError(f"GEMM {field} must be finite and non-negative")
    for field in ("dispatches", "submissions", "completions", "waits", "fallbacks", "transfer_count", "explicit_transfers", "vulkan_copies"):
        if not isinstance(artifact[field], int) or isinstance(artifact[field], bool) or artifact[field] < 0:
            raise ValueError(f"GEMM {field} counter must be a non-negative integer")
    if not (artifact["dispatches"] == artifact["submissions"] == artifact["completions"] == artifact["waits"] == 1):
        raise ValueError("GEMM dispatch/submission/completion/wait counters are inconsistent")
    if artifact["fallbacks"] != 0 or artifact["transfer_count"] != 0 or artifact["explicit_transfers"] != 0 or artifact["vulkan_copies"] != 0:
        raise ValueError("GEMM transfer, copy, and fallback counters must be zero")
    host = artifact["host_time"]
    if not isinstance(host, dict) or host.get("samples_ns") != [artifact["host_total_ns"]] or host.get("mean_ns") != artifact["host_total_ns"]:
        raise ValueError("GEMM host timing samples and mean are inconsistent")
    source = artifact["timing_source"]
    gpu = artifact["gpu_time"]
    if source == "gpu_timestamp":
        if artifact["timing_status"] != "available" or not isinstance(gpu, dict) or not isinstance(gpu.get("samples_ns"), list) or len(gpu["samples_ns"]) != 1:
            raise ValueError("GEMM GPU timing source requires one available sample")
        if gpu["mean_ns"] != gpu["samples_ns"][0] or artifact["gpu_time_ns"] != gpu["mean_ns"] or gpu["mean_ns"] <= 0:
            raise ValueError("GEMM GPU timing samples and mean are inconsistent")
        expected_tflops = artifact["arithmetic_operations"] / (gpu["mean_ns"] * 1e3)
        if not isinstance(artifact["effective_tflops"], (int, float)) or not math.isclose(artifact["effective_tflops"], expected_tflops, rel_tol=1e-9):
            raise ValueError("GEMM effective TFLOP/s is inconsistent with GPU timing")
    elif source == "unavailable":
        if artifact["timing_status"] != "unavailable" or artifact["gpu_time_ns"] is not None or artifact["effective_tflops"] is not None or artifact["gpu_time"] != {"status": "unavailable", "samples_ns": None, "mean_ns": None}:
            raise ValueError("GEMM unavailable timing source has inconsistent fields")
    else:
        raise ValueError("GEMM timing source is invalid")
    if artifact["validation"].get("status") != "blocked" or artifact["cpu_parity"].get("status") != "not_measured":
        raise ValueError("GEMM blocked artifact must explicitly report missing parity")
    if artifact.get("cpu_parity", {}).get("status") != "measured":
        raise ValueError("GEMM qualification requires measured CPU parity")
    if artifact.get("validation", {}).get("status") != "passed":
        raise ValueError("GEMM qualification requires passed validation")
    if artifact.get("submissions") != artifact.get("completions") or artifact.get("submissions") != artifact.get("waits"):
        raise ValueError("GEMM submission/completion/wait counters do not match")
    return artifact


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
    arithmetic_operations = 2 * LEFT_SHAPE[0] * RIGHT_SHAPE[1] * LEFT_SHAPE[1]
    effective_tflops = (
        arithmetic_operations / (sum(sample["gpu_time_ns"] for sample in gpu_samples if sample["available"]) * 1e3)
        if timed else None
    )
    result_path.write_text(
        json.dumps(
            {
                "schema_version": SCHEMA_VERSION,
                "status": "blocked",
                "timing_status": "available" if timed else "unavailable",
                "timing_reason": "available" if timed else timing_validation_reason,
                "timing_source": "gpu_timestamp" if timed else "unavailable",
                "seed": 47,
                "repetitions": 1,
                "shape": list(expected_shape),
                "seconds": elapsed,
                "host_total_ns": int(elapsed * 1_000_000_000),
                "gpu_time_ns": sum(
                    sample["gpu_time_ns"] for sample in gpu_samples if sample["available"]
                ) if timed else None,
                "host_time": {
                    "samples_ns": [int(elapsed * 1_000_000_000)],
                    "mean_ns": int(elapsed * 1_000_000_000),
                },
                "gpu_time": {
                    "status": "available" if timed else "unavailable",
                    "samples_ns": [sum(sample["gpu_time_ns"] for sample in gpu_samples if sample["available"])] if timed else None,
                    "mean_ns": sum(sample["gpu_time_ns"] for sample in gpu_samples if sample["available"]) if timed else None,
                },
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
                "explicit_transfers": counters[2],
                "vulkan_copies": counters[1],
                "arithmetic_operations": arithmetic_operations,
                "effective_tflops": effective_tflops,
                "validation": {
                    "status": "blocked",
                    "reason": "large GEMM payload readback is prohibited; CPU output parity is unavailable",
                },
                "cpu_parity": {
                    "status": "not_measured",
                    "reason": "payload readback is excluded from the GEMM timing contract",
                },
                "transfer_count": counters[2],
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
