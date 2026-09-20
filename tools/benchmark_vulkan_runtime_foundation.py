#!/usr/bin/env python3
"""Measure runtime-foundation workload evidence without performance claims."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Make the prescribed standalone invocation independent of the caller's
# PYTHONPATH while still loading the build-local extension.
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "build"), str(ROOT)]

import torch

import pytorch_vulkan


WORKLOAD_SHAPES = {
    "small": (16, 32, 8),
    "medium": (64, 128, 32),
    "large": (256, 512, 128),
    "skinny": (8, 1024, 8),
    "irregular": (37, 59, 13),
}


def _device_probe(device):
    if not pytorch_vulkan.is_available():
        return False, "no suitable Vulkan device is available"
    try:
        torch.ones(1).to(device)
    except Exception as error:  # pragma: no cover - hardware-dependent
        return False, str(error)
    return True, None


def _run_shape(name, shape, device, repetitions):
    m, k, n = shape
    torch.manual_seed(100 + len(name))
    left = torch.randn(m, k).to(device)
    right = torch.randn(k, n).to(device)
    addend = torch.randn(m, n).to(device)
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.reset_gpu_timing()
    pytorch_vulkan._C.reset_descriptor_resource_counters()
    resource_before = pytorch_vulkan._C.live_resource_snapshot()
    service_before = pytorch_vulkan._C.shared_service_snapshot()
    samples = []
    operation_sample_counts = []
    host_latencies = []
    peak_resources = list(resource_before)
    peak_service = {
        "descriptor_pending": service_before["descriptor_pending"],
        "descriptor_quarantined": service_before["descriptor_quarantined"],
        "pipeline_pending_destructions": service_before["pipeline_pending_destructions"],
        "timestamp_quarantined": int(pytorch_vulkan._C.timestamp_query_snapshot()[3]),
    }

    def observe_service_state():
        snapshot = pytorch_vulkan._C.shared_service_snapshot()
        peak_service["descriptor_pending"] = max(
            peak_service["descriptor_pending"], snapshot["descriptor_pending"]
        )
        peak_service["descriptor_quarantined"] = max(
            peak_service["descriptor_quarantined"], snapshot["descriptor_quarantined"]
        )
        peak_service["pipeline_pending_destructions"] = max(
            peak_service["pipeline_pending_destructions"],
            snapshot["pipeline_pending_destructions"],
        )
        peak_service["timestamp_quarantined"] = max(
            peak_service["timestamp_quarantined"],
            int(pytorch_vulkan._C.timestamp_query_snapshot()[3]),
        )

    for _ in range(repetitions):
        start = time.perf_counter_ns()
        pytorch_vulkan._C.reset_gpu_timing()
        product = torch.mm(left, right)
        operation_samples = pytorch_vulkan._C.gpu_timing_snapshot()
        samples.extend(operation_samples)
        operation_sample_counts.append(len(operation_samples))
        observe_service_state()
        pytorch_vulkan._C.reset_gpu_timing()
        torch.add(product, addend)
        operation_samples = pytorch_vulkan._C.gpu_timing_snapshot()
        samples.extend(operation_samples)
        operation_sample_counts.append(len(operation_samples))
        observe_service_state()
        pytorch_vulkan._C.reset_gpu_timing()
        torch.sum(product, dim=1)
        operation_samples = pytorch_vulkan._C.gpu_timing_snapshot()
        samples.extend(operation_samples)
        operation_sample_counts.append(len(operation_samples))
        observe_service_state()
        host_latencies.append(time.perf_counter_ns() - start)
        current_resources = pytorch_vulkan._C.live_resource_snapshot()
        peak_resources = [max(before, current) for before, current in zip(peak_resources, current_resources)]
    resource_after = pytorch_vulkan._C.live_resource_snapshot()
    service_after = pytorch_vulkan._C.shared_service_snapshot()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    timing_supported = pytorch_vulkan._C.timestamp_queries_supported()
    timing_reason = pytorch_vulkan._C.timestamp_query_support_reason()
    valid_samples = [sample for sample in samples if sample.get("available")]
    service_metrics = {
        "pipeline_entries_delta": service_after["pipeline_entries"] - service_before["pipeline_entries"],
        "pipeline_hits_delta": service_after["pipeline_hits"] - service_before["pipeline_hits"],
        "pipeline_misses_delta": service_after["pipeline_misses"] - service_before["pipeline_misses"],
        "pipeline_evictions_delta": service_after["pipeline_evictions"] - service_before["pipeline_evictions"],
        "shader_hits_delta": service_after["shader_hits"] - service_before["shader_hits"],
        "shader_misses_delta": service_after["shader_misses"] - service_before["shader_misses"],
        "descriptor_reuses_delta": service_after["descriptor_reuses"] - service_before["descriptor_reuses"],
    }
    return {
        "shape": list(shape),
        "warmup": {"count": 0, "excluded_from_steady_state": True},
        "host_latency_ns": host_latencies,
        "gpu_timestamp_intervals_ns": [sample["gpu_time_ns"] for sample in valid_samples],
        "gpu_timestamp_sample_count": len(valid_samples),
        "operation_sample_counts": operation_sample_counts,
        "timing_status": "available" if timing_supported and valid_samples else "unavailable",
        "timing_reason": "timestamp queries completed" if timing_supported else timing_reason,
        "dispatches": counters[0],
        "transfers": counters[1],
        "explicit_transfers": counters[2],
        "fallbacks": counters[3],
        "submissions": pytorch_vulkan._C.compute_submitted_count(),
        "completions": pytorch_vulkan._C.compute_completed_count(),
        "waits": pytorch_vulkan._C.compute_wait_count(),
        "resource_bounds": {
            "before": list(resource_before),
            "after": list(resource_after),
            "peak": peak_resources,
            "peak_live_allocations": peak_resources[6],
            "descriptor_pool_limit": service_after["descriptor_pool_limit"],
            "pending_transfers_after": resource_after[4],
            "pending_compute_after": resource_after[5],
            "descriptor_pending_after": service_after["descriptor_pending"],
            "descriptor_quarantined_after": service_after["descriptor_quarantined"],
            "pipeline_pending_destructions_after": service_after["pipeline_pending_destructions"],
            "timestamp_quarantined_after": pytorch_vulkan._C.timestamp_query_snapshot()[3],
            "peak_service": peak_service,
        },
        "cache_metrics": service_metrics,
    }


def _qualification_failures(workloads, repetitions, timing_supported):
    failures = []
    if set(workloads) != set(WORKLOAD_SHAPES):
        failures.append("required workload shapes are incomplete")
    for name, evidence in workloads.items():
        expected_dispatches = repetitions * 3
        if len(evidence["host_latency_ns"]) != repetitions:
            failures.append(f"{name}: host sample count is inconsistent")
        if evidence["dispatches"] != expected_dispatches:
            failures.append(f"{name}: dispatch count is inconsistent")
        if evidence["submissions"] != expected_dispatches:
            failures.append(f"{name}: submission count is inconsistent")
        if evidence["completions"] != expected_dispatches:
            failures.append(f"{name}: completion count is inconsistent")
        if evidence["waits"] != expected_dispatches:
            failures.append(f"{name}: wait count is inconsistent")
        if evidence["transfers"] or evidence["explicit_transfers"] or evidence["fallbacks"]:
            failures.append(f"{name}: transfer or fallback activity was observed")
        if not timing_supported:
            failures.append(f"{name}: GPU timing is unsupported ({evidence['timing_reason']})")
        elif evidence["timing_status"] != "available":
            failures.append(f"{name}: GPU timing is unavailable")
        if timing_supported and evidence["gpu_timestamp_sample_count"] != expected_dispatches:
            failures.append(f"{name}: GPU sample count is inconsistent")
        if evidence["operation_sample_counts"] != [1] * expected_dispatches:
            failures.append(f"{name}: operation sample counts are inconsistent")
        bounds = evidence["resource_bounds"]
        if bounds["peak"][0] > bounds["descriptor_pool_limit"] or bounds["peak"][1] > bounds["descriptor_pool_limit"] * 64:
            failures.append(f"{name}: descriptor resource bound exceeded")
        if any(bounds[key] for key in (
            "pending_transfers_after", "pending_compute_after",
            "descriptor_pending_after", "descriptor_quarantined_after",
            "pipeline_pending_destructions_after", "timestamp_quarantined_after",
        )):
            failures.append(f"{name}: deferred or quarantined resources remain")
        if any(value < 0 for value in evidence["cache_metrics"].values()):
            failures.append(f"{name}: cache counter delta is negative")
    return failures


def run(output, device="vk:0", repetitions=3):
    if repetitions <= 0:
        raise ValueError("repetitions must be greater than zero")
    available, reason = _device_probe(device)
    result = {
        "schema_version": 1,
        "qualification": "phase-1-runtime-foundation",
        "device": device,
        "warmups_excluded": True,
        "performance_claims": "timing evidence only; no faster-than-CPU claim",
    }
    if not available:
        result.update({"status": "skipped", "reason": reason, "workloads": {}})
    else:
        workloads = {}
        try:
            for name, shape in WORKLOAD_SHAPES.items():
                workloads[name] = _run_shape(name, shape, device, repetitions)
            timing_supported = pytorch_vulkan._C.timestamp_queries_supported()
            failures = _qualification_failures(workloads, repetitions, timing_supported)
            result.update({
                "status": "qualified" if not failures else "failed",
                "reason": "supported workload evidence completed"
                if not failures else "; ".join(failures),
                "workloads": workloads,
            })
        except Exception as error:  # pragma: no cover - hardware-dependent
            result.update({
                "status": "failed",
                "reason": f"benchmark execution failed: {error}",
                "workloads": workloads,
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default=os.environ.get("VULKAN_DEVICE", "vk:0"))
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    try:
        result = run(args.output, args.device, args.repetitions)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps({"status": result["status"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
