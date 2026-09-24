#!/usr/bin/env python3
"""Paired CPU/Vulkan microbenchmarks for registered Vulkan operators."""

from __future__ import annotations

import math
import argparse
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path


SCHEMA_VERSION = 1
TIMING_SCOPES = {"operator", "gemm", "training"}


def _timed(call):
    started = time.perf_counter_ns()
    output = call()
    return output, time.perf_counter_ns() - started


def _aggregate_operator_gpu_time(samples):
    """Sum top-level submission scopes, excluding nested diagnostic intervals."""
    aggregate = [sample for sample in samples if sample.get("scope") in TIMING_SCOPES]
    return sum(sample["gpu_time_ns"] for sample in aggregate) if aggregate else None


def _run_vulkan(case, payload):
    api = case["vulkan_api"]
    previous_samples = api.gpu_timing_snapshot()
    submission_floor = max(
        (sample.get("submission_id", 0) for sample in previous_samples), default=0
    )
    api.reset_execution_counters()
    api.reset_timing()
    api.reset_gpu_timing()
    api.reset_descriptor_resource_counters()
    output, host_ns = _timed(lambda: case["vulkan_call"](payload))
    counters = api.execution_counter_snapshot()
    samples = api.gpu_timing_snapshot()
    current = [
        sample
        for sample in samples
        if sample.get("available") and sample.get("submission_id", 0) > submission_floor
    ]
    completed = api.compute_completed_count()
    aggregate_samples = [
        sample
        for sample in current
        if sample.get("scope") in case.get("expected_scopes", TIMING_SCOPES)
        and sample.get("scope") in TIMING_SCOPES
    ]
    valid = bool(aggregate_samples) and completed > 0 and all(
        sample.get("scope") in case.get("expected_scopes", TIMING_SCOPES)
        and sample.get("gpu_time_ns", -1) >= 0
        for sample in aggregate_samples
    )
    gpu_ns = _aggregate_operator_gpu_time(current) if valid else None
    return output, host_ns, {
        "gpu_time_ns": gpu_ns,
        "timing_valid": valid,
        "timing_reason": "available" if valid else "no valid completed timestamp for this operator",
        "counters": {
            "dispatches": counters[0],
            "transfers": counters[1],
            "explicit_transfers": counters[2],
            "fallbacks": counters[3],
            "submissions": api.compute_submitted_count(),
            "completions": completed,
            "waits": api.compute_wait_count(),
            "descriptor_pools": api.descriptor_pool_creation_count(),
            "descriptor_sets": api.descriptor_set_allocation_count(),
            "descriptor_reuses": api.descriptor_set_reuse_count(),
        },
        "timestamp_samples": current,
    }


def measure_case(case, warmups=2, repetitions=10, order_seed=1729):
    if warmups < 0 or repetitions < 1:
        raise ValueError("warmups must be non-negative and repetitions must be positive")
    api = case.get("vulkan_api")
    timing_supported = api.timestamp_queries_supported()
    timing_reason = api.timestamp_query_support_reason()

    for _ in range(warmups):
        cpu_payload, vulkan_payload = case["prepare_pair"](case.get("seed", 1729))
        case["cpu_call"](cpu_payload)
        case["vulkan_call"](vulkan_payload)

    cpu_samples = []
    vulkan_host_samples = []
    vulkan_gpu_samples = []
    counter_samples = []
    timestamp_samples = []
    parity = None
    for repetition in range(repetitions):
        cpu_payload, vulkan_payload = case["prepare_pair"](case.get("seed", 1729))
        cpu_result = None
        vulkan_result = None
        vulkan_timing = None
        cpu_first = (repetition + order_seed) % 2 == 0
        operations = ("cpu", "vulkan") if cpu_first else ("vulkan", "cpu")
        for operation in operations:
            if operation == "cpu":
                cpu_result, elapsed = _timed(lambda: case["cpu_call"](cpu_payload))
                cpu_samples.append(elapsed)
            else:
                vulkan_result, elapsed, vulkan_timing = _run_vulkan(case, vulkan_payload)
                vulkan_host_samples.append(elapsed)
                vulkan_gpu_samples.append(vulkan_timing["gpu_time_ns"])
                counter_samples.append(vulkan_timing["counters"])
                timestamp_samples.append(vulkan_timing["timestamp_samples"])
        parity = case["compare"](cpu_result, vulkan_result, 3e-3, 3e-3)

    gpu_not_applicable = case.get("gpu_timing_not_applicable", False)
    gpu_valid = timing_supported and all(value is not None for value in vulkan_gpu_samples)
    result = {
        "schema_version": SCHEMA_VERSION,
        "schema": case["schema"],
        "family": case["family"],
        "phase": case["phase"],
        "seed": case.get("seed", 1729),
        "warmups": warmups,
        "repetitions": repetitions,
        "input_metadata": case["input_metadata"],
        "cpu_time_ns": {
            "samples": cpu_samples,
            "mean": statistics.mean(cpu_samples),
            "median": statistics.median(cpu_samples),
        },
        "vulkan_host_time_ns": {
            "samples": vulkan_host_samples,
            "mean": statistics.mean(vulkan_host_samples),
            "median": statistics.median(vulkan_host_samples),
        },
        "vulkan_gpu_time_ns": {
            "status": "not_applicable" if gpu_not_applicable else "available" if gpu_valid else "unavailable",
            "reason": "metadata-only operator" if gpu_not_applicable else "completed timestamps" if gpu_valid else timing_reason,
            "samples": None if gpu_not_applicable else vulkan_gpu_samples if gpu_valid else None,
            "mean": None if gpu_not_applicable else statistics.mean(vulkan_gpu_samples) if gpu_valid else None,
            "median": None if gpu_not_applicable else statistics.median(vulkan_gpu_samples) if gpu_valid else None,
        },
        "counters": counter_samples,
        "timestamp_samples": timestamp_samples,
        "parity": parity,
        "status": case.get("status", "measurable"),
    }
    result["qualification"] = qualify_case(result, rtol=3e-3, atol=3e-3)["qualification"]
    return result


def qualify_case(result, rtol, atol):
    if result.get("status") == "non_comparable":
        qualification = "non_comparable"
        reason = result.get("reason", "operator has no fair standalone comparison")
    elif any(sample.get("fallbacks", 0) for sample in result.get("counters", [])):
        qualification = "fallback_detected"
        reason = "one or more Vulkan repetitions used fallback"
    elif not result.get("parity", {}).get("passed", False):
        qualification = "parity_failed"
        reason = f"output or gradient parity failed at rtol={rtol}, atol={atol}"
    elif result.get("vulkan_gpu_time_ns", {}).get("status") not in {"available", "not_applicable"}:
        qualification = "timing_unavailable"
        reason = result.get("vulkan_gpu_time_ns", {}).get("reason", "GPU timestamps unavailable")
    else:
        qualification = "qualified"
        reason = "parity passed, no fallbacks, complete GPU timing"
    return {"qualification": qualification, "reason": reason}


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be exactly {SCHEMA_VERSION}")
    required = {"device", "warmups", "repetitions", "cpu_threads", "results"}
    missing = required - artifact.keys()
    if missing:
        raise ValueError(f"artifact missing fields: {sorted(missing)}")
    if not isinstance(artifact["device"], dict) or not artifact["device"].get("name"):
        raise ValueError("device.name is required")
    if not isinstance(artifact["cpu_threads"], dict):
        raise ValueError("cpu_threads must be an object")
    if not isinstance(artifact["results"], list):
        raise ValueError("results must be a list")
    warmups = artifact["warmups"]
    repetitions = artifact["repetitions"]
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 0:
        raise ValueError("warmups must be a non-negative integer")
    if not isinstance(repetitions, int) or isinstance(repetitions, bool) or repetitions < 1:
        raise ValueError("repetitions must be a positive integer")
    for index, result in enumerate(artifact["results"]):
        for field in ("schema", "family", "phase", "input_metadata", "cpu_time_ns", "vulkan_host_time_ns", "vulkan_gpu_time_ns", "counters", "parity", "qualification"):
            if field not in result:
                raise ValueError(f"results[{index}].{field} is required")
        for timing_field in ("cpu_time_ns", "vulkan_host_time_ns"):
            timing = result[timing_field]
            samples = timing.get("samples")
            if not isinstance(samples, list) or len(samples) != repetitions:
                raise ValueError(f"results[{index}].{timing_field}.samples must match repetitions")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in samples):
                raise ValueError(f"results[{index}].{timing_field} samples must be finite and non-negative")
            if not math.isclose(timing.get("mean", math.nan), statistics.mean(samples), rel_tol=1e-6, abs_tol=1.0):
                raise ValueError(f"results[{index}].{timing_field}.mean does not match samples")
            if not math.isclose(timing.get("median", math.nan), statistics.median(samples), rel_tol=1e-6, abs_tol=1.0):
                raise ValueError(f"results[{index}].{timing_field}.median does not match samples")
        gpu_timing = result["vulkan_gpu_time_ns"]
        if gpu_timing.get("status") == "available":
            gpu_samples = gpu_timing.get("samples")
            if not isinstance(gpu_samples, list) or len(gpu_samples) != repetitions:
                raise ValueError(f"results[{index}].vulkan_gpu_time_ns.samples must match repetitions")
            if any(not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 for value in gpu_samples):
                raise ValueError(f"results[{index}].vulkan_gpu_time_ns samples must be finite and non-negative")
            if not math.isclose(gpu_timing.get("mean", math.nan), statistics.mean(gpu_samples), rel_tol=1e-6, abs_tol=1.0):
                raise ValueError(f"results[{index}].vulkan_gpu_time_ns.mean does not match samples")
            if not math.isclose(gpu_timing.get("median", math.nan), statistics.median(gpu_samples), rel_tol=1e-6, abs_tol=1.0):
                raise ValueError(f"results[{index}].vulkan_gpu_time_ns.median does not match samples")
        elif gpu_timing.get("status") not in {"unavailable", "not_applicable"}:
            raise ValueError(f"results[{index}].vulkan_gpu_time_ns.status is invalid")
        elif any(gpu_timing.get(key) is not None for key in ("samples", "mean", "median")):
            raise ValueError(f"results[{index}].unavailable GPU timing must not contain values")
        counters = result["counters"]
        if not isinstance(counters, list) or len(counters) != repetitions:
            raise ValueError(f"results[{index}].counters must match repetitions")
        for sample in counters:
            for counter in (
                "dispatches", "transfers", "explicit_transfers", "fallbacks",
                "submissions", "completions", "waits",
            ):
                value = sample.get(counter)
                if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                    raise ValueError(f"results[{index}].counters.{counter} must be a non-negative integer")
        if not isinstance(result["parity"].get("passed"), bool):
            raise ValueError(f"results[{index}].parity.passed must be boolean")
        if result["qualification"] not in {
            "qualified", "parity_failed", "fallback_detected", "timing_unavailable", "non_comparable"
        }:
            raise ValueError(f"results[{index}].qualification is invalid")
    return artifact


def build_scorecards(results):
    eligible = [
        result for result in results
        if result.get("qualification") == "qualified"
        and result.get("cpu_time_ns", {}).get("mean", 0) > 0
    ]

    def rows(field):
        ranked = []
        for result in eligible:
            timing = result["vulkan_gpu_time_ns"] if field == "gpu" else result["vulkan_host_time_ns"]
            if field == "gpu" and timing.get("status") != "available":
                continue
            value = timing.get("mean")
            if value is None:
                continue
            cpu = result["cpu_time_ns"]["mean"]
            row = {
                "schema": result["schema"], "family": result["family"],
                "phase": result.get("phase", "forward"),
                "shape_profile": result.get("input_metadata", {}).get("shape_profile"),
                "vulkan_to_cpu_ratio": value / cpu,
                "cpu_time_ns": cpu, "vulkan_time_ns": value,
            }
            if field == "host":
                gpu = result["vulkan_gpu_time_ns"]
                row["host_minus_gpu_ns"] = value - gpu["mean"] if gpu.get("status") == "available" else None
                row["host_minus_gpu_label"] = "diagnostic estimate"
            ranked.append(row)
        return sorted(ranked, key=lambda row: (-row["vulkan_to_cpu_ratio"], row["family"], row["schema"], row["phase"], row["shape_profile"] or ""))

    return {"host_ratio": rows("host"), "gpu_ratio": rows("gpu")}


def _device_name():
    try:
        output = subprocess.run(["vulkaninfo", "--summary"], capture_output=True, text=True, timeout=10, check=True).stdout
        for line in output.splitlines():
            if "deviceName" in line:
                return line.split("=", 1)[-1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return "vk:0 (device name unavailable via vulkaninfo)"


def _render_summary(cards):
    lines = ["Vulkan operator benchmark rankings", "", "Host time / CPU time (host_minus_gpu is diagnostic):"]
    for row in cards["host_ratio"]:
        lines.append(f"{row['family']} {row['schema']} [{row['phase']}, {row['shape_profile']}]: {row['vulkan_to_cpu_ratio']:.4f}x; host-GPU={row['host_minus_gpu_ns']}")
    lines.extend(["", "GPU time / CPU time:"])
    for row in cards["gpu_ratio"]:
        lines.append(f"{row['family']} {row['schema']} [{row['phase']}, {row['shape_profile']}]: {row['vulkan_to_cpu_ratio']:.4f}x")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="vk:0")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family")
    parser.add_argument("--schema")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.repetitions < 1:
        parser.error("warmups must be non-negative and repetitions positive")
    if args.device != "vk:0":
        parser.error("this benchmark build supports only --device vk:0")
    root = Path(__file__).resolve().parents[1]
    from tools.vulkan_operator_benchmark_cases import iter_cases, load_operator_cases

    try:
        catalog = load_operator_cases(root)
        entries = [e for e in catalog["entries"] if (not args.family or e["family"] == args.family) and (not args.schema or e["schema"] == args.schema)]
        if args.schema and not entries:
            raise ValueError(f"unknown schema selection: {args.schema}")
        if args.validate_only:
            print(f"Catalog validated: {len(catalog['entries'])} registrations")
            return 0
        import torch
        import pytorch_vulkan
        if not pytorch_vulkan.is_available():
            raise RuntimeError("Vulkan device is unavailable")
        from tools.vulkan_operator_benchmark_cases import _runtime
        _, vulkan_module, _ = _runtime()
        extension = vulkan_module._C
        results = []
        for case in iter_cases({"entries": entries}, profiles=("small", "large")):
            if case.get("status") == "non_comparable":
                results.append({
                    "schema_version": SCHEMA_VERSION, "schema": case["schema"], "family": case["family"],
                    "phase": case["phase"], "input_metadata": {}, "status": "non_comparable",
                    "cpu_time_ns": {"samples": [0] * args.repetitions, "mean": 0, "median": 0},
                    "vulkan_host_time_ns": {"samples": [0] * args.repetitions, "mean": 0, "median": 0},
                    "vulkan_gpu_time_ns": {"status": "unavailable", "reason": case["reason"], "samples": None, "mean": None, "median": None},
                    "counters": [{key: 0 for key in ("dispatches", "transfers", "explicit_transfers", "fallbacks", "submissions", "completions", "waits")} for _ in range(args.repetitions)],
                    "parity": {"passed": False, "reason": case["reason"]}, "qualification": "non_comparable",
                    "qualification_reason": case["reason"],
                })
            else:
                results.append(measure_case(case, warmups=args.warmups, repetitions=args.repetitions))
    except Exception as error:
        parser.exit(2, f"benchmark failed: {error}\n")
    try:
        cpu_threads = {"intraop": torch.get_num_threads(), "interop": torch.get_num_interop_threads(),
                       "omp_num_threads": os.environ.get("OMP_NUM_THREADS")}
        artifact = {
            "schema_version": SCHEMA_VERSION,
            "device": {"requested": args.device, "name": _device_name(), "timestamp_queries_supported": extension.timestamp_queries_supported(), "timestamp_query_reason": extension.timestamp_query_support_reason()},
            "environment": {
                "os": platform.platform(), "python": platform.python_version(), "pytorch": torch.__version__,
                "source_revision": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True).stdout.strip(),
                "working_tree_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, check=True).stdout.strip()),
            },
            "cpu_threads": cpu_threads, "warmups": args.warmups, "repetitions": args.repetitions,
            "results": results, "scorecards": build_scorecards(results),
        }
        validate_artifact(artifact)
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
        path.with_suffix(path.suffix + ".summary.txt").write_text(_render_summary(artifact["scorecards"]))
    except Exception as error:
        parser.exit(2, f"artifact generation failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
