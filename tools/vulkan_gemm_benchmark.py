#!/usr/bin/env python3
"""Run the production GEMM large-shape workload in an isolated process."""

import argparse
import json
import multiprocessing
import time
from pathlib import Path


LEFT_SHAPE = (2048, 4096)
RIGHT_SHAPE = (4096, 2048)


def _probe_device(torch, pytorch_vulkan):
    if not pytorch_vulkan.is_available():
        return None, "no suitable Vulkan device is available"
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        return None, f"Vulkan tensor setup is unavailable: {error}"
    return device, None


def _child(result_path):
    import torch
    import pytorch_vulkan

    device, skip_reason = _probe_device(torch, pytorch_vulkan)
    if skip_reason:
        result_path.write_text(json.dumps({"status": "skipped", "reason": skip_reason}))
        return

    torch.manual_seed(47)
    left = torch.randn(*LEFT_SHAPE).to(device)
    right = torch.randn(*RIGHT_SHAPE).to(device)
    pytorch_vulkan._C.reset_execution_counters()
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
    if counters != (1, 0, 0, 0) or (submissions, completions, waits) != (1, 1, 1):
        raise RuntimeError(
            "large GEMM counter contract failed: "
            f"counters={counters}, submissions={submissions}, "
            f"completions={completions}, waits={waits}"
        )
    result_path.write_text(
        json.dumps(
            {
                "status": "completed",
                "shape": list(expected_shape),
                "seconds": elapsed,
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
    arguments = parser.parse_args()
    output = arguments.output or Path("vulkan_gemm_benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    if multiprocessing.current_process().name != "MainProcess":
        _child(output)
        return
    context = multiprocessing.get_context("spawn")
    process = context.Process(target=_child, args=(output,), name="vulkan-gemm")
    process.start()
    process.join()
    if process.exitcode != 0:
        raise SystemExit(process.exitcode)
    print(output.read_text())


if __name__ == "__main__":
    main()
