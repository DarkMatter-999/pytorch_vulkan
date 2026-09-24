import importlib.util
import subprocess
import sys
import textwrap
import os
from pathlib import Path

import pytest
import torch

import pytorch_vulkan


def _training_benchmark_module():
    script = (
        Path(__file__).resolve().parents[2] / "tools" / "vulkan_training_benchmark.py"
    )
    spec = importlib.util.spec_from_file_location("vulkan_gemm_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _gemm_benchmark_module():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vulkan_gemm_benchmark", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def timestamp_queries():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    if not pytorch_vulkan._C.timestamp_queries_supported():
        reason = pytorch_vulkan._C.timestamp_query_support_reason()
        pytest.skip(f"Vulkan timestamp queries unsupported: {reason}")
    return "vk:0"


def _assert_completed_sample(sample, scope):
    assert sample["scope"] == scope
    assert sample["available"] is True
    assert sample["gpu_time_ns"] >= 0
    assert sample["submission_id"] > 0


def test_gemm_timing_is_correlated_with_completed_submission(timestamp_queries):
    left = torch.randn(5, 7).to(timestamp_queries)
    right = torch.randn(7, 3).to(timestamp_queries)

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.reset_gpu_timing()
    output = torch.mm(left, right)
    samples = pytorch_vulkan._C.gpu_timing_snapshot()

    assert output.shape == (5, 3)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1
    assert samples
    _assert_completed_sample(samples[-1], "gemm")


def test_training_timing_is_one_completed_scope_without_transfer_or_fallback(
    timestamp_queries,
):
    first = torch.ones(2, 8).to(timestamp_queries)
    second = torch.full((2, 8), 2.0).to(timestamp_queries)
    output = torch.empty_like(first)

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.reset_gpu_timing()
    pytorch_vulkan._C.begin_training_step()
    try:
        torch.add(first, second, out=output)
        torch.neg(output, out=output)
    finally:
        pytorch_vulkan._C.end_training_step()

    samples = pytorch_vulkan._C.gpu_timing_snapshot()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (2, 0, 0, 0)
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert samples
    _assert_completed_sample(samples[-1], "training")


def test_training_benchmark_contract_distinguishes_host_and_gpu_timing(timestamp_queries):
    benchmark = _training_benchmark_module()
    baseline = benchmark.make_model("mlp", "cpu", 17).state_dict()
    result, _, _, _ = benchmark.run(
        "mlp",
        "sync",
        "fused",
        timestamp_queries,
        0,
        1,
        1,
        3,
        17,
        baseline,
        "forward",
    )
    assert result["scope"] == "forward"
    assert set(result["scope_labels"]) == {"gemm", "operator"}
    assert result["host_total_ns"][0] > 0
    assert result["gpu_time_ns"][0] >= 0
    assert result["transfer_time_ns"] == [0]
    assert result["warmup"] == {"count": 0, "excluded_from_steady_state": True}


def test_timestamp_ring_reuses_slots_after_completed_submissions(timestamp_queries):
    pytorch_vulkan._C.reset_gpu_timing()
    submission_ids = []
    for _ in range(5):
        left = torch.ones(2, 8).to(timestamp_queries)
        right = torch.full((2, 8), 2.0).to(timestamp_queries)
        torch.add(left, right)
        samples = pytorch_vulkan._C.gpu_timing_snapshot()
        assert samples
        sample = samples[-1]
        _assert_completed_sample(sample, "operator")
        submission_ids.append(sample["submission_id"])
    assert submission_ids == sorted(set(submission_ids))
    assert len(submission_ids) == 5


def test_pending_timing_uses_bounded_context_owned_query_ring(timestamp_queries):
    first = torch.ones(2, 8).to(timestamp_queries)
    second = torch.full((2, 8), 2.0).to(timestamp_queries)
    output = torch.empty_like(first)
    pytorch_vulkan._C.begin_training_step()
    try:
        torch.add(first, second, out=output)
        capacity, in_use, supported, quarantined = (
            pytorch_vulkan._C.timestamp_query_snapshot()
        )
        assert (capacity, supported, quarantined) == (20, True, False)
        assert in_use == 2
    finally:
        pytorch_vulkan._C.cancel_training_step()


def test_unavailable_timing_is_explicit_and_not_zero_duration():
    benchmark = _training_benchmark_module()
    assert benchmark.timing_status(False, "timestamp queries unavailable") == {
        "status": "unsupported",
        "reason": "timestamp queries unavailable",
    }
    assert benchmark.timing_fields(False, "timestamp queries unavailable") == {
        "gpu_time_ns": None,
        "timing_status": "unsupported",
        "timing_reason": "timestamp queries unavailable",
    }
    assert benchmark.validate_timing_samples([], {"training"}, 3, 1) == (
        False,
        "no completed timestamp sample is available",
    )
    assert benchmark.validate_timing_samples(
        [{"available": True, "scope": "training", "submission_id": 4, "gpu_time_ns": 0}],
        {"training"},
        3,
        1,
    ) == (True, "available")
    assert benchmark.validate_timing_samples(
        [{"available": True, "scope": "operator", "submission_id": 4, "gpu_time_ns": 1}],
        {"training"},
        3,
        1,
    ) == (False, "timestamp sample scope is unexpected")


def test_training_benchmark_reports_unsupported_timing_without_fake_zero(
    timestamp_queries, monkeypatch
):
    benchmark = _training_benchmark_module()
    monkeypatch.setattr(benchmark._C, "timestamp_queries_supported", lambda: False)
    monkeypatch.setattr(
        benchmark._C,
        "timestamp_query_support_reason",
        lambda: "test device has no timestamp queries",
    )
    baseline = benchmark.make_model("mlp", "cpu", 17).state_dict()
    result, _, _, _ = benchmark.run(
        "mlp", "sync", "fused", timestamp_queries, 0, 1, 1, 3, 17, baseline, "forward"
    )
    assert result["timing_status"] == "unsupported"
    assert result["timing_reason"] == "test device has no timestamp queries"
    assert result["gpu_time_ns"] is None
    assert result["transfer_time_semantics"].startswith("zero means")


def test_training_benchmark_downgrades_supported_timing_without_valid_sample(
    timestamp_queries, monkeypatch
):
    benchmark = _training_benchmark_module()
    monkeypatch.setattr(benchmark._C, "timestamp_queries_supported", lambda: True)
    monkeypatch.setattr(benchmark._C, "timestamp_query_support_reason", lambda: "supported")
    monkeypatch.setattr(benchmark._C, "gpu_timing_snapshot", lambda: [])
    baseline = benchmark.make_model("mlp", "cpu", 17).state_dict()
    result, _, _, _ = benchmark.run(
        "mlp", "sync", "fused", timestamp_queries, 0, 1, 1, 3, 17, baseline, "forward"
    )
    assert result["timing_status"] == "unavailable"
    assert result["timing_reason"] == "no completed timestamp sample is available"
    assert result["gpu_time_ns"] is None


def test_gemm_timing_validation_requires_scope_availability_and_submission():
    benchmark = _gemm_benchmark_module()
    valid = [{"available": True, "scope": "gemm", "submission_id": 4, "gpu_time_ns": 9}]
    assert benchmark.validate_timing_samples(valid, 4) == (True, "available")
    assert benchmark.validate_timing_samples([], 4) == (
        False,
        "no completed timestamp sample is available",
    )
    assert benchmark.validate_timing_samples(
        [{"available": True, "scope": "operator", "submission_id": 4}], 4
    ) == (False, "timestamp sample scope is not gemm")


def test_device_loss_quarantines_timing_context_and_rejects_new_work(timestamp_queries):
    script = textwrap.dedent(
        """
        import torch
        import pytorch_vulkan

        first = torch.ones(2, 8).to("vk:0")
        second = torch.full((2, 8), 2.0).to("vk:0")
        output = torch.empty_like(first)
        pytorch_vulkan._C.begin_training_step()
        torch.add(first, second, out=output)
        assert pytorch_vulkan._C.timestamp_query_snapshot() == (20, 2, True, False)
        pytorch_vulkan._C.test_inject_device_loss()
        assert pytorch_vulkan._C.timestamp_query_snapshot() == (20, 0, True, True)
        try:
            torch.ones(1).to("vk:0")
        except RuntimeError as error:
            assert "Vulkan device lost" in str(error)
        else:
            raise AssertionError("device-loss context accepted new work")
        assert pytorch_vulkan._C.gpu_timing_snapshot() == []
        print("timing-device-loss-ok")
        """
    )
    environment = os.environ.copy()
    # Keep the synthetic loss child isolated from validation-layer driver aborts.
    environment.pop("VK_INSTANCE_LAYERS", None)
    module_root = Path(pytorch_vulkan.__file__).resolve().parent.parent
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(module_root), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "timing-device-loss-ok" in result.stdout


def test_runtime_foundation_benchmark_schema_is_machine_readable():
    benchmark_path = Path(__file__).resolve().parents[2] / "tools" / "benchmark_vulkan_runtime_foundation.py"
    spec = importlib.util.spec_from_file_location("runtime_foundation_benchmark", benchmark_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.WORKLOAD_SHAPES == {
        "small": (16, 32, 8),
        "medium": (64, 128, 32),
        "large": (256, 512, 128),
        "skinny": (8, 1024, 8),
        "irregular": (37, 59, 13),
    }
