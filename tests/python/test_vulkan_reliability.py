import os
import pytest
import subprocess
import sys
import textwrap

import pytorch_vulkan


VALIDATION_ERROR_MARKERS = ("vuid-", "validation error")


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return os.environ.get("VULKAN_DEVICE", "vk:0")


def test_large_resident_training_reuses_only_valid_execution_state(vulkan_backend):
    script = textwrap.dedent(
        """
        import os
        import torch
        import pytorch_vulkan

        device = os.environ.get("VULKAN_DEVICE", "vk:0")
        torch.manual_seed(123)
        cpu_model = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
        ).to(dtype=torch.float32)
        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
        ).to(device=device, dtype=torch.float32)
        model.load_state_dict(cpu_model.state_dict())
        cpu_inputs = torch.randn(128, 2048, dtype=torch.float32)
        cpu_targets = torch.randn(128, 2048, dtype=torch.float32)
        inputs = cpu_inputs.to(device)
        targets = cpu_targets.to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=1e-4)
        baseline_resources = pytorch_vulkan._C.live_resource_snapshot()
        baseline_services = pytorch_vulkan._C.shared_service_snapshot()
        baseline_counters = pytorch_vulkan._C.execution_counter_snapshot()
        pytorch_vulkan._C.reset_descriptor_resource_counters()

        lost = False
        for _ in range(3):
            pytorch_vulkan._C.reset_execution_counters()
            try:
                pytorch_vulkan._C.begin_training_step()
                assert pytorch_vulkan._C.training_step_active()
                optimizer.zero_grad(set_to_none=True)
                hidden = torch.ops.pytorch_vulkan.linear_relu(
                    inputs, model[0].weight, model[0].bias
                )
                output = torch.nn.functional.linear(
                    hidden, model[2].weight, model[2].bias
                )
                loss = (output - targets).mul(output - targets).sum()
                loss.backward()
                optimizer.step()
                assert pytorch_vulkan._C.training_step_active()
                assert pytorch_vulkan._C.compute_submitted_count() == 0
                assert pytorch_vulkan._C.compute_completed_count() == 0
                assert pytorch_vulkan._C.compute_wait_count() == 0
                pytorch_vulkan._C.end_training_step()
                assert not pytorch_vulkan._C.training_step_active()
                assert pytorch_vulkan._C.compute_submitted_count() == 1
                assert pytorch_vulkan._C.compute_completed_count() == 1
                assert pytorch_vulkan._C.compute_wait_count() == 1
                cpu_optimizer.zero_grad(set_to_none=True)
                cpu_output = cpu_model(cpu_inputs)
                cpu_loss = (cpu_output - cpu_targets).mul(cpu_output - cpu_targets).sum()
                cpu_loss.backward()
                cpu_optimizer.step()
                for actual_parameter, expected_parameter in zip(
                    model.parameters(), cpu_model.parameters()
                ):
                    torch.testing.assert_close(
                        actual_parameter.cpu(), expected_parameter,
                        rtol=2e-3, atol=2e-3
                    )
            except BaseException as error:
                pytorch_vulkan._C.cancel_training_step()
                if "Vulkan device lost" not in str(error):
                    raise
                lost = True
            counters = pytorch_vulkan._C.execution_counter_snapshot()
            if not lost:
                assert counters[0] > 0
                assert counters[1] == 0
            assert counters[2] == 0
            assert counters[3] == 0
            assert pytorch_vulkan._C.pending_compute_count() == 0
            if not lost:
                services = pytorch_vulkan._C.shared_service_snapshot()
                assert services["pipeline_entries"] == baseline_services["pipeline_entries"]
                assert services["pipeline_hits"] == baseline_services["pipeline_hits"]
                assert services["pipeline_misses"] == baseline_services["pipeline_misses"]
                assert services["shader_modules"] == baseline_services["shader_modules"]
                assert services["shader_hits"] == baseline_services["shader_hits"]
                assert services["shader_misses"] == baseline_services["shader_misses"]
                assert services["descriptor_pool_creations"] - baseline_services["descriptor_pool_creations"] >= 1
                assert (
                    services["descriptor_allocations"]
                    - baseline_services["descriptor_allocations"]
                    + services["descriptor_reuses"]
                    - baseline_services["descriptor_reuses"]
                    >= 1
                )
                assert services["descriptor_pools"] <= services["descriptor_pool_limit"]
                resources = pytorch_vulkan._C.live_resource_snapshot()
                assert resources[2] == baseline_resources[2]
                assert resources[3] == baseline_resources[3]
                assert pytorch_vulkan._C.compute_submitted_count() == 1
                assert pytorch_vulkan._C.compute_completed_count() == 1
                assert pytorch_vulkan._C.compute_wait_count() == 1
        if lost:
            before_rejection = pytorch_vulkan._C.execution_counter_snapshot()
            try:
                pytorch_vulkan._C.begin_training_step()
            except BaseException as error:
                assert "Vulkan device lost" in str(error)
            else:
                raise AssertionError("device-loss context accepted a new submission")
            try:
                torch.empty(1, dtype=torch.float32).to(device)
            except BaseException as error:
                assert "Vulkan device lost" in str(error)
            else:
                raise AssertionError("device-loss context accepted synchronization work")
            assert pytorch_vulkan._C.execution_counter_snapshot() == before_rejection
        print("reliability-ok")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath("build"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "reliability-ok" in result.stdout


def test_descriptor_and_allocation_pressure_reuse_is_bounded(vulkan_backend):
    script = textwrap.dedent(
        """
        import gc
        import os
        import torch
        import pytorch_vulkan

        device = os.environ.get("VULKAN_DEVICE", "vk:0")
        left = torch.ones(16, 8, dtype=torch.float32).to(device)
        right = torch.full((16, 8), 2.0, dtype=torch.float32).to(device)
        output = torch.empty_like(left)
        baseline_resources = pytorch_vulkan._C.live_resource_snapshot()
        assert baseline_resources[2] > 0
        assert baseline_resources[3] > 0

        pytorch_vulkan._C.begin_training_step()
        retained = torch.empty(128, 8, dtype=torch.float32).to(device)
        del retained
        deferred_resources = pytorch_vulkan._C.live_resource_snapshot()
        assert deferred_resources[6] >= baseline_resources[6] + 1
        pytorch_vulkan._C.end_training_step()
        gc.collect()
        assert pytorch_vulkan._C.live_resource_snapshot()[6] == baseline_resources[6]

        pytorch_vulkan._C.reset_execution_counters()
        pytorch_vulkan._C.reset_descriptor_resource_counters()
        pytorch_vulkan._C.begin_training_step()
        try:
            for _ in range(130):
                torch.add(left, right, out=output)
                torch.mul(left, right, out=output)
                torch.neg(left, out=output)
                torch.relu(left)
            pytorch_vulkan._C.end_training_step()
        except BaseException:
            pytorch_vulkan._C.cancel_training_step()
            raise

        pytorch_vulkan._C.begin_training_step()
        try:
            torch.add(left, right, out=output)
            pytorch_vulkan._C.end_training_step()
        except BaseException:
            pytorch_vulkan._C.cancel_training_step()
            raise

        descriptor_pools = pytorch_vulkan._C.descriptor_pool_creation_count()
        descriptor_sets = pytorch_vulkan._C.descriptor_set_allocation_count()
        descriptor_reuses = pytorch_vulkan._C.descriptor_set_reuse_count()
        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert descriptor_pools <= 12, descriptor_pools
        assert descriptor_sets <= 12 * 64, descriptor_sets
        assert descriptor_sets + descriptor_reuses >= 1
        assert counters[2:] == (0, 0)
        assert pytorch_vulkan._C.pending_compute_count() == 0
        live_resources = pytorch_vulkan._C.live_resource_snapshot()
        assert live_resources[0] <= 12
        assert live_resources[1] <= 12 * 64
        assert live_resources[2] == baseline_resources[2]
        assert live_resources[3] == baseline_resources[3]
        assert live_resources[4] == 0
        assert live_resources[5] == 0

        peak_allocations = baseline_resources[6]
        for _ in range(32):
            values = [torch.empty(128, 8, dtype=torch.float32).to(device) for _ in range(4)]
            peak_allocations = max(
                peak_allocations, pytorch_vulkan._C.live_resource_snapshot()[6]
            )
            del values
            gc.collect()
        assert pytorch_vulkan._C.pending_compute_count() == 0
        assert pytorch_vulkan._C.fallback_count() == 0
        final_resources = pytorch_vulkan._C.live_resource_snapshot()
        assert final_resources[2:6] == baseline_resources[2:6]
        assert peak_allocations >= baseline_resources[6] + 4
        assert final_resources[6] == baseline_resources[6]
        print("pressure-ok", descriptor_pools, descriptor_sets, descriptor_reuses, peak_allocations)
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath("build"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "pressure-ok" in result.stdout


def test_mixed_workload_stress_has_bounded_runtime_resources(vulkan_backend):
    import gc
    import torch

    device = vulkan_backend
    torch.manual_seed(29)
    left = torch.randn(8, 16).to(device)
    right = torch.randn(16, 8).to(device)
    pointwise = torch.empty_like(left)
    reduction = torch.empty(8).to(device)
    model = torch.nn.Sequential(torch.nn.Linear(16, 8), torch.nn.ReLU()).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    target = torch.zeros(8, 8).to(device)

    # Scope the resource delta to the repeated workload, not its resident model.
    baseline_resources = pytorch_vulkan._C.live_resource_snapshot()
    baseline_services = pytorch_vulkan._C.shared_service_snapshot()
    peak_resources = list(baseline_resources)
    peak_service = {
        "descriptor_pending": baseline_services["descriptor_pending"],
        "descriptor_quarantined": baseline_services["descriptor_quarantined"],
        "pipeline_pending_destructions": baseline_services["pipeline_pending_destructions"],
        "pipeline_count": baseline_services["pipeline_count"],
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
        peak_service["pipeline_count"] = max(
            peak_service["pipeline_count"], snapshot["pipeline_count"]
        )
        peak_service["timestamp_quarantined"] = max(
            peak_service["timestamp_quarantined"],
            int(pytorch_vulkan._C.timestamp_query_snapshot()[3]),
        )

    pytorch_vulkan._C.reset_descriptor_resource_counters()

    for _ in range(3):
        pytorch_vulkan._C.reset_execution_counters()
        pytorch_vulkan._C.reset_gpu_timing()
        pytorch_vulkan._C.begin_training_step()
        try:
            gemm = torch.mm(left, right)
            observe_service_state()
            torch.add(left, left, out=pointwise)
            observe_service_state()
            pointwise = torch.relu(pointwise)
            observe_service_state()
            torch.sum(gemm, dim=1, out=reduction)
            observe_service_state()
            loss = (model(left) - target).mul(model(left) - target).sum()
            observe_service_state()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            observe_service_state()
            optimizer.step()
            pytorch_vulkan._C.end_training_step()
            observe_service_state()
        except BaseException:
            pytorch_vulkan._C.cancel_training_step()
            raise

        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert counters[0] >= 5
        assert counters[1:] == (0, 0, 0)
        assert (pytorch_vulkan._C.compute_submitted_count(),
                pytorch_vulkan._C.compute_completed_count(),
                pytorch_vulkan._C.compute_wait_count()) == (1, 1, 1)
        assert pytorch_vulkan._C.pending_compute_count() == 0
        current_resources = pytorch_vulkan._C.live_resource_snapshot()
        peak_resources = [max(before, after) for before, after in zip(peak_resources, current_resources)]

    optimizer.zero_grad(set_to_none=True)
    del gemm, pointwise, reduction, loss
    gc.collect()
    final_resources = pytorch_vulkan._C.live_resource_snapshot()
    services = pytorch_vulkan._C.shared_service_snapshot()
    assert final_resources[2:6] == baseline_resources[2:6]
    assert final_resources[6] <= baseline_resources[6] + 2
    assert peak_resources[0] <= services["descriptor_pool_limit"]
    assert peak_resources[1] <= services["descriptor_pool_limit"] * 64
    assert services["pipeline_entries"] >= baseline_services["pipeline_entries"]
    assert services["shader_modules"] >= baseline_services["shader_modules"]
    assert services["descriptor_pending"] == 0
    assert services["descriptor_quarantined"] == 0
    assert services["pipeline_pending_destructions"] == 0
    assert services["pipeline_count"] <= final_resources[2]
    assert not services["pipeline_invalidated"]
    assert not services["descriptor_invalidated"]
    assert pytorch_vulkan._C.timestamp_query_snapshot()[3] is False
    assert peak_service["descriptor_quarantined"] == 0
    assert peak_service["timestamp_quarantined"] == 0
    assert peak_service["descriptor_pending"] <= peak_resources[1]
    assert peak_service["pipeline_pending_destructions"] <= peak_resources[2]
    assert peak_service["pipeline_count"] <= peak_resources[2]
    print("TASK5_MIXED_WORKLOAD_COMPLETED", peak_resources, peak_service, services)


def test_mixed_workload_validation_layer_gate():
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [os.path.abspath("build"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    environment["VK_INSTANCE_LAYERS"] = "VK_LAYER_KHRONOS_validation"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-s",
            "-q",
            "tests/python/test_vulkan_reliability.py::test_mixed_workload_stress_has_bounded_runtime_resources",
        ],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostics = result.stdout + result.stderr
    assert result.returncode == 0, diagnostics
    lowered = diagnostics.lower()
    if "task5_mixed_workload_completed" not in lowered and "skipped" in lowered:
        pytest.skip("mixed workload skipped because Vulkan is unavailable")
    assert "task5_mixed_workload_completed" in lowered, diagnostics
    assert not any(marker in lowered for marker in VALIDATION_ERROR_MARKERS), diagnostics
    print("validation-layer-clean")
