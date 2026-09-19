import os
import pytest
import subprocess
import sys
import textwrap

import pytorch_vulkan


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
        model = torch.nn.Sequential(
            torch.nn.Linear(2048, 4096),
            torch.nn.ReLU(),
            torch.nn.Linear(4096, 2048),
        ).to(device=device, dtype=torch.float32)
        inputs = torch.randn(128, 2048, dtype=torch.float32).to(device)
        targets = torch.randn(128, 2048, dtype=torch.float32).to(device)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

        lost = False
        for _ in range(3):
            pytorch_vulkan._C.reset_execution_counters()
            try:
                pytorch_vulkan._C.begin_training_step()
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
                pytorch_vulkan._C.end_training_step()
            except BaseException as error:
                pytorch_vulkan._C.cancel_training_step()
                if "Vulkan device lost" not in str(error):
                    raise
                lost = True
            counters = pytorch_vulkan._C.execution_counter_snapshot()
            assert counters[2] == 0
            assert counters[3] == 0
            assert pytorch_vulkan._C.pending_compute_count() == 0
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
        assert descriptor_reuses > 0, descriptor_reuses
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
