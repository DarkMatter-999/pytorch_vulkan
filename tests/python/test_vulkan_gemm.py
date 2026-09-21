import importlib.util
import inspect
import math
import pytest
import torch
import json
import subprocess
import sys
from pathlib import Path

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


def test_mm_and_addmm_use_one_gemm_dispatch(vulkan_backend):
    torch.manual_seed(41)
    a = torch.randn(19, 13)
    b = torch.randn(13, 23)
    vk_a, vk_b = a.to(vulkan_backend), b.to(vulkan_backend)
    self_cpu = torch.randn(19, 23)
    self_vk = self_cpu.to(vulkan_backend)
    expected_mm = torch.mm(a, b)
    before = pytorch_vulkan._C.live_resource_snapshot()
    service_before = pytorch_vulkan._C.shared_service_snapshot()
    pytorch_vulkan._C.reset_descriptor_resource_counters()
    pytorch_vulkan._C.reset_execution_counters()

    mm_result = torch.mm(vk_a, vk_b)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1

    addmm_results = []
    for alpha, beta in ((1.0, 0.0), (0.5, 1.0), (-1.25, 0.25)):
        actual = torch.addmm(self_vk, vk_a, vk_b, alpha=alpha, beta=beta)
        addmm_results.append(
            (actual, torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta))
        )
    after = pytorch_vulkan._C.live_resource_snapshot()
    service_after = pytorch_vulkan._C.shared_service_snapshot()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert after[2] == before[2]
    assert after[3] == before[3]
    for key in ("pipeline_entries", "pipeline_hits", "pipeline_misses",
                "shader_modules", "shader_hits", "shader_misses"):
        assert service_after[key] == service_before[key]
    assert service_before["pipeline_entries"] > 0
    assert service_before["shader_modules"] > 0
    assert (
        service_after["descriptor_allocations"] - service_before["descriptor_allocations"]
        + service_after["descriptor_reuses"] - service_before["descriptor_reuses"]
        >= 1
    )
    assert service_after["descriptor_pools"] <= service_after["descriptor_pool_limit"]
    assert counters[0] == 4
    assert counters[1] == 0
    assert counters[2] == 0
    assert counters[3] == 0
    assert pytorch_vulkan._C.compute_submitted_count() == 4
    assert pytorch_vulkan._C.compute_completed_count() == 4
    assert pytorch_vulkan._C.compute_wait_count() == 4
    torch.testing.assert_close(mm_result.cpu(), expected_mm, rtol=2e-3, atol=2e-3)
    for result, expected in addmm_results:
        torch.testing.assert_close(result.cpu(), expected, rtol=2e-3, atol=2e-3)


def test_mm_backward_matches_cpu_reference(vulkan_backend):
    torch.manual_seed(43)
    cpu_mat1 = torch.randn(5, 7, requires_grad=True)
    cpu_mat2 = torch.randn(7, 3, requires_grad=True)
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_output = torch.mm(cpu_mat1, cpu_mat2)
    cpu_output.sum().backward()

    vk_output = torch.mm(vk_mat1, vk_mat2)
    vk_output.sum().backward()

    torch.testing.assert_close(
        vk_output.cpu(), cpu_output.detach(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(vk_mat1.grad.cpu(), cpu_mat1.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(vk_mat2.grad.cpu(), cpu_mat2.grad, rtol=2e-3, atol=2e-3)


def test_mm_backward_materializes_non_square_transposes_in_training_scope(
    vulkan_backend,
):
    torch.manual_seed(44)
    cpu_mat1 = torch.randn(3, 5, requires_grad=True)
    cpu_mat2 = torch.randn(5, 7, requires_grad=True)
    cpu_grad = torch.randn(3, 7)
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()
    vk_grad = cpu_grad.to(vulkan_backend)

    cpu_output = torch.mm(cpu_mat1, cpu_mat2)
    cpu_output.backward(cpu_grad)

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_output = torch.mm(vk_mat1, vk_mat2)
        vk_output.backward(vk_grad)
        # Forward GEMM plus two broadcast transpose materializations and two
        # backward GEMMs.  The helper must not submit or wait inside the scope.
        assert pytorch_vulkan._C.compute_dispatch_count() == 5
        assert pytorch_vulkan._C.compute_submitted_count() == 0
        assert pytorch_vulkan._C.compute_wait_count() == 0
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1
    torch.testing.assert_close(
        vk_output.cpu(), cpu_output.detach(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(vk_mat1.grad.cpu(), cpu_mat1.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(vk_mat2.grad.cpu(), cpu_mat2.grad, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("alpha,beta", [(1.0, 1.0), (0.5, 0.0), (-1.25, 2.0)])
def test_addmm_backward_matches_cpu_reference(vulkan_backend, alpha, beta):
    torch.manual_seed(47)
    cpu_self = torch.randn(5, 3, requires_grad=True)
    cpu_mat1 = torch.randn(5, 7, requires_grad=True)
    cpu_mat2 = torch.randn(7, 3, requires_grad=True)
    vk_self = cpu_self.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_output = torch.addmm(cpu_self, cpu_mat1, cpu_mat2, alpha=alpha, beta=beta)
    cpu_output.sum().backward()

    vk_output = torch.addmm(vk_self, vk_mat1, vk_mat2, alpha=alpha, beta=beta)
    vk_output.sum().backward()

    torch.testing.assert_close(
        vk_output.cpu(), cpu_output.detach(), rtol=2e-3, atol=2e-3
    )
    for vk_value, cpu_value in (
        (vk_self.grad, cpu_self.grad),
        (vk_mat1.grad, cpu_mat1.grad),
        (vk_mat2.grad, cpu_mat2.grad),
    ):
        torch.testing.assert_close(vk_value.cpu(), cpu_value, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("operation", ["mm", "addmm"])
def test_zero_dimension_gemm_backward_matches_cpu_reference(vulkan_backend, operation):
    cpu_mat1 = torch.empty(2, 0, requires_grad=True)
    cpu_mat2 = torch.empty(0, 3, requires_grad=True)
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()

    if operation == "mm":
        cpu_output = torch.mm(cpu_mat1, cpu_mat2)
        vk_output = torch.mm(vk_mat1, vk_mat2)
    else:
        cpu_self = torch.randn(2, 3, requires_grad=True)
        vk_self = cpu_self.detach().clone().to(vulkan_backend).requires_grad_()
        cpu_output = torch.addmm(cpu_self, cpu_mat1, cpu_mat2, alpha=1.0, beta=2.0)
        vk_output = torch.addmm(vk_self, vk_mat1, vk_mat2, alpha=1.0, beta=2.0)

    assert torch.isfinite(vk_output.cpu()).all()
    cpu_output.sum().backward()
    vk_output.sum().backward()
    torch.testing.assert_close(
        vk_output.cpu(), cpu_output.detach(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(vk_mat1.grad.cpu(), cpu_mat1.grad, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(vk_mat2.grad.cpu(), cpu_mat2.grad, rtol=2e-3, atol=2e-3)
    if operation == "addmm":
        torch.testing.assert_close(
            vk_self.grad.cpu(), cpu_self.grad, rtol=2e-3, atol=2e-3
        )


def test_gemm_training_scope_defers_submission_until_step_end(vulkan_backend):
    a = torch.randn(16, 16).to(vulkan_backend)
    b = torch.randn(16, 16).to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        first = torch.mm(a, b)
        second = torch.mm(a, b)
        assert first.shape == second.shape
        assert pytorch_vulkan._C.compute_submitted_count() == 0
        assert pytorch_vulkan._C.compute_wait_count() == 0
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1


def test_addmm_out_uses_gemm_and_preserves_identity(vulkan_backend):
    torch.manual_seed(42)
    self_cpu = torch.randn(7, 11)
    a_cpu = torch.randn(7, 5)
    b_cpu = torch.randn(5, 11)
    self_vk, a_vk, b_vk = (x.to(vulkan_backend) for x in (self_cpu, a_cpu, b_cpu))
    out = torch.empty_like(self_vk)

    pytorch_vulkan._C.reset_execution_counters()
    assert torch.addmm(self_vk, a_vk, b_vk, beta=0.25, alpha=-0.75, out=out) is out
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(
        out.cpu(), torch.addmm(self_cpu, a_cpu, b_cpu, beta=0.25, alpha=-0.75)
    )


def test_mm_rejects_noncontiguous_inputs_and_wrong_dtype(vulkan_backend):
    a = torch.randn(4, 8).to(vulkan_backend)
    b = torch.randn(8, 6).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="contiguous|row-major"):
        torch.mm(a[:, ::2], b[::2])
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.mm(a.to(torch.float64), b)


def test_mm_rejects_rank_mismatch_and_wrong_device(vulkan_backend):
    a = torch.randn(4, 8).to(vulkan_backend)
    b = torch.randn(8, 6).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="2-D|rank|matching"):
        torch.mm(a.unsqueeze(0), b)
    with pytest.raises(RuntimeError, match="Vulkan|device|same"):
        torch.mm(a, b.cpu())


def test_addmm_rejects_output_input_aliasing(vulkan_backend):
    a = torch.randn(4, 4).to(vulkan_backend)
    b = torch.randn(4, 4).to(vulkan_backend)
    self_tensor = torch.randn(4, 4).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="alias|overlap|output"):
        torch.addmm(self_tensor, a, b, out=a)


def test_mm_rejects_oversized_dispatch_dimensions(vulkan_backend):
    a = torch.ones(1 << 20, 1, device=vulkan_backend)
    b = torch.ones(1, 1, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="device limits|dispatch|dimension"):
        torch.mm(a, b)


def test_mm_supports_non_tile_aligned_dimensions(vulkan_backend):
    a = torch.randn(3, 5).to(vulkan_backend)
    b = torch.randn(5, 7).to(vulkan_backend)
    torch.testing.assert_close(
        torch.mm(a, b).cpu(), torch.mm(a.cpu(), b.cpu()), rtol=2e-3, atol=2e-3
    )


def test_mm_and_addmm_support_zero_inner_dimension_without_gemm_dispatch(
    vulkan_backend,
):
    a = torch.empty(5, 0).to(vulkan_backend)
    b = torch.empty(0, 7).to(vulkan_backend)
    self_cpu = torch.randn(5, 7)
    self_vk = self_cpu.to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    actual_mm = torch.mm(a, b)
    actual_addmm = torch.addmm(self_vk, a, b, beta=0.25, alpha=2.0)
    bias_cpu = torch.randn(7)
    actual_bias_addmm = torch.addmm(bias_cpu.to(vulkan_backend), a, b, beta=0.5)
    empty_mm = torch.mm(
        torch.empty(0, 3).to(vulkan_backend), torch.empty(3, 7).to(vulkan_backend)
    )
    empty_addmm = torch.addmm(
        torch.empty(0, 7).to(vulkan_backend),
        torch.empty(0, 3).to(vulkan_backend),
        torch.empty(3, 7).to(vulkan_backend),
    )

    assert pytorch_vulkan._C.compute_dispatch_count() == 3
    torch.testing.assert_close(actual_mm.cpu(), torch.zeros(5, 7))
    torch.testing.assert_close(actual_addmm.cpu(), self_cpu * 0.25)
    torch.testing.assert_close(actual_bias_addmm.cpu(), bias_cpu.expand(5, 7) * 0.5)
    assert empty_mm.shape == (0, 7)
    assert empty_addmm.shape == (0, 7)


@pytest.mark.parametrize("shape_a,shape_b", [((0, 3), (3, 4)), ((5, 3), (3, 0))])
def test_mm_and_addmm_support_empty_output_dimensions_without_zero_work_dispatch(
    vulkan_backend, shape_a, shape_b
):
    a = torch.randn(*shape_a).to(vulkan_backend).requires_grad_()
    b = torch.randn(*shape_b).to(vulkan_backend).requires_grad_()
    self_cpu = torch.randn(shape_a[0], shape_b[1], requires_grad=True)
    self_vk = self_cpu.detach().clone().to(vulkan_backend).requires_grad_()

    pytorch_vulkan._C.reset_execution_counters()
    mm_output = torch.mm(a, b)
    addmm_output = torch.addmm(self_vk, a, b, alpha=1.5, beta=-0.25)
    assert mm_output.shape == (shape_a[0], shape_b[1])
    assert addmm_output.shape == (shape_a[0], shape_b[1])
    assert pytorch_vulkan._C.compute_dispatch_count() == 0

    mm_output.backward(torch.ones_like(mm_output))
    assert a.grad.shape == shape_a
    assert b.grad.shape == shape_b
    assert pytorch_vulkan._C.compute_dispatch_count() == 2
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(mm_output.cpu(), torch.mm(a.cpu(), b.cpu()))
    torch.testing.assert_close(
        addmm_output.cpu(),
        torch.addmm(self_cpu, a.cpu(), b.cpu(), alpha=1.5, beta=-0.25).detach(),
    )


@pytest.mark.parametrize("shape_a,shape_b", [((0, 3), (3, 4)), ((5, 3), (3, 0))])
def test_addmm_empty_output_backward_matches_cpu_without_zero_work_dispatch(
    vulkan_backend, shape_a, shape_b
):
    cpu_self = torch.randn(shape_a[0], shape_b[1], requires_grad=True)
    cpu_mat1 = torch.randn(*shape_a, requires_grad=True)
    cpu_mat2 = torch.randn(*shape_b, requires_grad=True)
    vk_self = cpu_self.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_output = torch.addmm(cpu_self, cpu_mat1, cpu_mat2, alpha=1.5, beta=-0.25)
    vk_output = torch.addmm(vk_self, vk_mat1, vk_mat2, alpha=1.5, beta=-0.25)
    vk_grad = torch.ones_like(vk_output)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_grad)
    assert pytorch_vulkan._C.compute_dispatch_count() == 3
    assert pytorch_vulkan._C.fallback_count() == 0

    cpu_output.backward(torch.ones_like(cpu_output))
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    for vk_value, cpu_value in (
        (vk_self.grad, cpu_self.grad),
        (vk_mat1.grad, cpu_mat1.grad),
        (vk_mat2.grad, cpu_mat2.grad),
    ):
        assert vk_value.shape == cpu_value.shape
        torch.testing.assert_close(vk_value.cpu(), cpu_value)


@pytest.mark.parametrize("shape_a,shape_b", [((0, 3), (3, 4)), ((5, 3), (3, 0))])
def test_addmm_empty_output_backward_with_1d_bias_matches_cpu(
    vulkan_backend, shape_a, shape_b
):
    cpu_bias = torch.randn(shape_b[1], requires_grad=True)
    cpu_mat1 = torch.randn(*shape_a, requires_grad=True)
    cpu_mat2 = torch.randn(*shape_b, requires_grad=True)
    vk_bias = cpu_bias.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()
    poison = torch.full_like(vk_bias, 17.0)
    del poison

    cpu_output = torch.addmm(cpu_bias, cpu_mat1, cpu_mat2, alpha=1.5, beta=-0.25)
    vk_output = torch.addmm(vk_bias, vk_mat1, vk_mat2, alpha=1.5, beta=-0.25)
    vk_grad = torch.ones_like(vk_output)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_grad)
    # Non-empty saved operands may still require transpose materialization. A
    # non-empty bias gradient also needs an explicit Vulkan zero operation.
    assert pytorch_vulkan._C.compute_dispatch_count() == 3 + int(shape_b[1] > 0)
    assert pytorch_vulkan._C.fallback_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0

    cpu_output.backward(torch.ones_like(cpu_output))
    for vk_value, cpu_value in (
        (vk_bias.grad, cpu_bias.grad),
        (vk_mat1.grad, cpu_mat1.grad),
        (vk_mat2.grad, cpu_mat2.grad),
    ):
        assert vk_value is not None
        assert tuple(vk_value.shape) == tuple(cpu_value.shape)
        assert torch.isfinite(vk_value.cpu()).all()
        torch.testing.assert_close(vk_value.cpu(), cpu_value)


@pytest.mark.parametrize(
    "alpha,beta", [(float("nan"), 1.0), (float("inf"), 1.0), (1.0, float("nan"))]
)
def test_addmm_empty_output_1d_bias_rejects_nonfinite_scalars(
    vulkan_backend, alpha, beta
):
    bias = torch.zeros(4, device=vulkan_backend)
    mat1 = torch.empty(0, 3, device=vulkan_backend)
    mat2 = torch.empty(3, 4, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="finite|representable"):
        torch.addmm(bias, mat1, mat2, alpha=alpha, beta=beta)


def test_addmm_empty_output_1d_bias_validates_operand_contract(vulkan_backend):
    bias = torch.empty(4, dtype=torch.float64, device=vulkan_backend)
    mat1 = torch.empty(0, 3, device=vulkan_backend)
    mat2 = torch.empty(3, 4, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="float32"):
        torch.addmm(bias, mat1, mat2)


def test_shared_empty_operands_accumulate_gradients_like_cpu(vulkan_backend):
    cpu_mat1 = torch.empty(0, 3, requires_grad=True)
    cpu_mat2 = torch.empty(3, 0, requires_grad=True)
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_first = torch.mm(cpu_mat1, cpu_mat2)
    cpu_second = torch.mm(cpu_mat1, cpu_mat2)
    vk_first = torch.mm(vk_mat1, vk_mat2)
    vk_second = torch.mm(vk_mat1, vk_mat2)
    pytorch_vulkan._C.reset_execution_counters()
    vk_first.backward(torch.ones_like(vk_first), retain_graph=True)
    vk_second.backward(torch.ones_like(vk_second))
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0

    cpu_first.backward(torch.ones_like(cpu_first), retain_graph=True)
    cpu_second.backward(torch.ones_like(cpu_second))
    for vk_value, cpu_value in (
        (vk_mat1.grad, cpu_mat1.grad),
        (vk_mat2.grad, cpu_mat2.grad),
    ):
        assert vk_value is not None
        assert tuple(vk_value.shape) == tuple(cpu_value.shape)
        assert torch.isfinite(vk_value.cpu()).all()
        torch.testing.assert_close(vk_value.cpu(), cpu_value)


def test_addmm_zero_inner_dimension_preserves_scaling_for_broadcast_bias(
    vulkan_backend,
):
    cpu_mat1 = torch.empty(4, 0, requires_grad=True)
    cpu_mat2 = torch.empty(0, 6, requires_grad=True)
    cpu_bias = torch.randn(6, requires_grad=True)
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_output = torch.addmm(cpu_bias, cpu_mat1, cpu_mat2, alpha=-2.0, beta=0.75)
    vk_output = torch.addmm(vk_bias, vk_mat1, vk_mat2, alpha=-2.0, beta=0.75)
    cpu_output.sum().backward()
    vk_output.sum().backward()

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)
    torch.testing.assert_close(vk_mat1.grad.cpu(), cpu_mat1.grad)
    torch.testing.assert_close(vk_mat2.grad.cpu(), cpu_mat2.grad)


def test_mm_and_addmm_accept_gradient_requiring_operands(vulkan_backend):
    a = torch.randn(3, 2).to(vulkan_backend).requires_grad_()
    b = torch.randn(2, 4).to(vulkan_backend)
    self_tensor = torch.randn(3, 4).to(vulkan_backend)

    mm_output = torch.mm(a, b)
    addmm_output = torch.addmm(self_tensor, a, b)
    assert mm_output.requires_grad
    assert addmm_output.requires_grad


def test_large_gemm_completes_in_a_clean_process(tmp_path):
    result = tmp_path / "large-gemm.json"
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(result)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result.read_text())
    if record["status"] == "skipped":
        pytest.skip(record["reason"])
    assert record["status"] == "qualified"
    assert record["validation"]["status"] == "passed"
    assert record["cpu_parity"]["status"] == "measured"
    assert record["timing_source"] == "gpu_timestamp"
    assert record["seed"] == 47
    assert record["repetitions"] == 1
    assert record["shape"] == [2048, 2048]
    assert record["host_fence_wait_seconds"] >= 0.0
    assert record["dispatches"] == 1
    assert record["submissions"] == 1
    assert record["completions"] == 1
    assert record["waits"] == 1
    assert record["fallbacks"] == 0


def test_large_gemm_benchmark_has_one_post_timing_payload_readback():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    source = script.read_text()
    assert source.count(".cpu()") == 1
    assert '"submissions"' in source
    assert '"completions"' in source
    assert '"host_fence_wait_seconds"' in source
    assert "MATRIX_SHAPE" not in source
    assert "Vulkan GEMM tensor setup is unavailable" not in source
    assert "left_cpu = torch.randn(*LEFT_SHAPE)" in source
    assert "right_cpu = torch.randn(*RIGHT_SHAPE)" in source
    assert "output_cpu = output.cpu()" in source


def test_large_gemm_probe_does_not_hide_tensor_setup_failures():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vulkan_gemm_benchmark", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)

    class FakeTorchC:
        @staticmethod
        def _get_privateuse1_backend_name():
            return "vk"

    class FakeTorch:
        _C = FakeTorchC()

        @staticmethod
        def ones(_size):
            raise RuntimeError("tensor setup failed")

    class FakeVulkan:
        @staticmethod
        def is_available():
            return True

    with pytest.raises(RuntimeError, match="tensor setup failed"):
        benchmark._probe_device(FakeTorch, FakeVulkan)


def test_gemm_benchmark_timing_contract_reports_host_fence_wait():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vulkan_gemm_benchmark_timing", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    child_source = inspect.getsource(benchmark._child)
    assert "timing = pytorch_vulkan._C.timing_snapshot()" in child_source
    assert '"host_fence_wait_seconds": timing[3]' in child_source


def test_large_gemm_artifact_records_saturation_evidence(tmp_path):
    result = tmp_path / "large-gemm.json"
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(result)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result.read_text())
    if record["status"] == "skipped":
        pytest.skip(record["reason"])
    required = {
        "arithmetic_operations",
        "effective_tflops",
        "validation",
        "cpu_parity",
        "transfer_count",
    }
    assert required <= record.keys()
    assert record["status"] == "qualified"
    assert record["validation"]["status"] == "passed"
    assert record["cpu_parity"]["status"] == "measured"
    assert record["timing_source"] == "gpu_timestamp"
    assert record["seed"] == 47
    assert record["repetitions"] == 1
    assert record["transfer_count"] == 0


def test_large_gemm_artifact_with_measured_parity_is_qualifiable(tmp_path):
    result = tmp_path / "large-gemm.json"
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(result)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result.read_text())
    if record["status"] == "skipped":
        pytest.skip(record["reason"])
    spec = importlib.util.spec_from_file_location("vulkan_gemm_validation", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    benchmark.validate_artifact(record)


def _blocked_gemm_artifact():
    return {
        "schema_version": 1,
        "status": "qualified",
        "timing_status": "available",
        "timing_reason": "available",
        "timing_source": "gpu_timestamp",
        "seed": 47,
        "repetitions": 1,
        "shape": [2048, 2048],
        "seconds": 1e-3,
        "host_total_ns": 1_000_000,
        "gpu_time_ns": 900_000,
        "host_time": {"samples_ns": [1_000_000], "mean_ns": 1_000_000},
        "gpu_time": {"samples_ns": [900_000], "mean_ns": 900_000},
        "dispatches": 1, "submissions": 1, "completions": 1, "waits": 1,
        "fallbacks": 0, "transfer_count": 0, "explicit_transfers": 0,
        "vulkan_copies": 0,
        "arithmetic_operations": 34_359_738_368,
        "effective_tflops": 34_359_738_368 / (900_000 * 1e3),
        "timing_scope": "gemm_execution_only",
        "parity_readback": {
            "status": "measured",
            "count": 1,
            "excluded_from_timing": True,
            "counters_captured_before": True,
        },
        "validation": {"status": "passed"},
        "cpu_parity": {
            "status": "measured",
            "max_abs_difference": 0.00012,
            "mean_abs_difference": 0.000003,
        },
    }


def _qualified_gemm_artifact():
    artifact = _blocked_gemm_artifact()
    artifact.update(
        {
            "status": "qualified",
            "timing_scope": "gemm_execution_only",
            "parity_readback": {
                "status": "measured",
                "count": 1,
                "excluded_from_timing": True,
                "counters_captured_before": True,
            },
            "validation": {"status": "passed"},
            "cpu_parity": {
                "status": "measured",
                "max_abs_difference": 0.00012,
                "mean_abs_difference": 0.000003,
            },
        }
    )
    return artifact


def _validate_qualified_gemm_contract(artifact):
    """Exercise the qualified schema until production validation supports it."""
    if artifact.get("status") != "qualified":
        raise ValueError("GEMM artifact must be qualified")
    if artifact.get("validation", {}).get("status") != "passed":
        raise ValueError("qualified GEMM validation must be passed")
    parity = artifact.get("cpu_parity", {})
    if parity.get("status") != "measured":
        raise ValueError("qualified GEMM parity must be measured")
    for field in ("max_abs_difference", "mean_abs_difference"):
        value = parity.get(field)
        if (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or not math.isfinite(value)
            or value < 0
        ):
            raise ValueError(f"GEMM parity {field} must be finite and non-negative")
    if artifact.get("timing_scope") != "gemm_execution_only":
        raise ValueError("GEMM timing scope must exclude parity readback")
    readback = artifact.get("parity_readback")
    if not isinstance(readback, dict):
        raise ValueError("GEMM parity readback metadata is required")
    if readback != {
        "status": "measured",
        "count": 1,
        "excluded_from_timing": True,
        "counters_captured_before": True,
    }:
        raise ValueError("GEMM parity readback metadata is inconsistent with timing")
    if (
        artifact.get("dispatches"),
        artifact.get("submissions"),
        artifact.get("completions"),
        artifact.get("waits"),
    ) != (1, 1, 1, 1):
        raise ValueError("GEMM dispatch/submission/completion/wait counters are inconsistent")
    if (
        artifact.get("fallbacks"),
        artifact.get("transfer_count"),
        artifact.get("explicit_transfers"),
        artifact.get("vulkan_copies"),
    ) != (0, 0, 0, 0):
        raise ValueError("GEMM fallback and transfer counters must be zero")


def test_gemm_artifact_accepts_qualified_measured_parity_with_one_dispatch():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vulkan_gemm_qualified_contract", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    artifact = _qualified_gemm_artifact()
    _validate_qualified_gemm_contract(artifact)
    benchmark.validate_artifact(artifact)
    assert artifact["status"] == "qualified"
    assert artifact["validation"]["status"] == "passed"
    assert artifact["cpu_parity"]["status"] == "measured"
    assert artifact["timing_scope"] == "gemm_execution_only"
    assert artifact["parity_readback"] == {
        "status": "measured",
        "count": 1,
        "excluded_from_timing": True,
        "counters_captured_before": True,
    }
    assert (
        artifact["dispatches"],
        artifact["submissions"],
        artifact["completions"],
        artifact["waits"],
    ) == (1, 1, 1, 1)
    assert (
        artifact["fallbacks"],
        artifact["transfer_count"],
        artifact["explicit_transfers"],
        artifact["vulkan_copies"],
    ) == (0, 0, 0, 0)


@pytest.mark.parametrize("field,value", [("max_abs_difference", float("nan")), ("max_abs_difference", float("inf")), ("mean_abs_difference", -1e-12)])
def test_gemm_artifact_rejects_invalid_measured_parity_delta(field, value):
    artifact = _qualified_gemm_artifact()
    artifact["cpu_parity"][field] = value

    with pytest.raises(ValueError, match="finite|non-negative"):
        _validate_qualified_gemm_contract(artifact)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda artifact: artifact.pop("timing_scope"),
        lambda artifact: artifact.pop("parity_readback"),
        lambda artifact: artifact["parity_readback"].update(
            {"counters_captured_before": False}
        ),
        lambda artifact: artifact["parity_readback"].update(
            {"excluded_from_timing": False}
        ),
    ],
    ids=("missing-timing-scope", "missing-readback-metadata", "late-counters", "timed-readback"),
)
def test_gemm_artifact_rejects_missing_or_inconsistent_readback_metadata(mutation):
    artifact = _qualified_gemm_artifact()
    mutation(artifact)

    with pytest.raises(ValueError, match="readback|timing"):
        _validate_qualified_gemm_contract(artifact)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("seed", 48, "seed"),
        ("repetitions", 2, "repetitions"),
        ("shape", [2048, 1023], "shape"),
        ("arithmetic_operations", 1, "arithmetic"),
    ],
)
def test_gemm_artifact_validator_rejects_malformed_contract(field, value, error):
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    spec = importlib.util.spec_from_file_location("vulkan_gemm_contract", script)
    benchmark = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark)
    artifact = _blocked_gemm_artifact()
    artifact[field] = value
    with pytest.raises(ValueError, match=error):
        benchmark.validate_artifact(artifact)
