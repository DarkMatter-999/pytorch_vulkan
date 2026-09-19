import importlib.util
import inspect
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

    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.mm(vk_a, vk_b)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(actual.cpu(), torch.mm(a, b), rtol=2e-3, atol=2e-3)

    self_cpu = torch.randn(19, 23)
    self_vk = self_cpu.to(vulkan_backend)
    for alpha, beta in ((1.0, 0.0), (0.5, 1.0), (-1.25, 0.25)):
        pytorch_vulkan._C.reset_execution_counters()
        actual = torch.addmm(self_vk, vk_a, vk_b, alpha=alpha, beta=beta)
        assert pytorch_vulkan._C.compute_dispatch_count() == 1
        torch.testing.assert_close(
            actual.cpu(),
            torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta),
            rtol=2e-3,
            atol=2e-3,
        )


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
    assert record["status"] == "completed"
    assert record["shape"] == [2048, 2048]
    assert record["host_fence_wait_seconds"] >= 0.0
    assert record["dispatches"] == 1
    assert record["submissions"] == 1
    assert record["completions"] == 1
    assert record["waits"] == 1
    assert record["fallbacks"] == 0


def test_large_gemm_benchmark_has_no_payload_readback():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    source = script.read_text()
    assert ".cpu(" not in source
    assert '"submissions"' in source
    assert '"completions"' in source
    assert '"host_fence_wait_seconds"' in source
    assert "MATRIX_SHAPE" not in source
    assert "Vulkan GEMM tensor setup is unavailable" not in source
    assert "torch.randn(*LEFT_SHAPE).to(device)" in source
    assert "torch.randn(*RIGHT_SHAPE).to(device)" in source


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
