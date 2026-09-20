import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _vk(values, backend, requires_grad=False):
    return (
        torch.tensor(values, dtype=torch.float32)
        .to(backend)
        .requires_grad_(requires_grad)
    )


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
def test_binary_tensor_gradients_match_cpu(vulkan_backend, operation):
    cpu_lhs = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    cpu_rhs = torch.tensor([-3.0, 4.0, 2.0], requires_grad=True)
    vk_lhs = cpu_lhs.detach().clone().to(vulkan_backend).requires_grad_()
    vk_rhs = cpu_rhs.detach().clone().to(vulkan_backend).requires_grad_()
    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_grad = torch.ones_like(cpu_result).to(vulkan_backend)

    before = pytorch_vulkan._C.live_resource_snapshot()
    service_before = pytorch_vulkan._C.shared_service_snapshot()
    pytorch_vulkan._C.reset_descriptor_resource_counters()
    pytorch_vulkan._C.reset_execution_counters()

    vk_result = operation(vk_lhs, vk_rhs)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(vk_grad)

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
    descriptor_allocations = service_after["descriptor_allocations"] - service_before["descriptor_allocations"]
    descriptor_reuses = service_after["descriptor_reuses"] - service_before["descriptor_reuses"]
    assert descriptor_allocations + descriptor_reuses >= 1
    assert service_after["descriptor_pools"] <= service_after["descriptor_pool_limit"]
    assert counters[0] == 3
    assert counters[1] == 0
    assert counters[2] == 0
    assert counters[3] == 0
    assert pytorch_vulkan._C.compute_submitted_count() == 3
    assert pytorch_vulkan._C.compute_completed_count() == 3
    assert pytorch_vulkan._C.compute_wait_count() == 3
    torch.testing.assert_close(vk_result.cpu(), cpu_result)

    torch.testing.assert_close(vk_lhs.grad.cpu(), cpu_lhs.grad)
    torch.testing.assert_close(vk_rhs.grad.cpu(), cpu_rhs.grad)


@pytest.mark.parametrize("operation", [torch.add, torch.sub])
@pytest.mark.parametrize("requires_grad_operand", ["lhs", "rhs"])
def test_binary_tensor_single_operand_gradient_matches_cpu(
    vulkan_backend, operation, requires_grad_operand
):
    cpu_lhs = torch.tensor(
        [1.5, -2.0, 0.25], requires_grad=requires_grad_operand == "lhs"
    )
    cpu_rhs = torch.tensor(
        [-3.0, 4.0, 2.0], requires_grad=requires_grad_operand == "rhs"
    )
    vk_lhs = (
        cpu_lhs.detach()
        .clone()
        .to(vulkan_backend)
        .requires_grad_(requires_grad_operand == "lhs")
    )
    vk_rhs = (
        cpu_rhs.detach()
        .clone()
        .to(vulkan_backend)
        .requires_grad_(requires_grad_operand == "rhs")
    )

    cpu_result = operation(cpu_lhs, cpu_rhs)
    vk_result = operation(vk_lhs, vk_rhs)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    if requires_grad_operand == "lhs":
        torch.testing.assert_close(vk_lhs.grad.cpu(), cpu_lhs.grad)
        assert vk_rhs.grad is None
    else:
        assert vk_lhs.grad is None
        torch.testing.assert_close(vk_rhs.grad.cpu(), cpu_rhs.grad)


@pytest.mark.parametrize("operation", [torch.add, torch.sub, torch.mul])
@pytest.mark.parametrize("scalar_left", [False, True])
def test_binary_scalar_gradient_matches_cpu(vulkan_backend, operation, scalar_left):
    cpu_tensor = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    vk_tensor = cpu_tensor.detach().clone().to(vulkan_backend).requires_grad_()
    scalar = 2.5
    cpu_operands = (scalar, cpu_tensor) if scalar_left else (cpu_tensor, scalar)
    vk_operands = (scalar, vk_tensor) if scalar_left else (vk_tensor, scalar)

    cpu_result = operation(*cpu_operands)
    vk_result = operation(*vk_operands)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_tensor.grad.cpu(), cpu_tensor.grad)


def test_rsub_scalar_left_gradient_matches_cpu(vulkan_backend):
    cpu_tensor = torch.tensor([1.5, -2.0, 0.25], requires_grad=True)
    vk_tensor = cpu_tensor.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_result = torch.rsub(cpu_tensor, 2.5)
    vk_result = torch.rsub(vk_tensor, 2.5)
    torch.testing.assert_close(vk_result.cpu(), cpu_result)
    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))

    torch.testing.assert_close(vk_tensor.grad.cpu(), cpu_tensor.grad)


def test_autograd_sub_scalar_rejects_invalid_alpha_without_work(vulkan_backend):
    tensor = _vk([1.5, -2.0], vulkan_backend, requires_grad=True)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match=r"alpha == 1"):
        torch.sub(tensor, 2.5, alpha=2.0)

    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_binary_rejections_remain_explicit(vulkan_backend):
    lhs = torch.empty((2, 1), device=vulkan_backend)
    rhs = torch.empty((2, 3), device=vulkan_backend)
    with pytest.raises(RuntimeError, match="broadcast"):
        torch.mul(lhs, rhs)

    equal_lhs = torch.empty((2,), device=vulkan_backend)
    equal_rhs = torch.empty((2,), device=vulkan_backend)
    torch.testing.assert_close(
        torch.add(equal_lhs, equal_rhs, alpha=2).cpu(),
        torch.add(equal_lhs.cpu(), equal_rhs.cpu(), alpha=2),
    )

    grad_lhs = equal_lhs.detach().requires_grad_()
    grad_rhs = equal_rhs.detach().requires_grad_()
    with pytest.raises((RuntimeError, NotImplementedError), match="out|requires grad"):
        torch.mul(grad_lhs, grad_rhs, out=torch.empty_like(equal_lhs))


def test_binary_inplace_is_explicitly_rejected_outside_optimizer_step(vulkan_backend):
    tensor = _vk([1.0, 2.0], vulkan_backend)
    other = _vk([3.0, 4.0], vulkan_backend)
    with pytest.raises(
        RuntimeError, match=r"Vulkan mul_.*in-place operations are unsupported"
    ):
        tensor.mul_(other)


@pytest.mark.parametrize("operation", [torch.add, torch.mul])
def test_bool_binary_result_does_not_require_grad(vulkan_backend, operation):
    lhs = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)
    rhs = torch.tensor([False, True], dtype=torch.bool, device=vulkan_backend)

    result = operation(lhs, rhs)

    assert result.dtype is torch.bool
    assert not result.requires_grad
    assert result.grad_fn is None
