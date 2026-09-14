import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


@pytest.mark.parametrize("operation", [torch.sum, torch.mean])
@pytest.mark.parametrize("dim,keepdim", [(0, True), ((1, 2), True), (0, False), ((0, 2), False), (None, False)])
def test_float32_reduction_matches_cpu_and_preserves_gradient(vulkan_backend, operation, dim, keepdim):
    cpu_input = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).requires_grad_()
    vk_input = cpu_input.detach().clone().to(vulkan_backend).requires_grad_()
    kwargs = {"keepdim": keepdim} if dim is not None else {}
    if dim is not None:
        kwargs["dim"] = dim

    cpu_result = operation(cpu_input, **kwargs)
    vk_result = operation(vk_input, **kwargs)
    torch.testing.assert_close(vk_result.cpu(), cpu_result, rtol=1e-6, atol=1e-6)
    assert vk_result.dtype is torch.float32
    assert vk_result.is_contiguous()

    cpu_result.backward(torch.ones_like(cpu_result))
    vk_result.backward(torch.ones_like(cpu_result).to(vulkan_backend))
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.sum, torch.mean])
def test_reduction_out_matches_cpu(vulkan_backend, operation):
    cpu_input = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    vk_input = cpu_input.to(vulkan_backend)
    out = torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend)
    result = operation(vk_input, dim=1, keepdim=True, out=out)
    expected = operation(cpu_input, dim=1, keepdim=True)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", [torch.sum, torch.mean])
def test_reduction_rejects_bool_and_float16(vulkan_backend, operation):
    with pytest.raises(RuntimeError, match="Vulkan|bool|float16"):
        operation(torch.ones((2, 3), dtype=torch.bool).to(vulkan_backend), dim=1)
    with pytest.raises(RuntimeError, match="float16"):
        operation(torch.ones((2, 3), dtype=torch.float16).to(vulkan_backend), dim=1)


@pytest.mark.parametrize("view", [
    lambda x: x.t(),
    lambda x: x[:, 1:],
    lambda x: x.as_strided((2, 2), (3, 1), 1),
])
def test_reduction_matches_cpu_for_general_views(vulkan_backend, view):
    cpu_base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cpu_input = view(cpu_base)
    vk_input = view(cpu_base.to(vulkan_backend))
    torch.testing.assert_close(torch.sum(vk_input, dim=-1).cpu(), torch.sum(cpu_input, dim=-1))
    torch.testing.assert_close(torch.mean(vk_input, dim=0).cpu(), torch.mean(cpu_input, dim=0))


def test_reduction_rejects_overlapping_input(vulkan_backend):
    input = torch.arange(6, dtype=torch.float32, device="cpu").reshape(2, 3).to(vulkan_backend)
    overlapping = input.as_strided((2, 2), (1, 1))
    with pytest.raises(RuntimeError, match="overlap|overlapping"):
        torch.sum(overlapping, dim=1)


def test_reduction_rejects_rank_nine(vulkan_backend):
    input = torch.empty((1,) * 9, dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|ranks"):
        torch.sum(input, dim=0)


@pytest.mark.parametrize("operation", [torch.sum, torch.mean])
def test_reduction_matches_cpu_for_empty_reduced_dimension(vulkan_backend, operation):
    input = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    result = operation(input, dim=0)
    expected = operation(input.cpu(), dim=0)
    assert result.device == input.device
    assert result.device.type == "vk"
    torch.testing.assert_close(result.cpu(), expected, equal_nan=True)
