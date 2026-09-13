import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _pool_input(device, requires_grad=False):
    torch.manual_seed(23)
    value = torch.randn(2, 4, 3, 5, dtype=torch.float32).to(device)
    return value.requires_grad_(requires_grad)


def test_adaptive_avg_pool2d_global_forward_matches_cpu(vulkan_backend):
    cpu_input = _pool_input("cpu")
    result = torch.nn.functional.adaptive_avg_pool2d(cpu_input.to(vulkan_backend), (1, 1))
    expected = torch.nn.functional.adaptive_avg_pool2d(cpu_input, (1, 1))

    assert result.device == torch.device("vk:0")
    assert result.dtype is torch.float32
    assert result.shape == (2, 4, 1, 1)
    assert result.is_contiguous()
    assert result.storage_offset() == 0
    torch.testing.assert_close(result.cpu(), expected)


def test_adaptive_avg_pool2d_global_backward_matches_cpu(vulkan_backend):
    cpu_input = _pool_input("cpu", requires_grad=True)
    vk_input = _pool_input(vulkan_backend, requires_grad=True)
    cpu_output = torch.nn.functional.adaptive_avg_pool2d(cpu_input, (1, 1))
    vk_output = torch.nn.functional.adaptive_avg_pool2d(vk_input, (1, 1))
    grad_output = torch.randn_like(cpu_output)

    cpu_output.backward(grad_output)
    vk_output.backward(grad_output.to(vulkan_backend))

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach())
    assert vk_input.grad.device == torch.device("vk:0")
    assert vk_input.grad.is_contiguous()
    assert vk_input.grad.storage_offset() == 0
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)


@pytest.mark.parametrize("output_size", [(2, 2), (1, 2), (2, 1), [1], [1, 1, 1]])
def test_adaptive_avg_pool2d_rejects_non_global_output_size(vulkan_backend, output_size):
    with pytest.raises(RuntimeError, match="output|size|global|support|fixed"):
        torch.ops.aten._adaptive_avg_pool2d.default(_pool_input(vulkan_backend), output_size)


@pytest.mark.parametrize("shape", [(), (4,), (2, 4, 3), (2, 4, 3, 5, 1)])
def test_adaptive_avg_pool2d_rejects_non_rank4(vulkan_backend, shape):
    with pytest.raises(RuntimeError, match="rank|shape|4|dimension"):
        torch.ops.aten._adaptive_avg_pool2d.default(torch.empty(shape, device=vulkan_backend), [1, 1])


def test_adaptive_avg_pool2d_rejects_empty_input(vulkan_backend):
    with pytest.raises(RuntimeError, match="empty|nonempty|numel|size"):
        torch.nn.functional.adaptive_avg_pool2d(torch.empty((0, 4, 3, 5), device=vulkan_backend), (1, 1))


def test_adaptive_avg_pool2d_rejects_dtype_layout_device_and_offset(vulkan_backend):
    value = _pool_input(vulkan_backend)
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.nn.functional.adaptive_avg_pool2d(value.to(torch.float64), (1, 1))
    with pytest.raises(RuntimeError, match="contiguous|layout"):
        torch.nn.functional.adaptive_avg_pool2d(value.transpose(2, 3), (1, 1))
    offset = torch.empty((3, 4, 3, 5), device=vulkan_backend)[1:]
    assert offset.storage_offset() != 0
    with pytest.raises(RuntimeError, match="offset|zero"):
        torch.nn.functional.adaptive_avg_pool2d(offset, (1, 1))


def test_adaptive_avg_pool2d_preserves_deferred_formatter_and_fill_contract(vulkan_backend):
    value = _pool_input(vulkan_backend)
    with pytest.raises(RuntimeError, match="Double"):
        torch.arange(4, dtype=torch.float64).to(vulkan_backend)
    with pytest.raises((RuntimeError, TypeError), match="fill|unsupported|Vulkan"):
        value.fill_(1.0)


def test_adaptive_avg_pool2d_rejects_unsupported_overload(vulkan_backend):
    value = _pool_input(vulkan_backend)
    with pytest.raises((RuntimeError, TypeError, AttributeError), match="out|unsupported|overload|schema"):
        torch.ops.aten._adaptive_avg_pool2d.out(value, [1, 1], torch.empty((2, 4, 1, 1), device=vulkan_backend))


def test_adaptive_avg_pool2d_accumulated_leaf_and_higher_order_are_limited(vulkan_backend):
    value = _pool_input(vulkan_backend, requires_grad=True)
    output = torch.nn.functional.adaptive_avg_pool2d(value, (1, 1))
    gradient = torch.ones_like(output.cpu()).to(vulkan_backend)
    output.backward(gradient)
    with pytest.raises((RuntimeError, TypeError), match="leaf|accumul|grad"):
        output.backward(gradient)

    value = _pool_input(vulkan_backend, requires_grad=True)
    output = torch.nn.functional.adaptive_avg_pool2d(value, (1, 1))
    higher_order = torch.autograd.grad(output, value, torch.ones_like(output.cpu()).to(vulkan_backend), create_graph=True)[0]
    assert higher_order.device == torch.device("vk:0")
    assert not higher_order.requires_grad
