import pytest
import torch
import pytorch_vulkan


@pytest.fixture
def vk():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def tensors(device, requires_grad=False):
    x = torch.tensor([[1., -2.], [3., 4.]], device=device, requires_grad=requires_grad)
    y = torch.tensor([[.5, 1.], [2., 5.]], device=device)
    return x, y


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_mse_forward_backward_parity_and_residency(vk, reduction):
    x, y = tensors(vk, True)
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.nn.functional.mse_loss(x, y, reduction=reduction)
    expected = torch.nn.functional.mse_loss(x.cpu().detach(), y.cpu(), reduction=reduction)
    torch.testing.assert_close(result.cpu(), expected)
    result.backward(torch.ones_like(result))
    assert x.grad.device.type == "vk"
    assert pytorch_vulkan._C.compute_dispatch_count() >= 2
    assert pytorch_vulkan._C.fallback_count() == 0


def test_mse_training_step(vk):
    x, target = tensors(vk)
    parameter = torch.tensor([1., -1.], device=vk, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=.1)
    loss = torch.nn.functional.mse_loss(parameter, torch.zeros_like(parameter), reduction="mean")
    loss.backward()
    optimizer.step()
    assert parameter.device.type == "vk"


@pytest.mark.parametrize("case", ["shape", "reduction", "target_dtype", "target_device", "target_layout"])
def test_mse_rejections_before_vulkan_work(vk, case):
    x, y = tensors(vk)
    if case == "shape": y = torch.ones(3, device=vk)
    if case == "reduction":
        op = lambda: torch.ops.aten.mse_loss.default(x, y, -1)
    elif case == "target_dtype":
        y = torch.ones((2, 2), dtype=torch.int32)
        op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    elif case == "target_device": y = y.cpu(); op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    elif case == "target_layout": y = y.t(); op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    else: op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match="Vulkan|reduction|shape|contiguous|dtype"):
        op()
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_mse_higher_order_is_rejected(vk):
    x, y = tensors(vk, True)
    loss = torch.nn.functional.mse_loss(x, y)
    with pytest.raises(RuntimeError, match="higher-order"):
        loss.backward(create_graph=True)
