import pytest
import torch
import pytorch_vulkan


@pytest.fixture
def vk():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def tensors(device, requires_grad=False):
    x = torch.tensor([[1.0, -2.0], [3.0, 4.0]]).to(device).requires_grad_(requires_grad)
    y = torch.tensor([[0.5, 1.0], [2.0, 5.0]]).to(device)
    return x, y


@pytest.mark.parametrize("reduction", ["none", "sum", "mean"])
def test_mse_forward_backward_parity_and_residency(vk, reduction):
    x, y = tensors(vk, True)
    cpu_x, cpu_y = tensors("cpu")
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.nn.functional.mse_loss(x, y, reduction=reduction)
    expected = torch.nn.functional.mse_loss(
        cpu_x, cpu_y, reduction=reduction
    )
    result.backward(torch.ones_like(result))
    assert x.grad.device.type == "vk"
    assert pytorch_vulkan._C.compute_dispatch_count() >= 2
    assert pytorch_vulkan._C.fallback_count() == 0
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(x.grad.cpu(),
                               (2 * (cpu_x - cpu_y) / cpu_x.numel()
                                if reduction == "mean" else 2 * (cpu_x - cpu_y)))


@pytest.mark.parametrize("reduction", ["sum", "mean"])
def test_mse_wide_reduction_matches_cpu_without_fallback(vk, reduction):
    cpu_input = torch.linspace(-3.0, 3.0, 65536).reshape(1024, 64)
    cpu_target = torch.cos(cpu_input)
    vk_input = cpu_input.to(vk)
    vk_target = cpu_target.to(vk)
    pytorch_vulkan._C.reset_execution_counters()

    result = torch.nn.functional.mse_loss(vk_input, vk_target, reduction=reduction)
    expected = torch.nn.functional.mse_loss(cpu_input, cpu_target, reduction=reduction)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    assert pytorch_vulkan._C.fallback_count() == 0
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(result.cpu(), expected, rtol=1e-5, atol=1e-6)


def test_mse_training_step(vk):
    x, target = tensors(vk)
    parameter = torch.tensor([1.0, -1.0]).to(vk).requires_grad_()
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = torch.nn.functional.mse_loss(
        parameter, torch.zeros_like(parameter), reduction="mean"
    )
    loss.backward()
    optimizer.step()
    assert parameter.device.type == "vk"
    pytorch_vulkan._C.synchronize()


@pytest.mark.parametrize(
    "case", ["shape", "reduction", "target_dtype", "target_device"]
)
def test_mse_rejections_before_vulkan_work(vk, case):
    x, y = tensors(vk)
    if case == "shape":
        y = torch.ones(3, device=vk)
    if case == "reduction":
        op = lambda: torch.ops.aten.mse_loss.default(x, y, -1)
    elif case == "target_dtype":
        y = torch.ones((2, 2), dtype=torch.int32)
        op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    elif case == "target_device":
        y = y.cpu()
        op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    else:
        op = lambda: torch.ops.aten.mse_loss.default(x, y, 1)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(
        (RuntimeError, ValueError), match="Vulkan|reduction|shape|contiguous|dtype"
    ):
        op()
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_mse_readable_transposed_target(vk):
    cpu_x, cpu_y = tensors("cpu")
    x, y = cpu_x.to(vk), cpu_y.to(vk).t()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.nn.functional.mse_loss(x, y)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot()[2:] == (0, 0)
    torch.testing.assert_close(result.cpu(), torch.nn.functional.mse_loss(cpu_x, cpu_y.t()))
