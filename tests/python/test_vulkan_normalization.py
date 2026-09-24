import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _batch_norm_inputs(device, requires_grad=False):
    torch.manual_seed(211)
    x = torch.randn(2, 4, dtype=torch.float32).to(device)
    weight = torch.randn(4, dtype=torch.float32).to(device)
    bias = torch.randn(4, dtype=torch.float32).to(device)
    running_mean = torch.randn(4, dtype=torch.float32).to(device)
    running_var = (torch.rand(4, dtype=torch.float32) + 1.0).to(device)
    if requires_grad:
        x.requires_grad_()
        weight.requires_grad_()
        bias.requires_grad_()
    return x, weight, bias, running_mean, running_var


def test_native_batch_norm_fixed_training_contract_matches_cpu(vulkan_backend):
    cpu = _batch_norm_inputs("cpu")
    vk = tuple(value.to(vulkan_backend) for value in cpu)
    expected = torch.ops.aten.native_batch_norm.default(*cpu, True, 0.1, 1e-5)
    actual = torch.ops.aten.native_batch_norm.default(*vk, True, 0.1, 1e-5)
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference)


def test_native_batch_norm_4d_benchmark_case_matches_cpu(vulkan_backend):
    torch.manual_seed(1729)
    cpu = (
        torch.randn(2, 4, 2, 2),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.rand(4) + 1,
    )
    vk = tuple(value.to(vulkan_backend) for value in cpu)
    expected = torch.ops.aten.native_batch_norm.default(*cpu, True, 0.1, 1e-5)
    actual = torch.ops.aten.native_batch_norm.default(*vk, True, 0.1, 1e-5)
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference)


def test_native_batch_norm_rejects_eval_and_wrong_eps_without_vulkan_work(
    vulkan_backend,
):
    inputs = _batch_norm_inputs(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="training=True|eps"):
        torch.ops.aten.native_batch_norm.default(*inputs, False, 0.1, 1e-5)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_native_batch_norm_backward_returns_resident_gradients(vulkan_backend):
    cpu = _batch_norm_inputs("cpu", requires_grad=True)
    vk = tuple(
        value.to(vulkan_backend).requires_grad_(value.requires_grad) for value in cpu
    )
    cpu_forward = torch.ops.aten.native_batch_norm.default(*cpu, True, 0.1, 1e-5)
    vk_forward = torch.ops.aten.native_batch_norm.default(*vk, True, 0.1, 1e-5)
    grad = torch.randn_like(cpu[0])
    expected = torch.ops.aten.native_batch_norm_backward.default(
        grad,
        cpu[0],
        cpu[1],
        cpu[3],
        cpu[4],
        cpu_forward[1],
        cpu_forward[2],
        True,
        1e-5,
        [True, True, True],
    )
    actual = torch.ops.aten.native_batch_norm_backward.default(
        grad.to(vulkan_backend),
        vk[0],
        vk[1],
        vk[3],
        vk[4],
        vk_forward[1],
        vk_forward[2],
        True,
        1e-5,
        [True, True, True],
    )
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference)


def test_native_batch_norm_4d_backward_nonuniform_grad_matches_cpu(vulkan_backend):
    torch.manual_seed(1729)
    cpu = (
        torch.randn(2, 4, 2, 2),
        torch.randn(4),
        torch.randn(4),
        torch.randn(4),
        torch.rand(4) + 1,
    )
    vk = tuple(value.to(vulkan_backend) for value in cpu)
    cpu_forward = torch.ops.aten.native_batch_norm.default(*cpu, True, 0.1, 1e-5)
    vk_forward = torch.ops.aten.native_batch_norm.default(*vk, True, 0.1, 1e-5)
    grad = torch.randn_like(cpu[0])
    expected = torch.ops.aten.native_batch_norm_backward.default(
        grad, cpu[0], cpu[1], cpu[3], cpu[4], cpu_forward[1], cpu_forward[2],
        True, 1e-5, [True, True, True],
    )
    actual = torch.ops.aten.native_batch_norm_backward.default(
        grad.to(vulkan_backend), vk[0], vk[1], vk[3], vk[4],
        vk_forward[1], vk_forward[2], True, 1e-5, [True, True, True],
    )
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference)
