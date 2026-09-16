import pytest
import torch

import pytorch_vulkan


def _fused(input, weight, bias):
    return torch.ops.pytorch_vulkan.linear_relu(input, weight, bias)


@pytest.fixture
def vulkan_device():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return torch.device("vk:0")


def test_linear_relu_forward_matches_cpu_and_is_one_dispatch(vulkan_device):
    torch.manual_seed(101)
    cpu_input = torch.randn(3, 8)
    cpu_weight = torch.randn(16, 8)
    cpu_bias = torch.randn(16)
    vk_input = cpu_input.to(vulkan_device)
    vk_weight = cpu_weight.to(vulkan_device)
    vk_bias = cpu_bias.to(vulkan_device)

    pytorch_vulkan._C.reset_execution_counters()
    actual = _fused(vk_input, vk_weight, vk_bias)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(actual.cpu(), torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias)))


def test_linear_relu_backward_and_optimizer_match_cpu(vulkan_device):
    torch.manual_seed(103)
    cpu_input = torch.randn(4, 8, requires_grad=True)
    cpu_weight = torch.randn(16, 8, requires_grad=True)
    cpu_bias = torch.randn(16, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_device).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_device).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_device).requires_grad_()
    cpu_optimizer = torch.optim.SGD([cpu_weight, cpu_bias], lr=0.03, momentum=0.8)
    vk_optimizer = torch.optim.SGD([vk_weight, vk_bias], lr=0.03, momentum=0.8)
    cpu_output = torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))
    cpu_loss = cpu_output.mul(cpu_output).sum()
    cpu_loss.backward()
    vk_input_before = vk_input.detach().cpu().clone()

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = _fused(vk_input, vk_weight, vk_bias)
    vk_loss = vk_output.mul(vk_output).sum()
    vk_loss.backward()
    vk_optimizer.step()
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)
    cpu_optimizer.step()
    torch.testing.assert_close(vk_weight.cpu(), cpu_weight)
    torch.testing.assert_close(vk_bias.cpu(), cpu_bias)
    torch.testing.assert_close(vk_input.cpu(), vk_input_before)


def test_linear_relu_backward_masks_inactive_outputs(vulkan_device):
    cpu_input = torch.ones(2, 2, requires_grad=True)
    cpu_weight = torch.ones(3, 2, requires_grad=True)
    cpu_bias = torch.full((3,), -10.0, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_device).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_device).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_device).requires_grad_()

    cpu_output = torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))
    cpu_output.backward(torch.ones_like(cpu_output))
    vk_output = _fused(vk_input, vk_weight, vk_bias)
    vk_output.backward(torch.ones_like(vk_output))

    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)
    torch.testing.assert_close(vk_weight.grad.cpu(), cpu_weight.grad)
    torch.testing.assert_close(vk_bias.grad.cpu(), cpu_bias.grad)


def test_linear_relu_input_gradient_non_square_active_matches_cpu(vulkan_device):
    cpu_input = torch.tensor([[1.0, -2.0, 0.5], [-0.25, 2.0, 1.5]], requires_grad=True)
    cpu_weight = torch.tensor(
        [[1.0, 2.0, -1.0], [-2.0, 1.0, 0.5], [0.5, -1.0, 2.0], [1.5, 0.25, -0.5]],
        requires_grad=True,
    )
    cpu_bias = torch.tensor([0.5, -0.25, 1.0, -2.0], requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_device).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_device).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_device).requires_grad_()

    cpu_output = torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))
    cpu_output.backward(torch.ones_like(cpu_output))
    vk_output = _fused(vk_input, vk_weight, vk_bias)
    vk_output.backward(torch.ones_like(vk_output))

    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad)


def test_compiler_fuses_fixed_mlp_linear_relu(vulkan_device):
    torch.manual_seed(109)
    model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).to(vulkan_device)
    inputs = torch.randn(3, 8).to(vulkan_device)
    eager = model(inputs)
    eager_dispatches = pytorch_vulkan._C.compute_dispatch_count()
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True)
    pytorch_vulkan._C.reset_execution_counters()
    actual = compiled(inputs)
    assert pytorch_vulkan._C.compute_dispatch_count() < eager_dispatches
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(actual.cpu(), eager.cpu())


def test_compiled_fixed_mlp_training_matches_cpu_and_reduces_dispatches(vulkan_device):
    torch.manual_seed(111)
    cpu_model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4))
    vk_model = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)).to(vulkan_device)
    vk_model.load_state_dict({name: value.to(vulkan_device) for name, value in cpu_model.state_dict().items()})
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.02, momentum=0.7)
    vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.02, momentum=0.7)
    cpu_input = torch.randn(3, 8)
    cpu_target = torch.randn(3, 4)
    vk_input = cpu_input.to(vulkan_device)
    vk_target = cpu_target.to(vulkan_device)
    torch._dynamo.reset()
    compiled = torch.compile(vk_model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True)
    for _ in range(2):
        cpu_optimizer.zero_grad()
        cpu_error = cpu_model(cpu_input) - cpu_target
        cpu_loss = cpu_error.mul(cpu_error).sum()
        cpu_loss.backward()
        cpu_optimizer.step()
        pytorch_vulkan._C.reset_execution_counters()
        pytorch_vulkan._C.begin_training_step()
        try:
            vk_optimizer.zero_grad()
            vk_output = compiled(vk_input)
            assert pytorch_vulkan._C.compute_dispatch_count() == 2
            vk_error = vk_output - vk_target
            vk_loss = vk_error.mul(vk_error).sum()
            vk_loss.backward()
            vk_optimizer.step()
        finally:
            pytorch_vulkan._C.end_training_step()
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
        assert pytorch_vulkan._C.compute_submission_count() == 1
        assert pytorch_vulkan._C.pending_compute_count() == 0
        torch.testing.assert_close(vk_loss.cpu(), cpu_loss)
    for cpu_parameter, vk_parameter in zip(cpu_model.parameters(), vk_model.parameters()):
        torch.testing.assert_close(vk_parameter.cpu(), cpu_parameter)
        for name, vk_state in vk_optimizer.state[vk_parameter].items():
            cpu_state = cpu_optimizer.state[cpu_parameter][name]
            if isinstance(vk_state, torch.Tensor):
                torch.testing.assert_close(vk_state.cpu(), cpu_state)
            else:
                assert vk_state == cpu_state
