import pytest
import torch

import pytorch_vulkan
from tools.vulkan_model_benchmark import (
    MODEL_PARITY_ABS_TOLERANCE,
    RNNFixture,
    _move_model,
    _run_row,
)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _resident(value):
    assert value.device == torch.device("vk:0")
    assert value.dtype is torch.float32
    assert value.is_contiguous()


def _pair(device, batch=1, sequence_length=64):
    fixture = RNNFixture(batch=batch, sequence_length=sequence_length)
    cpu = fixture.make_cpu()
    vk = _move_model(fixture.make_cpu(), device)
    vk.load_state_dict({name: value.to(device) for name, value in cpu.state_dict().items()})
    inputs, target = fixture.make_inputs()
    return fixture, cpu, vk, inputs.requires_grad_(), inputs.detach().clone().to(device).requires_grad_(), target, target.to(device)


def test_rnn_forward_backward_and_optimizer_match_cpu(vulkan_backend):
    fixture, cpu, vk, cpu_input, vk_input, cpu_target, vk_target = _pair(vulkan_backend)
    cpu_optimizer = torch.optim.SGD(cpu.parameters(), lr=0.01)
    vk_optimizer = torch.optim.SGD(vk.parameters(), lr=0.01)
    cpu_loss = fixture.loss(cpu(cpu_input), cpu_target)
    cpu_loss.backward()
    pytorch_vulkan._C.reset_execution_counters()
    vk_loss = fixture.loss(vk(vk_input), vk_target)
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    copies = pytorch_vulkan._C.vulkan_copy_count()
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    cpu_optimizer.step()
    vk_optimizer.step()
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss.detach(), rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-3, atol=3e-3)
    for actual, expected in zip(vk.parameters(), cpu.parameters()):
        _resident(actual)
        _resident(actual.grad)
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)
    _resident(vk_loss)
    _resident(vk_input.grad)
    for state in vk.last_states:
        _resident(state)
        assert state.grad is not None
        _resident(state.grad)
    for actual, expected in zip(vk.last_states, cpu.last_states):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    assert dispatches > 0
    assert 0 < copies <= 3 * fixture.sequence_length
    assert transfers == 0
    assert fallbacks == 0


def test_rnn_benchmark_one_step_preserves_loss_parity(vulkan_backend):
    fixture = RNNFixture()
    initial_state = {
        name: value.detach().clone() for name, value in fixture.make_cpu().state_dict().items()
    }
    cpu_row, cpu_state = _run_row("rnn", fixture, "cpu", 0, 1, initial_state, True)
    vulkan_row, vulkan_state = _run_row("rnn", fixture, "vk:0", 0, 1, initial_state, True)
    torch.testing.assert_close(
        torch.tensor(vulkan_row["loss"]),
        torch.tensor(cpu_row["loss"]),
        rtol=3e-3,
        atol=3e-3,
    )
    assert max(
        float((vulkan_state[name] - cpu_state[name]).abs().max()) for name in cpu_state
    ) <= MODEL_PARITY_ABS_TOLERANCE
    assert vulkan_row["dispatches"] > 0
    assert vulkan_row["explicit_transfers"] == 0
    assert vulkan_row["fallbacks"] == 0


def test_rnn_rejects_batch_above_bounded_contract_before_work(vulkan_backend):
    with pytest.raises(ValueError, match="batch"):
        RNNFixture(batch=17)
    fixture = RNNFixture()
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs = torch.empty(
        (17, fixture.sequence_length, fixture.input_dim), dtype=fixture.dtype, device=vulkan_backend
    )
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match="batch"):
        model(inputs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("batch", [1, 16])
@pytest.mark.parametrize("sequence_length", [1, 64])
def test_rnn_supported_boundaries_stay_resident(vulkan_backend, batch, sequence_length):
    fixture, _, model, _, inputs, _, _ = _pair(vulkan_backend, batch, sequence_length)
    output = model(inputs)
    assert output.shape == (batch, sequence_length, fixture.hidden_dim)
    _resident(output)
    for state in model.last_states:
        _resident(state)


@pytest.mark.parametrize("mutator, message", [
    (lambda x: x.to(torch.float64), "float32|dtype|type"),
    (lambda x: x.transpose(1, 2), "contiguous|layout|shape"),
    (lambda x: torch.empty((x.shape[0], 65, x.shape[2]), device=x.device, dtype=x.dtype), "shape|sequence"),
])
def test_rnn_rejects_unsupported_input_before_work(vulkan_backend, mutator, message):
    fixture, _, model, _, inputs, _, _ = _pair(vulkan_backend)
    bad_inputs = mutator(inputs)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=message):
        model(bad_inputs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_rnn_rejects_higher_order_gradient_before_work(vulkan_backend):
    value = torch.ones((1, 8), device=vulkan_backend, requires_grad=True)
    output = torch.tanh(value)
    grad_output = torch.ones_like(output)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match="higher|second|order"):
        torch.autograd.grad(output, value, grad_output, create_graph=True)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
