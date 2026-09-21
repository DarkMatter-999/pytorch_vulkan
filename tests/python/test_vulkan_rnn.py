import pytest
import torch

import pytorch_vulkan
from tools.vulkan_model_benchmark import (
    MODEL_PARITY_ABS_TOLERANCE,
    RNNFixture,
    _move_model,
    _run_row,
)


RNN_PRE_OPTIMIZATION_COPY_OPERATIONS = 64
RNN_PRE_OPTIMIZATION_COPY_COMMANDS = 64


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
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss = fixture.loss(vk(vk_input), vk_target)
        vk_loss.backward()
        vk_optimizer.step()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    copies = pytorch_vulkan._C.vulkan_copy_count()
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    submissions = pytorch_vulkan._C.compute_submitted_count()
    completions = pytorch_vulkan._C.compute_completed_count()
    waits = pytorch_vulkan._C.compute_wait_count()
    cpu_optimizer.step()
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
    assert submissions == 1
    assert completions == 1
    assert waits == 1


def test_training_scope_cleans_up_after_forward_failure(vulkan_backend):
    fixture, _, model, _, inputs, _, target = _pair(vulkan_backend, batch=1)
    pytorch_vulkan._C.begin_training_step()
    try:
        with pytest.raises((RuntimeError, ValueError), match="sequence|shape"):
            model(torch.empty((1, 65, fixture.input_dim), device=vulkan_backend))
    finally:
        pytorch_vulkan._C.cancel_training_step()

    assert not pytorch_vulkan._C.training_step_active()
    pytorch_vulkan._C.begin_training_step()
    try:
        loss = fixture.loss(model(inputs), target)
        loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    assert not pytorch_vulkan._C.training_step_active()


def test_stack_64_states_has_one_transfer_lifecycle_and_backward_parity(vulkan_backend):
    torch.manual_seed(1729)
    cpu_states = [torch.randn(2, 8, requires_grad=True) for _ in range(64)]
    vk_states = [state.detach().clone().to(vulkan_backend).requires_grad_() for state in cpu_states]
    grad = torch.randn((2, 64, 8), dtype=torch.float32)
    vk_grad = grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_output = torch.stack(cpu_states, dim=1)
    vk_output = torch.stack(vk_states, dim=1)
    operations = pytorch_vulkan._C.transfer_operation_count()
    submissions = pytorch_vulkan._C.transfer_submission_count()
    completions = pytorch_vulkan._C.transfer_completion_count()
    waits = pytorch_vulkan._C.transfer_wait_count()
    copy_commands = pytorch_vulkan._C.copy_command_count()
    cpu_output.backward(grad)
    vk_output.backward(vk_grad)

    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=0, atol=0)
    for vk_state, cpu_state in zip(vk_states, cpu_states):
        torch.testing.assert_close(vk_state.grad.cpu(), cpu_state.grad, rtol=0, atol=0)
    assert operations < RNN_PRE_OPTIMIZATION_COPY_OPERATIONS
    assert copy_commands == RNN_PRE_OPTIMIZATION_COPY_COMMANDS
    assert submissions == completions == 1
    assert waits >= submissions


def test_stack_64_states_dim_zero_has_one_transfer_lifecycle(vulkan_backend):
    states = [torch.full((2, 8), float(index), device=vulkan_backend) for index in range(64)]
    expected = torch.stack([state.cpu() for state in states], dim=0)
    pytorch_vulkan._C.reset_execution_counters()

    output = torch.stack(states, dim=0)
    operations = pytorch_vulkan._C.transfer_operation_count()
    submissions = pytorch_vulkan._C.transfer_submission_count()
    completions = pytorch_vulkan._C.transfer_completion_count()
    waits = pytorch_vulkan._C.transfer_wait_count()

    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0)
    assert operations == 1
    assert submissions == 1
    assert completions == 1
    assert waits == 1


@pytest.mark.parametrize("case", ["rank", "dtype", "noncontiguous"])
def test_stack_rejects_unsupported_layout_before_work(vulkan_backend, case):
    first = torch.ones((2, 8), dtype=torch.float32, device=vulkan_backend)
    if case == "rank":
        states, message = [first.reshape(2, 8, 1)], "contiguous"
    elif case == "dtype":
        states, message = [first, torch.ones((2, 8), dtype=torch.float64)], "matching"
    else:
        states, message = [first.transpose(0, 1)], "contiguous"
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises((RuntimeError, ValueError), match=message):
        torch.stack(states, dim=0)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


def test_stack_rejects_mixed_device_before_work(vulkan_backend):
    states = [torch.ones((2, 8), dtype=torch.float32, device=vulkan_backend),
              torch.ones((2, 8), dtype=torch.float32)]
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises((RuntimeError, ValueError), match="matching|device|same"):
        torch.stack(states, dim=0)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


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
    assert vulkan_row["submissions"] == vulkan_row["repetitions"]
    assert vulkan_row["completions"] == vulkan_row["repetitions"]
    assert vulkan_row["waits"] == vulkan_row["repetitions"]
    assert vulkan_row["compute_submissions"] == vulkan_row["repetitions"]
    assert vulkan_row["transfer_operations"] == vulkan_row["vulkan_copies"]
    assert vulkan_row["transfer_operations"] > 0
    assert vulkan_row["transfer_submissions"] <= vulkan_row["transfer_operations"]
    assert vulkan_row["transfer_completions"] == vulkan_row["transfer_submissions"]
    assert vulkan_row["transfer_waits"] >= vulkan_row["transfer_submissions"]
    assert vulkan_row["fallback_status"] is False
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
