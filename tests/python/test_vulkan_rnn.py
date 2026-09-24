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


def _resident_view(value):
    assert value.device == torch.device("vk:0")
    assert value.dtype is torch.float32


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
        _resident_view(state)
        assert state.grad is not None
        _resident_view(state.grad)
    for actual, expected in zip(vk.last_states, cpu.last_states):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    assert dispatches > 0
    assert copies == 0
    assert transfers == 0
    assert fallbacks == 0
    assert submissions == 1
    assert completions == 1
    assert waits == 1


def test_default_rnn_fixture_hidden_256_training_path(vulkan_backend):
    fixture, cpu, model, cpu_inputs, inputs, cpu_target, target = _pair(vulkan_backend)
    cpu_optimizer = torch.optim.SGD(cpu.parameters(), lr=0.01)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    pytorch_vulkan._C.reset_execution_counters()
    cpu_output = cpu(cpu_inputs)
    cpu_loss = fixture.loss(cpu_output, cpu_target)
    cpu_loss.backward()
    pytorch_vulkan._C.begin_training_step()
    try:
        output = model(inputs)
        loss = fixture.loss(output, target)
        loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    submissions = pytorch_vulkan._C.compute_submitted_count()
    completions = pytorch_vulkan._C.compute_completed_count()
    waits = pytorch_vulkan._C.compute_wait_count()
    copies = pytorch_vulkan._C.vulkan_copy_count()
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    optimizer.step()
    cpu_optimizer.step()
    torch.testing.assert_close(output.cpu(), cpu_output, rtol=3e-3, atol=3e-3)
    assert torch.isfinite(output.cpu()).all()
    torch.testing.assert_close(loss.cpu(), cpu_loss.detach(), rtol=3e-3, atol=3e-3)
    assert torch.isfinite(loss.cpu()).all()
    assert dispatches > 0
    assert submissions == 1
    assert completions == 1
    assert waits == 1
    assert copies == 0
    assert transfers == 0
    assert fallbacks == 0
    torch.testing.assert_close(inputs.grad.cpu(), cpu_inputs.grad, rtol=3e-3, atol=3e-3)
    assert torch.isfinite(inputs.grad.cpu()).all()
    for actual, expected in zip(model.parameters(), cpu.parameters()):
        assert actual.grad is not None
        assert torch.isfinite(actual.grad.cpu()).all()
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
        assert torch.isfinite(actual.cpu()).all()
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)


def test_rnn_state_gradient_hook_is_generation_local(vulkan_backend):
    fixture, _, model, _, inputs, _, target = _pair(vulkan_backend, batch=1, sequence_length=3)
    first_output = model(inputs)
    first_states = list(model.last_states)
    second_output = model(inputs.detach().clone().requires_grad_())
    second_states = list(model.last_states)
    assert first_states[0].untyped_storage().data_ptr() != second_states[0].untyped_storage().data_ptr()

    pytorch_vulkan._C.begin_training_step()
    try:
        fixture.loss(first_output, target).backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    assert all(state.grad is not None for state in first_states)
    assert all(state.grad is None for state in second_states)


def test_rnn_last_states_are_fused_output_views_with_cpu_matching_gradients(vulkan_backend):
    fixture, cpu, vk, cpu_input, vk_input, cpu_target, vk_target = _pair(
        vulkan_backend, batch=2, sequence_length=3
    )
    cpu_output = cpu(cpu_input)
    vk_output = vk(vk_input)
    cpu_loss = fixture.loss(cpu_output, cpu_target)
    vk_loss = fixture.loss(vk_output, vk_target)
    cpu_loss.backward()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    output_storage = vk_output.untyped_storage().data_ptr()
    assert len(vk.last_states) == fixture.sequence_length
    for index, state in enumerate(vk.last_states):
        assert state.untyped_storage().data_ptr() == output_storage
        assert tuple(state.shape) == (2, fixture.hidden_dim)
        assert tuple(state.stride()) == (fixture.sequence_length * fixture.hidden_dim, 1)
        assert state.storage_offset() == vk_output.storage_offset() + index * fixture.hidden_dim
        assert state.grad is not None
    for actual, expected in zip(vk.last_states, cpu.last_states):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)


def test_rnn_backward_exposes_non_overlapping_mode_timestamps(vulkan_backend):
    fixture, _, model, _, inputs, _, target = _pair(
        vulkan_backend, batch=16, sequence_length=64
    )
    pytorch_vulkan._C.reset_gpu_timing()
    pytorch_vulkan._C.begin_training_step()
    try:
        fixture.loss(model(inputs), target).backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    samples = pytorch_vulkan._C.gpu_timing_snapshot()
    modes = [sample for sample in samples if sample["scope"].startswith("rnn_backward_mode_")]
    assert {sample["scope"] for sample in modes} == {
        "rnn_backward_mode_1", "rnn_backward_mode_2",
        "rnn_backward_mode_3", "rnn_backward_mode_4",
    }
    assert all(sample["available"] and sample["gpu_time_ns"] > 0 for sample in modes)
    assert all(sum(sample["scope"] == scope for sample in samples) == 1
               for scope in {sample["scope"] for sample in modes})
    assert sum(sample["scope"] == "training" for sample in samples) == 1


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
    assert vulkan_row["transfer_operations"] == 0
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
        _resident_view(state)


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
