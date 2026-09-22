import subprocess
import re
import os
import sys
import textwrap
import gc
from pathlib import Path

import pytest
import torch
import pytorch_vulkan


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


def _without_comments(source):
    return re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)


def test_rnn_sequence_shader_contract_and_integrity():
    source = (ROOT / "src/vulkan/shaders/glsl/rnn_sequence.comp").read_text(
        encoding="ascii"
    )
    code = _without_comments(source)
    for token in (
        "binding = 0",
        "binding = 1",
        "binding = 2",
        "binding = 3",
        "binding = 4",
        "binding = 5",
        "binding = 6",
        "binding = 7",
        "binding = 8",
        "binding = 9",
        "binding = 10",
        "binding = 11",
        "binding = 12",
        "binding = 13",
        "output_buffer",
        "saved_state",
        "gradient_output",
        "gradient_input",
        "tanh(",
        "params.mode == 1u",
        "params.mode == 2u",
        "params.mode == 3u",
        "params.mode == 4u",
    ):
        assert token in code
    assert re.search(
        r"for\s*\(\s*uint\s+sequence_index\s*=.*?sequence_index\s*<\s*params\.sequence",
        code,
        flags=re.DOTALL,
    )

    result = subprocess.run(
        ["python3", str(ROOT / "tools/verify_rnn_sequence_spv.py")],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "rnn_sequence shader contract=ok" in result.stdout


def _cpu_reference(input_tensor, weight, recurrent_weight, bias):
    state = torch.zeros(input_tensor.size(0), weight.size(0), dtype=input_tensor.dtype)
    outputs = []
    for step in range(input_tensor.size(1)):
        next_state = torch.empty_like(state)
        for batch_index in range(input_tensor.size(0)):
            for hidden in range(weight.size(0)):
                value = bias[hidden].item()
                for input_index in range(weight.size(1)):
                    value += (input_tensor[batch_index, step, input_index] *
                              weight[hidden, input_index]).item()
                for previous_hidden in range(weight.size(0)):
                    value += (state[batch_index, previous_hidden] *
                              recurrent_weight[hidden, previous_hidden]).item()
                next_state[batch_index, hidden] = torch.tanh(torch.tensor(value))
        state = next_state
        outputs.append(state)
    return torch.stack(outputs, dim=1)


def _differentiable_cpu_reference(input_tensor, weight, recurrent_weight, bias):
    state = torch.zeros(
        input_tensor.size(0), weight.size(0), dtype=input_tensor.dtype,
        device=input_tensor.device,
    )
    outputs = []
    for step in range(input_tensor.size(1)):
        state = torch.tanh(
            input_tensor[:, step, :] @ weight.t() + state @ recurrent_weight.t() + bias
        )
        outputs.append(state)
    return torch.stack(outputs, dim=1)


@pytest.mark.parametrize("batch,sequence", [(1, 1), (1, 64), (16, 1), (16, 64)])
def test_rnn_sequence_forward_parity_and_single_dispatch(vulkan_backend, batch, sequence):
    torch.manual_seed(41 + batch + sequence)
    input_cpu = torch.randn(batch, sequence, 8)
    weight_cpu = torch.randn(16, 8) * 0.1
    recurrent_cpu = torch.randn(16, 16) * 0.1
    bias_cpu = torch.randn(16) * 0.1
    input_vk = input_cpu.to(vulkan_backend)
    weight_vk = weight_cpu.to(vulkan_backend)
    recurrent_vk = recurrent_cpu.to(vulkan_backend)
    bias_vk = bias_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.ops.pytorch_vulkan.rnn_sequence(
        input_vk, weight_vk, recurrent_vk, bias_vk
    )
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] == 1
    assert counters[2:] == (0, 0)
    services = pytorch_vulkan._C.shared_service_snapshot()
    resources = pytorch_vulkan._C.live_resource_snapshot()
    # Six legacy facade-owned resources are intentionally outside the shared
    # services. RNN resources must be represented by the shared snapshots only.
    assert resources[2] == services["pipeline_count"] + 6
    assert resources[3] == services["shader_modules"] + 6
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1
    torch.testing.assert_close(
        result.cpu(), _cpu_reference(input_cpu, weight_cpu, recurrent_cpu, bias_cpu),
        rtol=3e-3,
        atol=3e-3,
    )


def _assert_rnn_rejected_without_activity(call):
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError)):
        call()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def _rnn_inputs(device, *, batch=2, sequence=3, input_dim=4, hidden=5):
    return (
        torch.randn(batch, sequence, input_dim).to(device),
        torch.randn(hidden, input_dim).to(device),
        torch.randn(hidden, hidden).to(device),
        torch.randn(hidden).to(device),
    )


def test_rnn_sequence_rejects_wrong_device_dtype_and_rank_without_activity(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend)
    cpu_inputs = _rnn_inputs("cpu")
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(*cpu_inputs)
    )
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            torch.empty_like(inputs[0], dtype=torch.float64),
            inputs[1],
            inputs[2],
            inputs[3],
        )
    )
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            inputs[0][0], inputs[1], inputs[2], inputs[3]
        )
    )


def test_rnn_sequence_rejects_noncontiguous_and_mismatched_dimensions_without_activity(
    vulkan_backend,
):
    inputs = _rnn_inputs(vulkan_backend)
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            inputs[0].transpose(1, 2), inputs[1], inputs[2], inputs[3]
        )
    )
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            inputs[0], torch.randn(9, 5, device=vulkan_backend), inputs[2], inputs[3]
        )
    )
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            inputs[0], inputs[1], inputs[2], torch.randn(6, device=vulkan_backend)
        )
    )
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(
            inputs[0], inputs[1], torch.randn(6, 6, device=vulkan_backend), inputs[3]
        )
    )


@pytest.mark.parametrize("batch,sequence", [(17, 3), (2, 65)])
def test_rnn_sequence_rejects_contract_limits_without_activity(vulkan_backend, batch, sequence):
    inputs = _rnn_inputs(vulkan_backend, batch=batch, sequence=sequence)
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
    )


def test_rnn_sequence_rejects_wrong_device_index_without_activity(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend)
    backend_name = torch._C._get_privateuse1_backend_name()

    def call():
        try:
            wrong_input = torch.empty(
                inputs[0].shape, dtype=torch.float32, device=f"{backend_name}:1"
            )
        except RuntimeError:
            # A single-device runner cannot materialize :1.  Use the stable
            # mixed-device operator validation path instead, while still
            # reaching the fused operator call.
            wrong_input = torch.empty_like(inputs[0], device="cpu")
        return torch.ops.pytorch_vulkan.rnn_sequence(
            wrong_input, inputs[1], inputs[2], inputs[3]
        )

    _assert_rnn_rejected_without_activity(call)


def test_rnn_sequence_rejects_oversized_workgroup_dimension_without_activity(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend, hidden=257)
    _assert_rnn_rejected_without_activity(
        lambda: torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
    )


def test_rnn_sequence_training_scope_defers_submission_and_wait(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        result = torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
        assert result.shape == (2, 3, 5)
        assert pytorch_vulkan._C.compute_submitted_count() == 0
        assert pytorch_vulkan._C.compute_completed_count() == 0
        assert pytorch_vulkan._C.compute_wait_count() == 0
        assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1


def test_rnn_sequence_cancellation_cleans_pending_work_and_allows_next_step(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend)
    # Prime the reusable descriptor pools so the baseline describes the
    # steady-state shared resources rather than first-use pool creation.
    pytorch_vulkan._C.begin_training_step()
    warmup = torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
    pytorch_vulkan._C.end_training_step()
    del warmup
    # Discard cyclic autograd/runtime objects from preceding suite tests before
    # capturing the exact live-allocation baseline.
    gc.collect()
    baseline_resources = pytorch_vulkan._C.live_resource_snapshot()
    baseline_services = pytorch_vulkan._C.shared_service_snapshot()
    baseline_timestamp = pytorch_vulkan._C.timestamp_query_snapshot()
    baseline_pending_compute = pytorch_vulkan._C.pending_compute_count()
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    output = None
    with pytest.raises((RuntimeError, ValueError), match="sequence|shape"):
        output = torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
        assert pytorch_vulkan._C.execution_counter_snapshot()[0] > 0
        bad_input = torch.empty((2, 65, 4), device=vulkan_backend)
        torch.ops.pytorch_vulkan.rnn_sequence(
            bad_input, inputs[1], inputs[2], inputs[3]
        )
    pytorch_vulkan._C.cancel_training_step()
    del output
    del bad_input
    gc.collect()
    assert not pytorch_vulkan._C.training_step_active()
    assert pytorch_vulkan._C.pending_compute_count() == baseline_pending_compute
    resources = pytorch_vulkan._C.live_resource_snapshot()
    assert resources[:6] == baseline_resources[:6]
    assert resources[6] == baseline_resources[6]
    services = pytorch_vulkan._C.shared_service_snapshot()
    for key in (
        "descriptor_sets",
        "descriptor_pending",
        "descriptor_quarantined",
        "pipeline_pending_destructions",
        "pipeline_count",
    ):
        assert services[key] == baseline_services[key], key
    assert pytorch_vulkan._C.timestamp_query_snapshot() == baseline_timestamp

    pytorch_vulkan._C.begin_training_step()
    try:
        torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    assert not pytorch_vulkan._C.training_step_active()
    assert pytorch_vulkan._C.pending_compute_count() == 0
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1


def test_rnn_sequence_reuses_descriptors_and_shared_resources(vulkan_backend):
    inputs = _rnn_inputs(vulkan_backend)
    baseline_resources = pytorch_vulkan._C.live_resource_snapshot()
    baseline_services = pytorch_vulkan._C.shared_service_snapshot()
    pytorch_vulkan._C.reset_descriptor_resource_counters()

    for _ in range(2):
        pytorch_vulkan._C.reset_execution_counters()
        pytorch_vulkan._C.begin_training_step()
        try:
            torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
            pytorch_vulkan._C.end_training_step()
        except BaseException:
            pytorch_vulkan._C.cancel_training_step()
            raise
        assert pytorch_vulkan._C.pending_compute_count() == 0

    services = pytorch_vulkan._C.shared_service_snapshot()
    resources = pytorch_vulkan._C.live_resource_snapshot()
    assert services["descriptor_pools"] <= services["descriptor_pool_limit"]
    assert services["descriptor_allocations"] + services["descriptor_reuses"] >= 1
    assert services["pipeline_entries"] >= baseline_services["pipeline_entries"]
    assert services["shader_modules"] >= baseline_services["shader_modules"]
    assert resources[2] == baseline_resources[2]
    assert resources[3] == baseline_resources[3]
    assert resources[4] == 0
    assert resources[5] == 0
    assert resources == baseline_resources
    for key in (
        "descriptor_sets",
        "descriptor_pending",
        "descriptor_quarantined",
        "pipeline_pending_destructions",
        "pipeline_count",
    ):
        assert services[key] == baseline_services[key], key


def test_vanilla_rnn_training_accepts_standard_linear_parameter_layout(vulkan_backend):
    from tools.vulkan_model_benchmark import _VanillaRNN, _move_model

    model = _move_model(_VanillaRNN(4, 5), vulkan_backend)
    inputs = torch.randn(2, 3, 4).to(vulkan_backend).requires_grad_()
    target = torch.randn(2, 3, 5).to(vulkan_backend)
    pytorch_vulkan._C.begin_training_step()
    try:
        loss = torch.nn.functional.mse_loss(model(inputs), target)
        loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    assert model.input.weight.grad is not None
    assert model.hidden.weight.grad is not None
    assert model.input.weight.grad.shape == model.input.weight.shape
    assert model.hidden.weight.grad.shape == model.hidden.weight.shape


@pytest.mark.parametrize("batch,sequence", [(1, 1), (1, 64), (16, 1), (16, 64)])
def test_rnn_sequence_gradient_parity_at_supported_boundaries(vulkan_backend, batch, sequence):
    torch.manual_seed(701 + batch + sequence)
    cpu_inputs = [
        torch.randn(batch, sequence, 8, requires_grad=True),
        (torch.randn(16, 8) * 0.1).requires_grad_(),
        (torch.randn(16, 16) * 0.1).requires_grad_(),
        (torch.randn(16) * 0.1).requires_grad_(),
    ]
    vk_inputs = [value.detach().clone().to(vulkan_backend).requires_grad_() for value in cpu_inputs]
    cpu_output = _differentiable_cpu_reference(*cpu_inputs)
    grad_output = torch.randn_like(cpu_output)
    grad_output_vk = grad_output.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output = torch.ops.pytorch_vulkan.rnn_sequence(*vk_inputs)
    cpu_output.backward(grad_output)
    vk_output.backward(grad_output_vk)

    assert pytorch_vulkan._C.execution_counter_snapshot() == (5, 0, 0, 0)
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=3e-3, atol=3e-3)
    for actual, expected in zip(vk_inputs, cpu_inputs):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    assert pytorch_vulkan._C.compute_dispatch_count() == 5


def test_rnn_sequence_retained_last_state_view_gradients_match(vulkan_backend):
    torch.manual_seed(733)
    cpu_inputs = [
        torch.randn(2, 3, 4, requires_grad=True),
        (torch.randn(5, 4) * 0.1).requires_grad_(),
        (torch.randn(5, 5) * 0.1).requires_grad_(),
        (torch.randn(5) * 0.1).requires_grad_(),
    ]
    vk_inputs = [value.detach().clone().to(vulkan_backend).requires_grad_() for value in cpu_inputs]
    cpu_output = _differentiable_cpu_reference(*cpu_inputs)
    vk_output = torch.ops.pytorch_vulkan.rnn_sequence(*vk_inputs)
    cpu_states = [cpu_output[:, -1, :]]
    vk_states = [vk_output[:, -1, :]]
    for state in cpu_states + vk_states:
        state.retain_grad()
    cpu_loss = cpu_states[0].sum()
    vk_loss = vk_states[0].sum()
    cpu_loss.backward()
    vk_loss.backward()

    torch.testing.assert_close(vk_states[0].grad.cpu(), cpu_states[0].grad,
                               rtol=3e-3, atol=3e-3)


def test_rnn_sequence_rejects_higher_order_gradient(vulkan_backend):
    inputs = [value.detach().clone().to(vulkan_backend).requires_grad_()
              for value in _rnn_inputs(vulkan_backend, batch=1, sequence=2)]
    output = torch.ops.pytorch_vulkan.rnn_sequence(*inputs)
    grad_output = torch.ones_like(output)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match="higher|second|order"):
        torch.autograd.grad(output, inputs, grad_outputs=grad_output, create_graph=True)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.compute_submitted_count() == 0
    assert pytorch_vulkan._C.compute_completed_count() == 0
    assert pytorch_vulkan._C.compute_wait_count() == 0
    assert not pytorch_vulkan._C.training_step_active()


def test_rnn_sequence_rejects_new_submission_after_device_loss(vulkan_backend):
    script = textwrap.dedent(
        """
        import torch
        import pytorch_vulkan

        device = "vk:0"
        values = [
            torch.randn(1, 2, 4).to(device),
            torch.randn(5, 4).to(device),
            torch.randn(5, 5).to(device),
            torch.randn(5).to(device),
        ]
        pytorch_vulkan._C.begin_training_step()
        torch.ops.pytorch_vulkan.rnn_sequence(*values)
        pytorch_vulkan._C.test_inject_device_loss()
        before = pytorch_vulkan._C.execution_counter_snapshot()
        try:
            torch.ops.pytorch_vulkan.rnn_sequence(*values)
        except RuntimeError as error:
            assert "Vulkan device lost" in str(error)
        else:
            raise AssertionError("device-loss context accepted fused RNN work")
        finally:
            pytorch_vulkan._C.cancel_training_step()
        assert not pytorch_vulkan._C.training_step_active()
        assert pytorch_vulkan._C.pending_compute_count() == 0
        assert pytorch_vulkan._C.execution_counter_snapshot() == before
        print("fused-rnn-device-loss-ok")
        """
    )
    environment = os.environ.copy()
    environment.pop("VK_INSTANCE_LAYERS", None)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "build"), environment.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    result = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "fused-rnn-device-loss-ok" in result.stdout


def test_rnn_sequence_training_step_updates_parameters_with_one_lifecycle(vulkan_backend):
    torch.manual_seed(761)
    cpu_parameters = [
        (torch.randn(5, 4) * 0.1).detach().requires_grad_(),
        (torch.randn(5, 5) * 0.1).detach().requires_grad_(),
        (torch.randn(5) * 0.1).detach().requires_grad_(),
    ]
    vk_parameters = [value.detach().clone().to(vulkan_backend).requires_grad_()
                     for value in cpu_parameters]
    cpu_input = torch.randn(2, 3, 4, requires_grad=True)
    vk_input = cpu_input.detach().clone().to(vulkan_backend).requires_grad_()
    cpu_optimizer = torch.optim.SGD(cpu_parameters, lr=0.05)
    vk_optimizer = torch.optim.SGD(vk_parameters, lr=0.05)
    cpu_loss = _differentiable_cpu_reference(
        cpu_input, cpu_parameters[0], cpu_parameters[1], cpu_parameters[2]
    )
    cpu_loss = (cpu_loss * cpu_loss).sum()
    cpu_loss.backward()
    cpu_optimizer.step()

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss = torch.ops.pytorch_vulkan.rnn_sequence(
            vk_input, vk_parameters[0], vk_parameters[1], vk_parameters[2]
        )
        vk_loss = (vk_loss * vk_loss).sum()
        vk_loss.backward()
        vk_optimizer.step()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    lifecycle = (
        pytorch_vulkan._C.compute_submitted_count(),
        pytorch_vulkan._C.compute_completed_count(),
        pytorch_vulkan._C.compute_wait_count(),
    )
    assert counters[2:] == (0, 0)
    assert lifecycle == (1, 1, 1)

    for actual, expected in zip(vk_parameters, cpu_parameters):
        torch.testing.assert_close(actual.cpu(), expected, rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-3, atol=3e-3)
    assert pytorch_vulkan._C.compute_dispatch_count() == 15
    assert counters[1] == 0
    assert counters[2] == 0
    assert pytorch_vulkan._C.fallback_count() == 0
