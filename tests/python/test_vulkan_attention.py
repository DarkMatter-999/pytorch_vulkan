import pytest
import torch

import pytorch_vulkan


# Documented three-repetition baseline normalized to one bounded step.
ATTENTION_PRE_OPTIMIZATION_DISPATCHES_PER_STEP = 94
ATTENTION_PRE_OPTIMIZATION_SUBMISSIONS_PER_STEP = 33
from tools.vulkan_model_benchmark import AttentionFixture, _move_model


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _assert_resident(value):
    assert value.device == torch.device("vk:0")
    assert value.dtype is torch.float32
    assert value.is_contiguous()


def _pair(device, batch=4):
    fixture = AttentionFixture(batch=batch)
    cpu = fixture.make_cpu()
    vk = _move_model(fixture.make_cpu(), device)
    inputs, target = fixture.make_inputs()
    for name, parameter in list(vk.named_parameters()):
        parent_name, _, parameter_name = name.rpartition(".")
        parent = vk.get_submodule(parent_name) if parent_name else vk
        setattr(parent, parameter_name, torch.nn.Parameter(cpu.state_dict()[name].clone().to(device)))
    vk_input = inputs.to(device).requires_grad_()
    vk_input.retain_grad()
    return fixture, cpu, vk, inputs.requires_grad_(), vk_input, target, target.to(device)


def test_attention_forward_backward_matches_cpu_and_stays_resident(vulkan_backend):
    fixture, cpu, vk, cpu_input, vk_input, cpu_target, vk_target = _pair(vulkan_backend)
    cpu_output = cpu(cpu_input)
    cpu_loss = fixture.loss(cpu_output, cpu_target)
    cpu_loss.backward()
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_output = vk(vk_input)
        vk_loss = fixture.loss(vk_output, vk_target)
        vk_loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    compute_submissions = pytorch_vulkan._C.compute_submitted_count()
    transfer_operations = pytorch_vulkan._C.transfer_operation_count()
    transfer_submissions = pytorch_vulkan._C.transfer_submission_count()
    transfer_completions = pytorch_vulkan._C.transfer_completion_count()
    transfer_waits = pytorch_vulkan._C.transfer_wait_count()
    assert vk_output.shape == (4, 128, 256)
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss.detach(), rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-3, atol=3e-3)
    for actual, expected in zip(vk.parameters(), cpu.parameters()):
        _assert_resident(actual)
        _assert_resident(actual.grad)
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    for value in (vk_output, vk_loss, vk_input, vk_input.grad):
        _assert_resident(value)
    assert dispatches > 0
    assert transfers == 0
    assert fallbacks == 0
    assert compute_submissions == 1
    assert transfer_operations == 0
    assert transfer_submissions == 0
    assert transfer_completions == 0
    assert transfer_waits == 0


@pytest.mark.parametrize("rows,features,outputs", [(37, 13, 11), (129, 32, 19), (1025, 64, 33)])
def test_linear_dinput_generic_shapes_match_cpu(vulkan_backend, rows, features, outputs):
    cpu_weight = torch.randn(outputs, features)
    cpu_grad = torch.randn(rows, outputs)
    vk_weight = cpu_weight.to(vulkan_backend)
    vk_grad = cpu_grad.to(vulkan_backend)
    vk_input = torch.empty((rows, features), device=vulkan_backend, requires_grad=True)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_grad)
    actual = vk_input.grad
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    expected = cpu_grad @ cpu_weight

    actual_cpu = actual.cpu()
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(actual_cpu, expected, rtol=3e-3, atol=3e-3)
    assert transfers == 0
    assert fallbacks == 0
    assert dispatches == 1



@pytest.mark.parametrize("rows,features,outputs", [(37, 13, 11), (129, 32, 19), (1025, 64, 33), (1024, 256, 256), (8192, 256, 256)])
def test_linear_dweight_generic_shapes_match_cpu(vulkan_backend, rows, features, outputs):
    cpu_input = torch.randn(rows, features)
    cpu_grad = torch.randn(rows, outputs)
    vk_input = cpu_input.to(vulkan_backend)
    vk_weight = torch.randn(outputs, features).to(vulkan_backend).requires_grad_()
    vk_output = torch.nn.functional.linear(vk_input, vk_weight)
    vk_grad = cpu_grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_grad)
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    actual_cpu = vk_weight.grad.cpu()
    expected = cpu_grad.T @ cpu_input
    assert torch.isfinite(actual_cpu).all()
    torch.testing.assert_close(actual_cpu, expected, rtol=3e-3, atol=3e-3)
    assert transfers == 0
    assert fallbacks == 0
    assert dispatches == 1


def test_linear_dweight_uses_gemm_scope_when_timestamps_supported(vulkan_backend):
    cpu_input = torch.randn(37, 13)
    cpu_grad = torch.randn(37, 11)
    vk_input = cpu_input.to(vulkan_backend)
    vk_weight = torch.randn(11, 13).to(vulkan_backend).requires_grad_()
    vk_grad = cpu_grad.to(vulkan_backend)
    vk_output = torch.nn.functional.linear(vk_input, vk_weight)
    if not pytorch_vulkan._C.timestamp_queries_supported():
        pytest.skip("Vulkan timestamp queries unsupported")
    pytorch_vulkan._C.reset_gpu_timing()
    pytorch_vulkan._C.reset_execution_counters()
    vk_output.backward(vk_grad)
    samples = pytorch_vulkan._C.gpu_timing_snapshot()
    assert samples
    assert [sample["scope"] for sample in samples] == ["gemm"]

@pytest.mark.parametrize(
    "requires_grad,has_bias,expected_dispatches",
    [
        ((True, False, False), True, 1),
        ((False, True, False), True, 1),
        ((False, False, True), True, 1),
        ((True, True, False), True, 2),
        ((True, True, True), True, 3),
        ((False, False, False), True, 0),
        ((True, False, False), False, 1),
        ((False, True, False), False, 1),
    ],
)
def test_linear_backward_only_computes_requested_gradients(
    vulkan_backend, requires_grad, has_bias, expected_dispatches
):
    torch.manual_seed(451)
    rows, features, outputs = 5, 7, 3
    x = torch.randn(rows, features)
    weight = torch.randn(outputs, features)
    bias = torch.randn(outputs) if has_bias else None
    grad = torch.randn(rows, outputs)
    vk_x = x.to(vulkan_backend).requires_grad_(requires_grad[0])
    vk_weight = weight.to(vulkan_backend).requires_grad_(requires_grad[1])
    vk_bias = bias.to(vulkan_backend).requires_grad_(requires_grad[2]) if bias is not None else None
    cpu_x = x.requires_grad_(requires_grad[0])
    cpu_weight = weight.requires_grad_(requires_grad[1])
    cpu_bias = bias.requires_grad_(requires_grad[2]) if bias is not None else None

    cpu_out = torch.nn.functional.linear(cpu_x, cpu_weight, cpu_bias)
    vk_out = torch.nn.functional.linear(vk_x, vk_weight, vk_bias)
    vk_grad = grad.to(vulkan_backend)
    if any(requires_grad):
        cpu_out.backward(grad)
    pytorch_vulkan._C.reset_execution_counters()
    if any(requires_grad):
        vk_out.backward(vk_grad)
    assert pytorch_vulkan._C.compute_dispatch_count() == expected_dispatches
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0

    for vk_leaf, cpu_leaf, requested in (
        (vk_x, cpu_x, requires_grad[0]),
        (vk_weight, cpu_weight, requires_grad[1]),
        (vk_bias, cpu_bias, requires_grad[2] if has_bias else False),
    ):
        if vk_leaf is None:
            continue
        assert (vk_leaf.grad is not None) == requested
        if requested:
            torch.testing.assert_close(vk_leaf.grad.cpu(), cpu_leaf.grad, rtol=3e-3, atol=3e-3)




def test_attention_supported_additive_mask_matches_cpu_and_gradients(vulkan_backend):
    fixture, cpu, vk, cpu_input, vk_input, cpu_target, vk_target = _pair(vulkan_backend)
    mask = cpu.causal_mask
    cpu_output = cpu(cpu_input, mask)
    cpu_loss = fixture.loss(cpu_output, cpu_target)
    cpu_loss.backward()
    pytorch_vulkan._C.reset_execution_counters()
    vk_mask = vk.causal_mask
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_output = vk(vk_input, vk_mask)
        vk_loss = fixture.loss(vk_output, vk_target)
        vk_loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=3e-3, atol=3e-3)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=3e-3, atol=3e-3)
    for actual, expected in zip(vk.parameters(), cpu.parameters()):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("batch", [8, 64])
def test_attention_training_dweight_batches_match_cpu_in_single_scope(vulkan_backend, batch):
    fixture, cpu, vk, cpu_input, vk_input, cpu_target, vk_target = _pair(
        vulkan_backend, batch=batch
    )
    cpu_loss = fixture.loss(cpu(cpu_input), cpu_target)
    cpu_loss.backward()

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss = fixture.loss(vk(vk_input), vk_target)
        vk_loss.backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()

    counters = pytorch_vulkan._C.execution_counter_snapshot()
    for actual, expected in zip(vk.parameters(), cpu.parameters()):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=3e-3, atol=3e-3)
    assert torch.isfinite(vk_input.grad.cpu()).all()
    assert counters[1] == 0
    assert counters[2] == 0
    assert counters[3] == 0
    assert pytorch_vulkan._C.compute_submitted_count() == 1
    assert pytorch_vulkan._C.compute_completed_count() == 1
    assert pytorch_vulkan._C.compute_wait_count() == 1


def test_attention_causal_mask_changes_future_scores(vulkan_backend):
    fixture = AttentionFixture(batch=1)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    valid = inputs.to(vulkan_backend)
    unmasked = model(valid)
    masked = model(valid, model.causal_mask)
    assert not torch.equal(unmasked.cpu(), masked.cpu())


def test_attention_rejects_invalid_causal_mask_values_before_work(vulkan_backend):
    fixture = AttentionFixture()
    model = fixture.make_cpu()
    inputs, _ = fixture.make_inputs()
    mask = fixture.make_mask()
    mask[0, 0, 1] = -1.0
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="causal|mask"):
        model(inputs, mask)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_mask_intermediates_stay_resident(vulkan_backend):
    fixture = AttentionFixture(batch=1)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    output = model(inputs.to(vulkan_backend), model.causal_mask)
    _assert_resident(output)
    for value in model.last_intermediates.values():
        _assert_resident(value)


def test_attention_backward_accumulates_gradients_out_of_place(vulkan_backend):
    fixture = AttentionFixture(batch=1)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, target = fixture.make_inputs()
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        output = model(inputs.to(vulkan_backend).requires_grad_())
        fixture.loss(output, target.to(vulkan_backend)).backward()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    pytorch_vulkan._C.end_training_step()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("batch", [1, 4])
def test_attention_supported_batch_boundaries(vulkan_backend, batch):
    fixture = AttentionFixture(batch=batch)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    output = model(inputs.to(vulkan_backend))
    _assert_resident(output)
    assert output.shape == (batch, 128, 256)


@pytest.mark.parametrize("mutator, message", [
    (lambda x: torch.empty(x.shape, device=x.device, dtype=torch.float64), "float32|dtype|type"),
    (lambda x: x.transpose(1, 2), "contiguous|layout|shape"),
    (lambda x: x[:, :-1], "shape"),
])
def test_attention_rejects_unsupported_contract_before_work(vulkan_backend, mutator, message):
    fixture = AttentionFixture()
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    valid = inputs.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=message):
        model(mutator(valid))
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("mutator, message", [
    (lambda mask: torch.empty(mask.shape, device=mask.device, dtype=torch.float64), "float32|dtype|type"),
    (lambda mask: mask[:, :-1], "shape"),
    (lambda mask: mask.transpose(1, 2), "contiguous|layout"),
])
def test_attention_rejects_unsupported_mask_before_work(vulkan_backend, mutator, message):
    fixture = AttentionFixture()
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    valid_input = inputs.to(vulkan_backend)
    valid_mask = model.causal_mask
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=message):
        model(valid_input, mutator(valid_mask))
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_accepts_only_registered_mask_identity(vulkan_backend):
    fixture = AttentionFixture(batch=1)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    valid_input = fixture.make_inputs()[0].to(vulkan_backend)
    clone = model.causal_mask.clone()
    altered = model.causal_mask.clone()
    altered[0, 0, 1] = 0.0
    for mask in (clone, altered):
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(RuntimeError, match="registered|identity|mask"):
            model(valid_input, mask)
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
        assert pytorch_vulkan._C.vulkan_copy_count() == 0
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
        assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_rejects_mask_on_wrong_device_before_work(vulkan_backend):
    fixture = AttentionFixture()
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    valid_input = inputs.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="device"):
        model(valid_input, fixture.make_mask())
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_rejects_malformed_bmm_and_linear_before_work(vulkan_backend):
    fixture = AttentionFixture(batch=1)
    model = fixture.make_cpu()
    valid = fixture.make_inputs()[0].to(vulkan_backend)
    weight = model.query.weight.to(vulkan_backend)
    bad_bias = torch.zeros((255,), device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="bias|linear"):
        torch.nn.functional.linear(valid, weight, bad_bias)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    bad_rhs = torch.ones((1, 255, 128), device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="bmm|shape"):
        torch.bmm(valid, bad_rhs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_rejects_unsupported_bmm_slice_layout_before_work(vulkan_backend):
    lhs = torch.ones((1, 2, 4), device=vulkan_backend)
    rhs = torch.ones((1, 4, 2), device=vulkan_backend)
    padded = torch.empty((1, 2, 8), device=vulkan_backend)
    lhs_strided = padded[..., ::2]
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="layout|contiguous|transpose|bmm"):
        torch.bmm(lhs_strided, rhs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_bmm_uses_one_batched_dispatch_for_contiguous_layouts(vulkan_backend):
    torch.manual_seed(71)
    lhs_cpu = torch.randn(3, 5, 7)
    rhs_cpu = torch.randn(3, 7, 4)
    lhs = lhs_cpu.to(vulkan_backend)
    rhs = rhs_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.bmm(lhs, rhs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    assert pytorch_vulkan._C.compute_dispatch_count() < ATTENTION_PRE_OPTIMIZATION_DISPATCHES_PER_STEP
    assert pytorch_vulkan._C.compute_submitted_count() < ATTENTION_PRE_OPTIMIZATION_SUBMISSIONS_PER_STEP
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(actual.cpu(), torch.bmm(lhs_cpu, rhs_cpu), rtol=2e-3, atol=2e-3)


def test_attention_bmm_rejects_batch_count_above_vulkan_z_limit_before_work(
    vulkan_backend,
):
    lhs = torch.ones((1 << 16, 1, 1), device=vulkan_backend)
    rhs = torch.ones((1 << 16, 1, 1), device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="device limits|batch|workgroup"):
        torch.bmm(lhs, rhs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_attention_bmm_uses_one_batched_dispatch_without_transpose_materialization(
    vulkan_backend,
):
    torch.manual_seed(73)
    query_cpu = torch.randn(3, 5, 7)
    key_cpu = torch.randn(3, 4, 7)
    query = query_cpu.to(vulkan_backend)
    key = key_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.bmm(query, key.transpose(1, 2))
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    assert pytorch_vulkan._C.compute_dispatch_count() < ATTENTION_PRE_OPTIMIZATION_DISPATCHES_PER_STEP
    assert pytorch_vulkan._C.compute_submitted_count() < ATTENTION_PRE_OPTIMIZATION_SUBMISSIONS_PER_STEP
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(
        actual.cpu(), torch.bmm(query_cpu, key_cpu.transpose(1, 2)), rtol=2e-3, atol=2e-3
    )


def test_attention_bmm_backward_rejects_unsupported_grad_layout_before_work(vulkan_backend):
    lhs = torch.ones((1, 2, 4), device=vulkan_backend, requires_grad=True)
    rhs = torch.ones((1, 4, 2), device=vulkan_backend)
    actual = torch.bmm(lhs, rhs)
    padded_grad = torch.ones((1, 2, 4), device=vulkan_backend)
    unsupported_grad = padded_grad[..., ::2]

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="layout|contiguous|transpose|bmm"):
        actual.backward(unsupported_grad)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("batch", [8, 64])
@pytest.mark.parametrize(
    "needs_lhs, needs_rhs", [(False, False), (False, True), (True, False), (True, True)]
)
def test_attention_bmm_backward_uses_transposed_contiguous_operands_without_extra_work(
    vulkan_backend, batch, needs_lhs, needs_rhs
):
    """Catches a BMM backward regression that materializes transpose views or falls back."""
    torch.manual_seed(79 + batch)
    lhs_cpu = torch.randn(batch, 5, 7, requires_grad=needs_lhs)
    key_cpu = torch.randn(batch, 4, 7, requires_grad=needs_rhs)
    lhs = lhs_cpu.detach().to(vulkan_backend).requires_grad_(needs_lhs)
    key = key_cpu.detach().to(vulkan_backend).requires_grad_(needs_rhs)
    expected = torch.bmm(lhs_cpu, key_cpu.transpose(1, 2))
    grad_output_cpu = torch.ones_like(expected)
    grad_output = grad_output_cpu.to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.bmm(lhs, key.transpose(1, 2))
    forward_dispatches = pytorch_vulkan._C.compute_dispatch_count()
    forward_copies = pytorch_vulkan._C.vulkan_copy_count()
    forward_transfers = pytorch_vulkan._C.explicit_transfer_count()
    forward_fallbacks = pytorch_vulkan._C.fallback_count()

    assert actual.shape == expected.shape
    assert forward_dispatches == 1
    assert forward_copies == 0
    assert forward_transfers == 0
    assert forward_fallbacks == 0

    if not (needs_lhs or needs_rhs):
        assert not actual.requires_grad
        assert lhs.grad is None
        assert key.grad is None
        return

    expected.backward(grad_output_cpu)
    actual.backward(grad_output)
    backward_dispatches = pytorch_vulkan._C.compute_dispatch_count() - forward_dispatches
    backward_copies = pytorch_vulkan._C.vulkan_copy_count() - forward_copies
    backward_transfers = pytorch_vulkan._C.explicit_transfer_count() - forward_transfers
    backward_fallbacks = pytorch_vulkan._C.fallback_count() - forward_fallbacks

    assert backward_dispatches == int(needs_lhs) + int(needs_rhs)
    assert backward_copies == 0
    assert backward_transfers == 0
    assert backward_fallbacks == 0
    if needs_lhs:
        torch.testing.assert_close(lhs.grad.cpu(), lhs_cpu.grad, rtol=2e-3, atol=2e-3)
    else:
        assert lhs.grad is None
    if needs_rhs:
        torch.testing.assert_close(key.grad.cpu(), key_cpu.grad, rtol=2e-3, atol=2e-3)
    else:
        assert key.grad is None
