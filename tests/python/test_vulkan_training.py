import pytest
import torch

import pytorch_vulkan


def _make_mlp(device, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 4),
    ).to(device=device, dtype=torch.float32)


def _squared_error_loss(output, target):
    error = output - target
    return error.mul(error).sum()


def _assert_vk_f32_contiguous(tensor):
    assert tensor.device == torch.device("vk:0")
    assert tensor.dtype is torch.float32
    assert tensor.is_contiguous()


def _execution_counters():
    return (
        pytorch_vulkan._C.compute_dispatch_count(),
        pytorch_vulkan._C.explicit_transfer_count(),
    )


def _readback_with_counter_assertion(tensor, expected_transfer_count):
    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == expected_transfer_count
    return tensor.cpu()


def _make_training_pairs(device, seed):
    cpu_model = _make_mlp("cpu", seed)
    cpu_input = torch.randn(2, 8, dtype=torch.float32, requires_grad=True)
    cpu_target = torch.randn(2, 4, dtype=torch.float32)

    vk_model = _make_mlp(device, seed)
    vk_model.load_state_dict(
        {
            name: parameter.detach().clone().to(device)
            for name, parameter in cpu_model.state_dict().items()
        }
    )
    vk_input = cpu_input.detach().clone().to(device).requires_grad_()
    vk_target = cpu_target.detach().clone().to(device)
    return cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_mlp_fixture_has_expected_shape_dtype_and_device():
    model = _make_mlp("cpu", seed=17)

    assert [tuple(parameter.shape) for parameter in model.parameters()] == [
        (16, 8),
        (16,),
        (4, 16),
        (4,),
    ]
    for parameter in model.parameters():
        assert parameter.dtype is torch.float32
        assert parameter.device == torch.device("cpu")
        assert parameter.is_contiguous()


def test_mlp_fixture_seed_is_deterministic():
    first = _make_mlp("cpu", seed=23)
    second = _make_mlp("cpu", seed=23)

    for first_parameter, second_parameter in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_squared_error_loss_uses_scalar_sum():
    output = torch.tensor([[1.0, 3.0]], dtype=torch.float32)
    target = torch.tensor([[2.0, 1.0]], dtype=torch.float32)

    loss = _squared_error_loss(output, target)

    assert loss.shape == ()
    assert loss.dtype is torch.float32
    assert loss.item() == pytest.approx(5.0)


def test_vulkan_fixture_contract(vulkan_backend):
    model = _make_mlp(vulkan_backend, seed=29)

    for parameter in model.parameters():
        _assert_vk_f32_contiguous(parameter)


def test_cpu_and_vulkan_training_pairs_share_state(vulkan_backend):
    cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target = (
        _make_training_pairs(vulkan_backend, seed=31)
    )

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = vk_model(vk_input)
    cpu_output = cpu_model(cpu_input)

    _assert_vk_f32_contiguous(vk_output)
    for vk_parameter in vk_model.parameters():
        _assert_vk_f32_contiguous(vk_parameter)
    _assert_vk_f32_contiguous(vk_input)
    _assert_vk_f32_contiguous(vk_target)

    transfer_count = 0
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_output, transfer_count), cpu_output
    )
    transfer_count += 1
    for cpu_parameter, vk_parameter in zip(cpu_model.parameters(), vk_model.parameters()):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_parameter, transfer_count),
            cpu_parameter,
        )
        transfer_count += 1
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_input, transfer_count), cpu_input
    )
    transfer_count += 1
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_target, transfer_count), cpu_target
    )
    assert vk_input.requires_grad
    assert all(parameter.requires_grad for parameter in vk_model.parameters())


def test_counter_snapshot_precedes_final_readback(vulkan_backend):
    value = torch.ones(2, 8, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.neg(value)

    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0

    result.cpu()
    assert pytorch_vulkan._C.explicit_transfer_count() == 1


def test_fixed_mlp_forward_backward_runs_on_vulkan(vulkan_backend):
    cpu_model = _make_mlp("cpu", seed=17)
    vk_model = _make_mlp(vulkan_backend, seed=17)
    cpu_input = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)
    cpu_target = torch.randn(3, 4, dtype=torch.float32)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_target = cpu_target.to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = vk_model(vk_input)
    vk_loss = _squared_error_loss(vk_output, vk_target)
    _assert_vk_f32_contiguous(vk_output)
    _assert_vk_f32_contiguous(vk_loss)
    vk_loss.backward()

    for parameter in vk_model.parameters():
        _assert_vk_f32_contiguous(parameter.grad)
    _assert_vk_f32_contiguous(vk_input.grad)
    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0


def run_training_step(model, optimizer, inputs, targets):
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("training model must have parameters on vk:0")
    model_device = parameters[0].device
    if model_device.type == "vk" and model_device != torch.device("vk:0"):
        raise RuntimeError("training model must be on vk:0")
    if inputs.device != model_device or targets.device != model_device:
        raise RuntimeError(f"training inputs and targets must be on {model_device}")
    if inputs.dtype is not torch.float32 or targets.dtype is not torch.float32:
        raise RuntimeError("training inputs and targets must be float32")
    if not inputs.is_contiguous() or not targets.is_contiguous():
        raise RuntimeError("training inputs and targets must be contiguous")
    if inputs.ndim != 2 or inputs.shape[1] != 8:
        raise RuntimeError("training inputs must have shape (batch, 8)")
    if targets.ndim != 2 or targets.shape != (inputs.shape[0], 4):
        raise RuntimeError("training targets must have shape (batch, 4)")

    modules = list(model.children())
    if not isinstance(model, torch.nn.Sequential) or [type(module) for module in modules] != [
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]:
        raise RuntimeError("training model must be the fixed MLP")
    if (
        modules[0].in_features != 8
        or modules[0].out_features != 16
        or modules[2].in_features != 16
        or modules[2].out_features != 4
    ):
        raise RuntimeError("training model must be Linear(8,16)->ReLU->Linear(16,4)")
    for parameter in parameters:
        if model_device.type == "vk":
            _assert_vk_f32_contiguous(parameter)

    optimizer.zero_grad(set_to_none=False)
    output = model(inputs)
    loss = _squared_error_loss(output, targets)
    loss.backward()
    if model_device.type == "vk":
        _assert_vk_f32_contiguous(output)
        _assert_vk_f32_contiguous(loss)
        for parameter in parameters:
            _assert_vk_f32_contiguous(parameter.grad)
        if inputs.requires_grad:
            _assert_vk_f32_contiguous(inputs.grad)
    optimizer.step()
    return loss


def _make_sgd(parameters):
    return torch.optim.SGD(
        parameters,
        lr=0.1,
        momentum=0.9,
        dampening=0.0,
        weight_decay=0.01,
        nesterov=False,
        maximize=False,
        foreach=False,
        differentiable=False,
    )


def _make_adam(parameters):
    return torch.optim.Adam(
        parameters,
        lr=0.01,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        foreach=False,
        maximize=False,
        capturable=False,
        differentiable=False,
        fused=False,
    )


def _assert_optimizer_state_vulkan(optimizer):
    for state in optimizer.state.values():
        for name, value in state.items():
            if not isinstance(value, torch.Tensor):
                continue
            if name == "step":
                assert isinstance(optimizer, torch.optim.Adam)
                assert value.device == torch.device("cpu")
                assert value.ndim == 0
                assert value.dtype is torch.float32
            else:
                _assert_vk_f32_contiguous(value)


def test_fixed_mlp_forward_backward_matches_cpu(vulkan_backend):
    cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target = (
        _make_training_pairs(vulkan_backend, seed=41)
    )
    cpu_output = cpu_model(cpu_input)
    cpu_loss = _squared_error_loss(cpu_output, cpu_target)
    cpu_loss.backward()

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = vk_model(vk_input)
    vk_loss = _squared_error_loss(vk_output, vk_target)
    _assert_vk_f32_contiguous(vk_output)
    _assert_vk_f32_contiguous(vk_loss)
    vk_loss.backward()
    for parameter in vk_model.parameters():
        _assert_vk_f32_contiguous(parameter.grad)
    _assert_vk_f32_contiguous(vk_input.grad)
    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0

    transfer_count = 0
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_output, transfer_count), cpu_output.detach()
    )
    transfer_count += 1
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_loss, transfer_count), cpu_loss.detach()
    )
    transfer_count += 1
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_input.grad, transfer_count), cpu_input.grad
    )
    transfer_count += 1
    for cpu_parameter, vk_parameter in zip(cpu_model.parameters(), vk_model.parameters()):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_parameter.grad, transfer_count),
            cpu_parameter.grad,
        )
        transfer_count += 1


def _run_fixed_mlp(device, optimizer_factory, inputs, targets, steps):
    model = _make_mlp(device, seed=53)
    optimizer = optimizer_factory(model.parameters())
    losses = []
    if device != "cpu":
        counters = []
    for _ in range(steps):
        if device != "cpu":
            pytorch_vulkan._C.reset_execution_counters()
        loss = run_training_step(model, optimizer, inputs, targets)
        if device != "cpu":
            for parameter in model.parameters():
                _assert_vk_f32_contiguous(parameter)
                _assert_vk_f32_contiguous(parameter.grad)
            _assert_optimizer_state_vulkan(optimizer)
            counters.append(_execution_counters())
        losses.append(loss.detach())
    return model, optimizer, losses, counters if device != "cpu" else None


@pytest.mark.parametrize("optimizer_factory", [_make_sgd, _make_adam])
def test_fixed_mlp_sgd_and_adam_match_cpu(optimizer_factory, vulkan_backend):
    cpu_input = torch.randn(3, 8, dtype=torch.float32)
    cpu_target = torch.randn(3, 4, dtype=torch.float32)
    vk_input = cpu_input.to(vulkan_backend)
    vk_target = cpu_target.to(vulkan_backend)
    cpu_model, cpu_optimizer, cpu_losses, _ = _run_fixed_mlp(
        "cpu", optimizer_factory, cpu_input, cpu_target, steps=4
    )
    vk_model, vk_optimizer, vk_losses, counters = _run_fixed_mlp(
        vulkan_backend, optimizer_factory, vk_input, vk_target, steps=4
    )

    assert all(dispatches > 0 and transfers == 0 for dispatches, transfers in counters)
    transfer_count = 0
    for vk_loss, cpu_loss in zip(vk_losses, cpu_losses):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_loss, transfer_count), cpu_loss
        )
        transfer_count += 1
    for cpu_parameter, vk_parameter in zip(cpu_model.parameters(), vk_model.parameters()):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_parameter, transfer_count), cpu_parameter
        )
        transfer_count += 1
    for cpu_parameter, parameter in zip(cpu_model.parameters(), vk_model.parameters()):
        for name, value in vk_optimizer.state[parameter].items():
            expected = cpu_optimizer.state[cpu_parameter][name]
            if isinstance(value, torch.Tensor) and value.device.type == "vk":
                torch.testing.assert_close(
                    _readback_with_counter_assertion(value, transfer_count),
                    expected,
                )
                transfer_count += 1
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected)
            else:
                assert value == expected


@pytest.mark.parametrize(
    "inputs, targets, model_factory, message",
    [
        ("float64", "float32", _make_mlp, "float32"),
        ("float32", "float32", _make_mlp, "vk:0"),
        ("float32", "float32", torch.nn.Linear, "fixed MLP"),
    ],
)
def test_run_training_step_rejects_invalid_contract_without_side_effects(
    inputs, targets, model_factory, message, vulkan_backend
):
    model = model_factory(vulkan_backend, seed=61) if model_factory is _make_mlp else model_factory(8, 4).to(vulkan_backend)
    optimizer = _make_sgd(model.parameters())
    input_dtype = torch.float64 if inputs == "float64" else torch.float32
    vk_inputs = (
        torch.empty(3, 8, dtype=input_dtype, device=vulkan_backend)
        if input_dtype is torch.float64
        else torch.ones(3, 8, dtype=input_dtype).to(vulkan_backend)
    )
    vk_targets = torch.ones(3, 4, dtype=torch.float32).to(vulkan_backend)
    if message == "vk:0":
        vk_inputs = vk_inputs.cpu()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        run_training_step(model, optimizer, vk_inputs, vk_targets)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize(
    "input_shape, target_shape, message",
    [
        ((3, 7), (3, 4), "inputs"),
        ((3, 8), (3, 5), "targets"),
        ((3, 8, 1), (3, 4), "inputs"),
    ],
)
def test_run_training_step_rejects_invalid_shapes_without_side_effects(
    input_shape, target_shape, message, vulkan_backend
):
    model = _make_mlp(vulkan_backend, seed=67)
    optimizer = _make_sgd(model.parameters())
    inputs = torch.ones(input_shape).to(vulkan_backend)
    targets = torch.ones(target_shape).to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        run_training_step(model, optimizer, inputs, targets)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


def test_run_training_step_rejects_wrong_fixed_mlp_dimensions_without_side_effects(
    vulkan_backend,
):
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 15),
        torch.nn.ReLU(),
        torch.nn.Linear(15, 4),
    ).to(device=vulkan_backend, dtype=torch.float32)
    optimizer = _make_sgd(model.parameters())
    inputs = torch.ones(3, 8).to(vulkan_backend)
    targets = torch.ones(3, 4).to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=r"Linear\(8,16\)"):
        run_training_step(model, optimizer, inputs, targets)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
