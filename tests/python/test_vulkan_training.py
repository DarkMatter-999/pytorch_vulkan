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


def _make_mnist_classifier(device, seed):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(784, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ).to(device=device, dtype=torch.float32)


def _make_mnist_batch(seed, batch_size):
    torch.manual_seed(seed)
    inputs = torch.randn(batch_size, 1, 28, 28, dtype=torch.float32)
    labels = torch.randint(10, (batch_size,), dtype=torch.int64)
    targets = torch.nn.functional.one_hot(labels, num_classes=10).to(
        dtype=torch.float32
    )
    assert torch.all(targets.eq(1).sum(dim=1).eq(1))
    assert torch.all(targets.eq(0).sum(dim=1).eq(9))
    return inputs, targets


def _squared_error_loss(output, target):
    error = output - target
    return error.mul(error).sum()


def _assert_vk_f32_contiguous(tensor):
    assert tensor.device == torch.device("vk:0")
    assert tensor.dtype is torch.float32
    assert tensor.is_contiguous()


def _validate_mnist_training_contract(model, optimizer, inputs, targets, loss_fn):
    parameters = list(model.parameters())
    if not parameters:
        raise RuntimeError("MNIST training model must have parameters on vk:0")
    model_device = parameters[0].device
    if model_device != torch.device("vk:0"):
        raise RuntimeError("MNIST training model must be on vk:0")
    if inputs.dtype is not torch.float32 or targets.dtype is not torch.float32:
        raise RuntimeError("MNIST training inputs and targets must be float32")
    if inputs.device != model_device or targets.device != model_device:
        raise RuntimeError("MNIST training inputs and targets must be on vk:0")
    if not inputs.is_contiguous() or not targets.is_contiguous():
        raise RuntimeError("MNIST training inputs and targets must be contiguous")
    if inputs.ndim != 4 or tuple(inputs.shape[1:]) != (1, 28, 28):
        raise RuntimeError("MNIST training inputs must have shape (batch, 1, 28, 28)")
    if targets.ndim != 2 or tuple(targets.shape) != (inputs.shape[0], 10):
        raise RuntimeError("MNIST training targets must have shape (batch, 10)")
    modules = list(model.children())
    if not isinstance(model, torch.nn.Sequential) or [
        type(module) for module in modules
    ] != [
        torch.nn.Flatten,
        torch.nn.Linear,
        torch.nn.ReLU,
        torch.nn.Linear,
    ]:
        raise RuntimeError("MNIST training model must be Flatten->Linear->ReLU->Linear")
    if (
        modules[0].start_dim != 1
        or modules[1].in_features != 784
        or modules[1].out_features != 32
        or modules[3].in_features != 32
        or modules[3].out_features != 10
    ):
        raise RuntimeError(
            "MNIST training model must be Flatten->Linear(784,32)->ReLU->Linear(32,10)"
        )
    if loss_fn is not _squared_error_loss:
        raise RuntimeError("MNIST training supports only float32 squared-error loss")
    for parameter in parameters:
        _assert_vk_f32_contiguous(parameter)
    if not isinstance(optimizer, (torch.optim.SGD, torch.optim.Adam)):
        raise RuntimeError("MNIST training supports only SGD and Adam")
    optimizer_parameters = [
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ]
    if optimizer_parameters != parameters:
        raise RuntimeError("MNIST optimizer parameters must match the training model")
    unsupported = (
        ("nesterov", torch.optim.SGD),
        ("maximize", (torch.optim.SGD, torch.optim.Adam)),
        ("foreach", (torch.optim.SGD, torch.optim.Adam)),
        ("differentiable", (torch.optim.SGD, torch.optim.Adam)),
        ("fused", (torch.optim.SGD, torch.optim.Adam)),
        ("amsgrad", torch.optim.Adam),
        ("capturable", torch.optim.Adam),
    )
    for option, optimizer_type in unsupported:
        if isinstance(optimizer, optimizer_type) and optimizer.defaults.get(
            option, False
        ):
            raise RuntimeError(
                f"MNIST training does not support optimizer option {option}"
            )


def run_mnist_training_step(
    model, optimizer, inputs, targets, loss_fn=_squared_error_loss
):
    _validate_mnist_training_contract(model, optimizer, inputs, targets, loss_fn)
    pytorch_vulkan._C.begin_training_step()
    try:
        optimizer.zero_grad(set_to_none=False)
        output = model(inputs)
        loss = loss_fn(output, targets)
        grad_output = torch.empty_like(loss)
        grad_output.fill_(1.0)
        loss.backward(grad_output)
        _assert_vk_f32_contiguous(output)
        _assert_vk_f32_contiguous(loss)
        _assert_vk_f32_contiguous(inputs.grad)
        for parameter in model.parameters():
            _assert_vk_f32_contiguous(parameter.grad)
        optimizer.step()
        pytorch_vulkan._C.end_training_step()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    return loss


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

    for first_parameter, second_parameter in zip(
        first.parameters(), second.parameters()
    ):
        torch.testing.assert_close(first_parameter, second_parameter)


def test_mnist_batch_seed_is_deterministic():
    first_inputs, first_targets = _make_mnist_batch(seed=23, batch_size=3)
    second_inputs, second_targets = _make_mnist_batch(seed=23, batch_size=3)

    torch.testing.assert_close(first_inputs, second_inputs)
    torch.testing.assert_close(first_targets, second_targets)


def test_mnist_forward_matches_cpu_at_explicit_readback(vulkan_backend):
    cpu_model = _make_mnist_classifier("cpu", seed=71)
    vk_model = _make_mnist_classifier(vulkan_backend, seed=71)
    vk_model.load_state_dict(
        {
            name: parameter.detach().clone().to(vulkan_backend)
            for name, parameter in cpu_model.state_dict().items()
        }
    )
    cpu_input, cpu_target = _make_mnist_batch(seed=73, batch_size=3)
    vk_input = cpu_input.to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = vk_model(vk_input)

    assert vk_output.shape == (3, 10)
    _assert_vk_f32_contiguous(vk_output)
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0

    cpu_output = cpu_model(cpu_input)
    torch.testing.assert_close(
        _readback_with_counter_assertion(vk_output, expected_transfer_count=0),
        cpu_output,
    )
    assert cpu_target.shape == (3, 10)
    assert cpu_target.dtype is torch.float32
    assert cpu_target.is_contiguous()


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
    for cpu_parameter, vk_parameter in zip(
        cpu_model.parameters(), vk_model.parameters()
    ):
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
    value = torch.ones(2, 8).to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.neg(value)

    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0
    assert pytorch_vulkan._C.compute_submission_count() == 1
    assert pytorch_vulkan._C.compute_submission_count() == 1

    result.cpu()
    assert pytorch_vulkan._C.explicit_transfer_count() == 1


def test_training_scope_records_two_dispatches_and_completes_once(vulkan_backend):
    first = torch.ones(2, 8).to(vulkan_backend)
    second = torch.full((2, 8), 2.0).to(vulkan_backend)
    result = torch.empty_like(first)

    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        torch.add(first, second, out=result)
        torch.neg(result, out=result)
        assert pytorch_vulkan._C.compute_dispatch_count() == 2
        assert pytorch_vulkan._C.compute_submission_count() == 0
    finally:
        pytorch_vulkan._C.end_training_step()

    assert pytorch_vulkan._C.compute_submission_count() == 1
    torch.testing.assert_close(result.cpu(), torch.full((2, 8), -3.0))


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
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss.backward()
        pytorch_vulkan._C.end_training_step()
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise

    for parameter in vk_model.parameters():
        _assert_vk_f32_contiguous(parameter.grad)
    _assert_vk_f32_contiguous(vk_input.grad)
    assert pytorch_vulkan._C.compute_submission_count() > 0
    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0


def test_fixed_mlp_bounded_lifecycle_stress(vulkan_backend):
    for iteration in range(4):
        cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target = (
            _make_training_pairs(vulkan_backend, seed=900 + iteration)
        )
        cpu_loss = _squared_error_loss(cpu_model(cpu_input), cpu_target)
        cpu_loss.backward()
        cpu_optimizer = _make_sgd(cpu_model.parameters())
        cpu_optimizer.step()
        optimizer = _make_sgd(vk_model.parameters())
        pytorch_vulkan._C.reset_execution_counters()
        run_training_step(vk_model, optimizer, vk_input, vk_target)
        assert pytorch_vulkan._C.compute_submission_count() == 1
        assert pytorch_vulkan._C.pending_compute_count() == 0
        for (cpu_name, cpu_parameter), (vk_name, vk_parameter) in zip(
            cpu_model.named_parameters(), vk_model.named_parameters()
        ):
            assert vk_name == cpu_name
            torch.testing.assert_close(
                vk_parameter.cpu(), cpu_parameter, rtol=2e-4, atol=2e-4
            )
            torch.testing.assert_close(
                vk_parameter.grad.cpu(), cpu_parameter.grad, rtol=2e-4, atol=2e-4
            )
        for cpu_parameter, vk_parameter in zip(
            cpu_model.parameters(), vk_model.parameters()
        ):
            for name, vk_value in optimizer.state[vk_parameter].items():
                cpu_value = cpu_optimizer.state[cpu_parameter][name]
                if isinstance(vk_value, torch.Tensor):
                    torch.testing.assert_close(
                        vk_value.cpu(), cpu_value, rtol=2e-4, atol=2e-4
                    )
                else:
                    assert vk_value == cpu_value


def test_fused_linear_relu_training_scope_retires_resources(vulkan_backend):
    torch.manual_seed(107)
    inputs = torch.randn(3, 8).to(vulkan_backend).requires_grad_()
    weight = torch.randn(16, 8).to(vulkan_backend).requires_grad_()
    bias = torch.randn(16).to(vulkan_backend).requires_grad_()
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.begin_training_step()
    try:
        output = torch.ops.pytorch_vulkan.linear_relu(inputs, weight, bias)
        loss = output.mul(output).sum()
        loss.backward()
        assert pytorch_vulkan._C.compute_submitted_count() == 0
        assert pytorch_vulkan._C.compute_completed_count() == 0
    finally:
        pytorch_vulkan._C.end_training_step()
    assert pytorch_vulkan._C.compute_submission_count() == 1
    assert pytorch_vulkan._C.pending_compute_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


def test_fused_linear_relu_optimizer_state_matches_cpu_for_mnist_shape(vulkan_backend):
    torch.manual_seed(127)
    cpu_input = torch.randn(1, 784, requires_grad=True)
    cpu_weight = torch.randn(32, 784, requires_grad=True)
    cpu_bias = torch.randn(32, requires_grad=True)
    vk_input = cpu_input.detach().to(vulkan_backend).requires_grad_()
    vk_weight = cpu_weight.detach().to(vulkan_backend).requires_grad_()
    vk_bias = cpu_bias.detach().to(vulkan_backend).requires_grad_()
    cpu_optimizer = torch.optim.SGD([cpu_weight, cpu_bias], lr=0.01, momentum=0.9)
    vk_optimizer = torch.optim.SGD([vk_weight, vk_bias], lr=0.01, momentum=0.9)

    cpu_output = torch.relu(torch.nn.functional.linear(cpu_input, cpu_weight, cpu_bias))
    cpu_loss = cpu_output.mul(cpu_output).sum()
    cpu_loss.backward()
    cpu_optimizer.step()

    pytorch_vulkan._C.reset_execution_counters()
    vk_output = torch.ops.pytorch_vulkan.linear_relu(vk_input, vk_weight, vk_bias)
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_output.mul(vk_output).sum().backward()
        pytorch_vulkan._C.end_training_step()
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise
    vk_optimizer.step()

    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    for vk_gradient, cpu_gradient in zip(
        (vk_input.grad, vk_weight.grad, vk_bias.grad),
        (cpu_input.grad, cpu_weight.grad, cpu_bias.grad),
    ):
        torch.testing.assert_close(
            vk_gradient.cpu(), cpu_gradient, rtol=2e-4, atol=2e-4
        )
    for vk_parameter, cpu_parameter in zip(
        (vk_weight, vk_bias), (cpu_weight, cpu_bias)
    ):
        torch.testing.assert_close(
            vk_parameter.cpu(), cpu_parameter, rtol=2e-4, atol=2e-4
        )
    for vk_parameter, cpu_parameter in zip(
        (vk_weight, vk_bias), (cpu_weight, cpu_bias)
    ):
        torch.testing.assert_close(
            vk_optimizer.state[vk_parameter]["momentum_buffer"].cpu(),
            cpu_optimizer.state[cpu_parameter]["momentum_buffer"],
            rtol=2e-4,
            atol=2e-4,
        )


def test_user_facing_vulkan_inplace_add_remains_rejected(vulkan_backend):
    value = torch.randn(2, 8).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="in-place operations are unsupported"):
        value.add_(1.0)


def test_batch_norm_classification_training_matches_cpu(vulkan_backend):
    class FixedClassifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(4))
            self.bias = torch.nn.Parameter(torch.zeros(4))
            self.register_buffer("running_mean", torch.zeros(4))
            self.register_buffer("running_var", torch.ones(4))
            self.classifier = torch.nn.Linear(4, 3)

        def forward(self, value):
            value = torch.ops.aten.native_batch_norm.default(
                value,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                True,
                0.1,
                1e-5,
            )[0]
            return self.classifier(value)

    torch.manual_seed(419)
    cpu_model = FixedClassifier()
    vk_model = FixedClassifier().to(vulkan_backend)
    vk_model.load_state_dict(
        {
            name: value.detach().clone().to(vulkan_backend)
            for name, value in cpu_model.state_dict().items()
        }
    )
    cpu_input = torch.randn(2, 4, requires_grad=True)
    cpu_labels = torch.tensor([1, 2], dtype=torch.int64)
    vk_input = cpu_input.detach().clone().to(vulkan_backend).requires_grad_()
    vk_labels = cpu_labels.to(vulkan_backend)
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
    vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.01)
    cpu_loss = torch.nn.functional.cross_entropy(cpu_model(cpu_input), cpu_labels)
    cpu_loss.backward()
    cpu_optimizer.step()
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss = torch.nn.functional.cross_entropy(vk_model(vk_input), vk_labels)
        vk_loss.backward()
        vk_optimizer.step()
        pytorch_vulkan._C.end_training_step()
    except BaseException:
        pytorch_vulkan._C.cancel_training_step()
        raise
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss.detach(), rtol=2e-4, atol=2e-4)
    for actual, expected in zip(vk_model.parameters(), cpu_model.parameters()):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)


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
    if not isinstance(model, torch.nn.Sequential) or [
        type(module) for module in modules
    ] != [
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

    scoped = model_device.type == "vk"
    if scoped:
        pytorch_vulkan._C.begin_training_step()
    try:
        optimizer.zero_grad(set_to_none=False)
        output = model(inputs)
        loss = _squared_error_loss(output, targets)
        grad_output = torch.empty_like(loss)
        grad_output.fill_(1.0)
        loss.backward(grad_output)
        if scoped:
            _assert_vk_f32_contiguous(output)
            _assert_vk_f32_contiguous(loss)
            for parameter in parameters:
                _assert_vk_f32_contiguous(parameter.grad)
            if inputs.requires_grad:
                _assert_vk_f32_contiguous(inputs.grad)
        optimizer.step()
        if scoped:
            pytorch_vulkan._C.end_training_step()
    except BaseException:
        if scoped:
            pytorch_vulkan._C.cancel_training_step()
        raise
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
    pytorch_vulkan._C.begin_training_step()
    try:
        vk_loss.backward()
        pytorch_vulkan._C.end_training_step()
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise
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
    for cpu_parameter, vk_parameter in zip(
        cpu_model.parameters(), vk_model.parameters()
    ):
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
            counters.append(
                (*_execution_counters(), pytorch_vulkan._C.compute_submission_count())
            )
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

    assert all(
        dispatches > 0 and transfers == 0 and submissions == 1
        for dispatches, transfers, submissions in counters
    )
    transfer_count = 0
    for vk_loss, cpu_loss in zip(vk_losses, cpu_losses):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_loss, transfer_count), cpu_loss
        )
        transfer_count += 1
    for cpu_parameter, vk_parameter in zip(
        cpu_model.parameters(), vk_model.parameters()
    ):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_parameter, transfer_count),
            cpu_parameter,
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


def test_sgd_first_momentum_clone_matches_cpu(vulkan_backend):
    cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target = (
        _make_training_pairs(vulkan_backend, seed=53)
    )
    cpu_optimizer = _make_sgd(cpu_model.parameters())
    vk_optimizer = _make_sgd(vk_model.parameters())

    run_training_step(cpu_model, cpu_optimizer, cpu_input, cpu_target)
    run_training_step(vk_model, vk_optimizer, vk_input, vk_target)

    cpu_parameter = cpu_model[0].weight
    vk_parameter = vk_model[0].weight
    torch.testing.assert_close(
        vk_optimizer.state[vk_parameter]["momentum_buffer"].cpu(),
        cpu_optimizer.state[cpu_parameter]["momentum_buffer"],
    )
    torch.testing.assert_close(vk_parameter.cpu(), cpu_parameter)


def test_training_exception_cancels_recorded_work(vulkan_backend):
    model = _make_mlp(vulkan_backend, seed=71)
    inputs = torch.randn(3, 8).to(vulkan_backend)
    targets = torch.randn(3, 4).to(vulkan_backend)
    optimizer = _make_sgd(model.parameters())
    initial_parameters = [parameter.cpu().clone() for parameter in model.parameters()]
    original_step = optimizer.step

    def fail_step(*args, **kwargs):
        raise RuntimeError("injected optimizer failure")

    optimizer.step = fail_step
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        run_training_step(model, optimizer, inputs, targets)
    assert pytorch_vulkan._C.compute_submission_count() == 0
    assert not optimizer.state
    for parameter, initial in zip(model.parameters(), initial_parameters):
        torch.testing.assert_close(parameter.cpu(), initial)

    optimizer.step = original_step
    pytorch_vulkan._C.reset_execution_counters()
    run_training_step(model, optimizer, inputs, targets)
    assert pytorch_vulkan._C.compute_submission_count() == 1


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
    model = (
        model_factory(vulkan_backend, seed=61)
        if model_factory is _make_mlp
        else model_factory(8, 4).to(vulkan_backend)
    )
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


def _mnist_training_pairs(device, seed=79):
    cpu_model = _make_mnist_classifier("cpu", seed)
    cpu_inputs, cpu_targets = _make_mnist_batch(seed + 1, batch_size=3)
    vk_model = _make_mnist_classifier(device, seed)
    vk_model.load_state_dict(
        {
            name: value.detach().clone().to(device)
            for name, value in cpu_model.state_dict().items()
        }
    )
    return (
        cpu_model,
        vk_model,
        cpu_inputs,
        cpu_inputs.detach().clone().to(device).requires_grad_(),
        cpu_targets,
        cpu_targets.to(device),
    )


def test_mnist_shaped_training_runs_forward_and_backward_on_vulkan(vulkan_backend):
    _, vk_model, _, vk_inputs, _, vk_targets = _mnist_training_pairs(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    loss = run_mnist_training_step(
        vk_model, _make_sgd(vk_model.parameters()), vk_inputs, vk_targets
    )

    _assert_vk_f32_contiguous(loss)
    _assert_vk_f32_contiguous(vk_inputs.grad)
    for parameter in vk_model.parameters():
        _assert_vk_f32_contiguous(parameter.grad)
    assert pytorch_vulkan._C.compute_submission_count() > 0
    dispatches, transfers = _execution_counters()
    assert dispatches > 0
    assert transfers == 0
    assert pytorch_vulkan._C.compute_submission_count() == 1


def test_mnist_training_accepts_arbitrary_float32_targets_before_readback(
    vulkan_backend,
):
    _, vk_model, _, vk_inputs, _, _ = _mnist_training_pairs(vulkan_backend)
    cpu_targets = torch.linspace(-1.0, 1.0, steps=30, dtype=torch.float32).reshape(
        3, 10
    )
    vk_targets = cpu_targets.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    loss = run_mnist_training_step(
        vk_model, _make_sgd(vk_model.parameters()), vk_inputs, vk_targets
    )

    assert _execution_counters()[0] > 0
    assert _execution_counters()[1] == 0
    assert loss.cpu().shape == ()
    assert pytorch_vulkan._C.explicit_transfer_count() == 1


def _run_mnist_training(device, optimizer_factory, inputs, targets, steps=4):
    model = _make_mnist_classifier(device, seed=79)
    optimizer = optimizer_factory(model.parameters())
    losses = []
    counters = []
    for _ in range(steps):
        pytorch_vulkan._C.reset_execution_counters()
        losses.append(
            run_mnist_training_step(model, optimizer, inputs, targets).detach()
        )
        counters.append(_execution_counters())
        counters[-1] = (*counters[-1], pytorch_vulkan._C.compute_submission_count())
        _assert_optimizer_state_vulkan(optimizer)
    return model, optimizer, losses, counters


@pytest.mark.parametrize("optimizer_factory", [_make_sgd, _make_adam])
def test_mnist_shaped_sgd_and_adam_match_cpu(optimizer_factory, vulkan_backend):
    cpu_model, _, cpu_inputs, vk_inputs, cpu_targets, vk_targets = (
        _mnist_training_pairs(vulkan_backend)
    )
    cpu_optimizer = optimizer_factory(cpu_model.parameters())
    cpu_losses = []
    for _ in range(2):
        cpu_optimizer.zero_grad(set_to_none=False)
        cpu_loss = _squared_error_loss(cpu_model(cpu_inputs), cpu_targets)
        cpu_loss.backward()
        cpu_optimizer.step()
        cpu_losses.append(cpu_loss.detach())
    vk_model, vk_optimizer, vk_losses, counters = _run_mnist_training(
        vulkan_backend, optimizer_factory, vk_inputs, vk_targets, steps=2
    )

    assert all(
        dispatches > 0 and transfers == 0 and submissions == 1
        for dispatches, transfers, submissions in counters
    )
    transfer_count = 0
    for vk_loss, cpu_loss in zip(vk_losses, cpu_losses):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_loss, transfer_count),
            cpu_loss.detach(),
            rtol=1e-4,
            atol=1e-4,
        )
        transfer_count += 1
    for cpu_parameter, vk_parameter in zip(
        cpu_model.parameters(), vk_model.parameters()
    ):
        torch.testing.assert_close(
            _readback_with_counter_assertion(vk_parameter, transfer_count),
            cpu_parameter,
            rtol=1e-4,
            atol=1e-4,
        )
        transfer_count += 1
    for cpu_parameter, vk_parameter in zip(
        cpu_model.parameters(), vk_model.parameters()
    ):
        for name, value in vk_optimizer.state[vk_parameter].items():
            expected = cpu_optimizer.state[cpu_parameter][name]
            if isinstance(value, torch.Tensor) and value.device.type == "vk":
                torch.testing.assert_close(
                    _readback_with_counter_assertion(value, transfer_count),
                    expected,
                    rtol=1e-4,
                    atol=1e-4,
                )
                transfer_count += 1
            elif isinstance(value, torch.Tensor):
                torch.testing.assert_close(value, expected)
            else:
                assert value == expected


@pytest.mark.parametrize(
    "case",
    [
        "cross_entropy",
        "long_labels",
        "convolution",
        "batch_norm",
        "float64",
        "cpu_input",
        "unsupported_optimizer",
        "noncontiguous_targets",
    ],
)
def test_mnist_unsupported_boundaries_reject_without_side_effects(case, vulkan_backend):
    _, vk_model, _, vk_inputs, _, vk_targets = _mnist_training_pairs(vulkan_backend)
    rejected_model = vk_model
    if case == "cross_entropy":
        action = lambda: run_mnist_training_step(
            vk_model, optimizer, vk_inputs, vk_targets, torch.nn.CrossEntropyLoss()
        )
        message = "squared-error"
    elif case == "long_labels":
        long_labels = torch.zeros(3, dtype=torch.int64)
        action = lambda: run_mnist_training_step(
            vk_model, optimizer, vk_inputs, long_labels
        )
        message = "float32"
    elif case == "convolution":
        convolution_model = torch.nn.Sequential(torch.nn.Conv2d(1, 4, 3)).to(
            vulkan_backend
        )
        rejected_model = convolution_model
        action = lambda: run_mnist_training_step(
            convolution_model, optimizer, vk_inputs, vk_targets
        )
        message = "Flatten"
    elif case == "batch_norm":
        batch_norm_model = torch.nn.Sequential(
            torch.nn.BatchNorm2d(1, track_running_stats=False)
        ).to(vulkan_backend)
        rejected_model = batch_norm_model
        action = lambda: run_mnist_training_step(
            batch_norm_model, optimizer, vk_inputs, vk_targets
        )
        message = "Flatten"
    elif case == "float64":
        float64_inputs = vk_inputs.to(dtype=torch.float64)
        action = lambda: run_mnist_training_step(
            vk_model, optimizer, float64_inputs, vk_targets
        )
        message = "float32"
    elif case == "cpu_input":
        cpu_inputs = vk_inputs.cpu()
        action = lambda: run_mnist_training_step(
            vk_model, optimizer, cpu_inputs, vk_targets
        )
        message = "vk:0"
    else:
        action = None
        message = "nesterov"

    if case == "noncontiguous_targets":
        noncontiguous_targets = torch.ones(3, 20).to(vulkan_backend)[:, ::2]
        action = lambda: run_mnist_training_step(
            vk_model, optimizer, vk_inputs, noncontiguous_targets
        )
        message = "contiguous"

    rejected_model.register_buffer(
        "rejection_sentinel",
        torch.tensor([3.0], dtype=torch.float32).to(vulkan_backend),
    )
    optimizer = _make_sgd(rejected_model.parameters())
    for parameter in rejected_model.parameters():
        optimizer.state[parameter]["momentum_buffer"] = torch.ones_like(parameter)
    if case == "unsupported_optimizer":
        optimizer.defaults["nesterov"] = True
        action = lambda: run_mnist_training_step(
            rejected_model, optimizer, vk_inputs, vk_targets
        )

    def snapshot_tensor(tensor):
        return {
            "value": tensor.detach().cpu().clone(),
            "version": tensor._version,
            "device": tensor.device,
            "dtype": tensor.dtype,
            "shape": tuple(tensor.shape),
            "contiguous": tensor.is_contiguous(),
        }

    def snapshot_state():
        return {
            parameter: {
                name: snapshot_tensor(value)
                if isinstance(value, torch.Tensor)
                else value
                for name, value in state.items()
            }
            for parameter, state in optimizer.state.items()
        }

    before_parameters = {
        parameter: snapshot_tensor(parameter)
        for parameter in rejected_model.parameters()
    }
    before_buffers = {
        name: snapshot_tensor(value) for name, value in rejected_model.named_buffers()
    }
    before_inputs = snapshot_tensor(vk_inputs if case != "cpu_input" else cpu_inputs)
    before_targets = snapshot_tensor(
        noncontiguous_targets
        if case == "noncontiguous_targets"
        else vk_targets
        if case != "long_labels"
        else long_labels
    )
    before_state = snapshot_state()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, TypeError), match=message):
        action()
    assert _execution_counters() == (0, 0)
    for parameter, expected in before_parameters.items():
        actual = snapshot_tensor(parameter)
        assert actual["version"] == expected["version"]
        assert actual["device"] == expected["device"]
        assert actual["dtype"] == expected["dtype"]
        assert actual["shape"] == expected["shape"]
        assert actual["contiguous"] == expected["contiguous"]
        torch.testing.assert_close(actual["value"], expected["value"])
    for name, expected in before_buffers.items():
        actual = snapshot_tensor(dict(rejected_model.named_buffers())[name])
        assert actual["version"] == expected["version"]
        assert actual["device"] == expected["device"]
        torch.testing.assert_close(actual["value"], expected["value"])
    actual_inputs = snapshot_tensor(vk_inputs if case != "cpu_input" else cpu_inputs)
    actual_targets = snapshot_tensor(
        noncontiguous_targets
        if case == "noncontiguous_targets"
        else vk_targets
        if case != "long_labels"
        else long_labels
    )
    assert actual_inputs["version"] == before_inputs["version"]
    assert actual_targets["version"] == before_targets["version"]
    torch.testing.assert_close(actual_inputs["value"], before_inputs["value"])
    torch.testing.assert_close(actual_targets["value"], before_targets["value"])
    assert optimizer.state.keys() == before_state.keys()
    for parameter, expected_state in before_state.items():
        actual_state = optimizer.state[parameter]
        assert actual_state.keys() == expected_state.keys()
        for name, expected in expected_state.items():
            actual = actual_state[name]
            if isinstance(expected, dict):
                snapshot = snapshot_tensor(actual)
                assert snapshot["version"] == expected["version"]
                assert snapshot["device"] == expected["device"]
                assert snapshot["dtype"] == expected["dtype"]
                torch.testing.assert_close(snapshot["value"], expected["value"])
            else:
                assert actual == expected
