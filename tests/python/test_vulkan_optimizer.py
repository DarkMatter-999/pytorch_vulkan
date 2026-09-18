"""PyTorch 2.4 scalar optimizer dispatch and parity contract."""

import pytest
import torch
import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def make_sgd(parameters, **overrides):
    options = {
        "lr": 0.1,
        "momentum": 0.9,
        "dampening": 0.0,
        "weight_decay": 0.01,
        "nesterov": False,
        "maximize": False,
        "foreach": False,
        "differentiable": False,
    }
    options.update(overrides)
    options.pop("require_vulkan", None)
    return torch.optim.SGD(
        parameters,
        **options,
    )


def make_adam(parameters, **overrides):
    options = {
        "lr": 0.01,
        "betas": (0.9, 0.999),
        "eps": 1e-8,
        "weight_decay": 0.01,
        "amsgrad": False,
        "foreach": False,
        "maximize": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
    }
    options.update(overrides)
    options.pop("require_vulkan", None)
    return torch.optim.Adam(
        parameters,
        **options,
    )


def _state_snapshot(optimizer, parameter):
    state = optimizer.state[parameter]
    return {
        name: (
            value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        )
        for name, value in state.items()
    }


def _complete_state_snapshot(optimizer, parameter):
    snapshot = {}
    for name, value in optimizer.state[parameter].items():
        if isinstance(value, torch.Tensor):
            snapshot[name] = (
                "tensor",
                value.detach().cpu().clone(),
                value._version,
                value.device.type,
            )
        else:
            snapshot[name] = ("scalar", value)
    return snapshot


def _assert_state_snapshot(optimizer, parameter, snapshot):
    assert optimizer.state[parameter].keys() == snapshot.keys()
    for name, expected in snapshot.items():
        value = optimizer.state[parameter][name]
        if expected[0] == "tensor":
            _, expected_value, expected_version, _ = expected
            assert value._version == expected_version
            torch.testing.assert_close(value.detach().cpu(), expected_value)
        else:
            assert value == expected[1]


def run_training_steps(device, optimizer_factory, steps):
    parameter = torch.tensor([1.0, -2.0], device=device, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    losses = []
    counters = []
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=False)
        loss = (parameter * parameter).sum()
        if device == "cpu":
            loss.backward()
        else:
            pytorch_vulkan._C.begin_training_step()
            try:
                loss.backward()
                pytorch_vulkan._C.end_training_step()
            except Exception:
                pytorch_vulkan._C.cancel_training_step()
                raise
        pytorch_vulkan._C.reset_execution_counters() if device != "cpu" else None
        optimizer.step()
        if device != "cpu":
            counters.append(
                (
                    pytorch_vulkan._C.compute_dispatch_count(),
                    pytorch_vulkan._C.explicit_transfer_count(),
                )
            )
        losses.append(loss.detach().cpu().clone())
    return (
        parameter.detach().cpu().clone(),
        _state_snapshot(optimizer, parameter),
        losses,
        counters,
    )


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_optimizer_steps_match_cpu(optimizer_factory, vulkan_backend):
    cpu_parameter, cpu_state, cpu_losses, _ = run_training_steps(
        "cpu", optimizer_factory, 1
    )
    vk_parameter, vk_state, vk_losses, counters = run_training_steps(
        vulkan_backend, optimizer_factory, 1
    )
    torch.testing.assert_close(vk_parameter, cpu_parameter)
    torch.testing.assert_close(vk_losses[0], cpu_losses[0])
    assert counters == [(value, 0) for value, _ in counters]
    assert counters[0][0] > 0
    for name in ("momentum_buffer", "exp_avg", "exp_avg_sq"):
        if name in cpu_state:
            torch.testing.assert_close(vk_state[name], cpu_state[name])


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_optimizer_multi_step_state_matches_cpu(optimizer_factory, vulkan_backend):
    cpu_parameter, cpu_state, cpu_losses, _ = run_training_steps(
        "cpu", optimizer_factory, 4
    )
    vk_parameter, vk_state, vk_losses, counters = run_training_steps(
        vulkan_backend, optimizer_factory, 4
    )
    torch.testing.assert_close(vk_parameter, cpu_parameter)
    torch.testing.assert_close(torch.stack(vk_losses), torch.stack(cpu_losses))
    assert all(dispatches > 0 and transfers == 0 for dispatches, transfers in counters)
    expected_names = (
        ("momentum_buffer",)
        if optimizer_factory is make_sgd
        else ("step", "exp_avg", "exp_avg_sq")
    )
    for name in expected_names:
        assert name in vk_state
        if name != "step":
            assert isinstance(vk_state[name], torch.Tensor)
            assert vk_state[name].device == torch.device("cpu")
            torch.testing.assert_close(vk_state[name], cpu_state[name])
    if optimizer_factory is make_adam:
        assert (
            not isinstance(vk_state["step"], torch.Tensor)
            or vk_state["step"].device.type == "cpu"
        )
        torch.testing.assert_close(vk_state["step"], cpu_state["step"])


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_optimizer_repeated_updates_survive_allocator_size_transitions(
    optimizer_factory, vulkan_backend
):
    def run_allocation_phase(device, shape, steps):
        parameter = torch.tensor(
            [1.0, -2.0] if shape == (2,) else [0.25] * shape[0],
            device=device,
            requires_grad=True,
        )
        optimizer = optimizer_factory([parameter])
        counters = []
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=False)
            loss = (parameter * parameter).sum()
            if device == "cpu":
                loss.backward()
            else:
                pytorch_vulkan._C.begin_training_step()
                try:
                    loss.backward()
                    pytorch_vulkan._C.end_training_step()
                except Exception:
                    pytorch_vulkan._C.cancel_training_step()
                    raise
            if device != "cpu":
                pytorch_vulkan._C.reset_execution_counters()
            optimizer.step()
            if device != "cpu":
                counters.append(
                    (
                        pytorch_vulkan._C.compute_dispatch_count(),
                        pytorch_vulkan._C.explicit_transfer_count(),
                    )
                )
        if device != "cpu":
            assert all(
                dispatches > 0 and transfers == 0 for dispatches, transfers in counters
            )
        return parameter.detach().cpu().clone(), _state_snapshot(optimizer, parameter)

    for shape, steps in [((2,), 32), ((257,), 2), ((2,), 32)]:
        cpu_parameter, cpu_state = run_allocation_phase("cpu", shape, steps)
        vk_parameter, vk_state = run_allocation_phase(vulkan_backend, shape, steps)
        torch.testing.assert_close(vk_parameter, cpu_parameter)
        for name, expected in cpu_state.items():
            torch.testing.assert_close(vk_state[name], expected)


def test_basic_sgd_and_adam_reach_vulkan_update_path(vulkan_backend):
    for optimizer_factory in (make_sgd, make_adam):
        run_training_steps(vulkan_backend, optimizer_factory, 1)


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_basic_sgd_and_adam_cpu_control(optimizer_factory):
    run_training_steps("cpu", optimizer_factory, 1)


def test_cpu_optimizer_options_preserve_pytorch_behavior():
    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    sgd = torch.optim.SGD([parameter], 0.1, 0.9, 0.0, 0.0, True)
    parameter.square().sum().backward()
    sgd.step()

    parameter = torch.tensor([1.0, -2.0], requires_grad=True)
    adam = torch.optim.Adam([parameter], amsgrad=True)
    parameter.square().sum().backward()
    adam.step()


@pytest.mark.parametrize(
    "constructor, args, options, message",
    [
        (torch.optim.SGD, (0.1, 0.9, 0.0, 0.0, True), {}, "nesterov"),
        (torch.optim.Adam, (), {"fused": True}, "fused"),
    ],
)
def test_unsupported_vulkan_options_reject_in_positional_or_keyword_form(
    constructor, args, options, message, vulkan_backend
):
    parameter = torch.ones(2, device=vulkan_backend, requires_grad=True)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        constructor([parameter], *args, **options)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_vulkan_optimizer_accepts_parameter_groups(optimizer_factory, vulkan_backend):
    parameter = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([{"params": [parameter]}])
    parameter.grad = torch.tensor([2.0, -4.0], device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    optimizer.step()
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert parameter.device == torch.device("vk:0")


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_repeated_backward_accumulates_vulkan_gradient(
    optimizer_factory, vulkan_backend
):
    cpu = torch.tensor([1.0, -2.0], requires_grad=True)
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    cpu_optimizer = optimizer_factory([cpu])
    vk_optimizer = optimizer_factory([vk])
    (cpu * cpu).sum().backward()
    (cpu * cpu).sum().backward()
    pytorch_vulkan._C.begin_training_step()
    try:
        (vk * vk).sum().backward()
        (vk * vk).sum().backward()
        pytorch_vulkan._C.end_training_step()
    except Exception:
        pytorch_vulkan._C.cancel_training_step()
        raise
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad)
    assert vk.grad.device == torch.device("vk:0")
    cpu_optimizer.step()
    pytorch_vulkan._C.reset_execution_counters()
    vk_optimizer.step()
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(vk.detach().cpu(), cpu.detach())
    cpu_optimizer.zero_grad(set_to_none=False)
    vk_optimizer.zero_grad(set_to_none=False)
    assert vk.grad.device == torch.device("vk:0")
    torch.testing.assert_close(vk.grad.cpu(), torch.zeros_like(cpu))


@pytest.mark.parametrize(
    "optimizer_factory, state_names",
    [
        (make_sgd, ("momentum_buffer",)),
        (make_adam, ("exp_avg", "exp_avg_sq")),
    ],
)
def test_optimizer_state_tensors_remain_vulkan_resident(
    optimizer_factory, state_names, vulkan_backend
):
    parameter = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    parameter.grad = torch.tensor([2.0, -4.0], device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    optimizer.step()
    state = optimizer.state[parameter]
    for name in state_names:
        assert state[name].device == torch.device("vk:0")
    if optimizer_factory is make_adam:
        assert (
            not isinstance(state["step"], torch.Tensor)
            or state["step"].device.type == "cpu"
        )
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize(
    "optimizer_factory, option",
    [
        (make_sgd, "nesterov"),
        (make_sgd, "maximize"),
        (make_sgd, "foreach"),
        (make_sgd, "differentiable"),
        (make_sgd, "fused"),
        (make_adam, "amsgrad"),
        (make_adam, "maximize"),
        (make_adam, "foreach"),
        (make_adam, "fused"),
        (make_adam, "differentiable"),
        (make_adam, "capturable"),
    ],
)
def test_unsupported_optimizer_options_reject_before_dispatch(
    optimizer_factory, option, vulkan_backend
):
    parameter = torch.ones(2, device=vulkan_backend, requires_grad=True)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=option):
        optimizer_factory([parameter], **{option: True})
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize(
    "optimizer_factory, option",
    [
        (make_sgd, "nesterov"),
        (make_sgd, "maximize"),
        (make_sgd, "foreach"),
        (make_sgd, "differentiable"),
        (make_sgd, "fused"),
        (make_adam, "amsgrad"),
        (make_adam, "maximize"),
        (make_adam, "foreach"),
        (make_adam, "fused"),
        (make_adam, "differentiable"),
        (make_adam, "capturable"),
    ],
)
def test_unsupported_vulkan_group_options_reject_before_dispatch(
    optimizer_factory, option, vulkan_backend
):
    parameter = torch.ones(2, device=vulkan_backend, requires_grad=True)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=option):
        optimizer_factory([{"params": [parameter], option: True}])
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize(
    "optimizer_factory, parameter_factory, message",
    [
        (
            make_sgd,
            lambda device: torch.empty(2, dtype=torch.float64, device=device),
            "float32",
        ),
        (
            make_adam,
            lambda device: torch.empty(2, dtype=torch.float64, device=device),
            "float32",
        ),
        (make_sgd, lambda device: torch.empty((2, 2), device=device).t(), "contiguous"),
        (
            make_adam,
            lambda device: torch.empty((2, 2), device=device).t(),
            "contiguous",
        ),
        (make_sgd, lambda device: torch.ones(2), "vk:0"),
        (make_adam, lambda device: torch.ones(2), "vk:0"),
    ],
)
def test_invalid_optimizer_parameters_reject_before_dispatch(
    optimizer_factory, parameter_factory, message, vulkan_backend
):
    parameter = parameter_factory(
        vulkan_backend if message != "vk:0" else "cpu"
    ).requires_grad_()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        if message == "vk:0":
            pytorch_vulkan.validate_basic_optimizer([parameter], require_vulkan=True)
        else:
            optimizer_factory([parameter])
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_nonzero_offset_optimizer_parameter_rejects_without_side_effects(
    optimizer_factory, vulkan_backend
):
    base = torch.ones(3, device=vulkan_backend, requires_grad=True)
    parameter = base[1:]
    before = parameter.detach().cpu().clone()
    version = parameter._version
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="zero-offset"):
        optimizer_factory([parameter])
    assert parameter._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(parameter.cpu(), before)


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_overlapping_optimizer_parameter_rejects_without_side_effects(
    optimizer_factory, vulkan_backend
):
    parameter = torch.as_strided(torch.ones(2, device=vulkan_backend), (2,), (0,))
    before = parameter.detach().cpu().clone()
    version = parameter._version
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="overlapping"):
        optimizer_factory([parameter])
    assert parameter._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(parameter.cpu(), before)


@pytest.mark.parametrize(
    "optimizer_factory, state_factory, message",
    [
        (
            make_sgd,
            lambda device: torch.empty(2, dtype=torch.float64, device=device),
            "float32",
        ),
        (
            make_adam,
            lambda device: torch.empty(2, dtype=torch.float64, device=device),
            "float32",
        ),
        (make_sgd, lambda device: torch.empty((2, 2), device=device).t(), "contiguous"),
        (
            make_adam,
            lambda device: torch.empty((2, 2), device=device).t(),
            "contiguous",
        ),
        (make_sgd, lambda device: torch.empty(3, device=device)[1:], "zero-offset"),
        (make_adam, lambda device: torch.empty(3, device=device)[1:], "zero-offset"),
        (
            make_sgd,
            lambda device: torch.as_strided(torch.empty(2, device=device), (2,), (0,)),
            "overlapping",
        ),
        (
            make_adam,
            lambda device: torch.as_strided(torch.empty(2, device=device), (2,), (0,)),
            "overlapping",
        ),
    ],
)
def test_invalid_vulkan_optimizer_state_rejects_before_dispatch(
    optimizer_factory, state_factory, message, vulkan_backend
):
    parameter_shape = (2, 2) if message == "contiguous" else (2,)
    parameter = torch.ones(parameter_shape, device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state_name = "momentum_buffer" if optimizer_factory is make_sgd else "exp_avg"
    state = optimizer.state[parameter][state_name]
    before = parameter.detach().cpu().clone()
    version = parameter._version
    optimizer.state[parameter][state_name] = state_factory(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        optimizer.step()
    assert parameter._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(parameter.cpu(), before)
    optimizer.state[parameter][state_name] = state


@pytest.mark.parametrize(
    "optimizer_factory, state_name",
    [
        (make_sgd, "momentum_buffer"),
        (make_adam, "exp_avg"),
    ],
)
def test_cpu_tensor_optimizer_state_rejects_before_mutation(
    optimizer_factory, state_name, vulkan_backend
):
    parameter = torch.ones(2, device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    optimizer.state[parameter][state_name] = torch.full(parameter.shape, 7.0)
    parameter.grad = torch.ones_like(parameter)
    before_parameter = parameter.detach().cpu().clone()
    parameter_version = parameter._version
    state_snapshot = _complete_state_snapshot(optimizer, parameter)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="Vulkan.*state.*resident"):
        optimizer.step()

    assert parameter._version == parameter_version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(parameter.cpu(), before_parameter)
    _assert_state_snapshot(optimizer, parameter, state_snapshot)


def _install_malformed_gradient(parameter, kind):
    if kind == "dtype":
        parameter.grad.data = torch.empty(2, dtype=torch.float64, device="vk")
    elif kind == "device":
        parameter.grad = torch.ones(2)
    elif kind == "layout":
        parameter.grad.data = torch.empty((2, 2), device="vk").t()
    elif kind == "offset":
        parameter.grad.data = torch.empty(3, device="vk")[1:]
    elif kind == "overlap":
        parameter.grad.data = torch.as_strided(torch.empty(2, device="vk"), (2,), (0,))
    elif kind == "shape":
        parameter.grad.data = torch.empty(3, device="vk")
    else:
        raise AssertionError(f"unknown malformed gradient kind: {kind}")


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
@pytest.mark.parametrize(
    "kind, message",
    [
        ("dtype", "float32"),
        ("device", "device"),
        ("layout", "contiguous"),
        ("offset", "zero-offset"),
        ("overlap", "overlapping"),
        ("shape", "size|shape"),
    ],
)
def test_malformed_vulkan_gradient_rejects_before_dispatch(
    optimizer_factory, kind, message, vulkan_backend
):
    parameter_shape = (2, 2) if kind == "layout" else (2,)
    parameter = torch.ones(parameter_shape, device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    cpu_parameter = torch.ones(parameter_shape, requires_grad=True)
    cpu_optimizer = optimizer_factory([cpu_parameter])
    cpu_parameter.grad = torch.ones_like(cpu_parameter)
    cpu_optimizer.step()
    vk_state = _complete_state_snapshot(optimizer, parameter)
    cpu_state = _complete_state_snapshot(cpu_optimizer, cpu_parameter)
    if optimizer_factory is make_adam:
        assert "step" in optimizer.state[parameter]
        assert "step" in cpu_optimizer.state[cpu_parameter]
    before = parameter.detach().cpu().clone()
    version = parameter._version
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        _install_malformed_gradient(parameter, kind)
        optimizer.step()
    assert parameter._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(parameter.cpu(), before)
    _assert_state_snapshot(optimizer, parameter, vk_state)
    _assert_state_snapshot(cpu_optimizer, cpu_parameter, cpu_state)


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_float16_vulkan_parameters_reject_before_optimizer_work(
    optimizer_factory, vulkan_backend
):
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="float16|float32|deferred"):
        parameter = torch.empty(
            2, dtype=torch.float16, device=vulkan_backend
        ).requires_grad_()
        optimizer_factory([parameter])
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize("optimizer_factory", [make_sgd, make_adam])
def test_mixed_cpu_vulkan_parameters_reject_before_dispatch(
    optimizer_factory, vulkan_backend
):
    parameters = [
        torch.ones(2, device=vulkan_backend, requires_grad=True),
        torch.ones(2, requires_grad=True),
    ]
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="vk:0"):
        optimizer_factory(parameters)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize(
    "optimizer_factory, state_name",
    [
        (make_sgd, "momentum_buffer"),
        (make_adam, "exp_avg"),
    ],
)
def test_wrong_shaped_optimizer_state_rejects_before_dispatch(
    optimizer_factory, state_name, vulkan_backend
):
    parameter = torch.ones(2, device=vulkan_backend, requires_grad=True)
    optimizer = optimizer_factory([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.state[parameter][state_name] = torch.ones(3, device=vulkan_backend)
    parameter.grad = torch.ones_like(parameter)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="shape|size|matching"):
        optimizer.step()
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


def test_nonzero_vulkan_device_index_rejects_before_dispatch(vulkan_backend):
    if pytorch_vulkan.device_count() < 2:
        pytest.skip("requires a second Vulkan device")
    parameter = torch.ones(2, device="vk:1", requires_grad=True)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="vk:0"):
        make_sgd([parameter])
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize("operation", ["add_", "sub_", "mul_"])
def test_public_inplace_pointwise_is_rejected(vulkan_backend, operation):
    cpu = torch.tensor([1.0, -2.0, 3.0])
    vk = cpu.to(vulkan_backend)
    other_vk = torch.tensor([0.5, 2.0, -4.0], device=vulkan_backend)
    with pytest.raises(
        RuntimeError,
        match=rf"Vulkan {operation}.*in-place operations are unsupported",
    ):
        getattr(vk, operation)(other_vk)


def test_public_inplace_pointwise_scalar_is_rejected(vulkan_backend):
    for operation, value in (("add_", 0.5), ("sub_", 0.5), ("mul_", 2.5)):
        cpu = torch.tensor([1.0, -2.0, 3.0])
        vk = cpu.to(vulkan_backend)
        with pytest.raises(
            RuntimeError,
            match=rf"Vulkan {operation}.*in-place operations are unsupported",
        ):
            getattr(vk, operation)(value)


def test_zero_and_fill_mutate_vulkan_storage(vulkan_backend):
    cpu = torch.full((4,), 3.0)
    vk = cpu.to(vulkan_backend)
    before = vk.data_ptr()
    cpu.zero_()
    result = vk.zero_()
    assert result.data_ptr() == before
    torch.testing.assert_close(vk.cpu(), cpu)
    cpu.fill_(2.5)
    result = vk.fill_(2.5)
    assert result.data_ptr() == before
    torch.testing.assert_close(vk.cpu(), cpu)


def test_inplace_pointwise_rejects_partial_overlap_before_dispatch(vulkan_backend):
    cpu = torch.arange(6.0)
    vk = cpu.to(vulkan_backend)
    self_vk = vk[:4]
    other_vk = vk[1:5]
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(
        RuntimeError, match="Vulkan add_.*in-place operations are unsupported"
    ):
        self_vk.add_(other_vk)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    torch.testing.assert_close(vk.cpu(), cpu)


@pytest.mark.parametrize("view", [slice(1, 5)])
def test_public_inplace_pointwise_rejects_valid_view_layouts(vulkan_backend, view):
    cpu = torch.arange(6.0)
    vk = cpu.to(vulkan_backend)
    self_vk = vk[view]
    other_vk = torch.full_like(self_vk, 2.0)
    with pytest.raises(
        RuntimeError, match="Vulkan add_.*in-place operations are unsupported"
    ):
        self_vk.add_(other_vk)


def test_inplace_pointwise_rejects_uncertain_noncontiguous_view(vulkan_backend):
    vk = torch.arange(6.0).to(vulkan_backend)[::2]
    other = torch.ones(3, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(
        RuntimeError, match="Vulkan add_.*in-place operations are unsupported"
    ):
        vk.add_(other)

    assert pytorch_vulkan._C.compute_dispatch_count() == 0


def test_fill_and_zero_increment_version_without_cpu_payload_transfer(vulkan_backend):
    vk = torch.full((4,), 3.0, device=vulkan_backend)
    before = vk._version
    pytorch_vulkan._C.reset_execution_counters()

    vk.fill_(2.5)
    assert vk._version == before + 1
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0

    before = vk._version
    vk.zero_()
    assert vk._version == before + 1
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_matches_cpu_without_readback(vulkan_backend, name):
    cpu = torch.tensor([2.0, 4.0, 8.0])
    vk = cpu.to(vulkan_backend)
    a_cpu = torch.tensor([1.0, 3.0, 5.0])
    b_cpu = torch.tensor([2.0, 2.0, 2.0])
    a_vk, b_vk = a_cpu.to(vulkan_backend), b_cpu.to(vulkan_backend)
    before = vk.data_ptr()
    pytorch_vulkan._C.reset_execution_counters()

    getattr(cpu, name)(a_cpu, b_cpu, value=0.25)
    result = getattr(vk, name)(a_vk, b_vk, value=0.25)

    assert result.data_ptr() == before
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(vk.cpu(), cpu)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_rejects_mismatched_shapes_without_side_effects(
    vulkan_backend, name
):
    self_vk = torch.ones(3, device=vulkan_backend)
    tensor1 = torch.ones(2, device=vulkan_backend)
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = self_vk.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="matching.*(shape|operands)|equal.*shape"):
        getattr(self_vk, name)(tensor1, tensor2)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(self_vk.cpu(), before)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_rejects_non_float32_without_side_effects(vulkan_backend, name):
    self_vk = torch.ones(3, device=vulkan_backend)
    tensor1 = torch.empty(3, dtype=torch.bool, device=vulkan_backend)
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = self_vk.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="float32"):
        getattr(self_vk, name)(tensor1, tensor2)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(self_vk.cpu(), before)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_rejects_noncontiguous_without_side_effects(
    vulkan_backend, name
):
    self_vk = torch.ones(3, device=vulkan_backend)
    tensor1 = torch.ones(6, device=vulkan_backend)[::2]
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = self_vk.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="contiguous"):
        getattr(self_vk, name)(tensor1, tensor2)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(self_vk.cpu(), before)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_rejects_partial_overlap_without_side_effects(
    vulkan_backend, name
):
    base = torch.ones(4, device=vulkan_backend)
    self_vk = base[:3]
    tensor1 = base[1:]
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = base.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="overlap"):
        getattr(self_vk, name)(tensor1, tensor2)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(base.cpu(), before)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
def test_compound_update_rejects_uncertain_overlap_without_side_effects(
    vulkan_backend, name
):
    self_vk = torch.ones(3, device=vulkan_backend)
    tensor1 = torch.as_strided(torch.ones(6, device=vulkan_backend), (3,), (2,))
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = self_vk.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="contiguous|overlap"):
        getattr(self_vk, name)(tensor1, tensor2)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(self_vk.cpu(), before)


@pytest.mark.parametrize("name", ["addcmul_", "addcdiv_"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 1e-50])
def test_compound_update_rejects_invalid_scalar_without_side_effects(
    vulkan_backend, name, value
):
    self_vk = torch.ones(3, device=vulkan_backend)
    tensor1 = torch.ones(3, device=vulkan_backend)
    tensor2 = torch.ones(3, device=vulkan_backend)
    before = self_vk.cpu()
    version = self_vk._version
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="non-finite|outside float32|underflows"):
        getattr(self_vk, name)(tensor1, tensor2, value=value)

    assert self_vk._version == version
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(self_vk.cpu(), before)
