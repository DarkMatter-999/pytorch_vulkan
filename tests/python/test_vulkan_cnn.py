import pytest
import torch

import pytorch_vulkan
from tools.vulkan_model_benchmark import CNNFixture, _move_model, _run_row, run


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _pair(device, batch=8):
    fixture = CNNFixture(batch=batch)
    cpu_model = fixture.make_cpu()
    inputs, target = fixture.make_inputs()
    vk_model = _move_model(fixture.make_cpu(), device)
    vk_inputs = inputs.to(device).requires_grad_()
    vk_target = target.to(device)
    for name, parameter in list(vk_model.named_parameters()):
        parent_name, _, parameter_name = name.rpartition(".")
        parent = vk_model.get_submodule(parent_name) if parent_name else vk_model
        setattr(
            parent,
            parameter_name,
            torch.nn.Parameter(cpu_model.state_dict()[name].detach().clone().to(device)),
        )
    return fixture, cpu_model, vk_model, inputs.requires_grad_(), vk_inputs, target, vk_target


def _assert_resident(value):
    assert value.device == torch.device("vk:0")
    assert value.dtype is torch.float32
    assert value.is_contiguous()


def test_cnn_forward_backward_training_matches_cpu_and_stays_on_vulkan(vulkan_backend):
    fixture, cpu_model, vk_model, cpu_input, vk_input, cpu_target, vk_target = _pair(
        vulkan_backend
    )
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
    vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.01)

    cpu_optimizer.zero_grad(set_to_none=False)
    cpu_output = cpu_model(cpu_input)
    cpu_loss = fixture.loss(cpu_output, cpu_target)
    cpu_loss.backward()
    pytorch_vulkan._C.reset_execution_counters()
    vk_optimizer.zero_grad(set_to_none=False)
    captured = []
    handles = [
        module.register_forward_hook(lambda _module, _inputs, output: captured.append(output))
        for module in vk_model
    ]
    vk_output = vk_model(vk_input)
    vk_loss = fixture.loss(vk_output, vk_target)
    vk_loss.backward()
    dispatches = pytorch_vulkan._C.compute_dispatch_count()
    transfers = pytorch_vulkan._C.explicit_transfer_count()
    fallbacks = pytorch_vulkan._C.fallback_count()
    for handle in handles:
        handle.remove()

    for actual, expected in zip(vk_model.parameters(), cpu_model.parameters()):
        torch.testing.assert_close(actual.grad.cpu(), expected.grad, rtol=2e-4, atol=2e-4)
    cpu_optimizer.step()
    vk_optimizer.step()

    for value in [vk_output, vk_loss, vk_input, vk_input.grad, *captured]:
        _assert_resident(value)
    for parameter in vk_model.parameters():
        _assert_resident(parameter)
        _assert_resident(parameter.grad)
    torch.testing.assert_close(vk_output.cpu(), cpu_output.detach(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(vk_loss.cpu(), cpu_loss.detach(), rtol=2e-4, atol=2e-4)
    torch.testing.assert_close(vk_input.grad.cpu(), cpu_input.grad, rtol=2e-4, atol=2e-4)
    for actual, expected in zip(vk_model.parameters(), cpu_model.parameters()):
        torch.testing.assert_close(actual.cpu(), expected, rtol=2e-4, atol=2e-4)
    assert dispatches > 0
    assert transfers == 0
    assert fallbacks == 0


def test_cnn_rejects_wrong_device_and_overlapping_operands_before_work(vulkan_backend):
    fixture = CNNFixture(batch=8)
    inputs, _ = fixture.make_inputs()
    model = fixture.make_cpu()
    weight = model[0].weight.detach().to(vulkan_backend)
    bias = model[0].bias.detach().to(vulkan_backend)
    valid = inputs.to(vulkan_backend)
    cases = [
        (valid, weight.cpu(), bias, "device|vk"),
        (valid[:1].expand_as(valid), weight, bias, "overlap|layout"),
        (valid, weight[:1].expand_as(weight), bias, "overlap|layout"),
        (valid, weight, bias[:1].expand_as(bias), "overlap|layout"),
    ]
    for value, case_weight, case_bias, message in cases:
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(RuntimeError, match=message):
            torch.nn.functional.conv2d(value, case_weight, case_bias, padding=1)
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
        assert pytorch_vulkan._C.fallback_count() == 0

    grad = torch.ones((1, 8, 32, 32), device=vulkan_backend).expand(8, 8, 32, 32)
    input_value = valid.requires_grad_()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="overlap|layout"):
        torch.ops.aten.convolution_backward.default(
            grad,
            input_value,
            weight,
            [8],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_cnn_backward_rejects_noncontiguous_grad_output_before_work(vulkan_backend):
    fixture, _, model, _, vk_input, _, _ = _pair(vulkan_backend)
    weight = model[0].weight
    grad_output = torch.ones((8, 8, 32, 32), device=vulkan_backend).transpose(2, 3)
    assert not grad_output.is_contiguous()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="contiguous|layout"):
        torch.ops.aten.convolution_backward.default(
            grad_output,
            vk_input,
            weight,
            [8],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_cnn_backward_rejects_legacy_bias_size_before_work(vulkan_backend):
    fixture, _, model, _, vk_input, _, _ = _pair(vulkan_backend)
    weight = model[0].weight
    grad_output = torch.ones((8, 8, 32, 32), device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="bias"):
        torch.ops.aten.convolution_backward.default(
            grad_output,
            vk_input,
            weight,
            [4],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_cnn_benchmark_initializes_backend_before_cpu_reference(vulkan_backend):
    artifact = run("cnn", "vk:0", warmups=0, repetitions=1)
    assert artifact["rows"][1]["dispatches"] > 0


@pytest.mark.parametrize("batch", [1, 8])
def test_cnn_supported_batch_boundaries(vulkan_backend, batch):
    fixture = CNNFixture(batch=batch)
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    output = model(inputs.to(vulkan_backend))
    _assert_resident(output)
    assert output.shape == (batch, 10)


@pytest.mark.parametrize(
    "mutator, message",
    [
        (lambda x: x.to(torch.float64), "float32|type|dtype"),
        (lambda x: x[:, :, :, :-1], "shape"),
        (lambda x: x.transpose(2, 3), "layout|contiguous"),
    ],
)
def test_cnn_rejects_wrong_schema_before_vulkan_work(vulkan_backend, mutator, message):
    fixture = CNNFixture()
    model = _move_model(fixture.make_cpu(), vulkan_backend)
    inputs, _ = fixture.make_inputs()
    bad_inputs = mutator(inputs.to(vulkan_backend))
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises((RuntimeError, ValueError), match=message):
        model(bad_inputs)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
