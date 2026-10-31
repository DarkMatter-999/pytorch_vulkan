import copy

import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_nll_loss_mean_fixed_contract_matches_cpu(vulkan_backend):
    torch.manual_seed(311)
    cpu_logits = torch.randn(2, 3, dtype=torch.float32)
    cpu_labels = torch.tensor([1, -100], dtype=torch.int64)
    logits = cpu_logits.to(vulkan_backend)
    labels = cpu_labels.to(vulkan_backend)
    expected = torch.ops.aten.nll_loss_forward.default(
        torch.log_softmax(cpu_logits, dim=1), cpu_labels, None, 1, -100
    )
    actual = torch.ops.aten.nll_loss_forward.default(
        torch.log_softmax(logits, dim=1), labels, None, 1, -100
    )
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference)


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_nll_loss_accepts_weights_and_every_reduction(vulkan_backend, reduction):
    # Replaces the fixed-slice rejection test: weights and all reductions are now supported.
    torch.manual_seed(5)
    log_probs = torch.log_softmax(torch.randn(6, 4), dim=1)
    cpu_labels = torch.tensor([0, 1, 2, 3, -100, 1], dtype=torch.int64)
    cpu_weight = torch.rand(4) + 0.25
    expected = torch.ops.aten.nll_loss_forward.default(
        log_probs, cpu_labels, cpu_weight, reduction, -100
    )
    actual = torch.ops.aten.nll_loss_forward.default(
        log_probs.to(vulkan_backend), cpu_labels.to(vulkan_backend),
        cpu_weight.to(vulkan_backend), reduction, -100,
    )
    pytorch_vulkan._C.synchronize()
    for result, reference in zip(actual, expected):
        assert result.device == torch.device(vulkan_backend)
        torch.testing.assert_close(result.cpu(), reference, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "labels, message",
    [
        (torch.tensor([0, 1], dtype=torch.int64), "vk:0"),
        (torch.ones(2, dtype=torch.float32), "int64"),
        (torch.tensor([0, 1, 2], dtype=torch.int64).to("cpu"), "size mismatch"),
    ],
)
def test_nll_loss_rejects_malformed_labels_before_vulkan_work(
    labels, message, vulkan_backend
):
    logits = torch.ones(2, 3).to(vulkan_backend)
    if labels.device.type == "cpu" and message != "vk:0":
        labels = labels.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        torch.ops.aten.nll_loss_forward.default(logits, labels, None, 1, -100)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "labels",
    [
        torch.tensor([-1, 1], dtype=torch.int64),
        torch.tensor([3, 1], dtype=torch.int64),
        torch.tensor([255, 1], dtype=torch.int64),
        torch.tensor([2**40, -(2**63)], dtype=torch.int64),
    ],
)
def test_cpu_label_transfer_accepts_every_int64_label(labels, vulkan_backend):
    # Replaces the negative-label upload rejection: validity is judged at the NLL call,
    # where the class count and ignore_index are known.
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    labels.to(vulkan_backend)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 1, 0)


def test_nll_loss_backward_keeps_gradient_on_vulkan(vulkan_backend):
    cpu_logits = torch.randn(2, 3, dtype=torch.float32)
    cpu_labels = torch.tensor([0, 2], dtype=torch.int64)
    log_cpu = torch.log_softmax(cpu_logits, dim=1)
    loss_cpu, total_cpu = torch.ops.aten.nll_loss_forward.default(
        log_cpu, cpu_labels, None, 1, -100
    )
    logits = cpu_logits.to(vulkan_backend)
    labels = cpu_labels.to(vulkan_backend)
    log_vk = torch.log_softmax(logits, dim=1)
    loss_vk, total_vk = torch.ops.aten.nll_loss_forward.default(
        log_vk, labels, None, 1, -100
    )
    expected = torch.ops.aten.nll_loss_backward.default(
        torch.ones_like(loss_cpu), log_cpu, cpu_labels, None, 1, -100, total_cpu
    )
    actual = torch.ops.aten.nll_loss_backward.default(
        torch.ones_like(loss_vk), log_vk, labels, None, 1, -100, total_vk
    )
    assert actual.device == torch.device(vulkan_backend)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
def test_composed_cross_entropy_initializes_all_descriptors(
    vulkan_backend, reduction, capfd
):
    generator = torch.Generator(device="cpu").manual_seed(12012)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(generator.get_state())
        reference = torch.nn.Sequential(
            torch.nn.Flatten(1),
            torch.nn.Linear(784, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3),
        )
    model = copy.deepcopy(reference).to(vulkan_backend)
    cpu_input = torch.rand(
        (2, 1, 28, 28),
        generator=torch.Generator(device="cpu").manual_seed(12015),
    )
    cpu_labels = torch.tensor([0, 2], dtype=torch.int64)
    loss_fn = torch.nn.CrossEntropyLoss(reduction=reduction)
    expected = loss_fn(reference(cpu_input), cpu_labels)
    expected.backward(torch.ones_like(expected))
    inputs = cpu_input.to(vulkan_backend)
    labels = cpu_labels.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()

    actual = loss_fn(model(inputs), labels)
    actual.backward(torch.ones_like(actual))
    pytorch_vulkan._C.synchronize()
    dispatches, vulkan_copies, transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert dispatches > 0
    assert vulkan_copies >= 0
    assert (transfers, fallbacks) == (0, 0)
    assert actual.device == torch.device(vulkan_backend)
    torch.testing.assert_close(
        actual.detach().cpu(), expected.detach(), rtol=2e-4, atol=2e-4
    )
    for parameter, cpu_parameter in zip(model.parameters(), reference.parameters()):
        assert parameter.grad.device == torch.device(vulkan_backend)
        torch.testing.assert_close(
            parameter.grad.cpu(), cpu_parameter.grad, rtol=2e-4, atol=2e-4
        )
    captured = capfd.readouterr()
    assert "Validation Error:" not in captured.out + captured.err
