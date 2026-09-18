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


def test_nll_loss_rejects_weights_and_sum_without_vulkan_work(vulkan_backend):
    logits = torch.ones(2, 3).to(vulkan_backend)
    labels = torch.tensor([0, 1], dtype=torch.int64).to(vulkan_backend)
    weight = torch.ones(3).to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="class weights|reduction"):
        torch.ops.aten.nll_loss_forward.default(
            logits, labels, weight, 0, -100
        )
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize(
    "labels, message",
    [(torch.tensor([0, 1], dtype=torch.int64), "vk:0"),
     (torch.ones(2, dtype=torch.float32), "int64"),
     (torch.tensor([0, 1, 2], dtype=torch.int64).to("cpu"), "shape")],
)
def test_nll_loss_rejects_invalid_labels_before_vulkan_work(labels, message, vulkan_backend):
    logits = torch.ones(2, 3).to(vulkan_backend)
    if labels.device.type == "cpu" and message != "vk:0":
        labels = labels.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=message):
        torch.ops.aten.nll_loss_forward.default(logits, labels, None, 1, -100)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


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
