import pytest
import torch

import pytorch_vulkan
from pytorch_vulkan.compiler import VulkanCompilerError


@pytest.fixture
def vulkan_device():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def _mlp(device):
    torch.manual_seed(17)
    return torch.nn.Sequential(
        torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
    ).to(device=device, dtype=torch.float32)


def _mnist_model(device):
    torch.manual_seed(19)
    return torch.nn.Sequential(
        torch.nn.Flatten(start_dim=1),
        torch.nn.Linear(784, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, 10),
    ).to(device=device, dtype=torch.float32)


def _compile_and_check(model, cpu_model, cpu_input, device):
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True)
    vk_input = cpu_input.to(device)
    pytorch_vulkan._C.reset_execution_counters()
    first = compiled(vk_input)
    first_counters = (
        pytorch_vulkan._C.compute_dispatch_count(),
        pytorch_vulkan._C.compute_submitted_count(),
        pytorch_vulkan._C.compute_completed_count(),
        pytorch_vulkan._C.compute_wait_count(),
        pytorch_vulkan._C.explicit_transfer_count(),
    )
    pytorch_vulkan._C.reset_execution_counters()
    second = compiled(vk_input)
    second_counters = (
        pytorch_vulkan._C.compute_dispatch_count(),
        pytorch_vulkan._C.compute_submitted_count(),
        pytorch_vulkan._C.compute_completed_count(),
        pytorch_vulkan._C.compute_wait_count(),
        pytorch_vulkan._C.explicit_transfer_count(),
    )

    expected = cpu_model(cpu_input)
    for output in (first, second):
        assert output.device == torch.device("vk:0")
        assert output.is_contiguous()
        torch.testing.assert_close(output.cpu(), expected)
    for counters in (first_counters, second_counters):
        dispatches, submissions, completions, waits, transfers = counters
        assert dispatches > 0
        assert submissions > 0
        assert completions > 0
        assert waits > 0
        assert transfers == 0

    stats = pytorch_vulkan.compiler_stats()
    assert stats["node_count"] > 0
    assert stats["calls"] == 2
    assert stats["setup_time"] >= 0
    assert stats["replay_time"] >= 0
    assert len(stats["call_metrics"]) == 2
    for metrics in stats["call_metrics"]:
        assert metrics["dispatches"] > 0
        assert metrics["submissions"] > 0
        assert metrics["completions"] > 0
        assert metrics["waits"] > 0
        assert metrics["transfers"] == 0


def test_compile_linear_relu_linear_replays_on_vulkan_without_transfer(vulkan_device):
    cpu_model = _mlp("cpu")
    model = _mlp(vulkan_device)
    model.load_state_dict({name: value.to(vulkan_device) for name, value in cpu_model.state_dict().items()})
    _compile_and_check(model, cpu_model, torch.randn(2, 8), vulkan_device)
    assert pytorch_vulkan.compiler_stats()["fusion_applied"] is True


def test_compile_same_module_sequence_with_altered_wiring_is_not_fused(vulkan_device):
    class AlteredWiring(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(8, 16)
            self.activation = torch.nn.ReLU()
            self.second = torch.nn.Linear(16, 4)

        def forward(self, value):
            first = self.first(value)
            self.activation(first)
            return self.second(first)

    torch.manual_seed(17)
    model = AlteredWiring().to(device=vulkan_device, dtype=torch.float32)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True)
    compiled(torch.randn(2, 8).to(vulkan_device))

    assert pytorch_vulkan.compiler_stats()["fusion_applied"] is False


def test_compile_valid_chain_with_earlier_output_is_not_fused(vulkan_device):
    class EarlierOutput(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.Linear(8, 16)
            self.activation = torch.nn.ReLU()
            self.second = torch.nn.Linear(16, 4)

        def forward(self, value):
            first = self.first(value)
            activated = self.activation(first)
            self.second(activated)
            return activated

    torch.manual_seed(17)
    model = EarlierOutput().to(device=vulkan_device, dtype=torch.float32)
    cpu_model = EarlierOutput()
    cpu_model.load_state_dict({name: value.cpu() for name, value in model.state_dict().items()})
    cpu_input = torch.randn(2, 8)
    torch._dynamo.reset()
    compiled = torch.compile(model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True)
    result = compiled(cpu_input.to(vulkan_device))

    torch.testing.assert_close(result.cpu(), cpu_model(cpu_input))
    assert pytorch_vulkan.compiler_stats()["fusion_applied"] is False


def test_compile_mnist_shaped_graph_replays_on_vulkan_without_transfer(vulkan_device):
    cpu_model = _mnist_model("cpu")
    model = _mnist_model(vulkan_device)
    model.load_state_dict({name: value.to(vulkan_device) for name, value in cpu_model.state_dict().items()})
    _compile_and_check(model, cpu_model, torch.randn(2, 1, 28, 28), vulkan_device)
    assert pytorch_vulkan.compiler_stats()["fusion_applied"] is True


def test_vulkan_backend_rejects_unsupported_fx_nodes():
    class Unsupported(torch.nn.Module):
        def forward(self, value):
            return torch.sigmoid(value)

    graph = torch.fx.symbolic_trace(Unsupported())
    with pytest.raises(VulkanCompilerError, match="unsupported FX node"):
        pytorch_vulkan.vulkan_backend(graph, (torch.ones(2, 8, device="meta"),))


def test_vulkan_backend_rejects_dynamic_symbolic_shapes(vulkan_device):
    model = _mnist_model(vulkan_device)
    torch._dynamo.reset()
    compiled = torch.compile(
        model, backend=pytorch_vulkan.vulkan_backend, fullgraph=True, dynamic=True
    )
    vk_input = torch.randn(2, 1, 28, 28).to(vulkan_device)
    with pytest.raises((VulkanCompilerError, torch._dynamo.exc.BackendCompilerFailed),
                       match="dynamic symbolic shapes are unsupported"):
        compiled(vk_input)
