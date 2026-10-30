"""CPU ownership regression for the Vulkan RNN fixture's state hook."""

import gc
import weakref

import torch

from tools.vulkan_model_benchmark import _VanillaRNN


class _VulkanInput(torch.Tensor):
    @property
    def device(self):
        return torch.device("privateuseone:0")


def _run_fixture(monkeypatch):
    # Replace only the device-specific leaf; execute the actual fixture branch
    # and autograd hook using a regular CPU output tensor.
    output = torch.tensor([[[1.0], [2.0]]], requires_grad=True) * 1
    monkeypatch.setattr(torch.ops.pytorch_vulkan, "rnn_sequence", lambda *args: output, raising=False)
    model = _VanillaRNN(1, 1)
    inputs = torch.zeros(1, 2, 1).as_subclass(_VulkanInput)
    assert model(inputs) is output
    return model, output


def test_state_hook_does_not_keep_discarded_views_or_output_alive(monkeypatch):
    model, output = _run_fixture(monkeypatch)
    states = [weakref.ref(state) for state in model.last_states]
    output_ref = weakref.ref(output)
    del model.last_states
    gc.collect()
    assert all(state() is None for state in states)
    # The hook still exists on a live output, but must not retain its views.
    output.sum().backward()
    monkeypatch.undo()
    del output, model
    gc.collect()
    assert output_ref() is None


def test_state_hook_keeps_retained_states_updated_across_backward(monkeypatch):
    model, output = _run_fixture(monkeypatch)
    states = model.last_states
    del model.last_states
    output.backward(torch.tensor([[[3.0], [5.0]]]), retain_graph=True)
    torch.testing.assert_close(states[0].grad, torch.tensor([[3.0]]))
    torch.testing.assert_close(states[1].grad, torch.tensor([[5.0]]))
    output.backward(torch.tensor([[[7.0], [11.0]]]))
    torch.testing.assert_close(states[0].grad, torch.tensor([[10.0]]))
    torch.testing.assert_close(states[1].grad, torch.tensor([[16.0]]))
