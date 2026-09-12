import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_cpu_view_and_print_remain_normal_pytorch_behavior(capsys):
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    viewed = tensor.view(-1)

    assert viewed.shape == (4,)
    assert "tensor(" in repr(tensor)
    print(tensor)
    assert "1." in capsys.readouterr().out


def test_vulkan_device_index_one_is_rejected_at_supported_device_boundary():
    with pytest.raises(RuntimeError, match="only device index 0"):
        torch.empty((4,), dtype=torch.float32, device="vk:1")


@pytest.mark.parametrize(
    "source",
    [
        torch.tensor([1.0, -2.5, 3.25, 0.0], dtype=torch.float32),
        torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.float32),
    ],
)
def test_vulkan_repr_shows_values_and_device(vulkan_backend, source):
    tensor = source.to(vulkan_backend)

    representation = repr(tensor)
    assert "tensor(" in representation
    assert "1." in representation
    assert "-2.5" in representation
    assert "device='vk:0'" in representation or "device='privateuseone:0'" in representation


@pytest.mark.parametrize(
    "source",
    [
        torch.tensor([1.0, -2.5, 3.25, 0.0], dtype=torch.float32),
        torch.tensor([[1.0, -2.5], [3.25, 0.0]], dtype=torch.float32),
    ],
)
def test_vulkan_print_shows_values_and_device(capsys, vulkan_backend, source):
    tensor = source.to(vulkan_backend)

    print(tensor)
    printed = capsys.readouterr().out

    assert "tensor(" in printed
    assert "1." in printed
    assert "-2.5" in printed
    assert "device='vk:0'" in printed or "device='privateuseone:0'" in printed


def test_vulkan_abs_out_preserves_supplied_output(vulkan_backend):
    source = torch.tensor([1.0, -2.5, 0.0], dtype=torch.float32).to(vulkan_backend)
    output = torch.empty_like(source)
    returned = torch.abs(source, out=output)

    assert returned is output
    assert output.device == source.device
    torch.testing.assert_close(output.to("cpu"), torch.tensor([1.0, 2.5, 0.0]))


def test_vulkan_abs_out_resizes_empty_formatter_buffer(vulkan_backend):
    source = torch.tensor([1.0, -2.5], dtype=torch.float32).to(vulkan_backend)
    output = torch.empty((0,), dtype=torch.float32, device=vulkan_backend)
    returned = torch.abs(source, out=output)

    assert returned is output
    assert tuple(output.shape) == (2,)
    torch.testing.assert_close(output.to("cpu"), torch.tensor([1.0, 2.5]))
