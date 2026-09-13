import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_print_and_repr_reject_deferred_double_formatting(vulkan_backend):
    values = torch.tensor([-3.5, 0.0, 2.25, 1.0e20], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    with pytest.raises(RuntimeError, match="Double"):
        repr(tensor)


def test_multidimensional_print_rejects_deferred_double_formatting(vulkan_backend):
    cpu = torch.tensor([[-1.0, 0.0], [3.5, 7.25]], dtype=torch.float32)
    tensor = cpu.to(vulkan_backend)

    with pytest.raises(RuntimeError, match="Double"):
        print(tensor)


def test_isfinite_accepts_float32_max(vulkan_backend):
    values = torch.tensor([torch.finfo(torch.float32).max], dtype=torch.float32).to(vulkan_backend)

    result = torch.isfinite(values)

    torch.testing.assert_close(result.cpu(), torch.tensor([True]))


def test_print_flattening_view_rejects_deferred_double_formatting(vulkan_backend):
    source = torch.tensor([[-2.0, 0.0], [4.0, 8.0]], dtype=torch.float32).to(vulkan_backend)
    flattened = source.view(-1)

    assert flattened.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(flattened.shape) == (4,)
    assert tuple(flattened.stride()) == (1,)
    with pytest.raises(RuntimeError, match="Double"):
        repr(flattened)


def test_masked_select_compacts_f32_values_on_vulkan(vulkan_backend):
    values = torch.tensor([-2.0, 0.0, 4.5, 8.0], dtype=torch.float32).to(vulkan_backend)
    mask = torch.tensor([True, False, True, False], dtype=torch.bool).to(vulkan_backend)

    result = torch.masked_select(values, mask)

    assert result.device == values.device
    assert result.dtype == torch.float32
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), torch.tensor([-2.0, 4.5]))


@pytest.mark.parametrize("selected", [0, 4])
def test_masked_select_handles_empty_and_full_masks(vulkan_backend, selected):
    values = torch.arange(4, dtype=torch.float32).to(vulkan_backend)
    mask = torch.tensor([index < selected for index in range(4)], dtype=torch.bool).to(vulkan_backend)

    result = torch.masked_select(values, mask)

    assert result.device == values.device
    assert result.shape == (selected,)
    torch.testing.assert_close(result.cpu(), torch.arange(selected, dtype=torch.float32))


def test_masked_select_rejects_broadcasting_and_unsupported_metadata(vulkan_backend):
    values = torch.arange(4, dtype=torch.float32).to(vulkan_backend)
    mask = torch.ones(4, dtype=torch.bool).to(vulkan_backend)

    with pytest.raises(RuntimeError, match="equal shapes"):
        torch.masked_select(values.view(2, 2), mask)


@pytest.mark.parametrize(
    "mask_factory, message",
    [
        (lambda values: torch.ones(4, dtype=torch.float32).to(values.device), "bool"),
        (lambda values: torch.as_strided(torch.ones(4, dtype=torch.bool).to(values.device), (2, 2), (1, 2)), "contiguous"),
        (lambda values: torch.as_strided(torch.ones(5, dtype=torch.bool).to(values.device), (4,), (1,), 1), "storage_offset"),
    ],
)
def test_masked_select_rejects_unsupported_mask_forms(vulkan_backend, mask_factory, message):
    values = torch.arange(4, dtype=torch.float32).to(vulkan_backend)
    mask = mask_factory(values)

    with pytest.raises(RuntimeError, match=message):
        torch.masked_select(values, mask)


def test_masked_select_rejects_cpu_mask(vulkan_backend):
    values = torch.arange(4, dtype=torch.float32).to(vulkan_backend)

    with pytest.raises(RuntimeError, match="vk:0"):
        torch.masked_select(values, torch.ones(4, dtype=torch.bool))


@pytest.mark.parametrize(
    "values_factory, message",
    [
        (lambda device: torch.as_strided(torch.arange(5, dtype=torch.float32).to(device), (2, 2), (1, 2)), "contiguous"),
        (lambda device: torch.as_strided(torch.arange(5, dtype=torch.float32).to(device), (4,), (1,), 1), "storage_offset"),
    ],
)
def test_masked_select_rejects_unsupported_input_forms(vulkan_backend, values_factory, message):
    values = values_factory(vulkan_backend)
    mask = torch.ones(values.shape, dtype=torch.bool).to(vulkan_backend)

    with pytest.raises(RuntimeError, match=message):
        torch.masked_select(values, mask)


def test_masked_select_preserves_global_double_rejection(vulkan_backend):
    with pytest.raises(RuntimeError, match="Double"):
        torch.arange(4, dtype=torch.float64).to(vulkan_backend)
