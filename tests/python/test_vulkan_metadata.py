import io

import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _source(device):
    return torch.arange(12, dtype=torch.float32).reshape(3, 4).to(device)


def test_contiguous_f32_metadata_contract(vulkan_backend):
    tensor = _source(vulkan_backend)

    assert tensor.dtype is torch.float32
    assert tensor.device == torch.device("vk:0")
    assert tensor.layout == torch.strided
    assert tensor.dim() == 2
    assert tuple(tensor.size()) == (3, 4)
    assert tuple(tensor.stride()) == (4, 1)
    assert tensor.numel() == 12
    assert tensor.storage_offset() == 0
    assert tensor.is_contiguous()


def test_positive_stride_nonzero_offset_view_aliases_storage(vulkan_backend):
    source = _source(vulkan_backend)
    view = torch.as_strided(source, (2, 2), (1, 2), storage_offset=1)

    assert view.storage_offset() == 1
    assert tuple(view.stride()) == (1, 2)
    assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    torch.testing.assert_close(view.cpu(), source.cpu().as_strided((2, 2), (1, 2), 1))


def test_view_reshape_and_as_strided_preserve_aliasing(vulkan_backend):
    source = _source(vulkan_backend)

    for view in (source.view(12), source.reshape(12), torch.as_strided(source, (3, 4), (4, 1))):
        assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()


def test_consumers_reject_overlapping_views(vulkan_backend):
    source = _source(vulkan_backend)
    overlapping = torch.as_strided(source, (2, 4), (0, 1))

    with pytest.raises(RuntimeError, match="overlap"):
        overlapping.copy_(source[:2])


def test_views_reject_negative_stride_and_invalid_reachable_range(vulkan_backend):
    source = _source(vulkan_backend)

    with pytest.raises(RuntimeError, match="negative stride"):
        torch.as_strided(source, (2, 2), (-1, 1))
    with pytest.raises(RuntimeError, match="outside its Vulkan allocation"):
        torch.as_strided(source, (2, 2), (4, 1), storage_offset=8)


def test_serialization_rejects_overlapping_metadata(vulkan_backend):
    source = _source(vulkan_backend)
    overlapping = torch.as_strided(source, (2, 4), (0, 1))

    with pytest.raises(ValueError, match="overlap"):
        pytorch_vulkan.save(overlapping, io.BytesIO())


def test_vulkan_rejects_unsupported_dtype(vulkan_backend):
    with pytest.raises((RuntimeError, ValueError), match="(unsupported|supports|Double|float64|dtype)"):
        torch.empty((2,), dtype=torch.int32, device=vulkan_backend)
