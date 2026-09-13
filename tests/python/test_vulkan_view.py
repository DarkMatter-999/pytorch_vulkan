import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _source(vulkan_backend):
    return torch.arange(6, dtype=torch.float32).reshape(2, 3).to(vulkan_backend)


@pytest.mark.parametrize(
    "make_view, expected_shape, expected_stride",
    [
        (lambda source: torch.as_strided(source, source.size(), source.stride()), (2, 3), (3, 1)),
        (lambda source: source.view(-1), (6,), (1,)),
        (lambda source: torch.reshape(source, (6,)), (6,), (1,)),
    ],
)
def test_metadata_only_views_preserve_storage_and_values(
    vulkan_backend, make_view, expected_shape, expected_stride
):
    source = _source(vulkan_backend)
    result = make_view(source)

    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(result.shape) == expected_shape
    assert tuple(result.stride()) == expected_stride
    assert tuple(source.shape) == (2, 3)
    torch.testing.assert_close(
        result.cpu(), torch.arange(6, dtype=torch.float32).reshape(expected_shape)
    )


def _assert_view_rejected(operation, message):
    with pytest.raises(RuntimeError, match=message):
        operation()


def test_metadata_only_views_reject_unsupported_metadata(vulkan_backend):
    source = _source(vulkan_backend)
    _assert_view_rejected(lambda: source.view(5), "invalid for input")
    _assert_view_rejected(
        lambda: torch.as_strided(source, (2, 3), (-1, 1)),
        "negative stride",
    )
    _assert_view_rejected(
        lambda: torch.as_strided(source, (2, 3), (1, 3)),
        "outside its Vulkan allocation",
    )


def test_as_strided_accepts_general_in_allocation_metadata(vulkan_backend):
    source = _source(vulkan_backend)
    views = [
        torch.as_strided(source, (2,), (2,)),
        torch.as_strided(source, (2, 3), (1, 2)),
        torch.as_strided(source, (2, 3), (0, 1)),
        torch.as_strided(source, (2,), (1,), storage_offset=1),
    ]
    for result in views:
        assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(views[0].shape) == (2,)
    assert tuple(views[1].stride()) == (1, 2)
    assert tuple(views[2].stride()) == (0, 1)
    assert views[3].storage_offset() == 1


def test_strided_views_cover_transpose_slice_zero_stride_and_empty(vulkan_backend):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(vulkan_backend)
    transpose = source.transpose(0, 1)
    sliced = source[:, 1:]
    broadcast = torch.as_strided(source, (2, 3), (0, 1), storage_offset=2)
    empty = torch.as_strided(source, (0, 4), (4, 1), storage_offset=12)

    for view in (transpose, sliced, broadcast, empty):
        assert view.device == source.device
        assert view.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(transpose.stride()) == (1, 4)
    assert tuple(sliced.shape) == (3, 3) and tuple(sliced.stride()) == (4, 1)
    assert tuple(broadcast.stride()) == (0, 1) and broadcast.storage_offset() == 2
    assert empty.numel() == 0
