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


def _assert_no_implicit_transfer_or_fallback():
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0


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


def test_as_strided_rejects_reachable_uint32_address_overflow(vulkan_backend):
    source = _source(vulkan_backend)
    _assert_view_rejected(
        lambda: torch.as_strided(source, (2,), (2**32 - 1,), storage_offset=1),
        "shader address",
    )
    _assert_view_rejected(
        lambda: torch.as_strided(source, (2, 2), (2**31, 2**31)),
        "shader address",
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


def test_empty_strided_view_validates_storage_boundary(vulkan_backend):
    source = torch.arange(4, dtype=torch.float32).to(vulkan_backend)
    valid = torch.as_strided(source, (0,), (1,), storage_offset=4)
    assert valid.numel() == 0
    _assert_view_rejected(
        lambda: torch.as_strided(source, (0,), (1,), storage_offset=5),
        "outside its Vulkan allocation",
    )
    _assert_view_rejected(
        lambda: torch.as_strided(source, (0,), (1,), storage_offset=2**32 + 1),
        "outside its Vulkan allocation",
    )


def test_reshape_alias_and_incompatible_reshape_copy(vulkan_backend):
    source = _source(vulkan_backend)
    alias = source.reshape(6)
    transposed = source.transpose(0, 1)
    copied = transposed.reshape(6)

    assert alias.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert copied.untyped_storage().data_ptr() != transposed.untyped_storage().data_ptr()
    assert alias.device == source.device == copied.device

    del alias, transposed, source
    torch.testing.assert_close(copied.cpu(), torch.tensor([0., 3., 1., 4., 2., 5.]))


def test_reshape_alias_accepts_compute_stride_noncontiguous_layout(vulkan_backend):
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(vulkan_backend)
    source = base[:, :, :2]
    result = source.reshape(6, 2)

    assert not source.is_contiguous()
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(result.shape) == (6, 2)
    assert tuple(result.stride()) == (4, 1)


def test_reshape_copy_rejects_unsupported_layout_before_transfer(vulkan_backend):
    source = _source(vulkan_backend)
    overlapping = torch.as_strided(source, (2, 3), (0, 1))
    with pytest.raises(RuntimeError, match="non-overlapping"):
        overlapping.reshape(6)


def test_copy_preserves_transposed_and_sliced_logical_order(vulkan_backend):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = cpu.to(vulkan_backend).transpose(0, 1)[:, 1:3]
    destination = torch.empty_strided(source.shape, (1, 4), device=vulkan_backend)

    destination.copy_(source)

    assert tuple(destination.stride()) == (1, 4)
    torch.testing.assert_close(destination.cpu(), cpu.t().contiguous()[:, 1:3])


def test_contiguous_materializes_general_vulkan_view(vulkan_backend):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = cpu.to(vulkan_backend).transpose(0, 1)

    result = source.contiguous()

    assert result.is_contiguous()
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    assert result.device == source.device and result.dtype == source.dtype
    torch.testing.assert_close(result.cpu(), cpu.t())


def test_copy_rejects_overlapping_destination(vulkan_backend):
    source = torch.arange(4, dtype=torch.float32).to(vulkan_backend)
    destination = torch.as_strided(source, (2, 2), (0, 1))

    with pytest.raises(RuntimeError, match="internal overlap"):
        destination.copy_(source)


def test_contiguous_empty_general_view(vulkan_backend):
    source = torch.empty_strided((0, 3), (3, 1), device=vulkan_backend)
    result = source.contiguous()

    assert result.numel() == 0
    assert result.is_contiguous()


def test_chained_metadata_views_replay_autograd(vulkan_backend):
    cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = cpu.to(vulkan_backend).detach().requires_grad_()
    view = source.transpose(0, 1)
    view.retain_grad()

    result = torch.neg(view)
    result.backward(torch.ones(result.shape, dtype=result.dtype).to(vulkan_backend))

    assert view.grad is not None
    assert source.grad is not None
    torch.testing.assert_close(view.grad.cpu(), -torch.ones_like(view.cpu()))
    torch.testing.assert_close(source.grad.cpu(), -torch.ones_like(cpu))


def test_incompatible_reshape_copy_keeps_autograd(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = cpu.to(vulkan_backend).detach().requires_grad_()
    result = torch.neg(source.transpose(0, 1).reshape(6))

    result.backward(torch.ones(result.shape, dtype=result.dtype).to(vulkan_backend))

    assert source.grad is not None
    torch.testing.assert_close(source.grad.cpu(), -torch.ones_like(cpu))


def test_incompatible_reshape_copy_maps_nonuniform_gradient_like_cpu(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    cpu_result = cpu.transpose(0, 1).reshape(6)
    cpu_result.backward(torch.arange(1, 7, dtype=torch.float32))

    source = cpu.detach().to(vulkan_backend).requires_grad_()
    result = source.transpose(0, 1).reshape(6)
    result.backward(torch.arange(1, 7, dtype=torch.float32).to(vulkan_backend))

    assert source.grad is not None
    torch.testing.assert_close(source.grad.cpu(), cpu.grad, rtol=0, atol=0)


def test_view_preserves_gradient(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    grad = torch.arange(1, 7, dtype=torch.float32)
    vk_grad = grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_result = cpu.view(6)
    vk_result = vk.view(6)
    assert vk_result.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
    cpu_result.backward(grad)
    vk_result.backward(vk_grad)

    _assert_no_implicit_transfer_or_fallback()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=0, atol=0)


def test_as_strided_preserves_gradient(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    grad = torch.arange(1, 5, dtype=torch.float32).reshape(2, 2)
    vk_grad = grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_result = torch.as_strided(cpu, (2, 2), (2, 1), storage_offset=1)
    vk_result = torch.as_strided(vk, (2, 2), (2, 1), storage_offset=1)
    assert vk_result.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
    cpu_result.backward(grad)
    vk_result.backward(vk_grad)

    _assert_no_implicit_transfer_or_fallback()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=0, atol=0)


def test_reshape_alias_preserves_gradient(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    grad = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2)
    vk_grad = grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_result = torch.ops.aten._reshape_alias.default(cpu, [3, 2], [2, 1])
    vk_result = torch.ops.aten._reshape_alias.default(vk, [3, 2], [2, 1])
    assert vk_result.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
    cpu_result.backward(grad)
    vk_result.backward(vk_grad)

    _assert_no_implicit_transfer_or_fallback()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=0, atol=0)


def test_incompatible_reshape_preserves_gradient(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    vk = cpu.detach().clone().to(vulkan_backend).requires_grad_()
    grad = torch.arange(1, 7, dtype=torch.float32)
    vk_grad = grad.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_result = cpu.transpose(0, 1).reshape(6)
    vk_source = vk.transpose(0, 1)
    vk_result = vk_source.reshape(6)
    assert vk_result.untyped_storage().data_ptr() != vk_source.untyped_storage().data_ptr()
    cpu_result.backward(grad)
    vk_result.backward(vk_grad)

    _assert_no_implicit_transfer_or_fallback()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=0, atol=0)
