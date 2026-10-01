import pytest
import pytorch_vulkan
import torch


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
        (
            lambda source: torch.as_strided(source, source.size(), source.stride()),
            (2, 3),
            (3, 1),
        ),
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


def test_view_offset_singleton_empty_and_alias_version_match_cpu(vulkan_backend):
    cpu = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    vk = cpu.to(vulkan_backend)
    cases = [
        (cpu.as_strided((2, 2), (12, 1), storage_offset=1),
         vk.as_strided((2, 2), (12, 1), storage_offset=1)),
        (cpu[:, :, ::2].view(2, 6), vk[:, :, ::2].view(2, 6)),
        (cpu[:0].view(0, 3, 4), vk[:0].view(0, 3, 4)),
    ]
    for expected, result in cases:
        assert result.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
        assert tuple(result.shape) == tuple(expected.shape)
        assert tuple(result.stride()) == tuple(expected.stride())
        torch.testing.assert_close(result.cpu(), expected)

    singleton_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    singleton_vk = singleton_cpu.to(vulkan_backend)
    expected_singleton, result_singleton = singleton_cpu.view(2, 3), singleton_vk.view(2, 3)
    assert tuple(result_singleton.stride()) == tuple(expected_singleton.stride())
    assert result_singleton.untyped_storage().data_ptr() == singleton_vk.untyped_storage().data_ptr()
    torch.testing.assert_close(result_singleton.cpu(), expected_singleton)

    cpu_leaf = cpu.clone().requires_grad_()
    vk_leaf = vk.detach().requires_grad_()
    cpu_view, vk_view = cpu_leaf.as_strided((2, 2), (12, 1), storage_offset=1), vk_leaf.as_strided(
        (2, 2), (12, 1), storage_offset=1
    )
    cpu_version, vk_version = cpu_leaf._version, vk_leaf._version
    with torch.no_grad():
        cpu_view[0, 0].add_(100)
        vk_view[0, 0].add_(100)
    assert cpu_leaf._version == cpu_version + 1
    assert vk_leaf._version == vk_version + 1
    assert cpu_view.untyped_storage().data_ptr() == cpu_leaf.untyped_storage().data_ptr()
    assert vk_view.untyped_storage().data_ptr() == vk_leaf.untyped_storage().data_ptr()
    torch.testing.assert_close(vk_leaf.cpu(), cpu_leaf)


def test_incompatible_view_rejects_before_device_work(vulkan_backend):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4).to(vulkan_backend).transpose(0, 1)
    cpu_source = torch.arange(8, dtype=torch.float32).reshape(2, 4).transpose(0, 1)
    with pytest.raises(RuntimeError, match="not compatible"):
        cpu_source.view(2, 4)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="not compatible"):
        source.view(2, 4)
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0


def test_fake_tensor_metadata_redispatch_preserves_view_geometry():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        source = torch.empty((2, 3), device="vk:0")
        result = torch.ops.aten.as_strided.default(source, [2, 2], [3, 1], 1)

    assert type(result) is type(source)
    assert result.device == source.device
    assert tuple(result.shape) == (2, 2)
    assert tuple(result.stride()) == (3, 1)


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
        assert (
            result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
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
    assert (
        copied.untyped_storage().data_ptr() != transposed.untyped_storage().data_ptr()
    )
    assert alias.device == source.device == copied.device

    del alias, transposed, source
    torch.testing.assert_close(
        copied.cpu(), torch.tensor([0.0, 3.0, 1.0, 4.0, 2.0, 5.0])
    )


def test_reshape_alias_accepts_compute_stride_noncontiguous_layout(vulkan_backend):
    base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).to(vulkan_backend)
    source = base[:, :, :2]
    result = source.reshape(6, 2)

    assert not source.is_contiguous()
    assert result.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
    assert tuple(result.shape) == (6, 2)
    assert tuple(result.stride()) == (4, 1)


def test_reshape_copy_materializes_readable_zero_stride_layout(vulkan_backend):
    cpu = _source("cpu")
    source = _source(vulkan_backend)
    cpu_overlapping = torch.as_strided(cpu, (2, 3), (0, 1))
    overlapping = torch.as_strided(source, (2, 3), (0, 1))
    pytorch_vulkan._C.reset_execution_counters()

    expected = cpu_overlapping.reshape(6)
    result = overlapping.reshape(6)

    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0
    assert result.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()
    torch.testing.assert_close(result.cpu(), expected)


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


def test_view_trainable_seed_second_reverse_matches_cpu(vulkan_backend):
    base_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    seed_cpu = torch.arange(1, 13, dtype=torch.float32).requires_grad_()
    second_seed_cpu = torch.arange(21, 33, dtype=torch.float32).reshape(3, 4)
    base_vk = base_cpu.detach().to(vulkan_backend).requires_grad_()
    seed_vk = seed_cpu.detach().to(vulkan_backend).requires_grad_()
    second_seed_vk = second_seed_cpu.to(vulkan_backend)
    square_base_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    square_base_vk = square_base_cpu.detach().to(vulkan_backend).requires_grad_()
    square_seed_cpu = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4)
    square_seed_vk = square_seed_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    view_cpu, view_vk = base_cpu.view(12), base_vk.view(12)
    first_cpu = torch.autograd.grad(
        view_cpu, base_cpu, seed_cpu, create_graph=True
    )[0]
    first_vk = torch.autograd.grad(
        view_vk, base_vk, seed_vk, create_graph=True
    )[0]
    second_cpu = torch.autograd.grad(first_cpu, seed_cpu, second_seed_cpu)[0]
    second_vk = torch.autograd.grad(first_vk, seed_vk, second_seed_vk)[0]

    square_cpu = square_base_cpu.view(12)
    square_vk = square_base_vk.view(12)
    loss_cpu = (square_cpu * square_cpu).sum()
    loss_vk = (square_vk * square_vk).sum()
    square_first_cpu = torch.autograd.grad(
        loss_cpu, square_base_cpu, create_graph=True
    )[0]
    square_first_vk = torch.autograd.grad(
        loss_vk, square_base_vk, create_graph=True
    )[0]
    square_second_cpu = torch.autograd.grad(
        square_first_cpu, square_base_cpu, square_seed_cpu
    )[0]
    square_second_vk = torch.autograd.grad(
        square_first_vk, square_base_vk, square_seed_vk
    )[0]

    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0

    assert first_vk.requires_grad
    assert tuple(first_vk.shape) == (3, 4)
    assert tuple(second_vk.shape) == (12,)
    torch.testing.assert_close(first_vk.cpu(), first_cpu)
    torch.testing.assert_close(second_vk.cpu(), second_cpu)
    torch.testing.assert_close(second_vk.cpu(), torch.arange(21, 33, dtype=torch.float32))
    torch.testing.assert_close(square_first_vk.cpu(), 2 * square_base_cpu.detach())
    torch.testing.assert_close(square_second_vk.cpu(), 2 * square_seed_cpu)
    torch.testing.assert_close(square_first_vk.cpu(), square_first_cpu)
    torch.testing.assert_close(square_second_vk.cpu(), square_second_cpu)


def test_stock_reshape_copy_transpose_trainable_seed_second_reverse(vulkan_backend):
    cpu_base = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    cpu_seed = torch.arange(1, 7, dtype=torch.float32).requires_grad_()
    cpu_second_seed = torch.tensor([[11., 12., 13.], [14., 15., 16.]])
    vk_base = cpu_base.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.detach().to(vulkan_backend).requires_grad_()
    vk_second_seed = cpu_second_seed.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_flat = cpu_base.transpose(0, 1).reshape(6)
    vk_flat = vk_base.transpose(0, 1).reshape(6)
    cpu_first = torch.autograd.grad(cpu_flat, cpu_base, cpu_seed, create_graph=True)[0]
    vk_first = torch.autograd.grad(vk_flat, vk_base, vk_seed, create_graph=True)[0]
    cpu_second = torch.autograd.grad(cpu_first, cpu_seed, cpu_second_seed)[0]
    vk_second = torch.autograd.grad(vk_first, vk_seed, vk_second_seed)[0]

    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0
    torch.testing.assert_close(vk_first.cpu(), cpu_first)
    torch.testing.assert_close(cpu_second, torch.tensor([11., 14., 12., 15., 13., 16.]))
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_stock_reshape_copy_offset_narrow_square_second_reverse(vulkan_backend):
    cpu = torch.arange(24, dtype=torch.float32).reshape(4, 6).requires_grad_()
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    second_seed_cpu = torch.arange(1, 25, dtype=torch.float32).reshape(4, 6)
    second_seed_vk = second_seed_cpu.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_view = cpu.transpose(0, 1).narrow(0, 1, 3)
    vk_view = vk.transpose(0, 1).narrow(0, 1, 3)
    cpu_result, vk_result = cpu_view.reshape(12), vk_view.reshape(12)
    assert tuple(vk_view.shape) == (3, 4)
    assert tuple(vk_view.stride()) == (1, 6) and vk_view.storage_offset() == 1
    assert vk_result.untyped_storage().data_ptr() != vk_view.untyped_storage().data_ptr()
    cpu_first = torch.autograd.grad((cpu_result * cpu_result).sum(), cpu, create_graph=True)[0]
    vk_first = torch.autograd.grad((vk_result * vk_result).sum(), vk, create_graph=True)[0]
    cpu_second = torch.autograd.grad(cpu_first, cpu, second_seed_cpu)[0]
    vk_second = torch.autograd.grad(vk_first, vk, second_seed_vk)[0]

    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0
    mask = torch.zeros_like(cpu)
    mask[:, 1:4] = 1
    torch.testing.assert_close(cpu_first, 2 * cpu * mask)
    torch.testing.assert_close(vk_first.cpu(), cpu_first)
    torch.testing.assert_close(cpu_second, 2 * second_seed_cpu * mask)
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_stock_reshape_rank_four_metadata_copy_path(vulkan_backend):
    cpu = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5).requires_grad_()
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    second_seed = torch.arange(1, 121, dtype=torch.float32).reshape(2, 3, 4, 5)
    vk_second_seed = second_seed.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    cpu_flat = cpu.transpose(1, 3).reshape(120)
    vk_view = vk.transpose(1, 3)
    vk_flat = vk_view.reshape(120)
    assert vk_flat.untyped_storage().data_ptr() != vk_view.untyped_storage().data_ptr()
    cpu_grad = torch.autograd.grad((cpu_flat * cpu_flat).sum(), cpu, create_graph=True)[0]
    vk_grad = torch.autograd.grad((vk_flat * vk_flat).sum(), vk, create_graph=True)[0]
    cpu_second = torch.autograd.grad(cpu_grad, cpu, second_seed)[0]
    vk_second = torch.autograd.grad(vk_grad, vk, vk_second_seed)[0]

    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == 0
    assert snapshot[3] == 0
    torch.testing.assert_close(vk_grad.cpu(), cpu_grad)
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_slice_backward_zero_fill_and_trainable_seed_second_reverse(vulkan_backend):
    cpu_base = torch.arange(12, dtype=torch.float32).reshape(3, 4).requires_grad_()
    cpu_seed = torch.arange(1, 7, dtype=torch.float32).reshape(3, 2).requires_grad_()
    cpu_second_seed = torch.arange(21, 33, dtype=torch.float32).reshape(3, 4)
    cpu_narrow_seed = torch.arange(31, 37, dtype=torch.float32).reshape(3, 2)
    vk_base = cpu_base.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.detach().to(vulkan_backend).requires_grad_()
    vk_second_seed = cpu_second_seed.to(vulkan_backend)
    vk_narrow_seed = cpu_narrow_seed.to(vulkan_backend)
    # Exercise both step-one narrow/slice closure and step-two slicing.
    cpu_narrow, vk_narrow = cpu_base.narrow(1, 1, 2), vk_base.narrow(1, 1, 2)
    cpu_even, vk_even = cpu_base[:, ::2], vk_base[:, ::2]
    pytorch_vulkan._C.reset_execution_counters()

    cpu_first = torch.autograd.grad(cpu_even, cpu_base, cpu_seed, create_graph=True)[0]
    vk_first = torch.autograd.grad(vk_even, vk_base, vk_seed, create_graph=True)[0]
    cpu_narrow_first = torch.autograd.grad(
        cpu_narrow, cpu_base, cpu_narrow_seed, create_graph=True
    )[0]
    vk_narrow_first = torch.autograd.grad(
        vk_narrow, vk_base, vk_narrow_seed, create_graph=True
    )[0]
    cpu_second = torch.autograd.grad(cpu_first, cpu_seed, cpu_second_seed)[0]
    vk_second = torch.autograd.grad(vk_first, vk_seed, vk_second_seed)[0]
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0

    assert torch.equal(cpu_first[:, 1::2], torch.zeros_like(cpu_first[:, 1::2]))
    assert torch.equal(cpu_narrow_first[:, :1], torch.zeros_like(cpu_narrow_first[:, :1]))
    assert torch.equal(cpu_narrow_first[:, 3:], torch.zeros_like(cpu_narrow_first[:, 3:]))
    torch.testing.assert_close(vk_first.cpu(), cpu_first)
    torch.testing.assert_close(vk_narrow_first.cpu(), cpu_narrow_first)
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_slice_backward_copy_into_fresh_nonleaf_preserves_history(vulkan_backend):
    cpu_src = torch.arange(1, 5, dtype=torch.float32).requires_grad_()
    cpu_carrier = torch.zeros(6, dtype=torch.float32, requires_grad=True)
    cpu_fresh = cpu_carrier * 2
    cpu_fresh.narrow(0, 1, 4).copy_(cpu_src)
    vk_src = cpu_src.detach().to(vulkan_backend).requires_grad_()
    vk_carrier = cpu_carrier.detach().to(vulkan_backend).requires_grad_()
    vk_fresh = vk_carrier * 2
    pytorch_vulkan._C.reset_execution_counters()
    vk_fresh.narrow(0, 1, 4).copy_(vk_src)
    cpu_grads = torch.autograd.grad(cpu_fresh.sum(), (cpu_src, cpu_carrier))
    vk_grads = torch.autograd.grad(vk_fresh.sum(), (vk_src, vk_carrier))
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0
    assert vk_fresh.grad_fn is not None
    torch.testing.assert_close(cpu_grads[0], torch.ones(4))
    torch.testing.assert_close(cpu_grads[1], torch.tensor([2., 0., 0., 0., 0., 2.]))
    for actual, expected in zip(vk_grads, cpu_grads):
        torch.testing.assert_close(actual.cpu(), expected)


def test_expanded_vector_reshape_materializes_with_graph(vulkan_backend):
    cpu = torch.tensor([2., 3., 4.], dtype=torch.float32, requires_grad=True)
    cpu_seed = torch.tensor([1., 2., 3.])
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = cpu.expand(2, 3).reshape(6)
    vk_result = vk.expand(2, 3).reshape(6)
    cpu_first = torch.autograd.grad((cpu_result * cpu_result).sum(), cpu, create_graph=True)[0]
    vk_first = torch.autograd.grad((vk_result * vk_result).sum(), vk, create_graph=True)[0]
    cpu_second = torch.autograd.grad(cpu_first, cpu, cpu_seed)[0]
    vk_second = torch.autograd.grad(vk_first, vk, vk_seed)[0]
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0
    assert tuple(vk.expand(2, 3).stride()) == (0, 1)
    assert vk_result.untyped_storage().data_ptr() != vk.untyped_storage().data_ptr()
    assert cpu_result.untyped_storage().data_ptr() != cpu.untyped_storage().data_ptr()
    torch.testing.assert_close(vk_first.cpu(), torch.tensor([8., 12., 16.]))
    torch.testing.assert_close(vk_second.cpu(), torch.tensor([4., 8., 12.]))
    torch.testing.assert_close(vk_first.cpu(), cpu_first)
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_expanded_scalar_reshape_matches_cpu_alias_and_derivatives(vulkan_backend):
    cpu = torch.tensor(2., dtype=torch.float32, requires_grad=True)
    cpu_seed = torch.tensor(1.)
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = cpu.expand(2, 3).reshape(6)
    vk_result = vk.expand(2, 3).reshape(6)
    cpu_first = torch.autograd.grad((cpu_result * cpu_result).sum(), cpu, create_graph=True)[0]
    vk_first = torch.autograd.grad((vk_result * vk_result).sum(), vk, create_graph=True)[0]
    cpu_second = torch.autograd.grad(cpu_first, cpu, cpu_seed)[0]
    vk_second = torch.autograd.grad(vk_first, vk, vk_seed)[0]
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0
    assert tuple(cpu_result.stride()) == tuple(vk_result.stride()) == (0,)
    assert cpu_result.untyped_storage().data_ptr() == cpu.untyped_storage().data_ptr()
    assert vk_result.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
    torch.testing.assert_close(vk_first.cpu(), torch.tensor(24.))
    torch.testing.assert_close(vk_second.cpu(), torch.tensor(12.))
    torch.testing.assert_close(vk_first.cpu(), cpu_first)
    torch.testing.assert_close(vk_second.cpu(), cpu_second)


def test_expanded_clone_and_contiguous_keep_history_and_storage_contracts(vulkan_backend):
    cpu = torch.tensor([2., 3., 4.], dtype=torch.float32, requires_grad=True)
    cpu_seed = torch.tensor([1., 2., 3.])
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    vk_seed = cpu_seed.to(vulkan_backend)
    cpu_expanded, vk_expanded = cpu.expand(2, 3), vk.expand(2, 3)
    cpu_transposed, vk_transposed = cpu_expanded.transpose(0, 1), vk_expanded.transpose(0, 1)
    pytorch_vulkan._C.reset_execution_counters()
    results = []
    for cpu_materialized, vk_materialized in (
        (cpu_expanded.clone(), vk_expanded.clone()),
        (cpu_expanded.contiguous(), vk_expanded.contiguous()),
        (cpu_transposed.contiguous(), vk_transposed.contiguous()),
    ):
        cpu_first = torch.autograd.grad((cpu_materialized * cpu_materialized).sum(), cpu,
                                        create_graph=True, retain_graph=True)[0]
        vk_first = torch.autograd.grad((vk_materialized * vk_materialized).sum(), vk,
                                       create_graph=True, retain_graph=True)[0]
        cpu_second = torch.autograd.grad(cpu_first, cpu, cpu_seed, retain_graph=True)[0]
        vk_second = torch.autograd.grad(vk_first, vk, vk_seed, retain_graph=True)[0]
        results.append((cpu_materialized, vk_materialized, cpu_first, vk_first, cpu_second, vk_second))
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0
    for cpu_value, vk_value, cpu_first, vk_first, cpu_second, vk_second in results:
        assert vk_value.untyped_storage().data_ptr() != vk.untyped_storage().data_ptr()
        assert cpu_value.untyped_storage().data_ptr() != cpu.untyped_storage().data_ptr()
        torch.testing.assert_close(vk_first.cpu(), cpu_first)
        torch.testing.assert_close(vk_second.cpu(), cpu_second)
    cpu_contig, vk_contig = cpu.contiguous(), vk.contiguous()
    assert cpu_contig.untyped_storage().data_ptr() == cpu.untyped_storage().data_ptr()
    assert vk_contig.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()


def test_saved_view_mutation_and_copy_alias_contracts_match_cpu(vulkan_backend):
    cpu = torch.arange(6, dtype=torch.float32).requires_grad_()
    vk = cpu.detach().to(vulkan_backend).requires_grad_()
    cpu_view, vk_view = cpu.narrow(0, 1, 4), vk.narrow(0, 1, 4)
    assert cpu_view.untyped_storage().data_ptr() == cpu.untyped_storage().data_ptr()
    assert vk_view.untyped_storage().data_ptr() == vk.untyped_storage().data_ptr()
    cpu_loss, vk_loss = (cpu_view * cpu_view).sum(), (vk_view * vk_view).sum()
    with torch.no_grad():
        cpu.add_(1)
        vk.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation") as cpu_error:
        cpu_loss.backward()
    with pytest.raises(RuntimeError, match="modified by an inplace operation") as vk_error:
        vk_loss.backward()
    assert "version" in str(cpu_error.value).lower()
    assert "version" in str(vk_error.value).lower()

    cpu_source = torch.arange(6, dtype=torch.float32)
    vk_source = cpu_source.to(vulkan_backend)
    cpu_copy, vk_copy = cpu_source.reshape(2, 3).clone(), vk_source.reshape(2, 3).clone()
    cpu_copy[0, 0] = -1
    vk_copy[0, 0] = -1
    assert cpu_copy.untyped_storage().data_ptr() != cpu_source.untyped_storage().data_ptr()
    assert vk_copy.untyped_storage().data_ptr() != vk_source.untyped_storage().data_ptr()
    torch.testing.assert_close(vk_source.cpu(), cpu_source)


def test_invalid_view_and_copy_mutations_reject_before_device_work(vulkan_backend):
    leaf = torch.arange(6, dtype=torch.float32).to(vulkan_backend).requires_grad_()
    independent = torch.ones(6, dtype=torch.float32).to(vulkan_backend)
    overlapping = torch.as_strided(independent, (2, 2), (0, 1))
    overlap_source = torch.ones((2, 2), dtype=torch.float32).to(vulkan_backend)
    noncontiguous = torch.ones((2, 4), dtype=torch.float32).to(vulkan_backend).transpose(0, 1)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="leaf Variable"):
        leaf.copy_(independent)
    with pytest.raises(RuntimeError, match="internal overlap"):
        overlapping.copy_(overlap_source)
    with pytest.raises(RuntimeError, match="not compatible"):
        noncontiguous.view(2, 4)
    with pytest.raises(RuntimeError, match="negative stride"):
        torch.as_strided(independent, (2,), (-1,))
    with pytest.raises(RuntimeError, match="outside its Vulkan allocation"):
        torch.as_strided(independent, (7,), (1,))
    pytorch_vulkan._C.synchronize()
    snapshot = pytorch_vulkan._C.execution_counter_snapshot()
    assert snapshot[2] == snapshot[3] == 0


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
    assert (
        vk_result.untyped_storage().data_ptr() != vk_source.untyped_storage().data_ptr()
    )
    cpu_result.backward(grad)
    vk_result.backward(vk_grad)

    _assert_no_implicit_transfer_or_fallback()
    torch.testing.assert_close(vk.grad.cpu(), cpu.grad, rtol=0, atol=0)
