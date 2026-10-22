import ctypes

import pytest
import pytorch_vulkan
import torch


@pytest.fixture
def vk():
    assert pytorch_vulkan.is_available(), "expected qualified Vulkan device unavailable"
    return "vk"


def test_copy_broadcasts_into_strided_destination(vk):
    source = torch.arange(3, dtype=torch.float32)
    base = torch.full((3, 2), -7.0, device=vk)
    destination = base.transpose(0, 1)

    assert destination.copy_(source) is destination
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(destination.cpu(), source.expand(2, 3), rtol=0, atol=0)
    torch.testing.assert_close(base.cpu(), source.expand(2, 3).t(), rtol=0, atol=0)


class TestPinnedHost:
    """Vulkan host pinning uses the PrivateUse1 fallback only in CPU-only builds.

    PyTorch's allocator priority is CUDA > XPU > PrivateUse1, and
    empty(pin_memory=True) may initialize Vulkan; initialization errors must
    propagate rather than silently falling back to pageable CPU memory.
    """

    @staticmethod
    def _layout_cases():
        base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
        return [
            ("contiguous", base),
            ("transpose", base.t()),
            ("offset_slice", base[1:, 1::2]),
            ("expanded", base[:1, :3].expand(4, 3)),
            ("stride_holes", base[:, ::2]),
            ("zero_size", base[:0, :]),
            ("empty_view_offset", base[2:2, :]),
        ]

    @staticmethod
    def _copy_reference(source):
        expected = torch.empty_strided(
            tuple(source.shape), tuple(source.stride()), dtype=source.dtype
        )
        expected.copy_(source)
        return expected

    def test_cpu_pin_preserves_layout_extent_and_matches_copy_overlap_oracle(self):
        for label, source in self._layout_cases():
            reference_error = None
            try:
                expected = self._copy_reference(source)
            except RuntimeError as error:
                reference_error = error

            if reference_error is not None:
                with pytest.raises(RuntimeError) as actual_error:
                    source.pin_memory(device="vk")
                assert str(actual_error.value) == str(reference_error), label
                continue

            actual = source.pin_memory(device="vk")
            expected_bytes = expected.untyped_storage().nbytes()
            assert actual.device.type == "cpu", label
            assert tuple(actual.size()) == tuple(source.size()), label
            assert tuple(actual.stride()) == tuple(source.stride()), label
            assert actual.storage_offset() == 0, label
            assert actual.untyped_storage().nbytes() == expected_bytes, label
            assert actual.is_pinned(device="vk") is (expected_bytes != 0), label
            assert actual.untyped_storage().resizable() is False, label
            assert torch.equal(actual, expected), label

    def test_pin_preserves_generated_autograd_identity_derivative(self):
        source = torch.tensor([[1.0, -2.0], [3.0, 4.0]], requires_grad=True)
        pinned = source.pin_memory(device="vk")
        assert pinned.requires_grad is True
        assert pinned.grad_fn is not None
        assert "PinMemoryBackward" in type(pinned.grad_fn).__name__
        seed = torch.tensor([[0.25, -3.0], [2.0, 5.5]])
        (gradient,) = torch.autograd.grad(pinned, source, grad_outputs=seed)
        # PyTorch 2.4 derivatives.yaml declares `_pin_memory`'s `self: grad`.
        torch.testing.assert_close(gradient, seed, rtol=0, atol=0)

    def test_vulkan_source_backendselect_guard_and_cpu_then_pin_composition(self, vk):
        source_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        source = source_cpu.to(vk)
        pytorch_vulkan._C.synchronize()
        meta = torch.empty((2,), device="meta")
        with pytest.raises(RuntimeError, match="cannot pin '.*' only dense CPU tensors can be pinned") as meta_error:
            meta.pin_memory()
        assert "cannot pin 'MetaFloatType'" in str(meta_error.value)
        for device in (None, "vk", "cpu"):
            with pytest.raises(RuntimeError, match="cannot pin '.*' only dense CPU tensors can be pinned") as vk_error:
                source.pin_memory() if device is None else source.pin_memory(device=device)
            assert "cannot pin 'vkFloatType'" in str(vk_error.value)
        assert source.is_pinned() is False
        assert source.is_pinned(device="vk") is False

        pinned = source.cpu().pin_memory(device="vk")
        assert pinned.device.type == "cpu"
        assert pinned.is_pinned(device="vk") is True
        assert (tuple(pinned.shape), tuple(pinned.stride()), pinned.storage_offset()) == (
            tuple(source.shape), tuple(source.stride()), 0
        )
        assert pinned.untyped_storage().nbytes() == source_cpu.untyped_storage().nbytes()
        assert torch.equal(pinned, source_cpu)

    def test_pinned_recognition_tracks_storage_provenance_and_views(self):
        pinned = torch.arange(8, dtype=torch.float32).pin_memory(device="vk")
        assert pinned.is_pinned(device="vk") is True
        assert pinned[:0].is_pinned(device="vk") is True
        assert pinned[1:].is_pinned(device="vk") is True
        assert pinned.clone().is_pinned(device="vk") is False
        assert torch.empty((3,)).is_pinned(device="vk") is False

        raw = (ctypes.c_float * pinned.numel()).from_address(pinned.data_ptr())
        foreign = torch.from_numpy(__import__("numpy").ctypeslib.as_array(raw))
        assert torch.equal(foreign, pinned)
        assert foreign.is_pinned(device="vk") is False

        zero = torch.empty((0,), dtype=torch.float32).pin_memory(device="vk")
        assert zero.untyped_storage().nbytes() == 0
        assert zero.is_pinned(device="vk") is False
        assert pinned.contiguous() is pinned
        assert pinned.contiguous().is_pinned(device="vk") is True

    def test_public_defaults_and_global_pinned_factories(self):
        cpu = torch.arange(4, dtype=torch.float32)
        with pytest.raises(NotImplementedError):
            cpu.pin_memory()
        assert cpu.is_pinned() is False
        assert cpu.is_pinned(device="cpu") is False
        with pytest.raises(RuntimeError):
            cpu.pin_memory(device="cpu")

        pinned = torch.empty((5,), dtype=torch.float32, pin_memory=True)
        strided = torch.empty_strided((2, 3), (4, 1), dtype=torch.float32,
                                      pin_memory=True)
        assert pinned.device.type == strided.device.type == "cpu"
        assert pinned.is_pinned(device="vk") is True
        assert strided.is_pinned(device="vk") is True
        assert strided.stride() == (4, 1)

    def test_blocking_copy_surface_and_existing_rejections_are_unchanged(self, vk):
        source = torch.arange(6, dtype=torch.float32)
        pinned = source.pin_memory(device="vk")
        device = pinned.to(vk)
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(device.cpu(), source, rtol=0, atol=0)

        destination = torch.empty((6,), dtype=torch.float32).pin_memory(device="vk")
        destination.copy_(device)
        torch.testing.assert_close(destination, source, rtol=0, atol=0)

        with pytest.raises(RuntimeError, match="non_blocking=True"):
            pinned.to(vk, non_blocking=True)
        with pytest.raises(RuntimeError, match="non_blocking=True"):
            destination.copy_(device, non_blocking=True)
        with pytest.raises(RuntimeError, match="pinned"):
            torch.ops.aten._to_copy.default(
                device, device=torch.device("cpu"), pin_memory=True
            )

        torch.testing.assert_close(pinned + 2.0, source + 2.0, rtol=0, atol=0)
        torch.testing.assert_close(pinned.sum(), source.sum(), rtol=0, atol=0)


def test_set_empty_resets_f32_storage_but_preserves_old_alias(vk):
    base = torch.arange(6, dtype=torch.float32).to(vk)
    alias = base[1:4]
    assert base.set_() is base
    assert tuple(base.shape) == (0,)
    assert base.untyped_storage().nbytes() == 0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(alias.cpu(), torch.tensor([1.0, 2.0, 3.0]))


def test_set_tensor_shares_source_storage_geometry_and_writes(vk):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).to(vk)
    target = torch.empty((1,), dtype=torch.float32, device=vk)
    result = target.set_(source)
    assert result is target
    assert target.untyped_storage()._cdata == source.untyped_storage()._cdata
    assert (tuple(target.shape), tuple(target.stride()), target.storage_offset()) == (
        tuple(source.shape), tuple(source.stride()), source.storage_offset()
    )
    target[0, 0] = -5.0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(source[0, 0].cpu(), torch.tensor(-5.0))
    torch.testing.assert_close(
        target.cpu(),
        torch.tensor([[-5.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0], [8.0, 9.0, 10.0, 11.0]]),
    )


def test_set_tensor_accepts_noncontiguous_source_and_same_impl_noop(vk):
    base_cpu = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    base_vk = base_cpu.to(vk)
    source_cpu = base_cpu.as_strided((2, 3), (1, 4), storage_offset=1)
    source_vk = base_vk.as_strided((2, 3), (1, 4), storage_offset=1)
    target_cpu = torch.empty((1,), dtype=torch.float32)
    target_vk = torch.empty((1,), dtype=torch.float32, device=vk)

    assert target_cpu.set_(source_cpu) is target_cpu
    assert target_vk.set_(source_vk) is target_vk
    for target, source in ((target_cpu, source_cpu), (target_vk, source_vk)):
        assert target.untyped_storage()._cdata == source.untyped_storage()._cdata
        assert (tuple(target.shape), tuple(target.stride()), target.storage_offset()) == (
            tuple(source.shape), tuple(source.stride()), source.storage_offset()
        )

    assert source_cpu.set_(source_cpu) is source_cpu
    assert source_vk.set_(source_vk) is source_vk
    assert (tuple(source_vk.shape), tuple(source_vk.stride()), source_vk.storage_offset()) == (
        (2, 3), (1, 4), 1
    )

    target_cpu[1, 2] = -7.0
    target_vk[1, 2] = -7.0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(target_vk.cpu(), target_cpu)
    torch.testing.assert_close(base_vk.cpu(), base_cpu)


def test_set_tensor_offset_composite_keeps_noncontiguous_rejection(vk):
    source = torch.arange(20, dtype=torch.float32).reshape(4, 5).to(vk).t()
    target = torch.empty((1,), dtype=torch.float32, device=vk)
    before = (target.shape, target.stride(), target.storage_offset(), target.untyped_storage()._cdata)
    with pytest.raises(RuntimeError, match="passed in tensor to be used as storage must be contiguous"):
        target.set_(source, 0, source.shape, source.stride())
    assert (target.shape, target.stride(), target.storage_offset(), target.untyped_storage()._cdata) == before


def test_set_storage_uses_whole_storage_and_defaults(vk):
    source = torch.arange(6, dtype=torch.float32).to(vk)
    target = torch.empty((1,), dtype=torch.float32, device=vk)
    assert target.set_(source.untyped_storage()) is target
    assert target.untyped_storage()._cdata == source.untyped_storage()._cdata
    assert tuple(target.shape) == (6,)
    assert tuple(target.stride()) == (1,)
    target[2] = 17.0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(source[2].cpu(), torch.tensor(17.0))


def test_set_tensor_offset_sizes_strides_adds_source_offset(vk):
    source = torch.arange(12, dtype=torch.float32).to(vk)[2:]
    target = torch.empty((1,), dtype=torch.float32, device=vk)
    assert target.set_(source, 1, (2, 2), (3, 1)) is target
    assert target.untyped_storage()._cdata == source.untyped_storage()._cdata
    assert (tuple(target.shape), tuple(target.stride()), target.storage_offset()) == (
        (2, 2), (3, 1), source.storage_offset() + 1
    )
    target[1, 1] = 29.0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(source[5].cpu(), torch.tensor(29.0))


def test_set_storage_offset_sizes_strides_and_defaults(vk):
    source = torch.arange(8, dtype=torch.float32).to(vk)
    target = torch.empty((1,), dtype=torch.float32, device=vk)
    assert target.set_(source.untyped_storage(), 1, (2, 2), (3, 1)) is target
    assert target.untyped_storage()._cdata == source.untyped_storage()._cdata
    assert (tuple(target.shape), tuple(target.stride()), target.storage_offset()) == (
        (2, 2), (3, 1), 1
    )
    target[1, 1] = 31.0
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(source[5].cpu(), torch.tensor(31.0))


def test_set_storage_growth_errors_and_autograd_match_cpu(vk):
    for device in ("cpu", vk):
        source = torch.arange(2, dtype=torch.float32).to(device)
        target = torch.empty((1,), dtype=torch.float32, device=device)
        storage = source.untyped_storage()
        assert target.set_(storage, 0, (4,), (1,)) is target
        assert storage.nbytes() >= 4 * torch.tensor([], dtype=torch.float32).element_size()
        if device == vk:
            pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(target[:2].cpu(), torch.tensor([0.0, 1.0]))

        before = (target.shape, target.stride(), target.storage_offset(), storage.nbytes())
        with pytest.raises(RuntimeError):
            target.set_(storage, -1, (1,), (1,))
        assert (target.shape, target.stride(), target.storage_offset(), storage.nbytes()) == before

    leaf = torch.ones((2,), dtype=torch.float32, device=vk, requires_grad=True)
    source = torch.ones((2,), dtype=torch.float32, device=vk)
    with torch.enable_grad(), pytest.raises(RuntimeError, match="leaf Variable"):
        leaf.set_(source)
    version = leaf._version
    with torch.no_grad():
        assert leaf.set_(source) is leaf
    assert leaf.requires_grad
    assert leaf._version == version + 1
    pytorch_vulkan._C.synchronize()


def test_resize_growth_preserves_storage_alias(vk):
    base = torch.arange(6, dtype=torch.float32).to(vk)
    alias = base[1:4]
    storage_id = base.untyped_storage()._cdata
    base.resize_(12)
    assert base.untyped_storage()._cdata == storage_id
    assert alias.untyped_storage()._cdata == storage_id
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(alias.cpu(), torch.tensor([1.0, 2.0, 3.0]))
    torch.testing.assert_close(base[:6].cpu(), torch.arange(6, dtype=torch.float32))


def test_resize_out_factories_preserve_output_identity(vk):
    for factory, value in ((torch.full, 2.5), (torch.zeros, 0.0), (torch.ones, 1.0)):
        for shape, initial_shape in (((2, 3), (2,)), ((2, 3), (2, 3)), ((0, 3), (2,))):
            out = torch.full(initial_shape, -4.0, device=vk)
            storage_id = out.untyped_storage()._cdata
            result = factory(shape, value, out=out) if factory is torch.full else factory(shape, out=out)
            assert result is out
            assert out.untyped_storage()._cdata == storage_id
            pytorch_vulkan._C.synchronize()
            torch.testing.assert_close(out.cpu(), torch.full(shape, value), rtol=0, atol=0)


def test_resize_shrink_regrow_and_resize_as_keep_alias_storage(vk):
    base = torch.arange(12, dtype=torch.float32).to(vk)
    alias = base[2:7]
    storage_id = base.untyped_storage()._cdata
    capacity = base.untyped_storage().nbytes()
    base.resize_(4)
    assert base.untyped_storage()._cdata == storage_id
    assert base.untyped_storage().nbytes() == capacity
    base.resize_(12)
    assert base.untyped_storage()._cdata == storage_id
    template = torch.empty((3, 4), dtype=torch.float32)
    assert base.resize_as_(template) is base
    assert base.shape == template.shape
    assert base.untyped_storage()._cdata == storage_id
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(alias.cpu(), torch.arange(2, 7, dtype=torch.float32))


def test_resize_validates_before_mutation_and_rejects_grad(vk):
    tensor = torch.arange(6, dtype=torch.float32).to(vk)
    storage_id = tensor.untyped_storage()._cdata
    shape = tensor.shape
    with pytest.raises(RuntimeError):
        tensor.resize_((-1, 4))
    assert tensor.shape == shape
    assert tensor.untyped_storage()._cdata == storage_id

    requires_grad = torch.ones((2,), dtype=torch.float32, device=vk, requires_grad=True)
    with pytest.raises(RuntimeError, match="cannot resize variables that require grad"):
        requires_grad.resize_((4,))
    with torch.no_grad(), pytest.raises(RuntimeError, match="cannot resize variables that require grad"):
        requires_grad.resize_((4,))

    channels_last_invalid = torch.arange(6, dtype=torch.float32).to(vk)
    before = channels_last_invalid.shape
    with pytest.raises(RuntimeError, match="channels_last memory format requires a 4D tensor"):
        channels_last_invalid.resize_((2, 3), memory_format=torch.channels_last)
    assert channels_last_invalid.shape == before


def test_resize_as_channels_last_and_empty_growth(vk):
    empty = torch.empty((0,), dtype=torch.float32, device=vk)
    empty.resize_((2, 3, 4, 5))
    template = torch.empty((2, 3, 4, 5), dtype=torch.float32).to(memory_format=torch.channels_last)
    assert empty.resize_as_(template, memory_format=torch.preserve_format) is empty
    assert empty.stride() == template.stride()
    pytorch_vulkan._C.synchronize()


def test_resize_growth_preserves_pending_write(vk):
    base = torch.arange(6, dtype=torch.float32).to(vk)
    base.add_(10.0)
    base.resize_(12)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(base[:6].cpu(), torch.arange(6, dtype=torch.float32) + 10.0)


def test_resize_offset_view_preserves_storage_and_prefix(vk):
    base = torch.arange(12, dtype=torch.float32).to(vk)
    view = base.as_strided((3,), (2,), 2)
    storage_id = base.untyped_storage()._cdata
    view.resize_((6,))
    assert view.storage_offset() == 2
    assert view.stride() == (1,)
    assert view.untyped_storage()._cdata == storage_id
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(base[:12].cpu(), torch.arange(12, dtype=torch.float32))


def test_named_resize_matches_native_noop_and_rejections(vk):
    for device in ("cpu", vk):
        named = torch.empty((2,), dtype=torch.float32, device=device, names=("N",))
        assert named.resize_((2,)) is named
        assert named.names == ("N",)
        with pytest.raises(RuntimeError, match="Cannot resize named tensor"):
            named.resize_((3,))
        assert named.shape == (2,)
        with pytest.raises(RuntimeError, match="Unsupported memory format for named tensor resize"):
            named.resize_((2,), memory_format=torch.contiguous_format)
        assert named.shape == (2,)
    pytorch_vulkan._C.synchronize()


def test_resize_deterministic_fill_matches_cpu_tail(vk):
    previous_algorithms = torch.are_deterministic_algorithms_enabled()
    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    try:
        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = True
        for device in ("cpu", vk):
            tensor = torch.arange(6, dtype=torch.float32).to(device)
            tensor.resize_((9,))
            if device == vk:
                pytorch_vulkan._C.synchronize()
            values = tensor.cpu()
            torch.testing.assert_close(values[:6], torch.arange(6, dtype=torch.float32))
            assert torch.isnan(values[6:]).all()
    finally:
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill
        torch.use_deterministic_algorithms(previous_algorithms)


def test_resize_storage_direct_growth_shrink_and_zero(vk):
    for device in ("cpu", vk):
        tensor = torch.arange(6, dtype=torch.float32).to(device)
        storage = tensor.untyped_storage()
        assert storage.nbytes() == 24
        storage.resize_(48)
        assert storage.nbytes() == 48
        torch.testing.assert_close(tensor.cpu(), torch.arange(6, dtype=torch.float32))
        tensor.resize_((2,))
        storage.resize_(8)
        assert storage.nbytes() == 8
        torch.testing.assert_close(tensor.cpu(), torch.tensor([0.0, 1.0]))
        storage.resize_(0)
        assert storage.nbytes() == 0
    pytorch_vulkan._C.synchronize()


def test_resize_boundaries_format_version_and_bad_out(vk):
    for device in ("cpu", vk):
        tensor = torch.arange(6, dtype=torch.float32).to(device)
        capacity = tensor.untyped_storage().nbytes()
        version = tensor._version
        assert tensor.resize_((0,)) is tensor
        assert tensor.untyped_storage().nbytes() == capacity
        assert tensor._version == version + 1

    for shape, memory_format in (
        ((2, 3, 4, 5), torch.channels_last),
        ((2, 3, 4, 5, 6), torch.channels_last_3d),
    ):
        empty = torch.empty((0,), dtype=torch.float32, device=vk)
        empty.resize_(shape, memory_format=memory_format)
        assert empty.is_contiguous(memory_format=memory_format)

    tensor = torch.arange(3, dtype=torch.float32).to(vk)
    before = (tensor.shape, tensor.untyped_storage().nbytes(), tensor._version)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        tensor.resize_((1 << 62,))
    assert (tensor.shape, tensor.untyped_storage().nbytes(), tensor._version) == before
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)

    out = torch.full((2, 3), -7.0, dtype=torch.float32, device=vk)
    before = (out.shape, out.untyped_storage().nbytes(), out._version)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.full((-1, 3), 1.0, out=out)
    assert (out.shape, out.untyped_storage().nbytes(), out._version) == before
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    pytorch_vulkan._C.synchronize()


def test_resize_as_requires_grad_rejects_even_with_no_grad(vk):
    tensor = torch.ones((2,), dtype=torch.float32, device=vk, requires_grad=True)
    template = torch.empty((4,), dtype=torch.float32)
    with torch.no_grad(), pytest.raises(RuntimeError, match="cannot resize variables that require grad"):
        tensor.resize_as_(template)


def test_resize_factory_rejects_bad_out_without_mutation(vk):
    sparse_out = torch.sparse_coo_tensor(
        torch.empty((2, 0), dtype=torch.int64), torch.empty((0,), dtype=torch.float32), (2, 3)
    )
    sparse_before = (sparse_out.shape, sparse_out._nnz())
    with pytest.raises(RuntimeError, match=r"full\(\.\.\.\) is not implemented for sparse layout"):
        torch.full((2, 3), 1.0, out=sparse_out)
    assert (sparse_out.shape, sparse_out._nnz()) == sparse_before

    out = torch.empty((2,), dtype=torch.float64, device=vk)
    before = (out.shape, out.untyped_storage().nbytes(), out._version)
    with pytest.raises(RuntimeError, match="currently supports only float32"):
        torch.zeros((2, 3), out=out)
    assert (out.shape, out.untyped_storage().nbytes(), out._version) == before


def test_copy_rejects_equal_numel_incompatible_shapes_before_work(vk):
    destination = torch.full((2, 3), -4.0, device=vk)
    source = torch.arange(6, dtype=torch.float32).reshape(3, 2)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        destination.copy_(source)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    torch.testing.assert_close(destination.cpu(), torch.full((2, 3), -4.0))


@pytest.mark.parametrize("factory", [torch.zeros_like, torch.ones_like, torch.full_like])
@pytest.mark.parametrize("shape", [(), (0, 3), (2, 3)])
def test_deterministic_like_factories_preserve_shapes(vk, factory, shape):
    source = torch.empty(shape, dtype=torch.float32)
    if factory is torch.full_like:
        actual = factory(source, 2.5, device=vk)
        value = 2.5
    else:
        actual = factory(source, device=vk)
        value = 1 if factory is torch.ones_like else 0
    pytorch_vulkan._C.synchronize()
    assert actual.shape == source.shape
    torch.testing.assert_close(
        actual.cpu(), torch.full(shape, value, dtype=source.dtype), rtol=0, atol=0
    )


def test_preserve_format_to_copy_keeps_dense_transpose_layout(vk):
    source = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    actual = source.to(vk, copy=True)
    assert actual.stride() == source.stride()
    assert actual.untyped_storage().data_ptr() != source.to(vk).untyped_storage().data_ptr()
    torch.testing.assert_close(actual.cpu(), source, rtol=0, atol=0)


def test_ordinary_deterministic_factories_and_fill_out(vk):
    assert torch.empty((2, 3), device=vk).shape == (2, 3)
    strided = torch.empty_strided((2, 3), (1, 2), device=vk)
    assert strided.stride() == (1, 2)
    like = torch.empty_like(strided)
    assert like.stride() == strided.stride()
    for actual, expected in (
        (torch.zeros((2, 3), device=vk), torch.zeros((2, 3))),
        (torch.ones((2, 3), device=vk), torch.ones((2, 3))),
        (torch.full((2, 3), 2.5, device=vk), torch.full((2, 3), 2.5)),
    ):
        pytorch_vulkan._C.synchronize()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    out = torch.empty((2, 3), device=vk)
    assert out.fill_(3.0) is out
    torch.testing.assert_close(out.cpu(), torch.full((2, 3), 3.0), rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
def test_preserve_format_dense_transpose_is_direction_independent(vk, direction):
    source_base_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source_cpu = source_base_cpu.t()
    if direction == "h2d":
        source = source_cpu
        target = vk
    elif direction == "d2h":
        source = source_base_cpu.to(vk).t()
        target = "cpu"
    else:
        source = source_base_cpu.to(vk).t()
        target = vk
    actual = source.to(target, copy=True)
    expected = source_cpu.to("cpu", copy=True)
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
    if direction == "d2d":
        assert actual.untyped_storage().data_ptr() != source.untyped_storage().data_ptr()


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
def test_copy_broadcasts_each_direction_and_rejects_before_work(vk, direction):
    if direction == "h2d":
        source = torch.tensor([2.0, 3.0, 4.0])
        base = torch.full((3, 2), -9.0, device=vk)
    elif direction == "d2h":
        source = torch.tensor([2.0, 3.0, 4.0]).to(vk)
        base = torch.full((3, 2), -9.0)
    else:
        source = torch.tensor([2.0, 3.0, 4.0]).to(vk)
        base = torch.full((3, 2), -9.0, device=vk)
    destination = base.transpose(0, 1)
    assert destination.copy_(source) is destination
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(destination.cpu(), source.cpu().expand(2, 3), rtol=0, atol=0)
    torch.testing.assert_close(base.cpu(), source.cpu().expand(2, 3).t(), rtol=0, atol=0)

    invalid = [
        (torch.ones((2, 3)), (3, 2)),
        (torch.ones((1, 2, 3)), (2, 3)),
        (torch.empty((0, 2)), (0, 3)),
    ]
    for source_cpu, destination_shape in invalid:
        invalid_source = source_cpu if direction == "h2d" else source_cpu.to(vk)
        invalid_base = torch.full(destination_shape, -5.0)
        if direction != "d2h":
            invalid_base = invalid_base.to(vk)
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(RuntimeError):
            invalid_base.copy_(invalid_source)
        assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
        torch.testing.assert_close(invalid_base.cpu(), torch.full(destination_shape, -5.0))


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
@pytest.mark.parametrize("source_shape", [((), (2, 3)), ((2, 1), (2, 3)), ((1,), (0, 3))])
def test_copy_broadcast_scalar_and_empty_shapes(vk, direction, source_shape):
    shape, output_shape = source_shape
    count = 1
    for dimension in shape:
        count *= dimension
    source_cpu = torch.arange(count, dtype=torch.float32).reshape(shape)
    source = source_cpu if direction == "h2d" else source_cpu.to(vk)
    target_device = "cpu" if direction == "d2h" else vk
    destination = torch.full(output_shape, -6.0, dtype=torch.float32, device=target_device)
    assert destination.copy_(source) is destination
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(destination.cpu(), source_cpu.expand(output_shape), rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
def test_broadcast_copy_preserves_gapped_destination_holes(vk, direction):
    source_cpu = torch.tensor([8.0, 9.0, 10.0])
    source = source_cpu if direction == "h2d" else source_cpu.to(vk)
    device = "cpu" if direction == "d2h" else vk
    base = torch.full((12,), -11.0, device=device)
    destination = torch.as_strided(base, (2, 3), (5, 1), storage_offset=1)
    destination.copy_(source)
    pytorch_vulkan._C.synchronize()
    expected = torch.full((12,), -11.0)
    expected_view = torch.as_strided(expected, (2, 3), (5, 1), storage_offset=1)
    expected_view.copy_(source_cpu.expand(2, 3))
    torch.testing.assert_close(base.cpu(), expected, rtol=0, atol=0)


def test_exact_self_copy_is_noop_and_expanded_h2d_destination_rejects(vk):
    tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=vk)
    pytorch_vulkan._C.reset_execution_counters()
    assert tensor.copy_(tensor) is tensor
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)

    base = torch.full((1, 3), -2.0, device=vk)
    destination = base.expand(2, 3)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="internal overlap"):
        destination.copy_(torch.ones((2, 3)))
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    torch.testing.assert_close(base.cpu(), torch.full((1, 3), -2.0))


def test_non_f32_copy_does_not_gain_broadcast_support(vk):
    destination = torch.empty((2, 3), dtype=torch.bool, device=vk)
    source = torch.ones((3,), dtype=torch.bool)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="matching sizes"):
        destination.copy_(source)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_non_f32_to_copy_layout_policy_remains_contiguous(vk):
    source = torch.tensor([True, False] * 6).reshape(3, 4).t()
    actual = source.to(vk)
    assert actual.is_contiguous()


def test_copy_from_formatter_route_normalizes_f32_broadcast(vk):
    source = torch.arange(3, dtype=torch.float32).to(vk)
    destination = torch.empty((2, 3), dtype=torch.float32)
    result = torch.ops.aten._copy_from.default(source, destination, False)
    torch.testing.assert_close(result, torch.arange(3, dtype=torch.float32).expand(2, 3), rtol=0, atol=0)

    invalid = torch.arange(6, dtype=torch.float32).reshape(3, 2).to(vk)
    sentinel = torch.full((2, 3), -3.0)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.ops.aten._copy_from.default(invalid, sentinel, False)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)
    torch.testing.assert_close(sentinel, torch.full((2, 3), -3.0))


def test_copy_from_and_resize_populates_empty_cpu_destination(vk):
    source_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    source_cpu = source_cpu.as_strided((2, 3), (6, 2), 1)
    source = source_cpu.to(vk)
    destination = torch.empty((0,), dtype=torch.float32)

    expected = torch.empty((0,), dtype=torch.float32)
    expected.resize_as_(source_cpu)
    expected.copy_(source_cpu)
    result = torch.ops.aten._copy_from_and_resize.default(source, destination)

    assert result is destination
    assert tuple(destination.shape) == tuple(expected.shape)
    assert tuple(destination.stride()) == tuple(expected.stride())
    torch.testing.assert_close(destination, expected, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["cpu_to_vk", "vk_to_vk", "vk_to_cpu"])
def test_copy_from_and_resize_empty_destination_direction_matrix(vk, direction):
    source_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = source_cpu if direction == "cpu_to_vk" else source_cpu.to(vk)
    target_device = "cpu" if direction == "vk_to_cpu" else vk
    destination = torch.empty((0,), dtype=torch.float32, device=target_device)

    expected = torch.empty((0,), dtype=torch.float32)
    expected.resize_as_(source_cpu)
    expected.copy_(source_cpu)
    result = torch.ops.aten._copy_from_and_resize.default(source, destination)
    pytorch_vulkan._C.synchronize()

    assert result is destination
    assert tuple(destination.shape) == tuple(expected.shape)
    assert tuple(destination.stride()) == tuple(expected.stride())
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)


def test_copy_from_and_resize_nonempty_destination_broadcasts_without_resize(vk):
    source = torch.tensor([2.0, 3.0, 4.0])
    destination = torch.full((2, 3), -5.0, device=vk)
    original_storage = destination.untyped_storage()._cdata
    expected = torch.full((2, 3), -5.0)
    expected.copy_(source)

    result = torch.ops.aten._copy_from_and_resize.default(source, destination)
    pytorch_vulkan._C.synchronize()

    assert result is destination
    assert destination.untyped_storage()._cdata == original_storage
    assert tuple(destination.shape) == (2, 3)
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)


def test_copy_from_and_resize_rejects_incompatible_nonempty_shape_without_mutation(vk):
    source = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    destination = torch.full((2, 3), -7.0, device=vk)
    before = (tuple(destination.shape), tuple(destination.stride()),
              destination.storage_offset(), destination.untyped_storage()._cdata,
              destination.untyped_storage().nbytes(), destination._version)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError):
        torch.ops.aten._copy_from_and_resize.default(source, destination)
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    pytorch_vulkan._C.synchronize()

    assert counters == (0, 0, 0, 0)
    assert (tuple(destination.shape), tuple(destination.stride()),
            destination.storage_offset(), destination.untyped_storage()._cdata,
            destination.untyped_storage().nbytes(), destination._version) == before
    torch.testing.assert_close(destination.cpu(), torch.full((2, 3), -7.0))


def test_copy_from_and_resize_empty_source_and_exact_self_are_noops(vk):
    destination = torch.tensor([[1.0, 2.0], [3.0, 4.0]], device=vk)
    empty_source = torch.empty((0, 3), dtype=torch.float32)
    empty_before = (tuple(destination.shape), tuple(destination.stride()),
                    destination.storage_offset(), destination.untyped_storage()._cdata,
                    destination.untyped_storage().nbytes(), destination._version)
    pytorch_vulkan._C.reset_execution_counters()
    empty_result = torch.ops.aten._copy_from_and_resize.default(empty_source, destination)
    empty_counters = pytorch_vulkan._C.execution_counter_snapshot()

    self_before = (tuple(destination.shape), tuple(destination.stride()),
                   destination.storage_offset(), destination.untyped_storage()._cdata,
                   destination.untyped_storage().nbytes(), destination._version)
    pytorch_vulkan._C.reset_execution_counters()
    self_result = torch.ops.aten._copy_from_and_resize.default(destination, destination)
    self_counters = pytorch_vulkan._C.execution_counter_snapshot()
    pytorch_vulkan._C.synchronize()

    assert empty_result is destination
    assert empty_counters == (0, 0, 0, 0)
    assert (tuple(destination.shape), tuple(destination.stride()),
            destination.storage_offset(), destination.untyped_storage()._cdata,
            destination.untyped_storage().nbytes(), destination._version) == empty_before
    assert self_result is destination
    assert self_counters == (0, 0, 0, 0)
    assert (tuple(destination.shape), tuple(destination.stride()),
            destination.storage_offset(), destination.untyped_storage()._cdata,
            destination.untyped_storage().nbytes(), destination._version) == self_before
    torch.testing.assert_close(destination.cpu(), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_copy_from_and_resize_rejects_nonf32_source_before_mutating_destination(vk):
    source = torch.tensor([True, False])
    destination = torch.empty((0,), dtype=torch.float32, device=vk)
    before = (tuple(destination.shape), tuple(destination.stride()),
              destination.storage_offset(), destination.untyped_storage()._cdata,
              destination.untyped_storage().nbytes(), destination._version)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(RuntimeError, match="only float32"):
        torch.ops.aten._copy_from_and_resize.default(source, destination)
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    pytorch_vulkan._C.synchronize()

    assert counters == (0, 0, 0, 0)
    assert (tuple(destination.shape), tuple(destination.stride()),
            destination.storage_offset(), destination.untyped_storage()._cdata,
            destination.untyped_storage().nbytes(), destination._version) == before


def test_copy_from_and_resize_out_populates_empty_cpu_out(vk):
    source_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    source = source_cpu.to(vk)
    destination = torch.empty((0,), dtype=torch.float32)
    out = torch.empty((0,), dtype=torch.float32)

    expected = torch.empty((0,), dtype=torch.float32)
    expected.resize_as_(source_cpu)
    expected.copy_(source_cpu)
    result = torch.ops.aten._copy_from_and_resize.out(source, destination, out=out)
    pytorch_vulkan._C.synchronize()

    assert result is out
    assert result is not destination
    assert tuple(destination.shape) == tuple(expected.shape)
    assert tuple(out.shape) == tuple(expected.shape)
    assert tuple(out.stride()) == tuple(expected.stride())
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_copy_from_and_resize_out_resizes_mismatched_preexisting_out(vk):
    source_cpu = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    source = source_cpu.to(vk)
    destination = torch.empty((0,), dtype=torch.float32)
    out = torch.full((1,), -9.0, dtype=torch.float32)

    expected = torch.empty((0,), dtype=torch.float32)
    expected.resize_as_(source_cpu)
    expected.copy_(source_cpu)
    result = torch.ops.aten._copy_from_and_resize.out(source, destination, out=out)
    pytorch_vulkan._C.synchronize()

    assert result is out
    assert tuple(destination.shape) == tuple(expected.shape)
    assert tuple(out.shape) == tuple(expected.shape)
    assert tuple(out.stride()) == tuple(expected.stride())
    torch.testing.assert_close(destination.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)


def test_copy_from_and_resize_out_copies_unchanged_destination_for_empty_source(vk):
    source = torch.empty((0, 3), dtype=torch.float32, device=vk)
    destination = torch.tensor([[4.0, 5.0], [6.0, 7.0]])
    out = torch.empty((0,), dtype=torch.float32)
    before = (tuple(destination.shape), tuple(destination.stride()),
              destination.storage_offset(), destination.untyped_storage()._cdata,
              destination.untyped_storage().nbytes(), destination._version)
    pytorch_vulkan._C.reset_execution_counters()

    result = torch.ops.aten._copy_from_and_resize.out(source, destination, out=out)
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    pytorch_vulkan._C.synchronize()

    assert result is out
    assert counters == (0, 0, 0, 0)
    assert (tuple(destination.shape), tuple(destination.stride()),
            destination.storage_offset(), destination.untyped_storage()._cdata,
            destination.untyped_storage().nbytes(), destination._version) == before
    assert tuple(out.shape) == tuple(destination.shape)
    torch.testing.assert_close(destination, torch.tensor([[4.0, 5.0], [6.0, 7.0]]))
    torch.testing.assert_close(out, destination)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
@pytest.mark.parametrize("memory_format", [None, torch.preserve_format, torch.contiguous_format])
def test_to_copy_formats_for_dense_views(vk, direction, memory_format):
    base_cpu = torch.arange(30, dtype=torch.float32).reshape(5, 6)
    base_vk = base_cpu.to(vk)
    cases_cpu = [
        base_cpu.t(),
        base_cpu.as_strided((2, 2), (6, 2), 1),
        base_cpu[:1, :].expand(3, 6),
        torch.arange(0, dtype=torch.float32).reshape(0, 3),
        torch.tensor(7.0),
        torch.arange(48, dtype=torch.float32).reshape(2, 3, 2, 4).to(memory_format=torch.channels_last),
        torch.arange(144, dtype=torch.float32).reshape(2, 3, 2, 3, 4).to(memory_format=torch.channels_last_3d),
    ]
    cases_vk = [
        base_vk.t(),
        base_vk.as_strided((2, 2), (6, 2), 1),
        base_vk[:1, :].expand(3, 6),
        torch.empty((0, 3), dtype=torch.float32, device=vk),
        torch.tensor(7.0).to(vk),
        cases_cpu[5].to(vk),
        cases_cpu[6].to(vk),
    ]
    for case_cpu, case_vk in zip(cases_cpu, cases_vk):
        source = case_cpu if direction == "h2d" else case_vk
        target = "cpu" if direction == "d2h" else vk
        kwargs = {} if memory_format is None else {"memory_format": memory_format}
        actual = source.to(target, copy=True, **kwargs)
        expected = case_cpu.to("cpu", copy=True, **kwargs)
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
@pytest.mark.parametrize("rank, memory_format", [(4, torch.channels_last), (5, torch.channels_last_3d)])
def test_to_copy_channels_last_formats(vk, direction, rank, memory_format):
    shape = (2, 3, 2, 4) if rank == 4 else (2, 3, 2, 3, 4)
    base = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    source_cpu = base.to(memory_format=memory_format)
    source = source_cpu if direction == "h2d" else source_cpu.to(vk)
    target = "cpu" if direction == "d2h" else vk
    actual = source.to(target, copy=True, memory_format=memory_format)
    expected = source_cpu.to("cpu", copy=True, memory_format=memory_format)
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
@pytest.mark.parametrize(
    "shape,memory_format",
    [((2, 3, 4), torch.channels_last), ((2, 3, 4, 5), torch.channels_last_3d)],
)
def test_invalid_channels_last_rank_fails_before_device_work(vk, direction, shape, memory_format):
    source = torch.ones(shape)
    if direction != "h2d":
        source = source.to(vk)
    target = "cpu" if direction == "d2h" else vk
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        source.to(target, copy=True, memory_format=memory_format)
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize("direction", ["h2d", "d2h"])
def test_f32_to_copy_rejects_pinned_memory_before_work(vk, direction):
    source = torch.arange(3, dtype=torch.float32)
    if direction == "d2h":
        source = source.to(vk)
        target = "cpu"
    else:
        target = vk
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="pinned"):
        torch.ops.aten._to_copy.default(
            source, device=torch.device(target), pin_memory=True
        )
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize("direction", ["h2d", "d2h", "d2d"])
def test_f32_copy_preserves_raw_bits_including_gather(vk, direction):
    signed_bits = torch.tensor(
        [0, -2147483648, 1, -2147483647, 0x7FC12345, -8388608], dtype=torch.int32
    )
    source_cpu = signed_bits.view(torch.float32)
    if direction == "h2d":
        source = source_cpu
        target = vk
    elif direction == "d2h":
        source = source_cpu.to(vk)
        target = "cpu"
    else:
        source = source_cpu.to(vk)
        target = vk
    if direction == "d2d":
        source = source.reshape(2, 3).t()
        destination = torch.empty((3, 2), device=vk)
        expected = source_cpu.reshape(2, 3).t()
    else:
        destination = torch.empty(source.shape, device=target)
        expected = source_cpu
    destination.copy_(source)
    pytorch_vulkan._C.synchronize()
    assert torch.equal(destination.cpu().view(torch.int32), expected.view(torch.int32))


@pytest.mark.parametrize("memory_format", [torch.channels_last, torch.channels_last_3d])
def test_initialized_like_factories_preserve_channels_last_values(vk, memory_format):
    shape = (1, 2, 2, 3) if memory_format is torch.channels_last else (1, 2, 2, 2, 3)
    source = torch.empty(shape).to(memory_format=memory_format)
    for factory, value in ((torch.zeros_like, 0.0), (torch.ones_like, 1.0),
                           (lambda t, **kw: torch.full_like(t, 2.5, **kw), 2.5)):
        actual = factory(source, device=vk, memory_format=memory_format)
        pytorch_vulkan._C.synchronize()
        expected = torch.full(shape, value).to(memory_format=memory_format)
        assert actual.stride() == expected.stride()
        torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["gapped", "expanded"])
@pytest.mark.parametrize("factory", [torch.zeros_like, torch.ones_like])
def test_initialized_like_factories_match_gapped_and_expanded_cpu_layout(vk, layout, factory):
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    source = base.as_strided((2, 2), (4, 2), 1) if layout == "gapped" else base[:1, :].expand(3, 4)
    actual = factory(source, device=vk, memory_format=torch.preserve_format)
    expected = factory(source, device="cpu", memory_format=torch.preserve_format)
    pytorch_vulkan._C.synchronize()
    assert actual.stride() == expected.stride()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
