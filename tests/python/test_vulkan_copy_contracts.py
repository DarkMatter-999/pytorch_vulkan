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
