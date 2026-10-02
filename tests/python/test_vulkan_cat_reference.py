import math

import pytest
import torch


@pytest.mark.parametrize("rank", range(1, 5))
def test_cpu_cat_all_axes_and_negative_equivalents(rank):
    shape = tuple(range(2, rank + 2))
    count = math.prod(shape)
    tensors = [
        torch.arange(count, dtype=torch.float32).reshape(shape),
        torch.arange(count, dtype=torch.float32).reshape(shape) + 1000,
    ]
    for dim in range(rank):
        positive = torch.cat(tensors, dim=dim)
        negative = torch.cat(tensors, dim=dim - rank)
        torch.testing.assert_close(positive, negative, rtol=0, atol=0)
        assert positive.shape[dim] == 2 * shape[dim]


@pytest.mark.parametrize("dim", [4, -5])
def test_cpu_cat_rejects_invalid_rank_four_dimensions(dim):
    with pytest.raises(IndexError):
        torch.cat([torch.ones(1, 2, 3, 4)], dim=dim)


@pytest.mark.parametrize("tensors, dim", [
    ([torch.ones(2, 3), torch.ones(4, 4)], 1),
    ([torch.ones(2, 3), torch.ones(2, 4, 1)], 1),
])
def test_cpu_cat_rejects_incompatible_extents(tensors, dim):
    with pytest.raises(RuntimeError):
        torch.cat(tensors, dim=dim)


def test_cpu_cat_rejects_scalar_and_empty_list():
    with pytest.raises(RuntimeError, match="zero-dimensional"):
        torch.cat([torch.tensor(1.0)])
    with pytest.raises(RuntimeError, match="non-empty list"):
        torch.cat([])


def test_cpu_cat_singleton_allocates_fresh_storage():
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    result = torch.cat([value], dim=0)
    torch.testing.assert_close(result, value, rtol=0, atol=0)
    assert result.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("rank", range(1, 5))
def test_cpu_cat_skips_rank_one_empty_with_rank_one_through_four(rank):
    shape = (2,) * rank
    value = torch.arange(2**rank, dtype=torch.float32).reshape(shape)
    empty = torch.empty((0,))
    for dim in range(rank):
        result = torch.cat([empty, value], dim=dim)
        negative_result = torch.cat([value, empty], dim=dim - rank)
        torch.testing.assert_close(result, value, rtol=0, atol=0)
        torch.testing.assert_close(negative_result, value, rtol=0, atol=0)
        assert tuple(result.shape) == shape
        assert tuple(negative_result.shape) == shape


@pytest.mark.parametrize("dim", range(-6, 7))
def test_cpu_cat_all_skipped_rank_one_empty_accepts_raw_dimension(dim):
    result = torch.cat([torch.empty(0), torch.empty(0)], dim=dim)
    assert tuple(result.shape) == (0,)
    assert tuple(result.stride()) == (1,)


@pytest.mark.parametrize(
    "tensors, dim, expected_shape",
    [
        ([torch.empty(2, 0), torch.ones(2, 3)], 1, (2, 3)),
        ([torch.empty(0, 3), torch.empty(0, 3)], 0, (0, 3)),
        ([torch.empty(2, 0), torch.empty(2, 0)], 1, (2, 0)),
    ],
)
def test_cpu_cat_empty_extents_not_special_rank_one_skip(tensors, dim, expected_shape):
    result = torch.cat(tensors, dim=dim)
    assert tuple(result.shape) == expected_shape


def test_cpu_cat_reads_expanded_zero_stride_inputs():
    base = torch.tensor([[1.0, 2.0, 3.0]])
    expanded = base.expand(4, 3)
    assert expanded.stride() == (0, 1)
    result = torch.cat([expanded, expanded], dim=1)
    expected = torch.empty((4, 6), dtype=base.dtype)
    expected[:, :3] = expanded
    expected[:, 3:] = expanded
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_offset_view_cat_cpu_oracle():
    bases = [torch.arange(n, dtype=torch.float32).reshape(shape)
             for n, shape in ((180, (2, 3, 5, 6)), (120, (2, 2, 5, 6)))]
    views = [x.transpose(2, 3)[:, :, 1:5, 1:4] for x in bases]
    assert [(tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in views] == [
        ((2, 3, 4, 3), (90, 30, 1, 6), 7),
        ((2, 2, 4, 3), (60, 30, 1, 6), 7),
    ]
    y = torch.cat(views, dim=-3)
    assert (tuple(y.shape), tuple(y.stride()), y.storage_offset()) == (
        (2, 5, 4, 3), (60, 12, 3, 1), 0
    )
    assert y.data_ptr() != views[0].data_ptr()


def test_cpu_cat_promotes_mixed_float_dtypes_as_reference_behavior():
    result = torch.cat([torch.ones(2, dtype=torch.float32),
                        torch.ones(3, dtype=torch.float64)])
    assert result.dtype is torch.float64
    assert result.tolist() == [1.0] * 5


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_cpu_cat_channels_last_inputs_preserve_format(dim):
    shape = [2, 3, 4, 5]
    second_shape = shape.copy()
    second_shape[dim] += 1
    first = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    second = (
        torch.arange(math.prod(second_shape), dtype=torch.float32).reshape(second_shape)
        + 1000
    )
    first = first.contiguous(memory_format=torch.channels_last)
    second = second.contiguous(memory_format=torch.channels_last)
    result = torch.cat([first, second], dim=dim)
    expected_shape = list(shape)
    expected_shape[dim] += second_shape[dim]
    expected = torch.empty(
        expected_shape, dtype=torch.float32, memory_format=torch.channels_last
    )
    expected.narrow(dim, 0, shape[dim]).copy_(first)
    expected.narrow(dim, shape[dim], second_shape[dim]).copy_(second)
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", [0, 1, 2, 3])
def test_cpu_cat_mixed_channels_last_and_contiguous_selects_contiguous(dim):
    first_shape = [2, 3, 4, 5]
    second_shape = first_shape.copy()
    second_shape[dim] += 1
    first = torch.arange(math.prod(first_shape), dtype=torch.float32).reshape(first_shape)
    first = first.contiguous(memory_format=torch.channels_last)
    second = (
        torch.arange(math.prod(second_shape), dtype=torch.float32).reshape(second_shape)
        + 1000
    )
    expected_shape = first_shape.copy()
    expected_shape[dim] += second_shape[dim]
    expected = torch.empty(expected_shape, dtype=torch.float32)
    expected.narrow(dim, 0, first_shape[dim]).copy_(first)
    expected.narrow(
        dim, first_shape[dim], second_shape[dim]
    ).copy_(second)
    result = torch.cat([first, second], dim=dim)
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_cpu_cat_singleton_ambiguous_channels_last_uses_suggested_format():
    first = torch.ones((1, 3, 1, 1)).contiguous(memory_format=torch.channels_last)
    second = torch.ones((1, 2, 1, 1)).contiguous(memory_format=torch.channels_last)
    result = torch.cat([first, second], dim=1)
    assert result.stride() == (5, 1, 1, 1)


def test_cpu_cat_skipped_empty_suggestion_changes_channels_last_selection():
    empty = torch.empty((0,))
    first = torch.ones((2, 3, 4, 5)).contiguous(memory_format=torch.channels_last)
    second = torch.ones((2, 2, 4, 5)).contiguous(memory_format=torch.channels_last)
    result = torch.cat([first, empty, second], dim=1)
    assert result.stride() == (100, 20, 5, 1)


def test_cpu_to_memory_format_reference_for_channels_last_cat_inputs():
    value = torch.arange(120, dtype=torch.float32).reshape(2, 3, 4, 5)
    channels_last = value.contiguous(memory_format=torch.channels_last)
    expected = channels_last.stride()
    assert channels_last.to("cpu", copy=True).stride() == expected
    assert channels_last.to("cpu", copy=True,
                            memory_format=torch.preserve_format).stride() == expected
    assert channels_last.to("cpu", copy=True,
                            memory_format=torch.channels_last).stride() == expected
