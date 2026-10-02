import pytest
import pytorch_vulkan
import torch


def test_functional_cat_dispatch_and_values():
    a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    b = torch.arange(8, dtype=torch.float32).reshape(2, 4) + 20
    av, bv = a.to("vk:0"), b.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    y = torch.cat([av, bv], dim=-1)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    assert y.device == av.device
    assert y.device.type == torch._C._get_privateuse1_backend_name()
    assert y.grad_fn is None
    assert y.data_ptr() != av.data_ptr()
    torch.testing.assert_close(y.cpu(), torch.cat([a, b], dim=1))


def test_functional_cat_accepts_expanded_zero_stride_input():
    base = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    expanded = base.to("vk:0").expand(4, 3)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.cat([expanded, expanded], dim=1)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    assert result.device.type == torch._C._get_privateuse1_backend_name()
    expected = torch.cat([base.expand(4, 3), base.expand(4, 3)], dim=1)
    torch.testing.assert_close(result.cpu(), expected)


@pytest.mark.parametrize("rank", range(1, 5))
def test_functional_cat_all_axes_negative_dimensions_and_layouts(rank):
    import math

    shape = tuple(range(2, rank + 2))
    first_cpu = torch.arange(math.prod(shape), dtype=torch.float32).reshape(shape)
    pairs = []
    for dim in range(rank):
        second_shape = list(shape)
        second_shape[dim] += 1
        second_cpu = torch.arange(math.prod(second_shape), dtype=torch.float32).reshape(second_shape) + 100
        pairs.append((first_cpu, second_cpu, first_cpu.to("vk:0"), second_cpu.to("vk:0"), dim))
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    results = []
    for cpu_first, cpu_second, first, second, dim in pairs:
        result = torch.cat([first, second], dim=dim)
        negative = torch.cat([first, second], dim=dim - rank)
        results.append((result, negative, torch.cat([cpu_first, cpu_second], dim=dim)))
        pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] == 4 * rank
    assert counters[1:] == (0, 0, 0)
    for result, negative, expected in results:
        torch.testing.assert_close(result.cpu(), expected)
        torch.testing.assert_close(negative.cpu(), expected)


def test_functional_cat_offset_views_preserve_storage_geometry():
    bases_cpu = [torch.arange(n, dtype=torch.float32).reshape(shape).requires_grad_()
                 for n, shape in ((180, (2, 3, 5, 6)), (120, (2, 2, 5, 6)))]
    bases = [x.detach().to("vk:0").requires_grad_() for x in bases_cpu]
    views = [x.transpose(2, 3).narrow(2, 1, 4).narrow(3, 1, 3) for x in bases]
    cpu_views = [x.transpose(2, 3).narrow(2, 1, 4).narrow(3, 1, 3) for x in bases_cpu]
    seed_cpu = torch.arange(1, 121, dtype=torch.float32).reshape(2, 5, 4, 3)
    seed = seed_cpu.to("vk:0")
    assert [(tuple(x.shape), tuple(x.stride()), x.storage_offset()) for x in views] == [
        ((2, 3, 4, 3), (90, 30, 1, 6), 7),
        ((2, 2, 4, 3), (60, 30, 1, 6), 7),
    ]
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.cat(views, dim=-3)
    expected = torch.cat(cpu_views, dim=1)
    grads = torch.autograd.grad(result, views, seed)
    cpu_grads = torch.autograd.grad(expected, cpu_views, seed_cpu)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    assert tuple(result.stride()) == (60, 12, 3, 1)
    torch.testing.assert_close(result.cpu(), expected)
    for actual, reference in zip(grads, cpu_grads):
        torch.testing.assert_close(actual.cpu(), reference)
        assert (tuple(actual.shape), tuple(actual.stride()), actual.storage_offset()) == (
            tuple(reference.shape), tuple(reference.stride()), reference.storage_offset()
        )


def test_functional_cat_trainable_rank_one_empty_and_unused_input_match_cpu_none():
    cpu_empty = torch.empty((0,), dtype=torch.float32, requires_grad=True)
    cpu_used = torch.arange(1, 5, dtype=torch.float32, requires_grad=True)
    cpu_unused = torch.tensor([9.0], dtype=torch.float32, requires_grad=True)
    empty = cpu_empty.detach().to("vk:0").requires_grad_()
    used = cpu_used.detach().to("vk:0").requires_grad_()
    unused = cpu_unused.detach().to("vk:0").requires_grad_()
    seed_cpu = torch.arange(1, 5, dtype=torch.float32)
    seed = seed_cpu.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = torch.cat([cpu_empty, cpu_used])
    result = torch.cat([empty, used])
    cpu_empty_grad, cpu_used_grad, cpu_unused_grad = torch.autograd.grad(
        cpu_result, (cpu_empty, cpu_used, cpu_unused), seed_cpu, allow_unused=True
    )
    empty_grad, used_grad, unused_grad = torch.autograd.grad(
        result, (empty, used, unused), seed, allow_unused=True
    )
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    assert cpu_unused_grad is None and unused_grad is None
    assert (tuple(empty_grad.shape), tuple(empty_grad.stride()), empty_grad.storage_offset()) == (
        tuple(cpu_empty_grad.shape), tuple(cpu_empty_grad.stride()), cpu_empty_grad.storage_offset()
    )
    assert tuple(used_grad.stride()) == tuple(cpu_used_grad.stride())
    assert used_grad.storage_offset() == cpu_used_grad.storage_offset()
    torch.testing.assert_close(used_grad.cpu(), cpu_used_grad)


@pytest.mark.parametrize(
    "shape,dim",
    [((2, 0), 1), ((2, 0, 3, 4), 1)],
)
@pytest.mark.parametrize("empty_first", [True, False])
def test_functional_cat_non_skipped_empty_leading_or_trailing_segment(
    shape, dim, empty_first
):
    empty_cpu = torch.empty(shape, dtype=torch.float32, requires_grad=True)
    value_shape = list(shape)
    value_shape[dim] = 2 if len(shape) == 4 else 3
    value_cpu = torch.arange(1, torch.tensor(value_shape).prod().item() + 1,
                             dtype=torch.float32).reshape(value_shape).requires_grad_()
    empty_vk = empty_cpu.detach().to("vk:0").requires_grad_()
    value_vk = value_cpu.detach().to("vk:0").requires_grad_()
    cpu_inputs = [empty_cpu, value_cpu] if empty_first else [value_cpu, empty_cpu]
    vk_inputs = [empty_vk, value_vk] if empty_first else [value_vk, empty_vk]
    seed_cpu = torch.arange(1, value_cpu.numel() + 1, dtype=torch.float32).reshape(value_shape)
    seed_vk = seed_cpu.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    expected = torch.cat(cpu_inputs, dim=dim)
    result = torch.cat(vk_inputs, dim=dim)
    cpu_grads = torch.autograd.grad(expected, cpu_inputs, seed_cpu)
    grads = torch.autograd.grad(result, vk_inputs, seed_vk)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    assert tuple(result.shape) == tuple(expected.shape)
    assert tuple(result.stride()) == tuple(expected.stride())
    torch.testing.assert_close(result.cpu(), expected)
    for actual, reference in zip(grads, cpu_grads):
        assert (tuple(actual.shape), tuple(actual.stride()), actual.storage_offset()) == (
            tuple(reference.shape), tuple(reference.stride()), reference.storage_offset()
        )
        torch.testing.assert_close(actual.cpu(), reference)


def test_functional_cat_native_generated_vjp_and_hvp():
    cpu_a = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).requires_grad_()
    cpu_b = torch.arange(11, 19, dtype=torch.float32).reshape(2, 4).requires_grad_()
    a = cpu_a.detach().to("vk:0").requires_grad_()
    b = cpu_b.detach().to("vk:0").requires_grad_()
    da = torch.arange(1, 7, dtype=torch.float32).reshape(2, 3)
    db = torch.arange(7, 15, dtype=torch.float32).reshape(2, 4)
    da_vk, db_vk = da.to("vk:0"), db.to("vk:0")
    seed_cpu = torch.arange(1, 15, dtype=torch.float32).reshape(2, 7).requires_grad_()
    seed = seed_cpu.detach().to("vk:0").requires_grad_()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_y, y = torch.cat([cpu_a, cpu_b], 1), torch.cat([a, b], 1)
    cpu_ga, cpu_gb = torch.autograd.grad((cpu_y * cpu_y).sum(), (cpu_a, cpu_b), create_graph=True)
    ga, gb = torch.autograd.grad((y * y).sum(), (a, b), create_graph=True)
    cpu_ha, cpu_hb = torch.autograd.grad((cpu_ga * da).sum() + (cpu_gb * db).sum(), (cpu_a, cpu_b))
    ha, hb = torch.autograd.grad((ga * da_vk).sum() + (gb * db_vk).sum(), (a, b))
    # Rebuild independent graphs for the trainable-seed direction after HVP.
    cpu_seed_y, seed_y = torch.cat([cpu_a, cpu_b], 1), torch.cat([a, b], 1)
    cpu_seed_a, cpu_seed_b = torch.autograd.grad(
        cpu_seed_y, (cpu_a, cpu_b), seed_cpu, create_graph=True)
    seed_a, seed_b = torch.autograd.grad(seed_y, (a, b), seed, create_graph=True)
    cpu_gseed = torch.autograd.grad((cpu_seed_a * da).sum() + (cpu_seed_b * db).sum(), seed_cpu)[0]
    gseed = torch.autograd.grad((seed_a * da_vk).sum() + (seed_b * db_vk).sum(), seed)[0]
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[2:] == (0, 0)
    for actual, expected in ((ga, cpu_ga), (gb, cpu_gb)):
        torch.testing.assert_close(actual.cpu(), expected)
    for actual, expected in ((ha, cpu_ha), (hb, cpu_hb), (gseed, cpu_gseed)):
        torch.testing.assert_close(actual.cpu(), expected)


def test_functional_cat_native_crop_dependency_vjp():
    import torch.nn.functional as F

    generator = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn((1, 4, 6, 7), dtype=torch.float32, generator=generator)
    weight = torch.randn((6, 2, 3, 2), dtype=torch.float32, generator=generator)
    grad_output = torch.randn((1, 6, 3, 5), dtype=torch.float32, generator=generator)
    grad_input = torch.randn((1, 4, 6, 7), dtype=torch.float32, generator=generator)
    native_y = F.conv2d(x, weight, stride=(2, 1), padding=(1, 0),
                        dilation=(1, 2), groups=2)
    assert tuple(native_y.shape) == (1, 6, 3, 5)
    # Recreate the exact two ordinary CPU grouped-backward intermediates from
    # the source trace. The Vulkan portion starts only after these CPU leaves.
    cpu_groups = []
    for group in range(2):
        transformed_input = grad_input[:, group * 2:(group + 1) * 2].permute(1, 0, 2, 3)
        transformed_grad_output = grad_output[:, group * 3:(group + 1) * 3].permute(1, 0, 2, 3)
        assert tuple(transformed_input.shape) == (2, 1, 6, 7)
        assert tuple(transformed_grad_output.shape) == (3, 1, 3, 5)
        value = F.conv2d(transformed_input, transformed_grad_output,
                         stride=(1, 2), padding=(1, 0), dilation=(2, 1))
        assert tuple(value.shape) == (2, 3, 4, 2)
        assert tuple(value.stride()) == (24, 8, 2, 1)
        cpu_groups.append(value.detach().requires_grad_())
    a, b = [value.detach().to("vk:0").requires_grad_() for value in cpu_groups]
    seed_cpu = torch.arange(1, 73, dtype=torch.float32).reshape(6, 2, 3, 2)
    seed = seed_cpu.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_cat = torch.cat(cpu_groups, 1)
    cat_value = torch.cat([a, b], 1)
    assert tuple(cpu_cat.shape) == tuple(cat_value.shape) == (2, 6, 4, 2)
    cpu_y = cpu_cat.transpose(0, 1).narrow(2, 0, 3)
    y = cat_value.transpose(0, 1).narrow(2, 0, 3)
    assert tuple(cpu_y.shape) == tuple(y.shape) == (6, 2, 3, 2)
    cpu_ga, cpu_gb = torch.autograd.grad(cpu_y, cpu_groups, seed_cpu)
    ga, gb = torch.autograd.grad(y, (a, b), seed)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[0] >= 2
    assert counters[2:] == (0, 0)
    torch.testing.assert_close(y.cpu(), cpu_y)
    torch.testing.assert_close(ga.cpu(), cpu_ga)
    torch.testing.assert_close(gb.cpu(), cpu_gb)
    assert torch.count_nonzero(cpu_ga[:, :, 3, :]) == 0
    assert torch.count_nonzero(cpu_gb[:, :, 3, :]) == 0


def test_functional_cat_dispatch_count_scales_with_inputs_not_size():
    submissions = []
    for rows in (2, 16):
        for offset_view in (False, True):
            if offset_view:
                cpu_bases = [torch.arange(2 * (rows + 2) * 12, dtype=torch.float32)
                             .reshape(2, rows + 2, 4, 3) + i * 1000 for i in range(2)]
                cpu_inputs = [x.transpose(2, 3).narrow(1, 1, rows) for x in cpu_bases]
                vk_bases = [x.to("vk:0") for x in cpu_bases]
                vk_inputs = [x.transpose(2, 3).narrow(1, 1, rows) for x in vk_bases]
                assert all(x.storage_offset() > 0 for x in vk_inputs)
            else:
                cpu_inputs = [torch.arange(2 * rows * 12, dtype=torch.float32)
                              .reshape(2, rows, 3, 4) + i * 1000 for i in range(2)]
                vk_inputs = [x.to("vk:0") for x in cpu_inputs]
            pytorch_vulkan._C.synchronize()
            pytorch_vulkan._C.reset_execution_counters()
            result = torch.cat(vk_inputs, dim=1)
            pytorch_vulkan._C.synchronize()
            counters = pytorch_vulkan._C.execution_counter_snapshot()
            submitted = pytorch_vulkan._C.compute_submitted_count()
            assert counters == (2, 0, 0, 0)
            assert submitted <= counters[0]
            torch.testing.assert_close(result.cpu(), torch.cat(cpu_inputs, dim=1))
            submissions.append(submitted)
    assert len(set(submissions)) == 1


def test_functional_cat_errors_dtype_and_empty_rules():
    one = torch.tensor([1.0, 2.0], dtype=torch.float32).to("vk:0")
    unsupported = torch.tensor([4, 5], dtype=torch.int64).to("vk:0")
    scalar_vk = torch.tensor(7.0, dtype=torch.float32).to("vk:0")
    empty = torch.empty((0,), dtype=torch.float32).to("vk:0")
    mismatch_a = torch.ones(2, 3).to("vk:0")
    mismatch_b = torch.ones(4, 3).to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.cat([])
    with pytest.raises(RuntimeError):
        torch.cat([torch.ones(())])
    with pytest.raises(RuntimeError, match="zero-dimensional"):
        torch.cat([scalar_vk])
    with pytest.raises(IndexError):
        torch.cat([one], dim=1)
    with pytest.raises(RuntimeError):
        torch.cat([mismatch_a, mismatch_b], dim=1)
    with pytest.raises(RuntimeError):
        torch.cat([unsupported, unsupported], dim=0)
    with pytest.raises(RuntimeError):
        torch.cat([one, unsupported], dim=0)
    # [0] list skipping is the sole empty-shape special case, including raw dims.
    for dim in range(-6, 7):
        result = torch.cat([empty, empty], dim=dim)
        assert tuple(result.shape) == (0,)
        assert tuple(result.stride()) == (1,)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (0, 0, 0, 0)
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.cat([empty, one], dim=-1)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    torch.testing.assert_close(result.cpu(), torch.tensor([1.0, 2.0]))


@pytest.mark.parametrize("rank", range(1, 5))
def test_functional_cat_skips_rank_one_empty_with_all_ranks(rank):
    shape = tuple(range(2, rank + 2))
    value_cpu = torch.arange(torch.tensor(shape).prod(), dtype=torch.float32).reshape(shape)
    empty_cpu = torch.empty((0,), dtype=torch.float32)
    value = value_cpu.to("vk:0")
    empty = empty_cpu.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    outputs = []
    for dim in range(rank):
        outputs.append((torch.cat([empty, value], dim=dim),
                        torch.cat([value, empty], dim=dim - rank),
                        torch.cat([empty_cpu, value_cpu], dim=dim)))
        pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2 * rank, 0, 0, 0)
    for positive, negative, expected in outputs:
        torch.testing.assert_close(positive.cpu(), expected)
        torch.testing.assert_close(negative.cpu(), expected)


def test_functional_cat_mixed_frozen_and_zero_length_trainable_segments():
    cpu_a = torch.arange(6, dtype=torch.float32).reshape(2, 3).requires_grad_()
    cpu_frozen = (torch.arange(4, dtype=torch.float32).reshape(2, 2) + 20)
    cpu_empty = torch.empty((2, 0), dtype=torch.float32).requires_grad_()
    cpu_b = (torch.arange(8, dtype=torch.float32).reshape(2, 4) + 40).requires_grad_()
    a = cpu_a.detach().to("vk:0").requires_grad_()
    frozen = cpu_frozen.to("vk:0")
    empty = cpu_empty.detach().to("vk:0").requires_grad_()
    b = cpu_b.detach().to("vk:0").requires_grad_()
    cpu_seed = torch.arange(1, 19, dtype=torch.float32).reshape(2, 9)
    seed = cpu_seed.to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = torch.cat([cpu_a, cpu_frozen, cpu_empty, cpu_b], dim=1)
    result = torch.cat([a, frozen, empty, b], dim=1)
    cpu_ga, cpu_ge, cpu_gb = torch.autograd.grad(
        cpu_result, (cpu_a, cpu_empty, cpu_b), cpu_seed)
    ga, ge, gb = torch.autograd.grad(result, (a, empty, b), seed)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (3, 0, 0, 0)
    torch.testing.assert_close(result.cpu(), cpu_result)
    for actual, expected in ((ga, cpu_ga), (ge, cpu_ge), (gb, cpu_gb)):
        torch.testing.assert_close(actual.cpu(), expected)
    assert ge.shape == (2, 0)


def test_functional_cat_65_inputs_bounded_descriptor_resources():
    cpu_inputs = [torch.tensor([float(i)], dtype=torch.float32) for i in range(65)]
    inputs = [x.to("vk:0") for x in cpu_inputs]
    pytorch_vulkan._C.synchronize()
    service_before = pytorch_vulkan._C.shared_service_snapshot()
    pytorch_vulkan._C.reset_execution_counters()
    pytorch_vulkan._C.reset_descriptor_resource_counters()
    result = torch.cat(inputs)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    services = pytorch_vulkan._C.shared_service_snapshot()
    assert counters == (65, 0, 0, 0)
    assert pytorch_vulkan._C.compute_submitted_count() <= 65
    descriptor_work = (
        services["descriptor_allocations"] - service_before["descriptor_allocations"]
        + services["descriptor_reuses"] - service_before["descriptor_reuses"]
    )
    assert descriptor_work == 65
    assert services["descriptor_pools"] <= services["descriptor_pool_limit"]
    assert services["descriptor_sets"] <= services["descriptor_pool_limit"] * 64
    torch.testing.assert_close(result.cpu(), torch.cat(cpu_inputs))


def test_functional_cat_empty_extents_and_singleton_freshness():
    singleton_cpu = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    singleton = singleton_cpu.to("vk:0")
    zero_cat = torch.empty((2, 0), dtype=torch.float32).to("vk:0")
    zero_noncat_a = torch.empty((0, 3), dtype=torch.float32).to("vk:0")
    zero_noncat_b = torch.empty((0, 3), dtype=torch.float32).to("vk:0")
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    fresh = torch.cat([singleton], dim=1)
    zero_cat_result = torch.cat([zero_cat, zero_cat], dim=1)
    zero_noncat_result = torch.cat([zero_noncat_a, zero_noncat_b], dim=0)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (1, 0, 0, 0)
    assert tuple(zero_cat_result.shape) == (2, 0)
    assert tuple(zero_noncat_result.shape) == (0, 3)
    assert zero_cat_result.stride() == torch.cat(
        [torch.empty((2, 0)), torch.empty((2, 0))], dim=1).stride()
    assert zero_noncat_result.stride() == torch.cat(
        [torch.empty((0, 3)), torch.empty((0, 3))], dim=0).stride()
    assert fresh.data_ptr() != singleton.data_ptr()
    torch.testing.assert_close(fresh.cpu(), singleton_cpu)


@pytest.mark.parametrize(
    "memory_format",
    [None, torch.preserve_format, torch.channels_last],
    ids=["default", "preserve", "explicit-channels-last"],
)
def test_vulkan_to_channels_last_format_preserves_layout_for_cat(memory_format):
    shape = (2, 3, 4, 5)
    second_shape = (2, 4, 4, 5)
    cpu_a = torch.arange(120, dtype=torch.float32).reshape(shape).contiguous(
        memory_format=torch.channels_last)
    cpu_b = (torch.arange(160, dtype=torch.float32).reshape(second_shape) + 1000).contiguous(
        memory_format=torch.channels_last)
    transfer_options = {} if memory_format is None else {"memory_format": memory_format}
    a = cpu_a.to("vk:0", **transfer_options)
    b = cpu_b.to("vk:0", **transfer_options)
    assert a.stride() == cpu_a.stride()
    assert b.stride() == cpu_b.stride()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.cat([a, b], dim=1)
    expected = torch.cat([cpu_a, cpu_b], dim=1)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", range(4))
@pytest.mark.parametrize("mixed", [False, True], ids=["all-channels-last", "mixed"])
def test_functional_cat_rank_four_format_selection(dim, mixed):
    first_shape = [2, 3, 4, 5]
    second_shape = first_shape.copy()
    second_shape[dim] += 1
    first_cpu = torch.arange(120, dtype=torch.float32).reshape(first_shape).contiguous(
        memory_format=torch.channels_last)
    second_cpu = (torch.arange(torch.tensor(second_shape).prod(), dtype=torch.float32)
                  .reshape(second_shape) + 1000)
    if not mixed:
        second_cpu = second_cpu.contiguous(memory_format=torch.channels_last)

    def explicit_layout_upload(value):
        device_value = torch.empty_strided(
            value.shape, value.stride(), dtype=value.dtype, device="vk:0")
        device_value.copy_(value)
        return device_value

    first = explicit_layout_upload(first_cpu)
    second = explicit_layout_upload(second_cpu)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.cat([first, second], dim=dim)
    expected = torch.cat([first_cpu, second_cpu], dim=dim)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (2, 0, 0, 0)
    assert result.stride() == expected.stride()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("dim", range(4))
def test_functional_cat_ambiguous_singleton_and_skipped_empty_format(dim):
    cpu_singletons = [
        torch.ones((1, 3, 1, 1)).contiguous(memory_format=torch.channels_last),
        torch.ones((1, 2, 1, 1)).contiguous(memory_format=torch.channels_last),
    ]
    vk_singletons = [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="vk:0")
                     for x in cpu_singletons]
    for target, source in zip(vk_singletons, cpu_singletons):
        target.copy_(source)
    cpu_empty = torch.empty((0,), dtype=torch.float32)
    vk_empty = cpu_empty.to("vk:0")
    first_shape = [2, 3, 4, 5]
    second_shape = first_shape.copy()
    second_shape[dim] += 1
    cpu_spatial = [
        torch.ones(first_shape).contiguous(memory_format=torch.channels_last),
        torch.ones(second_shape).contiguous(memory_format=torch.channels_last),
    ]
    vk_spatial = [torch.empty_strided(x.shape, x.stride(), dtype=x.dtype, device="vk:0")
                  for x in cpu_spatial]
    for target, source in zip(vk_spatial, cpu_spatial):
        target.copy_(source)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    singleton_result = torch.cat(vk_singletons, dim=1)
    skipped_result = torch.cat([vk_spatial[0], vk_empty, vk_spatial[1]], dim=dim)
    singleton_expected = torch.cat(cpu_singletons, dim=1)
    skipped_expected = torch.cat([cpu_spatial[0], cpu_empty, cpu_spatial[1]], dim=dim)
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters == (4, 0, 0, 0)
    assert singleton_result.stride() == singleton_expected.stride()
    assert skipped_result.stride() == skipped_expected.stride()
    torch.testing.assert_close(singleton_result.cpu(), singleton_expected)
    torch.testing.assert_close(skipped_result.cpu(), skipped_expected)
