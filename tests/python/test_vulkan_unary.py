import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


UNARY_OPERATIONS = [torch.neg, torch.abs, torch.relu]
ACTIVATION_OPERATIONS = [torch.sigmoid, torch.tanh]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (torch.neg, [3.5, 0.0, -0.0, -2.25]),
        (torch.abs, [3.5, 0.0, 0.0, 2.25]),
        (torch.relu, [-0.0, -0.0, 0.0, 2.25]),
    ],
)
def test_cpu_only_unary_remains_normal_pytorch_behavior(operation, expected):
    values = torch.tensor([-3.5, -0.0, 0.0, 2.25], dtype=torch.float32)

    torch.testing.assert_close(
        operation(values), torch.tensor(expected, dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_functional_values_and_metadata(vulkan_backend, operation):
    values = torch.tensor([-3.5, -0.0, 0.0, 2.25], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    result = operation(tensor)

    expected = operation(values)
    assert result.device.type == "vk"
    assert result.device.index == 0
    assert result.shape == values.shape
    assert result.dtype is torch.float32
    assert result.is_contiguous()
    assert result.data_ptr() != tensor.data_ptr()
    torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(tensor.cpu(), values, rtol=0, atol=0)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_functional_multidimensional_values(vulkan_backend, operation):
    values = torch.tensor([[-3.5, 0.0], [2.25, -0.0]], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    result = operation(tensor)

    assert result.shape == (2, 2)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), operation(values), rtol=0, atol=0)
    torch.testing.assert_close(tensor.cpu(), values, rtol=0, atol=0)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_repeated_unary_operations_are_independent(vulkan_backend, operation):
    tensor = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32).to(vulkan_backend)
    expected = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32)

    for _ in range(32):
        tensor = operation(tensor)
        expected = operation(expected)

    torch.testing.assert_close(tensor.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_empty_unary_result_preserves_metadata(vulkan_backend, operation):
    tensor = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)

    result = operation(tensor)

    assert result.device == tensor.device
    assert result.shape == (0, 3)
    assert result.dtype is torch.float32
    assert result.numel() == 0
    assert result.is_contiguous()


def _assert_unary_rejected(operation, message=None):
    with pytest.raises((RuntimeError, TypeError), match=message):
        operation()


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_out_variant_is_supported(vulkan_backend, operation):
    tensor = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    output = torch.empty_like(tensor)

    if operation is torch.relu:
        assert torch.ops.aten.relu.out(tensor, out=output) is output
    else:
        assert operation(tensor, out=output) is output


@pytest.mark.parametrize(
    ("operation", "method"),
    [(torch.neg, "neg_"), (torch.abs, "abs_"), (torch.relu, "relu_")],
)
def test_unary_inplace_variant_is_rejected(vulkan_backend, operation, method):
    tensor = torch.tensor([1.0, -2.0], dtype=torch.float32, device=vulkan_backend)
    before = tensor.cpu()

    _assert_unary_rejected(lambda: getattr(tensor, method)(), "in-place")
    torch.testing.assert_close(tensor.cpu(), before)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_float64_input_is_rejected(vulkan_backend, operation):
    if pytorch_vulkan.formatter_double_supported():
        torch.empty((2,), dtype=torch.float64, device=vulkan_backend)
    else:
        with pytest.raises(RuntimeError, match="shaderFloat64"):
            torch.empty((2,), dtype=torch.float64, device=vulkan_backend)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_stride_aware_inputs_match_cpu(vulkan_backend, operation):
    base_cpu = torch.tensor([[-3.0, 1.0, 2.0], [4.0, -5.0, 6.0]], dtype=torch.float32)
    base = base_cpu.to(vulkan_backend)
    cpu_views = [
        base_cpu.transpose(0, 1),
        base_cpu[:, 1:],
        torch.as_strided(base_cpu, (2, 3), (0, 1), storage_offset=0),
        torch.as_strided(base_cpu, (2, 2), (1, 2), storage_offset=1),
    ]
    views = [
        base.transpose(0, 1),
        base[:, 1:],
        torch.as_strided(base, (2, 3), (0, 1), storage_offset=0),
        torch.as_strided(base, (2, 2), (1, 2), storage_offset=1),
    ]

    for tensor, cpu_view in zip(views, cpu_views):
        result = operation(tensor)
        expected = operation(cpu_view)
        torch.testing.assert_close(result.cpu(), expected, rtol=0, atol=0)


def test_unary_rejects_rank_above_eight_before_dispatch(vulkan_backend):
    tensor = torch.empty((1,) * 9, dtype=torch.float32, device=vulkan_backend)

    _assert_unary_rejected(lambda: torch.neg(tensor), "rank.*8")


def test_ceil_nonzero_storage_offset_matches_expected(vulkan_backend):
    base = torch.tensor([0.25, 1.25, 2.25], dtype=torch.float32, device=vulkan_backend)
    tensor = torch.as_strided(base, (2,), (1,), storage_offset=1)
    assert tensor.storage_offset() != 0

    torch.testing.assert_close(torch.ceil(tensor).cpu(), torch.tensor([2.0, 3.0]))


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_second_vulkan_device_is_rejected(vulkan_backend, operation):
    _assert_unary_rejected(
        lambda: operation(torch.empty((2,), dtype=torch.float32, device="vk:1")),
        "only device index 0",
    )


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_zero_dimensional_input_is_supported(vulkan_backend, operation):
    tensor = torch.empty((), dtype=torch.float32, device=vulkan_backend)

    result = operation(tensor)
    assert result.dim() == 0


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_bool_input_is_rejected_by_capability_matrix(vulkan_backend, operation):
    tensor = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)

    _assert_unary_rejected(lambda: operation(tensor), "bool|float32|support")


@pytest.mark.parametrize("operation", ACTIVATION_OPERATIONS)
def test_activation_functional_values_stay_on_vulkan(vulkan_backend, operation):
    values = torch.tensor([-3.5, -0.0, 0.0, 2.25], dtype=torch.float32)
    tensor = values.to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = operation(tensor)

    assert result.device == tensor.device
    assert result.dtype is torch.float32
    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0
    torch.testing.assert_close(result.cpu(), operation(values), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize(
    "operation", [torch.sigmoid, torch.tanh, torch.nn.functional.gelu]
)
def test_activation_empty_result_is_vulkan_resident(vulkan_backend, operation):
    tensor = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    if operation is torch.nn.functional.gelu:
        result = operation(tensor, approximate="tanh")
    else:
        result = operation(tensor)

    assert result.device == tensor.device
    assert result.shape == (0, 3)
    assert result.numel() == 0
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


def test_gelu_tanh_matches_cpu_and_rejects_exact_mode(vulkan_backend):
    values = torch.tensor([-4.0, -1.5, 0.0, 1.5, 4.0], dtype=torch.float32)
    tensor = values.to(vulkan_backend)

    result = torch.nn.functional.gelu(tensor, approximate="tanh")
    torch.testing.assert_close(
        result.cpu(),
        torch.nn.functional.gelu(values, approximate="tanh"),
        rtol=2e-5,
        atol=2e-6,
    )

    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="approximate.*tanh|GELU"):
        torch.nn.functional.gelu(tensor, approximate="none")
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize(
    "operation",
    [torch.sigmoid, torch.tanh, torch.nn.functional.gelu],
)
def test_activation_unsupported_overloads_do_not_dispatch(vulkan_backend, operation):
    tensor = torch.ones(2, dtype=torch.float32, device=vulkan_backend)
    output = torch.empty_like(tensor)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(
        (RuntimeError, TypeError), match="Vulkan|Could not run|unsupported"
    ):
        if operation is torch.nn.functional.gelu:
            operation(tensor, approximate="none", out=output)
        else:
            operation(tensor, out=output)
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("method", ["sigmoid_", "tanh_"])
def test_activation_inplace_variant_is_rejected(vulkan_backend, method):
    tensor = torch.ones(2, dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    with pytest.raises(
        (RuntimeError, TypeError), match="Vulkan|Could not run|unsupported"
    ):
        getattr(tensor, method)()
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("operation", [torch.sigmoid, torch.tanh])
def test_activation_rejects_bool_and_float16(vulkan_backend, operation):
    bool_tensor = torch.tensor([True, False], dtype=torch.bool, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="float32|dtype|support"):
        operation(bool_tensor)
    if pytorch_vulkan.formatter_double_supported():
        with pytest.raises(RuntimeError, match="float16|float32|dtype"):
            float16_tensor = torch.empty(2, dtype=torch.float16, device=vulkan_backend)
            operation(float16_tensor)


@pytest.mark.parametrize(
    "operation",
    [
        torch.sigmoid,
        torch.tanh,
        lambda value: torch.nn.functional.gelu(value, approximate="tanh"),
    ],
)
def test_activation_rejects_device_and_rank_limits(vulkan_backend, operation):
    with pytest.raises(RuntimeError, match="device index 0"):
        operation(torch.empty((2,), dtype=torch.float32, device="vk:1"))
    with pytest.raises(RuntimeError, match="rank.*8"):
        operation(torch.empty((1,) * 9, dtype=torch.float32, device=vulkan_backend))
