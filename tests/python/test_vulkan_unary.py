import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


UNARY_OPERATIONS = [torch.neg, torch.abs, torch.relu]


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
    values = torch.tensor(
        [[-3.5, 0.0], [2.25, -0.0]], dtype=torch.float32
    )
    tensor = values.to(vulkan_backend)

    result = operation(tensor)

    assert result.shape == (2, 2)
    assert result.is_contiguous()
    torch.testing.assert_close(result.cpu(), operation(values), rtol=0, atol=0)
    torch.testing.assert_close(tensor.cpu(), values, rtol=0, atol=0)


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_repeated_unary_operations_are_independent(vulkan_backend, operation):
    tensor = torch.tensor([1.0, -2.0, 0.0], dtype=torch.float32).to(
        vulkan_backend
    )
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
    tensor = torch.empty((2,), dtype=torch.float64, device=vulkan_backend)

    _assert_unary_rejected(lambda: operation(tensor), "float32")


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_non_contiguous_input_is_rejected(vulkan_backend, operation):
    tensor = torch.empty_strided(
        (3, 2), (1, 3), dtype=torch.float32, device=vulkan_backend
    )
    assert not tensor.is_contiguous()

    _assert_unary_rejected(lambda: operation(tensor), "contiguous")


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_nonzero_storage_offset_is_rejected(vulkan_backend, operation):
    base = torch.empty((3,), dtype=torch.float32, device=vulkan_backend)
    try:
        tensor = torch.as_strided(base, (2,), (1,), storage_offset=1)
    except NotImplementedError:
        pytest.skip("as_strided is unavailable for the Vulkan backend")
    assert tensor.storage_offset() != 0

    _assert_unary_rejected(lambda: operation(tensor), "zero.*offset|offset.*zero")


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_second_vulkan_device_is_rejected(vulkan_backend, operation):
    _assert_unary_rejected(
        lambda: operation(
            torch.empty((2,), dtype=torch.float32, device="vk:1")
        ),
        "only device index 0",
    )


@pytest.mark.parametrize("operation", UNARY_OPERATIONS)
def test_unary_zero_dimensional_input_is_rejected(vulkan_backend, operation):
    tensor = torch.empty((), dtype=torch.float32, device=vulkan_backend)

    _assert_unary_rejected(lambda: operation(tensor), "zero-dimensional")
