import re

import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def _assert_rejected(operation, message):
    with pytest.raises(RuntimeError, match=message):
        operation()


def _assert_no_vulkan_work():
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


FLOAT16_ALLOCATION_ERROR = (
    "Vulkan float16 support is deferred; allocation cannot use float16 until its "
    "storage, transfer, shader, promotion, and autograd contracts are implemented"
)
FLOAT16_STORAGE_ERROR = (
    "Vulkan float16 support is deferred; storage cannot use float16 until its "
    "storage, transfer, shader, promotion, and autograd contracts are implemented"
)


def _exact_error(message):
    return re.escape(message)


@pytest.mark.parametrize(
    "source",
    [
        torch.tensor([True, False, True], dtype=torch.bool),
        torch.tensor(True, dtype=torch.bool),
        torch.tensor([[True, False], [False, True]], dtype=torch.bool),
        torch.empty((0, 3), dtype=torch.bool),
    ],
)
def test_bool_cpu_vulkan_cpu_round_trip_preserves_metadata(vulkan_backend, source):
    device_tensor = source.to(vulkan_backend)
    result = device_tensor.to("cpu")

    assert device_tensor.dtype is torch.bool
    assert device_tensor.shape == source.shape
    assert device_tensor.stride() == source.stride()
    assert device_tensor.is_contiguous()
    assert result.dtype is torch.bool
    assert result.shape == source.shape
    assert torch.equal(result, source)


def test_bool_copy_is_in_place_and_synchronous(vulkan_backend):
    source = torch.tensor([True, False, True], dtype=torch.bool)
    destination = torch.empty_like(source, device=vulkan_backend)
    metadata = (destination.dtype, destination.shape, destination.stride())

    assert destination.copy_(source) is destination
    assert (destination.dtype, destination.shape, destination.stride()) == metadata
    assert torch.equal(destination.cpu(), source)


def test_bool_non_contiguous_transfer_is_supported(vulkan_backend):
    source = torch.tensor([[True, False], [False, True]], dtype=torch.bool).t()
    assert not source.is_contiguous()
    result = source.to(vulkan_backend)
    assert torch.equal(result.cpu(), source)


def test_bool_non_blocking_transfer_is_rejected(vulkan_backend):
    source = torch.tensor([True, False], dtype=torch.bool)
    _assert_rejected(
        lambda: source.to(vulkan_backend, non_blocking=True), "non_blocking"
    )


def test_bool_vulkan_to_vulkan_transfer_is_supported(vulkan_backend):
    source = torch.tensor([True, False], dtype=torch.bool).to(vulkan_backend)
    destination = torch.empty_like(source)
    destination.copy_(source)
    assert torch.equal(destination.cpu(), source.cpu())


def test_bool_mismatched_dtype_transfer_is_rejected(vulkan_backend):
    destination = torch.empty((2,), dtype=torch.bool, device=vulkan_backend)
    source = torch.ones((2,), dtype=torch.float32)
    _assert_rejected(lambda: destination.copy_(source), "matching dtypes")


def test_bool_non_contiguous_destination_transfer_is_supported(vulkan_backend):
    source = torch.tensor([[True, False], [False, True]], dtype=torch.bool)
    destination = torch.empty((2, 2), dtype=torch.bool, device=vulkan_backend).t()
    assert not destination.is_contiguous()
    destination.copy_(source)
    assert torch.equal(destination.cpu(), source)


def test_bool_invalid_device_index_is_rejected(vulkan_backend):
    _assert_rejected(
        lambda: torch.empty((2,), dtype=torch.bool, device="vk:1"),
        "only device index 0",
    )


def test_float16_contiguous_allocation_is_rejected_before_materialization(
    vulkan_backend,
):
    _assert_rejected(
        lambda: torch.empty((2,), dtype=torch.float16, device=vulkan_backend),
        _exact_error(FLOAT16_ALLOCATION_ERROR),
    )


def test_float16_non_contiguous_allocation_is_rejected_before_materialization(
    vulkan_backend,
):
    _assert_rejected(
        lambda: torch.empty_strided(
            (2, 2), (1, 2), dtype=torch.float16, device=vulkan_backend
        ),
        _exact_error(FLOAT16_ALLOCATION_ERROR),
    )


def test_float16_cpu_to_vulkan_transfer_is_rejected_before_dispatch(vulkan_backend):
    source = torch.empty((2,), dtype=torch.float16)
    pytorch_vulkan._C.reset_execution_counters()
    _assert_rejected(
        lambda: source.to(vulkan_backend), _exact_error(FLOAT16_ALLOCATION_ERROR)
    )
    _assert_no_vulkan_work()


def test_float16_non_contiguous_cpu_to_vulkan_transfer_is_rejected(
    vulkan_backend,
):
    source = torch.empty((2, 2), dtype=torch.float16).t()
    assert not source.is_contiguous()
    _assert_rejected(
        lambda: source.to(vulkan_backend), _exact_error(FLOAT16_ALLOCATION_ERROR)
    )


def test_float16_vulkan_to_cpu_transfer_is_rejected_at_allocation_boundary(
    vulkan_backend,
):
    _assert_rejected(
        lambda: torch.empty((2,), dtype=torch.float16, device=vulkan_backend).to("cpu"),
        _exact_error(FLOAT16_ALLOCATION_ERROR),
    )


def test_float16_cpu_to_vulkan_copy_is_rejected_before_dispatch(vulkan_backend):
    destination = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    source = torch.empty((2,), dtype=torch.float16)
    pytorch_vulkan._C.reset_execution_counters()
    _assert_rejected(
        lambda: destination.copy_(source), _exact_error(FLOAT16_STORAGE_ERROR)
    )
    _assert_no_vulkan_work()


def test_float16_vulkan_to_cpu_copy_is_rejected_before_dispatch(vulkan_backend):
    source = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    destination = torch.empty((2,), dtype=torch.float16)
    pytorch_vulkan._C.reset_execution_counters()
    _assert_rejected(
        lambda: destination.copy_(source), _exact_error(FLOAT16_STORAGE_ERROR)
    )
    _assert_no_vulkan_work()


def test_integer_copy_to_vulkan_float32_is_rejected_before_dispatch(vulkan_backend):
    destination = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
    source = torch.ones((2,), dtype=torch.int64)
    pytorch_vulkan._C.reset_execution_counters()

    _assert_rejected(lambda: destination.copy_(source), "dtype")
    _assert_no_vulkan_work()


def test_vulkan_float32_to_cpu_double_conversion_is_rejected_before_dispatch(
    vulkan_backend,
):
    source = torch.ones((2,), dtype=torch.float32, device=vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    _assert_rejected(
        lambda: source.to(device="cpu", dtype=torch.float64),
        "requested dtype does not match source dtype",
    )
    _assert_no_vulkan_work()


def test_cpu_float32_to_vulkan_bool_conversion_is_rejected_before_dispatch(
    vulkan_backend,
):
    source = torch.ones((2,), dtype=torch.float32)
    pytorch_vulkan._C.reset_execution_counters()

    _assert_rejected(
        lambda: source.to(device=vulkan_backend, dtype=torch.bool),
        "requested dtype does not match source dtype",
    )
    _assert_no_vulkan_work()


@pytest.mark.parametrize(
    "operation, expected_dtype, expected_shape",
    [
        (
            lambda: torch.add(torch.tensor([1], dtype=torch.float16), 1.0),
            torch.float16,
            (1,),
        ),
        (
            lambda: torch.add(1.0, torch.tensor([1], dtype=torch.float16)),
            torch.float16,
            (1,),
        ),
        (
            lambda: torch.mul(torch.tensor([1], dtype=torch.float16), 2),
            torch.float16,
            (1,),
        ),
        (
            lambda: torch.sub(
                torch.tensor([1], dtype=torch.float16),
                torch.tensor([2], dtype=torch.float32),
            ),
            torch.float32,
            (1,),
        ),
        (
            lambda: torch.add(torch.tensor(1, dtype=torch.float16), 1.0),
            torch.float16,
            (),
        ),
    ],
)
def test_float16_cpu_promotion_observations(operation, expected_dtype, expected_shape):
    result = operation()
    assert result.dtype is expected_dtype
    assert result.shape == expected_shape


def test_float16_operator_request_is_rejected_before_dispatch(vulkan_backend):
    _assert_rejected(
        lambda: torch.neg(
            torch.empty((2,), dtype=torch.float16, device=vulkan_backend)
        ),
        _exact_error(FLOAT16_ALLOCATION_ERROR),
    )


def test_float16_scalar_form_is_rejected_before_dispatch(vulkan_backend):
    _assert_rejected(
        lambda: torch.add(
            torch.empty((2,), dtype=torch.float16, device=vulkan_backend), 1.0
        ),
        _exact_error(FLOAT16_ALLOCATION_ERROR),
    )
