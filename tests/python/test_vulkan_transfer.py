import gc

import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_cpu_vk_cpu_round_trip(vulkan_backend):
    source = torch.tensor([1.0, -2.5, 3.25, 0.0], dtype=torch.float32)
    device_tensor = source.to("vk")
    result = device_tensor.to("cpu")
    assert device_tensor.device.type == "vk"
    assert device_tensor.device.index == 0
    assert device_tensor.shape == source.shape
    assert device_tensor.dtype == torch.float32
    assert device_tensor.is_contiguous()
    torch.testing.assert_close(result, source, rtol=0, atol=0)


def test_cpu_vk_cpu_round_trip_scalar_like(vulkan_backend):
    source = torch.tensor(1.25, dtype=torch.float32)
    result = source.to(vulkan_backend).to("cpu")
    torch.testing.assert_close(result, source, rtol=0, atol=0)


def test_cpu_vk_cpu_round_trip_multidimensional_contiguous(vulkan_backend):
    source = torch.tensor(
        [[1.0, -2.5], [3.25, 0.0]], dtype=torch.float32
    ).contiguous()
    device_tensor = source.to(vulkan_backend)
    result = device_tensor.to("cpu")
    assert device_tensor.shape == source.shape
    assert device_tensor.is_contiguous()
    torch.testing.assert_close(result, source, rtol=0, atol=0)


def test_independent_vulkan_tensor_lifetimes_and_repeated_round_trips(
    vulkan_backend,
):
    sources = [
        torch.tensor([float(index), -float(index)], dtype=torch.float32)
        for index in range(4)
    ]
    device_tensors = [source.to(vulkan_backend) for source in sources]
    for index in (3, 1, 0, 2):
        torch.testing.assert_close(
            device_tensors[index].to("cpu"), sources[index], rtol=0, atol=0
        )
    del device_tensors
    gc.collect()

    for index in range(32):
        source = torch.tensor([float(index)], dtype=torch.float32)
        device_tensor = source.to(vulkan_backend)
        torch.testing.assert_close(
            device_tensor.to("cpu"), source, rtol=0, atol=0
        )
        del device_tensor
    gc.collect()


def test_empty_float32_tensor_transfer_is_a_no_op(vulkan_backend):
    source = torch.empty((0,), dtype=torch.float32)
    device_tensor = source.to(vulkan_backend)
    result = device_tensor.to("cpu")
    assert device_tensor.numel() == 0
    assert result.shape == source.shape
    assert result.dtype == torch.float32


def _assert_transfer_rejected(operation, message):
    with pytest.raises(RuntimeError, match=message):
        operation()


def test_float64_transfer_is_rejected(vulkan_backend):
    _assert_transfer_rejected(
        lambda: torch.ones((2,), dtype=torch.float64).to(vulkan_backend),
        "float32",
    )


def test_integer_transfer_is_rejected(vulkan_backend):
    _assert_transfer_rejected(
        lambda: torch.ones((2,), dtype=torch.int64).to(vulkan_backend),
        "float32",
    )


def test_non_contiguous_transfer_is_rejected(vulkan_backend):
    source = torch.arange(8, dtype=torch.float32).reshape(2, 4).t()
    assert not source.is_contiguous()
    _assert_transfer_rejected(lambda: source.to(vulkan_backend), "contiguous")


def test_non_blocking_transfer_is_rejected(vulkan_backend):
    source = torch.ones((2,), dtype=torch.float32)
    _assert_transfer_rejected(
        lambda: source.to(vulkan_backend, non_blocking=True), "non_blocking"
    )


def test_vulkan_to_vulkan_transfer_is_rejected(vulkan_backend):
    def transfer():
        source = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
        source.to(vulkan_backend, copy=True)

    _assert_transfer_rejected(transfer, "Vulkan-to-Vulkan")


def test_mismatched_sizes_are_rejected(vulkan_backend):
    def transfer():
        destination = torch.empty((2,), dtype=torch.float32, device=vulkan_backend)
        source = torch.ones((3,), dtype=torch.float32)
        destination.copy_(source)

    _assert_transfer_rejected(transfer, "matching sizes")


def test_second_vulkan_device_is_rejected(vulkan_backend):
    _assert_transfer_rejected(
        lambda: torch.empty((2,), dtype=torch.float32, device="vk:1"),
        "only device index 0",
    )
