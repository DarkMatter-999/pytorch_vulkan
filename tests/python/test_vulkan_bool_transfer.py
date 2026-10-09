"""Byte-exact Bool copy contracts and physical bulk-copy scaling."""

import pytest
import pytorch_vulkan
import torch


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


@pytest.mark.parametrize("size", [0, 1, 3, 4, 5, 64, 512, 4096])
@pytest.mark.parametrize("offset", [1, 2, 3, 4])
@pytest.mark.parametrize("direction", ["upload", "readback", "device"])
def test_bool_contiguous_offset_and_tail_copy(vulkan_backend, size, offset, direction):
    # CPU data_ptr already includes its view offset; Vulkan offsets are relative
    # to the backing allocation. Different offsets exercise both conventions.
    source_base = torch.arange(size + offset + 8).remainder(3).eq(0)
    destination_base = ~torch.arange(size + offset + 9).remainder(2).eq(0)
    expected = destination_base.clone()
    source_slice = source_base[offset:offset + size]
    expected[offset + 1:offset + 1 + size].copy_(source_slice)
    source = source_base if direction == "upload" else source_base.to(vulkan_backend)
    destination = destination_base if direction == "readback" else destination_base.to(vulkan_backend)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    view = destination[offset + 1:offset + 1 + size]
    assert view.copy_(source[offset:offset + size]) is view
    api = pytorch_vulkan._C
    assert api.transfer_operation_count() == int(size != 0)
    expected_counters = ((0, 0, 0, 0) if size == 0 else
                         (0, 1, 0, 0) if direction == "device" else (0, 0, 1, 0))
    assert api.execution_counter_snapshot() == expected_counters
    api.synchronize()
    result = destination if direction == "readback" else destination.cpu()
    assert torch.equal(result, expected)  # Entire backing tensor, including neighbors.
    source_result = source if direction == "upload" else source.cpu()
    assert torch.equal(source_result, source_base)


@pytest.mark.parametrize("size", [64, 512, 4096])
@pytest.mark.parametrize("direction", ["upload", "readback", "device"])
def test_bool_bulk_has_constant_physical_submissions(vulkan_backend, size, direction):
    cpu = torch.arange(size).remainder(3).eq(0)
    source = cpu if direction == "upload" else cpu.to(vulkan_backend)
    destination = torch.empty_like(cpu, device="cpu" if direction == "readback" else vulkan_backend)
    api = pytorch_vulkan._C
    api.synchronize()
    api.reset_execution_counters()
    destination.copy_(source)
    submissions = api.transfer_submission_count()
    completions = api.transfer_completion_count()
    waits = api.transfer_wait_count()
    commands = api.copy_command_count()
    assert submissions <= 1, (size, direction, submissions, completions, waits, commands)
    assert completions == submissions
    assert waits <= 2
    assert commands <= 1
    assert api.transfer_operation_count() == 1
    assert api.execution_counter_snapshot() == ((0, 1, 0, 0) if direction == "device" else (0, 0, 1, 0))
    api.synchronize()
    assert torch.equal(destination if direction == "readback" else destination.cpu(), cpu)


def test_bool_alias_overlap_and_strided_contracts(vulkan_backend):
    cpu = torch.tensor([True, False, False, True, False, True, True, False])
    device = cpu.to(vulkan_backend)
    api = pytorch_vulkan._C
    api.synchronize()
    api.reset_execution_counters()
    device.copy_(device)
    assert api.execution_counter_snapshot() == (0, 0, 0, 0)
    assert api.transfer_operation_count() == 0
    with pytest.raises(RuntimeError, match="partially overlap"):
        device[1:5].copy_(device[:4])
    with pytest.raises(RuntimeError, match="internal overlap"):
        device[:1].expand(4).copy_(device[:4])
    with pytest.raises(RuntimeError, match="non_blocking"):
        device.copy_(cpu, non_blocking=True)
    with pytest.raises(RuntimeError, match="matching dtypes"):
        device.copy_(cpu.float())
    api.synchronize()
    assert torch.equal(device.cpu(), cpu)
    expected = cpu.clone()
    expected[::2].copy_(cpu[1::2])
    device[::2].copy_(cpu[1::2])
    api.synchronize()
    assert torch.equal(device.cpu(), expected)
    readback = torch.empty((4, 2), dtype=torch.bool).t()
    readback.copy_(device.reshape(2, 4))
    assert torch.equal(readback, expected.reshape(2, 4))
    other = cpu.to(vulkan_backend)
    other[::2].copy_(device[1::2])
    api.synchronize()
    other_expected = cpu.clone()
    other_expected[::2].copy_(expected[1::2])
    assert torch.equal(other.cpu(), other_expected)
    assert api.fallback_count() == 0


def test_bool_bulk_copy_inside_training_recording_preserves_lifetime(vulkan_backend):
    cpu = torch.arange(13).remainder(3).eq(0)
    source = cpu.to(vulkan_backend)
    expected = ~cpu
    destination = expected.to(vulkan_backend)
    expected[3:10].copy_(cpu[1:8])
    api = pytorch_vulkan._C
    api.synchronize()
    api.reset_execution_counters()
    api.begin_training_step()
    try:
        destination[3:10].copy_(source[1:8])
        assert api.copy_command_count() == 1
        assert api.execution_counter_snapshot() == (0, 1, 0, 0)
        del source
        api.end_training_step()
    except BaseException:
        api.cancel_training_step()
        raise
    api.synchronize()
    assert torch.equal(destination.cpu(), expected)
