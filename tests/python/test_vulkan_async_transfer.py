"""Public F32 non_blocking transfer witnesses (Task 5 steps 1-2).

Pending cases use the completion-hold seam and are meaningful only in async
execution mode (PYTORCH_VULKAN_ASYNC_EXECUTION unset/1); in sync mode every
accepted non_blocking=True must instead complete before return. Run one mode per
invocation. While a hold is active only enqueue/query/drop-reference operations
may run: release before any host read or synchronize.
"""
import gc
import threading
import time

import pytest
import pytorch_vulkan
import torch

api = pytorch_vulkan._C


@pytest.fixture
def vk():
    assert pytorch_vulkan.is_available(), "expected qualified Vulkan device unavailable"
    return "vk"


def _async():
    return api.execution_mode() == "async"


@pytest.fixture
def hold():
    # Pending-transfer witnesses are meaningful only in async mode. Evaluated at run
    # time, never at import/collection (nothing here may touch the device).
    if api.execution_mode() != "async":
        pytest.skip("pending-transfer witnesses require async execution mode")
    state = {"on": False}

    def start():
        api.test_hold_completion(True)
        state["on"] = True

    yield start
    api.test_hold_completion(False)
    api.synchronize()
    assert api.completion_hold_expiry_count() == 0


def _pending():
    return api.live_resource_snapshot()[4]  # pending_transfers


def _pinned_allocations():
    return api.live_resource_snapshot()[7]  # pinned_host_allocations


def _waits():
    return api.execution_fence_wait_count(), api.transfer_wait_count()


def _pinned(values):
    return torch.as_tensor(values, dtype=torch.float32).pin_memory(device="vk")


def _complete_or_pending_after(call):
    """Run call() and return its result; in sync mode assert completion before return."""
    result = call()
    if not _async():
        assert api.query() is True
    return result


@pytest.mark.parametrize("flag_view", ["whole", "offset"])
def test_pinned_h2d_nonblocking_returns_before_completion(vk, hold, flag_view):
    src = _pinned(torch.arange(24, dtype=torch.float32))
    view = src if flag_view == "whole" else src[4:20]
    expected = view.clone().cpu()
    api.synchronize()
    before = _waits()
    hold()
    dst = view.to(vk, non_blocking=True)
    assert _waits() == before and _pending() == 1
    assert api.query() is False
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(dst.cpu(), expected, rtol=0, atol=0)
    assert _pending() == 0


def test_pinned_and_device_wrappers_deleted_while_pending_keep_resources(vk, hold):
    src = _pinned(torch.arange(16, dtype=torch.float32))
    expected = src.clone().cpu()
    keeper = torch.zeros(16).to(vk)
    api.synchronize()
    pinned_before = _pinned_allocations()
    hold()
    dst = src.to(vk, non_blocking=True)
    del src
    gc.collect()
    assert _pinned_allocations() == pinned_before  # lease keeps the pinned allocation
    del dst
    gc.collect()
    assert _pending() == 1
    api.test_hold_completion(False)
    api.synchronize()
    assert _pending() == 0 and _pinned_allocations() < pinned_before
    # a surviving device copy path is still usable
    survivor = (keeper + 3.0).cpu()
    torch.testing.assert_close(survivor, torch.full((16,), 3.0), rtol=0, atol=0)
    assert expected.numel() == 16


def test_destination_resize_growth_after_enqueue_preserves_prefix(vk, hold):
    src = _pinned(torch.arange(8, dtype=torch.float32))
    expected = src.clone().cpu()
    api.synchronize()
    hold()
    dst = src.to(vk, non_blocking=True)
    dst.resize_(16)  # hold active: ordering through the existing copy-resize path
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(dst.cpu()[:8], expected, rtol=0, atol=0)


def test_pinned_d2h_into_copy_destination_valid_after_synchronize(vk, hold):
    device = torch.arange(12, dtype=torch.float32).to(vk)
    destination = torch.zeros(12).pin_memory(device="vk")
    api.synchronize()
    hold()
    returned = destination.copy_(device, non_blocking=True)
    assert returned is destination and api.query() is False and _pending() == 1
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(destination, torch.arange(12, dtype=torch.float32), rtol=0, atol=0)


def test_auto_pinned_readback_result_is_pinned_and_valid_after_synchronize(vk):
    device = torch.arange(6, dtype=torch.float32).to(vk)
    api.synchronize()
    result = _complete_or_pending_after(lambda: device.to("cpu", non_blocking=True))
    assert result.is_pinned(device="vk")  # D4: CUDA pin_out analog
    api.synchronize()
    torch.testing.assert_close(result, torch.arange(6, dtype=torch.float32), rtol=0, atol=0)


def test_pageable_h2d_snapshot_is_independent_of_later_cpu_writes(vk, hold):
    cpu = torch.tensor([2.0, 7.0, 11.0])
    api.synchronize()
    hold()
    dst = cpu.to(vk, non_blocking=True)
    cpu.fill_(-1.0)
    assert api.query() is False
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(dst.cpu(), torch.tensor([2.0, 7.0, 11.0]), rtol=0, atol=0)


def test_pageable_d2h_and_ineligible_layouts_complete_before_return(vk):
    device = torch.arange(6, dtype=torch.float32).to(vk)
    pageable = torch.zeros(6)
    api.synchronize()
    pageable.copy_(device, non_blocking=True)
    assert api.query() is True
    torch.testing.assert_close(pageable, torch.arange(6, dtype=torch.float32), rtol=0, atol=0)
    # Ineligible layouts per the Task 0 oracle (async-t0-cpu-probe-03.json): stride-0
    # expand and a stride hole. A transposed dense source is ELIGIBLE (preserve-format
    # output keeps its strides), so it is deliberately not used here.
    expanded = torch.tensor([1.0, 2.0]).expand(3, 2)
    dst = expanded.to(vk, non_blocking=True)
    assert api.query() is True
    torch.testing.assert_close(dst.cpu(), expanded, rtol=0, atol=0)
    holed = torch.arange(6, dtype=torch.float32).reshape(2, 3)[:, :2]
    dst = holed.to(vk, non_blocking=True)
    assert api.query() is True
    torch.testing.assert_close(dst.cpu(), holed, rtol=0, atol=0)


@pytest.mark.parametrize("flag", [False, True])
def test_d2d_returns_without_host_wait_in_async_mode(vk, flag):
    source = torch.arange(8, dtype=torch.float32).to(vk)
    destination = torch.empty(8, device=vk)
    api.synchronize()
    before = _waits()
    destination.copy_(source, non_blocking=flag)
    if _async():
        assert _waits() == before
    api.synchronize()
    torch.testing.assert_close(destination.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)


def test_queue_ordering_with_compute_producers_and_consumers(vk):
    a = torch.ones(16).to(vk)
    producer = a + 1.0
    sink = torch.zeros(16).pin_memory(device="vk")
    sink.copy_(producer, non_blocking=True)  # D2H ordered after the producer
    consumer = producer * 2.0  # compute ordered after / alongside the D2H
    api.synchronize()
    torch.testing.assert_close(sink, torch.full((16,), 2.0), rtol=0, atol=0)
    torch.testing.assert_close(consumer.cpu(), torch.full((16,), 4.0), rtol=0, atol=0)
    host = _pinned(torch.arange(16, dtype=torch.float32))
    uploaded = host.to(vk, non_blocking=True)  # H2D then compute
    result = uploaded + 1.0
    api.synchronize()
    torch.testing.assert_close(result.cpu(), torch.arange(16, dtype=torch.float32) + 1.0, rtol=0, atol=0)


def test_three_enqueue_backpressure_releases_in_order(vk, hold):
    sources = [_pinned(torch.full((8,), float(i + 1))) for i in range(3)]
    results = [None] * 3
    errors = []
    api.synchronize()
    waits_before = api.transfer_reservation_wait_count()
    hold()
    results[0] = sources[0].to(vk, non_blocking=True)
    results[1] = sources[1].to(vk, non_blocking=True)

    def third():
        try:
            results[2] = sources[2].to(vk, non_blocking=True)
        except BaseException as error:  # reported to the main thread
            errors.append(error)

    worker = threading.Thread(target=third)
    worker.start()
    deadline = time.monotonic() + 5.0
    while api.transfer_reservation_wait_count() < waits_before + 1 and time.monotonic() < deadline:
        time.sleep(0.01)
    assert api.transfer_reservation_wait_count() >= waits_before + 1
    api.test_hold_completion(False)
    worker.join(5.0)
    assert not worker.is_alive() and not errors
    api.synchronize()
    for index, result in enumerate(results):
        torch.testing.assert_close(result.cpu(), torch.full((8,), float(index + 1)), rtol=0, atol=0)


def test_overlap_and_mismatch_rejected_before_any_enqueue(vk):
    api.synchronize()
    pending = _pending()
    operations = api.transfer_operation_count()
    with pytest.raises(RuntimeError, match="expanded size"):
        torch.empty(2, device=vk).copy_(torch.ones(4), non_blocking=True)
    overlapping = torch.empty(1, device=vk).expand(4)
    with pytest.raises(RuntimeError, match="internal overlap"):
        overlapping.copy_(torch.ones(4), non_blocking=True)
    assert _pending() == pending and api.transfer_operation_count() == operations


def test_empty_tensors_create_no_transfer_record(vk):
    api.synchronize()
    operations = api.transfer_operation_count()
    empty = torch.empty(0).to(vk, non_blocking=True)
    assert empty.numel() == 0
    torch.empty(0, device=vk).to("cpu", non_blocking=True)
    assert api.transfer_operation_count() == operations and _pending() == 0


def test_non_float32_non_blocking_is_still_rejected(vk):
    with pytest.raises(RuntimeError, match="non_blocking"):
        torch.ones(2, dtype=torch.bool).to(vk, non_blocking=True)
    with pytest.raises(RuntimeError, match="non_blocking"):
        torch.ones(2, dtype=torch.int64).to(vk, non_blocking=True)
    with pytest.raises(RuntimeError, match="float32"):
        torch.ones(2, dtype=torch.float64).to(vk, non_blocking=True)


def test_to_and_copy_autograd_names_and_first_reverse(vk):
    leaf = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    moved = leaf.to(vk, non_blocking=True)
    assert moved.grad_fn.name() == "ToCopyBackward0"
    moved.sum().backward()
    api.synchronize()
    torch.testing.assert_close(leaf.grad, torch.ones(3), rtol=0, atol=0)
    source = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    destination = torch.zeros(3).to(vk)
    destination.copy_(source, non_blocking=True)
    assert destination.grad_fn.name() == "torch::autograd::CopyBackwards"
    destination.sum().backward()
    api.synchronize()
    torch.testing.assert_close(source.grad, torch.ones(3), rtol=0, atol=0)
