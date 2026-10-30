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


def _allocations():
    return api.live_resource_snapshot()[6]  # Vulkan allocations


def _waits():
    return api.execution_fence_wait_count(), api.transfer_wait_count()


def _blocking_call(call):
    """Run a route that must complete before return and prove it before any synchronize.

    Never call this under a completion hold: a blocking route waits on the fence, and a
    held fence reports NOT_READY (query() False) until release or the 20 s watchdog
    expiry. Evidence: pending_transfers is 0 and query() is True right after the call;
    in async mode every host access also bumps a wait counter (acquire_host_access
    always queue-waits; copy_buffer_sync counts its own wait), whereas the async routes
    (pinned H2D/D2H, D2D) never wait. fallback_count is deliberately not used anywhere in
    this module: it is a seam-only counter that no production path increments.
    """
    api.synchronize()
    before = _waits()
    result = call()
    after = _waits()  # read before any other API call that could wait
    assert _pending() == 0 and api.query() is True
    if _async():
        assert sum(after) > sum(before)
    return result


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
    survivor_dst = torch.empty(16, device=vk)
    api.synchronize()
    pinned_before = _pinned_allocations()
    allocations_before = _allocations()  # keeper + survivor_dst
    hold()
    dst = src.to(vk, non_blocking=True)
    survivor_dst.copy_(dst, non_blocking=True)  # D2D that must outlive both wrappers
    del src
    gc.collect()
    assert _pinned_allocations() == pinned_before  # lease keeps the pinned allocation
    del dst
    gc.collect()
    assert _pending() == 1
    assert _allocations() == allocations_before + 1  # lease keeps the deleted destination
    api.test_hold_completion(False)
    api.synchronize()
    assert _pending() == 0 and _pinned_allocations() < pinned_before
    assert _allocations() == allocations_before  # released after completion
    # the surviving D2D copy carries the pinned upload's values
    torch.testing.assert_close(survivor_dst.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close((keeper + 3.0).cpu(), torch.full((16,), 3.0), rtol=0, atol=0)


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
    _blocking_call(lambda: pageable.copy_(device, non_blocking=True))
    torch.testing.assert_close(pageable, torch.arange(6, dtype=torch.float32), rtol=0, atol=0)
    # Ineligible layouts per the Task 0 oracle (async-t0-cpu-probe-03.json): stride-0
    # expand and a stride hole. A transposed dense source is ELIGIBLE (preserve-format
    # output keeps its strides), so it is deliberately not used here.
    expanded = torch.tensor([1.0, 2.0]).expand(3, 2)
    dst = _blocking_call(lambda: expanded.to(vk, non_blocking=True))
    torch.testing.assert_close(dst.cpu(), expanded, rtol=0, atol=0)
    holed = torch.arange(6, dtype=torch.float32).reshape(2, 3)[:, :2]
    dst = _blocking_call(lambda: holed.to(vk, non_blocking=True))
    torch.testing.assert_close(dst.cpu(), holed, rtol=0, atol=0)


def test_wait_counters_discriminate_blocking_from_async_routes(vk):
    if not _async():
        pytest.skip("wait-counter discrimination needs async mode (sync mode waits everywhere)")
    host = _pinned(torch.arange(8, dtype=torch.float32))
    device = torch.arange(8, dtype=torch.float32).to(vk)
    destination = torch.empty(8, device=vk)
    api.synchronize()
    before = _waits()
    uploaded = host.to(vk, non_blocking=True)  # pinned H2D (async)
    destination.copy_(device, non_blocking=True)  # D2D (async)
    assert _waits() == before  # the async routes never waited
    api.synchronize()
    # Blocking counterparts do wait (asserted inside _blocking_call).
    _blocking_call(lambda: torch.zeros(8).copy_(device, non_blocking=True))  # pageable D2H
    _blocking_call(lambda: torch.arange(8, dtype=torch.float32).to(vk))  # blocking H2D
    torch.testing.assert_close(uploaded.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)


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


# --- Task 6: connected second reverse, backward blocking, saved version, layouts -----
# Task 0 CPU oracle: .superpowers/sdd/2026-10-20-vulkan-f32-copy-layout/receipts/
# async-t0-cpu-probe-03.json (result.stdout "autograd"): x=[1,2,3], independent nonzero
# seeds [2,3,5] / [7,11,13]; first reverse 2*x*seed1, connected second reverse 2*seed1*seed2.
# This is the one connected witness the plan names; general higher-order, forward-mode
# and transform support remain unproven.
TASK0_X = [1.0, 2.0, 3.0]
TASK0_SEED1 = [2.0, 3.0, 5.0]
TASK0_SEED2 = [7.0, 11.0, 13.0]
TASK0_FIRST = [4.0, 12.0, 30.0]
TASK0_SECOND = [28.0, 66.0, 130.0]


def _second_reverse_chain(kind, device, check_names=False):
    """first = d(mid*mid)/dx with create_graph=True, second = d(first)/dx; real, not inferred."""
    x = torch.tensor(TASK0_X, requires_grad=True)
    if kind == "to":
        mid = x.to(device, copy=True, non_blocking=True)
        if check_names:
            assert mid.grad_fn.name() == "ToCopyBackward0"
    else:
        mid = torch.empty(3, device=device)
        mid.copy_(x, non_blocking=True)  # preallocated destination
        if check_names:
            assert mid.grad_fn.name() == "torch::autograd::CopyBackwards"
    (first,) = torch.autograd.grad(
        mid * mid, x, grad_outputs=torch.tensor(TASK0_SEED1).to(device), create_graph=True
    )
    assert first.requires_grad  # the first reverse stays connected to x
    (second,) = torch.autograd.grad(first, x, grad_outputs=torch.tensor(TASK0_SEED2))
    return first.detach(), second


def _second_reverse_reference(kind):
    return _second_reverse_chain(kind, "cpu")


@pytest.mark.parametrize("kind", ["to", "copy_"])
def test_connected_second_reverse_matches_cpu_and_task0_witness(vk, kind):
    expected_first, expected_second = _second_reverse_reference(kind)
    torch.testing.assert_close(expected_first, torch.tensor(TASK0_FIRST), rtol=0.003, atol=0.003)
    torch.testing.assert_close(expected_second, torch.tensor(TASK0_SECOND), rtol=0.003, atol=0.003)
    first, second = _second_reverse_chain(kind, vk, check_names=True)
    api.synchronize()
    torch.testing.assert_close(first, expected_first, rtol=0.003, atol=0.003)
    torch.testing.assert_close(second, expected_second, rtol=0.003, atol=0.003)


def test_backward_transfers_are_blocking_by_reference(vk):
    leaf = torch.tensor(TASK0_X, requires_grad=True)
    api.synchronize()
    before_forward = _waits()
    moved = leaf.to(vk, non_blocking=True)
    loss = (moved * moved).sum()  # leaves pending compute for the backward readback
    if _async():
        assert _waits() == before_forward  # nonblocking forward calls never waited
    before_backward = _waits()
    (grad,) = torch.autograd.grad(loss, leaf)
    assert sum(_waits()) > sum(before_backward)  # the backward readback is a blocking transfer
    torch.testing.assert_close(grad, 2.0 * torch.tensor(TASK0_X), rtol=0.003, atol=0.003)


def test_saved_version_is_rejected_after_inplace_nonblocking_copy(vk):
    leaf = torch.tensor(TASK0_X, requires_grad=True)
    destination = torch.empty(3, device=vk)
    destination.copy_(leaf, non_blocking=True)
    out = destination * destination  # saves destination at its current version
    destination.copy_(torch.ones(3), non_blocking=True)  # in-place nonblocking mutation
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        out.sum().backward()
    api.synchronize()


# Degraded (ineligible) layouts per the Task 0 oracle complete before return in both
# flags. A transposed DENSE .to() source is eligible (preserve-format output keeps its
# strides); the ineligible transposed case is copy_ into a contiguous destination.
def _degrade_cases():
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return {
        "stride0_scalar_broadcast": lambda flag, vk: (torch.tensor([2.5]).expand(3, 4), None),
        "offset_non_contiguous": lambda flag, vk: (base[1:, 1::2], None),
        "gapped": lambda flag, vk: (base[:, ::2], None),
        "transposed_copy_into_contiguous": lambda flag, vk: (matrix.t(), torch.empty((3, 2), device=vk)),
    }


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("label", ["stride0_scalar_broadcast", "offset_non_contiguous", "gapped",
                                   "transposed_copy_into_contiguous"])
def test_layout_degrade_matrix_completes_before_return(vk, label, flag):
    source, destination = _degrade_cases()[label](flag, vk)
    if destination is None:
        result = _blocking_call(lambda: source.to(vk, non_blocking=flag))
    else:
        result = _blocking_call(lambda: destination.copy_(source, non_blocking=flag))
    torch.testing.assert_close(result.cpu(), source, rtol=0, atol=0)


# Vulkan -> CPU mirror of the degrade matrix. Sources are as_strided views of a Vulkan
# base (as_strided is a registered, evidenced vk view); expected values come from the
# equivalent CPU views. `.to('cpu', non_blocking=True)` allocates a pinned result but
# these layouts are ineligible, so the call still completes before return.
def _d2h_degrade_cases(vk):
    base = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    base_vk = base.to(vk)
    scalar_vk = torch.tensor([2.5]).to(vk)
    matrix = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    matrix_vk = matrix.to(vk)
    return {
        "stride0_scalar_broadcast": (torch.as_strided(scalar_vk, (3, 4), (0, 0)),
                                     torch.tensor([2.5]).expand(3, 4), None),
        "offset_non_contiguous": (torch.as_strided(base_vk, (3, 3), (6, 2), 7),
                                  base[1:, 1::2], None),
        "gapped": (torch.as_strided(base_vk, (4, 3), (6, 2)), base[:, ::2], None),
        "transposed_copy_into_contiguous": (torch.as_strided(matrix_vk, (3, 2), (1, 3)),
                                            matrix.t(), torch.empty((3, 2))),
    }


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("label", ["stride0_scalar_broadcast", "offset_non_contiguous", "gapped",
                                   "transposed_copy_into_contiguous"])
def test_layout_degrade_matrix_d2h_completes_before_return(vk, label, flag):
    source_vk, expected, destination = _d2h_degrade_cases(vk)[label]
    if destination is None:
        result = _blocking_call(lambda: source_vk.to("cpu", non_blocking=flag))
    else:
        result = _blocking_call(lambda: destination.copy_(source_vk, non_blocking=flag))
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


def test_no_boxed_privateuse1_fallback_is_registered(vk):
    # The real discriminator for "no CPU fallback": with no boxed PrivateUse1 backend
    # fallback registered, an op lacking a Vulkan kernel raises NotImplementedError (as
    # aten::arange did in b1) instead of silently running on the CPU.
    assert not torch._C._dispatch_has_backend_fallback(torch._C.DispatchKey.PrivateUse1)


def _eligible_cases():
    return {
        "transposed_dense": torch.arange(6, dtype=torch.float32).reshape(2, 3).t(),
        "channels_last": torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        .contiguous(memory_format=torch.channels_last),
        "channels_last_3d": torch.arange(48, dtype=torch.float32).reshape(1, 2, 2, 3, 4)
        .contiguous(memory_format=torch.channels_last_3d),
    }


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("label", ["transposed_dense", "channels_last", "channels_last_3d"])
def test_preserve_format_dense_layouts_are_async_eligible(vk, label, flag):
    source = _eligible_cases()[label]
    api.synchronize()
    copies = api.copy_command_count()
    held = _async() and flag  # eligible route: the call returns while completion is held
    if held:
        api.test_hold_completion(True)
    try:
        result = source.to(vk, non_blocking=flag)
        if held:
            # Pending proof without any extra copy, and the single dense region is
            # already recorded (the engine records at enqueue).
            assert _pending() == 1 and api.query() is False
            assert api.copy_command_count() - copies == 1  # one dense region
    finally:
        if held:
            api.test_hold_completion(False)
    api.synchronize()
    if held:
        assert api.completion_hold_expiry_count() == 0
    # Measure the H2D window BEFORE the readback so a multi-command readback cannot
    # be mistaken for (or hide) the upload's region count.
    upload_copies = api.copy_command_count() - copies
    assert result.stride() == source.stride()  # E2: preserve-format output keeps strides
    readback_before = api.copy_command_count()
    torch.testing.assert_close(result.cpu(), source, rtol=0, atol=0)
    readback_copies = api.copy_command_count() - readback_before
    if held:
        assert upload_copies == 1, (label, upload_copies, readback_copies)


def test_repeated_enqueue_sync_cycles_return_to_baseline(vk):
    def cycle():
        host = _pinned(torch.arange(32, dtype=torch.float32))
        up = host.to(vk, non_blocking=True)
        down = up.to("cpu", non_blocking=True)
        copy = torch.empty(32, device=vk)
        copy.copy_(up, non_blocking=True)
        pageable = torch.arange(32, dtype=torch.float32).to(vk, non_blocking=True)
        api.synchronize()
        expected = torch.arange(32, dtype=torch.float32)
        for value in (down, copy.cpu(), pageable.cpu()):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)

    cycle()  # warm shared caches before taking the baseline
    gc.collect()
    api.synchronize()
    baseline = api.live_resource_snapshot()[4:]  # pending transfers/compute, allocations, pinned
    for _ in range(5):
        cycle()
    gc.collect()
    api.synchronize()
    assert api.live_resource_snapshot()[4:] == baseline


def test_sync_mode_accepted_non_blocking_completes_before_return(vk):
    if _async():
        pytest.skip("sync-mode completion witness; async mode asserts pending state instead")
    host = _pinned(torch.arange(8, dtype=torch.float32))
    uploaded = host.to(vk, non_blocking=True)  # pinned H2D
    assert api.query() is True and _pending() == 0
    sink = torch.zeros(8).pin_memory(device="vk")
    sink.copy_(uploaded, non_blocking=True)  # pinned D2H
    assert api.query() is True and _pending() == 0
    torch.testing.assert_close(sink, torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
    pageable = torch.arange(8, dtype=torch.float32).to(vk, non_blocking=True)  # pageable H2D
    assert api.query() is True and _pending() == 0
    clone = torch.empty(8, device=vk)
    clone.copy_(pageable, non_blocking=True)  # D2D
    assert api.query() is True
    torch.testing.assert_close(clone.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)


# --- Review N4 gaps: set_, D2D consumer/lifetime/resize while pending ----------------
def test_set_rebinding_after_enqueue_defers_allocation_release_until_completion(vk, hold):
    # No alias view: after dst.set_(other) the pending destination allocation is
    # reachable only through the engine's lease, so the allocation counter [6] shows
    # whether the lease keeps it alive until completion. The landed bytes of that
    # unreachable allocation are not observable; this proves lifetime, not contents.
    src = _pinned(torch.arange(8, dtype=torch.float32))
    other = torch.zeros(8).to(vk)
    api.synchronize()
    allocations = _allocations()  # includes `other`
    hold()
    dst = src.to(vk, non_blocking=True)
    assert _allocations() == allocations + 1  # the pending destination allocation
    dst.set_(other)  # drops dst's own reference to that allocation
    assert _allocations() == allocations + 1  # only the lease keeps it alive
    assert _pending() == 1
    api.test_hold_completion(False)
    api.synchronize()
    assert _allocations() == allocations  # released once the transfer completed
    torch.testing.assert_close(dst.cpu(), torch.zeros(8), rtol=0, atol=0)


def test_d2d_consumer_enqueued_after_pending_copy_sees_copied_values(vk, hold):
    source = torch.arange(8, dtype=torch.float32).to(vk)
    destination = torch.empty(8, device=vk)
    api.synchronize()
    hold()
    destination.copy_(source, non_blocking=True)
    consumer = destination + 1.0  # compute ordered after the pending D2D
    assert api.query() is False
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(destination.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
    torch.testing.assert_close(consumer.cpu(), torch.arange(8, dtype=torch.float32) + 1.0, rtol=0, atol=0)


def test_d2d_source_and_destination_deleted_while_pending(vk, hold):
    source = torch.arange(8, dtype=torch.float32).to(vk)
    destination = torch.empty(8, device=vk)
    tail = torch.empty(8, device=vk)
    api.synchronize()
    allocations = _allocations()  # source + destination + tail
    hold()
    destination.copy_(source, non_blocking=True)
    tail.copy_(destination, non_blocking=True)  # chained D2D that outlives both wrappers
    del source, destination
    gc.collect()
    assert _allocations() == allocations  # leases keep both deleted allocations alive
    api.test_hold_completion(False)
    api.synchronize()
    assert _allocations() == allocations - 2
    torch.testing.assert_close(tail.cpu(), torch.arange(8, dtype=torch.float32), rtol=0, atol=0)


def test_d2d_then_resize_growth_preserves_prefix(vk, hold):
    source = torch.arange(8, dtype=torch.float32).to(vk)
    destination = torch.empty(8, device=vk)
    api.synchronize()
    hold()
    destination.copy_(source, non_blocking=True)
    destination.resize_(16)  # hold active, like the accepted H2D resize witness
    api.test_hold_completion(False)
    api.synchronize()
    torch.testing.assert_close(destination.cpu()[:8], torch.arange(8, dtype=torch.float32), rtol=0, atol=0)
