import json
import os

import pytest
import torch
import pytorch_vulkan

from mse_autograd_cases import (MSE_AUTOGRAD_CASES, make_mse_case, mse_forward,
                                mse_graph, mse_reference, seeded)


@pytest.fixture
def vk():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def checked_sync():
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[2:] == (0, 0), counters
    return counters


@pytest.mark.parametrize("api", ["functional", "module"])
@pytest.mark.parametrize("case", MSE_AUTOGRAD_CASES, ids=lambda c: c["id"])
def test_public_mse_selected_reverse_and_independent_fd(vk, case, api):
    reference = mse_reference(case, api=api)
    context = make_mse_case(case, vk)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    graph = mse_graph(case, context, api=api)
    assert graph["loss"].grad_fn is not None
    assert all(g.requires_grad and g.grad_fn is not None for g in graph["first"])
    assert type(graph["loss"].grad_fn).__name__ == type(reference["graph"]["loss"].grad_fn).__name__
    assert [type(g.grad_fn).__name__ for g in graph["first"]] == [
        type(g.grad_fn).__name__ for g in reference["graph"]["first"]]
    second = torch.autograd.grad(graph["contraction"], context["bases"])
    values = (graph["loss"], *graph["first"], *second)
    expected = (reference["graph"]["loss"], *reference["graph"]["first"],
                *reference["second"])
    assert all(v.device == torch.device(vk) for v in values)
    counters = checked_sync()
    errors = []
    for actual, want in zip(values, expected):
        readback = actual.cpu()
        torch.testing.assert_close(readback, want, rtol=.003, atol=.003,
                                   equal_nan=True)
        difference = (readback.detach() - want.detach()).abs()
        errors.append(float(difference[torch.isfinite(difference)].max())
                      if torch.isfinite(difference).any() else 0.)
    for row in reference["fd"]:
        torch.testing.assert_close(torch.tensor(row["analytic"]),
                                   torch.tensor(row["fd"]), rtol=.008, atol=.002)
        if case["kind"] != "empty":
            assert abs(row["fd"]) > 1e-6, row
    print("MSE_METRICS " + json.dumps({
        "case": case["id"], "api": api, "counters_before_readback": counters,
        "mode": pytorch_vulkan._C.execution_mode(),
        "max_abs_error": max(errors),
        "fd_max_abs_error": max(abs(r["analytic"] - r["fd"]) for r in reference["fd"]),
        "fd_nonzero": sum(abs(r["fd"]) > 1e-6 for r in reference["fd"]),
        "fd_min_nonzero": min((abs(r["fd"]) for r in reference["fd"]
                               if abs(r["fd"]) > 1e-6), default=0.),
    }))
    assert pytorch_vulkan._C.execution_mode() == (
        "sync" if os.getenv("PYTORCH_VULKAN_ASYNC_EXECUTION") == "0" else "async")


@pytest.mark.parametrize("case", MSE_AUTOGRAD_CASES[::3], ids=lambda c: c["id"])
def test_public_mse_no_grad(vk, case):
    cpu = make_mse_case(case)
    context = make_mse_case(case, vk)
    pytorch_vulkan._C.reset_execution_counters()
    with torch.no_grad():
        actual = mse_forward(case, context["bases"], "module")
        expected = mse_forward(case, cpu["bases"], "module")
    assert not actual.requires_grad and actual.grad_fn is None
    checked_sync()
    torch.testing.assert_close(actual.cpu(), expected, rtol=.003, atol=.003,
                               equal_nan=True)


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("layout", ["scalar", "singleton", "offset", "tail", "strided-singleton"])
def test_direct_backward_scalar_upstream_avoids_copy(vk, reduction, layout):
    x, t = seeded((2, 3), 101), seeded((2, 3), 211)
    base = seeded((3,), 307) if layout in ("offset", "tail", "strided-singleton") else seeded((), 307)
    def view(b):
        if layout == "offset":
            return b[1:2]
        if layout == "tail":
            return b[2:3]
        if layout == "strided-singleton":
            return b[1:2:2].reshape(1, 1)
        return b.reshape(1, 1) if layout == "singleton" else b
    u = view(base)
    xx, tt, bb = x.to(vk), t.to(vk), base.to(vk)
    uu = view(bb)
    expected = torch.ops.aten.mse_loss_backward(u, x, t, reduction)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.mse_loss_backward(uu, xx, tt, reduction)
    counters = checked_sync()
    assert counters == (1, 0, 0, 0), counters
    torch.testing.assert_close(actual.cpu(), expected, rtol=.003, atol=.003)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_public_nonuniform_selected_second_seed(vk, reduction):
    results = []
    for device in ("cpu", vk):
        x = seeded((2, 3), 101).to(device).requires_grad_()
        t = seeded((2, 3), 211).to(device).requires_grad_()
        u = seeded((2, 3) if reduction == "none" else (), 307).to(device).requires_grad_()
        p = seeded((2, 3), 401).to(device)
        if device == vk:
            pytorch_vulkan._C.synchronize()
            pytorch_vulkan._C.reset_execution_counters()
        loss = torch.nn.functional.mse_loss(x, t, reduction=reduction)
        first = torch.autograd.grad(loss, (x, t), u, create_graph=True)
        assert all(g.requires_grad and g.grad_fn is not None for g in first)
        second = torch.autograd.grad(first[0], (x, t, u), p)
        if device == vk:
            checked_sync()
        results.append(tuple(g.detach().cpu() for g in (*first, *second)))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=.003, atol=.003)
    assert torch.count_nonzero(results[0][2]) == 6
    assert torch.unique(results[0][2]).numel() > 1


@pytest.mark.parametrize("reduction", [0, 1, 2])
def test_direct_empty_backward_scalar_tail_has_no_device_work(vk, reduction):
    x, t, base = torch.empty(0, 3), torch.empty(0, 3), seeded((3,), 307)
    xx, tt, bb = x.to(vk), t.to(vk), base.to(vk)
    expected = torch.ops.aten.mse_loss_backward(base[2:3], x, t, reduction)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.mse_loss_backward(bb[2:3], xx, tt, reduction)
    assert checked_sync() == (0, 0, 0, 0)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize("reduction", [0, 1, 2])
@pytest.mark.parametrize("layout", ["scalar", "row", "full", "expanded", "transpose"])
def test_direct_backward_broadcast_gradient_addressing(vk, reduction, layout):
    x, t = seeded((2, 3), 101), seeded((2, 3), 211)
    u = seeded((), 307) if layout == "scalar" else seeded((1, 3), 307) if layout in ("row", "expanded") else seeded((2, 3), 307)
    xx, tt, uu = x.to(vk), t.to(vk), u.to(vk)
    if layout == "expanded":
        u, uu = u.expand(2, 3), uu.expand(2, 3)
    if layout == "transpose":
        x, t, u = x.t(), t.t(), u.t()
        xx, tt, uu = xx.t(), tt.t(), uu.t()
    expected = torch.ops.aten.mse_loss_backward(u, x, t, reduction)
    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.ops.aten.mse_loss_backward(uu, xx, tt, reduction)
    counters = checked_sync()
    assert counters[1] == {"scalar": 0, "row": 1, "full": 0,
                           "expanded": 1, "transpose": 3}[layout], counters
    torch.testing.assert_close(actual.cpu(), expected, rtol=.003, atol=.003)


@pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
def test_public_sum_seeded_second_reverse(vk, reduction):
    case = {"kind": "ordinary", "reduction": reduction, "selection": "input"}
    outputs = []
    for device in ("cpu", vk):
        context = make_mse_case(case, device)
        if device == vk:
            pytorch_vulkan._C.reset_execution_counters()
        loss = mse_forward(case, context["bases"])
        first = torch.autograd.grad(loss, context["bases"][0], context["bases"][2],
                                    create_graph=True)[0]
        second = torch.autograd.grad(first.sum(), context["bases"])
        if device == vk:
            checked_sync()
        outputs.append(tuple(g.detach().cpu() for g in second))
    for actual, expected in zip(outputs[1], outputs[0]):
        torch.testing.assert_close(actual, expected, rtol=.003, atol=.003)


@pytest.mark.parametrize("kind", ["ordinary", "broadcast"])
@pytest.mark.parametrize("phase", ["before-first", "before-second"])
@pytest.mark.parametrize("target", [0, 1, 2])
def test_public_mse_saved_original_base_versions(vk, kind, phase, target):
    case = {"kind": kind, "reduction": "mean", "selection": "both"}
    outcomes = []
    for device in ("cpu", vk):
        context = make_mse_case(case, device)
        bases = context["bases"]
        if phase == "before-first":
            loss = mse_forward(case, bases)
        else:
            graph = mse_graph(case, context)
        version = bases[target]._version
        if device == vk:
            pytorch_vulkan._C.reset_execution_counters()
        with torch.no_grad():
            bases[target].add_(.125)
        assert bases[target]._version == version + 1
        try:
            result = (torch.autograd.grad(loss, bases[:2], bases[2], create_graph=True)
                      if phase == "before-first" else
                      torch.autograd.grad(graph["contraction"], bases))
        except RuntimeError as error:
            assert "modified by an inplace operation" in str(error)
            outcomes.append(None)
            if device == vk:
                checked_sync()
        else:
            if device == vk:
                checked_sync()
            outcomes.append(tuple(g.detach().cpu() for g in result))
    assert (outcomes[0] is None) == (outcomes[1] is None)
    if outcomes[0] is not None:
        for actual, expected in zip(outcomes[1], outcomes[0]):
            torch.testing.assert_close(actual, expected, rtol=.003, atol=.003)


@pytest.mark.parametrize("case", ["shape", "dtype", "device", "reduction"])
def test_backward_invalid_contract_preflight(vk, case):
    x, t = seeded((2, 3), 101).to(vk), seeded((2, 3), 211).to(vk)
    u = seeded((4,) if case == "shape" else (), 307)
    if case == "dtype":
        u = u.to(torch.bool)
    if case != "device":
        u = u.to(vk)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        torch.ops.aten.mse_loss_backward(u, x, t, -1 if case == "reduction" else 1)
    assert checked_sync()[0] == 0
