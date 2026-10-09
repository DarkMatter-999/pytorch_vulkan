"""Finite batched/broadcast matmul acceptance through stock generated autograd."""
import contextlib
import json

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from general_matmul_cases import (
    GENERAL_MATMUL_CASES, call_matmul, cpu_reference, make_cpu_bases,
    make_operands, random_tensor, tensor_fact,
)

ERRORS = []
ROUTES = []


@pytest.fixture(scope="module")
def backend():
    import pytorch_vulkan

    assert pytorch_vulkan.is_available(), "general matmul requires the selected Vulkan device"
    yield pytorch_vulkan
    pytorch_vulkan._C.synchronize()
    print("GENERAL_MATMUL_ACCEPTANCE " + json.dumps({"max_tensor_error": max(ERRORS, default=0.),
        "comparisons": len(ERRORS), "routes": ROUTES}, sort_keys=True))


def compare(backend, actual, expected):
    backend._C.synchronize()
    assert str(actual.device) == "vk:0"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.requires_grad == expected.requires_grad
    assert (actual.grad_fn is None) == (expected.grad_fn is None)
    readback = actual.detach().cpu()
    reference = expected.detach()
    ERRORS.append((readback-reference).abs().max().item() if readback.numel() else 0.)
    torch.testing.assert_close(readback, reference, rtol=.003, atol=.003)


@pytest.mark.parametrize("case", GENERAL_MATMUL_CASES, ids=lambda c: c["id"])
def test_general_matmul(backend, case):
    cpu = make_cpu_bases(case)
    inputs = make_operands(case, cpu)
    vk = tuple(x.to("vk:0") for x in cpu)
    operands = make_operands(case, vk)
    for x, y in zip(inputs, operands):
        assert tuple(x.shape) == tuple(y.shape)
        assert x.stride() == y.stride()
        assert x.storage_offset() == y.storage_offset()
    mode = torch.no_grad if case["no_grad"] else contextlib.nullcontext
    with mode():
        with profile(activities=[ProfilerActivity.CPU]) as cpu_trace:
            expected = call_matmul(case, *inputs)
        backend._C.synchronize()
        backend._C.reset_execution_counters()
        with profile(activities=[ProfilerActivity.CPU]) as trace:
            actual = call_matmul(case, *operands)
        backend._C.synchronize()
        counters = backend._C.execution_counter_snapshot()
        assert counters[2:] == (0, 0) or list(counters[2:]) == [0, 0]
        events = {event.key for event in trace.key_averages()}
        cpu_events = {event.key for event in cpu_trace.key_averages()}
        assert "aten::" + case["route"] in events
        assert "aten::" + case["route"] in cpu_events
        if case["copy_route"]:
            assert {"aten::clone", "aten::copy_"} <= events
            assert {"aten::clone", "aten::copy_"} <= cpu_events
        assert counters[1] >= case["materialization_minimum"]
        ROUTES.append({"id": case["id"], "route": case["route"],
                       "cpu": sorted(cpu_events), "vk": sorted(events),
                       "counters": list(counters)})
    compare(backend, actual, expected)
    targets = tuple(i for i, x in enumerate(cpu) if x.requires_grad)
    if targets and not case["no_grad"]:
        seed = random_tensor(expected.shape, case["seed"] + 1)
        cg = torch.autograd.grad(expected, tuple(cpu[i] for i in targets), seed, create_graph=True)
        vk_seed = seed.to("vk:0")
        backend._C.synchronize()
        backend._C.reset_execution_counters()
        vg = torch.autograd.grad(actual, tuple(vk[i] for i in targets), vk_seed, create_graph=True)
        backend._C.synchronize()
        assert list(backend._C.execution_counter_snapshot()[2:]) == [0, 0]
        for i, a, b in zip(targets, vg, cg):
            assert a.shape == cpu[i].shape
            compare(backend, a, b)


MIXED = tuple(c for c in GENERAL_MATMUL_CASES if c["mixed"])


@pytest.mark.parametrize("case", MIXED, ids=lambda c: c["id"])
def test_mixed_reverse_independent_fd(backend, case):
    reference = cpu_reference(case)
    bases = tuple(x.to("vk:0") for x in make_cpu_bases(case))
    output = call_matmul(case, *make_operands(case, bases))
    seed = random_tensor(output.shape, case["seed"] + 1).to("vk:0")
    gradients = torch.autograd.grad(output, bases, seed, create_graph=True)
    for row in reference["mixed"]:
        first, opposite = row["direction"]
        actual = torch.autograd.grad(gradients[first], bases[opposite],
                                     row["probe"].to("vk:0"), retain_graph=True)[0]
        compare(backend, actual, row["tensor"])
        assert row["tensor"].count_nonzero().item() > 0
        assert abs(row["projection"]) > 1e-5
        torch.testing.assert_close(torch.tensor(row["projection"]), torch.tensor(row["fd"]),
                                   rtol=.008, atol=.002)


MUTATION = tuple(c for c in GENERAL_MATMUL_CASES
                 if c["selection"] == "both" and c["api"] == "matmul"
                 and not c["no_grad"] and c["family"] in
                 ("batch-matrix", "offset-fold", "cross-broadcast"))


@pytest.mark.parametrize("case", MUTATION, ids=lambda c: c["id"])
@pytest.mark.parametrize("mutated", (0, 1))
def test_saved_storage_mutation(backend, case, mutated):
    reference = cpu_reference(case)
    for device in ("cpu", "vk:0"):
        bases = tuple(x.to(device) for x in make_cpu_bases(case))
        output = call_matmul(case, *make_operands(case, bases))
        versions = tuple(x._version for x in bases)
        before = bases[mutated].detach().clone()
        with torch.no_grad():
            bases[mutated].add_(1.)
        assert bases[mutated]._version == versions[mutated] + 1
        assert bases[1-mutated]._version == versions[1-mutated]
        if device == "vk:0":
            backend._C.synchronize()
        torch.testing.assert_close(bases[mutated].detach().cpu(), before.cpu() + 1)
        seed = random_tensor(output.shape, case["seed"] + 1).to(device)
        if case["family"] != "cross-broadcast":
            with pytest.raises(RuntimeError, match="modified by an inplace operation"):
                torch.autograd.grad(output, bases, seed)
        else:
            gradients = torch.autograd.grad(output, bases, seed)
            for actual, expected in zip(gradients, reference["gradients"]):
                if device == "vk:0":
                    compare(backend, actual, expected.detach())
                else:
                    torch.testing.assert_close(actual, expected.detach())


@pytest.mark.parametrize("shapes", [((2,3,4),(2,5,2)), ((2,3,4),(3,4,2)),
    ((),(2,3,4)), ((2,3,4),()), ((2,0,4),(2,5,2)),
    ((0,3,4),(3,4,2)), ((2,3,0),(2,1,2))])
@pytest.mark.parametrize("api", ("matmul", "@"))
def test_malformed_preflight(backend, shapes, api):
    cpu = tuple(random_tensor(s, 901+i) for i, s in enumerate(shapes))
    case = {"api": api}
    with pytest.raises(RuntimeError):
        call_matmul(case, *cpu)
    vk = tuple(x.to("vk:0") for x in cpu)
    backend._C.synchronize()
    backend._C.reset_execution_counters()
    before = backend._C.execution_counter_snapshot()
    with pytest.raises(RuntimeError):
        call_matmul(case, *vk)
    backend._C.synchronize()
    assert backend._C.execution_counter_snapshot() == before


@pytest.mark.parametrize("boundary", ("dtype", "cpu-left", "cpu-right"))
def test_checked_leaf_boundaries(backend, boundary):
    a = random_tensor((2,3,4), 991)
    b = random_tensor((4,2), 992)
    if boundary == "dtype":
        b = b.double()
        with pytest.raises(RuntimeError):
            a @ b
    a = a if boundary == "cpu-left" else a.to("vk:0")
    b = (b if boundary == "cpu-right" else
         (b.float().to("vk:0").double() if boundary == "dtype" else b.to("vk:0")))
    backend._C.synchronize()
    backend._C.reset_execution_counters()
    with pytest.raises(RuntimeError):
        a @ b
    backend._C.synchronize()
    assert backend._C.compute_dispatch_count() == 0
    assert backend._C.fallback_count() == 0


def test_source_owned_finite_metadata():
    assert len({c["id"] for c in GENERAL_MATMUL_CASES}) == len(GENERAL_MATMUL_CASES)
    for case in GENERAL_MATMUL_CASES:
        bases = make_cpu_bases(case)
        operands = make_operands(case, bases)
        assert tuple(tensor_fact(x) for x in operands) == case["operand_facts"]
        assert tuple(call_matmul(case, *operands).shape) == case["output_shape"]
