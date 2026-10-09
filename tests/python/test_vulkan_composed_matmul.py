"""Connected ordinary-module acceptance; CPU recipes are separately importable."""
import json
import os
from contextlib import contextmanager

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from composed_matmul_cases import (
    COMPOSED_MATMUL_SCENARIOS, make_model_case, model_graph, model_reference,
    model_metadata, mutation_case, training_steps,
)


@pytest.fixture(scope="module")
def backend():
    import pytorch_vulkan
    assert pytorch_vulkan.is_available(), "connected models require the selected Vulkan device"
    yield pytorch_vulkan
    pytorch_vulkan._C.synchronize()


@pytest.mark.parametrize("scenario", COMPOSED_MATMUL_SCENARIOS, ids=lambda s: s["id"])
def test_composed_model(backend, scenario):
    cpu_routes, vk_routes = {}, {}
    cpu_shapes, vk_shapes = {}, {}

    def phases(destination, shapes):
        @contextmanager
        def phase(name):
            try:
                with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as trace:
                    yield
            finally:
                destination[name] = sorted(e.key for e in trace.key_averages())
                shapes[name] = [{"op": e.name, "inputs": e.input_shapes}
                                for e in trace.events() if e.name in
                                {"aten::mm", "aten::bmm", "aten::mv"}]
        return phase

    reference = model_reference(scenario, phase=phases(cpu_routes, cpu_shapes))
    context = make_model_case("vk:0")
    assert {**model_metadata(context), "scenario": dict(scenario)} == reference["metadata"]
    backend._C.synchronize()
    backend._C.reset_execution_counters()
    if scenario["kind"] == "training":
        actual = training_steps(context, scenario["set_to_none"], phases(vk_routes, vk_shapes))
    elif scenario["kind"] == "mutation":
        actual = mutation_case(context, scenario["mutated"], phases(vk_routes, vk_shapes))
    else:
        actual = model_graph(context, scenario, phases(vk_routes, vk_shapes))
    backend._C.synchronize()
    counters = backend._C.execution_counter_snapshot()
    assert tuple(counters[2:]) == (0, 0), counters
    assert backend._C.execution_mode() == (
        "sync" if os.getenv("PYTORCH_VULKAN_ASYNC_EXECUTION") == "0" else "async")
    assert cpu_routes.keys() == vk_routes.keys()
    for name, routes in vk_routes.items():
        expected_routes = set(cpu_routes[name]) & {"aten::mm", "aten::bmm", "aten::mv",
                                                  "aten::mse_loss", "aten::mse_loss_backward"}
        assert expected_routes <= set(routes), (name, expected_routes, routes)
        if name.endswith("forward"):
            assert {"aten::mm", "aten::bmm", "aten::mv"} <= set(routes)
            witnesses = [{"op": "aten::mm", "inputs": [[4, 4], [4, 3]]},
                         {"op": "aten::bmm", "inputs": [[6, 2, 3], [6, 3, 2]]},
                         {"op": "aten::bmm", "inputs": [[6, 2, 2], [6, 2, 3]]},
                         {"op": "aten::mv", "inputs": [[12, 3], [3]]}]
            assert all(w in cpu_shapes[name] and w in vk_shapes[name] for w in witnesses)
    errors = []
    histories, value_signals = {}, {}

    def compare(a, b, path="values"):
        if isinstance(b, torch.Tensor):
            assert a.device == torch.device("vk:0")
            assert a.shape == b.shape and a.dtype == b.dtype
            assert a.requires_grad == b.requires_grad
            assert (a.grad_fn is None) == (b.grad_fn is None)
            if b.grad_fn is not None:
                assert type(a.grad_fn).__name__ == type(b.grad_fn).__name__
            readback = a.detach().cpu()
            torch.testing.assert_close(readback, b.detach(), rtol=.003, atol=.003)
            errors.append(float((readback - b.detach()).abs().max()))
            histories[path] = {"requires_grad": a.requires_grad,
                               "grad_fn": None if a.grad_fn is None else type(a.grad_fn).__name__,
                               "shape": list(a.shape), "device": str(a.device)}
            value_signals[path] = {"cpu_max_abs": float(b.detach().abs().max()),
                                  "vk_max_abs": float(readback.abs().max()),
                                  "cpu_nonzero": int(b.detach().count_nonzero()),
                                  "vk_nonzero": int(readback.count_nonzero())}
            if ".first." in path or ".second." in path or ".gradients." in path:
                assert b.detach().count_nonzero() > 0
            if path == "values.first.input":
                assert b.detach()[0].count_nonzero() == 0
                assert readback[0].count_nonzero() == 0
        elif isinstance(b, dict):
            assert a.keys() == b.keys()
            for key in b:
                compare(a[key], b[key], f"{path}.{key}")
        elif isinstance(b, (list, tuple)):
            assert len(a) == len(b)
            for index, (av, bv) in enumerate(zip(a, b)):
                compare(av, bv, f"{path}.{index}")
        else:
            assert a == b

    compare(actual, reference["values"])
    for row in reference["fd"]:
        assert abs(row["fd"]) > .002, row
        torch.testing.assert_close(torch.tensor(row["analytic"]), torch.tensor(row["fd"]),
                                   rtol=.008, atol=.002)
    if scenario["kind"] in ("first", "mixed"):
        assert actual["first"]["input"].shape == (3, 1, 4, 2)
        assert set(actual["first"]) == {"input", "shared", "down", "up", "readout"}
        assert all(g.requires_grad and g.grad_fn is not None for g in actual["first"].values())
    if scenario["kind"] == "hvp":
        assert set(actual["first"]) == set(actual["second"]) == {"shared", "down", "up", "readout"}
        assert all(g.requires_grad and g.grad_fn is not None for g in actual["first"].values())
    if scenario["kind"] == "training":
        options = actual["optimizer_options"]
        assert options["foreach"] is None and options["fused"] is None
        for index, step in enumerate(reference["values"]["steps"]):
            assert step["state_parameter_keys"] == [0, 1, 2, 3]
            assert all(set(state) == {"momentum_buffer"} for state in step["state"].values())
            for gradient in step["reset"].values():
                if index == 0 or scenario["set_to_none"]:
                    assert gradient is None
                else:
                    assert gradient is not None and gradient.count_nonzero() == 0
        for gradient in reference["values"]["final_reset"].values():
            assert gradient is None if scenario["set_to_none"] else gradient.count_nonzero() == 0
    print("COMPOSED_MODEL_METRICS " + json.dumps({
        "id": scenario["id"], "mode": backend._C.execution_mode(),
        "counters_before_readback": counters, "routes": vk_routes, "cpu_routes": cpu_routes,
        "max_abs_error": max(errors, default=0.), "fd": reference["fd"],
        "metadata": reference["metadata"], "histories": histories,
        "value_signals": value_signals, "cpu_route_shapes": cpu_shapes,
        "vk_route_shapes": vk_shapes,
        "mutation_outcome": actual.get("outcome"),
        "mutation_version_delta": actual.get("version_delta"),
        "optimizer_options": actual.get("optimizer_options"),
    }, sort_keys=True))
