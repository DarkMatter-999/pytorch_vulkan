import torch
import torch.nn as nn
import pytest
import os

import vulkan_workload_conformance as workload


def test_source_contract_has_exact_six_scenario_mode_pairs():
    expected_ids = {
        "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none",
        "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero",
        "hvp.grouped-depthwise.output-energy.parameters",
    }
    assert {scenario.workload_id for scenario in workload.SCENARIOS} == expected_ids
    assert all(scenario.required_modes == ("async", "sync") for scenario in workload.SCENARIOS)
    assert {scenario.workload_id: scenario.reset_mode for scenario in workload.SCENARIOS} == {
        "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none": "none",
        "classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero": "zero",
        "hvp.grouped-depthwise.output-energy.parameters": None,
    }
    assert workload.REQUIRED_RECORD_KEYS == frozenset(
        {
            ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none", "none", "async"),
            ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-none", "none", "sync"),
            ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero", "zero", "async"),
            ("classifier.grouped-depthwise.ce.sgd-momentum.zero-grad-zero", "zero", "sync"),
            ("hvp.grouped-depthwise.output-energy.parameters", None, "async"),
            ("hvp.grouped-depthwise.output-energy.parameters", None, "sync"),
        }
    )


def test_fixtures_are_stock_modules_with_local_seeded_cpu_inputs():
    torch.manual_seed(91)
    rng_before = torch.random.get_rng_state().clone()
    classifier, classifier_input, labels = workload.classifier_fixture()
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert isinstance(classifier, nn.Sequential)
    assert [type(layer) for layer in classifier] == [
        nn.Conv2d, nn.ReLU, nn.Conv2d, nn.ReLU, nn.Flatten, nn.Linear
    ]
    assert tuple(classifier_input.shape) == (2, 4, 5, 5)
    assert classifier_input.dtype == torch.float32 and classifier_input.device.type == "cpu"
    assert labels.dtype == torch.int64 and labels.tolist() == [0, 2]

    hvp_model, hvp_input, directions = workload.hvp_fixture()
    assert isinstance(hvp_model, nn.Sequential)
    assert [type(layer) for layer in hvp_model] == [nn.Conv2d, nn.Conv2d]
    assert tuple(hvp_input.shape) == (2, 4, 5, 5)
    params = tuple(hvp_model.parameters())
    assert len(directions) == len(params) == 3
    assert all(direction.shape == parameter.shape for direction, parameter in zip(directions, params))
    assert all(direction.dtype == torch.float32 and direction.device.type == "cpu" for direction in directions)
    for index, (parameter, direction) in enumerate(zip(params, directions)):
        expected = torch.randn(
            parameter.shape,
            dtype=torch.float32,
            generator=torch.Generator(device="cpu").manual_seed(200 + index),
        )
        assert torch.equal(direction, expected)


def test_zero_grad_boundary_observes_reference_reset_contract():
    none_run = workload.run_cpu_classifier("none")
    zero_run = workload.run_cpu_classifier("zero")
    assert len(none_run["steps"]) == len(zero_run["steps"]) == 3
    assert [step["loss"]["values"][0] for step in none_run["steps"]] == [
        1.0897274017333984,
        1.0827380418777466,
        1.0696792602539062,
    ]
    assert none_run["reset_observations"][1]["all_grad_none"] is True
    assert zero_run["reset_observations"][0]["all_grad_none"] is True
    assert zero_run["reset_observations"][2]["same_grad_objects"] is True
    assert zero_run["reset_observations"][2]["all_grad_zero"] is True
    assert none_run["initial_state_keys"] == zero_run["initial_state_keys"] == {}
    assert none_run["steps"][0]["state_keys"] == {
        "0.weight": ["momentum_buffer"],
        "2.weight": ["momentum_buffer"],
        "2.bias": ["momentum_buffer"],
        "5.weight": ["momentum_buffer"],
        "5.bias": ["momentum_buffer"],
    }
    for none_step, zero_step in zip(none_run["steps"], zero_run["steps"]):
        assert none_step["loss"] == zero_step["loss"]
        assert none_step["gradients"] == zero_step["gradients"]
        assert none_step["parameters"] == zero_step["parameters"]
        assert none_step["momentum"] == zero_step["momentum"]


def test_cpu_classifier_restores_thread_count_and_rng_after_oracle():
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    torch.manual_seed(438)
    rng_before = torch.random.get_rng_state().clone()
    try:
        workload.run_cpu_classifier("none")
        assert torch.get_num_threads() == 2
        assert torch.equal(torch.random.get_rng_state(), rng_before)
    finally:
        torch.set_num_threads(original_threads)


def test_hvp_cpu_oracle_checks_each_parameter_role_against_centered_fd():
    original_threads = torch.get_num_threads()
    torch.manual_seed(438)
    rng_before = torch.random.get_rng_state().clone()
    oracle = workload.run_cpu_hvp()
    assert torch.get_num_threads() == original_threads
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert set(oracle["hvp"]) == {"0.weight", "1.weight", "1.bias"}
    assert all(oracle["hvp"][name]["nonzero_count"] > 0 for name in oracle["hvp"])
    assert set(oracle["first_gradients"]) == set(oracle["hvp"])
    assert all(oracle["history"]["first_gradients_require_grad"].values())
    assert oracle["finite_difference"]["checked"] is True
    assert oracle["finite_difference"]["epsilon"] == 1e-4


def test_fixture_metadata_pins_public_module_and_loss_defaults():
    classifier = workload.fixture_metadata(workload.CLASSIFIER_NONE_ID)
    hvp = workload.fixture_metadata(workload.HVP_ID)
    assert classifier["model"][0]["stride"] == [1, 1]
    assert classifier["model"][0]["dilation"] == [1, 1]
    assert classifier["model"][0]["padding_mode"] == "zeros"
    assert classifier["model"][1]["inplace"] is False
    assert classifier["model"][4]["end_dim"] == -1
    assert classifier["loss_options"] == {
        "weight": None, "size_average": None, "ignore_index": -100,
        "reduce": None, "reduction": "mean", "label_smoothing": 0.0,
    }
    assert classifier["optimizer"] == {
        "type": "SGD", "lr": 0.01, "momentum": 0.9, "dampening": 0.0,
        "weight_decay": 0.0, "nesterov": False, "maximize": False,
        "foreach": None, "differentiable": False, "fused": None,
    }
    assert hvp["model"][0]["stride"] == [1, 1]
    assert hvp["model"][1]["padding_mode"] == "zeros"


def test_snapshot_helper_orders_sync_snapshot_before_readback():
    events = []
    counters, payload, order = workload._synchronized_observation(
        lambda: events.append("sync"),
        lambda: events.append("snapshot") or (4, 0, 0, 0),
        lambda snapshot: events.append("readback") or "payload",
    )
    assert events == ["sync", "snapshot", "readback"]
    assert counters == (4, 0, 0, 0)
    assert payload == "payload"
    assert order == ["sync", "snapshot", "readback"]


def _vulkaninfo_device_block(index, *, name, driver_name, driver_info, fields_first=False):
    fields = [
        f"\tdeviceName         = {name}",
        f"\tdriverName         = {driver_name}",
        f"\tdriverInfo         = {driver_info}",
    ]
    if fields_first:
        fields.reverse()
    return f"GPU{index}:\n" + "\n".join(fields)


def test_vulkaninfo_metadata_parser_binds_complete_fields_to_one_gpu_block():
    summary = """VULKANINFO
deviceName = false header
Devices:
========
""" + _vulkaninfo_device_block(
        0, name="AMD Radeon Graphics", driver_name="radv", driver_info="Mesa 26.2.3",
        fields_first=True,
    )

    assert workload._parse_vulkan_device_summary(summary) == {
        "hardware": "AMD Radeon Graphics",
        "driver": "radv Mesa 26.2.3",
    }


@pytest.mark.parametrize("blocks", [
    _vulkaninfo_device_block(0, name="unsupported", driver_name="", driver_info="")
    .replace("\tdriverName         = \n", "")
    .replace("\tdriverInfo         = \n", "") + "\n" +
    _vulkaninfo_device_block(1, name="usable", driver_name="radv", driver_info="Mesa"),
    _vulkaninfo_device_block(0, name="first", driver_name="radv", driver_info="Mesa") + "\n" +
    _vulkaninfo_device_block(1, name="second", driver_name="", driver_info=""),
])
def test_vulkaninfo_metadata_parser_rejects_multigpu_summaries(blocks):
    with pytest.raises(RuntimeError, match="exactly one GPU device block"):
        workload._parse_vulkan_device_summary("Devices:\n========\n" + blocks)


@pytest.mark.parametrize("summary", [
    "Devices:\n========\n",
    "Devices:\n========\n" + _vulkaninfo_device_block(0, name="gpu", driver_name="radv", driver_info="Mesa") + "\nGPU0:\n",
    "Devices:\n========\n" + _vulkaninfo_device_block(0, name="gpu", driver_name="radv", driver_info="Mesa") + "\n\tdriverInfo = duplicate",
    "Devices:\n========\nGPU0:\n\tdeviceName = gpu\n\tdriverName = radv",
    "deviceName = fake\nGPU0:\n\tdeviceName = fake\n\tdriverName = radv\n\tdriverInfo = Mesa",
])
def test_vulkaninfo_metadata_parser_rejects_missing_duplicate_or_out_of_section_fields(summary):
    with pytest.raises(RuntimeError):
        workload._parse_vulkan_device_summary(summary)


@pytest.fixture
def workload_device():
    import pytorch_vulkan
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_stock_sgd_executes_three_steps_in_both_reset_modes(workload_device):
    import pytorch_vulkan
    expected = "sync" if os.getenv("PYTORCH_VULKAN_ASYNC_EXECUTION") == "0" else "async"
    assert pytorch_vulkan._C.execution_mode() == expected
    for reset_mode in ("none", "zero"):
        record = workload.run_vulkan_classifier(reset_mode)
        workload.validate_workload_record(record)
        assert record["execution_mode"] == expected
        assert len(record["vulkan"]["steps"]) == 3
        assert all(step["output"]["device"] == "vk:0" for step in record["vulkan"]["steps"])
        assert [window["phase"] for window in record["execution"]] == [
            phase for _ in range(3) for phase in ("reset", "forward_backward", "optimizer")
        ]
        for window in record["execution"]:
            assert window["observation_order"] == ["sync", "snapshot", "readback"]
            assert window["synchronized_before_snapshot"] and window["snapshot_before_readback"]
            assert window["fallbacks"] == window["explicit_transfers"] == 0
            if window["phase"] == "forward_backward":
                assert window["dispatches"] > 0
            else:
                assert window["dispatches"] >= 0
        resets = record["vulkan"]["reset_observations"]
        assert len(resets) == 3
        assert all(entry["all_grad_none"] == (reset_mode == "none" or entry["index"] == 1) for entry in resets)
        if reset_mode == "zero":
            assert all(resets[index]["all_grad_zero"] for index in (1, 2))
            assert all(resets[index]["same_grad_objects"] for index in (1, 2))
        assert record["comparison"]["passed"] is True


def test_parameter_hvp_executes_with_named_live_history(workload_device, monkeypatch):
    import pytorch_vulkan
    expected = "sync" if os.getenv("PYTORCH_VULKAN_ASYNC_EXECUTION") == "0" else "async"
    assert pytorch_vulkan._C.execution_mode() == expected

    capture_window = workload._capture_window
    contraction_dispatch_counts = []

    def capture_window_spy(api, phase, step, run_operation, observe):
        if phase != "second_reverse":
            return capture_window(api, phase, step, run_operation, observe)

        def operation_after_window_reset():
            original_grad = torch.autograd.grad

            def grad_after_contraction(*args, **kwargs):
                dispatches = api._C.execution_counter_snapshot()[0]
                contraction_dispatch_counts.append(dispatches)
                assert dispatches > 0, "HVP contraction must dispatch inside the second_reverse window"
                return original_grad(*args, **kwargs)

            with monkeypatch.context() as scoped:
                scoped.setattr(torch.autograd, "grad", grad_after_contraction)
                return run_operation()

        return capture_window(api, phase, step, operation_after_window_reset, observe)

    monkeypatch.setattr(workload, "_capture_window", capture_window_spy)
    record = workload.run_vulkan_hvp()
    workload.validate_workload_record(record)
    assert len(contraction_dispatch_counts) == 1
    assert record["execution_mode"] == expected
    assert set(record["vulkan"]["hvp"]) == {"0.weight", "1.weight", "1.bias"}
    assert all(value["nonzero_count"] > 0 for value in record["vulkan"]["hvp"].values())
    assert record["vulkan"]["history"]["first_gradients_require_grad"] == {
        name: True for name in ("0.weight", "1.weight", "1.bias")
    }
    assert set(record["vulkan"]["history"]["grad_fns"]) == {"0.weight", "1.weight", "1.bias"}
    assert [window["phase"] for window in record["execution"]] == ["first_reverse", "second_reverse"]
    assert all(window["dispatches"] > 0 for window in record["execution"])
    assert all(window["fallbacks"] == window["explicit_transfers"] == 0 for window in record["execution"])
    assert all(window["observation_order"] == ["sync", "snapshot", "readback"] for window in record["execution"])
    assert record["vulkan"]["finite_difference"] is None
    assert record["comparison"]["passed"] is True
