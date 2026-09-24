from pathlib import Path
import copy
import importlib.util

import pytest


ROOT = Path(__file__).parents[2]
_BENCHMARK_SPEC = importlib.util.spec_from_file_location(
    "vulkan_operator_benchmark", ROOT / "tools/vulkan_operator_benchmark.py"
)
_BENCHMARK = importlib.util.module_from_spec(_BENCHMARK_SPEC)
_BENCHMARK_SPEC.loader.exec_module(_BENCHMARK)


def test_nested_diagnostic_gpu_scopes_do_not_change_operator_aggregate():
    samples = [
        {"scope": "operator", "gpu_time_ns": 100},
        {"scope": "rnn_backward_mode_1", "gpu_time_ns": 90},
        {"scope": "rnn_backward_mode_2", "gpu_time_ns": 5},
        {"scope": "operator", "gpu_time_ns": 20},
    ]
    assert _BENCHMARK._aggregate_operator_gpu_time(samples) == 120


class _FakeVulkanApi:
    def __init__(self, *, timing=True, fallbacks=0):
        self.timing = timing
        self.fallbacks = fallbacks
        self.submission = 0
        self.resets = 0

    def timestamp_queries_supported(self):
        return self.timing

    def timestamp_query_support_reason(self):
        return "fake timestamp status"

    def reset_execution_counters(self):
        self.resets = 0

    def reset_timing(self):
        pass

    def reset_gpu_timing(self):
        pass

    def reset_descriptor_resource_counters(self):
        pass

    def gpu_timing_snapshot(self):
        if not self.timing:
            return []
        return [
            {
                "available": True,
                "scope": "operator",
                "submission_id": self.submission,
                "gpu_time_ns": 17,
            }
        ]

    def execution_counter_snapshot(self):
        return (1, 0, 0, self.fallbacks)

    def compute_submitted_count(self):
        return self.submission

    def compute_completed_count(self):
        return self.submission

    def compute_wait_count(self):
        return self.submission

    def descriptor_pool_creation_count(self):
        return 0

    def descriptor_set_allocation_count(self):
        return 1

    def descriptor_set_reuse_count(self):
        return 0


def _fake_case(api=None, *, equal=True):
    events = []
    payload_pairs = []

    def prepare_pair(seed):
        source = [seed]
        pair = (source.copy(), source.copy())
        payload_pairs.append(pair)
        return pair

    def cpu_call(values):
        events.append("cpu")
        return [values[0] + 1]

    def vulkan_call(values):
        events.append("vulkan")
        if api is not None:
            api.submission += 1
        return [values[0] + (1 if equal else 2)]

    def compare(cpu_result, vulkan_result, rtol, atol):
        return {"passed": cpu_result == vulkan_result, "max_abs_difference": abs(cpu_result[0] - vulkan_result[0])}

    return {
        "schema": "aten::example.default",
        "family": "pointwise",
        "phase": "forward",
        "status": "measurable",
        "input_metadata": {"shape": [1], "dtype": "float32", "stride": [1]},
        "prepare_pair": prepare_pair,
        "cpu_call": cpu_call,
        "vulkan_call": vulkan_call,
        "compare": compare,
        "vulkan_api": api,
        "events": events,
        "payload_pairs": payload_pairs,
    }


def test_measure_case_pairs_inputs_alternates_order_and_excludes_warmups():
    from tools.vulkan_operator_benchmark import measure_case

    api = _FakeVulkanApi()
    case = _fake_case(api)
    result = measure_case(case, warmups=1, repetitions=2, order_seed=1)

    assert case["events"] == ["cpu", "vulkan", "vulkan", "cpu", "cpu", "vulkan"]
    assert result["cpu_time_ns"]["samples"] and len(result["cpu_time_ns"]["samples"]) == 2
    assert len(result["vulkan_host_time_ns"]["samples"]) == 2
    assert result["warmups"] == 1
    assert all(sample == 17 for sample in result["vulkan_gpu_time_ns"]["samples"])
    assert result["parity"]["passed"] is True
    assert len(case["payload_pairs"]) == 3
    assert all(cpu == vulkan and cpu is not vulkan for cpu, vulkan in case["payload_pairs"])


@pytest.mark.parametrize(
    ("equal", "timing", "fallbacks", "expected"),
    [
        (False, True, 0, "parity_failed"),
        (True, False, 0, "timing_unavailable"),
        (True, True, 1, "fallback_detected"),
    ],
)
def test_qualification_rejects_invalid_cases(equal, timing, fallbacks, expected):
    from tools.vulkan_operator_benchmark import measure_case, qualify_case

    result = measure_case(
        _fake_case(_FakeVulkanApi(timing=timing, fallbacks=fallbacks), equal=equal),
        warmups=0,
        repetitions=1,
    )

    assert qualify_case(result, rtol=3e-3, atol=3e-3)["qualification"] == expected


def test_artifact_validation_requires_complete_metadata_and_finite_samples():
    from tools.vulkan_operator_benchmark import validate_artifact

    with pytest.raises(ValueError, match="schema_version"):
        validate_artifact({})

    artifact = {
        "schema_version": 1,
        "device": {"name": "Renoir"},
        "warmups": 0,
        "repetitions": 1,
        "cpu_threads": {"intraop": 1},
        "results": [
            {
                "schema": "aten::example.default",
                "family": "pointwise",
                "phase": "forward",
                "input_metadata": {"shape": [1]},
                "cpu_time_ns": {"samples": [10], "mean": 10, "median": 10},
                "vulkan_host_time_ns": {"samples": [20], "mean": 20, "median": 20},
                "vulkan_gpu_time_ns": {
                    "status": "available",
                    "samples": [15],
                    "mean": 15,
                    "median": 15,
                },
                "counters": [{
                    "dispatches": 1,
                    "transfers": 0,
                    "explicit_transfers": 0,
                    "fallbacks": 0,
                    "submissions": 1,
                    "completions": 1,
                    "waits": 1,
                }],
                "parity": {"passed": True},
                "qualification": "qualified",
            }
        ],
    }
    assert validate_artifact(artifact) is artifact

    artifact["results"][0]["cpu_time_ns"]["samples"] = [float("nan")]
    with pytest.raises(ValueError, match="finite and non-negative"):
        validate_artifact(artifact)


def test_artifact_validation_checks_sample_counts_and_counters():
    from tools.vulkan_operator_benchmark import validate_artifact

    valid = {
        "schema_version": 1,
        "device": {"name": "Renoir"},
        "warmups": 0,
        "repetitions": 1,
        "cpu_threads": {"intraop": 1},
        "results": [{
            "schema": "aten::example.default",
            "family": "pointwise",
            "phase": "forward",
            "input_metadata": {"shape": [1]},
            "cpu_time_ns": {"samples": [10], "mean": 10, "median": 10},
            "vulkan_host_time_ns": {"samples": [20], "mean": 20, "median": 20},
            "vulkan_gpu_time_ns": {"status": "available", "samples": [15], "mean": 15, "median": 15},
            "counters": [{"dispatches": 1, "transfers": 0, "explicit_transfers": 0, "fallbacks": 0, "submissions": 1, "completions": 1, "waits": 1}],
            "parity": {"passed": True},
            "qualification": "qualified",
        }],
    }
    bad_count = copy.deepcopy(valid)
    bad_count["repetitions"] = 2
    with pytest.raises(ValueError, match="match repetitions"):
        validate_artifact(bad_count)

    bad_counter = copy.deepcopy(valid)
    del bad_counter["results"][0]["counters"][0]["waits"]
    with pytest.raises(ValueError, match="counters.waits"):
        validate_artifact(bad_counter)


@pytest.mark.parametrize(
    "schema",
    [
        "aten::relu.default",
        "aten::gelu.default",
        "aten::add.Tensor",
        "aten::div.Tensor",
        "aten::mm.default",
        "aten::convolution.default",
        "aten::_adaptive_avg_pool2d.default",
        "aten::mean.dim",
        "aten::native_batch_norm.default",
        "aten::mse_loss.default",
        "aten::nll_loss_forward.default",
        "aten::view.default",
        "aten::gelu_backward.grad_input",
        "aten::sigmoid_backward.grad_input",
        "aten::tanh_backward.grad_input",
        "aten::_adaptive_avg_pool2d_backward.default",
        "aten::_softmax_backward_data.out",
        "aten::_log_softmax_backward_data.out",
        "aten::mse_loss_backward.default",
        "aten::convolution_backward.default",
        "aten::native_batch_norm_backward.default",
        "aten::nll_loss_backward.default",
        "aten::add.Scalar_out",
        "aten::add.out",
        "aten::add_.Tensor",
        "aten::sub.Scalar_out",
        "aten::sub.out",
        "aten::rsub.Scalar_out",
        "aten::mul.Scalar_out",
        "aten::mul.out",
        "aten::mul_.Scalar",
        "aten::lerp.Scalar_out",
        "aten::lerp_.Scalar",
        "aten::sqrt.out",
        "aten::amax.out",
        "aten::amin.out",
        "aten::prod.int_out",
        "aten::_softmax.out",
        "aten::_log_softmax.out",
        "aten::addmm.out",
        "aten::copy_.default",
        "aten::masked_select.default",
        "pytorch_vulkan::rnn_sequence",
    ],
)
def test_representative_family_cases_match_cpu(schema):
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark_cases import build_case

    case = build_case(schema, "small", seed=41)
    cpu_payload, vulkan_payload = case["prepare_pair"](case["seed"])
    cpu_result = case["cpu_call"](cpu_payload)
    vulkan_result = case["vulkan_call"](vulkan_payload)

    assert case["family"]
    assert case["input_metadata"]["dtype"]
    assert case["compare"](cpu_result, vulkan_result, 3e-3, 3e-3)["passed"]


@pytest.mark.parametrize(
    ("profile", "size", "selected"),
    [
        ("small", 32, None), ("large", 262144, None),
        ("zero", 0, 0),
        ("medium_none", 4096, 0), ("medium_sparse", 4096, 64),
        ("medium_half", 4096, 2048), ("medium_dense", 4096, 4032),
        ("wide_none", 65536, 0), ("wide_sparse", 65536, 1024),
        ("wide_half", 65536, 32768), ("wide_dense", 65536, 64512),
    ],
)
def test_masked_select_profiles_have_deterministic_shapes_and_density(profile, size, selected):
    import torch
    from tools.vulkan_operator_benchmark_cases import _arguments

    values, mask = _arguments(torch, "aten::masked_select.default", profile, 42)
    replay_values, replay_mask = _arguments(torch, "aten::masked_select.default", profile, 42)
    assert values.shape == mask.shape == (size,)
    assert torch.equal(values, replay_values)
    assert torch.equal(mask, replay_mask)
    assert mask.dtype == torch.bool
    if selected is not None:
        assert int(mask.sum()) == selected
        if "sparse" in profile:
            assert torch.equal(mask.nonzero().flatten(), torch.arange(0, size, 64))
        if "dense" in profile:
            assert torch.equal((~mask).nonzero().flatten(), torch.arange(0, size, 64))


def test_masked_select_catalog_exposes_density_profiles_without_changing_other_operators():
    from tools.vulkan_operator_benchmark_cases import iter_cases

    entries = [
        {"schema": schema, "family": "pointwise", "status": "measurable"}
        for schema in ("aten::masked_select.default", "aten::relu.default")
    ]
    cases = list(iter_cases({"entries": entries}))
    profiles = [case["input_metadata"]["shape_profile"] for case in cases
                if case["schema"] == "aten::masked_select.default"]
    assert profiles == [
        "small", "large", "zero", "medium_none", "medium_sparse", "medium_half",
        "medium_dense", "wide_none", "wide_sparse", "wide_half", "wide_dense",
    ]
    assert [case["input_metadata"]["shape_profile"] for case in cases
             if case["schema"] == "aten::relu.default" and case["phase"] == "forward"] == ["small", "large"]


def test_masked_select_prepare_pair_reuses_vulkan_mask_for_same_seed():
    import torch
    import pytorch_vulkan
    from tools.vulkan_operator_benchmark_cases import build_case

    case = build_case("aten::masked_select.default", "wide_half", seed=123)
    before = pytorch_vulkan._C.transfer_operation_count()
    cpu_args1, vk_args1 = case["prepare_pair"](123)
    after_first = pytorch_vulkan._C.transfer_operation_count()
    cpu_args2, vk_args2 = case["prepare_pair"](123)
    after_second = pytorch_vulkan._C.transfer_operation_count()

    assert after_first - before == 2
    assert after_second - after_first == 1
    assert torch.equal(cpu_args1[1], cpu_args2[1])
    assert torch.equal(cpu_args1[1], vk_args1[1].cpu())
    assert torch.equal(cpu_args2[1], vk_args2[1].cpu())


def test_masked_select_zero_profile_runs_with_empty_parity_and_no_dispatch():
    import torch
    import pytorch_vulkan
    from tools.vulkan_operator_benchmark_cases import build_case

    case = build_case("aten::masked_select.default", "zero", seed=1729)
    cpu_args, vk_args = case["prepare_pair"](1729)
    pytorch_vulkan._C.reset_execution_counters()
    cpu_result = case["cpu_call"](cpu_args)
    vk_result = case["vulkan_call"](vk_args)
    dispatches, _, _, fallbacks = pytorch_vulkan._C.execution_counter_snapshot()

    assert cpu_result.numel() == vk_result.numel() == 0
    assert case["compare"](cpu_result, vk_result, 0, 0)["passed"]
    assert dispatches == 0
    assert fallbacks == 0


def test_iter_cases_accounts_for_catalog_entries_and_backward_cases():
    from tools.vulkan_operator_benchmark_cases import iter_cases, load_operator_cases

    catalog = load_operator_cases(ROOT)
    cases = list(iter_cases(catalog, ("small",)))

    assert {case["schema"] for case in cases} == {
        entry["schema"] for entry in catalog["entries"]
    }
    assert any(case.get("phase") == "backward" for case in cases)
    assert all(case.get("reason") for case in cases if case["status"] == "non_comparable")
    add_scalar_inplace = next(
        case for case in cases if case["schema"] == "aten::add_.Scalar"
    )
    assert add_scalar_inplace["status"] == "non_comparable"
    assert "deferred" in add_scalar_inplace["reason"]


def test_backward_case_times_vulkan_backward_dispatches():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark import measure_case
    from tools.vulkan_operator_benchmark_cases import build_case

    backward = build_case("aten::relu.default", "small", seed=71)["backward_case"]
    result = measure_case(backward, warmups=0, repetitions=1)

    assert result["phase"] == "backward"
    assert result["counters"][0]["dispatches"] > 0
    assert result["parity"]["passed"] is True


def test_metadata_operator_is_host_rankable_with_gpu_timing_not_applicable():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark import measure_case
    from tools.vulkan_operator_benchmark_cases import build_case

    result = measure_case(
        build_case("aten::view.default", "small", seed=13),
        warmups=0,
        repetitions=1,
    )

    assert result["vulkan_gpu_time_ns"]["status"] == "not_applicable"
    assert result["qualification"] == "qualified"


def test_gemm_benchmark_qualifies_completed_gemm_scope():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark import measure_case
    from tools.vulkan_operator_benchmark_cases import build_case

    result = measure_case(
        build_case("aten::mm.default", "small", seed=17),
        warmups=0,
        repetitions=1,
    )

    assert result["vulkan_gpu_time_ns"]["status"] == "available"
    assert result["qualification"] == "qualified"
    assert any(
        sample["scope"] == "gemm" and sample["available"]
        for samples in result["timestamp_samples"]
        for sample in samples
    )


def test_stack_backward_case_tracks_list_tensor_gradients():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark import measure_case
    from tools.vulkan_operator_benchmark_cases import build_case

    backward = build_case("aten::stack.default", "small", seed=17)["backward_case"]
    result = measure_case(backward, warmups=0, repetitions=1)

    assert result["counters"][0]["dispatches"] > 0
    assert result["parity"]["passed"] is True


def test_div_tensor_backward_keeps_scalar_denominator_on_cpu():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark_cases import build_case

    backward = build_case("aten::div.Tensor", "small", seed=29)["backward_case"]
    _, vulkan_payload = backward["prepare_pair"](29)

    assert vulkan_payload["args"][1].device.type == "cpu"
    from tools.vulkan_operator_benchmark import measure_case

    result = measure_case(backward, warmups=0, repetitions=1)
    assert result["parity"]["passed"] is True
    assert result["counters"][0]["fallbacks"] == 0


@pytest.mark.parametrize(
    "schema",
    ["aten::mse_loss.default", "aten::native_batch_norm.default"],
)
def test_loss_and_batch_norm_backward_cases_are_differentiable(schema):
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark import measure_case
    from tools.vulkan_operator_benchmark_cases import build_case

    backward = build_case(schema, "small", seed=31)["backward_case"]
    _, vulkan_payload = backward["prepare_pair"](31)
    if schema == "aten::native_batch_norm.default":
        assert not vulkan_payload["args"][3].requires_grad
        assert not vulkan_payload["args"][4].requires_grad
    result = measure_case(backward, warmups=0, repetitions=1)

    assert result["parity"]["passed"] is True
    assert result["counters"][0]["fallbacks"] == 0


def test_large_rnn_benchmark_case_uses_a_numerically_stable_reference():
    import pytorch_vulkan

    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    from tools.vulkan_operator_benchmark_cases import build_case

    case = build_case("pytorch_vulkan::rnn_sequence", "large", seed=43)
    cpu_payload, vulkan_payload = case["prepare_pair"](43)
    cpu_result = case["cpu_call"](cpu_payload)
    vulkan_result = case["vulkan_call"](vulkan_payload)

    assert case["compare"](cpu_result, vulkan_result, 3e-3, 3e-3)["passed"]


def test_operator_catalog_covers_registered_aten_and_custom_schemas():
    from tools.validate_vulkan_capabilities import _source_registration_inventory
    from tools.vulkan_operator_benchmark_cases import (
        _custom_registration_inventory,
        load_operator_cases,
    )

    catalog = load_operator_cases(ROOT)
    schemas = [entry["schema"] for entry in catalog["entries"]]
    aten_registrations, _ = _source_registration_inventory(ROOT)
    custom_registrations = _custom_registration_inventory(ROOT)

    assert len(schemas) == len(set(schemas))
    assert aten_registrations <= set(schemas)
    assert custom_registrations <= set(schemas)
    assert {"pytorch_vulkan::rnn_sequence", "pytorch_vulkan::linear_relu"} <= custom_registrations
    assert catalog["schema_version"] == 1
    assert all(entry["family"] and entry["reason"] for entry in catalog["entries"])
    custom_entries = [entry for entry in catalog["entries"] if entry["schema"].startswith("pytorch_vulkan::")]
    assert all(entry["status"] == "measurable" for entry in custom_entries)


def test_operator_coverage_reports_missing_and_extra_schemas():
    from tools.vulkan_operator_benchmark_cases import validate_operator_coverage

    catalog = {
        "entries": [
            {
                "schema": "aten::known.default",
                "family": "pointwise",
                "status": "non_comparable",
                "reason": "no standalone CPU counterpart",
            },
            {
                "schema": "aten::unexpected.default",
                "family": "pointwise",
                "status": "non_comparable",
                "reason": "test entry",
            },
        ]
    }

    with pytest.raises(ValueError, match="missing=.*aten::missing.default.*extra=.*aten::unexpected.default"):
        validate_operator_coverage(
            catalog,
            {"aten::known.default", "aten::missing.default"},
            set(),
        )


def test_non_comparable_operator_requires_reason():
    from tools.vulkan_operator_benchmark_cases import validate_operator_coverage

    catalog = {
        "entries": [
            {
                "schema": "aten::example.default",
                "family": "pointwise",
                "status": "non_comparable",
                "reason": "",
            }
        ]
    }

    with pytest.raises(ValueError, match="reason"):
        validate_operator_coverage(catalog, {"aten::example.default"}, set())


def test_scorecards_rank_host_and_gpu_ratios_separately_and_exclude_invalid():
    from tools.vulkan_operator_benchmark import build_scorecards

    def result(schema, cpu, host, gpu, qualification="qualified"):
        return {
            "schema": schema,
            "family": "pointwise",
            "phase": "forward",
            "qualification": qualification,
            "cpu_time_ns": {"mean": cpu},
            "vulkan_host_time_ns": {"mean": host},
            "vulkan_gpu_time_ns": {"status": "available", "mean": gpu},
        }

    cards = build_scorecards([
        result("aten::a.default", 10, 100, 30),
        result("aten::b.default", 20, 120, 100),
        result("aten::bad.default", 1, 1000, 1, "parity_failed"),
    ])

    assert [row["schema"] for row in cards["host_ratio"]] == [
        "aten::a.default", "aten::b.default"
    ]
    assert [row["schema"] for row in cards["gpu_ratio"]] == [
        "aten::b.default", "aten::a.default"
    ]
    assert cards["host_ratio"][0]["host_minus_gpu_ns"] == 70
    assert cards["host_ratio"][0]["host_minus_gpu_label"] == "diagnostic estimate"


def test_ranked_summary_includes_shape_profile():
    from tools.vulkan_operator_benchmark import _render_summary

    row = {
        "schema": "aten::relu.default", "family": "pointwise", "phase": "forward",
        "shape_profile": "large", "vulkan_to_cpu_ratio": 2.0,
        "host_minus_gpu_ns": 10, "vulkan_time_ns": 20,
    }
    summary = _render_summary({"host_ratio": [row], "gpu_ratio": [row]})

    assert "[forward, large]" in summary
