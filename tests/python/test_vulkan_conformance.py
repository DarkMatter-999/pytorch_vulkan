import os
import re
from dataclasses import replace
from types import SimpleNamespace

import pytest
import pytorch_vulkan
import torch
import vulkan_conformance as vc
from vulkan_conformance import (
    ALL_CASES,
    DECLARED_OPERATION_MANIFEST,
    MANIFEST_CASE_NAMES,
    ROADMAP_DEFERRED_SCHEMAS,
    SUPPORTED_CASES,
    ConformanceCase,
    assert_gradients,
    assert_no_vulkan_work,
    assert_reverse_second_order,
    assert_vulkan_result,
    coverage_snapshot,
    record_coverage,
    run_and_compare,
    run_case,
    to_vulkan_inputs,
)
from tools.vulkan_capability_declarations import STOCK_LINEAR_ROUTE_CASES

REVERSE_SECOND_ORDER_CASES = {
    "arithmetic.autograd.add-tensor",
    "arithmetic.autograd.add-scalar",
    "arithmetic.autograd.mul-tensor",
    "arithmetic.autograd.mul-scalar",
    "arithmetic.autograd.sum-default",
    "arithmetic.autograd.sum-dim",
    "view.view.trainable-seed",
    "view.reshape.copy.trainable-seed",
    "view.reshape.offset-copy.second-order",
    "cat.rank4.native-seed",
    "cat.offset.trainable-seed",
}


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = os.environ.get(
        "VULKAN_DEVICE", f"{torch._C._get_privateuse1_backend_name()}:0"
    )
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


@pytest.fixture(scope="module", autouse=True)
def record_conformance_coverage():
    with vc.coverage_recording():
        yield


def test_registry_names_are_unique():
    names = [case.name for case in ALL_CASES]
    assert len(names) == len(set(names))


def test_registry_cases_are_typed_and_have_cpu_references():
    assert ALL_CASES
    assert all(isinstance(case, ConformanceCase) for case in ALL_CASES)
    assert all(case.cpu_reference is not None for case in ALL_CASES)
    stock_linear_route_names = {case["name"] for case in STOCK_LINEAR_ROUTE_CASES}
    assert {case.name for case in ALL_CASES} == MANIFEST_CASE_NAMES - stock_linear_route_names


def test_registry_supported_and_deferred_schemas_match_manifest():
    supported = {case.declaration_id for case in ALL_CASES if case.supported}
    assert supported == DECLARED_OPERATION_MANIFEST
    assert not supported & ROADMAP_DEFERRED_SCHEMAS


def test_ordinary_output_padding_case_is_registered_as_supported_convolution():
    case = next(
        case for case in ALL_CASES
        if case.name == "convolution.parameters.output-padding.ignored"
    )
    assert case.supported
    assert case.declaration_id == "aten::convolution.default"
    assert case.operation is vc._ordinary_output_padding_forward
    assert case.cpu_reference is vc._ordinary_output_padding_reference
    assert all(
        item.name != "convolution.parameters.output-padding.rejected"
        for item in ALL_CASES
    )


def test_differentiable_value_cases_declare_gradient_checks():
    for name in ("masked-select.bool-mask",):
        case = next(case for case in ALL_CASES if case.name == name)
        assert not case.autograd_supported
        assert case.requires_grad_inputs


@pytest.mark.parametrize(
    "case",
    [case for case in ALL_CASES if not case.autograd_supported],
    ids=lambda case: case.name,
)
def test_declared_autograd_rejection_is_explicit(vulkan_backend, case):
    inputs = to_vulkan_inputs(case.inputs(), vulkan_backend)
    result = case.operation(*inputs, *case.args, **(case.kwargs or {}))
    assert inputs[0].requires_grad
    if case.name == "masked-select.bool-mask":
        assert not inputs[1].requires_grad
    gradient = torch.ones(result.shape, dtype=result.dtype).to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(case.autograd_error_type, match=case.autograd_error_pattern):
        result.backward(gradient)
    if case.name == "masked-select.bool-mask":
        assert pytorch_vulkan._C.compute_dispatch_count() <= 1
    else:
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    assert pytorch_vulkan._C.fallback_count() == 0


@pytest.mark.parametrize("case", SUPPORTED_CASES, ids=lambda case: case.name)
def test_supported_case_matches_cpu_and_stays_vulkan(vulkan_backend, case):
    convolution_context = {} if case.convolution_direction is not None else None
    graph_context = {} if case.graph_autograd_case is not None else None
    result, cpu_result, inputs = run_and_compare(
        case, vulkan_backend, return_inputs=True,
        convolution_context_out=convolution_context,
        graph_autograd_context_out=graph_context,
    )
    assert_vulkan_result(result, case)
    assert case.execution_mode in {"compute", "copy", "metadata", "empty"}
    if graph_context is not None:
        # The graph executor captures phase counters before synchronized readbacks.
        # Its final return values are then copied to CPU, so process-global totals
        # include those deliberate readbacks and are not the runtime evidence.
        phases = graph_context["graph_autograd"]["execution"]
        assert sum(phase["compute_dispatches"] for phase in phases.values()) > 0
        for phase in phases.values():
            if not case.name.startswith("matrix.graph."):
                assert phase["vulkan_copies"] == 0
            assert phase["explicit_transfers"] == 0
            assert phase["fallbacks"] == 0
    elif case.execution_mode == "compute":
        if case.convolution_direction == "backward" or case.name.startswith("convolution.backward.bias-"):
            assert pytorch_vulkan._C.compute_dispatch_count() == sum(case.args[-1])
            execution = vc._CASE_EXECUTION[case.name]
            if not any(case.args[-1]):
                assert execution["vulkan_copies"] == 0
                assert execution["explicit_transfers"] == 0
                assert execution["fallbacks"] == 0
                assert execution["buffer_creations_delta"] == 0
                assert execution["live_allocations_delta"] == 0
        else:
            assert pytorch_vulkan._C.compute_dispatch_count() > 0
        if case.name not in {"linear.forward", "linear.forward.strided"}:
            assert pytorch_vulkan._C.vulkan_copy_count() == 0
    elif case.execution_mode == "copy":
        assert pytorch_vulkan._C.vulkan_copy_count() > 0
        if case.name not in {
            "masked-select.strided-value-view",
            "reduction.softmax.dim",
            "reduction.log-softmax.dim",
            "reduction.softmax.backward",
            "reduction.log-softmax.backward",
            "unary.sigmoid.backward",
            "unary.tanh.backward",
            "unary.gelu.tanh.backward",
        }:
            assert pytorch_vulkan._C.compute_dispatch_count() == 0
    else:
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
        assert pytorch_vulkan._C.vulkan_copy_count() == 0
    if graph_context is None:
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
    vc.assert_result_parity(result, cpu_result, case)
    vc.mark_executed(case.name)
    reverse_autograd = (
        assert_reverse_second_order(case, inputs)
        if case.name in REVERSE_SECOND_ORDER_CASES
        else None
    )
    record_coverage(
        case,
        inputs,
        result,
        gradients=case.check_gradients,
        parity=True,
        reverse_autograd=reverse_autograd,
        convolution_context=convolution_context,
        graph_autograd=(graph_context or {}).get("graph_autograd"),
    )


def test_executed_cases_match_recorded_cases():
    """Executed and recorded cases must be symmetric for this test run."""
    recorded = set(coverage_snapshot())
    executed = vc.executed_snapshot()
    missing = executed - recorded
    unexpected = recorded - executed
    assert not missing and not unexpected, (
        f"executed but not recorded: {sorted(missing)}; "
        f"recorded but not executed: {sorted(unexpected)}"
    )


def test_recorded_shapes_use_the_declared_shape_convention():
    """Recorded shapes are compared against declared_shapes in a later task, and
    declared_shapes is x-delimited and SHAPE_PATTERN-validated. A different
    delimiter would make every declared shape look unexercised."""
    from tools.validate_vulkan_capabilities import SHAPE_PATTERN

    snapshot = coverage_snapshot()
    # No "the suite must have run" assertion: that would make this test
    # order-dependent, and it fails in isolation whenever nothing has run yet.
    for name, record in snapshot.items():
        for shape in record["input_shapes"]:
            assert re.fullmatch(r"[0-9]+(?:x[0-9]+)*", shape), (
                f"{name} recorded shape {shape!r} is not x-delimited"
            )
            if "x" in shape and all(int(extent) > 0 for extent in shape.split("x")):
                assert SHAPE_PATTERN.fullmatch(shape), (
                    f"{name} recorded shape {shape!r} does not match SHAPE_PATTERN"
                )


def test_cat_tensor_list_evidence_tracks_actual_order_and_metadata():
    case = next(case for case in ALL_CASES if case.name == "cat.rank2.forward")
    tensors = [torch.ones((2, 3)), torch.ones((2, 4))]
    output = torch.empty((2, 7))
    with vc.coverage_recording():
        vc.mark_executed(case.name)
        vc.record_coverage(case, (tensors,), output, gradients=False, parity=True)
        original = vc.coverage_snapshot()[case.name]
        assert [item["shape"] for item in original["tensor_lists"][0]["tensors"]] == [[2, 3], [2, 4]]
        variants = (
            [tensors[1], tensors[0]],
            [tensors[0].to(torch.float64), tensors[1]],
            [tensors[0].reshape(6), tensors[1]],
            [tensors[0], torch.ones((2, 5))],
        )
        for changed in variants:
            vc.reset_coverage()
            vc.record_coverage(case, (changed,), output, gradients=False, parity=True)
            assert vc.coverage_snapshot()[case.name]["tensor_lists"] != original["tensor_lists"]


@pytest.mark.parametrize(
    "case",
    [case for case in ALL_CASES if case.check_gradients],
    ids=lambda case: case.name,
)
def test_declared_autograd_case_matches_cpu(vulkan_backend, case):
    assert_gradients(case, vulkan_backend)


@pytest.mark.parametrize("case", ALL_CASES, ids=lambda case: case.name)
def test_rejected_case_matches_declared_pattern(vulkan_backend, case):
    if case.supported:
        pytest.skip("positive case")
    pytorch_vulkan._C.reset_execution_counters()
    inputs = case.inputs()
    try:
        if case.convert_inputs:
            inputs = to_vulkan_inputs(inputs, vulkan_backend)
        if case.setup_inputs is not None:
            inputs = case.setup_inputs(inputs, vulkan_backend)
    except RuntimeError as error:
        assert re.search(case.error_pattern, str(error))
        assert_no_vulkan_work()
        return
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=case.error_pattern):
        case.operation(*inputs, *case.args, **(case.kwargs or {}))
    assert_no_vulkan_work()


def test_assert_vulkan_result_checks_device_and_dtype(vulkan_backend):
    case = next(case for case in ALL_CASES if case.supported)
    result = run_case(case, vulkan_backend)
    assert_vulkan_result(result, case)


def test_tuple_result_parity_preserves_tensor_and_none_slots():
    case = ConformanceCase(
        "tuple.test", "test", "aten::convolution_backward.default",
        lambda: None, lambda: (), lambda: None,
        expected_shapes=((2,), None, (1,)),
    )
    actual = (torch.tensor([1.0, 2.0]), None, torch.tensor([3.0]))
    expected = (torch.tensor([1.0, 2.0]), None, torch.tensor([3.0]))

    vc.assert_result_parity(actual, expected, case)

    with pytest.raises(AssertionError, match="slot 1"):
        vc.assert_result_parity((actual[0], torch.empty(0), actual[2]), expected, case)


def test_tuple_result_parity_checks_slot_shapes():
    case = ConformanceCase(
        "tuple.shape", "test", "aten::convolution_backward.default",
        lambda: None, lambda: (), lambda: None,
        expected_shapes=((2,), None, (1,)),
    )
    with pytest.raises(AssertionError, match="shape"):
        vc.assert_result_parity((torch.ones(3), None, torch.ones(1)),
                                (torch.ones(2), None, torch.ones(1)), case)


def test_named_convolution_bias_and_mask_evidence_executes_independently(vulkan_backend):
    from tools.validate_vulkan_capabilities import (
        CONVOLUTION_REQUIRED_CASES,
        validate_convolution_evidence,
    )

    cases = {case.name for case in SUPPORTED_CASES
             if case.name.startswith("convolution.forward.bias-")
             or case.name.startswith("convolution.backward.bias-")
             or case.convolution_direction is not None}
    coverage = vc.run_convolution_evidence_cases()
    expected = set(CONVOLUTION_REQUIRED_CASES)
    assert len(expected) == 36
    assert cases == expected
    assert set(coverage) == expected
    validate_convolution_evidence(coverage, require_complete=True)


def test_transposed_convolution_registry_declares_all_directional_roles():
    cases = {case.name: case for case in ALL_CASES if case.name.startswith("convolution.transposed.")}
    assert len(cases) == 18
    forward = cases["convolution.transposed.forward.bias-present"]
    assert forward.convolution_direction == "forward"
    assert forward.convolution_numerical_operations == (1,)
    assert forward.convolution_result_slots == (0,)
    assert forward.convolution_operand_roles == ("input", "weight", "bias")
    backward = cases["convolution.transposed.backward.bias-present.mask-010"]
    assert backward.convolution_direction == "backward"
    assert backward.convolution_numerical_operations == (2,)
    assert backward.convolution_result_slots == (1,)
    assert backward.convolution_operand_roles == ("grad_output", "input", "weight")
    assert cases["convolution.transposed.backward.bias-absent.mask-000"].convolution_numerical_operations == ()
    for state in ("present", "absent"):
        for bits in range(8):
            mask = tuple(bool(bits & (1 << shift)) for shift in range(3))
            case = cases[f"convolution.transposed.backward.bias-{state}.mask-{''.join('1' if b else '0' for b in mask)}"]
            assert case.args[-1] == list(mask)
            assert case.convolution_numerical_operations == tuple(
                op for op, requested in zip((0, 2, 3), mask) if requested
            )
            assert case.convolution_result_slots == tuple(
                index for index, requested in enumerate(mask) if requested
            )
    hints = {repr(case.args[0]) for case in cases.values()
             if case.convolution_direction == "backward"}
    assert hints == {"None", "[]", "[999]"}


def test_convolution_context_metadata_reads_actual_tensor_device_and_layout():
    value = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    metadata = vc._convolution_tensor_metadata(value, str(value.device))
    assert metadata == {
        "defined": True,
        "dtype": "float32",
        "rank": 2,
        "shape": [4, 3],
        "strides": [1, 4],
        "storage_offset": 0,
        "device": "cpu",
    }
    absent = vc._convolution_tensor_metadata(None, "cpu")
    assert absent == {
        "defined": False, "dtype": None, "rank": None, "shape": None,
        "strides": None, "storage_offset": None, "device": "cpu",
    }


def test_transposed_context_warmup_uses_actual_bias_and_preserves_schema_inputs():
    case = next(case for case in ALL_CASES
                if case.name == "convolution.transposed.backward.bias-present.mask-000")
    inputs = case.inputs()
    returned, capture = case.convolution_context_setup(inputs, "cpu")
    assert len(returned) == 3
    assert all(actual is expected for actual, expected in zip(returned, inputs))
    assert set(capture) == {"warm_forward_bias"}
    assert capture["warm_forward_bias"] == vc._convolution_tensor_metadata(
        torch.randn((6,), generator=torch.Generator(device="cpu").manual_seed(9917)), "cpu"
    )


def test_transposed_run_clears_caller_context_when_capture_contract_is_missing():
    case = next(case for case in ALL_CASES
                if case.name == "convolution.transposed.forward.bias-present")
    output = {"stale": {"warm_forward_bias": "previous case"}}
    with pytest.raises(ValueError, match="requires fresh context capture"):
        run_and_compare(replace(case, convolution_context_setup=None), convolution_context_out=output)
    assert output == {}


def test_transposed_context_setup_exception_leaves_no_execution_or_context():
    case = next(case for case in ALL_CASES
                if case.name == "convolution.transposed.forward.bias-present")

    def fail_setup(_inputs, _device):
        raise RuntimeError("context setup failed")

    output = {"stale": True}
    with pytest.raises(RuntimeError, match="context setup failed"):
        run_and_compare(replace(case, convolution_context_setup=fail_setup),
                        convolution_context_out=output)
    assert output == {}
    assert case.name not in vc._CASE_EXECUTION


def test_transposed_operation_exception_leaves_no_execution_or_context(monkeypatch):
    case = next(case for case in ALL_CASES
                if case.name == "convolution.transposed.forward.bias-present")
    _install_fake_convolution_runtime(monkeypatch, (0, 0, 0, 0))

    def fail_operation(*_args):
        raise RuntimeError("measured operation failed")

    output = {"stale": True}
    with pytest.raises(RuntimeError, match="measured operation failed"):
        run_and_compare(replace(case, operation=fail_operation),
                        convolution_context_out=output)
    assert output == {}
    assert case.name not in vc._CASE_EXECUTION


def test_transposed_nonzero_vulkan_copies_do_not_publish_execution_or_context(monkeypatch):
    case = next(case for case in ALL_CASES
                if case.name == "convolution.transposed.forward.bias-present")
    _install_fake_convolution_runtime(monkeypatch, (1, 1, 0, 0))
    output = {"stale": True}
    with pytest.raises(AssertionError, match="vulkan_copies.*0"):
        run_and_compare(case, convolution_context_out=output)
    assert output == {}
    assert case.name not in vc._CASE_EXECUTION


def _install_fake_convolution_runtime(monkeypatch, counters):
    class FakeRuntime:
        @staticmethod
        def synchronize():
            pass

        @staticmethod
        def reset_execution_counters():
            pass

        @staticmethod
        def timing_breakdown():
            return {"buffer_creations": 0}

        @staticmethod
        def live_resource_snapshot():
            return [0] * 7

        @staticmethod
        def execution_counter_snapshot():
            return counters

    monkeypatch.setattr(vc, "pytorch_vulkan", SimpleNamespace(_C=FakeRuntime()))
    monkeypatch.setattr(vc, "to_vulkan_inputs", lambda inputs, _device: tuple(inputs))


def test_nll_forward_none_writes_zero_total_weight(vulkan_backend):
    case = next(
        case for case in ALL_CASES
        if case.name == "classification.nll-forward.none.wide"
    )
    logits, labels = case.inputs()
    log_probs = torch.log_softmax(logits, dim=1).to(vulkan_backend)
    labels = labels.to(vulkan_backend)
    loss, total_weight = torch.ops.aten.nll_loss_forward.default(
        log_probs, labels, None, 0, -100
    )
    pytorch_vulkan._C.synchronize()
    expected = torch.nn.functional.nll_loss(
        torch.log_softmax(logits, dim=1), labels.cpu(), reduction="none"
    )
    actual = loss.cpu()
    assert actual.numel() == 512
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=5e-7)
    assert total_weight.cpu().item() == 0.0


def test_execution_counters_count_dispatch_and_explicit_cpu_transfer(vulkan_backend):
    source = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = torch.neg(source)

    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    result.cpu()
    assert pytorch_vulkan._C.explicit_transfer_count() == 1


@pytest.mark.parametrize(
    "case_name",
    ["linear.forward", "mm.forward", "addmm.forward", "addmm.out"],
)
def test_gemm_frontends_complete_one_dispatch(vulkan_backend, case_name):
    case = next(case for case in ALL_CASES if case.name == case_name)
    run_and_compare(case, vulkan_backend)
    pytorch_vulkan._C.synchronize()
    expected = 3 if case_name == "linear.forward" else 1
    assert pytorch_vulkan._C.compute_dispatch_count() == expected
    # Async batching may coalesce multiple dispatches into fewer submissions.
    assert pytorch_vulkan._C.pending_compute_count() == 0
    # completed/submitted/waits count different things: submitted counts
    # vkQueueSubmit calls, completed counts fence completions observed from
    # either the blocking-wait or the poll path. Async can place several
    # fences in one submit, so completed may exceed submitted. Only the
    # dispatch count and pending==0 are mode-agnostic exact invariants.
    assert pytorch_vulkan._C.compute_completed_count() >= 1
    assert pytorch_vulkan._C.compute_submitted_count() >= 1
    assert pytorch_vulkan._C.fallback_count() == 0


def test_zero_dimension_mm_and_addmm_gradients_are_finite(vulkan_backend):
    cpu_mat1 = torch.empty(2, 0, requires_grad=True)
    cpu_mat2 = torch.empty(0, 3, requires_grad=True)
    cpu_self = torch.ones(2, 3, requires_grad=True)
    cpu_addmm_mat1 = cpu_mat1.detach().clone().requires_grad_()
    cpu_addmm_mat2 = cpu_mat2.detach().clone().requires_grad_()
    vk_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()
    vk_addmm_mat1 = cpu_mat1.detach().clone().to(vulkan_backend).requires_grad_()
    vk_addmm_mat2 = cpu_mat2.detach().clone().to(vulkan_backend).requires_grad_()
    vk_self = cpu_self.detach().clone().to(vulkan_backend).requires_grad_()

    cpu_mm = torch.mm(cpu_mat1, cpu_mat2)
    cpu_addmm = torch.addmm(cpu_self, cpu_addmm_mat1, cpu_addmm_mat2, beta=2.0)
    vk_mm = torch.mm(vk_mat1, vk_mat2)
    vk_addmm = torch.addmm(vk_self, vk_addmm_mat1, vk_addmm_mat2, beta=2.0)

    assert torch.isfinite(vk_mm.cpu()).all()
    assert torch.isfinite(vk_addmm.cpu()).all()
    cpu_mm.sum().backward()
    cpu_addmm.sum().backward()
    vk_mm.sum().backward()
    vk_addmm.sum().backward()
    torch.testing.assert_close(vk_mm.cpu(), cpu_mm, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(vk_addmm.cpu(), cpu_addmm, rtol=2e-3, atol=2e-3)
    for vk_gradient, cpu_gradient in (
        (vk_mat1.grad, cpu_mat1.grad),
        (vk_mat2.grad, cpu_mat2.grad),
        (vk_addmm_mat1.grad, cpu_addmm_mat1.grad),
        (vk_addmm_mat2.grad, cpu_addmm_mat2.grad),
    ):
        assert vk_gradient is not None
        assert cpu_gradient is not None
        assert torch.isfinite(vk_gradient.cpu()).all()
        torch.testing.assert_close(
            vk_gradient.cpu(), cpu_gradient, rtol=2e-3, atol=2e-3
        )
    assert vk_self.grad is not None
    assert torch.isfinite(vk_self.grad.cpu()).all()
    torch.testing.assert_close(vk_self.grad.cpu(), cpu_self.grad, rtol=2e-3, atol=2e-3)


def test_execution_counter_snapshot_accounts_for_fallbacks(vulkan_backend):
    source = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.set_strict_mode(True)
    try:
        pytorch_vulkan._C.reset_execution_counters()
        torch.neg(source)
        assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    finally:
        pytorch_vulkan._C.set_strict_mode(False)


def test_unsupported_operation_does_not_fallback(vulkan_backend):
    source = torch.ones(2, dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.set_strict_mode(True)
    try:
        pytorch_vulkan._C.reset_execution_counters()
        with pytest.raises(RuntimeError, match="Vulkan"):
            source.neg_()
        assert_no_vulkan_work()
    finally:
        pytorch_vulkan._C.set_strict_mode(False)


def test_optimizer_step_preserves_an_outer_training_scope(vulkan_backend):
    parameter = torch.tensor([1.0, -2.0], device=vulkan_backend, requires_grad=True)
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    loss = (parameter * parameter).sum()
    pytorch_vulkan._C.begin_training_step()
    try:
        loss.backward()
        optimizer.step()
        assert pytorch_vulkan._C.training_step_active()
    finally:
        pytorch_vulkan._C.cancel_training_step()


@pytest.mark.parametrize(
    "case_name", ["optimizer.add.inplace", "optimizer.mul.scalar.inplace"]
)
def test_direct_optimizer_inplace_schema_preserves_an_outer_training_scope(
    vulkan_backend, case_name
):
    case = next(case for case in ALL_CASES if case.name == case_name)
    inputs = to_vulkan_inputs(case.inputs(), vulkan_backend)
    pytorch_vulkan._C.begin_training_step()
    try:
        case.operation(*inputs, *case.args, **(case.kwargs or {}))
        assert pytorch_vulkan._C.training_step_active()
    finally:
        pytorch_vulkan._C.cancel_training_step()
