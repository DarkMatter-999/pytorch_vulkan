import re

import pytest
import torch

import pytorch_vulkan
from vulkan_conformance import (
    ALL_CASES,
    ConformanceCase,
    SUPPORTED_CASES,
    assert_gradients,
    assert_vulkan_result,
    run_case,
    run_and_compare,
    to_vulkan_inputs,
)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk:0"


def test_registry_names_are_unique():
    names = [case.name for case in ALL_CASES]
    assert len(names) == len(set(names))


def test_registry_cases_are_typed_and_have_cpu_references():
    assert ALL_CASES
    assert all(isinstance(case, ConformanceCase) for case in ALL_CASES)
    assert all(case.cpu_reference is not None for case in ALL_CASES)


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


@pytest.mark.parametrize("case", SUPPORTED_CASES, ids=lambda case: case.name)
def test_supported_case_matches_cpu_and_stays_vulkan(vulkan_backend, case):
    result, cpu_result = run_and_compare(case, vulkan_backend)
    assert_vulkan_result(result, case)
    assert case.execution_mode in {"compute", "copy", "metadata", "empty"}
    if case.execution_mode == "compute":
        assert pytorch_vulkan._C.compute_dispatch_count() > 0
        assert pytorch_vulkan._C.vulkan_copy_count() == 0
    elif case.execution_mode == "copy":
        assert pytorch_vulkan._C.vulkan_copy_count() > 0
        if case.name != "masked-select.strided-value-view":
            assert pytorch_vulkan._C.compute_dispatch_count() == 0
    else:
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
        assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    torch.testing.assert_close(
        result.cpu(), cpu_result, rtol=case.rtol, atol=case.atol, equal_nan=True
    )


@pytest.mark.parametrize(
    "case", [case for case in ALL_CASES if case.check_gradients], ids=lambda case: case.name
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
        assert pytorch_vulkan._C.compute_dispatch_count() == 0
        assert pytorch_vulkan._C.vulkan_copy_count() == 0
        assert pytorch_vulkan._C.explicit_transfer_count() == 0
        return
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match=case.error_pattern):
        case.operation(*inputs, *case.args, **(case.kwargs or {}))
    assert pytorch_vulkan._C.compute_dispatch_count() == 0
    assert pytorch_vulkan._C.vulkan_copy_count() == 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0


def test_assert_vulkan_result_checks_device_and_dtype(vulkan_backend):
    case = next(case for case in ALL_CASES if case.supported)
    result = run_case(case, vulkan_backend)
    assert_vulkan_result(result, case)


def test_execution_counters_count_dispatch_and_explicit_cpu_transfer(vulkan_backend):
    source = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32).to(vulkan_backend)
    pytorch_vulkan._C.reset_execution_counters()

    result = torch.neg(source)

    assert pytorch_vulkan._C.compute_dispatch_count() > 0
    assert pytorch_vulkan._C.explicit_transfer_count() == 0
    result.cpu()
    assert pytorch_vulkan._C.explicit_transfer_count() == 1
