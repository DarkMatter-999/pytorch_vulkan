import pytest
import torch
import pytorch_vulkan


@pytest.mark.parametrize("layout", ["dense", "transpose", "offset", "scalar", "empty"])
@pytest.mark.parametrize("divisor", [3.0, -2.5])
def test_scalar_division_alias_version_and_cpu_parity(layout, divisor):
    assert pytorch_vulkan.is_available()
    base = torch.arange(12, dtype=torch.float32).reshape(3, 4) - 4.5
    if layout == "scalar":
        base = torch.tensor(.75)
    if layout == "empty":
        base = torch.empty(0, 3)
    cpu, vk = base.clone(), base.to("vk:0")
    if layout == "transpose":
        cpu, vk = cpu.t(), vk.t()
    if layout == "offset":
        cpu, vk = cpu[1:], vk[1:]
    before, pointer = vk._version, vk.data_ptr()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    result = torch.ops.aten.div_.Scalar(vk, divisor)
    cpu.div_(divisor)
    assert result is vk and result.data_ptr() == pointer
    assert result._version == before + 1
    pytorch_vulkan._C.synchronize()
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    assert counters[2:] == (0, 0)
    assert counters[0] == (0 if layout == "empty" else 1)
    torch.testing.assert_close(result.cpu(), cpu)


def test_scalar_division_rejects_expanded_write_before_dispatch():
    base = torch.tensor([[.2, -.4, .7]]).to("vk:0")
    expanded = base.expand(2, 3)
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="overlap"):
        torch.ops.aten.div_.Scalar(expanded, 2)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_scalar_division_preserves_generated_autograd_leaf_preflight():
    leaf = torch.tensor([.2, -.4]).to("vk:0").requires_grad_()
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="leaf Variable"):
        torch.ops.aten.div_.Scalar(leaf, 2)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


@pytest.mark.parametrize("case", ["bool", "complex", "nonfinite", "overflow", "underflow"])
def test_scalar_division_narrow_dtype_scalar_preflight(case):
    base = torch.tensor([True, False]) if case == "bool" else torch.tensor([.2, -.4])
    tensor = base.to("vk:0")
    divisor = {"bool": 2, "complex": 1 + 2j, "nonfinite": float("inf"),
               "overflow": 1e100, "underflow": 1e-100}[case]
    pytorch_vulkan._C.reset_execution_counters()
    with pytest.raises(RuntimeError, match="float32|real numeric|non-finite|underflows"):
        torch.ops.aten.div_.Scalar(tensor, divisor)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (0, 0, 0, 0)


def test_direct_scalar_division_zero_divisor_matches_cpu_ieee_values():
    cpu = torch.tensor([-1., 0., 1.])
    tensor = cpu.to("vk:0")
    pytorch_vulkan._C.reset_execution_counters()
    torch.ops.aten.div_.Scalar(tensor, 0.)
    cpu.div_(0.)
    pytorch_vulkan._C.synchronize()
    assert pytorch_vulkan._C.execution_counter_snapshot() == (1, 0, 0, 0)
    torch.testing.assert_close(tensor.cpu(), cpu, equal_nan=True)
