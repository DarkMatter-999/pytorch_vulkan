"""Float32 alpha-add numerical and optional-capability regression boundaries."""

import ctypes
import pathlib
import re

import pytest
import pytorch_vulkan
import torch


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_shared_pointwise_modules_need_no_optional_arithmetic_capabilities():
    root = pathlib.Path(__file__).resolve().parents[2]
    header = (root / "src/vulkan/shaders/generated/pointwise_spv.h").read_text()
    modules = re.findall(r"uint32_t (\w+)\[\] = \{(.*?)\};", header, re.S)
    assert len(modules) == 10
    for name, body in modules:
        words = [int(word, 16) for word in re.findall(r"0x([0-9a-f]+)U", body)]
        index = 5
        capabilities = set()
        while index < len(words):
            count, opcode = words[index] >> 16, words[index] & 0xFFFF
            assert count > 0
            if opcode == 17:  # OpCapability
                capabilities.add(words[index + 1])
            index += count
        assert index == len(words)
        assert not capabilities.intersection({9, 10, 11}), (name, capabilities)


def _fmaf(lhs, rhs, alpha):
    # An independently rounded IEEE float32 fused operation, not Python double.
    function = ctypes.CDLL("libm.so.6").fmaf
    function.argtypes = [ctypes.c_float] * 3
    function.restype = ctypes.c_float
    return function(rhs, alpha, lhs)


@pytest.mark.parametrize("alpha", [1.0000001192092896, 1.1, 2.0**-149, 2.0**-100, 2.0**100, -1.0, 0.0, 1.0])
def test_alpha_add_exponent_and_cancellation_precision(vulkan_backend, alpha):
    generator = torch.Generator().manual_seed(1701)
    bits = torch.randint(0, 2**32, (128, 2), dtype=torch.int64, generator=generator)
    # Keep all tensor inputs finite, while retaining subnormal and huge exponents.
    bits = torch.where((bits & 0x7F800000) == 0x7F800000, bits & ~0x00800000, bits)
    values = bits.to(torch.int32).view(torch.float32)
    lhs, rhs = values[:, 0].contiguous(), values[:, 1].contiguous()
    cpu = torch.add(lhs, rhs, alpha=alpha)
    oracle = torch.tensor([_fmaf(a, b, alpha) for a, b in zip(lhs.tolist(), rhs.tolist())])
    torch.testing.assert_close(cpu, oracle, rtol=0, atol=0, equal_nan=True)
    actual = torch.add(lhs.to("vk:0"), rhs.to("vk:0"), alpha=alpha)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), cpu, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("direction", ["tensor", "right", "left", "explicit"])
@pytest.mark.parametrize("exponent", [-140, -100, -40, 0, 40, 100])
def test_alpha_add_cancellation_scaled(vulkan_backend, direction, exponent):
    alpha = 1.0000001192092896
    lhs = torch.tensor([-10000001024.0 * 2.0**exponent], dtype=torch.float32)
    rhs = torch.tensor([1e10 * 2.0**exponent], dtype=torch.float32)
    if not torch.isfinite(lhs).all() or not torch.isfinite(rhs).all():
        # Keep the full exponent range with a smaller significand.
        lhs = torch.tensor([-1.000000238418579 * 2.0**exponent])
        rhs = torch.tensor([1.0000001192092896 * 2.0**exponent])
    if direction == "tensor":
        cpu = torch.add(lhs, rhs, alpha=alpha)
        actual = torch.add(lhs.to("vk:0"), rhs.to("vk:0"), alpha=alpha)
    elif direction == "left":
        cpu = torch.add(rhs.item(), lhs, alpha=alpha)
        actual = torch.add(rhs.item(), lhs.to("vk:0"), alpha=alpha)
    elif direction == "explicit":
        cpu = torch.ops.aten.add.Scalar(lhs, rhs.item(), alpha)
        actual = torch.ops.aten.add.Scalar(lhs.to("vk:0"), rhs.item(), alpha)
    else:
        cpu = torch.add(lhs, rhs.item(), alpha=alpha)
        actual = torch.add(lhs.to("vk:0"), rhs.item(), alpha=alpha)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("direction", ["tensor", "right", "left", "explicit"])
def test_alpha_add_rounding_midpoint_and_subnormals(vulkan_backend, direction):
    alpha = 1.0000001192092896
    lhs = torch.tensor([-2.0**-100, 2.0**-149, -2.0**-149, 0.0])
    rhs = torch.tensor([1.5, 2.0**-126, 2.0**-126, 2.0**-149])
    cpu = torch.add(lhs, rhs, alpha=alpha)
    # A double product+sum loses the tiny negative addend at the rounding midpoint.
    double_result = (rhs.double() * alpha + lhs.double()).float()
    assert cpu[0].item() != double_result[0].item()
    if direction == "tensor":
        actual = torch.add(lhs.to("vk:0"), rhs.to("vk:0"), alpha=alpha)
    else:
        results = []
        for a, b in zip(lhs, rhs):
            if direction == "right":
                result = torch.add(a.reshape(1).to("vk:0"), b.item(), alpha=alpha)
            elif direction == "left":
                result = torch.add(a.item(), b.reshape(1).to("vk:0"), alpha=alpha)
            else:
                result = torch.ops.aten.add.Scalar(a.reshape(1).to("vk:0"), b.item(), alpha)
            pytorch_vulkan._C.synchronize()
            results.append(result.cpu().item())
        actual = torch.tensor(results)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize("alpha", [0.0, 1.0, 2.0, -2.0])
def test_alpha_add_nonfinite_tensor_behavior(vulkan_backend, alpha):
    lhs = torch.tensor([float("inf"), -float("inf"), float("nan"), 1.0, -0.0])
    rhs = torch.tensor([-float("inf"), 1.0, 2.0, float("inf"), -0.0])
    lhs = torch.cat((lhs, torch.tensor([-float("inf"), float("inf")])))
    rhs = torch.cat((rhs, torch.tensor([torch.finfo(torch.float32).max] * 2)))
    actual = torch.add(lhs.to("vk:0"), rhs.to("vk:0"), alpha=alpha)
    pytorch_vulkan._C.synchronize()
    torch.testing.assert_close(actual.cpu(), torch.add(lhs, rhs, alpha=alpha), rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("alpha", [1.0, 1.0000001192092896, 2.0, -2.0, 2.0**-149])
def test_alpha_add_overflow_cancellation_and_signed_zero(vulkan_backend, alpha):
    largest = torch.finfo(torch.float32).max
    lhs = torch.tensor([-largest, largest, -0.0, -0.0, 0.0, 2.0**-149, 0.0])
    rhs = torch.tensor([largest, -largest, -0.0, 0.0, -0.0, -2.0**-149, -2.0**-149])
    expected = torch.add(lhs, rhs, alpha=alpha)
    actual = torch.add(lhs.to("vk:0"), rhs.to("vk:0"), alpha=alpha)
    pytorch_vulkan._C.synchronize()
    assert torch.equal(actual.cpu().view(torch.int32), expected.view(torch.int32))
