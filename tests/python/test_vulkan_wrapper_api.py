"""CPU-only checks for the wrapper facade and import-first startup boundary."""

import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _run_python(source, *, environment=None):
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(source)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def test_upstream_identity_and_package_names():
    import pytorch_vulkan as wrapper
    import torch as upstream
    from pytorch_vulkan import _C, compiler, compiler_metadata, serialization

    assert wrapper.Tensor is upstream.Tensor
    assert wrapper.nn is upstream.nn
    assert wrapper.optim is upstream.optim
    assert wrapper.autograd is upstream.autograd
    assert wrapper.tensor is upstream.tensor
    assert wrapper.float32 is upstream.float32
    assert wrapper.version is upstream.version
    assert wrapper.save is serialization.save
    assert wrapper.load is serialization.load
    assert wrapper.save is not upstream.save
    assert wrapper.load is not upstream.load
    assert wrapper._C is _C
    assert wrapper._C is not upstream._C
    for name in (
        "current_device", "device_count", "formatter_double_supported",
        "is_available", "set_device",
    ):
        assert getattr(wrapper, name) is getattr(_C, name)
    assert wrapper.compiler_stats is compiler.compiler_stats
    assert wrapper.vulkan_backend is compiler.vulkan_backend
    assert (
        wrapper.validate_compiler_tensor_metadata
        is compiler_metadata.validate_compiler_tensor_metadata
    )
    assert callable(wrapper.device_count)
    assert callable(wrapper.compiler_stats)
    assert wrapper.optim.SGD is wrapper._VulkanValidatedSGD
    assert wrapper.optim.Adam is wrapper._VulkanValidatedAdam
    assert set(wrapper.__all__) == {
        "compiler_stats", "device_count", "formatter_double_supported",
        "is_available", "load", "save", "validate_compiler_tensor_metadata",
        "vulkan_backend",
    }
    assert 'nn' in dir(wrapper)
    assert {
        name for name in dir(upstream) if not name.startswith('_')
    } <= set(dir(wrapper))
    with pytest.raises(AttributeError, match="pytorch_vulkan"):
        getattr(wrapper, 'not_a_pytorch_api_91837')
    with pytest.raises(AttributeError, match="pytorch_vulkan"):
        getattr(wrapper, '__not_a_module_protocol__')
    with pytest.raises(AttributeError, match="pytorch_vulkan"):
        getattr(wrapper, '_debug_has_internal_overlap')
    with pytest.raises(AttributeError, match="pytorch_vulkan"):
        getattr(wrapper, '__version__')


@pytest.mark.parametrize("prior_torch_import", [False, True])
def test_supported_import_history_without_backward(prior_torch_import):
    prefix = """
        import torch as upstream
        assert upstream.autograd.is_multithreading_enabled()
    """ if prior_torch_import else ""
    _run_python(textwrap.dedent(prefix) + textwrap.dedent("""
        import sys
        def reject_autograd_execution(frame, event, arg):
            if (event == 'call'
                    and frame.f_code.co_name in {'backward', 'grad', '_engine_run_backward'}
                    and '/torch/' in frame.f_code.co_filename):
                raise AssertionError('wrapper import must not execute autograd')
        sys.setprofile(reject_autograd_execution)
        import pytorch_vulkan as wrapper
        sys.setprofile(None)
        internal = sys.modules['torch']
        assert wrapper.torch is internal
        assert internal.__name__ == 'torch'
        assert internal.autograd.is_multithreading_enabled()
        assert internal._C._get_privateuse1_backend_name() == 'vk'
        assert internal.vk is wrapper._VulkanDeviceModule
        assert wrapper._C is not internal._C
        assert wrapper.nn is internal.nn
        assert wrapper.optim is internal.optim
        assert wrapper.autograd is internal.autograd
    """) + (textwrap.dedent("""
        assert internal is upstream
    """) if prior_torch_import else ""))


def test_wrapper_only_cpu_model_and_sgd():
    _run_python("""
        import pytorch_vulkan as torch
        assert torch.autograd.is_multithreading_enabled()
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[0.5, -0.25]]))
            model.bias.copy_(torch.tensor([0.1]))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs = torch.tensor([[1.0, 2.0], [-1.0, 1.0]])
        targets = torch.tensor([[0.3], [-0.2]])
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.MSELoss()(model(inputs), targets)
        torch.testing.assert_close(loss, torch.tensor(0.12125))
        loss.backward()
        torch.testing.assert_close(model.weight.grad, torch.tensor([[0.25, -0.85]]))
        torch.testing.assert_close(model.bias.grad, torch.tensor([-0.65]))
        optimizer.step()
        torch.testing.assert_close(model.weight, torch.tensor([[0.475, -0.165]]))
        torch.testing.assert_close(model.bias, torch.tensor([0.165]))
        assert all(parameter.device.type == 'cpu' for parameter in model.parameters())
        assert torch.autograd.is_multithreading_enabled()
    """)


def test_missing_vulkan_driver_preserves_unavailable_behavior(tmp_path):
    environment = os.environ.copy()
    environment["VK_ICD_FILENAMES"] = str(tmp_path / "missing-vulkan-driver.json")
    _run_python("""
        import pytorch_vulkan as wrapper
        assert wrapper.is_available() is False
        assert wrapper.torch.autograd.is_multithreading_enabled()
    """, environment=environment)
