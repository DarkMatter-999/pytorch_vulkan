import pytest

import pytorch_vulkan

from mnist_training_parity import run_synthetic_pair


@pytest.fixture
def backend():
    assert pytorch_vulkan.is_available(), "Vulkan backend is required for MNIST qualification"
    return pytorch_vulkan


@pytest.mark.parametrize("batch_size", [1, 2, 16])
@pytest.mark.parametrize("set_to_none", [True, False])
def test_synthetic_mnist_first_order(backend, batch_size, set_to_none, capfd):
    result = run_synthetic_pair(batch_size, set_to_none, backend=backend)
    captured = capfd.readouterr()
    assert result["updates"] == 3
    assert result["batch_size"] == batch_size
    assert result["parity"]["passed"] is True
    assert result["parameter_names"] == ["1.weight", "1.bias", "3.weight", "3.bias"]
    assert "Validation Error:" not in captured.out + captured.err
