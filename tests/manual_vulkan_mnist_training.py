"""Explicit, real-data MNIST training acceptance gate (never auto-downloads)."""

import os
from pathlib import Path

import pytest

import pytorch_vulkan

from mnist_training_parity import assert_vulkan_tensor_metadata, run_real_pair


@pytest.fixture
def backend():
    assert pytorch_vulkan.is_available(), "Vulkan backend is required for MNIST qualification"
    return pytorch_vulkan


def _require_data_root():
    configured = os.environ.get("PYTORCH_VULKAN_MNIST_ROOT")
    assert configured, "PYTORCH_VULKAN_MNIST_ROOT must name a prepared MNIST data root"
    root = Path(configured).expanduser().resolve()
    assert root.is_dir(), f"MNIST data root does not exist: {root}"
    return root


@pytest.mark.parametrize("set_to_none", [True, False])
def test_real_mnist_parity(backend, set_to_none, capfd):
    root = _require_data_root()
    result = run_real_pair(root, set_to_none, backend=backend)
    captured = capfd.readouterr()
    assert result["updates"] == 34
    assert result["train_batch_sizes"] == ([16] * 16 + [1]) * 2
    assert result["evaluation_counts"] == [129, 129]
    assert result["evaluation_batch_sizes"] == [[16] * 8 + [1]] * 2
    assert result["parity"]["passed"] is True
    assert len(result["evaluations"]) == 2
    assert all(item["comparison"]["passed"] for item in result["evaluations"])
    assert result["metric_device"] == "cpu"
    assert_vulkan_tensor_metadata(result["vulkan"]["tensor_metadata"])
    assert "Validation Error:" not in captured.out + captured.err
