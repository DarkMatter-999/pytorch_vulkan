"""Validation at the boundary between Vulkan tensors and the compiler."""

import torch


def _is_symbolic_dimension(dimension):
    """Return whether a shape dimension is symbolic without materializing it."""
    is_symbolic = getattr(dimension, "is_symbolic", None)
    if callable(is_symbolic) and is_symbolic():
        return True
    return isinstance(dimension, torch.SymInt)


def validate_compiler_tensor_metadata(tensor) -> None:
    """Validate the static tensor contract required by the Vulkan compiler."""
    if tensor.device != torch.device("vk:0"):
        raise RuntimeError("Vulkan compiler requires device vk:0")
    if tensor.dtype != torch.float32 or tensor.layout != torch.strided:
        raise RuntimeError("Vulkan compiler requires strided contiguous F32 tensors")
    if any(_is_symbolic_dimension(dim) for dim in tensor.shape):
        raise RuntimeError("Vulkan compiler requires static contiguous F32 metadata")
    if not tensor.is_contiguous() or tensor.storage_offset() != 0:
        raise RuntimeError("Vulkan compiler requires zero-offset contiguous tensors")
