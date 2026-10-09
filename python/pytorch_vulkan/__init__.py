import os

import torch
import inspect

from . import _C
from ._C import current_device, device_count, formatter_double_supported, is_available, set_device
from .compiler import compiler_stats, vulkan_backend
from .compiler_metadata import validate_compiler_tensor_metadata
from .serialization import load, save


_INITIAL_PID = os.getpid()


class _VulkanDeviceModule:
    @staticmethod
    def is_available():
        return is_available()

    @staticmethod
    def device_count():
        return device_count()

    @staticmethod
    def current_device():
        return current_device()

    @staticmethod
    def set_device(index):
        return set_device(index)

    @staticmethod
    def synchronize():
        return _C.synchronize()

    @staticmethod
    def _is_in_bad_fork():
        # Shim over the PID recorded at import. The backend already tracks this
        # authoritatively via a pthread_atfork child handler
        # (pytorch_vulkan::inherited_fork_state in src/vulkan_platform.cpp), but
        # that symbol is not exposed to Python yet. Replace this with a binding
        # to inherited_fork_state() so the two cannot disagree.
        return os.getpid() != _INITIAL_PID

    # manual_seed_all is deliberately NOT implemented. Seeding a device RNG is
    # meaningless here: the Vulkan backend has no device random ops
    # (src/random_ops.cpp is not in the build, and torch.randn(device="vk:0")
    # is unsupported). PyTorch's warning that seeding "does not take effect" is
    # therefore accurate and is left in place rather than silenced by a stub
    # that would appear to work while doing nothing. Adding it properly requires
    # a device RNG, which is out of scope.


torch.utils.rename_privateuse1_backend("vk")
torch._register_device_module("vk", _VulkanDeviceModule)


def _materialize_optimizer_parameters(parameters):
    if isinstance(parameters, torch.Tensor):
        return (parameters,)
    parameters = tuple(parameters)
    if parameters and isinstance(parameters[0], dict):
        return tuple({**group, "params": tuple(group["params"])} for group in parameters)
    return parameters


def validate_basic_optimizer(parameters, options=None, require_vulkan=False, unsupported=()):
    parameters = _materialize_optimizer_parameters(parameters)
    normalized = tuple(
        parameter for group in parameters for parameter in group["params"]
    ) if parameters and isinstance(parameters[0], dict) else parameters
    has_vk = any(parameter.device.type == "vk" for parameter in normalized)
    if require_vulkan and not has_vk:
        raise RuntimeError("Vulkan basic optimizer requires vk:0 parameters")
    if not has_vk:
        return parameters
    for name in unsupported:
        if options.get(name, False):
            raise RuntimeError(f"Vulkan basic optimizer rejects {name}")
    if parameters and isinstance(parameters[0], dict):
        for group in parameters:
            for name in unsupported:
                if group.get(name, False):
                    raise RuntimeError(f"Vulkan basic optimizer rejects {name}")
    for parameter in normalized:
        if parameter.device.type != "vk" or parameter.device.index != 0:
            raise RuntimeError("Vulkan basic optimizer supports only vk:0 parameters")
        if parameter.dtype != torch.float32:
            raise RuntimeError("Vulkan basic optimizer supports only float32 parameters")
        if torch._debug_has_internal_overlap(parameter) != 0:
            raise RuntimeError("Vulkan basic optimizer rejects overlapping parameters")
        if parameter.layout != torch.strided or not parameter.is_contiguous():
            raise RuntimeError("Vulkan basic optimizer requires contiguous strided parameters")
        if parameter.storage_offset() != 0:
            raise RuntimeError("Vulkan basic optimizer requires zero-offset parameters")
    return parameters


def _validate_basic_optimizer_state(optimizer):
    for parameter in optimizer.param_groups[0]["params"] if len(optimizer.param_groups) == 1 else (
        parameter for group in optimizer.param_groups for parameter in group["params"]
    ):
        if parameter.device.type != "vk":
            continue
        validate_basic_optimizer([parameter], require_vulkan=True)
        gradient = parameter.grad
        if gradient is not None:
            validate_basic_optimizer([gradient], require_vulkan=True)
            if tuple(gradient.shape) != tuple(parameter.shape):
                raise RuntimeError("Vulkan basic optimizer gradient shape must match parameter")
        for name, value in optimizer.state[parameter].items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.device.type != "vk":
                if name == "step" and value.device.type == "cpu" and value.numel() == 1:
                    continue
                raise RuntimeError(
                    "Vulkan basic optimizer state tensors must remain Vulkan-resident"
                )
            validate_basic_optimizer([value], require_vulkan=True)
            if tuple(value.shape) != tuple(parameter.shape):
                raise RuntimeError(
                    f"Vulkan basic optimizer state {name} shape must match parameter"
                )


_ORIGINAL_SGD = torch.optim.SGD
_ORIGINAL_ADAM = torch.optim.Adam


class _VulkanValidatedSGD(torch.optim.SGD):
    def __init__(self, params, *args, **kwargs):
        signature = inspect.signature(_ORIGINAL_SGD.__init__)
        bound = signature.bind(self, params, *args, **kwargs)
        bound.apply_defaults()
        params = validate_basic_optimizer(
            bound.arguments["params"], bound.arguments,
            unsupported=("nesterov", "maximize", "foreach", "differentiable", "fused"),
        )
        super().__init__(params, *args, **kwargs)

    def step(self, closure=None):
        _validate_basic_optimizer_state(self)
        owns_training_step = not _C.training_step_active()
        if owns_training_step:
            _C.begin_training_step()
        try:
            result = super().step(closure)
            if owns_training_step:
                _C.end_training_step()
            return result
        except Exception:
            if owns_training_step:
                _C.cancel_training_step()
            raise


class _VulkanValidatedAdam(torch.optim.Adam):
    def __init__(self, params, *args, **kwargs):
        signature = inspect.signature(_ORIGINAL_ADAM.__init__)
        bound = signature.bind(self, params, *args, **kwargs)
        bound.apply_defaults()
        params = validate_basic_optimizer(
            bound.arguments["params"], bound.arguments,
            unsupported=("amsgrad", "maximize", "foreach", "fused", "differentiable", "capturable"),
        )
        super().__init__(params, *args, **kwargs)

    def step(self, closure=None):
        _validate_basic_optimizer_state(self)
        owns_training_step = not _C.training_step_active()
        if owns_training_step:
            _C.begin_training_step()
        try:
            result = super().step(closure)
            if owns_training_step:
                _C.end_training_step()
            return result
        except Exception:
            if owns_training_step:
                _C.cancel_training_step()
            raise


torch.optim.SGD = _VulkanValidatedSGD
torch.optim.Adam = _VulkanValidatedAdam

__all__ = ["compiler_stats", "device_count", "formatter_double_supported", "is_available", "load",
           "save", "validate_compiler_tensor_metadata", "vulkan_backend"]


def __getattr__(name):
    if name.startswith('_'):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    try:
        return getattr(torch, name)
    except AttributeError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None


def __dir__():
    return sorted(set(globals()) | {n for n in dir(torch) if not n.startswith('_')})
