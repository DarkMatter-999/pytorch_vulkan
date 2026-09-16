"""Explicit serialization for Vulkan tensors.

Generic ``torch.save`` remains a CPU-only fallback: materialize Vulkan tensors
with ``tensor.cpu()`` before passing them to PyTorch serialization.
"""

import torch

from ._C import is_available as _is_vulkan_available


_FORMAT = "pytorch_vulkan"
_VERSION = 1
_NODE_TAG = "pytorch_vulkan.node.v1"
_NODE_KINDS = {"tensor", "dict", "list", "tuple"}
_STORAGE_KEYS = {"dtype", "nbytes", "data"}


def _is_vk(tensor):
    return tensor.device.type in {"privateuseone", "vulkan", "vk"}


def _is_primitive(value):
    return value is None or isinstance(value, (bool, int, float, str))


def _checked_layout_values(dtype, layout, sizes, strides, offset,
                           storage_elements, label):
    if dtype != torch.float32:
        raise ValueError(f"{label}: unsupported dtype {dtype}; only float32 is supported")
    if layout != torch.strided:
        raise ValueError(f"{label}: only strided tensors are supported")
    if (not isinstance(sizes, (list, tuple)) or
            not isinstance(strides, (list, tuple)) or
            any(type(value) is not int for value in sizes) or
            any(type(value) is not int for value in strides) or
            type(offset) is not int):
        raise ValueError(f"{label}: integer layout metadata is required")
    if offset < 0 or any(size < 0 for size in sizes):
        raise ValueError(f"{label}: sizes and storage offset must be non-negative")
    if any(stride < 0 for stride in strides):
        raise ValueError(f"{label}: strides must be non-negative")
    if len(sizes) != len(strides):
        raise ValueError(f"{label}: sizes and strides must have matching ranks")
    if offset > storage_elements:
        raise ValueError(f"{label}: storage offset is outside the serialized storage")
    if not any(size == 0 for size in sizes):
        reachable = offset + sum((size - 1) * stride for size, stride in zip(sizes, strides))
        if reachable >= storage_elements:
            raise ValueError(f"{label}: view reaches outside the serialized storage")
        dimensions = sorted((stride, size) for size, stride in zip(sizes, strides)
                            if size > 1)
        span = 0
        for stride, size in dimensions:
            if stride <= span:
                if any(stride == 0 for stride, _ in dimensions):
                    raise ValueError(f"{label}: overlapping view metadata is unsupported")
                element_count = 1
                for value in sizes:
                    element_count *= value
                if element_count > 1 << 20:
                    raise ValueError(f"{label}: overlap classification is unsupported")
                addresses = set()
                for linear in range(element_count):
                    remaining = linear
                    address = 0
                    for size, item_stride in reversed(tuple(zip(sizes, strides))):
                        coordinate = remaining % size
                        remaining //= size
                        address += coordinate * item_stride
                    if address in addresses:
                        raise ValueError(f"{label}: overlapping view metadata is unsupported")
                    addresses.add(address)
                break
            span += (size - 1) * stride
    return tuple(sizes), tuple(strides), offset


def _checked_layout(tensor, storage_elements, label):
    return _checked_layout_values(
        tensor.dtype, tensor.layout, tensor.size(), tensor.stride(),
        tensor.storage_offset(), storage_elements, label)


def _validate_vulkan_tensor_metadata(tensor, label):
    if not _is_vk(tensor):
        raise ValueError(f"{label}: expected a vk:0 tensor, got {tensor.device}")
    if tensor.device.index not in (None, 0):
        raise ValueError(f"{label}: Vulkan serialization supports only vk:0, got {tensor.device}")
    storage_elements = int(tensor.untyped_storage().nbytes()) // tensor.element_size()
    return _checked_layout(tensor, storage_elements, label)


def _storage_owner(tensor):
    owner = tensor
    while isinstance(getattr(owner, "_base", None), torch.Tensor):
        owner = owner._base
    storage_elements = int(owner.untyped_storage().nbytes()) // owner.element_size()
    if (owner.storage_offset() != 0 or not owner.is_contiguous() or
            owner.numel() != storage_elements):
        raise ValueError("Vulkan storage must have a contiguous offset-zero owner")
    return owner


def _node(kind, payload):
    return (_NODE_TAG, kind, payload)


def save(obj, file):
    """Save tensors through CPU-owned bytes without registering a torch package."""
    storages = []
    storage_ids = {}
    active_containers = set()

    def encode_container(value, encoder):
        identity = id(value)
        if identity in active_containers:
            raise ValueError("cannot serialize cyclic dict/list/tuple objects")
        active_containers.add(identity)
        try:
            return encoder()
        finally:
            active_containers.remove(identity)

    def encode(value):
        if isinstance(value, torch.Tensor):
            if not _is_vk(value):
                if value.device.type != "cpu":
                    raise TypeError(f"unsupported tensor device {value.device}")
                return value
            _validate_vulkan_tensor_metadata(value, "Vulkan tensor")
            storage_id = value.untyped_storage()._cdata
            if storage_id not in storage_ids:
                owner = _storage_owner(value)
                cpu_owner = torch.empty(owner.numel(), dtype=value.dtype).copy_(owner)
                storage_ids[storage_id] = len(storages)
                storages.append({
                    "dtype": "float32",
                    "nbytes": int(cpu_owner.numel() * cpu_owner.element_size()),
                    "data": cpu_owner,
                })
            index = storage_ids[storage_id]
            sizes, strides, offset = _checked_layout(
                value, int(storages[index]["data"].numel()), "Vulkan tensor"
            )
            return _node("tensor", (
                index, "float32", sizes, strides, offset, "vk:0", value.requires_grad,
            ))
        if isinstance(value, dict):
            def encode_dict():
                entries = []
                for key, item in value.items():
                    if not _is_primitive(key):
                        raise TypeError(f"unsupported dictionary key type {type(key).__name__}")
                    entries.append((key, encode(item)))
                return _node("dict", tuple(entries))
            return encode_container(value, encode_dict)
        if isinstance(value, list):
            return encode_container(value, lambda: _node("list", tuple(encode(item) for item in value)))
        if isinstance(value, tuple):
            return encode_container(value, lambda: _node("tuple", tuple(encode(item) for item in value)))
        if _is_primitive(value):
            return value
        raise TypeError(f"unsupported object type {type(value).__name__}")

    torch.save({"format": _FORMAT, "version": _VERSION,
                "storages": storages, "object": encode(obj)}, file)


def _validate_tensor_node(node, storage_sizes):
    if (not isinstance(node, tuple) or len(node) != 3 or
            node[0] != _NODE_TAG or node[1] != "tensor" or
            not isinstance(node[2], tuple) or len(node[2]) != 7):
        raise ValueError("invalid encoded tensor node")
    index, dtype, sizes, strides, offset, device, requires_grad = node[2]
    if type(index) is not int or index < 0 or index >= len(storage_sizes):
        raise ValueError("tensor references an invalid storage")
    if dtype != "float32":
        raise ValueError("tensor has unsupported dtype")
    if device != "vk:0" or type(requires_grad) is not bool:
        raise ValueError("tensor has unsupported or malformed metadata")
    return _checked_layout_values(
        torch.float32, torch.strided, sizes, strides, offset,
        storage_sizes[index], "serialized tensor")


def _scan(value, storage_sizes, active):
    if _is_primitive(value):
        return False
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("unsupported encoded tensor device")
        return False
    if not isinstance(value, tuple) or len(value) != 3 or value[0] != _NODE_TAG:
        if isinstance(value, (dict, list, tuple)):
            identity = id(value)
            if identity in active:
                raise ValueError("cyclic encoded object graph")
            active.add(identity)
            try:
                items = value.items() if isinstance(value, dict) else value
                for item in items:
                    if isinstance(value, dict):
                        _scan(item[0], storage_sizes, active)
                        _scan(item[1], storage_sizes, active)
                    else:
                        _scan(item, storage_sizes, active)
            finally:
                active.remove(identity)
        raise ValueError("unsupported encoded object graph or leaf")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic encoded object graph")
    active.add(identity)
    try:
        kind, payload = value[1], value[2]
        if kind == "tensor":
            _validate_tensor_node(value, storage_sizes)
            return True
        if kind not in _NODE_KINDS or not isinstance(payload, tuple):
            raise ValueError("unsupported encoded node")
        if kind == "dict":
            contains_vk = False
            for entry in payload:
                if (not isinstance(entry, tuple) or len(entry) != 2 or
                        not _is_primitive(entry[0])):
                    raise ValueError("unsupported encoded dictionary entry")
                contains_vk = _scan(entry[1], storage_sizes, active) or contains_vk
            return contains_vk
        contains_vk = False
        for item in payload:
            contains_vk = _scan(item, storage_sizes, active) or contains_vk
        return contains_vk
    finally:
        active.remove(identity)


def _validate_payload(payload):
    if not isinstance(payload, dict) or payload.get("format") != _FORMAT:
        raise ValueError("invalid pytorch_vulkan serialization format")
    if payload.get("version") != _VERSION:
        raise ValueError("unsupported pytorch_vulkan serialization version")
    storages = payload.get("storages")
    if not isinstance(storages, list):
        raise ValueError("invalid serialized storage table")
    storage_sizes = []
    for index, record in enumerate(storages):
        if not isinstance(record, dict) or set(record) != _STORAGE_KEYS or record.get("dtype") != "float32":
            raise ValueError(f"storage {index}: unsupported dtype or metadata")
        data = record["data"]
        nbytes = record["nbytes"]
        if (not isinstance(data, torch.Tensor) or data.device.type != "cpu" or
                data.dtype != torch.float32 or data.layout != torch.strided or
                data.dim() != 1 or not data.is_contiguous() or
                type(nbytes) is not int or nbytes < 0 or
                nbytes != data.numel() * data.element_size()):
            raise ValueError(f"storage {index}: invalid byte count or storage data")
        storage_sizes.append(data.numel())
    return _contains_vk(payload.get("object"), storage_sizes)


def load(file, map_location=None):
    """Load explicit Vulkan serialization to CPU or ``vk:0`` storage."""
    if map_location is None:
        target = None
    elif isinstance(map_location, str) and map_location in {"cpu", "vk", "vk:0"}:
        target = "cpu" if map_location == "cpu" else "vk:0"
    elif isinstance(map_location, torch.device) and map_location.type == "cpu":
        target = "cpu"
    elif (isinstance(map_location, torch.device) and
          map_location.type in {"vk", "privateuseone", "vulkan"}):
        if map_location.index not in (None, 0):
            raise ValueError("map_location must use only vk:0")
        target = "vk:0"
    else:
        raise ValueError("map_location must be None, 'cpu', or 'vk'")

    payload = torch.load(file, map_location="cpu", weights_only=True)
    contains_vk = _validate_payload(payload)
    target = target or ("vk:0" if contains_vk else "cpu")
    if target == "vk:0" and not _is_vulkan_available():
        raise RuntimeError("serialized tensor requires Vulkan device vk:0, but Vulkan is unavailable")

    storage_values = [record["data"] for record in payload["storages"]]
    if target == "vk:0":
        storage_values = [torch.empty_like(data, device="vk").copy_(data)
                          for data in storage_values]
    return _decode(payload["object"], storage_values,
                   [data.numel() for data in storage_values], set())


def _decode(value, storage_values, storage_sizes, active):
    if _is_primitive(value):
        return value
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise ValueError("unsupported encoded tensor device")
        return value
    if not isinstance(value, tuple) or len(value) != 3 or value[0] != _NODE_TAG:
        raise ValueError("unsupported encoded object graph or leaf")
    identity = id(value)
    if identity in active:
        raise ValueError("cyclic encoded object graph")
    active.add(identity)
    try:
        kind, payload = value[1], value[2]
        if kind == "tensor":
            _validate_tensor_node(value, storage_sizes)
            index, _, sizes, strides, offset, _, requires_grad = payload
            tensor = torch.as_strided(storage_values[index], tuple(sizes),
                                      tuple(strides), offset)
            if requires_grad:
                tensor.requires_grad_(True)
            return tensor
        if kind not in {"dict", "list", "tuple"} or not isinstance(payload, tuple):
            raise ValueError("unsupported encoded node")
        if kind == "dict":
            result = {}
            for entry in payload:
                if (not isinstance(entry, tuple) or len(entry) != 2 or
                        not _is_primitive(entry[0])):
                    raise ValueError("unsupported encoded dictionary entry")
                result[entry[0]] = _decode(entry[1], storage_values, storage_sizes, active)
            return result
        values = [_decode(item, storage_values, storage_sizes, active) for item in payload]
        return values if kind == "list" else tuple(values)
    finally:
        active.remove(identity)


def _contains_vk(value, storage_sizes):
    return _scan(value, storage_sizes, set())
