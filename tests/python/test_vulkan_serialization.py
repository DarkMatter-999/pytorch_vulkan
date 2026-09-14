import io

import pytest
import torch

import pytorch_vulkan
import pytorch_vulkan.serialization as serialization


def _source():
    return torch.arange(12, dtype=torch.float32, device="cpu").reshape(3, 4)


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="no Vulkan device")
def test_explicit_round_trip_preserves_views_and_shared_storage():
    source = _source().to("vk")
    view = source.as_strided((2, 2), (1, 2), 1)

    buffer = io.BytesIO()
    pytorch_vulkan.save({"base": source, "view": view}, buffer)

    buffer.seek(0)
    restored = pytorch_vulkan.load(buffer, map_location="cpu")
    assert restored["base"].device.type == "cpu"
    assert restored["base"].dtype == source.dtype
    assert tuple(restored["base"].size()) == (3, 4)
    assert tuple(restored["base"].stride()) == (4, 1)
    assert restored["view"].storage_offset() == 1
    assert tuple(restored["view"].stride()) == (1, 2)
    assert restored["view"].untyped_storage()._cdata == restored["base"].untyped_storage()._cdata
    torch.testing.assert_close(restored["base"], source.cpu(), rtol=0, atol=0)
    expected_view = source.cpu().as_strided((2, 2), (1, 2), 1)
    torch.testing.assert_close(restored["view"], expected_view, rtol=0, atol=0)

    buffer.seek(0)
    restored_vk = pytorch_vulkan.load(buffer, map_location="vk")
    assert restored_vk["base"].device == torch.device("vk:0")
    assert restored_vk["view"].untyped_storage()._cdata == restored_vk["base"].untyped_storage()._cdata
    torch.testing.assert_close(
        restored_vk["base"].cpu().as_strided((2, 2), (1, 2), 1),
        expected_view,
        rtol=0,
        atol=0,
    )


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="no Vulkan device")
def test_explicit_round_trip_preserves_zero_element_tensor():
    tensor = torch.empty((0, 3), dtype=torch.float32, device="vk")
    buffer = io.BytesIO()
    pytorch_vulkan.save(tensor, buffer)
    buffer.seek(0)
    restored = pytorch_vulkan.load(buffer, map_location="cpu")
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.float32
    assert tuple(restored.size()) == (0, 3)
    assert tuple(restored.stride()) == tuple(tensor.stride())
    assert restored.storage_offset() == tensor.storage_offset()


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="no Vulkan device")
def test_explicit_round_trip_preserves_requires_grad():
    tensor = _source().to("vk")
    tensor.requires_grad_(True)
    buffer = io.BytesIO()
    pytorch_vulkan.save(tensor, buffer)

    buffer.seek(0)
    restored_cpu = pytorch_vulkan.load(buffer, map_location="cpu")
    assert restored_cpu.requires_grad is True

    buffer.seek(0)
    restored_vk = pytorch_vulkan.load(buffer, map_location="vk")
    assert restored_vk.requires_grad is True


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="no Vulkan device")
def test_explicit_round_trip_preserves_chained_view_metadata():
    source = _source().to("vk")
    view = source.transpose(0, 1)[:, 1:3]

    buffer = io.BytesIO()
    pytorch_vulkan.save({"source": source, "view": view}, buffer)
    buffer.seek(0)
    restored = pytorch_vulkan.load(buffer, map_location="vk")

    assert tuple(restored["view"].shape) == tuple(view.shape)
    assert tuple(restored["view"].stride()) == tuple(view.stride())
    assert restored["view"].storage_offset() == view.storage_offset()
    assert restored["view"].untyped_storage()._cdata == restored["source"].untyped_storage()._cdata
    torch.testing.assert_close(restored["view"].cpu(), view.cpu(), rtol=0, atol=0)


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="no Vulkan device")
def test_generic_torch_save_requires_cpu_materialization():
    tensor = _source().to("vk")
    with pytest.raises((NotImplementedError, RuntimeError)):
        torch.save(tensor, io.BytesIO())

    buffer = io.BytesIO()
    torch.save(tensor.cpu(), buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location="cpu", weights_only=True)
    torch.testing.assert_close(restored, tensor.cpu(), rtol=0, atol=0)


def test_load_rejects_malformed_metadata_before_vulkan_restore():
    payload = {
        "format": "pytorch_vulkan",
        "version": 1,
        "storages": [{"dtype": "float32", "nbytes": 4, "data": torch.zeros(1)}],
        "object": ("pytorch_vulkan.node.v1", "tensor",
                   (0, "float64", [1], [1], 0, "vk:0", False)),
    }
    source = io.BytesIO()
    torch.save(payload, source)
    source.seek(0)
    with pytest.raises(ValueError, match="unsupported dtype"):
        pytorch_vulkan.load(source, map_location="vk")


def test_load_rejects_invalid_map_location():
    with pytest.raises(ValueError, match="map_location"):
        pytorch_vulkan.load(io.BytesIO(), map_location="cuda")
    with pytest.raises(ValueError, match="vk:0"):
        pytorch_vulkan.load(io.BytesIO(), map_location=torch.device("vk:1"))


def test_save_rejects_unsupported_and_cyclic_objects():
    with pytest.raises(TypeError, match="unsupported object type"):
        pytorch_vulkan.save(object(), io.BytesIO())

    cyclic = []
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match="cyclic"):
        pytorch_vulkan.save(cyclic, io.BytesIO())


def test_user_dictionary_marker_collision_is_not_a_tensor():
    value = {"__pytorch_vulkan_tensor__": "tensor-v1", "storage": 0,
             "dtype": "float32", "sizes": [1], "strides": [1],
             "storage_offset": 0, "device": "vk:0", "requires_grad": False}
    source = io.BytesIO()
    pytorch_vulkan.save(value, source)
    source.seek(0)
    assert pytorch_vulkan.load(source, map_location="cpu") == value


def test_load_rejects_cyclic_and_unsupported_graphs():
    cyclic = []
    cyclic.append(cyclic)
    payload = {"format": "pytorch_vulkan", "version": 1,
               "storages": [], "object": cyclic}
    source = io.BytesIO()
    torch.save(payload, source)
    source.seek(0)
    with pytest.raises(ValueError, match="cyclic"):
        pytorch_vulkan.load(source, map_location="cpu")

    source = io.BytesIO()
    torch.save({"format": "pytorch_vulkan", "version": 1,
                "storages": [], "object": {"unsupported": 1}}, source)
    source.seek(0)
    with pytest.raises(ValueError, match="unsupported encoded"):
        pytorch_vulkan.load(source, map_location="cpu")


@pytest.mark.parametrize("sizes, strides, offset", [
    ([1.0], [1], 0),
    ([1], ["1"], 0),
    ([1], [1], False),
])
def test_load_rejects_non_integer_layout_metadata(sizes, strides, offset):
    node = ("pytorch_vulkan.node.v1", "tensor",
            (0, "float32", sizes, strides, offset, "vk:0", False))
    payload = {
        "format": "pytorch_vulkan", "version": 1,
        "storages": [{"dtype": "float32", "nbytes": 4, "data": torch.zeros(1)}],
        "object": node,
    }
    source = io.BytesIO()
    torch.save(payload, source)
    source.seek(0)
    with pytest.raises(ValueError, match="integer layout metadata"):
        pytorch_vulkan.load(source, map_location="cpu")


def test_load_rejects_invalid_storage_byte_count():
    payload = {
        "format": "pytorch_vulkan",
        "version": 1,
        "storages": [{"dtype": "float32", "nbytes": 8, "data": torch.zeros(1)}],
        "object": None,
    }
    source = io.BytesIO()
    torch.save(payload, source)
    source.seek(0)
    with pytest.raises(ValueError, match="invalid byte count"):
        pytorch_vulkan.load(source, map_location="vk")


@pytest.mark.skipif(not pytorch_vulkan.is_available(), reason="requires a Vulkan device")
def test_default_restore_rejects_unavailable_vulkan(monkeypatch):
    payload = {
        "format": "pytorch_vulkan",
        "version": 1,
        "storages": [{"dtype": "float32", "nbytes": 4, "data": torch.zeros(1)}],
        "object": ("pytorch_vulkan.node.v1", "tensor",
                   (0, "float32", [], [], 0, "vk:0", False)),
    }
    source = io.BytesIO()
    torch.save(payload, source)
    source.seek(0)
    monkeypatch.setattr(serialization, "_is_vulkan_available", lambda: False)
    with pytest.raises(RuntimeError, match="Vulkan.*unavailable"):
        pytorch_vulkan.load(source)
