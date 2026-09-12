import multiprocessing
import os

import pytest
import torch

import pytorch_vulkan


def _spawn_vulkan_child(queue, result_queue):
    tensor = queue.get()
    vulkan_tensor = tensor.to("vk")
    result = torch.add(vulkan_tensor, 2.0).cpu()
    result_queue.put((result.tolist(), result.device.type, pytorch_vulkan.is_available()))


def _fork_vulkan_child(write_fd, tensor):
    messages = []
    try:
        torch.add(tensor, 1.0)
        messages.append("operator unexpectedly succeeded")
    except Exception as error:
        messages.append(str(error))
    try:
        tensor.cpu()
        messages.append("CPU transfer unexpectedly succeeded")
    except Exception as error:
        messages.append(str(error))
    os.write(write_fd, "\n".join(messages).encode("utf-8"))
    del tensor
    os.close(write_fd)
    os._exit(0)


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_spawn_cpu_queue_boundary_and_child_vulkan_execution(vulkan_backend):
    context = multiprocessing.get_context("spawn")
    queue = context.Queue()
    result_queue = context.Queue()
    process = context.Process(target=_spawn_vulkan_child, args=(queue, result_queue))
    queue.put(torch.tensor([1.0, -2.5, 3.25], dtype=torch.float32))
    process.start()
    values, device_type, available = result_queue.get(timeout=30)
    process.join(timeout=30)

    assert process.exitcode == 0
    assert values == [3.0, -0.5, 5.25]
    assert device_type == "cpu"
    assert available is True


def test_fork_after_vulkan_initialization_is_rejected(vulkan_backend):
    tensor = torch.ones(1, dtype=torch.float32).to(vulkan_backend)
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(read_fd)
        _fork_vulkan_child(write_fd, tensor)

    os.close(write_fd)
    message = os.read(read_fd, 4096).decode("utf-8")
    os.close(read_fd)
    _, status = os.waitpid(pid, 0)

    assert os.waitstatus_to_exitcode(status) == 0
    assert "fork" in message.lower()
    assert "spawn" in message.lower()
    assert "cpu" in message.lower()
    assert len(message.splitlines()) == 2


def test_repeated_spawned_children_exit_cleanly(vulkan_backend):
    context = multiprocessing.get_context("spawn")
    for value in range(4):
        queue = context.Queue()
        result_queue = context.Queue()
        process = context.Process(target=_spawn_vulkan_child, args=(queue, result_queue))
        queue.put(torch.tensor([float(value)], dtype=torch.float32))
        process.start()
        values, device_type, available = result_queue.get(timeout=30)
        process.join(timeout=30)

        assert process.exitcode == 0
        assert values == [float(value) + 2.0]
        assert device_type == "cpu"
        assert available is True
