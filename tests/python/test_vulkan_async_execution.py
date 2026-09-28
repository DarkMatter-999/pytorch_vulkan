import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[2]


def _execution_mode(value):
    env = os.environ.copy()
    env.pop("PYTORCH_VULKAN_ASYNC_EXECUTION", None)
    if value is not None:
        env["PYTORCH_VULKAN_ASYNC_EXECUTION"] = value
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'build'}"
    result = subprocess.run(
        [sys.executable, "-c", "import pytorch_vulkan; print(pytorch_vulkan._C.execution_mode())"],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_execution_mode_is_selected_before_platform_initialization():
    # Async is the default; sync is opt-in via PYTORCH_VULKAN_ASYNC_EXECUTION=0.
    assert _execution_mode(None) == "async"
    assert _execution_mode("1") == "async"
    assert _execution_mode("0") == "sync"


def test_async_gpu_chain_and_ordered_transfer_boundaries():
    env = os.environ.copy()
    env["PYTORCH_VULKAN_ASYNC_EXECUTION"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'build'}"
    script = r"""
import torch
import pytorch_vulkan
api = pytorch_vulkan._C
assert api.execution_mode() == 'async'
x = torch.ones(64, device='vk')
y = torch.full((64,), 2.0, device='vk')
api.reset_execution_counters()
for _ in range(20):
    x = torch.add(x, y)
assert api.compute_submitted_count() == 1, api.compute_submitted_count()
assert api.compute_wait_count() == 0, api.compute_wait_count()
assert torch.equal(x.cpu(), torch.full((64,), 41.0)), x.cpu()

# Host-side validation failures before recording preserve an earlier prefix.
api.reset_execution_counters()
prefix = torch.add(torch.ones(64, device='vk'), y)
assert api.compute_submitted_count() == 0, api.compute_submitted_count()
bad_rhs = torch.ones(3, device='vk')
try:
    torch.add(prefix, bad_rhs)
except RuntimeError:
    pass
else:
    raise AssertionError('shape mismatch unexpectedly recorded')
assert torch.equal(prefix.cpu(), torch.full((64,), 3.0)), prefix.cpu()
assert api.compute_submitted_count() >= 1, api.compute_submitted_count()

# An upload is a direct host-visible or staging boundary and must follow prior work.
api.reset_execution_counters()
prior = torch.add(torch.ones(64, device='vk'), y)
seed = torch.full((64,), 5.0)
uploaded = seed.to('vk')
z = torch.add(uploaded, y)
assert torch.equal(prior.cpu(), torch.full((64,), 3.0)), prior.cpu()
assert torch.equal(z.cpu(), torch.full((64,), 7.0)), z.cpu()

# Device copy and empty-reduction fill are independently submitted boundaries.
pending_copy = torch.add(uploaded, y)
copied = torch.empty_like(pending_copy)
copied.copy_(pending_copy)
pending_fill = torch.add(uploaded, y)
empty = torch.empty((2, 0), device='vk')
filled = empty.sum(dim=1)
assert torch.equal(copied.cpu(), torch.full((64,), 7.0)), copied.cpu()
assert torch.equal(pending_fill.cpu(), torch.full((64,), 7.0)), pending_fill.cpu()
assert torch.equal(filled.cpu(), torch.zeros(2)), filled.cpu()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_async_host_readback_and_explicit_synchronization_boundaries():
    env = os.environ.copy()
    env["PYTORCH_VULKAN_ASYNC_EXECUTION"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'build'}"
    script = r"""
import gc
import torch
import pytorch_vulkan
api = pytorch_vulkan._C
expected = torch.full((32,), 7.0)

# All public CPU-visible reads must wait for pending producers.
result = torch.add(torch.full((32,), 3.0, device='vk'), 4.0)
assert torch.equal(result.to('cpu'), expected)
assert torch.add(torch.tensor(3.0, device='vk'), 4.0).item() == 7.0
printed = torch.add(torch.full((4,), 5.0, device='vk'), 2.0)
assert '7.' in repr(printed)
print(printed)

# Explicit hook shares stream synchronization, including recording work.
pending = torch.add(torch.full((32,), 2.0, device='vk'), 5.0)
api.synchronize()
assert torch.equal(pending.cpu(), expected)

# Stream query flushes recording but never blocks for its fence.
api.reset_execution_counters()
queried = torch.add(torch.full((32,), 1.0, device='vk'), 6.0)
assert isinstance(api.query(), bool)
assert api.compute_wait_count() == 0, api.compute_wait_count()
assert torch.equal(queried.cpu(), expected)

# Tensor storage may be released by Python before the explicit completion
# boundary; pending source/destination references remain valid until retirement.
source = torch.full((32,), 3.0, device='vk')
destination = torch.add(source, 4.0)
del source, destination
gc.collect()
api.synchronize()
survivor = torch.add(torch.full((32,), 1.0, device='vk'), 6.0)
assert torch.equal(survivor.cpu(), expected)
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_async_training_parity_for_cnn_rnn_attention_and_repeated_lifecycle():
    env = os.environ.copy()
    env["PYTORCH_VULKAN_ASYNC_EXECUTION"] = "1"
    env["PYTHONPATH"] = f"{ROOT}:{ROOT / 'build'}"
    script = r"""
import torch
import pytorch_vulkan
from tools.vulkan_model_benchmark import (
    AttentionFixture, CNNFixture, RNNFixture, _move_model,
)

api = pytorch_vulkan._C
assert api.execution_mode() == 'async'
fixtures = (
    ('cnn', CNNFixture(batch=1)),
    ('rnn', RNNFixture(batch=1, sequence_length=2)),
    ('attention', AttentionFixture(batch=1)),
)
for name, fixture in fixtures:
    inputs, targets = fixture.make_inputs()
    for scoped in (True, False):
        cpu_model = fixture.make_cpu()
        vk_model = _move_model(fixture.make_cpu(), 'vk:0')
        cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01)
        vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.01)
        vk_inputs, vk_targets = inputs.to('privateuseone:0'), targets.to('privateuseone:0')
        api.reset_execution_counters()
        for _ in range(2):
            cpu_optimizer.zero_grad(set_to_none=True)
            cpu_loss = fixture.loss(cpu_model(inputs), targets)
            cpu_loss.backward()
            cpu_optimizer.step()

            started = False
            try:
                if scoped:
                    api.begin_training_step()
                    started = True
                vk_optimizer.zero_grad(set_to_none=True)
                vk_loss = fixture.loss(vk_model(vk_inputs), vk_targets)
                vk_loss.backward()
                vk_optimizer.step()
                if started:
                    api.end_training_step()
                    started = False
                else:
                    api.synchronize()
            except BaseException:
                if started:
                    api.cancel_training_step()
                raise
            assert abs(float(vk_loss.cpu()) - float(cpu_loss)) < 3e-3, (name, scoped)
        api.synchronize()
        assert api.fallback_count() == 0, (name, scoped, api.fallback_count())
        for (parameter_name, cpu_parameter), (_, vk_parameter) in zip(
            cpu_model.named_parameters(), vk_model.named_parameters()
        ):
            delta = (cpu_parameter - vk_parameter.cpu()).abs().max().item()
            assert delta < 3e-3, (name, scoped, parameter_name, delta)
        assert api.compute_completed_count() == api.compute_submitted_count(), (
            name, scoped, api.compute_completed_count(), api.compute_submitted_count()
        )
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
