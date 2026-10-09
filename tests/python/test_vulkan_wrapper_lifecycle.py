"""Fresh-process entry-point lifecycle qualification; positive cases never skip."""

import os

from wrapper_subprocess import run_script


TRANSFER_SCRIPT = r'''
import pytorch_vulkan as torch
import json
from pathlib import Path
from composed_evidence_common import phases, history_nodes, combine_phases, validate_phases, validate_counters

RESIDENT = False

assert torch.autograd.is_multithreading_enabled()
assert torch.is_available(), 'selected acceptance device vk:0 is unavailable'
print('wrapper_file=' + str(Path(torch.__file__).resolve()), flush=True)
print('default_threading=True', flush=True)

def cpu_view(seed):
    base = torch.randn((2, 3), generator=torch.Generator().manual_seed(seed), requires_grad=True)
    target = torch.randn((2, 3), generator=torch.Generator().manual_seed(seed + 1))
    probe = torch.randn((2, 3), generator=torch.Generator().manual_seed(seed + 2))
    loss = torch.nn.functional.mse_loss(base.t(), target.t())
    g, = torch.autograd.grad(loss, base, create_graph=True)
    h, = torch.autograd.grad(g, base, probe)
    torch.testing.assert_close(g, (base - target) * (2 / base.numel()))
    torch.testing.assert_close(h, probe * (2 / base.numel()))
    assert g.requires_grad and history_nodes(g).count('TBackward0') == 2
    assert 'MseLossBackwardBackward0' in history_nodes(g)
    print('cpu_view=' + json.dumps({'seed': seed, 'first': g.tolist(), 'second': h.tolist(),
                                  'nodes': history_nodes(g)}), flush=True)

def derivatives(device, seed):
    global last_routes
    base = torch.randn(3, generator=torch.Generator().manual_seed(seed))
    target = torch.randn(3, generator=torch.Generator().manual_seed(seed + 1))
    probe = torch.randn(3, generator=torch.Generator().manual_seed(seed + 2))
    if RESIDENT and device != 'cpu':
        base = base.to(device).requires_grad_()
        probe = probe.to(device)
    else:
        base.requires_grad_()
    x = base if device == 'cpu' or RESIDENT else base.to(device)
    target = target.to(device)
    routes, counters = {}, {}
    phase = phases(routes)
    def start():
        if device != 'cpu':
            torch._C.synchronize()
            torch._C.reset_execution_counters()
    def finish(name):
        if device != 'cpu':
            torch._C.synchronize()
            counters[name] = list(torch._C.execution_counter_snapshot())
    start()
    with phase('forward'):
        loss = torch.nn.functional.mse_loss(x, target)
    finish('forward')
    start()
    with phase('first'):
        g, = torch.autograd.grad(loss, base, create_graph=True)
    finish('first')
    assert g.requires_grad and g.grad_fn is not None
    start()
    with phase('second'):
        h, = torch.autograd.grad(g, base, probe)
    finish('second')
    expected_device = device if RESIDENT else 'cpu'
    assert str(base.device) == str(g.device) == str(h.device) == expected_device
    assert 'MseLossBackwardBackward0' in history_nodes(g)
    if device != 'cpu':
        assert history_nodes(g).count('ToCopyBackward0') == (0 if RESIDENT else 2)
    if device != 'cpu':
        for name, count in counters.items():
            if RESIDENT:
                validate_counters(count, minimum_copies=0)
            else:
                # The normal CopyBackward route returns to the CPU original base:
                # one transfer in first, CPU probe + result transfer in second.
                assert count[0] > 0 and count[1:] == [0, {'forward': 0, 'first': 1, 'second': 2}[name], 0]
    record = {'seed': seed, 'device': device, 'base_device': str(base.device),
              'gradient_devices': [str(g.device), str(h.device)],
              'fixture': {'base': base.detach().cpu().tolist(), 'target': target.cpu().tolist(),
                          'first_seed': 1.0, 'second_seed': probe.cpu().tolist()},
              'first': g.detach().cpu().tolist(),
              'second': h.detach().cpu().tolist(), 'nodes': history_nodes(g),
              'routes': routes, 'counters': counters}
    print('derivatives=' + json.dumps(record), flush=True)
    last_routes = routes
    return g.detach().cpu(), h.detach().cpu()

def mutation(device, before_second):
    base = torch.tensor([.2, -.4, .7]).to(device).requires_grad_()
    target = torch.tensor([.1, .3, -.2]).to(device)
    loss = torch.nn.functional.mse_loss(base, target)
    if before_second:
        g, = torch.autograd.grad(loss, base, create_graph=True)
    version = base._version
    with torch.no_grad():
        base.add_(.125)
    assert base._version == version + 1
    try:
        if before_second:
            torch.autograd.grad(g, base, torch.tensor([.3, -.2, .5]).to(device))
        else:
            torch.autograd.grad(loss, base, create_graph=True)
    except RuntimeError as error:
        assert 'modified by an inplace operation' in str(error)
        print('mutation=' + json.dumps({'device': device, 'before_second': before_second,
                                      'version': base._version, 'error': str(error)}), flush=True)
    else:
        raise AssertionError('saved-variable mutation was accepted')
    if device != 'cpu':
        torch._C.synchronize()

for seed in (101, 307):
    cpu_view(seed)
    cpu = derivatives('cpu', seed)
    cpu_routes = last_routes
    vk = derivatives('vk:0', seed)
    validate_phases(combine_phases(cpu_routes, last_routes), cpu_routes)
    cpu_again = derivatives('cpu', seed)
    for expected, actual, repeated in zip(cpu, vk, cpu_again):
        torch.testing.assert_close(actual, expected, rtol=.003, atol=.003)
        torch.testing.assert_close(repeated, expected)
        print('max_error=' + str(float((actual - expected).abs().max())), flush=True)
for device in ('cpu', 'vk:0'):
    for before_second in (False, True):
        mutation(device, before_second)
assert torch.autograd.is_multithreading_enabled()
print('lifecycle_complete=True', flush=True)
'''


def test_wrapper_first_cpu_base_transfer_lifecycle():
    result = run_script(TRANSFER_SCRIPT, dict(os.environ))
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'lifecycle_complete=True' in result.stdout


def test_wrapper_first_vulkan_base_resident_lifecycle():
    result = run_script(TRANSFER_SCRIPT.replace('RESIDENT = False', 'RESIDENT = True'), dict(os.environ))
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'lifecycle_complete=True' in result.stdout


def test_upstream_import_without_autograd_then_wrapper():
    result = run_script('import torch\n' + TRANSFER_SCRIPT, dict(os.environ))
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'lifecycle_complete=True' in result.stdout


LATE_IMPORT_SCRIPT = r'''
import torch
base = torch.tensor([.2, -.4, .7], requires_grad=True)
torch.nn.functional.mse_loss(base, torch.tensor([.1, .3, -.2])).backward()
print('cpu_backward_before_wrapper=True', flush=True)
import pytorch_vulkan as torch
assert torch.autograd.is_multithreading_enabled()
assert torch.is_available(), 'selected acceptance device vk:0 is unavailable'
base = torch.tensor([.2, -.4, .7]).to('vk:0').requires_grad_()
loss = torch.nn.functional.mse_loss(base, torch.tensor([.1, .3, -.2]).to('vk:0'))
print('about_to_execute_late_import_vulkan_backward=True', flush=True)
torch.autograd.grad(loss, base, create_graph=True)
torch._C.synchronize()
'''


def test_documented_unsupported_late_import_retains_engine_failure():
    """Diagnostic contract only: this does NOT qualify late import as supported."""
    result = run_script(LATE_IMPORT_SCRIPT, dict(os.environ))
    print('unsupported_late_import_exit=' + str(result.returncode))
    print(result.stdout)
    print(result.stderr)
    assert result.returncode == 1, result.stdout + result.stderr
    assert 'about_to_execute_late_import_vulkan_backward=True' in result.stdout
    assert 'RuntimeError: 0 <= device.index()' in result.stderr
    assert 'device_ready_queues_.size()' in result.stderr
    assert 'INTERNAL ASSERT FAILED' in result.stderr
    assert 'torch/csrc/autograd/engine.cpp":1448' in result.stderr
