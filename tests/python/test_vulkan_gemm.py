import pytest
import torch
import json
import subprocess
import sys
from pathlib import Path

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    device = f"{torch._C._get_privateuse1_backend_name()}:0"
    try:
        torch.ones(1).to(device)
    except (NotImplementedError, RuntimeError) as error:
        pytest.skip(f"Vulkan tensor setup is unavailable: {error}")
    return device


def test_mm_and_addmm_use_one_gemm_dispatch(vulkan_backend):
    torch.manual_seed(41)
    a = torch.randn(19, 13)
    b = torch.randn(13, 23)
    vk_a, vk_b = a.to(vulkan_backend), b.to(vulkan_backend)

    pytorch_vulkan._C.reset_execution_counters()
    actual = torch.mm(vk_a, vk_b)
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(actual.cpu(), torch.mm(a, b), rtol=2e-3, atol=2e-3)

    self_cpu = torch.randn(19, 23)
    self_vk = self_cpu.to(vulkan_backend)
    for alpha, beta in ((1.0, 0.0), (0.5, 1.0), (-1.25, 0.25)):
        pytorch_vulkan._C.reset_execution_counters()
        actual = torch.addmm(self_vk, vk_a, vk_b, alpha=alpha, beta=beta)
        assert pytorch_vulkan._C.compute_dispatch_count() == 1
        torch.testing.assert_close(
            actual.cpu(),
            torch.addmm(self_cpu, a, b, alpha=alpha, beta=beta),
            rtol=2e-3,
            atol=2e-3,
        )


def test_addmm_out_uses_gemm_and_preserves_identity(vulkan_backend):
    torch.manual_seed(42)
    self_cpu = torch.randn(7, 11)
    a_cpu = torch.randn(7, 5)
    b_cpu = torch.randn(5, 11)
    self_vk, a_vk, b_vk = (x.to(vulkan_backend) for x in (self_cpu, a_cpu, b_cpu))
    out = torch.empty_like(self_vk)

    pytorch_vulkan._C.reset_execution_counters()
    assert torch.addmm(self_vk, a_vk, b_vk, beta=0.25, alpha=-0.75, out=out) is out
    assert pytorch_vulkan._C.compute_dispatch_count() == 1
    torch.testing.assert_close(
        out.cpu(), torch.addmm(self_cpu, a_cpu, b_cpu, beta=0.25, alpha=-0.75)
    )


def test_mm_rejects_noncontiguous_inputs_and_wrong_dtype(vulkan_backend):
    a = torch.randn(4, 8).to(vulkan_backend)
    b = torch.randn(8, 6).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="contiguous|row-major"):
        torch.mm(a[:, ::2], b[::2])
    with pytest.raises(RuntimeError, match="float32|dtype"):
        torch.mm(a.to(torch.float64), b)


def test_mm_rejects_rank_mismatch_and_wrong_device(vulkan_backend):
    a = torch.randn(4, 8).to(vulkan_backend)
    b = torch.randn(8, 6).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="2-D|rank|matching"):
        torch.mm(a.unsqueeze(0), b)
    with pytest.raises(RuntimeError, match="Vulkan|device|same"):
        torch.mm(a, b.cpu())


def test_addmm_rejects_output_input_aliasing(vulkan_backend):
    a = torch.randn(4, 4).to(vulkan_backend)
    b = torch.randn(4, 4).to(vulkan_backend)
    self_tensor = torch.randn(4, 4).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="alias|overlap|output"):
        torch.addmm(self_tensor, a, b, out=a)


def test_mm_rejects_oversized_dispatch_dimensions(vulkan_backend):
    a = torch.ones(1 << 20, 1, device=vulkan_backend)
    b = torch.ones(1, 1, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="device limits|dispatch|dimension"):
        torch.mm(a, b)


def test_mm_supports_non_tile_aligned_dimensions(vulkan_backend):
    a = torch.randn(3, 5).to(vulkan_backend)
    b = torch.randn(5, 7).to(vulkan_backend)
    torch.testing.assert_close(
        torch.mm(a, b).cpu(), torch.mm(a.cpu(), b.cpu()), rtol=2e-3, atol=2e-3
    )


def test_large_gemm_completes_in_a_clean_process(tmp_path):
    result = tmp_path / "large-gemm.json"
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    completed = subprocess.run(
        [sys.executable, str(script), "--output", str(result)],
        cwd=script.parents[1],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    record = json.loads(result.read_text())
    if record["status"] == "skipped":
        pytest.skip(record["reason"])
    assert record["status"] == "completed"
    assert record["shape"] == [2048, 2048]
    assert record["dispatches"] == 1
    assert record["submissions"] == 1
    assert record["completions"] == 1
    assert record["waits"] == 1
    assert record["fallbacks"] == 0


def test_large_gemm_benchmark_has_no_payload_readback():
    script = Path(__file__).resolve().parents[2] / "tools" / "vulkan_gemm_benchmark.py"
    source = script.read_text()
    assert ".cpu(" not in source
    assert '"submissions"' in source
    assert '"completions"' in source
