import pytest
import torch

import pytorch_vulkan


@pytest.fixture
def vulkan_backend():
    if not pytorch_vulkan.is_available():
        pytest.skip("no suitable Vulkan device is available")
    return "vk"


def test_argmax_matches_cpu_with_keepdim(vulkan_backend):
    cpu_input = torch.tensor([[1.0, 5.0, 2.0], [9.0, 4.0, 3.0]])
    vk_input = cpu_input.to(vulkan_backend)
    result = torch.argmax(vk_input, dim=1, keepdim=True)
    assert result.dtype is torch.int64
    assert result.is_contiguous()
    torch.testing.assert_close(
        result.cpu(), torch.argmax(cpu_input, dim=1, keepdim=True)
    )


def test_argmax_without_dim_flattens_input(vulkan_backend):
    cpu_input = torch.tensor([[1.0, 8.0], [9.0, 3.0]])
    result = torch.argmax(cpu_input.to(vulkan_backend))
    assert result.shape == torch.Size([])
    assert result.cpu().item() == torch.argmax(cpu_input).item()


def test_argmax_without_dim_keepdim_preserves_original_rank(vulkan_backend):
    cpu_input = torch.tensor([[1.0, 8.0], [9.0, 3.0]])
    result = torch.argmax(cpu_input.to(vulkan_backend), keepdim=True)
    assert result.shape == torch.Size([1, 1])
    torch.testing.assert_close(result.cpu(), torch.argmax(cpu_input, keepdim=True))


def test_argmax_nan_and_tie_matches_cpu(vulkan_backend):
    cpu_input = torch.tensor([[float("nan"), 4.0, 4.0], [2.0, 2.0, 1.0]])
    result = torch.argmax(cpu_input.to(vulkan_backend), dim=1)
    torch.testing.assert_close(result.cpu(), torch.argmax(cpu_input, dim=1))


def test_argmax_matches_cpu_for_transpose_slice_and_offset(vulkan_backend):
    cpu_base = torch.tensor(
        [[1.0, 8.0, 2.0, 7.0], [9.0, 3.0, 6.0, 4.0], [5.0, 0.0, 11.0, 10.0]]
    )
    for cpu_input in (
        cpu_base.t(),
        cpu_base[:, 1:],
        cpu_base.as_strided((2, 3), (4, 1), 1),
    ):
        result = torch.argmax(cpu_input.to(vulkan_backend), dim=-1)
        torch.testing.assert_close(result.cpu(), torch.argmax(cpu_input, dim=-1))


def test_argmax_rejects_overlapping_input(vulkan_backend):
    input = (
        torch.arange(6, dtype=torch.float32, device="cpu")
        .reshape(2, 3)
        .to(vulkan_backend)
    )
    overlapping = input.as_strided((2, 2), (1, 1))
    with pytest.raises(RuntimeError, match="overlap|overlapping"):
        torch.argmax(overlapping, dim=1)
    with pytest.raises(RuntimeError, match="overlap|overlapping"):
        torch.argmax(overlapping)


def test_argmax_rejects_bool_and_float16(vulkan_backend):
    with pytest.raises(RuntimeError, match="Vulkan|bool"):
        torch.argmax(torch.ones((2, 3), dtype=torch.bool).to(vulkan_backend), dim=1)
    with pytest.raises(RuntimeError, match="float16"):
        torch.argmax(torch.ones((2, 3), dtype=torch.float16).to(vulkan_backend), dim=1)


def test_argmax_rejects_invalid_dim(vulkan_backend):
    input = torch.ones((2, 3), dtype=torch.float32).to(vulkan_backend)
    with pytest.raises(RuntimeError, match="dim|dimension"):
        torch.argmax(input, dim=2)


def test_argmax_rejects_rank_nine(vulkan_backend):
    input = torch.empty((1,) * 9, dtype=torch.float32, device=vulkan_backend)
    with pytest.raises(RuntimeError, match="rank|ranks"):
        torch.argmax(input, dim=0)


def test_argmax_returns_empty_result_for_empty_non_reduced_dimension(vulkan_backend):
    input = torch.empty((0, 3), dtype=torch.float32, device=vulkan_backend)
    result = torch.argmax(input, dim=1)
    assert result.shape == (0,)
    assert result.numel() == 0


def test_arbitrary_vulkan_int64_allocation_is_rejected(vulkan_backend):
    with pytest.raises(RuntimeError, match="dtype|float32|bool"):
        torch.empty((2,), dtype=torch.int64, device=vulkan_backend)


def test_argmax_out_validates_metadata_before_dispatch(vulkan_backend):
    input = torch.tensor([[1.0, 2.0], [4.0, 3.0]], device=vulkan_backend)
    with pytest.raises(RuntimeError, match="int64|dtype"):
        torch.argmax(
            input,
            dim=1,
            out=torch.empty((2, 1), dtype=torch.float32, device=vulkan_backend),
        )
    with pytest.raises(RuntimeError, match="device"):
        valid = torch.argmax(input, dim=1)
        torch.argmax(input, dim=1, out=torch.empty((2, 1), dtype=torch.int64))
    with pytest.raises(RuntimeError, match="contiguous"):
        valid = torch.argmax(input, dim=1)
        torch.argmax(input, dim=1, out=valid.expand(2, 2)[:, :1])


def test_argmax_out_rejects_wrong_shape_before_dispatch(vulkan_backend):
    input = torch.tensor([[1.0, 2.0], [4.0, 3.0]], device=vulkan_backend)
    out = torch.argmax(input, dim=1, keepdim=True).view(2)
    with pytest.raises(RuntimeError, match="shape|metadata"):
        torch.argmax(input, dim=1, keepdim=True, out=out)
