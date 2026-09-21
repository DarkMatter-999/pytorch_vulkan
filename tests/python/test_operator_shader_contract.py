import pathlib
import re
import subprocess
import tempfile

import pytest

from tools.verify_gemm_spv import decode_embedded_spirv, verify_source, verify_spirv


ROOT = pathlib.Path(__file__).resolve().parents[2]


def _source(name):
    return (ROOT / "src/vulkan/shaders/glsl" / name).read_text(encoding="ascii")


def test_normalization_shader_contract_keeps_read_write_saved_statistics():
    source = _source("normalization.comp")
    for binding in range(10):
        assert re.search(
            rf"layout\(set = 0, binding = {binding}, std430\) buffer ", source
        )
    assert "writeonly buffer" not in source
    assert "d[index] =" in source and "e[index] =" in source
    assert "float m = f[ch]" in source and "float r = g[ch]" in source
    assert "uint mode; uint batch; uint channels; uint spatial; uint classes;" in source
    assert (
        "uint ignore_index; uint padding0; uint padding1; float momentum; float eps;"
        in source
    )


def test_classification_shader_contract_keeps_int64_labels_and_bounds_guards():
    source = _source("classification.comp")
    assert "readonly buffer Labels { int64_t labels[]; }" in source
    assert "label < 0" in source
    assert "label >= int64_t(p.classes)" in source
    assert "p.mode == 4u" not in source


def test_gemm_shader_contract_freezes_bindings_and_push_constant_offsets():
    source = _source("gemm.comp")
    expected_bindings = [
        (0, "readonly", "A"),
        (1, "readonly", "B"),
        (2, "readonly", "C"),
        (3, "writeonly", "D"),
        (4, "readonly", "Bias"),
    ]
    for binding, access, block in expected_bindings:
        assert re.search(
            rf"layout\(set = 0, binding = {binding}, std430\) {access} buffer {block} ",
            source,
        )

    expected_members = {
        "uint M": 0,
        "uint N": 4,
        "uint K": 8,
        "uint a_row_stride": 12,
        "uint a_col_stride": 16,
        "uint b_row_stride": 20,
        "uint b_col_stride": 24,
        "uint c_row_stride": 28,
        "uint c_col_stride": 32,
        "uint d_row_stride": 36,
        "uint d_col_stride": 40,
        "uint bias_stride": 44,
        "float alpha": 48,
        "float beta": 52,
        "uint has_bias": 56,
        "uint batch_count": 60,
        "uint batch_stride_a": 64,
        "uint batch_stride_b": 68,
        "uint batch_stride_c": 72,
        "uint batch_stride_d": 76,
    }
    for declaration, offset in expected_members.items():
        assert f"/* offset {offset} */ {declaration};" in source


def test_gemm_shader_contract_guards_tiles_and_epilogue():
    source = _source("gemm.comp")
    assert "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1)" in source
    assert "row >= params.M" in source or "row < params.M" in source
    assert "col >= params.N" in source or "col < params.N" in source
    assert "k_index >= params.K" in source
    assert "barrier();" in source
    assert "params.alpha" in source
    assert "params.beta" in source
    assert "params.has_bias" in source
    assert "if (row < params.M && col < params.N)" in source
    assert "d.values[" in source


def test_gemm_verifier_source_contract_accepts_batched_strides():
    source = _source("gemm.comp")
    verify_source(source)
    assert "params.reserved" not in source


def test_gemm_verifier_checks_compiled_spirv_descriptor_contract():
    source_path = ROOT / "src/vulkan/shaders/glsl/gemm.comp"
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "gemm.comp.spv"
        disassembly = pathlib.Path(directory) / "gemm.comp.spvasm"
        subprocess.run(["glslc", "-o", str(binary), str(source_path)], check=True)
        subprocess.run(["spirv-dis", str(binary), "-o", str(disassembly)], check=True)
        text = disassembly.read_text(encoding="ascii")

    verify_spirv(text)
    with pytest.raises(SystemExit, match="descriptor set"):
        verify_spirv(text.replace("DescriptorSet 0", "DescriptorSet 1", 1))
    with pytest.raises(SystemExit, match="storage-buffer descriptor"):
        verify_spirv(text.replace("BufferBlock", "Block", 1))


def test_gemm_verifier_decodes_embedded_spirv_words_exactly():
    header = (ROOT / "src/vulkan/shaders/generated/gemm_spv.h").read_text(
        encoding="ascii"
    )
    embedded = decode_embedded_spirv(header)
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "gemm.comp.spv"
        subprocess.run(
            [
                "glslc",
                "-Os",
                "-o",
                str(binary),
                str(ROOT / "src/vulkan/shaders/glsl/gemm.comp"),
            ],
            check=True,
        )
        assert embedded == binary.read_bytes()
        subprocess.run(["spirv-val", str(binary)], check=True)
