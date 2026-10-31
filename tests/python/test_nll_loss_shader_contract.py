"""Contract tests for nll_loss.comp, its generator and its structural verifier.

The verifier proves float32 contraction control (NoContraction on every float
add/sub/mul), the division and atomic budgets, the descriptor/push-constant layout
and the embedded-vs-compiled equality.  These tests mutate the shader to show each hazard is actually rejected.
"""
import pathlib
import re
import shutil
import subprocess

import pytest

from tools import generate_nll_loss_spv as generator
from tools import verify_nll_loss_spv as verifier

ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE = ROOT / verifier.SOURCE_NAME


def _source():
    return SOURCE.read_text(encoding="ascii")


def _compile(text, tmp_path, name="mutant"):
    path = tmp_path / f"{name}.comp"
    path.write_text(text, encoding="ascii")
    binary = tmp_path / f"{name}.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(path)], check=True)
    return binary.read_bytes()


def _replace(text, old, new):
    assert old in text, f"mutation anchor missing: {old!r}"
    return text.replace(old, new, 1)


def _temp_root(tmp_path):
    root = tmp_path / "root"
    (root / "src/vulkan/shaders/glsl").mkdir(parents=True)
    (root / "src/vulkan/shaders/generated").mkdir(parents=True)
    shutil.copy(SOURCE, root / verifier.SOURCE_NAME)
    return root


def test_committed_source_passes_the_source_contract():
    verifier.verify_source(_source())


def test_unmodified_source_compiles_to_a_fully_decorated_module(tmp_path):
    count = verifier.verify_spirv(_compile(_source(), tmp_path, "good"))
    assert count >= verifier.MIN_NO_CONTRACTION_ARITHMETIC


def test_every_precise_qualifier_removed_is_rejected(tmp_path):
    text = re.sub(r"\bprecise\s+", "", _source())
    with pytest.raises(SystemExit, match="without NoContraction"):
        verifier.verify_spirv(_compile(text, tmp_path))
    with pytest.raises(SystemExit, match="precise qualifiers"):
        verifier.verify_source(text)


def test_removing_precise_from_the_reduction_chain_is_rejected(tmp_path):
    text = _source()
    text = _replace(text, "precise float tile_reduce(", "float tile_reduce(")
    text = _replace(text, "precise float acc = 0.0;", "float acc = 0.0;")
    text = _replace(text, "precise float value = 0.0;", "float value = 0.0;")
    text = _replace(text, "precise float result = partial[0];", "float result = partial[0];")
    with pytest.raises(SystemExit, match="without NoContraction"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_removing_precise_from_the_term_product_is_rejected(tmp_path):
    text = _replace(_source(), "precise float term = params.has_weight", "float term = params.has_weight")
    text = _replace(text, "precise float row_value(", "float row_value(")
    text = _replace(text, "precise float w = params.has_weight != 0u ? weights[params.weight_offset + cls] : 1.0;",
                    "float w = params.has_weight != 0u ? weights[params.weight_offset + cls] : 1.0;")
    with pytest.raises(SystemExit, match="without NoContraction"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_fused_multiply_add_is_rejected(tmp_path):
    text = _replace(_source(), "(-x) * w : -x;", "fma(-x, w, 0.0) : -x;")
    with pytest.raises(SystemExit, match="Fma"):
        verifier.verify_spirv(_compile(text, tmp_path))
    with pytest.raises(SystemExit, match="forbidden construct"):
        verifier.verify_source(text)


def test_an_extra_division_is_rejected(tmp_path):
    text = _replace(_source(), "(-x) * w : -x;", "(-x) / w : -x;")
    with pytest.raises(SystemExit, match="exactly 2 OpFDiv"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_a_missing_division_is_rejected(tmp_path):
    text = _replace(_source(), "out0[0] = divide(loss, weight_sum);", "out0[0] = loss;")
    with pytest.raises(SystemExit, match="exactly 2 OpFDiv"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_mix_is_rejected(tmp_path):
    text = _replace(_source(), "(-x) * w : -x;", "mix(-x, x, w) : -x;")
    with pytest.raises(SystemExit, match="FMix"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_a_second_atomic_is_rejected(tmp_path):
    text = _replace(_source(), "if (gid < 4u) status[gid] = 0u;", "if (gid < 4u) atomicAdd(status[gid], 1u);")
    with pytest.raises(SystemExit, match="exactly one OpAtomicUMax"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_the_status_atomic_must_be_umax(tmp_path):
    text = _replace(_source(), "atomicMax(status[0], 0xFFFFFFFFu - gid);", "atomicMin(status[0], 0xFFFFFFFFu - gid);")
    with pytest.raises(SystemExit, match="exactly one OpAtomicUMax"):
        verifier.verify_spirv(_compile(text, tmp_path))


def _append_decoration(data, *words):
    import struct

    body = struct.pack("<%dI" % (len(words) + 1), ((len(words) + 1) << 16) | 71, *words)
    return data + body


def test_relaxed_precision_and_fast_math_decorations_are_rejected(tmp_path):
    good = _compile(_source(), tmp_path, "good")
    target = next(
        ops[0] for opcode, ops in verifier._instructions(good) if opcode == 133
    )
    verifier.verify_spirv(good)
    with pytest.raises(SystemExit, match="RelaxedPrecision"):
        verifier.verify_spirv(_append_decoration(good, target, verifier.DECORATION_RELAXED_PRECISION))
    with pytest.raises(SystemExit, match="RelaxedPrecision or FPFastMathMode"):
        verifier.verify_spirv(_append_decoration(good, target, verifier.DECORATION_FP_FAST_MATH_MODE, 0x10))


def test_stripped_no_contraction_decoration_is_rejected(tmp_path):
    import struct

    good = _compile(_source(), tmp_path, "good")
    words = list(struct.unpack("<%dI" % (len(good) // 4), good))
    multiplies = {ops[1] for opcode, ops in verifier._instructions(good) if opcode == 133}
    index, stripped = 5, False
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xFFFF
        if (
            opcode == 71
            and words[index + 2] == verifier.DECORATION_NO_CONTRACTION
            and words[index + 1] in multiplies
            and not stripped
        ):
            for offset in range(count):
                words[index + offset] = 1 << 16  # one-word OpNop
            stripped = True
        index += count
    assert stripped
    with pytest.raises(SystemExit, match="without NoContraction"):
        verifier.verify_spirv(struct.pack("<%dI" % len(words), *words))


def test_changed_binding_is_rejected(tmp_path):
    text = _replace(_source(), "binding = 3, std430", "binding = 9, std430")
    with pytest.raises(SystemExit, match="bindings must be"):
        verifier.verify_spirv(_compile(text, tmp_path))
    with pytest.raises(SystemExit, match="missing binding 3"):
        verifier.verify_source(text)


def test_output_buffer_must_stay_writeonly(tmp_path):
    text = _replace(_source(), "writeonly buffer Out0", "buffer Out0")
    with pytest.raises(SystemExit, match="writeonly"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_input_buffer_must_stay_readonly(tmp_path):
    text = _replace(_source(), "readonly buffer Weight", "buffer Weight")
    with pytest.raises(SystemExit, match="readonly"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_status_buffer_must_stay_read_write(tmp_path):
    text = _replace(_source(), "layout(set = 0, binding = 7, std430) buffer Status", "layout(set = 0, binding = 7, std430) readonly buffer Status")
    text = _replace(text, "atomicMax(status[0], 0xFFFFFFFFu - gid);", "if (status[0] == 7u) return;")
    text = _replace(text, "if (gid < 4u) status[gid] = 0u;", "if (gid < 4u) out0[gid] = 0.0;")
    text = _replace(text, "      status[2] = uint(label);\n      status[3] = uint(label >> 32);\n", "      out0[2] = float(uint(label));\n")
    with pytest.raises(SystemExit):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_push_constant_layout_change_is_rejected(tmp_path):
    text = _replace(_source(), "uint total_offset; uint count;\n}", "uint total_offset; uint count; uint extra;\n}")
    text = _replace(text, "if (gid >= params.count) return;", "if (gid >= params.count || params.extra == 0xffffffffu) return;")
    with pytest.raises(SystemExit, match="13 members"):
        verifier.verify_spirv(_compile(text, tmp_path))
    with pytest.raises(SystemExit, match="push-constant fields"):
        verifier.verify_source(text)


def test_push_constant_member_type_change_is_rejected(tmp_path):
    text = _replace(_source(), "uint ignore_lo; uint ignore_hi;", "float ignore_lo; uint ignore_hi;")
    text = _replace(text, "uint64_t(params.ignore_lo)", "uint64_t(floatBitsToUint(params.ignore_lo))")
    with pytest.raises(SystemExit, match="wrong type"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_workgroup_size_change_is_rejected(tmp_path):
    text = _replace(_source(), "local_size_x = 256", "local_size_x = 128")
    with pytest.raises(SystemExit, match="256x1x1"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_float64_is_rejected(tmp_path):
    text = _replace(
        _source(),
        "#version 450\n",
        "#version 450\n#extension GL_EXT_shader_explicit_arithmetic_types_float64 : require\n",
    )
    text = _replace(text, "float divide(float a, float b) {", "float divide(float a, float b) {\n  double wide = double(a);\n  if (wide != wide) return a;")
    with pytest.raises(SystemExit, match="capabilities|non-32-bit"):
        verifier.verify_spirv(_compile(text, tmp_path))


def test_generator_and_verifier_round_trip(tmp_path, capsys):
    root = _temp_root(tmp_path)
    generator.generate(root)
    verifier.verify(root)
    output = capsys.readouterr().out
    assert "no_contraction_arithmetic=" in output and "header_sha256=" in output
    header = (root / verifier.HEADER_NAME).read_text(encoding="ascii")
    assert "vulkan_nll_loss_shader" in header and "kCodeSize" in header


def test_verifier_rejects_a_modified_header_word(tmp_path):
    root = _temp_root(tmp_path)
    generator.generate(root)
    header = root / verifier.HEADER_NAME
    text = header.read_text(encoding="ascii")
    match = re.search(r"0x[0-9a-f]{8}U", text[text.index("kCode") + 200:])
    tampered = text.replace(match.group(0), "0xdeadbeefU", 1) if match else text
    header.write_text(tampered, encoding="ascii")
    with pytest.raises(SystemExit):
        verifier.verify(root)


def test_verifier_rejects_a_stale_manifest(tmp_path):
    root = _temp_root(tmp_path)
    generator.generate(root)
    manifest = root / verifier.MANIFEST_NAME
    lines = manifest.read_text(encoding="ascii").splitlines()
    lines[0] = "source_sha256=" + "0" * 64
    manifest.write_text("\n".join(lines) + "\n", encoding="ascii")
    with pytest.raises(SystemExit, match="source_sha256 mismatch"):
        verifier.verify(root)


def test_verifier_rejects_source_edited_after_generation(tmp_path):
    root = _temp_root(tmp_path)
    generator.generate(root)
    source = root / verifier.SOURCE_NAME
    source.write_text(source.read_text(encoding="ascii") + "\n// edited\n", encoding="ascii")
    with pytest.raises(SystemExit, match="mismatch"):
        verifier.verify(root)


def test_missing_products_ask_for_the_generator(tmp_path):
    root = _temp_root(tmp_path)
    with pytest.raises(SystemExit, match="generate_nll_loss_spv"):
        verifier.verify(root)


def test_generator_refuses_to_publish_a_non_compliant_shader(tmp_path):
    root = _temp_root(tmp_path)
    source = root / verifier.SOURCE_NAME
    source.write_text(re.sub(r"\bprecise\s+", "", source.read_text(encoding="ascii")), encoding="ascii")
    with pytest.raises(SystemExit):
        generator.generate(root)
    assert not (root / verifier.HEADER_NAME).exists()
    assert not (root / verifier.MANIFEST_NAME).exists()


def test_committed_product_is_structurally_compliant():
    header = (ROOT / verifier.HEADER_NAME).read_text(encoding="ascii")
    count = verifier.verify_spirv(verifier.decode_embedded_spirv(header))
    assert count >= verifier.MIN_NO_CONTRACTION_ARITHMETIC
