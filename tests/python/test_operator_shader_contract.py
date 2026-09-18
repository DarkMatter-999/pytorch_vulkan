import pathlib
import re


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
