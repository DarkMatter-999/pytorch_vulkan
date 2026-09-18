#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
NORMALIZATION = ROOT / "src/vulkan/shaders/glsl/normalization.comp"
CLASSIFICATION = ROOT / "src/vulkan/shaders/glsl/classification.comp"


def assert_contracts():
    normalization = NORMALIZATION.read_text(encoding="ascii")
    bindings = re.findall(
        r"layout\(set = 0, binding = (\d+), std430\) (\w+)", normalization
    )
    assert [(int(binding), access) for binding, access in bindings] == [
        (i, "buffer") for i in range(10)
    ]
    assert "writeonly buffer" not in normalization
    assert "d[index] =" in normalization and "e[index] =" in normalization
    assert "float m = f[ch]" in normalization and "float r = g[ch]" in normalization
    assert (
        "uint mode; uint batch; uint channels; uint spatial; uint classes;"
        in normalization
    )
    assert (
        "uint ignore_index; uint padding0; uint padding1; float momentum; float eps;"
        in normalization
    )

    classification = CLASSIFICATION.read_text(encoding="ascii")
    assert "readonly buffer Labels { int64_t labels[]; }" in classification
    assert "label < 0" in classification
    assert "label >= int64_t(p.classes)" in classification
    assert "p.mode == 4u" not in classification


assert_contracts()
subprocess.run(
    [str(ROOT / ".venv/bin/python"), str(ROOT / "tools/generate_operator_spv.py")],
    check=True,
)
manifest = (
    (ROOT / "src/vulkan/shaders/generated/operator_spv.sha256")
    .read_text(encoding="ascii")
    .splitlines()
)
header_hash = hashlib.sha256(
    (ROOT / "src/vulkan/shaders/generated/operator_spv.h").read_bytes()
).hexdigest()
assert manifest[-1] == "header_sha256=" + header_hash
print("operator shader integrity=ok")
