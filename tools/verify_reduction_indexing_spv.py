#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
GENERATED = ROOT / "src/vulkan/shaders/generated/reduction_indexing_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/reduction_indexing_spv.sha256"
SHADERS = ["reduction.comp", "indexing.comp", "broadcast.comp"]

def digest(data):
    return hashlib.sha256(data).hexdigest()

expected = {}
for line in MANIFEST.read_text(encoding="ascii").splitlines():
    key, value = line.split("=", 1)
    expected[key] = value
with tempfile.TemporaryDirectory() as directory:
    binaries = []
    for shader in SHADERS:
        output = pathlib.Path(directory) / (shader + ".spv")
        subprocess.run(["glslc", "-Os", "-o", str(output),
                        str(ROOT / "src/vulkan/shaders/glsl" / shader)], check=True)
        binaries.append(output.read_bytes())
source = b"".join((ROOT / "src/vulkan/shaders/glsl" / shader).read_bytes() for shader in SHADERS)
actual = {
    "reduction_source_sha256": digest(source),
    "spirv_sha256": digest(b"".join(binaries)),
    "header_sha256": digest(GENERATED.read_bytes()),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
