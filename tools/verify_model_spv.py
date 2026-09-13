#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/model.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/model_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/model_spv.sha256"

def digest(data):
    return hashlib.sha256(data).hexdigest()

expected = dict(line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines())
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "model.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()
actual = {"source_sha256": digest(SOURCE.read_bytes()), "spirv_sha256": digest(spirv),
          "header_sha256": digest(GENERATED.read_bytes())}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
