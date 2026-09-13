#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/formatter_double.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/formatter_double_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/formatter_double_spv.sha256"
expected = dict(line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines())
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "formatter_double.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()
actual = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "spirv_sha256": hashlib.sha256(spirv).hexdigest(),
    "header_sha256": hashlib.sha256(GENERATED.read_bytes()).hexdigest(),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
