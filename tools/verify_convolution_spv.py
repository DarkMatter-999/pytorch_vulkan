#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/convolution.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/convolution_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/convolution_spv.sha256"
source_text = SOURCE.read_text(encoding="ascii")
for operation in ("params.operation == 1u", "params.operation == 2u"):
    if operation not in source_text:
        raise SystemExit(f"missing convolution backward operation: {operation}")
if "} else {" not in source_text or "index >= params.output_channels" not in source_text:
    raise SystemExit("missing convolution bias backward operation")
digest = lambda data: hashlib.sha256(data).hexdigest()
expected = dict(line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines())
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "convolution.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()
actual = {"source_sha256": digest(SOURCE.read_bytes()), "spirv_sha256": digest(spirv), "header_sha256": digest(GENERATED.read_bytes())}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
