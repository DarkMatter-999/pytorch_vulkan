#!/usr/bin/env python3
import hashlib
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/linear_relu_backward.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/linear_relu_backward_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/linear_relu_backward_spv.sha256"


def digest(data):
    return hashlib.sha256(data).hexdigest()


source = SOURCE.read_text(encoding="ascii")
required = [
    "binding = 0", "binding = 1", "binding = 2", "binding = 3",
    "binding = 4", "binding = 5", "binding = 6", "binding = 7",
    "dInput", "dWeight", "dBias", "activation_value > 0.0",
    "input_region_offset", "weight_region_offset", "bias_region_offset",
]
for token in required:
    if token not in source:
        raise SystemExit(f"shader contract missing: {token}")
if "masked_gradient" in source or "masked_grad" in source:
    raise SystemExit("shader contract contains an intermediate masked gradient")

expected = dict(line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines())
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "linear_relu_backward.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()

header = GENERATED.read_bytes()
if not re.search(rb"namespace vulkan_backward_shader", header):
    raise SystemExit("generated header has the wrong namespace")
actual = {
    "source_sha256": digest(SOURCE.read_bytes()),
    "spirv_sha256": digest(spirv),
    "header_sha256": digest(header),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
print("linear_relu_backward shader contract=ok")
