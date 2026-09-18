#!/usr/bin/env python3
import hashlib
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/loss.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/loss_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/loss_spv.sha256"
source = SOURCE.read_text(encoding="ascii")
for token in (
    "binding = 0",
    "binding = 1",
    "binding = 2",
    "binding = 3",
    "mse_loss",
    "params.element_count",
    "params.reduction",
    "params.backward",
):
    if token not in source:
        raise SystemExit(f"loss shader contract missing: {token}")
expected = dict(
    line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines()
)
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "loss.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()
header = GENERATED.read_bytes()
if not re.search(rb"namespace vulkan_loss_shader", header):
    raise SystemExit("generated loss header has the wrong namespace")
actual = {
    "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    "spirv_sha256": hashlib.sha256(spirv).hexdigest(),
    "header_sha256": hashlib.sha256(header).hexdigest(),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
print("loss shader contract=ok")
