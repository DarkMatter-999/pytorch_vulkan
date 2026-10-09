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
    "params.auxiliary_scalar_offset",
):
    if token not in source:
        raise SystemExit(f"loss shader contract missing: {token}")
expected = dict(
    line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines()
)
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "loss.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    subprocess.run(["spirv-val", str(binary)], check=True)
    assembly = subprocess.check_output(["spirv-dis", str(binary)], text=True)
    spirv = binary.read_bytes()
header = GENERATED.read_bytes()
if not re.search(rb"namespace vulkan_loss_shader", header):
    raise SystemExit("generated loss header has the wrong namespace")
words = re.findall(rb"0x([0-9a-fA-F]{8})", header)
embedded = b"".join(int(word, 16).to_bytes(4, "little") for word in words)
if embedded != spirv:
    raise SystemExit("embedded loss SPIR-V differs from fresh compilation")
bindings = re.findall(r"OpDecorate (%\S+) Binding (\d+)", assembly)
sets = dict(re.findall(r"OpDecorate (%\S+) DescriptorSet (\d+)", assembly))
if sorted(int(n) for _, n in bindings) != [0, 1, 2, 3] or any(sets.get(var) != "0" for var, _ in bindings):
    raise SystemExit("loss descriptor ABI must be set0 bindings0-3")
push_variables = re.findall(r"(%\S+) = OpVariable (%\S+) PushConstant", assembly)
if len(push_variables) != 1:
    raise SystemExit("loss shader must have exactly one push block")
pointer = push_variables[0][1]
block = re.search(re.escape(pointer) + r" = OpTypePointer PushConstant (%\S+)", assembly).group(1)
offsets = re.findall(r"OpMemberDecorate " + re.escape(block) + r" (\d+) Offset (\d+)", assembly)
if offsets != [("0", "0"), ("1", "4"), ("2", "8"), ("3", "12")]:
    raise SystemExit("loss push ABI must have active offsets0/4/8/12")
if not re.search(r"OpExecutionMode %\S+ LocalSize 256 1 1", assembly):
    raise SystemExit("loss local size must be256x1x1")
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
print("loss shader ABI=set0 bindings0-3 push offsets0/4/8/12 range16 embedded=fresh spirv-val=ok")
