#!/usr/bin/env python3
import hashlib
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/convolution.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/convolution_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/convolution_spv.sha256"
source_text = SOURCE.read_text(encoding="ascii")
parameter_names = (
    "batch input_channels input_height input_width output_channels output_height "
    "output_width kernel_height kernel_width operation stride_height stride_width "
    "padding_height padding_width dilation_height dilation_width groups has_bias"
).split()
push_block = re.search(
    r"layout\s*\(\s*push_constant\s*\)\s*uniform\s+Params\s*\{(.*?)\}\s*params\s*;",
    source_text,
    re.S,
)
if push_block is None:
    raise SystemExit("missing convolution push-constant Params block")
source_members = re.findall(r"\buint\s+(\w+)\s*;", push_block.group(1))
if source_members != parameter_names:
    raise SystemExit(
        f"convolution GLSL members/order mismatch: expected {parameter_names}, got {source_members}"
    )
optional_bias_branch = re.search(
    r"if\s*\(\s*params\.has_bias\s*!=\s*0u\s*\)\s*\{([^{}]*)\}",
    source_text,
    re.S,
)
if optional_bias_branch is None:
    raise SystemExit("missing branch guarding optional convolution bias")
bias_load = "bias_values[bm.storage_offset + oc * bm.strides[0]]"
if bias_load not in optional_bias_branch.group(1):
    raise SystemExit("optional convolution bias load is not inside has_bias branch")
if len(re.findall(r"\bbias_values\s*\[(?!\])", source_text)) != 1:
    raise SystemExit("unexpected convolution bias read outside the guarded forward load")
for operation in ("params.operation == 1u", "params.operation == 2u"):
    if operation not in source_text:
        raise SystemExit(f"missing convolution backward operation: {operation}")
if (
    "} else if (params.operation == 3u) {" not in source_text
    or "gl_WorkGroupID.x" not in source_text
    or "gl_LocalInvocationID.x" not in source_text
    or "reduction_values[lane]" not in source_text
    or "barrier();" not in source_text
):
    raise SystemExit("missing parallel convolution bias-gradient reduction")

with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "convolution.comp.spv"
    assembly = pathlib.Path(directory) / "convolution.comp.spvasm"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()
    subprocess.run(["spirv-dis", str(binary), "-o", str(assembly)], check=True)
    disassembly = assembly.read_text(encoding="utf-8")

type_ints = {
    match.group(1): (int(match.group(2)), int(match.group(3)))
    for match in re.finditer(
        r"^\s*(%\S+)\s*=\s*OpTypeInt\s+(\d+)\s+(\d+)\s*$",
        disassembly,
        re.M,
    )
}
type_structs = {
    match.group(1): (match.group(2) or "").split()
    for match in re.finditer(
        r"^\s*(%\S+)\s*=\s*OpTypeStruct(?:\s+(.*))?$", disassembly, re.M
    )
}
pointers = {
    match.group(1): (match.group(2), match.group(3))
    for match in re.finditer(
        r"^\s*(%\S+)\s*=\s*OpTypePointer\s+(\S+)\s+(%\S+)\s*$",
        disassembly,
        re.M,
    )
}
push_variables = re.findall(
    r"^\s*(%\S+)\s*=\s*OpVariable\s+(%\S+)\s+PushConstant\s*$",
    disassembly,
    re.M,
)
if len(push_variables) != 1:
    raise SystemExit(f"expected one PushConstant variable, found {len(push_variables)}")
_, pointer_id = push_variables[0]
if pointer_id not in pointers or pointers[pointer_id][0] != "PushConstant":
    raise SystemExit("PushConstant variable does not use a PushConstant pointer")
push_struct_id = pointers[pointer_id][1]
if push_struct_id not in type_structs:
    raise SystemExit("PushConstant pointer pointee is not an OpTypeStruct")
push_members = type_structs[push_struct_id]
if len(push_members) != len(parameter_names):
    raise SystemExit(
        f"compiled convolution push struct has {len(push_members)} members, expected 18"
    )
member_offsets = {
    int(member_index): int(offset)
    for struct_id, member_index, offset in re.findall(
        r"^\s*OpMemberDecorate\s+(%\S+)\s+(\d+)\s+Offset\s+(\d+)\s*$",
        disassembly,
        re.M,
    )
    if struct_id == push_struct_id
}
expected_offsets = list(range(0, 72, 4))
actual_offsets = [member_offsets.get(index) for index in range(18)]
if actual_offsets != expected_offsets:
    raise SystemExit(
        f"compiled convolution push-constant offsets mismatch: {actual_offsets}"
    )
for index, member_type in enumerate(push_members):
    if type_ints.get(member_type) != (32, 0):
        raise SystemExit(
            f"compiled convolution member {index} is not a 32-bit unsigned integer"
        )
payload_size = max(
    member_offsets[index] + type_ints[push_members[index]][0] // 8
    for index in range(18)
)
if payload_size != 72:
    raise SystemExit(
        f"compiled convolution push-constant payload is {payload_size}, expected 72"
    )
print(f"compiled_push_constant_offsets={actual_offsets}")
print(f"compiled_push_constant_payload_bytes={payload_size}")

digest = lambda data: hashlib.sha256(data).hexdigest()
expected = dict(
    line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines()
)
actual = {
    "source_sha256": digest(SOURCE.read_bytes()),
    "spirv_sha256": digest(spirv),
    "header_sha256": digest(GENERATED.read_bytes()),
}
for key, value in actual.items():
    if expected.get(key) != value:
        raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
    print(f"{key}={value}")
