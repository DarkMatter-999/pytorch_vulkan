#!/usr/bin/env python3
"""Read-only exact-byte and compiled host-ABI verification of operator shaders."""
import argparse
import hashlib
import pathlib
import re
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
NAMES = ("normalization", "classification")
HOST_FIELDS = [("uint32_t", n) for n in ("mode", "batch", "channels", "spatial", "classes")]
HOST_FIELDS += [("int32_t", "ignore_index"), ("uint32_t", "padding0"),
                ("uint32_t", "padding1"), ("float", "momentum"), ("float", "eps")]
ABI = {
    "normalization": (10, [0, 4, 8, 12, 32, 36], ["uint", "uint", "uint", "uint", "float", "float"]),
    "classification": (6, [0, 4, 16, 20], ["uint", "uint", "uint", "int"]),
}


def assert_contracts(root):
    normalization = (root / "src/vulkan/shaders/glsl/normalization.comp").read_text(encoding="ascii")
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

    classification = (root / "src/vulkan/shaders/glsl/classification.comp").read_text(encoding="ascii")
    assert "readonly buffer Labels { int64_t labels[]; }" in classification
    assert "label < 0" in classification
    assert "label >= int64_t(p.classes)" in classification
    assert "p.mode == 4u" in classification and "p.mode == 5u" in classification


def verify_host(root):
    host = (root / "src/vulkan_compute.cpp").read_text()
    block = re.search(r"struct OperatorParams\s*\{([^}]+)\}", host)
    fields = re.findall(r"(\w+)\s+(\w+)\s*;", block.group(1)) if block else []
    if fields != HOST_FIELDS:
        raise ValueError("operator host push ABI must be ten 32-bit members, 40 bytes")
    for name, (count, _, _) in ABI.items():
        call = re.search(r'create_extra\("' + name + r'"\s*,(.*?)\);', host, re.S)
        if not call or not re.search(r"sizeof\(OperatorParams\)\s*,\s*" + str(count) + r"\s*$", call.group(1)):
            raise ValueError(f"{name}: host descriptor/push ABI mismatch")


def verify_binary(data, name):
    if len(data) < 20 or len(data) % 4:
        raise ValueError(f"{name}: malformed SPIR-V length")
    words = struct.unpack(f"<{len(data) // 4}I", data)
    if words[0] != 0x07230203 or words[1] != 0x00010000 or not words[3] or words[4]:
        raise ValueError(f"{name}: malformed/unsupported SPIR-V header")
    index = 5
    while index < len(words):
        count = words[index] >> 16
        if count == 0 or index + count > len(words):
            raise ValueError(f"{name}: malformed SPIR-V instruction")
        index += count


def verify_abi(assembly, name):
    count, wanted_offsets, wanted_types = ABI[name]
    bindings = re.findall(r"OpDecorate (%\S+) Binding (\d+)", assembly)
    sets = dict(re.findall(r"OpDecorate (%\S+) DescriptorSet (\d+)", assembly))
    if sorted(int(b) for _, b in bindings) != list(range(count)) or any(sets.get(v) != "0" for v, _ in bindings):
        raise ValueError(f"{name}: compiled descriptor ABI must be set0 bindings0-{count - 1}")
    if re.findall(r"OpExecutionMode %\S+ LocalSize (\d+) (\d+) (\d+)", assembly) != [("256", "1", "1")]:
        raise ValueError(f"{name}: compiled local size must be 256x1x1")
    variables = re.findall(r"%\S+ = OpVariable (%\S+) PushConstant", assembly)
    if len(variables) != 1:
        raise ValueError(f"{name}: compiled push ABI must have exactly one block")
    pointer = re.search(re.escape(variables[0]) + r" = OpTypePointer PushConstant (%\S+)", assembly)
    if not pointer:
        raise ValueError(f"{name}: malformed compiled push pointer")
    block = pointer.group(1)
    offsets = re.findall(r"OpMemberDecorate " + re.escape(block) + r" (\d+) Offset (\d+)", assembly)
    if offsets != [(str(i), str(offset)) for i, offset in enumerate(wanted_offsets)]:
        raise ValueError(f"{name}: compiled push offsets must be {wanted_offsets}, host range40")
    members = re.search(re.escape(block) + r" = OpTypeStruct ([^\n]+)", assembly)
    types = {}
    for id, width, signed in re.findall(r"(%\S+) = OpTypeInt (\d+) (\d+)", assembly):
        if width == "32":
            types[id] = "int" if signed == "1" else "uint"
    for id, width in re.findall(r"(%\S+) = OpTypeFloat (\d+)", assembly):
        if width == "32":
            types[id] = "float"
    if not members or [types.get(id) for id in members.group(1).split()] != wanted_types:
        raise ValueError(f"{name}: compiled push member types disagree with host32-bit ABI")


def verify(root=ROOT, compiler="glslc"):
    generated = root / "src/vulkan/shaders/generated"
    text = (generated / "operator_spv.sha256").read_text(encoding="ascii")
    entries = [line.split("=", 1) for line in text.splitlines()]
    if (len(entries) != 2 or any(len(entry) != 2 for entry in entries)
            or [entry[0] for entry in entries] != ["source_sha256", "header_sha256"]
            or any(re.fullmatch(r"[0-9a-f]{64}", entry[1]) is None for entry in entries)):
        raise ValueError("operator manifest must contain exactly valid source_sha256/header_sha256")
    expected = dict(entries)
    header = (generated / "operator_spv.h").read_bytes()
    sources = [root / f"src/vulkan/shaders/glsl/{name}.comp" for name in NAMES]
    actual = {"source_sha256": hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest(),
              "header_sha256": hashlib.sha256(header).hexdigest()}
    for key, value in actual.items():
        if expected[key] != value:
            raise ValueError(f"operator {key} mismatch: expected {expected[key]}, got {value}; regenerate explicitly")
    arrays = re.findall(rb"namespace vulkan_(\w+)_shader\s*\{\s*inline constexpr uint32_t (\w+)\[\]\s*=\s*\{([^}]+)\}", header)
    if ([(n.decode(), a.decode()) for n, a, _ in arrays] != [(n, "kCode") for n in NAMES]
            or len(re.findall(rb"uint32_t\s+\w+\[\]\s*=", header)) != 2):
        raise ValueError("operator embedded array inventory must be normalization/classification kCode")
    assert_contracts(root)
    verify_host(root)
    with tempfile.TemporaryDirectory(prefix="operator-verify-") as directory:
        for source, (name, _, body) in zip(sources, arrays):
            name = name.decode()
            if re.fullmatch(rb"\s*0x[0-9a-fA-F]{8}U(?:\s*,\s*0x[0-9a-fA-F]{8}U)*\s*,?\s*", body) is None:
                raise ValueError(f"{name}: malformed embedded SPIR-V array")
            embedded = b"".join(int(w, 16).to_bytes(4, "little") for w in re.findall(rb"0x([0-9a-fA-F]{8})", body))
            verify_binary(embedded, name)
            binary = pathlib.Path(directory) / (name + ".spv")
            try:
                subprocess.run([compiler, "-Os", "-o", str(binary), str(source)], check=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f"{name}: shader compiler failed: {error}") from error
            fresh = binary.read_bytes()
            verify_binary(fresh, name)
            if fresh != embedded:
                raise ValueError(f"{name}: embedded SPIR-V differs from selected compiler; select the matching complete toolchain or regenerate explicitly")
            try:
                subprocess.run(["spirv-val", str(binary)], check=True)
                assembly = subprocess.check_output(["spirv-dis", str(binary)], text=True)
            except (OSError, subprocess.CalledProcessError) as error:
                raise ValueError(f"{name}: SPIR-V validation/reflection failed: {error}") from error
            verify_abi(assembly, name)
            print(f"{name}: embedded=fresh spirv-val=ok set0 bindings0-{ABI[name][0]-1} push_offsets={ABI[name][1]} host_range=40")
    for key, value in actual.items():
        print(key + "=" + value)
    print("operator shader integrity=ok (read-only)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--glslc", default="glslc")
    args = parser.parse_args()
    try:
        verify(args.root, args.glslc)
    except (OSError, ValueError, AssertionError) as error:
        raise SystemExit(f"operator verification: {error}")


if __name__ == "__main__":
    main()
