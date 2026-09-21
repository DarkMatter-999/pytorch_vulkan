#!/usr/bin/env python3
import hashlib
import pathlib
import re
import struct
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/gemm.comp"
GENERATED = ROOT / "src/vulkan/shaders/generated/gemm_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/gemm_spv.sha256"


def fail(message):
    raise SystemExit(f"gemm shader contract missing: {message}")


def verify_source(source):
    bindings = re.findall(
        r"layout\(set = 0, binding = (\d+), std430\) (readonly|writeonly) buffer (\w+)",
        source,
    )
    if bindings != [
        ("0", "readonly", "A"),
        ("1", "readonly", "B"),
        ("2", "readonly", "C"),
        ("3", "writeonly", "D"),
        ("4", "readonly", "Bias"),
    ]:
        fail(f"descriptor ABI {bindings!r}")
    if "layout(local_size_x = 16, local_size_y = 16, local_size_z = 1)" not in source:
        fail("16x16x1 workgroup")
    for token in (
        "row < params.M",
        "col < params.N",
        "a_col < params.K",
        "b_row < params.K",
        "k_index >= params.K",
    ):
        if token not in source:
            fail(f"bounds guard {token}")
    if source.count("barrier();") < 2:
        fail("shared-memory barriers")
    for token in ("params.alpha", "params.beta", "params.has_bias"):
        if token not in source:
            fail(f"epilogue branch {token}")
    if (
        "if (row < params.M && col < params.N)" not in source
        or "d.values[" not in source
    ):
        fail("guarded output store")
    expected_members = {
        "M": 0,
        "N": 4,
        "K": 8,
        "a_row_stride": 12,
        "a_col_stride": 16,
        "b_row_stride": 20,
        "b_col_stride": 24,
        "c_row_stride": 28,
        "c_col_stride": 32,
        "d_row_stride": 36,
        "d_col_stride": 40,
        "bias_stride": 44,
        "alpha": 48,
        "beta": 52,
        "has_bias": 56,
        "batch_count": 60,
        "batch_stride_a": 64,
        "batch_stride_b": 68,
        "batch_stride_c": 72,
        "batch_stride_d": 76,
    }
    for name, offset in expected_members.items():
        if f"/* offset {offset} */" not in source or f" {name};" not in source:
            fail(f"push constant {name} at offset {offset}")


def verify_spirv(disassembly):
    descriptor_ids = {}
    descriptor_sets = {}
    access = {}
    for identifier, binding in re.findall(
        r"OpDecorate (%\w+) Binding (\d+)", disassembly
    ):
        descriptor_ids.setdefault(int(binding), []).append(identifier)
    for identifier, descriptor_set in re.findall(
        r"OpDecorate (%\w+) DescriptorSet (\d+)", disassembly
    ):
        descriptor_sets[identifier] = int(descriptor_set)
    for identifier, qualifier in re.findall(
        r"OpDecorate (%\w+) (NonWritable|NonReadable)", disassembly
    ):
        access[identifier] = qualifier
    if set(descriptor_ids) != set(range(5)) or any(
        len(identifiers) != 1 for identifiers in descriptor_ids.values()
    ):
        fail(f"SPIR-V fixed bindings {sorted(descriptor_ids)}")
    descriptor_ids = {
        binding: identifiers[0] for binding, identifiers in descriptor_ids.items()
    }
    pointer_types = {
        identifier: (storage_class, struct)
        for identifier, storage_class, struct in re.findall(
            r"(%\w+) = OpTypePointer (\w+) (%\w+)", disassembly
        )
    }
    variables = {
        identifier: (pointer, storage_class)
        for identifier, pointer, storage_class in re.findall(
            r"(%\w+) = OpVariable (%\w+) (\w+)", disassembly
        )
    }
    storage_buffer_structs = set(
        re.findall(r"OpDecorate (%\w+) BufferBlock", disassembly)
    )
    for binding, identifier in descriptor_ids.items():
        if descriptor_sets.get(identifier) != 0:
            fail(f"descriptor set for binding {binding}")
        pointer, storage_class = variables.get(identifier, (None, None))
        if storage_class != "Uniform" or pointer not in pointer_types:
            fail(f"storage-buffer descriptor for binding {binding}")
        pointer_storage, struct = pointer_types[pointer]
        if pointer_storage != "Uniform" or struct not in storage_buffer_structs:
            fail(f"storage-buffer descriptor for binding {binding}")
    expected_access = {
        0: "NonWritable",
        1: "NonWritable",
        2: "NonWritable",
        3: "NonReadable",
        4: "NonWritable",
    }
    for binding, qualifier in expected_access.items():
        if access.get(descriptor_ids[binding]) != qualifier:
            fail(f"SPIR-V binding {binding} access")

    push_struct = re.search(
        r"OpTypePointer PushConstant (%\w+)\n\s+%\w+ = OpVariable %\w+ PushConstant",
        disassembly,
    )
    if not push_struct:
        fail("push-constant block")
    struct_id = push_struct.group(1)
    offsets = [
        int(offset)
        for member, offset in re.findall(
            rf"OpMemberDecorate {re.escape(struct_id)} (\d+) Offset (\d+)",
            disassembly,
        )
    ]
    if offsets != list(range(0, 80, 4)):
        fail(f"push-constant offsets {offsets}")


def decode_embedded_spirv(header):
    match = re.search(r"kCode\[\]\s*=\s*\{(?P<words>.*?)\};", header, re.DOTALL)
    if not match:
        fail("embedded kCode array")
    words = [
        int(word, 16) for word in re.findall(r"0x([0-9a-fA-F]+)U", match.group("words"))
    ]
    if not words:
        fail("embedded kCode is empty")
    return b"".join(struct.pack("<I", word) for word in words)


def main():
    source_bytes = SOURCE.read_bytes()
    verify_source(source_bytes.decode("ascii"))
    if not GENERATED.exists() or not MANIFEST.exists():
        raise SystemExit(
            "gemm shader artifact missing: run tools/generate_gemm_spv.py in Task 2"
        )
    header = GENERATED.read_bytes()
    embedded = decode_embedded_spirv(header.decode("ascii"))
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "gemm.comp.spv"
        contract_binary = pathlib.Path(directory) / "gemm.contract.spv"
        disassembly = pathlib.Path(directory) / "gemm.comp.spvasm"
        subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
        subprocess.run(["glslc", "-o", str(contract_binary), str(SOURCE)], check=True)
        subprocess.run(
            ["spirv-dis", str(contract_binary), "-o", str(disassembly)], check=True
        )
        verify_spirv(disassembly.read_text(encoding="ascii"))
        spirv = binary.read_bytes()
        embedded_binary = pathlib.Path(directory) / "gemm.embedded.spv"
        embedded_binary.write_bytes(embedded)
        subprocess.run(["spirv-val", str(embedded_binary)], check=True)
    manifest = dict(
        line.split("=", 1) for line in MANIFEST.read_text(encoding="ascii").splitlines()
    )
    if embedded != spirv:
        fail("embedded SPIR-V bytes differ from freshly compiled SPIR-V")
    actual = {
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "spirv_sha256": hashlib.sha256(spirv).hexdigest(),
        "embedded_spirv_sha256": hashlib.sha256(embedded).hexdigest(),
        "header_sha256": hashlib.sha256(header).hexdigest(),
    }
    for key, value in actual.items():
        if key == "embedded_spirv_sha256":
            expected = manifest.get("spirv_sha256")
        else:
            expected = manifest.get(key)
        if expected != value:
            raise SystemExit(f"{key} mismatch: expected {expected}, got {value}")
        print(f"{key}={value}")
    print("gemm shader contract=ok")


if __name__ == "__main__":
    main()
