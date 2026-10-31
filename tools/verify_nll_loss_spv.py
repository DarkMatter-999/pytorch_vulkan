#!/usr/bin/env python3
"""Verify the nll_loss shader source, its SPIR-V and the generated header.

The float contract of nll_loss.comp is that every add, subtract and multiply is separately
rounded and happens in a fixed order: FMA contraction would change float32 bits that the
numpy oracle (tests/python/nll_loss_reference.py) reproduces exactly.  This verifier proves,
on both the freshly compiled SPIR-V and the array embedded in nll_loss_spv.h, that

* every OpFAdd/OpFSub/OpFMul result carries NoContraction (and enough of them exist
  that the check cannot pass vacuously);
* the only float division is the explicit-zero-guarded `divide` helper: exactly two OpFDiv
  (reduce-mean and backward-mean); no OpFRem/OpFMod, no vector/matrix arithmetic, no
  GLSL.std.450 instruction at all, and no RelaxedPrecision or FPFastMathMode decoration;
* the only atomic is one OpAtomicUMax (the integer error-status word); atomics never touch
  float data;
* only the Shader and Int64 capabilities are used, floats are 32-bit and integers 32/64-bit;
* the descriptor bindings (0-4 readonly, 5-6 writeonly, 7 read-write, all in set 0), the
  52-byte push-constant layout of 13 uint members and the 256-wide workgroup match the host
  contract.
"""
import argparse
import hashlib
import pathlib
import re
import shutil
import struct
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE_NAME = "src/vulkan/shaders/glsl/nll_loss.comp"
HEADER_NAME = "src/vulkan/shaders/generated/nll_loss_spv.h"
MANIFEST_NAME = "src/vulkan/shaders/generated/nll_loss_spv.sha256"

MAGIC = 0x07230203
OP_EXT_INST_IMPORT, OP_EXT_INST = 11, 12
OP_EXECUTION_MODE, OP_CAPABILITY = 16, 17
OP_TYPE_INT, OP_TYPE_FLOAT = 21, 22
OP_TYPE_STRUCT, OP_TYPE_POINTER = 30, 32
OP_VARIABLE = 59
OP_DECORATE, OP_MEMBER_DECORATE = 71, 72
ARITHMETIC = {129: "OpFAdd", 131: "OpFSub", 133: "OpFMul"}
FORBIDDEN_OPS = {
    140: "OpFRem", 141: "OpFMod", 142: "OpVectorTimesScalar",
    143: "OpMatrixTimesScalar", 144: "OpVectorTimesMatrix", 145: "OpMatrixTimesVector",
    146: "OpMatrixTimesMatrix", 147: "OpOuterProduct", 148: "OpDot",
}
ALLOWED_EXT_INSTRUCTIONS = {}
EXT_INSTRUCTION_NAMES = {50: "Fma", 46: "FMix", 43: "FClamp", 66: "Ldexp", 31: "Sqrt"}
DECORATION_RELAXED_PRECISION, DECORATION_NO_CONTRACTION = 0, 42
DECORATION_FP_FAST_MATH_MODE = 40
DECORATION_BUFFER_BLOCK, DECORATION_BLOCK = 3, 2
DECORATION_BINDING, DECORATION_DESCRIPTOR_SET, DECORATION_OFFSET = 33, 34, 35
DECORATION_NON_WRITABLE, DECORATION_NON_READABLE = 24, 25
STORAGE_UNIFORM, STORAGE_PUSH_CONSTANT, STORAGE_BUFFER = 2, 9, 12
OP_FDIV, OP_ATOMIC_UMAX = 136, 239
ATOMIC_OPS = set(range(227, 243))  # OpAtomicLoad .. OpAtomicXor
EXPECTED_FDIV = 2
MIN_NO_CONTRACTION_ARITHMETIC = 10
PUSH_CONSTANT_OFFSETS = [4 * i for i in range(13)]
PUSH_CONSTANT_FLOAT_MEMBERS = set()
EXPECTED_BINDINGS = {0: "readonly", 1: "readonly", 2: "readonly", 3: "readonly",
                     4: "readonly", 5: "writeonly", 6: "writeonly", 7: "readwrite"}
SOURCE_PUSH_FIELDS = [
    "operation", "n", "c", "reduction", "has_weight", "ignore_lo", "ignore_hi",
    "target_offset", "input_offset", "weight_offset", "grad_offset", "total_offset", "count",
]
MIN_SOURCE_PRECISE = 10


def fail(message):
    raise SystemExit(message)


def decode_embedded_spirv(header):
    match = re.search(r"kCode\[\]\s*=\s*\{(?P<words>.*?)\};", header, re.DOTALL)
    if not match:
        fail("embedded kCode array")
    words = [int(word, 16) for word in re.findall(r"0x([0-9a-fA-F]+)U", match.group("words"))]
    if not words:
        fail("embedded kCode is empty")
    return b"".join(struct.pack("<I", word) for word in words)


def _instructions(data):
    if len(data) % 4 or len(data) < 20:
        fail("SPIR-V binary has an invalid length")
    words = struct.unpack("<%dI" % (len(data) // 4), data)
    if words[0] != MAGIC:
        fail("SPIR-V binary has a bad magic number")
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xFFFF
        if count == 0 or index + count > len(words):
            fail("SPIR-V binary has a malformed instruction")
        yield opcode, list(words[index + 1:index + count])
        index += count


def _string(words):
    raw = b"".join(struct.pack("<I", word) for word in words)
    return raw.split(b"\0", 1)[0].decode("ascii")


def verify_spirv(data):
    """Structural float / binding contract of a compiled nll_loss shader."""
    decorations, member_decorations = {}, {}
    no_contraction, arithmetic = set(), []
    ext_sets, capabilities = {}, set()
    pointers, variables, structs, scalars = {}, {}, {}, {}
    local_size = None
    fdiv_count, atomics = 0, []
    for opcode, ops in _instructions(data):
        if opcode in FORBIDDEN_OPS:
            fail(f"nll_loss shader uses forbidden {FORBIDDEN_OPS[opcode]}")
        if opcode == OP_FDIV:
            fdiv_count += 1
        if opcode in ATOMIC_OPS:
            atomics.append(opcode)
        if opcode == OP_CAPABILITY:
            capabilities.add(ops[0])
        elif opcode == OP_EXT_INST_IMPORT:
            ext_sets[ops[0]] = _string(ops[1:])
        elif opcode == OP_EXT_INST:
            if ext_sets.get(ops[2]) != "GLSL.std.450":
                fail("nll_loss shader imports an unexpected instruction set")
            if ops[3] not in ALLOWED_EXT_INSTRUCTIONS:
                name = EXT_INSTRUCTION_NAMES.get(ops[3], f"instruction {ops[3]}")
                fail(f"nll_loss shader uses forbidden GLSL.std.450 {name}")
        elif opcode == OP_EXECUTION_MODE and ops[1] == 17:
            local_size = tuple(ops[2:5])
        elif opcode == OP_TYPE_INT:
            scalars[ops[0]] = ("int", ops[1], ops[2])
        elif opcode == OP_TYPE_FLOAT:
            scalars[ops[0]] = ("float", ops[1])
        elif opcode == OP_TYPE_POINTER:
            pointers[ops[0]] = (ops[1], ops[2])
        elif opcode == OP_TYPE_STRUCT:
            structs[ops[0]] = ops[1:]
        elif opcode == OP_VARIABLE:
            variables[ops[1]] = (ops[0], ops[2])
        elif opcode == OP_DECORATE:
            target, decoration = ops[0], ops[1]
            if decoration in (DECORATION_RELAXED_PRECISION, DECORATION_FP_FAST_MATH_MODE):
                fail("nll_loss shader uses RelaxedPrecision or FPFastMathMode")
            if decoration == DECORATION_NO_CONTRACTION:
                no_contraction.add(target)
            decorations.setdefault(target, {}).setdefault(decoration, []).append(ops[2:])
        elif opcode == OP_MEMBER_DECORATE:
            target, member, decoration = ops[0], ops[1], ops[2]
            if decoration == DECORATION_RELAXED_PRECISION:
                fail("nll_loss shader uses RelaxedPrecision or FPFastMathMode")
            member_decorations.setdefault((target, member), {}).setdefault(decoration, []).append(ops[3:])
        elif opcode in ARITHMETIC:
            arithmetic.append((opcode, ops[1]))
    if capabilities != {1, 11}:
        fail(f"nll_loss shader capabilities must be exactly Shader and Int64, got {sorted(capabilities)}")
    for kind in scalars.values():
        if kind[0] == "float" and kind[1] != 32:
            fail("nll_loss shader uses a non-32-bit float type")
        if kind[0] == "int" and kind[1] not in (32, 64):
            fail("nll_loss shader uses an unsupported integer width")
    if fdiv_count != EXPECTED_FDIV:
        fail(f"nll_loss shader must contain exactly {EXPECTED_FDIV} OpFDiv, got {fdiv_count}")
    if atomics != [OP_ATOMIC_UMAX]:
        fail(f"nll_loss shader must contain exactly one OpAtomicUMax and no other atomic, got {atomics}")
    if local_size != (256, 1, 1):
        fail(f"nll_loss shader workgroup must be 256x1x1, got {local_size}")
    for opcode, result in arithmetic:
        if result not in no_contraction:
            fail(f"nll_loss shader has {ARITHMETIC[opcode]} without NoContraction")
    if len(arithmetic) < MIN_NO_CONTRACTION_ARITHMETIC:
        fail(
            f"nll_loss shader has only {len(arithmetic)} float add/sub/mul "
            f"instructions; expected at least {MIN_NO_CONTRACTION_ARITHMETIC}"
        )
    _verify_layout(decorations, member_decorations, pointers, variables, structs, scalars)
    return len(arithmetic)


def _pointee_struct(variable, pointers, structs):
    pointer_type, storage = variable
    storage_class, pointee = pointers[pointer_type]
    if pointee not in structs:
        fail("nll_loss shader resource is not a struct")
    return storage_class, pointee


def _verify_layout(decorations, member_decorations, pointers, variables, structs, scalars):
    bindings, push_structs = {}, []
    for identifier, variable in variables.items():
        storage = variable[1]
        if storage in (STORAGE_UNIFORM, STORAGE_BUFFER):
            decorated = decorations.get(identifier, {})
            if DECORATION_BINDING not in decorated or DECORATION_DESCRIPTOR_SET not in decorated:
                fail("nll_loss shader buffer lacks a descriptor binding")
            if decorated[DECORATION_DESCRIPTOR_SET][0][0] != 0:
                fail("nll_loss shader must use descriptor set 0")
            bindings[decorated[DECORATION_BINDING][0][0]] = (identifier, variable)
        elif storage == STORAGE_PUSH_CONSTANT:
            push_structs.append(_pointee_struct(variable, pointers, structs)[1])
    if sorted(bindings) != sorted(EXPECTED_BINDINGS):
        fail(f"nll_loss shader bindings must be {sorted(EXPECTED_BINDINGS)}, got {sorted(bindings)}")
    for binding, access in EXPECTED_BINDINGS.items():
        identifier, variable = bindings[binding]
        storage_class, struct = _pointee_struct(variable, pointers, structs)
        block = decorations.get(struct, {})
        if storage_class == STORAGE_UNIFORM and DECORATION_BUFFER_BLOCK not in block:
            fail("nll_loss shader requires a storage-buffer descriptor")
        if storage_class == STORAGE_BUFFER and DECORATION_BLOCK not in block:
            fail("nll_loss shader requires a storage-buffer descriptor")
        member = member_decorations.get((struct, 0), {})
        if access == "readonly" and (DECORATION_NON_WRITABLE not in member or DECORATION_NON_READABLE in member):
            fail(f"nll_loss shader binding {binding} must be readonly")
        if access == "writeonly" and (DECORATION_NON_READABLE not in member or DECORATION_NON_WRITABLE in member):
            fail(f"nll_loss shader binding {binding} must be writeonly")
        if access == "readwrite" and (DECORATION_NON_READABLE in member or DECORATION_NON_WRITABLE in member):
            fail(f"nll_loss shader binding {binding} must be read-write")
    if len(push_structs) != 1:
        fail("nll_loss shader must declare exactly one push-constant block")
    struct = push_structs[0]
    members = structs[struct]
    if len(members) != len(PUSH_CONSTANT_OFFSETS):
        fail(f"nll_loss push constants must have {len(PUSH_CONSTANT_OFFSETS)} members")
    for index, offset in enumerate(PUSH_CONSTANT_OFFSETS):
        found = member_decorations.get((struct, index), {}).get(DECORATION_OFFSET)
        if not found or found[0][0] != offset:
            fail(f"nll_loss push-constant member {index} must be at offset {offset}")
        kind = scalars.get(members[index])
        want = "float" if index in PUSH_CONSTANT_FLOAT_MEMBERS else "int"
        if kind is None or kind[0] != want or (want == "int" and kind[2] != 0):
            fail(f"nll_loss push-constant member {index} has the wrong type")


def verify_source(text):
    """Textual guards that complement the SPIR-V proof."""
    code = re.sub(r"/\*.*?\*/", "", re.sub(r"//[^\n]*", "", text), flags=re.DOTALL)
    for binding in EXPECTED_BINDINGS:
        if not re.search(rf"layout\(set = 0, binding = {binding}[,)]", code):
            fail(f"nll_loss source missing binding {binding}")
    block = re.search(r"uniform Params\s*\{(?P<body>.*?)\}\s*params", code, re.DOTALL)
    if not block:
        fail("nll_loss source missing push-constant block")
    fields = re.findall(r"\b(?:uint|float)\s+(\w+)\s*;", block.group("body"))
    if fields != SOURCE_PUSH_FIELDS:
        fail(f"nll_loss push-constant fields are {fields}")
    # The float-division budget is enforced authoritatively as OpFDiv in the SPIR-V.
    for token in (r"\bfma\s*\(", r"\bmix\s*\(", r"\bclamp\s*\(", r"\bsqrt\s*\(", r"#pragma"):
        if re.search(token, code):
            fail(f"nll_loss source uses forbidden construct {token}")
    if len(re.findall(r"\bprecise\b", code)) < MIN_SOURCE_PRECISE:
        fail("nll_loss source lost its precise qualifiers")


def compile_source(source, compiler="glslc"):
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "nll_loss.comp.spv"
        subprocess.run([compiler, "-Os", "-o", str(binary), str(source)], check=True)
        return binary.read_bytes()


def verify(root=ROOT, compiler="glslc"):
    source = root / SOURCE_NAME
    header_path, manifest_path = root / HEADER_NAME, root / MANIFEST_NAME
    source_text = source.read_text(encoding="ascii")
    verify_source(source_text)
    if not header_path.exists() or not manifest_path.exists():
        fail("nll_loss shader products are missing: run tools/generate_nll_loss_spv.py")
    compiled = compile_source(source, compiler)
    header_bytes = header_path.read_bytes()
    embedded = decode_embedded_spirv(header_bytes.decode("ascii"))
    count = verify_spirv(compiled)
    if verify_spirv(embedded) != count:
        fail("embedded SPIR-V differs structurally from the compiled SPIR-V")
    if embedded != compiled:
        fail("embedded SPIR-V bytes differ from freshly compiled SPIR-V")
    validator = shutil.which("spirv-val")
    if validator is None:
        fail("spirv-val is required to verify the nll_loss shader")
    with tempfile.TemporaryDirectory() as directory:
        path = pathlib.Path(directory) / "nll_loss.embedded.spv"
        path.write_bytes(embedded)
        subprocess.run([validator, str(path)], check=True)
    expected = dict(line.split("=", 1) for line in manifest_path.read_text(encoding="ascii").splitlines())
    actual = {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "spirv_sha256": hashlib.sha256(compiled).hexdigest(),
        "header_sha256": hashlib.sha256(header_bytes).hexdigest(),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            fail(f"{key} mismatch: expected {expected.get(key)}, got {value}")
        print(f"{key}={value}")
    print(f"no_contraction_arithmetic={count}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    parser.add_argument("--glslc", default="glslc")
    args = parser.parse_args()
    verify(args.root, args.glslc)


if __name__ == "__main__":
    main()
