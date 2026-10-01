#!/usr/bin/env python3
"""Verify checked-in pointwise SPIR-V against the source and manifest."""

import argparse
import hashlib
import pathlib
import re
import struct
import subprocess
import tempfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def verify_arithmetic_capabilities(binary, name):
    """All shared modes must load without optional 16/64-bit arithmetic."""
    words = struct.unpack(f"<{len(binary) // 4}I", binary)
    index = 5
    while index < len(words):
        count, opcode = words[index] >> 16, words[index] & 0xFFFF
        if count == 0 or index + count > len(words):
            raise SystemExit(f"{name}: malformed SPIR-V instruction")
        if opcode == 17 and words[index + 1] in {9, 10, 11}:
            raise SystemExit(f"{name}: optional arithmetic capability {words[index + 1]}")
        index += count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path)
    root = parser.parse_args().root or pathlib.Path(__file__).resolve().parents[1]
    source = root / "src/vulkan/shaders/glsl/pointwise.comp"
    generated = root / "src/vulkan/shaders/generated/pointwise_spv.h"
    manifest = root / "src/vulkan/shaders/generated/pointwise_spv.sha256"
    expected = dict(
        line.split("=", 1)
        for line in manifest.read_text(encoding="ascii").splitlines()
        if "=" in line
    )
    version = (
        subprocess.check_output(["glslc", "--version"], text=True)
        .splitlines()[0]
        .strip()
    )
    if (
        expected.get("compiler") != f"glslc {version}"
        or re.fullmatch(r"\d+\.\d+", version) is None
    ):
        raise SystemExit(
            f"pointwise manifest compiler mismatch: expected glslc {version}"
        )

    names = [
        ("kTensorTensorCode", 0, False, False),
        ("kTensorScalarCode", 1, False, False),
        ("kScalarTensorCode", 2, False, False),
        ("kUnaryCode", 3, False, False),
        ("kBoolTensorTensorCode", 0, True, False),
        ("kBoolOutputTensorScalarCode", 1, False, True),
        ("kBoolOutputUnaryCode", 3, False, True),
        ("kBoolOutputTensorTensorCode", 0, False, True),
        ("kCompoundMulCode", 4, False, False),
        ("kCompoundDivCode", 5, False, False),
    ]
    with tempfile.TemporaryDirectory() as directory:
        binaries = []
        for name, mode, bool_dtype, bool_output in names:
            path = (
                pathlib.Path(directory)
                / f"pointwise_{mode}_{int(bool_dtype)}_{int(bool_output)}.spv"
            )
            command = ["glslc", "-Os", f"-DPOINTWISE_MODE={mode}"]
            if bool_dtype:
                command.append("-DPOINTWISE_BOOL")
            if bool_output:
                command.append("-DPOINTWISE_BOOL_OUTPUT")
            subprocess.run(command + ["-o", str(path), str(source)], check=True)
            binary = path.read_bytes()
            verify_arithmetic_capabilities(binary, name)
            binaries.append(binary)

    actual = {
        "source_sha256": sha256(source.read_bytes()),
        "spirv_sha256": sha256(b"".join(binaries)),
        "header_sha256": sha256(generated.read_bytes()),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise SystemExit(
                f"{key} mismatch: expected {expected.get(key)}, got {value}"
            )
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
