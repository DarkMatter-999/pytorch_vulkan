#!/usr/bin/env python3
"""Compile the shared pointwise shader and emit its checked-in C++ header."""

import hashlib
import pathlib
import subprocess
import tempfile
import re

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/pointwise.comp"
OUTPUT = ROOT / "src/vulkan/shaders/generated/pointwise_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/pointwise_spv.sha256"
EXPECTED_VERSION = "2026.3"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    expected = {}
    for line in MANIFEST.read_text(encoding="ascii").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            expected[key] = value
    expected_compiler = f"glslc {EXPECTED_VERSION}"
    if expected.get("compiler") != expected_compiler:
        raise SystemExit(
            f"manifest compiler mismatch: expected {expected_compiler}, "
            f"got {expected.get('compiler')}"
        )

    version = subprocess.check_output(["glslc", "--version"], text=True)
    first_line = version.splitlines()[0].strip()
    if re.fullmatch(r"\d+\.\d+", first_line) is None or first_line != EXPECTED_VERSION:
        raise SystemExit(f"expected glslc {EXPECTED_VERSION}, got {version!r}")

    names = [("kTensorTensorCode", 0), ("kTensorScalarCode", 1),
             ("kScalarTensorCode", 2)]
    with tempfile.TemporaryDirectory() as directory:
        binaries = []
        for _, mode in names:
            path = pathlib.Path(directory) / f"pointwise_{mode}.spv"
            subprocess.run(
                ["glslc", "-Os", f"-DPOINTWISE_MODE={mode}", "-o", str(path),
                 str(SOURCE)], check=True)
            binaries.append(path.read_bytes())

    with OUTPUT.open("w", encoding="ascii") as header:
        header.write("#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\n")
        header.write("namespace vulkan_pointwise_shader {\n")
        for (name, _), binary in zip(names, binaries):
            words = [int.from_bytes(binary[index:index + 4], "little")
                     for index in range(0, len(binary), 4)]
            header.write(f"inline constexpr uint32_t {name}[] = {{\n")
            for index in range(0, len(words), 8):
                values = ", ".join(f"0x{word:08x}U" for word in words[index:index + 8])
                header.write(f"    {values},\n")
            header.write("};\n")
            header.write(f"inline constexpr std::size_t {name}Size = sizeof({name});\n")
        header.write("} // namespace vulkan_pointwise_shader\n")

    actual = {
        "source_sha256": sha256(SOURCE.read_bytes()),
        "spirv_sha256": sha256(b"".join(binaries)),
        "header_sha256": sha256(OUTPUT.read_bytes()),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
