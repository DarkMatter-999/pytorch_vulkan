#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/linear_relu_backward.comp"
BINARY = pathlib.Path("/tmp/linear_relu_backward.comp.spv")
HEADER = ROOT / "src/vulkan/shaders/generated/linear_relu_backward_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/linear_relu_backward_spv.sha256"

subprocess.run(["glslc", "-Os", "-o", str(BINARY), str(SOURCE)], check=True)
data = BINARY.read_bytes()
words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
with HEADER.open("w", encoding="ascii") as header:
    header.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
    header.write("namespace vulkan_backward_shader {\ninline constexpr uint32_t kCode[] = {\n")
    for i in range(0, len(words), 8):
        header.write("    " + ", ".join(f"0x{x:08x}U" for x in words[i:i + 8]) + ",\n")
    header.write("};\ninline constexpr std::size_t kCodeSize = sizeof(kCode);\n}\n")

MANIFEST.write_text(
    "source_sha256=" + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + "\n"
    "spirv_sha256=" + hashlib.sha256(data).hexdigest() + "\n"
    "header_sha256=" + hashlib.sha256(HEADER.read_bytes()).hexdigest() + "\n",
    encoding="ascii",
)
