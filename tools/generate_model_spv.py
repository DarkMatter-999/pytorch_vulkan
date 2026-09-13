#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess

root = pathlib.Path(__file__).resolve().parents[1]
source = root / "src/vulkan/shaders/glsl/model.comp"
binary = pathlib.Path("/tmp/model.comp.spv")
header_path = root / "src/vulkan/shaders/generated/model_spv.h"
manifest_path = root / "src/vulkan/shaders/generated/model_spv.sha256"
subprocess.run(["glslc", "-Os", "-o", str(binary), str(source)], check=True)
data = binary.read_bytes()
words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
with header_path.open("w", encoding="ascii") as header:
    header.write("#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace vulkan_model_shader {\n")
    header.write("inline constexpr uint32_t kLinearCode[] = {\n")
    for i in range(0, len(words), 8):
        header.write("    " + ", ".join(f"0x{x:08x}U" for x in words[i:i + 8]) + ",\n")
    header.write("};\ninline constexpr std::size_t kLinearCodeSize = sizeof(kLinearCode);\n}\n")
header_data = header_path.read_bytes()
manifest_path.write_text(
    "source_sha256=" + hashlib.sha256(source.read_bytes()).hexdigest() + "\n"
    "spirv_sha256=" + hashlib.sha256(data).hexdigest() + "\n"
    "header_sha256=" + hashlib.sha256(header_data).hexdigest() + "\n",
    encoding="ascii",
)
