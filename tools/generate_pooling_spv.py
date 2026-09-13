#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import tempfile

root = pathlib.Path(__file__).resolve().parents[1]
source = root / "src/vulkan/shaders/glsl/pooling.comp"
header_path = root / "src/vulkan/shaders/generated/pooling_spv.h"
manifest_path = root / "src/vulkan/shaders/generated/pooling_spv.sha256"
with tempfile.TemporaryDirectory() as directory:
    binary = pathlib.Path(directory) / "pooling.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(source)], check=True)
    data = binary.read_bytes()
words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
with header_path.open("w", encoding="ascii") as header:
    header.write("#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace vulkan_pooling_shader {\ninline constexpr uint32_t kCode[] = {\n")
    for i in range(0, len(words), 8):
        header.write("    " + ", ".join(f"0x{x:08x}U" for x in words[i:i + 8]) + ",\n")
    header.write("};\ninline constexpr std::size_t kCodeSize = sizeof(kCode);\n}\n")
header_data = header_path.read_bytes()
manifest_path.write_text("source_sha256=" + hashlib.sha256(source.read_bytes()).hexdigest() + "\n" +
                         "spirv_sha256=" + hashlib.sha256(data).hexdigest() + "\n" +
                         "header_sha256=" + hashlib.sha256(header_data).hexdigest() + "\n", encoding="ascii")
