#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = (ROOT / "src/vulkan/shaders/glsl/normalization.comp", ROOT / "src/vulkan/shaders/glsl/classification.comp")
HEADER = ROOT / "src/vulkan/shaders/generated/operator_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/operator_spv.sha256"
words = []
source_hash = hashlib.sha256()
for source in SOURCES:
    source_hash.update(source.read_bytes())
    binary = pathlib.Path("/tmp") / (source.stem + ".operator.spv")
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(source)], check=True)
    data = binary.read_bytes()
    words.append((source.stem, [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]))
with HEADER.open("w", encoding="ascii") as header:
    header.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
    for name, code in words:
        namespace = "vulkan_normalization_shader" if name == "normalization" else "vulkan_classification_shader"
        header.write(f"namespace {namespace} {{ inline constexpr uint32_t kCode[] = {{\n")
        for i in range(0, len(code), 8): header.write("    " + ", ".join(f"0x{x:08x}U" for x in code[i:i + 8]) + ",\n")
        header.write("}; inline constexpr std::size_t kCodeSize = sizeof(kCode); }\n")
MANIFEST.write_text("source_sha256=" + source_hash.hexdigest() + "\nheader_sha256=" + hashlib.sha256(HEADER.read_bytes()).hexdigest() + "\n", encoding="ascii")
