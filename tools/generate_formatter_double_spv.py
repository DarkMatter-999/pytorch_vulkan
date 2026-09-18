#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess
import struct
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/formatter_double.comp"
DEFAULT_OUTPUT = ROOT / "src/vulkan/shaders/generated"

directory = pathlib.Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_OUTPUT
directory.mkdir(parents=True, exist_ok=True)
HEADER = directory / "formatter_double_spv.h"
MANIFEST = directory / "formatter_double_spv.sha256"
binary_directory = directory if len(sys.argv) > 1 else pathlib.Path("/tmp")
with tempfile.TemporaryDirectory(
    dir=binary_directory if binary_directory.exists() else None
) as temp:
    binary = pathlib.Path(temp) / "formatter_double.comp.spv"
    subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
    spirv = binary.read_bytes()

    words = struct.unpack(f"<{len(spirv) // 4}I", spirv)
    lines = [
        "#pragma once",
        "#include <cstddef>",
        "#include <cstdint>",
        "namespace vulkan_formatter_double_shader {",
        "inline constexpr uint32_t kCode[] = {",
    ]
    lines.extend(
        "    " + ", ".join(f"0x{word:08x}U" for word in words[i : i + 8]) + ","
        for i in range(0, len(words), 8)
    )
    lines.extend(
        ["};", "inline constexpr std::size_t kCodeSize = sizeof(kCode);", "}", ""]
    )
    HEADER.write_text("\n".join(lines), encoding="ascii")
    MANIFEST.write_text(
        f"source_sha256={hashlib.sha256(SOURCE.read_bytes()).hexdigest()}\n"
        f"spirv_sha256={hashlib.sha256(spirv).hexdigest()}\n"
        f"header_sha256={hashlib.sha256(HEADER.read_bytes()).hexdigest()}\n",
        encoding="ascii",
    )
