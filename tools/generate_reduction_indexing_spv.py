#!/usr/bin/env python3
import pathlib
import subprocess
import hashlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "src/vulkan/shaders/generated/reduction_indexing_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/reduction_indexing_spv.sha256"
SHADERS = [
    ("kReductionCode", "reduction.comp"),
    ("kReductionBackwardCode", "reduction_backward.comp"),
    ("kIndexingCode", "indexing.comp"),
    ("kBroadcastCode", "broadcast.comp"),
]

with OUT.open("w", encoding="ascii") as header:
    header.write(
        "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\nnamespace vulkan_reduction_shader {\n"
    )
    for name, shader in SHADERS:
        path = OUT.parent.parent / "glsl" / shader
        binary = pathlib.Path("/tmp") / (shader + ".spv")
        subprocess.run(["glslc", "-Os", "-o", str(binary), str(path)], check=True)
        words = [
            int.from_bytes(binary.read_bytes()[i : i + 4], "little")
            for i in range(0, binary.stat().st_size, 4)
        ]
        header.write(f"inline constexpr uint32_t {name}[] = {{\n")
        for i in range(0, len(words), 8):
            header.write(
                "    " + ", ".join(f"0x{x:08x}U" for x in words[i : i + 8]) + ",\n"
            )
        header.write(
            f"}};\ninline constexpr std::size_t {name}Size = sizeof({name});\n"
        )
    header.write("} // namespace vulkan_reduction_shader\n")

spirv = b"".join(
    (pathlib.Path("/tmp") / (shader + ".spv")).read_bytes() for _, shader in SHADERS
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


MANIFEST.write_text(
    "reduction_source_sha256="
    + digest(
        b"".join(
            (OUT.parent.parent / "glsl" / shader).read_bytes() for _, shader in SHADERS
        )
    )
    + "\n"
    "spirv_sha256=" + digest(spirv) + "\n"
    "header_sha256=" + digest(OUT.read_bytes()) + "\n",
    encoding="ascii",
)
