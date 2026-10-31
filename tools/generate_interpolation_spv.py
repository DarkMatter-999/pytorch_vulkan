#!/usr/bin/env python3
"""Compile interpolation.comp and publish interpolation_spv.h / .sha256.

Run it under the pinned shader profile, e.g.
  tools/shader_toolchain.py run --profile <profile.json> -- python tools/generate_interpolation_spv.py
Nothing is published unless the compiled SPIR-V passes the structural float/binding
contract in verify_interpolation_spv.py (NoContraction on every float add/sub/mul, ...).
"""
import argparse
import hashlib
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import verify_interpolation_spv as verifier  # noqa: E402


def generate(root, compiler="glslc"):
    source = root / verifier.SOURCE_NAME
    header_path = root / verifier.HEADER_NAME
    manifest_path = root / verifier.MANIFEST_NAME
    verifier.verify_source(source.read_text(encoding="ascii"))
    data = verifier.compile_source(source, compiler)
    verifier.verify_spirv(data)
    words = [int.from_bytes(data[i:i + 4], "little") for i in range(0, len(data), 4)]
    lines = [
        "#pragma once\n#include <cstddef>\n#include <cstdint>\n"
        "namespace vulkan_interpolation_shader {\ninline constexpr uint32_t kCode[] = {\n"
    ]
    for i in range(0, len(words), 8):
        lines.append("    " + ", ".join(f"0x{x:08x}U" for x in words[i:i + 8]) + ",\n")
    lines.append("};\ninline constexpr std::size_t kCodeSize = sizeof(kCode);\n}\n")
    header_data = "".join(lines).encode("ascii")
    manifest = (
        f"source_sha256={hashlib.sha256(source.read_bytes()).hexdigest()}\n"
        f"spirv_sha256={hashlib.sha256(data).hexdigest()}\n"
        f"header_sha256={hashlib.sha256(header_data).hexdigest()}\n"
    )
    with tempfile.TemporaryDirectory(dir=header_path.parent) as staging:
        staged_header = pathlib.Path(staging) / header_path.name
        staged_manifest = pathlib.Path(staging) / manifest_path.name
        staged_header.write_bytes(header_data)
        staged_manifest.write_text(manifest, encoding="ascii")
        staged_header.replace(header_path)
        staged_manifest.replace(manifest_path)
    print(manifest, end="")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=pathlib.Path, default=verifier.ROOT)
    parser.add_argument("--glslc", default="glslc")
    args = parser.parse_args()
    generate(args.root, args.glslc)


if __name__ == "__main__":
    main()
