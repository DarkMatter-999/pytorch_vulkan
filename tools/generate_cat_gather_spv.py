#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/cat_gather.comp"
DEFAULT_OUTPUT = ROOT / "src/vulkan/shaders/generated"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def generate(output):
    output.mkdir(parents=True, exist_ok=True)
    header = output / "cat_gather_spv.h"
    with tempfile.TemporaryDirectory() as temp:
        binary_path = pathlib.Path(temp) / "cat_gather.spv"
        subprocess.run(["glslc", "-Os", "-o", str(binary_path), str(SOURCE)], check=True)
        binary = binary_path.read_bytes()
    words = [int.from_bytes(binary[i:i + 4], "little") for i in range(0, len(binary), 4)]
    with header.open("w", encoding="ascii", newline="\n") as f:
        f.write("#pragma once\n#include <cstddef>\n#include <cstdint>\nnamespace vulkan_cat_gather_shader {\ninline constexpr uint32_t kCode[] = {\n")
        for i in range(0, len(words), 8):
            f.write("    " + ", ".join(f"0x{w:08x}U" for w in words[i:i + 8]) + ",\n")
        f.write("};\ninline constexpr std::size_t kCodeSize = sizeof(kCode);\n}\n")
    manifest = output / "cat_gather_spv.sha256"
    manifest.write_text(f"source_sha256={digest(SOURCE.read_bytes())}\nspirv_sha256={digest(binary)}\nheader_sha256={digest(header.read_bytes())}\n", encoding="ascii", newline="\n")
    print(manifest.read_text(), end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    generate(parser.parse_args().output_dir.resolve())
