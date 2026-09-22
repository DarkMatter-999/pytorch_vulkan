#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/rnn_sequence.comp"
DEFAULT_OUTPUT = ROOT / "src/vulkan/shaders/generated"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    header = output / "rnn_sequence_spv.h"
    manifest = output / "rnn_sequence_spv.sha256"
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "rnn_sequence.comp.spv"
        subprocess.run(["glslc", "-Os", "-o", str(binary), str(SOURCE)], check=True)
        data = binary.read_bytes()
    words = [int.from_bytes(data[i : i + 4], "little") for i in range(0, len(data), 4)]
    with header.open("w", encoding="ascii") as generated:
        generated.write("#pragma once\n#include <cstddef>\n#include <cstdint>\n")
        generated.write(
            "namespace vulkan_rnn_sequence_shader {\ninline constexpr uint32_t kCode[] = {\n"
        )
        for i in range(0, len(words), 8):
            generated.write("    " + ", ".join(f"0x{x:08x}U" for x in words[i : i + 8]) + ",\n")
        generated.write("};\ninline constexpr std::size_t kCodeSize = sizeof(kCode);\n}\n")
    manifest.write_text(
        "source_sha256=" + hashlib.sha256(SOURCE.read_bytes()).hexdigest() + "\n"
        "spirv_sha256=" + hashlib.sha256(data).hexdigest() + "\n"
        "header_sha256=" + hashlib.sha256(header.read_bytes()).hexdigest() + "\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
