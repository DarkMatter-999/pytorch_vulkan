#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/gemm.comp"
DEFAULT_OUTPUT = ROOT / "src/vulkan/shaders/generated"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_header(path, binary):
    words = [
        int.from_bytes(binary[index : index + 4], "little")
        for index in range(0, len(binary), 4)
    ]
    with path.open("w", encoding="ascii", newline="\n") as header:
        header.write(
            "#pragma once\n\n"
            "#include <cstddef>\n"
            "#include <cstdint>\n\n"
            "namespace vulkan_gemm_shader {\n"
            "inline constexpr uint32_t kCode[] = {\n"
        )
        for index in range(0, len(words), 8):
            header.write(
                "    "
                + ", ".join(f"0x{word:08x}U" for word in words[index : index + 8])
                + ",\n"
            )
        header.write(
            "};\n"
            "inline constexpr std::size_t kCodeSize = sizeof(kCode);\n"
            "} // namespace vulkan_gemm_shader\n"
        )


def generate(output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    header = output_dir / "gemm_spv.h"
    manifest = output_dir / "gemm_spv.sha256"
    with tempfile.TemporaryDirectory() as directory:
        binary_path = pathlib.Path(directory) / "gemm.comp.spv"
        subprocess.run(
            ["glslc", "-Os", "-o", str(binary_path), str(SOURCE)],
            check=True,
        )
        binary = binary_path.read_bytes()
    write_header(header, binary)
    manifest.write_text(
        "source_sha256="
        + sha256(SOURCE.read_bytes())
        + "\nspirv_sha256="
        + sha256(binary)
        + "\nheader_sha256="
        + sha256(header.read_bytes())
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    for key, value in (
        ("source_sha256", sha256(SOURCE.read_bytes())),
        ("spirv_sha256", sha256(binary)),
        ("header_sha256", sha256(header.read_bytes())),
    ):
        print(f"{key}={value}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=pathlib.Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    generate(args.output_dir.resolve())


if __name__ == "__main__":
    main()
