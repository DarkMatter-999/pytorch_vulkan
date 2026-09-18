#!/usr/bin/env python3
import hashlib
import pathlib
import re
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/vulkan/shaders/glsl/masked_select.comp"
OUTPUT = ROOT / "src/vulkan/shaders/generated/masked_select_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/masked_select_spv.sha256"
EXPECTED_VERSION = "2026.3"


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    expected = dict(
        line.split("=", 1)
        for line in MANIFEST.read_text(encoding="ascii").splitlines()
        if "=" in line
    )
    if expected.get("compiler") != f"glslc {EXPECTED_VERSION}":
        raise SystemExit("masked-select manifest compiler mismatch")
    version = (
        subprocess.check_output(["glslc", "--version"], text=True)
        .splitlines()[0]
        .strip()
    )
    if re.fullmatch(r"\d+\.\d+", version) is None or version != EXPECTED_VERSION:
        raise SystemExit(f"expected glslc {EXPECTED_VERSION}, got {version!r}")
    with tempfile.TemporaryDirectory() as directory:
        binaries = []
        for name, define in (
            ("kCountCode", "MASKED_SELECT_COUNT"),
            ("kCompactCode", None),
        ):
            path = pathlib.Path(directory) / f"{name}.spv"
            command = (
                ["glslc", "-Os"]
                + ([f"-D{define}"] if define else [])
                + ["-o", str(path), str(SOURCE)]
            )
            subprocess.run(command, check=True)
            binaries.append(path.read_bytes())
    with OUTPUT.open("w", encoding="ascii") as header:
        header.write(
            "#pragma once\n\n#include <cstddef>\n#include <cstdint>\n\nnamespace vulkan_masked_select_shader {\n"
        )
        for name, binary in zip(("kCountCode", "kCompactCode"), binaries):
            words = [
                int.from_bytes(binary[i : i + 4], "little")
                for i in range(0, len(binary), 4)
            ]
            header.write(f"inline constexpr uint32_t {name}[] = {{\n")
            for i in range(0, len(words), 8):
                header.write(
                    "    " + ", ".join(f"0x{x:08x}U" for x in words[i : i + 8]) + ",\n"
                )
            header.write(
                f"}};\ninline constexpr std::size_t {name}Size = sizeof({name});\n"
            )
        header.write("} // namespace vulkan_masked_select_shader\n")
    actual = {
        "source_sha256": sha256(SOURCE.read_bytes()),
        "spirv_sha256": sha256(b"".join(binaries)),
        "header_sha256": sha256(OUTPUT.read_bytes()),
    }
    for key, value in actual.items():
        if expected.get(key) is not None and expected.get(key) != value:
            raise SystemExit(
                f"{key} mismatch: expected {expected.get(key)}, got {value}"
            )
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
