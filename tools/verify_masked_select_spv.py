#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import re
import subprocess
import tempfile


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path)
    args = parser.parse_args()
    root = args.root or pathlib.Path(__file__).resolve().parents[1]
    source = root / "src/vulkan/shaders/glsl/masked_select.comp"
    generated = root / "src/vulkan/shaders/generated/masked_select_spv.h"
    manifest = root / "src/vulkan/shaders/generated/masked_select_spv.sha256"
    expected = dict(
        line.split("=", 1)
        for line in manifest.read_text(encoding="ascii").splitlines()
        if "=" in line
    )
    expected_version = "2026.3"
    if expected.get("compiler") != f"glslc {expected_version}":
        raise SystemExit("masked-select manifest compiler mismatch")
    version = (
        subprocess.check_output(["glslc", "--version"], text=True)
        .splitlines()[0]
        .strip()
    )
    if re.fullmatch(r"\d+\.\d+", version) is None or version != expected_version:
        raise SystemExit(f"expected glslc {expected_version}, got {version!r}")
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
                + ["-o", str(path), str(source)]
            )
            subprocess.run(command, check=True)
            binaries.append(path.read_bytes())
    actual = {
        "source_sha256": sha256(source.read_bytes()),
        "spirv_sha256": sha256(b"".join(binaries)),
        "header_sha256": sha256(generated.read_bytes()),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise SystemExit(
                f"{key} mismatch: expected {expected.get(key)}, got {value}"
            )
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
