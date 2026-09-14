#!/usr/bin/env python3
"""Verify checked-in pointwise SPIR-V against the source and manifest."""

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
    root = parser.parse_args().root or pathlib.Path(__file__).resolve().parents[1]
    source = root / "src/vulkan/shaders/glsl/pointwise.comp"
    generated = root / "src/vulkan/shaders/generated/pointwise_spv.h"
    manifest = root / "src/vulkan/shaders/generated/pointwise_spv.sha256"
    expected = dict(
        line.split("=", 1)
        for line in manifest.read_text(encoding="ascii").splitlines()
        if "=" in line
    )
    version = subprocess.check_output(["glslc", "--version"], text=True).splitlines()[0].strip()
    if expected.get("compiler") != f"glslc {version}" or re.fullmatch(r"\d+\.\d+", version) is None:
        raise SystemExit(f"pointwise manifest compiler mismatch: expected glslc {version}")

    names = [("kTensorTensorCode", 0, False, False),
             ("kTensorScalarCode", 1, False, False),
             ("kScalarTensorCode", 2, False, False),
             ("kUnaryCode", 3, False, False),
             ("kBoolTensorTensorCode", 0, True, False),
             ("kBoolOutputTensorScalarCode", 1, False, True),
             ("kBoolOutputUnaryCode", 3, False, True),
             ("kBoolOutputTensorTensorCode", 0, False, True)]
    with tempfile.TemporaryDirectory() as directory:
        binaries = []
        for _, mode, bool_dtype, bool_output in names:
            path = pathlib.Path(directory) / f"pointwise_{mode}_{int(bool_dtype)}_{int(bool_output)}.spv"
            command = ["glslc", "-Os", f"-DPOINTWISE_MODE={mode}"]
            if bool_dtype:
                command.append("-DPOINTWISE_BOOL")
            if bool_output:
                command.append("-DPOINTWISE_BOOL_OUTPUT")
            subprocess.run(command + ["-o", str(path), str(source)], check=True)
            binaries.append(path.read_bytes())

    actual = {
        "source_sha256": sha256(source.read_bytes()),
        "spirv_sha256": sha256(b"".join(binaries)),
        "header_sha256": sha256(generated.read_bytes()),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
