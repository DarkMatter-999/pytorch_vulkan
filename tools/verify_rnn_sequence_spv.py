#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import re
import subprocess
import tempfile


ROOT = pathlib.Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    args = parser.parse_args()
    root = args.root
    source_path = root / "src/vulkan/shaders/glsl/rnn_sequence.comp"
    generated_path = root / "src/vulkan/shaders/generated/rnn_sequence_spv.h"
    manifest_path = root / "src/vulkan/shaders/generated/rnn_sequence_spv.sha256"
    source = source_path.read_text(encoding="ascii")
    source_without_comments = re.sub(r"//[^\n]*|/\*.*?\*/", "", source, flags=re.DOTALL)
    required = [
        "layout(local_size_x = 256) in",
        *[f"binding = {binding}" for binding in range(14)],
        "output_buffer",
        "saved_state",
        "gradient_output",
        "gradient_input",
        "gradient_weight",
        "gradient_recurrent_weight",
        "gradient_bias",
        "tanh(",
        "params.mode == 1u",
        "params.mode == 2u",
        "params.mode == 3u",
        "params.mode == 4u",
    ]
    for token in required:
        if token not in source_without_comments:
            raise SystemExit(f"shader contract missing: {token}")
    if not re.search(
        r"for\s*\(\s*uint\s+sequence_index\s*=.*?sequence_index\s*<\s*params\.sequence",
        source_without_comments,
        flags=re.DOTALL,
    ):
        raise SystemExit("shader contract missing executable sequence loop")
    expected = dict(line.split("=", 1) for line in manifest_path.read_text(encoding="ascii").splitlines())
    with tempfile.TemporaryDirectory() as directory:
        binary = pathlib.Path(directory) / "rnn_sequence.comp.spv"
        subprocess.run(["glslc", "-Os", "-o", str(binary), str(source_path)], check=True)
        spirv = binary.read_bytes()
    header = generated_path.read_bytes()
    if not re.search(rb"namespace vulkan_rnn_sequence_shader", header):
        raise SystemExit("generated header has the wrong namespace")
    actual = {
        "source_sha256": digest(source_path.read_bytes()),
        "spirv_sha256": digest(spirv),
        "header_sha256": digest(header),
    }
    for key, value in actual.items():
        if expected.get(key) != value:
            raise SystemExit(f"{key} mismatch: expected {expected.get(key)}, got {value}")
        print(f"{key}={value}")
    print("rnn_sequence shader contract=ok")


if __name__ == "__main__":
    main()
