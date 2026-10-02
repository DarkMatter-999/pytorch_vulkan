#!/usr/bin/env python3
import argparse
import hashlib
import pathlib
import re
import shutil
import subprocess
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]

def sha(data): return hashlib.sha256(data).hexdigest()

def verify(root):
    source = root / "src/vulkan/shaders/glsl/cat_gather.comp"
    header = root / "src/vulkan/shaders/generated/cat_gather_spv.h"
    manifest = root / "src/vulkan/shaders/generated/cat_gather_spv.sha256"
    hashes = dict(line.split("=", 1) for line in manifest.read_text().splitlines())
    assert hashes["source_sha256"] == sha(source.read_bytes()), "source_sha256 mismatch"
    assert hashes["header_sha256"] == sha(header.read_bytes()), "header_sha256 mismatch"
    glslc = shutil.which("glslc")
    dis = shutil.which("spirv-dis")
    assert glslc and dis, "glslc and spirv-dis are required"
    with tempfile.TemporaryDirectory() as tmp:
        binary = pathlib.Path(tmp) / "cat.spv"
        subprocess.run([glslc, "-Os", "-o", str(binary), str(source)], check=True)
        blob = binary.read_bytes()
        assert hashes["spirv_sha256"] == sha(blob), "spirv_sha256 mismatch"
        assembly = subprocess.check_output([dis, str(binary)], text=True)
        words = [int(value, 16) for value in re.findall(r"0x([0-9a-fA-F]{8})U", header.read_text())]
        embedded = b"".join(word.to_bytes(4, "little") for word in words)
        assert embedded == blob, "embedded SPIR-V differs from freshly compiled shader"
    assert "OpExecutionMode %4 LocalSize 256 1 1" in assembly
    bindings = dict((int(b), var) for var, b in re.findall(r"OpDecorate %(\S+) Binding (\d+)", assembly))
    assert set(bindings) == {0, 1, 2}
    for binding, variable in bindings.items():
        assert f"OpDecorate %{variable} DescriptorSet 0" in assembly
        if binding in (0, 2):
            assert f"OpDecorate %{variable} NonWritable" in assembly
        if binding == 1:
            assert f"OpDecorate %{variable} NonReadable" in assembly
    assert "OpCapability Float64" not in assembly and "OpCapability Int64" not in assembly
    text = source.read_text()
    for word in ("words[0]", "words[1]", "words[2]", "words[3]", "words[4u", "words[8u", "words[12]", "words[17u"):
        assert word in text, f"metadata ABI access missing: {word}"
    assert "if (n - 1u - ordinal < step) break;" in text
    assert "ordinal += step;" in text
    assert "dst[dst_index] = src[src_index];" in text
    # Model all ordinals near UINT32_MAX with clamped groups; avoid allocations.
    n = 0xFFFFFFFE
    groups = min((n + 255) // 256, 0xFFFFFFFF // 256)
    step = groups * 256
    for start in (0, 1, step - 1, step, n - 1):
        ordinal = start
        visited = 0
        while ordinal < n:
            visited += 1
            if n - 1 - ordinal < step:
                break
            ordinal += step
        assert visited >= 1 and ordinal < n
    print("cat_gather shader contract=ok")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=pathlib.Path, default=ROOT)
    verify(parser.parse_args().root.resolve())
