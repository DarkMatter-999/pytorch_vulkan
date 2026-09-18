import hashlib
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


def test_gemm_generator_is_deterministic_and_records_hashes(tmp_path):
    generator = ROOT / "tools/generate_gemm_spv.py"
    outputs = []
    for name in ("first", "second"):
        output = tmp_path / name
        subprocess.run(
            ["python3", str(generator), "--output-dir", str(output)],
            check=True,
        )
        outputs.append(output)

    for filename in ("gemm_spv.h", "gemm_spv.sha256"):
        assert (outputs[0] / filename).read_bytes() == (
            outputs[1] / filename
        ).read_bytes()

    manifest = dict(
        line.split("=", 1)
        for line in (outputs[0] / "gemm_spv.sha256")
        .read_text(encoding="ascii")
        .splitlines()
    )
    assert list(manifest) == ["source_sha256", "spirv_sha256", "header_sha256"]
    assert (
        manifest["source_sha256"]
        == hashlib.sha256(
            (ROOT / "src/vulkan/shaders/glsl/gemm.comp").read_bytes()
        ).hexdigest()
    )
    with tempfile.TemporaryDirectory() as directory:
        spirv = Path(directory) / "gemm.comp.spv"
        subprocess.run(
            [
                "glslc",
                "-Os",
                "-o",
                str(spirv),
                str(ROOT / "src/vulkan/shaders/glsl/gemm.comp"),
            ],
            check=True,
        )
        assert (
            manifest["spirv_sha256"] == hashlib.sha256(spirv.read_bytes()).hexdigest()
        )
    assert (
        manifest["header_sha256"]
        == hashlib.sha256((outputs[0] / "gemm_spv.h").read_bytes()).hexdigest()
    )
    (outputs[0] / "gemm_spv.h").read_text(encoding="ascii")


def test_masked_select_verifier_rejects_tampered_header(tmp_path):
    if shutil.which("glslc") is None:
        pytest.skip("glslc is not available")

    for relative in (
        "src/vulkan/shaders/glsl/masked_select.comp",
        "src/vulkan/shaders/generated/masked_select_spv.h",
        "src/vulkan/shaders/generated/masked_select_spv.sha256",
    ):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)

    verifier = ROOT / "tools/verify_masked_select_spv.py"
    subprocess.run(["python3", str(verifier), "--root", str(tmp_path)], check=True)

    header = tmp_path / "src/vulkan/shaders/generated/masked_select_spv.h"
    original = header.read_bytes()
    header.write_bytes(original + b"\n")
    result = subprocess.run(
        ["python3", str(verifier), "--root", str(tmp_path)],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "header_sha256 mismatch" in result.stderr
    assert header.read_bytes() == original + b"\n"
