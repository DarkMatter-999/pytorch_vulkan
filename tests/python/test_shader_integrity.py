import shutil
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]


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
