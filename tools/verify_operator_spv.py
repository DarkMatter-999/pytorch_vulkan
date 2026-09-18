#!/usr/bin/env python3
import hashlib
import pathlib
import subprocess

ROOT = pathlib.Path(__file__).resolve().parents[1]
subprocess.run([str(ROOT / ".venv/bin/python"), str(ROOT / "tools/generate_operator_spv.py")], check=True)
manifest = (ROOT / "src/vulkan/shaders/generated/operator_spv.sha256").read_text(encoding="ascii").splitlines()
header_hash = hashlib.sha256((ROOT / "src/vulkan/shaders/generated/operator_spv.h").read_bytes()).hexdigest()
assert manifest[-1] == "header_sha256=" + header_hash
print("operator shader integrity=ok")
