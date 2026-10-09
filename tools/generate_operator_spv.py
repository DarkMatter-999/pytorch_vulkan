#!/usr/bin/env python3
import hashlib
import argparse
import pathlib
import subprocess
import tempfile
import os
import shutil
import fcntl

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCES = (
    ROOT / "src/vulkan/shaders/glsl/normalization.comp",
    ROOT / "src/vulkan/shaders/glsl/classification.comp",
)
HEADER = ROOT / "src/vulkan/shaders/generated/operator_spv.h"
MANIFEST = ROOT / "src/vulkan/shaders/generated/operator_spv.sha256"


def publish_products(directory, header, manifest):
    """Stage both products; roll back caught publication failures.

    Each rename is atomic, NOT the pair. An observer may see a mixed pair until
    the second rename; interruption/power loss or failed rollback needs manual
    recovery from retained backups. Cooperating generators serialize on the
    directory lock. Verification never treats a mixed pair as valid.
    """
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    staging = None
    retain = False
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        targets = [directory / HEADER.name, directory / MANIFEST.name]
        if any(p.is_symlink() or (p.exists() and not p.is_file()) for p in targets):
            raise ValueError("operator publication refuses symlink/non-file outputs")
        staging = pathlib.Path(tempfile.mkdtemp(prefix=".operator-stage-", dir=directory))
        existed = [p.exists() for p in targets]
        for index, (target, data) in enumerate(zip(targets, (header, manifest))):
            if existed[index]:
                shutil.copy2(target, staging / (target.name + ".backup"))
            staged = staging / target.name
            staged.write_bytes(data)
            if existed[index]:
                staged.chmod(target.stat().st_mode & 0o777)
            with staged.open("rb") as file:
                os.fsync(file.fileno())
        published = []
        try:
            for index, target in enumerate(targets):
                os.replace(staging / target.name, target)
                published.append(index)
            os.fsync(descriptor)
        except BaseException:
            try:
                for index in reversed(published):
                    target = targets[index]
                    if existed[index]:
                        os.replace(staging / (target.name + ".backup"), target)
                    else:
                        target.unlink()
                os.fsync(descriptor)
            except BaseException as recovery_error:
                retain = True
                raise RuntimeError(f"operator rollback failed; recover retained backups in {staging}") from recovery_error
            raise
    finally:
        os.close(descriptor)
        if staging is not None and not retain:
            shutil.rmtree(staging)  # Only this invocation's private staging tree.


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-directory", type=pathlib.Path, default=HEADER.parent)
    parser.add_argument("--glslc", default="glslc")
    parser.add_argument("--expected-header-sha256")
    parser.add_argument("--expected-manifest-sha256")
    args = parser.parse_args()
    words = []
    source_hash = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix="operator-generate-") as directory:
        for source in SOURCES:
            source_hash.update(source.read_bytes())
            binary = pathlib.Path(directory) / (source.stem + ".spv")
            subprocess.run([args.glslc, "-Os", "-o", str(binary), str(source)], check=True)
            data = binary.read_bytes()
            words.append((source.stem, [int.from_bytes(data[i:i+4], "little")
                                        for i in range(0, len(data), 4)]))
    lines = ["#pragma once\n#include <cstddef>\n#include <cstdint>\n"]
    for name, code in words:
        lines.append(f"namespace vulkan_{name}_shader {{ inline constexpr uint32_t kCode[] = {{\n")
        for i in range(0, len(code), 8):
            lines.append("    " + ", ".join(f"0x{x:08x}U" for x in code[i:i+8]) + ",\n")
        lines.append("}; inline constexpr std::size_t kCodeSize = sizeof(kCode); }\n")
    header = "".join(lines).encode("ascii")
    manifest = ("source_sha256=" + source_hash.hexdigest() + "\nheader_sha256="
                + hashlib.sha256(header).hexdigest() + "\n").encode("ascii")
    for name, data, expected in (("header", header, args.expected_header_sha256),
                                 ("manifest", manifest, args.expected_manifest_sha256)):
        if expected is not None and hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"operator generation: expected {name} hash mismatch; outputs not written")
    publish_products(args.output_directory, header, manifest)


if __name__ == "__main__":
    main()
