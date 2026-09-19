#!/usr/bin/env python3
"""Record the supported Vulkan build environment as machine-readable JSON."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _command(command: list[str]) -> dict[str, str]:
    name = command[0]
    try:
        result = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
    ) as exc:
        if isinstance(exc, subprocess.CalledProcessError):
            detail = (exc.stderr or exc.stdout or "command failed").strip()
        elif isinstance(exc, subprocess.TimeoutExpired):
            detail = "command timed out"
        else:
            detail = f"executable not found: {name}"
        return {"status": "unavailable", "error": detail}

    output = (result.stdout or result.stderr).strip().splitlines()
    return {"status": "available", "version": output[0] if output else "unknown"}


def _git(args: list[str]) -> dict[str, str]:
    return _command(["git", *args])


def _repository() -> dict[str, object]:
    root_result = _git(["rev-parse", "--show-toplevel"])
    git_root = root_result.get("version", str(ROOT))
    revision = _git(["rev-parse", "HEAD"])
    submodules = []
    result = subprocess.run(
        ["git", "submodule", "status", "--recursive"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        for line in result.stdout.splitlines():
            match = re.match(r"^[-+U ]?([0-9a-f]+)\s+([^ ]+)", line)
            if match:
                submodules.append({"path": match.group(2), "revision": match.group(1)})
    return {
        "git_root": str(Path(git_root).resolve()),
        "revision": revision.get("version", ""),
        "submodules": submodules,
    }


def _torch() -> dict[str, object]:
    try:
        import torch
    except Exception as exc:  # Optional for provenance on incomplete environments.
        return {
            "status": "unavailable",
            "error": f"{type(exc).__name__}: {exc}",
            "supported_2_4": None,
        }

    version = str(torch.__version__)
    supported = _supported_torch_version(version)
    return {
        "status": "available",
        "version": version,
        "supported_2_4": supported,
        "cuda": getattr(torch.version, "cuda", None),
    }


def _supported_torch_version(version: str) -> bool:
    return version.split("+", 1)[0] == "2.4.0"


def record(extra_commands: list[str]) -> dict[str, object]:
    commands = {
        "cmake": ["cmake", "--version"],
        "compiler": ["c++", "--version"],
        "vulkaninfo": ["vulkaninfo", "--summary"],
        "glslc": ["glslc", "--version"],
    }
    probes = {name: _command(command) for name, command in commands.items()}
    for command in extra_commands:
        probes[command] = _command([command, "--version"])
    return {
        "schema_version": 1,
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "torch": _torch(),
        "tools": probes,
        "probes": {name: probes[name] for name in extra_commands},
        "repository": _repository(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-command", action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.dumps(record(args.probe_command), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
