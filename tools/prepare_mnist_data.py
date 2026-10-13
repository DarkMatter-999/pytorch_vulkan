#!/usr/bin/env python3
"""Explicitly acquire and validate the canonical MNIST gzip IDX resources."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

from tools.mnist_data import (
    DEFAULT_DATA_ROOT,
    MAX_COMPRESSED_BYTES,
    MNIST_BASE_URL,
    RESOURCE_TABLE,
    _read_resource,
    external_data_root,
    validate_compressed_resource,
)


def _download_and_validate(resource):
    url = MNIST_BASE_URL + resource.filename
    with urlopen(url, timeout=30) as response:
        compressed = response.read(MAX_COMPRESSED_BYTES + 1)
    validate_compressed_resource(compressed, resource)
    return compressed


def prepare(root: Path, *, repo_root: Path) -> dict:
    resolved_root = external_data_root(root, repo_root)
    existing = set()
    for name, resource in RESOURCE_TABLE.items():
        path = resolved_root / name
        if path.exists():
            if not path.is_file():
                raise ValueError(f"MNIST resource path is not a file: {path}")
            # Invalid existing resources are errors, never candidates for repair.
            _read_resource(path, resource)
            existing.add(name)

    pending = {}
    for name, resource in RESOURCE_TABLE.items():
        if name not in existing:
            pending[name] = _download_and_validate(resource)

    if pending:
        resolved_root.mkdir(parents=True, exist_ok=True)
        staged = {}
        try:
            for name, contents in pending.items():
                fd, temporary_name = tempfile.mkstemp(prefix=".mnist-", dir=resolved_root)
                temporary = Path(temporary_name)
                staged[name] = temporary
                with os.fdopen(fd, "wb") as stream:
                    stream.write(contents)
                    stream.flush()
                    os.fsync(stream.fileno())
            for name, temporary in staged.items():
                # Hard-link publication is atomic and refuses to overwrite a file
                # that appeared after the initial existence check.
                os.link(temporary, resolved_root / name)
        finally:
            for temporary in staged.values():
                temporary.unlink(missing_ok=True)

    identities = {}
    for name, resource in RESOURCE_TABLE.items():
        path = resolved_root / name
        _, identity = _read_resource(path, resource)
        identities[name] = identity
    return {
        "dataset": "MNIST",
        "root": str(resolved_root),
        "resources": identities,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    result = prepare(args.root, repo_root=Path(__file__).resolve().parents[1])
    for name, identity in result["resources"].items():
        print(f"{name}: md5={identity['md5']} sha256={identity['sha256']}")


if __name__ == "__main__":
    main()
