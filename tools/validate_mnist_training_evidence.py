#!/usr/bin/env python3
"""CPU-only validator for source-owned MNIST evidence documents."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/python"))
sys.path.insert(0, str(ROOT))
from mnist_training_evidence import _json, validate_current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("document", type=Path)
    parser.add_argument("--receipts", type=Path, action="append", default=[])
    args = parser.parse_args(argv)
    document = _json(args.document)
    validate_current(document, repo_root=ROOT, receipt_roots=args.receipts)
    print("MNIST current identities and available CPU replay validated; see data-unavailable warnings; not new GPU proof")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
