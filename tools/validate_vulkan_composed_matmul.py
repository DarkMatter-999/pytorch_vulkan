#!/usr/bin/env python3
"""CPU-only connected-model replay; --live explicitly qualifies the loaded runtime."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/python"))
from composed_matmul_evidence import validate_composed_evidence, qualify_composed_current_runtime


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, default=ROOT / "docs/vulkan_composed_matmul_coverage.json")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    document = json.loads(args.coverage.read_text())
    validate_composed_evidence(document, ROOT)
    if args.live:
        qualify_composed_current_runtime(document, ROOT)
    print(f"Connected models: {len(document['records'])} records validated ({'live' if args.live else 'CPU-only historical replay'})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
