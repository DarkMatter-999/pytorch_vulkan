#!/usr/bin/env python3
"""Capture and source-validate only the finite stock Linear route records."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests/python"))

from stock_linear_route_evidence import capture_all_stock_linear_routes  # noqa: E402
from tools import generate_vulkan_capabilities as generator  # noqa: E402
from tools.validate_vulkan_capabilities import validate_stock_linear_route_evidence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="vk:0")
    parser.add_argument("--coverage", type=Path, default=ROOT / "docs/vulkan_coverage.json")
    parser.add_argument("--manifest", type=Path, default=ROOT / "docs/vulkan_capabilities.json")
    args = parser.parse_args()

    coverage = json.loads(args.coverage.read_text())
    records = capture_all_stock_linear_routes(args.device)
    existing_route_ids = {name for name in coverage if name.startswith("linear.rank")}
    for name in existing_route_ids - set(records):
        del coverage[name]
    coverage.update(records)
    validate_stock_linear_route_evidence(coverage)
    manifest_text = generator.render(generator.build_coverage_manifest(coverage))
    coverage_text = json.dumps(coverage, indent=2, sort_keys=True) + "\n"

    for output, text in ((args.coverage, coverage_text), (args.manifest, manifest_text)):
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(text)
        os.replace(temporary, output)
    print(f"captured and validated {len(records)} stock Linear route records")


if __name__ == "__main__":
    main()
