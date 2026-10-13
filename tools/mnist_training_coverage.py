#!/usr/bin/env python3
"""Guarded MNIST training evidence capture and publication entry point."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/python"))
sys.path.insert(0, str(ROOT))
from mnist_training_evidence import _json, aggregate_attempts, capture, publish, select_execution_mode, validate_current


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cap = commands.add_parser("capture")
    cap.add_argument("--stage", choices=("synthetic", "real"), required=True)
    cap.add_argument("--mode", choices=("async", "sync"), required=True)
    cap.add_argument("--output", type=Path, required=True)
    cap.add_argument("--reference-receipt", type=Path, required=True)
    cap.add_argument("--mnist-root", type=Path)
    val = commands.add_parser("validate")
    val.add_argument("path", type=Path)
    val.add_argument("--receipts", type=Path, action="append", default=[])
    pub = commands.add_parser("publish")
    pub.add_argument("--attempt", type=Path, action="append", required=True)
    pub.add_argument("--archive", type=Path, required=True)
    pub.add_argument("--output", type=Path, required=True)
    agg = commands.add_parser("aggregate", help="validate and combine four captures in memory; emit candidate JSON only")
    agg.add_argument("--attempt", type=Path, action="append", required=True)
    args = parser.parse_args(argv)
    if args.command == "capture":
        if not args.output.is_absolute(): parser.error("--output must be absolute")
        if args.stage == "real" and args.mnist_root is None: parser.error("real capture requires --mnist-root")
        if args.stage == "synthetic" and args.mnist_root is not None: parser.error("synthetic capture rejects --mnist-root")
        select_execution_mode(args.mode)
        capture(args.stage, args.mode, args.mnist_root, args.output,
                reference_receipt=args.reference_receipt)
    elif args.command == "validate":
        document = _json(args.path)
        validate_current(document, repo_root=ROOT, receipt_roots=args.receipts)
    elif args.command == "aggregate":
        if len(args.attempt) != 4: parser.error("aggregate requires four --attempt values")
        document = aggregate_attempts(args.attempt, repo_root=ROOT)
        # stdout is a transient candidate stream, not a publication destination.
        print(json.dumps(document, sort_keys=True, indent=2, allow_nan=False))
    else:
        if len(args.attempt) != 4: parser.error("publish requires four --attempt values")
        publish(args.attempt, args.output, archive=args.archive)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
