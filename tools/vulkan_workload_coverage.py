"""Capture, merge, and validate Stage G workload execution evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "tests" / "python"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:  # package import in tests, direct script import in the CLI
    from tools.validate_vulkan_workload_contract import (
        REQUIRED_MODES, REQUIRED_WORKLOADS,
        SCHEMA_VERSION, validate_collection, validate_document, validate_record,
        validate_required_matrix, installed_torch_metadata,
    )
except ImportError:  # pragma: no cover - selected by direct script execution
    from validate_vulkan_workload_contract import (
        REQUIRED_MODES, REQUIRED_WORKLOADS,
        SCHEMA_VERSION, validate_collection, validate_document, validate_record,
        validate_required_matrix, installed_torch_metadata,
    )


def _document(records):
    torch_version, torch_git_revision = installed_torch_metadata()
    return {
        "schema_version": SCHEMA_VERSION,
        "torch_version": torch_version,
        "torch_git_revision": torch_git_revision,
        "records": records,
    }


def merge_documents(async_document, sync_document):
    validate_collection(async_document, "async")
    validate_collection(sync_document, "sync")
    if (async_document["torch_version"], async_document["torch_git_revision"]) != (
        sync_document["torch_version"], sync_document["torch_git_revision"]
    ):
        raise ValueError("mode captures use different PyTorch versions/revisions")
    records = sorted(
        async_document["records"] + sync_document["records"],
        key=lambda record: (record["workload_id"], record["execution_mode"]),
    )
    merged = _document(records)
    validate_document(merged)
    return merged


def dumps_document(document):
    validate_document(document)
    return json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _read(path):
    with Path(path).open(encoding="utf-8") as source:
        return json.load(source, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"invalid JSON number {value}")))


def _write(path, document):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(dumps_document(document), encoding="utf-8")


def _write_collection(path, document, mode):
    validate_collection(document, mode)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _record(mode, output):
    if mode not in REQUIRED_MODES:
        raise ValueError("mode must be async or sync")
    installed_torch_metadata()
    import pytorch_vulkan
    actual_mode = pytorch_vulkan._C.execution_mode()
    if actual_mode != mode:
        raise ValueError(f"requested execution mode {mode}, initialized runtime reports {actual_mode}")
    import vulkan_workload_conformance as workload

    validate_required = {(wid, reset) for wid, reset in REQUIRED_WORKLOADS}
    if {(scenario.workload_id, scenario.reset_mode) for scenario in workload.SCENARIOS} != validate_required:
        raise ValueError("source-owned required workload scenario matrix is incomplete")
    records = []
    for workload_id, reset_mode in REQUIRED_WORKLOADS:
        result = (
            workload.run_vulkan_hvp()
            if reset_mode is None
            else workload.run_vulkan_classifier(reset_mode)
        )
        if result["execution_mode"] != mode:
            raise ValueError(f"{workload_id}: runner mode differs from requested {mode}")
        validate_record(result)
        records.append(result)
    document = _document(records)
    validate_collection(document, mode)
    _write_collection(output, document, mode)
    print(f"captured {len(records)} source-owned workloads in {mode} mode to {output}")


def _merge(async_path, sync_path, output):
    merged = merge_documents(_read(async_path), _read(sync_path))
    _write(output, merged)
    print(f"merged {len(merged['records'])} source-owned workload/mode records to {output}")


def _validate(path):
    document = _read(path)
    validate_document(document)
    print(f"validated {len(document['records'])} source-owned workload/mode records in {path}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    record = commands.add_parser("record", help="run and capture one initialized Vulkan mode")
    record.add_argument("--mode", choices=REQUIRED_MODES, required=True)
    record.add_argument("--output", required=True)
    merge = commands.add_parser("merge", help="merge exact async and sync mode captures")
    merge.add_argument("--async", dest="async_path", required=True)
    merge.add_argument("--sync", dest="sync_path", required=True)
    merge.add_argument("--output", default=str(ROOT / "docs" / "vulkan_workload_coverage.json"))
    validate = commands.add_parser("validate", help="validate the complete six-record artifact")
    validate.add_argument("path", nargs="?", default=str(ROOT / "docs" / "vulkan_workload_coverage.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            _record(args.mode, args.output)
        elif args.command == "merge":
            _merge(args.async_path, args.sync_path, args.output)
        else:
            _validate(args.path)
    except (ValueError, AssertionError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
