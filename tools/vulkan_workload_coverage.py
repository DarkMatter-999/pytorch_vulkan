"""Capture, merge, and validate source-owned Vulkan workload evidence."""

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
        validate_fragment,
        validate_historical_stage_g_base,
    )
except ImportError:  # pragma: no cover - selected by direct script execution
    from validate_vulkan_workload_contract import (
        REQUIRED_MODES, REQUIRED_WORKLOADS,
        SCHEMA_VERSION, validate_collection, validate_document, validate_record,
        validate_required_matrix, installed_torch_metadata,
        validate_fragment,
        validate_historical_stage_g_base,
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
    if len(async_document.get("records", [])) != len(REQUIRED_WORKLOADS):
        validate_fragment(async_document, "async", [r["workload_id"] for r in async_document.get("records", [])])
    else:
        validate_collection(async_document, "async")
    if len(sync_document.get("records", [])) != len(REQUIRED_WORKLOADS):
        validate_fragment(sync_document, "sync", [r["workload_id"] for r in sync_document.get("records", [])])
    else:
        validate_collection(sync_document, "sync")
    if (async_document["torch_version"], async_document["torch_git_revision"]) != (
        sync_document["torch_version"], sync_document["torch_git_revision"]
    ):
        raise ValueError("mode captures use different PyTorch versions/revisions")
    records = async_document["records"] + sync_document["records"]
    identities = [(r["workload_id"], r["reset_mode"], r["execution_mode"]) for r in records]
    if len(set(identities)) != len(identities):
        raise ValueError("merge fragments contain duplicate workload/mode identities")
    records = sorted(
        records,
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


def _write_collection(path, document, mode, workload_ids=None):
    if workload_ids is None:
        validate_collection(document, mode)
    else:
        validate_fragment(document, mode, workload_ids)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _record(mode, output, workload_ids=None):
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
    selected = list(workload_ids) if workload_ids is not None else [wid for wid, _ in REQUIRED_WORKLOADS]
    known = {scenario.workload_id for scenario in workload.SCENARIOS}
    if not selected or len(set(selected)) != len(selected) or not set(selected) <= known:
        raise ValueError("--workload IDs must be unique source-owned workload IDs")
    records = []
    for workload_id in selected:
        scenario = next(item for item in workload.SCENARIOS if item.workload_id == workload_id)
        reset_mode = scenario.reset_mode
        if workload_id == workload.MATRIX_HVP_ID:
            result = workload.run_vulkan_matrix_hvp()
        elif workload_id == workload.HVP_ID:
            result = workload.run_vulkan_hvp()
        elif workload_id in (workload.MATRIX_NONE_ID, workload.MATRIX_ZERO_ID):
            result = workload.run_vulkan_matrix_sgd(reset_mode)
        else:
            result = workload.run_vulkan_classifier(reset_mode)
        if result["execution_mode"] != mode:
            raise ValueError(f"{workload_id}: runner mode differs from requested {mode}")
        validate_record(result)
        records.append(result)
    document = _document(records)
    validate_fragment(document, mode, selected)
    _write_collection(output, document, mode, selected)
    print(f"captured {len(records)} source-owned workloads in {mode} mode to {output}")


def _merge(async_path, sync_path, output, base_path=None):
    async_doc, sync_doc = _read(async_path), _read(sync_path)
    if base_path is None:
        merged = merge_documents(async_doc, sync_doc)
    else:
        base = _read(base_path)
        validate_historical_stage_g_base(base)
        validate_fragment(async_doc, "async", [r["workload_id"] for r in async_doc["records"]])
        validate_fragment(sync_doc, "sync", [r["workload_id"] for r in sync_doc["records"]])
        records = base["records"] + async_doc["records"] + sync_doc["records"]
        keys = [(r["workload_id"], r["reset_mode"], r["execution_mode"]) for r in records]
        if len(set(keys)) != len(keys):
            raise ValueError("base and fragments contain duplicate workload/mode identities")
        merged = _document(sorted(records, key=lambda r:(r["workload_id"], r["execution_mode"])))
        validate_document(merged)
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
    record.add_argument("--workload", action="append", dest="workload_ids")
    merge = commands.add_parser("merge", help="merge exact async and sync mode captures")
    merge.add_argument("--async", dest="async_path", required=True)
    merge.add_argument("--sync", dest="sync_path", required=True)
    merge.add_argument("--base")
    merge.add_argument("--output", default=str(ROOT / "docs" / "vulkan_workload_coverage.json"))
    validate = commands.add_parser("validate", help="validate the complete source-owned artifact")
    validate.add_argument("path", nargs="?", default=str(ROOT / "docs" / "vulkan_workload_coverage.json"))
    args = parser.parse_args(argv)
    try:
        if args.command == "record":
            _record(args.mode, args.output, args.workload_ids)
        elif args.command == "merge":
            _merge(args.async_path, args.sync_path, args.output, args.base)
        else:
            _validate(args.path)
    except (ValueError, AssertionError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")


if __name__ == "__main__":
    main()
