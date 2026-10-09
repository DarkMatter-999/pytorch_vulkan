#!/usr/bin/env python3
"""CPU-only strict wrapper-first provenance and guarded evidence publication.

This is a separate gate, not a substitute for the numerical evidence validators.
No backend bootstrap is imported. Publication validates both actual capture
receipts, semantic evidence and preservation before replacing any published file.
Multi-file rollback handles Python exceptions; this is NOT crash-atomic.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SIDECAR = "docs/vulkan_bootstrap_provenance.json"
IDENTITY_PATHS = (
    "tools/vulkan_wrapper_pytest.py", "tools/capture_vulkan_composed_evidence.py",
    "python/pytorch_vulkan/__init__.py", "CMakeLists.txt",
    "tools/run_vulkan_qualification.py", "tools/validate_vulkan_bootstrap_provenance.py",
)
OUTPUT_PATHS = (
    "docs/vulkan_coverage.json", "docs/vulkan_mse_sync_coverage.json",
    "docs/vulkan_composed_matmul_coverage.json", "docs/vulkan_capabilities.json",
    "docs/vulkan_general_matmul_sync_coverage.json", "docs/vulkan_workload_coverage.json",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def identity(root=ROOT):
    return {p: digest((root / p).read_bytes()) for p in IDENTITY_PATHS}


def mse_ids():
    return sorted(f"mse-ad.{kind}-{reduction}-{selection}.{api}"
                  for kind in ("scalar", "ordinary", "broadcast", "expanded-upstream", "empty")
                  for reduction in ("none", "mean", "sum")
                  for selection in ("input", "target", "both")
                  for api in ("functional", "module"))


def model_ids(mode):
    return sorted(f"g3.{name}.{mode}" for name in (
        "mixed-down-up", "mixed-input-shared", "mse-parameter-hvp", "mutation-down",
        "mutation-input", "mutation-shared", "output-first", "sgd-none", "sgd-zero"))


def fields(data, keys, label):
    if type(data) is not dict or data.keys() != set(keys):
        raise ValueError(label + ": wrong fields")


def check_command(command, mode, root):
    if (type(command) is not list or len(command) != 8
            or any(type(x) is not str or not x for x in command)
            or not Path(command[0]).is_absolute()
            or (root / command[1]).resolve() != (root / "tools/capture_vulkan_composed_evidence.py").resolve()
            or command[2:7] != ["--mode", mode, "--device", "vk:0", "--output"]
            or not Path(command[7]).is_absolute()):
        raise ValueError("wrong capture entrypoint/command/mode")


def validate_document(data, root=ROOT, *, output_root=None):
    output_root = output_root or root
    keys = {"schema_version", "entry_point", "source_identity", "extension", "outputs", "captures"}
    if "artifact_transition" in data:
        keys.add("artifact_transition")
    fields(data, keys, "provenance")
    if type(data["schema_version"]) is not int or data["schema_version"] != 1 or data["entry_point"] != "wrapper-first":
        raise ValueError("wrong provenance schema/entrypoint")
    if data["source_identity"] != identity(root):
        raise ValueError("stale bootstrap source identity")
    fields(data["extension"], {"path", "sha256"}, "extension")
    extension = Path(data["extension"]["path"])
    if not extension.is_absolute() or not extension.is_file() or digest(extension.read_bytes()) != data["extension"]["sha256"]:
        raise ValueError("stale loaded extension identity")
    if data["outputs"] != {p: digest((output_root / p).read_bytes()) for p in OUTPUT_PATHS}:
        raise ValueError("stale/tampered provenance outputs")
    fields(data["captures"], {"async", "sync"}, "captures")
    for mode, capture in data["captures"].items():
        fields(capture, {"command", "mse_record_ids", "model_record_ids"}, "capture")
        check_command(capture["command"], mode, root)
        if capture["mse_record_ids"] != mse_ids() or capture["model_record_ids"] != model_ids(mode):
            raise ValueError("missing/conflicting capture record sets")
    if "artifact_transition" in data:
        transition = data["artifact_transition"]
        validate_transition_document(transition)
        if (transition["new_extension"] != data["extension"]
                or transition["capture_source_identity"] != data["source_identity"]):
            raise ValueError("artifact transition differs from published provenance")
        for mode, capture in data["captures"].items():
            path = Path(capture["command"][-1])
            receipt, _, _ = read_attempt(path, mode, root)
            if (digest((path / "receipt.json").read_bytes()) != transition["receipt_sha256"][mode]
                    or receipt["extension"] != data["extension"]
                    or any(receipt[key] != capture[key] for key in capture)):
                raise ValueError("artifact transition receipt differs from published provenance")
    return data


def load(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"missing/invalid provenance input: {path}") from error


def validate_file(path, root=ROOT, *, output_root=None):
    output_root = output_root or root
    data = validate_document(load(path), root, output_root=output_root)
    # Independently bind the sidecar inventories and loaded extension to ACTUAL
    # published record dictionaries, not only declared record-ID lists.
    for mode, name in (("async", OUTPUT_PATHS[0]), ("sync", OUTPUT_PATHS[1])):
        coverage = load(output_root / name)
        if sorted(n for n in coverage if n.startswith("mse-ad.")) != mse_ids():
            raise ValueError("published MSE inventory differs from provenance")
        for key in mse_ids():
            check_runtime(coverage[key]["mse_autograd_evidence"]["runtime_identity"], mode, data["extension"])
    models = load(output_root / OUTPUT_PATHS[2])["records"]
    if sorted(models) != sorted(model_ids("async") + model_ids("sync")):
        raise ValueError("published model inventory differs from provenance")
    for mode in ("async", "sync"):
        for key in model_ids(mode):
            check_runtime(models[key]["runtime_identity"], mode, data["extension"])
    return data


def check_runtime(runtime, mode, extension):
    if (runtime["execution_mode"] != mode or runtime["extension_sha256"] != extension["sha256"]
            or runtime["extension_path"] != extension["path"]):
        raise ValueError("record runtime differs from capture provenance")


def preserve_measurements(original, fresh, *, extension_transition=None):
    """A startup recapture may change provenance, not historical measurements.

    Unexpected values/counters/routes/runtime hardware changes require human
    investigation, never automatic publication as an import-order repair.
    """
    def measurements(record):
        record = deepcopy(record)
        evidence = record.get("mse_autograd_evidence", record)
        evidence.pop("source_identity", None)
        for key in ("checkout_head", "extension_path"):
            evidence["runtime_identity"].pop(key, None)
        return record
    old, new = measurements(original), measurements(fresh)
    if extension_transition is not None:
        old_runtime = old.get("mse_autograd_evidence", old)["runtime_identity"]
        new_runtime = new.get("mse_autograd_evidence", new)["runtime_identity"]
        if (old_runtime.get("extension_sha256"), new_runtime.get("extension_sha256")) != extension_transition:
            raise ValueError("record artifact differs from approved extension transition")
        # Only a separately authenticated old->new pair can change this one field.
        new_runtime["extension_sha256"] = old_runtime["extension_sha256"]
    if old != new:
        raise ValueError("recapture measurement changed beyond approved provenance")


def publication_root(root):
    """Resolve the explicit trusted authority once, never destination parents."""
    try:
        root = Path(root).resolve(strict=True)
        root_stat = root.lstat()
    except OSError as error:
        raise ValueError("unsafe/missing publication root") from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ValueError("unsafe publication root is not a directory")
    return root, (root_stat.st_dev, root_stat.st_ino)


def publication_paths(root, names, authority):
    """Reject noncanonical names, parent aliases and nonregular existing leaves.

    Root is the already-resolved caller-supplied authority, not implicitly ROOT.
    Do not resolve destination paths: even confined symlink aliases are forbidden.
    This is an exclusive-writer check, not a hostile-race-safe dirfd protocol.
    """
    try:
        root_stat = root.lstat()
        if not stat.S_ISDIR(root_stat.st_mode) or (root_stat.st_dev, root_stat.st_ino) != authority:
            raise ValueError("unsafe publication root authority changed")
        destinations = {}
        for name in names:
            if (type(name) is not str or not name or "\x00" in name
                    or name.startswith("/") or any(p in ("", ".", "..") for p in name.split("/"))):
                raise ValueError("unsafe publication destination name")
            parts = name.split("/")
            parent = root
            for part in parts[:-1]:
                parent = parent / part
                if not stat.S_ISDIR(parent.lstat().st_mode):
                    raise ValueError("unsafe publication destination parent")
            destination = parent / parts[-1]
            try:
                leaf = destination.lstat()
            except FileNotFoundError:
                pass  # A new regular output is allowed, but its parents must exist.
            else:
                if not stat.S_ISREG(leaf.st_mode):
                    raise ValueError("unsafe publication destination leaf")
            destinations[name] = destination
        return destinations
    except OSError as error:
        raise ValueError("unsafe/missing publication destination parent or root") from error


def publish_bytes(root, payloads, validate):
    """Preflight the whole set; validate; stage; replace with guarded rollback.

    Caller supplies a trusted root and must own publication exclusively. Checks
    are repeated before writes but do NOT defeat same-UID hostile concurrent
    filesystem races. No crash-atomic claim. Failed rollback retains recovery
    bytes in the named staging directory, rather than writing through an alias.
    """
    root, authority = publication_root(root)
    destinations = publication_paths(root, payloads, authority)
    validate()
    destinations = publication_paths(root, payloads, authority)
    before = {name: path.read_bytes() if path.exists() else None for name, path in destinations.items()}
    staged = Path(tempfile.mkdtemp(prefix=".composed-publication-", dir=root))
    stage_stat = staged.lstat()
    stage_authority = (stage_stat.st_dev, stage_stat.st_ino)
    retain = False

    def check_stage():
        publication_paths(root, [staged.name + "/recovery.json"], authority)
        current = staged.lstat()
        if (current.st_dev, current.st_ino) != stage_authority:
            raise ValueError("unsafe publication staging authority changed")

    try:
        check_stage()
        originals = {}
        indices = {name: index for index, name in enumerate(payloads)}
        for name, index in indices.items():
            (staged / f"new-{index}").write_bytes(payloads[name])
            originals[name] = None if before[name] is None else f"before-{index}"
            if before[name] is not None:
                (staged / originals[name]).write_bytes(before[name])
        (staged / "recovery.json").write_text(json.dumps(
            {"root": str(root), "originals": originals}, indent=2, sort_keys=True) + "\n")
        changed = []
        try:
            for name, index in indices.items():
                destinations = publication_paths(root, payloads, authority)
                check_stage()
                os.replace(staged / f"new-{index}", destinations[name])
                changed.append(name)
        except BaseException:
            try:
                for name in reversed(changed):
                    destinations = publication_paths(root, payloads, authority)
                    check_stage()
                    if before[name] is None:
                        destinations[name].unlink()
                    else:
                        index = indices[name]
                        backup = staged / f"rollback-{index}"
                        publication_paths(root, [staged.name + "/" + originals[name],
                                                 staged.name + "/" + backup.name], authority)
                        backup.write_bytes((staged / originals[name]).read_bytes())
                        os.replace(backup, destinations[name])
            except BaseException as error:
                retain = True
                raise RuntimeError(f"publication rollback failed; recovery retained at {staged}") from error
            raise
    finally:
        if not retain:
            check_stage()
            shutil.rmtree(staged)


def read_attempt(path, mode, root=ROOT):
    path = Path(path).resolve()
    if not path.is_relative_to(root / ".superpowers"):
        raise ValueError("capture attempt must be excluded within checkout")
    receipt = load(path / "receipt.json")
    fields(receipt, {"schema_version", "entry_point", "mode", "command", "source_identity",
                     "extension", "outputs", "mse_record_ids", "model_record_ids"}, "receipt")
    if (type(receipt["schema_version"]) is not int or receipt["schema_version"] != 1
            or receipt["entry_point"] != "wrapper-first" or receipt["mode"] != mode
            or receipt["source_identity"] != identity(root)):
        raise ValueError("stale/wrong capture receipt")
    check_command(receipt["command"], mode, root)
    if receipt["command"][-1] != str(path):
        raise ValueError("capture command output differs from attempt")
    if receipt["outputs"] != {name: digest((path / name).read_bytes()) for name in ("mse.json", "models.json")}:
        raise ValueError("capture output hash mismatch")
    mse, models = load(path / "mse.json"), load(path / "models.json")
    fields(models, {"schema_version", "records"}, "model capture")
    if type(models["schema_version"]) is not int or models["schema_version"] != 1:
        raise ValueError("wrong model capture schema")
    if (receipt["mse_record_ids"] != mse_ids() or sorted(mse) != mse_ids()
            or receipt["model_record_ids"] != model_ids(mode) or sorted(models["records"]) != model_ids(mode)):
        raise ValueError("capture record inventory mismatch")
    for record in mse.values():
        check_runtime(record["mse_autograd_evidence"]["runtime_identity"], mode, receipt["extension"])
    for record in models["records"].values():
        check_runtime(record["runtime_identity"], mode, receipt["extension"])
    return receipt, mse, models


def read_archive(archive, root):
    """Authenticate exact published baseline bytes and every archived record."""
    manifest = load(archive / "manifest.json")
    originals = {}
    for name in OUTPUT_PATHS:
        short = Path(name).name
        payload = (archive / short).read_bytes()
        entry = manifest["files"][short]
        if digest(payload) != entry["sha256"] or len(payload) != entry["size"]:
            raise ValueError("immutable archive hash/size mismatch")
        if (root / name).read_bytes() != payload:
            raise ValueError("published baseline changed since immutable archive; refusing overwrite")
        originals[name] = load(archive / short)
        records = originals[name].get("records", originals[name])
        if isinstance(records, list):
            records = {str(i): record for i, record in enumerate(records)}
        hashes = {str(key): digest(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
                  for key, record in records.items()}
        if hashes != entry["record_payload_sha256"]:
            raise ValueError("immutable archive record payload hash mismatch")
    return originals


def validate_transition_document(transition):
    fields(transition, {"schema_version", "kind", "baseline_manifest_sha256", "old_extension_sha256",
                        "new_extension", "source_delta", "receipt_sha256", "capture_source_identity"},
           "artifact transition")
    if (type(transition["schema_version"]) is not int or transition["schema_version"] != 1
            or transition["kind"] != "contiguous-bool-transfer"):
        raise ValueError("wrong artifact transition schema/kind")
    fields(transition["new_extension"], {"path", "sha256"}, "transition extension")
    fields(transition["source_delta"], {"src/vulkan_transfer.cpp"}, "approved transfer source delta")
    delta = transition["source_delta"]["src/vulkan_transfer.cpp"]
    fields(delta, {"before", "after"}, "transfer source delta")
    fields(transition["receipt_sha256"], {"async", "sync"}, "transition receipts")
    fields(transition["capture_source_identity"], set(IDENTITY_PATHS), "transition capture controls")
    hashes = [transition["baseline_manifest_sha256"], transition["old_extension_sha256"],
              transition["new_extension"]["sha256"], *delta.values(),
              *transition["receipt_sha256"].values(), *transition["capture_source_identity"].values()]
    if any(type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value) for value in hashes):
        raise ValueError("invalid artifact transition hash")
    path = transition["new_extension"]["path"]
    if (type(path) is not str or not Path(path).is_absolute() or delta["before"] == delta["after"]
            or transition["old_extension_sha256"] == transition["new_extension"]["sha256"]):
        raise ValueError("artifact transition must identify a real source/binary change")


def validate_artifact_transition(transition, archive, originals, receipts, fresh, root):
    """Explicit approval binds one transfer delta to an archive and two real receipts.

    This CPU-only integrity gate does not manufacture GPU execution evidence.
    Semantic validators still independently validate all fresh MSE/model records.
    """
    validate_transition_document(transition)
    if digest((archive / "manifest.json").read_bytes()) != transition["baseline_manifest_sha256"]:
        raise ValueError("artifact transition baseline manifest differs from approval")
    # Recheck actual baseline, including unrelated records; never trust caller copies.
    if json.dumps(read_archive(archive, root), sort_keys=True) != json.dumps(originals, sort_keys=True):
        raise ValueError("artifact transition baseline records differ from archive")
    if transition["capture_source_identity"] != identity(root):
        raise ValueError("artifact transition capture controls are stale")
    extension = transition["new_extension"]
    if digest(Path(extension["path"]).read_bytes()) != extension["sha256"]:
        raise ValueError("artifact transition current extension is stale")
    pair = (transition["old_extension_sha256"], extension["sha256"])
    observed_runtime, actual_sources = None, {}
    for mode in ("async", "sync"):
        receipt = receipts[mode]
        if receipt["extension"] != extension:
            raise ValueError("artifact transition capture modes used different extensions")
        attempt = Path(receipt["command"][-1]).resolve()
        actual_receipt, mse, models = read_attempt(attempt, mode, root)
        if (digest((attempt / "receipt.json").read_bytes()) != transition["receipt_sha256"][mode]
                or actual_receipt != receipt
                or json.dumps((mse, models), sort_keys=True) != json.dumps(fresh[mode], sort_keys=True)):
            raise ValueError("artifact transition receipt/payload differs from approval")
        coverage_name = OUTPUT_PATHS[0 if mode == "async" else 1]
        old_models = originals[OUTPUT_PATHS[2]]["records"]
        if sorted(old_models) != sorted(model_ids("async") + model_ids("sync")):
            raise ValueError("artifact transition baseline model inventory mismatch")
        pairs = [(originals[coverage_name][key], mse[key]) for key in mse_ids()]
        pairs += [(old_models[key], models["records"][key]) for key in model_ids(mode)]
        for original, record in pairs:
            old = original.get("mse_autograd_evidence", original)
            new = record.get("mse_autograd_evidence", record)
            old_sources, new_sources = old["source_identity"], new["source_identity"]
            if old_sources.keys() != new_sources.keys():
                raise ValueError("artifact transition semantic source inventory changed")
            delta = {path: {"before": old_sources[path], "after": value}
                     for path, value in new_sources.items() if old_sources[path] != value}
            if delta != transition["source_delta"]:
                raise ValueError("artifact transition semantic source delta is not the approved transfer change")
            for path, value in new_sources.items():
                if path not in actual_sources:
                    actual_sources[path] = digest((root / path).read_bytes())
                if actual_sources[path] != value:
                    raise ValueError("artifact transition semantic source is stale")
            runtime = {key: value for key, value in new["runtime_identity"].items() if key != "execution_mode"}
            if observed_runtime is not None and observed_runtime != runtime:
                raise ValueError("artifact transition inconsistent current runtime")
            observed_runtime = runtime
            preserve_measurements(original, record, extension_transition=pair)
    return pair


def publish_attempts(async_attempt, sync_attempt, archive, root=ROOT, *, artifact_transition=None):
    """CPU validation only; publish a complete two-mode candidate or nothing."""
    root, authority = publication_root(root)
    publication_paths(root, (*OUTPUT_PATHS, SIDECAR), authority)
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "tests/python"))
    from mse_capability_evidence import validate_mse_evidence
    from composed_matmul_evidence import validate_composed_evidence
    from tools import generate_vulkan_capabilities as generator
    from tools.validate_vulkan_capabilities import validate_manifest_data

    archive = Path(archive).resolve()
    originals = read_archive(archive, root)
    receipts, fresh = {}, {}
    for mode, attempt in (("async", async_attempt), ("sync", sync_attempt)):
        receipt, mse, models = read_attempt(attempt, mode, root)
        validate_mse_evidence(mse, root, require_complete=True)
        receipts[mode], fresh[mode] = receipt, (mse, models)
    if receipts["async"]["extension"] != receipts["sync"]["extension"]:
        raise ValueError("capture modes used different extensions")
    transition, extension_transition = None, None
    if artifact_transition is not None:
        artifact_transition = Path(artifact_transition).resolve()
        if not artifact_transition.is_relative_to(root / ".superpowers"):
            raise ValueError("artifact transition must be retained excluded within checkout")
        transition = load(artifact_transition)
        extension_transition = validate_artifact_transition(transition, archive, originals, receipts, fresh, root)
    candidates = dict(originals)
    for mode, name in (("async", OUTPUT_PATHS[0]), ("sync", OUTPUT_PATHS[1])):
        candidates[name] = {**originals[name], **fresh[mode][0]}
        for key in mse_ids():
            preserve_measurements(originals[name][key], fresh[mode][0][key], extension_transition=extension_transition)
        if any(candidates[name][key] != value for key, value in originals[name].items() if key not in mse_ids()):
            raise ValueError("unaffected coverage payload changed")
        current = load(root / name)
        if any(current.get(key) != value for key, value in originals[name].items() if key not in mse_ids()):
            raise ValueError("unaffected current coverage differs from archive")
    candidates[OUTPUT_PATHS[2]] = {"schema_version": 1, "records": {
        **fresh["async"][1]["records"], **fresh["sync"][1]["records"]}}
    validate_composed_evidence(candidates[OUTPUT_PATHS[2]], root)
    for key, record in candidates[OUTPUT_PATHS[2]]["records"].items():
        preserve_measurements(originals[OUTPUT_PATHS[2]]["records"][key], record, extension_transition=extension_transition)
    capabilities = generator.build_coverage_manifest(candidates[OUTPUT_PATHS[0]])
    payloads = {name: (json.dumps(candidates[name], indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
                for name in OUTPUT_PATHS[:3]}
    payloads[OUTPUT_PATHS[3]] = generator.render(capabilities).encode()
    # Preserve unaffected whole-file bytes, never reserialize them.
    for name in OUTPUT_PATHS[4:]:
        payloads[name] = (archive / Path(name).name).read_bytes()
        if (root / name).read_bytes() != payloads[name]:
            raise ValueError("unaffected published document differs from archive")
    sidecar = {"schema_version": 1, "entry_point": "wrapper-first", "source_identity": identity(root),
               "extension": receipts["async"]["extension"],
               "outputs": {name: digest(payload) for name, payload in payloads.items()},
               "captures": {mode: {key: receipt[key] for key in ("command", "mse_record_ids", "model_record_ids")}
                             for mode, receipt in receipts.items()}}
    if transition is not None:
        sidecar["artifact_transition"] = transition
    payloads[SIDECAR] = (json.dumps(sidecar, indent=2, sort_keys=True) + "\n").encode()
    # Candidate view uses current source, not a copy or relabel of historical maps.
    with tempfile.TemporaryDirectory(prefix="composed-candidate-", dir=root / ".superpowers") as directory:
        candidate_root = Path(directory)
        for directory_name in ("src", "python", "tests", "tools"):
            (candidate_root / directory_name).symlink_to(root / directory_name, target_is_directory=True)
        (candidate_root / "CMakeLists.txt").symlink_to(root / "CMakeLists.txt")
        (candidate_root / "docs").mkdir()
        for path in (root / "docs").iterdir():
            name = "docs/" + path.name
            if name not in payloads:
                (candidate_root / name).symlink_to(path, target_is_directory=path.is_dir())
        for name, payload in payloads.items():
            path = candidate_root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        # Command source paths refer to the actual checkout, not this temporary view.
        def validate():
            if transition is not None:
                if load(artifact_transition) != transition:
                    raise ValueError("artifact transition approval changed before publication")
                validate_artifact_transition(transition, archive, originals, receipts, fresh, root)
            validate_manifest_data(capabilities, candidate_root)
            validate_composed_evidence(candidates[OUTPUT_PATHS[2]], candidate_root)
            validate_file(candidate_root / SIDECAR, root, output_root=candidate_root)
            current_outputs = {p: digest((candidate_root / p).read_bytes()) for p in OUTPUT_PATHS}
            if current_outputs != sidecar["outputs"]:
                raise ValueError("candidate output hash changed")
            if identity(root) != sidecar["source_identity"]:
                raise ValueError("source changed before publication")
            extension = Path(sidecar["extension"]["path"])
            if digest(extension.read_bytes()) != sidecar["extension"]["sha256"]:
                raise ValueError("extension changed before publication")
        publish_bytes(root, payloads, validate)
    return sidecar


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provenance", type=Path, default=ROOT / SIDECAR)
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--async-attempt", type=Path)
    parser.add_argument("--sync-attempt", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--artifact-transition", type=Path,
                        help="explicit approved transfer artifact transition, bound to archive and both receipts")
    args = parser.parse_args()
    try:
        if args.publish:
            if not all((args.async_attempt, args.sync_attempt, args.archive)):
                parser.error("--publish requires both attempts and immutable --archive")
            if args.provenance.resolve() != (ROOT / SIDECAR).resolve():
                parser.error("--publish writes only the canonical provenance sidecar")
            publish_attempts(args.async_attempt, args.sync_attempt, args.archive,
                             artifact_transition=args.artifact_transition)
        else:
            if args.artifact_transition is not None:
                parser.error("--artifact-transition requires --publish")
            validate_file(args.provenance)
    except (ValueError, KeyError, TypeError, OSError) as error:
        print(f"bootstrap provenance failed: {error}", file=sys.stderr)
        return 1
    print("wrapper-first bootstrap provenance validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
