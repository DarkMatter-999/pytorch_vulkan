#!/usr/bin/env python3
"""Capture public MSE and connected models in one real execution mode.

Run each mode in a fresh, bounded process owned by the qualification orchestrator.
Publication is a separate CPU-only validation step. Importing this module does
not bootstrap PyTorch or execute a capture.
"""
import argparse
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IDENTITY_PATHS = (
    "tools/vulkan_wrapper_pytest.py", "tools/capture_vulkan_composed_evidence.py",
    "python/pytorch_vulkan/__init__.py", "CMakeLists.txt",
    "tools/run_vulkan_qualification.py", "tools/validate_vulkan_bootstrap_provenance.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture(mode, device):
    if device != "vk:0":
        raise ValueError("composed evidence capture requires device vk:0")
    wrapper = importlib.import_module("pytorch_vulkan")
    if mode not in ("async", "sync") or wrapper._C.execution_mode() != mode:
        raise ValueError("capture execution mode does not match requested mode")
    # Helper imports/reference backward must come AFTER successful bootstrap.
    mse = importlib.import_module("mse_capability_evidence")
    models = importlib.import_module("composed_matmul_evidence")
    return (mse.capture_all_mse_cases(device=device),
            models.capture_composed_evidence(device=device))


def new_attempt(output, root=ROOT):
    output = Path(output).resolve()
    if not output.is_relative_to(root / ".superpowers"):
        raise ValueError("capture output must be a fresh excluded .superpowers attempt")
    relative = output.relative_to(root)
    ignored = subprocess.run(["git", "check-ignore", "-q", str(relative)], cwd=root)
    if ignored.returncode:
        raise ValueError("capture output must be git-excluded")
    if output.exists():
        raise ValueError("capture attempt already exists; refusing overwrite")
    output.mkdir(parents=True, exist_ok=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("async", "sync"), required=True)
    parser.add_argument("--device", choices=("vk:0",), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sys.argv[1:] != ["--mode", args.mode, "--device", args.device,
                        "--output", str(args.output.resolve())]:
        parser.error("use --mode MODE --device vk:0 --output ABSOLUTE_PATH in that order")
    output = new_attempt(args.output)
    sys.path.insert(0, str(ROOT / "tests/python"))
    before = {p: digest((ROOT / p).read_bytes()) for p in IDENTITY_PATHS}
    mse, models = capture(args.mode, args.device)
    wrapper = sys.modules["pytorch_vulkan"]
    if digest(Path(wrapper.__file__).read_bytes()) != before["python/pytorch_vulkan/__init__.py"]:
        raise ValueError("loaded wrapper init does not match capture source identity")
    extension = Path(wrapper._C.__file__).resolve()
    extension_hash = digest(extension.read_bytes())
    runtime_records = [r["mse_autograd_evidence"]["runtime_identity"] for r in mse.values()]
    runtime_records += [r["runtime_identity"] for r in models["records"].values()]
    if any(r["extension_sha256"] != extension_hash or r["execution_mode"] != args.mode
           for r in runtime_records):
        raise ValueError("captured runtime does not match loaded extension/mode")
    after = {p: digest((ROOT / p).read_bytes()) for p in IDENTITY_PATHS}
    if before != after:
        raise ValueError("capture source identity changed during execution")
    outputs = {}
    for name, data in (("mse.json", mse), ("models.json", models)):
        payload = (json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        (output / name).write_bytes(payload)
        outputs[name] = digest(payload)
    # Receipt exists only after both real capture functions returned successfully.
    receipt = {"schema_version": 1, "entry_point": "wrapper-first", "mode": args.mode,
               "command": [sys.executable, *sys.argv],
               "source_identity": after,
               "extension": {"path": str(extension), "sha256": extension_hash},
               "outputs": outputs, "mse_record_ids": sorted(mse),
               "model_record_ids": sorted(models["records"])}
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
