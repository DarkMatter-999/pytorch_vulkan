"""CPU-only fixtures. No fixture below is a genuine GPU capture/publication."""
from copy import deepcopy
import hashlib
import json
import importlib.machinery
import os
from pathlib import Path
import sys

import pytest
import torch
from torch.utils.data import TensorDataset

import mnist_training_evidence as evidence
from mnist_training_cases import make_model, synthetic_batch, train
from mnist_training_parity import TrainingObserver, _observer_result, _real_observers
from tools.mnist_data import RESOURCE_TABLE, mnist_loader


def _fixture_runtime(mode):
    return {"pytorch_version": str(torch.__version__), "pytorch_git_revision": torch.version.git_version,
            "checkout_head": "0" * 40, "extension_sha256": "1" * 64,
            "extension_path": "/CPU-FIXTURE/_C.so", "device": "vk:0",
            "hardware": "CPU fixture, NOT GPU evidence", "driver": "CPU fixture",
            "vulkan_api_version": "1.1", "vulkan_instance_version": "1.1", "execution_mode": mode}


def _fixture_data():
    resources = {name: {"md5": item.md5, "sha256": "b" * 64, "compressed_bytes": 1}
                 for name, item in RESOURCE_TABLE.items()}
    return {split: {"dataset": "MNIST", "split": split, "count": count,
                    "indices": list(range(count)),
                    "label_histogram": {str(i): sum(j % 10 == i for j in range(count)) for i in range(10)},
                    "input_digest_sha256": "5" * 64,
                    "data_identity": {"canonical_resource_table": "torchvision-v0.19.0-MNIST",
                                      "resources": resources}}
            for split, count in (("train", 257), ("test", 129))}


def _vk_fixture_tree(value):
    """TEST ONLY: project CPU observer output into labelled Vulkan-schema fixtures."""
    if isinstance(value, dict):
        if "device" in value:
            return {**value, "device": "vk:0"}
        return {k: _vk_fixture_tree(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_vk_fixture_tree(v) for v in value]
    return value


@pytest.fixture(scope="module")
def produced():
    # Real recipe/observer producers, not validator-derived metadata or phase order.
    pairs = {}
    for sid in evidence.SCENARIOS:
        real = sid.startswith("mnist.real.")
        batch = 16 if real else int(sid.split("batch")[1].split(".")[0])
        none = sid.endswith("sgd-none")
        observer = TrainingObserver("cpu")
        with observer.phase("setup.model-staging"):
            model = make_model()
        optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
        if real:
            def loader(count):
                x = torch.zeros(count, 1, 28, 28)
                y = torch.arange(count, dtype=torch.int64) % 10
                return mnist_loader(TensorDataset(x, y))
            observer, sizes, eval_sizes, metrics = _real_observers(
                model, optimizer, "cpu", None, loader(257), loader(129), none, observer=observer)
        else:
            train(model, optimizer, lambda: [synthetic_batch(batch)] * 3, device="cpu",
                  epochs=1, set_to_none=none, observer=observer)
        cpu = _observer_result(observer)
        vk = _vk_fixture_tree(cpu)
        vk["counters"] = {name: ([1, 0, 0, 0] if route == "compute" else [0, 0, 0, 0])
                          for name, route in vk["phase_routes"].items()}
        pair = {"updates": 34 if real else 3, "batch_size": batch, "set_to_none": none,
                "parameter_names": list(evidence.PARAMETERS), "cpu": cpu, "vulkan": vk,
                "diagnostic_snapshots": {"cpu": observer.snapshots, "vulkan": observer.snapshots}}
        if real:
            pair.update({"train_batch_sizes": sizes, "evaluation_batch_sizes": [eval_sizes[:9], eval_sizes[9:]],
                         "data_identity": _fixture_data(), "metric_device": "cpu", "evaluations": [
                             {"epoch": epoch, "cpu": item, "vulkan": item,
                              "comparison": {"passed": True, **{k: item[k] for k in ("correct", "count", "accuracy")}}}
                             for epoch, item in enumerate(metrics)]})
        diagnostics = {side: {event: evidence._tree(value, pair[side]["tensor_metadata"][event])
                              for event, value in observer.snapshots.items()} for side in ("cpu", "vulkan")}
        pairs[sid] = (pair, diagnostics)
    return pairs


@pytest.fixture(scope="module")
def records(produced):
    reference = evidence._resolve_reference(evidence.REFERENCE)
    output = []
    for sid in evidence.SCENARIOS:
        for mode in evidence.MODES:
            pair, diagnostics = produced[sid]
            record = evidence._record_from_pair(sid, mode, pair, _fixture_runtime(mode),
                                                {"CPU-FIXTURE.py": "2" * 64}, reference, diagnostics)
            record["raw_receipt"] = {"attempt": f"/CPU-FIXTURE/{record['recipe']['kind']}-{mode}",
                                     "file": "capture.json", "record_id": sid}
            output.append(record)
    return output


def _document(records):
    return {"schema_version": 1, "pytorch_version": str(torch.__version__),
            "pytorch_git_revision": torch.version.git_version, "records": records}


def _fresh_process_modules(monkeypatch):
    # These CPU bootstrap unit tests model a fresh *child*, even if an outer
    # wrapper-first pytest process already imported its own wrapper. Restore
    # every entry afterward; never execute or reset that outer backend.
    for name in list(sys.modules):
        if name == "pytorch_vulkan" or name.startswith("pytorch_vulkan."):
            monkeypatch.delitem(sys.modules, name)


def test_genuine_cpu_producer_serializer_schema(records):
    evidence.validate_structure(_document(records))
    update = records[0]["steps"][0]["cpu"]["update"]
    assert "optimizer_options" in update
    assert update["parameters"]["1.weight"]["requires_grad"] is True


def test_real_evaluation_phase_order_matches_actual_cpu_recipe(produced):
    pair, _ = produced["mnist.real.subset257-129.sgd-none"]
    assert evidence._expected_phases(True, 34) == pair["cpu"]["phase_order"]


def test_near_zero_boundary_has_no_relative_floor():
    cpu = {"shape": [1], "dtype": "torch.float32", "values": [0.0]}
    evidence._compare_tensor_pair(cpu, {**cpu, "values": [2e-4]}, "logits")
    with pytest.raises(ValueError, match="tolerance"):
        evidence._compare_tensor_pair(cpu, {**cpu, "values": [2e-4 + 1e-8]}, "logits")


def test_relative_error_reporting_clamps_only_its_denominator():
    cpu = {"shape": [1], "dtype": "torch.float32", "values": [0.0]}
    vk = {**cpu, "values": [2e-4]}
    summary = evidence._numeric_parity({"event": {"logits": cpu}}, {"event": {"logits": vk}})
    assert summary["errors"]["event.logits"] == {
        "maximum_absolute_error": 2e-4, "maximum_relative_error": 1.0}
    with pytest.raises(ValueError, match="tolerance"):
        evidence._numeric_parity({"event": {"logits": cpu}},
                                 {"event": {"logits": {**vk, "values": [2e-4 + 1e-8]}}})


@pytest.mark.parametrize("mutate", [
    lambda r: r["source_snapshots"]["aten/src/ATen/native/LossNLL.cpp"].update(line_references={}),
    lambda r: r["installed"]["schemas"].update({"aten::addmm": "unavailable: fixture"}),
    lambda r: r["installed"]["dispatch_tables"].pop("aten::relu"),
    lambda r: r["optimizer_defaults"].update(momentum=0),
    lambda r: r["profiler_phase_operators_and_shapes"].pop("backward"),
    lambda r: r["cpu_comparisons"]["1"].update(finite=False),
    lambda r: r["numerical_rationale"]["training_parity_ceilings"]["parameters"].update(atol=.01),
])
def test_reference_semantic_closure_is_fail_closed(tmp_path, mutate):
    receipt = json.loads(evidence.REFERENCE.read_text())
    mutate(receipt)
    path = tmp_path / "CPU-FIXTURE-reference.json"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        evidence.validate_reference_receipt(path)


@pytest.mark.parametrize("section, mutate", [
    ("phase_counters", lambda r: r["cpu"]["tensor_metadata"].update(images=[1, 2])),
    ("phase_counters", lambda r: r["cpu"]["counter_categories"]["compute"].clear()),
    ("parity", lambda r: r.update(images=[[0] * 28])),
    ("parity", lambda r: r.update(comparisons="not a list")),
])
def test_nested_metadata_and_parity_payloads_rejected(records, section, mutate):
    changed = dict(records[0])
    changed[section] = deepcopy(changed[section])
    mutate(changed[section])
    with pytest.raises(ValueError):
        evidence.validate_record(changed)


@pytest.mark.parametrize("mutate", [
    lambda item: item["cpu"].update(correct=True, accuracy=1 / 129),
    lambda item: item["comparison"].update(passed=False),
    lambda item: (item["cpu"].update(predictions=[9] * 129), item["vulkan"].update(predictions=[9] * 129)),
])
def test_real_metrics_are_derived_not_self_asserted(records, mutate):
    changed = dict(records[-1])
    changed["evaluation"] = deepcopy(changed["evaluation"])
    mutate(changed["evaluation"][0])
    with pytest.raises(ValueError):
        evidence.validate_record(changed)


def test_error_summary_covers_every_primary_tensor_and_is_recomputed(records):
    parity = records[0]["parity"]
    assert set(parity) == {"tolerances", "errors"}
    assert "step-0.backward.loss" in parity["errors"]
    assert parity["errors"]["step-0.backward.loss"] == {
        "maximum_absolute_error": 0.0, "maximum_relative_error": 0.0}
    changed = dict(records[0])
    changed["parity"] = deepcopy(parity)
    changed["parity"]["errors"]["step-0.backward.loss"]["maximum_absolute_error"] = .01
    with pytest.raises(ValueError):
        evidence.validate_record(changed)


@pytest.fixture(scope="module")
def filesystem_fixture(tmp_path_factory, records):
    """Authenticated *filesystem* fixture, NOT a GPU receipt or a real binary.

    Real source inventory/reference validator/PathFinder/hash checks run against
    controlled files. The fake extension is never imported; no authentication
    function is mocked. Actual CPU synthetic snapshots are used for replay.
    """
    root = tmp_path_factory.mktemp("CPU-FIXTURE-repository")
    for name in evidence.source_paths():
        target = root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((evidence.ROOT / name).read_bytes())
    # Real git exclusion is tested without a commit or inherited repository work.
    import subprocess
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / ".git/info/exclude").write_text(".superpowers/\n")
    package = root / "build/pytorch_vulkan"
    package.mkdir(parents=True)
    (package / "__init__.py").write_bytes((root / "python/pytorch_vulkan/__init__.py").read_bytes())
    for name in ("compiler.py", "compiler_metadata.py", "serialization.py"):
        (package / name).write_bytes((root / "python/pytorch_vulkan" / name).read_bytes())
    extension = package / ("_C" + importlib.machinery.EXTENSION_SUFFIXES[0])
    extension.write_bytes(b"CPU TEST FIXTURE: NOT A VULKAN BINARY\n")
    sources = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in evidence.source_paths()}
    attempts = []
    for stage in ("synthetic", "real"):
        for mode in evidence.MODES:
            attempt = root / ".superpowers" / f"{stage}-{mode}"
            attempt.mkdir(parents=True)
            selected = []
            for record in records:
                if record["recipe"]["kind"] != stage or record["execution_mode"] != mode:
                    continue
                record = {**record, "runtime": {**record["runtime"], "extension_path": str(extension),
                          "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()},
                          "source_identity": sources,
                          "raw_receipt": {"attempt": str(attempt), "file": "capture.json", "record_id": record["scenario_id"]}}
                selected.append(record)
            reference = selected[0]["reference_receipt"]
            external = tmp_path_factory.getbasetemp() / "explicitly-UNAVAILABLE-mnist"
            command = [sys.executable, str(root / "tools/mnist_training_coverage.py"), "capture",
                       "--stage", stage, "--mode", mode, "--output", str(attempt),
                       "--reference-receipt", reference["path"]]
            if stage == "real":
                command.extend(["--mnist-root", str(external)])
            environment = {"PYTORCH_VULKAN_ASYNC_EXECUTION": "1" if mode == "async" else "0",
                           "PYTHONPATH": os.pathsep.join(str(root / p) for p in (".", "build", "tests/python")),
                           "VK_INSTANCE_LAYERS": "VK_LAYER_KHRONOS_validation",
                           "PYTORCH_VULKAN_MNIST_ROOT": str(external) if stage == "real" else None}
            fragment = {**_document(selected), "stage": stage, "execution_mode": mode}
            raw = json.dumps(fragment, separators=(",", ":"), allow_nan=False).encode()
            (attempt / "capture.json").write_bytes(raw)
            receipt = {"schema_version": 1, "complete": True, "stage": stage, "mode": mode,
                       "command": command, "working_directory": str(root), "environment": environment,
                       "source_identity": sources, "runtime": selected[0]["runtime"],
                       "reference_receipt": reference,
                       "outputs": {"capture.json": {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}}}
            (attempt / "receipt.json").write_text(json.dumps(receipt))
            attempts.append(attempt)
    return root, attempts, extension


def test_publisher_cannot_reach_archive_with_stale_sources(filesystem_fixture, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    changed = root / "tools/mnist_training_coverage.py"
    before = changed.read_bytes()
    changed.write_bytes(before + b"\n# stale semantic control\n")
    output = root / "CPU-FIXTURE-output.json"
    output.write_bytes(b"original bytes\n")
    archive = root / ".superpowers" / "must-not-exist"
    try:
        with pytest.raises(ValueError, match="source"):
            evidence.publish(attempts, output, archive=archive)
        assert not archive.exists()
        assert output.read_bytes() == b"original bytes\n"
    finally:
        changed.write_bytes(before)


@pytest.mark.parametrize("mutation", [
    lambda r: r.update(command=["python", "capture"]),
    lambda r: r["environment"].update(PYTORCH_VULKAN_ASYNC_EXECUTION="0"),
    lambda r: r["outputs"].update({"extra.json": {"sha256": "0" * 64, "size": 1}}),
])
def test_completed_attempt_exact_provenance(filesystem_fixture, monkeypatch, mutation):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    path = attempts[0] / "receipt.json"
    original = path.read_bytes()
    receipt = json.loads(original)
    mutation(receipt)
    path.write_text(json.dumps(receipt))
    try:
        with pytest.raises(ValueError):
            evidence._validate_attempt(attempts[0])
    finally:
        path.write_bytes(original)


@pytest.mark.parametrize("mode, selected", [("async", "1"), ("sync", "0")])
def test_fresh_cli_selects_mode_before_capture(monkeypatch, mode, selected):
    from tools import mnist_training_coverage as cli
    _fresh_process_modules(monkeypatch)
    monkeypatch.delenv("PYTORCH_VULKAN_ASYNC_EXECUTION", raising=False)
    calls = []
    monkeypatch.setattr(cli, "capture", lambda *args, **kwargs: calls.append(os.environ.get("PYTORCH_VULKAN_ASYNC_EXECUTION")))
    cli.main(["capture", "--stage", "synthetic", "--mode", mode,
              "--output", "/CPU-FIXTURE/attempt", "--reference-receipt", str(evidence.REFERENCE)])
    assert calls == [selected]


def test_replay_rejects_matching_but_fabricated_cpu_and_vk_trajectory(records):
    changed = deepcopy(records[0])
    for side in ("cpu", "vulkan"):
        changed["steps"][0][side]["backward"]["loss"]["values"][0] += .01
    # Equal forged arrays can pass structural parity; replay must reject them.
    with pytest.raises(ValueError, match="CPU replay"):
        evidence.replay_cpu_reference(_document([changed]), real_root=None, repo_root=evidence.ROOT)


def test_replay_labels_missing_real_data_without_downloading(records, tmp_path):
    document = _document([records[-1]])
    with pytest.warns(RuntimeWarning, match="data-unavailable"):
        status = evidence.replay_cpu_reference(document, real_root=tmp_path / "absent", repo_root=evidence.ROOT)
    assert status[records[-1]["scenario_id"]] == "data-unavailable"
    assert not (tmp_path / "absent").exists()


def test_real_current_authentication_uses_files_and_replay_without_wrapper(filesystem_fixture, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    imported_before = "pytorch_vulkan" in sys.modules
    before = {(p, name): hashlib.sha256((p / name).read_bytes()).hexdigest()
              for p in attempts for name in ("capture.json", "receipt.json")}
    with pytest.warns(RuntimeWarning, match="data-unavailable"):
        document = evidence.aggregate_attempts(attempts, repo_root=root)
    with pytest.warns(RuntimeWarning, match="data-unavailable"):
        evidence.validate_current(document, repo_root=root, receipt_roots=attempts)
    assert ("pytorch_vulkan" in sys.modules) is imported_before
    assert len(document["records"]) == 16
    assert before == {(p, name): hashlib.sha256((p / name).read_bytes()).hexdigest()
                      for p in attempts for name in ("capture.json", "receipt.json")}


def test_current_rejects_changed_selected_binary(filesystem_fixture, monkeypatch):
    root, attempts, extension = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    original = extension.read_bytes()
    extension.write_bytes(original + b"tampered")
    try:
        with pytest.raises(ValueError, match="extension SHA256"):
            evidence.aggregate_attempts(attempts, repo_root=root)
    finally:
        extension.write_bytes(original)


def test_raw_locator_is_rejected_never_repaired(filesystem_fixture, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    fragment_path, receipt_path = attempts[0] / "capture.json", attempts[0] / "receipt.json"
    raw, receipt_raw = fragment_path.read_bytes(), receipt_path.read_bytes()
    fragment = json.loads(raw)
    fragment["records"][0]["raw_receipt"]["attempt"] = str(attempts[1])
    edited = json.dumps(fragment).encode()
    fragment_path.write_bytes(edited)
    receipt = json.loads(receipt_raw)
    receipt["outputs"]["capture.json"] = {"sha256": hashlib.sha256(edited).hexdigest(), "size": len(edited)}
    receipt_path.write_text(json.dumps(receipt))
    try:
        with pytest.raises(ValueError, match="immutable raw receipt locator"):
            evidence._validate_attempt(attempts[0])
        assert fragment_path.read_bytes() == edited
    finally:
        fragment_path.write_bytes(raw)
        receipt_path.write_bytes(receipt_raw)


def test_writer_contention_blocks_before_authentication(filesystem_fixture, monkeypatch):
    import fcntl
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    output = root / "CPU-FIXTURE-lock-test.json"
    archive = root / ".superpowers" / "CPU-FIXTURE-lock-archive"
    with output.with_name(output.name + ".lock").open("a+b") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another publisher"):
            evidence.publish(attempts, output, archive=archive)
    assert not archive.exists()
    assert not output.exists()


def test_publication_rollback_uses_explicit_mocked_authentication_boundary(filesystem_fixture, records, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    # This test proves replacement/rollback ONLY. Actual authentication is tested
    # above against real control/reference/binary bytes; never claim fixture GPU proof.
    monkeypatch.setattr(evidence, "aggregate_attempts", lambda *args, **kwargs: _document([]))
    monkeypatch.setattr(evidence, "_publication_recheck", lambda *args: None)
    output = root / "CPU-FIXTURE-rollback.json"
    original = b'{ "preserve": "exact bytes" }\n'
    output.write_bytes(original)
    archive = root / ".superpowers" / "CPU-FIXTURE-rollback-archive"
    replace = evidence.os.replace
    calls = []

    def injected(source, target):
        calls.append(str(source))
        if len(calls) == 1:
            # Simulate a caught error *after* replacement, not merely before it.
            replace(source, target)
            raise OSError("CPU fixture injected post-replacement failure")
        return replace(source, target)

    monkeypatch.setattr(evidence.os, "replace", injected)
    with pytest.raises(OSError, match="injected"):
        evidence.publish(attempts, output, archive=archive)
    assert output.read_bytes() == original
    assert (archive / "baseline.bin").read_bytes() == original
    manifest = json.loads((archive / "archive.json").read_text())
    assert manifest["baseline"] == {"present": True, "sha256": hashlib.sha256(original).hexdigest(), "size": len(original)}
    assert (archive / "baseline.bin").stat().st_mode & 0o222 == 0


def test_first_fixture_publication_archives_explicit_absence(filesystem_fixture, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    # Replacement mechanics only, explicit authenticated-assembly test boundary.
    monkeypatch.setattr(evidence, "aggregate_attempts", lambda *args, **kwargs: _document([]))
    monkeypatch.setattr(evidence, "_publication_recheck", lambda *args: None)
    output = root / "CPU-FIXTURE-first-publication.json"
    archive = root / ".superpowers" / "CPU-FIXTURE-first-archive"
    evidence.publish(attempts, output, archive=archive)
    assert json.loads((archive / "archive.json").read_text())["baseline"] == {
        "present": False, "sha256": None, "size": None}
    assert not (archive / "baseline.bin").exists()
    assert archive.stat().st_mode & 0o222 == 0


@pytest.mark.parametrize("inherited", ["0", "true", "garbage"])
def test_conflicting_mode_is_rejected_before_bootstrap(monkeypatch, inherited):
    _fresh_process_modules(monkeypatch)
    monkeypatch.setenv("PYTORCH_VULKAN_ASYNC_EXECUTION", inherited)
    with pytest.raises(ValueError, match="conflicts"):
        evidence.select_execution_mode("async")


def test_capture_completion_waits_for_fragment_validation(filesystem_fixture, produced, monkeypatch):
    import types
    import vector_matmul_capability_evidence as runtime_helper
    import mnist_training_parity as parity

    root, _, extension = filesystem_fixture
    attempt = root / ".superpowers" / "CPU-FIXTURE-rejected-capture"
    monkeypatch.setattr(evidence, "ROOT", root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _fresh_process_modules(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", evidence._pythonpath(root))
    monkeypatch.setenv("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
    monkeypatch.delenv("PYTORCH_VULKAN_ASYNC_EXECUTION", raising=False)
    monkeypatch.delenv("PYTORCH_VULKAN_MNIST_ROOT", raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", [str(root / "tools/mnist_training_coverage.py"), "capture",
                        "--stage", "synthetic", "--mode", "async", "--output", str(attempt),
                        "--reference-receipt", str(evidence.REFERENCE)])
    wrapper = types.SimpleNamespace(_C=types.SimpleNamespace(execution_mode=lambda: "async", __file__=str(extension)))
    original_import = evidence.importlib.import_module
    monkeypatch.setattr(evidence.importlib, "import_module", lambda name, *a, **k:
                        wrapper if name == "pytorch_vulkan" else original_import(name, *a, **k))
    runtime = {**_fixture_runtime("async"), "extension_path": str(extension),
               "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}
    monkeypatch.setattr(runtime_helper, "runtime_identity", lambda *a: runtime)
    monkeypatch.setattr(parity, "run_synthetic_pair", lambda batch, none, **kwargs:
                        produced[f"mnist.synthetic.batch{batch}.sgd-{'none' if none else 'zero'}"][0])

    def invalid(fragment):
        raise ValueError("CPU fixture injected fragment rejection")

    monkeypatch.setattr(evidence, "validate_fragment", invalid)
    with pytest.raises(ValueError, match="fragment rejection"):
        evidence.capture("synthetic", "async", None, attempt, reference_receipt=evidence.REFERENCE)
    assert attempt.is_dir()  # retained partial attempt
    assert not (attempt / "receipt.json").exists()
    assert not (attempt / "capture.json").exists()


def test_capture_serializes_actual_installed_torch_version(filesystem_fixture, produced, monkeypatch):
    """CPU control regression: real version/recipe values, injected GPU boundaries."""
    import types
    import vector_matmul_capability_evidence as runtime_helper
    import mnist_training_parity as parity

    assert isinstance(torch.__version__, str)
    assert type(torch.__version__) is not str  # exercise installed TorchVersion, not a plain-string mock
    root, _, extension = filesystem_fixture
    attempt = root / ".superpowers" / "CPU-FIXTURE-installed-version-capture"
    monkeypatch.setattr(evidence, "ROOT", root)
    monkeypatch.setattr(sys, "path", list(sys.path))
    _fresh_process_modules(monkeypatch)
    monkeypatch.setenv("PYTHONPATH", evidence._pythonpath(root))
    monkeypatch.setenv("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation")
    monkeypatch.delenv("PYTORCH_VULKAN_ASYNC_EXECUTION", raising=False)
    monkeypatch.delenv("PYTORCH_VULKAN_MNIST_ROOT", raising=False)
    monkeypatch.chdir(root)
    monkeypatch.setattr(sys, "argv", [str(root / "tools/mnist_training_coverage.py"), "capture",
                        "--stage", "synthetic", "--mode", "async", "--output", str(attempt),
                        "--reference-receipt", str(evidence.REFERENCE)])
    wrapper = types.SimpleNamespace(_C=types.SimpleNamespace(execution_mode=lambda: "async", __file__=str(extension)))
    original_import = evidence.importlib.import_module
    monkeypatch.setattr(evidence.importlib, "import_module", lambda name, *a, **k:
                        wrapper if name == "pytorch_vulkan" else original_import(name, *a, **k))
    runtime = {**_fixture_runtime("async"), "pytorch_version": torch.__version__,
               "extension_path": str(extension),
               "extension_sha256": hashlib.sha256(extension.read_bytes()).hexdigest()}
    monkeypatch.setattr(runtime_helper, "runtime_identity", lambda *a: runtime)
    monkeypatch.setattr(parity, "run_synthetic_pair", lambda batch, none, **kwargs:
                        produced[f"mnist.synthetic.batch{batch}.sgd-{'none' if none else 'zero'}"][0])

    # Keep actual fragment/type validation and completion serialization active.
    fragment = evidence.capture("synthetic", "async", None, attempt, reference_receipt=evidence.REFERENCE)
    assert type(fragment["pytorch_version"]) is str
    assert fragment["pytorch_version"] == str(torch.__version__)
    assert len(fragment["records"]) == 6
    for record in fragment["records"]:
        assert type(record["runtime"]["pytorch_version"]) is str
        assert type(record["reference_receipt"]["pytorch_version"]) is str
        assert record["runtime"]["pytorch_version"] == str(torch.__version__)
    assert runtime["pytorch_version"] is torch.__version__  # producer metadata is not mutated
    receipt = json.loads((attempt / "receipt.json").read_text())
    assert receipt["complete"] is True
    assert receipt["runtime"]["pytorch_version"] == str(torch.__version__)
    assert json.loads((attempt / "capture.json").read_text()) == fragment
    with pytest.raises(ValueError, match="PyTorch version: wrong type"):
        evidence.validate_fragment({**fragment, "pytorch_version": torch.__version__})


@pytest.mark.parametrize("mutate", [
    lambda r: r["steps"][0]["vulkan"]["update"]["state"]["1.weight"].clear(),
    lambda r: r["steps"][0]["vulkan"]["backward"]["loss"].update(values=[float("inf")]),
    lambda r: r["steps"][0]["vulkan"]["backward"]["logits"].update(device="cpu"),
    lambda r: r["steps"][0]["vulkan"]["backward"]["loss"].update(history=False),
    lambda r: r["steps"][0]["vulkan"]["update"]["optimizer_options"].update(foreach=True),
    lambda r: r["steps"][0]["vulkan"]["update"]["parameters"]["1.bias"].update(requires_grad=False),
    lambda r: r["steps"][1]["vulkan"]["reset"]["retained_objects"].update({"1.weight": True}),
    lambda r: r["phase_counters"]["vulkan"]["counters"].pop("setup.model-staging"),
    lambda r: r["phase_counters"]["vulkan"]["counters"].update({"step-0.update": [1, 0, 1, 0]}),
    lambda r: r["phase_counters"]["vulkan"]["counters"].update({"step-0.update": [True, 0, 0, 0]}),
    lambda r: r["phase_counters"]["vulkan"]["counters"].update({"setup.model-staging": [0, 0, 1, 1]}),
    lambda r: r["recipe"].update(model_seed=12013),
    lambda r: r["recipe"]["optimizer"].update(momentum=.0),
    lambda r: r["recipe"]["train_batch_sizes"].pop(),
    lambda r: r["parity"]["tolerances"]["logits"].update(atol=.003),
    lambda r: r["parity"]["errors"].pop("step-0.backward.loss"),
    lambda r: r["reference_receipt"]["source_snapshots"].clear(),
    lambda r: r["source_identity"].update({"CPU-FIXTURE.py": "z" * 64}),
])
def test_strict_record_negative_contracts(records, mutate):
    changed = deepcopy(records[0])
    mutate(changed)
    with pytest.raises(ValueError):
        evidence.validate_record(changed)


@pytest.mark.parametrize("kind", ["missing", "duplicate", "missing-mode"])
def test_complete_matrix_rejects_fragments_and_duplicates(records, kind):
    if kind == "missing":
        selected = records[:-1]
    elif kind == "duplicate":
        selected = [*records, records[0]]
    else:
        selected = [r for r in records if r["execution_mode"] == "async"]
    with pytest.raises(ValueError, match="matrix"):
        evidence.validate_structure(_document(selected))


def test_source_owned_inventory_covers_semantics_and_new_controls():
    from composed_evidence_common import DEPENDENCIES
    inventory = set(evidence.source_paths())
    assert set(DEPENDENCIES) <= inventory
    required = {"tools/mnist_training_coverage.py", "tools/validate_mnist_training_evidence.py",
                "tests/python/test_mnist_training_evidence.py", "tests/python/mnist_training_evidence.py",
                "tests/python/mnist_training_parity.py", "tests/python/mnist_training_cases.py",
                "tools/probe_mnist_training_reference.py", "tools/mnist_data.py", "CMakeLists.txt",
                "src/vulkan/shaders/generated/linear_relu_backward_spv.h",
                "src/vulkan/shaders/generated/linear_relu_backward_spv.sha256",
                "src/vulkan/shaders/generated/operator_spv.h", "src/vulkan/shaders/generated/operator_spv.sha256",
                "src/vulkan/descriptor_arena.cpp", "src/vulkan/pipeline_cache.cpp",
                "python/pytorch_vulkan/compiler.py", "python/pytorch_vulkan/compiler_metadata.py",
                "python/pytorch_vulkan/serialization.py",
                "tools/run_vulkan_qualification.py", "tools/qualification_process_supervisor.py",
                "tests/python/test_qualification_process_supervisor.py",
                "tests/python/test_vulkan_qualification.py"}
    assert required <= inventory
    assert all((evidence.ROOT / name).is_file() for name in inventory)
    assert not any(name.startswith((".superpowers/", "docs/")) for name in inventory)


def test_reference_preflight_checks_actual_installed_schema(monkeypatch):
    original = torch._C._get_schema
    monkeypatch.setattr(torch._C, "_get_schema", lambda name, overload:
                        "unavailable: CPU fixture changed schema" if name == "aten::addmm" else original(name, overload))
    with pytest.raises(ValueError, match="installed"):
        evidence.validate_reference_receipt(evidence.REFERENCE)


@pytest.mark.parametrize("mutation", [
    lambda f, r: f.update(stage="real"),
    lambda f, r: f["records"].pop(),
    lambda f, r: f["records"].append(f["records"][0]),
    lambda f, r: r.update(stage="real"),
    lambda f, r: r["reference_receipt"].update(path="/CPU-FIXTURE/substituted-reference.json"),
    lambda f, r: r["command"].__setitem__(6, "sync"),
    lambda f, r: r["environment"].update(PYTHONPATH="relative:build:tests/python"),
])
def test_completed_stage_and_receipt_mutations_rejected(filesystem_fixture, monkeypatch, mutation):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    capture, receipt_path = attempts[0] / "capture.json", attempts[0] / "receipt.json"
    original, receipt_bytes = capture.read_bytes(), receipt_path.read_bytes()
    fragment, receipt = json.loads(original), json.loads(receipt_bytes)
    mutation(fragment, receipt)
    raw = json.dumps(fragment).encode()
    capture.write_bytes(raw)
    receipt["outputs"]["capture.json"] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    receipt_path.write_text(json.dumps(receipt))
    try:
        with pytest.raises(ValueError):
            evidence._validate_attempt(attempts[0])
    finally:
        capture.write_bytes(original)
        receipt_path.write_bytes(receipt_bytes)


def test_aggregate_rejects_mixed_runtime_even_with_hash_consistent_receipts(filesystem_fixture, monkeypatch):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    capture, receipt_path = attempts[0] / "capture.json", attempts[0] / "receipt.json"
    original, receipt_bytes = capture.read_bytes(), receipt_path.read_bytes()
    fragment, receipt = json.loads(original), json.loads(receipt_bytes)
    for record in fragment["records"]:
        record["runtime"]["hardware"] = "a different CPU-FIXTURE device"
    receipt["runtime"]["hardware"] = "a different CPU-FIXTURE device"
    raw = json.dumps(fragment).encode()
    capture.write_bytes(raw)
    receipt["outputs"]["capture.json"] = {"sha256": hashlib.sha256(raw).hexdigest(), "size": len(raw)}
    receipt_path.write_text(json.dumps(receipt))
    try:
        with pytest.raises(ValueError, match="common runtime"):
            evidence.aggregate_attempts(attempts, repo_root=root)
    finally:
        capture.write_bytes(original)
        receipt_path.write_bytes(receipt_bytes)


@pytest.mark.parametrize("name", ["capture.json", "receipt.json"])
def test_current_requires_complete_raw_files(filesystem_fixture, monkeypatch, name):
    root, attempts, _ = filesystem_fixture
    monkeypatch.setattr(evidence, "ROOT", root)
    path = attempts[0] / name
    held = path.with_suffix(".CPU-FIXTURE-held")
    path.rename(held)
    try:
        with pytest.raises(ValueError, match="missing"):
            evidence.aggregate_attempts(attempts, repo_root=root)
    finally:
        held.rename(path)


def test_real_replay_available_branch_checks_loader_metadata_and_metrics(records, tmp_path, monkeypatch):
    import tools.mnist_data as data
    calls = []

    def verified_loader_boundary(root, *, split, repo_root):
        # Explicit dataset-verification TEST boundary; canonical verification
        # itself is covered by test_mnist_data.py, not claimed by these fixtures.
        calls.append(split)
        count = 257 if split == "train" else 129
        return TensorDataset(torch.zeros(count, 1, 28, 28), torch.arange(count, dtype=torch.int64) % 10), _fixture_data()[split]

    monkeypatch.setattr(data, "load_mnist_subset", verified_loader_boundary)
    real_records = [r for r in records if r["recipe"]["kind"] == "real"]
    result = evidence.replay_cpu_reference(_document(real_records), real_root=tmp_path, repo_root=evidence.ROOT)
    assert calls == ["train", "test"]
    assert set(result.values()) == {"verified"}
    wrong = dict(real_records[0])
    wrong["data_identity"] = deepcopy(wrong["data_identity"])
    wrong["data_identity"]["train"]["input_digest_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CPU replay verified dataset identity"):
        evidence.replay_cpu_reference(_document([wrong]), real_root=tmp_path, repo_root=evidence.ROOT)
