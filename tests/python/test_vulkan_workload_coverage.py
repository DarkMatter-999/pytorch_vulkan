import copy
import json

import pytest
import torch

import vulkan_workload_conformance as workload
from tools import vulkan_workload_coverage as coverage


def _windows(is_hvp):
    phases = [(None, phase) for phase in ("first_reverse", "second_reverse")] if is_hvp else [
        (step, phase)
        for step in (1, 2, 3)
        for phase in ("reset", "forward_backward", "optimizer")
    ]
    return [
        {
            "phase": phase,
            "step": step,
            "dispatches": int(phase not in ("reset", "optimizer")),
            "fallbacks": 0,
            "explicit_transfers": 0,
            "observation_order": ["sync", "snapshot", "readback"],
            "synchronized_before_snapshot": True,
            "snapshot_before_readback": True,
        }
        for step, phase in phases
    ]


def _tensor_paths(cpu, vk, path=""):
    if isinstance(cpu, dict) and isinstance(vk, dict):
        if {"dtype", "shape", "device", "values"} <= cpu.keys():
            if vk["values"] is not None:
                yield path, cpu, vk
            return
        assert cpu.keys() == vk.keys()
        for key in cpu:
            yield from _tensor_paths(cpu[key], vk[key], f"{path}.{key}" if path else key)
    elif isinstance(cpu, list):
        assert len(cpu) == len(vk)
        for index, (left, right) in enumerate(zip(cpu, vk)):
            yield from _tensor_paths(left, right, f"{path}[{index}]")


def _synthetic_record(scenario, mode):
    if scenario.workload_id == workload.MATRIX_HVP_ID: cpu = workload.run_cpu_matrix_hvp()
    elif scenario.reset_mode is None: cpu = workload.run_cpu_hvp()
    elif scenario.workload_id in (workload.MATRIX_NONE_ID, workload.MATRIX_ZERO_ID): cpu = workload.run_cpu_matrix_sgd(scenario.reset_mode)
    else: cpu = workload.run_cpu_classifier(scenario.reset_mode)
    vk = copy.deepcopy(cpu)
    is_hvp = scenario.reset_mode is None

    def set_vk(node):
        if isinstance(node, dict):
            if {"dtype", "shape", "device", "values"} <= node.keys():
                node["device"] = "vk:0"
            for value in node.values():
                set_vk(value)
        elif isinstance(node, list):
            for value in node:
                set_vk(value)

    set_vk(vk)
    if is_hvp:
        vk["finite_difference"] = None
    errors = {path: 0.0 for path, _, _ in _tensor_paths(cpu, vk)}
    return {
        "workload_id": scenario.workload_id,
        "reset_mode": scenario.reset_mode,
        "execution_mode": mode,
        "device": "vk:0",
        "hardware": "synthetic-test-device",
        "driver": "synthetic-test-driver",
        "extension_sha256": "f" * 64,
        "fixture": workload.fixture_metadata(scenario.workload_id),
        "cpu": cpu,
        "vulkan": vk,
        "comparison": {"rtol": 3e-3, "atol": 3e-3, "passed": True, "max_abs_errors": errors},
        "execution": _windows(is_hvp),
    }


@pytest.fixture
def valid_document():
    scenarios = tuple(workload.SCENARIOS)
    return {
        "schema_version": 1,
        "torch_version": "2.4.0+cpu",
        "torch_git_revision": "e4ee3be4063b7c430974252fdf7db42273388d86",
        "records": [
            _synthetic_record(scenario, mode)
            for scenario in scenarios
            for mode in ("async", "sync")
        ],
    }


def _mode_capture(document, mode):
    return {**{key: value for key, value in document.items() if key != "records"},
            "records": [record for record in document["records"] if record["execution_mode"] == mode]}


def test_valid_source_bound_document_and_deterministic_merge(valid_document):
    coverage.validate_document(valid_document)
    async_capture = _mode_capture(valid_document, "async")
    sync_capture = _mode_capture(valid_document, "sync")
    merged = coverage.merge_documents(async_capture, sync_capture)
    assert len(merged["records"]) == 12
    assert len({(r["workload_id"], r["reset_mode"], r["execution_mode"]) for r in merged["records"]}) == 12
    encoded = coverage.dumps_document(merged)
    assert encoded.endswith("\n")
    assert coverage.dumps_document(coverage.merge_documents(async_capture, sync_capture)) == encoded


def test_selective_record_cli_accepts_repeatable_workload_ids_and_merges_base(monkeypatch, tmp_path):
    # Parser-level contract: selected captures are partial, while --base merge
    # retains a complete historical artifact and validates the resulting union.
    calls = []
    monkeypatch.setattr(coverage, "_record", lambda mode, output, workload_ids=None: calls.append((mode, output, workload_ids)))
    assert coverage.main(["record", "--mode", "async", "--output", "fragment.json",
                          "--workload", "dense.two-linear.output-energy.sgd-momentum.zero-grad-none",
                          "--workload", "hvp.dense.two-linear.output-energy.parameters"]) is None
    assert calls == [("async", "fragment.json", [
        "dense.two-linear.output-energy.sgd-momentum.zero-grad-none",
        "hvp.dense.two-linear.output-energy.parameters",
    ])]


def test_matrix_workload_ids_are_source_owned_and_have_immutable_mode_keys():
    expected = {
        "dense.two-linear.output-energy.sgd-momentum.zero-grad-none",
        "dense.two-linear.output-energy.sgd-momentum.zero-grad-zero",
        "hvp.dense.two-linear.output-energy.parameters",
    }
    assert expected <= {item.workload_id for item in workload.SCENARIOS}
    assert {("dense.two-linear.output-energy.sgd-momentum.zero-grad-none", "none", mode)
            for mode in ("async", "sync")} <= workload.REQUIRED_RECORD_KEYS


def test_selective_fragments_merge_over_exact_historical_six_records(valid_document, tmp_path):
    legacy_ids = {workload.CLASSIFIER_NONE_ID, workload.CLASSIFIER_ZERO_ID, workload.HVP_ID}
    matrix_ids = {workload.MATRIX_NONE_ID, workload.MATRIX_ZERO_ID, workload.MATRIX_HVP_ID}
    base = {key: value for key, value in valid_document.items() if key != "records"}
    base["records"] = [record for record in valid_document["records"] if record["workload_id"] in legacy_ids]
    async_fragment = {**{k:v for k,v in valid_document.items() if k != "records"},
                      "records": [r for r in valid_document["records"] if r["workload_id"] in matrix_ids and r["execution_mode"] == "async"]}
    sync_fragment = {**{k:v for k,v in valid_document.items() if k != "records"},
                     "records": [r for r in valid_document["records"] if r["workload_id"] in matrix_ids and r["execution_mode"] == "sync"]}
    base_path, async_path, sync_path, output = (tmp_path / name for name in ("base.json", "async.json", "sync.json", "merged.json"))
    for path, document in ((base_path, base), (async_path, async_fragment), (sync_path, sync_fragment)):
        path.write_text(json.dumps(document), encoding="utf-8")
    coverage._merge(str(async_path), str(sync_path), str(output), str(base_path))
    merged = json.loads(output.read_text(encoding="utf-8"))
    assert len(merged["records"]) == 12
    records_by_id = {(r["workload_id"], r["reset_mode"], r["execution_mode"]):r for r in merged["records"]}
    for record in base["records"]:
        key = (record["workload_id"], record["reset_mode"], record["execution_mode"])
        assert records_by_id[key] == record
    assert coverage.dumps_document(merged) == output.read_text(encoding="utf-8")


def test_partial_fragment_rejects_missing_duplicate_and_wrong_mode(valid_document):
    selected = [workload.MATRIX_NONE_ID]
    fragment = _mode_capture(valid_document, "async")
    fragment["records"] = [r for r in fragment["records"] if r["workload_id"] == selected[0]]
    coverage.validate_fragment(fragment, "async", selected)
    with pytest.raises(ValueError, match="exactly one"):
        coverage.validate_fragment({**fragment, "records": []}, "async", selected)
    with pytest.raises(ValueError, match="duplicate"):
        coverage.validate_fragment({**fragment, "records": fragment["records"] * 2}, "async", selected)
    wrong = copy.deepcopy(fragment); wrong["records"][0]["execution_mode"] = "sync"
    with pytest.raises(ValueError, match="identity/mode"):
        coverage.validate_fragment(wrong, "async", selected)


@pytest.mark.parametrize("mutation", ["fixture", "reset", "residency", "fallback", "history"])
def test_matrix_record_validator_rejects_semantic_mutations(valid_document, mutation):
    document = copy.deepcopy(valid_document)
    record = next(r for r in document["records"] if r["workload_id"] == workload.MATRIX_NONE_ID)
    if mutation == "fixture":
        record["fixture"]["input"]["shape"] = [4, 3]
    elif mutation == "reset":
        record["reset_mode"] = "zero"
    elif mutation == "residency":
        record["vulkan"]["steps"][0]["output"]["device"] = "cpu"
    elif mutation == "fallback":
        record["execution"][1]["fallbacks"] = 1
    else:
        record = next(r for r in document["records"] if r["workload_id"] == workload.MATRIX_HVP_ID)
        record["vulkan"]["history"]["first_gradients_require_grad"]["0.weight"] = False
    with pytest.raises(ValueError):
        coverage.validate_document(document)


@pytest.mark.parametrize("entrypoint", ["record", "collection", "document"])
@pytest.mark.parametrize(("field", "value"), [
    ("workload_id", []),
    ("workload_id", {"bad": "id"}),
    ("workload_id", 7),
    ("workload_id", None),
    ("execution_mode", []),
    ("execution_mode", {"bad": "mode"}),
    ("execution_mode", 1),
    ("execution_mode", None),
    ("reset_mode", []),
    ("reset_mode", {"bad": "reset"}),
    ("reset_mode", 1),
])
def test_identity_fields_reject_malformed_scalar_types_with_field_specific_errors(
    valid_document, entrypoint, field, value
):
    record = copy.deepcopy(valid_document["records"][0])
    record[field] = value
    if entrypoint == "record":
        call = lambda: coverage.validate_record(record)
    elif entrypoint == "collection":
        capture = _mode_capture(valid_document, "async")
        capture["records"][0][field] = value
        call = lambda: coverage.validate_collection(capture, "async")
    else:
        document = copy.deepcopy(valid_document)
        document["records"][0][field] = value
        call = lambda: coverage.validate_document(document)
    with pytest.raises(ValueError, match=field):
        call()


@pytest.mark.parametrize("entrypoint", ["record", "collection", "document"])
def test_identity_entrypoints_reject_non_object_records(valid_document, entrypoint):
    if entrypoint == "record":
        call = lambda: coverage.validate_record([])
    elif entrypoint == "collection":
        capture = _mode_capture(valid_document, "async")
        capture["records"][0] = []
        call = lambda: coverage.validate_collection(capture, "async")
    else:
        document = copy.deepcopy(valid_document)
        document["records"][0] = []
        call = lambda: coverage.validate_document(document)
    with pytest.raises(ValueError, match="record must be an object"):
        call()


def test_cli_bad_identity_is_a_clear_exit_two_error_without_traceback(valid_document, tmp_path, capsys):
    document = copy.deepcopy(valid_document)
    document["records"][0]["workload_id"] = []
    path = tmp_path / "bad-identity.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        coverage.main(["validate", str(path)])
    assert exit_info.value.code == 2
    stderr = capsys.readouterr().err
    assert "workload_id" in stderr
    assert "Traceback" not in stderr


def test_literal_source_matrix_survives_coordinated_source_and_record_deletion(monkeypatch, valid_document):
    document = copy.deepcopy(valid_document)
    removed = document["records"][-1]["workload_id"]
    document["records"] = [r for r in document["records"] if r["workload_id"] != removed]
    monkeypatch.setattr(workload, "SCENARIOS", tuple(s for s in workload.SCENARIOS if s.workload_id != removed))
    with pytest.raises(ValueError, match="source-owned required workload"):
        coverage.validate_document(document)


def test_literal_source_matrix_cannot_be_erased(valid_document):
    with pytest.raises(ValueError, match="source-owned required workload"):
        coverage.validate_required_matrix(())


@pytest.mark.parametrize("mutation", [
    "missing_tensor", "boolean", "nan", "spoofed_error", "wrong_device", "wrong_shape",
    "wrong_dtype", "truncated_step", "false_history", "invalid_grad_fn", "zero_hvp",
    "reset_identity", "nonzero_reset", "missing_momentum", "extra_state", "wrong_order",
    "fallback", "changed_tolerance", "wrong_mode", "coordinated_payload",
])
def test_strict_validator_rejects_tampered_payloads(valid_document, mutation):
    document = copy.deepcopy(valid_document)
    record = document["records"][0]
    if mutation == "missing_tensor":
        del record["vulkan"]["steps"][0]["loss"]
    elif mutation == "boolean":
        record["vulkan"]["steps"][0]["loss"]["values"] = [True]
    elif mutation == "nan":
        record["vulkan"]["steps"][0]["loss"]["values"] = [float("nan")]
    elif mutation == "spoofed_error":
        record["comparison"]["max_abs_errors"]["steps[0].loss"] = 1.0
    elif mutation == "wrong_device":
        record["vulkan"]["steps"][0]["loss"]["device"] = "cpu"
    elif mutation == "wrong_shape":
        record["vulkan"]["steps"][0]["loss"]["shape"] = [True]
    elif mutation == "wrong_dtype":
        record["vulkan"]["steps"][0]["loss"]["dtype"] = "torch.float64"
    elif mutation == "truncated_step":
        record["execution"].pop()
    elif mutation == "false_history":
        record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
        record["vulkan"]["history"]["first_gradients_require_grad"]["0.weight"] = False
    elif mutation == "invalid_grad_fn":
        record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
        record["vulkan"]["history"]["grad_fns"]["0.weight"] = "NoneType"
    elif mutation == "zero_hvp":
        record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
        for tensor in record["vulkan"]["hvp"].values():
            tensor["values"] = [0.0] * len(tensor["values"])
            tensor["nonzero_count"] = 0
    elif mutation == "reset_identity":
        record["vulkan"]["reset_observations"][1]["same_grad_objects"] = False
    elif mutation == "nonzero_reset":
        record = next(r for r in document["records"] if r["reset_mode"] == "zero")
        record["vulkan"]["reset_observations"][1]["gradients"]["0.weight"]["values"][0] = 1.0
    elif mutation == "missing_momentum":
        del record["vulkan"]["steps"][0]["momentum"]["0.weight"]
    elif mutation == "extra_state":
        record["vulkan"]["steps"][0]["state_keys"]["0.weight"].append("step")
    elif mutation == "wrong_order":
        record["execution"][0]["observation_order"] = ["readback", "sync", "snapshot"]
    elif mutation == "fallback":
        record["execution"][1]["fallbacks"] = 1
    elif mutation == "changed_tolerance":
        record["comparison"]["atol"] = 0.3
    elif mutation == "wrong_mode":
        record["execution_mode"] = "sync" if record["execution_mode"] == "async" else "async"
    else:
        record["cpu"]["steps"][0]["loss"]["values"] = [9.0]
        record["vulkan"]["steps"][0]["loss"]["values"] = [9.0]
    with pytest.raises(ValueError):
        coverage.validate_document(document)


def test_merge_rejects_mode_mismatch_and_incoherent_builds(valid_document):
    async_capture = _mode_capture(valid_document, "async")
    sync_capture = _mode_capture(valid_document, "sync")
    async_capture["records"][0]["execution_mode"] = "sync"
    with pytest.raises(ValueError):
        coverage.merge_documents(async_capture, sync_capture)
    async_capture = _mode_capture(valid_document, "async")
    async_capture["records"][0]["extension_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        coverage.merge_documents(async_capture, sync_capture)


def test_every_hvp_execution_phase_requires_its_own_positive_dispatch(valid_document):
    document = copy.deepcopy(valid_document)
    record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
    record["execution"][1]["dispatches"] = 0
    with pytest.raises(ValueError, match="dispatch"):
        coverage.validate_document(document)


def test_classifier_execution_step_rejects_boolean_integer_alias(valid_document):
    document = copy.deepcopy(valid_document)
    record = next(r for r in document["records"] if r["workload_id"] != workload.HVP_ID)
    record["execution"][0]["step"] = True
    with pytest.raises(ValueError, match="phase/step"):
        coverage.validate_document(document)


def _set_first_loss_delta(document, delta):
    record = next(r for r in document["records"] if r["workload_id"] == workload.CLASSIFIER_NONE_ID)
    cpu_loss = record["cpu"]["steps"][0]["loss"]["values"][0]
    vk_loss = cpu_loss + delta
    record["vulkan"]["steps"][0]["loss"]["values"][0] = vk_loss
    record["comparison"]["max_abs_errors"]["steps[0].loss"] = abs(vk_loss - cpu_loss)


def test_parity_uses_torch_assert_close_additive_rtol_atol_boundary(valid_document):
    document = copy.deepcopy(valid_document)
    _set_first_loss_delta(document, 0.005)
    coverage.validate_document(document)


def test_parity_rejects_values_outside_torch_assert_close_boundary(valid_document):
    document = copy.deepcopy(valid_document)
    _set_first_loss_delta(document, 0.007)
    with pytest.raises(ValueError, match="parity tolerance"):
        coverage.validate_document(document)


@pytest.mark.parametrize(
    ("version", "revision"),
    [("0.0.0-test", "e4ee3be4063b7c430974252fdf7db42273388d86"),
     ("2.4.0+cpu", "bad-revision")],
)
def test_document_validation_requires_actual_supported_torch_build(valid_document, monkeypatch, version, revision):
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "git_version", revision)
    with pytest.raises(ValueError, match="installed PyTorch"):
        coverage.validate_document(valid_document)


def test_valid_vulkan_hvp_history_may_use_different_grad_fn_class_names(valid_document):
    document = copy.deepcopy(valid_document)
    record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
    record["vulkan"]["history"]["grad_fns"] = {
        name: "DifferentDifferentiableBackward" for name in workload.run_cpu_hvp()["hvp"]
    }
    coverage.validate_document(document)


@pytest.mark.parametrize("mutation", ["missing", "extra", "non_bool", "none_type"])
def test_vulkan_hvp_history_requires_exact_named_boolean_and_valid_grad_fns(valid_document, mutation):
    document = copy.deepcopy(valid_document)
    record = next(r for r in document["records"] if r["workload_id"] == workload.HVP_ID)
    history = record["vulkan"]["history"]
    if mutation == "missing":
        del history["first_gradients_require_grad"]["0.weight"]
    elif mutation == "extra":
        history["grad_fns"]["extra"] = "SomeBackward"
    elif mutation == "non_bool":
        history["first_gradients_require_grad"]["0.weight"] = 1
    else:
        history["grad_fns"]["0.weight"] = "NoneType"
    with pytest.raises(ValueError):
        coverage.validate_document(document)
