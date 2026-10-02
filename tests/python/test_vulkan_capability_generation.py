import json
from pathlib import Path

import pytest
import pytorch_vulkan
import torch

from tools import generate_vulkan_capabilities as generator
from tools import validate_vulkan_capabilities as capability_validator
from tools.validate_vulkan_capabilities import (
    load_manifest,
    schema_exists,
    validate_tensor_list_evidence,
)

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "docs/vulkan_capabilities.json"
COVERAGE_COMMITTED = ROOT / "docs/vulkan_coverage.json"
DERIVED_FIELDS = frozenset({"schema", "test_cases", "tests", "witnesses"})


def test_every_declared_schema_exists_in_pytorch_dispatcher():
    """Supported and deferred claims must name real dispatcher schemas."""
    manifest = load_manifest(Path("docs/vulkan_capabilities.json"))
    missing = [
        entry["schema"]
        for entry in manifest["entries"]
        if entry["status"] != "rejected" and not schema_exists(entry["schema"])
    ]
    assert missing == [], f"manifest names schemas absent from torch.ops.aten: {missing}"


def test_schema_exists_helper_rejects_phantoms():
    """The helper itself must discriminate: real schemas yes, phantoms no."""
    assert schema_exists("aten::relu.out")
    assert schema_exists("aten::abs.default")
    assert schema_exists("aten::convolution.default")
    assert not schema_exists("aten::isfinite.out")
    assert not schema_exists("aten::_cat.default")


def _committed_entries() -> dict[str, dict]:
    data = json.loads(COMMITTED.read_text())
    return {entry["schema"]: entry for entry in data["entries"]}


def test_generated_manifest_reproduces_every_derived_field():
    committed = _committed_entries()
    generated = {
        entry["schema"]: entry for entry in generator.build_manifest()["entries"]
    }
    assert set(generated) == set(committed)
    for schema, entry in committed.items():
        for field in DERIVED_FIELDS:
            assert generated[schema][field] == entry[field], f"{schema}.{field}"


def test_generated_manifest_reproduces_every_declared_field():
    committed = _committed_entries()
    generated = {
        entry["schema"]: entry for entry in generator.build_manifest()["entries"]
    }
    for schema, entry in committed.items():
        for field in set(entry) - DERIVED_FIELDS:
            assert generated[schema][field] == entry[field], f"{schema}.{field}"


def test_generator_is_deterministic():
    assert generator.render(generator.build_manifest()) == generator.render(
        generator.build_manifest()
    )


def test_render_ends_with_exactly_one_newline():
    text = generator.render(generator.build_manifest())
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_render_has_no_schema_specific_formatting():
    manifest = {
        "entries": [
            {
                "schema": "aten::unregistered.default",
                "shape_constraints": ["example_shape"],
                "test_cases": [{"name": "example", "supported": True}],
            }
        ],
        "version": 1,
    }
    assert generator.render(manifest) == json.dumps(
        manifest, indent=2, sort_keys=True
    ) + "\n"


def test_manifest_drift_comparison_detects_tampered_text():
    committed_text = COMMITTED.read_text()
    generated = generator.build_manifest()
    tampered_text = committed_text.replace('"version": 1', '"version": 2', 1)
    assert generator.manifest_matches(committed_text, generated)
    assert not generator.manifest_matches(tampered_text, generated)


def test_missing_declaration_names_the_registered_schema(monkeypatch):
    monkeypatch.setattr(
        generator,
        "_source_registration_inventory",
        lambda _root: (set(generator.DECLARATIONS) | {"aten::nonexistent.default"}, set()),
    )
    with pytest.raises(SystemExit, match="missing declarations for: aten::nonexistent"):
        generator._validate_declarations()


def test_declaration_with_a_typoed_key_is_rejected(monkeypatch):
    monkeypatch.setattr(
        generator,
        "_source_registration_inventory",
        lambda _root: (set(generator.DECLARATIONS), set()),
    )
    broken = dict(generator.DECLARATIONS["aten::nll_loss_forward.default"])
    broken["rank"] = broken.pop("ranks")
    monkeypatch.setitem(generator.DECLARATIONS, "aten::nll_loss_forward.default", broken)
    with pytest.raises(SystemExit, match="declaration keys must be exactly"):
        generator._validate_declarations()


def test_unregistered_declarations_are_roadmap_not_errors():
    roadmap = generator._roadmap_schemas()
    assert len(roadmap) == 86
    assert "aten::concat.default" in roadmap
    assert "aten::reshape.default" not in roadmap


def test_roadmap_declarations_have_no_implementation_witnesses():
    committed = _committed_entries()
    for schema in generator._roadmap_schemas():
        if schema not in committed:
            continue
        declaration = generator.DECLARATIONS[schema]
        if declaration["status"] == "rejected":
            continue
        assert declaration["status"] == "deferred"
        assert declaration["reason"] == "deferred_contract"
        assert committed[schema]["test_cases"] == []


def test_every_supported_schema_has_a_conformance_case():
    supported = {
        schema
        for schema, declaration in generator.DECLARATIONS.items()
        if declaration["status"] == "supported"
    }
    witnessed = {
        schema
        for schema, case in generator._case_index().items()
        if any(item["supported"] for item in case["test_cases"])
    }
    assert supported == witnessed


def test_witnesses_field_is_derived_from_executed_coverage():
    committed = _committed_entries()
    generated = {
        entry["schema"]: entry
        for entry in generator.build_manifest()["entries"]
    }
    for schema, entry in committed.items():
        assert "witnesses" in entry, f"{schema} has no witnesses field"
        assert generated[schema]["witnesses"] == entry["witnesses"], f"{schema}.witnesses"


def test_supported_entries_only_declare_what_witnesses_exercise():
    committed = _committed_entries()
    for schema, entry in committed.items():
        if entry["status"] != "supported":
            continue
        witnesses = entry["witnesses"]
        assert witnesses["dtypes"], f"{schema} is supported with no dtype witness"
        unproven_dtypes = set(entry["dtypes"]["inputs"]) - set(witnesses["dtypes"])
        assert not unproven_dtypes, (
            f"{schema} declares input dtypes {sorted(unproven_dtypes)} that no case exercised"
        )
        proven = set(witnesses["ranks"])
        assert proven, f"{schema} is supported with no rank witness"
        for rank in range(entry["ranks"]["min"], entry["ranks"]["max"] + 1):
            assert rank in proven, (
                f"{schema} declares ranks {entry['ranks']['min']}-{entry['ranks']['max']} "
                f"but rank {rank} was never exercised"
            )


def test_auxiliary_and_output_ranks_do_not_witness_primary_input_rank():
    coverage = {
        "synthetic.convolution": {
            "schema": "aten::convolution.default",
            "primary_input": {"dtype": "float32", "rank": 4},
            "operands": [
                {"role": "primary_input", "dtype": "float32", "rank": 4},
                {"role": "operand", "dtype": "float32", "rank": 4},
                {"role": "operand", "dtype": "float32", "rank": 1},
            ],
            "input_dtypes": ["float32"],
            "input_ranks": [1, 4],
            "output_dtype": "float32",
            "output_rank": 1,
            "parity": True,
        }
    }
    witness = generator._witnesses_by_schema(coverage)["aten::convolution.default"]
    assert witness["ranks"] == [4]
    assert witness["dtypes"] == ["float32"]


def test_reverse_second_order_witness_requires_executed_metadata_not_gradient_flag():
    name = "arithmetic.autograd.add-tensor"
    schema = "aten::add.Tensor"
    base = {
        "schema": schema,
        "primary_input": {"dtype": "float32", "rank": 1},
        "parity": True,
        "gradients": True,
    }
    assert "reverse_second_order_cases" not in generator._witnesses_by_schema({name: base})[schema]
    witnessed = dict(base, reverse_autograd={"order": 2, "graph_preserved": True})
    assert generator._witnesses_by_schema({name: witnessed})[schema]["reverse_second_order_cases"] == [name]


def test_reverse_second_order_witness_reproduction_uses_case_specific_coverage():
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    witnessed = {
        name: record for name, record in coverage.items()
        if record.get("reverse_autograd") == {"order": 2, "graph_preserved": True}
    }
    assert len(witnessed) == 11
    for name, record in witnessed.items():
        assert name.startswith(("arithmetic.autograd.", "view.", "cat."))
        assert record["schema"] in {
            "aten::add.Tensor", "aten::add.Scalar", "aten::mul.Tensor",
            "aten::mul.Scalar", "aten::sum.default", "aten::sum.dim_IntList",
            "aten::view.default", "aten::reshape.default", "aten::cat.default",
        }


def test_cat_tensor_list_evidence_validator_binds_aggregates_to_members():
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    record = coverage["cat.rank2.forward"]
    validate_tensor_list_evidence({"cat.rank2.forward": record})
    changed = json.loads(json.dumps(record))
    changed["tensor_lists"][0]["tensors"].reverse()
    with pytest.raises(ValueError, match="malformed tensor-list member|aggregate coverage metadata"):
        validate_tensor_list_evidence({"cat.rank2.forward": changed})
    changed = json.loads(json.dumps(record))
    changed["primary_input"]["rank"] = 4
    with pytest.raises(ValueError, match="aggregate coverage metadata"):
        validate_tensor_list_evidence({"cat.rank2.forward": changed})


def test_cat_tensor_list_validator_requires_exact_named_witnesses():
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    for name in list(coverage):
        if name.startswith("cat."):
            coverage[name].pop("tensor_lists", None)
    with pytest.raises(ValueError, match="required executed cat.default TensorList witness"):
        validate_tensor_list_evidence(coverage, require_complete=True)

    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    coverage["cat.rank4.native-seed"] = json.loads(json.dumps(coverage["cat.rank2.forward"]))
    with pytest.raises(ValueError, match="named witness"):
        validate_tensor_list_evidence(coverage, require_complete=True)


def test_cat_generation_refuses_missing_named_witness():
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    del coverage["cat.rank1.forward"]
    with pytest.raises(ValueError, match="required executed cat.default TensorList witness"):
        generator.build_coverage_manifest(coverage)


@pytest.mark.parametrize("mutation", ["secondary_dtype", "output_dtype", "all_float64"])
def test_cat_float32_tensorlist_identity_is_mandatory_for_validation_and_generation(
    mutation, monkeypatch
):
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    record = coverage["cat.rank4.native-seed"]
    tensors = record["tensor_lists"][0]["tensors"]
    if mutation == "secondary_dtype":
        tensors[1]["dtype"] = "float64"
        record["operands"][1]["dtype"] = "float64"
        record["input_dtypes"] = ["float32", "float64"]
    elif mutation == "output_dtype":
        record["output_dtype"] = "float64"
    else:
        for tensor, operand in zip(tensors, record["operands"]):
            tensor["dtype"] = "float64"
            operand["dtype"] = "float64"
        record["input_dtypes"] = ["float64"]
        record["primary_input"]["dtype"] = "float64"

    with pytest.raises(ValueError, match="float32"):
        validate_tensor_list_evidence(coverage, require_complete=True)
    with pytest.raises(ValueError, match="float32"):
        generator.build_coverage_manifest(coverage)
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="float32"):
        capability_validator.validate_manifest_data(load_manifest(COMMITTED), ROOT)


def _convolution_record(mask, bias_present):
    name = "convolution.backward.bias-present.mask-" if bias_present else "convolution.backward.bias-absent.mask-"
    name += "".join("1" if bit else "0" for bit in mask)
    shapes = {
        "input": [1, 4, 5, 6],
        "weight": [4, 2, 2, 2],
        "bias": [4],
        "grad_output": [1, 4, 4, 5],
    }
    operands = {}
    for role, shape in shapes.items():
        if role == "bias" and not bias_present:
            operands[role] = {"defined": False}
            continue
        strides = [shape[1] * shape[2] * shape[3], shape[2] * shape[3], shape[3], 1] if len(shape) == 4 else [1]
        operands[role] = {
            "defined": True, "dtype": "float32", "rank": len(shape),
            "shape": shape, "strides": strides, "storage_offset": 0,
            "format": "contiguous",
        }
    return name, {
        "schema": "aten::convolution_backward.default",
        "primary_input": {"dtype": "float32", "rank": 4},
        "input_dtypes": ["float32"], "input_ranks": [4],
        "input_shapes": ["1x4x4x5", "1x4x5x6", "4x2x2x2"],
        "operands": [{"role": "primary_input", "dtype": "float32", "rank": 4}],
        "output_slots": [
            {"index": index, "defined": bit, **({"dtype": "float32", "rank": len(shapes[role]), "shape": shapes[role]} if bit else {})}
            for index, (bit, role) in enumerate(zip(mask, ("input", "weight", "bias")))
        ],
        "convolution_context": {
            "device": "vk:0", "cpu_oracle": "aten::convolution_backward.default",
            "forward_bias_present": bias_present, "output_mask": mask,
        },
        "convolution_operands": operands,
        "schema_args": {
            "bias_sizes": [4], "stride": [1, 1], "padding": [0, 0],
            "dilation": [1, 1], "transposed": False,
            "output_padding": [0, 0], "groups": 2, "output_mask": mask,
        },
        "execution": {
            "compute_dispatches": sum(mask), "vulkan_copies": 0,
            "explicit_transfers": 0, "fallbacks": 0,
            "buffer_creations_delta": sum(mask), "live_allocations_delta": sum(mask),
        },
        "gradients": False, "parity": True,
    }


def test_convolution_evidence_validator_binds_case_identity_to_actual_metadata():
    from tools.validate_vulkan_capabilities import validate_convolution_evidence

    name = "convolution.backward.bias-absent.mask-000"
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    validate_convolution_evidence(coverage, require_complete=True)
    generator.build_coverage_manifest(coverage)
    record = coverage[name]

    changed = json.loads(json.dumps(record))
    changed["convolution_context"]["output_mask"] = [True, False, False]
    changed["output_slots"][0] = {"index": 0, "defined": True, "dtype": "float32", "rank": 4, "shape": [1, 4, 5, 6]}
    changed["schema_args"]["output_mask"] = [True, False, False]
    changed["execution"]["compute_dispatches"] = 1
    coverage[name] = changed
    with pytest.raises(ValueError, match="case identity"):
        validate_convolution_evidence(coverage)
    with pytest.raises(ValueError, match="case identity"):
        generator.build_coverage_manifest(coverage)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing", "dtype", "slot", "slot_dtype", "defined", "counter",
        "negative_counter", "mask000_copy", "mask000_resource", "metadata",
        "device", "oracle", "parity",
    ],
)
def test_convolution_evidence_rejects_consistent_looking_tampering(mutation, monkeypatch):
    from tools.validate_vulkan_capabilities import validate_convolution_evidence

    name = (
        "convolution.backward.bias-absent.mask-001"
        if mutation == "slot_dtype" else
        "convolution.backward.bias-absent.mask-000"
    )
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    record = coverage[name]
    if mutation == "missing":
        del coverage[name]
        with pytest.raises(ValueError, match="required executed convolution witness"):
            validate_convolution_evidence(coverage, require_complete=True)
        with pytest.raises(ValueError, match="required executed convolution witness"):
            generator.build_coverage_manifest(coverage)
        monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
        with pytest.raises(ValueError, match="required executed convolution witness"):
            capability_validator.validate_manifest_data(load_manifest(COMMITTED), ROOT)
        return
    if mutation == "dtype":
        def rewrite_dtype(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if key == "dtype" and child == "float32":
                        value[key] = "float64"
                    else:
                        rewrite_dtype(child)
            elif isinstance(value, list):
                for child in value:
                    rewrite_dtype(child)
        rewrite_dtype(record)
        record["input_dtypes"] = ["float64"]
    elif mutation == "slot":
        record["output_slots"][0]["shape"] = [9, 9, 9, 9]
    elif mutation == "slot_dtype":
        record["output_slots"][2]["dtype"] = "float64"
    elif mutation == "defined":
        record["output_slots"][2] = {"index": 2, "defined": True, "dtype": "float32", "rank": 1, "shape": [4]}
    elif mutation == "counter":
        record["execution"]["compute_dispatches"] = 1
    elif mutation == "negative_counter":
        record["execution"]["fallbacks"] = -1
    elif mutation == "mask000_copy":
        record["execution"]["vulkan_copies"] = 1
    elif mutation == "mask000_resource":
        record["execution"]["buffer_creations_delta"] = 1
    elif mutation == "metadata":
        record["convolution_operands"]["weight"]["strides"] = [1, 1, 1, 1]
    elif mutation == "device":
        record["convolution_context"]["device"] = "vk:1"
    elif mutation == "oracle":
        record["convolution_context"]["cpu_oracle"] = "torch.nn.functional.conv2d"
    else:
        record["parity"] = False
    with pytest.raises(ValueError):
        validate_convolution_evidence(coverage)
    with pytest.raises(ValueError):
        generator.build_coverage_manifest(coverage)
    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError):
        capability_validator.validate_manifest_data(load_manifest(COMMITTED), ROOT)


@pytest.mark.parametrize(
    ("schema", "case", "mutation"),
    [
        ("aten::convolution.default", "convolution.forward.bias-absent", "input_dtype"),
        ("aten::convolution.default", "convolution.forward.bias-present", "output_dtype"),
        ("aten::convolution.default", "convolution.forward.bias-absent", "rank_pair"),
        ("aten::convolution.default", "convolution.forward.bias-present", "case_link"),
        ("aten::convolution.default", "convolution.forward.bias-absent", "unsupported_case"),
        ("aten::convolution_backward.default", "convolution.backward.bias-absent.mask-000", "input_dtype"),
        ("aten::convolution_backward.default", "convolution.backward.bias-present.mask-111", "output_dtype"),
        ("aten::convolution_backward.default", "convolution.backward.bias-absent.mask-000", "rank_pair"),
        ("aten::convolution_backward.default", "convolution.backward.bias-absent.mask-000", "case_link"),
        ("aten::convolution_backward.default", "convolution.backward.bias-present.mask-000", "unsupported_case"),
    ],
)
def test_convolution_manifest_claims_are_bound_to_coverage(schema, case, mutation, monkeypatch):
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    manifest = load_manifest(COMMITTED)
    entry = next(item for item in manifest["entries"] if item["schema"] == schema)
    if mutation == "input_dtype":
        entry["dtypes"]["inputs"] = ["float64"]
        entry["witnesses"]["dtypes"] = ["float64"]
        entry["witnesses"]["pairs"] = [["float64", 4]]
    elif mutation == "output_dtype":
        entry["dtypes"]["outputs"] = ["float64"]
    elif mutation == "rank_pair":
        entry["ranks"] = {"min": 3, "max": 3}
        entry["witnesses"]["ranks"] = [3]
        entry["witnesses"]["pairs"] = [["float32", 3]]
    elif mutation == "case_link":
        entry["witnesses"]["cases"].remove(case)
        entry["test_cases"] = [item for item in entry["test_cases"] if item["name"] != case]
    else:
        next(item for item in entry["test_cases"] if item["name"] == case)["supported"] = False

    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="convolution"):
        capability_validator.validate_manifest_data(manifest, ROOT)


def test_convolution_manifest_cannot_erase_the_complete_forward_witness_set(monkeypatch):
    coverage = json.loads(COVERAGE_COMMITTED.read_text())
    manifest = load_manifest(COMMITTED)
    entry = next(
        item for item in manifest["entries"]
        if item["schema"] == "aten::convolution.default"
    )
    required = {"convolution.forward.bias-present", "convolution.forward.bias-absent"}
    for name in required:
        del coverage[name]
    entry["witnesses"]["cases"] = [
        name for name in entry["witnesses"]["cases"] if name not in required
    ]
    entry["test_cases"] = [
        case for case in entry["test_cases"] if case["name"] not in required
    ]

    monkeypatch.setattr(capability_validator, "load_coverage_evidence", lambda _root: coverage)
    with pytest.raises(ValueError, match="required executed convolution witness"):
        capability_validator.validate_manifest_data(manifest, ROOT)


def test_declared_shapes_were_actually_exercised():
    import vulkan_conformance as vc

    coverage = _run_all_supported_cases()
    for case in vc.SUPPORTED_CASES:
        if not case.declared_shapes:
            continue
        exercised = set(coverage[case.name]["input_shapes"])
        for shape in case.declared_shapes:
            assert shape in exercised, (
                f"{case.name} declares shape {shape} but only exercised {sorted(exercised)}"
            )


def test_supported_entries_report_their_witness_count():
    committed = _committed_entries()
    thin = [
        schema
        for schema, entry in committed.items()
        if entry["status"] == "supported" and len(entry["witnesses"]["cases"]) == 1
    ]
    assert thin, "expected some entries to rest on a single witness"


def test_committed_manifest_matches_regenerated_output():
    if not _vulkan_device_available():
        pytest.skip("no Vulkan device: coverage cannot be recorded without executing cases")
    coverage = _run_all_supported_cases()
    committed_coverage = json.loads(COVERAGE_COMMITTED.read_text())
    assert coverage == committed_coverage, _coverage_drift_message(
        committed_coverage, coverage
    )
    assert COMMITTED.read_text() == generator.render(
        generator.build_coverage_manifest(coverage)
    )


def _coverage_drift_message(committed: dict, observed: dict) -> str:
    for case in sorted(set(committed) | set(observed)):
        if committed.get(case) != observed.get(case):
            return (
                f"coverage drift for {case}: committed={committed.get(case)!r}, "
                f"observed={observed.get(case)!r}"
            )
    return "coverage record differs"


def _probe_vk_device():
    try:
        torch.ones(1).to("vk:0")
        pytorch_vulkan._C.synchronize()
        return True
    except Exception:
        return False


def _vulkan_device_available() -> bool:
    try:
        return _probe_vk_device()
    except Exception:
        return False


def _run_all_supported_cases():
    import vulkan_conformance as vc

    with vc.coverage_recording():
        for case in vc.SUPPORTED_CASES:
            result, expected, inputs = vc.run_and_compare(case, return_inputs=True)
            vc.assert_result_parity(result, expected, case)
            reverse_names = {
                "arithmetic.autograd.add-tensor", "arithmetic.autograd.add-scalar",
                "arithmetic.autograd.mul-tensor", "arithmetic.autograd.mul-scalar",
                "arithmetic.autograd.sum-default", "arithmetic.autograd.sum-dim",
                "view.view.trainable-seed", "view.reshape.copy.trainable-seed",
                "view.reshape.offset-copy.second-order",
                "cat.rank4.native-seed", "cat.offset.trainable-seed",
            }
            reverse_autograd = vc.assert_reverse_second_order(case, inputs) if case.name in reverse_names else None
            vc.mark_executed(case.name)
            vc.record_coverage(
                case,
                inputs,
                result,
                gradients=case.check_gradients,
                parity=True,
                reverse_autograd=reverse_autograd,
            )
        return vc.coverage_snapshot()


def test_nll_schema_declares_real_shapes():
    committed = _committed_entries()
    for schema in ("aten::nll_loss_forward.default", "aten::nll_loss_backward.default"):
        shapes = committed[schema]["shape_constraints"]
        assert shapes == sorted(shapes)
        assert "unwitnessed" not in shapes
        expected = ["2x3", "8x5"]
        if schema == "aten::nll_loss_forward.default":
            expected.append("512x5")
        assert shapes == sorted(expected)
