import json
from pathlib import Path

import pytest
import pytorch_vulkan
import torch

from tools import generate_vulkan_capabilities as generator
from tools.validate_vulkan_capabilities import load_manifest, schema_exists

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
    assert len(witnessed) == 9
    for name, record in witnessed.items():
        assert name.startswith(("arithmetic.autograd.", "view."))
        assert record["schema"] in {
            "aten::add.Tensor", "aten::add.Scalar", "aten::mul.Tensor",
            "aten::mul.Scalar", "aten::sum.default", "aten::sum.dim_IntList",
            "aten::view.default", "aten::reshape.default",
        }


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
            torch.testing.assert_close(
                result.cpu(), expected, rtol=case.rtol, atol=case.atol, equal_nan=True
            )
            pytorch_vulkan._C.synchronize()
            reverse_names = {
                "arithmetic.autograd.add-tensor", "arithmetic.autograd.add-scalar",
                "arithmetic.autograd.mul-tensor", "arithmetic.autograd.mul-scalar",
                "arithmetic.autograd.sum-default", "arithmetic.autograd.sum-dim",
                "view.view.trainable-seed", "view.reshape.copy.trainable-seed",
                "view.reshape.offset-copy.second-order",
            }
            reverse_autograd = vc.assert_reverse_second_order(case, inputs) if case.name in reverse_names else None
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
