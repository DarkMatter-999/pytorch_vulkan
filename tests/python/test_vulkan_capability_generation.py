import json
from pathlib import Path

import pytest

from tools import generate_vulkan_capabilities as generator

ROOT = Path(__file__).resolve().parents[2]
COMMITTED = ROOT / "docs/vulkan_capabilities.json"
DERIVED_FIELDS = frozenset({"schema", "test_cases", "tests", "shape_constraints"})


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
    assert len(roadmap) == 85
    assert "aten::_cat.default" in roadmap


def test_roadmap_declarations_have_no_implementation_witnesses():
    committed = _committed_entries()
    for schema in generator._roadmap_schemas():
        declaration = generator.DECLARATIONS[schema]
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
    assert len(supported) == 77
    assert supported == witnessed
