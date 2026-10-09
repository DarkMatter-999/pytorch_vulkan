"""Fresh composed generation compares captures; published history stays immutable."""
import json
from pathlib import Path

from composed_matmul_evidence import capture_composed_evidence, validate_composed_evidence
from mse_capability_evidence import capture_all_mse_cases, qualify_mse_current_runtime

ROOT = Path(__file__).resolve().parents[2]


def test_fresh_composed_manifest_and_model_generation():
    import pytorch_vulkan
    import vulkan_conformance as vc
    from tools import generate_vulkan_capabilities as generator
    mode = pytorch_vulkan._C.execution_mode()
    historical = json.loads((ROOT / "docs/vulkan_coverage.json").read_text())
    expected_mode = historical if mode == "async" else json.loads((ROOT / "docs/vulkan_mse_sync_coverage.json").read_text())
    fresh = capture_all_mse_cases()
    qualify_mse_current_runtime(fresh, ROOT)
    for name, record in fresh.items():
        expected = expected_mode[name]
        # HEAD and path are provenance, never execution identity requirements.
        for key in ("checkout_head", "extension_path"):
            record["mse_autograd_evidence"]["runtime_identity"][key] = expected["mse_autograd_evidence"]["runtime_identity"][key]
        assert record == expected
    with vc.coverage_recording():
        for case in vc.ALL_CASES:
            if case.name.startswith("scalar-division.") or case.name == "loss.mse.rank1":
                result, expected, inputs = vc.run_and_compare(case, return_inputs=True)
                vc.assert_result_parity(result, expected, case)
                if case.check_gradients:
                    vc.assert_gradients(case)
                vc.record_coverage(case, inputs, result, gradients=case.check_gradients, parity=True)
        fresh.update(vc.coverage_snapshot())
    candidate = {**historical, **fresh}
    assert all(candidate[k] == v for k, v in historical.items() if k not in fresh)
    assert (ROOT / "docs/vulkan_capabilities.json").read_text() == generator.render(generator.build_coverage_manifest(candidate))
    published = json.loads((ROOT / "docs/vulkan_composed_matmul_coverage.json").read_text())
    validate_composed_evidence(published, ROOT)
    for name, record in capture_composed_evidence()["records"].items():
        expected = published["records"][name]
        for key in ("checkout_head", "extension_path"):
            record["runtime_identity"][key] = expected["runtime_identity"][key]
        assert record == expected
