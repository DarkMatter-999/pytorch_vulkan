import pytest
import copy

from tools.vulkan_model_benchmark import (
    AttentionFixture,
    CNNFixture,
    RNNFixture,
    validate_artifact,
)


def test_cnn_slice_schema_is_exact():
    fixture = CNNFixture()
    assert fixture.shape == (8, 3, 32, 32)
    assert fixture.target_shape == (8, 10)
    assert fixture.expected_parameter_shapes() == {
        "0.weight": (8, 3, 3, 3),
        "0.bias": (8,),
        "3.weight": (10, 8 * 32 * 32),
        "3.bias": (10,),
    }


@pytest.mark.parametrize("batch", [0, -1])
def test_cnn_slice_rejects_invalid_batch(batch):
    with pytest.raises(ValueError, match="batch"):
        CNNFixture(batch=batch)


def test_attention_slice_schema_is_exact():
    fixture = AttentionFixture()
    model = fixture.make_cpu()
    assert fixture.shape == (4, 128, 256)
    assert fixture.target_shape == (4, 128, 256)
    assert fixture.expected_parameter_shapes() == {
        "query.weight": (256, 256), "query.bias": (256,),
        "key.weight": (256, 256), "key.bias": (256,),
        "value.weight": (256, 256), "value.bias": (256,),
        "output.weight": (256, 256), "output.bias": (256,),
    }
    assert fixture.mask_shape == (4, 128, 128)
    assert model.causal_mask.shape == fixture.mask_shape
    assert model.causal_mask.dtype is fixture.dtype
    assert model.causal_mask.is_contiguous()
    assert model.causal_mask[0, 0, 0] == 0
    assert model.causal_mask[0, 0, 1] == -10000


def test_rnn_slice_schema_is_exact():
    fixture = RNNFixture()
    assert fixture.shape == (16, 64, 128)
    assert fixture.target_shape == (16, 64, 256)
    assert fixture.expected_parameter_shapes() == {
        "input.weight": (256, 128),
        "input.bias": (256,),
        "hidden.weight": (256, 256),
    }


def test_model_artifact_schema_requires_saturation_evidence():
    artifact = {
        "schema_version": 1,
        "seed": 1729,
        "rows": [
            {
                "model": "cnn", "mode": "cpu", "device": "cpu",
                "dtype": "float32", "shape": [8, 3, 32, 32], "batch": 8,
                "sequence_length": None, "warmups": 2, "repetitions": 1,
                "seed": 1729, "host_time": {"samples_ns": [1], "mean_ns": 1},
                "gpu_time": {"status": "not_applicable", "samples_ns": None, "mean_ns": None},
                "loss": 1.0,
                "parity": {"status": "reference", "loss_abs_difference": None, "parameter_max_abs_difference": None},
                "dispatches": 0, "vulkan_copies": 0, "explicit_transfers": 0,
                "fallbacks": 0, "submissions": 0, "completions": 0, "waits": 0,
            },
            {
                "model": "cnn", "mode": "vulkan", "device": "vk:0",
                "dtype": "float32", "shape": [8, 3, 32, 32], "batch": 8,
                "sequence_length": None, "warmups": 2, "repetitions": 1,
                "seed": 1729, "host_time": {"samples_ns": [1], "mean_ns": 1},
                "gpu_time": {"status": "unavailable", "samples_ns": None, "mean_ns": None},
                "loss": 1.0,
                "parity": {"status": "measured", "loss_abs_difference": 0.0, "parameter_max_abs_difference": 0.0},
                "dispatches": 1, "vulkan_copies": 0, "explicit_transfers": 0,
                "fallbacks": 0, "submissions": 1, "completions": 1, "waits": 1,
            },
        ],
    }
    with pytest.raises(ValueError, match="saturation|arithmetic|validation"):
        validate_artifact(copy.deepcopy(artifact))


def test_model_aggregate_schema_requires_all_nested_model_artifacts():
    from tools.vulkan_model_benchmark import validate_aggregate_artifact

    with pytest.raises(ValueError, match="cnn|attention|rnn|models"):
        validate_aggregate_artifact({"schema_version": 1, "seed": 1729, "models": {}})


def test_model_aggregate_schema_validates_each_nested_result():
    from tools.vulkan_model_benchmark import FIXTURES, validate_aggregate_artifact

    models = {}
    for name, fixture_type in FIXTURES.items():
        fixture = fixture_type()
        common = {
            "model": name, "dtype": "float32", "shape": list(fixture.shape),
            "batch": fixture.batch, "sequence_length": getattr(fixture, "sequence_length", None),
            "warmups": 0, "repetitions": 1, "seed": 1729,
            "host_time": {"samples_ns": [1], "mean_ns": 1},
            "arithmetic_operations": fixture.arithmetic_operations(),
            "validation": "passed", "transfer_count": 0,
            "vulkan_copies": 0, "explicit_transfers": 0, "fallbacks": 0,
            "transfer_operations": 0, "transfer_submissions": 0,
            "transfer_completions": 0, "transfer_waits": 0,
            "compute_submissions": 0, "fallback_status": False,
        }
        cpu = {
            **common, "mode": "cpu", "device": "cpu", "gpu_time": {
                "status": "not_applicable", "samples_ns": None, "mean_ns": None,
            }, "timing_source": "host_wall", "effective_tflops": 1.0,
            "loss": 1.0, "cpu_parity": {"status": "reference"},
            "parity": {"status": "reference", "loss_abs_difference": None, "parameter_max_abs_difference": None},
            "dispatches": 0, "submissions": 0, "compute_submissions": 0,
            "completions": 0, "waits": 0,
        }
        vulkan = {
            **common, "mode": "vulkan", "device": "vk:0", "gpu_time": {
                "status": "unavailable", "samples_ns": None, "mean_ns": None,
            }, "timing_source": "unavailable", "effective_tflops": None,
            "loss": 1.0, "cpu_parity": {"status": "measured"},
            "parity": {"status": "measured", "loss_abs_difference": 0.0, "parameter_max_abs_difference": 0.0},
            "dispatches": 1, "submissions": 1, "compute_submissions": 1,
            "completions": 1, "waits": 1,
        }
        models[name] = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    artifact = validate_aggregate_artifact({"schema_version": 1, "seed": 1729, "models": models})
    assert set(artifact["models"]) == {"cnn", "attention", "rnn"}
    pending = copy.deepcopy(artifact)
    pending["models"]["cnn"]["rows"][1]["cpu_parity"] = {"status": "pending"}
    with pytest.raises(ValueError, match="parity|validation"):
        validate_aggregate_artifact(pending)
    host_labeled_gpu = copy.deepcopy(artifact)
    host_labeled_gpu["models"]["cnn"]["rows"][1]["timing_source"] = "host_wall"
    with pytest.raises(ValueError, match="timing source|effective"):
        validate_aggregate_artifact(host_labeled_gpu)


def test_model_artifact_requires_explicit_transfer_and_scope_counters():
    from tools.vulkan_model_benchmark import FIXTURES, validate_aggregate_artifact

    models = {}
    for name, fixture_type in FIXTURES.items():
        fixture = fixture_type()
        common = {
            "model": name, "dtype": "float32", "shape": list(fixture.shape),
            "batch": fixture.batch, "sequence_length": getattr(fixture, "sequence_length", None),
            "warmups": 0, "repetitions": 1, "seed": 1729,
            "host_time": {"samples_ns": [1], "mean_ns": 1},
            "arithmetic_operations": fixture.arithmetic_operations(),
            "validation": "passed", "transfer_count": 0,
            "vulkan_copies": 0, "explicit_transfers": 0, "fallbacks": 0,
            "transfer_operations": 0, "transfer_submissions": 0,
            "transfer_completions": 0, "transfer_waits": 0,
            "compute_submissions": 0, "fallback_status": False,
        }
        cpu = {
            **common, "mode": "cpu", "device": "cpu", "gpu_time": {
                "status": "not_applicable", "samples_ns": None, "mean_ns": None,
            }, "timing_source": "host_wall", "effective_tflops": 1.0,
            "loss": 1.0, "cpu_parity": {"status": "reference"},
            "parity": {"status": "reference", "loss_abs_difference": None, "parameter_max_abs_difference": None},
            "dispatches": 0, "submissions": 0, "compute_submissions": 0,
            "completions": 0, "waits": 0,
        }
        vulkan = {
            **common, "mode": "vulkan", "device": "vk:0", "gpu_time": {
                "status": "unavailable", "samples_ns": None, "mean_ns": None,
            }, "timing_source": "unavailable", "effective_tflops": None,
            "loss": 1.0, "cpu_parity": {"status": "measured"},
            "parity": {"status": "measured", "loss_abs_difference": 0.0, "parameter_max_abs_difference": 0.0},
            "dispatches": 1, "submissions": 1, "compute_submissions": 1,
            "completions": 1, "waits": 1,
        }
        models[name] = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    artifact = validate_aggregate_artifact({"schema_version": 1, "seed": 1729, "models": models})
    assert artifact["models"]["rnn"]["rows"][1]["fallback_status"] is False
    missing = copy.deepcopy(artifact)
    del missing["models"]["attention"]["rows"][1]["transfer_operations"]
    with pytest.raises(ValueError, match="transfer_operations"):
        validate_aggregate_artifact(missing)
