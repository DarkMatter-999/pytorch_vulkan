import pytest
import torch

from tools.vulkan_model_benchmark import (
    AttentionFixture,
    CNNFixture,
    RNNFixture,
    _run_row,
    _apply_parity,
    TIMING_MEAN_REL_TOLERANCE,
    validate_artifact,
)


def _vulkan_row(cpu, **overrides):
    """Build a schema-valid Vulkan row so each test mutates one contract."""
    row = dict(cpu)
    row.update(
        {
            "mode": "vulkan",
            "device": "vk:0",
            "gpu_time": {"status": "unavailable", "samples_ns": None, "mean_ns": None},
            "timing_source": "unavailable",
            "effective_tflops": None,
            "validation": "passed",
            "cpu_parity": {"status": "measured"},
            "parity": {
                "status": "measured",
                "loss_abs_difference": 0.0,
                "parameter_max_abs_difference": 0.0,
            },
            "vulkan_copies": 0,
            "explicit_transfers": 0,
            "transfer_count": 0,
            "fallbacks": 0,
            "transfer_operations": 0,
            "transfer_submissions": 0,
            "transfer_completions": 0,
            "transfer_waits": 0,
            "compute_submissions": 1,
            "fallback_status": False,
            "dispatches": 1,
            "submissions": 1,
            "completions": 1,
            "waits": 1,
        }
    )
    row.update(overrides)
    return row


@pytest.mark.parametrize(
    ("fixture", "input_shape", "target_shape"),
    [
        (CNNFixture(), (8, 3, 32, 32), (8, 10)),
        (AttentionFixture(), (4, 128, 256), (4, 128, 256)),
        (RNNFixture(), (16, 64, 128), (16, 64, 256)),
    ],
    ids=("cnn", "attention", "rnn"),
)
def test_model_fixture_has_exact_f32_inputs_and_backward_trace(fixture, input_shape, target_shape):
    model = fixture.make_cpu()
    inputs, target = fixture.make_inputs()

    assert tuple(inputs.shape) == input_shape
    assert tuple(target.shape) == target_shape
    assert inputs.dtype is torch.float32
    assert target.dtype is torch.float32
    assert all(parameter.dtype is torch.float32 for parameter in model.parameters())
    assert {
        name: tuple(parameter.shape) for name, parameter in model.named_parameters()
    } == fixture.expected_parameter_shapes()

    output = model(inputs)
    loss = fixture.loss(output, target)
    loss.backward()

    trace = fixture.trace()
    expected = fixture.expected_schema_set()
    assert expected["forward"] <= set(trace["forward"])
    assert expected["backward"] <= set(trace["backward"])


def test_attention_benchmark_execution_row_excludes_loss_readback_transfer():
    row = _run_row("attention", AttentionFixture(), "vk:0", warmups=0, repetitions=1)
    assert row["explicit_transfers"] == 0
    assert row["dispatches"] > 0
    assert row["compute_submissions"] == row["repetitions"]
    assert row["transfer_operations"] == row["vulkan_copies"]
    assert row["transfer_submissions"] == 0
    assert row["transfer_completions"] == 0
    assert row["transfer_waits"] == 0
    assert row["fallback_status"] is False


def test_rnn_benchmark_training_step_uses_one_fused_lifecycle():
    row = _run_row("rnn", RNNFixture(sequence_length=2), "vk:0", warmups=0, repetitions=1)

    assert row["dispatches"] == 11
    assert row["compute_submissions"] == 1
    assert row["transfer_operations"] == row["vulkan_copies"]
    assert row["transfer_completions"] == row["transfer_submissions"]
    assert row["transfer_waits"] >= row["transfer_submissions"]
    assert row["explicit_transfers"] == 0
    assert row["fallbacks"] == 0


def test_fixture_contracts_reject_unsupported_dtype_and_dimensions():
    assert CNNFixture().shape == (8, 3, 32, 32)
    assert AttentionFixture().shape == (4, 128, 256)
    assert RNNFixture().shape == (16, 64, 128)

    with pytest.raises(ValueError, match="float32"):
        CNNFixture(dtype=torch.float64)
    with pytest.raises(ValueError, match="batch"):
        AttentionFixture(batch=0)
    with pytest.raises(ValueError, match="hidden_dim"):
        RNNFixture(hidden_dim=0)


def test_benchmark_artifact_requires_matching_cpu_and_vulkan_rows():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    vulkan["parity"] = {"status": "measured", "loss_abs_difference": 0.001, "parameter_max_abs_difference": 0.001}
    validate_artifact(artifact)
    vulkan["shape"] = [1]
    with pytest.raises(ValueError, match="invalid fixture shape"):
        validate_artifact(artifact)


def test_benchmark_artifact_rejects_parity_outside_model_tolerance():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu, parity={"status": "measured", "loss_abs_difference": 0.2437055, "parameter_max_abs_difference": 0.0})
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    with pytest.raises(ValueError, match="parity tolerance"):
        validate_artifact(artifact)
    vulkan["parity"]["loss_abs_difference"] = 0.0
    vulkan["parity"]["parameter_max_abs_difference"] = 0.2437055
    with pytest.raises(ValueError, match="parity tolerance"):
        validate_artifact(artifact)


def test_artifact_rejects_duplicate_rows_and_counter_activity():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan, dict(vulkan)]}
    with pytest.raises(ValueError, match="exactly one CPU and one Vulkan"):
        validate_artifact(artifact)
    artifact["rows"] = [cpu, vulkan]
    vulkan["vulkan_copies"] = 1
    with pytest.raises(ValueError, match="counter contract|transfer operation"):
        validate_artifact(artifact)


def test_artifact_rejects_device_sequence_seed_and_timing_drift():
    cpu = _run_row("attention", AttentionFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    for field, value in (("device", "vulkan:0"), ("sequence_length", 1), ("seed", 3)):
        vulkan[field] = value
        with pytest.raises(ValueError):
            validate_artifact(artifact)
        vulkan[field] = "vk:0" if field == "device" else cpu[field]
    vulkan["gpu_time"] = {"status": "available", "samples_ns": None, "mean_ns": None}
    with pytest.raises(ValueError, match="timing"):
        validate_artifact(artifact)


def test_artifact_rejects_nonfinite_loss_and_host_mean():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    vulkan["loss"] = float("nan")
    with pytest.raises(ValueError, match="loss"):
        validate_artifact(artifact)
    vulkan["loss"] = cpu["loss"]
    vulkan["host_time"]["mean_ns"] = float("nan")
    with pytest.raises(ValueError, match="timing"):
        validate_artifact(artifact)


def test_vk_device_contract_rejects_alias_before_allocation():
    with pytest.raises(ValueError, match="vk:0"):
        _run_row("cnn", CNNFixture(), "vk:1", warmups=0, repetitions=1)


def test_parity_helper_records_measured_differences():
    cpu = {"loss": 2.0}
    vulkan = {"loss": 2.25}
    cpu_state = {"weight": torch.tensor([1.0, 2.0])}
    vulkan_state = {"weight": torch.tensor([1.5, 2.0])}
    _apply_parity(vulkan, cpu, cpu_state, vulkan_state)
    assert vulkan["parity"] == {"status": "measured", "loss_abs_difference": 0.25, "parameter_max_abs_difference": 0.5}


def test_artifact_rejects_schema_model_and_exact_fixture_shape_drift():
    cpu = _run_row("attention", AttentionFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    for mutation, message in (({"schema_version": 2}, "schema"), ({"model": "unknown"}, "model"), ({"shape": [4, 128, 255]}, "invalid fixture shape"), ({"batch": 3}, "invalid fixture shape"), ({"sequence_length": 127}, "invalid fixture shape")):
        candidate = dict(artifact)
        candidate["rows"] = [dict(cpu), dict(vulkan)]
        candidate.update({key: value for key, value in mutation.items() if key == "schema_version"})
        if "model" in mutation or "shape" in mutation or "batch" in mutation or "sequence_length" in mutation:
            candidate["rows"][1].update(mutation)
        with pytest.raises(ValueError, match=message):
            validate_artifact(candidate)


def test_artifact_rejects_malformed_counter_types_and_cpu_activity():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    for row, field, value in ((cpu, "dispatches", 1), (vulkan, "dispatches", True), (vulkan, "waits", -1), (vulkan, "fallbacks", 1)):
        row[field] = value
        with pytest.raises(ValueError, match="counter"):
            validate_artifact(artifact)
        row[field] = 0 if row is cpu or field in {"fallbacks", "dispatches", "waits"} else value
    vulkan.update({"dispatches": 1, "waits": 1, "fallbacks": 0})


def test_artifact_rejects_inconsistent_transfer_lifecycle_counters():
    cpu = _run_row("attention", AttentionFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    invalid_rows = (
        {"transfer_operations": 1, "vulkan_copies": 0},
        {"transfer_submissions": 2, "transfer_completions": 1},
        {"transfer_submissions": 1, "transfer_completions": 1, "transfer_waits": 0},
        {"transfer_operations": 1, "transfer_submissions": 0, "transfer_completions": 0, "transfer_waits": 0},
    )
    for mutation in invalid_rows:
        candidate = {"schema_version": 1, "seed": 1729, "rows": [dict(cpu), dict(vulkan)]}
        candidate["rows"][1].update(mutation)
        with pytest.raises(ValueError, match="transfer|counter"):
            validate_artifact(candidate)

    valid = {"schema_version": 1, "seed": 1729, "rows": [dict(cpu), dict(vulkan)]}
    valid["rows"][1].update({
        "vulkan_copies": 1,
        "transfer_operations": 1,
        "transfer_submissions": 1,
        "transfer_completions": 1,
        "transfer_waits": 2,
    })
    validate_artifact(valid)


def test_artifact_accepts_compute_scoped_transfer_operations_without_standalone_submission():
    cpu = _run_row("attention", AttentionFixture(), "cpu", warmups=0, repetitions=3)
    vulkan = _vulkan_row(
        cpu,
        vulkan_copies=3,
        transfer_operations=3,
        transfer_submissions=0,
        transfer_completions=0,
        transfer_waits=0,
    )
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    validate_artifact(artifact)


def test_artifact_rejects_standalone_transfer_lifecycle_over_operation_count():
    cpu = _run_row("rnn", RNNFixture(), "cpu", warmups=0, repetitions=3)
    vulkan = _vulkan_row(
        cpu,
        vulkan_copies=198,
        transfer_operations=198,
        transfer_submissions=405_504,
        transfer_completions=405_504,
        transfer_waits=405_699,
    )
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}

    with pytest.raises(ValueError, match="transfer submissions exceed transfer operations"):
        validate_artifact(artifact)


def test_artifact_requires_consistent_timing_means():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    vulkan["host_time"]["mean_ns"] = cpu["host_time"]["samples_ns"][0] + max(1.0, abs(cpu["host_time"]["samples_ns"][0]) * TIMING_MEAN_REL_TOLERANCE) * 2
    with pytest.raises(ValueError, match="timing mean"):
        validate_artifact(artifact)


def test_artifact_validates_available_gpu_timing_shape_and_mean():
    cpu = _run_row("cnn", CNNFixture(), "cpu", warmups=0, repetitions=1)
    vulkan = _vulkan_row(cpu, gpu_time={"status": "available", "samples_ns": [10.0], "mean_ns": 10.0}, timing_source="gpu_timestamp", effective_tflops=1.0)
    artifact = {"schema_version": 1, "seed": 1729, "rows": [cpu, vulkan]}
    validate_artifact(artifact)
    vulkan["gpu_time"]["samples_ns"] = []
    with pytest.raises(ValueError, match="GPU timing samples"):
        validate_artifact(artifact)
