from contextlib import contextmanager
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from mnist_training_cases import evaluate, make_model, synthetic_batch, train
from mnist_training_parity import (
    TrainingObserver,
    _real_observers,
    _compare_evaluations,
    _compare_metadata,
    _device_matches,
    _observer_result,
    assert_vulkan_tensor_metadata,
    compare_runs,
)


class _FakeCounters:
    def __init__(self, counters=None):
        self.current = tuple(counters or (0, 0, 0, 0))
        self.events = []

    def synchronize(self):
        self.events.append("synchronize")

    def reset_execution_counters(self):
        self.events.append("reset")
        self.current = (0, 0, 0, 0)

    def execution_counter_snapshot(self):
        self.events.append("snapshot-counters")
        return self.current


class _FakeBackend:
    def __init__(self, counters=None):
        self._C = _FakeCounters(counters)


def test_phase_synchronizes_resets_and_captures_counts_without_nested_phase():
    backend = _FakeBackend()
    observer = TrainingObserver("cpu", backend=backend)
    with observer.phase("step-0.forward-backward"):
        assert backend._C.events == ["synchronize", "reset"]
        backend._C.current = (4, 2, 0, 0)

    assert backend._C.events == [
        "synchronize", "reset", "synchronize", "snapshot-counters"
    ]
    assert observer.counters == {"step-0.forward-backward": [4, 2, 0, 0]}


def test_observer_rejects_nested_counter_reset_intervals():
    backend = _FakeBackend()
    observer = TrainingObserver("cpu", backend=backend)
    with observer.phase("step-0.forward-backward"):
        with pytest.raises(AssertionError, match="nested observer phase"):
            with observer.phase("step-0.backward-diagnostics"):
                pass
    assert backend._C.events.count("reset") == 1


def test_observe_validates_live_tensor_before_snapshot_and_keeps_metadata(monkeypatch):
    backend = _FakeBackend()
    observer = TrainingObserver("cpu", backend=backend)
    ordering = backend._C.events
    original_to = torch.Tensor.to

    def record_readback(tensor, *args, **kwargs):
        ordering.append("readback")
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_readback)
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    live = source * 3
    logits = live + 1
    with observer.phase("step-0.backward-diagnostics"):
        observer.observe("step-0.backward", {"logits": logits, "loss": live})
    assert ordering.index("synchronize") < ordering.index("readback")
    source.data.zero_()

    assert torch.equal(observer.snapshots["step-0.backward"]["loss"], torch.tensor([3.0, 6.0]))
    assert observer.tensor_metadata["step-0.backward"]["loss"] == {
        "device": "cpu", "dtype": "torch.float32", "shape": [2],
        "requires_grad": True, "history": True,
    }


def test_observe_rejects_wrong_residency_before_readback():
    observer = TrainingObserver("vk:0")
    with pytest.raises(AssertionError, match="residency"):
        observer.observe("step-0.backward", {"loss": torch.tensor(1.0)})
    assert observer.snapshots == {}


def test_device_matching_uses_exact_torch_device_object():
    cpu_tensor = torch.tensor([1.0])
    assert _device_matches(cpu_tensor, torch.device("cpu"))
    assert not _device_matches(cpu_tensor, torch.device("cpu:1"))
    assert not _device_matches(cpu_tensor, torch.device("meta"))


def test_reset_zero_metric_runs_only_after_cpu_diagnostic_readback(monkeypatch):
    observer = TrainingObserver("cpu")
    events = []
    original_to = torch.Tensor.to
    original_count_nonzero = torch.count_nonzero

    def record_readback(tensor, *args, **kwargs):
        events.append("readback")
        return original_to(tensor, *args, **kwargs)

    def record_metric(tensor, *args, **kwargs):
        events.append("metric")
        assert tensor.device.type == "cpu"
        return original_count_nonzero(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_readback)
    monkeypatch.setattr(torch, "count_nonzero", record_metric)
    with observer.phase("step-0.reset-diagnostics"):
        observer.observe("step-0.reset", {"gradients": {"weight": torch.zeros(2)}})

    assert events.index("readback") < events.index("metric")


def test_recipe_observations_follow_exact_reset_backward_update_order():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    observer = TrainingObserver("cpu")
    train(model, optimizer, lambda: [synthetic_batch(1)], device="cpu", epochs=1,
          set_to_none=True, observer=observer)

    assert observer.phase_order == [
        "step-0.staging", "step-0.reset", "step-0.reset-diagnostics",
        "step-0.forward-backward", "step-0.backward-diagnostics",
        "step-0.update", "step-0.update-diagnostics",
    ]
    assert list(observer.snapshots) == [
        "step-0.reset", "step-0.backward", "step-0.update",
    ]


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda row: row["parameters"].update({"extra": torch.tensor([1.0])}), "keys"),
        (lambda row: row["state"].update({"extra": {"momentum_buffer": torch.ones(2)}}), "keys"),
        (lambda row: row["parameters"].update({"weight": torch.ones(2, dtype=torch.float64)}), "dtype"),
        (lambda row: row["parameters"].update({"weight": torch.ones(2, 1)}), "shape"),
        (lambda row: row["state"]["weight"].update({"momentum_buffer": None}), "state"),
    ],
)
def test_compare_runs_rejects_structure_dtype_shape_and_state_mismatches(change, message):
    expected = TrainingObserver("cpu")
    actual = TrainingObserver("vk:0")
    expected.snapshots["step-0.update"] = {
        "parameters": {"weight": torch.ones(2)},
        "state": {"weight": {"momentum_buffer": torch.ones(2)}},
        "optimizer_options": {"parameter_names": ["weight"], "lr": 0.01},
    }
    actual.snapshots["step-0.update"] = {
        "parameters": {"weight": torch.ones(2)},
        "state": {"weight": {"momentum_buffer": torch.ones(2)}},
        "optimizer_options": {"parameter_names": ["weight"], "lr": 0.01},
    }
    change(actual.snapshots["step-0.update"])

    with pytest.raises(AssertionError, match=message):
        compare_runs(actual, expected)


def test_compare_runs_rejects_missing_gradient_or_momentum_and_nonfinite_values():
    expected = TrainingObserver("cpu")
    actual = TrainingObserver("vk:0")
    expected.snapshots["step-0.backward"] = {"gradients": {"weight": torch.ones(2)}}
    actual.snapshots["step-0.backward"] = {"gradients": {"weight": None}}
    with pytest.raises(AssertionError, match="gradient"):
        compare_runs(actual, expected)

    actual.snapshots["step-0.backward"] = {"gradients": {"weight": torch.tensor([float("nan"), 1.0])}}
    expected.snapshots["step-0.backward"] = {"gradients": {"weight": torch.ones(2)}}
    with pytest.raises(AssertionError, match="finite"):
        compare_runs(actual, expected)

    actual.snapshots.clear()
    expected.snapshots.clear()
    actual.snapshots["step-0.update"] = {
        "parameters": {"weight": torch.ones(2)},
        "state": {"weight": {"momentum_buffer": None}},
        "optimizer_options": {"parameter_names": ["weight"]},
    }
    expected.snapshots["step-0.update"] = {
        "parameters": {"weight": torch.ones(2)},
        "state": {"weight": {"momentum_buffer": torch.ones(2)}},
        "optimizer_options": {"parameter_names": ["weight"]},
    }
    with pytest.raises(AssertionError, match="momentum"):
        compare_runs(actual, expected)


def test_compare_runs_uses_parameter_tolerance_and_reset_none_vs_zero_contract():
    expected = TrainingObserver("cpu")
    actual = TrainingObserver("vk:0")
    expected.snapshots["step-0.update"] = {"parameters": {"weight": torch.ones(2)}}
    actual.snapshots["step-0.update"] = {"parameters": {"weight": torch.tensor([1.0, 1.0002])}}
    with pytest.raises(AssertionError, match="close"):
        compare_runs(actual, expected)

    actual.snapshots.clear()
    expected.snapshots.clear()
    expected.snapshots["step-0.reset"] = {"gradients": {"weight": None}}
    actual.snapshots["step-0.reset"] = {"gradients": {"weight": torch.zeros(2)}}
    with pytest.raises(AssertionError, match="reset"):
        compare_runs(actual, expected)


def test_diagnostic_transfers_are_recorded_separately_from_compute_accounting():
    observer = TrainingObserver("vk:0", backend=_FakeBackend())
    with observer.phase("step-0.forward-backward"):
        observer.backend._C.current = (3, 0, 0, 0)
    with observer.phase("step-0.backward-diagnostics"):
        observer.backend._C.current = (0, 0, 1, 0)

    assert observer.counters["step-0.forward-backward"] == [3, 0, 0, 0]
    assert observer.counters["step-0.backward-diagnostics"] == [0, 0, 1, 0]
    assert observer.counter_categories["compute"] == {"step-0.forward-backward"}
    assert observer.counter_categories["diagnostics"] == {"step-0.backward-diagnostics"}
    observer.validate_accounting()


def test_model_setup_transfer_counters_survive_first_training_reset():
    backend = _FakeBackend()
    observer = TrainingObserver("vk:0", backend=backend)
    with observer.phase("setup.model-staging"):
        backend._C.current = (0, 2, 2, 0)
    with observer.phase("step-0.forward-backward"):
        backend._C.current = (3, 0, 0, 0)

    assert observer.phase_routes["setup.model-staging"] == "setup"
    assert observer.counter_categories["setup"] == {"setup.model-staging"}
    assert observer.counters["setup.model-staging"] == [0, 2, 2, 0]
    assert observer.counters["step-0.forward-backward"] == [3, 0, 0, 0]
    observer.validate_accounting()
    summary = _observer_result(observer)
    assert summary["phase_routes"]["setup.model-staging"] == "setup"
    assert summary["counter_categories"]["setup"] == ["setup.model-staging"]
    assert summary["counters"]["setup.model-staging"] == [0, 2, 2, 0]


def test_model_setup_counter_interval_rejects_fallback_but_allows_transfers():
    observer = TrainingObserver("vk:0", backend=_FakeBackend())
    with observer.phase("setup.model-staging"):
        observer.backend._C.current = (0, 1, 1, 1)

    with pytest.raises(AssertionError, match="setup fallback"):
        observer.validate_accounting()


def test_metadata_comparison_rejects_sequence_length_and_type_mismatches():
    metadata = {"dtype": "torch.float32", "shape": [1], "requires_grad": False, "history": False}
    with pytest.raises(AssertionError, match="sequence length"):
        _compare_metadata([metadata], [], ("metadata",))
    with pytest.raises(AssertionError, match="sequence type"):
        _compare_metadata([metadata], (metadata,), ("metadata",))


def test_manual_acceptance_residency_assertion_walks_nested_metadata_and_rejects_cpu():
    metadata = {
        "step-0.backward": {
            "gradients": {
                "weight": {"device": "vk:0", "dtype": "torch.float32"},
                "bias": [{"device": "vk:0", "dtype": "torch.float32"}],
            }
        },
        "epoch-0.evaluation-0": {
            "logits": {"device": "vk:0", "dtype": "torch.float32"}
        },
    }
    assert_vulkan_tensor_metadata(metadata)

    metadata["step-0.backward"]["gradients"]["bias"][0]["device"] = "cpu"
    with pytest.raises(AssertionError):
        assert_vulkan_tensor_metadata(metadata)


def test_cpu_training_process_does_not_import_or_initialize_vulkan():
    repository = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(repository), str(repository / "tests/python")))
    script = r"""
import sys
import torch
from mnist_training_cases import make_model, synthetic_batch

assert "pytorch_vulkan" not in sys.modules
assert torch.optim.SGD.__module__ == "torch.optim.sgd"
model = make_model("cpu")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
images, labels = synthetic_batch(2)
torch.nn.CrossEntropyLoss()(model(images), labels).backward()
optimizer.step()
assert all(parameter.device.type == "cpu" for parameter in model.parameters())
assert "pytorch_vulkan" not in sys.modules
assert torch.optim.SGD.__module__ == "torch.optim.sgd"
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=repository, env=env,
        text=True, capture_output=True, timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_two_epoch_callbacks_evaluate_in_order_and_restore_training_mode():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    observer = TrainingObserver("cpu")
    evaluations = []

    def after_epoch(epoch):
        assert epoch == len(evaluations)
        result = evaluate(
            model, lambda: [synthetic_batch(2), synthetic_batch(1)],
            device="cpu", observer=observer, epoch=epoch,
        )
        evaluations.append(result)
        assert model.training

    train(model, optimizer, lambda: [synthetic_batch(1)], device="cpu", epochs=2,
          set_to_none=True, observer=observer, after_epoch=after_epoch)

    events = list(observer.snapshots)
    assert [name for name in events if name.endswith(".update")] == ["step-0.update", "step-1.update"]
    assert [name for name in events if name.startswith("epoch-")] == [
        "epoch-0.evaluation-0", "epoch-0.evaluation-1",
        "epoch-1.evaluation-0", "epoch-1.evaluation-1",
    ]
    assert events.index("step-0.update") < events.index("epoch-0.evaluation-0")
    assert events.index("epoch-0.evaluation-1") < events.index("step-1.reset")
    assert len(evaluations) == 2 and all(result["count"] == 3 for result in evaluations)


def test_evaluation_is_no_grad_and_does_not_mutate_existing_gradients():
    model = make_model()
    observer = TrainingObserver("cpu")
    for index, parameter in enumerate(model.parameters()):
        parameter.grad = torch.full_like(parameter, index + 1.0)
    before = [parameter.grad.clone() for parameter in model.parameters()]
    result = evaluate(model, lambda: [synthetic_batch(2)], device="cpu", observer=observer)

    assert model.training
    assert result["count"] == 2
    assert all(torch.equal(parameter.grad, prior) for parameter, prior in zip(model.parameters(), before))
    logits_metadata = observer.tensor_metadata["epoch-0.evaluation-0"]["logits"]
    assert logits_metadata["requires_grad"] is False and logits_metadata["history"] is False


def test_evaluation_aggregates_nine_batches_including_single_item_final_batch():
    observer = TrainingObserver("cpu")
    batches = [synthetic_batch(16) for _ in range(8)] + [synthetic_batch(1)]
    result = evaluate(make_model(), lambda: batches, device="cpu", observer=observer)

    assert [name for name in observer.snapshots if name.startswith("epoch-")] == [
        f"epoch-0.evaluation-{index}" for index in range(9)
    ]
    assert result["count"] == 129
    assert len(result["predictions"]) == 129
    assert result["correct"] == sum(
        prediction == int(label)
        for prediction, label in zip(result["predictions"], torch.cat([labels for _, labels in batches]).tolist())
    )


@pytest.mark.parametrize(
    "actual,expected,message",
    [
        ({"predictions": [1], "correct": 1, "count": 1}, {"predictions": [0], "correct": 0, "count": 1}, "predictions"),
        ({"predictions": [1], "correct": 0, "count": 1}, {"predictions": [1], "correct": 1, "count": 1}, "correct"),
        ({"predictions": [1], "correct": 1, "count": 2}, {"predictions": [1], "correct": 1, "count": 1}, "count"),
    ],
)
def test_evaluation_comparison_rejects_exact_prediction_and_metric_mismatches(actual, expected, message):
    with pytest.raises(AssertionError, match=message):
        _compare_evaluations(actual, expected)


@pytest.mark.parametrize("set_to_none", [True, False])
def test_real_observers_cpu_integration_runs_bounded_two_epoch_pair(set_to_none):
    generator = torch.Generator(device="cpu").manual_seed(42012)
    train_images = torch.rand((257, 1, 28, 28), generator=generator, dtype=torch.float32)
    train_labels = torch.arange(257, dtype=torch.int64) % 10
    test_images = torch.rand((129, 1, 28, 28), generator=generator, dtype=torch.float32)
    test_labels = torch.arange(129, dtype=torch.int64) % 10
    train_loader = DataLoader(
        TensorDataset(train_images, train_labels), batch_size=16, shuffle=False,
        drop_last=False, num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(test_images, test_labels), batch_size=16, shuffle=False,
        drop_last=False, num_workers=0,
    )
    model = make_model("cpu")

    evaluation_gradients = []

    def before_forward(module, inputs):
        if not module.training:
            assert not torch.is_grad_enabled()
            evaluation_gradients.append(
                [parameter.grad.detach().clone() for parameter in module.parameters()]
            )

    def after_forward(module, inputs, output):
        if not module.training:
            for parameter, before in zip(module.parameters(), evaluation_gradients[-1]):
                assert torch.equal(parameter.grad, before)

    pre_hook = model.register_forward_pre_hook(before_forward)
    post_hook = model.register_forward_hook(after_forward)
    try:
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        observer, train_sizes, evaluation_sizes, evaluations = _real_observers(
            model, optimizer, "cpu", None, train_loader, test_loader, set_to_none
        )
    finally:
        pre_hook.remove()
        post_hook.remove()

    assert model.training
    assert len(evaluations) == 2
    assert [result["count"] for result in evaluations] == [129, 129]
    assert train_sizes == ([16] * 16 + [1]) * 2
    assert evaluation_sizes == ([16] * 8 + [1]) * 2
    assert len(evaluation_gradients) == 18
    assert sum(name.endswith(".update") for name in observer.snapshots) == 34
    assert sum(name.endswith(".backward") for name in observer.snapshots) == 34
    assert observer.phase_order.index("epoch-0.evaluation-staging-0") > observer.phase_order.index(
        "step-16.update-diagnostics"
    )
    assert observer.phase_order.index("step-17.staging") > observer.phase_order.index(
        "epoch-0.evaluation-readback-8"
    )
    assert observer.phase_order.index("epoch-1.evaluation-staging-0") > observer.phase_order.index(
        "step-33.update-diagnostics"
    )
    assert [name for name in observer.snapshots if name.startswith("epoch-")] == [
        f"epoch-{epoch}.evaluation-{batch}" for epoch in range(2) for batch in range(9)
    ]
    for name, metadata in observer.tensor_metadata.items():
        if name.startswith("epoch-"):
            assert metadata["logits"]["requires_grad"] is False
            assert metadata["logits"]["history"] is False
            assert observer.snapshots[name]["logits"].shape[0] in (1, 16)
            assert set(observer.snapshots[name]) == {"logits"}
