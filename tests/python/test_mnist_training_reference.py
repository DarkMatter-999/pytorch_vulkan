import copy
import hashlib
from contextlib import contextmanager

import pytest
import torch

from mnist_training_cases import evaluate, make_model, synthetic_batch, train
from tools import probe_mnist_training_reference as reference_probe


def test_recipe_and_rng_are_reproducible():
    before = torch.random.get_rng_state().clone()
    a, b = make_model(), make_model()
    assert torch.equal(before, torch.random.get_rng_state())
    assert sum(p.numel() for p in a.parameters()) == 25450
    assert list(a.state_dict()) == ["1.weight", "1.bias", "3.weight", "3.bias"]
    for key in a.state_dict():
        torch.testing.assert_close(a.state_dict()[key], b.state_dict()[key], rtol=0, atol=0)
    x, y = synthetic_batch(2)
    assert x.shape == (2, 1, 28, 28) and y.tolist() == [0, 9]
    assert y.dtype == torch.int64 and x.device.type == "cpu"
    assert a(x).shape == (2, 10)


def test_model_seeding_avoids_all_device_manual_seed_and_preserves_old_weights(monkeypatch):
    before = torch.random.get_rng_state().clone()

    def reject_all_device_seed(*args, **kwargs):
        raise AssertionError("all-device torch.manual_seed must not run")

    monkeypatch.setattr(torch, "manual_seed", reject_all_device_seed)
    model = make_model()
    fingerprint = hashlib.sha256(
        b"".join(
            tensor.detach().contiguous().numpy().tobytes()
            for tensor in model.state_dict().values()
        )
    ).hexdigest()

    assert fingerprint == "02e5f53ffc56e435b5c085b774e1701aca7c5ecc75821627b0f11b385e4dc402"
    assert torch.equal(before, torch.random.get_rng_state())


@pytest.mark.parametrize("batch_size,labels", [(1, [9]), (2, [0, 9]), (16, list(range(10)) + list(range(6)))])
def test_synthetic_fixture_has_versioned_cpu_inputs(batch_size, labels):
    images, targets = synthetic_batch(batch_size)
    generator = torch.Generator(device="cpu").manual_seed(12013 + batch_size)
    expected = torch.rand((batch_size, 1, 28, 28), generator=generator, dtype=torch.float32)
    torch.testing.assert_close(images, expected, rtol=0, atol=0)
    assert images.dtype == torch.float32 and images.device.type == "cpu"
    assert targets.dtype == torch.int64 and targets.tolist() == labels


def test_synthetic_fixture_rejects_unqualified_batch_sizes():
    with pytest.raises(ValueError, match="batch_size"):
        synthetic_batch(3)


class _RecordingObserver:
    def __init__(self):
        self.events = []
        self.phases = []
        self.active_phases = []
        self.received = []
        self.parameters = None
        self.live_gradient_identity = {}
        self.live_parameter_identity = {}
        self.live_state_identity = {}

    @contextmanager
    def phase(self, name):
        self.phases.append(name)
        self.active_phases.append(name)
        try:
            yield
        finally:
            assert self.active_phases.pop() == name

    def observe(self, name, values):
        self.received.append((name, values, tuple(self.active_phases)))
        if self.parameters is not None and name.endswith(".backward"):
            self.live_gradient_identity[name] = {
                key: values["gradients"][key] is parameter.grad
                for key, parameter in self.parameters.items()
            }
        if self.parameters is not None and name.endswith(".update"):
            self.live_parameter_identity[name] = {
                key: values["parameters"][key] is parameter
                for key, parameter in self.parameters.items()
            }
            self.live_state_identity[name] = {
                key: values["state"][key]["momentum_buffer"] is
                self._optimizer.state[parameter]["momentum_buffer"]
                for key, parameter in self.parameters.items()
            }
        self.events.append((name, _snapshot_for_test(values)))


def _snapshot_for_test(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot_for_test(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot_for_test(item) for item in value)
    return value


@pytest.mark.parametrize("set_to_none", [True, False])
def test_three_steps_observe_reset_backward_and_update_snapshots(set_to_none):
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    observer = _RecordingObserver()
    batches = lambda: [synthetic_batch(2)] * 3

    train(model, optimizer, batches, device="cpu", epochs=1,
          set_to_none=set_to_none, observer=observer)

    resets = [values for name, values in observer.events if name.endswith(".reset")]
    assert len(resets) == 3
    assert all(value is None for value in resets[0]["gradients"].values())
    for reset in resets[1:]:
        if set_to_none:
            assert all(value is None for value in reset["gradients"].values())
            assert not any(reset["retained_objects"].values())
        else:
            assert all(torch.count_nonzero(value).item() == 0 for value in reset["gradients"].values())
            assert all(reset["retained_objects"].values())

    backward = [values for name, values in observer.events if name.endswith(".backward")]
    updates = [values for name, values in observer.events if name.endswith(".update")]
    assert len(backward) == len(updates) == 3
    assert all(tuple(item["gradients"]) == ("1.weight", "1.bias", "3.weight", "3.bias") for item in backward)
    assert all("momentum_buffer" in state for update in updates for state in update["state"].values())
    assert all(t.device.type == "cpu" for item in backward for t in item["gradients"].values())
    assert "step-0.staging" in observer.phases and "step-2.update" in observer.phases


def test_training_observes_live_values_in_distinct_diagnostic_phases_and_preserves_snapshots():
    model = make_model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    observer = _RecordingObserver()
    observer.parameters = dict(model.named_parameters())
    observer._optimizer = optimizer
    train(model, optimizer, lambda: [synthetic_batch(2)] * 3, device="cpu",
          epochs=1, set_to_none=True, observer=observer)

    received_backward = next(item for item in observer.received if item[0] == "step-0.backward")
    _, raw_backward, active = received_backward
    assert active == ("step-0.backward-diagnostics",)
    assert raw_backward["logits"].requires_grad and raw_backward["logits"].grad_fn is not None
    assert raw_backward["loss"].requires_grad and raw_backward["loss"].grad_fn is not None
    assert all(observer.live_gradient_identity["step-0.backward"].values())

    received_update = next(item for item in observer.received if item[0] == "step-0.update")
    _, raw_update, active = received_update
    assert active == ("step-0.update-diagnostics",)
    assert raw_update["parameters"]["1.weight"] is model[1].weight
    assert raw_update["state"]["1.weight"]["momentum_buffer"] is optimizer.state[model[1].weight]["momentum_buffer"]
    assert all(observer.live_parameter_identity["step-0.update"].values())
    assert all(observer.live_state_identity["step-0.update"].values())

    first_snapshot = next(values for name, values in observer.events if name == "step-0.update")
    first_weight = first_snapshot["parameters"]["1.weight"].clone()
    assert not first_snapshot["parameters"]["1.weight"].requires_grad
    assert not torch.equal(first_weight, model[1].weight.detach())


def test_evaluate_observes_original_logits_in_unique_diagnostic_phase_without_staging_labels(monkeypatch):
    model = make_model()
    observer = _RecordingObserver()
    first = synthetic_batch(2)
    second = synthetic_batch(1)
    batches = lambda: [first, second]
    transferred = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        transferred.append(id(tensor))
        return original_to(tensor, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_to)
    result = evaluate(model, batches, device="cpu", observer=observer, epoch=7)

    evaluation_events = [item for item in observer.received if item[0].startswith("epoch-7.evaluation-")]
    assert [name for name, _, _ in evaluation_events] == [
        "epoch-7.evaluation-0", "epoch-7.evaluation-1"
    ]
    for batch_index, (_, payload, active) in enumerate(evaluation_events):
        assert active == (f"epoch-7.evaluation-diagnostics-{batch_index}",)
        assert set(payload) == {"logits"}
        assert payload["logits"].device.type == "cpu"
        assert payload["logits"].grad_fn is None
    assert [name for name in observer.phases if "evaluation-staging" in name] == [
        "epoch-7.evaluation-staging-0", "epoch-7.evaluation-staging-1"
    ]
    assert all("label" not in name for name in observer.phases)
    assert id(first[0]) in transferred and id(second[0]) in transferred
    assert id(first[1]) not in transferred and id(second[1]) not in transferred
    assert result["count"] == 3


def test_reference_inventory_rejects_mutated_source_bytes(tmp_path):
    name = "sample.py"
    path = tmp_path / name
    expected = b"def expected_route(): pass\n"
    path.write_bytes(b"def changed_route(): pass\n")
    identities = {name: {"sha256": hashlib.sha256(expected).hexdigest(), "url": "fixture://sample"}}
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        reference_probe.source_inventory(tmp_path, identities=identities,
                                         source_lines={name: ["expected_route"]})


def test_reference_inventory_rejects_absent_expected_source_reference(tmp_path):
    name = "sample.py"
    path = tmp_path / name
    content = b"def unrelated_route(): pass\n"
    path.write_bytes(content)
    identities = {name: {"sha256": hashlib.sha256(content).hexdigest(), "url": "fixture://sample"}}
    with pytest.raises(ValueError, match="missing expected source reference"):
        reference_probe.source_inventory(tmp_path, identities=identities,
                                         source_lines={name: ["required_route"]})


def test_reference_probe_requires_installed_build_git_revision():
    reference_probe.validate_versions(reference_probe.VERSION, reference_probe.REVISION)
    with pytest.raises(RuntimeError, match="git revision"):
        reference_probe.validate_versions(reference_probe.VERSION, "different-build-revision")
    with pytest.raises(RuntimeError, match="torch version"):
        reference_probe.validate_versions("9.9.9", reference_probe.REVISION)


def test_both_reset_modes_produce_identical_three_step_updates():
    initial = make_model().state_dict()
    outputs = []
    for set_to_none in (True, False):
        model = make_model()
        model.load_state_dict(copy.deepcopy(initial))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        train(model, optimizer, lambda: [synthetic_batch(2)] * 3,
              device="cpu", epochs=1, set_to_none=set_to_none,
              observer=_RecordingObserver())
        outputs.append((model.state_dict(), optimizer.state_dict()))
    for key in outputs[0][0]:
        torch.testing.assert_close(outputs[0][0][key], outputs[1][0][key], rtol=0, atol=0)
    for state_key in outputs[0][1]["state"]:
        for field in outputs[0][1]["state"][state_key]:
            torch.testing.assert_close(outputs[0][1]["state"][state_key][field],
                                       outputs[1][1]["state"][state_key][field], rtol=0, atol=0)


def test_evaluate_preserves_mode_and_returns_only_aggregated_metrics():
    model = make_model()
    observer = _RecordingObserver()
    result = evaluate(model, lambda: [synthetic_batch(2)], device="cpu",
                      observer=observer, epoch=4)
    assert model.training
    assert set(result) == {"predictions", "correct", "count", "accuracy"}
    assert result["count"] == len(result["predictions"]) == 2
    assert result["correct"] == sum(p == y for p, y in zip(result["predictions"], [0, 9]))
    assert result["accuracy"] == result["correct"] / 2
    assert all("labels" not in values for _, values in observer.events)
    assert "epoch-4.evaluation-0" in observer.phases
