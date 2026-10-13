"""Observation and CPU/Vulkan parity helpers for ordinary MNIST training."""

from contextlib import contextmanager
import math
from pathlib import Path

import torch

from mnist_training_cases import evaluate, make_model, synthetic_batch, train
from tools.mnist_data import load_mnist_subset, mnist_loader


def _device_matches(tensor, device):
    try:
        expected_device = torch.device(device)
    except (RuntimeError, TypeError):
        return False
    return tensor.device == expected_device


def _metadata(tensor):
    return {
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "requires_grad": bool(tensor.requires_grad),
        "history": tensor.grad_fn is not None,
    }


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(device="cpu").clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_snapshot(item) for item in value]
    return value


def _tensor_tree(value, path=()):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _tensor_tree(item, path + (key,))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _tensor_tree(item, path + (index,))


def _metadata_tree(value):
    if isinstance(value, torch.Tensor):
        return _metadata(value)
    if isinstance(value, dict):
        return {key: _metadata_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_metadata_tree(item) for item in value)
    if isinstance(value, list):
        return [_metadata_tree(item) for item in value]
    return None


def assert_vulkan_tensor_metadata(tensor_metadata):
    """Assert every nested tensor metadata leaf names the selected Vulkan device."""
    leaves = list(_iter_tensor_metadata(tensor_metadata))
    assert leaves
    assert all(metadata["device"] == "vk:0" for metadata in leaves)


def _iter_tensor_metadata(value):
    if isinstance(value, dict):
        if "device" in value:
            yield value
        else:
            for item in value.values():
                yield from _iter_tensor_metadata(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _iter_tensor_metadata(item)


class TrainingObserver:
    """Validate live callback values, then preserve diagnostic CPU snapshots."""

    def __init__(self, device: str, *, backend=None):
        self.device = device
        self.backend = backend
        self.snapshots = {}
        self.tensor_metadata = {}
        self.counters = {}
        self.phase_routes = {}
        self.phase_order = []
        self.active_phases = []
        self.counter_categories = {
            "setup": set(),
            "compute": set(),
            "reset": set(),
            "staging": set(),
            "diagnostics": set(),
            "readback": set(),
        }

    @staticmethod
    def _category(name):
        if name.startswith("setup."):
            return "setup"
        if "diagnostics" in name:
            return "diagnostics"
        if "staging" in name:
            return "staging"
        if "readback" in name:
            return "readback"
        if name.endswith(".reset"):
            return "reset"
        if name.endswith(".forward-backward") or (
            ".evaluation-" in name and "diagnostics" not in name
        ) or name.endswith(".update"):
            return "compute"
        return "other"

    @contextmanager
    def phase(self, name):
        if self.active_phases:
            raise AssertionError(f"nested observer phase {name!r} inside {self.active_phases[-1]!r}")
        category = self._category(name)
        self.phase_order.append(name)
        self.phase_routes[name] = category
        if category in self.counter_categories:
            self.counter_categories[category].add(name)
        if self.backend is not None:
            self.backend._C.synchronize()
            self.backend._C.reset_execution_counters()
        self.active_phases.append(name)
        try:
            yield
        finally:
            if self.active_phases.pop() != name:
                raise AssertionError("observer phase stack was corrupted")
            if self.backend is not None:
                self.backend._C.synchronize()
                self.counters[name] = list(
                    self.backend._C.execution_counter_snapshot()
                )

    def observe(self, name, values):
        metadata = _metadata_tree(values)
        for path, tensor in _tensor_tree(values):
            if not _device_matches(tensor, self.device):
                raise AssertionError(
                    f"{name} {'.'.join(map(str, path))}: wrong residency "
                    f"{tensor.device} for {self.device}"
                )

        if name.endswith(".backward"):
            for key in ("logits", "loss"):
                tensor = values[key]
                if not tensor.requires_grad or tensor.grad_fn is None:
                    raise AssertionError(f"{name} {key} is missing gradient history")
        if name.startswith("epoch-") and ".evaluation-" in name:
            logits = values["logits"]
            if logits.requires_grad or logits.grad_fn is not None:
                raise AssertionError(f"{name} evaluation logits unexpectedly have history")

        if name.endswith(".reset"):
            expected_phase = name[:-len(".reset")] + ".reset-diagnostics"
        elif name.endswith(".backward"):
            expected_phase = name[:-len(".backward")] + ".backward-diagnostics"
        elif name.endswith(".update"):
            expected_phase = name[:-len(".update")] + ".update-diagnostics"
        elif name.startswith("epoch-") and ".evaluation-" in name:
            epoch, batch_index = name.split(".evaluation-", 1)
            expected_phase = f"{epoch}.evaluation-diagnostics-{batch_index}"
        else:
            raise AssertionError(f"unrecognized observation event {name!r}")
        if self.active_phases != [expected_phase]:
            raise AssertionError(
                f"{name}: expected active diagnostic phase {expected_phase!r}, "
                f"got {self.active_phases!r}"
            )

        self.tensor_metadata[name] = metadata
        snapshot = _snapshot(values)
        if name.endswith(".reset"):
            for path, tensor in _tensor_tree(snapshot):
                if torch.count_nonzero(tensor).item() != 0:
                    location = ".".join(map(str, path))
                    raise AssertionError(
                        f"{name} {location}: reset gradient is not exactly zero"
                    )
        self.snapshots[name] = snapshot

    def validate_accounting(self):
        from composed_evidence_common import validate_counters

        for name, category in self.phase_routes.items():
            if name not in self.counters:
                continue
            counters = self.counters[name]
            if category == "compute":
                validate_counters(counters)
            elif category == "setup":
                if counters[3] != 0:
                    raise AssertionError(f"{name}: setup fallback is not allowed: {counters}")
            elif category == "reset":
                if counters[2:] != [0, 0]:
                    raise AssertionError(f"{name}: reset caused transfer/fallback: {counters}")
            elif category in ("staging", "diagnostics", "readback"):
                if counters[3] != 0:
                    raise AssertionError(f"{name}: fallback in {category}: {counters}")


def _compare_values(actual, expected, path, field, errors):
    label = ".".join(map(str, path)) or "value"
    if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
        if actual is None or expected is None:
            if any("reset" in str(part) for part in path):
                raise AssertionError(f"reset mismatch at {label}")
            if field == "gradients":
                raise AssertionError(f"missing gradient at {label}")
            if field == "state":
                raise AssertionError(f"missing momentum/state at {label}")
            raise AssertionError(f"reset mismatch at {label}")
        if not isinstance(actual, torch.Tensor) or not isinstance(expected, torch.Tensor):
            raise AssertionError(f"tensor mismatch at {label}")
        if actual.shape != expected.shape:
            raise AssertionError(f"shape mismatch at {label}: {actual.shape} != {expected.shape}")
        if actual.dtype != expected.dtype:
            raise AssertionError(f"dtype mismatch at {label}: {actual.dtype} != {expected.dtype}")
        if any("reset" in str(part) for part in path) and not torch.equal(actual, expected):
            raise AssertionError(f"reset mismatch at {label}")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"nonfinite tensor at {label}")
        rtol, atol = (1e-4, 1e-5) if field == "parameters" else (2e-4, 2e-4)
        try:
            torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
        except AssertionError as error:
            raise AssertionError(f"not close at {label}: {error}") from error
        delta = (actual - expected).abs()
        maximum_absolute = float(delta.max().item()) if delta.numel() else 0.0
        denominator = expected.abs().clamp_min(atol)
        maximum_relative = float((delta / denominator).max().item()) if delta.numel() else 0.0
        errors[label] = {
            "maximum_absolute_error": maximum_absolute,
            "maximum_relative_error": maximum_relative,
        }
        return

    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict) or actual.keys() != expected.keys():
            raise AssertionError(f"keys mismatch at {label}")
        for key in actual:
            child_field = key if key in ("parameters", "gradients", "state") else field
            _compare_values(actual[key], expected[key], path + (key,), child_field, errors)
        return
    if isinstance(actual, (tuple, list)) or isinstance(expected, (tuple, list)):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise AssertionError(f"sequence mismatch at {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_values(left, right, path + (index,), field, errors)
        return
    if isinstance(actual, float) or isinstance(expected, float):
        if not math.isfinite(float(actual)) or not math.isfinite(float(expected)):
            raise AssertionError(f"nonfinite value at {label}")
        if actual != expected:
            raise AssertionError(f"value mismatch at {label}: {actual!r} != {expected!r}")
        return
    if actual != expected:
        if field == "gradients":
            raise AssertionError(f"gradient mismatch at {label}")
        if field == "state":
            raise AssertionError(f"momentum/state mismatch at {label}")
        if "reset" in path:
            raise AssertionError(f"reset mismatch at {label}")
        raise AssertionError(f"value mismatch at {label}: {actual!r} != {expected!r}")


def compare_runs(actual: TrainingObserver, expected: TrainingObserver) -> dict:
    if actual.snapshots.keys() != expected.snapshots.keys():
        raise AssertionError("snapshot event keys/order mismatch")
    if actual.phase_order != expected.phase_order:
        raise AssertionError("phase order mismatch")
    errors = {}
    for name in expected.snapshots:
        left, right = actual.snapshots[name], expected.snapshots[name]
        _compare_values(left, right, (name,), "", errors)
        left_meta = actual.tensor_metadata.get(name)
        right_meta = expected.tensor_metadata.get(name)
        if left_meta is not None and right_meta is not None:
            _compare_metadata(left_meta, right_meta, (name,))
    return {"passed": True, "errors": errors}


def _compare_metadata(actual, expected, path):
    label = ".".join(map(str, path))
    if isinstance(actual, dict) or isinstance(expected, dict):
        if not isinstance(actual, dict) or not isinstance(expected, dict):
            raise AssertionError(f"metadata structure mismatch at {label}")
        if actual.keys() != expected.keys():
            raise AssertionError(f"metadata keys mismatch at {label}")
        if "dtype" in actual or "dtype" in expected:
            if "dtype" not in actual or "dtype" not in expected:
                raise AssertionError(f"metadata tensor structure mismatch at {label}")
            for key in ("shape", "dtype", "requires_grad", "history"):
                if actual[key] != expected[key]:
                    raise AssertionError(f"metadata {key} mismatch at {label}")
            return
        for key in actual:
            _compare_metadata(actual[key], expected[key], path + (key,))
        return
    if isinstance(actual, (list, tuple)) or isinstance(expected, (list, tuple)):
        if type(actual) is not type(expected):
            raise AssertionError(f"metadata sequence type mismatch at {label}")
        if len(actual) != len(expected):
            raise AssertionError(f"metadata sequence length mismatch at {label}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare_metadata(left, right, path + (index,))
        return
    if actual != expected:
        raise AssertionError(f"metadata value mismatch at {label}: {actual!r} != {expected!r}")


def _compare_evaluations(actual, expected):
    """Compare host-side classification results without numerical tolerance."""
    for key in ("predictions", "correct", "count"):
        if actual[key] != expected[key]:
            raise AssertionError(f"evaluation {key} mismatch")
    expected_accuracy = expected["correct"] / expected["count"] if expected["count"] else 0.0
    actual_accuracy = actual["correct"] / actual["count"] if actual["count"] else 0.0
    if actual_accuracy != expected_accuracy:
        raise AssertionError("evaluation accuracy mismatch")
    return {
        "passed": True,
        "correct": actual["correct"],
        "count": actual["count"],
        "accuracy": actual_accuracy,
    }


def _real_observers(
    model, optimizer, device, backend, train_loader, test_loader, set_to_none,
    *, observer=None,
):
    if observer is None:
        observer = TrainingObserver(device, backend=backend)
    elif observer.device != device or observer.backend is not backend:
        raise ValueError("injected observer device/backend does not match the training run")
    train_batch_sizes = []
    evaluation_batch_sizes = []
    evaluations = []

    def training_batches():
        for images, labels in train_loader:
            train_batch_sizes.append(int(labels.numel()))
            yield images, labels

    def after_epoch(epoch):
        def evaluation_batches():
            for images, labels in test_loader:
                evaluation_batch_sizes.append(int(labels.numel()))
                yield images, labels

        evaluations.append(evaluate(
            model, evaluation_batches, device=device, observer=observer, epoch=epoch
        ))

    train(model, optimizer, training_batches, device=device, epochs=2,
          set_to_none=set_to_none, observer=observer, after_epoch=after_epoch)
    observer.validate_accounting()
    return observer, train_batch_sizes, evaluation_batch_sizes, evaluations


def run_real_pair(root: Path, set_to_none: bool, *, backend) -> dict:
    """Compare ordinary CPU/Vulkan training on the approved bounded MNIST prefix."""
    root = Path(root)
    repository_root = Path(__file__).resolve().parents[2]
    train_dataset, train_metadata = load_mnist_subset(
        root, split="train", repo_root=repository_root
    )
    test_dataset, test_metadata = load_mnist_subset(
        root, split="test", repo_root=repository_root
    )
    train_loader = mnist_loader(train_dataset)
    test_loader = mnist_loader(test_dataset)

    cpu_observer = TrainingObserver("cpu")
    with cpu_observer.phase("setup.model-staging"):
        cpu_model = make_model("cpu")
    vk_observer = TrainingObserver("vk:0", backend=backend)
    with vk_observer.phase("setup.model-staging"):
        vk_model = make_model("cpu").to("vk:0")
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01, momentum=0.9)
    vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.01, momentum=0.9)
    cpu_observer, cpu_train_sizes, cpu_eval_sizes, cpu_evaluations = _real_observers(
        cpu_model, cpu_optimizer, "cpu", None, train_loader, test_loader, set_to_none,
        observer=cpu_observer,
    )
    vk_observer, vk_train_sizes, vk_eval_sizes, vk_evaluations = _real_observers(
        vk_model, vk_optimizer, "vk:0", backend, train_loader, test_loader, set_to_none,
        observer=vk_observer,
    )

    parity = compare_runs(vk_observer, cpu_observer)
    evaluation_comparisons = [
        _compare_evaluations(vk_metrics, cpu_metrics)
        for vk_metrics, cpu_metrics in zip(vk_evaluations, cpu_evaluations)
    ]
    if len(cpu_evaluations) != 2 or len(vk_evaluations) != 2:
        raise AssertionError("expected exactly two evaluation epochs")
    return {
        "updates": sum(name.endswith(".update") for name in cpu_observer.snapshots),
        "set_to_none": bool(set_to_none),
        "train_batch_sizes": cpu_train_sizes,
        "evaluation_batch_sizes": [cpu_eval_sizes[:9], cpu_eval_sizes[9:]],
        "evaluation_counts": [item["count"] for item in cpu_evaluations],
        "metric_device": "cpu",
        "data_identity": {"train": train_metadata, "test": test_metadata},
        "evaluations": [
            {"epoch": epoch, "comparison": evaluation_comparisons[epoch],
             "cpu": cpu_metrics, "vulkan": vk_metrics}
            for epoch, (cpu_metrics, vk_metrics) in enumerate(zip(cpu_evaluations, vk_evaluations))
        ],
        "parity": parity,
        "parameter_names": ["1.weight", "1.bias", "3.weight", "3.bias"],
        # Source-owned capture can serialize these model diagnostics without
        # repeating training. Inputs and labels never enter observer snapshots.
        "diagnostic_snapshots": {
            "cpu": cpu_observer.snapshots,
            "vulkan": vk_observer.snapshots,
        },
        "cpu": _observer_result(cpu_observer),
        "vulkan": _observer_result(vk_observer),
    }


def _observer_result(observer):
    return {
        "phase_order": list(observer.phase_order),
        "phase_routes": dict(observer.phase_routes),
        "counters": {key: list(value) for key, value in observer.counters.items()},
        "counter_categories": {
            key: sorted(value) for key, value in observer.counter_categories.items()
        },
        "tensor_metadata": observer.tensor_metadata,
    }


def run_synthetic_pair(batch_size: int, set_to_none: bool, *, backend) -> dict:
    """Run three ordinary SGD updates from identical CPU-initialized models."""
    if batch_size not in (1, 2, 16):
        raise ValueError("batch_size must be one of 1, 2, or 16")
    cpu_observer = TrainingObserver("cpu")
    with cpu_observer.phase("setup.model-staging"):
        cpu_model = make_model("cpu")
    vk_observer = TrainingObserver("vk:0", backend=backend)
    with vk_observer.phase("setup.model-staging"):
        vk_model = make_model("cpu").to("vk:0")
    cpu_optimizer = torch.optim.SGD(cpu_model.parameters(), lr=0.01, momentum=0.9)
    vk_optimizer = torch.optim.SGD(vk_model.parameters(), lr=0.01, momentum=0.9)
    batches = lambda: [synthetic_batch(batch_size) for _ in range(3)]
    train(cpu_model, cpu_optimizer, batches, device="cpu", epochs=1,
          set_to_none=set_to_none, observer=cpu_observer)
    train(vk_model, vk_optimizer, batches, device="vk:0", epochs=1,
          set_to_none=set_to_none, observer=vk_observer)
    vk_observer.validate_accounting()
    parity = compare_runs(vk_observer, cpu_observer)
    return {
        "updates": 3,
        "batch_size": batch_size,
        "set_to_none": bool(set_to_none),
        "parity": parity,
        "parameter_names": ["1.weight", "1.bias", "3.weight", "3.bias"],
        # Keep the existing observer values alive for source-owned evidence;
        # metadata remains separate so CPU copies cannot misstate live residency.
        "diagnostic_snapshots": {
            "cpu": cpu_observer.snapshots,
            "vulkan": vk_observer.snapshots,
        },
        "cpu": _observer_result(cpu_observer),
        "vulkan": _observer_result(vk_observer),
    }
