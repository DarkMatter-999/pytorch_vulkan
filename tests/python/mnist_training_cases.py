"""Ordinary CPU/PyTorch recipe shared by staged MNIST training tests."""

from contextlib import nullcontext
from typing import Callable, ContextManager, Iterable, Protocol

import torch
from torch import Tensor, nn


class Observer(Protocol):
    """Instrumentation boundary; implementations must not alter live tensors."""

    def phase(self, name: str) -> ContextManager[None]: ...

    def observe(self, name: str, values: dict) -> None: ...


class CPUObserver:
    """No-op observer; recipe values are live until observe() returns."""

    def phase(self, name: str) -> ContextManager[None]:
        return nullcontext()

    def observe(self, name: str, values: dict) -> None:
        return None


def make_model(device: str = "cpu") -> nn.Sequential:
    """Build the seeded CPU model without changing the caller's CPU RNG."""
    with torch.random.fork_rng(devices=[]):
        generator = torch.Generator(device="cpu").manual_seed(12012)
        torch.set_rng_state(generator.get_state())
        model = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(784, 32),
            nn.ReLU(),
            nn.Linear(32, 10),
        )
    return model.to(device)


def synthetic_batch(batch_size: int) -> tuple[Tensor, Tensor]:
    """Return the fixed CPU F32/int64 training fixtures for qualified sizes."""
    if batch_size not in (1, 2, 16):
        raise ValueError("batch_size must be one of 1, 2, or 16")
    generator = torch.Generator(device="cpu").manual_seed(12013 + batch_size)
    images = torch.rand((batch_size, 1, 28, 28), generator=generator, dtype=torch.float32)
    if batch_size == 1:
        labels = [9]
    elif batch_size == 2:
        labels = [0, 9]
    else:
        labels = [index % 10 for index in range(16)]
    return images, torch.tensor(labels, dtype=torch.int64, device="cpu")


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batches: Callable[[], Iterable[tuple[Tensor, Tensor]]],
    *,
    device: str,
    epochs: int,
    set_to_none: bool,
    observer: Observer,
    after_epoch: Callable[[int], None] | None = None,
) -> None:
    """Train with ordinary CrossEntropy and synchronously observe live diagnostics."""
    loss_fn = nn.CrossEntropyLoss()
    step = 0
    for epoch in range(epochs):
        model.train()
        for x_cpu, y_cpu in batches():
            with observer.phase(f"step-{step}.staging"):
                x, y = x_cpu.to(device), y_cpu.to(device)
            previous = {key: parameter.grad for key, parameter in model.named_parameters()}
            with observer.phase(f"step-{step}.reset"):
                optimizer.zero_grad(set_to_none=set_to_none)
            named_parameters = dict(model.named_parameters())
            with observer.phase(f"step-{step}.reset-diagnostics"):
                observer.observe(f"step-{step}.reset", {
                    "gradients": {key: parameter.grad for key, parameter in named_parameters.items()},
                    "retained_objects": {
                        key: previous[key] is parameter.grad
                        for key, parameter in named_parameters.items()
                        if previous[key] is not None
                    },
                })
            with observer.phase(f"step-{step}.forward-backward"):
                logits = model(x)
                loss = loss_fn(logits, y)
                loss.backward()
            with observer.phase(f"step-{step}.backward-diagnostics"):
                observer.observe(f"step-{step}.backward", {
                    "logits": logits,
                    "loss": loss,
                    "gradients": {key: parameter.grad for key, parameter in named_parameters.items()},
                })
            with observer.phase(f"step-{step}.update"):
                optimizer.step()
            options = dict(optimizer.param_groups[0])
            options.pop("params", None)
            options["parameter_names"] = list(named_parameters)
            with observer.phase(f"step-{step}.update-diagnostics"):
                observer.observe(f"step-{step}.update", {
                    "parameters": named_parameters,
                    "state": {
                        key: dict(optimizer.state[parameter])
                        for key, parameter in named_parameters.items()
                    },
                    "optimizer_options": options,
                })
            step += 1
        if after_epoch is not None:
            after_epoch(epoch)


def evaluate(
    model: nn.Module,
    batches: Callable[[], Iterable[tuple[Tensor, Tensor]]],
    *,
    device: str,
    observer: Observer,
    epoch: int = 0,
) -> dict:
    """Evaluate via normal module inference and disclose aggregate CPU metrics."""
    was_training = model.training
    predictions: list[int] = []
    correct = 0
    count = 0
    try:
        model.eval()
        with torch.no_grad():
            for batch_index, (x_cpu, y_cpu) in enumerate(batches()):
                with observer.phase(f"epoch-{epoch}.evaluation-staging-{batch_index}"):
                    x = x_cpu.to(device)
                with observer.phase(f"epoch-{epoch}.evaluation-{batch_index}"):
                    logits = model(x)
                with observer.phase(f"epoch-{epoch}.evaluation-diagnostics-{batch_index}"):
                    observer.observe(f"epoch-{epoch}.evaluation-{batch_index}", {"logits": logits})
                with observer.phase(f"epoch-{epoch}.evaluation-readback-{batch_index}"):
                    logits_cpu = logits.detach().to(device="cpu").clone()
                batch_predictions = logits_cpu.argmax(dim=1)
                predictions.extend(int(value) for value in batch_predictions.tolist())
                correct += int(batch_predictions.eq(y_cpu).sum().item())
                count += int(y_cpu.numel())
    finally:
        model.train(was_training)
    return {
        "predictions": predictions,
        "correct": correct,
        "count": count,
        "accuracy": correct / count if count else 0.0,
    }
