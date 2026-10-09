"""Deterministic public MSE reference fixtures; imports no Vulkan backend.

The three original bases always require gradients, including for selected-only
first reverses: cross-target second derivatives must remain observable.
"""
import itertools
import warnings

import torch
import torch.nn.functional as F


MSE_AUTOGRAD_CASES = tuple(
    {"id": f"{kind}-{reduction}-{selection}", "kind": kind,
     "reduction": reduction, "selection": selection}
    for kind, reduction, selection in itertools.product(
        ("scalar", "ordinary", "empty", "broadcast", "expanded-upstream"),
        ("none", "mean", "sum"), ("input", "target", "both")
    )
)


def seeded(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed)) * 0.3


def make_mse_case(case, device="cpu"):
    kind, reduction = case["kind"], case["reduction"]
    shape = () if kind == "scalar" else (0, 3) if kind == "empty" else (2, 3)
    target_shape = (1, 3) if kind == "broadcast" else shape
    upstream_shape = shape if reduction == "none" else ()
    if kind == "expanded-upstream" and reduction == "none":
        upstream_shape = (1, 3)
    shapes = (shape, target_shape, upstream_shape)
    bases = tuple(seeded(s, seed).to(device).requires_grad_()
                  for s, seed in zip(shapes, (101, 211, 307)))
    probes = tuple(seeded(s, seed).to(device)
                   for s, seed in zip(shapes[:2], (401, 503)))
    directions = tuple(seeded(s, seed).to(device)
                       for s, seed in zip(shapes, (607, 709, 811)))
    return {"bases": bases, "probes": probes, "directions": directions}


def mse_forward(case, bases, api="functional"):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Using a target size")
        if api == "module":
            return torch.nn.MSELoss(reduction=case["reduction"])(*bases[:2])
        return F.mse_loss(*bases[:2], reduction=case["reduction"])


def mse_graph(case, context, *, api="functional", create_graph=True):
    bases = context["bases"]
    loss = mse_forward(case, bases, api)
    upstream = bases[2]
    if case["kind"] == "expanded-upstream" and case["reduction"] == "none":
        upstream = upstream.expand_as(loss)
    selected = {"input": (0,), "target": (1,), "both": (0, 1)}[case["selection"]]
    first = torch.autograd.grad(loss, tuple(bases[i] for i in selected), upstream,
                                create_graph=create_graph)
    contraction = sum((g * context["probes"][i]).sum()
                      for i, g in zip(selected, first))
    return {"loss": loss, "upstream": upstream, "selected": selected,
            "first": first, "contraction": contraction}


def mse_reference(case, *, api="functional", epsilon=0.001):
    context = make_mse_case(case)
    graph = mse_graph(case, context, api=api)
    second = torch.autograd.grad(graph["contraction"], context["bases"])
    fd = []
    for target, direction in enumerate(context["directions"]):
        values = []
        for sign in (-1, 1):
            fresh = dict(context)
            fresh["bases"] = tuple(
                (base.detach() + (sign * epsilon * direction if i == target else 0))
                .requires_grad_() for i, base in enumerate(context["bases"])
            )
            values.append(float(mse_graph(case, fresh, api=api,
                                          create_graph=False)["contraction"]))
        fd.append({"target": target, "minus": values[0], "plus": values[1],
                   "fd": (values[1] - values[0]) / (2 * epsilon),
                   "analytic": float((second[target] * direction).sum())})
    return {"context": context, "graph": graph, "second": second, "fd": fd}
