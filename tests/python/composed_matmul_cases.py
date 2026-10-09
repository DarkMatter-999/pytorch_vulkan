"""Source-owned connected-model recipes and CPU replay, without Vulkan import.

Tensor trees use original-base names input/shared/down/up/readout; gradients never
target reconstructed views. Scenarios and return keys are stable capture interfaces.
Only selected second reverse is qualified; this is not arbitrary-order/forward AD.
"""
from contextlib import nullcontext

import torch


PARAMETER_SHAPES = {"shared": (4, 3), "down": (3, 3, 2),
                    "up": (1, 3, 3, 2), "readout": (3,)}
COMPOSED_MATMUL_SCENARIOS = (
    {"id": "g3.sgd-none", "kind": "training", "set_to_none": True},
    {"id": "g3.sgd-zero", "kind": "training", "set_to_none": False},
    {"id": "g3.output-first", "kind": "first"},
    {"id": "g3.mixed-down-up", "kind": "mixed", "first": "down", "second": "up"},
    {"id": "g3.mixed-input-shared", "kind": "mixed", "first": "input", "second": "shared"},
    {"id": "g3.mse-parameter-hvp", "kind": "hvp"},
    {"id": "g3.mutation-shared", "kind": "mutation", "mutated": "shared"},
    {"id": "g3.mutation-input", "kind": "mutation", "mutated": "input"},
    {"id": "g3.mutation-down", "kind": "mutation", "mutated": "down"},
)


class MultiHeadLowRankRegressor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        shapes = {"shared": (4, 3), "down": (3, 3, 2),
                  "up": (1, 3, 3, 2), "readout": (3,)}
        generator = torch.Generator().manual_seed(3101)
        for name, shape in shapes.items():
            value = torch.randn(shape, generator=generator) * 0.3
            self.register_parameter(name, torch.nn.Parameter(value))

    def forward(self, x_base):
        x = x_base[1:].transpose(-1, -2)
        a = x @ self.shared
        c = a @ self.down
        d = c @ self.up.transpose(-1, -2)
        return d @ self.readout


def seeded(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def make_model_case(device="cpu"):
    """Normal CPU construction and model.to; independent seeds, original leaf input."""
    model = MultiHeadLowRankRegressor().to(device)
    bases = {"input": seeded((3, 1, 4, 2), 3102).to(device).requires_grad_(),
             **dict(model.named_parameters())}
    probes = {"down": seeded(PARAMETER_SHAPES["down"], 3105).to(device),
              "input": seeded((3, 1, 4, 2), 3107).to(device)}
    directions = {"up": seeded(PARAMETER_SHAPES["up"], 3106).to(device),
                  "shared": seeded(PARAMETER_SHAPES["shared"], 3108).to(device)}
    return {"model": model, "bases": bases,
            "target": seeded((2, 3, 2), 3103).to(device),
            "upstream": seeded((2, 3, 2), 3104).to(device),
            "probes": probes, "directions": directions,
            "hvp_directions": {k: seeded(s, 3109+i).to(device)
                               for i, (k, s) in enumerate(PARAMETER_SHAPES.items())},
            "hvp_probes": {k: seeded(s, 3113+i).to(device)
                           for i, (k, s) in enumerate(PARAMETER_SHAPES.items())}}


def layout(tensor):
    return {"shape": list(tensor.shape), "stride": list(tensor.stride()),
            "offset": tensor.storage_offset()}


def model_metadata(context):
    return {"parameter_count": sum(p.numel() for p in context["model"].parameters()),
            "seeds": {"parameters": 3101, "input": 3102, "target": 3103,
                      "upstream": 3104, "probes_directions": list(range(3105, 3117))},
            "input_scale": 1., "parameter_scale": .3,
            "bases": {k: layout(v) for k, v in context["bases"].items()},
            "views": {"input": layout(context["bases"]["input"][1:].transpose(-1, -2)),
                      "up": layout(context["bases"]["up"].transpose(-1, -2))}}


def model_graph(context, scenario, phase=lambda name: nullcontext()):
    bases = context["bases"]
    with phase("forward"):
        output = context["model"](bases["input"])
        loss = torch.nn.MSELoss()(output, context["target"]) if scenario["kind"] == "hvp" else None
    keys = tuple(PARAMETER_SHAPES) if loss is not None else tuple(bases)
    with phase("first"):
        gradients = torch.autograd.grad(loss if loss is not None else output,
                                        tuple(bases[k] for k in keys),
                                        None if loss is not None else context["upstream"],
                                        create_graph=True)
    first = dict(zip(keys, gradients))
    values = {"output": output, "first": first}
    if loss is not None:
        values["loss"] = loss
        with phase("second"):
            contraction = sum((first[k]*context["hvp_directions"][k]).sum() for k in keys)
            values["second"] = dict(zip(keys, torch.autograd.grad(
                contraction, tuple(bases[k] for k in keys))))
    elif scenario["kind"] == "mixed":
        with phase("second"):
            contraction = (first[scenario["first"]]*context["probes"][scenario["first"]]).sum()
            key = scenario["second"]
            values["second"] = {key: torch.autograd.grad(contraction, bases[key])[0]}
    return values


def training_steps(context, set_to_none, phase=lambda name: nullcontext()):
    model = context["model"]
    optimizer = torch.optim.SGD(model.parameters(), lr=.01, momentum=.9)
    loss_fn = torch.nn.MSELoss()
    parameters = dict(model.named_parameters())
    steps = []
    for step in range(3):
        optimizer.zero_grad(set_to_none=set_to_none)
        reset = {k: None if p.grad is None else p.grad.detach().clone() for k, p in parameters.items()}
        with phase(f"step-{step}-forward"):
            output = model(context["bases"]["input"])
            loss = loss_fn(output, context["target"])
        with phase(f"step-{step}-backward"):
            loss.backward()
        gradients = {k: p.grad.detach().clone() for k, p in parameters.items()}
        with phase(f"step-{step}-update"):
            optimizer.step()
        steps.append({"output": output, "loss": loss, "reset": reset,
                      "gradients": gradients,
                      "parameters": {k: p.detach().clone() for k, p in parameters.items()},
                      "state": {k: {sk: sv.detach().clone() for sk, sv in optimizer.state[p].items()}
                                for k, p in parameters.items()},
                      "state_parameter_keys": list(optimizer.state_dict()["state"]),
                      "parameter_keys": list(model.state_dict())})
    optimizer.zero_grad(set_to_none=set_to_none)
    return {"steps": steps,
            "final_reset": {k: None if p.grad is None else p.grad.detach().clone()
                            for k, p in parameters.items()},
            "optimizer_options": dict(optimizer.state_dict()["param_groups"][0])}


def mutation_case(context, mutated, phase=lambda name: nullcontext()):
    bases = context["bases"]
    with phase("forward"):
        output = context["model"](bases["input"])
    before = {k: v._version for k, v in bases.items()}
    with torch.no_grad():
        bases[mutated].add_(1.)
    after = {k: v._version for k, v in bases.items()}
    mutation = {"output": output, "mutated_base": bases[mutated].detach().clone(),
                "version_delta": {k: after[k]-before[k] for k in bases}}
    try:
        with phase("backward"):
            gradients = torch.autograd.grad(output, tuple(bases.values()), context["upstream"])
    except RuntimeError as error:
        if "modified by an inplace operation" not in str(error):
            raise
        return {**mutation, "outcome": "saved-version-rejection", "first": {}}
    return {**mutation, "outcome": "pre-mutation-gradient", "first": dict(zip(bases, gradients))}


def model_reference(scenario, epsilon=.001, phase=lambda name: nullcontext()):
    """CPU oracle: FD always recomputes fresh first gradients at perturbed bases."""
    context = make_model_case()
    metadata = model_metadata(context)
    metadata["scenario"] = dict(scenario)
    kind = scenario["kind"]
    if kind == "training":
        values = training_steps(context, scenario["set_to_none"], phase)
    elif kind == "mutation":
        baseline_context = make_model_case()
        baseline_output = baseline_context["model"](baseline_context["bases"]["input"])
        baseline = dict(zip(baseline_context["bases"], torch.autograd.grad(
            baseline_output, tuple(baseline_context["bases"].values()), baseline_context["upstream"])))
        values = mutation_case(context, scenario["mutated"], phase)
        for key, gradient in values["first"].items():
            torch.testing.assert_close(gradient, baseline[key])
    else:
        values = model_graph(context, scenario, phase)
    fd = []
    if kind in ("mixed", "hvp"):
        ends = []
        for sign in (-1, 1):
            fresh = make_model_case()
            directions = fresh["hvp_directions"] if kind == "hvp" else {
                scenario["second"]: fresh["directions"][scenario["second"]]}
            with torch.no_grad():
                for key, direction in directions.items():
                    fresh["bases"][key].add_(direction, alpha=sign*epsilon)
            y = fresh["model"](fresh["bases"]["input"])
            keys = tuple(PARAMETER_SHAPES) if kind == "hvp" else (scenario["first"],)
            objective = torch.nn.MSELoss()(y, fresh["target"]) if kind == "hvp" else y
            first = torch.autograd.grad(objective, tuple(fresh["bases"][k] for k in keys),
                                        None if kind == "hvp" else fresh["upstream"])
            ends.append(dict(zip(keys, first)))
        for key, second in values["second"].items():
            first_key = key if kind == "hvp" else scenario["first"]
            probe = context["hvp_probes"][key] if kind == "hvp" else context["probes"][first_key]
            projection = context["hvp_probes"][key] if kind == "hvp" else context["directions"][key]
            minus, plus = (float((end[first_key]*probe).sum()) for end in ends)
            fd.append({"target": key, "minus": minus, "plus": plus,
                       "fd": (plus-minus)/(2*epsilon), "analytic": float((second*projection).sum())})
    return {"values": values, "fd": fd, "metadata": metadata}
