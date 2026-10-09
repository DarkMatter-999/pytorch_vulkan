"""Source-owned finite G2 fixtures and CPU-only numerical oracles.

Route prediction mirrors PyTorch 2.4 LinearAlgebra.cpp should_fold/_matmul_impl
(e4ee3be4063b7c430974252fdf7db42273388d86), not profiler observations.
No backend import or device probing is performed by this module.
"""
import math
import torch


SHAPES = {
    "batch-vector": ((2,3,4),(4,)),
    "vector-batch": ((4,),(2,4,3)),
    "batch-matrix": ((2,3,4),(4,2)),
    "matrix-batch": ((3,4),(2,4,2)),
    "equal-batch": ((2,3,4),(2,4,2)),
    "singleton-left": ((1,3,4),(2,4,2)),
    "singleton-right": ((2,3,4),(1,4,2)),
    "unequal-rank": ((2,1,3,4),(5,4,2)),
    "cross-broadcast": ((2,1,3,4),(1,5,4,2)),
}
for rank in range(4, 9):
    leading = (2,) + (1,)*(rank-3)
    SHAPES[f"rank-{rank}"] = (leading+(2,3), (1,)*(rank-2)+(3,2))
SHAPES.update({
    "rank8-vector": ((2,1,1,1,1,1,2,3),(3,)),
    "vector-rank8": ((3,),(2,1,1,1,1,1,3,2)),
    "rank8-rank3": ((2,1,1,1,1,1,2,3),(1,3,2)),
    "rank3-rank8": ((1,2,3),(2,1,1,1,1,1,3,2)),
    "transpose-matrix": ((2,4,3),(4,2)),
    "nonfoldable": ((3,2,4),(4,2)),
    "nonfoldable-vector": ((3,2,4),(4,)),
    "offset-fold": ((3,3,4),(4,2)),
    "expanded": ((1,3,4),(4,2)),
    "cross-transposed": ((2,1,4,3),(1,5,2,4)),
    "empty-M": ((2,0,4),(1,4,2)),
    "empty-N": ((2,3,4),(1,4,0)),
    "empty-K": ((2,3,0),(1,0,2)),
    "zero-batch": ((0,3,4),(1,4,2)),
    "cross-zero": ((0,1,3,4),(1,5,4,2)),
})
RECIPES = {
    "transpose-matrix": ("transpose", "identity"),
    "nonfoldable": ("batch-row-swap", "identity"),
    "nonfoldable-vector": ("batch-row-swap", "identity"),
    "offset-fold": ("offset", "identity"),
    "expanded": ("expand", "identity"),
    "cross-transposed": ("transpose", "transpose"),
}
SELECTIVE = {"batch-vector", "vector-batch", "batch-matrix", "matrix-batch",
             "singleton-left", "singleton-right", "cross-broadcast",
             "transpose-matrix", "nonfoldable", "nonfoldable-vector", "expanded"}


def random_tensor(shape, seed):
    # In-place CPU scaling preserves the canonical empty allocation strides too.
    return torch.randn(tuple(shape), generator=torch.Generator().manual_seed(seed)).mul_(.3)


def tensor_fact(tensor):
    return {"shape": tuple(tensor.shape), "stride": tuple(tensor.stride()),
            "offset": tensor.storage_offset(), "dtype": str(tensor.dtype),
            "rank": tensor.ndim, "requires_grad": tensor.requires_grad,
            "history": tensor.grad_fn is not None}


def _view(tensor, recipe):
    if recipe == "transpose":
        return tensor.transpose(-1, -2)
    if recipe == "batch-row-swap":
        return tensor.transpose(0, 1)
    if recipe == "offset":
        return tensor[1:]
    if recipe == "expand":
        return tensor.expand(2, 3, 4)
    assert recipe == "identity"
    return tensor


def make_cpu_bases(case: dict) -> tuple[torch.Tensor, torch.Tensor]:
    return tuple(random_tensor(shape, case["seed"] + 10 + i).requires_grad_(
        case["selection"] == "both" or case["selection"] == ("a", "b")[i])
        for i, shape in enumerate(case["base_shapes"]))


def make_operands(case: dict, bases: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    assert tuple(tuple(x.shape) for x in bases) == case["base_shapes"]
    operands = tuple(_view(x, recipe) for x, recipe in zip(bases, case["recipes"]))
    if "operand_facts" in case:
        for operand, expected in zip(operands, case["operand_facts"]):
            fact = tensor_fact(operand)
            for key in ("shape", "stride", "offset", "dtype", "rank", "requires_grad"):
                assert fact[key] == expected[key], (case["id"], key, fact, expected)
    return operands


def call_matmul(case: dict, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if case["api"] == "matmul":
        return torch.matmul(a, b)
    assert case["api"] == "@"
    return a @ b


def _reshape_needs_copy(tensor, shape):
    """Pinned stock reshape uses a view when computeStride admits this geometry.

    Ordinary CPU view establishes viewability without executing a reshape copy.
    Empty reshapes remain aliasable; offset/zero-stride views are not presumed
    to require copying merely because they are noncontiguous.
    """
    try:
        tensor.view(shape)
        return False
    except RuntimeError:
        return True


def _route(a, b):
    """Predict leaf and minimum stock fold/broadcast/output materializations.

    LinearAlgebra.cpp:2076,2084,2138–2151 prescribe these reshape/contiguous
    geometries. The minimum counts only those required stock compositions;
    backend leaf metadata or additional device read planning is not constrained.
    """
    larger_left = a.ndim >= b.ndim
    large = a if larger_left else b.mT
    small = b if larger_left else a
    foldable = (large.ndim >= 3 and small.ndim <= 2 and
                (small.requires_grad or (a.ndim != 2 and
                 (large.numel() == 0 or all(
                     large.stride(i) == large.stride(i+1)*large.shape[i+1]
                     for i in range(large.ndim-2))))))
    if foldable:
        copies = int(_reshape_needs_copy(large, (math.prod(large.shape[:-1]), large.shape[-1])))
        if not larger_left and small.ndim == 2:
            # Stock right-large matrix fold returns output.mT().contiguous().
            shape = tuple(large.shape[:-1]) + (small.shape[-2],)
            copies += int(not torch.empty(shape).mT.is_contiguous())
        return "mm" if small.ndim == 2 else "mv", copies
    if a.ndim == b.ndim == 3 and a.shape[0] != b.shape[0]:
        if a.shape[0] == 1 and a.requires_grad:
            return _route(a.squeeze(0), b)
        if b.shape[0] == 1 and b.requires_grad:
            return _route(a, b.squeeze(0))
    batch = tuple(torch.broadcast_shapes(a.shape[:-2] if a.ndim > 1 else (),
                                        b.shape[:-2] if b.ndim > 1 else ()))
    count = math.prod(batch)
    n, m1 = (a.shape[-2] if a.ndim > 1 else 1), a.shape[-1]
    m2, p = (b.shape[-2], b.shape[-1]) if b.ndim > 1 else (b.shape[-1], 1)
    left = a.expand(batch + (n, m1))
    right = (b if b.ndim > 1 else b.unsqueeze(-1)).expand(batch + (m2, p))
    copies = (int(_reshape_needs_copy(left, (count, n, m1)))
              + int(_reshape_needs_copy(right, (count, m2, p))))
    return "bmm", copies


def _cases():
    cases = []
    for index, (family, shapes) in enumerate(SHAPES.items()):
        selections = ("none", "both", "a", "b") if family in SELECTIVE else ("none", "both")
        for selection, no_grad in [(s, False) for s in selections] + [("both", True)]:
            for api in ("matmul", "@"):
                case = {"id": f"g2.{family}.{selection}.{'no-grad' if no_grad else 'grad-mode'}.{api}",
                        "family": family, "base_shapes": shapes,
                        "recipes": RECIPES.get(family, ("identity", "identity")),
                        "seed": 1101 + index*37, "selection": selection,
                        "api": api, "no_grad": no_grad,
                        "mixed": family in ("batch-matrix", "cross-broadcast")
                                 and selection == "both" and not no_grad}
                operands = make_operands(case, make_cpu_bases(case))
                case["operand_facts"] = tuple(tensor_fact(x) for x in operands)
                a, b = operands
                batch = torch.broadcast_shapes(a.shape[:-2] if a.ndim > 1 else (),
                                               b.shape[:-2] if b.ndim > 1 else ())
                case["output_shape"] = tuple(batch) + ((a.shape[-2],) if a.ndim > 1 else ()) + ((b.shape[-1],) if b.ndim > 1 else ())
                case["route"], case["materialization_minimum"] = _route(a, b)
                case["copy_route"] = case["materialization_minimum"] > 0
                cases.append(case)
    return tuple(cases)


GENERAL_MATMUL_CASES: tuple[dict, ...] = _cases()


def cpu_reference(case: dict) -> dict:
    """Recompute first/mixed tensors plus an independent centered CPU FD.

    FD uses only fresh first-gradient evaluations at shifted opposite bases. It
    never consumes the analytic mixed tensor, projection, or captured summaries.
    """
    bases = make_cpu_bases(case)
    operands = make_operands(case, bases)
    with torch.set_grad_enabled(not case["no_grad"]):
        output = call_matmul(case, *operands)
    seed = random_tensor(output.shape, case["seed"] + 1)
    targets = tuple(i for i, b in enumerate(bases) if b.requires_grad)
    gradients = (torch.autograd.grad(output, tuple(bases[i] for i in targets), seed,
                                    create_graph=True, retain_graph=True)
                 if targets and not case["no_grad"] else ())
    result = {"bases": bases, "inputs": operands, "output": output,
              "base_facts": tuple(tensor_fact(x) for x in bases),
              "input_facts": tuple(tensor_fact(x) for x in operands),
              "output_fact": tensor_fact(output), "targets": targets,
              "seed": seed, "gradients": gradients,
              "gradient_facts": tuple(tensor_fact(x) for x in gradients), "mixed": []}
    if case["mixed"]:
        for first, opposite in ((0, 1), (1, 0)):
            probe = random_tensor(bases[first].shape, case["seed"] + 101 + first)
            direction = random_tensor(bases[opposite].shape, case["seed"] + 201 + first)
            mixed = torch.autograd.grad(gradients[first], bases[opposite], probe,
                                        retain_graph=True)[0]
            def shifted_projection(delta):
                fresh = [x.detach().clone() for x in bases]
                fresh[opposite].add_(direction, alpha=delta)
                fresh[first].requires_grad_()
                value = call_matmul(case, *tuple(_view(x, r) for x, r in zip(fresh, case["recipes"])))
                gradient = torch.autograd.grad(value, fresh[first], seed)[0]
                return (gradient * probe).sum().item()
            fd = (shifted_projection(.001) - shifted_projection(-.001)) / .002
            projection = (mixed * direction).sum().item()
            result["mixed"].append({"direction": (first, opposite), "tensor": mixed,
                "probe": probe, "perturbation": direction, "projection": projection,
                "fd": fd, "fd_error": abs(fd-projection), "epsilon": .001})
    return result
