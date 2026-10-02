"""Small, executable operator cases shared by Vulkan conformance tests."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import pytorch_vulkan
import torch

from tools.validate_vulkan_capabilities import load_manifest

TensorFactory = Callable[[], tuple[Any, ...]]


_MANIFEST = load_manifest(Path(__file__).resolve().parents[2] / "docs/vulkan_capabilities.json")
DECLARED_OPERATION_MANIFEST = frozenset(
    entry["schema"] for entry in _MANIFEST["entries"] if entry["status"] == "supported"
)

ROADMAP_DEFERRED_SCHEMAS = frozenset(
    entry["schema"] for entry in _MANIFEST["entries"] if entry["status"] == "deferred"
)

ROADMAP_OPERATION_FAMILIES = {"manifest-deferred": ROADMAP_DEFERRED_SCHEMAS}
ROADMAP_DEFERRED_REASON_BY_FAMILY = {"manifest-deferred": "deferred_contract"}

@dataclass(frozen=True)
class ConformanceCase:
    name: str
    family: str
    declaration_id: str
    operation: Callable[..., Any]
    input_factory: TensorFactory
    cpu_reference: Callable[..., Any]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] | None = None
    supported: bool = True
    error_pattern: str = r"Vulkan"
    expected_dtype: torch.dtype = torch.float32
    expected_shape: tuple[int, ...] | None = None
    check_gradients: bool = False
    autograd_supported: bool = True
    autograd_error_type: type[Exception] | None = None
    autograd_error_pattern: str | None = None
    requires_grad_inputs: bool = False
    rtol: float = 1e-5
    atol: float = 1e-8
    execution_mode: str = "compute"
    convert_inputs: bool = True
    setup_inputs: Callable[[Any, str], Any] | None = None
    declared_shapes: tuple[str, ...] = ()
    expected_shapes: tuple[tuple[int, ...] | None, ...] | None = None
    convolution_operand_roles: tuple[str, ...] = ()

    def inputs(self) -> tuple[Any, ...]:
        return self.input_factory(
            requires_grad=self.check_gradients or self.requires_grad_inputs
        )


COVERAGE_RECORDING = False
_COVERAGE: dict[str, dict[str, object]] = {}
_EXECUTED: set[str] = set()
_CASE_EXECUTION: dict[str, dict[str, object]] = {}
_STOCK_ROUTE_CASES = {
    "view.reshape.copy.trainable-seed",
    "view.reshape.offset-copy.second-order",
}


def mark_executed(case_name: str) -> None:
    _EXECUTED.add(case_name)


def executed_snapshot() -> set[str]:
    return set(_EXECUTED)


def reset_executed() -> None:
    _EXECUTED.clear()


def record_coverage(
    case: ConformanceCase,
    inputs: tuple[Any, ...],
    result: Any,
    *,
    gradients: bool,
    parity: bool,
    reverse_autograd: dict | None = None,
) -> None:
    """Record what a case actually exercised using its runtime tensors."""
    if not COVERAGE_RECORDING:
        return
    schema = _MANIFEST_CASES[case.name][0]
    def walk(value, path):
        if isinstance(value, torch.Tensor):
            return [(value, path)]
        if isinstance(value, (list, tuple)):
            return [item for index, child in enumerate(value) for item in walk(child, f"{path}[{index}]")]
        if isinstance(value, dict):
            return [item for key, child in value.items() for item in walk(child, f"{path}[{key!r}]")]
        return []

    walked = [item for index, value in enumerate(inputs) for item in walk(value, f"inputs[{index}]")]
    direct_tensors = [item for item in inputs if isinstance(item, torch.Tensor)]
    has_nested_inputs = any(isinstance(item, (list, tuple, dict)) for item in inputs)
    tensors = [tensor for tensor, _ in walked] if has_nested_inputs else direct_tensors
    positional_tensors = [
        item for item in (*inputs, *case.args) if isinstance(item, torch.Tensor)
    ]
    primary = positional_tensors[0] if positional_tensors else None
    operands = [
        {
            "role": "primary_input" if tensor is primary else "operand",
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "rank": tensor.dim(),
        }
        for tensor in positional_tensors
    ]
    record = {
        "schema": schema,
        "operands": operands,
        "primary_input": (
            {"dtype": operands[0]["dtype"], "rank": operands[0]["rank"]}
            if operands
            else None
        ),
        "input_dtypes": sorted(
            {str(tensor.dtype).removeprefix("torch.") for tensor in tensors}
        ),
        "input_ranks": sorted({tensor.dim() for tensor in tensors}),
        # A rank-0 tensor has no extents to join, so it contributes no entry
        # rather than a sentinel. "unwitnessed" already means "no shape witness
        # exists" in shape_constraints, and reusing it here would make a scalar
        # that genuinely ran indistinguishable from an entry with no witness.
        # Its rank is recorded separately in input_ranks.
        "input_shapes": sorted(
            {
                "x".join(str(extent) for extent in tensor.shape)
                for tensor in tensors
                if tensor.dim() > 0
            }
        ),
        "gradients": bool(gradients),
        "parity": bool(parity),
    }
    if isinstance(result, torch.Tensor):
        record["output_dtype"] = str(result.dtype).removeprefix("torch.")
        record["output_rank"] = result.dim()
    if reverse_autograd is not None:
        if reverse_autograd != {"order": 2, "graph_preserved": True}:
            raise ValueError("reverse_autograd evidence must report order 2 and preserved history")
        record["reverse_autograd"] = dict(reverse_autograd)
    if inputs and isinstance(inputs[0], (list, tuple)):
        listed = list(inputs[0])
        if listed and all(isinstance(item, torch.Tensor) for item in listed):
            if case.name not in _EXECUTED:
                raise ValueError(f"{case.name}: TensorList coverage requires a marked executed case")
            if schema != "aten::cat.default":
                raise ValueError(f"{case.name}: TensorList cat evidence has unexpected schema {schema}")
            def skipped(item):
                return item.numel() == 0 and item.dim() == 1
            primary_tensor = next((item for item in listed if not skipped(item)), listed[0])
            record["primary_input"] = {
                "dtype": str(primary_tensor.dtype).removeprefix("torch."),
                "rank": primary_tensor.dim(),
            }
            record["operands"] = [
                {"role": "primary_input" if tensor is primary_tensor else "operand",
                 "dtype": str(tensor.dtype).removeprefix("torch."), "rank": tensor.dim()}
                for tensor in listed
            ]
            record["input_dtypes"] = sorted({str(t.dtype).removeprefix("torch.") for t in tensors})
            record["input_ranks"] = sorted({t.dim() for t in tensors})
            record["input_shapes"] = sorted({"x".join(map(str, t.shape)) for t in tensors if t.dim() > 0})
            record["tensor_lists"] = [{
                "role": "primary_input", "path": "inputs[0]",
                "tensors": [{"position": i, "dtype": str(t.dtype).removeprefix("torch."),
                             "rank": t.dim(), "shape": list(t.shape)} for i, t in enumerate(listed)],
            }]
    if case.name in _STOCK_ROUTE_CASES:
        execution = _CASE_EXECUTION.get(case.name)
        if execution is None:
            raise ValueError(f"{case.name}: actual forward execution evidence was not captured")
        record["execution"] = dict(execution)
    if case.name.startswith("convolution.forward.bias-") or case.name.startswith("convolution.backward.bias-"):
        execution = _CASE_EXECUTION.get(case.name)
        if execution is None:
            raise ValueError(f"{case.name}: actual execution evidence was not captured")
        record.update(_convolution_coverage(case, inputs, result, execution))
    _COVERAGE[case.name] = record


def assert_reverse_second_order(case: ConformanceCase, inputs: tuple[Any, ...]) -> dict:
    """Compare seeded first/second reverse derivatives for the six named schemas."""
    recipes = {
        "arithmetic.autograd.add-tensor": ("aten::add.Tensor", torch.ops.aten.add.Tensor, ()),
        "arithmetic.autograd.add-scalar": ("aten::add.Scalar", torch.ops.aten.add.Scalar, (2.0,)),
        "arithmetic.autograd.mul-tensor": ("aten::mul.Tensor", torch.ops.aten.mul.Tensor, ()),
        "arithmetic.autograd.mul-scalar": ("aten::mul.Scalar", torch.ops.aten.mul.Scalar, (2.0,)),
        "arithmetic.autograd.sum-default": ("aten::sum.default", torch.ops.aten.sum.default, ()),
        "arithmetic.autograd.sum-dim": ("aten::sum.dim_IntList", torch.ops.aten.sum.dim_IntList, ([1], False)),
    }
    metadata_names = {
        "view.view.trainable-seed": ("aten::view.default", "view"),
        "view.reshape.copy.trainable-seed": ("aten::reshape.default", "reshape-copy"),
        "view.reshape.offset-copy.second-order": ("aten::reshape.default", "reshape-offset"),
    }
    if case.name in {"cat.rank4.native-seed", "cat.offset.trainable-seed"}:
        if case.declaration_id != "aten::cat.default":
            raise AssertionError(f"{case.name} maps to {case.declaration_id}")
        cpu_sources = case.inputs()[0]
        cpu_bases = [value.detach().clone().requires_grad_(True) for value in cpu_sources]
        vk_views = inputs[0]
        cpu_views = cpu_bases
        if case.name == "cat.offset.trainable-seed":
            cpu_views = [value.transpose(2, 3)[:, :, 1:5, 1:4] for value in cpu_bases]
            assert [(tuple(t.shape), tuple(t.stride()), t.storage_offset()) for t in vk_views] == [
                ((2, 3, 4, 3), (90, 30, 1, 6), 7), ((2, 2, 4, 3), (60, 30, 1, 6), 7)]
            assert [(tuple(t.shape), tuple(t.stride()), t.storage_offset()) for t in cpu_views] == [
                (tuple(t.shape), tuple(t.stride()), t.storage_offset()) for t in vk_views]
        cpu_dim = case.args[0]
        cpu_output = torch.cat(cpu_views, dim=cpu_dim)
        cpu_seed = torch.arange(1, cpu_output.numel() + 1, dtype=torch.float32).reshape(cpu_output.shape).div(7).requires_grad_()
        vk_seed = cpu_seed.detach().to(vk_views[0].device).requires_grad_()
        probes = [torch.arange(1, g.numel() + 1, dtype=torch.float32).reshape(g.shape).div(5 + i)
                  for i, g in enumerate(cpu_views)]
        vk_probes = [probe.to(vk_views[0].device) for probe in probes]
        pytorch_vulkan._C.synchronize()
        pytorch_vulkan._C.reset_execution_counters()
        vk_output = torch.cat(vk_views, dim=cpu_dim)
        cpu_first = torch.autograd.grad(cpu_output, cpu_views, cpu_seed, create_graph=True)
        vk_first = torch.autograd.grad(vk_output, vk_views, vk_seed, create_graph=True)
        cpu_second = torch.autograd.grad(sum((g * probe).sum() for g, probe in zip(cpu_first, probes)), cpu_seed)[0]
        vk_second = torch.autograd.grad(sum((g * probe).sum() for g, probe in zip(vk_first, vk_probes)), vk_seed)[0]
        pytorch_vulkan._C.synchronize()
        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert counters[2:] == (0, 0), f"{case.name} transfer/fallback counters: {counters}"
        torch.testing.assert_close(vk_output.cpu(), cpu_output, rtol=case.rtol, atol=case.atol)
        for actual, expected in zip(vk_first, cpu_first):
            torch.testing.assert_close(actual.cpu(), expected, rtol=case.rtol, atol=case.atol)
        torch.testing.assert_close(vk_second.cpu(), cpu_second, rtol=case.rtol, atol=case.atol)
        return {"order": 2, "graph_preserved": True}
    if case.name in metadata_names:
        expected_schema, recipe = metadata_names[case.name]
        if case.declaration_id != expected_schema:
            raise AssertionError(f"{case.name} maps to {case.declaration_id}, expected {expected_schema}")
        cpu_base = inputs[0].cpu().detach().clone().requires_grad_(True)
        vk_base = inputs[0].detach().requires_grad_(True)

        def apply(base):
            if recipe == "view":
                return torch.ops.aten.view.default(base, [12])
            if recipe == "reshape-copy":
                return torch.reshape(base.t(), [6])
            return torch.reshape(base.t().narrow(0, 1, 3), [12])

        cpu_output, vk_output = apply(cpu_base), apply(vk_base)
        cpu_seed = (torch.arange(1, cpu_output.numel() + 1, dtype=torch.float32) / 3).reshape(cpu_output.shape).requires_grad_()
        vk_seed = cpu_seed.detach().to(vk_output.device).requires_grad_()
        probe = torch.arange(1, cpu_base.numel() + 1, dtype=torch.float32).reshape(cpu_base.shape) / 5
        vk_probe = probe.to(vk_output.device)
        pytorch_vulkan._C.reset_execution_counters()
        cpu_first = torch.autograd.grad(cpu_output, cpu_base, cpu_seed, create_graph=True)[0]
        vk_first = torch.autograd.grad(vk_output, vk_base, vk_seed, create_graph=True)[0]
        cpu_seed_second = torch.autograd.grad((cpu_first * probe).sum(), cpu_seed, allow_unused=True)[0]
        vk_seed_second = torch.autograd.grad((vk_first * vk_probe).sum(), vk_seed, allow_unused=True)[0]
        cpu_hvp = vk_hvp = None
        if recipe == "reshape-offset":
            cpu_loss = (apply(cpu_base) * apply(cpu_base)).sum()
            vk_loss_output = apply(vk_base)
            vk_loss = (vk_loss_output * vk_loss_output).sum()
            cpu_grad = torch.autograd.grad(cpu_loss, cpu_base, create_graph=True)[0]
            vk_grad = torch.autograd.grad(vk_loss, vk_base, create_graph=True)[0]
            cpu_hvp = torch.autograd.grad((cpu_grad * probe.reshape(cpu_grad.shape)).sum(), cpu_base)[0]
            vk_hvp = torch.autograd.grad((vk_grad * vk_probe.reshape(vk_grad.shape)).sum(), vk_base)[0]
        pytorch_vulkan._C.synchronize()
        counters = pytorch_vulkan._C.execution_counter_snapshot()
        assert counters[2:] == (0, 0), f"{case.name} explicit transfer/fallback counters: {counters}"
        torch.testing.assert_close(vk_first.cpu(), cpu_first, rtol=case.rtol, atol=case.atol)
        assert (cpu_seed_second is None) == (vk_seed_second is None)
        if cpu_seed_second is not None:
            torch.testing.assert_close(vk_seed_second.cpu(), cpu_seed_second, rtol=case.rtol, atol=case.atol)
        if recipe == "reshape-offset":
            torch.testing.assert_close(vk_grad.cpu(), cpu_grad, rtol=case.rtol, atol=case.atol)
            torch.testing.assert_close(vk_hvp.cpu(), cpu_hvp, rtol=case.rtol, atol=case.atol)
        return {"order": 2, "graph_preserved": True}
    if case.name not in recipes:
        raise ValueError(f"no reverse second-order recipe for {case.name!r}")
    expected_schema, operation, args = recipes[case.name]
    if case.declaration_id != expected_schema:
        raise AssertionError(f"{case.name} maps to {case.declaration_id}, expected {expected_schema}")
    cpu_inputs = tuple(item.cpu().detach().clone().requires_grad_(True) for item in inputs if isinstance(item, torch.Tensor))
    vk_inputs = tuple(item.detach().requires_grad_(True) for item in inputs if isinstance(item, torch.Tensor))
    cpu_output = operation(*cpu_inputs, *args)
    vk_output = operation(*vk_inputs, *args)
    cpu_seed = torch.arange(1, cpu_output.numel() + 1, dtype=cpu_output.dtype).reshape(cpu_output.shape) / 3
    vk_seed = cpu_seed.to(vk_output.device).detach().requires_grad_(True)
    cpu_seed = cpu_seed.detach().requires_grad_(True)
    cpu_first = torch.autograd.grad(cpu_output, cpu_inputs, cpu_seed, create_graph=True)
    vk_first = torch.autograd.grad(vk_output, vk_inputs, vk_seed, create_graph=True)
    cpu_terms = []
    vk_terms = []
    for index, (cpu_gradient, vk_gradient) in enumerate(zip(cpu_first, vk_first)):
        assert cpu_gradient.requires_grad == vk_gradient.requires_grad
        torch.testing.assert_close(vk_gradient.cpu(), cpu_gradient, rtol=case.rtol, atol=case.atol)
        probe = torch.arange(1, cpu_gradient.numel() + 1, dtype=cpu_gradient.dtype).reshape(cpu_gradient.shape) / (index + 5)
        cpu_terms.append((cpu_gradient * probe).sum())
        vk_terms.append((vk_gradient * probe.to(vk_gradient.device)).sum())
    cpu_second = torch.autograd.grad(sum(cpu_terms), (*cpu_inputs, cpu_seed), allow_unused=True)
    vk_second = torch.autograd.grad(sum(vk_terms), (*vk_inputs, vk_seed), allow_unused=True)
    for cpu_gradient, vk_gradient in zip(cpu_second, vk_second):
        assert (cpu_gradient is None) == (vk_gradient is None)
        if cpu_gradient is not None:
            torch.testing.assert_close(vk_gradient.cpu(), cpu_gradient, rtol=case.rtol, atol=case.atol)
    pytorch_vulkan._C.synchronize()
    return {"order": 2, "graph_preserved": True}


def coverage_snapshot() -> dict[str, dict[str, object]]:
    return {name: dict(entry) for name, entry in _COVERAGE.items()}


def reset_coverage() -> None:
    _COVERAGE.clear()


@contextmanager
def coverage_recording():
    global COVERAGE_RECORDING
    previous_enabled = COVERAGE_RECORDING
    previous_coverage = coverage_snapshot()
    previous_executed = executed_snapshot()
    previous_execution = dict(_CASE_EXECUTION)
    reset_coverage()
    reset_executed()
    _CASE_EXECUTION.clear()
    COVERAGE_RECORDING = True
    try:
        yield
    finally:
        reset_coverage()
        _COVERAGE.update(previous_coverage)
        reset_executed()
        _EXECUTED.update(previous_executed)
        _CASE_EXECUTION.clear()
        _CASE_EXECUTION.update(previous_execution)
        COVERAGE_RECORDING = previous_enabled


class CpuExpression(str):
    """Human-readable CPU expression with an executable reference."""

    def __new__(cls, expression: str):
        return str.__new__(cls, expression)

    def __call__(self, contract, tensor: torch.Tensor, rhs: torch.Tensor | None = None):
        operation = contract.schema.split("::", 1)[1].split(".", 1)[0]
        out = torch.empty_like(tensor) if contract.output_mode == "out" else None
        if contract.operand_order == "tensor-tensor":
            if operation == "add":
                return torch.add(
                    tensor, rhs, alpha=contract.alpha, out=out
                ) if out is not None else torch.add(tensor, rhs, alpha=contract.alpha)
            if operation == "sub":
                return torch.sub(
                    tensor, rhs, alpha=contract.alpha, out=out
                ) if out is not None else torch.sub(tensor, rhs, alpha=contract.alpha)
            return torch.mul(tensor, rhs, out=out) if out is not None else torch.mul(
                tensor, rhs
            )
        if operation == "rsub":
            return torch.ops.aten.rsub.Scalar_out(
                tensor, contract.scalar, alpha=contract.alpha, out=out
            ) if out is not None else torch.ops.aten.rsub.Scalar(
                tensor, contract.scalar, alpha=contract.alpha
            )
        if operation == "add":
            return torch.add(
                tensor, contract.scalar, alpha=contract.alpha, out=out
            ) if out is not None else torch.add(
                tensor, contract.scalar, alpha=contract.alpha
            )
        if operation == "sub":
            return torch.sub(
                tensor, contract.scalar, alpha=contract.alpha, out=out
            ) if out is not None else torch.sub(
                tensor, contract.scalar, alpha=contract.alpha
            )
        return torch.mul(
            tensor, contract.scalar, out=out
        ) if out is not None else torch.mul(tensor, contract.scalar)


@dataclass(frozen=True)
class ScalarOutContract:
    """Frozen contract metadata for the Phase 2 scalar/``out=`` schemas."""

    schema: str
    case_name: str
    status: str
    operand_order: str
    scalar: float
    alpha: float
    cpu_expression: CpuExpression
    output_mode: str
    check_gradients: bool
    empty_supported: bool
    expected_counters: dict[str, int | str]
    empty_expected_counters: dict[str, int | str]
    rejection_boundaries: frozenset[str]

    def cpu_reference(self, tensor: torch.Tensor, rhs: torch.Tensor | None = None):
        return self.cpu_expression(self, tensor, rhs)


_SCALAR_OUT_COUNTERS = {
    "compute": "positive",
    "vulkan_copy": 0,
    "explicit_transfer": 0,
    "fallback": 0,
}
_SCALAR_OUT_EMPTY_COUNTERS = {
    "compute": 0,
    "vulkan_copy": 0,
    "explicit_transfer": 0,
    "fallback": 0,
}


SCALAR_OUT_CONTRACT_MATRIX = {
    "aten::add.Scalar": ScalarOutContract(
        "aten::add.Scalar",
        "scalar.add.float32",
        "supported",
        "tensor-scalar",
        2.0,
        1.0,
        CpuExpression("torch.add(tensor, scalar, alpha=alpha)"),
        "functional",
        True,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"alpha", "dtype", "device", "non_finite"}),
    ),
    "aten::add.Scalar_out": ScalarOutContract(
        "aten::add.Scalar_out",
        "out.add.scalar.float32",
        "supported",
        "tensor-scalar",
        -2.0,
        1.0,
        CpuExpression("torch.add(tensor, scalar, alpha=alpha, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "broadcast"}),
    ),
    "aten::add.out": ScalarOutContract(
        "aten::add.out",
        "out.add.tensor.float32",
        "supported",
        "tensor-tensor",
        0.0,
        1.0,
        CpuExpression("torch.add(lhs, rhs, alpha=alpha, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "broadcast"}),
    ),
    "aten::sub.Scalar": ScalarOutContract(
        "aten::sub.Scalar",
        "scalar.sub.float32",
        "supported",
        "tensor-scalar",
        -2.0,
        1.0,
        CpuExpression("torch.sub(tensor, scalar, alpha=alpha)"),
        "functional",
        True,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"alpha", "dtype", "device", "non_finite"}),
    ),
    "aten::rsub.Scalar": ScalarOutContract(
        "aten::rsub.Scalar",
        "scalar.rsub.float32",
        "supported",
        "scalar-tensor",
        2.0,
        1.0,
        CpuExpression("torch.sub(scalar, tensor, alpha=alpha)"),
        "functional",
        True,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"alpha", "dtype", "device", "non_finite"}),
    ),
    "aten::sub.Scalar_out": ScalarOutContract(
        "aten::sub.Scalar_out",
        "out.sub.scalar.float32",
        "supported",
        "tensor-scalar",
        0.0,
        1.0,
        CpuExpression("torch.sub(tensor, scalar, alpha=alpha, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "non_finite"}),
    ),
    "aten::rsub.Scalar_out": ScalarOutContract(
        "aten::rsub.Scalar_out",
        "out.rsub.scalar.float32",
        "supported",
        "scalar-tensor",
        -2.0,
        1.0,
        CpuExpression("torch.sub(scalar, tensor, alpha=alpha, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "non_finite"}),
    ),
    "aten::sub.out": ScalarOutContract(
        "aten::sub.out",
        "out.sub.tensor.float32",
        "supported",
        "tensor-tensor",
        2.0,
        1.0,
        CpuExpression("torch.sub(lhs, rhs, alpha=alpha, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"alpha", "dtype", "device", "overlap", "broadcast"}),
    ),
    "aten::mul.Scalar": ScalarOutContract(
        "aten::mul.Scalar",
        "scalar.mul.float32",
        "supported",
        "tensor-scalar",
        -2.0,
        1.0,
        CpuExpression("torch.mul(tensor, scalar)"),
        "functional",
        True,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "non_finite"}),
    ),
    "aten::mul.Scalar_out": ScalarOutContract(
        "aten::mul.Scalar_out",
        "out.mul.scalar.float32",
        "supported",
        "tensor-scalar",
        0.0,
        1.0,
        CpuExpression("torch.mul(tensor, scalar, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "non_finite"}),
    ),
    "aten::mul.out": ScalarOutContract(
        "aten::mul.out",
        "out.mul.tensor.float32",
        "supported",
        "tensor-tensor",
        2.0,
        1.0,
        CpuExpression("torch.mul(lhs, rhs, out=out)"),
        "out",
        False,
        True,
        _SCALAR_OUT_COUNTERS,
        _SCALAR_OUT_EMPTY_COUNTERS,
        frozenset({"dtype", "device", "overlap", "broadcast"}),
    ),
}

SCALAR_OUT_FUNCTIONAL_CASES = tuple(
    case for case in SCALAR_OUT_CONTRACT_MATRIX.values() if case.output_mode == "functional"
)
SCALAR_OUT_OUT_CASES = tuple(
    case for case in SCALAR_OUT_CONTRACT_MATRIX.values() if case.output_mode == "out"
)

PROMOTED_SCALAR_OUT_SCHEMAS = frozenset(SCALAR_OUT_CONTRACT_MATRIX)
REQUIRED_SCALAR_OUT_REJECTION_BOUNDARIES = frozenset(
    {"dtype", "device"}
)


def invoke_scalar_out_contract(
    contract: ScalarOutContract,
    tensor: torch.Tensor,
    *,
    rhs: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    scalar: float | None = None,
    alpha: float | None = None,
) -> torch.Tensor:
    """Execute one matrix row using its declared schema and output mode."""
    scalar = contract.scalar if scalar is None else scalar
    alpha = contract.alpha if alpha is None else alpha
    operation = contract.schema.split("::", 1)[1].split(".", 1)[0]
    is_out = contract.output_mode == "out"
    if is_out:
        out = torch.empty_like(tensor) if out is None else out

    if contract.operand_order == "tensor-tensor":
        if rhs is None:
            rhs = torch.empty_like(tensor)
        if operation == "add":
            return (
                torch.ops.aten.add.out(tensor, rhs, alpha=alpha, out=out)
                if is_out
                else torch.ops.aten.add.Tensor(tensor, rhs, alpha=alpha)
            )
        if operation == "sub":
            return (
                torch.ops.aten.sub.out(tensor, rhs, alpha=alpha, out=out)
                if is_out
                else torch.ops.aten.sub.Tensor(tensor, rhs, alpha=alpha)
            )
        return (
            torch.ops.aten.mul.out(tensor, rhs, out=out)
            if is_out
            else torch.ops.aten.mul.Tensor(tensor, rhs)
        )

    if operation == "rsub":
        if is_out:
            return torch.ops.aten.rsub.Scalar_out(
                tensor, scalar, alpha=alpha, out=out
            )
        return torch.ops.aten.rsub.Scalar(tensor, scalar, alpha=alpha)
    if operation == "add":
        return (
            torch.ops.aten.add.Scalar_out(tensor, scalar, alpha=alpha, out=out)
            if is_out
            else torch.ops.aten.add.Scalar(tensor, scalar, alpha=alpha)
        )
    if operation == "sub":
        return (
            torch.ops.aten.sub.Scalar_out(tensor, scalar, alpha=alpha, out=out)
            if is_out
            else torch.ops.aten.sub.Scalar(tensor, scalar, alpha=alpha)
        )
    return (
        torch.ops.aten.mul.Scalar_out(tensor, scalar, out=out)
        if is_out
        else torch.ops.aten.mul.Scalar(tensor, scalar)
    )


def assert_scalar_out_counters(expected: dict[str, int | str]) -> None:
    actual = {
        "compute": pytorch_vulkan._C.compute_dispatch_count(),
        "vulkan_copy": pytorch_vulkan._C.vulkan_copy_count(),
        "explicit_transfer": pytorch_vulkan._C.explicit_transfer_count(),
        "fallback": pytorch_vulkan._C.fallback_count(),
    }
    for name, expected_value in expected.items():
        if expected_value == "positive":
            assert actual[name] > 0, f"expected positive {name} counter, got {actual[name]}"
        else:
            assert actual[name] == expected_value, (
                f"expected {name} counter {expected_value}, got {actual[name]}"
            )


def vulkan_backend() -> str:
    """Return the canonical device string used by the test harness."""
    return "vk:0"


def to_vulkan_inputs(value: Any, device: str = "vk:0") -> Any:
    """Move tensor operands recursively, leaving scalar metadata untouched."""
    if isinstance(value, torch.Tensor):
        if value.dim() == 0:
            return value
        converted = value.to(device)
        if (
            tuple(value.stride()) != tuple(converted.stride())
            or value.storage_offset() != converted.storage_offset()
        ):
            converted = torch.empty_strided(
                value.size(), value.stride(), dtype=value.dtype, device=device
            )
            converted.copy_(value)
        return converted.detach().requires_grad_(value.requires_grad)
    if isinstance(value, tuple):
        return tuple(to_vulkan_inputs(item, device) for item in value)
    if isinstance(value, list):
        return [to_vulkan_inputs(item, device) for item in value]
    if isinstance(value, dict):
        return {key: to_vulkan_inputs(item, device) for key, item in value.items()}
    return value


def _walk_result(actual: Any, expected: Any, case: ConformanceCase, path: str = "result") -> None:
    if actual is None or expected is None:
        assert actual is expected, f"{case.name}: {path} None slot mismatch"
        return
    if isinstance(expected, (tuple, list)):
        assert isinstance(actual, type(expected)), f"{case.name}: {path} tuple structure mismatch"
        assert len(actual) == len(expected), f"{case.name}: {path} slot count mismatch"
        for index, (actual_child, expected_child) in enumerate(zip(actual, expected)):
            _walk_result(actual_child, expected_child, case, f"{path} slot {index}")
        return
    assert isinstance(actual, torch.Tensor) and isinstance(expected, torch.Tensor), (
        f"{case.name}: {path} must contain matching Tensor leaves"
    )
    assert tuple(actual.shape) == tuple(expected.shape), (
        f"{case.name}: {path} shape {tuple(actual.shape)} != {tuple(expected.shape)}"
    )
    assert actual.dtype == expected.dtype, (
        f"{case.name}: {path} dtype {actual.dtype} != {expected.dtype}"
    )
    torch.testing.assert_close(actual.cpu() if actual.device.type != "cpu" else actual,
                               expected.cpu() if expected.device.type != "cpu" else expected,
                               rtol=case.rtol, atol=case.atol, equal_nan=True)


def assert_result_parity(actual: Any, expected: Any, case: ConformanceCase) -> None:
    """Compare Tensor leaves recursively while preserving exact None positions."""
    if case.expected_shapes is not None:
        assert isinstance(actual, (tuple, list)), f"{case.name}: expected tuple result"
        assert len(actual) == len(case.expected_shapes), f"{case.name}: output slot count mismatch"
        for index, (value, shape) in enumerate(zip(actual, case.expected_shapes)):
            if shape is None:
                assert value is None, f"{case.name}: output slot {index} must be None"
            else:
                assert isinstance(value, torch.Tensor), f"{case.name}: output slot {index} must be a Tensor"
                assert tuple(value.shape) == shape, f"{case.name}: output slot {index} shape mismatch"
    _walk_result(actual, expected, case)


def _tensor_format(value: torch.Tensor) -> str:
    if value.is_contiguous():
        return "contiguous"
    if value.dim() == 4 and value.is_contiguous(memory_format=torch.channels_last):
        return "channels_last"
    return "other"


def _convolution_coverage(case, inputs, result, execution):
    roles = {}
    for role, value in zip(case.convolution_operand_roles, inputs):
        if value is None:
            roles[role] = {"defined": False}
        elif isinstance(value, torch.Tensor):
            roles[role] = {
                "defined": True,
                "dtype": str(value.dtype).removeprefix("torch."),
                "rank": value.dim(),
                "shape": list(value.shape),
                "strides": list(value.stride()),
                "storage_offset": value.storage_offset(),
                "format": _tensor_format(value),
            }
        else:
            raise ValueError(f"{case.name}: role {role} is not an actual Tensor/None")
    forward = case.name.startswith("convolution.forward.")
    bias = roles["bias"]["defined"]
    context = {
        "device": "vk:0",
        "cpu_oracle": "torch.nn.functional.conv2d" if forward else "aten::convolution_backward.default",
        "forward_bias_present": bias,
    }
    if forward:
        schema_args = dict(zip(("stride", "padding", "dilation", "groups"), case.args))
        schema_args = {key: list(value) if isinstance(value, tuple) else value for key, value in schema_args.items()}
    else:
        # The direct schema has bias_sizes, stride, padding, dilation, transposed,
        # output_padding, groups, output_mask; preserve those exact positional args.
        schema_args = dict(zip(
            ("bias_sizes", "stride", "padding", "dilation", "transposed", "output_padding", "groups", "output_mask"),
            case.args,
        ))
        schema_args = {key: list(value) if isinstance(value, (tuple, list)) else value for key, value in schema_args.items()}
        context["output_mask"] = list(case.args[-1])
    record = {"convolution_context": context, "convolution_operands": roles,
              "schema_args": schema_args, "execution": dict(execution)}
    if not isinstance(result, torch.Tensor):
        record["output_slots"] = [
            {"index": index, "defined": value is not None}
            if value is None else {"index": index, "defined": True,
                                   "dtype": str(value.dtype).removeprefix("torch."),
                                   "rank": value.dim(), "shape": list(value.shape)}
            for index, value in enumerate(result)
        ]
    return record


def run_case(case: ConformanceCase, device: str = "vk:0") -> Any:
    inputs = case.inputs()
    if case.convert_inputs:
        inputs = to_vulkan_inputs(inputs, device)
    kwargs = case.kwargs or {}
    return case.operation(*inputs, *case.args, **kwargs)


def run_and_compare(
    case: ConformanceCase, device: str = "vk:0", *, return_inputs: bool = False
) -> tuple[Any, ...]:
    """Compute the CPU reference and Vulkan result with counters scoped to execution."""
    cpu_inputs = case.inputs()
    reference_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in cpu_inputs
    )
    if case.setup_inputs is not None:
        reference_inputs = case.setup_inputs(reference_inputs, "cpu")
    if case.name.startswith("convolution.backward.bias-"):
        with torch.backends.mkldnn.flags(enabled=False):
            cpu_result = case.cpu_reference(
                *reference_inputs, *case.args, **(case.kwargs or {})
            )
    else:
        cpu_result = case.cpu_reference(
            *reference_inputs, *case.args, **(case.kwargs or {})
        )
    inputs = to_vulkan_inputs(cpu_inputs, device)
    if case.setup_inputs is not None:
        inputs = case.setup_inputs(inputs, device)
    if case.name.startswith("convolution."):
        pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    before_timing = pytorch_vulkan._C.timing_breakdown()
    before_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    result = case.operation(*inputs, *case.args, **(case.kwargs or {}))
    pytorch_vulkan._C.synchronize()
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    after_timing = pytorch_vulkan._C.timing_breakdown()
    after_live = pytorch_vulkan._C.live_resource_snapshot()[6]
    execution = {
        "mode": case.execution_mode,
        "compute_dispatches": dispatches,
        "vulkan_copies": vulkan_copies,
        "explicit_transfers": explicit_transfers,
        "fallbacks": fallbacks,
    }
    if case.name.startswith("convolution.forward.bias-") or case.name.startswith("convolution.backward.bias-"):
        execution = {
            "compute_dispatches": dispatches,
            "vulkan_copies": vulkan_copies,
            "explicit_transfers": explicit_transfers,
            "fallbacks": fallbacks,
            "buffer_creations_delta": after_timing["buffer_creations"] - before_timing["buffer_creations"],
            "live_allocations_delta": after_live - before_live,
        }
        _CASE_EXECUTION[case.name] = execution
    if case.name in _STOCK_ROUTE_CASES:
        if case.execution_mode != "copy" or dispatches != 0 or vulkan_copies <= 0:
            raise AssertionError(f"{case.name}: stock reshape copy route did not execute Vulkan copy work")
        _CASE_EXECUTION[case.name] = execution
    if case.execution_mode == "compute":
        if case.name.startswith("convolution.backward.bias-"):
            assert dispatches == sum(case.args[-1])
        else:
            assert dispatches > 0
    elif case.execution_mode == "copy":
        assert vulkan_copies > 0
    else:
        assert dispatches == 0 and vulkan_copies == 0
    assert explicit_transfers == 0
    assert fallbacks == 0
    return (result, cpu_result, inputs) if return_inputs else (result, cpu_result)


def run_convolution_evidence_cases(cases=None) -> dict[str, dict[str, object]]:
    """Execute and record only the named bias/mask witnesses after parity."""
    selected = tuple(cases) if cases is not None else tuple(
        case for case in ALL_CASES
        if case.name.startswith("convolution.forward.bias-")
        or case.name.startswith("convolution.backward.bias-")
    )
    with coverage_recording():
        for case in selected:
            result, expected, inputs = run_and_compare(case, return_inputs=True)
            assert_result_parity(result, expected, case)
            mark_executed(case.name)
            record_coverage(case, inputs, result, gradients=False, parity=True)
        return coverage_snapshot()


def assert_cpu_parity(case: ConformanceCase, result: torch.Tensor) -> None:
    inputs = case.inputs()
    expected = case.cpu_reference(*inputs, *case.args, **(case.kwargs or {}))
    torch.testing.assert_close(result.cpu(), expected)


def assert_vulkan_result(result: torch.Tensor, case: ConformanceCase) -> None:
    if case.expected_shapes is not None:
        assert isinstance(result, (tuple, list))
        assert len(result) == len(case.expected_shapes)
        for index, (value, shape) in enumerate(zip(result, case.expected_shapes)):
            if shape is None:
                assert value is None, f"{case.name}: output slot {index} must be None"
            else:
                assert isinstance(value, torch.Tensor)
                assert value.device == torch.device(
                    f"{torch._C._get_privateuse1_backend_name()}:0"
                )
                assert value.dtype == case.expected_dtype
                assert tuple(value.shape) == shape
        return
    assert isinstance(result, torch.Tensor)
    assert result.device == torch.device(
        f"{torch._C._get_privateuse1_backend_name()}:0"
    )
    assert result.dtype == case.expected_dtype
    if case.expected_shape is not None:
        assert tuple(result.shape) == case.expected_shape


def assert_no_vulkan_work() -> None:
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert dispatches == 0
    assert vulkan_copies == 0
    assert explicit_transfers == 0
    assert fallbacks == 0


def assert_gradients(case: ConformanceCase, device: str = "vk:0") -> None:
    cpu_inputs = case.inputs()
    vk_inputs = to_vulkan_inputs(cpu_inputs, device)
    kwargs = case.kwargs or {}
    cpu_result = case.cpu_reference(*cpu_inputs, *case.args, **kwargs)
    gradient = torch.ones_like(cpu_result)
    vk_gradient = gradient.to(device)
    pytorch_vulkan._C.reset_execution_counters()
    vk_result = case.operation(*vk_inputs, *case.args, **kwargs)
    cpu_result.backward(gradient)
    pytorch_vulkan._C.reset_execution_counters()
    vk_result.backward(vk_gradient)
    dispatches, vulkan_copies, explicit_transfers, fallbacks = (
        pytorch_vulkan._C.execution_counter_snapshot()
    )
    assert case.execution_mode in {"compute", "copy", "metadata"}
    if case.name not in {
        "view.view.trainable-seed",
        "view.reshape.copy.trainable-seed",
        "view.reshape.offset-copy.second-order",
    }:
        assert dispatches > 0 or vulkan_copies > 0, (
            f"{case.name} backward performed neither Vulkan dispatch nor copy work"
        )
    assert explicit_transfers == 0, f"{case.name} backward transferred explicitly"
    assert fallbacks == 0, f"{case.name} backward used implicit CPU fallback"
    for cpu_input, vk_input in zip(cpu_inputs, vk_inputs):
        if not isinstance(cpu_input, torch.Tensor) or not cpu_input.requires_grad:
            continue
        assert cpu_input.grad is not None
        assert vk_input.grad is not None
        assert vk_input.grad.device == torch.device(device)
        torch.testing.assert_close(
            vk_input.grad.cpu(), cpu_input.grad, rtol=case.rtol, atol=case.atol
        )


def _unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (
        torch.tensor(
            [-2.0, 0.5, 3.0], dtype=torch.float32, requires_grad=requires_grad
        ),
    )


def _unary_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    return (value.detach().requires_grad_(requires_grad),)


def _binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, -2.0], dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([3.0, 4.0], dtype=torch.float32, requires_grad=requires_grad),
    )


def _stack(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(6, dtype=torch.float32).reshape(2, 3).detach().requires_grad_(requires_grad),
        torch.arange(6, 12, dtype=torch.float32).reshape(2, 3).detach().requires_grad_(requires_grad),
    )


def _binary_strided(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    lhs = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    rhs = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, 1::2]
    return (
        lhs.detach().requires_grad_(requires_grad),
        rhs.detach().requires_grad_(requires_grad),
    )


def _loss(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[1.0, -2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=requires_grad
        ),
        torch.tensor([[0.5, 1.0], [2.0, 5.0]], dtype=torch.float32),
    )


def _mse_none(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="none")


def _mse_sum(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="sum")


def _mse_mean(input, target):
    return torch.nn.functional.mse_loss(input, target, reduction="mean")


def _mse_backward(*, requires_grad=False):
    input, target = _loss(requires_grad=False)
    return (torch.ones_like(input), input, target)


def _mse_backward_op(grad, input, target):
    return torch.ops.aten.mse_loss_backward.default(grad, input, target, 0)


def _reduction(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (
        torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32, requires_grad=requires_grad
        ),
    )


def _reduction_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4).transpose(0, 1)
    return (value.detach().requires_grad_(requires_grad),)


def _indexing(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([[1.0, 5.0], [9.0, 3.0]], dtype=torch.float32),)


def _indexing_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.tensor(
        [[1.0, 8.0, 2.0, 7.0], [9.0, 3.0, 6.0, 4.0], [5.0, 0.0, 11.0, 10.0]],
        dtype=torch.float32,
    ).t()
    return (value.detach().requires_grad_(requires_grad),)


def _view(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return (value.t().detach().requires_grad_(requires_grad),)


def _metadata_view(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    return (value.detach().requires_grad_(requires_grad),)


def _view_seed_input(*, requires_grad=False):
    return (torch.arange(1, 13, dtype=torch.float32).reshape(3, 4).requires_grad_(requires_grad),)


def _reshape_copy_seed_input(*, requires_grad=False):
    return (torch.arange(1, 7, dtype=torch.float32).reshape(2, 3).requires_grad_(requires_grad),)


def _reshape_offset_seed_input(*, requires_grad=False):
    return (torch.arange(1, 25, dtype=torch.float32).reshape(4, 6).requires_grad_(requires_grad),)


def _reshape_offset_copy(value):
    return torch.reshape(value.t().narrow(0, 1, 3), [12])


def _reshape_transpose_copy(value, shape=(6,)):
    return torch.reshape(value.t(), shape)


def _linear(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    return (
        value.detach().requires_grad_(requires_grad),
        torch.ones((3, 4), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(3, dtype=torch.float32, requires_grad=requires_grad),
    )


def _linear_strided(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value = torch.arange(8, dtype=torch.float32).reshape(2, 4)[:, ::2]
    weight = torch.arange(12, dtype=torch.float32).reshape(3, 4)[:, ::2]
    bias = torch.arange(6, dtype=torch.float32)[::2]
    return tuple(
        item.detach().requires_grad_(requires_grad) for item in (value, weight, bias)
    )


def _mm(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.arange(10, dtype=torch.float32).reshape(2, 5),
        torch.ones((5, 3), dtype=torch.float32),
    )


def _bmm(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    values = (
        torch.arange(20, dtype=torch.float32).reshape(2, 2, 5),
        torch.ones((2, 5, 3), dtype=torch.float32),
    )
    return tuple(value.detach().requires_grad_(requires_grad) for value in values)


def _addmm(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.zeros((2, 3), dtype=torch.float32),
        torch.arange(10, dtype=torch.float32).reshape(2, 5),
        torch.ones((5, 3), dtype=torch.float32),
    )


def _addmm_out(value, mat1, mat2):
    return torch.addmm(value, mat1, mat2, out=torch.empty_like(value))


def _cpu_addmm(value, mat1, mat2):
    return torch.addmm(value, mat1, mat2)


def _cpu_mm(value, other):
    return torch.mm(value, other)


def _cnn_convolution(*, requires_grad=False):
    return (
        torch.ones((8, 3, 32, 32), dtype=torch.float32, requires_grad=requires_grad),
        torch.ones((8, 3, 3, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(8, dtype=torch.float32, requires_grad=requires_grad),
    )


def _cpu_cnn_convolution(value, weight, bias, **kwargs):
    return torch.nn.functional.conv2d(value, weight, bias, **kwargs)


def _convolution(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.ones((2, 1, 8, 8), dtype=torch.float32, requires_grad=requires_grad),
        torch.ones((4, 1, 3, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _convolution_strided(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    value, weight, bias = _convolution()
    value = value.transpose(2, 3)
    weight = weight.transpose(2, 3)
    return tuple(
        item.detach().requires_grad_(requires_grad) for item in (value, weight, bias)
    )


def _conv_general(*, requires_grad=False):
    return (
        torch.randn(4, 5, 9, 9, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(3, 5, 3, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(3, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_stride_2(*, requires_grad=False):
    return (
        torch.randn(2, 3, 12, 12, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(4, 3, 3, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_padding_0(*, requires_grad=False):
    return (
        torch.randn(2, 3, 10, 10, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(4, 3, 3, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_dilation_2(*, requires_grad=False):
    return (
        torch.randn(2, 3, 12, 12, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(4, 3, 3, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_combined(*, requires_grad=False):
    return (
        torch.randn(2, 3, 12, 12, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(4, 3, 3, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_kernel_1x1(*, requires_grad=False):
    return (
        torch.randn(2, 4, 10, 10, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(6, 4, 1, 1, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(6, dtype=torch.float32, requires_grad=requires_grad),
    )


def _conv_kernel_5x5(*, requires_grad=False):
    return (
        torch.randn(2, 3, 12, 12, dtype=torch.float32, requires_grad=requires_grad),
        torch.randn(4, 3, 5, 5, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
    )


def _channel_mismatch_convolution(*, requires_grad=False):
    value, weight, bias = _convolution()
    return torch.cat((value, value), dim=1), weight, bias


def _convolution_backward_inputs(*, requires_grad=False) -> tuple[torch.Tensor, ...]:
    value, weight, _ = _convolution()
    return torch.ones((2, 4, 8, 8), dtype=torch.float32), value, weight


def _convolution_backward_mismatched_grad_inputs(*, requires_grad=False):
    value, weight, _ = _convolution()
    return torch.ones((2, 4, 7, 8), dtype=torch.float32), value, weight


def _convolution_backward_mismatched_batch_inputs(*, requires_grad=False):
    value, weight, _ = _convolution()
    return torch.ones((1, 4, 8, 8), dtype=torch.float32), value, weight


def _convolution_backward(grad, value, weight):
    return torch.ops.aten.convolution_backward.default(
        grad,
        value,
        weight,
        [4],
        [1, 1],
        [1, 1],
        [1, 1],
        False,
        [0, 0],
        1,
        [True, True, True],
    )[0]


def _convolution_backward_general_inputs(*, requires_grad=False):
    value, weight, _ = _conv_kernel_1x1()
    grad = torch.ones((2, 6, 12, 12), dtype=torch.float32)
    return grad, value, weight


def _convolution_backward_stride2_inputs(*, requires_grad=False):
    value, weight, _ = _conv_stride_2()
    return torch.ones((2, 4, 6, 6), dtype=torch.float32), value, weight


def _convolution_backward_dilation2_inputs(*, requires_grad=False):
    value, weight, _ = _conv_dilation_2()
    return torch.ones((2, 4, 12, 12), dtype=torch.float32), value, weight


def _convolution_backward_output(index):
    def operation(grad, value, weight, *, stride=1, padding=1, dilation=1, groups=1):
        return torch.ops.aten.convolution_backward.default(
            grad, value, weight, [weight.shape[0]], [stride, stride],
            [padding, padding] if isinstance(padding, int) else padding,
            [dilation, dilation], False, [0, 0], groups, [True, True, True],
        )[index]

    return operation


def _grouped_convolution_inputs(groups, outputs, *, backward=False):
    def factory(*, requires_grad=False):
        generator = torch.Generator().manual_seed(3100 + groups + outputs)
        value = torch.randn((2, 4, 8, 8), generator=generator)
        weight = torch.randn((outputs, 4 // groups, 3, 3), generator=generator)
        bias = torch.randn((outputs,), generator=generator)
        if backward:
            grad = torch.randn((2, outputs, 8, 8), generator=generator)
            return grad, value, weight
        return tuple(t.requires_grad_(requires_grad) for t in (value, weight, bias))

    return factory


def _grouped_nondivisible_inputs(*, requires_grad=False):
    return torch.ones((2, 4, 8, 8)), torch.ones((5, 2, 3, 3)), torch.ones(5)


def _convolution_parameter_guard(value, weight, bias, *, transposed=False, output_padding=(0, 0)):
    return torch.ops.aten.convolution.default(
        value, weight, bias, [2, 2], [1, 1], [1, 1], transposed,
        list(output_padding), 1,
    )


def _cpu_convolution_backward_output(
    index, grad, value, weight, stride=1, padding=1, dilation=1, groups=1
):
    if index == 0:
        return torch.nn.grad.conv2d_input(
            value.shape, weight, grad, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    if index == 1:
        return torch.nn.grad.conv2d_weight(
            value, weight.shape, grad, stride=stride, padding=padding, dilation=dilation, groups=groups
        )
    return grad.sum(dim=(0, 2, 3))


def _pooling(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4),)


def _pooling_strided(*, requires_grad=False) -> tuple[torch.Tensor]:
    value = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4).transpose(2, 3)
    return (value.detach().requires_grad_(requires_grad),)


def _masked_select(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([True, False, True]),
    )


def _masked_select_view(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    mask = torch.tensor([[True, False], [False, True], [True, False]])
    return (value.detach().requires_grad_(requires_grad), mask)


def _empty_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),)


def _gelu_tanh(value):
    return torch.nn.functional.gelu(value, approximate="tanh")


def _activation_backward(*, requires_grad=False):
    value = torch.tensor([-2.0, 0.5, 3.0], dtype=torch.float32)
    return torch.ones_like(value), value


def _sigmoid_backward_op(grad, output):
    return torch.ops.aten.sigmoid_backward.grad_input(
        grad, output, grad_input=torch.empty_like(output)
    )


def _tanh_backward_op(grad, output):
    return torch.ops.aten.tanh_backward.grad_input(
        grad, output, grad_input=torch.empty_like(output)
    )


def _gelu_backward_op(grad, input):
    return torch.ops.aten.gelu_backward.grad_input(
        grad, input, approximate="tanh", grad_input=torch.empty_like(input)
    )


def _sigmoid_backward_cpu(grad, output):
    return grad * output * (1.0 - output)


def _tanh_backward_cpu(grad, output):
    return grad * (1.0 - output * output)


def _gelu_backward_cpu(grad, input):
    input = input.detach().requires_grad_()
    result = torch.nn.functional.gelu(input, approximate="tanh")
    return torch.autograd.grad(result, input, grad)[0]


def _empty_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),
        torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),
    )


def _empty_reduction(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty((0, 3), dtype=torch.float32, requires_grad=requires_grad),)


def _bool_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([True, False]),)


def _float16_unary(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.0, -2.0], dtype=torch.float16),)


def _double_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (torch.ones(2, dtype=torch.float64), torch.ones(2, dtype=torch.float64))


def _mixed_device_add(value):
    return torch.add(value, torch.tensor([1.0, 1.0], dtype=value.dtype))


def _broadcast_binary(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.ones((2, 1), dtype=torch.float32),
        torch.ones((1, 2), dtype=torch.float32),
    )


def _optimizer_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor(2.0, dtype=torch.float32),
    )


def _optimizer_single(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.0, 2.0], dtype=torch.float32),)


def _optimizer_triple(
    *, requires_grad=False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([0.5, 1.5], dtype=torch.float32),
        torch.tensor([2.0, 3.0], dtype=torch.float32),
    )


def _optimizer_value_pair(*, requires_grad=False) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([1.0, 2.0], dtype=torch.float32),
        torch.tensor([3.0, 4.0], dtype=torch.float32),
    )


def _wrong_offset(*, requires_grad=False) -> tuple[torch.Tensor]:
    base = torch.ones(3, dtype=torch.float32)
    return (base[1:], torch.tensor(2.0, dtype=torch.float32))


def _setup_double_offset(inputs, device):
    value, _ = to_vulkan_inputs(inputs, device)
    value = value.to(torch.float64)
    return (value[1:], value[0])


def _expanded_read(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.tensor([1.5, -2.0], dtype=torch.float32).unsqueeze(1).expand(2, 2),)


def _mixed_expanded_overlap(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.empty_strided((2, 2, 2), (0, 1, 1), dtype=torch.float32),)


def _invalid_out(value):
    return torch.neg(value, out=torch.empty_like(value, dtype=torch.int64))


def _cpu_out(value):
    return torch.neg(value, out=torch.empty(value.shape, dtype=value.dtype))


def _bad_pool_params(value):
    return torch.ops.aten._adaptive_avg_pool2d.default(value, (2, 2))


def _argmax_bad_out(value):
    out = torch.empty((1,), dtype=torch.int64, device=value.device)
    return torch.argmax(value, dim=1, out=out)


def _argmax_offset_out(value):
    out = torch.empty((3,), dtype=torch.int64, device=value.device)[1:]
    return torch.argmax(value, dim=1, out=out)


def _unsupported_overload(value):
    return value.neg_()


def _adaptive_pool(*, requires_grad=False) -> tuple[torch.Tensor]:
    return (torch.ones((2, 4, 3, 5), dtype=torch.float32, requires_grad=requires_grad),)


def _adaptive_pool_backward_inputs(*, requires_grad=False) -> tuple[torch.Tensor, ...]:
    return torch.ones((2, 4, 1, 1), dtype=torch.float32), _adaptive_pool()[0]


def _adaptive_pool_backward(grad, value):
    return torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, value)


def _normalization(*, requires_grad=False):
    return (
        torch.randn(2, 4, dtype=torch.float32, requires_grad=requires_grad),
        torch.ones(4, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32, requires_grad=requires_grad),
        torch.zeros(4, dtype=torch.float32),
        torch.ones(4, dtype=torch.float32),
    )


def _batch_norm_2d(*, requires_grad=False):
    return (torch.randn(7, 3, dtype=torch.float32, requires_grad=requires_grad),)


def _batch_norm_4d(*, requires_grad=False):
    return (torch.randn(3, 4, 5, 7, dtype=torch.float32, requires_grad=requires_grad),)


def _batch_norm_4d_single_channel(*, requires_grad=False):
    return (torch.randn(5, 1, 3, 3, dtype=torch.float32, requires_grad=requires_grad),)


def _batch_norm_parameters(channels, device=None):
    return (
        torch.ones(channels, dtype=torch.float32, device=device),
        torch.zeros(channels, dtype=torch.float32, device=device),
        torch.zeros(channels, dtype=torch.float32, device=device),
        torch.ones(channels, dtype=torch.float32, device=device),
    )


def _cpu_native_batch_norm(value, weight, bias, running_mean, running_var,
                           training, momentum, eps):
    return torch.ops.aten.native_batch_norm(
        value, weight, bias, running_mean, running_var, training, momentum, eps
    )


def _native_batch_norm(value):
    weight, bias, running_mean, running_var = _batch_norm_parameters(
        value.shape[1], value.device
    )
    return torch.ops.aten.native_batch_norm(
        value, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )[0]


def _cpu_native_batch_norm_with_parameters(value):
    weight, bias, running_mean, running_var = _batch_norm_parameters(value.shape[1])
    return _cpu_native_batch_norm(
        value, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )[0]


def _native_batch_norm_output(input, weight, bias, running_mean, running_var):
    return torch.ops.aten.native_batch_norm.default(
        input, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )[0]


def _normalization_backward(*, requires_grad=False):
    input, weight, bias, running_mean, running_var = _normalization()
    _, save_mean, save_inv = torch.ops.aten.native_batch_norm.default(
        input, weight, bias, running_mean, running_var, True, 0.1, 1e-5
    )
    return (
        torch.ones_like(input),
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_inv,
    )


def _normalization_backward_4d(*, requires_grad=False):
    input = torch.arange(420, dtype=torch.float32).reshape(3, 4, 5, 7) / 16
    input.requires_grad_(requires_grad)
    weight, _, running_mean, running_var = _batch_norm_parameters(input.shape[1])
    _, save_mean, save_inv = torch.ops.aten.native_batch_norm.default(
        input, weight, torch.zeros_like(weight), running_mean, running_var,
        True, 0.1, 1e-5
    )
    return (
        (
            torch.arange(420, dtype=torch.float32).remainder(7).reshape(3, 4, 5, 7)
            - 3
        ) / 4,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_inv,
    )


def _native_batch_norm_backward_output(
    grad, input, weight, running_mean, running_var, save_mean, save_inv
):
    return torch.ops.aten.native_batch_norm_backward.default(
        grad,
        input,
        weight,
        running_mean,
        running_var,
        save_mean,
        save_inv,
        True,
        1e-5,
        [True, True, True],
    )[0]


def _nll_forward(*, requires_grad=False):
    return (
        torch.randn(2, 3, dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([1, 2], dtype=torch.int64),
    )


def _nll_forward_output(logits, labels):
    return torch.ops.aten.nll_loss_forward.default(
        torch.log_softmax(logits, dim=1), labels, None, 1, -100
    )[0]


def _nll_backward(*, requires_grad=False):
    logits = torch.randn(2, 3, dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.int64)
    log_probs = torch.log_softmax(logits, dim=1)
    _, total = torch.ops.aten.nll_loss_forward.default(log_probs, labels, None, 1, -100)
    return torch.ones(1, dtype=torch.float32), log_probs, labels, total.reshape(1)


def _nll_forward_wide(*, requires_grad=False):
    return (
        torch.randn(8, 5, dtype=torch.float32, requires_grad=requires_grad),
        torch.tensor([0, 4, 2, 1, 3, 2, 4, 0], dtype=torch.int64),
    )


def _nll_forward_none_large(*, requires_grad=False):
    logits = torch.randn(512, 5, dtype=torch.float32, requires_grad=requires_grad)
    labels = torch.arange(512, dtype=torch.int64) % 5
    return logits, labels


def _nll_backward_wide(*, requires_grad=False):
    logits = torch.randn(8, 5, dtype=torch.float32)
    labels = torch.tensor([0, 4, 2, 1, 3, 2, 4, 0], dtype=torch.int64)
    log_probs = torch.log_softmax(logits, dim=1)
    _, total = torch.ops.aten.nll_loss_forward.default(log_probs, labels, None, 1, -100)
    return torch.ones(1, dtype=torch.float32), log_probs, labels, total.reshape(1)


def _cpu_nll_backward_wide(grad, log_probs, labels, total):
    return _cpu_nll_backward_reduction(grad, log_probs, labels, "mean")


def _cpu_nll_backward_reduction(grad, log_probs, labels, reduction):
    log_probs = log_probs.detach().requires_grad_()
    reference = torch.nn.functional.nll_loss(
        log_probs, labels, reduction=reduction, ignore_index=-100
    )
    grad_outputs = grad.reshape(()) if reduction != "none" else grad
    return torch.autograd.grad(reference, log_probs, grad_outputs=grad_outputs)[0]


def _nll_backward_reduction_inputs(reduction):
    def factory(*, requires_grad=False):
        logits, labels = _nll_forward_wide()
        log_probs = torch.log_softmax(logits, dim=1)
        _, total = torch.ops.aten.nll_loss_forward.default(log_probs, labels, None, 1, -100)
        grad = torch.arange(1, 9, dtype=torch.float32) if reduction == "none" else torch.ones(1, dtype=torch.float32)
        return grad, log_probs, labels, total.reshape(1)
    return factory


def _nll_backward_reduction_output(reduction):
    reduction_code = {"none": 0, "mean": 1, "sum": 2}[reduction]
    def operation(grad, log_probs, labels, total):
        return torch.ops.aten.nll_loss_backward.default(
            grad if reduction_code == 0 else grad.reshape(()), log_probs, labels,
            None, reduction_code, -100, total.reshape(())
        )
    return operation


def _nll_backward_output(grad, log_probs, labels, total):
    return torch.ops.aten.nll_loss_backward.default(
        grad.reshape(()).to(log_probs.device), log_probs, labels, None, 1, -100,
        total.reshape(()).to(log_probs.device)
    )


def _cpu_neg(value):
    return torch.neg(value)


def _cpu_add(lhs, rhs):
    return torch.add(lhs, rhs)


def _cpu_lerp(lhs, rhs):
    return torch.lerp(lhs, rhs, 0.25)


def _lerp_out(lhs, rhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.lerp.Scalar_out(lhs, rhs, 0.25, out=out)


def _lerp_inplace(lhs, rhs):
    return torch.ops.aten.lerp_.Scalar(lhs, rhs, 0.25)


def _addcmul_inplace(lhs, tensor1, tensor2):
    return torch.ops.aten.addcmul_.default(lhs, tensor1, tensor2, value=0.25)


def _addcdiv_inplace(lhs, tensor1, tensor2):
    return torch.ops.aten.addcdiv_.default(lhs, tensor1, tensor2, value=0.25)


def _add_inplace(lhs, rhs):
    owns_training_step = not pytorch_vulkan._C.training_step_active()
    if owns_training_step:
        pytorch_vulkan._C.begin_training_step()
    try:
        result = torch.ops.aten.add_.Tensor(lhs, rhs, alpha=1.0)
        if owns_training_step:
            pytorch_vulkan._C.end_training_step()
        return result
    except Exception:
        if owns_training_step:
            pytorch_vulkan._C.cancel_training_step()
        raise


def _mul_scalar_inplace(lhs):
    owns_training_step = not pytorch_vulkan._C.training_step_active()
    if owns_training_step:
        pytorch_vulkan._C.begin_training_step()
    try:
        result = torch.ops.aten.mul_.Scalar(lhs, 2.0)
        if owns_training_step:
            pytorch_vulkan._C.end_training_step()
        return result
    except Exception:
        if owns_training_step:
            pytorch_vulkan._C.cancel_training_step()
        raise


def _zero_inplace(lhs):
    return torch.ops.aten.zero_.default(lhs)


def _sqrt_out(lhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.sqrt.out(lhs, out=out)


def _sqrt_out(lhs):
    out = torch.empty_like(lhs)
    return torch.ops.aten.sqrt.out(lhs, out=out)


def _cpu_sum(value, dim, keepdim=False):
    return torch.sum(value, dim=dim, keepdim=keepdim)


def _cpu_mean(value, dim=None, keepdim=False):
    return torch.mean(value, dim=dim, keepdim=keepdim)


def _sum_dims_out(value):
    destination = torch.empty((), dtype=value.dtype, device=value.device)
    return torch.ops.aten.sum.IntList_out(value, [0, 1], out=destination)


def _mean_dims_out(value):
    destination = torch.empty((), dtype=value.dtype, device=value.device)
    return torch.ops.aten.mean.out(value, [0, 1], out=destination)


def _cpu_sum_dims_out(value):
    return torch.sum(value, dim=[0, 1])


def _cpu_mean_dims_out(value):
    return torch.mean(value, dim=[0, 1])


def _cpu_argmax(value, dim=None, keepdim=False):
    return torch.argmax(value, dim=dim, keepdim=keepdim)


def _cpu_prod(value, dim, keepdim=False):
    return torch.prod(value, dim=dim, keepdim=keepdim)


def _softmax(value, dim):
    return torch.softmax(value, dim=dim)


def _log_softmax(value, dim):
    return torch.log_softmax(value, dim=dim)


def _amax_out(value, dim):
    return torch.ops.aten.amax.out(
        value,
        [dim],
        False,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _amin_out(value, dim):
    return torch.ops.aten.amin.out(
        value,
        [dim],
        False,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _prod_int_out(value, dim):
    return torch.ops.aten.prod.int_out(
        value,
        dim,
        False,
        dtype=value.dtype,
        out=torch.empty((value.size(0),), dtype=value.dtype, device=value.device),
    )


def _softmax_out(value, dim):
    return torch.ops.aten._softmax.out(value, dim, False, out=torch.empty_like(value))


def _log_softmax_out(value, dim):
    return torch.ops.aten._log_softmax.out(
        value, dim, False, out=torch.empty_like(value)
    )


def _softmax_backward_inputs(*, requires_grad=False):
    output = torch.softmax(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32), dim=1
    )
    grad = torch.ones_like(output)
    return grad, output


def _softmax_backward_out(grad, output):
    return torch.ops.aten._softmax_backward_data.out(
        grad, output, 1, torch.float32, grad_input=torch.empty_like(output)
    )


def _log_softmax_backward_out(grad, output):
    return torch.ops.aten._log_softmax_backward_data.out(
        grad, output, 1, torch.float32, out=torch.empty_like(output)
    )


def _cpu_reshape(value, shape):
    return torch.reshape(value, shape)


def _cpu_as_strided(value, size, stride):
    return torch.as_strided(value, size, stride)


def _cpu_linear(value, weight, bias):
    return torch.nn.functional.linear(value, weight, bias)


def _linear_attention(*, requires_grad=False):
    return (
        torch.arange(128 * 256, dtype=torch.float32).reshape(1, 128, 256).detach().requires_grad_(requires_grad),
        torch.arange(256 * 256, dtype=torch.float32).reshape(256, 256).detach().requires_grad_(requires_grad),
        torch.arange(256, dtype=torch.float32).detach().requires_grad_(requires_grad),
    )


def _cpu_convolution(value, weight, bias=None, **kwargs):
    return torch.nn.functional.conv2d(value, weight, bias, **kwargs)


def _cpu_max_pool(value, kernel_size):
    return torch.nn.functional.max_pool2d(value, kernel_size)


def _cpu_masked_select(value, mask):
    return torch.masked_select(value, mask)


def _cpu_adaptive_pool(value, output_size):
    return torch.nn.functional.adaptive_avg_pool2d(value, output_size)


def _abs_out_inputs(*, requires_grad=False):
    return torch.randn(3, 4), torch.empty(3, 4)


def _cpu_abs_out(value, out_buffer):
    out = out_buffer.clone()
    return torch.ops.aten.abs.out(value, out=out)


def _abs_out(value, out_buffer):
    return torch.ops.aten.abs.out(value, out=out_buffer)


def _out_inputs(*, requires_grad=False, dtype=torch.float32, shape=(3, 4)):
    value = torch.randn(shape, dtype=dtype)
    return value, torch.empty(shape, dtype=dtype)


def _bool_out_inputs(*, requires_grad=False):
    value = torch.randn(3, 4)
    other = torch.randn(3, 4)
    return value, other, torch.empty((3, 4), dtype=torch.bool)


def _scalar_bool_out_inputs(*, requires_grad=False):
    value = torch.randn(3, 4)
    return value, torch.empty((3, 4), dtype=torch.bool)


def _bitwise_bool_out_inputs(*, requires_grad=False):
    value = torch.rand(3, 4) < 0.5
    other = torch.rand(3, 4) < 0.5
    return value, other, torch.empty((3, 4), dtype=torch.bool)


def _inplace_inputs(*, requires_grad=False):
    return (torch.arange(1, 13, dtype=torch.float32).reshape(3, 4),)


def _inplace_binary_inputs(*, requires_grad=False):
    return (
        torch.arange(1, 13, dtype=torch.float32).reshape(3, 4),
        torch.full((3, 4), 2.0, dtype=torch.float32),
    )


def _cpu_inplace(op, value, *operands):
    out = value.clone()
    before = out.clone()
    op(out, *operands)
    assert not torch.equal(out, before), "in-place reference did not mutate its input"
    return out


def _cpu_out_via_op(op, *values):
    out = values[-1].clone()
    return op(*values[:-1], out=out)


def _cpu_argmax_out(value, dim, out_buffer):
    out = out_buffer.clone()
    return torch.ops.aten.argmax.out(value, dim, False, out=out)


def _argmax_inputs(*, requires_grad=False):
    value = torch.randn(3, 4)
    return value, torch.zeros((3,), dtype=torch.int64)


def _ne_scalar_out(value, out):
    return torch.ops.aten.ne.Scalar_out(value, 0.0, out=out)


def _neg_out(value, out):
    return torch.ops.aten.neg.out(value, out=out)


def _relu_out(value, out):
    return torch.ops.aten.relu.out(value, out=out)


def _eq_tensor_out(value, other, out):
    return torch.ops.aten.eq.Tensor_out(value, other, out=out)


def _bitwise_and_out(value, other, out):
    return torch.ops.aten.bitwise_and.Tensor_out(value, other, out=out)


def _argmax_out(value, out, dim, keepdim=False):
    return torch.ops.aten.argmax.out(value, dim, keepdim, out=out)


_MANIFEST_CASES = {
    case["name"]: (entry["schema"], case["supported"])
    for entry in _MANIFEST["entries"]
    for case in entry["test_cases"]
}
_MANIFEST_CASES.update({
    "convolution.forward.bias-present": ("aten::convolution.default", True),
    "convolution.forward.bias-absent": ("aten::convolution.default", True),
    **{
        f"convolution.backward.bias-{state}.mask-{mask:03b}":
        ("aten::convolution_backward.default", True)
        for state in ("present", "absent") for mask in range(8)
    },
})
MANIFEST_CASE_NAMES = frozenset(_MANIFEST_CASES)


def _case(
    name,
    family,
    operation,
    factory,
    *,
    args=(),
    kwargs=None,
    supported=True,
    error_pattern=r"Vulkan",
    expected_dtype=torch.float32,
    expected_shape=None,
    check_gradients=False,
    cpu_reference=None,
    autograd_supported=True,
    autograd_error_type=None,
    autograd_error_pattern=None,
    requires_grad_inputs=False,
    rtol=1e-5,
    atol=1e-8,
    execution_mode="compute",
    convert_inputs=True,
    setup_inputs=None,
    declared_shapes=(),
    expected_shapes=None,
    convolution_operand_roles=(),
):
    manifest_case = _MANIFEST_CASES.get(name)
    if manifest_case is None:
        return None
    declaration_id, manifest_supported = manifest_case
    if supported != manifest_supported:
        raise ValueError(f"case {name} support flag disagrees with capability manifest")
    case = ConformanceCase(
        name,
        family,
        declaration_id,
        operation,
        factory,
        cpu_reference,
        args,
        kwargs,
        manifest_supported,
        error_pattern,
        expected_dtype,
        expected_shape,
        check_gradients,
        autograd_supported,
        autograd_error_type,
        autograd_error_pattern,
        requires_grad_inputs,
        rtol,
        atol,
        execution_mode,
        convert_inputs,
        setup_inputs,
        declared_shapes,
        expected_shapes,
        convolution_operand_roles,
    )
    return case


def _transfer_pair(*, requires_grad=False):
    source = torch.arange(4, dtype=torch.float32, requires_grad=requires_grad)
    destination = torch.empty_like(source)
    return destination, source


def _copy_from(destination, source):
    return torch.ops.aten._copy_from.default(source, destination)


def _copy_from_cpu_reference(destination, source):
    return destination.copy_(source)


def _copy_in_place(destination, source):
    return torch.ops.aten.copy_.default(destination, source)


def _empty_like_template(template):
    return torch.empty_like(template, device=template.device)


def _empty_strided_like_template(template):
    return torch.empty_strided(
        template.size(), template.stride(), dtype=template.dtype, device=template.device
    )


def _empty_strided_zero_template(*, requires_grad=False):
    return (torch.empty_strided((0, 2), (1, 2), dtype=torch.float32),)


def _scalar_add(value):
    return torch.add(value, 2.0, alpha=1.0)


def _scalar_sub(value):
    return torch.sub(value, 2.0, alpha=1.0)


def _scalar_mul(value):
    return torch.mul(value, 2.0)


def _scalar_add_out(value):
    return torch.add(value, 2.0, alpha=1.0, out=torch.empty_like(value))


def _scalar_sub_out(value):
    return torch.sub(value, 2.0, alpha=1.0, out=torch.empty_like(value))


def _scalar_mul_out(value):
    return torch.mul(value, 2.0, out=torch.empty_like(value))


def _tensor_add_out(lhs, rhs):
    return torch.add(lhs, rhs, alpha=1.0, out=torch.empty_like(lhs))


def _tensor_sub_out(lhs, rhs):
    return torch.sub(lhs, rhs, alpha=1.0, out=torch.empty_like(lhs))


def _tensor_mul_out(lhs, rhs):
    return torch.mul(lhs, rhs, out=torch.empty_like(lhs))


def _rscalar(value):
    return torch.ops.aten.rsub.Scalar(value, 2.0, 1.0)


def _rscalar_out(value):
    return torch.ops.aten.rsub.Scalar_out(value, 2.0, 1.0, out=torch.empty_like(value))


def _cat_values(shapes, *, requires_grad=False, channels_last=False):
    values = []
    for index, shape in enumerate(shapes):
        value = (torch.arange(1, torch.tensor(shape).prod().item() + 1, dtype=torch.float32)
                 .reshape(shape) + index * 30)
        if channels_last:
            value = value.contiguous(memory_format=torch.channels_last)
        values.append(value.detach().requires_grad_(requires_grad))
    return (values,)


def _cat_native(*, requires_grad=False):
    import torch.nn.functional as F
    gen = torch.Generator(device="cpu").manual_seed(73)
    x = torch.randn((1, 4, 6, 7), generator=gen)
    weight = torch.randn((6, 2, 3, 2), generator=gen)
    grad_output = torch.randn((1, 6, 3, 5), generator=gen)
    grad_input = torch.randn((1, 4, 6, 7), generator=gen)
    assert tuple(F.conv2d(x, weight, stride=(2, 1), padding=(1, 0), dilation=(1, 2), groups=2).shape) == (1, 6, 3, 5)
    groups = []
    for group in range(2):
        vi = grad_input[:, group * 2:(group + 1) * 2].permute(1, 0, 2, 3)
        go = grad_output[:, group * 3:(group + 1) * 3].permute(1, 0, 2, 3)
        groups.append(F.conv2d(vi, go, stride=(1, 2), padding=(1, 0), dilation=(2, 1)).detach().requires_grad_(requires_grad))
    return (groups,)


def _cat_offset(*, requires_grad=False):
    return ([torch.arange(n, dtype=torch.float32).reshape(shape).detach().requires_grad_(requires_grad)
             for n, shape in ((180, (2, 3, 5, 6)), (120, (2, 2, 5, 6)))],)


def _cat_offset_setup(inputs, device):
    del device
    return [base.transpose(2, 3)[:, :, 1:5, 1:4] for base in inputs[0]],


def _cat_call(tensors, dim):
    return torch.cat(tensors, dim=dim)


def _cat_reference(tensors, dim):
    return torch.cat(tensors, dim=dim)


def _cat_channels_last(*, requires_grad=False):
    return _cat_values(((2, 2, 3, 4), (2, 2, 3, 4)), requires_grad=requires_grad, channels_last=True)


def _cat_empty(*, requires_grad=False):
    del requires_grad
    return ([torch.empty((0,), dtype=torch.float32), torch.arange(6, dtype=torch.float32).reshape(2, 3)],)


def _convolution_evidence_inputs(bias_present: bool, *, backward: bool):
    generator = torch.Generator(device="cpu").manual_seed(8042 + int(bias_present) + 3 * int(backward))
    value = torch.randn((1, 4, 5, 6), generator=generator, dtype=torch.float32)
    weight = torch.randn((4, 2, 2, 2), generator=generator, dtype=torch.float32)
    bias = torch.randn((4,), generator=generator, dtype=torch.float32) if bias_present else None
    if backward:
        grad_output = torch.randn((1, 4, 4, 5), generator=generator, dtype=torch.float32)
        return grad_output, value, weight, bias
    return value, weight, bias


def _convolution_context_setup(inputs, device):
    value, weight, bias = inputs[1:4] if len(inputs) == 4 else inputs
    torch.nn.functional.conv2d(
        value, weight, bias, stride=(1, 1), padding=(0, 0),
        dilation=(1, 1), groups=2,
    )
    return inputs


def _convolution_backward_with_context(grad_output, value, weight, _bias,
                                       bias_sizes, stride, padding, dilation,
                                       transposed, output_padding, groups, output_mask):
    return torch.ops.aten.convolution_backward.default(
        grad_output, value, weight, bias_sizes, stride, padding, dilation,
        transposed, output_padding, groups, output_mask,
    )


def _convolution_evidence_cases():
    cases = []
    forward_args = ((1, 1), (0, 0), (1, 1), 2)
    for bias_present in (True, False):
        state = "present" if bias_present else "absent"
        cases.append(ConformanceCase(
            f"convolution.forward.bias-{state}", "convolution",
            "aten::convolution.default", torch.nn.functional.conv2d,
            lambda requires_grad=False, present=bias_present: _convolution_evidence_inputs(present, backward=False),
            torch.nn.functional.conv2d, forward_args, None, True, r"Vulkan",
            torch.float32, (1, 4, 4, 5), False, True, None, None, False,
            2e-4, 2e-4, "compute", True, _convolution_context_setup,
            (), None, ("input", "weight", "bias"),
        ))
        for mask_bits in range(8):
            mask = tuple(bool(mask_bits & (1 << shift)) for shift in range(3))
            mask_name = "".join("1" if bit else "0" for bit in mask)
            schema_args = ([4], [1, 1], [0, 0], [1, 1], False, [0, 0], 2, list(mask))
            shapes = ((1, 4, 5, 6), (4, 2, 2, 2), (4,))
            cases.append(ConformanceCase(
                f"convolution.backward.bias-{state}.mask-{mask_name}", "convolution",
                "aten::convolution_backward.default", _convolution_backward_with_context,
                lambda requires_grad=False, present=bias_present: _convolution_evidence_inputs(present, backward=True),
                _convolution_backward_with_context, schema_args, None, True, r"Vulkan",
                torch.float32, None, False, True, None, None, False,
                3e-4, 3e-4, "compute", True, _convolution_context_setup,
                (),
                tuple(shape if bit else None for shape, bit in zip(shapes, mask)),
                ("grad_output", "input", "weight", "bias"),
            ))
    return tuple(cases)


ALL_CASES = tuple(
    case for case in (
    _case("cat.rank1.forward", "cat", _cat_call, lambda **kw: _cat_values(((2,), (3,)), **kw), args=(0,), cpu_reference=_cat_reference, expected_shape=(5,)),
    _case("cat.rank2.forward", "cat", _cat_call, lambda **kw: _cat_values(((2, 3), (2, 4)), **kw), args=(1,), cpu_reference=_cat_reference, expected_shape=(2, 7), declared_shapes=("2x3", "2x4")),
    _case("cat.rank3.forward", "cat", _cat_call, lambda **kw: _cat_values(((2, 2, 3), (2, 2, 4)), **kw), args=(2,), cpu_reference=_cat_reference, expected_shape=(2, 2, 7), declared_shapes=("2x2x3", "2x2x4")),
    _case("cat.rank4.native-seed", "cat", _cat_call, _cat_native, args=(1,), cpu_reference=_cat_reference, expected_shape=(2, 6, 4, 2), requires_grad_inputs=True, declared_shapes=("2x3x4x2",)),
    _case("cat.offset.trainable-seed", "cat", _cat_call, _cat_offset, args=(1,), cpu_reference=_cat_reference, expected_shape=(2, 5, 4, 3), requires_grad_inputs=True, setup_inputs=_cat_offset_setup, declared_shapes=("2x3x4x3", "2x2x4x3")),
    _case("cat.channels-last.forward", "cat", _cat_call, _cat_channels_last, args=(1,), cpu_reference=_cat_reference, expected_shape=(2, 4, 3, 4), declared_shapes=("2x2x3x4",)),
    _case("cat.empty.reference", "cat", _cat_call, _cat_empty, args=(1,), cpu_reference=_cat_reference, expected_shape=(2, 3), execution_mode="compute", declared_shapes=("2x3",)),
    _case(
        "unary.neg.float32",
        "unary",
        torch.neg,
        _unary,
        cpu_reference=_cpu_neg,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.neg.float32.strided",
        "unary",
        torch.neg,
        _unary_strided,
        cpu_reference=_cpu_neg,
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "unary.abs.float32",
        "unary",
        torch.abs,
        _unary,
        cpu_reference=torch.abs,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.relu.float32.empty",
        "unary",
        torch.relu,
        _empty_unary,
        cpu_reference=torch.relu,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "unary.sigmoid.float32",
        "unary",
        torch.sigmoid,
        _unary,
        cpu_reference=torch.sigmoid,
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "unary.tanh.float32",
        "unary",
        torch.tanh,
        _unary,
        cpu_reference=torch.tanh,
        expected_shape=(3,),
        check_gradients=True,
        rtol=1e-4,
        atol=2e-5,
    ),
    _case(
        "sequence.stack.float32",
        "sequence",
        lambda first, second: torch.stack((first, second), dim=1),
        _stack,
        cpu_reference=lambda first, second: torch.stack((first, second), dim=1),
        expected_shape=(2, 2, 3),
        check_gradients=True,
        execution_mode="copy",
    ),
    _case(
        "unary.gelu.tanh.float32",
        "unary",
        _gelu_tanh,
        _unary,
        cpu_reference=_gelu_tanh,
        expected_shape=(3,),
        check_gradients=True,
        rtol=2e-5,
        atol=2e-6,
    ),
    _case(
        "unary.sigmoid.float32.empty",
        "unary",
        torch.sigmoid,
        _empty_unary,
        cpu_reference=torch.sigmoid,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "unary.sigmoid.backward",
        "unary",
        _sigmoid_backward_op,
        _activation_backward,
        cpu_reference=_sigmoid_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "unary.tanh.backward",
        "unary",
        _tanh_backward_op,
        _activation_backward,
        cpu_reference=_tanh_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
        rtol=1e-4,
        atol=2e-5,
    ),
    _case(
        "unary.gelu.tanh.backward",
        "unary",
        _gelu_backward_op,
        _activation_backward,
        cpu_reference=_gelu_backward_cpu,
        expected_shape=(3,),
        execution_mode="copy",
        rtol=3e-4,
        atol=2e-5,
    ),
    _case(
        "binary.add.float32",
        "binary",
        torch.add,
        _binary,
        cpu_reference=_cpu_add,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "binary.sub.float32.strided",
        "binary",
        torch.sub,
        _binary_strided,
        cpu_reference=torch.sub,
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "binary.mul.float32.empty",
        "binary",
        torch.mul,
        _empty_binary,
        cpu_reference=torch.mul,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "loss.mse.none",
        "loss",
        _mse_none,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="none"),
        expected_shape=(2, 2),
        check_gradients=True,
    ),
    _case(
        "loss.mse.sum",
        "loss",
        _mse_sum,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="sum"),
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "loss.mse.mean",
        "loss",
        _mse_mean,
        _loss,
        cpu_reference=lambda x, y: torch.nn.functional.mse_loss(x, y, reduction="mean"),
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "loss.mse.backward",
        "loss",
        _mse_backward_op,
        _mse_backward,
        cpu_reference=lambda g, x, y: torch.ops.aten.mse_loss_backward.default(
            g, x, y, 0
        ),
        expected_shape=(2, 2),
    ),
    _case(
        "scalar.add.float32",
        "scalar and out",
        _scalar_add,
        _unary,
        cpu_reference=lambda value: torch.add(value, 2.0, alpha=1.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "scalar.sub.float32",
        "scalar and out",
        _scalar_sub,
        _unary,
        cpu_reference=lambda value: torch.sub(value, 2.0, alpha=1.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "scalar.mul.float32",
        "scalar and out",
        _scalar_mul,
        _unary,
        cpu_reference=lambda value: torch.mul(value, 2.0),
        expected_shape=(3,),
        check_gradients=True,
    ),
    _case(
        "out.add.scalar.float32",
        "scalar and out",
        _scalar_add_out,
        _unary,
        cpu_reference=lambda value: torch.add(
            value, 2.0, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "out.sub.scalar.float32",
        "scalar and out",
        _scalar_sub_out,
        _unary,
        cpu_reference=lambda value: torch.sub(
            value, 2.0, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "out.mul.scalar.float32",
        "scalar and out",
        _scalar_mul_out,
        _unary,
        cpu_reference=lambda value: torch.mul(value, 2.0, out=torch.empty_like(value)),
        expected_shape=(3,),
    ),
    _case(
        "out.add.tensor.float32",
        "scalar and out",
        _tensor_add_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.add(
            lhs, rhs, alpha=1.0, out=torch.empty_like(lhs)
        ),
        expected_shape=(2,),
    ),
    _case(
        "out.sub.tensor.float32",
        "scalar and out",
        _tensor_sub_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.sub(
            lhs, rhs, alpha=1.0, out=torch.empty_like(lhs)
        ),
        expected_shape=(2,),
    ),
    _case(
        "out.mul.tensor.float32",
        "scalar and out",
        _tensor_mul_out,
        _binary,
        cpu_reference=lambda lhs, rhs: torch.mul(lhs, rhs, out=torch.empty_like(lhs)),
        expected_shape=(2,),
    ),
    _case(
        "scalar.rsub.float32",
        "scalar and out",
        _rscalar,
        _unary,
        cpu_reference=lambda value: torch.sub(2.0, value, alpha=1.0),
        expected_shape=(3,),
    ),
    _case(
        "out.rsub.scalar.float32",
        "scalar and out",
        _rscalar_out,
        _unary,
        cpu_reference=lambda value: torch.sub(
            2.0, value, alpha=1.0, out=torch.empty_like(value)
        ),
        expected_shape=(3,),
    ),
    _case(
        "arithmetic.autograd.add-tensor", "arithmetic autograd",
        torch.ops.aten.add.Tensor, _binary,
        cpu_reference=torch.ops.aten.add.Tensor, check_gradients=True,
        expected_shape=(2,),
    ),
    _case(
        "arithmetic.autograd.add-scalar", "arithmetic autograd",
        torch.ops.aten.add.Scalar, _unary, args=(2.0,),
        cpu_reference=torch.ops.aten.add.Scalar, check_gradients=True,
        expected_shape=(3,),
    ),
    _case(
        "arithmetic.autograd.mul-tensor", "arithmetic autograd",
        torch.ops.aten.mul.Tensor, _stack,
        cpu_reference=torch.ops.aten.mul.Tensor, check_gradients=True,
        expected_shape=(2, 3),
    ),
    _case(
        "arithmetic.autograd.mul-scalar", "arithmetic autograd",
        torch.ops.aten.mul.Scalar, _unary, args=(2.0,),
        cpu_reference=torch.ops.aten.mul.Scalar, check_gradients=True,
        expected_shape=(3,),
    ),
    _case(
        "arithmetic.autograd.sum-default", "arithmetic autograd",
        torch.ops.aten.sum.default, _reduction,
        cpu_reference=torch.ops.aten.sum.default, check_gradients=True,
        expected_shape=(),
    ),
    _case(
        "arithmetic.autograd.sum-dim", "arithmetic autograd",
        torch.ops.aten.sum.dim_IntList, _reduction, args=([1], False),
        cpu_reference=torch.ops.aten.sum.dim_IntList, check_gradients=True,
        expected_shape=(2,),
    ),
    _case(
        "reduction.sum.dim",
        "reduction",
        torch.sum,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_sum,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "reduction.mean.dim",
        "reduction",
        torch.mean,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_mean,
        expected_shape=(2,),
        check_gradients=True,
    ),
    _case(
        "reduction.sum.default",
        "reduction",
        torch.sum,
        _reduction,
        cpu_reference=lambda value: torch.sum(value),
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "reduction.mean.default",
        "reduction",
        torch.mean,
        _reduction,
        cpu_reference=lambda value: torch.mean(value),
        expected_shape=(),
        check_gradients=True,
        # The Vulkan reduction sums through a shared-memory tree, so the result
        # differs from the CPU's sequential order by up to one float32 ULP.
        rtol=1e-6,
        atol=1e-6,
    ),
    _case(
        "reduction.sum.intlist.out",
        "reduction",
        _sum_dims_out,
        _reduction,
        cpu_reference=_cpu_sum_dims_out,
        expected_shape=(),
    ),
    _case(
        "reduction.mean.out",
        "reduction",
        _mean_dims_out,
        _reduction,
        cpu_reference=_cpu_mean_dims_out,
        expected_shape=(),
        rtol=1e-6,
        atol=1e-6,
    ),
    _case(
        "reduction.sum.keepdim.strided",
        "reduction",
        torch.sum,
        _reduction_strided,
        args=(1,),
        kwargs={"keepdim": True},
        cpu_reference=_cpu_sum,
        expected_shape=(3, 1, 4),
        check_gradients=True,
    ),
    _case(
        "reduction.mean.optional-dim",
        "reduction",
        torch.mean,
        _reduction,
        cpu_reference=_cpu_mean,
        expected_shape=(),
        check_gradients=True,
    ),
    _case(
        "reduction.sum.empty-dim",
        "reduction",
        torch.sum,
        _empty_reduction,
        args=(0,),
        cpu_reference=_cpu_sum,
        expected_shape=(3,),
        execution_mode="empty",
    ),
    _case(
        "reduction.mean.empty-dim",
        "reduction",
        torch.mean,
        _empty_reduction,
        args=(0,),
        cpu_reference=_cpu_mean,
        expected_shape=(3,),
        execution_mode="empty",
        rtol=0,
        atol=0,
    ),
    _case(
        "reduction.amax.dim",
        "reduction",
        _amax_out,
        _reduction,
        args=(1,),
        cpu_reference=torch.amax,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amax.default",
        "reduction",
        torch.amax,
        _reduction,
        args=(1,),
        cpu_reference=torch.amax,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amin.dim",
        "reduction",
        _amin_out,
        _reduction,
        args=(1,),
        cpu_reference=torch.amin,
        expected_shape=(2,),
    ),
    _case(
        "reduction.amin.default",
        "reduction",
        torch.amin,
        _reduction,
        args=(1,),
        cpu_reference=torch.amin,
        expected_shape=(2,),
    ),
    _case(
        "reduction.prod.dim",
        "reduction",
        _prod_int_out,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_prod,
        expected_shape=(2,),
    ),
    _case(
        "reduction.prod.default",
        "reduction",
        torch.prod,
        _reduction,
        args=(1,),
        cpu_reference=_cpu_prod,
        expected_shape=(2,),
    ),
    _case(
        "reduction.softmax.dim",
        "reduction",
        _softmax_out,
        _reduction,
        args=(1,),
        cpu_reference=_softmax,
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.softmax.default",
        "reduction",
        _softmax,
        _reduction,
        args=(1,),
        cpu_reference=_softmax,
        expected_shape=(2, 2),
    ),
    _case(
        "reduction.log-softmax.dim",
        "reduction",
        _log_softmax_out,
        _reduction,
        args=(1,),
        cpu_reference=_log_softmax,
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.log-softmax.default",
        "reduction",
        _log_softmax,
        _reduction,
        args=(1,),
        cpu_reference=_log_softmax,
        expected_shape=(2, 2),
    ),
    _case(
        "reduction.softmax.backward",
        "reduction",
        _softmax_backward_out,
        _softmax_backward_inputs,
        cpu_reference=lambda grad, output: torch.ops.aten._softmax_backward_data(
            grad, output, 1, torch.float32
        ),
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "reduction.log-softmax.backward",
        "reduction",
        _log_softmax_backward_out,
        _softmax_backward_inputs,
        cpu_reference=lambda grad, output: torch.ops.aten._log_softmax_backward_data(
            grad,
            output,
            1,
            torch.float32,
        ),
        expected_shape=(2, 2),
        execution_mode="copy",
    ),
    _case(
        "indexing.argmax.dim",
        "indexing",
        torch.argmax,
        _indexing,
        args=(1,),
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(2,),
    ),
    _case(
        "indexing.argmax.optional-dim.keepdim",
        "indexing",
        torch.argmax,
        _indexing,
        kwargs={"keepdim": True},
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(1, 1),
    ),
    _case(
        "indexing.argmax.strided",
        "indexing",
        torch.argmax,
        _indexing_strided,
        args=(-1,),
        cpu_reference=_cpu_argmax,
        expected_dtype=torch.int64,
        expected_shape=(4,),
    ),
    _case(
        "view.reshape.float32",
        "view",
        torch.reshape,
        _view,
        args=((6,),),
        cpu_reference=_cpu_reshape,
        expected_shape=(6,),
        check_gradients=True,
        execution_mode="copy",
    ),
    _case(
        "view.as-strided.metadata",
        "view",
        torch.as_strided,
        _metadata_view,
        args=((2, 3), (3, 1)),
        cpu_reference=_cpu_as_strided,
        expected_shape=(2, 3),
        execution_mode="metadata",
    ),
    _case(
        "view.view.metadata",
        "view",
        torch.ops.aten.view.default,
        _metadata_view,
        args=((6,),),
        cpu_reference=lambda value, shape: torch.ops.aten.view.default(value, shape),
        expected_shape=(6,),
        execution_mode="metadata",
    ),
    _case(
        "view.view.trainable-seed",
        "view",
        torch.ops.aten.view.default,
        _view_seed_input,
        args=((12,),),
        cpu_reference=lambda value, shape: torch.ops.aten.view.default(value, shape),
        expected_shape=(12,),
        check_gradients=True,
        execution_mode="metadata",
        declared_shapes=("3x4",),
    ),
    _case(
        "view.reshape.copy.trainable-seed",
        "view",
        _reshape_transpose_copy,
        _reshape_copy_seed_input,
        args=((6,),),
        cpu_reference=_reshape_transpose_copy,
        expected_shape=(6,),
        check_gradients=True,
        execution_mode="copy",
        declared_shapes=("2x3",),
    ),
    _case(
        "view.reshape.offset-copy.second-order",
        "view",
        _reshape_offset_copy,
        _reshape_offset_seed_input,
        cpu_reference=_reshape_offset_copy,
        expected_shape=(12,),
        check_gradients=True,
        execution_mode="copy",
        declared_shapes=("4x6",),
    ),
    _case(
        "view.reshape-alias.metadata",
        "view",
        torch.ops.aten._reshape_alias.default,
        _metadata_view,
        args=((2, 3), (3, 1)),
        cpu_reference=lambda value, size, stride: torch.ops.aten._reshape_alias.default(
            value, size, stride
        ),
        expected_shape=(2, 3),
        execution_mode="metadata",
    ),
    _case(
        "linear.forward",
        "linear",
        torch.nn.functional.linear,
        _linear,
        cpu_reference=_cpu_linear,
        expected_shape=(2, 3),
        check_gradients=True,
    ),
    _case(
        "linear.forward.strided",
        "linear",
        torch.nn.functional.linear,
        _linear_strided,
        cpu_reference=_cpu_linear,
        expected_shape=(2, 3),
        check_gradients=True,
    ),
    _case(
        "linear.attention.3d",
        "linear",
        torch.nn.functional.linear,
        _linear_attention,
        cpu_reference=_cpu_linear,
        expected_shape=(1, 128, 256),
        check_gradients=True,
    ),
    _case(
        "mm.forward",
        "linear",
        torch.mm,
        _mm,
        cpu_reference=_cpu_mm,
        expected_shape=(2, 3),
    ),
    _case(
        "attention.bmm.forward_backward",
        "linear",
        torch.bmm,
        _bmm,
        cpu_reference=torch.bmm,
        expected_shape=(2, 2, 3),
        check_gradients=True,
    ),
    _case(
        "addmm.forward",
        "linear",
        torch.addmm,
        _addmm,
        cpu_reference=_cpu_addmm,
        expected_shape=(2, 3),
    ),
    _case(
        "addmm.out",
        "linear",
        _addmm_out,
        _addmm,
        cpu_reference=_addmm_out,
        expected_shape=(2, 3),
    ),
    _case(
        "convolution.forward",
        "convolution",
        torch.nn.functional.conv2d,
        _convolution,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 8, 8),
        check_gradients=True,
    ),
    _case(
        "convolution.forward.strided",
        "convolution",
        torch.nn.functional.conv2d,
        _convolution_strided,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 8, 8),
        check_gradients=True,
    ),
    _case(
        "convolution.cnn.forward",
        "convolution",
        torch.nn.functional.conv2d,
        _cnn_convolution,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_cnn_convolution,
        expected_shape=(8, 8, 32, 32),
        check_gradients=True,
    ),
    _case(
        "convolution.general.forward",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_general,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(4, 3, 9, 9),
        declared_shapes=("4x5x9x9",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.stride-2",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_stride_2,
        kwargs={"stride": 2, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 6, 6),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    *(
        _case(
            f"convolution.groups.{label}.forward",
            "convolution",
            torch.nn.functional.conv2d,
            _grouped_convolution_inputs(groups, outputs),
            kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": groups},
            supported=True,
            cpu_reference=_cpu_convolution,
            expected_shape=(2, outputs, 8, 8),
            declared_shapes=("2x4x8x8",),
            rtol=2e-4,
            atol=2e-4,
        )
        for label, groups, outputs in (("2", 2, 6), ("4", 4, 8), ("depthwise", 4, 4))
    ),
    *(
        _case(
            f"convolution.groups.{label}.{gradient}",
            "convolution",
            _convolution_backward_output(index),
            _grouped_convolution_inputs(groups, outputs, backward=True),
            kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": groups},
            supported=True,
            cpu_reference=_convolution_backward_output(index),
            expected_shape=shape,
            declared_shapes=(f"2x{outputs}x8x8",),
            rtol=2e-4,
            atol=2e-4,
        )
        for label, groups, outputs in (("2", 2, 6), ("4", 4, 8), ("depthwise", 4, 4))
        for index, gradient, shape in (
            (0, "grad-input", (2, 4, 8, 8)),
            (1, "grad-weight", (outputs, 4 // groups, 3, 3)),
            (2, "grad-bias", (outputs,)),
        )
    ),
    _case(
        "convolution.groups.3.rejected",
        "convolution",
        torch.nn.functional.conv2d,
        _grouped_convolution_inputs(2, 6),
        kwargs={"padding": 1, "groups": 3},
        supported=False,
        cpu_reference=_cpu_convolution,
        error_pattern=r"groups.*divide",
    ),
    _case(
        "convolution.groups.nondivisible.rejected",
        "convolution",
        torch.nn.functional.conv2d,
        _grouped_nondivisible_inputs,
        kwargs={"padding": 1, "groups": 2},
        supported=False,
        cpu_reference=_cpu_convolution,
        error_pattern=r"groups.*divide",
    ),
    *(
        _case(
            f"convolution.parameters.{label}.rejected",
            "convolution",
            _convolution_parameter_guard,
            _grouped_convolution_inputs(1, 4),
            kwargs=kwargs,
            supported=False,
            cpu_reference=_convolution_parameter_guard,
            error_pattern=r"non-transposed.*zero.*output_padding",
        )
        for label, kwargs in (
            ("transposed", {"transposed": True}),
            ("output-padding", {"output_padding": (1, 0)}),
        )
    ),
    _case(
        "convolution.parameters.padding-0",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_padding_0,
        kwargs={"stride": 1, "padding": 0, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 8, 8),
        declared_shapes=("2x3x10x10",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.dilation-2",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_dilation_2,
        kwargs={"stride": 1, "padding": 2, "dilation": 2, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 12, 12),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.combined",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_combined,
        kwargs={"stride": 2, "padding": (1, 0), "dilation": 2, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 5, 4),
        declared_shapes=("2x3x12x12",),
        check_gradients=True,
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.stride-2.grad-input",
        "convolution",
        _convolution_backward_output(0),
        _convolution_backward_stride2_inputs,
        kwargs={"stride": 2, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=lambda grad, value, weight, **kwargs: _cpu_convolution_backward_output(
            0, grad, value, weight, **kwargs
        ),
        expected_shape=(2, 3, 12, 12),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.stride-2.grad-weight",
        "convolution",
        _convolution_backward_output(1),
        _convolution_backward_stride2_inputs,
        kwargs={"stride": 2, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=lambda grad, value, weight, **kwargs: _cpu_convolution_backward_output(
            1, grad, value, weight, **kwargs
        ),
        expected_shape=(4, 3, 3, 3),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.parameters.dilation-2.grad-bias",
        "convolution",
        _convolution_backward_output(2),
        _convolution_backward_dilation2_inputs,
        kwargs={"stride": 1, "padding": 2, "dilation": 2, "groups": 1},
        cpu_reference=lambda grad, value, weight, **kwargs: grad.sum(dim=(0, 2, 3)),
        expected_shape=(4,),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.general.kernel-1x1",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_kernel_1x1,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 6, 12, 12),
        declared_shapes=("2x4x10x10",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.general.kernel-5x5",
        "convolution",
        torch.nn.functional.conv2d,
        _conv_kernel_5x5,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        cpu_reference=_cpu_convolution,
        expected_shape=(2, 4, 10, 10),
        declared_shapes=("2x3x12x12",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.backward",
        "convolution",
        _convolution_backward,
        _convolution_backward_inputs,
        cpu_reference=lambda grad,
        value,
        weight: torch.ops.aten.convolution_backward.default(
            grad,
            value,
            weight,
            [4],
            [1, 1],
            [1, 1],
            [1, 1],
            False,
            [0, 0],
            1,
            [True, True, True],
        )[0],
        expected_shape=(2, 1, 8, 8),
    ),
    _case(
        "convolution.backward.general.grad-input",
        "convolution",
        _convolution_backward_output(0),
        _convolution_backward_general_inputs,
        cpu_reference=lambda grad, value, weight: _cpu_convolution_backward_output(
            0, grad, value, weight
        ),
        expected_shape=(2, 4, 10, 10),
        declared_shapes=("2x4x10x10",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.backward.kernel-1x1.grad-weight",
        "convolution",
        _convolution_backward_output(1),
        _convolution_backward_general_inputs,
        cpu_reference=lambda grad, value, weight: _cpu_convolution_backward_output(
            1, grad, value, weight
        ),
        expected_shape=(6, 4, 1, 1),
        declared_shapes=("2x4x10x10",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "convolution.backward.general.grad-bias",
        "convolution",
        _convolution_backward_output(2),
        _convolution_backward_general_inputs,
        cpu_reference=lambda grad, value, weight: _cpu_convolution_backward_output(
            2, grad, value, weight
        ),
        expected_shape=(6,),
        declared_shapes=("2x4x10x10",),
        rtol=2e-4,
        atol=2e-4,
    ),
    _case(
        "pooling.max.rejected",
        "pooling",
        torch.nn.functional.max_pool2d,
        _pooling,
        args=(2,),
        cpu_reference=_cpu_max_pool,
        supported=False,
        error_pattern=r"Vulkan pooling is not declared",
    ),
    _case(
        "masked-select.bool-mask",
        "masked-select",
        torch.masked_select,
        _masked_select,
        cpu_reference=_cpu_masked_select,
        expected_shape=(2,),
        autograd_supported=False,
        autograd_error_type=NotImplementedError,
        autograd_error_pattern=r"aten::(zero_|masked_scatter_)",
        requires_grad_inputs=True,
    ),
    _case(
        "masked-select.strided-value-view",
        "masked-select",
        torch.masked_select,
        _masked_select_view,
        cpu_reference=_cpu_masked_select,
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "optimizer.div.scalar",
        "optimizer",
        torch.ops.aten.div.Tensor,
        _optimizer_pair,
        cpu_reference=torch.div,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.lerp.out",
        "optimizer",
        _lerp_out,
        _optimizer_value_pair,
        cpu_reference=_cpu_lerp,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.lerp.inplace",
        "optimizer",
        _lerp_inplace,
        _optimizer_value_pair,
        cpu_reference=_cpu_lerp,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.sqrt.out",
        "optimizer",
        _sqrt_out,
        _optimizer_single,
        cpu_reference=torch.sqrt,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.add.inplace",
        "optimizer",
        _add_inplace,
        _optimizer_value_pair,
        cpu_reference=torch.add,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.mul.scalar.inplace",
        "optimizer",
        _mul_scalar_inplace,
        _optimizer_single,
        cpu_reference=lambda value: value * 2,
        expected_shape=(2,),
    ),
    _case(
        "optimizer.addcmul.inplace",
        "optimizer",
        _addcmul_inplace,
        _optimizer_triple,
        cpu_reference=lambda lhs, a, b: torch.addcmul(lhs, a, b, value=0.25),
        expected_shape=(2,),
    ),
    _case(
        "optimizer.addcdiv.inplace",
        "optimizer",
        _addcdiv_inplace,
        _optimizer_triple,
        cpu_reference=lambda lhs, a, b: torch.addcdiv(lhs, a, b, value=0.25),
        expected_shape=(2,),
    ),
    _case(
        "optimizer.zero.inplace",
        "optimizer",
        _zero_inplace,
        _optimizer_single,
        cpu_reference=lambda value: value.zero_(),
        expected_shape=(2,),
    ),
    _case(
        "aten._adaptive_avg_pool2d.global",
        "pooling",
        torch.ops.aten._adaptive_avg_pool2d.default,
        _adaptive_pool,
        args=((1, 1),),
        cpu_reference=_cpu_adaptive_pool,
        expected_shape=(2, 4, 1, 1),
        check_gradients=True,
    ),
    _case(
        "aten._adaptive_avg_pool2d.global.strided",
        "pooling",
        torch.ops.aten._adaptive_avg_pool2d.default,
        _pooling_strided,
        args=((1, 1),),
        cpu_reference=_cpu_adaptive_pool,
        expected_shape=(1, 1, 1, 1),
        check_gradients=True,
    ),
    _case(
        "aten._adaptive_avg_pool2d.backward",
        "pooling",
        _adaptive_pool_backward,
        _adaptive_pool_backward_inputs,
        cpu_reference=lambda grad,
        value: torch.ops.aten._adaptive_avg_pool2d_backward.default(grad, value),
        expected_shape=(2, 4, 3, 5),
    ),
    _case(
        "normalization.native-batch-norm",
        "normalization",
        _native_batch_norm_output,
        _normalization,
        cpu_reference=_native_batch_norm_output,
        expected_shape=(2, 4),
        # Batch norm's per-channel statistics are float32 reductions; Vulkan
        # and CPU summation order differ, so these outputs cannot meet a
        # near-exact tolerance. The measured absolute error is below 1e-6,
        # while relative error reached 3.7e-5, so both bounds need loosening.
        rtol=1e-4,
        atol=1e-6,
    ),
    _case(
        "normalization.native-batch-norm.2d",
        "normalization",
        _native_batch_norm,
        _batch_norm_2d,
        cpu_reference=_cpu_native_batch_norm_with_parameters,
        expected_shape=(7, 3),
        declared_shapes=("7x3",),
        # Batch norm's statistics are float32 reductions whose result depends
        # on summation order; 40 sync runs exceeded the default relative bound.
        rtol=1e-4,
        atol=1e-6,
    ),
    _case(
        "normalization.native-batch-norm.4d",
        "normalization",
        _native_batch_norm,
        _batch_norm_4d,
        cpu_reference=_cpu_native_batch_norm_with_parameters,
        expected_shape=(3, 4, 5, 7),
        declared_shapes=("3x4x5x7",),
        # Batch norm's statistics are float32 reductions whose result depends
        # on summation order; 40 sync runs exceeded the default tolerance.
        rtol=1e-4,
        atol=1e-6,
    ),
    _case(
        "normalization.native-batch-norm.4d-single-channel",
        "normalization",
        _native_batch_norm,
        _batch_norm_4d_single_channel,
        cpu_reference=_cpu_native_batch_norm_with_parameters,
        expected_shape=(5, 1, 3, 3),
        declared_shapes=("5x1x3x3",),
        # Batch norm's statistics are float32 reductions whose result depends
        # on summation order; a sync run exceeded the default relative bound.
        rtol=1e-4,
        atol=1e-6,
    ),
    _case(
        "normalization.native-batch-norm-backward",
        "normalization",
        _native_batch_norm_backward_output,
        _normalization_backward,
        cpu_reference=_native_batch_norm_backward_output,
        expected_shape=(2, 4),
    ),
    _case(
        "normalization.native-batch-norm-backward.4d",
        "normalization",
        _native_batch_norm_backward_output,
        _normalization_backward_4d,
        cpu_reference=_native_batch_norm_backward_output,
        expected_shape=(3, 4, 5, 7),
        declared_shapes=("3x4x5x7",),
    ),
    _case(
        "classification.nll-forward",
        "cross-entropy/NLL",
        _nll_forward_output,
        _nll_forward,
        cpu_reference=_nll_forward_output,
        expected_shape=(),
        declared_shapes=("2x3",),
    ),
    _case(
        "classification.nll-backward",
        "cross-entropy/NLL",
        _nll_backward_output,
        _nll_backward,
        cpu_reference=_nll_backward_output,
        expected_shape=(2, 3),
        declared_shapes=("2x3",),
    ),
    _case(
        "classification.nll-forward.wide",
        "classification",
        _nll_forward_output,
        _nll_forward_wide,
        cpu_reference=lambda logits, labels: torch.nn.functional.nll_loss(
            torch.log_softmax(logits, dim=1), labels, reduction="mean"
        ),
        expected_shape=(),
        declared_shapes=("8x5",),
    ),
    _case(
        "classification.nll-backward.wide",
        "classification",
        _nll_backward_output,
        _nll_backward_wide,
        cpu_reference=_cpu_nll_backward_wide,
        expected_shape=(8, 5),
        declared_shapes=("8x5",),
    ),
    _case(
        "classification.nll-forward.none.wide",
        "classification",
        lambda logits, labels: torch.ops.aten.nll_loss_forward.default(
            torch.log_softmax(logits, dim=1), labels, None, 0, -100
        )[0],
        _nll_forward_none_large,
        cpu_reference=lambda logits, labels: torch.nn.functional.nll_loss(
            torch.log_softmax(logits, dim=1), labels, reduction="none"
        ),
        expected_shape=(512,),
        declared_shapes=("512x5",),
    ),
    _case(
        "classification.nll-backward.sum",
        "classification",
        _nll_backward_reduction_output("sum"),
        _nll_backward_reduction_inputs("sum"),
        cpu_reference=lambda grad, log_probs, labels, total: _cpu_nll_backward_reduction(
            grad, log_probs, labels, "sum"
        ),
        expected_shape=(8, 5),
        declared_shapes=("8x5",),
    ),
    _case(
        "classification.nll-backward.none",
        "classification",
        _nll_backward_reduction_output("none"),
        _nll_backward_reduction_inputs("none"),
        cpu_reference=lambda grad, log_probs, labels, total: _cpu_nll_backward_reduction(
            grad, log_probs, labels, "none"
        ),
        expected_shape=(8, 5),
        declared_shapes=("8x5",),
    ),
    _case(
        "unary.neg.bool.rejected",
        "unary",
        torch.neg,
        _bool_unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"supports only float32 tensors",
    ),
    _case(
        "unary.neg.float16.rejected",
        "unary",
        torch.neg,
        _float16_unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"float16 support is deferred",
    ),
    _case(
        "binary.add.double.rejected",
        "binary",
        torch.add,
        _double_binary,
        supported=False,
        convert_inputs=True,
        cpu_reference=_cpu_add,
        error_pattern=r"formatter conversion supports only Vulkan float32 to Vulkan Double",
    ),
    _case(
        "binary.add.mixed-device.rejected",
        "binary",
        _mixed_device_add,
        _unary,
        supported=False,
        cpu_reference=_cpu_add,
        error_pattern=r"requires operands on the same device; got",
    ),
    _case(
        "binary.add.broadcast.rejected",
        "binary",
        torch.add,
        _broadcast_binary,
        supported=False,
        cpu_reference=_cpu_add,
        error_pattern=r"requires equal tensor sizes; broadcasting is unsupported",
    ),
    _case(
        "reduction.sum.expanded-readable-input",
        "reduction",
        torch.sum,
        _expanded_read,
        args=(1,),
        supported=True,
        cpu_reference=_cpu_sum,
    ),
    _case(
        "reduction.sum.mixed-expanded-overlap.rejected",
        "reduction",
        torch.sum,
        _mixed_expanded_overlap,
        args=(1,),
        supported=False,
        cpu_reference=_cpu_sum,
        error_pattern=r"rejects overlapping input views",
    ),
    _case(
        "comparison.ne.nonzero-offset-input.rejected",
        "comparison",
        torch.ne,
        _wrong_offset,
        supported=False,
        cpu_reference=torch.ne,
        error_pattern=r"formatter Double ne\.Tensor requires a contiguous zero-offset strided input",
        setup_inputs=_setup_double_offset,
    ),
    _case(
        "unary.neg.invalid-out.rejected",
        "unary",
        _invalid_out,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"supports only float32 and bool tensors, got Long",
    ),
    _case(
        "unary.neg.cpu-out.rejected",
        "unary",
        _cpu_out,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"requires a Vulkan output tensor",
    ),
    _case(
        "pooling.parameters.rejected",
        "pooling",
        _bad_pool_params,
        _adaptive_pool,
        args=(),
        supported=False,
        cpu_reference=_cpu_adaptive_pool,
        error_pattern=r"output size|\(1, 1\)",
    ),
    _case(
        "convolution.channels.shape.rejected",
        "convolution",
        torch.nn.functional.conv2d,
        _channel_mismatch_convolution,
        kwargs={"stride": 1, "padding": 1, "dilation": 1, "groups": 1},
        supported=False,
        cpu_reference=_cpu_convolution,
        error_pattern=r"channel|size|shape",
    ),
    _case(
        "convolution.backward.grad-output-spatial-shape.rejected",
        "convolution",
        _convolution_backward,
        _convolution_backward_mismatched_grad_inputs,
        supported=False,
        cpu_reference=_convolution_backward,
        error_pattern=r"Vulkan convolution backward grad_output shape expected .* actual",
    ),
    _case(
        "convolution.backward.grad-output-batch.rejected",
        "convolution",
        _convolution_backward,
        _convolution_backward_mismatched_batch_inputs,
        supported=False,
        cpu_reference=_convolution_backward,
        error_pattern=r"Vulkan convolution backward grad_output shape expected .* actual",
    ),
    _case(
        "unary.neg_.unsupported-overload.rejected",
        "unary",
        _unsupported_overload,
        _unary,
        supported=False,
        cpu_reference=_cpu_neg,
        error_pattern=r"Vulkan neg in-place variants are unsupported",
    ),
    _case(
        "transfer.copy_from.float32",
        "transfer",
        _copy_from,
        _transfer_pair,
        cpu_reference=_copy_from_cpu_reference,
        expected_shape=(4,),
        execution_mode="copy",
    ),
    _case(
        "transfer.to_copy.float32",
        "transfer",
        lambda value: value.to(value.device, copy=True),
        _unary,
        cpu_reference=lambda value: value.to(value.device, copy=True),
        expected_shape=(3,),
        execution_mode="copy",
    ),
    _case(
        "transfer.copy.float32",
        "transfer",
        _copy_in_place,
        _transfer_pair,
        cpu_reference=_copy_in_place,
        expected_shape=(4,),
        execution_mode="copy",
    ),
    _case(
        "transfer.empty.float32",
        "transfer",
        _empty_like_template,
        _empty_unary,
        cpu_reference=_empty_like_template,
        expected_shape=(0, 3),
        execution_mode="empty",
    ),
    _case(
        "transfer.empty_strided.float32",
        "transfer",
        _empty_strided_like_template,
        _empty_strided_zero_template,
        cpu_reference=_empty_strided_like_template,
        expected_shape=(0, 2),
        execution_mode="empty",
    ),
    _case(
        "unary.abs-out", "unary", _abs_out, _abs_out_inputs,
        cpu_reference=_cpu_abs_out, supported=True,
        expected_dtype=torch.float32, expected_shape=(3, 4),
    ),
    _case("unary.neg-out", "unary", _neg_out, _out_inputs,
          cpu_reference=lambda value, out: _cpu_out_via_op(torch.ops.aten.neg.out, value, out), supported=True,
          expected_dtype=torch.float32, expected_shape=(3, 4)),
    _case("unary.relu-out", "unary", _relu_out, _out_inputs,
          cpu_reference=lambda value, out: _cpu_out_via_op(torch.ops.aten.relu.out, value, out), supported=True,
          expected_dtype=torch.float32, expected_shape=(3, 4)),
    _case("comparison.ne-tensor", "comparison", torch.ne,
          lambda **kwargs: (torch.randn(3, 4), torch.randn(3, 4)), supported=True,
          expected_dtype=torch.bool, expected_shape=(3, 4), cpu_reference=torch.ne),
    _case("comparison.isfinite", "comparison", torch.ops.aten.isfinite.default,
          lambda **kwargs: (torch.tensor([[0.0, float("inf")], [float("-inf"), float("nan")]]),),
          supported=True, expected_dtype=torch.bool,
          cpu_reference=lambda value: torch.isfinite(value.clone()),
          declared_shapes=("2x2",)),
    _case("comparison.eq-tensor-out", "comparison", _eq_tensor_out,
          _bool_out_inputs, cpu_reference=lambda a, b, out: torch.ops.aten.eq.Tensor_out(a, b, out=out.clone()),
          supported=True, expected_dtype=torch.bool, expected_shape=(3, 4)),
    _case("comparison.ne-scalar-out", "comparison", _ne_scalar_out,
          _scalar_bool_out_inputs, cpu_reference=lambda a, out: torch.ops.aten.ne.Scalar_out(a, 0.0, out=out.clone()),
          supported=True, expected_dtype=torch.bool, expected_shape=(3, 4)),
    _case("comparison.argmax-out", "comparison", _argmax_out, _argmax_inputs,
          args=(1, False), cpu_reference=lambda value, out, dim, keepdim=False: _cpu_argmax_out(value, dim, out),
          supported=True, expected_dtype=torch.int64, expected_shape=(3,)),
    _case("comparison.bitwise-and-out", "comparison", _bitwise_and_out,
          _bitwise_bool_out_inputs, cpu_reference=lambda a, b, out: torch.ops.aten.bitwise_and.Tensor_out(a, b, out=out.clone()),
          supported=True, expected_dtype=torch.bool, expected_shape=(3, 4)),
    _case("inplace.add-scalar", "inplace", torch.ops.aten.add_.Scalar,
          _inplace_inputs, args=(2.0,), cpu_reference=lambda value, scalar: _cpu_inplace(torch.Tensor.add_, value, scalar), supported=True, expected_shape=(3, 4)),
    _case("inplace.sub-scalar", "inplace", torch.ops.aten.sub_.Scalar,
          _inplace_inputs, args=(2.0,), cpu_reference=lambda value, scalar: _cpu_inplace(torch.Tensor.sub_, value, scalar), supported=True, expected_shape=(3, 4)),
    _case("inplace.mul-tensor", "inplace", torch.ops.aten.mul_.Tensor,
          _inplace_binary_inputs, cpu_reference=lambda value, other: _cpu_inplace(torch.Tensor.mul_, value, other), supported=True, expected_shape=(3, 4)),
    _case("inplace.sub-tensor", "inplace", torch.ops.aten.sub_.Tensor,
          _inplace_binary_inputs, cpu_reference=lambda value, other: _cpu_inplace(torch.Tensor.sub_, value, other), supported=True, expected_shape=(3, 4)),
    _case("inplace.fill-scalar", "inplace", torch.ops.aten.fill_.Scalar,
          _inplace_inputs, args=(0.0,), cpu_reference=lambda value, scalar: _cpu_inplace(torch.Tensor.fill_, value, scalar), supported=True, expected_shape=(3, 4)),
    )
    if case is not None
) + _convolution_evidence_cases()


SUPPORTED_CASES = tuple(case for case in ALL_CASES if case.supported)
