"""Small, executable operator cases shared by Vulkan conformance tests."""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

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
    convolution_direction: Literal["forward", "backward"] | None = None
    convolution_numerical_operations: tuple[int, ...] = ()
    convolution_result_slots: tuple[int, ...] = ()
    convolution_context_setup: Callable[[tuple[Any, ...], str], tuple[tuple[Any, ...], dict[str, Any]]] | None = None
    graph_autograd_case: str | None = None

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
    convolution_context: dict[str, Any] | None = None,
    graph_autograd: dict[str, object] | None = None,
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
    if (case.name.startswith("convolution.forward.bias-")
            or case.name.startswith("convolution.backward.bias-")
            or case.convolution_direction is not None):
        execution = _CASE_EXECUTION.get(case.name)
        if execution is None:
            raise ValueError(f"{case.name}: actual execution evidence was not captured")
        record.update(_convolution_coverage(
            case, inputs, result, execution, convolution_context
        ))
    if graph_autograd is not None:
        record["graph_autograd"] = graph_autograd
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


def _convolution_tensor_metadata(value: torch.Tensor | None, context_device: str) -> dict[str, Any]:
    if value is None:
        return {
            "defined": False, "dtype": None, "rank": None, "shape": None,
            "strides": None, "storage_offset": None, "device": context_device,
        }
    return {
        "defined": True,
        "dtype": str(value.dtype).removeprefix("torch."),
        "rank": value.dim(),
        "shape": list(value.shape),
        "strides": list(value.stride()),
        "storage_offset": value.storage_offset(),
        "device": str(value.device),
    }


def _assert_unchanged_schema_inputs(actual, expected, case_name):
    if (not isinstance(actual, tuple) or len(actual) != len(expected)
            or any(value is not original for value, original in zip(actual, expected))):
        raise ValueError(f"{case_name}: context setup must return unchanged schema inputs")


def _convolution_coverage(case, inputs, result, execution, convolution_context=None):
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
    transposed = case.convolution_direction is not None
    forward = case.convolution_direction == "forward" if transposed else case.name.startswith("convolution.forward.")
    bias = roles["bias"]["defined"] if not transposed else bool(
        convolution_context and convolution_context.get("vulkan", {}).get("defined")
    )
    context = {
        "device": (
            str(result.device) if transposed and isinstance(result, torch.Tensor)
            else str(inputs[0].device) if transposed else "vk:0"
        ),
        "cpu_oracle": (
            "torch.nn.functional.conv_transpose2d" if transposed and forward
            else "torch.nn.functional.conv2d" if forward
            else "aten::convolution_backward.default"
        ),
        "forward_bias_present": bias,
    }
    if transposed:
        if not isinstance(convolution_context, dict) or set(convolution_context) != {"cpu", "vulkan"}:
            raise ValueError(f"{case.name}: actual warm forward bias context is required")
        context.update({
            "direction": case.convolution_direction,
            "warm_forward_bias": convolution_context,
            "expected_numerical_operations": list(case.convolution_numerical_operations),
            "expected_result_slots": list(case.convolution_result_slots),
            "mapping_id": "transposed-convolution-stage-e-v1",
        })
    if forward:
        arg_names = (("stride", "padding", "dilation", "transposed", "output_padding", "groups")
                     if transposed else ("stride", "padding", "dilation", "groups"))
        schema_args = dict(zip(arg_names, case.args))
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
    if transposed and isinstance(result, torch.Tensor):
        record = {"convolution_context": context, "convolution_operands": roles,
                  "schema_args": schema_args, "execution": dict(execution),
                  "output_dtype": str(result.dtype).removeprefix("torch."),
                  "output_rank": result.dim(), "output_shape": list(result.shape)}
        return record
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
    case: ConformanceCase, device: str = "vk:0", *, return_inputs: bool = False,
    convolution_context_out: dict[str, Any] | None = None,
    graph_autograd_context_out: dict[str, object] | None = None,
) -> tuple[Any, ...]:
    """Compute the CPU reference and Vulkan result with counters scoped to execution."""
    if convolution_context_out is not None:
        convolution_context_out.clear()
    if graph_autograd_context_out is not None:
        graph_autograd_context_out.clear()
    if case.graph_autograd_case is not None:
        vulkan_output, cpu_output, actual_inputs, graph_payload = run_graph_autograd_case(case, device)
        if graph_autograd_context_out is not None:
            graph_autograd_context_out["graph_autograd"] = graph_payload
        return (vulkan_output, cpu_output, actual_inputs) if return_inputs else (vulkan_output, cpu_output)
    transposed = case.convolution_direction is not None
    if transposed:
        _CASE_EXECUTION.pop(case.name, None)
    if transposed and (convolution_context_out is None or case.convolution_context_setup is None):
        raise ValueError(f"{case.name}: transposed convolution requires fresh context capture")
    cpu_inputs = case.inputs()
    reference_inputs = tuple(
        value.clone() if isinstance(value, torch.Tensor) else value
        for value in cpu_inputs
    )
    if case.setup_inputs is not None:
        reference_inputs = case.setup_inputs(reference_inputs, "cpu")
    cpu_context = None
    if transposed:
        original_reference_inputs = reference_inputs
        reference_inputs, cpu_context = case.convolution_context_setup(reference_inputs, "cpu")
        _assert_unchanged_schema_inputs(reference_inputs, original_reference_inputs, case.name)
    if (transposed and case.convolution_direction == "backward") or case.name.startswith("convolution.backward.bias-"):
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
    vulkan_context = None
    if transposed:
        original_inputs = inputs
        inputs, vulkan_context = case.convolution_context_setup(inputs, device)
        _assert_unchanged_schema_inputs(inputs, original_inputs, case.name)
        if (not isinstance(cpu_context, dict) or set(cpu_context) != {"warm_forward_bias"}
                or not isinstance(vulkan_context, dict) or set(vulkan_context) != {"warm_forward_bias"}):
            raise ValueError(f"{case.name}: context setup did not capture actual warm bias")
        cpu_bias = cpu_context["warm_forward_bias"]
        vulkan_bias = vulkan_context["warm_forward_bias"]
        if (not isinstance(cpu_bias, dict) or not isinstance(vulkan_bias, dict)
                or {k: v for k, v in cpu_bias.items() if k != "device"}
                != {k: v for k, v in vulkan_bias.items() if k != "device"}):
            raise ValueError(f"{case.name}: CPU/Vulkan warm bias metadata differs")
    if case.name.startswith("convolution.") or transposed:
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
    if (case.name.startswith("convolution.forward.bias-")
            or case.name.startswith("convolution.backward.bias-") or transposed):
        execution = {
            "compute_dispatches": dispatches,
            "vulkan_copies": vulkan_copies,
            "explicit_transfers": explicit_transfers,
            "fallbacks": fallbacks,
            "buffer_creations_delta": after_timing["buffer_creations"] - before_timing["buffer_creations"],
            "live_allocations_delta": after_live - before_live,
        }
        if not transposed:
            _CASE_EXECUTION[case.name] = execution
    if case.name in _STOCK_ROUTE_CASES:
        if case.execution_mode != "copy" or dispatches != 0 or vulkan_copies <= 0:
            raise AssertionError(f"{case.name}: stock reshape copy route did not execute Vulkan copy work")
        _CASE_EXECUTION[case.name] = execution
    if case.execution_mode == "compute":
        if case.convolution_direction == "backward" or case.name.startswith("convolution.backward.bias-"):
            assert dispatches == sum(case.args[-1])
            if transposed and not any(case.args[-1]):
                assert execution["buffer_creations_delta"] == 0
                assert execution["live_allocations_delta"] == 0
        elif transposed:
            assert dispatches == 1
        else:
            assert dispatches > 0
    elif case.execution_mode == "copy":
        assert vulkan_copies > 0
    else:
        assert dispatches == 0 and vulkan_copies == 0
    assert explicit_transfers == 0
    assert fallbacks == 0
    if transposed:
        assert vulkan_copies == 0, (
            f"{case.name}: expected vulkan_copies 0, got {vulkan_copies}"
        )
        convolution_context_out.update({
            "cpu": cpu_context["warm_forward_bias"],
            "vulkan": vulkan_context["warm_forward_bias"],
        })
        _CASE_EXECUTION[case.name] = execution
    return (result, cpu_result, inputs) if return_inputs else (result, cpu_result)


def run_convolution_evidence_cases(cases=None) -> dict[str, dict[str, object]]:
    """Execute and record only the named bias/mask witnesses after parity."""
    selected = tuple(cases) if cases is not None else tuple(
        case for case in ALL_CASES
        if case.name.startswith("convolution.forward.bias-")
        or case.name.startswith("convolution.backward.bias-")
        or case.convolution_direction is not None
    )
    with coverage_recording():
        for case in selected:
            context = {} if case.convolution_direction is not None else None
            result, expected, inputs = run_and_compare(
                case, return_inputs=True, convolution_context_out=context
            )
            assert_result_parity(result, expected, case)
            mark_executed(case.name)
            record_coverage(case, inputs, result, gradients=False, parity=True,
                            convolution_context=context)
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


def _ordinary_output_padding_inputs(*, requires_grad=False):
    generator = torch.Generator(device="cpu").manual_seed(4601)
    return (torch.randn((1, 2, 4, 5), generator=generator),
            torch.randn((3, 2, 2, 3), generator=generator), None)


def _ordinary_output_padding_forward(value, weight, bias, stride, padding, dilation,
                                     transposed, output_padding, groups):
    return torch.ops.aten.convolution.default(
        value, weight, bias, stride, padding, dilation, transposed,
        list(output_padding), groups,
    )


def _ordinary_output_padding_reference(value, weight, bias, stride, padding, dilation,
                                       transposed, output_padding, groups):
    assert transposed is False
    return torch.nn.functional.conv2d(
        value, weight, bias, stride, padding, dilation, groups,
    )


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


GRAPH_AUTOGRAD_CASES = {
    "matrix.graph.mm.generated": {
        "schema": "aten::mm.default", "forward_schema": "aten::mm.default",
        "generated_backward_schema": "aten::mm.default",
        "native_double_backward_schema": None,
        "directions": ["first_reverse", "second_reverse_mixed"],
        "derivative_contract": {"first_reverse": {"targets": ["input", "weight"], "requires_grad": True},
                               "second_reverse_mixed": {"targets": ["input", "weight"], "requires_grad": False}},
        "graph_levels": [1, 2],
        "geometry": {"shapes": [[2, 3], [3, 4]], "layouts": ["contiguous", "contiguous"]},
    },
    "matrix.graph.addmm.generated": {
        "schema": "aten::addmm.default", "forward_schema": "aten::addmm.default",
        "generated_backward_schema": "aten::mm.default",
        "native_double_backward_schema": None,
        "directions": ["first_reverse", "second_reverse_mixed"],
        "derivative_contract": {"first_reverse": {"targets": ["bias", "input", "weight"], "requires_grad": True},
                               "second_reverse_mixed": {"targets": ["bias", "input", "weight"], "requires_grad": False}},
        "graph_levels": [1, 2],
        "geometry": {"shapes": [[4], [2, 3], [3, 4]], "layouts": ["contiguous"] * 3,
                     "alpha": 1.75, "beta": -0.5, "self_shape": [4]},
    },
    "matrix.graph.bmm.generated": {
        "schema": "aten::bmm.default", "forward_schema": "aten::bmm.default",
        "generated_backward_schema": "aten::bmm.default",
        "native_double_backward_schema": None,
        "directions": ["first_reverse", "second_reverse_mixed"],
        "derivative_contract": {"first_reverse": {"targets": ["input", "weight"], "requires_grad": True},
                               "second_reverse_mixed": {"targets": ["input", "weight"], "requires_grad": False}},
        "graph_levels": [1, 2],
        "geometry": {"shapes": [[2, 3, 4], [2, 4, 5]], "layouts": ["contiguous", "contiguous"]},
    },
    "convolution.graph.grouped.ggI-ggW-ggb": {
        "schema": "aten::convolution.default", "forward_schema": "aten::convolution.default",
        "generated_backward_schema": "aten::convolution_backward.default",
        "native_double_backward_schema": "aten::_convolution_double_backward.default",
        "directions": ["first_reverse", "second_reverse_ggI", "second_reverse_ggW", "second_reverse_ggb", "second_reverse_combined"],
        "graph_levels": [1, 2], "geometry": {"stride": [2, 1], "padding": [1, 0], "dilation": [1, 2], "transposed": False, "output_padding": [0, 0], "groups": 2},
    },
    "convolution.graph.grouped.selected-third": {
        "schema": "aten::convolution.default", "forward_schema": "aten::convolution.default",
        "generated_backward_schema": "aten::convolution_backward.default",
        "native_double_backward_schema": "aten::_convolution_double_backward.default",
        "directions": ["first_reverse", "second_reverse_ggW", "selected_third_d_g_d_x_d_w"],
        "graph_levels": [1, 2, 3], "geometry": {"stride": [2, 1], "padding": [1, 0], "dilation": [1, 2], "transposed": False, "output_padding": [0, 0], "groups": 2},
    },
    "convolution.graph.depthwise.selected-third": {
        "schema": "aten::convolution.default", "forward_schema": "aten::convolution.default",
        "generated_backward_schema": "aten::convolution_backward.default",
        "native_double_backward_schema": "aten::_convolution_double_backward.default",
        "directions": ["first_reverse", "second_reverse_ggW", "selected_third_d_g_d_x_d_w"],
        "graph_levels": [1, 2, 3], "geometry": {"stride": [1, 2], "padding": [0, 1], "dilation": [1, 1], "transposed": False, "output_padding": [0, 0], "groups": 2},
    },
    "convolution.graph.transposed.second": {
        "schema": "aten::convolution.default", "forward_schema": "aten::convolution.default",
        "generated_backward_schema": "aten::convolution_backward.default",
        "native_double_backward_schema": "aten::_convolution_double_backward.default",
        "directions": ["first_reverse", "second_reverse_ggI", "second_reverse_ggW", "second_reverse_ggb", "second_reverse_combined"],
        "graph_levels": [1, 2], "geometry": {"stride": [2, 2], "padding": [0, 0], "dilation": [1, 1], "transposed": True, "output_padding": [1, 0], "groups": 1},
    },
    "convolution.graph.overrideable.first": {
        "schema": "aten::convolution_overrideable.default", "forward_schema": "aten::convolution_overrideable.default",
        "generated_backward_schema": "aten::convolution_backward_overrideable.default",
        "native_double_backward_schema": None, "directions": ["first_reverse"],
        "graph_levels": [1], "geometry": {"stride": [1, 1], "padding": [0, 0], "dilation": [1, 1], "transposed": False, "output_padding": [0, 0], "groups": 1},
    },
}
GRAPH_AUTOGRAD_REQUIRED_CASES = frozenset(GRAPH_AUTOGRAD_CASES)
MATRIX_GRAPH_MODE_OBSERVATION_CASES = frozenset({
    "matrix.graph.mm.generated",
    "matrix.graph.bmm.generated",
})
MATRIX_GRAPH_MODE_EXECUTION_CONTRACT = {
    "async": {
        "first_backward": {
            "compute_dispatches": 4, "vulkan_copies": 0,
            "explicit_transfers": 0, "fallbacks": 0,
            "buffer_creations_delta": 6, "live_allocations_delta": 2,
        },
        "higher_order": {
            "compute_dispatches": 33, "vulkan_copies": 0,
            "explicit_transfers": 0, "fallbacks": 0,
            "buffer_creations_delta": 55, "live_allocations_delta": 5,
        },
    },
    "sync": {
        "first_backward": {
            "compute_dispatches": 4, "vulkan_copies": 0,
            "explicit_transfers": 0, "fallbacks": 0,
            "buffer_creations_delta": 6, "live_allocations_delta": 2,
        },
        "higher_order": {
            "compute_dispatches": 33, "vulkan_copies": 0,
            "explicit_transfers": 0, "fallbacks": 0,
            "buffer_creations_delta": 54, "live_allocations_delta": 5,
        },
    },
}


def record_graph_autograd_coverage(case_name, route, geometry, operands, directions,
                                   graph_levels, derivative_results,
                                   route_observations, execution, numerical_errors=None):
    """Build the bounded, source-owned graph witness envelope."""
    contract = GRAPH_AUTOGRAD_CASES[case_name]
    if route != {key: contract[key] for key in (
            "forward_schema", "generated_backward_schema", "native_double_backward_schema")}:
        raise ValueError(f"{case_name}: graph route is not source-owned")
    if geometry != contract["geometry"] or directions != contract["directions"] or graph_levels != contract["graph_levels"]:
        raise ValueError(f"{case_name}: graph recipe is not source-owned")
    payload = {"route": route, "geometry": geometry, "operands": operands,
            "directions": directions, "graph_levels": graph_levels,
            "derivative_results": derivative_results,
            "route_observations": route_observations, "execution": execution}
    if numerical_errors is not None:
        payload["numerical_errors"] = numerical_errors
    return payload


def _graph_tensor_meta(value):
    if value is None:
        return {"defined": False, "dtype": None, "rank": None, "shape": None,
                "strides": None, "storage_offset": None, "device": None,
                "requires_grad": False}
    return {"defined": True, "dtype": str(value.dtype).removeprefix("torch."),
            "rank": value.dim(), "shape": list(value.shape),
            "strides": list(value.stride()), "storage_offset": value.storage_offset(),
            "device": str(value.device), "requires_grad": bool(value.requires_grad)}


def _graph_grad_fn(value):
    if value is None or value.grad_fn is None:
        return None
    return type(value.grad_fn).__name__


def _graph_nonzero(value):
    return bool(torch.count_nonzero(value.detach()).item()) if value is not None else False


def _graph_snap():
    pytorch_vulkan._C.synchronize()
    timing = pytorch_vulkan._C.timing_breakdown()
    live = pytorch_vulkan._C.live_resource_snapshot()[6]
    counters = pytorch_vulkan._C.execution_counter_snapshot()
    return timing["buffer_creations"], live, counters


def _graph_cpu_readback(value):
    """Synchronize each measured Vulkan result before its single CPU readback."""
    pytorch_vulkan._C.synchronize()
    return value.detach().cpu()


def _graph_delta(before, after):
    return {"compute_dispatches": after[2][0] - before[2][0],
            "vulkan_copies": after[2][1] - before[2][1],
            "explicit_transfers": after[2][2] - before[2][2],
            "fallbacks": after[2][3] - before[2][3],
            "buffer_creations_delta": after[0] - before[0],
            "live_allocations_delta": after[1] - before[1]}


def _run_matrix_graph_autograd_case(case: ConformanceCase, device: str):
    name = case.graph_autograd_case
    contract = GRAPH_AUTOGRAD_CASES[name]
    seed = {"matrix.graph.mm.generated": 4301,
            "matrix.graph.addmm.generated": 4302,
            "matrix.graph.bmm.generated": 4303}[name]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if name == "matrix.graph.mm.generated":
        shapes, roles = ((2, 3), (3, 4)), ("input", "weight")
        operation = torch.mm
    elif name == "matrix.graph.addmm.generated":
        shapes, roles = ((4,), (2, 3), (3, 4)), ("bias", "input", "weight")
        operation = lambda bias, a, b: torch.addmm(bias, a, b, alpha=1.75, beta=-0.5)
    else:
        shapes, roles = ((2, 3, 4), (2, 4, 5)), ("input", "weight")
        operation = torch.bmm
    cpu_inputs = tuple(torch.randn(shape, generator=generator, dtype=torch.float32,
                                   requires_grad=True) for shape in shapes)
    vk_inputs = tuple(value.detach().to(device).requires_grad_() for value in cpu_inputs)
    cpu_output, vk_output = operation(*cpu_inputs), operation(*vk_inputs)
    generator_output = torch.Generator(device="cpu").manual_seed(seed + 100)
    cpu_seed = torch.randn(cpu_output.shape, generator=generator_output).requires_grad_()
    vk_seed = cpu_seed.detach().to(device).requires_grad_()
    directions = tuple(torch.randn(value.shape, generator=generator_output) + 0.25
                       for value in cpu_inputs)
    vk_directions = tuple(value.to(device) for value in directions)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    first_before = _graph_snap()
    vk_first = torch.autograd.grad(vk_output, vk_inputs, vk_seed, create_graph=True)
    pytorch_vulkan._C.synchronize()
    first_after = _graph_snap()
    cpu_first = torch.autograd.grad(cpu_output, cpu_inputs, cpu_seed, create_graph=True)
    for actual, expected in zip(vk_first, cpu_first):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
        assert actual.requires_grad == expected.requires_grad
    cpu_energy_output, vk_energy_output = operation(*cpu_inputs), operation(*vk_inputs)
    cpu_energy = (cpu_energy_output * cpu_energy_output).sum()
    vk_energy = (vk_energy_output * vk_energy_output).sum()
    cpu_energy_first = torch.autograd.grad(cpu_energy, cpu_inputs, create_graph=True)
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    higher_before = _graph_snap()
    vk_energy_first = torch.autograd.grad(vk_energy, vk_inputs, create_graph=True)
    cpu_directional = sum((gradient * direction).sum()
                          for gradient, direction in zip(cpu_energy_first, directions))
    vk_directional = sum((gradient * direction).sum()
                         for gradient, direction in zip(vk_energy_first, vk_directions))
    cpu_hvp = torch.autograd.grad(cpu_directional, cpu_inputs)
    vk_hvp = torch.autograd.grad(vk_directional, vk_inputs)
    pytorch_vulkan._C.synchronize()
    higher_after = _graph_snap()
    for actual, expected in zip(vk_hvp, cpu_hvp):
        torch.testing.assert_close(actual.cpu(), expected, rtol=0.003, atol=0.003)
    base = tuple(value.detach() for value in cpu_inputs)
    eps = 0.001
    def directional_first(step):
        shifted = tuple((value + step * direction).requires_grad_()
                        for value, direction in zip(base, directions))
        output = operation(*shifted)
        gradients = torch.autograd.grad((output * output).sum(), shifted)
        return sum((gradient * direction).sum()
                   for gradient, direction in zip(gradients, directions))
    fd = (directional_first(eps) - directional_first(-eps)) / (2 * eps)
    analytic = sum((gradient * direction).sum()
                   for gradient, direction in zip(cpu_hvp, directions))
    torch.testing.assert_close(analytic, fd, rtol=0.008, atol=0.002)
    first_error = max((actual.detach().cpu() - expected.detach()).abs().max().item()
                      for actual, expected in zip(vk_first, cpu_first))
    hvp_error = max((actual.detach().cpu() - expected.detach()).abs().max().item()
                    for actual, expected in zip(vk_hvp, cpu_hvp))
    fd_error = abs((analytic - fd).item())
    torch.testing.assert_close(vk_output.cpu(), cpu_output, rtol=0.003, atol=0.003)
    execution = {"first_backward": _graph_delta(first_before, first_after),
                "higher_order": _graph_delta(higher_before, higher_after)}
    for measured in execution.values():
        if measured["compute_dispatches"] <= 0 or any(
                measured[key] for key in ("explicit_transfers", "fallbacks")):
            raise AssertionError(f"{name}: graph route used no compute or incurred host/fallback work: {measured}")
    operand_metadata = {}
    for role, cpu_value, vk_value in zip(roles, cpu_inputs, vk_inputs):
        operand_metadata[role] = {"cpu": _graph_tensor_meta(cpu_value),
                                  "vulkan": _graph_tensor_meta(vk_value)}
    operand_metadata["grad_output"] = {"cpu": _graph_tensor_meta(cpu_seed),
                                        "vulkan": _graph_tensor_meta(vk_seed)}
    derivative_results = []
    for direction, actuals, expecteds in (("first_reverse", vk_first, cpu_first),
                                          ("second_reverse_mixed", vk_hvp, cpu_hvp)):
        for slot, (actual, expected) in enumerate(zip(actuals, expecteds)):
            derivative_results.append({"direction": direction, "target": roles[slot],
                                       "slot": slot, "cpu_defined": True,
                                       "vulkan_defined": True,
                                       "cpu_shape": list(expected.shape),
                                       "vulkan_shape": list(actual.shape),
                                       "cpu_requires_grad": bool(expected.requires_grad),
                                       "vulkan_requires_grad": bool(actual.requires_grad),
                                       "cpu_grad_fn": _graph_grad_fn(expected),
                                       "vulkan_grad_fn": _graph_grad_fn(actual),
                                       "cpu_nonzero": _graph_nonzero(expected),
                                       "vulkan_nonzero": _graph_nonzero(actual.detach().cpu()),
                                       "oracle": "cpu"})
    route = {key: contract[key] for key in (
        "forward_schema", "generated_backward_schema", "native_double_backward_schema")}
    payload = record_graph_autograd_coverage(
        name, route, contract["geometry"], operand_metadata,
        contract["directions"], contract["graph_levels"], derivative_results,
        {"cpu": [], "vulkan": []}, execution,
         {"first_reverse_max_abs": first_error,
          "second_reverse_hvp_max_abs": hvp_error,
          "cpu_centered_fd_abs": fd_error})
    if name in MATRIX_GRAPH_MODE_OBSERVATION_CASES:
        payload["execution_mode_observation"] = {
            "execution_mode": pytorch_vulkan._C.execution_mode(),
            "execution": execution,
        }
    return vk_output, cpu_output, vk_inputs, payload


class _GraphTrace(torch.utils._python_dispatch.TorchDispatchMode):
    def __init__(self, phase):
        super().__init__()
        self.phase, self.events = phase, []
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        output = func(*args, **kwargs)
        def metadata(value):
            if isinstance(value, torch.Tensor):
                return {key: _graph_tensor_meta(value)[key] for key in ("shape", "strides", "storage_offset", "device")}
            if isinstance(value, (tuple, list)):
                return [item for child in value if (item := metadata(child)) is not None]
            return None
        def collect(value):
            if isinstance(value, torch.Tensor): return [metadata(value)]
            if isinstance(value, (tuple, list)): return [item for child in value for item in collect(child)]
            return []
        self.events.append({"phase": self.phase, "operator": str(func),
                            "arguments": collect(args), "output": collect(output)})
        return output


def run_graph_autograd_case(case: ConformanceCase, device: str = "vk:0"):
    """Execute one of the five finite public-API graph recipes and record actual work."""
    name = case.graph_autograd_case
    if name.startswith("matrix.graph."):
        return _run_matrix_graph_autograd_case(case, device)
    contract = GRAPH_AUTOGRAD_CASES[name]
    seed = {"convolution.graph.grouped.ggI-ggW-ggb": 4701,
            "convolution.graph.grouped.selected-third": 4701,
            "convolution.graph.depthwise.selected-third": 4702,
            "convolution.graph.transposed.second": 4703,
            "convolution.graph.overrideable.first": 4704}[name]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    shapes = {
        "convolution.graph.grouped.ggI-ggW-ggb": ((1, 4, 6, 7), (6, 2, 3, 2), (6,), (1, 6, 3, 5)),
        "convolution.graph.grouped.selected-third": ((1, 4, 6, 7), (6, 2, 3, 2), (6,), (1, 6, 3, 5)),
        "convolution.graph.depthwise.selected-third": ((1, 2, 5, 6), (4, 1, 2, 3), None, (1, 4, 4, 3)),
        "convolution.graph.transposed.second": ((1, 2, 3, 3), (2, 3, 2, 2), (3,), (1, 3, 7, 6)),
        "convolution.graph.overrideable.first": ((1, 2, 4, 5), (3, 2, 2, 3), (3,), (1, 3, 3, 3)),
    }[name]
    base_shapes = tuple(shape for shape in shapes[:3] if shape is not None)
    cpu_inputs = tuple(torch.randn(shape, generator=gen, dtype=torch.float32,
                                   requires_grad=True) for shape in base_shapes)
    cpu_inputs = (*cpu_inputs, torch.randn(shapes[3], generator=gen,
                                            dtype=torch.float32, requires_grad=True))
    x, w, b, g = cpu_inputs if len(cpu_inputs) == 4 else (*cpu_inputs[:2], None, cpu_inputs[-1])
    cpu_leaves = (x, w) if b is None else (x, w, b)
    seeds = tuple(torch.randn(value.shape, generator=gen) for value in cpu_leaves)
    sx = torch.randn(x.shape, generator=gen) if "selected_third_d_g_d_x_d_w" in contract["directions"] else None
    vk_inputs = tuple(value.detach().to(device).requires_grad_() for value in cpu_inputs)
    xv, wv = vk_inputs[:2]
    bv = vk_inputs[2] if len(vk_inputs) == 4 else None
    gv = vk_inputs[-1]
    vk_seeds = tuple(value.to(device) for value in seeds)
    sxv = sx.to(device) if sx is not None else None
    geom = contract["geometry"]
    params = dict(stride=tuple(geom["stride"]), padding=tuple(geom["padding"]),
                  dilation=tuple(geom["dilation"]), groups=geom["groups"])
    if name.endswith("overrideable.first"):
        cpu_forward = lambda: case.cpu_reference(x, w, b, **params)
    else:
        cpu_forward = lambda: case.operation(x, w, b)
    vk_forward = lambda: case.operation(xv, wv, bv)
    cpu_y = cpu_forward()
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    vk_y = vk_forward()
    pytorch_vulkan._C.synchronize()
    first_before = _graph_snap()
    cpu_first_trace, vk_first_trace = _GraphTrace("first_reverse"), _GraphTrace("first_reverse")
    vk_leaves = (xv, wv) if bv is None else (xv, wv, bv)
    with cpu_first_trace:
        cpu_first = torch.autograd.grad(cpu_y, cpu_leaves, g, create_graph=True, retain_graph=True)
    with vk_first_trace:
        vk_first = torch.autograd.grad(vk_y, vk_leaves, gv, create_graph=True, retain_graph=True)
    first_after = _graph_snap()
    results = []
    pending_values = []
    first_vk_cpu = []
    role_names = ("input", "weight") if b is None else ("input", "weight", "bias")
    def add_results(direction, targets, actuals, expecteds, oracle="cpu"):
        for slot, (target, actual, expected) in enumerate(zip(targets, actuals, expecteds)):
            entry = {"direction": direction, "target": target, "slot": slot,
                            "cpu_defined": expected is not None, "vulkan_defined": actual is not None,
                            "cpu_shape": list(expected.shape) if expected is not None else None,
                            "vulkan_shape": list(actual.shape) if actual is not None else None,
                            "cpu_requires_grad": bool(expected.requires_grad) if expected is not None else False,
                            "vulkan_requires_grad": bool(actual.requires_grad) if actual is not None else False,
                            "cpu_grad_fn": _graph_grad_fn(expected), "vulkan_grad_fn": _graph_grad_fn(actual),
                            "cpu_nonzero": False, "vulkan_nonzero": False, "oracle": oracle}
            results.append(entry)
            pending_values.append((entry, actual, expected))
    for index, (actual, expected) in enumerate(zip(vk_first, cpu_first)):
        actual_cpu = _graph_cpu_readback(actual)
        first_vk_cpu.append(actual_cpu)
        results.append({"direction": "first_reverse", "target": role_names[index], "slot": index,
                        "cpu_defined": True, "vulkan_defined": True,
                        "cpu_shape": list(expected.shape), "vulkan_shape": list(actual.shape),
                        "cpu_requires_grad": expected.requires_grad, "vulkan_requires_grad": actual.requires_grad,
                        "cpu_grad_fn": _graph_grad_fn(expected), "vulkan_grad_fn": _graph_grad_fn(actual),
                        "cpu_nonzero": _graph_nonzero(expected.detach()),
                        "vulkan_nonzero": _graph_nonzero(actual_cpu), "oracle": "cpu"})
    # First-result readbacks occur only after the first-phase snapshot. Exclude
    # those reads from the independently measured higher-order region.
    pytorch_vulkan._C.synchronize()
    pytorch_vulkan._C.reset_execution_counters()
    higher_before = _graph_snap()
    if len(contract["graph_levels"]) > 1:
        targets_all = (x, w, b, g) if b is not None else (x, w, g)
        vk_targets_all = (xv, wv, bv, gv) if bv is not None else (xv, wv, gv)
        for dir_name, idx, cpu_targets, vk_targets in (
            ("second_reverse_ggI", 0, (w, g), (wv, gv)),
            ("second_reverse_ggW", 1, (x, g), (xv, gv)),
        ):
            if dir_name not in contract["directions"]: continue
            ct, vt = _GraphTrace(dir_name), _GraphTrace(dir_name)
            with ct: cval = torch.autograd.grad((cpu_first[idx] * seeds[idx]).sum(), cpu_targets, allow_unused=True, retain_graph=True)
            with vt: vval = torch.autograd.grad((vk_first[idx] * vk_seeds[idx]).sum(), vk_targets, allow_unused=True, retain_graph=True)
            cpu_first_trace.events.extend(ct.events); vk_first_trace.events.extend(vt.events)
            targets = ("weight", "grad_output") if idx == 0 else ("input", "grad_output")
            add_results(dir_name, targets, vval, cval)
        if "second_reverse_ggb" in contract["directions"]:
            ct, vt = _GraphTrace("second_reverse_ggb"), _GraphTrace("second_reverse_ggb")
            cpu_targets, vk_targets = (x, w, g), (xv, wv, gv)
            with ct: cval = torch.autograd.grad((cpu_first[2] * seeds[2]).sum(), cpu_targets, allow_unused=True, retain_graph=True)
            with vt: vval = torch.autograd.grad((vk_first[2] * vk_seeds[2]).sum(), vk_targets, allow_unused=True, retain_graph=True)
            cpu_first_trace.events.extend(ct.events); vk_first_trace.events.extend(vt.events)
            add_results("second_reverse_ggb", ("input", "weight", "grad_output"), vval, cval)
        if "second_reverse_combined" in contract["directions"]:
            ct, vt = _GraphTrace("second_reverse_combined"), _GraphTrace("second_reverse_combined")
            with ct: cval = torch.autograd.grad(cpu_first, targets_all, grad_outputs=seeds if b is not None else (*seeds, torch.zeros_like(g)), allow_unused=True, retain_graph=True)
            with vt: vval = torch.autograd.grad(vk_first, vk_targets_all, grad_outputs=vk_seeds if bv is not None else (*vk_seeds, torch.zeros_like(gv)), allow_unused=True, retain_graph=True)
            cpu_first_trace.events.extend(ct.events); vk_first_trace.events.extend(vt.events)
            tgt = ("input", "weight", "bias", "grad_output") if b is not None else ("input", "weight", "grad_output")
            add_results("second_reverse_combined", tgt, vval, cval)
        if "selected_third_d_g_d_x_d_w" in contract["directions"]:
            idx = 1
            sw = seeds[idx]
            ct, vt = _GraphTrace("selected_third_d_g_d_x_d_w"), _GraphTrace("selected_third_d_g_d_x_d_w")
            with ct:
                cm = torch.autograd.grad((cpu_first[idx] * sw).sum(), x, create_graph=True, retain_graph=True)[0]
                cthird = torch.autograd.grad((cm * sx).sum(), g)[0]
            with vt:
                vm = torch.autograd.grad((vk_first[idx] * vk_seeds[idx]).sum(), xv, create_graph=True, retain_graph=True)[0]
                vthird = torch.autograd.grad((vm * sxv).sum(), gv)[0]
            cpu_first_trace.events.extend(ct.events); vk_first_trace.events.extend(vt.events)
            oracle = torch.nn.functional.conv2d(sx, sw, None, **params)
            torch.testing.assert_close(cthird, oracle, rtol=0, atol=0)
            add_results("selected_third_d_g_d_x_d_w", ("grad_output",), (vthird,), (cthird,), "cpu_and_analytical_conv")
        higher_after = _graph_snap()
    else:
        higher_after = _graph_snap()
    # Snapshots precede every readback used for parity/nonzero checks.
    for entry, actual, expected in pending_values:
        if actual is not None and expected is not None:
            actual_cpu = _graph_cpu_readback(actual)
            expected_cpu = expected.detach()
            torch.testing.assert_close(actual_cpu, expected_cpu, rtol=8e-4, atol=8e-4)
            entry["cpu_nonzero"] = _graph_nonzero(expected_cpu)
            entry["vulkan_nonzero"] = _graph_nonzero(actual_cpu)
    vk_y_cpu = _graph_cpu_readback(vk_y)
    if not torch.allclose(vk_y_cpu, cpu_y.detach(), rtol=case.rtol, atol=case.atol):
        raise AssertionError(f"{name}: forward CPU/Vulkan parity failed")
    for actual_cpu, expected in zip(first_vk_cpu, cpu_first):
        torch.testing.assert_close(actual_cpu, expected.detach(), rtol=8e-4, atol=8e-4)
    operands = {role: {"cpu": _graph_tensor_meta(tensor), "vulkan": _graph_tensor_meta(vkt)}
                for role, tensor, vkt in (("input", x, xv), ("weight", w, wv))}
    operands["bias"] = {"cpu": _graph_tensor_meta(b), "vulkan": _graph_tensor_meta(bv)}
    operands["grad_output"] = {"cpu": _graph_tensor_meta(g), "vulkan": _graph_tensor_meta(gv)}
    route = {key: contract[key] for key in ("forward_schema", "generated_backward_schema", "native_double_backward_schema")}
    execution = {"first_backward": _graph_delta(first_before, first_after),
                 "higher_order": _graph_delta(higher_before, higher_after)}
    payload = record_graph_autograd_coverage(name, route, geom, operands,
        contract["directions"], contract["graph_levels"], results,
        {"cpu": cpu_first_trace.events, "vulkan": vk_first_trace.events}, execution)
    return vk_y, cpu_y, vk_inputs, payload


def validate_graph_autograd_record(case_name, record):
    """Fail closed on graph payloads that do not match the source-owned witness."""
    if case_name not in GRAPH_AUTOGRAD_CASES:
        raise ValueError(f"{case_name}: unknown graph autograd witness")
    if not isinstance(record, dict):
        raise ValueError(f"{case_name}: coverage record must be an object")
    contract = GRAPH_AUTOGRAD_CASES[case_name]
    if case_name.startswith("matrix.graph."):
        payload = record.get("graph_autograd")
        if (record.get("schema") != contract["schema"] or record.get("parity") is not True
                or record.get("output_dtype") != "float32"):
            raise ValueError(f"{case_name}: direct matrix witness schema/dtype/parity mismatch")
        output_rank = 3 if case_name.endswith("bmm.generated") else 2
        if record.get("output_rank") != output_rank:
            raise ValueError(f"{case_name}: direct matrix witness output rank mismatch")
        route = {key: contract[key] for key in (
            "forward_schema", "generated_backward_schema", "native_double_backward_schema")}
        if (not isinstance(payload, dict)
                or payload.get("route") != route
                or payload.get("geometry") != contract["geometry"]
                or payload.get("directions") != contract["directions"]
                or payload.get("graph_levels") != contract["graph_levels"]):
            raise ValueError(f"{case_name}: direct matrix graph payload differs from its source contract")
        roles = payload.get("operands")
        expected_roles = {"input", "weight", "grad_output"} | ({"bias"} if case_name.endswith("addmm.generated") else set())
        if not isinstance(roles, dict) or set(roles) != expected_roles:
            raise ValueError(f"{case_name}: direct matrix operand roles mismatch")
        geometries = contract["geometry"]["shapes"]
        expected_shapes = {"input": geometries[0], "weight": geometries[1]}
        if case_name.endswith("addmm.generated"):
            expected_shapes = {"bias": contract["geometry"]["self_shape"],
                               "input": geometries[1], "weight": geometries[2]}
        expected_shapes["grad_output"] = [2, 3, 5] if output_rank == 3 else [2, 4]
        for role, pair in roles.items():
            if not isinstance(pair, dict) or set(pair) != {"cpu", "vulkan"}:
                raise ValueError(f"{case_name}: malformed {role} metadata pair")
            for side, meta in pair.items():
                if (not isinstance(meta, dict) or meta.get("defined") is not True
                        or meta.get("dtype") != "float32" or meta.get("device") != ("cpu" if side == "cpu" else "vk:0")
                        or type(meta.get("requires_grad")) is not bool
                        or meta.get("requires_grad") is not True
                        or meta.get("shape") != expected_shapes[role]
                        or meta.get("strides") != list(torch.empty(expected_shapes[role]).stride())
                        or meta.get("storage_offset") != 0):
                    raise ValueError(f"{case_name}: malformed {side} {role} metadata; requires_grad must be true")
        results = payload.get("derivative_results")
        derivative_contract = contract["derivative_contract"]
        expected_identity = {
            (direction, role, slot)
            for direction, derivative in derivative_contract.items()
            for slot, role in enumerate(derivative["targets"])
        }
        if not isinstance(results, list):
            raise ValueError(f"{case_name}: derivative results must be a list")
        actual_identity = []
        for result in results:
            if (not isinstance(result, dict)
                    or type(result.get("direction")) is not str
                    or type(result.get("target")) is not str
                    or type(result.get("slot")) is not int):
                raise ValueError(f"{case_name}: malformed derivative result identity")
            actual_identity.append((result["direction"], result["target"], result["slot"]))
        if len(actual_identity) != len(set(actual_identity)):
            raise ValueError(f"{case_name}: duplicate derivative result identity")
        if set(actual_identity) != expected_identity or len(actual_identity) != len(expected_identity):
            raise ValueError(f"{case_name}: derivative result identities differ from source slots")
        for result in results:
            direction, target = result["direction"], result["target"]
            expected_shape = expected_shapes[target]
            expected_history = derivative_contract[direction]["requires_grad"]
            for key in ("cpu_defined", "vulkan_defined", "cpu_nonzero", "vulkan_nonzero",
                        "cpu_requires_grad", "vulkan_requires_grad"):
                if type(result.get(key)) is not bool:
                    raise ValueError(f"{case_name}: malformed derivative result field {key}")
            if (result["cpu_defined"] is not True or result["vulkan_defined"] is not True
                    or result["cpu_nonzero"] is not True or result["vulkan_nonzero"] is not True
                    or result.get("oracle") != "cpu"
                    or result.get("cpu_shape") != expected_shape
                    or result.get("vulkan_shape") != expected_shape):
                raise ValueError(f"{case_name}: {direction}/{target} derivative dependency/shape differs from source")
            for side in ("cpu", "vulkan"):
                history = result[f"{side}_requires_grad"]
                grad_fn = result.get(f"{side}_grad_fn")
                if history is not expected_history or (expected_history and
                        (type(grad_fn) is not str or not grad_fn)) or (not expected_history and grad_fn is not None):
                    if direction == "second_reverse_mixed":
                        raise ValueError(f"{case_name}: second_reverse_mixed derivative must be graphless")
                    raise ValueError(f"{case_name}: first_reverse derivative must preserve graph history")
        if payload.get("route_observations") != {"cpu": [], "vulkan": []}:
            raise ValueError(f"{case_name}: matrix dispatcher trace is outside this finite witness schema")
        errors = payload.get("numerical_errors")
        if (not isinstance(errors, dict)
                or set(errors) != {"first_reverse_max_abs", "second_reverse_hvp_max_abs", "cpu_centered_fd_abs"}
                or any(not isinstance(value, (int, float)) or not torch.isfinite(torch.tensor(value))
                       or value < 0 for value in errors.values())):
            raise ValueError(f"{case_name}: missing measured matrix finite derivative errors")
        execution = payload.get("execution")
        if not isinstance(execution, dict) or set(execution) != {"first_backward", "higher_order"}:
            raise ValueError(f"{case_name}: missing direct matrix execution phases")
        counter_fields = {
            "compute_dispatches", "vulkan_copies", "explicit_transfers",
            "fallbacks", "buffer_creations_delta", "live_allocations_delta",
        }
        for counters in execution.values():
            if (not isinstance(counters, dict) or set(counters) != counter_fields
                    or any(type(value) is not int or value < 0 for value in counters.values())
                    or counters["compute_dispatches"] <= 0
                    or counters["explicit_transfers"] != 0 or counters["fallbacks"] != 0):
                raise ValueError(f"{case_name}: direct matrix phase lacks compute-only Vulkan evidence")
        if case_name in MATRIX_GRAPH_MODE_OBSERVATION_CASES:
            def validate_mode_execution(mode, measured):
                expected = MATRIX_GRAPH_MODE_EXECUTION_CONTRACT.get(mode)
                if expected is None or not isinstance(measured, dict):
                    raise ValueError(f"{case_name}: unknown or malformed runtime execution mode")
                if set(measured) != {"first_backward", "higher_order"}:
                    raise ValueError(f"{case_name}: runtime observation is missing a measured phase")
                counter_fields = {
                    "compute_dispatches", "vulkan_copies", "explicit_transfers",
                    "fallbacks", "buffer_creations_delta", "live_allocations_delta",
                }
                for phase, counters in measured.items():
                    if (not isinstance(counters, dict) or set(counters) != counter_fields
                            or any(type(value) is not int or value < 0
                                   for value in counters.values())):
                        raise ValueError(f"{case_name}: malformed {mode}/{phase} counters")
                if measured != expected:
                    raise ValueError(
                        f"{case_name}: {mode} runtime phases differ from the source-owned measurement"
                    )

            capture = payload.get("execution_mode_observation")
            mode_observations = payload.get("execution_modes")
            if capture is not None:
                if mode_observations is not None or not isinstance(capture, dict) or set(capture) != {
                    "execution_mode", "execution"
                }:
                    raise ValueError(f"{case_name}: malformed single-mode runtime capture")
                mode = capture["execution_mode"]
                if mode != pytorch_vulkan._C.execution_mode():
                    raise ValueError(f"{case_name}: runtime capture mode differs from initialized executor")
                if capture["execution"] != execution:
                    raise ValueError(f"{case_name}: runtime capture has no matching phase history")
                validate_mode_execution(mode, capture["execution"])
            else:
                if not isinstance(mode_observations, list) or len(mode_observations) != 2:
                    raise ValueError(f"{case_name}: async and sync runtime observations are required")
                by_mode = {}
                for observation in mode_observations:
                    if (not isinstance(observation, dict)
                            or set(observation) != {"execution_mode", "execution"}
                            or type(observation.get("execution_mode")) is not str):
                        raise ValueError(f"{case_name}: malformed mode-specific runtime observation")
                    mode = observation["execution_mode"]
                    if mode in by_mode:
                        raise ValueError(f"{case_name}: duplicate runtime mode observation")
                    validate_mode_execution(mode, observation["execution"])
                    by_mode[mode] = observation["execution"]
                if set(by_mode) != {"async", "sync"}:
                    raise ValueError(f"{case_name}: runtime observations must bind async and sync")
                if execution != by_mode["async"]:
                    raise ValueError(f"{case_name}: canonical execution must preserve the async observation")
        return True
    payload = record.get("graph_autograd")
    if not isinstance(payload, dict):
        raise ValueError(f"{case_name}: missing graph_autograd payload")
    if type(record.get("parity")) is not bool or record["parity"] is not True:
        raise ValueError(f"{case_name}: outer graph witness parity must be true")
    if record.get("output_dtype") != "float32" or type(record.get("output_rank")) is not int or record["output_rank"] != 4:
        raise ValueError(f"{case_name}: outer graph output must be float32 rank 4")
    if set(payload) != {"route", "geometry", "operands", "directions", "graph_levels", "derivative_results", "route_observations", "execution"}:
        raise ValueError(f"{case_name}: graph payload fields differ from schema")
    route = {key: contract[key] for key in ("forward_schema", "generated_backward_schema", "native_double_backward_schema")}
    for key, expected in (("schema", contract["schema"]), ("graph_autograd.route", route),
                          ("graph_autograd.geometry", contract["geometry"]),
                          ("graph_autograd.directions", contract["directions"]),
                          ("graph_autograd.graph_levels", contract["graph_levels"])):
        actual = record.get(key) if key == "schema" else payload.get(key.removeprefix("graph_autograd."))
        if actual != expected:
            raise ValueError(f"{case_name}: {key} does not match source-owned graph contract")
    roles = payload.get("operands")
    if not isinstance(roles, dict) or set(roles) != {"input", "weight", "bias", "grad_output"}:
        raise ValueError(f"{case_name}: graph roles must include input, weight, bias, grad_output")
    expected_shapes = {
        "convolution.graph.grouped.ggI-ggW-ggb": {"input": ([1,4,6,7],[168,42,7,1]), "weight": ([6,2,3,2],[12,6,2,1]), "bias": ([6],[1]), "grad_output": ([1,6,3,5],[90,15,5,1])},
        "convolution.graph.grouped.selected-third": {"input": ([1,4,6,7],[168,42,7,1]), "weight": ([6,2,3,2],[12,6,2,1]), "bias": ([6],[1]), "grad_output": ([1,6,3,5],[90,15,5,1])},
        "convolution.graph.depthwise.selected-third": {"input": ([1,2,5,6],[60,30,6,1]), "weight": ([4,1,2,3],[6,6,3,1]), "bias": None, "grad_output": ([1,4,4,3],[48,12,3,1])},
        "convolution.graph.transposed.second": {"input": ([1,2,3,3],[18,9,3,1]), "weight": ([2,3,2,2],[12,4,2,1]), "bias": ([3],[1]), "grad_output": ([1,3,7,6],[126,42,6,1])},
        "convolution.graph.overrideable.first": {"input": ([1,2,4,5],[40,20,5,1]), "weight": ([3,2,2,3],[12,6,3,1]), "bias": ([3],[1]), "grad_output": ([1,3,3,3],[27,9,3,1])},
    }[case_name]
    primary_input = record.get("primary_input")
    if not isinstance(primary_input, dict) or set(primary_input) != {"dtype", "rank"}:
        raise ValueError(f"{case_name}: malformed outer primary-input coverage")
    dtypes = set()
    for role, spec in expected_shapes.items():
        shape, strides = spec if spec is not None else (None, None)
        pair = roles.get(role)
        if not isinstance(pair, dict) or set(pair) != {"cpu", "vulkan"}:
            raise ValueError(f"{case_name}: malformed {role} device metadata")
        for side, meta in pair.items():
            if not isinstance(meta, dict) or set(meta) != {"defined", "dtype", "rank", "shape", "strides", "storage_offset", "device", "requires_grad"}:
                raise ValueError(f"{case_name}: malformed {side} {role} metadata fields")
            if type(meta["defined"]) is not bool or type(meta["requires_grad"]) is not bool:
                raise ValueError(f"{case_name}: malformed {side} {role} boolean metadata")
            if shape is None:
                if meta.get("defined") is not False or any(meta.get(field) is not None for field in ("dtype", "rank", "shape", "strides", "storage_offset", "device")):
                    raise ValueError(f"{case_name}: absent bias metadata must remain None")
                if meta["requires_grad"] is not False:
                    raise ValueError(f"{case_name}: undefined bias must have requires_grad=False")
                continue
            if (meta.get("defined") is not True or not isinstance(meta.get("dtype"), str) or
                    meta.get("rank") != len(shape) or meta.get("shape") != shape or
                    meta.get("strides") != strides or meta.get("storage_offset") != 0 or
                    meta.get("requires_grad") is not True):
                raise ValueError(f"{case_name}: invalid {side} {role} metadata")
            if side == "cpu" and meta.get("device") != "cpu":
                raise ValueError(f"{case_name}: CPU {role} device is not cpu")
            if side == "vulkan" and meta.get("device") != "vk:0":
                raise ValueError(f"{case_name}: Vulkan {role} device is not vk:0")
            dtypes.add(meta["dtype"])
    expected_ranks = sorted({len(spec[0]) for spec in expected_shapes.values() if spec is not None})
    expected_input_shapes = sorted({"x".join(map(str, spec[0])) for spec in expected_shapes.values() if spec is not None})
    if (dtypes != {"float32"} or record.get("input_dtypes") != ["float32"] or
            record.get("primary_input") != {"dtype": "float32", "rank": 4} or
            record.get("input_ranks") != expected_ranks or
            record.get("input_shapes") != expected_input_shapes):
        raise ValueError(f"{case_name}: graph witness must be float32 at every role")
    derivative_results = payload.get("derivative_results")
    if not isinstance(derivative_results, list):
        raise ValueError(f"{case_name}: missing derivative results")
    result_contracts = {
        "convolution.graph.grouped.ggI-ggW-ggb": {"first_reverse": {"input": True,"weight": True,"bias": True}, "second_reverse_ggI": {"weight": True,"grad_output": True}, "second_reverse_ggW": {"input": True,"grad_output": True}, "second_reverse_ggb": {"input": False,"weight": False,"grad_output": True}, "second_reverse_combined": {"input": True,"weight": True,"bias": False,"grad_output": True}},
        "convolution.graph.grouped.selected-third": {"first_reverse": {"input": True,"weight": True,"bias": True}, "second_reverse_ggW": {"input": True,"grad_output": True}, "selected_third_d_g_d_x_d_w": {"grad_output": True}},
        "convolution.graph.depthwise.selected-third": {"first_reverse": {"input": True,"weight": True}, "second_reverse_ggW": {"input": True,"grad_output": True}, "selected_third_d_g_d_x_d_w": {"grad_output": True}},
        "convolution.graph.transposed.second": {"first_reverse": {"input": True,"weight": True,"bias": True}, "second_reverse_ggI": {"weight": True,"grad_output": True}, "second_reverse_ggW": {"input": True,"grad_output": True}, "second_reverse_ggb": {"input": False,"weight": False,"grad_output": True}, "second_reverse_combined": {"input": True,"weight": True,"bias": False,"grad_output": True}},
        "convolution.graph.overrideable.first": {"first_reverse": {"input": True,"weight": True,"bias": True}},
    }[case_name]
    expected_pairs = {
        (direction, target, slot)
        for direction, targets in result_contracts.items()
        for slot, target in enumerate(targets)
    }
    result_fields = {"direction", "target", "slot", "cpu_defined", "vulkan_defined",
                     "cpu_shape", "vulkan_shape", "cpu_requires_grad", "vulkan_requires_grad",
                     "cpu_grad_fn", "vulkan_grad_fn", "cpu_nonzero", "vulkan_nonzero", "oracle"}
    actual_pairs = []
    for result in derivative_results:
        if not isinstance(result, dict) or set(result) != result_fields:
            raise ValueError(f"{case_name}: malformed derivative result")
        for key in ("cpu_defined", "vulkan_defined", "cpu_requires_grad", "vulkan_requires_grad", "cpu_nonzero", "vulkan_nonzero"):
            if type(result[key]) is not bool:
                raise ValueError(f"{case_name}: derivative result {key} must be bool")
        if (not isinstance(result["direction"], str) or not isinstance(result["target"], str)
                or type(result["slot"]) is not int
                or result["cpu_grad_fn"] is not None and not isinstance(result["cpu_grad_fn"], str)
                or result["vulkan_grad_fn"] is not None and not isinstance(result["vulkan_grad_fn"], str)):
            raise ValueError(f"{case_name}: malformed derivative result scalar")
        if not isinstance(result["oracle"], str) or result["oracle"] not in {"cpu", "cpu_and_analytical_conv"}:
            raise ValueError(f"{case_name}: unrecognized oracle")
        actual_pairs.append((result["direction"], result["target"], result["slot"]))
    if len(actual_pairs) != len(set(actual_pairs)):
        raise ValueError(f"{case_name}: duplicate derivative result direction/target/slot")
    if set(actual_pairs) != expected_pairs or len(actual_pairs) != len(expected_pairs):
        raise ValueError(f"{case_name}: unrecognized derivative direction or result target/slot set")
    result_by_pair = dict(zip(actual_pairs, derivative_results))
    for direction, targets in result_contracts.items():
        for slot, (target, defined) in enumerate(targets.items()):
            result = result_by_pair[(direction, target, slot)]
            if result["cpu_defined"] is not defined or result["vulkan_defined"] is not defined:
                raise ValueError(f"{case_name}: {direction}/{target} defined slot differs from CPU reference")
            target_spec = expected_shapes.get(target)
            target_shape = target_spec[0] if target_spec is not None else None
            if defined and (result["cpu_shape"] != target_shape or result["vulkan_shape"] != target_shape):
                raise ValueError(f"{case_name}: {direction}/{target} result shape differs from source contract")
            if not defined and (any(result[key] is not None for key in ("cpu_shape", "vulkan_shape", "cpu_grad_fn", "vulkan_grad_fn"))
                                or result["cpu_requires_grad"] is not False or result["vulkan_requires_grad"] is not False
                                or result["cpu_nonzero"] is not False or result["vulkan_nonzero"] is not False):
                raise ValueError(f"{case_name}: undefined derivative slot has fabricated history/value")
            if defined and (result["cpu_nonzero"] is not True or result["vulkan_nonzero"] is not True):
                raise ValueError(f"{case_name}: required derivative witness must be nonzero on CPU and Vulkan")
            if direction == "first_reverse":
                cpu_node = "ConvolutionBackwardBackward0"
                vk_node = ("ConvolutionBackwardOverrideableBackward0"
                          if case_name == "convolution.graph.overrideable.first"
                          else "ConvolutionBackwardBackward0")
                if (result["cpu_requires_grad"] is not True or result["vulkan_requires_grad"] is not True
                        or result["cpu_grad_fn"] != cpu_node or result["vulkan_grad_fn"] != vk_node):
                    raise ValueError(f"{case_name}: first_reverse requires_grad/native grad_fn differs from source contract")
            elif (result["cpu_requires_grad"] is not False or result["vulkan_requires_grad"] is not False
                    or result["cpu_grad_fn"] is not None or result["vulkan_grad_fn"] is not None):
                raise ValueError(f"{case_name}: returned second/third derivative history must be graphless")
            expected_oracle = "cpu_and_analytical_conv" if direction == "selected_third_d_g_d_x_d_w" else "cpu"
            if result["oracle"] != expected_oracle:
                raise ValueError(f"{case_name}: unrecognized oracle for {direction}")
            if result["cpu_requires_grad"] != result["vulkan_requires_grad"]:
                raise ValueError(f"{case_name}: CPU/Vulkan derivative history parity mismatch")
            if result["cpu_shape"] != result["vulkan_shape"] or result["cpu_defined"] != result["vulkan_defined"]:
                raise ValueError(f"{case_name}: CPU/Vulkan derivative defined/shape parity mismatch")
    for result in derivative_results:
        if result["cpu_defined"] != result["vulkan_defined"] or result["cpu_shape"] != result["vulkan_shape"]:
            raise ValueError(f"{case_name}: CPU/Vulkan derivative defined/shape parity mismatch")
    traces = payload.get("route_observations")
    if not isinstance(traces, dict) or set(traces) != {"cpu", "vulkan"}:
        raise ValueError(f"{case_name}: missing CPU/Vulkan route traces")
    allowed_operators = {
        "aten.convolution.default", "aten.convolution_backward.default",
        "aten.convolution_backward_overrideable.default", "aten.view.default",
        "aten.cat.default", "aten.expand.default", "aten.add.Tensor",
        "aten.slice.Tensor", "aten.transpose.int", "aten.mul.Tensor",
        "aten.ones_like.default", "aten.sum.default",
    }
    for side, events in traces.items():
        if not isinstance(events, list) or any(not isinstance(event, dict) or set(event) != {"phase", "operator", "arguments", "output"} for event in events):
            raise ValueError(f"{case_name}: malformed {side} trace")
        expected_device = "cpu" if side == "cpu" else "vk:0"
        for event in events:
            if event["phase"] not in contract["directions"]:
                raise ValueError(f"{case_name}: unrecognized route-observation phase")
            if not isinstance(event["operator"], str) or event["operator"] not in allowed_operators:
                raise ValueError(f"{case_name}: unrecognized canonical route-observation operator")
            for collection in (event["arguments"], event["output"]):
                if not isinstance(collection, list):
                    raise ValueError(f"{case_name}: route-observation metadata must be a list")
                for metadata in collection:
                    if (not isinstance(metadata, dict)
                            or set(metadata) != {"shape", "strides", "storage_offset", "device"}
                            or metadata["device"] != expected_device
                            or not isinstance(metadata["shape"], list)
                            or not isinstance(metadata["strides"], list)
                            or len(metadata["shape"]) != len(metadata["strides"])
                            or type(metadata["storage_offset"]) is not int
                            or any(type(value) is not int or value <= 0 for value in metadata["shape"])
                            or any(type(value) is not int or value < 0 for value in metadata["strides"])):
                        raise ValueError(f"{case_name}: malformed route-observation tensor metadata")
        first_operator = ("aten.convolution_backward.default"
                          if case_name != "convolution.graph.overrideable.first" or side == "cpu"
                          else "aten.convolution_backward_overrideable.default")
        if not any(event["phase"] == "first_reverse" and event["operator"] == first_operator
                   for event in events):
            raise ValueError(f"{case_name}: missing exact first_reverse dispatcher operation {first_operator}")
        if max(contract["graph_levels"]) > 1 and not any(
                event["phase"] != "first_reverse" and event["operator"] == "aten.convolution.default"
                for event in events):
            raise ValueError(f"{case_name}: missing exact higher-order convolution dispatcher operation")
        if case_name == "convolution.graph.grouped.ggI-ggW-ggb":
            ggi = [event for event in events if event["phase"] == "second_reverse_ggI"]
            ggw = [event for event in events if event["phase"] == "second_reverse_ggW"]
            comb = [event for event in events if event["phase"] == "second_reverse_combined"]
            ggi_pairs = {
                ((tuple(args[0]["shape"]), tuple(args[0]["strides"]), args[0]["storage_offset"]),
                 (tuple(args[1]["shape"]), tuple(args[1]["strides"]), args[1]["storage_offset"]))
                for event in ggi if event["operator"] == "aten.convolution.default"
                for args in (event["arguments"],) if len(args) >= 2
            }
            required_pairs = {
                (((2,1,6,7),(42,168,7,1),0), ((3,1,3,5),(15,90,5,1),0)),
                (((2,1,6,7),(42,168,7,1),84), ((3,1,3,5),(15,90,5,1),45)),
            }
            if not required_pairs <= ggi_pairs or not any(event["operator"] == "aten.cat.default" and any(meta.get("shape") == [2,6,4,2] for meta in event["output"]) for event in ggi):
                raise ValueError(f"{case_name}: missing grouped ggI per-group view/cat layout trace")
            if not any(event["operator"] == "aten.convolution.default" for event in ggw):
                raise ValueError(f"{case_name}: missing grouped ggW convolution event")
            ggb_ops = {event["operator"] for event in events if event["phase"] == "second_reverse_ggb"}
            if "aten.view.default" not in ggb_ops or "aten.expand.default" not in ggb_ops:
                raise ValueError(f"{case_name}: missing grouped ggb view/expand trace")
            comb_ops = {event["operator"] for event in comb}
            if not {"aten.add.Tensor", "aten.cat.default", "aten.convolution.default"} <= comb_ops:
                raise ValueError(f"{case_name}: missing combined accumulated bias add trace")
    execution = payload.get("execution")
    if (not isinstance(execution, dict) or set(execution) != {"first_backward", "higher_order"}
            or any(not isinstance(execution.get(phase), dict)
                   for phase in ("first_backward", "higher_order"))):
        raise ValueError(f"{case_name}: missing phase execution counters")
    expected_first_dispatches = len(result_contracts["first_reverse"])
    for phase, counters in execution.items():
        counter_fields = {"compute_dispatches", "vulkan_copies", "explicit_transfers", "fallbacks", "buffer_creations_delta", "live_allocations_delta"}
        if not isinstance(counters, dict) or set(counters) != counter_fields:
            raise ValueError(f"{case_name}: malformed {phase} counters")
        if any(type(counters[key]) is not int for key in counters):
            raise ValueError(f"{case_name}: non-runtime execution counter value")
        if any(counters[key] < 0 for key in counters):
            raise ValueError(f"{case_name}: negative execution counter delta")
        if counters["explicit_transfers"] != 0 or counters["fallbacks"] != 0:
            raise ValueError(f"{case_name}: transfers/fallbacks were observed")
    if execution["first_backward"]["compute_dispatches"] != expected_first_dispatches:
        raise ValueError(f"{case_name}: first-backward dispatch count differs from output mask")
    if execution["first_backward"]["vulkan_copies"] != 0:
        raise ValueError(f"{case_name}: first-backward region recorded Vulkan copies")
    if max(contract["graph_levels"]) > 1 and execution["higher_order"]["compute_dispatches"] <= 0:
        raise ValueError(f"{case_name}: higher-order region has no measured dispatch")
    if max(contract["graph_levels"]) == 1 and any(execution["higher_order"].values()):
        raise ValueError(f"{case_name}: first-order-only witness has unexpected higher-order execution")
    return True


def validate_graph_autograd_evidence(coverage, manifest_entries):
    for name in GRAPH_AUTOGRAD_REQUIRED_CASES:
        if name not in coverage:
            raise ValueError(f"required graph autograd witness {name!r} is missing")
        validate_graph_autograd_record(name, coverage[name])
        schema = GRAPH_AUTOGRAD_CASES[name]["schema"]
        entry = next((value for value in manifest_entries if value.get("schema") == schema), None)
        if entry is None or not any(case.get("name") == name and case.get("supported") is True for case in entry.get("test_cases", [])):
            raise ValueError(f"{name}: supported manifest link is missing")


_MANIFEST_CASES = {
    case["name"]: (entry["schema"], case["supported"])
    for entry in _MANIFEST["entries"]
    for case in entry["test_cases"]
}
_MANIFEST_CASES.update({
    "matrix.graph.mm.generated": ("aten::mm.default", True),
    "matrix.graph.addmm.generated": ("aten::addmm.default", True),
    "matrix.graph.bmm.generated": ("aten::bmm.default", True),
    "convolution.forward.bias-present": ("aten::convolution.default", True),
    "convolution.forward.bias-absent": ("aten::convolution.default", True),
    **{
        f"convolution.backward.bias-{state}.mask-{mask:03b}":
        ("aten::convolution_backward.default", True)
        for state in ("present", "absent") for mask in range(8)
    },
})
_MANIFEST_CASES.pop("convolution.parameters.output-padding.rejected", None)
_MANIFEST_CASES["convolution.parameters.output-padding.ignored"] = (
    "aten::convolution.default", True,
)
_MANIFEST_CASES["convolution.backward-overrideable.bias-present.mask-111"] = (
    "aten::convolution_backward_overrideable.default", True,
)
# The former rejected-transposed parameter case is obsolete now that the
# transposed operation has positive executable witnesses below.
_MANIFEST_CASES.pop("convolution.parameters.transposed.rejected", None)
_MANIFEST_CASES.update({
    "convolution.transposed.forward.bias-present": ("aten::convolution.default", True),
    "convolution.transposed.forward.bias-absent": ("aten::convolution.default", True),
    **{
        f"convolution.transposed.backward.bias-{state}.mask-{mask:03b}":
        ("aten::convolution_backward.default", True)
        for state in ("present", "absent") for mask in range(8)
    },
})
_GRAPH_SCHEMAS = {
    "convolution.graph.grouped.ggI-ggW-ggb": "aten::convolution.default",
    "convolution.graph.grouped.selected-third": "aten::convolution.default",
    "convolution.graph.depthwise.selected-third": "aten::convolution.default",
    "convolution.graph.transposed.second": "aten::convolution.default",
    "convolution.graph.overrideable.first": "aten::convolution_overrideable.default",
}
_MANIFEST_CASES.update({name: (schema, True) for name, schema in _GRAPH_SCHEMAS.items()})
DECLARED_OPERATION_MANIFEST = frozenset(
    set(DECLARED_OPERATION_MANIFEST)
    | set(_GRAPH_SCHEMAS.values())
    | {"aten::convolution_backward_overrideable.default"}
)
ROADMAP_DEFERRED_SCHEMAS = frozenset(
    schema for schema in ROADMAP_DEFERRED_SCHEMAS
    if schema not in set(_GRAPH_SCHEMAS.values()) | {"aten::convolution_backward_overrideable.default"}
)
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
    graph_autograd_case=None,
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
        None,
        (),
        (),
        None,
        graph_autograd_case,
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


def _transposed_convolution_inputs(bias_present: bool, *, backward: bool):
    generator = torch.Generator(device="cpu").manual_seed(
        9917 + int(bias_present) + 3 * int(backward)
    )
    value = torch.randn((2, 4, 3, 4), generator=generator, dtype=torch.float32)
    weight = torch.randn((4, 3, 2, 3), generator=generator, dtype=torch.float32)
    bias = torch.randn((6,), generator=generator, dtype=torch.float32) if bias_present else None
    if backward:
        grad_output = torch.randn((2, 6, 8, 6), generator=generator, dtype=torch.float32)
        return grad_output, value, weight
    return value, weight, bias


def _transposed_forward_cpu_oracle(x, weight, bias, stride, padding, dilation,
                                   transposed, output_padding, groups):
    assert transposed is True
    return torch.nn.functional.conv_transpose2d(
        x, weight, bias, stride, padding, output_padding, groups, dilation
    )


def _transposed_context_setup(bias_present: bool, *, backward: bool):
    def setup(inputs, device):
        if backward:
            _, x, weight = inputs
            bias = (
                torch.randn((6,), generator=torch.Generator(device="cpu").manual_seed(9917)).to(device)
                if bias_present else None
            )
        else:
            x, weight, bias = inputs
        output = torch.nn.functional.conv_transpose2d(
            x, weight, bias, stride=(2, 1), padding=(1, 1),
            output_padding=(2, 0), groups=2, dilation=(3, 2),
        )
        metadata = _convolution_tensor_metadata(bias, str(x.device))
        del output
        return inputs, {"warm_forward_bias": metadata}
    return setup


def _transposed_convolution_evidence_cases():
    cases = []
    for present in (True, False):
        state = "present" if present else "absent"
        forward_args = ([2, 1], [1, 1], [3, 2], True, [2, 0], 2)
        cases.append(ConformanceCase(
            f"convolution.transposed.forward.bias-{state}", "convolution",
            "aten::convolution.default", torch.ops.aten.convolution.default,
            lambda requires_grad=False, present=present: _transposed_convolution_inputs(present, backward=False),
            _transposed_forward_cpu_oracle, forward_args, None, True, r"Vulkan",
            torch.float32, (2, 6, 8, 6), False, True, None, None, False,
            3e-4, 3e-4, "compute", True, None,
            ("2x4x3x4", "4x3x2x3") + (("6",) if present else ()), None,
            ("input", "weight", "bias"), "forward", (1,), (0,),
            _transposed_context_setup(present, backward=False),
        ))
        for mask_bits in range(8):
            mask = tuple(bool(mask_bits & (1 << shift)) for shift in range(3))
            mask_name = "".join("1" if bit else "0" for bit in mask)
            bias_sizes = (None, [], [999])[mask_bits % 3]
            args = (bias_sizes, [2, 1], [1, 1], [3, 2], True, [2, 0], 2, list(mask))
            shapes = ((2, 4, 3, 4), (4, 3, 2, 3), (6,))
            operations = tuple(op for op, requested in zip((0, 2, 3), mask) if requested)
            slots = tuple(index for index, requested in enumerate(mask) if requested)
            cases.append(ConformanceCase(
                f"convolution.transposed.backward.bias-{state}.mask-{mask_name}",
                "convolution", "aten::convolution_backward.default",
                torch.ops.aten.convolution_backward.default,
                lambda requires_grad=False, present=present: _transposed_convolution_inputs(present, backward=True),
                torch.ops.aten.convolution_backward.default, args, None, True, r"Vulkan",
                torch.float32, None, False, True, None, None, False,
                4e-4, 4e-4, "compute", True, None,
                ("2x6x8x6", "2x4x3x4", "4x3x2x3"),
                tuple(shape if bit else None for shape, bit in zip(shapes, mask)),
                ("grad_output", "input", "weight"), "backward", operations, slots,
                _transposed_context_setup(present, backward=True),
            ))
    return tuple(cases)


def _graph_overrideable_forward(value, weight, bias):
    return torch.ops.aten.convolution_overrideable.default(
        value, weight, bias, [1, 1], [0, 0], [1, 1], False, [0, 0], 1
    )


_GRAPH_FORWARD_OPERATIONS = {
    "convolution.graph.grouped.ggI-ggW-ggb": lambda x,w,b: torch.nn.functional.conv2d(x,w,b,stride=(2,1),padding=(1,0),dilation=(1,2),groups=2),
    "convolution.graph.grouped.selected-third": lambda x,w,b: torch.nn.functional.conv2d(x,w,b,stride=(2,1),padding=(1,0),dilation=(1,2),groups=2),
    "convolution.graph.depthwise.selected-third": lambda x,w,b: torch.nn.functional.conv2d(x,w,b,stride=(1,2),padding=(0,1),dilation=(1,1),groups=2),
    "convolution.graph.transposed.second": lambda x,w,b: torch.nn.functional.conv_transpose2d(x,w,b,stride=(2,2),output_padding=(1,0)),
    "convolution.graph.overrideable.first": _graph_overrideable_forward,
}
_GRAPH_CPU_REFERENCES = {
    **{name: torch.nn.functional.conv2d for name in GRAPH_AUTOGRAD_CASES
       if name != "convolution.graph.transposed.second"},
    "convolution.graph.transposed.second": torch.nn.functional.conv_transpose2d,
}


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
    _case(
        "convolution.parameters.output-padding.ignored", "convolution",
        _ordinary_output_padding_forward, _ordinary_output_padding_inputs,
        args=([1, 1], [0, 0], [1, 1], False, [1, 0], 1),
        cpu_reference=_ordinary_output_padding_reference,
        expected_shape=(1, 3, 3, 3), rtol=3e-4, atol=3e-4,
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
) + tuple(
    _case(name, "linear", operation, factory, cpu_reference=operation,
          expected_shape=shape, requires_grad_inputs=True,
          rtol=0.003, atol=0.003,
          declared_shapes=declared_shapes, graph_autograd_case=name)
    for name, operation, factory, shape, declared_shapes in (
        ("matrix.graph.mm.generated", torch.mm,
         lambda **kw: (torch.empty(2, 3), torch.empty(3, 4)), (2, 4), ("2x3", "3x4")),
        ("matrix.graph.addmm.generated", torch.addmm,
         lambda **kw: (torch.empty(4), torch.empty(2, 3), torch.empty(3, 4)),
         (2, 4), ("4", "2x3", "3x4")),
        ("matrix.graph.bmm.generated", torch.bmm,
         lambda **kw: (torch.empty(2, 3, 4), torch.empty(2, 4, 5)),
         (2, 3, 5), ("2x3x4", "2x4x5")),
    )
) + _convolution_evidence_cases() + _transposed_convolution_evidence_cases() + tuple(
    _case(name, "convolution", _GRAPH_FORWARD_OPERATIONS[name],
          (lambda shapes=shapes: lambda **kw: tuple(torch.randn(s, generator=torch.Generator().manual_seed(71 + i), requires_grad=False) for i, s in enumerate(shapes)))(),
          cpu_reference=_GRAPH_CPU_REFERENCES[name], requires_grad_inputs=True,
          expected_shape=output, graph_autograd_case=name)
    for name, shapes, output in (
        ("convolution.graph.grouped.ggI-ggW-ggb", ((1,4,6,7),(6,2,3,2),(6,)), (1,6,3,5)),
        ("convolution.graph.grouped.selected-third", ((1,4,6,7),(6,2,3,2),(6,)), (1,6,3,5)),
        ("convolution.graph.depthwise.selected-third", ((1,2,5,6),(4,1,2,3)), (1,4,4,3)),
        ("convolution.graph.transposed.second", ((1,2,3,3),(2,3,2,2),(3,)), (1,3,7,6)),
        ("convolution.graph.overrideable.first", ((1,2,4,5),(3,2,2,3),(3,)), (1,3,3,3)),
    )
)


def _overrideable_backward_inputs(**kwargs):
    generator = torch.Generator(device="cpu").manual_seed(4711)
    return (torch.randn((1, 3, 3, 3), generator=generator),
            torch.randn((1, 2, 4, 5), generator=generator),
            torch.randn((3, 2, 2, 3), generator=generator))


def _overrideable_backward(grad, value, weight, stride, padding, dilation,
                           transposed, output_padding, groups, output_mask):
    return torch.ops.aten.convolution_backward_overrideable.default(
        grad, value, weight, stride, padding, dilation, transposed,
        output_padding, groups, output_mask)


def _overrideable_backward_cpu(grad, value, weight, stride, padding, dilation,
                               transposed, output_padding, groups, output_mask):
    with torch.backends.mkldnn.flags(enabled=False):
        return torch.ops.aten.convolution_backward.default(
            grad, value, weight, None, stride, padding, dilation, transposed,
            output_padding, groups, output_mask)


ALL_CASES += (_case(
    "convolution.backward-overrideable.bias-present.mask-111", "convolution",
    _overrideable_backward, _overrideable_backward_inputs,
    args=([1, 1], [0, 0], [1, 1], False, [0, 0], 1, [True, True, True]),
    cpu_reference=_overrideable_backward_cpu,
    expected_shapes=((1, 2, 4, 5), (3, 2, 2, 3), (3,)),
    rtol=8e-4, atol=8e-4,
),)


SUPPORTED_CASES = tuple(case for case in ALL_CASES if case.supported)
