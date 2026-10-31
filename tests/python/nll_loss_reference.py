"""NLL / cross-entropy reference model for the Vulkan nll_loss kernels.

Two roles:

* CPU-contract helpers (``cpu_forward``/``cpu_backward``) expressed with the exact
  float32 operation order measured on PyTorch 2.4.0 CPU (spec: ``bits.json``).
* ``reduce_oracle``: a numpy model of the Vulkan reduction (one 256-wide workgroup,
  tiles of 2048 = 256 x 8, fixed binary tree, tile sums re-reduced).  The device must
  match it bitwise for ``sum`` and ``total_weight``; ``mean`` is the oracle quotient
  within ``MEAN_ULPS`` float32 ulp.

Tolerance constants are frozen from the Task-1 measurement
(``tmp/nll-cross-entropy-20261031/tolerance2.json``, units of u * sum|terms|).
"""
import numpy as np

F32 = np.float32
U = 2.0 ** -24

WORKGROUP = 256
CHAIN = 8
TILE = WORKGROUP * CHAIN
MAX_ELEMENTS = 1 << 22  # at most 2048 tile sums fit one level-1 tile

# Measured maxima over the frozen seed grid (u * S_abs units); kept beside the bounds.
MEASURED = {"oracle_vs_float64": 2.40, "cpu_vs_float64": 3.85, "oracle_vs_cpu": 4.46}
ORACLE_DESIGN_BOUND = 4.0   # oracle error vs float64
PARITY_BOUND = 8.0          # |Vulkan - CPU| for sum and total_weight
MEAN_ULPS = 3               # Vulkan mean vs oracle quotient
BACKWARD_MEAN_ULPS = 4      # Vulkan backward-mean vs CPU
assert MEASURED["oracle_vs_cpu"] < PARITY_BOUND
assert MEASURED["oracle_vs_float64"] < ORACLE_DESIGN_BOUND


def _tree(partials):
    acc = partials.copy()
    stride = WORKGROUP // 2
    while stride >= 1:
        acc[..., :stride] = (acc[..., :stride] + acc[..., stride:2 * stride]).astype(F32)
        stride //= 2
    return acc[..., 0]


def _level(values):
    n = len(values)
    tiles = (n + TILE - 1) // TILE
    padded = np.concatenate([values, np.zeros(tiles * TILE - n, F32)])
    grid = padded.reshape(tiles, CHAIN, WORKGROUP)
    acc = np.zeros((tiles, WORKGROUP), F32)
    for j in range(CHAIN):
        acc = (acc + grid[:, j, :]).astype(F32)
    return _tree(acc).copy()


def reduce_oracle(terms):
    """Fixed-order float32 sum of ``terms`` exactly as the shader orders it."""
    values = np.asarray(terms, dtype=F32).reshape(-1)
    if values.size == 0:
        return F32(0.0)
    if values.size > MAX_ELEMENTS:
        raise ValueError("reduction exceeds the supported element count")
    while True:
        values = _level(values)
        if values.size == 1:
            return values[0]


def terms_and_weights(log_probs, target, weight, ignore_index):
    """Per-row loss terms (float32) and per-row weights; ignored rows are zero."""
    lp = np.asarray(log_probs, dtype=F32)
    if lp.ndim == 1:
        lp = lp[None, :]
        target = np.asarray(target).reshape(1)
    target = np.asarray(target, dtype=np.int64)
    keep = target != np.int64(ignore_index)
    safe = np.where(keep, target, 0)
    picked = lp[np.arange(len(target)), safe]
    if weight is None:
        w = np.ones(len(target), F32)
        terms = (-picked).astype(F32)
    else:
        w = np.asarray(weight, dtype=F32)[safe]
        terms = ((-picked) * w).astype(F32)
    terms = np.where(keep, terms, F32(0.0)).astype(F32)
    w = np.where(keep, w, F32(0.0)).astype(F32)
    return terms, w, keep


def oracle_forward(log_probs, target, weight, reduction, ignore_index):
    """(output, total_weight) as numpy float32 using the device reduction order."""
    terms, w, keep = terms_and_weights(log_probs, target, weight, ignore_index)
    if reduction == 0:
        return terms, F32(0.0)
    # Ignored rows stay in place as +0.0 terms: the device adds them too, which fixes the
    # association of the remaining terms.
    total_terms = reduce_oracle(terms)
    total_weight = reduce_oracle(w)
    if reduction == 2:
        return total_terms, total_weight
    with np.errstate(all="ignore"):
        return F32(total_terms / total_weight), total_weight


def oracle_backward(grad_output, shape, target, weight, reduction, ignore_index,
                    total_weight):
    """grad_input as numpy float32 in the CPU-matching operation order."""
    batched = len(shape) == 2
    rows, classes = (shape if batched else (1, shape[0]))
    target = np.asarray(target, dtype=np.int64).reshape(rows)
    grad = np.zeros((rows, classes), F32)
    g = np.asarray(grad_output, dtype=F32).reshape(-1)
    for n in range(rows):
        t = int(target[n])
        if t == ignore_index:
            continue
        w = F32(1.0) if weight is None else F32(np.asarray(weight, F32)[t])
        if reduction == 1:
            with np.errstate(all="ignore"):
                q = F32(g[0] / F32(total_weight))
            grad[n, t] = (-w) * q
        else:
            gv = g[0] if reduction == 2 else g[n]
            grad[n, t] = (-w) * gv
    return grad if batched else grad[0]


def s_abs(log_probs, target, weight, ignore_index):
    terms, w, keep = terms_and_weights(log_probs, target, weight, ignore_index)
    return (float(np.abs(terms.astype(np.float64)).sum()),
            float(np.abs(w.astype(np.float64)).sum()))


def within_parity(actual, cpu, sabs):
    """|V - C| <= PARITY_BOUND * u * S_abs (NaN/inf must match exactly)."""
    actual, cpu = float(actual), float(cpu)
    if not np.isfinite(cpu) or not np.isfinite(actual):
        return (np.isnan(cpu) and np.isnan(actual)) or actual == cpu
    return abs(actual - cpu) <= PARITY_BOUND * U * sabs


def ulp_distance(a, b):
    a = np.asarray(a, dtype=F32).view(np.int32).astype(np.int64)
    b = np.asarray(b, dtype=F32).view(np.int32).astype(np.int64)
    a = np.where(a < 0, np.int64(-2 ** 31) - a, a)
    b = np.where(b < 0, np.int64(-2 ** 31) - b, b)
    return np.abs(a - b)


def lowest_invalid(target, classes, ignore_index):
    """(index, value) of the lowest target that is neither ignored nor in [0, C)."""
    t = np.asarray(target, dtype=np.int64).reshape(-1)
    bad = np.nonzero((t != np.int64(ignore_index)) & ((t < 0) | (t >= classes)))[0]
    return (int(bad[0]), int(t[bad[0]])) if bad.size else None
