"""Independent scalar F32 oracle for the documented GEMM evaluation policy."""

import math
import struct

import torch


def f32(value):
    value = float(value)
    try:
        return struct.unpack("=f", struct.pack("=f", value))[0]
    except OverflowError:
        return math.copysign(float("inf"), value)


def f32_mul(lhs, rhs):
    return f32(f32(lhs) * f32(rhs))


def f32_add(lhs, rhs):
    return f32(f32(lhs) + f32(rhs))


def addmm_post_alpha(a, b, self, alpha, beta):
    """Round a sequential F32 dot, apply finite alpha once, then beta*self.

    This small oracle deliberately uses explicit product/accumulator rounding.
    It is for stable extreme-value class witnesses, not a bitwise GEMM or FMA
    oracle for arbitrary matrices.
    """
    m, k = len(a), len(a[0])
    n = len(b[0])
    result = []
    for i in range(m):
        row = []
        for j in range(n):
            dot = f32(0.0)
            for inner in range(k):
                product = f32_mul(a[i][inner], b[inner][j])
                dot = f32_add(dot, product)
            scaled_dot = f32_mul(alpha, dot)
            scaled_self = f32(0.0) if beta == 0.0 else f32_mul(beta, self[i][j])
            row.append(f32_add(scaled_self, scaled_dot))
        result.append(row)
    return torch.tensor(result, dtype=torch.float32)


def f32_class(value):
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if torch.isposinf(tensor).all():
        return "+inf"
    if torch.isneginf(tensor).all():
        return "-inf"
    if torch.isnan(tensor).all():
        return "nan"
    if torch.isfinite(tensor).all():
        return "finite"
    return "mixed"
