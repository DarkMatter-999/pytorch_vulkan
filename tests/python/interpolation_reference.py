"""CPU-side references for 2-D interpolation on Vulkan (no GPU, no torch dispatch).

* ``scales_value`` / ``area_scale`` / ``dim_taps`` transcribe ATen's UpSample.h
  (PyTorch 2.4.0) in float32.  The index maps for nearest and nearest-exact were
  checked against CPU for every in/out/hint combination of a 57,600-case grid.
* ``forward`` / ``backward`` are the *shader-model oracle*: the exact unfused float32
  expressions and fixed blend / accumulation order that interpolation.comp is
  required to compute.  Vulkan must equal them bitwise; CPU bilinear is only a
  tolerance reference because its vectorised and scalar paths contract differently.
"""
import numpy as np

f32 = np.float32
MAX_DIM = 1 << 23
MODES = ("nearest", "nearest-exact", "bilinear")


def scales_value(hint, size_in, size_out):
    """at::native::compute_scales_value<float>."""
    if hint is not None and hint > 0.0:
        return f32(1.0 / hint)
    return f32(f32(size_in) / f32(size_out))


def area_scale(size_in, size_out, align_corners, hint):
    """at::native::area_pixel_compute_scale<float>."""
    if align_corners:
        return f32(f32(size_in - 1) / f32(size_out - 1)) if size_out > 1 else f32(0)
    return scales_value(hint, size_in, size_out)


def scale_for(mode, size_in, size_out, hint, align_corners=False):
    if mode == "bilinear":
        return area_scale(size_in, size_out, align_corners, hint)
    return scales_value(hint, size_in, size_out)


def dim_taps(mode, size_in, size_out, hint, align_corners=False, fused=False):
    """Per-output-coordinate taps for one dimension.

    Returns ``(i0, i1, l0, l1)``: int64 source indices and float32 weights.  Nearest
    modes return ``i1 == i0``, ``l0 == 1`` and ``l1 == 0``.  ``fused=True`` evaluates
    the bilinear source coordinate with a fused multiply-add (only used to build
    contraction canaries).
    """
    d = np.arange(size_out, dtype=np.int64)
    ones = np.ones(size_out, dtype=f32)
    zeros = np.zeros(size_out, dtype=f32)
    if mode != "bilinear":
        if mode == "nearest" and size_out == size_in:
            return d, d, ones, zeros
        if mode == "nearest" and size_out == 2 * size_in:
            return d >> 1, d >> 1, ones, zeros
        scale = scales_value(hint, size_in, size_out)
        df = d.astype(f32)
        product = (df + f32(0.5)) * scale if mode == "nearest-exact" else df * scale
        idx = np.minimum(np.floor(product), f32(size_in - 1)).astype(np.int64)
        return idx, idx, ones, zeros
    if size_out == size_in:
        return d, d, ones, zeros
    scale = area_scale(size_in, size_out, align_corners, hint)
    df = d.astype(f32)
    if align_corners:
        real = scale * df
    elif fused:
        exact = np.float64(scale) * (df + f32(0.5)).astype(np.float64) - 0.5
        real = np.maximum(exact.astype(f32), f32(0))
    else:
        real = np.maximum(scale * (df + f32(0.5)) - f32(0.5), f32(0))
    i0 = np.minimum(np.floor(real), f32(size_in - 1)).astype(np.int64)
    l1 = np.minimum(np.maximum(real - i0.astype(f32), f32(0)), f32(1))
    i1 = i0 + (i0 < size_in - 1)
    return i0, i1, f32(1) - l1, l1


def forward(x, out_hw, mode, hints=(None, None), align_corners=False):
    """Shader-model forward.  ``x`` is an (N, C, H, W) float32 array."""
    x = np.asarray(x, dtype=f32)
    height, width = x.shape[2:]
    oh, ow = out_hw
    y0, y1, ly0, ly1 = dim_taps(mode, height, oh, hints[0], align_corners)
    x0, x1, lx0, lx1 = dim_taps(mode, width, ow, hints[1], align_corners)
    if mode != "bilinear":
        return np.ascontiguousarray(x[:, :, y0][:, :, :, x0])
    top = lx0 * x[:, :, y0][:, :, :, x0] + lx1 * x[:, :, y0][:, :, :, x1]
    bottom = lx0 * x[:, :, y1][:, :, :, x0] + lx1 * x[:, :, y1][:, :, :, x1]
    lh0 = ly0.reshape(1, 1, oh, 1)
    lh1 = ly1.reshape(1, 1, oh, 1)
    return np.ascontiguousarray(lh0 * top + lh1 * bottom)


def _weights(taps, size_in):
    """Per input index: list of (output coordinate, float32 weight), ascending."""
    i0, i1, l0, l1 = taps
    table = [[] for _ in range(size_in)]
    for d in range(len(i0)):
        for ih in sorted({int(i0[d]), int(i1[d])}):
            w = f32(0)
            if int(i0[d]) == ih:
                w = w + l0[d]
            if int(i1[d]) == ih:
                w = w + l1[d]
            table[ih].append((d, f32(w)))
    return table


def backward(grad, in_hw, mode, hints=(None, None), align_corners=False):
    """Shader-model backward (gather, ascending (oh, ow) accumulation order)."""
    grad = np.asarray(grad, dtype=f32)
    n, c, oh, ow = grad.shape
    height, width = in_hw
    row = _weights(dim_taps(mode, height, oh, hints[0], align_corners), height)
    col = _weights(dim_taps(mode, width, ow, hints[1], align_corners), width)
    result = np.zeros((n, c, height, width), dtype=f32)
    for ih in range(height):
        for iw in range(width):
            acc = np.zeros((n, c), dtype=f32)
            for h, wh in row[ih]:
                for w, ww in col[iw]:
                    if mode == "bilinear":
                        acc = acc + grad[:, :, h, w] * f32(wh * ww)
                    else:
                        acc = acc + grad[:, :, h, w]
            result[:, :, ih, iw] = acc
    return result


def contraction_canaries(max_in=64, max_out=100):
    """Samples ``(size_in, size_out, hint, d)`` where fused and unfused bilinear
    source coordinates give different ramp outputs (so a contracting driver would be
    visible).  Only ``align_corners=False`` has a multiply-add to fuse."""
    found = []
    for size_in in range(2, max_in + 1, 3):
        ramp = np.arange(size_in, dtype=f32)
        for size_out in range(1, max_out + 1, 7):
            if size_out == size_in:
                continue
            for hint in (None, 1.3, 0.7):
                plain = dim_taps("bilinear", size_in, size_out, hint)
                fused = dim_taps("bilinear", size_in, size_out, hint, fused=True)
                out_plain = plain[2] * ramp[plain[0]] + plain[3] * ramp[plain[1]]
                out_fused = fused[2] * ramp[fused[0]] + fused[3] * ramp[fused[1]]
                for d in np.nonzero(out_plain != out_fused)[0]:
                    found.append((size_in, size_out, hint, int(d)))
    return found
