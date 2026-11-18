#!/usr/bin/env python3
"""fkl_loop's tensor from the raw dump of the window e19-fkl-checks against references that differ only in FKL's grey value (nothing is run on
the GPU).

  fkl_grey_check.py <dump folder> [--out report.txt]

Rebuilds the frame and the 30 crops as grey_rule_search.py does, and reads fkl_loop_tensor.f32 (crop by crop, row by row). Every reference
computes the box and the conversion as the program's fklReference does (in double: the float coefficient 1/9 times the exact sum of the 9 grey
values of the 3 x 3 window, border REPLICATE at the crop's edges, then (v / 255 - 0.45) / 0.225) from one of these grey values:
  exact float weights    0.299f R + 0.587f G + 0.114f B exactly (the program's reference: fk::RGB2Gray with a float output, as written)
  FMA chain              t = 0.587f G; t = fma(0.299f, R, t); t = fma(0.114f, B, t), in float (the chain of the kernel's SASS)
  FMA chain, rounded     the FMA chain rounded to the nearest integer, ties to even (__float2int_rn), as a float
  exact, rounded         the exact weighted sum rounded to the nearest integer, ties to even (as FKL's host branch, std::nearbyint)
For each: the largest absolute difference of the tensor, and how many values are farther than 1e-5.
"""
import argparse
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--out", default=None)
args = ap.parse_args()

W, H = 3840, 2160
n = W * H
A = np.cumprod(np.full(n, 1664525, dtype=np.uint32))
powers = np.concatenate(([np.uint32(1)], A[:-1]))
S = np.cumsum(powers, dtype=np.uint32)
with np.errstate(over="ignore"):
    state = A * np.uint32(12345) + np.uint32(1013904223) * S
frame = np.stack([(state >> 24) & 255, (state >> 16) & 255, (state >> 8) & 255], axis=1).astype(np.uint8).reshape(H, W, 3)
crops = np.stack([frame[64 + 400 * (i // 6): 64 + 400 * (i // 6) + 256, 64 + 600 * (i % 6): 64 + 600 * (i % 6) + 256] for i in range(30)])
R, G, B = (crops[..., c].astype(np.int64) for c in range(3))   # 30 x 256 x 256
fkl = np.fromfile(os.path.join(args.dump, "fkl_loop_tensor.f32"), dtype=np.float32).reshape(30, 256, 256).astype(np.float64)

f32 = np.float32
exact = np.float64(f32(0.299)) * R + np.float64(f32(0.587)) * G + np.float64(f32(0.114)) * B


def fma(p, q, r):
    return (p.astype(np.float64) * q.astype(np.float64) + r.astype(np.float64)).astype(f32)


t = (f32(0.587) * G.astype(f32)).astype(f32)
t = fma(np.full(t.shape, f32(0.299), dtype=f32), R.astype(f32), t)
t = fma(np.full(t.shape, f32(0.114), dtype=f32), B.astype(f32), t)
greys = {"exact float weights (the program's reference)": exact,
         "FMA chain (G, R, B) in float": t.astype(np.float64),
         "FMA chain, rounded to the nearest integer (ties to even)": np.rint(t).astype(np.float64),
         "exact float weights, rounded to the nearest integer (ties to even)": np.rint(exact)}
coefficient = np.float64(f32(1.0) / f32(9.0))
lines = [f"grey values where the rounded FMA chain and the rounded exact sum differ: {int((np.rint(t) != np.rint(exact)).sum())} of {t.size}"]
for name, g in greys.items():
    p = np.pad(g, ((0, 0), (1, 1), (1, 1)), mode="edge")
    s = sum(p[:, 1 + dy: 1 + dy + 256, 1 + dx: 1 + dx + 256] for dy in (-1, 0, 1) for dx in (-1, 0, 1))
    ref = (coefficient * s / 255.0 - 0.45) / 0.225
    d = np.abs(fkl - ref)
    lines.append(f"{name}: largest difference {d.max():.6g}, beyond 1e-5: {int((d > 1e-5).sum())} of {d.size}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
