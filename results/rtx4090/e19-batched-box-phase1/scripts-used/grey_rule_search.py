#!/usr/bin/env python3
"""NPP's grey rounding, from the raw dump of the E19 checks run (nothing is run on the GPU).

  grey_rule_search.py <dump folder> [--out report.txt]

Rebuilds the frame (the LCG of e19_batched_box.cu: seed 12345, bytes 31..24, 23..16, 15..8 of every state, one state per pixel, row by row) and
the 30 crops (crop i at x = 64 + 600 (i mod 6), y = 64 + 400 (i div 6), 256 x 256; channel 0 = R), reads npp_grey.u8 and opencv_cuda_grey.u8
(crop by crop, row by row), and counts, for every candidate rule, the values where the rule differs from NPP's grey image. The candidates:
  float   the weights 0.299f, 0.587f, 0.114f; the three products and two sums in every order, each rounded to float, or with the second and third
          term added by an FMA (one rounding for product and sum); then truncated, + 0.5f and truncated, or rounded to the nearest even
  fixed   (wR R + wG G + wB B + 2^(k-1)) >> k with w = round(weight * 2^k), for k = 8..20, and the same truncated (no 2^(k-1))
Also checks that the program's OpenCV reference (the fixed point with k = 15) gives 0 differences against OpenCV's grey image, as the run did.
For the best rules, the differing values: NPP's value against the exact weighted sum of the float weights (how far the sum lies from a half).
"""
import argparse
import itertools
import os

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("dump")
ap.add_argument("--out", default=None)
args = ap.parse_args()

W, H = 3840, 2160
n = W * H
a = np.full(n, 1664525, dtype=np.uint32)
A = np.cumprod(a)                                   # a^1 .. a^n mod 2^32
powers = np.concatenate(([np.uint32(1)], A[:-1]))  # a^0 .. a^(n-1)
S = np.cumsum(powers, dtype=np.uint32)              # 1 + a + ... + a^(m-1) for m = 1..n
with np.errstate(over="ignore"):
    state = A * np.uint32(12345) + np.uint32(1013904223) * S
frame = np.stack([(state >> 24) & 255, (state >> 16) & 255, (state >> 8) & 255], axis=1).astype(np.uint8).reshape(H, W, 3)
# a check of the rebuild against the plain loop for the first pixels
s = 12345
for p in range(4):
    s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
    assert tuple(frame.reshape(-1, 3)[p]) == ((s >> 24) & 255, (s >> 16) & 255, (s >> 8) & 255)
crops = np.stack([frame[64 + 400 * (i // 6): 64 + 400 * (i // 6) + 256, 64 + 600 * (i % 6): 64 + 600 * (i % 6) + 256] for i in range(30)])
R, G, B = (crops[..., c].reshape(-1).astype(np.int64) for c in range(3))
npp = np.fromfile(os.path.join(args.dump, "npp_grey.u8"), dtype=np.uint8).astype(np.int64)
ocv = np.fromfile(os.path.join(args.dump, "opencv_cuda_grey.u8"), dtype=np.uint8).astype(np.int64)
assert npp.size == R.size == ocv.size == 1966080

lines = []
ocv_ref = (9798 * R + 19235 * G + 3735 * B + (1 << 14)) >> 15
lines.append(f"OpenCV's grey against (9798 R + 19235 G + 3735 B + 2^14) >> 15: {int((ocv_ref != ocv).sum())} differing values (the run: 0)")

f32 = np.float32
w = {"R": f32(0.299), "G": f32(0.587), "B": f32(0.114)}
x = {"R": R.astype(f32), "G": G.astype(f32), "B": B.astype(f32)}


def fma(p, q, r):
    # exact in double: the product of a float weight and an 8-bit integer has at most 32 significant bits, and the sum spans fewer than 53
    return (p.astype(np.float64) * q.astype(np.float64) + r.astype(np.float64)).astype(f32)


def roundings(v):
    return {"truncated": np.trunc(v).astype(np.int64),
            "+ 0.5f, truncated": np.trunc(v + f32(0.5)).astype(np.int64),
            "nearest even": np.rint(v).astype(np.int64)}


results = []
for order in itertools.permutations("RGB"):
    o = "".join(order)
    p = [w[c] * x[c] for c in order]                  # each product rounded to float
    sums = {f"products and sums rounded, {o[0]} + {o[1]} + {o[2]}": (p[0] + p[1]) + p[2],
            f"FMA chain, {o[0]} * w, then FMA {o[1]}, then FMA {o[2]}": fma(w[order[2]], x[order[2]], fma(w[order[1]], x[order[1]], p[0])),
            f"{o[0]} * w + {o[1]} * w rounded, then FMA {o[2]}": fma(w[order[2]], x[order[2]], p[0] + p[1])}
    for sname, v in sums.items():
        for rname, g in roundings(v).items():
            results.append((int((np.minimum(g, 255) != npp).sum()), f"float: {sname}; {rname}"))
for k in range(8, 21):
    wr, wg, wb = (int(round(c * (1 << k))) for c in (0.299, 0.587, 0.114))
    acc = wr * R + wg * G + wb * B
    results.append((int((np.minimum((acc + (1 << (k - 1))) >> k, 255) != npp).sum()), f"fixed: ({wr} R + {wg} G + {wb} B + 2^{k - 1}) >> {k}"))
    results.append((int((np.minimum(acc >> k, 255) != npp).sum()), f"fixed: ({wr} R + {wg} G + {wb} B) >> {k}"))
results.sort()
lines.append(f"{len(results)} candidate rules against NPP's grey image (1,966,080 values); the 12 with the fewest differing values:")
for d, name in results[:12]:
    lines.append(f"  {d:7d}  {name}")
# the differing values of the best float rule and the best fixed rule: where the exact sum of the float weights lies
wr27, wg27, wb27 = (int(np.ldexp(np.float64(f32(c)), 27)) for c in (0.299, 0.587, 0.114))
exact = (wr27 * R + wg27 * G + wb27 * B) / 2.0 ** 27       # the exact weighted sum with the float weights
frac = exact - np.floor(exact)
lines.append("NPP's value against the exact sum of the float weights: rounded up where the fraction is (counts by fraction band)")
up = npp > np.floor(exact)
for lo, hi in ((0.0, 0.4999), (0.4999, 0.5), (0.5, 0.5001), (0.5001, 1.0)):
    m = (frac >= lo) & (frac < hi)
    lines.append(f"  fraction in [{lo}, {hi}): {int(m.sum())} values, {int((m & up).sum())} rounded up")
best_float = next(name for d, name in results if name.startswith("float"))
best_fixed = next(name for d, name in results if name.startswith("fixed"))
lines.append(f"best float rule: {best_float}; best fixed rule: {best_fixed}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
