#!/usr/bin/env python3
"""The E19 crop sweep's crop table (coordinator brief 2026-11-24), checked on the CPU (nothing is run on the GPU)."""
import numpy as np
SX, SY = [0, 320, 0, 320, 192], [0, 0, 200, 200, 100]
W, H, CW, CH = 3840, 2160, 256, 256
def pos(i):
    c, r, k = i % 6, (i // 6) % 5, i // 30
    return 64 + 600 * c + SX[k], 64 + 400 * r + SY[k]
P = [pos(i) for i in range(150)]
e19 = [(64 + 600 * (i % 6), 64 + 400 * (i // 6)) for i in range(30)]
print("crops 0-29 equal E19's:", P[:30] == e19)
print("inside the frame:", all(x >= 0 and y >= 0 and x + CW <= W and y + CH <= H for x, y in P))
print("150 distinct:", len(set(P)) == 150)
print("largest right edge", max(x + CW for x, _ in P), "largest bottom edge", max(y + CH for _, y in P))
def overl(a, b):
    return abs(a[0] - b[0]) < CW and abs(a[1] - b[1]) < CH
for N in [1] + list(range(10, 151, 10)):
    cov = np.zeros((H, W), np.int32)
    for x, y in P[:N]:
        cov[y:y + CH, x:x + CW] += 1
    pairs = sum(overl(P[i], P[j]) for i in range(N) for j in range(i + 1, N))
    alone = sum(not any(overl(P[i], P[j]) for j in range(N) if j != i) for i in range(N))
    print(f"N={N:3d}: overlapping pairs {pairs:3d}; crops overlapping no other {alone:3d}; most crops on one pixel {cov.max()}; pixels covered {int((cov>0).sum())}, by 2+ {int((cov>1).sum())}, by 3 {int((cov>2).sum())}")
# address alignment of each crop's start in a uchar3 frame (3 bytes per pixel), relative to the E19 crop it repeats (i mod 30), for a pitch that is a multiple of 128
for pitch in (11776, 11520):
    d = {m: sorted({((P[i][1] * pitch + P[i][0] * 3) - (P[i % 30][1] * pitch + P[i % 30][0] * 3)) % m for i in range(150)}) for m in (32, 64, 128, 256, 512)}
    print(f"pitch {pitch}: start offset minus E19's, mod 32/64/128/256/512:", d)
for k in range(5):
    print(f"k={k}: SX*3 = {SX[k]*3} bytes; mod 64 {SX[k]*3 % 64}, mod 128 {SX[k]*3 % 128}")
