# fklReference's grey rule against the emulation of fkl_grey_check.py (the FMA chain in float, then np.rint), all 2^24 (R, G, B)
import numpy as np
v = np.arange(1 << 24, dtype=np.int64)
R, G, B = (v >> 16) & 255, (v >> 8) & 255, v & 255
f32 = np.float32
def fma(p, q, r):
    return (p.astype(np.float64) * q.astype(np.float64) + r.astype(np.float64)).astype(f32)
w = lambda c: np.full(v.size, f32(c), dtype=f32)
t = (w(0.587) * G.astype(f32)).astype(f32)
t = fma(w(0.299), R.astype(f32), t)
t = fma(w(0.114), B.astype(f32), t)
emu = np.rint(t).astype(np.int64)
cpp = np.fromfile("fkl_rule_table.u8", dtype=np.uint8).astype(np.int64)
print(f"fklReference's grey rule (std::fma, std::nearbyint) against the emulation, all 2^24 RGB values: {int((cpp != emu).sum())} differ")
