#!/usr/bin/env python3
"""Task C-NV: compares the outputs of the before/after harness, base (FKL b10fc66) against head (FKL #435 at 48fbf3f), on the CPU.

  compare_harness.py <base output folder> <head output folder> [--out comparison.txt] [--csv comparison.csv]

Each folder holds the .bin files of pr435_image and pr435_attention (raw bytes, row order, see pr435_common.h).
1. Bit for bit, base against head: every file the two folders share, except the four mma_split_*_epilogue.bin, whose
   difference is the bug #435 fixes (2.). For a file that differs: its size on both sides, the number of differing 4-byte
   words and the first differing byte offset.
2. The split path (mma_split_<shape>_*, mma_unsplit_<shape>_*; float32, the oracle float64), the checks of the PR's test at
   48fbf3f (utest_flash_attention_mma.h testMmaSplit, TOL = 3e-3):
   head: split + combine (identity) within TOL of the oracle; split + combine with the epilogue o * 2 + 0.5 and the unsplit
         DPP with it within 2 * TOL of 2 * oracle + 0.5; also the unsplit DPP (identity) within TOL of the oracle, and the
         largest |split - unsplit| with and without the epilogue (reported).
   base: its split output with the epilogue against its split output with the identity (bit for bit: the epilogue
         dropped), and its errors against the oracle (reported; the base's split with the epilogue is expected to miss
         2 * oracle + 0.5).
   The oracle files must be the same at both commits (the harness's host code computes them).
   A NaN anywhere in a checked output fails its check (an element the kernel did not write keeps the 0xff pattern, a NaN).
Exit 0 when every comparison of 1. is equal, the folders hold the same files apart from the head-only mma_unsplit_*, and
every check of 2. that is marked "required" passes.
"""
import argparse
import csv
import hashlib
import math
import os
import struct
import sys

ap = argparse.ArgumentParser()
ap.add_argument("base"); ap.add_argument("head")
ap.add_argument("--out"); ap.add_argument("--csv")
args = ap.parse_args()
TOL = 3e-3


def files(d):
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".bin"))


def raw(d, name):
    with open(os.path.join(d, name + ".bin"), "rb") as fh:
        return fh.read()


def floats(b, fmt="f"):
    n = len(b) // struct.calcsize(fmt)
    return struct.unpack(f"<{n}{fmt}", b)


def maxerr(xs, ref, scale=1.0, offset=0.0):
    if len(xs) != len(ref):
        return float("inf")
    e = 0.0
    for x, r in zip(xs, ref):
        if math.isnan(x):
            return float("nan")
        e = max(e, abs(x - (scale * r + offset)))
    return e


B, H = set(files(args.base)), set(files(args.head))
split_tags = sorted({n[len("mma_split_"):-len("_oracle_f64")] for n in B | H if n.startswith("mma_split_") and n.endswith("_oracle_f64")})
bug_files = {f"mma_split_{t}_epilogue" for t in split_tags}
head_only_expected = {f"mma_unsplit_{t}_{e}" for t in split_tags for e in ("identity", "epilogue")}
lines = [f"base: {args.base}", f"head: {args.head}",
         f"files: base {len(B)}, head {len(H)}; only in base: {sorted(B - H)}; only in head: {sorted(H - B)}"]
ok = (B - H) == set() and (H - B) == head_only_expected
if not ok:
    lines.append(f"FILE SETS: expected only in head exactly {sorted(head_only_expected)}")
rows = []
lines.append("")
lines.append("1. bit for bit, base against head (every shared file except the mma_split_*_epilogue files):")
n_eq = n_diff = 0
for name in sorted(B & H):
    if name in bug_files:
        continue
    a, b = raw(args.base, name), raw(args.head, name)
    sa, sb = hashlib.sha256(a).hexdigest(), hashlib.sha256(b).hexdigest()
    if a == b:
        n_eq += 1
        status, detail = "equal", ""
    else:
        n_diff += 1
        ok = False
        words = sum(1 for i in range(0, min(len(a), len(b)) - 3, 4) if a[i:i + 4] != b[i:i + 4])
        first = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
        status, detail = "DIFFERENT", f"sizes {len(a)} / {len(b)} B, {words} differing 4-byte words, first differing byte {first}"
    rows.append([name, "bitwise", status, len(a), len(b), sa, sb, detail])
    lines.append(f"  {status:9s} {name}  {len(a)} B  {sa[:16]}" + (f"  {sb[:16]}  {detail}" if detail else ""))
lines.append(f"  -> {n_eq} equal, {n_diff} different")

lines.append("")
lines.append(f"2. the split path (TOL = {TOL:g}, as in utest_flash_attention_mma.h at 48fbf3f):")


def check(label, err, bound, required):
    global ok
    passed = (not math.isnan(err)) and err <= bound
    if required and not passed:
        ok = False
    verdict = ("PASS" if passed else "FAIL") if required else ("within" if passed else "outside")
    lines.append(f"    {verdict:7s} {label}: max error {err:.3e} (bound {bound:g}{'' if required else ', reported only'})")
    rows.append([label, "tolerance", verdict, "", "", "", "", f"max error {err:.6e}, bound {bound:g}"])


for t in split_tags:
    lines.append(f"  shape {t}:")
    ob, oh = raw(args.base, f"mma_split_{t}_oracle_f64"), raw(args.head, f"mma_split_{t}_oracle_f64")
    same = ob == oh
    ok = ok and same
    lines.append(f"    {'PASS' if same else 'FAIL':7s} the oracle is the same at both commits ({len(oh) // 8} values)")
    orc = floats(oh, "d")
    hs_id = floats(raw(args.head, f"mma_split_{t}_identity"))
    hs_ep = floats(raw(args.head, f"mma_split_{t}_epilogue"))
    hu_id = floats(raw(args.head, f"mma_unsplit_{t}_identity"))
    hu_ep = floats(raw(args.head, f"mma_unsplit_{t}_epilogue"))
    check("head split + combine, identity, vs oracle", maxerr(hs_id, orc), TOL, True)
    check("head split + combine, epilogue o*2+0.5, vs 2*oracle+0.5", maxerr(hs_ep, orc, 2.0, 0.5), 2 * TOL, True)
    check("head unsplit, epilogue o*2+0.5, vs 2*oracle+0.5", maxerr(hu_ep, orc, 2.0, 0.5), 2 * TOL, True)
    check("head unsplit, identity, vs oracle", maxerr(hu_id, orc), TOL, True)
    lines.append(f"    reported max |head split - head unsplit|: identity {maxerr(hs_id, hu_id):.3e}, epilogue {maxerr(hs_ep, hu_ep):.3e}")
    bs_id_raw, bs_ep_raw = raw(args.base, f"mma_split_{t}_identity"), raw(args.base, f"mma_split_{t}_epilogue")
    bs_id, bs_ep = floats(bs_id_raw), floats(bs_ep_raw)
    lines.append(f"    {'yes' if bs_ep_raw == bs_id_raw else 'NO':7s} base split with the epilogue o*2+0.5 is bit for bit its split with the identity "
                 f"(the epilogue dropped; reported)")
    rows.append([f"mma_split_{t}: base epilogue == base identity", "bitwise", "equal" if bs_ep_raw == bs_id_raw else "different",
                 len(bs_ep_raw), len(bs_id_raw), "", "", "the bug: the base's split path drops the epilogue"])
    check("base split, identity, vs oracle", maxerr(bs_id, orc), TOL, False)
    check("base split, epilogue o*2+0.5, vs 2*oracle+0.5", maxerr(bs_ep, orc, 2.0, 0.5), 2 * TOL, False)
    he = raw(args.head, f"mma_split_{t}_epilogue")
    lines.append(f"    reported: base and head mma_split_{t}_epilogue are {'equal' if he == bs_ep_raw else 'different'} "
                 f"(max |base - head| {maxerr(bs_ep, floats(he)):.3e})")
lines.append("")
lines.append(f"RESULT: {'PASS' if ok else 'FAIL'}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "kind", "result", "base_bytes", "head_bytes", "base_sha256", "head_sha256", "detail"])
        w.writerows(rows)
sys.exit(0 if ok else 1)
