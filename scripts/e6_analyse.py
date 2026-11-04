#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E6 (reduced set): the comparison across the two builds and the compact table.

  python scripts/e6_analyse.py results/rtx4090/e6-fused-vs-unfused results/cuda13.4/e6-raw

Reads <results>/fmad_on/e6_fused_vs_unfused_fmad_on.csv and <results>/fmad_off/e6_fused_vs_unfused_fmad_off.csv (the within-build comparisons written by the program) and the raw bit-pattern
files <raw>/fmad_on/*.bin and <raw>/fmad_off/*.bin. Writes in <results>:
  e6_cross_build.csv   every raw output of the build fmad_on against the same case of fmad_off: identical (memcmp of the bit patterns), differing elements, max ULP (NaN pairs excluded and counted), NaN counts
  e6_table.csv / .md   one row per pipeline case, the four columns of the paper's table: fused vs per-op with -fmad on; fused vs per-op with -fmad off; fused, -fmad on vs off; fused vs fp64 (max ULP
                       against the float rounding of the double result, both builds); every cell is the worst case over the 5 seeds, border and interior separately where the pipeline has a border
Bit identity is memcmp of the bit patterns. The ULP distance is the integer distance of the monotonic mapping of the bit patterns (the same as the program's).
"""
import csv
import os
import sys
from collections import defaultdict

import numpy as np


def mono(u):
    u = u.astype(np.int64)
    return np.where(u & 0x80000000, -(u & 0x7FFFFFFF), u)


def compare(a, b):
    """a, b: uint32 arrays of equal length -> (identical, differing, max_ulp, nan_a, nan_b, nan_mismatch)"""
    fa, fb = a.view(np.float32), b.view(np.float32)
    na, nb = np.isnan(fa), np.isnan(fb)
    both = ~na & ~nb
    ulp = np.abs(mono(a) - mono(b))
    inf_mismatch = (np.isinf(fa) | np.isinf(fb)) & (a != b)
    max_ulp = float("inf") if inf_mismatch.any() else (int(ulp[both].max()) if both.any() else 0)
    # new columns (run 2): max absolute difference, |b| at the element of maximum ULP error, max abs error over the largest |value|, elements with both values below 1e-3 (the cancelling zone),
    # the maximum ULP outside that zone
    fin = both & np.isfinite(fa) & np.isfinite(fb)
    d = np.abs(fa.astype(np.float64) - fb.astype(np.float64))
    max_abs = float(d[fin].max()) if fin.any() else 0.0
    mag = float(np.abs(fb[fin][np.argmax(ulp[fin])])) if fin.any() else 0.0
    max_out = float(max(np.abs(fa[fin]).max(), np.abs(fb[fin]).max())) if fin.any() else 0.0
    small = fin & (np.maximum(np.abs(fa), np.abs(fb)) < 1e-3)
    outside = fin & ~small
    max_ulp_out = int(ulp[outside].max()) if outside.any() else 0
    compare.extra = (max_abs, mag, (max_abs / max_out) if max_out > 0 else 0.0, int(small.sum()), max_ulp_out)
    return bool((a == b).all()), int((a != b).sum()), max_ulp, int(na.sum()), int(nb.sum()), int((na != nb).sum())


def read_within(path):
    with open(path) as f:
        lines = [l for l in f if not l.startswith("#")]
    return list(csv.DictReader(lines))


def main():
    res, raw = sys.argv[1], sys.argv[2]
    names = sorted(f[:-4] for f in os.listdir(os.path.join(raw, "fmad_on")) if f.endswith(".bin"))
    rows = []
    for n in names:
        pa, pb = os.path.join(raw, "fmad_on", n + ".bin"), os.path.join(raw, "fmad_off", n + ".bin")
        if not os.path.exists(pb):
            rows.append([n, "missing in fmad_off", "", "", "", "", ""])
            continue
        a, b = np.fromfile(pa, dtype=np.uint32), np.fromfile(pb, dtype=np.uint32)
        if a.size != b.size:
            rows.append([n, "size differs", a.size, b.size, "", "", ""])
            continue
        ident, nd, mu, nan_a, nan_b, nan_mm = compare(a, b)
        rows.append([n, int(ident), a.size, nd, mu, nan_a, nan_b] + list(compare.extra))
    with open(os.path.join(res, "e6_cross_build.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["case", "identical", "n", "differing", "max_ulp", "nan_fmad_on", "nan_fmad_off", "max_abs", "mag_at_max_ulp", "max_abs_over_max_out", "n_small_lt_1e-3", "max_ulp_outside_small"])
        w.writerows([r + [""] * (12 - len(r)) for r in rows])
    # determinism: the same build, run 1 against run 2 (same seeds): every raw file memcmp-identical?
    if len(sys.argv) > 3:
        det = []
        for tag in ("fmad_on", "fmad_off"):
            for n in names:
                p1, p2 = os.path.join(raw, tag, n + ".bin"), os.path.join(sys.argv[3], tag, n + ".bin")
                if os.path.exists(p2):
                    det.append([tag, n, int(open(p1, "rb").read() == open(p2, "rb").read())])
        with open(os.path.join(res, "e6_determinism_run1_vs_run2.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["build", "case", "identical"])
            w.writerows(det)
        print(f"determinism: {sum(r[2] for r in det)} of {len(det)} raw files identical between run 1 and run 2")
    cross = defaultdict(list)   # (pipeline, conv, path) -> rows
    for r in rows:
        parts = r[0].split("_")
        # p1_pairs1000_seed0_fused / P2_ASYMMETRIC_seed0_perop
        key = (parts[0].upper(), parts[1] if parts[0].upper() != "P1" else parts[1], parts[-1])
        cross[key].append(r)
    within = {t: read_within(os.path.join(res, t, f"e6_fused_vs_unfused_{t}.csv")) for t in ("fmad_on", "fmad_off")}

    def agg(tag, pipeline, case, comparison, region):
        sel = [r for r in within[tag] if r["pipeline"] == pipeline and r["case"] == case and r["comparison"] == comparison and r["region"] == region]
        if not sel:
            return None
        mu = max(float(r["max_ulp"]) for r in sel)
        return {"identical_all": all(r["identical"] == "1" for r in sel), "max_ulp": mu, "differing": sum(int(r["differing"]) for r in sel), "n": sum(int(r["n"]) for r in sel),
                "nan": sum(int(r["nan_a"]) + int(r["nan_b"]) for r in sel), "seeds": len(sel)}

    def cell(a):
        return "n/a" if a is None else ("identical, 0 ULP" if a["identical_all"] else f"{a['differing']:,} of {a['n']:,} differ, max {a['max_ulp']:g} ULP") + (f", NaN {a['nan']}" if a["nan"] else "")

    # run 2: the new columns per case, comparison and region (worst of the 5 seeds; n_small summed over the seeds)
    if "max_abs" in (within["fmad_on"][0] if within["fmad_on"] else {}):
        ex = defaultdict(list)
        for tag in ("fmad_on", "fmad_off"):
            for r in within[tag]:
                if r["pipeline"] in ("P1", "P2", "P3") and r["comparison"] in ("fused_vs_perop", "fused_vs_fp64_rounded"):
                    ex[(tag, r["pipeline"], r["case"], r["comparison"], r["region"])].append(r)
        with open(os.path.join(res, "e6_new_columns.csv"), "w", newline="") as f:
            w = csv.writer(f, lineterminator="\n")
            w.writerow(["build", "pipeline", "case", "comparison", "region", "seeds", "max_ulp", "mag_at_max_ulp_of_the_worst_seed", "max_abs", "max_abs_over_max_out", "n_small_lt_1e-3_all_seeds", "n", "max_ulp_outside_small"])
            for k, v in sorted(ex.items()):
                worst = max(v, key=lambda r: float(r["max_ulp"]))
                w.writerow(list(k) + [len(v), worst["max_ulp"], worst["mag_at_max_ulp"], max(float(r["max_abs"]) for r in v), max(float(r["max_abs_over_max_out"]) for r in v),
                                      sum(int(r["n_small_lt_1e-3"]) for r in v), sum(int(r["n"]) for r in v), max(float(r["max_ulp_outside_small"]) for r in v)])
    out = []
    cases = [("P1", "pairs=1000", ["all"]), ("P2", "ASYMMETRIC", ["interior", "border"]), ("P2", "HALF_PIXEL", ["interior", "border"]), ("P3", "ASYMMETRIC", ["interior", "border"]), ("P3", "HALF_PIXEL", ["interior", "border"])]
    for pipeline, case, regions in cases:
        for region in regions:
            on = agg("fmad_on", pipeline, case, "fused_vs_perop", region)
            off = agg("fmad_off", pipeline, case, "fused_vs_perop", region)
            # fused, fmad on vs off: from the cross-build rows of the fused outputs of this case (P1: all pairs=1000 seeds; P2/P3: the whole tensor, no region split)
            tag = f"pairs1000" if pipeline == "P1" else case
            cb = [r for r in rows if r[0].startswith(("p1_pairs1000_" if pipeline == "P1" else f"{pipeline}_{case}_")) and r[0].endswith("_fused") and r[1] in (0, 1)]
            cbs = {"identical_all": all(r[1] == 1 for r in cb), "max_ulp": max((float(r[4]) for r in cb), default=0), "differing": sum(int(r[3]) for r in cb), "n": sum(int(r[2]) for r in cb),
                   "nan": sum(int(r[5]) + int(r[6]) for r in cb)} if cb else None
            f64_on = agg("fmad_on", pipeline, case, "fused_vs_fp64_rounded", region)
            f64_off = agg("fmad_off", pipeline, case, "fused_vs_fp64_rounded", region)
            out.append([pipeline, case, region, cell(on), cell(off), cell(cbs) if region in ("all", "interior") else "(whole tensor above)",
                        "n/a" if f64_on is None else f"on {f64_on['max_ulp']:g}, off {f64_off['max_ulp']:g}"])
    with open(os.path.join(res, "e6_table.csv"), "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["pipeline", "case", "region", "fused_vs_perop_fmad_on", "fused_vs_perop_fmad_off", "fused_fmad_on_vs_off", "fused_vs_fp64_max_ulp"])
        w.writerows(out)
    md = ["| pipeline | case | region | fused vs per-op, -fmad on | fused vs per-op, -fmad off | fused, -fmad on vs off | fused vs fp64 (max ULP, worst of 5 seeds) |", "|---|---|---|---|---|---|---|"]
    md += ["| " + " | ".join(str(x) for x in r) + " |" for r in out]
    open(os.path.join(res, "e6_table.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
