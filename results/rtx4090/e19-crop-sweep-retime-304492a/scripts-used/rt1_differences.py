#!/usr/bin/env python3
"""RT-1: e19_crop_sweep_differences.csv, the largest difference of fkl_batch's tensor from each baseline per N, from the "between" rows that
e19_crop_sweep_cu writes in every run's checks CSV (CPU only: reads files).

  rt1_differences.py <window folder> <sweep>... [--gpu NAME] --out <e19_crop_sweep_differences.csv> [--report report.txt]

The rows are those of version fkl_batch, compared_with opencv_cuda or npp, steps float (output units) and float_grey_levels (x 57.375), in
<window folder>/<sweep>/N<NNN>/e19_batched_box_checks.csv. The program computes them from the tensors in memory: values, the values that
differ (not equal) and the largest absolute difference. Every sweep given must hold all 16 N, and at each N the rows must be the same in every
sweep (the program is deterministic on these tensors). The output has one row per N and baseline:
  gpu,crops,baseline,values,values_not_equal,max_abs_diff_output_units,max_abs_diff_grey_levels
with the numbers as the program wrote them.
"""
import argparse
import csv
import os

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("sweeps", nargs="+"); ap.add_argument("--gpu", default="NVIDIA GeForce RTX 4090")
ap.add_argument("--out", required=True); ap.add_argument("--report")
a = ap.parse_args()
CROPS = [1] + list(range(10, 151, 10))
BASE = ("opencv_cuda", "npp")


def between(sweep, n):
    p = os.path.join(a.folder, sweep, f"N{n:03d}", "e19_batched_box_checks.csv")
    lines = open(p).read().splitlines()
    if any(l.startswith("# INCOMPLETE") for l in lines):
        raise SystemExit(f"{p}: INCOMPLETE")
    got = {}
    for r in csv.DictReader(l for l in lines if not l.startswith("#")):
        if r["version"] == "fkl_batch" and r["compared_with"] in BASE and r["step"] in ("float", "float_grey_levels"):
            if int(r["crops"]) != n:
                raise SystemExit(f"{p}: crops {r['crops']} is not {n}")
            if r["non_finite"] not in ("0", ""):
                raise SystemExit(f"{p}: non-finite values in {r['step']} vs {r['compared_with']}")
            got[(r["compared_with"], r["step"])] = (r["values"], r["differing_values"], r["max_abs_difference"])
    if len(got) != 4:
        raise SystemExit(f"{p}: {len(got)} of the 4 between rows")
    return got


out, report = [], [f"e19_crop_sweep_differences from {a.folder}, sweeps {' '.join(a.sweeps)}"]
for n in CROPS:
    per = [between(s, n) for s in a.sweeps]
    same = all(p == per[0] for p in per)
    if not same:
        raise SystemExit(f"N {n}: the between rows differ between the sweeps")
    g = per[0]
    for b in BASE:
        vals, diff, units = g[(b, "float")]
        v2, d2, grey = g[(b, "float_grey_levels")]
        if (vals, diff) != (v2, d2):
            raise SystemExit(f"N {n} {b}: float and float_grey_levels count different values")
        out.append([a.gpu, n, b, vals, diff, units, grey])
    report.append(f"  N {n:3}: the 4 between rows are the same in all {len(a.sweeps)} sweeps; vs opencv_cuda {g[('opencv_cuda', 'float')][1]} of "
                  f"{g[('opencv_cuda', 'float')][0]} differ, max {g[('opencv_cuda', 'float_grey_levels')][2]} grey levels; vs npp "
                  f"{g[('npp', 'float')][1]} differ, max {g[('npp', 'float_grey_levels')][2]} grey levels")
with open(a.out, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["gpu", "crops", "baseline", "values", "values_not_equal", "max_abs_diff_output_units", "max_abs_diff_grey_levels"])
    w.writerows(out)
text = "\n".join(report)
print(text)
if a.report:
    open(a.report, "w").write(text + "\n")
