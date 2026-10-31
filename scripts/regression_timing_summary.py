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
"""Summary of the alternating-build timing window (results/rtx4090/e4-resize-regression/timings): the E4 program (fkl_fused, fkl_fused_half_pixel) and the E1c program (hand kernels, FKL)
built against the library at 38c4f32, d776713 and c8fcbe8, ROUNDS rounds each.

  python scripts/regression_timing_summary.py results/rtx4090/e4-resize-regression/timings --out results/rtx4090/e4-resize-regression/timing_summary.csv

Per (program, batch, version, tree): the mean of the ROUNDS means (gpu_mean_ms of 100 launches each), the smallest and the largest of the ROUNDS means, and the ratio to the tree 38c4f32 as the median of the
per-round ratios (each round runs the three trees one after the other, so a per-round ratio pairs measurements made within seconds). The hand-written kernels of E1c do not depend on the library: they are
the control of the method (a ratio different from 1 there is noise of the method, not an effect of the library).
"""
import argparse
import csv
import glob
import os
import statistics as stats

TREES = ["p38c4f32", "d776713", "c8fcbe8"]
LABEL = {"p38c4f32": "38c4f32", "d776713": "d776713", "c8fcbe8": "c8fcbe8"}


def read(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def collect(d):
    """{(program, batch, version): {tree: [mean per round]}}"""
    data = {}
    for tree in TREES:
        for run in sorted(glob.glob(os.path.join(d, f"e4_{tree}_r*"))):
            r = int(run.rsplit("_r", 1)[1])
            for row in read(os.path.join(run, "e4_preprocessing.csv")):
                data.setdefault(("E4", int(row["batch"]), row["version"]), {}).setdefault(tree, {})[r] = float(row["gpu_mean_ms"])
        for run in sorted(glob.glob(os.path.join(d, f"e1_{tree}_r*"))):
            r = int(run.rsplit("_r", 1)[1])
            for row in read(os.path.join(run, "e1_preproc.csv")):
                for col, name in (("hand_mean_ms", "hand_batch (control)"), ("hand_generic_mean_ms", "hand_generic (control)"), ("fkl_mean_ms", "fkl")):
                    data.setdefault(("E1c", int(row["batch"]), name), {}).setdefault(tree, {})[r] = float(row[col])
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    data = collect(args.dir)
    rows = []
    for key in sorted(data, key=lambda k: (k[0], k[2], k[1])):
        by = data[key]
        for tree in TREES:
            if tree not in by:
                continue
            vals = [by[tree][r] for r in sorted(by[tree])]
            ratios = [by[tree][r] / by["p38c4f32"][r] for r in sorted(by[tree]) if "p38c4f32" in by and r in by["p38c4f32"]]
            rows.append(dict(program=key[0], batch=key[1], version=key[2], tree=LABEL[tree], rounds=len(vals), mean_ms=stats.mean(vals), min_of_means_ms=min(vals), max_of_means_ms=max(vals),
                             ratio_to_38c4f32=stats.median(ratios) if ratios else float("nan"),
                             ratio_min=min(ratios) if ratios else float("nan"), ratio_max=max(ratios) if ratios else float("nan")))
    hdr = "program batch version                     tree     rounds  mean_ms   min     max     ratio to 38c4f32 (median, min..max of the per-round ratios)"
    print(hdr)
    for r in rows:
        print(f"{r['program']:7s}{r['batch']:5d}  {r['version']:27s}{r['tree']:9s}{r['rounds']:3d}  {r['mean_ms']:.5f} {r['min_of_means_ms']:.5f} {r['max_of_means_ms']:.5f}   {r['ratio_to_38c4f32']:.3f} ({r['ratio_min']:.3f}..{r['ratio_max']:.3f})")
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
