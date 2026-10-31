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
"""Summary of the quiet E4 sweeps (results/rtx4090/e4-preprocessing/quiet-run1 ... quiet-runN): median over the runs of every timing, with the run-to-run spread.

  python scripts/e4_summary.py results/rtx4090/e4-preprocessing

Writes e4_summary.csv (one row per experiment, batch, frame width and version: median of gpu_mean_ms over the runs, the smallest and the largest of the runs, spread_pct = (largest - smallest) / median,
the smallest single call of all runs, the median host_enqueue_us and host_launch_us, intermediate_bytes) and e4_speedups.csv (per point: the median over the runs of the per-run ratio of the baseline
to the fused kernel, with the smallest and largest per-run ratio; the baselines of the sampling convention of the fused kernel: fkl_fused (ASYMMETRIC) against opencv_percrop and opencv_batch,
fkl_fused_half_pixel (HALF_PIXEL) against npp).
"""
import csv
import glob
import os
import statistics as stats
import sys

BASELINES = [("opencv_percrop", "fkl_fused"), ("opencv_batch", "fkl_fused"), ("npp", "fkl_fused_half_pixel")]


def rd(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e4-preprocessing"
    runs = sorted(glob.glob(os.path.join(d, "quiet-run*")))
    data = {}
    for run in runs:
        for r in rd(os.path.join(run, "e4_preprocessing.csv")):
            data.setdefault((r["experiment"], int(r["batch"]), int(r["src_w"]), r["version"]), []).append(r)
    summary = []
    for key in sorted(data, key=lambda k: (k[0], k[1], k[2], k[3])):
        rs = data[key]
        means = [float(r["gpu_mean_ms"]) for r in rs]
        med = stats.median(means)
        summary.append({"experiment": key[0], "batch": key[1], "src_w": key[2], "version": key[3], "runs": len(rs), "gpu_mean_ms_median": f"{med:.6f}", "gpu_mean_ms_smallest_run": f"{min(means):.6f}",
                        "gpu_mean_ms_largest_run": f"{max(means):.6f}", "spread_pct": f"{100 * (max(means) - min(means)) / med:.1f}", "gpu_min_ms_all_runs": f"{min(float(r['gpu_min_ms']) for r in rs):.6f}",
                        "host_enqueue_us_median": f"{stats.median(float(r['host_enqueue_us']) for r in rs):.1f}", "host_launch_us_median": f"{stats.median(float(r['host_launch_us']) for r in rs):.1f}",
                        "intermediate_bytes": rs[0]["intermediate_bytes"]})
    with open(os.path.join(d, "e4_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    points = sorted({(k[0], k[1], k[2]) for k in data})
    sp = []
    for exp, batch, w_ in points:
        for base, fused in BASELINES:
            kb, kf = (exp, batch, w_, base), (exp, batch, w_, fused)
            if kb not in data or kf not in data:
                continue
            ratios = [float(b["gpu_mean_ms"]) / float(f_["gpu_mean_ms"]) for b, f_ in zip(data[kb], data[kf])]
            sp.append({"experiment": exp, "batch": batch, "src_w": w_, "baseline": base, "fused": fused, "runs": len(ratios), "speedup_median": f"{stats.median(ratios):.2f}",
                       "speedup_smallest_run": f"{min(ratios):.2f}", "speedup_largest_run": f"{max(ratios):.2f}"})
    with open(os.path.join(d, "e4_speedups.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sp[0]))
        w.writeheader()
        w.writerows(sp)
    print(f"{len(runs)} runs, {len(summary)} rows in e4_summary.csv, {len(sp)} rows in e4_speedups.csv")


if __name__ == "__main__":
    main()
