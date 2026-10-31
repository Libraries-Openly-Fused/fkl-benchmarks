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
"""Summary of the host-side check (results/rtx4090/e0-pin-transfer/host-check): the programs whose number is a host time (the cvGS / FKL column of cpu-time, wrapper-overhead, data-type, and the FK
column of cpu-time-npp) run alternately from the frozen binary and from the rebuild against c8fcbe8, ROUNDS rounds.

  python scripts/host_check_summary.py results/rtx4090/e0-pin-transfer/host-check

Per program and file: for every round the ratio rebuild / frozen of the mean time of the FKL column, row by row (rows = fused operations, or the batch of the experiment), then the median over the rows;
the table gives the median, the smallest and the largest of these per-round medians over the rounds, and the same for the OpenCV / NPP column (the baseline, which does not depend on the library: the
control of the method). A ratio of 1.00 is no change.
"""
import csv
import glob
import os
import statistics as stats
import sys

PROGRAMS = [("cpu-time-opencv", "test_cpu_batchresize_x_split3D.csv", "cvGS MeanTime", "OpenCV MeanTime"),
            ("wrapper-overhead", "test_cvGS_VS_fk_CPU_and_GPU.csv", "cvGS MeanTime", "OpenCV MeanTime"),
            ("wrapper-overhead (CPU enqueue)", "test_cvGS_VS_fk_CPU_and_GPU_CPU_enqueue.csv", "cvGS MeanTime", "OpenCV MeanTime"),
            ("data-type", "test_batchread_x_write3D.csv", "cvGS MeanTime", "OpenCV MeanTime"),
            ("cpu-time-npp", "test_NPP_cpu_batchresize_x_split3D.csv", "FK MeanTime", "NPP MeanTime")]


def read(path):
    with open(path, newline="") as f:
        lines = [l for l in f if not l.startswith("#")]
    rows = list(csv.reader(lines))
    head = [h.strip() for h in rows[0]]
    return head, [r for r in rows[1:] if len(r) == len(head) and r[0].strip()]


def per_round(d, prog_dir, fname, col, rnd):
    h0, r0 = read(os.path.join(d, f"{prog_dir}_frozen_r{rnd}", fname))
    h1, r1 = read(os.path.join(d, f"{prog_dir}_rebuild_r{rnd}", fname))
    i = h0.index(col)
    ratios = []
    for a, b in zip(r0, r1):
        if a[0].strip() == b[0].strip() and float(a[i]) > 0:
            ratios.append(float(b[i]) / float(a[i]))
    return stats.median(ratios), len(ratios)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e0-pin-transfer/host-check"
    out = []
    print(f"{'experiment':34s} {'rows':>4s} {'rounds':>6s}   FKL column: median [smallest-largest]   baseline column (control): median [smallest-largest]")
    for name, fname, fkl_col, base_col in PROGRAMS:
        prog_dir = {"cpu-time-opencv": "cpu_time_opencv", "wrapper-overhead": "wrapper_overhead", "wrapper-overhead (CPU enqueue)": "wrapper_overhead", "data-type": "data_type", "cpu-time-npp": "cpu_time_npp"}[name]
        rounds = sorted(int(p.rsplit("_r", 1)[1]) for p in glob.glob(os.path.join(d, f"{prog_dir}_frozen_r*")))
        fk, base, nrows = [], [], 0
        for r in rounds:
            m, nrows = per_round(d, prog_dir, fname, fkl_col, r)
            fk.append(m)
            base.append(per_round(d, prog_dir, fname, base_col, r)[0])
        out.append((name, nrows, len(rounds), stats.median(fk), min(fk), max(fk), stats.median(base), min(base), max(base)))
        print(f"{name:34s} {nrows:4d} {len(rounds):6d}   {stats.median(fk):.3f} [{min(fk):.3f}-{max(fk):.3f}]                   {stats.median(base):.3f} [{min(base):.3f}-{max(base):.3f}]")
    with open(os.path.join(d, "host_check_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["experiment", "rows", "rounds", "fkl_ratio_median", "fkl_ratio_smallest", "fkl_ratio_largest", "baseline_ratio_median", "baseline_ratio_smallest", "baseline_ratio_largest"])
        for r in out:
            w.writerow([r[0], r[1], r[2]] + [f"{x:.4f}" for x in r[3:]])


if __name__ == "__main__":
    main()
