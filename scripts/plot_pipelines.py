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
"""F5, fig:pipelines: the application pipeline (batched resize, crop, normalise, split) against OpenCV-CUDA and against NPP, two panels.

  python scripts/plot_pipelines.py --csv results/rtx4090 [--cpu-opencv DIR] [--cpu-npp DIR] [--npp DIR] [--width 3.4] [--out-dir figures] [--suffix _after45]

  --suffix is appended to the three file names (cputime_r2<suffix>.pdf, ...), so that a re-drawn figure does not overwrite the existing one. With a --cpu-npp / --npp folder whose path contains 'after45' (the re-run after
  FastNPP #45: results/rtx4090/{e0-cpu-time-npp,e0-npp}-after45/run1) the sidecar says that the FastNPP vs NPP series is that run and not the older one.

  cputime_r2.pdf    (a) CPU-time speedup against the batch size (2 to 152): "cvGS vs OpenCV-CUDA" (e0-cpu-time-opencv) and "FastNPP vs NPP" (e0-cpu-time-npp)
  npp_r2.pdf        (b) GPU speedup of FastNPP against NPP (batch 10 to 150): "without CPU overhead" and "with CPU overhead" (the two files of e0-npp)
  pipelines_r2.pdf  both panels in one row, 7.1 x 2.0 in (the figure*); each single panel is 3.4 x 2.0 in

y axes start at 0 and are linear (the ranges are 14x to 420x and 30x to 145x, below the 100x-ratio at which the specification asks for a log axis).
The FastNPP and NPP series must come from the runs made after the FastNPP resize-convention change (PR #45) is merged and the submodule is moved; the
sidecar .txt records the build (the '# fkl' line is the FKL commit, '# repo' the fkl-benchmarks commit) of every CSV it read.
"""
import csv
import glob
import os
import statistics as stats

import plot_common as pc
import matplotlib.pyplot as plt

CPU_OPENCV = "test_cpu_batchresize_x_split3D.csv"
CPU_NPP = "test_NPP_cpu_batchresize_x_split3D.csv"
NPP_WITH = "test_npp_batchresize_x_split3D.csv"
NPP_WITHOUT = "test_npp_batchresize_x_split3D_No_CPU_Overhead.csv"


def load(path):
    _, blocks = pc.read_blocks(path)
    block = blocks[0]
    return pc.column(block, block[0][0]), pc.column(block, "Mean Speedup")


def audit_notes(results, experiments):
    """What the load audit (results/rtx4090/e0-load-audit) records for the runs of these experiments."""
    d = os.path.join(results, "e0-load-audit")
    try:
        with open(os.path.join(d, "window_summary.csv"), newline="") as f:
            windows = [r for r in csv.DictReader(f) if r["experiment"] in experiments]
        with open(os.path.join(d, "overlaps.csv"), newline="") as f:
            overlaps = [r for r in csv.DictReader(f) if r["experiment"] in experiments]
        with open(os.path.join(d, "irregular_points.csv"), newline="") as f:
            irregular = [r for r in csv.DictReader(f) if r["run"] in experiments]
    except OSError:
        return ["load audit (results/rtx4090/e0-load-audit): not found, the runs are not covered by it"]
    out = []
    for w in windows:
        out.append(f"load audit: {w['experiment']} ran {w['start'][:10]} {w['start'][11:19]}-{w['end'][11:19]} ({w['seconds']} s); seconds with a loaded core on the E0 cores / the other cores of its CCD / CCD1 / unpinned: "
                   f"{w['loaded_s_E0_cores']} / {w['loaded_s_CCD0_other_cores']} / {w['loaded_s_CCD1']} / {w['loaded_s_unpinned']}; maximum parallel jobs {w['max_parallel_jobs']}")
    out.append(f"load audit: rows of these experiments in overlaps.csv: {len(overlaps)}, in irregular_points.csv: {len(irregular)}; no row of the CSVs read here is flagged"
               if not overlaps and not irregular else
               f"load audit: rows of these experiments in overlaps.csv: {len(overlaps)}, in irregular_points.csv: {len(irregular)} (see results/rtx4090/e0-load-audit)")
    return out


def host_check_notes(results, p_cv):
    """The frozen-vs-rebuilt A/B of the host-time experiments (results/rtx4090/e0-pin-transfer/host-check), for the OpenCV panel."""
    d = os.path.join(results, "e0-pin-transfer", "host-check")
    summary = os.path.join(d, "host_check_summary.csv")
    if not os.path.exists(summary):
        return ["frozen-vs-rebuilt host-time A/B (pin transfer): not run when this figure was made"]
    with open(summary, newline="") as f:
        r = {row["experiment"]: row for row in csv.DictReader(f)}["cpu-time-opencv"]
    out = [f"frozen-vs-rebuilt host-time A/B of this experiment (results/rtx4090/e0-pin-transfer/host-check): the binary that made the numbers of this figure (fkl-benchmarks 269d229, FKL fb9ac3a) against the rebuild "
           f"against the FKL pin c8fcbe8, {r['rounds']} rounds, alternating order, {r['rows']} batch sizes; rebuilt / frozen mean host time of the cvGS call, per round the median over the batch sizes: median {float(r['fkl_ratio_median']):.3f}, "
           f"smallest {float(r['fkl_ratio_smallest']):.3f}, largest {float(r['fkl_ratio_largest']):.3f}; same ratio of the OpenCV-CUDA calls (the control, which does not depend on the library): median "
           f"{float(r['baseline_ratio_median']):.3f}, smallest {float(r['baseline_ratio_smallest']):.3f}, largest {float(r['baseline_ratio_largest']):.3f}; the host times of this figure hold at c8fcbe8 within the run-to-run spread"]
    runs = [load(p) for p in sorted(glob.glob(os.path.join(d, "cpu_time_opencv_frozen_r*", CPU_OPENCV)))]
    x0, y0 = load(p_cv)
    if runs and all(x == x0 for x, _ in runs):
        med = [stats.median(y[i] for _, y in runs) for i in range(len(x0))]
        ratio = [a / b for a, b in zip(y0, med)]
        lo, hi = ratio.index(min(ratio)), ratio.index(max(ratio))
        out.append(f"the plotted file against the median of the {len(runs)} frozen-binary runs of the A/B (speedup per batch size, plotted / median): median {stats.median(ratio):.3f}, smallest {min(ratio):.3f} at batch {x0[lo]:.0f}, "
                   f"largest {max(ratio):.3f} at batch {x0[hi]:.0f} (plotted {y0[hi]:.1f}x, A/B median {med[hi]:.1f}x, A/B range {min(y[hi] for _, y in runs):.1f}x to {max(y[hi] for _, y in runs):.1f}x); the plotted run is one of the samples, the figure is not redrawn from the A/B")
    return out


def cpu_notes(results, p_cv, p_cn, after45=False):
    _, blocks = pc.read_blocks(p_cv)
    b = blocks[0]
    col, ocv, cvgs = pc.column(b, "Mean Speedup"), pc.column(b, "OpenCV MeanTime"), pc.column(b, "cvGS MeanTime")
    dev = [100.0 * (c / (o / g) - 1.0) for c, o, g in zip(col, ocv, cvgs)]
    out = ["what is measured: host time, not GPU time. The time the CPU spends in the calls that enqueue the work (std::chrono::steady_clock around the OpenCV-CUDA calls of one iteration and around the one cvGS::executeOperations "
           "call of the same iteration, no synchronisation inside the timed region), 100 iterations after a warm-up pass. It measures the launch cost of the GPU pipelines; it is not the CPU back-end of the library",
           f"speedup = host time of the baseline / host time of the fused call: the column 'Mean Speedup', the mean over the 100 iterations of the per-iteration ratio; the ratio of the two mean times is lower by "
           f"{min(dev):.1f} % to {max(dev):.1f} % over the batch sizes",
           "the OpenCV-CUDA side of this experiment is the per-crop loop: for each crop on one stream convertTo, resize, multiply, cvtColor (3 channels), subtract, divide, split = 7 enqueued calls per crop, 7 x batch in total "
           "(benchmark_CPU_OpenCV_vs_cvGS.cu); it is not the stacked 'opencv_batch' formulation of E4, which runs 3 x batch + 4 calls (e4_preprocessing.cu) and is not in this experiment",
           ]
    if after45:
        out.append(f"the FastNPP vs NPP series is RUN 1 of the re-run after the FastNPP resize-convention change (PR #45 merged; FastNPP a1b5e245, FKL c8fcbe8): {p_cn}; window of 2026-11-06 15:30:00 to 15:31:16 "
                   "(results/rtx4090/e9-cross-library/window-console.txt and load.log: load 0.03 before, no other job); it is not part of the A/B above; the cvGS vs OpenCV-CUDA series is the older single run of e0-cpu-time-opencv")
        out += audit_notes(results, ("e0-cpu-time-opencv",))
    else:
        out.append("the FastNPP vs NPP series is from the run made before the FastNPP resize-convention change (PR #45): it is to be re-run after #45 is merged and the submodule is moved, and it is not part of the A/B above")
        out += audit_notes(results, ("e0-cpu-time-opencv", "e0-cpu-time-npp"))
    out += host_check_notes(results, p_cv)
    return out


def panel(ax, curves, ylabel, legend):
    for i, (label, (x, y)) in enumerate(curves):
        ax.plot(x, y, color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=3.2, linewidth=1.0, label=label, clip_on=False)
    ax.set_xlim(0, 160)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 161, 20))
    ax.set_xlabel("Batch size")
    ax.set_ylabel(ylabel)
    ax.legend(loc=legend)


def main():
    ap = pc.arg_parser(__doc__, "pipelines_r2.pdf")
    ap.set_defaults(out=None)
    ap.add_argument("--out-dir", default=pc.FIGURES)
    ap.add_argument("--width", type=float, default=3.4, help="width of each panel in inches")
    ap.add_argument("--cpu-opencv", default=None)
    ap.add_argument("--cpu-npp", default=None)
    ap.add_argument("--npp", default=None)
    ap.add_argument("--suffix", default="", help="appended to the output file names, e.g. _after45")
    args = ap.parse_args()
    d_cv = args.cpu_opencv or os.path.join(args.csv, "e0-cpu-time-opencv")
    d_cn = args.cpu_npp or os.path.join(args.csv, "e0-cpu-time-npp")
    d_np = args.npp or os.path.join(args.csv, "e0-npp")
    p_cv, p_cn = os.path.join(d_cv, CPU_OPENCV), os.path.join(d_cn, CPU_NPP)
    p_without, p_with = os.path.join(d_np, NPP_WITHOUT), os.path.join(d_np, NPP_WITH)

    a = [("cvGS vs OpenCV-CUDA", load(p_cv)), ("FastNPP vs NPP", load(p_cn))]
    b = [("without CPU overhead", load(p_without)), ("with CPU overhead", load(p_with))]
    notes, notes_a = [], []
    for label, curves in (("a", a), ("b", b)):
        for name, (x, y) in curves:
            j, jmin = y.index(max(y)), y.index(min(y))
            lines = [f"({label}) {name}: {len(x)} points, batch {x[0]:.0f} to {x[-1]:.0f}; first {y[0]:.2f}x, maximum {max(y):.2f}x at batch {x[j]:.0f}; last {y[-1]:.2f}x",
                     f"({label}) {name}, for the text: minimum {min(y):.1f}x at batch {x[jmin]:.0f}, maximum {max(y):.1f}x at batch {x[j]:.0f}; at the first batch ({x[0]:.0f}) {y[0]:.1f}x, at the last batch ({x[-1]:.0f}) {y[-1]:.1f}x"]
            notes += lines
            if label == "a":
                notes_a += lines

    after45 = "after45" in (p_cn + p_without + p_with)
    cpu = cpu_notes(args.csv, p_cv, p_cn, after45)
    pc.apply_style()
    fig, ax = plt.subplots(figsize=(args.width, 2.0), constrained_layout=True)
    panel(ax, a, "CPU-time speedup", "upper left")
    pc.save(fig, os.path.join(args.out_dir, f"cputime_r2{args.suffix}.pdf"), [p_cv, p_cn], "plot_pipelines.py", notes_a + cpu)
    fig, ax = plt.subplots(figsize=(args.width, 2.0), constrained_layout=True)
    panel(ax, b, "GPU speedup vs NPP", "lower right")
    pc.save(fig, os.path.join(args.out_dir, f"npp_r2{args.suffix}.pdf"), [p_without, p_with], "plot_pipelines.py", notes)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0), constrained_layout=True)
    panel(axes[0], a, "CPU-time speedup", "upper left")
    panel(axes[1], b, "GPU speedup vs NPP", "lower right")
    pc.save(fig, os.path.join(args.out_dir, f"pipelines_r2{args.suffix}.pdf"), [p_cv, p_cn, p_without, p_with], "plot_pipelines.py", notes + cpu)


if __name__ == "__main__":
    main()
