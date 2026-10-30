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

  python scripts/plot_pipelines.py --csv results/rtx4090 [--cpu-opencv DIR] [--cpu-npp DIR] [--npp DIR] [--width 3.4] [--out-dir figures]

  cputime_r2.pdf    (a) CPU-time speedup against the batch size (2 to 152): "cvGS vs OpenCV-CUDA" (e0-cpu-time-opencv) and "FastNPP vs NPP" (e0-cpu-time-npp)
  npp_r2.pdf        (b) GPU speedup of FastNPP against NPP (batch 10 to 150): "without CPU overhead" and "with CPU overhead" (the two files of e0-npp)
  pipelines_r2.pdf  both panels in one row, 7.1 x 2.0 in (the figure*); each single panel is 3.4 x 2.0 in

y axes start at 0 and are linear (the ranges are 14x to 420x and 30x to 145x, below the 100x-ratio at which the specification asks for a log axis).
The FastNPP and NPP series must come from the runs made after the FastNPP resize-convention change (PR #45) is merged and the submodule is moved; the
sidecar .txt records the build (the '# fkl' line is the FKL commit, '# repo' the fkl-benchmarks commit) of every CSV it read.
"""
import os

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
    args = ap.parse_args()
    d_cv = args.cpu_opencv or os.path.join(args.csv, "e0-cpu-time-opencv")
    d_cn = args.cpu_npp or os.path.join(args.csv, "e0-cpu-time-npp")
    d_np = args.npp or os.path.join(args.csv, "e0-npp")
    p_cv, p_cn = os.path.join(d_cv, CPU_OPENCV), os.path.join(d_cn, CPU_NPP)
    p_without, p_with = os.path.join(d_np, NPP_WITHOUT), os.path.join(d_np, NPP_WITH)

    a = [("cvGS vs OpenCV-CUDA", load(p_cv)), ("FastNPP vs NPP", load(p_cn))]
    b = [("without CPU overhead", load(p_without)), ("with CPU overhead", load(p_with))]
    notes = []
    for label, curves in (("a", a), ("b", b)):
        for name, (x, y) in curves:
            j = y.index(max(y))
            notes.append(f"({label}) {name}: {len(x)} points, batch {x[0]:.0f} to {x[-1]:.0f}; first {y[0]:.2f}x, maximum {max(y):.2f}x at batch {x[j]:.0f}; last {y[-1]:.2f}x")

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(args.width, 2.0), constrained_layout=True)
    panel(ax, a, "CPU-time speedup", "upper left")
    pc.save(fig, os.path.join(args.out_dir, "cputime_r2.pdf"), [p_cv, p_cn], "plot_pipelines.py", notes)
    fig, ax = plt.subplots(figsize=(args.width, 2.0), constrained_layout=True)
    panel(ax, b, "GPU speedup of FastNPP vs NPP", "lower right")
    pc.save(fig, os.path.join(args.out_dir, "npp_r2.pdf"), [p_without, p_with], "plot_pipelines.py", notes)
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0), constrained_layout=True)
    panel(axes[0], a, "CPU-time speedup", "upper left")
    panel(axes[1], b, "GPU speedup of FastNPP vs NPP", "lower right")
    pc.save(fig, os.path.join(args.out_dir, "pipelines_r2.pdf"), [p_cv, p_cn, p_without, p_with], "plot_pipelines.py", notes)


if __name__ == "__main__":
    main()
