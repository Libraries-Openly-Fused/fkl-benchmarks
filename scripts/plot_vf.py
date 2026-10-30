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
"""F2, fig:VerticalOnly: vertical fusion without and with horizontal fusion, three panels (each its own PDF, plus a combined preview).

  python scripts/plot_vf.py --csv results/rtx4090 [--mul-add1 DIR] [--mul1 DIR] [--mul-add DIR] [--width 2.3] [--out-dir figures]

  vf_a_r2.pdf     (a) batch 1, 4096 x 2160 uchar -> float: speedup of cvGS against OpenCV-CUDA, series "Mul+Add chain" (e0-vf-mul-add1) and
                  "Mul+Mul chain" (e0-vf-mul1; the instance 'mul1' is Mul then Mul)
  vf_b_r2.pdf     (b) the same chains against OpenCV-CUDA + CUDA Graphs (the *_CUDAGraphs.csv files)
  vfhf_c_r2.pdf   (c) VF + HF, batch 50 of 60 x 120 crops (e0-vf-mul-add): speedup against OpenCV-CUDA and against OpenCV-CUDA + CUDA Graphs
  vfpanels_r2.pdf the three panels side by side at 7.1 in (a preview of the figure*; the LaTeX layout may place them differently)

--mul-add1 / --mul1 / --mul-add are the experiment directories (default: <csv>/e0-vf-mul-add1, e0-vf-mul1, e0-vf-mul-add); the re-run of
e0-vf-mul-add1 (the first run has one disturbed row, at 5,402 operations) is given with --mul-add1. Every point of the files is drawn.
x axis: number of fused operations, linear; y axis from 0, linear (as the old figures).
"""
import os

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

FILES = {
    "mul-add1": ("benchmark_vertical_fusion_loopMulAdd1.csv", "benchmark_vertical_fusion_loopMulAdd1_CUDAGraphs.csv"),
    "mul1": ("benchmark_vertical_fusion_loopMul1.csv", "benchmark_vertical_fusion_loopMul1_CUDAGraphs.csv"),
    "mul-add": ("benchmark_vertical_fusion_loopMulAdd.csv", "benchmark_vertical_fusion_loopMulAdd_CUDAGraphs.csv"),
}


def kfmt(v, _):
    return "0" if v == 0 else f"{v / 1000:g}k"


def load(path):
    _, blocks = pc.read_blocks(path)
    block = blocks[0]
    return pc.column(block, block[0][0]), pc.column(block, "Mean Speedup")


def panel(ax, curves, ylabel, legend="lower right"):
    for i, (label, (x, y)) in enumerate(curves):
        ax.plot(x, y, color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.8, markevery=max(1, len(x) // 8), linewidth=1.0, label=label, clip_on=False)
    ax.set_xlim(0, 20500)
    ax.set_ylim(bottom=0)
    ax.set_xticks([0, 5000, 10000, 15000, 20000])
    ax.xaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt) if max(max(y) for _, (_, y) in curves) > 2000 else FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Number of fused operations")
    ax.set_ylabel(ylabel)
    if legend:
        ax.legend(loc=legend)


def main():
    ap = pc.arg_parser(__doc__, "vf_a_r2.pdf")
    ap.set_defaults(out=None)
    ap.add_argument("--out-dir", default=pc.FIGURES)
    ap.add_argument("--width", type=float, default=2.3, help="width of each panel in inches (2.3 = a third of the figure*, 3.4 = one column)")
    for k in FILES:
        ap.add_argument(f"--{k}", default=None, help=f"experiment directory of e0-vf-{k}")
    args = ap.parse_args()
    dirs = {k: getattr(args, k.replace("-", "_")) or os.path.join(args.csv, f"e0-vf-{k}") for k in FILES}
    paths = {k: [os.path.join(dirs[k], f) for f in FILES[k]] for k in FILES}

    ma, ma_g = load(paths["mul-add1"][0]), load(paths["mul-add1"][1])
    mm, mm_g = load(paths["mul1"][0]), load(paths["mul1"][1])
    hf = load(paths["mul-add"][0])
    hf_g = load(paths["mul-add"][1]) if os.path.exists(paths["mul-add"][1]) else None

    a = [("Mul+Add chain", ma), ("Mul+Mul chain", mm)]
    b = [("Mul+Add chain", ma_g), ("Mul+Mul chain", mm_g)]
    c = [("vs OpenCV-CUDA", hf)] + ([("vs OpenCV-CUDA + CUDA Graphs", hf_g)] if hf_g else [])
    specs = [("vf_a_r2.pdf", a, "Speedup vs\nOpenCV-CUDA", "lower right"), ("vf_b_r2.pdf", b, "Speedup vs OpenCV-CUDA\n+ CUDA Graphs", "lower right"), ("vfhf_c_r2.pdf", c, "Speedup", "center right")]

    pc.apply_style()
    notes = []
    for label, curves in (("a", a), ("b", b), ("c", c)):
        for name, (x, y) in curves:
            j = y.index(max(y))
            notes.append(f"({label}) {name}: {len(x)} points, {x[0]:.0f} to {x[-1]:.0f} operations; maximum {max(y):.2f}x at {x[j]:.0f}; last {y[-1]:.2f}x")
    if not hf_g:
        notes.append("(c) WARNING: the CUDA Graphs file of e0-vf-mul-add does not exist yet; the panel has only the OpenCV-CUDA series")
    for (name, curves, ylabel, legend) in specs:
        fig, ax = plt.subplots(figsize=(args.width, args.width * 0.75), constrained_layout=True)
        panel(ax, curves, ylabel, legend)
        srcs = {"vf_a_r2.pdf": [paths["mul-add1"][0], paths["mul1"][0]], "vf_b_r2.pdf": [paths["mul-add1"][1], paths["mul1"][1]],
                "vfhf_c_r2.pdf": [paths["mul-add"][0]] + ([paths["mul-add"][1]] if hf_g else [])}[name]
        pc.save(fig, os.path.join(args.out_dir, name), srcs, "plot_vf.py", notes)
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 7.1 / 3 * 0.85), constrained_layout=True)
    for ax, (name, curves, ylabel, legend) in zip(axes, specs):
        panel(ax, curves, ylabel, legend)
    allsrc = [p for k in FILES for p in paths[k] if os.path.exists(p)]
    pc.save(fig, os.path.join(args.out_dir, "vfpanels_r2.pdf"), allsrc, "plot_vf.py", notes)


if __name__ == "__main__":
    main()
