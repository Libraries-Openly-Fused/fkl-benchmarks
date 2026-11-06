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
"""Portability figures: the RX 9060 overlaid on the RTX 4090, speedup of the fused kernel against the unfused baseline, 3.4 in wide. No new measurement.

  python scripts/plot_portability_speedup.py --what hf      [--out figures/portability_speedup_hf_r2.pdf]
  python scripts/plot_portability_speedup.py --what numinst [--out figures/portability_speedup_numinst_r2.pdf]

Same definition of the speedup for both cards (definition B): time of the baseline in its own loop / mean time of the fused kernel, both means of the same run (for the horizontal fusion the fused kernel is the one
timed in the plain-launch file of the same run); the curve is the median of the runs per point. One colour and marker per card (the RTX 4090 in the first colour of the existing figures), the baseline in the line style.
hf (horizontal fusion, 120 batch sizes 1, 11, ..., 1191, crops of 60 x 120 of a 3840 x 2160 image, the 'NO_CPU_OVERHEAD' files, GPU time only): three baselines, as scripts/plot_hf_b.py: one kernel per crop with plain
  launches (solid), the same launches as one graph of child nodes (dashed; the structure of the 2025 paper), the same launches as one graph from one stream capture (dotted). RTX 4090: results/rtx4090/e0-graph-structure/
  e0-loop-hf-flat/run1..3 (the three runs of the figure hf_b_r2); RX 9060: results/rx9060/fkl-self/horizontal_fusion_with_flat_graph/run1..3 (the same three baselines, three runs of the same program).
numinst (instructions per operation, 100 points M = 1, 6, ..., 496, 500 instructions in all): 'Separated Ops MeanTime' / 'All Ops MeanTime' of one run. RTX 4090: results/rtx4090/e0-numinstructions; RX 9060:
  results/rx9060/fkl-self/numinstructions_x_operation_repeat_22-25 (FKL c8fcbe8, the final pin; the first run, numinstructions_x_operation, differs from it by at most 2.6 % of the speedup, median -0.6 %).
"""
import argparse
import os
import statistics as st

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

HF = {
    "RTX 4090": dict(runs=[f"results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run{i}" for i in (1, 2, 3)], suffix="GPU_NVIDIA", graphs="CUDAGraphs"),
    "RX 9060": dict(runs=[f"results/rx9060/fkl-self/horizontal_fusion_with_flat_graph/run{i}" for i in (1, 2, 3)], suffix="GPU_AMD", graphs="HIPGraphs"),
}
NUMINST = {
    "RTX 4090": "results/rtx4090/e0-numinstructions/testNumInstPerOp_GPU_NVIDIA.csv",
    "RX 9060": "results/rx9060/fkl-self/numinstructions_x_operation_repeat_22-25/testNumInstPerOp_GPU_AMD.csv",
}
STRUCT = [("plain", "-", "one kernel per crop"), ("child", (0, (4, 2)), "graph, child nodes"), ("flat", (0, (1, 1.5)), "graph, one capture")]


def hf_table(path):
    _, blocks = pc.read_blocks(path)
    b = blocks[0]
    return {int(x): (t, f) for x, t, f in zip(pc.column(b, "Batch size"), pc.column(b, "Iterated Batch MeanTime"), pc.column(b, "Fused Batch MeanTime"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--what", choices=["hf", "numinst"], required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    out = a.out or os.path.join(pc.FIGURES, f"portability_speedup_{a.what}_r2.pdf")
    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.1 if a.what == "hf" else 2.2), constrained_layout=True)
    sources, notes = [], []
    cards = list(HF if a.what == "hf" else NUMINST)
    for ci, card in enumerate(cards):
        color, marker = pc.SERIES_COLORS[ci], pc.SERIES_MARKERS[ci]
        if a.what == "hf":
            cfg = HF[card]
            names = {"plain": f"benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_{cfg['suffix']}.csv", "child": f"benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_{cfg['graphs']}_{cfg['suffix']}.csv",
                     "flat": f"benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_{cfg['graphs']}_OneCapturedGraph_{cfg['suffix']}.csv"}
            per_run = []
            for d in cfg["runs"]:
                files = {k: os.path.join(d, v) for k, v in names.items()}
                sources += list(files.values())
                per_run.append({k: hf_table(p) for k, p in files.items()})
            batches = sorted(set.intersection(*[set(r[k]) for r in per_run for k in r]))
            for k, ls, _ in STRUCT:
                y = [st.median(r[k][b][0] / r["plain"][b][1] for r in per_run) for b in batches]
                ax.plot(batches, y, color=color, linestyle=ls, marker=marker if k == "plain" else None, markersize=2.6, markevery=8, linewidth=1.0, clip_on=False)
                j = y.index(max(y))
                spread = max((max(r[k][b][0] / r["plain"][b][1] for r in per_run) - min(r[k][b][0] / r["plain"][b][1] for r in per_run)) / st.median(r[k][b][0] / r["plain"][b][1] for r in per_run) * 100 for b in batches)
                notes.append(f"{card}, {k}: {len(batches)} points (batch {batches[0]} to {batches[-1]}); maximum {max(y):.2f}x at batch {batches[j]}; at batch 101 {y[batches.index(101)]:.2f}x, at batch {batches[-1]} {y[-1]:.2f}x; largest spread of the 3 runs {spread:.1f} % (max - min) / median")
            notes.append(f"{card}: runs {', '.join(cfg['runs'])}")
        else:
            path = NUMINST[card]
            sources.append(path)
            _, blocks = pc.read_blocks(path)
            b = blocks[0]
            x = pc.column(b, "Number of instructions per Operation")
            y = [s / f for s, f in zip(pc.column(b, "Separated Ops MeanTime"), pc.column(b, "All Ops MeanTime"))]
            ax.plot(x, y, color=color, marker=marker, markersize=2.0, linewidth=1.0, clip_on=False)
            notes.append(f"{card}: {len(x)} points, M {x[0]:.0f} to {x[-1]:.0f}; speedup {y[0]:.1f}x at M = {x[0]:.0f}, {y[-1]:.2f}x at M = {x[-1]:.0f}, minimum {min(y):.2f}x at M = {x[y.index(min(y))]:.0f}; one run, {path}")
    if a.what == "hf":
        ax.set_xlim(0, 1200)
        ax.set_ylim(0, 90)
        ax.set_yticks(range(0, 61, 10))
        ax.set_xticks(range(0, 1201, 200))
        ax.set_xlabel("Batch size")
        ax.set_ylabel("Speedup of one HF kernel\nvs one kernel per element")
        card_h = [Line2D([0], [0], color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.6, linewidth=1.0, label=c) for i, c in enumerate(cards)]
        struct_h = [Line2D([0], [0], color=pc.INK, linestyle=ls, linewidth=1.0, label=lab) for _, ls, lab in STRUCT]
        ax.legend(handles=card_h + struct_h, loc="upper center", ncol=2, handlelength=2.2, borderaxespad=0.15, labelspacing=0.2, columnspacing=1.0)
    else:
        ax.set_yscale("log")
        ax.set_xlim(0, 520)
        ax.set_ylim(1, 1000)
        ax.set_xticks([0, 100, 200, 300, 400, 500])
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(False, which="minor")
        ax.set_xlabel("Instructions per operation (M)")
        ax.set_ylabel("Speedup of one fused kernel\nvs " + r"$\lceil 500/M \rceil$" + " kernels")
        ax.legend(handles=[Line2D([0], [0], color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.6, linewidth=1.0, label=c) for i, c in enumerate(cards)], loc="upper right")
    pc.save(fig, out, sources, "plot_portability_speedup.py", ["definition B (the time of the baseline in its own loop over the mean time of the fused kernel of the same run, median of the runs for hf)"] + notes)


if __name__ == "__main__":
    main()
