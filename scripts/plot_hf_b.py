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
"""F3, fig:ResultsHorizontalFusion with the three structures of the baseline and the definition B of the speedup, 3.4 in wide.

  python scripts/plot_hf_b.py --runs results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run1 .../run2 .../run3 [--out figures/hf_b_r2.pdf]

Three runs of benchmark_horizontal_fusion_cu (E0 loop, GPU time only, the 'NO_CPU_OVERHEAD' files). The baseline is one kernel per crop, launched as plain launches, as one CUDA graph of child nodes (the
structure of the 2025 paper) or as one CUDA graph from one stream capture (the fastest structure). Definition B: the speedup of a baseline is its mean time in its own loop over the mean time of the fused
kernel in the plain-launch loop of the same run (benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv); the curve is the median of the three runs per batch size.
"""
import argparse
import os
import statistics as st

import plot_common as pc
import matplotlib.pyplot as plt

PLAIN = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv"
CHILD = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv"
FLAT = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_OneCapturedGraph_GPU_NVIDIA.csv"


def table(path):
    _, blocks = pc.read_blocks(path)
    block = blocks[0]
    batch = pc.column(block, "Batch size")
    return {int(b): (t, f) for b, t, f in zip(batch, pc.column(block, "Iterated Batch MeanTime"), pc.column(block, "Fused Batch MeanTime"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=os.path.join(pc.FIGURES, "hf_b_r2.pdf"))
    args = ap.parse_args()
    sources, per_run = [], []
    for d in args.runs:
        files = {k: os.path.join(d, v) for k, v in (("plain", PLAIN), ("child", CHILD), ("flat", FLAT))}
        sources += list(files.values())
        per_run.append({k: table(p) for k, p in files.items()})
    batches = sorted(set.intersection(*[set(r[k]) for r in per_run for k in r]))
    labels = {"plain": "vs kernel per element", "child": "vs CUDA Graph, child nodes", "flat": "vs CUDA Graph, one capture"}
    series = {}
    for k in ("plain", "child", "flat"):
        series[k] = [st.median(r[k][b][0] / r["plain"][b][1] for r in per_run) for b in batches]
    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
    notes = ["definition B: speedup = mean time of the baseline in its own loop / mean time of the fused kernel in the plain-launch loop of the same run; the curve is the median of the runs per batch size",
             f"runs read: {', '.join(args.runs)} ({len(per_run)} runs, 'Iterated Batch MeanTime' of the three files and 'Fused Batch MeanTime' of {PLAIN})"]
    for i, k in enumerate(("plain", "child", "flat")):
        y = series[k]
        ax.plot(batches, y, color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.8, markevery=8, linewidth=1.0, label=labels[k], clip_on=False)
        j = y.index(max(y))
        lo = min(r[k][batches[j]][0] / r["plain"][batches[j]][1] for r in per_run)
        hi = max(r[k][batches[j]][0] / r["plain"][batches[j]][1] for r in per_run)
        notes.append(f"{labels[k]}: {len(batches)} points, batch {batches[0]} to {batches[-1]}; maximum {max(y):.2f}x at batch {batches[j]} (runs {lo:.2f} to {hi:.2f}); at the last batch {y[-1]:.2f}x; median over the batches {st.median(y):.2f}x")
    ax.set_xlim(0, 1200)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 1201, 200))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup of one HF kernel\nvs one kernel per batch element")
    ax.legend(loc="lower right")
    pc.save(fig, args.out, sources, "plot_hf_b.py", notes)


if __name__ == "__main__":
    main()
