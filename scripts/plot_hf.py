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
"""F3, fig:ResultsHorizontalFusion: speedup of one horizontally fused kernel against one kernel per batch element, 3.4 in wide.

  python scripts/plot_hf.py --csv results/rtx4090/e0-hf [--out figures/hf_r2.pdf]

Both series are the 'NO_CPU_OVERHEAD' files of the E0 horizontal-fusion run (GPU time only, as the old figure): the baseline is the
launch of one kernel per crop, without and with CUDA Graphs (independent nodes). Column 'Mean Speedup' of each file.
"""
import plot_common as pc
import matplotlib.pyplot as plt

PLAIN = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv"
GRAPHS = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv"


def main():
    ap = pc.arg_parser(__doc__, "hf_r2.pdf")
    args = ap.parse_args()
    paths = [pc.resolve(args.csv, PLAIN), pc.resolve(args.csv, GRAPHS)]
    labels = ["vs kernel per element", "vs kernel per element + CUDA Graphs"]
    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
    notes = []
    for i, (p, label) in enumerate(zip(paths, labels)):
        _, blocks = pc.read_blocks(p)
        x = pc.column(blocks[0], "Batch size")
        y = pc.column(blocks[0], "Mean Speedup")
        ax.plot(x, y, color=pc.SERIES_COLORS[i], marker=pc.SERIES_MARKERS[i], markersize=2.8, markevery=8, linewidth=1.0, label=label, clip_on=False)
        j = y.index(max(y))
        notes.append(f"{label}: {len(x)} points, batch {x[0]:.0f} to {x[-1]:.0f}; maximum {max(y):.2f}x at batch {x[j]:.0f}; at the last batch {y[-1]:.2f}x")
    ax.set_xlim(0, 1200)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 1201, 200))
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Speedup of one HF kernel vs\none kernel per batch element")
    ax.legend(loc="lower right")
    pc.save(fig, args.out, paths, "plot_hf.py", notes)


if __name__ == "__main__":
    main()
