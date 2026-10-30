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
"""F7, fig:GPUSize: maximum speedup of VF + HF (batch 50) against the FLOP/B ratio of each system, 3.4 x 1.9 in.

  python scripts/plot_gpusize.py --csv results/rtx4090/e0-vf-mul-add [--yscale linear|log] [--out figures/gpusize_r2.pdf]

Systems 1 to 4 keep the values of the 2025 runs (FLOP/B from the systems table of the paper; maximum speedup from the interim figure
flops_speedup_r2.pdf that the paper agent drew, see revision2/figure-spec.md F7): they were measured on other machines and are not re-measured.
System 5 (RTX 4090, FLOP/B 81.76 from the same table) is the maximum of column 'Mean Speedup' (speedup against OpenCV-CUDA) of the new E0 run
e0-vf-mul-add (VF + HF, batch 50, 2 to 19,902 operations), read from the CSV. No fitted line. The labels S1 to S5 are placed with hand offsets
so that S2 and S3 (26 and 27.73 FLOP/B, 2.5k and 2.6k) do not overlap.
"""
import os

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAME = "benchmark_vertical_fusion_loopMulAdd.csv"

# FLOP/B, maximum speedup of the 2025 runs, label offset in points (dx, dy), horizontal and vertical alignment of the label
SYSTEMS_2025 = {
    "S1": (18.36, 1300.0, (6, 0), "left", "center"),
    "S2": (26.0, 2500.0, (-4, 4), "right", "bottom"),
    "S3": (27.73, 2600.0, (4, -3), "left", "top"),
    "S4": (16.75, 16600.0, (-5, 0), "right", "center"),
}
S5_FLOPB = 81.76
S5_LABEL = (-5, 0, "right", "center")


def kfmt(v, _):
    return "0" if v == 0 else (f"{v / 1000:g}k" if v >= 1000 else f"{v:g}")


def main():
    ap = pc.arg_parser(__doc__, "gpusize_r2.pdf")
    ap.add_argument("--yscale", choices=["linear", "log"], default="linear", help="linear from 0 is the specification; log if S1-S3 are not legible (they are, here: both are produced)")
    args = ap.parse_args()
    path = pc.resolve(args.csv, NAME)
    _, blocks = pc.read_blocks(path)
    x = pc.column(blocks[0], blocks[0][0][0])
    y = pc.column(blocks[0], "Mean Speedup")
    j = y.index(max(y))
    s5 = max(y)

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 1.9), constrained_layout=True)
    pts = {k: (v[0], v[1]) for k, v in SYSTEMS_2025.items()}
    pts["S5"] = (S5_FLOPB, s5)
    for k, (fx, sy) in pts.items():
        ax.plot([fx], [sy], color=pc.SERIES_COLORS[0], marker="o", markersize=4.5, linestyle="none", clip_on=False)
        if k == "S5":
            dx, dy, ha, va = S5_LABEL
        else:
            (dx, dy), ha, va = SYSTEMS_2025[k][2], SYSTEMS_2025[k][3], SYSTEMS_2025[k][4]
        ax.annotate(k, (fx, sy), xytext=(dx, dy), textcoords="offset points", ha=ha, va=va)
    ax.set_xlim(0, 90)
    ax.set_xticks(range(0, 91, 10))
    if args.yscale == "log":
        ax.set_yscale("log")
        ax.set_ylim(800, 30000)
        ax.grid(False, which="minor")
    else:
        ax.set_ylim(0, 24000)
        ax.set_yticks(range(0, 24001, 5000))
    ax.yaxis.set_major_formatter(FuncFormatter(kfmt))
    ax.set_xlabel("FLOP/B")
    ax.set_ylabel("Maximum speedup (VF+HF)")
    notes = ["S1-S4: FLOP/B and maximum speedup of the 2025 runs, not re-measured: " + ", ".join(f"{k} {v[0]:g} FLOP/B {v[1]:.0f}x" for k, v in SYSTEMS_2025.items()),
             f"S5: FLOP/B {S5_FLOPB}; maximum speedup {s5:.1f}x at {x[j]:.0f} operations (batch 50, vs OpenCV-CUDA), from {len(x)} points ({x[0]:.0f} to {x[-1]:.0f} operations)",
             f"y axis: {args.yscale}" + (" (S1-S3 are near the bottom of a linear axis; log keeps them legible)" if args.yscale == "log" else "")]
    if x[-1] < 19902:
        notes.append(f"WARNING: the CSV stops at {x[-1]:.0f} operations: the run is not finished, do NOT use this figure")
    pc.save(fig, args.out, [path], "plot_gpusize.py", notes)


if __name__ == "__main__":
    main()
