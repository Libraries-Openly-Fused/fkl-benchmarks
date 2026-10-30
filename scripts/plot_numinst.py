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
"""F4, fig:NumInstructionsPerOperation: speedup of one fused kernel against ceil(500/M) kernels, against the instructions per operation M.

  python scripts/plot_numinst.py --csv results/rtx4090/e0-numinstructions [--width 2.1] [--out figures/numinst_r2.pdf]

Drawn at 2.1 in wide by default (the 0.3\\textwidth inset) with 7.5 pt text, so it is legible there; if the paper prints it at
\\columnwidth the PDF is simply scaled up (--width 3.4 draws it at that width instead). Logarithmic y axis with gridlines at the
decades, the first and the last point annotated with their values (read from the CSV).
"""
import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAME = "testNumInstPerOp_GPU_NVIDIA.csv"


def main():
    ap = pc.arg_parser(__doc__, "numinst_r2.pdf")
    ap.add_argument("--width", type=float, default=2.1, help="printed width in inches (2.1 = 0.3\\textwidth, 3.4 = one column)")
    args = ap.parse_args()
    path = pc.resolve(args.csv, NAME)
    _, blocks = pc.read_blocks(path)
    x = pc.column(blocks[0], "Number of instructions per Operation")
    y = pc.column(blocks[0], "Mean Speedup")

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(args.width, args.width * 0.9), constrained_layout=True)
    ax.plot(x, y, color=pc.SERIES_COLORS[0], marker="o", markersize=1.8, linewidth=1.0)
    ax.set_yscale("log")
    ax.set_xlim(0, 520)
    ax.set_ylim(1, 1000)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.grid(False, which="minor")
    ax.set_xlabel("Instructions per operation (M)")
    ax.set_ylabel("Speedup of one\nfused kernel vs\n" + r"$\lceil 500/M \rceil$" + " kernels")
    ax.annotate(f"{y[0]:.0f}×", (x[0], y[0]), xytext=(5, -2), textcoords="offset points", ha="left", va="center")
    ax.annotate(f"{y[-1]:.2f}×", (x[-1], y[-1]), xytext=(-3, 6), textcoords="offset points", ha="right", va="bottom")
    notes = [f"{len(x)} points, M from {x[0]:.0f} to {x[-1]:.0f}; first speedup {y[0]:.3f}, last {y[-1]:.3f}, minimum {min(y):.3f} at M = {x[y.index(min(y))]:.0f}"]
    pc.save(fig, args.out, [path], "plot_numinst.py", notes)


if __name__ == "__main__":
    main()
