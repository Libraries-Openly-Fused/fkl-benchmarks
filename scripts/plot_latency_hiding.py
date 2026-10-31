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
"""F1, fig:LatencyHiding: kernel time against the number of instructions of the kernel (E0 latency hiding), 3.4 in wide.

  python scripts/plot_latency_hiding.py --csv results/rtx4090/e0-latency-hiding [--out figures/latencyhiding_r2.pdf]

The vertical dotted line is the knee of the measured curve, not a number taken from the old figure (which put it near 300): the
curve is fitted with "flat, then a straight line" (continuous at the knee, the knee anywhere between the second and the second to
last point, least squares) and the line goes where the fit puts the end of the plateau. The fit is printed and written to the sidecar.
Ticks are every 180 instructions starting at 1 (1, 181, 361, ...: a subset of the old figure's 1, 61, 121, ... that stays legible at
7.5 pt in 3.4 in; every tick is a measured point).
"""
import numpy as np

import plot_common as pc
import matplotlib.pyplot as plt

NAME = "testLatencyHiding_GPU_NVIDIA.csv"


def knee(x, y):
    """Continuous 'flat then linear' least-squares fit: returns (knee, plateau, slope, rms residual)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    best = None
    for b in np.linspace(x[1], x[-2], 4000):
        A = np.column_stack([np.ones_like(x), np.maximum(x - b, 0.0)])
        coef, res, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, b, coef[0], coef[1])
    sse, b, c, s = best
    return b, c, s, (sse / len(x)) ** 0.5


def main():
    ap = pc.arg_parser(__doc__, "latencyhiding_r2.pdf")
    args = ap.parse_args()
    path = pc.resolve(args.csv, NAME)
    _, blocks = pc.read_blocks(path)
    x = pc.column(blocks[0], "Number of Operations")
    y = pc.column(blocks[0], "MeanTime")
    b, plateau, slope, rms = knee(x, y)

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 1.85), constrained_layout=True)
    ax.plot(x, y, color=pc.SERIES_COLORS[0], marker="o", markersize=2.2, linewidth=1.0)
    ax.axvline(b, color=pc.INK, linestyle=(0, (1.5, 2.5)), linewidth=0.8)
    ymax = 2.0
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, ymax)
    ax.set_xticks(list(range(1, 1182, 180)))
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel("Number of instructions in the kernel")
    ax.set_ylabel("Execution time (ms)")
    ax.text(b / 2, 0.80 * ymax, "Memory-\nbound\nkernels", ha="center", va="center")
    ax.text(b + (1200 - b) * 0.55, 0.30 * ymax, "Compute-bound\nkernels", ha="center", va="center")
    notes = [f"knee of the measured curve (fit of 'flat, then linear'): {b:.0f} instructions; plateau {plateau:.4f} ms; slope after the knee {slope * 1000:.3f} us per instruction; rms residual {rms * 1000:.1f} us",
             f"{len(x)} points, x from {x[0]:.0f} to {x[-1]:.0f}, y = column MeanTime (ms)"]
    pc.save(fig, args.out, [path], "plot_latency_hiding.py", notes)


if __name__ == "__main__":
    main()
