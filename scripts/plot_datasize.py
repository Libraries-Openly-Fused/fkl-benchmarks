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
"""F6, fig:DataSize: execution time of OpenCV-CUDA and FKL against the number of elements, log-log, 3.4 in wide.

  python scripts/plot_datasize.py --csv results/rtx4090/e0-data-size [--pair CV_32FC1XCV_32FC1] [--fkl-label "FKL (cvGS)"] [--out figures/datasize_r2.pdf]

Type pair: CV_32FC1XCV_32FC1 (float to float). The specification suggested 8UC1 -> 32FC1 "the one the old figure used", but the
paper's text of the experiment says "a 1D number of float data elements" and the old figure starts at the same two points (100 and
282,370 elements), so the old figure was the float pair; the six pairs are all in the CSV and --pair selects another one.
The series are drawn as measured, every point, no smoothing. The legend says "FKL (cvGS)" because the column is the cvGS wrapper
(the specification's rule: "FKL (cvGS)" where the wrapper is what was run); --fkl-label "FKL" gives the shorter name.
"""
import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

NAME = "benchmark_image_resolution_MAD_loop.csv"


def main():
    ap = pc.arg_parser(__doc__, "datasize_r2.pdf")
    ap.add_argument("--pair", default="CV_32FC1XCV_32FC1")
    ap.add_argument("--fkl-label", default="FKL (cvGS)")
    ap.add_argument("--xscale", choices=["log", "linear"], default="log", help="log is the specification; linear is the old figure's axis")
    args = ap.parse_args()
    path = pc.resolve(args.csv, NAME)
    _, blocks = pc.read_blocks(path)
    block = next((b for b in blocks if b[0][0].endswith(f"({args.pair})")), None)
    if block is None:
        raise SystemExit(f"no block '{args.pair}' in {path}; blocks: {[b[0][0] for b in blocks]}")
    x = pc.column(block, block[0][0])
    cv = pc.column(block, "OpenCV MeanTime")
    fk = pc.column(block, "cvGS MeanTime")

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
    ax.plot(x, cv, color=pc.SERIES_COLORS[0], marker=pc.SERIES_MARKERS[0], markersize=2.6, markevery=5, linewidth=1.0, label="OpenCV-CUDA")
    ax.plot(x, fk, color=pc.SERIES_COLORS[1], marker=pc.SERIES_MARKERS[1], markersize=2.6, markevery=5, linewidth=1.0, label=args.fkl_label)
    ax.set_yscale("log")
    if args.xscale == "log":
        ax.set_xscale("log")
        ax.set_xlim(1e2, 10 ** 7.3)
    else:
        ax.set_xlim(0, 17.5e6)
        ax.set_xticks([0, 5e6, 10e6, 15e6])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_ylim(1e-3, 1e2)
    ax.grid(False, which="minor")
    ax.set_xlabel("Number of elements")
    ax.set_ylabel("Execution time (ms)")
    ax.legend(loc="upper left" if args.xscale == "log" else "lower right")
    slow = [i for i in range(len(x)) if int(x[i]) % 4 == 2]
    notes = [f"x axis: {args.xscale}", f"type pair {args.pair}; {len(x)} points, {x[0]:.0f} to {x[-1]:.0f} elements",
             f"at the last point: OpenCV-CUDA {cv[-1]:.4g} ms, {args.fkl_label} {fk[-1]:.4g} ms (speedup {cv[-1] / fk[-1]:.0f}x)",
             f"OpenCV-CUDA is bimodal in this run: its time at every point with N mod 4 = 2 is about 3x that of the neighbouring N mod 4 = 0 points "
             f"({len(slow)} of {len(x)} points have N mod 4 = 2); the curve is drawn as measured"]
    pc.save(fig, args.out, [path], "plot_datasize.py", notes)


if __name__ == "__main__":
    main()
