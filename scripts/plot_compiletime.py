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
"""F8, fig:compile_times: compile time and compiler memory against the number of fused operations (E3), two panels.

  python scripts/plot_compiletime.py --csv results/rtx4090/e3-compile [--layout wide|single] [--out figures/compiletime_r2.pdf]

(a) nvcc wall time, (b) peak memory of the compiler (the largest process of the nvcc run, GNU time), both logarithmic, one core,
nvcc 13.4.92, -O3, -std=c++17, sm_89 + PTX. Series:
  StaticLoop            the vertical-fusion translation units of the paper (cvGPUSpeedup mul_add1 chain, identical operations in a StaticLoop), 2 to 19,902
  distinct operations   variant A of the compile-time generator (one distinct IOp per operation, float): with the get_arg change (solid; the PR branch
                        where measured, else a copy of the tree with the same function) and without it (dotted, hollow markers)
The timeouts of the distinct-operations chain at N = 6,000 (3,600 s without the change, 5,400 s with it) are hollow triangles at the cap, in (a) only:
a killed compiler has no peak memory. The dotted vertical line is the first failing N of variant B (12 B per operation): ptxas refuses the kernel
parameter space (0x8508 bytes, limit 0x7ffc) at N = 3,848; variant A (4 B per operation) would reach that limit at N = 8,177 and was not compiled that far.
The register cliff at 1,024 operations is a row of the E3 table (e3_table_before_after.csv), not a curve here: it does not change the compile time.
"""
import csv

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(table, variant, trees):
    """{n: row} of the rows of the given trees (earlier trees win when both measured the same n), status ok, plus the timeouts."""
    out, timeouts = {}, {}
    for tree in trees:
        for r in table:
            if r["tree"] != tree or r["variant"] != variant:
                continue
            n = int(float(r["n"]))
            if r["status"] == "ok":
                out.setdefault(n, r)
            elif r["status"].startswith("timeout"):
                timeouts.setdefault(n, r)
    return dict(sorted(out.items())), timeouts


def main():
    ap = pc.arg_parser(__doc__, "compiletime_r2.pdf")
    ap.add_argument("--layout", choices=["wide", "single"], default="wide", help="wide: two panels side by side, 7.1 x 2.0 in; single: panel (a) only, one column, 3.4 x 2.1 in")
    args = ap.parse_args()
    chains = pc.resolve(args.csv, "e3_table_chains.csv")
    loops = pc.resolve(args.csv, "e3_table_staticloop.csv")
    table = rows(chains)
    sl = rows(loops)
    without, to_without = series(table, "A", ["current"])
    with_fix, to_with = series(table, "A", ["getarg-pr", "patched"])
    sl_n = [int(r["ops"]) for r in sl]

    pc.apply_style()
    if args.layout == "wide":
        fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.0), constrained_layout=True)
        ax_t, ax_m = axes
    else:
        fig, ax_t = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
        ax_m = None
    c_sl, c_a = pc.SERIES_COLORS[0], pc.SERIES_COLORS[1]

    def draw(ax, key, scale):
        ax.plot(sl_n, [float(r[key]) * scale for r in sl], color=c_sl, marker="o", markersize=3.0, linewidth=1.0, label="StaticLoop (identical operations)")
        ax.plot(list(without), [float(r[key]) * scale for r in without.values()], color=c_a, linestyle=(0, (1.5, 2.0)), marker="s", markersize=3.6,
                markerfacecolor="white", markeredgewidth=0.8, linewidth=1.0, label="distinct operations, without the change")
        ax.plot(list(with_fix), [float(r[key]) * scale for r in with_fix.values()], color=c_a, marker="s", markersize=2.4, linewidth=1.0,
                label="distinct operations, with the change")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 3e4)
        ax.axvline(3848, color=pc.INK, linestyle=(0, (1.5, 2.5)), linewidth=0.8, label="variant B fails at N = 3,848 (ptxas)" if ax is ax_t else None)
        ax.grid(False, which="minor")
        ax.set_xlabel("Number of fused operations")

    draw(ax_t, "nvcc_s", 1.0)
    ax_t.set_ylabel("nvcc wall time (s)")
    ax_t.set_ylim(1, 1e4)
    for to, cap_marker in ((to_without, dict(markerfacecolor="white")), (to_with, dict(markerfacecolor=c_a))):
        for n, r in to.items():
            ax_t.plot([n], [float(r["nvcc_s"])], color=c_a, marker="^", markersize=4.2, markeredgewidth=0.8, linestyle="none", **cap_marker)
    ax_t.text(6000 * 1.25, 4300, "timeout", ha="left", va="center")
    ax_t.legend(loc="upper left")

    if ax_m is not None:
        draw(ax_m, "peak_rss_mb", 1 / 1024.0)
        ax_m.set_ylabel("Peak compiler memory (GB)")
        ax_m.set_ylim(0.1, 30)
        ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    notes = ["variant A without the change: tree 'current' (LTS-C++17 before PR #417); with: " + "tree 'getarg-pr' (PR branch) where measured, else 'patched' (same function in a scratch copy)",
             f"distinct operations without: N {min(without)}..{max(without)} ({len(without)} points) + timeout at {sorted(to_without)}; with: N {min(with_fix)}..{max(with_fix)} ({len(with_fix)} points) + timeout at {sorted(to_with)}",
             f"StaticLoop: N {sl_n[0]}..{sl_n[-1]} ({len(sl_n)} points), nvcc {float(sl[0]['nvcc_s']):.1f} s to {float(sl[-1]['nvcc_s']):.1f} s",
             "derived by scripts/e3_table.py from e3_compile.csv, e3_compile_A_knee*.csv, e3_patched_A.csv, e3_getarg_pr_A.csv, compile-times-1-100-500-2000/*.csv, e3_staticloop_mul_add1.csv, e3_limit_B_384*.csv"]
    pc.save(fig, args.out, [chains, loops], "plot_compiletime.py", notes)


if __name__ == "__main__":
    main()
