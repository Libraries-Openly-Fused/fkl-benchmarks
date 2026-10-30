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
"""F8, fig:compile_times: nvcc compile time against the number of fused operations (E3), the library as it is now.

  python scripts/plot_compiletime.py --csv results/rtx4090/e3-compile [--layout single|wide] [--out figures/compiletime_single_r2.pdf]
  python scripts/plot_compiletime.py --csv results/rtx4090/e3-compile --history --out <not the paper folders>.pdf     (response letter only)

Paper-facing default (the paper describes the library as it is now): two series on a log-log plot, nvcc wall time (one core, nvcc 13.4.92, -O3, -std=c++17,
sm_89 + PTX):
  "StaticLoop (identical operations)"  the vertical-fusion translation units of the paper (cvGPUSpeedup mul_add1 chain in a StaticLoop), 2 to 19,902
  "distinct operations"                variant A of the compile-time generator: one distinct IOp (float) per fused operation, 1 to 4,000, then the time-out
The filled triangle is the time-out of the distinct-operations chain at N = 6,000 (5,400 s cap). The dotted vertical line is the first failing N of variant B
(float3 parameters, 12 B per operation): ptxas refuses the kernel parameter space (0x8508 B, limit 0x7ffc B) at N = 3,848; variant A (4 B per operation) would
reach that limit at N = 8,177 and was not compiled that far.
Which tree the distinct-operations series was compiled with (written to the sidecar): the library with the get_arg change of PR #417 (head a8a74ea on LTS-C++17
d776713); N = 1 ... 4,000 from the branch itself except N = 1,020, and the time-out and N = 1,020 from a copy of the pinned tree 38c4f32 with the same one-function change (the code is
identical, only comments and one static_assert message differ).
--layout wide adds the panel of peak compiler memory (7.1 x 2.0 in); --history adds the same series on the tree before the change (dotted, hollow markers),
only for the response letter and the README of the results folder, never for a paper-facing file.
"""
import csv

import plot_common as pc
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

CURRENT_TREES = ["getarg-pr", "patched"]     # the distinct-operations series of the library as it is now (see the docstring)
HISTORY_TREES = ["current"]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def series(table, variant, trees):
    """{n: row} of the rows of the given trees (earlier trees win when both measured the same n), status ok, plus the timeouts and the N taken from each tree."""
    out, timeouts, used = {}, {}, {}
    for tree in trees:
        for r in table:
            if r["tree"] != tree or r["variant"] != variant:
                continue
            n = int(float(r["n"]))
            if r["status"] == "ok" and n not in out:
                out[n] = r
                used.setdefault(tree, []).append(n)
            elif r["status"].startswith("timeout") and n not in timeouts:
                timeouts[n] = r
                used.setdefault(tree, []).append(n)
    return dict(sorted(out.items())), timeouts, used


def main():
    ap = pc.arg_parser(__doc__, "compiletime_single_r2.pdf")
    ap.add_argument("--layout", choices=["single", "wide"], default="single", help="single: nvcc time only, 3.4 x 2.1 in; wide: time and compiler memory, 7.1 x 2.0 in")
    ap.add_argument("--history", action="store_true", help="also draw the tree before the change (response letter only)")
    args = ap.parse_args()
    chains = pc.resolve(args.csv, "e3_table_chains.csv")
    loops = pc.resolve(args.csv, "e3_table_staticloop.csv")
    table = rows(chains)
    sl = rows(loops)
    cur, to_cur, used = series(table, "A", CURRENT_TREES)
    hist, to_hist, _ = series(table, "A", HISTORY_TREES)
    sl_n = [int(r["ops"]) for r in sl]

    pc.apply_style()
    if args.layout == "wide":
        fig, (ax_t, ax_m) = plt.subplots(1, 2, figsize=(7.1, 2.0), constrained_layout=True)
    else:
        fig, ax_t = plt.subplots(figsize=(3.4, 2.1), constrained_layout=True)
        ax_m = None
    c_sl, c_a = pc.SERIES_COLORS[0], pc.SERIES_COLORS[1]

    def draw(ax, key, scale, with_labels):
        ax.plot(sl_n, [float(r[key]) * scale for r in sl], color=c_sl, marker="o", markersize=3.0, linewidth=1.0, label="StaticLoop (identical operations)" if with_labels else None)
        if args.history:
            ax.plot(list(hist), [float(r[key]) * scale for r in hist.values()], color=c_a, linestyle=(0, (1.5, 2.0)), marker="s", markersize=3.6, markerfacecolor="white",
                    markeredgewidth=0.8, linewidth=1.0, label="distinct operations, before the change" if with_labels else None)
        ax.plot(list(cur), [float(r[key]) * scale for r in cur.values()], color=c_a, marker="s", markersize=2.4, linewidth=1.0,
                label=("distinct operations, after the change" if args.history else "distinct operations") if with_labels else None)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1, 3e4)
        ax.axvline(3848, color=pc.INK, linestyle=(0, (1.5, 2.5)), linewidth=0.8, label="float3 parameters fail at N = 3,848" if with_labels else None)
        ax.grid(False, which="minor")
        ax.set_xlabel("Number of fused operations")

    draw(ax_t, "nvcc_s", 1.0, True)
    ax_t.set_ylabel("nvcc wall time (s)")
    ax_t.set_ylim(1, 1e4)
    for n, r in to_cur.items():
        ax_t.plot([n], [float(r["nvcc_s"])], color=c_a, marker="^", markersize=4.2, markeredgewidth=0.8, linestyle="none")
        ax_t.text(n * 1.25, float(r["nvcc_s"]), "timeout", ha="left", va="center")
    if args.history:
        for n, r in to_hist.items():
            ax_t.plot([n], [float(r["nvcc_s"])], color=c_a, marker="^", markersize=4.2, markeredgewidth=0.8, linestyle="none", markerfacecolor="white")
    ax_t.legend(loc="upper left")
    if ax_m is not None:
        draw(ax_m, "peak_rss_mb", 1 / 1024.0, False)
        ax_m.set_ylabel("Peak compiler memory (GB)")
        ax_m.set_ylim(0.1, 30)
        ax_m.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))

    notes = ["distinct operations (variant A) compiled with the library as it is now: the get_arg change of PR #417 (LTS-C++17-get-arg-no-tuple, head a8a74eaaaee4, on LTS-C++17 d776713)",
             f"  N = {', '.join(str(n) for n in sorted(used.get('getarg-pr', [])))}: compiled from the branch worktree (include path recorded in e3_getarg_pr_A.csv, compile-times-1-100-500-2000/with_fix.csv and cubin-sizes/e3_cubin_A.csv; the get_arg header is identical in every commit of the branch: 4f71c94, 3c0eb19 and a8a74ea)",
             f"  N = {', '.join(str(n) for n in sorted(used.get('patched', [])))}: compiled from a copy of the pinned tree 38c4f32 with the same one-function change (the code of Nth/Skip is identical, only comments and one static_assert message differ); 6000 is the time-out",
             f"StaticLoop: N {sl_n[0]}..{sl_n[-1]} ({len(sl_n)} points), nvcc {float(sl[0]['nvcc_s']):.1f} s to {float(sl[-1]['nvcc_s']):.1f} s",
             "toolchain: CUDA 13.4.3, nvcc 13.4.92, g++ 15.2, one core, --threads 1, -O3 -DNDEBUG -std=c++17, sm_89 + PTX; fkl-benchmarks pins external/fkl at 38c4f32",
             "tables: results/rtx4090/e3-compile/e3_table_chains.csv, e3_table_staticloop.csv (derived by scripts/e3_table.py from the sweeps listed in its docstring)"]
    pc.save(fig, args.out, [chains, loops], "plot_compiletime.py", notes)


if __name__ == "__main__":
    main()
