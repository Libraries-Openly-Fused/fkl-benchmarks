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
"""Compact E3 tables for the paper (one column, 3.4 in at 8 pt), derived from the full tables of scripts/e3_table.py.

  python scripts/e3_compact.py results/rtx4090/e3-compile

  e3_compact_before_after.csv  variant A (one distinct float operation per fused operation, 4 B of kernel parameters each), selected N:
                               N, registers, spill bytes, nvcc wall time (s) and kernel time (ms), each without ("_b", before) and with ("_a", after) the get_arg change
  e3_compact_limits.csv        what stops a fused kernel and at which N: the register cliff (before the change), the ptxas parameter-space limit (variant B),
                               and the recursive fk::get of the old trees (the fold-versus-recursion rows)

A third file, e3_compact_current.csv, is the one for the paper: the library as it is now, variant A, without any comparison with an earlier state (columns N,
param_B, regs, spill_B, sass, nvcc_s, rss_GB, ms: the kernel time of the fused kernel where it was measured). The two files above compare the state before and after the
get_arg change of PR #417: they are material for the response letter and the README of the results folder, not for paper-facing files.

Columns of e3_table_before_after.csv that the compact table leaves out: param_space_bytes, sass_*, cicc_s_*, ptxas_s_*, rss_mb_*, status_*, with_tree
(the full table keeps them; the peak memory is in the compile-time figure).
"""
import csv
import os
import sys

ROWS_A = [1, 100, 500, 1000, 1023, 1024, 2000, 4000]


def fnum(text, digits):
    try:
        v = float(text)
    except ValueError:
        return ""
    return f"{v:.{digits}f}" if digits else f"{v:.0f}"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e3-compile"
    ba = {(r["variant"], int(r["n"])): r for r in csv.DictReader(open(os.path.join(d, "e3_table_before_after.csv")))}
    out = []
    for n in ROWS_A:
        r = ba[("A", n)]
        out.append({"N": n, "regs_b": r["registers_without"], "regs_a": r["registers_with"],
                    "spill_b": r["spill_bytes_without"], "spill_a": r["spill_bytes_with"],
                    "nvcc_s_b": fnum(r["nvcc_s_without"], 1), "nvcc_s_a": fnum(r["nvcc_s_with"], 1),
                    "ms_b": fnum(r["kernel_ms_without"], 3) if r["kernel_ms_without"] else "", "ms_a": fnum(r["kernel_ms_with"], 3) if r["kernel_ms_with"] else ""})
    with open(os.path.join(d, "e3_compact_before_after.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    cur = []
    for n in ROWS_A:
        r = ba[("A", n)]
        cur.append({"N": n, "param_B": r["param_space_bytes"], "regs": r["registers_with"], "spill_B": r["spill_bytes_with"], "sass": r["sass_with"],
                    "nvcc_s": fnum(r["nvcc_s_with"], 1), "rss_GB": fnum(float(r["rss_mb_with"]) / 1024.0, 1) if r["rss_mb_with"] else "",
                    "ms": fnum(r["kernel_ms_with"], 3) if r["kernel_ms_with"] else ""})
    with open(os.path.join(d, "e3_compact_current.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cur[0].keys()))
        w.writeheader()
        w.writerows(cur)

    chains = list(csv.DictReader(open(os.path.join(d, "e3_table_chains.csv"))))
    trees = list(csv.DictReader(open(os.path.join(d, "e3_table_trees.csv"))))
    lim = []
    # register cliff of variant A without the change: the first N of the table with 255 registers, and the last one below it with the register count there
    a_wo = sorted(((int(r["n"]), r) for r in chains if r["tree"] == "current" and r["variant"] == "A" and r["status"] == "ok"), key=lambda t: t[0])
    cliff = next(n for n, r in a_wo if int(r["registers"]) == 255)
    below = max(n for n, r in a_wo if n < cliff)
    lim.append({"case": "A: registers", "last_ok_N": below, "first_bad_N": cliff, "what": "255 registers and local-memory spills (17 before); with the change 17 registers at every N measured (to 4,000)"})
    # ptxas parameter space, variant B
    b_ok = max(int(r["n"]) for r in chains if r["tree"] == "current" and r["variant"] == "B" and r["status"] == "ok")
    b_bad = min(int(r["n"]) for r in chains if r["tree"] == "current" and r["variant"] == "B" and r["status"].startswith("failed"))
    lim.append({"case": "B: ptxas", "last_ok_N": b_ok, "first_bad_N": b_bad, "what": "parameter space 0x8508 B > 0x7ffc B (12 B per operation); variant A (4 B) would reach it at N = 8,177"})
    # the old trees: recursive fk::get
    for tree in ("07b2cb3^", "07b2cb3", "1f33224"):
        rows = [r for r in trees if r["tree"] == tree]
        bad = min(int(r["n"]) for r in rows if r["status"].startswith("failed"))
        ok = max(int(r["n"]) for r in rows if r["status"] == "ok" and int(r["n"]) < bad)
        raw = next(r["first_error"] for r in rows if r["status"].startswith("failed") and int(r["n"]) == bad)
        # e3_table.py keeps the start of the compiler's message; the raw sweeps (e3_limit_fine_*.csv, 300 characters) name the function fk::get<Index...
        assert "cannot deduce" in raw, raw[:200]
        lim.append({"case": f"old tree {tree}", "last_ok_N": ok, "first_bad_N": bad, "what": "nvcc: cannot deduce the return type of fk::get<Index...> (recursive, auto return type)"})
    with open(os.path.join(d, "e3_compact_limits.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(lim[0].keys()))
        w.writeheader()
        w.writerows(lim)
    print(f"{os.path.join(d, 'e3_compact_before_after.csv')} ({len(out)} rows), {os.path.join(d, 'e3_compact_limits.csv')} ({len(lim)} rows)")
    for r in lim:
        print("  ", r)


if __name__ == "__main__":
    main()
