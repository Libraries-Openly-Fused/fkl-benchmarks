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
"""Compact E3 table of the nvcc phases and the cubin size (one column, 3.4 in at 8 pt), for the library as it is now.

  python scripts/e3_compact_phases.py results/rtx4090/e3-compile

Rows: the distinct-operations chain (variant A) at N = 100, 1,000, 2,000 and 4,000, and the StaticLoop chain at 2, 10,002 and 19,902 operations. Columns:
chain, N, reps (compiles the median is taken over), nvcc_s (wall time, one core, --threads 1), frontend_s (cudafe++), cicc_s, ptxas_s, hostcc_s (the host compiler, g++,
that nvcc runs on the host half of the file), other_s (fatbinary, link, ... : nvcc_s minus the four phases), cubin_B (size of the sm_89 cubin).
Distinct operations: the PR #417 tree (the library as it is now): the compiles of e3_getarg_pr_A.csv, compile-times-1-100-500-2000/with_fix.csv and cubin-sizes/e3_cubin_A.csv.
Left out: peak memory, registers and spills (e3_compact_before_after.csv, the compile-time figure), SASS and text bytes (the full tables).
"""
import csv
import os
import statistics
import sys

FIELDS = ["cuda_wall_s", "phase_cudafe_s", "phase_cicc_s", "phase_ptxas_s", "phase_gcc_s", "phase_other_s", "cubin_bytes"]
DISTINCT_SOURCES = ["e3_getarg_pr_A.csv", "compile-times-1-100-500-2000/with_fix.csv", "cubin-sizes/e3_cubin_A.csv"]


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(l for l in f if not l.startswith("#")))


def median_row(rows):
    out = {}
    for k in FIELDS:
        vals = [float(r[k]) for r in rows if r.get(k, "") != ""]
        out[k] = statistics.median(vals) if vals else None
    return out


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e3-compile"
    distinct = []
    for s in DISTINCT_SOURCES:
        p = os.path.join(d, s)
        if os.path.exists(p):
            distinct += [r for r in load(p) if r["variant"] == "A" and r["status"] == "ok"]
    static = [r for r in load(os.path.join(d, "e3_staticloop_mul_add1.csv")) if r["status"] == "ok"]
    out = []
    for chain, rows, key, ns in (("distinct", distinct, "n", [100, 1000, 2000, 4000]), ("StaticLoop", static, "ops", [2, 10002, 19902])):
        for n in ns:
            sel = [r for r in rows if int(float(r[key])) == n]
            if not sel:
                print(f"no ok compile of {chain} at N = {n}", file=sys.stderr)
                continue
            m = median_row(sel)
            f = lambda k, nd: "" if m[k] is None else f"{m[k]:.{nd}f}"
            out.append({"chain": chain, "N": n, "reps": len(sel), "nvcc_s": f("cuda_wall_s", 1), "frontend_s": f("phase_cudafe_s", 1), "cicc_s": f("phase_cicc_s", 1),
                        "ptxas_s": f("phase_ptxas_s", 1), "hostcc_s": f("phase_gcc_s", 1), "other_s": f("phase_other_s", 1), "cubin_B": f("cubin_bytes", 0)})
    path = os.path.join(d, "e3_compact_phases.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(path)
    for r in out:
        print("  ", r)


if __name__ == "__main__":
    main()
