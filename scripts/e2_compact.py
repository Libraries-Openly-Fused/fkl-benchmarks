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
"""Compact E2 table for the paper (one column, 3.4 in at 8 pt): one row per implementation and chain length k, from e2_chain.csv and e2_chain_large.csv.

  python scripts/e2_compact.py results/rtx4090/e2-chain [results/rtx4090/e2-chain-run3 results/rtx4090/e2-chain-run4]

The first directory is written to (e2_compact.csv) and gives the columns that are not timings; with more directories every timing is the median of the means of the runs, spread_pct is (largest - smallest) / median
of the means of the runs, min_ms is the smallest single call of all runs, runs the number of runs.

Columns: k, impl, timing ("warm": the data of the previous call is still in the 72 MB L2; "L2-flushed": a kernel that reads 256 MB ran before the call, outside the timed interval; the
flushed rows exist for FKL, Triton and the three Inductor variants, eager has warm rows only), kernels (launched per call), gpu_ms (CUDA events, mean of 100 calls), compile_s (cold: a fresh process and fresh caches; FKL: nvcc, one core),
bitwise (fraction of output elements identical to FKL's), ulp_fkl (maximum difference to FKL in float32 ULPs of FKL's value), ulp_exact (maximum difference to an exact
float64 evaluation in ULPs), host_us (host time to launch one call, no synchronisation; not measured for FKL).
Rows: k = 10, 100, 500 and the bounded larger attempt k = 5,002. Left out: gpu_min_ms, the SASS counts (README), the 'eager captured in one CUDA graph by hand' version.
"""
import csv
import os
import statistics
import sys

NAMES = {"fkl": "FKL", "triton_hand": "Triton (hand-written)", "inductor_default": "TorchInductor", "inductor_wide": "TorchInductor (wide fusion window)",
         "inductor_graphs": "TorchInductor + CUDA Graphs", "eager": "PyTorch eager"}


def rows(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def main():
    dirs = sys.argv[1:] or ["results/rtx4090/e2-chain"]
    d = dirs[0]
    runs = [rows(os.path.join(x, "e2_chain.csv")) + rows(os.path.join(x, "e2_chain_large.csv")) for x in dirs]
    data = runs[0]

    def timing(k, ver, mean_key, min_key):
        """median of the means over the runs, spread, smallest single call, number of runs"""
        means, mins = [], []
        for run in runs:
            r = next((x for x in run if int(x["ops"]) == k and x["version"] == ver), None)
            if r is not None and r["status"] == "ok" and r.get(mean_key, "") != "":
                means.append(float(r[mean_key]))
                if r.get(min_key, "") != "":
                    mins.append(float(r[min_key]))
        med = statistics.median(means)
        return {"gpu_ms": f"{med:.4f}", "spread_pct": f"{100 * (max(means) - min(means)) / med:.1f}", "min_ms": f"{min(mins):.4f}" if mins else "", "runs": len(means)}

    out = []
    for k in (10, 100, 500, 5002):
        for ver, name in NAMES.items():
            r = next((x for x in data if int(x["ops"]) == k and x["version"] == ver), None)
            if r is None or r["status"] != "ok":
                out.append({"k": k, "impl": name, "timing": "", "kernels": "", "gpu_ms": "", "spread_pct": "", "min_ms": "", "runs": "", "compile_s": "", "bitwise": "", "ulp_fkl": "", "ulp_exact": "", "host_us": "", "status": r["status"] if r else "missing"})
                continue
            f = lambda key, fmt: fmt.format(float(r[key])) if r.get(key, "") != "" else ""
            base = {"k": k, "impl": name, "kernels": r["kernels"], "compile_s": f("compile_s", "{:.2f}"), "bitwise": f("bitexact_fraction_vs_fkl", "{:.4f}"),
                    "ulp_fkl": f("max_ulp_vs_fkl", "{:.3g}"), "ulp_exact": f("max_ulp_vs_exact", "{:.3g}"), "host_us": f("host_launch_us", "{:.1f}"), "status": "ok"}
            out.append(dict(base, timing="warm", **timing(k, ver, "gpu_mean_ms", "gpu_min_ms")))
            if ver != "eager" and r.get("gpu_flushed_mean_ms", "") != "":
                out.append(dict(base, timing="L2-flushed", **timing(k, ver, "gpu_flushed_mean_ms", "gpu_flushed_min_ms")))
    path = os.path.join(d, "e2_compact.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(path, len(out), "rows")


if __name__ == "__main__":
    main()
