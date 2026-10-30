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
"""Tables of the data-size parity test (results/rtx4090/e0-data-size-parity) from the Nsight Compute CSVs of opencv-cuda/parity/opencv_parity_probe.cu.

  python scripts/parity_tables.py results/rtx4090/e0-data-size-parity

Writes parity_ncu_selected_sizes.csv (per pair and size: the kernel family OpenCV-CUDA used for convertTo, multiply and add, their duration and DRAM throughput, and the cvGS
fused kernel's) and parity_ncu_sweep_32FC1.csv (the 60 sizes of the benchmark: the multiply kernel's family, duration and DRAM throughput, the fused kernel's) and prints the
summary. ncu serialises the kernels, flushes the caches between them and locks the clocks: these are isolated cold launches, not the back-to-back launches of the benchmark.
"""
import csv
import io
import os
import re
import sys

DUR, DRAM = "gpu__time_duration.sum", "dram__throughput.avg.pct_of_peak_sustained_elapsed"


def num(s):
    return float(s.replace(",", ""))


def read_ncu(path):
    text = open(path, errors="replace").read()
    m = re.search(r'^"?ID"?,', text, re.M)
    rows = list(csv.reader(io.StringIO(text[m.start():])))
    h = rows[0]
    unit = rows[1][h.index(DUR)]
    assert unit == "ns", unit
    return [dict(zip(h, r)) for r in rows[1:] if len(r) == len(h) and r[0].strip().isdigit()]


def family(k):
    return "transformSmart<4>" if "transformSmart" in k["Kernel Name"] else "transformSimple"


def pick(ks, key):
    return [k for k in ks if key in k["Kernel Name"]]


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/e0-data-size-parity"
    types = ["32FC1->32FC1", "8UC1->32FC1", "8UC3->32FC3"]
    sizes = [100, 282370, 564640, 846910, 9032740, 16654030]
    ks = read_ncu(os.path.join(d, "ncu_selected_sizes.csv"))
    conv, mul, add, fused = pick(ks, "Convertor"), pick(ks, "MulScalarOp"), pick(ks, "AddScalarOp"), pick(ks, "launchTransformDPP")
    out = []
    for ti, t in enumerate(types):
        for si, n in enumerate(sizes):
            i = ti * len(sizes) + si
            row = {"pair": t, "n": n, "n_mod_4": n % 4}
            for name, k in (("convert", conv[i]), ("multiply", mul[i]), ("add", add[i])):
                row[f"{name}_kernel"] = family(k)
                row[f"{name}_us"] = f"{num(k[DUR]) / 1000:.2f}"
                row[f"{name}_dram_pct"] = f"{num(k[DRAM]):.1f}"
            if ti < 2:
                f = fused[ti * len(sizes) + si]
                row["fused_us"], row["fused_dram_pct"] = f"{num(f[DUR]) / 1000:.2f}", f"{num(f[DRAM]):.1f}"
            out.append(row)
    cols = ["pair", "n", "n_mod_4", "convert_kernel", "convert_us", "convert_dram_pct", "multiply_kernel", "multiply_us", "multiply_dram_pct", "add_kernel", "add_us", "add_dram_pct", "fused_us", "fused_dram_pct"]
    with open(os.path.join(d, "parity_ncu_selected_sizes.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(out)
    sw = read_ncu(os.path.join(d, "ncu_benchmark_sizes_32FC1.csv"))
    smul, sfused = pick(sw, "MulScalarOp"), pick(sw, "launchTransformDPP")
    sweep = []
    for i, (m, f) in enumerate(zip(smul, sfused)):
        n = 100 + 282270 * i
        sweep.append({"n": n, "n_mod_4": n % 4, "multiply_kernel": family(m), "multiply_us": f"{num(m[DUR]) / 1000:.2f}", "multiply_dram_pct": f"{num(m[DRAM]):.1f}",
                      "fused_us": f"{num(f[DUR]) / 1000:.2f}", "fused_dram_pct": f"{num(f[DRAM]):.1f}"})
    with open(os.path.join(d, "parity_ncu_sweep_32FC1.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        w.writeheader()
        w.writerows(sweep)
    over = [r for r in sweep if float(r["multiply_dram_pct"]) > 40]
    print("sweep, float pair, 60 sizes of the benchmark:")
    print("  vectorised kernel (transformSmart<4>) <-> N mod 4 = 0 for all points:", all((r["multiply_kernel"] == "transformSmart<4>") == (r["n_mod_4"] == 0) for r in sweep))
    print("  first size where the OpenCV multiply kernel exceeds 40 % of peak DRAM bandwidth:", over[0]["n"], f"({over[0]['multiply_dram_pct']} %)")
    print("  maximum DRAM %: transformSmart<4>", max(float(r["multiply_dram_pct"]) for r in sweep if r["multiply_kernel"] == "transformSmart<4>"),
          "transformSimple", max(float(r["multiply_dram_pct"]) for r in sweep if r["multiply_kernel"] == "transformSimple"))
    print("  fused kernel: first size above 40 %:", next(r["n"] for r in sweep if float(r["fused_dram_pct"]) > 40), "maximum", max(float(r["fused_dram_pct"]) for r in sweep))


if __name__ == "__main__":
    main()
