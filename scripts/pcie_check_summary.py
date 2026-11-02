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
"""Summary of the PCIe link check (results/rtx4090/pcie-check): per phase (idle, load, copies, idle) the link generation and width, the pstate, the SM clock and the utilisation seen by the 100 ms sampler, and the copy bandwidth.

  python scripts/pcie_check_summary.py results/rtx4090/pcie-check
"""
import collections
import csv
import datetime
import os
import statistics as stats
import sys


def number(text):
    return float(text.strip().split()[0])


def main():
    folder = sys.argv[1]
    phases = {}
    for line in open(os.path.join(folder, "phases.txt")):
        if line.startswith("#"):
            continue
        t, name = line.split()
        phases[name] = float(t)
    smp = []
    with open(os.path.join(folder, "sampler.csv"), newline="") as f:
        for r in csv.reader(f):
            if len(r) < 9:
                continue
            try:
                t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
            except ValueError:
                continue
            smp.append((t, r[1].strip(), r[2].strip(), r[3].strip(), r[4].strip(), r[5].strip(), number(r[6]), number(r[7]), number(r[8])))
    out = [f"PCIe link check, {len(smp)} samples of {os.path.join(folder, 'sampler.csv')}; link maximum seen: generation {max(x[3] for x in smp)}, width {max(x[5] for x in smp)}", ""]
    for name in ("idle", "load", "copy", "idle2"):
        a, b = phases[f"{name}_start"], phases[f"{name}_end"]
        sel = [x for x in smp if a <= x[0] <= b]
        if not sel:
            out.append(f"{name}: no samples")
            continue
        cnt = lambda i: ", ".join(f"{k} ({v})" for k, v in collections.Counter(x[i] for x in sel).most_common())
        out.append(f"{name:6s} {b - a:5.1f} s, {len(sel):3d} samples: pstate {cnt(1)}; link generation current {cnt(2)} (max {sel[0][3]}); link width current {cnt(4)} (max {sel[0][5]}); SM clock median {stats.median(x[6] for x in sel):.0f} MHz; utilisation median {stats.median(x[7] for x in sel):.0f} %; power median {stats.median(x[8] for x in sel):.0f} W")
    bw = os.path.join(folder, "pcie_bandwidth.csv")
    if os.path.exists(bw):
        out.append("")
        for r in csv.DictReader([l for l in open(bw) if not l.startswith("#")]):
            out.append(f"copies {r['direction']} {r['memory']:8s}: {r['copies']:>4s} copies of {int(r['bytes']) // 2**20} MB in {float(r['seconds']):.1f} s, median {float(r['gbps_median']):.2f} GB/s (smallest {float(r['gbps_min']):.2f}, largest {float(r['gbps_max']):.2f})")
    text = "\n".join(out) + "\n"
    open(os.path.join(folder, "pcie_check_summary.txt"), "w").write(text)
    print(text)


if __name__ == "__main__":
    main()
