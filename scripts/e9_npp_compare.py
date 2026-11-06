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
"""The NPP re-runs after FastNPP #45 against the single runs of 2026-10-30, in the standing form: old / new / spread / exceeds / bound / expected.

  python scripts/e9_npp_compare.py [--out results/rtx4090/e9-cross-library/npp_comparison.md]

old: results/rtx4090/{e0-npp,e0-cpu-time-npp}/ (ONE run each, 2026-10-30, FastNPP 9dfc4f4 and the old FKL pin);
new: results/rtx4090/{e0-npp-after45,e0-cpu-time-npp-after45}/run{1,2,3}/ (FastNPP a1b5e245, FKL c8fcbe8), median of the 3 runs per batch.
spread = (max - min) / median of the 3 new runs; change = new median / old - 1; exceeds = |change| above the flag of the number (3 % for a GPU-bound time, 5 % for a host-bound one);
beyond_spread = |new median - old| > (max - min) of the new runs. 'expected' says whether #45 is expected to change that column: the FastNPP columns may change (the resize now samples at the pixel
centre), the NPP columns must not (the same NPP calls); an exceeding NPP number is a finding and is reported as such, not explained here.
"""
import argparse
import csv
import os
import statistics as st

R = "results/rtx4090"
# (label, old file, new file name, per-column (bound, flag))
PROGRAMS = [
    ("e0-npp, with CPU overhead", f"{R}/e0-npp/test_npp_batchresize_x_split3D.csv", "e0-npp-after45", "test_npp_batchresize_x_split3D.csv",
     {"NPP MeanTime": ("GPU-bound", 3.0), "FK MeanTime": ("host-bound", 5.0)}),
    ("e0-npp, GPU only (No_CPU_Overhead)", f"{R}/e0-npp/test_npp_batchresize_x_split3D_No_CPU_Overhead.csv", "e0-npp-after45", "test_npp_batchresize_x_split3D_No_CPU_Overhead.csv",
     {"NPP MeanTime": ("GPU-bound", 3.0), "FK MeanTime": ("GPU-bound", 3.0)}),
    ("e0-cpu-time-npp, CPU time", f"{R}/e0-cpu-time-npp/test_NPP_cpu_batchresize_x_split3D.csv", "e0-cpu-time-npp-after45", "test_NPP_cpu_batchresize_x_split3D.csv",
     {"NPP MeanTime": ("host-bound", 5.0), "FK MeanTime": ("host-bound", 5.0)}),
]
RUNS = (1, 2, 3)


def read(path):
    rows = [l for l in open(path) if l.strip() and not l.startswith("#")]
    header = [h.strip() for h in rows[0].split(",")]
    out = {}
    for r in csv.reader(rows[1:]):
        if len(r) != len(header):
            continue
        d = dict(zip(header, [float(x) for x in r]))
        out[int(d["Batch size"])] = d
    return out


def agreement(path):
    rows = [l for l in open(path) if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(rows))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=f"{R}/e9-cross-library/npp_comparison.md")
    a = ap.parse_args()
    lines = ["# NPP re-runs after FastNPP #45 against the single runs of 2026-10-30", "",
             "old = ONE run each (2026-10-30, FastNPP 9dfc4f4, the old FKL pin); new = median of 3 runs (FastNPP a1b5e245, FKL c8fcbe8). Times in ms. exceeds: |change| > 3 % (GPU-bound) or 5 % (host-bound).", ""]
    summary = []
    for label, old_path, new_dir, new_name, cols in PROGRAMS:
        old = read(old_path)
        new = [read(f"{R}/{new_dir}/run{i}/{new_name}") for i in RUNS]
        batches = sorted(set(old) & set.intersection(*[set(n) for n in new]))
        lines += [f"## {label}", "", f"{len(batches)} batch sizes in common (old {len(old)}, new {[len(n) for n in new]}).", ""]
        for col, (bound, flag) in cols.items():
            who = "FastNPP (the 'FK' column)" if col == "FK MeanTime" else "NPP"
            expected = "may change (#45: pixel-centre resize)" if col == "FK MeanTime" else "must NOT change (same NPP calls)"
            lines += [f"### {who}, {bound}, flag {flag:g} %, {expected}", "",
                      "| batch | old | new median | spread % | change % | exceeds | beyond spread |", "|---:|---:|---:|---:|---:|:---:|:---:|"]
            exceeds = beyond = 0
            changes = []
            spreads = []
            for b in batches:
                o = old[b][col]
                vals = [n[b][col] for n in new]
                med = st.median(vals)
                spread = (max(vals) - min(vals)) / med * 100
                change = (med / o - 1) * 100
                ex = abs(change) > flag
                bs = abs(med - o) > (max(vals) - min(vals))
                exceeds += ex
                beyond += bs
                changes.append(change)
                spreads.append(spread)
                lines.append(f"| {b} | {o:.6g} | {med:.6g} | {spread:.1f} | {change:+.1f} | {'YES' if ex else 'no'} | {'yes' if bs else 'no'} |")
            s = (f"{label} / {who}: {len(batches)} points; change min {min(changes):+.1f} %, median {st.median(changes):+.1f} %, max {max(changes):+.1f} %; "
                 f"exceeds the {flag:g} % flag in {exceeds}, beyond the spread of the 3 runs in {beyond}; largest spread of the 3 runs {max(spreads):.1f} %; {bound}; {expected}")
            summary.append(s)
            lines += ["", s, ""]
        # speedup column: derived from the two means, reported for the figure only
    lines += ["## Agreement (informational maximum difference FastNPP against NPP, and which convention each follows)", ""]
    for label, old_path, new_dir, new_name, _ in PROGRAMS[:1] + PROGRAMS[2:]:
        agr_old = agreement(old_path.replace(".csv", "_agreement.csv"))
        agr_new = [agreement(f"{R}/{new_dir}/run{i}/{new_name.replace('.csv', '_agreement.csv')}") for i in RUNS]
        fo = sorted(set(r["FastNPP_follows"] for r in agr_old))
        fn = sorted(set(r["FastNPP_follows"] for r in agr_new[0]) | set(r["FastNPP_follows"] for ag in agr_new[1:] for r in ag))
        no = sorted(set(r["NPP_follows"] for r in agr_old))
        nn = sorted(set(r["NPP_follows"] for ag in agr_new for r in ag))
        mo = [float(r["FastNPP_vs_NPP_max"]) for r in agr_old]
        mn = [float(r["FastNPP_vs_NPP_max"]) for ag in agr_new for r in ag]
        pn = [float(r["FastNPP_vs_ref_pixel_centre_max"]) for ag in agr_new for r in ag]
        nr = [float(r["NPP_vs_ref_pixel_centre_max"]) for ag in agr_new for r in ag]
        lines += [f"- {label}: FastNPP_follows old {fo} ({len(agr_old)} rows) -> new {fn} ({[len(ag) for ag in agr_new]} rows); NPP_follows old {no} -> new {nn}; "
                  f"FastNPP_vs_NPP_max old {min(mo):.6g} to {max(mo):.6g} -> new {min(mn):.6g} to {max(mn):.6g}; FastNPP_vs_ref_pixel_centre_max new max {max(pn):.6g}; NPP_vs_ref_pixel_centre_max new max {max(nr):.6g} "
                  f"(tolerance of the check 1e-3)"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(summary))
    print("\n".join(l for l in lines if l.startswith("- ")))
    print("written", a.out)


if __name__ == "__main__":
    main()
