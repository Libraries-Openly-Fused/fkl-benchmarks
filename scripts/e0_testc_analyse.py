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
"""Test C of results/rtx4090/e0-cpu-time-history: the two CPU-time programs built Release / None (empty build type) / Debug, 3 rounds each.

  python scripts/e0_testc_analyse.py [--dir results/rtx4090/e0-cpu-time-history/test-C]

Per program and build: the fused host time (microseconds) and the speedup per batch size (median over the 3 rounds), the run-to-run spread, the slope and the constant of the fused host time (batch 2 to 42 by the two
end points, and 2 to 152 by least squares), the baseline column as the control, Release against the old single runs of 2026-10-30 (host-bound: 5 % flag), and the speedups against the reading of the 2025 figure by the
coordinator (Test A of ../host-path-candidates.md, accuracy +-1 to +-2, NOT data). Writes testC_summary.md next to the data.
"""
import argparse
import csv
import os
import statistics as st

BUILDS = ["Release", "None", "Debug"]
PROGS = [("cvGS", "benchmark_CPU_OpenCV_vs_cvGS_cu", "test_cpu_batchresize_x_split3D.csv", "OpenCV MeanTime", "cvGS MeanTime", "results/rtx4090/e0-cpu-time-opencv/test_cpu_batchresize_x_split3D.csv", "OpenCV-CUDA"),
         ("FastNPP", "benchmark_CPU_NPP_vs_FastNPP_cu", "test_NPP_cpu_batchresize_x_split3D.csv", "NPP MeanTime", "FK MeanTime", "results/rtx4090/e0-cpu-time-npp/test_NPP_cpu_batchresize_x_split3D.csv", "NPP")]
READ2025 = {"cvGS": {2: 12, 12: 56, 22: 84, 32: 100.5, 42: 112}, "FastNPP": {2: 12, 12: 43, 22: 60, 32: 70, 42: 74.5}}


def read(path):
    rows = [l for l in open(path) if l.strip() and not l.startswith("#")]
    h, out = None, {}
    for r in csv.reader(rows):
        r = [x.strip() for x in r]
        if h is None:
            if r and r[0].lower().startswith("batch"):
                h = r
            continue
        if len(r) == len(h):
            try:
                d = dict(zip(h, map(float, r)))
                out[int(d[h[0]])] = d
            except ValueError:
                pass
    return out


def fit(xs, ys):
    mx, my = st.mean(xs), st.mean(ys)
    s = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return s, my - s * mx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="results/rtx4090/e0-cpu-time-history/test-C")
    a = ap.parse_args()
    L = ["# Test C, summary (scripts/e0_testc_analyse.py)", ""]
    for name, prog, fname, basecol, fusedcol, oldpath, basename in PROGS:
        data = {b: [read(os.path.join(a.dir, b, f"round{r}", prog, fname)) for r in (1, 2, 3)] for b in BUILDS}
        old = read(oldpath)
        batches = sorted(set.intersection(*[set(d) for b in BUILDS for d in data[b]]))
        med = lambda b, bt, col: st.median(d[bt][col] for d in data[b])
        spr = lambda b, bt, col: (max(d[bt][col] for d in data[b]) - min(d[bt][col] for d in data[b])) / med(b, bt, col) * 100
        L += [f"## {name} ({prog})", "", f"Fused host time (microseconds, median of 3 rounds) and speedup (column 'Mean Speedup', median of 3 rounds; the old single run of 2026-10-30 in the last columns); spread = (max - min) / median of the 3 rounds.", "",
              "| batch | fused Release | fused None | fused Debug | speedup Release | speedup None | speedup Debug | speedup old Release (2026-10-30) | Release/old fused | spread % R / N / D (speedup) |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
        for bt in batches:
            fu = {b: med(b, bt, fusedcol) * 1000 for b in BUILDS}
            sp = {b: med(b, bt, "Mean Speedup") for b in BUILDS}
            o = old.get(bt)
            L.append(f"| {bt} | {fu['Release']:.2f} | {fu['None']:.2f} | {fu['Debug']:.2f} | {sp['Release']:.1f} | {sp['None']:.1f} | {sp['Debug']:.1f} | " + (f"{o['Mean Speedup']:.1f} | {fu['Release'] / (o[fusedcol] * 1000):.3f}" if o else "| ") +
                     f" | {spr('Release', bt, 'Mean Speedup'):.1f} / {spr('None', bt, 'Mean Speedup'):.1f} / {spr('Debug', bt, 'Mean Speedup'):.1f} |")
        L += ["", "Slope and constant of the fused host time (microseconds per crop, microseconds):", "", "| build | slope batch 2..42 | constant (from 2..42) | slope 2..152 (least squares) | constant (least squares) |", "|---|---:|---:|---:|---:|"]
        for b in BUILDS:
            t2, t42 = med(b, 2, fusedcol) * 1000, med(b, 42, fusedcol) * 1000
            s42 = (t42 - t2) / 40
            s, c = fit(batches, [med(b, bt, fusedcol) * 1000 for bt in batches])
            L.append(f"| {b} | {s42:.4f} | {t2 - 2 * s42:.3f} | {s:.4f} | {c:.3f} |")
        L += ["", f"Control, the baseline column ({basename} host time per call, microseconds, median of 3 rounds) per build; ratio to Release:", "", f"| batch | {basename} Release | None | Debug | None/Release | Debug/Release |", "|---:|---:|---:|---:|---:|---:|"]
        for bt in batches:
            v = {b: med(b, bt, basecol) * 1000 for b in BUILDS}
            L.append(f"| {bt} | {v['Release']:.1f} | {v['None']:.1f} | {v['Debug']:.1f} | {v['None'] / v['Release']:.3f} | {v['Debug'] / v['Release']:.3f} |")
        ratios = [med('None', bt, basecol) / med('Release', bt, basecol) for bt in batches] + [med('Debug', bt, basecol) / med('Release', bt, basecol) for bt in batches]
        L += ["", f"The baseline column moves by {min(ratios):.3f} to {max(ratios):.3f} times the Release one over all batch sizes and both builds.", "",
              "Against the reading of the 2025 figure (coordinator, accuracy +-1 to +-2, NOT data): speedup None / Debug / Release / 2025 reading at batches 2, 12, 22, 32, 42:", ""]
        L += ["| batch | None | Debug | Release | 2025 reading |", "|---:|---:|---:|---:|---:|"] + [f"| {bt} | {med('None', bt, 'Mean Speedup'):.1f} | {med('Debug', bt, 'Mean Speedup'):.1f} | {med('Release', bt, 'Mean Speedup'):.1f} | {v} |" for bt, v in READ2025[name].items()]
        L += ["", f"None at batches 62 to 152: {', '.join(f'{bt}: {med('None', bt, 'Mean Speedup'):.0f}' for bt in batches if bt >= 62)}; Debug: {', '.join(f'{bt}: {med('Debug', bt, 'Mean Speedup'):.0f}' for bt in batches if bt >= 62)}; Release: {', '.join(f'{bt}: {med('Release', bt, 'Mean Speedup'):.0f}' for bt in batches if bt >= 62)}", ""]
    out = os.path.join(a.dir, "testC_summary.md")
    open(out, "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
