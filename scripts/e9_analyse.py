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
"""E9 summary in the order of the four facts of the paper: (a) launches, (b) bit identity, (c) time of the mixed kernels against the single-library ones, (d) registers and SASS lines.

  python scripts/e9_analyse.py [--dir results/rtx4090/e9-cross-library] [--binary build/local-rtx4090/bin/e9_cross_library_cu]

Reads run{0,1,2}/e9_agreement.csv, run{0,1,2}/e9_timing.csv, nsys/kernels.txt; runs cuobjdump -arch sm_89 on the binary for (d).
Time: graph_ms is the GPU time per call (a CUDA graph of 100 calls, the standing GPU-bound number, 3 % flag); launch_ms is the plain launch loop (host and GPU). For each run the ratio mixed / single-library is
formed; the table gives its median over the 3 runs and the spread (max - min over the runs, in % points); 'exceeds' = the ratio differs from 1 by more than the flag (3 % for the GPU time, 5 % for the host-bound launch loop).
"""
import argparse
import csv
import re
import statistics as st
import subprocess

PAIRS = [("M1", "R1"), ("M2", "R2")]
RUNS = (0, 1, 2)


def rows(path):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(lines))


def sass(binary):
    out = subprocess.run(["cuobjdump", "-arch", "sm_89", "-sass", binary], capture_output=True, text=True).stdout
    res = subprocess.run(["cuobjdump", "-arch", "sm_89", "-res-usage", binary], capture_output=True, text=True).stdout
    regs = {}
    for blk in re.split(r"\n(?=\s*Function )", res):
        m = re.match(r"\s*Function (\S+):", blk)
        if m:
            r = re.search(r"REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", blk)
            regs[m.group(1)] = tuple(int(x) for x in r.groups())
    n, cur = {}, None
    for l in out.splitlines():
        m = re.match(r"\s+Function : (\S+)", l)
        if m:
            cur = m.group(1)
            n[cur] = 0
        elif cur and re.match(r"\s+/\*[0-9a-f]{4}\*/", l):
            n[cur] += 1
    found = {}
    for k, v in n.items():
        d = subprocess.run(["c++filt", k], capture_output=True, text=True).stdout
        if "launchTransformDPP_Kernel" not in d:
            continue
        cv_resize = "PlanePolicy)1" in d
        swap_np = "VectorReorderRT" in d
        if "short_listing" in d:
            name = "M1 short listing"
        elif cv_resize:
            name = "M1 (= long listing)" if swap_np else "R1"
        else:
            name = "R2" if swap_np else "M2"
        found[name] = (regs[k][0], v, regs[k][1], regs[k][2], regs[k][3])
    return found


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dir", default="results/rtx4090/e9-cross-library")
    ap.add_argument("--binary", default="build/local-rtx4090/bin/e9_cross_library_cu")
    a = ap.parse_args()
    out = ["# E9 summary (the four facts, in the order of the paper)", ""]
    timing = {r: rows(f"{a.dir}/run{r}/e9_timing.csv") for r in RUNS}
    agree = {r: rows(f"{a.dir}/run{r}/e9_agreement.csv") for r in RUNS}

    out += ["## (a) Launches", "", "| variant | kernel nodes per call, runs 0 / 1 / 2 |", "|---|---|"]
    for v in ("M1", "M2", "R1", "R2"):
        ks = [next(x["kernels_per_call"] for x in timing[r] if x["variant"] == v) for r in RUNS]
        out.append(f"| {v} | {' / '.join(ks)} |")
    try:
        k = open(f"{a.dir}/nsys/kernels.txt").read().strip().splitlines()
        out += ["", "nsys cross-check (one call of each of M1, M2, R1, R2 and of the two listings inside the profiler range): " + k[-1], ""]
        out += ["```"] + [l[:160] for l in k[:-1]] + ["```"]
    except OSError:
        out += ["", "nsys: no kernels.txt"]

    out += ["", "## (b) Bit identity (function `compare()` of npp/e9_cross_library.cu: memcmp of the 4-byte bit pattern of each output float + NaN count of both outputs; output pre-filled with the NaN pattern)", "",
            "| comparison | runs x seeds | all identical | elements differing (min..max over rows) | max abs difference (max) | NaN count (max of the two outputs) |", "|---|---:|:---:|---|---:|---:|"]
    names = []
    for x in agree[0]:
        if x["comparison"] not in names:
            names.append(x["comparison"])
    for nme in names:
        rs = [x for r in RUNS for x in agree[r] if x["comparison"] == nme]
        ident = all(int(x["identical"]) == 1 for x in rs)
        diff = [int(x["differing"]) for x in rs]
        mx = max(float(x["max_abs_difference"]) for x in rs)
        nan = max(max(int(x["nan_a"]), int(x["nan_b"])) for x in rs)
        out.append(f"| {nme} | {len(rs)} | {'YES' if ident else 'NO'} | {min(diff)}..{max(diff)} of {rs[0]['n']} | {mx:.6g} | {nan} |")

    out += ["", "## (c) Time per call (ms), mixed against the single-library kernel it shares its resize with; THE run is run 0; order of the variants rotated per run", ""]
    for col, label, bound, flag in (("graph_ms_median", "GPU time per call (CUDA graph of 100 calls)", "GPU-bound, flag 3 %", 0.03), ("launch_ms_median", "launch loop (host and GPU)", "host-bound (the launch costs about as much as the kernel), flag 5 %", 0.05)):
        out += [f"### {label}, {bound}", "", "| variant | run 0 | run 1 | run 2 | median | spread over runs % | within-run block range (max/min - 1, max over runs) % |", "|---|---:|---:|---:|---:|---:|---:|"]
        vals = {}
        for v in ("M1", "R1", "M2", "R2"):
            vs = [float(next(x[col] for x in timing[r] if x["variant"] == v)) for r in RUNS]
            vals[v] = vs
            lo = col.replace("median", "min")
            hi = col.replace("median", "max")
            rng = max(float(next(x[hi] for x in timing[r] if x["variant"] == v)) / float(next(x[lo] for x in timing[r] if x["variant"] == v)) - 1 for r in RUNS) * 100
            med = st.median(vs)
            out.append(f"| {v} | {vs[0]:.5f} | {vs[1]:.5f} | {vs[2]:.5f} | {med:.5f} | {(max(vs) - min(vs)) / med * 100:.1f} | {rng:.1f} |")
        out += ["", "| pair (mixed / single-library) | ratio run 0 | run 1 | run 2 | median | spread over runs (max - min, % points) | exceeds the flag (median) | runs beyond the flag |", "|---|---:|---:|---:|---:|---:|:---:|---:|"]
        for m, s in PAIRS:
            ra = [vals[m][i] / vals[s][i] for i in range(3)]
            med = st.median(ra)
            out.append(f"| {m} / {s} | {ra[0]:.4f} | {ra[1]:.4f} | {ra[2]:.4f} | {med:.4f} | {(max(ra) - min(ra)) * 100:.1f} | {'YES' if abs(med - 1) > flag else 'no'} | {sum(abs(x - 1) > flag for x in ra)} of 3 |")
        out.append("")

    out += ["## (d) Registers and SASS lines of the final binary (cuobjdump -arch sm_89), before the window / before #45: M1 34 / 552, R1 32 / 544, M2 33 / 544, R2 32 / 560; after #45 (compile time): M2 552, R2 568", "",
            "| kernel | registers | SASS lines | stack | shared | local |", "|---|---:|---:|---:|---:|---:|"]
    for k, (reg, n, stack, shared, local) in sorted(sass(a.binary).items()):
        out.append(f"| {k} | {reg} | {n} | {stack} | {shared} | {local} |")
    open(f"{a.dir}/e9_summary.md", "w").write("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
