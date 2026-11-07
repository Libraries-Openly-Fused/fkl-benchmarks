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
"""Does the order in which the variants are timed change their times? (results/rtx4090/measurement-checks)

  python scripts/order_compare.py results/rtx4090/measurement-checks [--threshold 2]

For each program the same variants were timed in the normal order and with the order reversed (E1: E1_ORDER=reverse; E2: E2_ORDER=reverse and the versions of the Python driver
reversed; E4: every variant alone in its own process, in the reverse order, after 10 s of idle GPU). The table gives, per program, the median over all (point, variant) of the absolute difference between the
two orders, the largest, and every entry whose time differs by more than the threshold (percent). The time compared is the mean of the program's own measurement.
"""
import csv
import glob
import os
import statistics as st
import sys


def rd(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def report(name, pairs, thr):
    """pairs: {(point, variant): (normal, reversed)}"""
    devs = {k: 100 * (b / a - 1) for k, (a, b) in pairs.items() if a > 0}
    ab = sorted(abs(d) for d in devs.values())
    over = sorted(((k, d) for k, d in devs.items() if abs(d) > thr), key=lambda kd: -abs(kd[1]))
    print(f"{name}: {len(devs)} (point, variant) entries; median |difference| {st.median(ab):.2f} %, 90th percentile {ab[int(0.9 * (len(ab) - 1))]:.2f} %, largest {ab[-1]:.2f} %; {len(over)} above {thr} %")
    for (point, variant), d in over[:14]:
        a, b = pairs[(point, variant)]
        print(f"    {point} {variant}: normal {a:.5g}, reversed {b:.5g} ({d:+.1f} %)")
    return devs


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "results/rtx4090/measurement-checks"
    thr = 2.0
    if "--threshold" in sys.argv:
        thr = float(sys.argv[sys.argv.index("--threshold") + 1])
    lines = []
    # E4: normal run against every variant alone
    p = os.path.join(d, "e4", "normal", "e4_preprocessing.csv")
    if os.path.exists(p):
        normal = {(f"{x['experiment']} batch {x['batch']} src {x['src_w']}", x["version"]): float(x["gpu_mean_ms"]) for x in rd(p)}
        alone = {}
        for q in sorted(glob.glob(os.path.join(d, "e4", "alone_*", "e4_preprocessing.csv"))):
            for x in rd(q):
                alone[(f"{x['experiment']} batch {x['batch']} src {x['src_w']}", x["version"])] = float(x["gpu_mean_ms"])
        report("E4 (all variants in one process against every variant alone in its own process, reverse order, 10 s idle before each)", {k: (normal[k], alone[k]) for k in normal if k in alone}, thr)
    # E1
    for prog, key in (("e1_vf_chain_cu", "ops"), ("e1_hf_batch_cu", "batch"), ("e1_preproc_cu", "batch")):
        a, b = os.path.join(d, "e1", f"{prog}_normal"), os.path.join(d, "e1", f"{prog}_reverse")
        fa, fb = glob.glob(os.path.join(a, "*.csv")), glob.glob(os.path.join(b, "*.csv"))
        if not fa or not fb:
            continue
        ra, rb = {x[key]: x for x in rd(sorted(fa)[0])}, {x[key]: x for x in rd(sorted(fb)[0])}
        names = list(next(iter(ra.values())))
        # GPU times (as the launch runs and kernel-only); the host times of the preparation (and the totals that contain them) separately, against the
        # 5 % threshold of host-bound numbers
        cols = [c for c in names if "mean" in c.lower() and not any(w in c.lower() for w in ("over", "prep", "total"))]
        host = [c for c in names if "prep_mean" in c.lower()]
        report(f"E1 {prog} (columns {', '.join(cols)})", {(f"{key} {k}", c): (float(ra[k][c]), float(rb[k][c])) for k in ra if k in rb for c in cols}, thr)
        if host:
            report(f"E1 {prog}, host time of the preparation (columns {', '.join(host)}; threshold 5 %)",
                   {(f"{key} {k}", c): (float(ra[k][c]), float(rb[k][c])) for k in ra if k in rb for c in host}, 5.0)
    # E2 FKL program and the Python driver
    fa, fb = os.path.join(d, "e2", "fkl_normal", "e2_fkl_chain.csv"), os.path.join(d, "e2", "fkl_reverse", "e2_fkl_chain.csv")
    if os.path.exists(fa) and os.path.exists(fb):
        ra, rb = {x["ops"]: x for x in rd(fa)}, {x["ops"]: x for x in rd(fb)}
        gpu = [c for c in ("fkl_mean_ms", "fkl_flushed_mean_ms", "fkl_kernel_mean_ms") if c in next(iter(ra.values()))]
        report(f"E2 FKL program ({', '.join(gpu)})", {(f"k {k}", c): (float(ra[k][c]), float(rb[k][c])) for k in ra if k in rb for c in gpu}, thr)
        host = [c for c in ("fkl_prep_mean_ms", "fkl_flushed_prep_mean_ms") if c in next(iter(ra.values()))]
        if host:
            report(f"E2 FKL program, host time of the preparation ({', '.join(host)}; threshold 5 %)",
                   {(f"k {k}", c): (float(ra[k][c]), float(rb[k][c])) for k in ra if k in rb for c in host}, 5.0)
    pa, pb = os.path.join(d, "e2", "python_normal", "e2_chain.csv"), os.path.join(d, "e2", "python_reverse", "e2_chain.csv")
    if os.path.exists(pa) and os.path.exists(pb):
        ra, rb = {(x["version"], x["ops"]): x for x in rd(pa) if x["status"] == "ok"}, {(x["version"], x["ops"]): x for x in rd(pb) if x["status"] == "ok"}
        report("E2 Python driver (gpu_mean_ms, gpu_flushed_mean_ms and gpu_kernel_mean_ms; versions and chain lengths reversed)", {(f"k {k[1]}", f"{k[0]} {c}"): (float(ra[k][c]), float(rb[k][c])) for k in ra if k in rb and k[0] != "fkl" for c in ("gpu_mean_ms", "gpu_flushed_mean_ms", "gpu_kernel_mean_ms") if ra[k].get(c) not in (None, "") and rb[k].get(c) not in (None, "")}, thr)


if __name__ == "__main__":
    main()
