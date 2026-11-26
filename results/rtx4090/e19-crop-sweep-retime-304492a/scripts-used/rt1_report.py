#!/usr/bin/env python3
"""RT-1: the E19 crop sweep timed again, A (976e42b, FKL b10fc66) and B (main, FKL 304492a) alternating in one window, against the record
results/rtx4090/e19-crop-sweep/ (old). CPU only: reads the CSVs.

  rt1_report.py <old record folder> <window folder> [--out report.txt] [--csv table.csv] [--md table.md]

Reads <folder>/<sweep>/N<NNN>/e19_batched_box.csv (mean_ms of 100 calls per row and interval): the old record's quiet-run1..3, and the
window's quiet-run1-A..quiet-run3-A (A) and quiet-run1-B..quiet-run3-B (B); the window's warmup-A and warmup-B for reference only.
fkl_loop is in the raw CSVs only: no table and no ratio here uses it (Oscar 2026-11-20).
For every N, row (fkl_batch, opencv_cuda, npp) and interval (whole_call, kernel_only):
  old, A, B   the median of the 3 quiet means in us, and the spread (max - min) / median
  A vs old    the change of the medians in %, whether it exceeds the larger of the two spreads, whether the two 3-run ranges overlap
  B vs A      the same
  bound       whole_call / kernel_only of the medians of the newer side: within 3 % of 1 the number is GPU-bound, otherwise host-bound
  flag        a GPU-bound change above 3 %, or a host-bound change above 5 % (the standing report form); the cause is "not shown"
Ratios: opencv_cuda / fkl_batch and npp / fkl_batch per sweep, median, for old, A and B (informational).
"""
import argparse
import csv
import glob
import os
import re
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("old"); ap.add_argument("new"); ap.add_argument("--out"); ap.add_argument("--csv"); ap.add_argument("--md")
a = ap.parse_args()
ROWS, INTERVALS = ["fkl_batch", "opencv_cuda", "npp"], ["whole_call", "kernel_only"]
CROPS = [1] + list(range(10, 151, 10))


def read(folder, sweep):
    out = {}
    for d in sorted(glob.glob(os.path.join(folder, sweep, "N*"))):
        n = int(re.search(r"N(\d+)$", d).group(1))
        p = os.path.join(d, "e19_batched_box.csv")
        lines = open(p).read().splitlines()
        if any(l.startswith("# INCOMPLETE") for l in lines):
            raise SystemExit(f"{p}: INCOMPLETE")
        for r in csv.DictReader(l for l in lines if not l.startswith("#")):
            if int(r["crops"]) != n:
                raise SystemExit(f"{p}: crops {r['crops']} is not {n}")
            out[(n, r["version"], r["interval"])] = float(r["mean_ms"]) * 1000.0
    return out


def side(folder, sweeps):
    runs = [read(folder, s) for s in sweeps]
    for s, r in zip(sweeps, runs):
        missing = [(n, v, i) for n in CROPS for v in ROWS for i in INTERVALS if (n, v, i) not in r]
        if missing:
            raise SystemExit(f"{folder}/{s}: missing {missing[:4]}")
    return runs


OLD = side(a.old, ["quiet-run1", "quiet-run2", "quiet-run3"])
A = side(a.new, ["quiet-run1-A", "quiet-run2-A", "quiet-run3-A"])
B = side(a.new, ["quiet-run1-B", "quiet-run2-B", "quiet-run3-B"])
WARM = {s: read(a.new, f"warmup-{s}") for s in "AB" if os.path.isdir(os.path.join(a.new, f"warmup-{s}"))}


def stats(runs, key):
    v = [r[key] for r in runs]
    med = statistics.median(v)
    return dict(med=med, lo=min(v), hi=max(v), spread=(max(v) - min(v)) / med * 100.0, values=v)


def compare(x, y, bound_ratio):
    ch = (y["med"] - x["med"]) / x["med"] * 100.0
    larger = max(x["spread"], y["spread"])
    exceeds = abs(ch) > larger
    overlap = not (y["lo"] > x["hi"] or y["hi"] < x["lo"])
    gpu = abs(bound_ratio - 1.0) <= 0.03
    flag = abs(ch) > (3.0 if gpu else 5.0)
    return dict(change=ch, exceeds=exceeds, overlap=overlap, bound="GPU-bound" if gpu else "host-bound", flag=flag)


rows, lines = [], []
for n in CROPS:
    for v in ROWS:
        for i in INTERVALS:
            k = (n, v, i)
            o, sa, sb = stats(OLD, k), stats(A, k), stats(B, k)
            # the bound of each side from its own whole_call / kernel_only medians
            bnd = {}
            for nm, runs in (("A", A), ("B", B)):
                bnd[nm] = statistics.median(r[(n, v, "whole_call")] for r in runs) / statistics.median(r[(n, v, "kernel_only")] for r in runs)
            ao, ba = compare(o, sa, bnd["A"]), compare(sa, sb, bnd["B"])
            rows.append(dict(crops=n, row=v, interval=i,
                             old_us=o["med"], old_min=o["lo"], old_max=o["hi"], old_spread_pct=o["spread"],
                             A_us=sa["med"], A_min=sa["lo"], A_max=sa["hi"], A_spread_pct=sa["spread"],
                             B_us=sb["med"], B_min=sb["lo"], B_max=sb["hi"], B_spread_pct=sb["spread"],
                             A_vs_old_pct=ao["change"], A_vs_old_exceeds=ao["exceeds"], A_vs_old_overlap=ao["overlap"], A_bound=ao["bound"], A_vs_old_flag=ao["flag"],
                             B_vs_A_pct=ba["change"], B_vs_A_exceeds=ba["exceeds"], B_vs_A_overlap=ba["overlap"], B_bound=ba["bound"], B_vs_A_flag=ba["flag"],
                             warmup_A_vs_A_pct=((WARM["A"][k] - sa["med"]) / sa["med"] * 100.0) if "A" in WARM else "",
                             warmup_B_vs_B_pct=((WARM["B"][k] - sb["med"]) / sb["med"] * 100.0) if "B" in WARM else ""))


def yn(b):
    return "yes" if b else "no"


md = ["| N | row, interval | old, us (spread) | A, us (spread) | B, us (spread) | A vs old | exceeds? | B vs A | exceeds? | bound (B) | flag |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    flag = [x for x, f in (("A vs old", r["A_vs_old_flag"]), ("B vs A", r["B_vs_A_flag"])) if f]
    md.append(f"| {r['crops']} | {r['row']} {r['interval']} | {r['old_us']:.2f} ({r['old_spread_pct']:.2f} %) | {r['A_us']:.2f} ({r['A_spread_pct']:.2f} %) | "
              f"{r['B_us']:.2f} ({r['B_spread_pct']:.2f} %) | {r['A_vs_old_pct']:+.2f} % | {yn(r['A_vs_old_exceeds'])}{'' if r['A_vs_old_overlap'] else ' (no overlap)'} | "
              f"{r['B_vs_A_pct']:+.2f} % | {yn(r['B_vs_A_exceeds'])}{'' if r['B_vs_A_overlap'] else ' (no overlap)'} | {r['B_bound']} | {', '.join(flag) or '-'} |")
# summary
nf_ao = [r for r in rows if r["A_vs_old_flag"]]; nf_ba = [r for r in rows if r["B_vs_A_flag"]]
ex_ba = [r for r in rows if r["B_vs_A_exceeds"]]
lines.append(f"E19 crop sweep re-time: {len(rows)} numbers (16 N x 3 rows x 2 intervals); old = {a.old}, window = {a.new}")
for lbl, key in (("A vs old", "A_vs_old_pct"), ("B vs A", "B_vs_A_pct")):
    for v in ROWS:
        for i in INTERVALS:
            cs = [r[key] for r in rows if r["row"] == v and r["interval"] == i]
            cs10 = [r[key] for r in rows if r["row"] == v and r["interval"] == i and r["crops"] >= 10]
            lines.append(f"  {lbl:8} {v:11} {i:11}: {min(cs):+.2f} % to {max(cs):+.2f} % (N >= 10: {min(cs10):+.2f} % to {max(cs10):+.2f} %)")
lines.append(f"  flags (GPU-bound above 3 %, host-bound above 5 %): A vs old {len(nf_ao)}, B vs A {len(nf_ba)}")
for r in nf_ao:
    lines.append(f"    A vs old N {r['crops']} {r['row']} {r['interval']}: {r['A_vs_old_pct']:+.2f} % ({r['A_bound']}; spreads {r['old_spread_pct']:.2f} %, {r['A_spread_pct']:.2f} %)")
for r in nf_ba:
    lines.append(f"    B vs A   N {r['crops']} {r['row']} {r['interval']}: {r['B_vs_A_pct']:+.2f} % ({r['B_bound']}; spreads {r['A_spread_pct']:.2f} %, {r['B_spread_pct']:.2f} %)")
lines.append(f"  B vs A changes that exceed the larger spread: {len(ex_ba)} of {len(rows)}; of them, ranges not overlapping: {sum(1 for r in ex_ba if not r['B_vs_A_overlap'])}")
# ratios
lines.append("  ratios (median of the per-sweep ratios): N, interval: opencv_cuda / fkl_batch old, A, B | npp / fkl_batch old, A, B")
for n in CROPS:
    for i in INTERVALS:
        def rat(runs, v):
            return statistics.median(r[(n, v, i)] / r[(n, "fkl_batch", i)] for r in runs)
        lines.append(f"    {n:3} {i:11}: {rat(OLD, 'opencv_cuda'):.3f} {rat(A, 'opencv_cuda'):.3f} {rat(B, 'opencv_cuda'):.3f} | "
                     f"{rat(OLD, 'npp'):.3f} {rat(A, 'npp'):.3f} {rat(B, 'npp'):.3f}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
if a.md:
    open(a.md, "w").write("\n".join(md) + "\n")
if a.csv:
    with open(a.csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
