#!/usr/bin/env python3
"""Report of the cleanup window E (2026-11-08): the cleaned E2 FKL program with its passes in the order of the Python driver (warm, L2-flushed, kernel-only;
PR #28), against the old order of windows C and D (warm, kernel-only, L2-flushed), for every k; the SM clock of every run; the regenerated
e2_per_k_summary.csv (the FKL rows from this window, the Python rows from windows C and D). Reads only the result folders; writes the markdown report given
as the first argument, a CSV of its numbers next to it, and cleanup-window-e/cleaned/e2_per_k_summary.csv."""
import bisect
import csv
import datetime
import os
import re
import statistics
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
C, D, E = R + "/cleanup-window-c", R + "/cleanup-window-d", R + "/cleanup-window-e"
OUT = sys.argv[1]


def rows(path, expect):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    out = list(csv.DictReader(lines))
    assert len(out) == expect, (path, len(out))
    assert not any("INCOMPLETE" in l for l in open(path)), path
    return out


def by_ops(dirs):
    return [{int(r["ops"]): r for r in rows(f"{d}/e2_fkl_chain.csv", 6)} for d in dirs]


def st(values):
    med = statistics.median(values)
    return med, (max(values) - min(values)) / med * 100


def fmt(v):
    return f"{v:.4g}"


OLD_C = by_ops([f"{C}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)])
OLD_D = by_ops([f"{D}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)])
NEW = by_ops([f"{E}/cleaned/e2-fkl/run{i}" for i in (1, 2, 3)])
REF_C = by_ops([f"{C}/reference/e2-fkl/run{i}" for i in (1, 2, 3)])
REF_D = by_ops([f"{D}/reference/e2-fkl/run{i}" for i in (1, 2, 3)])
PUB = by_ops([f"{R}/e2-chain", f"{R}/e2-chain-run3", f"{R}/e2-chain-run4"])
report = ["# Cleanup window E, 2026-11-08: the E2 FKL program in the order of the Python driver", "",
          "The cleaned FKL program of windows C and D timed every k warm, kernel-only, L2-flushed (old order). Since PR #28 it times warm, L2-flushed, kernel-only, "
          "as triton/e2_chain.py (new order). The reference program has no kernel-only pass: its flushed pass follows its warm pass, as in the new order. "
          "Times in ms, the median of three runs, with the spread ((largest - smallest) / median) in brackets; change = new order against the old order of the "
          "window named.", ""]
CSV_ROWS = []
for col, title in (("fkl_flushed_mean_ms", "L2-flushed"), ("fkl_kernel_mean_ms", "kernel-only"), ("fkl_mean_ms", "warm")):
    report += [f"## {title}", "",
               "| k | old order, window C | old order, window D | new order, window E | new against C | new against D |" + (" reference C / D | published |" if col != "fkl_kernel_mean_ms" else ""),
               "|---|---|---|---|---|---|" + ("---|---|" if col != "fkl_kernel_mean_ms" else "")]
    for k in sorted(NEW[0]):
        c, d, n = st([float(x[k][col]) for x in OLD_C]), st([float(x[k][col]) for x in OLD_D]), st([float(x[k][col]) for x in NEW])
        line = (f"| {k} | {fmt(c[0])} ({c[1]:.1f} %) | {fmt(d[0])} ({d[1]:.1f} %) | {fmt(n[0])} ({n[1]:.1f} %) | {(n[0] - c[0]) / c[0] * 100:+.1f} % | "
                f"{(n[0] - d[0]) / d[0] * 100:+.1f} % |")
        if col != "fkl_kernel_mean_ms":
            rc, rd, p = st([float(x[k][col]) for x in REF_C]), st([float(x[k][col]) for x in REF_D]), st([float(x[k][col]) for x in PUB])
            line += f" {fmt(rc[0])} / {fmt(rd[0])} | {fmt(p[0])} ({p[1]:.1f} %) |"
        report.append(line)
        CSV_ROWS.append([title, k, c[0], c[1], d[0], d[1], n[0], n[1]])
    report.append("")

# FKL against the Python driver at k = 5,002 (the Python rows of window D: the driver's order has not changed)
PY = [{(int(r["ops"]), r["version"]): r for r in rows(f"{D}/cleaned/e2-python-k5002/run{i}/e2_chain.csv", 7)} for i in (1, 2, 3)]
PP = [{(int(r["ops"]), r["version"]): r for r in rows(f"{d}/e2_chain_large.csv", 7)} for d in (f"{R}/e2-chain", f"{R}/e2-chain-run3", f"{R}/e2-chain-run4")]
fn = {c: st([float(x[5002][c]) for x in NEW])[0] for c in ("fkl_mean_ms", "fkl_flushed_mean_ms", "fkl_kernel_mean_ms")}
fo = {c: st([float(x[5002][c]) for x in OLD_D])[0] for c in ("fkl_mean_ms", "fkl_flushed_mean_ms", "fkl_kernel_mean_ms")}
fp = {c: st([float(x[5002][c]) for x in PUB])[0] for c in ("fkl_mean_ms", "fkl_flushed_mean_ms")}
report += ["## FKL against the Python driver at k = 5,002 (FKL's time over the version's: > 1 FKL is slower)", "",
           "The Python rows are window D's (the driver's order is unchanged); published: the large runs 2, 3, 4.", "",
           "| version | flushed: published | flushed: old order (D) | flushed: new order (E) | kernel-only: old order (D) | kernel-only: new order (E) |", "|---|---|---|---|---|---|"]
for v in ("triton_hand", "inductor_default", "inductor_wide", "inductor_graphs", "eager", "eager_graphs"):
    pf = statistics.median(float(x[(5002, v)]["gpu_flushed_mean_ms"]) for x in PP)
    df = statistics.median(float(x[(5002, v)]["gpu_flushed_mean_ms"]) for x in PY)
    dk = statistics.median(float(x[(5002, v)]["gpu_kernel_mean_ms"]) for x in PY)
    report.append(f"| {v} | {fp['fkl_flushed_mean_ms'] / pf:.3f} | {fo['fkl_flushed_mean_ms'] / df:.3f} | {fn['fkl_flushed_mean_ms'] / df:.3f} | "
                  f"{fo['fkl_kernel_mean_ms'] / dk:.3f} | {fn['fkl_kernel_mean_ms'] / dk:.3f} |")
report.append("")

# the clock of every run of this window
SAMPLES = []
for line in open(f"{E}/gpu_clocks.csv"):
    f = [x.strip() for x in line.split(",")]
    if len(f) >= 7:
        t = datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f").astimezone()
        SAMPLES.append((t.timestamp() * 1000, float(f[1].split()[0]), float(f[5].split()[0]), float(f[6]), f[3]))
SAMPLES.sort()
TIMES = [s[0] for s in SAMPLES]
report += ["## The SM clock of every run (the 100 ms sampler, between the run's start and end)", "",
           "| run | start | s | SM MHz median | SM MHz min | SM MHz max | power W median | power W max | temperature max | pstates |", "|---|---|---|---|---|---|---|---|---|---|"]
for i in (1, 2, 3):
    txt = open(f"{E}/cleaned/e2-fkl/run{i}/load.txt").read()
    t0 = datetime.datetime.fromisoformat(re.search(r"start (\S+)", txt).group(1)).timestamp() * 1000
    t1 = datetime.datetime.fromisoformat(re.search(r"end (\S+)", txt).group(1)).timestamp() * 1000
    s = SAMPLES[bisect.bisect_left(TIMES, t0):bisect.bisect_right(TIMES, t1)]
    sm, pw = [x[1] for x in s], [x[2] for x in s]
    report.append(f"| cleaned/e2-fkl/run{i} | {datetime.datetime.fromtimestamp(t0 / 1000).strftime('%H:%M:%S')} | {(t1 - t0) / 1000:.0f} | {statistics.median(sm):.0f} | "
                  f"{min(sm):.0f} | {max(sm):.0f} | {statistics.median(pw):.0f} | {max(pw):.0f} | {max(x[3] for x in s):.0f} | {' '.join(sorted({x[4] for x in s}))} |")
report.append("")

# agreement
hashes = [sorted(l.split()[0] + " " + l.split()[1] for l in open(f"{E}/cleaned/e2-fkl/run{i}/outputs-sha256.txt") if l.strip()) for i in (1, 2, 3)]
pub_h = sorted(l.split()[0] + " " + l.split()[1] for l in open(f"{C}/cleaned/e2-fkl/run1/outputs-sha256.txt") if l.strip())
order_note = all("the passes of every k in the order warm, L2-flushed, kernel-only" in open(f"{E}/cleaned/e2-fkl/run{i}/e2_fkl_chain.csv").read() for i in (1, 2, 3))
report += ["## Agreement", "",
           f"- The outputs of the three runs: the same six hashes in every run: {len({tuple(h) for h in hashes}) == 1}; equal to windows C and D and to the published outputs: {hashes[0] == pub_h}",
           f"- The CSV note of every run gives the new order: {order_note}", ""]

# the regenerated e2_per_k_summary.csv: FKL from this window, the Python versions from windows C and D
VERSIONS = ("fkl", "triton_hand", "inductor_default", "inductor_wide", "inductor_graphs", "eager", "eager_graphs")
PYC = [{(int(r["ops"]), r["version"]): r for r in rows(f"{C}/cleaned/e2-python/run{i}/e2_chain.csv", 35)} for i in (1, 2, 3)]
summary = []
for k in (2, 10, 100, 500, 5002):
    for v in VERSIONS:
        row = {"k": k, "version": v}
        if v == "fkl":
            cols = (("fkl_mean_ms", "warm"), ("fkl_kernel_mean_ms", "kernel_only"), ("fkl_flushed_mean_ms", "flushed"))
            sets, src = NEW, "cleanup-window-e/cleaned/e2-fkl/run1..3 (the FKL program, passes in the order warm, flushed, kernel-only)"
            get = lambda x, c: float(x[k][c])
        else:
            cols = (("gpu_mean_ms", "warm"), ("gpu_kernel_mean_ms", "kernel_only"), ("gpu_flushed_mean_ms", "flushed"))
            sets = PYC if k != 5002 else PY
            src = ("cleanup-window-c/cleaned/e2-python/run1..3" if k != 5002 else "cleanup-window-d/cleaned/e2-python-k5002/run1..3") + " (triton/e2_chain.py, passes in the order warm, flushed, kernel-only)"
            for x in sets:
                assert x[(k, v)]["status"] == "ok", (k, v)
            get = lambda x, c: float(x[(k, v)][c])
        for c, name in cols:
            m, s = st([get(x, c) for x in sets])
            row[f"{name}_ms_median"] = f"{m:.6g}"
            row[f"{name}_spread_pct"] = f"{s:.2f}"
        row["source"] = src
        summary.append(row)
os.makedirs(f"{E}/cleaned", exist_ok=True)
with open(f"{E}/cleaned/e2_per_k_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
report += ["## e2_per_k_summary.csv", "",
           f"`cleaned/e2_per_k_summary.csv` ({len(summary)} rows) supersedes `../cleanup-window-d/cleaned/e2_per_k_summary.csv` for the paper. Every row names its source:",
           "the fkl rows come from this window's three runs; the Python rows come from window C (k = 2, 10, 100, 500) and window D (k = 5,002). Columns: warm "
           "with the launch, warm kernel-only and L2-flushed, each the median of three runs with the spread.", ""]
FIND = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cleanup_e_findings.md")
if os.path.exists(FIND):
    report[2:2] = [l.rstrip("\n") for l in open(FIND)] + [""]
open(OUT, "w").write("\n".join(report) + "\n")
with open(OUT.replace(".md", ".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["pass", "k", "old_order_c_median_ms", "old_order_c_spread_pct", "old_order_d_median_ms", "old_order_d_spread_pct", "new_order_e_median_ms", "new_order_e_spread_pct"])
    w.writerows(CSV_ROWS)
print(f"report written; e2_per_k_summary.csv: {len(summary)} rows")
