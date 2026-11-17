#!/usr/bin/env python3
"""The summary of the E18 box window on the RX 9060 (phase 2; written in phase 1, not run yet): per version and interval, the median of the
three quiet runs' means, with their min-max, and the speedups of FKL over RPP (the median of the three per-run ratios, with their min-max).
It is results/rtx4090/e18-stencil/scripts-used/e18_report.py with this program's file and versions.

  e18_box_report.py <window folder> [--runs quiet-run1,quiet-run2,quiet-run3]

Reads <run>/e18_box_GPU_AMD.csv (comment lines '#' skipped) and writes <window folder>/e18_box_summary.csv, e18_box_speedups.csv and
e18_box_report.md.
"""
import argparse
import csv
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--runs", default="quiet-run1,quiet-run2,quiet-run3")
args = ap.parse_args()
runs = args.runs.split(",")
data = []
for r in runs:
    with open(os.path.join(args.folder, r, "e18_box_GPU_AMD.csv")) as f:
        for d in csv.DictReader(line for line in f if not line.startswith("#")):
            d["run"] = r
            data.append(d)
order = list(dict.fromkeys(d["version"] for d in data if d["run"] == runs[0]))


def pick(run=None, version=None, interval=None):
    return [d for d in data if (run is None or d["run"] == run) and (version is None or d["version"] == version)
            and (interval is None or d["interval"] == interval)]


rows = []
for v in order:
    for interval in ("whole_call", "kernel_only"):
        s = pick(version=v, interval=interval)
        if len(s) != len(runs):
            raise SystemExit(f"{v} {interval}: {len(s)} rows for {len(runs)} runs")
        means = [float(d["mean_ms"]) for d in s]
        row = {"version": v, "interval": interval, "runs": len(means), "median_of_means_ms": statistics.median(means), "min_of_means_ms": min(means),
               "max_of_means_ms": max(means), "spread_pct": 100.0 * (max(means) - min(means)) / statistics.median(means),
               "min_of_mins_ms": min(float(d["min_ms"]) for d in s)}
        if interval == "kernel_only":
            row["host_prepare_median_ms"] = statistics.median(float(d["host_prepare_mean_ms"]) for d in s)
        rows.append(row)
cols = ["version", "interval", "runs", "median_of_means_ms", "min_of_means_ms", "max_of_means_ms", "spread_pct", "min_of_mins_ms", "host_prepare_median_ms"]
with open(os.path.join(args.folder, "e18_box_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
pairs = [("rpp", "fkl_box_halfpixel"), ("rpp", "fkl_half_once"), ("rpp_ltrb", "fkl_box_halfpixel"), ("rpp_ltrb", "fkl_half_once")]
sp = []
for base, fkl in pairs:
    for interval in ("whole_call", "kernel_only"):
        ratios = []
        for r in runs:
            b = float(pick(run=r, version=base, interval=interval)[0]["mean_ms"])
            f = float(pick(run=r, version=fkl, interval=interval)[0]["mean_ms"])
            ratios.append(b / f)
        sp.append({"baseline": base, "fkl": fkl, "interval": interval, "median_speedup": statistics.median(ratios), "min_speedup": min(ratios),
                   "max_speedup": max(ratios)})
with open(os.path.join(args.folder, "e18_box_speedups.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["baseline", "fkl", "interval", "median_speedup", "min_speedup", "max_speedup"])
    w.writeheader()
    w.writerows(sp)
md = ["| version | whole_call (ms) | kernel_only (ms) |", "|---|--:|--:|"]
for v in order:
    cell = {}
    for interval in ("whole_call", "kernel_only"):
        r = next(x for x in rows if x["version"] == v and x["interval"] == interval)
        cell[interval] = f"{r['median_of_means_ms']:.4f} ({r['min_of_means_ms']:.4f}-{r['max_of_means_ms']:.4f})"
    md.append(f"| {v} | {cell['whole_call']} | {cell['kernel_only']} |")
md += ["", "| baseline / FKL | whole_call | kernel_only |", "|---|--:|--:|"]
for base, fkl in pairs:
    cell = {}
    for interval in ("whole_call", "kernel_only"):
        s = next(x for x in sp if x["baseline"] == base and x["fkl"] == fkl and x["interval"] == interval)
        cell[interval] = f"{s['median_speedup']:.2f} ({s['min_speedup']:.2f}-{s['max_speedup']:.2f})"
    md.append(f"| {base} / {fkl} | {cell['whole_call']} | {cell['kernel_only']} |")
md.append("")
md.append(f"Largest spread of the three runs' means: {max(r['spread_pct'] for r in rows):.1f} %.")
with open(os.path.join(args.folder, "e18_box_report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
