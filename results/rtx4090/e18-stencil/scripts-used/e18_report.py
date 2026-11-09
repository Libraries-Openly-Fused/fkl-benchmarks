#!/usr/bin/env python3
"""The summary of the E18 window: per version and interval, the median of the three quiet runs' means, with their min-max, and the speedups of
FKL over the baseline of its convention (the median of the three per-run ratios, with their min-max).

  e18_report.py <window folder> [--runs quiet-run1,quiet-run2,quiet-run3]

Reads <run>/e18_stencil.csv (comment lines '#' skipped) and writes <window folder>/e18_summary.csv, e18_speedups.csv and e18_report.md.
"""
import argparse
import csv
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--runs", default="quiet-run1,quiet-run2,quiet-run3")
args = ap.parse_args()
runs = args.runs.split(",")
data = []   # dicts with the run name
for r in runs:
    with open(os.path.join(args.folder, r, "e18_stencil.csv")) as f:
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
        means = [float(d["mean_ms"]) for d in s]
        row = {"version": v, "sampling": s[0]["sampling"], "interval": interval, "runs": len(means),
               "median_of_means_ms": statistics.median(means), "min_of_means_ms": min(means), "max_of_means_ms": max(means),
               "spread_pct": 100.0 * (max(means) - min(means)) / statistics.median(means), "min_of_mins_ms": min(float(d["min_ms"]) for d in s)}
        if interval == "kernel_only":
            row["host_prepare_median_ms"] = statistics.median(float(d["host_prepare_mean_ms"]) for d in s)
        rows.append(row)
cols = ["version", "sampling", "interval", "runs", "median_of_means_ms", "min_of_means_ms", "max_of_means_ms", "spread_pct", "min_of_mins_ms",
        "host_prepare_median_ms"]
with open(os.path.join(args.folder, "e18_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
pairs = [("opencv_cuda", "fkl_stencil_asym"), ("npp", "fkl_stencil_halfpixel"),
         ("opencv_cuda", "fkl_stencil_asym_once"), ("npp", "fkl_stencil_halfpixel_once")]
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
with open(os.path.join(args.folder, "e18_speedups.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["baseline", "fkl", "interval", "median_speedup", "min_speedup", "max_speedup"])
    w.writeheader()
    w.writerows(sp)
md = ["| version | interval | median of the 3 means (ms) | min-max of the means (ms) | spread | host preparation (ms, median) |",
      "|---|---|--:|--:|--:|--:|"]
for r in rows:
    hp = f"{r['host_prepare_median_ms']:.4f}" if "host_prepare_median_ms" in r else ""
    md.append(f"| {r['version']} | {r['interval']} | {r['median_of_means_ms']:.4f} | {r['min_of_means_ms']:.4f}-{r['max_of_means_ms']:.4f} | "
              f"{r['spread_pct']:.1f} % | {hp} |")
md += ["", "| baseline / FKL | interval | median speedup | min-max |", "|---|---|--:|--:|"]
for s in sp:
    md.append(f"| {s['baseline']} / {s['fkl']} | {s['interval']} | {s['median_speedup']:.2f} | {s['min_speedup']:.2f}-{s['max_speedup']:.2f} |")
open(os.path.join(args.folder, "e18_report.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
