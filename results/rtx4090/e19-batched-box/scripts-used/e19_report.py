#!/usr/bin/env python3
"""The E19 timing window's summary from its CSVs (nothing is run).

  e19_report.py <window folder> [--runs quiet-run1,quiet-run2,quiet-run3] [--out report.txt] [--csv summary.csv]

Reads <window folder>/<run>/e19_batched_box.csv of the quiet runs (mean_ms of 100 calls per row and interval). Per row and interval: the 3 run
means in us, their median, min and max, and the spread (max - min) / median. Per interval: the ratios opencv_cuda / fkl_loop and
npp / fkl_loop, each computed in every run (from that run's means), with the median, min and max of the 3 per-run ratios. Also the warmup and
nsys runs' means, for reference only (not in the medians).
"""
import argparse
import csv
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--runs", default="quiet-run1,quiet-run2,quiet-run3")
ap.add_argument("--out", default=None)
ap.add_argument("--csv", default=None)
args = ap.parse_args()
ROWS = ["fkl_loop", "opencv_cuda", "npp"]
INTERVALS = ["whole_call", "kernel_only"]


def read(run):
    with open(os.path.join(args.folder, run, "e19_batched_box.csv")) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return {(r["version"], r["interval"]): float(r["mean_ms"]) * 1000.0 for r in rows}


runs = args.runs.split(",")
data = {run: read(run) for run in runs}
lines, out = [], []
lines.append(f"quiet runs: {', '.join(runs)}; every value is the mean of 100 calls of one frame (30 crops), in us")
for interval in INTERVALS:
    lines.append(f"{interval}:")
    for row in ROWS:
        v = [data[run][(row, interval)] for run in runs]
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med
        lines.append(f"  {row}: median {med:.2f} (min {min(v):.2f}, max {max(v):.2f}; runs {' / '.join(f'{x:.2f}' for x in v)}); spread {100 * spread:.2f} %")
        out.append({"kind": "time_us", "row": row, "interval": interval, "median": med, "min": min(v), "max": max(v), "spread_pct": 100 * spread,
                    **{run: x for run, x in zip(runs, v)}})
    for base in ("opencv_cuda", "npp"):
        r = [data[run][(base, interval)] / data[run][("fkl_loop", interval)] for run in runs]
        med = statistics.median(r)
        lines.append(f"  {base} / fkl_loop: median {med:.3f} (min {min(r):.3f}, max {max(r):.3f}; runs {' / '.join(f'{x:.3f}' for x in r)})")
        out.append({"kind": "ratio", "row": f"{base}/fkl_loop", "interval": interval, "median": med, "min": min(r), "max": max(r), "spread_pct": "",
                    **{run: x for run, x in zip(runs, r)}})
for extra in ("warmup", "nsys"):
    if os.path.exists(os.path.join(args.folder, extra, "e19_batched_box.csv")):
        d = read(extra)
        lines.append(f"{extra} (not used): " + "; ".join(f"{row} {interval} {d[(row, interval)]:.2f}" for interval in INTERVALS for row in ROWS))
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "row", "interval", "median", "min", "max", "spread_pct"] + runs)
        w.writeheader()
        for r in out:
            w.writerow(r)
