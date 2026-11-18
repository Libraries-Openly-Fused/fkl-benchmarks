#!/usr/bin/env python3
"""The E19 phase 3 timing window's summary from its CSVs (nothing is run).

  e19_report.py <window folder> [--runs quiet-run1,quiet-run2,quiet-run3] [--old <summary.csv of an earlier window>] [--out report.txt]
                [--csv summary.csv]

Reads <window folder>/<run>/e19_batched_box.csv of the quiet runs (mean_ms of 100 calls per row and interval). Per row and interval: the 3 run
means in us, their median, min and max, and the spread (max - min) / median. Per interval, ratios computed in every run (from that run's
means), with the median, min and max of the 3 per-run ratios: fkl_batch, opencv_cuda and npp over fkl_loop; fkl_loop, opencv_cuda and npp
over fkl_batch. Also the warmup and nsys runs' means, for reference only (not in the medians).

--old: the summary CSV of an earlier window (e19_report.py's --csv of results/rtx4090/e19-batched-box/). For every row and interval in both:
the old median, the new median, the change in %, both windows' spreads, whether the change exceeds the larger of the two spreads, whether the
new 3-run range overlaps the old one, and whether the change is above 3 % (the flag for a GPU-bound number). Every interval here is reported as
GPU-bound when whole_call and kernel_only of that row differ by less than 3 % in this window (the GPU, not the host's enqueue, sets whole_call).
"""
import argparse
import csv
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("folder")
ap.add_argument("--runs", default="quiet-run1,quiet-run2,quiet-run3")
ap.add_argument("--old", default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--csv", default=None)
args = ap.parse_args()
ROWS = ["fkl_loop", "fkl_batch", "opencv_cuda", "npp"]
INTERVALS = ["whole_call", "kernel_only"]
RATIOS = [("fkl_batch", "fkl_loop"), ("opencv_cuda", "fkl_loop"), ("npp", "fkl_loop"),
          ("fkl_loop", "fkl_batch"), ("opencv_cuda", "fkl_batch"), ("npp", "fkl_batch")]


def read(run):
    with open(os.path.join(args.folder, run, "e19_batched_box.csv")) as f:
        rows = list(csv.DictReader(l for l in f if not l.startswith("#")))
    return {(r["version"], r["interval"]): float(r["mean_ms"]) * 1000.0 for r in rows}


runs = args.runs.split(",")
data = {run: read(run) for run in runs}
lines, out = [], []
summary = {}
lines.append(f"quiet runs: {', '.join(runs)}; every value is the mean of 100 calls of one frame (30 crops), in us")
for interval in INTERVALS:
    lines.append(f"{interval}:")
    for row in ROWS:
        v = [data[run][(row, interval)] for run in runs]
        med = statistics.median(v)
        spread = (max(v) - min(v)) / med
        summary[(row, interval)] = (med, min(v), max(v), 100 * spread)
        lines.append(f"  {row}: median {med:.2f} (min {min(v):.2f}, max {max(v):.2f}; runs {' / '.join(f'{x:.2f}' for x in v)}); spread {100 * spread:.2f} %")
        out.append({"kind": "time_us", "row": row, "interval": interval, "median": med, "min": min(v), "max": max(v), "spread_pct": 100 * spread,
                    **{run: x for run, x in zip(runs, v)}})
    for num, den in RATIOS:
        r = [data[run][(num, interval)] / data[run][(den, interval)] for run in runs]
        med = statistics.median(r)
        lines.append(f"  {num} / {den}: median {med:.3f} (min {min(r):.3f}, max {max(r):.3f}; runs {' / '.join(f'{x:.3f}' for x in r)})")
        out.append({"kind": "ratio", "row": f"{num}/{den}", "interval": interval, "median": med, "min": min(r), "max": max(r), "spread_pct": "",
                    **{run: x for run, x in zip(runs, r)}})
lines.append("whole_call / kernel_only per row (medians): " + "; ".join(
    f"{row} {summary[(row, 'whole_call')][0] / summary[(row, 'kernel_only')][0]:.4f}" for row in ROWS))
for extra in ("warmup", "nsys"):
    if os.path.exists(os.path.join(args.folder, extra, "e19_batched_box.csv")):
        d = read(extra)
        lines.append(f"{extra} (not used): " + "; ".join(f"{row} {interval} {d[(row, interval)]:.2f}" for interval in INTERVALS for row in ROWS))
if args.old:
    with open(args.old) as f:
        old = {(r["row"], r["interval"]): r for r in csv.DictReader(f) if r["kind"] == "time_us"}
    lines.append(f"repeats against {args.old}: old median / new median / change / spread old, new / exceeds the larger spread? / ranges overlap? / above 3 %? / bound")
    for interval in INTERVALS:
        for row in ROWS:
            if (row, interval) not in old:
                continue
            o = old[(row, interval)]
            om, omin, omax, ospread = float(o["median"]), float(o["min"]), float(o["max"]), float(o["spread_pct"])
            nm, nmin, nmax, nspread = summary[(row, interval)]
            change = 100 * (nm - om) / om
            exceeds = abs(change) > max(ospread, nspread)
            overlap = nmin <= omax and omin <= nmax
            ratio = summary[(row, "whole_call")][0] / summary[(row, "kernel_only")][0]
            bound = "GPU-bound" if abs(ratio - 1) < 0.03 else f"host share in whole_call (whole_call / kernel_only {ratio:.3f})"
            lines.append(f"  {row} {interval}: {om:.2f} / {nm:.2f} us / {change:+.2f} % / {ospread:.2f} %, {nspread:.2f} % / "
                         f"{'EXCEEDS' if exceeds else 'within the spread'} / {'overlap' if overlap else 'NO overlap'} / "
                         f"{'ABOVE 3 %' if abs(change) > 3 else 'below 3 %'} / {bound}")
            out.append({"kind": "repeat_change_pct", "row": row, "interval": interval, "median": nm, "min": nmin, "max": nmax, "spread_pct": nspread,
                        "old_median": om, "old_spread_pct": ospread, "change_pct": change})
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "row", "interval", "median", "min", "max", "spread_pct"] + runs + ["old_median", "old_spread_pct", "change_pct"])
        w.writeheader()
        for r in out:
            w.writerow(r)
