#!/usr/bin/env python3
"""The summary of the E18 box window on the RX 9060 (phase 2; written in phase 1, not run yet): per version and interval, the median of the
three quiet runs' means, with their min-max, and the speedups of FKL over RPP (the median of the three per-run ratios, with their min-max).
It is results/rtx4090/e18-stencil/scripts-used/e18_report.py with this program's file and versions.

  e18_box_report.py <window folder> [--runs quiet-run1,quiet-run2,quiet-run3]

Reads <run>/e18_box_GPU_AMD.csv (comment lines '#' skipped) and writes <window folder>/e18_box_summary.csv, e18_box_speedups.csv and
e18_box_report.md.
The clock: the summary gives the lowest sclk at the interval boundaries (the columns sclk_before_mhz and sclk_after_mhz) of every version and
interval. e18_box_clock.csv gives, per run, the sclk samples of <window folder>/gpu_clocks.csv (time_ns,sclk_hz,mclk_hz) between the run's
timing start and end (its stdout.txt line "E18 timing (realtime ns): ..."). The criterion is at least SCLK_MIN_MHZ (3100) everywhere.
"""
import argparse
import csv
import os
import re
import statistics

SCLK_MIN_MHZ = 3100.0

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
        if all(d.get("sclk_before_mhz") not in (None, "") for d in s):
            row["sclk_min_at_boundaries_mhz"] = min(min(float(d["sclk_before_mhz"]), float(d["sclk_after_mhz"])) for d in s)
        rows.append(row)
cols = ["version", "interval", "runs", "median_of_means_ms", "min_of_means_ms", "max_of_means_ms", "spread_pct", "min_of_mins_ms", "host_prepare_median_ms",
        "sclk_min_at_boundaries_mhz"]
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
# The clock at the interval boundaries, and in gpu_clocks.csv during the timing of every run
boundary = [r["sclk_min_at_boundaries_mhz"] for r in rows if "sclk_min_at_boundaries_mhz" in r]
if boundary:
    md.append(f"Lowest sclk at an interval boundary: {min(boundary):.0f} MHz "
              f"({'at least' if min(boundary) >= SCLK_MIN_MHZ else 'BELOW'} {SCLK_MIN_MHZ:.0f} MHz).")
clock_path = os.path.join(args.folder, "gpu_clocks.csv")
if os.path.exists(clock_path):
    samples = []
    with open(clock_path) as f:
        for d in csv.DictReader(f):
            try:
                samples.append((int(d["time_ns"]), int(d["sclk_hz"]) / 1e6))
            except (TypeError, ValueError):
                pass
    clock_rows = []
    for r in runs + [x for x in ("rocprofv3",) if os.path.isdir(os.path.join(args.folder, x))]:
        text = open(os.path.join(args.folder, r, "stdout.txt")).read()
        m = re.search(r"E18 timing \(realtime ns\): warm-up start (\d+), timing start (\d+), timing end (\d+)", text)
        if not m:
            clock_rows.append({"run": r, "samples": 0})
            continue
        start, end = int(m.group(2)), int(m.group(3))
        inside = [s for t, s in samples if start <= t <= end]
        clock_rows.append({"run": r, "timing_seconds": (end - start) / 1e9, "samples": len(inside), "sclk_min_mhz": min(inside) if inside else "",
                           "sclk_median_mhz": statistics.median(inside) if inside else "", "sclk_max_mhz": max(inside) if inside else "",
                           "samples_below_min": sum(1 for s in inside if s < SCLK_MIN_MHZ)})
    with open(os.path.join(args.folder, "e18_box_clock.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run", "timing_seconds", "samples", "sclk_min_mhz", "sclk_median_mhz", "sclk_max_mhz", "samples_below_min"])
        w.writeheader()
        w.writerows(clock_rows)
    md.append("")
    md.append(f"sclk in gpu_clocks.csv during the timing of each run (criterion: at least {SCLK_MIN_MHZ:.0f} MHz):")
    for c in clock_rows:
        if not c["samples"]:
            md.append(f"- {c['run']}: no samples in its timing")
            continue
        md.append(f"- {c['run']}: {c['samples']} samples in {c['timing_seconds']:.2f} s, min {c['sclk_min_mhz']:.0f}, median {c['sclk_median_mhz']:.0f}, "
                  f"max {c['sclk_max_mhz']:.0f} MHz; below {SCLK_MIN_MHZ:.0f}: {c['samples_below_min']}")
with open(os.path.join(args.folder, "e18_box_report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
