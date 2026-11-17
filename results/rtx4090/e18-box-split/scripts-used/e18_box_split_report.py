#!/usr/bin/env python3
"""The summary of the E18 box split window, from the three quiet runs' e18_box_split.csv (CUDA-event means of 100 calls each).

  e18_box_split_report.py <window folder> --old <e18_box_summary.csv of the window e18-box> [--runs quiet-run1,quiet-run2,quiet-run3]

Writes to <window folder>:
  e18_box_split_summary.csv    per row and interval: the median of the three runs' means, their min and max, the spread ((max - min) / median),
                               the smallest single call, the median host preparation (kernel_only)
  e18_box_split_ratios.csv     per interval, the median of the three per-run ratios and their min-max:
                                 npp_box / fkl_box, npp_resize / fkl_resize, npp / fkl_trunc_halfpixel (time of the first over time of the second);
                                 (fkl_resize + fkl_box) / fkl_trunc_halfpixel and (npp_resize + npp_box) / npp (the two steps alone against the pipeline)
  e18_box_split_kernel_vs_whole.csv  per row: kernel_only mean over whole_call mean (the median of the three per-run ratios, min-max)
  e18_box_split_repeats.csv    the three repeated rows against the window e18-box: old and new medians, old and new spreads (max - min of the three
                               means), the change, whether the change exceeds the larger of the two spreads, and whether it exceeds 3 %
  e18_box_split_report.md      the same as tables
"""
import argparse
import csv
import os
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--old", required=True); ap.add_argument("--runs", default="quiet-run1,quiet-run2,quiet-run3")
args = ap.parse_args()
runs = args.runs.split(",")
US = 1000.0   # the CSVs are in ms; the report is in us


def read(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


data = []
for r in runs:
    for d in read(os.path.join(args.folder, r, "e18_box_split.csv")):
        d["run"] = r
        data.append(d)
order = list(dict.fromkeys(d["version"] for d in data if d["run"] == runs[0]))


def mean(run, version, interval):
    s = [d for d in data if d["run"] == run and d["version"] == version and d["interval"] == interval]
    if len(s) != 1:
        raise SystemExit(f"{run} {version} {interval}: {len(s)} rows")
    return float(s[0]["mean_ms"]) * US


summary = {}
rows = []
for v in order:
    for interval in ("whole_call", "kernel_only"):
        s = [d for d in data if d["version"] == v and d["interval"] == interval]
        means = [float(d["mean_ms"]) * US for d in s]
        if len(means) != len(runs):
            raise SystemExit(f"{v} {interval}: {len(means)} runs")
        med = statistics.median(means)
        row = {"version": v, "sampling": s[0]["sampling"], "interval": interval, "runs": len(means), "median_of_means_us": med,
               "min_of_means_us": min(means), "max_of_means_us": max(means), "spread_pct": 100.0 * (max(means) - min(means)) / med,
               "min_of_mins_us": min(float(d["min_ms"]) for d in s) * US,
               "host_prepare_median_us": statistics.median(float(d["host_prepare_mean_ms"]) for d in s) * US if interval == "kernel_only" else ""}
        rows.append(row)
        summary[(v, interval)] = row
cols = list(rows[0].keys())
with open(os.path.join(args.folder, "e18_box_split_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)

ratios = []
for interval in ("whole_call", "kernel_only"):
    defs = [("npp_box / fkl_box", lambda r: mean(r, "npp_box", interval) / mean(r, "fkl_box", interval)),
            ("npp_resize / fkl_resize", lambda r: mean(r, "npp_resize", interval) / mean(r, "fkl_resize", interval)),
            ("npp / fkl_trunc_halfpixel", lambda r: mean(r, "npp", interval) / mean(r, "fkl_trunc_halfpixel", interval)),
            ("(fkl_resize + fkl_box) / fkl_trunc_halfpixel",
             lambda r: (mean(r, "fkl_resize", interval) + mean(r, "fkl_box", interval)) / mean(r, "fkl_trunc_halfpixel", interval)),
            ("(npp_resize + npp_box) / npp", lambda r: (mean(r, "npp_resize", interval) + mean(r, "npp_box", interval)) / mean(r, "npp", interval))]
    for name, f in defs:
        x = [f(r) for r in runs]
        ratios.append({"ratio": name, "interval": interval, "median": statistics.median(x), "min": min(x), "max": max(x)})
with open(os.path.join(args.folder, "e18_box_split_ratios.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ratio", "interval", "median", "min", "max"])
    w.writeheader()
    w.writerows(ratios)

kw = []
for v in order:
    x = [mean(r, v, "kernel_only") / mean(r, v, "whole_call") for r in runs]
    kw.append({"version": v, "median_kernel_only_over_whole_call": statistics.median(x), "min": min(x), "max": max(x)})
with open(os.path.join(args.folder, "e18_box_split_kernel_vs_whole.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["version", "median_kernel_only_over_whole_call", "min", "max"])
    w.writeheader()
    w.writerows(kw)

old = {(d["version"], d["interval"]): d for d in read(args.old)}
rep = []
for v in ("fkl_box_asym", "fkl_trunc_halfpixel", "npp"):
    for interval in ("whole_call", "kernel_only"):
        o, n = old[(v, interval)], summary[(v, interval)]
        om = float(o["median_of_means_ms"]) * US
        ospread = (float(o["max_of_means_ms"]) - float(o["min_of_means_ms"])) * US
        nm = n["median_of_means_us"]
        nspread = n["max_of_means_us"] - n["min_of_means_us"]
        change = nm - om
        rep.append({"version": v, "interval": interval, "old_median_us": om, "old_min_us": float(o["min_of_means_ms"]) * US,
                    "old_max_us": float(o["max_of_means_ms"]) * US, "old_spread_us": ospread, "new_median_us": nm, "new_min_us": n["min_of_means_us"],
                    "new_max_us": n["max_of_means_us"], "new_spread_us": nspread, "change_us": change, "change_pct": 100.0 * change / om,
                    "exceeds_spread": int(abs(change) > max(ospread, nspread)), "above_3_pct": int(abs(100.0 * change / om) > 3.0)})
with open(os.path.join(args.folder, "e18_box_split_repeats.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rep[0].keys()))
    w.writeheader()
    w.writerows(rep)

md = ["| row | interval | median of the 3 means (us) | min-max of the means (us) | spread | host preparation (us, median) |", "|---|---|--:|--:|--:|--:|"]
for r in rows:
    hp = f"{r['host_prepare_median_us']:.3f}" if r["host_prepare_median_us"] != "" else ""
    md.append(f"| {r['version']} | {r['interval']} | {r['median_of_means_us']:.3f} | {r['min_of_means_us']:.3f}-{r['max_of_means_us']:.3f} | "
              f"{r['spread_pct']:.1f} % | {hp} |")
md += ["", "| ratio | interval | median of the 3 per-run ratios | min-max |", "|---|---|--:|--:|"]
for s in ratios:
    md.append(f"| {s['ratio']} | {s['interval']} | {s['median']:.3f} | {s['min']:.3f}-{s['max']:.3f} |")
md += ["", "| row | kernel_only / whole_call (median of the 3 per-run ratios) | min-max |", "|---|--:|--:|"]
for k in kw:
    md.append(f"| {k['version']} | {k['median_kernel_only_over_whole_call']:.4f} | {k['min']:.4f}-{k['max']:.4f} |")
md += ["", "| row | interval | old (window e18-box) median, min-max (us) | new median, min-max (us) | old / new spread (us) | change | exceeds the spread | above 3 % |",
       "|---|---|--:|--:|--:|--:|---|---|"]
for r in rep:
    md.append(f"| {r['version']} | {r['interval']} | {r['old_median_us']:.3f}, {r['old_min_us']:.3f}-{r['old_max_us']:.3f} | {r['new_median_us']:.3f}, "
              f"{r['new_min_us']:.3f}-{r['new_max_us']:.3f} | {r['old_spread_us']:.3f} / {r['new_spread_us']:.3f} | {r['change_us']:+.3f} us "
              f"({r['change_pct']:+.1f} %) | {'yes' if r['exceeds_spread'] else 'no'} | {'yes' if r['above_3_pct'] else 'no'} |")
largest = max(rows, key=lambda r: r["spread_pct"])
md += ["", f"Largest spread: {largest['spread_pct']:.1f} % ({largest['version']}, {largest['interval']})."]
open(os.path.join(args.folder, "e18_box_split_report.md"), "w").write("\n".join(md) + "\n")
print("\n".join(md))
