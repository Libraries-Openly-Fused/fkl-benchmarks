#!/usr/bin/env python3
"""The summary of the E19 RX 9060 repeat check (nothing is run; no GPU).

  repeat_check_report.py <window folder> <E19 timing folder> <crop sweep timing folder> <output folder>

The window folder holds quiet-run1..3-A and quiet-run1..3-B (A = E19's e19_batched_box_hip, B = e19_crop_sweep_hip with E19_CROPS=30), plus
unused-A/B, rocprofv3-A/B and gpu_clocks.csv. fkl_loop is left out (Oscar, 2026-11-20).
Writes to the output folder:
  summary.csv  per program (A, B), row (fkl_batch, rpp) and interval: the median of the three quiet runs' means, their min and max, the spread
               ((max - min) / median), the min of the mins, and the sclk at the interval's boundaries (min-max over the three runs)
  bound.csv    per program and row: whole_call against kernel_only, GPU-bound when they differ by less than 3 % (the crop sweep's criterion)
  compare.csv  A against E19's window (results/rx9060/e19-batched-box/e19_batched_box_summary.csv), B against the crop sweep's N = 30
               (results/rx9060/e19-crop-sweep/summary/summary.csv), and B against A in this window: old, new, change, both spreads, whether
               the change exceeds the larger spread, and the flag above 3 %
  clock.csv    per run: the hwmon samples during its timing (sclk min / median / max, below 3.1 GHz, median power, max temperatures) and the sclk
               after the 2 s warm-up
  unused.csv   unused-A/B and rocprofv3-A/B against the quiet median of their program
  report.md    the tables in Markdown
The reading rule, fixed before the window: if A and B agree within their spreads, the change is not the program; if they differ beyond them, it
follows the program. No further cause is claimed.
"""
import csv
import os
import re
import statistics
import sys

win, e19_dir, sweep_dir, out = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
ROWS = ["fkl_batch", "rpp"]
INTERVALS = ["whole_call", "kernel_only"]
PROGRAMS = ["A", "B"]
QUIET = ["quiet-run1", "quiet-run2", "quiet-run3"]
FLAG_PCT = 3.0


def read_times(folder):
    with open(os.path.join(folder, "e19_batched_box_GPU_AMD.csv")) as f:
        return {(r["version"], r["interval"]): r for r in csv.DictReader(line for line in f if not line.startswith("#"))}


def write_csv(name, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(new, old):
    return 100.0 * (new - old) / old


times = {(p, q): read_times(os.path.join(win, f"{q}-{p}")) for p in PROGRAMS for q in QUIET + ["unused", "rocprofv3"]}
summary, stat = [], {}
for p in PROGRAMS:
    for row in ROWS:
        for interval in INTERVALS:
            s = [times[(p, q)][(row, interval)] for q in QUIET]
            means = [float(d["mean_ms"]) for d in s]
            m = statistics.median(means)
            spread = 100.0 * (max(means) - min(means)) / m
            stat[(p, row, interval)] = (m, min(means), max(means), spread)
            before = [float(d["sclk_before_mhz"]) for d in s]
            after = [float(d["sclk_after_mhz"]) for d in s]
            summary.append({"program": p, "row": row, "interval": interval, "median_of_means_ms": f"{m:.6f}", "min_of_means_ms": f"{min(means):.6f}",
                            "max_of_means_ms": f"{max(means):.6f}", "spread_pct": f"{spread:.2f}",
                            "min_of_mins_ms": f"{min(float(d['min_ms']) for d in s):.6f}",
                            "sclk_before_mhz": f"{min(before):.0f}-{max(before):.0f}", "sclk_after_mhz": f"{min(after):.0f}-{max(after):.0f}",
                            **{f"mean_ms_{q}": f"{float(d['mean_ms']):.6f}" for q, d in zip(QUIET, s)}})
write_csv("summary.csv", summary)

# GPU- or host-bound, the crop sweep's criterion (the RTX 4090 report's): GPU-bound when kernel_only and whole_call differ by less than 3 %
bound = []
for p in PROGRAMS:
    for row in ROWS:
        w, k = stat[(p, row, "whole_call")][0], stat[(p, row, "kernel_only")][0]
        bound.append({"program": p, "row": row, "whole_call_ms": f"{w:.6f}", "kernel_only_ms": f"{k:.6f}",
                      "kernel_only_vs_whole_call_pct": f"{pct(k, w):+.2f}", "bound": "GPU" if abs(pct(k, w)) < FLAG_PCT else "host"})
write_csv("bound.csv", bound)

with open(os.path.join(e19_dir, "e19_batched_box_summary.csv")) as f:
    e19 = {(r["version"], r["interval"]): (float(r["median_of_means_ms"]), float(r["spread_pct"])) for r in csv.DictReader(f)}
with open(os.path.join(sweep_dir, "summary", "summary.csv")) as f:
    sweep = {(r["row"], r["interval"]): (float(r["median_of_means_ms"]), float(r["spread_pct"])) for r in csv.DictReader(f) if r["crops"] == "30"}
compare = []
for label, old_of, new_p in (("A against E19's window", lambda r, i: e19[(r, i)], "A"),
                             ("B against the crop sweep's N = 30", lambda r, i: sweep[(r, i)], "B"),
                             ("B against A, this window", lambda r, i: (stat[("A", r, i)][0], stat[("A", r, i)][3]), "B")):
    for row in ROWS:
        for interval in INTERVALS:
            om, osp = old_of(row, interval)
            nm, nsp = stat[(new_p, row, interval)][0], stat[(new_p, row, interval)][3]
            ch = pct(nm, om)
            compare.append({"comparison": label, "row": row, "interval": interval, "old_ms": f"{om:.4f}", "new_ms": f"{nm:.4f}", "change_pct": f"{ch:+.1f}",
                            "spread_old_pct": f"{osp:.1f}", "spread_new_pct": f"{nsp:.1f}",
                            "exceeds_larger_spread": "yes" if abs(ch) > max(osp, nsp) else "no", "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("compare.csv", compare)

samples = []
with open(os.path.join(win, "gpu_clocks.csv")) as f:
    for d in csv.DictReader(f):
        try:
            samples.append((int(d["time_ns"]), int(d["sclk_hz"]) / 1e6, int(d["power_uw"]) / 1e6, int(d["temp_edge_mc"]) / 1e3,
                            int(d["temp_junction_mc"]) / 1e3, int(d["temp_mem_mc"]) / 1e3))
        except (TypeError, ValueError, KeyError):
            pass
clock = []
for q in ["unused"] + QUIET + ["rocprofv3"]:
    for p in PROGRAMS:
        text = open(os.path.join(win, f"{q}-{p}", "stdout.txt")).read()
        m = re.search(r"E19 timing \(realtime ns\): warm-up start (\d+), timing start (\d+), timing end (\d+)", text)
        w = re.search(r"sclk ([\d.]+) MHz before it, ([\d.]+) MHz after it", text)
        start, end = int(m.group(2)), int(m.group(3))
        inside = [x for x in samples if start <= x[0] <= end]
        sc = [x[1] for x in inside]
        clock.append({"run": f"{q}-{p}", "timing_seconds": f"{(end - start) / 1e9:.2f}", "samples": len(inside),
                      "sclk_min_mhz": f"{min(sc):.0f}" if sc else "", "sclk_median_mhz": f"{statistics.median(sc):.0f}" if sc else "",
                      "sclk_max_mhz": f"{max(sc):.0f}" if sc else "", "samples_below_3100": sum(1 for s in sc if s < 3100.0),
                      "power_median_w": f"{statistics.median(x[2] for x in inside):.0f}" if inside else "",
                      "temp_edge_max_c": f"{max(x[3] for x in inside):.0f}" if inside else "",
                      "temp_junction_max_c": f"{max(x[4] for x in inside):.0f}" if inside else "",
                      "temp_mem_max_c": f"{max(x[5] for x in inside):.0f}" if inside else "",
                      "sclk_before_warmup_mhz": f"{float(w.group(1)):.0f}", "sclk_after_warmup_mhz": f"{float(w.group(2)):.0f}"})
write_csv("clock.csv", clock)

unused = []
for q in ["unused", "rocprofv3"]:
    for p in PROGRAMS:
        for row in ROWS:
            for interval in INTERVALS:
                v = float(times[(p, q)][(row, interval)]["mean_ms"])
                ch = pct(v, stat[(p, row, interval)][0])
                unused.append({"run": f"{q}-{p}", "row": row, "interval": interval, "mean_ms": f"{v:.6f}", "quiet_median_ms": f"{stat[(p, row, interval)][0]:.6f}",
                               "change_pct": f"{ch:+.2f}", "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("unused.csv", unused)

md = ["# E19 RX 9060 repeat check: the summary (A = E19's binary, B = the crop sweep's at N = 30)", "",
      "## The times: the median of the three quiet runs' means (min-max), ms", "",
      "| program | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |", "|---|--:|--:|--:|--:|"]
for p in PROGRAMS:
    cells = [f"{stat[(p, r, i)][0]:.4f} ({stat[(p, r, i)][1]:.4f}-{stat[(p, r, i)][2]:.4f})" for r in ROWS for i in INTERVALS]
    md.append(f"| {p} | " + " | ".join(cells) + " |")
md += ["", "Spreads (%): " + "; ".join(f"{p} {r} {i} {stat[(p, r, i)][3]:.1f}" for p in PROGRAMS for r in ROWS for i in INTERVALS), "",
       "## The comparisons (old / new / change / spread old / new / exceeds the larger spread / > 3 %)", "",
       "| comparison | row, interval | old (ms) | new (ms) | change | spread old / new | exceeds? | > 3 %? |", "|---|---|--:|--:|--:|--:|---|---|"]
for c in compare:
    md.append(f"| {c['comparison']} | {c['row']} {c['interval']} | {c['old_ms']} | {c['new_ms']} | {c['change_pct']} % | {c['spread_old_pct']} / "
              f"{c['spread_new_pct']} % | {c['exceeds_larger_spread']} | {c['flag_above_3pct']} |")
md += ["", "## GPU- or host-bound (kernel_only against whole_call; GPU-bound below 3 %)", ""]
for b in bound:
    md.append(f"- {b['program']} {b['row']}: whole_call {b['whole_call_ms']} ms, kernel_only {b['kernel_only_ms']} ms "
              f"({b['kernel_only_vs_whole_call_pct']} %): {b['bound']}")
md += ["", "## The clock in each run's timing (hwmon every 100 ms)", "",
       "| run | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the warm-up |",
       "|---|--:|---|--:|--:|---|--:|"]
for c in clock:
    md.append(f"| {c['run']} | {c['samples']} | {c['sclk_min_mhz']} / {c['sclk_median_mhz']} / {c['sclk_max_mhz']} | {c['samples_below_3100']} | "
              f"{c['power_median_w']} | {c['temp_edge_max_c']} / {c['temp_junction_max_c']} / {c['temp_mem_max_c']} | {c['sclk_after_warmup_mhz']} |")
md += ["", "## The unused runs against their program's quiet median", ""]
for u in unused:
    md.append(f"- {u['run']} {u['row']} {u['interval']}: {u['mean_ms']} ms ({u['change_pct']} %){' FLAG' if u['flag_above_3pct'] == 'FLAG' else ''}")
with open(os.path.join(out, "report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
