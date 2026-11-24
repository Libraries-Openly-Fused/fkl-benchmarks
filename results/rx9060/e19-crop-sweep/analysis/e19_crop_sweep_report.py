#!/usr/bin/env python3
"""The summary of the E19 crop sweep timing window on the RX 9060 (nothing is run; no GPU).

  e19_crop_sweep_report.py <window folder> <E19 timing folder (results/rx9060/e19-batched-box)> <output folder>

The window folder holds warmup/, quiet-run1..3/ and rocprofv3/, each with nNNN/ (N = 1, 10, ..., 150), gpu_clocks.csv and window-console.txt.
fkl_loop is out of the comparison (Oscar, 2026-11-20): it is in the raw CSVs only, and in no table or ratio written here.
Writes to the output folder:
  summary.csv     per N, row (fkl_batch, rpp) and interval: the median of the three quiet runs' means, their min and max, the spread
                  ((max - min) / median), the min of the mins, the median host preparation time (kernel_only), and the lowest sclk at the
                  interval's boundaries in the three runs
  ratios.csv      per N and interval: rpp / fkl_batch in each quiet run (from that run's means), and their median, min and max
  bound.csv       per N and row: kernel_only / whole_call of the medians; GPU-bound when they differ by less than 3 % (the GPU, not the
                  host's enqueue, sets whole_call), host-bound otherwise
  clock.csv       per N and quiet run: the hwmon samples (gpu_clocks.csv, every 100 ms) between the run's timing start and end (its stdout
                  line "E19 timing (realtime ns)"): sclk min / median / max and the samples below 3.1 GHz, the median power, the largest
                  edge, junction and memory temperatures; the sclk before and after the 2 s warm-up spin
  boundaries.csv  per N, row and interval: the sclk read before and after the interval (the CSV columns), min-max over the three quiet runs
  repeat.csv      N = 30 against the E19 timing window (its e19_batched_box_summary.csv), fkl_batch and rpp, both intervals: old, new, change,
                  both spreads, whether the change exceeds the larger spread, and the flag above 3 %
  unused.csv      the warm-up and rocprofv3 sweeps' means against the quiet median, per N, row and interval; flagged above 3 %
  waits.csv       the idle wait before every run (window-console.txt)
  report.md       the tables above in Markdown
"""
import csv
import os
import re
import statistics
import sys

win, old_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
SWEEP = [1] + list(range(10, 151, 10))
QUIET = ["quiet-run1", "quiet-run2", "quiet-run3"]
UNUSED = ["warmup", "rocprofv3"]
ROWS = ["fkl_batch", "rpp"]
INTERVALS = ["whole_call", "kernel_only"]
SCLK_MIN_MHZ = 3100.0
FLAG_PCT = 3.0


def folder(run, n):
    return os.path.join(win, run, f"n{n:03d}")


def read_times(run, n):
    with open(os.path.join(folder(run, n), "e19_batched_box_GPU_AMD.csv")) as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    for r in rows:
        if int(r["crops"]) != n:
            raise SystemExit(f"{run} n{n:03d}: the crops column is {r['crops']}")
    return {(r["version"], r["interval"]): r for r in rows}


times = {(run, n): read_times(run, n) for run in QUIET + UNUSED for n in SWEEP}


def write_csv(name, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(new, old):
    return 100.0 * (new - old) / old


# ---- summary, boundaries
summary, boundaries = [], []
med = {}
for n in SWEEP:
    for row in ROWS:
        for interval in INTERVALS:
            s = [times[(run, n)][(row, interval)] for run in QUIET]
            means = [float(d["mean_ms"]) for d in s]
            m = statistics.median(means)
            med[(n, row, interval)] = m
            spread = 100.0 * (max(means) - min(means)) / m
            before = [float(d["sclk_before_mhz"]) for d in s]
            after = [float(d["sclk_after_mhz"]) for d in s]
            summary.append({"crops": n, "row": row, "interval": interval, "median_of_means_ms": f"{m:.6f}", "min_of_means_ms": f"{min(means):.6f}",
                            "max_of_means_ms": f"{max(means):.6f}", "spread_pct": f"{spread:.2f}",
                            "min_of_mins_ms": f"{min(float(d['min_ms']) for d in s):.6f}",
                            "host_prepare_median_ms": f"{statistics.median(float(d['host_prepare_mean_ms']) for d in s):.6f}" if interval == "kernel_only" else "",
                            "sclk_min_at_boundaries_mhz": f"{min(before + after):.0f}"})
            boundaries.append({"crops": n, "row": row, "interval": interval, "sclk_before_min_mhz": f"{min(before):.0f}", "sclk_before_max_mhz": f"{max(before):.0f}",
                               "sclk_after_min_mhz": f"{min(after):.0f}", "sclk_after_max_mhz": f"{max(after):.0f}"})
write_csv("summary.csv", summary)
write_csv("boundaries.csv", boundaries)
spread_of = {(int(r["crops"]), r["row"], r["interval"]): float(r["spread_pct"]) for r in summary}
range_of = {(int(r["crops"]), r["row"], r["interval"]): (float(r["min_of_means_ms"]), float(r["max_of_means_ms"])) for r in summary}

# ---- ratios
ratios = []
ratio_of = {}
for n in SWEEP:
    for interval in INTERVALS:
        per_run = [float(times[(run, n)][("rpp", interval)]["mean_ms"]) / float(times[(run, n)][("fkl_batch", interval)]["mean_ms"]) for run in QUIET]
        ratio_of[(n, interval)] = per_run
        ratios.append({"crops": n, "interval": interval, **{f"rpp_over_fkl_batch_{run}": f"{r:.4f}" for run, r in zip(QUIET, per_run)},
                       "median": f"{statistics.median(per_run):.4f}", "min": f"{min(per_run):.4f}", "max": f"{max(per_run):.4f}"})
write_csv("ratios.csv", ratios)

# ---- GPU- or host-bound
bound = []
for n in SWEEP:
    for row in ROWS:
        w, k = med[(n, row, "whole_call")], med[(n, row, "kernel_only")]
        bound.append({"crops": n, "row": row, "whole_call_ms": f"{w:.6f}", "kernel_only_ms": f"{k:.6f}", "kernel_only_vs_whole_call_pct": f"{pct(k, w):+.2f}",
                      "bound": "GPU" if abs(pct(k, w)) < FLAG_PCT else "host"})
write_csv("bound.csv", bound)

# ---- clock
samples = []
with open(os.path.join(win, "gpu_clocks.csv")) as f:
    for d in csv.DictReader(f):
        try:
            samples.append((int(d["time_ns"]), int(d["sclk_hz"]) / 1e6, int(d["power_uw"]) / 1e6, int(d["temp_edge_mc"]) / 1e3,
                            int(d["temp_junction_mc"]) / 1e3, int(d["temp_mem_mc"]) / 1e3))
        except (TypeError, ValueError, KeyError):
            pass
clock = []
for n in SWEEP:
    for run in QUIET + UNUSED:
        text = open(os.path.join(folder(run, n), "stdout.txt")).read()
        m = re.search(r"E19 timing \(realtime ns\): warm-up start (\d+), timing start (\d+), timing end (\d+)", text)
        w = re.search(r"sclk ([\d.]+) MHz before it, ([\d.]+) MHz after it", text)
        start, end = int(m.group(2)), int(m.group(3))
        inside = [x for x in samples if start <= x[0] <= end]
        sc = [x[1] for x in inside]
        clock.append({"crops": n, "run": run, "used": "yes" if run in QUIET else "no", "timing_seconds": f"{(end - start) / 1e9:.2f}", "samples": len(inside),
                      "sclk_min_mhz": f"{min(sc):.0f}" if sc else "", "sclk_median_mhz": f"{statistics.median(sc):.0f}" if sc else "",
                      "sclk_max_mhz": f"{max(sc):.0f}" if sc else "", "samples_below_3100": sum(1 for s in sc if s < SCLK_MIN_MHZ),
                      "power_median_w": f"{statistics.median(x[2] for x in inside):.0f}" if inside else "",
                      "temp_edge_max_c": f"{max(x[3] for x in inside):.0f}" if inside else "",
                      "temp_junction_max_c": f"{max(x[4] for x in inside):.0f}" if inside else "",
                      "temp_mem_max_c": f"{max(x[5] for x in inside):.0f}" if inside else "",
                      "sclk_before_warmup_mhz": f"{float(w.group(1)):.0f}", "sclk_after_warmup_mhz": f"{float(w.group(2)):.0f}"})
write_csv("clock.csv", clock)

# ---- the N = 30 repeat against the E19 timing window
with open(os.path.join(old_dir, "e19_batched_box_summary.csv")) as f:
    old = {(r["version"], r["interval"]): r for r in csv.DictReader(f)}
repeat = []
for row in ROWS:
    for interval in INTERVALS:
        o = old[(row, interval)]
        om, osp = float(o["median_of_means_ms"]), float(o["spread_pct"])
        nm, nsp = med[(30, row, interval)], spread_of[(30, row, interval)]
        ch = pct(nm, om)
        repeat.append({"row": row, "interval": interval, "old_ms": f"{om:.4f}", "new_ms": f"{nm:.4f}", "change_pct": f"{ch:+.1f}", "spread_old_pct": f"{osp:.1f}",
                       "spread_new_pct": f"{nsp:.1f}", "exceeds_larger_spread": "yes" if abs(ch) > max(osp, nsp) else "no",
                       "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("repeat.csv", repeat)

# ---- the unused runs against the quiet medians
unused = []
for run in UNUSED:
    for n in SWEEP:
        for row in ROWS:
            for interval in INTERVALS:
                v = float(times[(run, n)][(row, interval)]["mean_ms"])
                ch = pct(v, med[(n, row, interval)])
                lo, hi = range_of[(n, row, interval)]
                unused.append({"run": run, "crops": n, "row": row, "interval": interval, "mean_ms": f"{v:.6f}", "quiet_median_ms": f"{med[(n, row, interval)]:.6f}",
                               "change_pct": f"{ch:+.2f}", "inside_quiet_range": "yes" if lo <= v <= hi else "no",
                               "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("unused.csv", unused)

# ---- the idle waits
waits = []
current = None
for line in open(os.path.join(win, "window-console.txt")):
    m = re.match(r"  idle after (\d+) s wait; loadavg ([\d.]+) ([\d.]+) ([\d.]+)", line)
    if m:
        current = (int(m.group(1)), m.group(2))
        continue
    m = re.match(r"# (\S+) (\S+/n\d{3}) ", line)
    if m and current is not None and "exit" not in line:
        waits.append({"run": m.group(2), "start": m.group(1), "idle_wait_s": current[0], "load_1min_at_start": current[1]})
        current = None
write_csv("waits.csv", waits)

# ---- report.md
md = ["# E19 crop sweep, RX 9060: the timing summary (fkl_loop is in the raw CSVs only)", "",
      "## The times: the median of the three quiet runs' means (min-max), ms", "",
      "| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |", "|--:|--:|--:|--:|--:|"]
for n in SWEEP:
    cells = []
    for row in ROWS:
        for interval in INTERVALS:
            lo, hi = range_of[(n, row, interval)]
            cells.append(f"{med[(n, row, interval)]:.4f} ({lo:.4f}-{hi:.4f})")
    md.append(f"| {n} | " + " | ".join(cells) + " |")
md += ["", f"The largest spread of the three runs' means: {max(float(r['spread_pct']) for r in summary):.1f} % "
       f"({max(summary, key=lambda r: float(r['spread_pct']))['row']}, {max(summary, key=lambda r: float(r['spread_pct']))['interval']}, "
       f"N = {max(summary, key=lambda r: float(r['spread_pct']))['crops']}).", "",
       "## rpp / fkl_batch: the median of the three per-run ratios (min-max)", "", "| N | whole_call | kernel_only |", "|--:|--:|--:|"]
for n in SWEEP:
    c = [f"{statistics.median(ratio_of[(n, i)]):.2f} ({min(ratio_of[(n, i)]):.2f}-{max(ratio_of[(n, i)]):.2f})" for i in INTERVALS]
    md.append(f"| {n} | {c[0]} | {c[1]} |")
md += ["", "## GPU- or host-bound: kernel_only against whole_call of the medians (GPU-bound below 3 %)", "",
       "| N | fkl_batch | rpp |", "|--:|--:|--:|"]
for n in SWEEP:
    c = [f"{b['kernel_only_vs_whole_call_pct']} % ({b['bound']})" for b in bound if b["crops"] == n]
    md.append(f"| {n} | {c[0]} | {c[1]} |")
md += ["", "## The clock, power and temperature in the quiet runs' timing (hwmon every 100 ms)", "",
       "| N | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the 2 s warm-up (MHz) |",
       "|--:|--:|---|--:|--:|---|---|"]
for n in SWEEP:
    c = [x for x in clock if x["crops"] == n and x["used"] == "yes"]
    sc_min = min(float(x["sclk_min_mhz"]) for x in c)
    sc_max = max(float(x["sclk_max_mhz"]) for x in c)
    sc_med = "/".join(x["sclk_median_mhz"] for x in c)
    md.append(f"| {n} | {sum(x['samples'] for x in c)} | {sc_min:.0f} / {sc_med} / {sc_max:.0f} | {sum(x['samples_below_3100'] for x in c)} | "
              f"{'/'.join(x['power_median_w'] for x in c)} | {max(int(x['temp_edge_max_c']) for x in c)} / {max(int(x['temp_junction_max_c']) for x in c)} / "
              f"{max(int(x['temp_mem_max_c']) for x in c)} | {'/'.join(x['sclk_after_warmup_mhz'] for x in c)} |")
md += ["", "## sclk at the interval boundaries (before-after, min-max over the three quiet runs, MHz)", "",
       "| N | fkl_batch whole_call | fkl_batch kernel_only | rpp whole_call | rpp kernel_only |", "|--:|---|---|---|---|"]
for n in SWEEP:
    c = []
    for row in ROWS:
        for interval in INTERVALS:
            b = next(x for x in boundaries if x["crops"] == n and x["row"] == row and x["interval"] == interval)
            c.append(f"{b['sclk_before_min_mhz']}-{b['sclk_before_max_mhz']} / {b['sclk_after_min_mhz']}-{b['sclk_after_max_mhz']}")
    md.append(f"| {n} | " + " | ".join(c) + " |")
md += ["", "## The N = 30 repeat against results/rx9060/e19-batched-box/", "",
       "| row, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | > 3 %? |", "|---|--:|--:|--:|--:|---|---|"]
for r in repeat:
    md.append(f"| {r['row']} {r['interval']} | {r['old_ms']} | {r['new_ms']} | {r['change_pct']} % | {r['spread_old_pct']} / {r['spread_new_pct']} % | "
              f"{r['exceeds_larger_spread']} | {r['flag_above_3pct']} |")
flagged = [u for u in unused if u["flag_above_3pct"] == "FLAG"]
md += ["", f"## The unused runs (warm-up, rocprofv3) against the quiet medians: {len(flagged)} of {len(unused)} values above 3 %", ""]
for run in UNUSED:
    us = [u for u in unused if u["run"] == run]
    ch = [float(u["change_pct"]) for u in us]
    md.append(f"- {run}: change {min(ch):+.1f} to {max(ch):+.1f} %; above 3 %: {sum(1 for u in us if u['flag_above_3pct'] == 'FLAG')} of {len(us)}")
for u in flagged:
    md.append(f"  - FLAG {u['run']} N = {u['crops']} {u['row']} {u['interval']}: {u['mean_ms']} ms against {u['quiet_median_ms']} ({u['change_pct']} %)")
for run in UNUSED:
    c = [x for x in clock if x["run"] == run]
    low = [x for x in c if x["samples_below_3100"]]
    md.append(f"- {run}: samples below 3.1 GHz in the timing: {sum(x['samples_below_3100'] for x in c)} "
              f"({', '.join('N = %d: %d' % (x['crops'], x['samples_below_3100']) for x in low) or 'none'})")
w = [x for x in waits if x["idle_wait_s"]]
md += ["", f"## The idle waits: {len(w)} of {len(waits)} runs started after a wait", ""]
for x in w:
    md.append(f"- {x['run']}: {x['idle_wait_s']} s")
with open(os.path.join(out, "report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
