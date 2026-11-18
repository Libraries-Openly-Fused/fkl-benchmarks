#!/usr/bin/env python3
"""The gpu_clocks.csv samples (sclk every ~100 ms) inside each timed interval of the quiet runs of E18 box windows 1 and 2. No GPU.

  interval_clocks.py <window-1 folder> <window-2 folder> --out interval_clocks.csv

The program records no per-interval host times, so each interval's position is ESTIMATED from the order (rpp, rpp_ltrb, fkl_box_halfpixel,
fkl_half_once; whole_call then kernel_only) and its duration from e18_box_GPU_AMD.csv:
  whole_call  = 101 calls x mean_ms (one warm-up call, then 100)
  kernel_only = 101 x (5 ms spin + mean_ms)
The estimates are scaled to fill the run's timing range:
  window 2: the range the program prints ("E18 timing (realtime ns): ... timing start S, timing end E");
  window 1 (no such line): it ends at the last sample >= 3000 MHz before the clock drops at the end of the run's timing, plus 50 ms, and has the
  length of the summed estimate.
The position error is about one sampling period (100 ms) or less; a 25 ms whole_call holds at most one sample.
"""
import argparse, csv, os, re
ap = argparse.ArgumentParser(); ap.add_argument("w1"); ap.add_argument("w2"); ap.add_argument("--out", required=True)
a = ap.parse_args()
ORDER = ["rpp", "rpp_ltrb", "fkl_box_halfpixel", "fkl_half_once"]
def samples(folder):
    out = []
    for d in csv.DictReader(open(os.path.join(folder, "gpu_clocks.csv"))):
        try: out.append((int(d["time_ns"]), int(d["sclk_hz"]) / 1e6))
        except (TypeError, ValueError): pass
    return out
def run_times(folder, run):
    t = {}
    for d in csv.DictReader(l for l in open(os.path.join(folder, run, "e18_box_GPU_AMD.csv")) if not l.startswith("#")):
        t[(d["version"], d["interval"])] = float(d["mean_ms"])
    return t
def window1_end(folder, run, smp):
    # the run's console times (to the second) bound it; the timing ends with the last sample >= 3000 MHz before the clock drops
    text = open(os.path.join(folder, "window-console.txt")).read()
    import datetime
    m = re.search(r"# (\S+) " + run + r" +\n", text); e = re.search(r"# (\S+) " + run + r" exit", text)
    s0 = datetime.datetime.fromisoformat(m.group(1)).timestamp() * 1e9; e0 = (datetime.datetime.fromisoformat(e.group(1)).timestamp() + 1) * 1e9
    inside = [(t, s) for t, s in smp if s0 <= t <= e0]
    high = [t for t, s in inside if s >= 3000]
    return max(high) + 50e6
rows = []
for window, folder in (("window 1", a.w1), ("window 2", a.w2)):
    smp = samples(folder)
    for run in ("quiet-run1", "quiet-run2", "quiet-run3"):
        times = run_times(folder, run)
        est = []
        for v in ORDER:
            est.append((v, "whole_call", 101 * times[(v, "whole_call")] * 1e6))
            est.append((v, "kernel_only", 101 * (5.0 + times[(v, "kernel_only")]) * 1e6))
        total = sum(d for _, _, d in est)
        m = re.search(r"timing start (\d+), timing end (\d+)", open(os.path.join(folder, run, "stdout.txt")).read())
        if m:
            start, end = int(m.group(1)), int(m.group(2)); basis = "program's timing range"
        else:
            end = window1_end(folder, run, smp); start = end - total; basis = "estimated from the clock drop at the end"
        scale = (end - start) / total
        t = start
        for v, iv, d in est:
            t0, t1 = t, t + d * scale; t = t1
            inside = [s for ts, s in smp if t0 <= ts <= t1]
            rows.append({"window": window, "run": run, "version": v, "interval": iv, "basis": basis, "estimated_ms": round((t1 - t0) / 1e6, 1),
                         "samples": len(inside), "sclk_min_mhz": min(inside) if inside else "", "sclk_max_mhz": max(inside) if inside else "",
                         "sclk_mean_mhz": round(sum(inside) / len(inside)) if inside else "", "mean_ms": times[(v, iv)]})
with open(a.out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
for r in rows:
    if r["interval"] == "kernel_only" or r["version"] == "rpp_ltrb":
        print(r["window"], r["run"], r["version"], r["interval"], f"{r['samples']} samples", r["sclk_min_mhz"], r["sclk_mean_mhz"], r["sclk_max_mhz"], r["mean_ms"])
