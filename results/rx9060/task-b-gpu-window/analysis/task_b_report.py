#!/usr/bin/env python3
"""The summary of the task B GPU window (nothing is run; no GPU).

  task_b_report.py <window folder> <output folder>

The window folder holds tests-after/ and tests-before/ (one folder per FKL test), checks-before/ and checks-after/ (E18, E19 and the E19 crop
sweep at N = 30, with E19's dumps), harness/ (unused-A/B and quiet-run1..3-A/B of filters_before_after_hip; A = against FKL 24478f0, before,
B = against the task B head, after) and gpu_clocks.csv. Writes to the output folder:
  summary.csv  per program (A, B), case and interval: the median of the three quiet runs' means, their min and max, the spread
               ((max - min) / median), the min of the mins, and the sclk at the interval's boundaries (min-max over the three runs)
  compare.csv  B against A: old (A), new (B), change, both spreads, whether the change exceeds the larger spread, and the flag above 3 %
  outputs.csv  per harness run: the sha256 of filters_outputs.bin (the outputs of the float filter instantiations) and whether it equals
               unused-A's
  fklb.csv     E18's output digests and the sha256 of every E19 dump, before and after, and whether they are equal
  clock.csv    per run: the hwmon samples during its timing (sclk min / median / max, below 3.1 GHz, median power, max temperatures) and the sclk
               after the 2 s warm-up
  unused.csv   unused-A/B against the quiet median of their program
  tests.csv    per test (after and before): exit code, PASS and FAIL lines, the last line
  report.md    the tables in Markdown
No cause is claimed for any change.
"""
import csv
import hashlib
import os
import re
import statistics
import sys

win, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
PROGRAMS = ["A", "B"]
QUIET = ["quiet-run1", "quiet-run2", "quiet-run3"]
INTERVALS = ["whole_call", "kernel_only"]
FLAG_PCT = 3.0


def read_times(folder):
    with open(os.path.join(folder, "filters_speed_GPU_AMD.csv")) as f:
        return {(r["case"], r["interval"]): r for r in csv.DictReader(line for line in f if not line.startswith("#"))}


def write_csv(name, rows):
    with open(os.path.join(out, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def pct(new, old):
    return 100.0 * (new - old) / old


H = os.path.join(win, "harness")
times = {(p, q): read_times(os.path.join(H, f"{q}-{p}")) for p in PROGRAMS for q in QUIET + ["unused"]}
CASES = []
for (case, _interval) in times[("A", "quiet-run1")]:
    if case not in CASES:
        CASES.append(case)

summary, stat = [], {}
for p in PROGRAMS:
    for case in CASES:
        for interval in INTERVALS:
            s = [times[(p, q)][(case, interval)] for q in QUIET]
            means = [float(d["mean_ms"]) for d in s]
            m = statistics.median(means)
            spread = 100.0 * (max(means) - min(means)) / m
            stat[(p, case, interval)] = (m, min(means), max(means), spread)
            before = [float(d["sclk_before_mhz"]) for d in s]
            after = [float(d["sclk_after_mhz"]) for d in s]
            summary.append({"program": p, "case": case, "interval": interval, "median_of_means_ms": f"{m:.6f}",
                            "min_of_means_ms": f"{min(means):.6f}", "max_of_means_ms": f"{max(means):.6f}", "spread_pct": f"{spread:.2f}",
                            "min_of_mins_ms": f"{min(float(d['min_ms']) for d in s):.6f}",
                            "sclk_before_mhz": f"{min(before):.0f}-{max(before):.0f}", "sclk_after_mhz": f"{min(after):.0f}-{max(after):.0f}",
                            **{f"mean_ms_{q}": f"{float(d['mean_ms']):.6f}" for q, d in zip(QUIET, s)}})
write_csv("summary.csv", summary)

compare = []
for case in CASES:
    for interval in INTERVALS:
        om, _, _, osp = stat[("A", case, interval)]
        nm, _, _, nsp = stat[("B", case, interval)]
        ch = pct(nm, om)
        compare.append({"case": case, "interval": interval, "old_ms": f"{om:.4f}", "new_ms": f"{nm:.4f}", "change_pct": f"{ch:+.1f}",
                        "spread_old_pct": f"{osp:.1f}", "spread_new_pct": f"{nsp:.1f}",
                        "exceeds_larger_spread": "yes" if abs(ch) > max(osp, nsp) else "no",
                        "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("compare.csv", compare)

outputs = []
runs = [f"{q}-{p}" for q in ["unused"] + QUIET for p in PROGRAMS]
digests = {}
for r in runs:
    with open(os.path.join(H, r, "filters_outputs.bin"), "rb") as f:
        digests[r] = hashlib.sha256(f.read()).hexdigest()
for r in runs:
    outputs.append({"run": r, "outputs_sha256": digests[r], "same_as_unused_A": "yes" if digests[r] == digests["unused-A"] else "NO"})
write_csv("outputs.csv", outputs)

samples = []
with open(os.path.join(win, "gpu_clocks.csv")) as f:
    for d in csv.DictReader(f):
        try:
            samples.append((int(d["time_ns"]), int(d["sclk_hz"]) / 1e6, int(d["power_uw"]) / 1e6, int(d["temp_edge_mc"]) / 1e3,
                            int(d["temp_junction_mc"]) / 1e3, int(d["temp_mem_mc"]) / 1e3))
        except (TypeError, ValueError, KeyError):
            pass
clock = []
for r in runs:
    text = open(os.path.join(H, r, "stdout.txt")).read()
    m = re.search(r"filters_before_after timing \(realtime ns\): warm-up start (\d+), timing start (\d+), timing end (\d+)", text)
    w = re.search(r"sclk ([\d.]+) MHz before it, ([\d.]+) MHz after it", text)
    start, end = int(m.group(2)), int(m.group(3))
    inside = [x for x in samples if start <= x[0] <= end]
    sc = [x[1] for x in inside]
    clock.append({"run": r, "timing_seconds": f"{(end - start) / 1e9:.2f}", "samples": len(inside),
                  "sclk_min_mhz": f"{min(sc):.0f}" if sc else "", "sclk_median_mhz": f"{statistics.median(sc):.0f}" if sc else "",
                  "sclk_max_mhz": f"{max(sc):.0f}" if sc else "", "samples_below_3100": sum(1 for s in sc if s < 3100.0),
                  "power_median_w": f"{statistics.median(x[2] for x in inside):.0f}" if inside else "",
                  "temp_edge_max_c": f"{max(x[3] for x in inside):.0f}" if inside else "",
                  "temp_junction_max_c": f"{max(x[4] for x in inside):.0f}" if inside else "",
                  "temp_mem_max_c": f"{max(x[5] for x in inside):.0f}" if inside else "",
                  "sclk_before_warmup_mhz": f"{float(w.group(1)):.0f}", "sclk_after_warmup_mhz": f"{float(w.group(2)):.0f}"})
write_csv("clock.csv", clock)

unused = []
for p in PROGRAMS:
    for case in CASES:
        for interval in INTERVALS:
            v = float(times[(p, "unused")][(case, interval)]["mean_ms"])
            ch = pct(v, stat[(p, case, interval)][0])
            unused.append({"run": f"unused-{p}", "case": case, "interval": interval, "mean_ms": f"{v:.6f}",
                           "quiet_median_ms": f"{stat[(p, case, interval)][0]:.6f}", "change_pct": f"{ch:+.2f}",
                           "flag_above_3pct": "FLAG" if abs(ch) > FLAG_PCT else "no"})
write_csv("unused.csv", unused)

tests = []
for side in ["after", "before"]:
    tdir = os.path.join(win, f"tests-{side}")
    for name in sorted(os.listdir(tdir)):
        lines = open(os.path.join(tdir, name, "stdout.txt")).read().splitlines()
        tests.append({"side": side, "test": name, "exit": open(os.path.join(tdir, name, "exit_code.txt")).read().strip(),
                      "pass_lines": sum(1 for line in lines if line.rstrip().endswith("PASS")),
                      "fail_lines": sum(1 for line in lines if "FAIL" in line), "last_line": lines[-1] if lines else ""})
write_csv("tests.csv", tests)

fklb = []
for program in ["e18_box", "e19_batched_box", "e19_crop_sweep_n030"]:
    found = {}
    for side in ["before", "after"]:
        d = os.path.join(win, f"checks-{side}", program)
        for line in open(os.path.join(d, "stdout.txt")).read().splitlines():
            m = re.match(r"digest (.+): ([0-9a-f]+)$", line)
            if m:
                found.setdefault(("digest " + m.group(1)), {})[side] = m.group(2)
        dumps = os.path.join(d, "dumps")
        if os.path.isdir(dumps):
            for name in sorted(os.listdir(dumps)):
                with open(os.path.join(dumps, name), "rb") as f:
                    found.setdefault("dump " + name, {})[side] = hashlib.sha256(f.read()).hexdigest()
    for what, sides in found.items():
        fklb.append({"program": program, "output": what, "before": sides.get("before", ""), "after": sides.get("after", ""),
                     "equal": "yes" if sides.get("before") == sides.get("after") and sides.get("before") else "NO"})
write_csv("fklb.csv", fklb)

md = ["# Task B on the RX 9060: before (A, FKL 24478f0) and after (B, the task B head)", "",
      "## The FKL filter tests", "", "| built at | test | exit | PASS lines | FAIL lines |", "|---|---|--:|--:|--:|"]
md += [f"| {'the PR head' if t['side'] == 'after' else '24478f0'} | {t['test']} | {t['exit']} | {t['pass_lines']} | {t['fail_lines']} |"
       for t in tests]
md += ["", "## fkl-benchmarks E18 and E19: the outputs of the checks, before and after", "",
       "| program | output | equal before and after |", "|---|---|---|"]
md += [f"| {f['program']} | {f['output']} | {f['equal']} |" for f in fklb]
md += ["", "## The times: the median of the three quiet runs' means (min-max), ms", "",
       "| case | interval | A (before) | B (after) |", "|---|---|--:|--:|"]
for case in CASES:
    for interval in INTERVALS:
        a, b = stat[("A", case, interval)], stat[("B", case, interval)]
        md.append(f"| {case} | {interval} | {a[0]:.4f} ({a[1]:.4f}-{a[2]:.4f}) | {b[0]:.4f} ({b[1]:.4f}-{b[2]:.4f}) |")
md += ["", "## B against A (old / new / change / spread old / new / exceeds the larger spread / > 3 %)", "",
       "| case, interval | old (ms) | new (ms) | change | spread old / new | exceeds? | > 3 %? |", "|---|--:|--:|--:|--:|---|---|"]
for c in compare:
    md.append(f"| {c['case']} {c['interval']} | {c['old_ms']} | {c['new_ms']} | {c['change_pct']} % | {c['spread_old_pct']} / {c['spread_new_pct']} % | "
              f"{c['exceeds_larger_spread']} | {c['flag_above_3pct']} |")
md += ["", "## The outputs of the float filter instantiations (filters_outputs.bin)", ""]
md += [f"- {o['run']}: {o['outputs_sha256'][:16]}... {'(the same as unused-A)' if o['same_as_unused_A'] == 'yes' else 'DIFFERS from unused-A'}"
       for o in outputs]
md += ["", "## The clock in each run's timing (hwmon every 100 ms)", "",
       "| run | samples | sclk min / median / max (MHz) | below 3.1 GHz | median power (W) | max edge / junction / memory (°C) | sclk after the warm-up |",
       "|---|--:|---|--:|--:|---|--:|"]
for c in clock:
    md.append(f"| {c['run']} | {c['samples']} | {c['sclk_min_mhz']} / {c['sclk_median_mhz']} / {c['sclk_max_mhz']} | {c['samples_below_3100']} | "
              f"{c['power_median_w']} | {c['temp_edge_max_c']} / {c['temp_junction_max_c']} / {c['temp_mem_max_c']} | {c['sclk_after_warmup_mhz']} |")
md += ["", "## The unused runs against their program's quiet median", ""]
md += [f"- {u['run']} {u['case']} {u['interval']}: {u['mean_ms']} ms ({u['change_pct']} %){' FLAG' if u['flag_above_3pct'] == 'FLAG' else ''}"
       for u in unused]
with open(os.path.join(out, "report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
