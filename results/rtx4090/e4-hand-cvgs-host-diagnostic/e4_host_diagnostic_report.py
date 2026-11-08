#!/usr/bin/env python3
"""Report of the diagnostic window F2 (e4-hand-cvgs-host-diagnostic, RTX 4090, 2026-11-08): is FKL's higher host time in the window e4-hand-cvgs the binary
or drift?

Reads only result folders:
  this window: results/rtx4090/e4-hand-cvgs-host-diagnostic/binary-{836a98e,2dac1c8}/run1..3 (E4a, fkl_fused and fkl_fused_half_pixel), gpu_clocks.csv
  for drift: results/rtx4090/cleanup-window-b/cleaned/e4-preprocessing/quiet-run1..3 (836a98e, 2026-11-07) and results/rtx4090/e4-hand-cvgs/quiet-run1..3 (2dac1c8)
Writes e4-hand-cvgs-host-diagnostic-report.md and .csv into the window folder.
old = 836a98e (window B's binary), new = 2dac1c8 (the binary of the window e4-hand-cvgs), both in this window, alternating; median of three runs, spread =
(largest - smallest) / median; change = new against old; 'exceeds' when |change| is larger than both spreads; FLAG above 3 % (GPU-bound) or 5 % (host-bound);
the bound rules of cleanup_ab_report.py (gpu_mean_ms: host when the median host_enqueue_us is at least 0.8 of it; the host times: host).
"""
import csv
import datetime
import os
import statistics

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
D = f"{R}/e4-hand-cvgs-host-diagnostic"
VERSIONS = ("fkl_fused", "fkl_fused_half_pixel")
METRICS = (("enq", "host_enqueue_us"), ("lau", "host_launch_us"), ("gpu", "gpu_mean_ms"))


def rows_of(path):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    return list(csv.DictReader(lines))


def load(dirs, expect):
    out = {}
    for d in dirs:
        rows = [r for r in rows_of(os.path.join(d, "e4_preprocessing.csv")) if r["experiment"] == "E4a" and r["version"] in VERSIONS]
        if len(rows) != expect:
            raise SystemExit(f"{d}: {len(rows)} E4a rows of the two FKL versions, expected {expect}")
        for r in rows:
            e = out.setdefault((int(r["batch"]), r["version"]), {"gpu": [], "enq": [], "lau": []})
            e["gpu"].append(float(r["gpu_mean_ms"])); e["enq"].append(float(r["host_enqueue_us"])); e["lau"].append(float(r["host_launch_us"]))
    return out


def stats(v):
    m = statistics.median(v)
    return m, (max(v) - min(v)) / m * 100 if m else None


def pct(a, b):
    return (a - b) / b * 100


def judge(change, spreads, bound):
    exceeds = abs(change) > max(s for s in spreads if s is not None)
    flag = abs(change) > (5.0 if bound == "host" else 3.0)
    return ("exceeds" if exceeds else "within") + (" FLAG" if flag else ""), flag and exceeds


old = load([f"{D}/binary-836a98e/run{i}" for i in (1, 2, 3)], 60)
new = load([f"{D}/binary-2dac1c8/run{i}" for i in (1, 2, 3)], 60)
winb = load([f"{R}/cleanup-window-b/cleaned/e4-preprocessing/quiet-run{i}" for i in (1, 2, 3)], 60)
winf = load([f"{R}/e4-hand-cvgs/quiet-run{i}" for i in (1, 2, 3)], 60)
batches = sorted({k[0] for k in old})
md, rows, flags = [], [], []
md.append("# Diagnostic window F2 (e4-hand-cvgs-host-diagnostic), RTX 4090: FKL's host time, binary or drift (report)")
md.append("")
md.append(f"Written by `e4_host_diagnostic_report.py` from the result folders only, {datetime.date.today().isoformat()}. old = 836a98e (window B's binary), new = 2dac1c8 "
          "(the binary of the window e4-hand-cvgs), alternating in this window; E4a, fkl_fused and fkl_fused_half_pixel only; median of three runs.")
md.append("")
findings = os.path.join(D, "e4-hand-cvgs-host-diagnostic-findings.md")
if os.path.exists(findings):
    md += open(findings).read().rstrip("\n").splitlines()
    md.append("")
md.append("## Summary over batches 50 to 300 (the median of the per-batch changes, with the smallest and the largest)")
md.append("")
md.append("| version | metric | new against old, this window | old this window against window B | new this window against e4-hand-cvgs |")
md.append("|---|---|---|---|---|")
for v in VERSIONS:
    for m, name in METRICS:
        a = [pct(statistics.median(new[(b, v)][m]), statistics.median(old[(b, v)][m])) for b in batches if b >= 50]
        bb = [pct(statistics.median(old[(b, v)][m]), statistics.median(winb[(b, v)][m])) for b in batches if b >= 50]
        c = [pct(statistics.median(new[(b, v)][m]), statistics.median(winf[(b, v)][m])) for b in batches if b >= 50]
        md.append(f"| {v} | {name} | {statistics.median(a):+.1f} % ({min(a):+.1f} to {max(a):+.1f}) | {statistics.median(bb):+.1f} % ({min(bb):+.1f} to {max(bb):+.1f}) | "
                  f"{statistics.median(c):+.1f} % ({min(c):+.1f} to {max(c):+.1f}) |")
        rows.append({"section": "summary 50-300", "version": v, "metric": name, "new_vs_old_median_pct": statistics.median(a), "new_vs_old_min_pct": min(a),
                     "new_vs_old_max_pct": max(a), "old_vs_window_b_median_pct": statistics.median(bb), "new_vs_e4_hand_cvgs_median_pct": statistics.median(c)})
md.append("")
for m, name in METRICS:
    md.append(f"## {name}, every batch (standing form: new against old in this window; then the drift columns)")
    md.append("")
    md.append("| batch | version | bound | old (spread) | new (spread) | change | | old against window B | new against e4-hand-cvgs |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for b in batches:
        for v in VERSIONS:
            o, n = old[(b, v)], new[(b, v)]
            bound = ("host" if statistics.median(n["enq"]) >= 0.8 * statistics.median(n["gpu"]) * 1000 else "GPU") if m == "gpu" else "host"
            om, os_ = stats(o[m]); nm, ns = stats(n[m])
            c = pct(nm, om)
            j, f = judge(c, [os_, ns], bound)
            db = pct(om, statistics.median(winb[(b, v)][m])); df = pct(nm, statistics.median(winf[(b, v)][m]))
            rows.append({"section": name, "batch": b, "version": v, "bound": bound, "old": om, "old_spread_pct": os_, "new": nm, "new_spread_pct": ns,
                         "change_pct": c, "judgement": j, "old_vs_window_b_pct": db, "new_vs_e4_hand_cvgs_pct": df})
            if f:
                flags.append((name, b, v, bound, c))
            md.append(f"| {b} | {v} | {bound} | {om:.4g} ({os_:.1f} %) | {nm:.4g} ({ns:.1f} %) | {c:+.1f} % | {j} | {db:+.1f} % | {df:+.1f} % |")
    md.append("")
md.append(f"Flagged (new against old, above the flag and both spreads): {len(flags)}")
md.append("")
for name, b, v, bound, c in flags:
    md.append(f"- {name} E4a {b} {v} ({bound}): {c:+.1f} %")
md.append("")
md.append("## The runs and the SM clock")
md.append("")
samples = []
for line in open(os.path.join(D, "gpu_clocks.csv")):
    f = [x.strip() for x in line.split(",")]
    try:
        samples.append((datetime.datetime.strptime(f[0], "%Y/%m/%d %H:%M:%S.%f"), int(f[1].split()[0])))
    except (ValueError, IndexError):
        continue
md.append(f"Sampler: {len(samples)} samples of nvidia-smi every 100 ms (gpu_clocks.csv).")
md.append("")
md.append("| run | start | end | SM clock median MHz | min MHz | samples |")
md.append("|---|---|---|---|---|---|")
for run in ["warmup"] + [f"binary-{b}/run{i}" for i in (1, 2, 3) for b in ("836a98e", "2dac1c8")]:
    t = open(os.path.join(D, run, "load.txt")).read()
    st, en = t.split("start ")[1].split()[0], t.split("end ")[1].split()[0]
    s0 = datetime.datetime.fromisoformat(st).replace(tzinfo=None); s1 = datetime.datetime.fromisoformat(en).replace(tzinfo=None)
    sel = [c for ts, c in samples if s0 <= ts <= s1]
    md.append(f"| {run} | {st[11:19]} | {en[11:19]} | {statistics.median(sel):,.0f} | {min(sel):,} | {len(sel)} |" if sel else f"| {run} | {st[11:19]} | {en[11:19]} | - | - | 0 |")
md.append("")
open(os.path.join(D, "e4-hand-cvgs-host-diagnostic-report.md"), "w").write("\n".join(md) + "\n")
keys = []
for r in rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(os.path.join(D, "e4-hand-cvgs-host-diagnostic-report.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in rows:
        w.writerow(r)
print(len(md), "lines;", len(rows), "rows;", len(flags), "flags")
