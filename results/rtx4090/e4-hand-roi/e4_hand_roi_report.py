#!/usr/bin/env python3
"""Report of the E4 window G (e4-hand-roi, RTX 4090, 2026-11-08), the final RTX 4090 data of Figure 15: the series of the figure, and every version against the
window e4-hand-cvgs.

Reads only result folders (nothing is measured or written elsewhere):
  new: results/rtx4090/e4-hand-roi/quiet-run1..3 (main 469aa6d: e4_preprocessing.csv, e4_preprocessing_agreement.csv, e4_preprocessing_pairs.csv), gpu_clocks.csv
  old: results/rtx4090/e4-hand-cvgs/quiet-run1..3 (main 2dac1c8, the window e4-hand-cvgs)
Writes e4-hand-roi-report.md and e4-hand-roi-report.csv into the window folder.

The series of the figure (E4a, every batch): per run, the ratio of the two versions' gpu_mean_ms (CUDA events around one call); the median of the three per-run
ratios with the smallest and the largest run, and the medians of the two times.
  hand_roi / FKL (the hand line of the figure), hand_batch / FKL and hand_padded / FKL, against fkl_fused, fkl_fused_late, fkl_fused_after and the mean of
  fkl_fused_late and fkl_fused_after (the two FKL times next to the hand-written kernels); hand_roi / hand_batch and hand_padded / hand_batch; fkl_fused_late / fkl_fused and fkl_fused_after / fkl_fused (the position);
  opencv_percrop / cvgs_fused (the speedup of cvGPUSpeedup) and cvgs_fused / fkl_fused.
Every version that both windows have (E4a: the nine versions of e4-hand-cvgs; E4c: four): old = the window e4-hand-cvgs, new = these runs; median, spread = (largest - smallest) /
median; change = new against old; 'exceeds' when |change| is larger than both spreads; FLAG above 3 % (GPU-bound) or 5 % (host-bound); bound of
gpu_mean_ms: host when the median host_enqueue_us is at least 0.8 of it, the host times are host-bound (the rules of cleanup_ab_report.py).
"""
import csv
import datetime
import os
import statistics
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
NEW = f"{R}/e4-hand-roi"
OLD = f"{R}/e4-hand-cvgs"
RUNS = ("quiet-run1", "quiet-run2", "quiet-run3")
EXISTING = ("fkl_fused", "fkl_fused_half_pixel", "opencv_percrop", "npp", "fkl_fused_late", "hand_batch", "hand_padded", "fkl_fused_after", "cvgs_fused")
SHOWN = (10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 200, 250, 300)


def rows_of(path, expect=None):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    table = list(csv.reader(lines))
    head = [c.strip() for c in table[0]]
    body = table[1:]
    bad = [r for r in body if len(r) != len(head)]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} rows with a field count other than the header's")
    out = [dict(zip(head, [c.strip() for c in r])) for r in body]
    if expect is not None and len(out) != expect:
        raise SystemExit(f"{path}: {len(out)} data rows, expected {expect}")
    return out


def stats(values):
    values = [v for v in values if v is not None]
    if not values:
        return None, None, 0
    med = statistics.median(values)
    spread = (max(values) - min(values)) / med * 100 if len(values) > 1 and med else None
    return med, spread, len(values)


def pct(a, b):
    return (a - b) / b * 100 if (a is not None and b not in (None, 0)) else None


def fmt(v, digits=4):
    if v is None:
        return "-"
    if abs(v) >= 1000:
        return f"{v:,.0f}"
    return f"{v:.{digits}g}"


def fp(v):
    return "-" if v is None else f"{v:+.1f} %"


def fs(v):
    return "one run" if v is None else f"{v:.1f} %"


def judge(change, spreads, bound):
    if change is None:
        return "-", False
    known = [s for s in spreads if s is not None]
    exceeds = bool(known) and abs(change) > max(known)
    limit = 5.0 if bound == "host" else 3.0
    flag = abs(change) > limit
    return ("exceeds" if exceeds else "within") + (" FLAG" if flag else ""), flag and exceeds


def table_of(dirs, fname, expect):
    out = {}
    for i, d in enumerate(dirs):
        for r in rows_of(os.path.join(d, fname), expect):
            k = (r["experiment"], int(r["batch"]), int(r["src_w"]), r["version"])
            e = out.setdefault(k, {"gpu": [None] * len(dirs), "enq": [None] * len(dirs), "lau": [None] * len(dirs)})
            e["gpu"][i] = float(r["gpu_mean_ms"]); e["enq"][i] = float(r["host_enqueue_us"]); e["lau"][i] = float(r["host_launch_us"])
    return out


new_dirs = [os.path.join(NEW, r) for r in RUNS]
old_dirs = [os.path.join(OLD, r) for r in RUNS]
new = table_of(new_dirs, "e4_preprocessing.csv", 30 * 10 + 3 * 4)
old = table_of(old_dirs, "e4_preprocessing.csv", 30 * 9 + 3 * 4)
batches = sorted({k[1] for k in new if k[0] == "E4a"})
csv_rows = []
md = []


def ratio_runs(a, b):
    return [x / y for x, y in zip(a, b)]


def ratio_line(label, values):
    m, s, _ = stats(values)
    return m, min(values), max(values), s


def gpu(version, batch):
    return new[("E4a", batch, 3840, version)]["gpu"]


md.append("# E4 window G (e4-hand-roi), RTX 4090: the final data of Figure 15 (report)")
md.append("")
md.append(f"Written by `e4_hand_roi_report.py` from the result folders only, {datetime.date.today().isoformat()}. Times: gpu_mean_ms, CUDA events around one call of the whole "
          "implementation (E4's protocol), 100 calls after one warm-up; per point the median of the three runs. Ratios: the median of the three per-run ratios, with "
          "the smallest and the largest run.")
md.append("")
findings = os.path.join(NEW, "e4-hand-roi-findings.md")   # written by hand from the tables below
if os.path.exists(findings):
    md += open(findings).read().rstrip("\n").splitlines()
    md.append("")

# ---------------------------------------------------------------- new series
md.append("## The new series (E4a)")
md.append("")
md.append("### hand_roi / FKL (the hand line of the figure: the hand-written kernel with its CPU function, over FKL)")
md.append("")
md.append("FKL time `adjacent` = the mean of fkl_fused_late (timed just before hand_batch) and fkl_fused_after (timed just after hand_roi).")
md.append("")
md.append("| batch | hand_roi ms | fkl_fused ms | adjacent FKL ms | hand_roi / adjacent (min - max) | hand_roi / fkl_fused | hand_roi / hand_batch |")
md.append("|---|---|---|---|---|---|---|")
for b in batches:
    hr, hb, ff, fl, fa = gpu("hand_roi", b), gpu("hand_batch", b), gpu("fkl_fused", b), gpu("fkl_fused_late", b), gpu("fkl_fused_after", b)
    adj = [(x + y) / 2 for x, y in zip(fl, fa)]
    r_adj = ratio_runs(hr, adj)
    row = {"section": "hand_roi / FKL", "batch": b, "hand_roi_ms": statistics.median(hr), "fkl_fused_ms": statistics.median(ff), "fkl_adjacent_ms": statistics.median(adj),
           "hand_roi_over_adjacent": statistics.median(r_adj), "hand_roi_over_adjacent_min": min(r_adj), "hand_roi_over_adjacent_max": max(r_adj),
           "hand_roi_over_fkl_fused": statistics.median(ratio_runs(hr, ff)), "hand_roi_over_hand_batch": statistics.median(ratio_runs(hr, hb))}
    csv_rows.append(row)
    if b in SHOWN:
        md.append(f"| {b} | {fmt(row['hand_roi_ms'])} | {fmt(row['fkl_fused_ms'])} | {fmt(row['fkl_adjacent_ms'])} | {row['hand_roi_over_adjacent']:.3f} "
                  f"({row['hand_roi_over_adjacent_min']:.3f} - {row['hand_roi_over_adjacent_max']:.3f}) | {row['hand_roi_over_fkl_fused']:.3f} | {row['hand_roi_over_hand_batch']:.3f} |")
md.append("")
md.append("### hand_batch / FKL (E1c's kernel with compile-time crop sizes, a repository diagnostic)")
md.append("")
md.append("")
md.append("| batch | hand_batch ms | fkl_fused ms | adjacent FKL ms | hand / fkl_fused | hand / adjacent (min - max) | hand / late | hand / after |")
md.append("|---|---|---|---|---|---|---|---|")
for b in batches:
    hb, ff, fl, fa = gpu("hand_batch", b), gpu("fkl_fused", b), gpu("fkl_fused_late", b), gpu("fkl_fused_after", b)
    adj = [(x + y) / 2 for x, y in zip(fl, fa)]
    r_ff, r_adj, r_l, r_a = ratio_runs(hb, ff), ratio_runs(hb, adj), ratio_runs(hb, fl), ratio_runs(hb, fa)
    m_adj, lo_adj, hi_adj, s_adj = ratio_line("", r_adj)
    row = {"section": "hand_batch / FKL", "batch": b, "hand_batch_ms": statistics.median(hb), "fkl_fused_ms": statistics.median(ff),
           "fkl_fused_late_ms": statistics.median(fl), "fkl_fused_after_ms": statistics.median(fa), "fkl_adjacent_ms": statistics.median(adj),
           "hand_over_fkl_fused": statistics.median(r_ff), "hand_over_adjacent": m_adj, "hand_over_adjacent_min": lo_adj, "hand_over_adjacent_max": hi_adj,
           "hand_over_late": statistics.median(r_l), "hand_over_after": statistics.median(r_a)}
    csv_rows.append(row)
    if b in SHOWN:
        md.append(f"| {b} | {fmt(row['hand_batch_ms'])} | {fmt(row['fkl_fused_ms'])} | {fmt(row['fkl_adjacent_ms'])} | {row['hand_over_fkl_fused']:.3f} | "
                  f"{m_adj:.3f} ({lo_adj:.3f} - {hi_adj:.3f}) | {row['hand_over_late']:.3f} | {row['hand_over_after']:.3f} |")
md.append("")
md.append("### hand_padded (the larger launch-parameter block) and the position of FKL")
md.append("")
md.append("| batch | hand_padded ms | hand_padded / adjacent FKL | hand_padded / hand_batch | fkl_fused_late / fkl_fused | fkl_fused_after / fkl_fused |")
md.append("|---|---|---|---|---|---|")
for b in batches:
    hp, hb, ff, fl, fa = gpu("hand_padded", b), gpu("hand_batch", b), gpu("fkl_fused", b), gpu("fkl_fused_late", b), gpu("fkl_fused_after", b)
    adj = [(x + y) / 2 for x, y in zip(fl, fa)]
    r = {"section": "hand_padded and position", "batch": b, "hand_padded_ms": statistics.median(hp),
         "hand_padded_over_adjacent": statistics.median(ratio_runs(hp, adj)), "hand_padded_over_hand_batch": statistics.median(ratio_runs(hp, hb)),
         "late_over_fkl_fused": statistics.median(ratio_runs(fl, ff)), "late_over_fkl_fused_min": min(ratio_runs(fl, ff)), "late_over_fkl_fused_max": max(ratio_runs(fl, ff)),
         "after_over_fkl_fused": statistics.median(ratio_runs(fa, ff)), "after_over_fkl_fused_min": min(ratio_runs(fa, ff)), "after_over_fkl_fused_max": max(ratio_runs(fa, ff))}
    csv_rows.append(r)
    if b in SHOWN:
        md.append(f"| {b} | {fmt(r['hand_padded_ms'])} | {r['hand_padded_over_adjacent']:.3f} | {r['hand_padded_over_hand_batch']:.3f} | "
                  f"{r['late_over_fkl_fused']:.3f} ({r['late_over_fkl_fused_min']:.3f} - {r['late_over_fkl_fused_max']:.3f}) | "
                  f"{r['after_over_fkl_fused']:.3f} ({r['after_over_fkl_fused_min']:.3f} - {r['after_over_fkl_fused_max']:.3f}) |")
md.append("")
md.append("### cvGPUSpeedup against OpenCV-CUDA per crop")
md.append("")
md.append("| batch | opencv_percrop ms | cvgs_fused ms | speedup opencv_percrop / cvgs_fused (min - max) | cvgs_fused / fkl_fused |")
md.append("|---|---|---|---|---|")
for b in batches:
    oc, cg, ff = gpu("opencv_percrop", b), gpu("cvgs_fused", b), gpu("fkl_fused", b)
    sp = ratio_runs(oc, cg)
    r = {"section": "cvGPUSpeedup", "batch": b, "opencv_percrop_ms": statistics.median(oc), "cvgs_fused_ms": statistics.median(cg),
         "speedup_cvgs": statistics.median(sp), "speedup_cvgs_min": min(sp), "speedup_cvgs_max": max(sp), "cvgs_over_fkl_fused": statistics.median(ratio_runs(cg, ff))}
    csv_rows.append(r)
    if b in SHOWN:
        md.append(f"| {b} | {fmt(r['opencv_percrop_ms'])} | {fmt(r['cvgs_fused_ms'])} | {r['speedup_cvgs']:.1f} ({r['speedup_cvgs_min']:.1f} - {r['speedup_cvgs_max']:.1f}) | "
                  f"{r['cvgs_over_fkl_fused']:.3f} |")
md.append("")
md.append("### Host time of one call (host_enqueue_us: the call returns, the stream idle before it; median of the three runs, us)")
md.append("")
HOSTV = ("fkl_fused", "fkl_fused_late", "fkl_fused_after", "hand_roi", "hand_batch", "hand_padded", "cvgs_fused", "opencv_percrop")
md.append("| batch | " + " | ".join(HOSTV) + " |")
md.append("|---|" + "---|" * len(HOSTV))
for b in batches:
    vals = {v: statistics.median(new[("E4a", b, 3840, v)]["enq"]) for v in HOSTV}
    csv_rows.append(dict({"section": "host_enqueue_us", "batch": b}, **{f"{v}_us": vals[v] for v in HOSTV}))
    if b in SHOWN:
        md.append(f"| {b} | " + " | ".join(fmt(vals[v]) for v in HOSTV) + " |")
md.append("")

# ---------------------------------------------------------------- existing series against window B
md.append("## Every version against the window e4-hand-cvgs (standing form)")
md.append("")
md.append("old = the window e4-hand-cvgs (main 2dac1c8, 2026-11-08 12:13), new = this window (main 469aa6d; the kernels of 2dac1c8 identical, plus hand_roi's); "
          "median of three runs, spread = (largest - smallest) / median. In every batch hand_roi now runs between hand_padded and fkl_fused_after; the "
          "diagnostic F2 showed that the host code of the same FKL call can change with the content of the translation unit (here: hand_roi's 30 kernels).")
md.append("")
flags = []
for metric, title in (("gpu", "gpu_mean_ms"), ("enq", "host_enqueue_us"), ("lau", "host_launch_us")):
    md.append(f"### {title}")
    md.append("")
    md.append("| point | version | bound | old (spread) | new (spread) | change | |")
    md.append("|---|---|---|---|---|---|---|")
    for k in sorted(old, key=lambda k: (k[0], k[1], k[2], EXISTING.index(k[3]))):
        if k not in new:
            continue
        exp, b, w, v = k
        o, n = old[k], new[k]
        if metric == "gpu":
            bound = "host" if statistics.median(n["enq"]) >= 0.8 * statistics.median(n["gpu"]) * 1000 else "GPU"
        else:
            bound = "host"
        om, os_, _ = stats(o[metric]); nm, ns, _ = stats(n[metric])
        ch = pct(nm, om)
        j, fl = judge(ch, [os_, ns], bound)
        point = f"E4a {b}" if exp == "E4a" else f"E4c {w}"
        csv_rows.append({"section": f"existing {title}", "point": point, "version": v, "bound": bound, "old": om, "old_spread_pct": os_, "new": nm,
                         "new_spread_pct": ns, "change_pct": ch, "judgement": j})
        if fl:
            flags.append((title, point, v, bound, om, os_, nm, ns, ch))
        if exp == "E4c" or b in (10, 50, 100, 150, 200, 250, 300):
            md.append(f"| {point} | {v} | {bound} | {fmt(om)} ({fs(os_)}) | {fmt(nm)} ({fs(ns)}) | {fp(ch)} | {j} |")
    md.append("")
md.append(f"Flagged over all 30 batches and 3 frames (|change| above the flag and above both spreads): {len(flags)}")
md.append("")
for t, point, v, bound, om, os_, nm, ns, ch in flags:
    md.append(f"- {t} {point} {v} ({bound}): {fp(ch)} (old {fmt(om)}, spread {fs(os_)}; new {fmt(nm)}, spread {fs(ns)})")
md.append("")

# ---------------------------------------------------------------- agreement and pairs
md.append("## Agreement")
md.append("")
bad = []
for d in new_dirs:
    for r in rows_of(os.path.join(d, "e4_preprocessing_agreement.csv"), 30 * 10 + 3 * 4):
        if r["ok_own"] != "1":
            bad.append((os.path.basename(d), r["experiment"], r["batch"], r["version"], r["max_abs_err_own_reference"]))
md.append(f"- ok_own (against the scalar reference of the version's own convention, 1e-4): {3 * (30 * 10 + 3 * 4) - len(bad)} of {3 * (30 * 10 + 3 * 4)} rows pass" +
          ("" if not bad else "; failing: " + "; ".join(str(x) for x in bad)))
pairs = {}
for d in new_dirs:
    for r in rows_of(os.path.join(d, "e4_preprocessing_pairs.csv"), 30 * 7):
        key = (r["version_a"], r["version_b"])
        p = pairs.setdefault(key, {"bitwise": 0, "rows": 0, "max": 0.0, "within": 0})
        p["rows"] += 1; p["bitwise"] += int(r["bitwise_equal"]); p["within"] += int(r["within_tolerance"]); p["max"] = max(p["max"], float(r["max_abs_diff"]))
md.append("- pairs (the outputs of the checked calls; 30 batches x 3 runs per pair):")
for (a, b), p in sorted(pairs.items()):
    md.append(f"  - {a} vs {b}: bitwise equal in {p['bitwise']} of {p['rows']}, largest |difference| {p['max']:.3g}, within 1e-4 in {p['within']} of {p['rows']}")
    csv_rows.append({"section": "pairs", "version": f"{a} vs {b}", "bitwise_rows": p["bitwise"], "rows": p["rows"], "max_abs_diff": p["max"], "within_rows": p["within"]})
md.append("")

# ---------------------------------------------------------------- clocks and runs
md.append("## The runs and the SM clock")
md.append("")


def parse_time(s):
    return datetime.datetime.strptime(s.strip(), "%Y/%m/%d %H:%M:%S.%f")


samples = []
clk = os.path.join(NEW, "gpu_clocks.csv")
if os.path.exists(clk):
    for line in open(clk):
        f = [x.strip() for x in line.split(",")]
        if len(f) < 4:
            continue
        try:
            samples.append((parse_time(f[0]), int(f[1].split()[0]), f[3]))
        except ValueError:
            continue
md.append(f"Sampler: {len(samples)} samples of nvidia-smi every 100 ms (gpu_clocks.csv).")
md.append("")
md.append("| run | start | end | SM clock median MHz | min MHz | samples | samples below 2,500 MHz |")
md.append("|---|---|---|---|---|---|---|")
for r in ("warmup",) + RUNS:
    lt = os.path.join(NEW, r, "load.txt")
    if not os.path.exists(lt):
        continue
    t = open(lt).read()
    st = t.split("start ")[1].split()[0]; en = t.split("end ")[1].split()[0]
    s0 = datetime.datetime.fromisoformat(st).replace(tzinfo=None); s1 = datetime.datetime.fromisoformat(en).replace(tzinfo=None)
    sel = [c for (ts, c, ps) in samples if s0 <= ts <= s1]
    if sel:
        md.append(f"| {r} | {st[11:19]} | {en[11:19]} | {statistics.median(sel):,.0f} | {min(sel):,} | {len(sel)} | {sum(1 for c in sel if c < 2500)} |")
md.append("")

# ---------------------------------------------------------------- write
md_path = os.path.join(NEW, "e4-hand-roi-report.md")
csv_path = os.path.join(NEW, "e4-hand-roi-report.csv")
open(md_path, "w").write("\n".join(md) + "\n")
keys = []
for r in csv_rows:
    for k in r:
        if k not in keys:
            keys.append(k)
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in csv_rows:
        w.writerow(r)
print(md_path, len(md), "lines;", csv_path, len(csv_rows), "rows;", len(flags), "flags")
