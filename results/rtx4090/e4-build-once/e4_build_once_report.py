#!/usr/bin/env python3
"""Report of the E4 window H (e4-build-once, RTX 4090): the build-once series of Figure 15, and every existing version against the window e4-hand-roi.

Reads only result folders (nothing is measured or written elsewhere):
  new: results/rtx4090/e4-build-once/quiet-run1..3 (e4_preprocessing.csv, e4_preprocessing_agreement.csv, e4_preprocessing_pairs.csv), gpu_clocks.csv
  old: results/rtx4090/e4-hand-roi/quiet-run1..3 (main 469aa6d, the window e4-hand-roi)
Writes e4-build-once-report.md and e4-build-once-report.csv into the window folder.

The build-once variants (E4a) build their crop views (or ROIs, or parameter block) and every operation ONCE, before the timing: every timed call is only the
launch path, and the variants are valid only while the crops (their positions and sizes) do not change. The series of the figure (E4a, every batch): per run,
the ratio of the two versions' gpu_mean_ms (CUDA events around one call); the median of the three per-run ratios with the smallest and the largest run, and
the medians of the two times:
  opencv_percrop / fkl_fused_once, npp / fkl_fused_half_pixel_once, opencv_percrop / cvgs_fused_once (speedups over the baselines, which are timed as a user
  calls them); hand_roi_once / FKL, against the mean of fkl_fused_once and fkl_fused_once_after (the two FKL calls next to it) and against fkl_fused_once.
Also: the build-once variants against their whole-call versions, the position of the bracket, and the host time of one call.
Every version that both windows have (E4a: the ten versions of e4-hand-roi; E4c: four): old = the window e4-hand-roi, new = these runs; median, spread =
(largest - smallest) / median; change = new against old; 'exceeds' when |change| is larger than both spreads; FLAG above 3 % (GPU-bound) or 5 % (host-bound);
bound of gpu_mean_ms: host when the median host_enqueue_us is at least 0.8 of it, the host times are host-bound (the rules of cleanup_ab_report.py).
"""
import csv
import datetime
import os
import statistics
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
NEW = f"{R}/e4-build-once"
OLD = f"{R}/e4-hand-roi"
RUNS = ("quiet-run1", "quiet-run2", "quiet-run3")
EXISTING = ("fkl_fused", "fkl_fused_half_pixel", "opencv_percrop", "npp", "fkl_fused_late", "hand_batch", "hand_padded", "hand_roi", "fkl_fused_after", "cvgs_fused")
ONCE = ("cvgs_fused_once", "fkl_fused_half_pixel_once", "fkl_fused_once", "hand_roi_once", "fkl_fused_once_after")
N_E4A_NEW, N_E4A_OLD, N_E4C = len(EXISTING) + len(ONCE), len(EXISTING), 4
PAIRS_NEW, PAIRS_OLD = 13, 7   # pairs rows per batch: the 7 of e4-hand-roi and the 6 of the build-once variants
SUMMARY = (10, 50, 100, 150, 200, 250, 300)


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
new = table_of(new_dirs, "e4_preprocessing.csv", 30 * N_E4A_NEW + 3 * N_E4C)
old = table_of(old_dirs, "e4_preprocessing.csv", 30 * N_E4A_OLD + 3 * N_E4C)
batches = sorted({k[1] for k in new if k[0] == "E4a"})
if len(batches) != 30:
    raise SystemExit(f"{len(batches)} E4a batches, expected 30")
csv_rows = []
md = []


def ratio_runs(a, b):
    return [x / y for x, y in zip(a, b)]


def gpu(version, batch):
    return new[("E4a", batch, 3840, version)]["gpu"]


def series(label, num, den, digits):
    """A table of num / den over every batch (num and den: functions of the batch giving the three per-run values) and its CSV rows."""
    md.append(f"| batch | {label[0]} ms | {label[1]} ms | ratio, median of the runs | smallest run | largest run | spread |")
    md.append("|---|---|---|---|---|---|---|")
    for b in batches:
        n, d = num(b), den(b)
        r = ratio_runs(n, d)
        m, s, _ = stats(r)
        csv_rows.append({"section": f"{label[0]} / {label[1]}", "batch": b, "numerator_ms": statistics.median(n), "denominator_ms": statistics.median(d),
                         "ratio": m, "ratio_min": min(r), "ratio_max": max(r), "ratio_spread_pct": s})
        md.append(f"| {b} | {fmt(statistics.median(n))} | {fmt(statistics.median(d))} | {m:.{digits}f} | {min(r):.{digits}f} | {max(r):.{digits}f} | {fs(s)} |")
    md.append("")


md.append("# E4 window H (e4-build-once), RTX 4090: the build-once series of Figure 15 (report)")
md.append("")
md.append(f"Written by `e4_build_once_report.py` from the result folders only, {datetime.date.today().isoformat()}. Times: gpu_mean_ms, CUDA events around one call "
          "(E4's protocol), 100 calls after one warm-up; per point the median of the three runs. Ratios: the median of the three per-run ratios, with the smallest "
          "and the largest run. The build-once variants (`*_once`) build their crop views, ROIs or parameter block and every operation once, before the timing: "
          "their interval holds only the launch path and the kernel, and they are valid only while the crops (their positions and sizes) do not change. The "
          "baselines (opencv_percrop, npp) are timed as a user calls them.")
md.append("")
findings = os.path.join(NEW, "e4-build-once-findings.md")   # written by hand from the tables below
if os.path.exists(findings):
    md += open(findings).read().rstrip("\n").splitlines()
    md.append("")

md.append("## The build-once series of Figure 15 (E4a, every batch)")
md.append("")
md.append("### opencv_percrop / fkl_fused_once (speedup of FKL, build once, over OpenCV-CUDA per crop)")
md.append("")
series(("opencv_percrop", "fkl_fused_once"), lambda b: gpu("opencv_percrop", b), lambda b: gpu("fkl_fused_once", b), 1)
md.append("### npp / fkl_fused_half_pixel_once (speedup of FKL with HALF_PIXEL, build once, over NPP)")
md.append("")
series(("npp", "fkl_fused_half_pixel_once"), lambda b: gpu("npp", b), lambda b: gpu("fkl_fused_half_pixel_once", b), 1)
md.append("### opencv_percrop / cvgs_fused_once (speedup of cvGPUSpeedup, build once, over OpenCV-CUDA per crop)")
md.append("")
series(("opencv_percrop", "cvgs_fused_once"), lambda b: gpu("opencv_percrop", b), lambda b: gpu("cvgs_fused_once", b), 1)
md.append("### hand_roi_once / FKL, build once (the hand line)")
md.append("")
md.append("Against the mean of fkl_fused_once (timed just before hand_roi_once) and fkl_fused_once_after (timed just after it):")
md.append("")
adj_once = lambda b: [(x + y) / 2 for x, y in zip(gpu("fkl_fused_once", b), gpu("fkl_fused_once_after", b))]
series(("hand_roi_once", "mean(fkl_fused_once, fkl_fused_once_after)"), lambda b: gpu("hand_roi_once", b), adj_once, 3)
md.append("Against fkl_fused_once:")
md.append("")
series(("hand_roi_once", "fkl_fused_once"), lambda b: gpu("hand_roi_once", b), lambda b: gpu("fkl_fused_once", b), 3)

md.append("## The build-once variants against their whole-call versions, and the position")
md.append("")
md.append("| batch | fkl_fused_once / fkl_fused | fkl_fused_half_pixel_once / fkl_fused_half_pixel | cvgs_fused_once / cvgs_fused | hand_roi_once / hand_roi | "
          "fkl_fused_once_after / fkl_fused_once (min - max) | cvgs_fused_once / fkl_fused_once |")
md.append("|---|---|---|---|---|---|---|")
for b in batches:
    r = {"section": "build once against whole call", "batch": b,
         "fkl_once_over_whole": statistics.median(ratio_runs(gpu("fkl_fused_once", b), gpu("fkl_fused", b))),
         "fkl_half_once_over_whole": statistics.median(ratio_runs(gpu("fkl_fused_half_pixel_once", b), gpu("fkl_fused_half_pixel", b))),
         "cvgs_once_over_whole": statistics.median(ratio_runs(gpu("cvgs_fused_once", b), gpu("cvgs_fused", b))),
         "hand_roi_once_over_whole": statistics.median(ratio_runs(gpu("hand_roi_once", b), gpu("hand_roi", b))),
         "after_over_once": statistics.median(ratio_runs(gpu("fkl_fused_once_after", b), gpu("fkl_fused_once", b))),
         "after_over_once_min": min(ratio_runs(gpu("fkl_fused_once_after", b), gpu("fkl_fused_once", b))),
         "after_over_once_max": max(ratio_runs(gpu("fkl_fused_once_after", b), gpu("fkl_fused_once", b))),
         "cvgs_once_over_fkl_once": statistics.median(ratio_runs(gpu("cvgs_fused_once", b), gpu("fkl_fused_once", b)))}
    csv_rows.append(r)
    md.append(f"| {b} | {r['fkl_once_over_whole']:.3f} | {r['fkl_half_once_over_whole']:.3f} | {r['cvgs_once_over_whole']:.3f} | {r['hand_roi_once_over_whole']:.3f} | "
              f"{r['after_over_once']:.3f} ({r['after_over_once_min']:.3f} - {r['after_over_once_max']:.3f}) | {r['cvgs_once_over_fkl_once']:.3f} |")
md.append("")
md.append("### Host time of one call (host_enqueue_us: the call returns, the stream idle before it; median of the three runs, us)")
md.append("")
HOSTV = ("fkl_fused", "fkl_fused_once", "fkl_fused_once_after", "fkl_fused_half_pixel_once", "hand_roi", "hand_roi_once", "cvgs_fused", "cvgs_fused_once")
md.append("| batch | " + " | ".join(HOSTV) + " |")
md.append("|---|" + "---|" * len(HOSTV))
for b in batches:
    vals = {v: statistics.median(new[("E4a", b, 3840, v)]["enq"]) for v in HOSTV}
    csv_rows.append(dict({"section": "host_enqueue_us", "batch": b}, **{f"{v}_us": vals[v] for v in HOSTV}))
    md.append(f"| {b} | " + " | ".join(fmt(vals[v]) for v in HOSTV) + " |")
md.append("")

md.append("## Every existing version against the window e4-hand-roi (standing form)")
md.append("")
md.append("old = the window e4-hand-roi (main 469aa6d), new = this window (the same 182 kernels; the build-once variants are timed after all the existing ones, "
          "which keep their order, code and buffers); median of three runs, spread = (largest - smallest) / median. The diagnostic F2 showed that the host code "
          "of the same call can change with the content of the translation unit.")
md.append("")
flags = []
for metric, title in (("gpu", "gpu_mean_ms"), ("enq", "host_enqueue_us"), ("lau", "host_launch_us")):
    md.append(f"### {title}")
    md.append("")
    md.append("| point | version | bound | old (spread) | new (spread) | change | |")
    md.append("|---|---|---|---|---|---|---|")
    medians = []
    for k in sorted(old, key=lambda k: (k[0], k[1], k[2], EXISTING.index(k[3]))):
        if k not in new:
            raise SystemExit(f"{k} is in the old window and not in the new one")
        exp, b, w, v = k
        o, n = old[k], new[k]
        bound = ("host" if statistics.median(n["enq"]) >= 0.8 * statistics.median(n["gpu"]) * 1000 else "GPU") if metric == "gpu" else "host"
        om, os_, _ = stats(o[metric]); nm, ns, _ = stats(n[metric])
        ch = pct(nm, om)
        j, fl = judge(ch, [os_, ns], bound)
        point = f"E4a {b}" if exp == "E4a" else f"E4c {w}"
        csv_rows.append({"section": f"existing {title}", "point": point, "version": v, "bound": bound, "old": om, "old_spread_pct": os_, "new": nm,
                         "new_spread_pct": ns, "change_pct": ch, "judgement": j})
        if exp == "E4a" and b >= 50:
            medians.append((v, ch))
        if fl:
            flags.append((title, point, v, bound, om, os_, nm, ns, ch))
        if exp == "E4c" or b in SUMMARY:
            md.append(f"| {point} | {v} | {bound} | {fmt(om)} ({fs(os_)}) | {fmt(nm)} ({fs(ns)}) | {fp(ch)} | {j} |")
    md.append("")
    md.append(f"Median change over E4a batches 50 to 300, per version ({title}): " +
              ", ".join(f"{v} {fp(statistics.median([c for (vv, c) in medians if vv == v]))}" for v in EXISTING))
    md.append("")
md.append(f"Flagged over all 30 batches and 3 frames (|change| above the flag and above both spreads): {len(flags)}")
md.append("")
for t, point, v, bound, om, os_, nm, ns, ch in flags:
    md.append(f"- {t} {point} {v} ({bound}): {fp(ch)} (old {fmt(om)}, spread {fs(os_)}; new {fmt(nm)}, spread {fs(ns)})")
md.append("")

md.append("## Agreement")
md.append("")
bad = []
n_agree = 30 * N_E4A_NEW + 3 * N_E4C
for d in new_dirs:
    for r in rows_of(os.path.join(d, "e4_preprocessing_agreement.csv"), n_agree):
        if r["ok_own"] != "1":
            bad.append((os.path.basename(d), r["experiment"], r["batch"], r["version"], r["max_abs_err_own_reference"]))
md.append(f"- ok_own (against the scalar reference of the version's own convention, 1e-4): {3 * n_agree - len(bad)} of {3 * n_agree} rows pass" +
          ("" if not bad else "; failing: " + "; ".join(str(x) for x in bad)))
pairs = {}
for d in new_dirs:
    for r in rows_of(os.path.join(d, "e4_preprocessing_pairs.csv"), 30 * PAIRS_NEW):
        key = (r["version_a"], r["version_b"])
        p = pairs.setdefault(key, {"bitwise": 0, "rows": 0, "max": 0.0, "within": 0})
        p["rows"] += 1; p["bitwise"] += int(r["bitwise_equal"]); p["within"] += int(r["within_tolerance"]); p["max"] = max(p["max"], float(r["max_abs_diff"]))
md.append("- pairs (the outputs of the checked calls; 30 batches x 3 runs per pair):")
for (a, b), p in sorted(pairs.items()):
    md.append(f"  - {a} vs {b}: bitwise equal in {p['bitwise']} of {p['rows']}, largest |difference| {p['max']:.3g}, within 1e-4 in {p['within']} of {p['rows']}")
    csv_rows.append({"section": "pairs", "version": f"{a} vs {b}", "bitwise_rows": p["bitwise"], "rows": p["rows"], "max_abs_diff": p["max"], "within_rows": p["within"]})
md.append("")

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

md_path = os.path.join(NEW, "e4-build-once-report.md")
csv_path = os.path.join(NEW, "e4-build-once-report.csv")
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
