#!/usr/bin/env python3
"""Report of the cleanup windows A and B (2026-11-07): published / reference-now / cleaned for every number, with run-to-run spreads,
drift (reference-now against published), cleanup effect (cleaned against reference-now), the 3 % (GPU-bound) / 5 % (host-bound) flags,
and the agreement rows. Reads only; writes the markdown report given as the first argument and a CSV of every number next to it."""
import csv
import os
import statistics
import sys

R = "/home/oscar/tfkl-bench/fkl-benchmarks/results/rtx4090"
A = R + "/cleanup-window-a"
B = R + "/cleanup-window-b"
OUT = sys.argv[1]
ALL_ROWS = []   # every number, for the CSV


def rows_of(path, expect=None):
    lines = [l for l in open(path) if l.strip() and not l.startswith("#")]
    table = list(csv.reader(lines))
    head = [c.strip() for c in table[0]]
    body = [r for r in table[1:]]
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
    """'exceeds' when |change| is larger than every spread involved (a missing spread is ignored); the flag at 3 % (GPU) or 5 % (host)."""
    if change is None:
        return "-", False
    known = [s for s in spreads if s is not None]
    exceeds = bool(known) and abs(change) > max(known)
    limit = 5.0 if bound == "host" else 3.0
    flag = abs(change) > limit
    text = ("exceeds" if exceeds else "within") + (" FLAG" if flag else "")
    return text, flag and exceeds


def section(title, key_name, numbers):
    """numbers: list of (label, bound, pub_values, ref_values, new_values, cause)."""
    lines = [f"### {title}", "",
             f"| {key_name} | bound | published (spread) | reference now (spread) | cleaned (spread) | drift ref vs pub | | cleanup cleaned vs ref | | cause |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    flagged = []
    for label, bound, pv, rv, nv, cause in numbers:
        pm, ps, pn = stats(pv)
        rm, rs, rn = stats(rv)
        nm, ns, nn = stats(nv)
        drift, effect = pct(rm, pm), pct(nm, rm)
        dj, dflag = judge(drift, [ps, rs], bound)
        ej, eflag = judge(effect, [rs, ns], bound)
        lines.append(f"| {label} | {bound} | {fmt(pm)} ({fs(ps) if pn else '-'}) | {fmt(rm)} ({fs(rs)}) | {fmt(nm)} ({fs(ns)}) | {fp(drift)} | {dj} | "
                     f"{fp(effect)} | {ej} | {cause if (dflag or eflag) else ''} |")
        ALL_ROWS.append([title, label, bound, pm, ps, pn, rm, rs, rn, nm, ns, nn, drift, dj, effect, ej, cause if (dflag or eflag) else ""])
        if dflag:
            flagged.append(f"{title} / {label} ({bound}): drift {fp(drift)} (published {fmt(pm)}, reference now {fmt(rm)}; spreads {fs(ps)}, {fs(rs)})")
        if eflag:
            flagged.append(f"{title} / {label} ({bound}): cleanup {fp(effect)} (reference now {fmt(rm)}, cleaned {fmt(nm)}; spreads {fs(rs)}, {fs(ns)}): {cause}")
    return lines + [""], flagged


def interp(points, x):
    """linear interpolation of {x: y} at x (clamped)"""
    xs = sorted(points)
    if x <= xs[0]:
        return points[xs[0]]
    if x >= xs[-1]:
        return points[xs[-1]]
    for a, b in zip(xs, xs[1:]):
        if a <= x <= b:
            return points[a] + (points[b] - points[a]) * (x - a) / (b - a)


report = ["# Cleanup windows A and B, 2026-11-07: published, reference now and cleaned", ""]
flags_all = []

# ---------------------------------------------------------------- window A: NPP GPU program, CPU-time programs
pub_npp = [f"{R}/e0-npp-after45/run{i}" for i in (1, 2, 3)]
ref_npp = [f"{A}/reference/e0-npp/run{i}" for i in (1, 2, 3)]
new_npp = [f"{A}/cleaned/e0-npp/run{i}" for i in (1, 2, 3)]
pub_cpu = [f"{R}/e0-cpu-time-npp-after45/run{i}" for i in (1, 2, 3)]
ref_cpu = [f"{A}/reference/e0-cpu-time-npp/run{i}" for i in (1, 2, 3)]
new_cpu = [f"{A}/cleaned/e0-cpu-time-npp/run{i}" for i in (1, 2, 3)]
pub_ocv = [f"{R}/e0-cpu-time-opencv"]
ref_ocv = [f"{A}/reference/e0-cpu-time-opencv/run{i}" for i in (1, 2, 3)]
new_ocv = [f"{A}/cleaned/e0-cpu-time-opencv/run{i}" for i in (1, 2, 3)]


def column(dirs, fname, key, col, expect):
    """{key value: [one value per run]}"""
    out = {}
    for d in dirs:
        for r in rows_of(os.path.join(d, fname), expect):
            out.setdefault(int(float(r[key])), []).append(float(r[col]))
    return out


# host times of the CPU-time NPP program (the same pipeline), median per batch, for the GPU-or-host classification of the GPU program
cpu_host = {}
for side, dirs in (("ref", ref_cpu), ("new", new_cpu)):
    for col in ("NPP MeanTime", "FK MeanTime"):
        c = column(dirs, "test_NPP_cpu_batchresize_x_split3D.csv", "Batch size", col, 16)
        cpu_host[(side, col)] = {b: statistics.median(v) for b, v in c.items()}

for fname, title in (("test_npp_batchresize_x_split3D.csv", "NPP GPU program, test_npp_batchresize_x_split3D (ms, CUDA events around one frame)"),
                     ("test_npp_batchresize_x_split3D_No_CPU_Overhead.csv", "NPP GPU program, No_CPU_Overhead (ms; FastNPP built once, valid only for fixed crops)")):
    numbers = []
    cols = {c: (column(pub_npp, fname, "Batch size", c, 15), column(ref_npp, fname, "Batch size", c, 15), column(new_npp, fname, "Batch size", c, 15))
            for c in ("NPP MeanTime", "FK MeanTime", "Mean Speedup")}
    for b in sorted(cols["NPP MeanTime"][1]):
        for c, name in (("NPP MeanTime", "NPP"), ("FK MeanTime", "FastNPP"), ("Mean Speedup", "speedup")):
            p, r, n = cols[c][0][b], cols[c][1][b], cols[c][2][b]
            if c == "Mean Speedup":
                bound = "ratio"
                cause = "follows FastNPP's time (see that row)"
            else:
                host = interp(cpu_host[("new", c)], b)
                event = statistics.median(n)
                bound = "host" if host >= 0.8 * event else "GPU"
                if name == "NPP":
                    cause = "NPP's own per-crop calls; its batch building (crop pointers) was already inside the interval"
                elif "No_CPU" in fname:
                    cause = "FastNPP built once in both builds; the stream is now idle before the first interval"
                else:
                    cause = "probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup"
            numbers.append((f"{b} {name}", bound if bound != "ratio" else "GPU", p, r, n, cause))
    lines, fl = section(title, "batch, side", numbers)
    report += lines
    flags_all += fl

# construction column of the No_CPU_Overhead test (new in the cleaned build)
cons = column(new_npp, "test_npp_batchresize_x_split3D_No_CPU_Overhead.csv", "Batch size", "FastNPP construction ms (once on host)", 15)
report += ["FastNPP's one-time construction (cleaned build only, host timer, ms; median over the three runs, spread):", "",
           "| batch | " + " | ".join(str(b) for b in sorted(cons)) + " |", "|---|" + "---|" * len(cons),
           "| ms | " + " | ".join(f"{fmt(stats(cons[b])[0], 3)} ({fs(stats(cons[b])[1])})" for b in sorted(cons)) + " |", ""]

for dirs_set, fname, title, key, cols_names in (
        ((pub_cpu, ref_cpu, new_cpu), "test_NPP_cpu_batchresize_x_split3D.csv", "CPU-time program, NPP against FastNPP (ms of host time per frame)", "Batch size",
         (("NPP MeanTime", "NPP", "NPP's per-crop calls; batch building already timed"),
          ("FK MeanTime", "FastNPP", "probable, not isolated: FastNPP now builds its batched resize from descriptor arrays written in the same call (stack arrays), instead of pinned host arrays (cudaMallocHost) filled once at setup"),
          ("Mean Speedup", "speedup", "follows FastNPP's time"))),
        ((pub_ocv, ref_ocv, new_ocv), "test_cpu_batchresize_x_split3D.csv", "CPU-time program, OpenCV-CUDA against cvGPUSpeedup (ms of host time per frame)",
         "Batch size (CV_8UC3XCV_32FC3)",
         (("OpenCV MeanTime", "OpenCV", "the GpuMat ROIs are now made in every iteration (by design); OpenCV's per-crop calls dominate"),
          ("cvGS MeanTime", "cvGS", "the GpuMat ROIs are now made in every iteration (by design; the reference made them once before the timed loop, a5dd9fa l.86-87): about 25 ns per crop"),
          ("Mean Speedup", "speedup", "follows cvGS's time")))):
    numbers = []
    data = {c: tuple(column(d, fname, key, c, 16) for d in dirs_set) for c, _, _ in cols_names}
    for b in sorted(data[cols_names[0][0]][1]):
        for c, name, cause in cols_names:
            numbers.append((f"{b} {name}", "host", data[c][0][b], data[c][1][b], data[c][2][b], cause))
    lines, fl = section(title, "batch, side", numbers)
    report += lines
    flags_all += fl

# E9
pub_e9 = [f"{R}/e9-cross-library/run{i}" for i in (0, 1, 2)]
ref_e9 = [f"{A}/reference/e9-cross-library/run{i}" for i in (0, 1, 2)]
new_e9 = [f"{A}/cleaned/e9-cross-library/run{i}" for i in (0, 1, 2)]


def e9col(dirs, col):
    out = {}
    for d in dirs:
        for r in rows_of(os.path.join(d, "e9_timing.csv"), 4):
            out.setdefault(r["variant"], []).append(float(r[col]))
    return out


numbers = []
lm = {s: e9col(d, "launch_ms_median") for s, d in (("p", pub_e9), ("r", ref_e9), ("n", new_e9))}
gm = {s: e9col(d, "graph_ms_median") for s, d in (("p", pub_e9), ("r", ref_e9), ("n", new_e9))}
kp = {s: e9col(d, "kernels_per_call") for s, d in (("p", pub_e9), ("r", ref_e9), ("n", new_e9))}
for v in sorted(lm["r"]):
    bound = "GPU" if statistics.median(gm["n"][v]) >= 0.8 * statistics.median(lm["n"][v]) else "host"
    numbers.append((f"{v} launch_ms", bound, lm["p"][v], lm["r"][v], lm["n"][v], "every call now builds its batch and operations (host work inside the interval, by design)"))
    numbers.append((f"{v} graph_ms", "GPU", gm["p"][v], gm["r"][v], gm["n"][v], "a graph replays only the kernel"))
lines, fl = section("E9 cross-library kernels (ms per call; median of each run, then over the runs)", "variant, column", numbers)
report += lines
flags_all += fl
report += ["kernels_per_call: " + ", ".join(f"{v} published {sorted(set(kp['p'][v]))}, reference {sorted(set(kp['r'][v]))}, cleaned {sorted(set(kp['n'][v]))}" for v in sorted(kp["r"])), ""]

# ---------------------------------------------------------------- window B: E4
pub_e4 = [f"{R}/e4-preprocessing/quiet-run{i}" for i in (1, 2, 3)]
ref_e4 = [f"{B}/reference/e4-preprocessing/quiet-run{i}" for i in (1, 2, 3)]
new_e4 = [f"{B}/cleaned/e4-preprocessing/quiet-run{i}" for i in (1, 2, 3)]


def e4col(dirs, expect):
    out = {}
    for d in dirs:
        for r in rows_of(os.path.join(d, "e4_preprocessing.csv"), expect):
            point = f"E4a {r['batch']}" if r["experiment"] == "E4a" else f"E4c {r['src_w']}x{r['src_h']}"
            k = (r["experiment"], int(r["batch"]), int(r["src_w"]), r["version"])
            out.setdefault(k, {"label": point, "gpu": [], "enq": [], "lau": []})
            out[k]["gpu"].append(float(r["gpu_mean_ms"]))
            out[k]["enq"].append(float(r["host_enqueue_us"]))
            out[k]["lau"].append(float(r["host_launch_us"]))
    return out


e4 = {"p": e4col(pub_e4, 162), "r": e4col(ref_e4, 162), "n": e4col(new_e4, 132)}
E4_CAUSE = {
    "fkl_fused": "FKL's crop views, read and operations are now built in the timed call (by design; the reference built them once before timing, a5dd9fa l.300-305): about 30 ns per crop",
    "fkl_fused_half_pixel": "as fkl_fused",
    "opencv_percrop": "OpenCV's ROIs are now made in the call (by design)",
    "npp": "NPP's stacks are now pitched cudaMallocPitch buffers (rows of 1,024 bytes instead of dense 720 and 768); crop pointers made in the call",
}
for metric, unit, title in (("gpu", "ms", "E4 pre-processing, gpu_mean_ms (CUDA events around one call)"),
                            ("enq", "us", "E4 pre-processing, host_enqueue_us (host time of one call, stream idle before it)"),
                            ("lau", "us", "E4 pre-processing, host_launch_us (100 calls back to back, wall time per call)")):
    numbers = []
    for k in sorted(e4["n"], key=lambda k: (k[0], k[1], k[2], k[3])):
        exp, batch, w, version = k
        if exp == "E4a" and batch not in (10, 50, 100, 150, 200, 250, 300):
            continue   # the table shows the batches of the published README; every batch is in the CSV
        new = e4["n"][k]
        if metric == "gpu":
            bound = "host" if statistics.median(new["enq"]) >= 0.8 * statistics.median(new["gpu"]) * 1000 else "GPU"
        else:
            bound = "host"
        numbers.append((f"{new['label']} {version}", bound, e4["p"][k][metric], e4["r"][k][metric], new[metric], E4_CAUSE[version]))
    lines, fl = section(title + ("; the table shows the batches of the published README, the CSV has all" if metric == "gpu" else ""), "point, version", numbers)
    report += lines
    flags_all += fl

# flags over ALL E4 batches (not only those shown)
e4_flag_all = []
for metric in ("gpu", "enq", "lau"):
    for k in sorted(e4["n"]):
        new = e4["n"][k]
        bound = ("host" if statistics.median(new["enq"]) >= 0.8 * statistics.median(new["gpu"]) * 1000 else "GPU") if metric == "gpu" else "host"
        pm, ps, _ = stats(e4["p"][k][metric]); rm, rs, _ = stats(e4["r"][k][metric]); nm, ns, _ = stats(new[metric])
        for name, change, spreads in (("drift", pct(rm, pm), [ps, rs]), ("cleanup", pct(nm, rm), [rs, ns])):
            j, f = judge(change, spreads, bound)
            if f:
                e4_flag_all.append((metric, name, k, change, bound))
report += ["E4: flagged changes over all 30 batches and 3 frames (metric, drift or cleanup, point, version, change, bound):", ""]
by = {}
for metric, name, k, change, bound in e4_flag_all:
    by.setdefault((metric, name, k[3], k[0]), []).append((k[1] if k[0] == "E4a" else k[2], change, bound))
for (metric, name, version, exp), items in sorted(by.items()):
    report.append(f"- {metric} {name} {exp} {version}: " + ", ".join(f"{p}: {c:+.1f} % ({b})" for p, c, b in items))
report.append("")

# ---------------------------------------------------------------- agreement rows
def data_lines(path):
    return [l.rstrip("\n") for l in open(path) if l.strip() and not l.startswith("#")]


report += ["## Agreement rows (data rows, comment lines left out)", ""]
for fname, pubdirs, refdirs, newdirs in (
        ("test_npp_batchresize_x_split3D_agreement.csv", pub_npp, ref_npp, new_npp),
        ("test_npp_batchresize_x_split3D_No_CPU_Overhead_agreement.csv", pub_npp, ref_npp, new_npp),
        ("test_NPP_cpu_batchresize_x_split3D_agreement.csv", pub_cpu, ref_cpu, new_cpu),
        ("e9_agreement.csv", pub_e9, ref_e9, new_e9)):
    pub = [data_lines(os.path.join(d, fname)) for d in pubdirs]
    for side, dirs in (("reference", refdirs), ("cleaned", newdirs)):
        same = [data_lines(os.path.join(d, fname)) == pub[i] for i, d in enumerate(dirs)]
        n = len(data_lines(os.path.join(dirs[0], fname))) - 1
        report.append(f"- {fname}, {side}: {n} data rows per run; identical to the published run of the same index: {same}")
# E4: the cleaned build has no opencv_batch; compare the rows of the versions both have
for side, dirs in (("reference", ref_e4), ("cleaned", new_e4)):
    res = []
    for i, d in enumerate(dirs):
        pub = data_lines(os.path.join(pub_e4[i], "e4_preprocessing_agreement.csv"))
        new = data_lines(os.path.join(d, "e4_preprocessing_agreement.csv"))
        head = pub[0].split(",")
        vi = head.index("version")
        pub_rows = [l for l in pub[1:] if l.split(",")[vi] != "opencv_batch"]
        new_rows = [l for l in new[1:] if l.split(",")[vi] != "opencv_batch"]
        res.append((len(new_rows), new[0] == pub[0], new_rows == pub_rows))
    report.append(f"- e4_preprocessing_agreement.csv, {side} (opencv_batch rows left out of both sides): (rows, same header, rows identical) per run: {res}")
for side, dirs in (("reference", ref_ocv), ("cleaned", new_ocv)):
    report.append(f"- CPU-time OpenCV program, {side}: " + ", ".join(open(os.path.join(d, "stdout.txt")).read().count("passed!!").__str__() + " passed" for d in dirs)
                  + " (pass/fail check only; the cleaned build reads seeded noise, the reference a constant image)")
report.append("")

# ---------------------------------------------------------------- the published flag list first
report[2:2] = ["## Changes above the spread and above the 3 % (GPU) or 5 % (host) threshold", ""] + [f"- {f}" for f in flags_all] + [
    f"- E4: {len(e4_flag_all)} flagged (metric, point) changes over all batches, listed at the end of the E4 tables" if e4_flag_all else "- E4: none", ""]
open(OUT, "w").write("\n".join(report) + "\n")
with open(OUT.replace(".md", ".csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table", "number", "bound", "published_median", "published_spread_pct", "published_runs", "reference_median", "reference_spread_pct", "reference_runs",
                "cleaned_median", "cleaned_spread_pct", "cleaned_runs", "drift_pct", "drift_judgement", "cleanup_pct", "cleanup_judgement", "cause"])
    w.writerows(ALL_ROWS)
print(f"{len(ALL_ROWS)} numbers; {len(flags_all)} flagged outside E4; {len(e4_flag_all)} flagged in E4 (all batches)")
