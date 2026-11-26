"""The summary of the E18 box re-time window of RT-A2 (window.sh): A = e18_box_hip at 976e42b with FKL b10fc66, B = at 73a9650 with FKL
304492a, and old = B of results/rx9060/e18-box-retime-b10fc66/ (#62: 2160c70 with FKL b10fc66; its window's quiet-run1..3-B, its
rocprofv3-B trace). Made after the window, without the GPU. This is #62's retime_report.py with only "old" pointed at #62's B runs.

usage: python3 -I retime_report.py <window dir> <#62's window dir (results/rx9060/e18-box-retime-b10fc66/window)> <out dir>

Per binary, version and interval: the median of the three quiet runs' means, with their min-max, and the spread = (max - min) / median.
The comparisons B against A, B against old and A against old: change = new / old - 1; it exceeds the spread when |change| is above the larger
of the two spreads. GPU- or host-bound is the crop sweep's criterion (results/rx9060/e19-repeat-check/): a version is GPU-bound when its
kernel_only and whole_call medians differ by less than 3 %, else host-bound; the label is that of the new side. A change is flagged above
3 % (GPU-bound) or 5 % (host-bound). No cause is computed here.
Also: the unused runs against their binary's quiet median; the clocks (the warm-up, the interval boundaries, and the gpu_clocks.csv samples
inside each run's timing); and whether the data rows of the checks, agreement, attribution and ties files of every run equal the record's.
The rocprofv3 runs (rocprofv3-A, rocprofv3-B) are kept apart, never in the quiet runs' times:
  - their means against their binary's quiet median (the cost of the tracing);
  - the kernels of every timed range and their GPU durations, for the record's trace (old) and A's and B's: per range and kernel, the number
    of dispatches and the median and mean of End - Start (us); per kernel_only range, the per-call GPU span after the spin and the sum of the
    kernel durations per call. A kernel belongs to a range when the host API call that launched it (by correlation id) started inside the
    range. This is the method of results/rx9060/e19-crop-sweep/analysis/kernel_durations.py, with E18's trace prefix and rows;
  - the record's rocprof_calls.py --spin (results/rx9060/e18-box-phase1/rocprofv3/) on A's and B's traces: the kernels of every call and
    the spin coverage of every kernel_only iteration."""
import csv
import gzip
import os
import re
import statistics
import subprocess
import sys

WIN, REC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(OUT, exist_ok=True)
VERSIONS = ["rpp", "rpp_ltrb", "fkl_box_halfpixel", "fkl_half_once"]
INTERVALS = ["whole_call", "kernel_only"]
ORDER = ["unused-A", "unused-B"] + [f"quiet-run{k}-{s}" for k in (1, 2, 3) for s in "AB"] + ["rocprofv3-A", "rocprofv3-B"]   # the window's order
OTHER = ["e18_box_checks_GPU_AMD.csv", "e18_box_agreement_GPU_AMD.csv", "e18_box_attribution_GPU_AMD.csv", "e18_box_ties_GPU_AMD.csv"]


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


def timing(run_dir):
    t = {(r["version"], r["interval"]): r for r in rows(os.path.join(run_dir, "e18_box_GPU_AMD.csv"))}
    missing = [(v, i) for v in VERSIONS for i in INTERVALS if (v, i) not in t]
    if missing:
        raise SystemExit(f"{run_dir}: rows missing {missing}")
    return t


RUN_DIRS = [os.path.join(WIN, r) for r in ORDER if os.path.isdir(os.path.join(WIN, r))]
runs = {"old": [os.path.join(REC, f"quiet-run{k}-B") for k in (1, 2, 3)],
        "A": [os.path.join(WIN, f"quiet-run{k}-A") for k in (1, 2, 3)],
        "B": [os.path.join(WIN, f"quiet-run{k}-B") for k in (1, 2, 3)]}
T = {side: [timing(d) for d in dirs] for side, dirs in runs.items()}

summary = {}
for side in ("old", "A", "B"):
    for v in VERSIONS:
        for i in INTERVALS:
            means = [float(t[(v, i)]["mean_ms"]) for t in T[side]]
            sclk = [float(t[(v, i)][k]) for t in T[side] for k in ("sclk_before_mhz", "sclk_after_mhz")]
            med = statistics.median(means)
            summary[(side, v, i)] = {"side": side, "version": v, "interval": i, "runs": len(means), "median_of_means_ms": med,
                                     "min_of_means_ms": min(means), "max_of_means_ms": max(means),
                                     "spread_pct": 100.0 * (max(means) - min(means)) / med,
                                     "min_of_mins_ms": min(float(t[(v, i)]["min_ms"]) for t in T[side]),
                                     "sclk_before_min_mhz": min(float(t[(v, i)]["sclk_before_mhz"]) for t in T[side]),
                                     "sclk_after_min_mhz": min(float(t[(v, i)]["sclk_after_mhz"]) for t in T[side]),
                                     "sclk_after_max_mhz": max(float(t[(v, i)]["sclk_after_mhz"]) for t in T[side]),
                                     "sclk_min_at_boundaries_mhz": min(sclk)}
with open(os.path.join(OUT, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(next(iter(summary.values())).keys()))
    w.writeheader()
    w.writerows(summary.values())

bound = {}
with open(os.path.join(OUT, "bound.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["side", "version", "whole_call_ms", "kernel_only_ms", "kernel_only_vs_whole_call_pct", "bound"])
    for side in ("old", "A", "B"):
        for v in VERSIONS:
            wc, ko = summary[(side, v, "whole_call")]["median_of_means_ms"], summary[(side, v, "kernel_only")]["median_of_means_ms"]
            d = 100.0 * (ko / wc - 1)
            bound[(side, v)] = "GPU" if abs(d) < 3.0 else "host"
            w.writerow([side, v, f"{wc:.6f}", f"{ko:.6f}", f"{d:+.2f}", bound[(side, v)]])

COMPARISONS = [("B against A, this window", "A", "B"), ("B against old", "old", "B"), ("A against old", "old", "A")]
comp = []
for label, o, n in COMPARISONS:
    for v in VERSIONS:
        for i in INTERVALS:
            so, sn = summary[(o, v, i)], summary[(n, v, i)]
            ch = 100.0 * (sn["median_of_means_ms"] / so["median_of_means_ms"] - 1)
            spread = max(so["spread_pct"], sn["spread_pct"])
            b = bound[(n, v)]
            limit = 3.0 if b == "GPU" else 5.0
            comp.append({"comparison": label, "version": v, "interval": i, "old_ms": so["median_of_means_ms"], "new_ms": sn["median_of_means_ms"],
                         "change_pct": ch, "spread_old_pct": so["spread_pct"], "spread_new_pct": sn["spread_pct"],
                         "exceeds_spread": "yes" if abs(ch) > spread else "no", "bound_new": b, "flag_limit_pct": limit,
                         "flag": "FLAG" if abs(ch) > limit else "no"})
with open(os.path.join(OUT, "compare.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(comp[0].keys()))
    w.writeheader()
    w.writerows(comp)

unused = []
for side in ("A", "B"):
    d = os.path.join(WIN, f"unused-{side}")
    if not os.path.isdir(d):
        continue
    t = timing(d)
    for v in VERSIONS:
        for i in INTERVALS:
            m = float(t[(v, i)]["mean_ms"])
            unused.append({"run": f"unused-{side}", "version": v, "interval": i, "mean_ms": m,
                           "vs_quiet_median_pct": 100.0 * (m / summary[(side, v, i)]["median_of_means_ms"] - 1)})
with open(os.path.join(OUT, "unused.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "version", "interval", "mean_ms", "vs_quiet_median_pct"])
    w.writeheader()
    w.writerows(unused)

# the clocks: the warm-up (console), and the gpu_clocks.csv samples inside each run's timing (the realtime ns the program prints)
WARM = re.compile(r"clock warm-up: one gpuSpin of (\d+) ms; sclk (\d+) MHz before it, (\d+) MHz after it")
SPAN = re.compile(r"E18 timing \(realtime ns\): warm-up start (\d+), timing start (\d+), timing end (\d+)")
samples = []
gc = os.path.join(WIN, "gpu_clocks.csv")
if os.path.exists(gc):
    with open(gc) as f:
        for r in csv.DictReader(f):
            try:
                samples.append((int(r["time_ns"]), int(r["sclk_hz"]) / 1e6, r.get("power_uw", ""), r.get("temp_edge_mc", ""),
                                r.get("temp_junction_mc", ""), r.get("temp_mem_mc", "")))
            except (ValueError, TypeError, KeyError):
                pass
clock = []
for d in RUN_DIRS:
    out = open(os.path.join(d, "stdout.txt")).read()
    wm, sm = WARM.search(out), SPAN.search(out)
    row = {"run": os.path.basename(d), "warmup_ms": wm.group(1) if wm else "", "sclk_before_warmup_mhz": wm.group(2) if wm else "",
           "sclk_after_warmup_mhz": wm.group(3) if wm else ""}
    if sm and samples:
        t0, t1 = int(sm.group(2)), int(sm.group(3))
        inside = [s for s in samples if t0 <= s[0] <= t1]
        sc = [s[1] for s in inside]
        pw = [int(s[2]) / 1e6 for s in inside if str(s[2]).isdigit()]
        tj = [int(s[4]) / 1000 for s in inside if str(s[4]).isdigit()]
        row.update({"timing_seconds": f"{(t1 - t0) / 1e9:.3f}", "samples": len(sc), "sclk_min_mhz": f"{min(sc):.0f}" if sc else "",
                    "sclk_median_mhz": f"{statistics.median(sc):.0f}" if sc else "", "sclk_max_mhz": f"{max(sc):.0f}" if sc else "",
                    "power_median_w": f"{statistics.median(pw):.1f}" if pw else "", "temp_junction_max_c": f"{max(tj):.1f}" if tj else ""})
    clock.append(row)
with open(os.path.join(OUT, "clock.csv"), "w", newline="") as f:
    keys = ["run", "warmup_ms", "sclk_before_warmup_mhz", "sclk_after_warmup_mhz", "timing_seconds", "samples", "sclk_min_mhz", "sclk_median_mhz",
            "sclk_max_mhz", "power_median_w", "temp_junction_max_c"]
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    for r in clock:
        w.writerow({k: r.get(k, "") for k in keys})


# the data rows of the other output files against the record's quiet-run1
def data(path):
    with open(path) as f:
        return [line for line in f if not line.startswith("#")]


ident = []
for d in RUN_DIRS:
    for name in OTHER:
        p, ref = os.path.join(d, name), os.path.join(REC, "quiet-run1-B", name)
        same = os.path.exists(p) and data(p) == data(ref)
        ident.append(f"{os.path.basename(d)} {name}: {'identical to the record' if same else 'DIFFERENT from the record'}")
with open(os.path.join(OUT, "checks-identity.txt"), "w") as f:
    f.write("\n".join(ident) + "\n")

md = ["# E18 box re-time after the pin move to FKL 304492a (RT-A2): the summary", "",
      "A = e18_box_hip at 976e42b with FKL b10fc66; B = at 73a9650 with FKL 304492a; old = B of `results/rx9060/e18-box-retime-b10fc66/` (#62, 2026-11-24).", "",
      "## The times: the median of the three quiet runs' means (min-max), ms", "",
      "| version | interval | old | A | B |", "|---|---|--:|--:|--:|"]
for v in VERSIONS:
    for i in INTERVALS:
        cells = [f"{summary[(s, v, i)]['median_of_means_ms']:.4f} ({summary[(s, v, i)]['min_of_means_ms']:.4f}-"
                 f"{summary[(s, v, i)]['max_of_means_ms']:.4f}), spread {summary[(s, v, i)]['spread_pct']:.1f} %" for s in ("old", "A", "B")]
        md.append(f"| `{v}` | {i} | " + " | ".join(cells) + " |")
md += ["", "## The comparisons", "",
       "| comparison | version, interval | old (ms) | new (ms) | change | spread old / new | exceeds the larger spread? | bound (new) | flag |",
       "|---|---|--:|--:|--:|--:|---|---|---|"]
for c in comp:
    md.append(f"| {c['comparison']} | `{c['version']}` {c['interval']} | {c['old_ms']:.4f} | {c['new_ms']:.4f} | {c['change_pct']:+.1f} % | "
              f"{c['spread_old_pct']:.1f} / {c['spread_new_pct']:.1f} % | {c['exceeds_spread']} | {c['bound_new']} | "
              f"{c['flag'] + (' (> ' + str(int(c['flag_limit_pct'])) + ' %)' if c['flag'] == 'FLAG' else '')} |")
md += ["", "## GPU- or host-bound (kernel_only against whole_call; GPU-bound below 3 %)", "", "| side | version | kernel_only vs whole_call | bound |",
       "|---|---|--:|---|"]
for side in ("old", "A", "B"):
    for v in VERSIONS:
        wc, ko = summary[(side, v, "whole_call")]["median_of_means_ms"], summary[(side, v, "kernel_only")]["median_of_means_ms"]
        md.append(f"| {side} | `{v}` | {100.0 * (ko / wc - 1):+.1f} % | {bound[(side, v)]} |")
md += ["", "## The unused runs against their binary's quiet median", ""]
for u in unused:
    md.append(f"- {u['run']} `{u['version']}` {u['interval']}: {u['vs_quiet_median_pct']:+.1f} %")
md += ["", "## The clocks (`clock.csv`; sclk at the interval boundaries in `summary.csv`)", ""]
for r in clock:
    md.append(f"- {r['run']}: sclk {r['sclk_before_warmup_mhz']} MHz before the warm-up, {r['sclk_after_warmup_mhz']} after; timing "
              f"{r.get('timing_seconds', '')} s, {r.get('samples', '')} samples, sclk min / median / max {r.get('sclk_min_mhz', '')} / "
              f"{r.get('sclk_median_mhz', '')} / {r.get('sclk_max_mhz', '')} MHz, power median {r.get('power_median_w', '')} W, "
              f"junction max {r.get('temp_junction_max_c', '')} °C")
# ---- the rocprofv3 runs: apart from the quiet runs
traced = []
for side in ("A", "B"):
    d = os.path.join(WIN, f"rocprofv3-{side}")
    if not os.path.isdir(d):
        continue
    t = timing(d)
    for v in VERSIONS:
        for i in INTERVALS:
            m = float(t[(v, i)]["mean_ms"])
            traced.append({"run": f"rocprofv3-{side}", "version": v, "interval": i, "mean_ms": m,
                           "vs_quiet_median_pct": 100.0 * (m / summary[(side, v, i)]["median_of_means_ms"] - 1)})
with open(os.path.join(OUT, "rocprofv3-times.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["run", "version", "interval", "mean_ms", "vs_quiet_median_pct"])
    w.writeheader()
    w.writerows(traced)


def trace_folder(top):
    """The folder under top that holds e18box_kernel_trace.csv (rocprofv3 may write it in a subfolder)."""
    for root, _, files in os.walk(top):
        if "e18box_kernel_trace.csv" in files or "e18box_kernel_trace.csv.gz" in files:
            return root
    return None


def trace_rows(folder, name):
    p = os.path.join(folder, name)
    f = gzip.open(p + ".gz", "rt", newline="") if os.path.exists(p + ".gz") else open(p, newline="")
    with f:
        return list(csv.DictReader(f))


RANGE = re.compile(r"(whole_call|kernel_only):(" + "|".join(VERSIONS) + r")$")
durations, spans = [], []
for label, top in (("old", os.path.join(REC, "rocprofv3-B", "trace")), ("A", os.path.join(WIN, "rocprofv3-A", "trace")),
                   ("B", os.path.join(WIN, "rocprofv3-B", "trace"))):
    folder = trace_folder(top) if os.path.isdir(top) else None
    if folder is None:
        continue
    marks = [m for m in trace_rows(folder, "e18box_marker_api_trace.csv") if RANGE.match(m["Function"])]
    api = {a["Correlation_Id"]: int(a["Start_Timestamp"]) for a in trace_rows(folder, "e18box_hip_api_trace.csv")}
    ks = trace_rows(folder, "e18box_kernel_trace.csv")
    for m in marks:
        s0, e0 = int(m["Start_Timestamp"]), int(m["End_Timestamp"])
        inside = sorted((k for k in ks if k["Correlation_Id"] in api and s0 <= api[k["Correlation_Id"]] <= e0), key=lambda k: int(k["Start_Timestamp"]))
        by = {}
        for k in inside:
            name = re.match(r"(?:void )?([\w:]+)", k["Kernel_Name"]).group(1)
            by.setdefault(name, []).append((int(k["End_Timestamp"]) - int(k["Start_Timestamp"])) / 1e3)
        for name, d in by.items():
            durations.append({"trace": label, "range": m["Function"], "kernel": name, "dispatches": len(d), "median_us": statistics.median(d),
                              "mean_us": statistics.mean(d)})
        if m["Function"].startswith("kernel_only"):
            calls, cur = [], None
            for k in inside:
                if "gpuSpin" in k["Kernel_Name"]:
                    cur = []
                    calls.append(cur)
                elif cur is not None:
                    cur.append(k)
            span = [(int(c[-1]["End_Timestamp"]) - int(c[0]["Start_Timestamp"])) / 1e3 for c in calls if c]
            busy = [sum(int(k["End_Timestamp"]) - int(k["Start_Timestamp"]) for k in c) / 1e3 for c in calls if c]
            if span:
                spans.append({"trace": label, "range": m["Function"], "calls": len(span), "gpu_span_median_us": statistics.median(span),
                              "kernel_sum_median_us": statistics.median(busy)})
with open(os.path.join(OUT, "kernel-durations.txt"), "w") as f:
    f.write("trace\trange\tkernel\tdispatches\tmedian_us\tmean_us\n")
    for r in durations:
        f.write(f"{r['trace']}\t{r['range']}\t{r['kernel']}\t{r['dispatches']}\t{r['median_us']:.2f}\t{r['mean_us']:.2f}\n")
    f.write("\n")
    for r in spans:
        f.write(f"{r['trace']}\t{r['range']}\tper call after the spin: GPU span median {r['gpu_span_median_us']:.2f} us, sum of kernel durations "
                f"median {r['kernel_sum_median_us']:.2f} us ({r['calls']} calls)\n")
CALLS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "e18-box-phase1", "rocprofv3", "rocprof_calls.py")
for side in ("A", "B"):
    folder = trace_folder(os.path.join(WIN, f"rocprofv3-{side}", "trace")) if os.path.isdir(os.path.join(WIN, f"rocprofv3-{side}", "trace")) else None
    if folder is None:
        continue
    r = subprocess.run([sys.executable, "-I", CALLS, folder, "e18box", "--spin", "--out", os.path.join(OUT, f"rocprofv3-{side}-calls.txt"),
                        "--tsv", os.path.join(OUT, f"rocprofv3-{side}-calls.tsv")], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"rocprof_calls.py on {folder}: exit {r.returncode}\n{r.stderr}")

md += ["", "## The rocprofv3 runs (apart: never in the times above)", "",
       "Their means against their binary's quiet median (`rocprofv3-times.csv`):", ""]
for u in traced:
    md.append(f"- {u['run']} `{u['version']}` {u['interval']}: {u['vs_quiet_median_pct']:+.1f} %")
md += ["", "The median GPU duration of each kernel per timed range, us (`kernel-durations.txt`; dispatches in brackets):", "",
       "| range | kernel | old | A | B |", "|---|---|--:|--:|--:|"]
keys = []
for r in durations:
    if (r["range"], r["kernel"]) not in keys:
        keys.append((r["range"], r["kernel"]))
for rng, kern in keys:
    cell = {}
    for r in durations:
        if (r["range"], r["kernel"]) == (rng, kern):
            cell[r["trace"]] = f"{r['median_us']:.2f} ({r['dispatches']})"
    md.append(f"| {rng} | `{kern}` | {cell.get('old', '')} | {cell.get('A', '')} | {cell.get('B', '')} |")
md += ["", "Per call after the spin (kernel_only ranges), median GPU span / sum of kernel durations, us:", "", "| range | old | A | B |", "|---|--:|--:|--:|"]
for rng in [f"kernel_only:{v}" for v in VERSIONS]:
    cell = {r["trace"]: f"{r['gpu_span_median_us']:.2f} / {r['kernel_sum_median_us']:.2f}" for r in spans if r["range"] == rng}
    md.append(f"| {rng} | {cell.get('old', '')} | {cell.get('A', '')} | {cell.get('B', '')} |")
md += ["", "The kernels of every call and the spin coverage of A's and B's traces: `rocprofv3-A-calls.txt`, `rocprofv3-B-calls.txt` (the record's "
       "rocprof_calls.py --spin)."]
md += ["", "## The check rows", "", f"{sum('identical' in x and 'DIFFERENT' not in x for x in ident)} of {len(ident)} files have the record's data rows "
       "(`checks-identity.txt`)."]
with open(os.path.join(OUT, "report.md"), "w") as f:
    f.write("\n".join(md) + "\n")
print("\n".join(md))
