#!/usr/bin/env python3
"""From the nsys trace of a complete E18 box split run (exported to SQLite): the GPU duration of every kernel of one call, per row and interval.

  kernel_durations.py <e18-box-split.sqlite> [--out report.txt] [--csv kernels.csv]

The calls are taken as kernel_time_by_interval.py takes them: in each NVTX range kernel_only:<row>, the kernels launched after each gpuSpin (the
100 timed calls; the warm-up call has no spin before it and is not counted); in each range whole_call:<row>, the row's kernels per call in GPU
order, the first call (fklb::timeIt's warm-up) left out. For every kernel position of a call: its name, and the median, min and max of its
duration (GPU end - GPU start) over the 100 timed calls. Then the sums of the medians that compare the split rows with the pipeline rows:
fkl_resize + fkl_box against fkl_trunc_halfpixel's fused kernel, and npp_resize + npp_box against npp's two kernels.
"""
import argparse
import csv
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
names = dict(c.execute("select id, value from StringIds"))
ranges = {t: (s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null")}
api = {corr: s for corr, s in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
kernels = sorted((s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))


def launched_in(rs, re_):
    return [k for k in kernels if k[3] in api and rs <= api[k[3]] <= re_]


rows = [t.split(":", 1)[1] for t in ranges if t.startswith("whole_call:")]
rows.sort(key=lambda v: ranges["whole_call:" + v][0])
table = []   # (row, interval, position, name, calls, median, min, max) in us
median = {}  # (row, interval) -> list of per-position medians
for v in rows:
    calls, cur = [], None
    for k in launched_in(*ranges["kernel_only:" + v]):
        if k[2] == "gpuSpin":
            if cur is not None:
                calls.append(cur)
            cur = []
        elif cur is not None:
            cur.append(k)
    if cur is not None:
        calls.append(cur)
    perCall = len(calls[0])
    wk = launched_in(*ranges["whole_call:" + v])
    wcalls = [wk[i:i + perCall] for i in range(0, len(wk), perCall)][1:]
    for interval, cs in (("whole_call", wcalls), ("kernel_only", calls)):
        if any(len(cl) != perCall for cl in cs):
            raise SystemExit(f"{v} {interval}: a call without {perCall} kernels")
        meds = []
        for p in range(perCall):
            pos_names = {cl[p][2] for cl in cs}
            if len(pos_names) != 1:
                raise SystemExit(f"{v} {interval}: position {p} has several kernels {pos_names}")
            d = [(cl[p][1] - cl[p][0]) / 1000.0 for cl in cs]
            m = statistics.median(d)
            meds.append(m)
            table.append((v, interval, p + 1, pos_names.pop(), len(cs), m, min(d), max(d)))
        median[(v, interval)] = meds
lines = ["row / interval / kernel position: kernel, calls, GPU duration median (min-max) in us"]
for r in table:
    lines.append(f"{r[0]} / {r[1]} / {r[2]}: {r[3]}, {r[4]} calls, {r[5]:.2f} ({r[6]:.2f}-{r[7]:.2f}) us")
lines.append("")
lines.append("sums of the kernel medians (us):")
for interval in ("whole_call", "kernel_only"):
    try:
        fr, fb, fused = sum(median[("fkl_resize", interval)]), sum(median[("fkl_box", interval)]), sum(median[("fkl_trunc_halfpixel", interval)])
        lines.append(f"{interval}: fkl_resize {fr:.2f} + fkl_box {fb:.2f} = {fr + fb:.2f} against the fused kernel of fkl_trunc_halfpixel {fused:.2f}: "
                     f"(fkl_resize + fkl_box) / fused = {(fr + fb) / fused:.4f}")
        nr, nb, npp = sum(median[("npp_resize", interval)]), sum(median[("npp_box", interval)]), median[("npp", interval)]
        lines.append(f"{interval}: npp_resize {nr:.2f} + npp_box {nb:.2f} = {nr + nb:.2f} against npp's two kernels {npp[0]:.2f} + {npp[1]:.2f} = "
                     f"{sum(npp):.2f}: (npp_resize + npp_box) / npp = {(nr + nb) / sum(npp):.4f}")
        lines.append(f"{interval}: fkl_box / npp_box {fb / nb:.4f}; fkl_resize / npp_resize {fr / nr:.4f}; fused / npp's two kernels {fused / sum(npp):.4f}")
    except KeyError as e:
        lines.append(f"{interval}: a row is missing from the trace: {e}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["row", "interval", "kernel_position", "kernel", "calls", "median_us", "min_us", "max_us"])
        for r in table:
            w.writerow([r[0], r[1], r[2], r[3], r[4], f"{r[5]:.3f}", f"{r[6]:.3f}", f"{r[7]:.3f}"])
