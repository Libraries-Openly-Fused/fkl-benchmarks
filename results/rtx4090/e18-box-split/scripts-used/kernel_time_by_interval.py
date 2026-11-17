#!/usr/bin/env python3
"""From the nsys trace of a complete E18 box run (exported to SQLite): the GPU durations of the kernels of one call in the two intervals.

  kernel_time_by_interval.py <e18-box.sqlite> [--out report.txt]

For every variant and each of its NVTX ranges whole_call:<version> and kernel_only:<version>, the calls after the first one (the first is the
warm-up of fklb::timeIt and of fklb::timePreparedLaunch) are taken from the kernels launched in the range, in GPU order (kernel_only: a call is the
kernels after one gpuSpin; whole_call: a call is the variant's kernels per call, counted from the kernel names). For each call:
  kernels  the sum of the GPU durations of its kernels (end - start of each)
  span     from the start of its first kernel to the end of its last kernel
The medians over the 100 timed calls are printed for both intervals, with kernel_only over whole_call.
"""
import argparse
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--out", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
names = dict(c.execute("select id, value from StringIds"))
ranges = {t: (s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null")}
api = {corr: s for corr, s in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
kernels = sorted((s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))


def launched_in(rs, re_):
    return [k for k in kernels if k[3] in api and rs <= api[k[3]] <= re_]


lines = ["version: median over the 100 timed calls, kernel durations summed (us) / first-kernel-start to last-kernel-end span (us); "
         "whole_call | kernel_only | kernel_only / whole_call"]
versions = [t.split(":", 1)[1] for t in ranges if t.startswith("whole_call:")]
for v in sorted(versions, key=lambda v: ranges["whole_call:" + v][0]):
    ks = [k for k in launched_in(*ranges["kernel_only:" + v])]
    calls, cur = [], None
    for k in ks:
        if k[2] == "gpuSpin":
            if cur is not None:
                calls.append(cur)
            cur = []
        elif cur is not None:
            cur.append(k)
    if cur is not None:
        calls.append(cur)
    perCall = len(calls[0])
    wk = [k for k in launched_in(*ranges["whole_call:" + v])]
    wcalls = [wk[i:i + perCall] for i in range(0, len(wk), perCall)][1:]   # the first call is timeIt's warm-up
    res = {}
    for label, cs in (("whole_call", wcalls), ("kernel_only", calls)):
        res[label] = (statistics.median(sum(k[1] - k[0] for k in cl) for cl in cs) / 1000.0,
                      statistics.median((cl[-1][1] - cl[0][0]) for cl in cs) / 1000.0, len(cs))
    w, k = res["whole_call"], res["kernel_only"]
    lines.append(f"{v}: {perCall} kernels per call; whole_call ({w[2]} calls) {w[0]:.2f} / {w[1]:.2f}; kernel_only ({k[2]} calls) {k[0]:.2f} / "
                 f"{k[1]:.2f}; ratio {k[0] / w[0]:.4f} / {k[1] / w[1]:.4f}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
