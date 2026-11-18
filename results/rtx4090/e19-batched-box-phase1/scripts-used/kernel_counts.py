#!/usr/bin/env python3
"""The kernels and the CUDA runtime calls of one call (one frame, 30 crops) of each E19 row, from the SQLite export of the profile-mode nsys run
(nothing is run).

  kernel_counts.py <run.sqlite> [--out report.txt]

The profile mode (E19_PROFILE=1) calls every row once outside any range (a warm-up), then E19_PROFILE_CALLS times, each call in the NVTX range
"<row>:call" (pushed and popped on the host around the call; the GPU work is not waited for inside it). A kernel, memset or memcpy belongs to a
call when the runtime API call that enqueued it (the same correlationId) starts inside the call's range. Per row:
  kernels per call (every call), and per crop (/ 30); the kernels by name per call; memsets and memcpys per call
  the runtime API calls per call by name
  host: the length of the call's range, the host's enqueue of one call (median, min, max)
  GPU: from the start of the call's first kernel to the end of its last kernel, and the sum of its kernels' durations (median, min, max)
"""
import argparse
import collections
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db")
ap.add_argument("--out", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
tables = {r[0] for r in c.execute("select name from sqlite_master where type='table'")}
names = dict(c.execute("select id, value from StringIds"))
ranges = sorted((s, e, t) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null"))
api = sorted((s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, nameId, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME"))
api_start = {corr: s for s, e, n, corr in api}
kernels = [(s, e, names.get(n, "?"), corr) for s, e, n, corr in
           c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL")]
memsets = list(c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMSET")) if "CUPTI_ACTIVITY_KIND_MEMSET" in tables else []
memcpys = list(c.execute("select start, end, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []


def inside(corr, s, e):
    return corr in api_start and s <= api_start[corr] <= e


def stats(v):
    return f"median {statistics.median(v):.1f}, min {min(v):.1f}, max {max(v):.1f}"


lines = [f"{len(kernels)} kernels, {len(memsets)} memsets, {len(memcpys)} memcpys, {len(api)} runtime API calls in the trace"]
rows = []
for s, e, t in ranges:
    if t.endswith(":call") and t[:-5] not in rows:
        rows.append(t[:-5])
for row in rows:
    calls = [(s, e) for s, e, t in ranges if t == row + ":call"]
    per_call, by_name, sets, cpys, host, span, busy, api_names = [], [], [], [], [], [], [], []
    for s, e in calls:
        ks = sorted(k for k in kernels if inside(k[3], s, e))
        per_call.append(len(ks))
        by_name.append(collections.Counter(k[2] for k in ks))
        sets.append(sum(1 for m in memsets if inside(m[2], s, e)))
        cpys.append(sum(1 for m in memcpys if inside(m[2], s, e)))
        api_names.append(collections.Counter(n for a0, a1, n, corr in api if s <= a0 <= e))
        host.append((e - s) / 1000.0)
        if ks:
            span.append((max(k[1] for k in ks) - ks[0][0]) / 1000.0)
            busy.append(sum(k[1] - k[0] for k in ks) / 1000.0)
    lines.append("")
    lines.append(f"{row}: {len(calls)} calls in the ranges '{row}:call'")
    lines.append(f"  kernels per call: {sorted(set(per_call))} (per crop: {sorted(set(n / 30 for n in per_call))}); memsets per call: "
                 f"{sorted(set(sets))}; memcpys per call: {sorted(set(cpys))}")
    same = all(b == by_name[0] for b in by_name)
    lines.append(f"  kernels by name, per call{' (the same in every call)' if same else ' (FIRST CALL; the calls differ)'}:")
    for n, k in sorted(by_name[0].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"    {k:4d}  {n}")
    same_api = all(a == api_names[0] for a in api_names)
    lines.append(f"  runtime API calls by name, per call{' (the same in every call)' if same_api else ' (FIRST CALL; the calls differ)'}:")
    for n, k in sorted(api_names[0].items(), key=lambda x: (-x[1], x[0])):
        lines.append(f"    {k:4d}  {n}")
    lines.append(f"  host, the call's range (the enqueue of one call), us: {stats(host)}")
    if span:
        lines.append(f"  GPU, first kernel start to last kernel end, us: {stats(span)}")
        lines.append(f"  GPU, sum of the kernel durations, us: {stats(busy)}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
