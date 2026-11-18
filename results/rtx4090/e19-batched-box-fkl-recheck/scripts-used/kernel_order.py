#!/usr/bin/env python3
"""The kernels of the first crop of each call of each E19 row (every row with ranges '<row>:call', in the order of the trace) in launch order, and every kernel name's GPU duration (median, min, max over the
calls' kernels), from the SQLite export of the profile-mode nsys run (nothing is run). A kernel belongs to a call when the runtime API call that
enqueued it starts inside the call's NVTX range '<row>:call'.

  kernel_order.py <run.sqlite> [--out report.txt]
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
names = dict(c.execute("select id, value from StringIds"))
ranges = sorted((s, e, t) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null"))
api_start = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
kernels = sorted((api_start.get(corr, -1), s, e, names.get(n, "?"), names.get(d, "?"), gx, gy, gz, bx, by, bz)
                 for s, e, n, d, corr, gx, gy, gz, bx, by, bz in
                 c.execute("select start, end, shortName, demangledName, correlationId, gridX, gridY, gridZ, blockX, blockY, blockZ "
                           "from CUPTI_ACTIVITY_KIND_KERNEL"))
lines = []
rows = []
for s, e, t in ranges:
    if t.endswith(":call") and t[:-5] not in rows:
        rows.append(t[:-5])
for row in rows:
    calls = [(s, e) for s, e, t in ranges if t == row + ":call"]
    ks_all = [[k for k in kernels if s <= k[0] <= e] for s, e in calls]
    per_crop = len(ks_all[0]) // 30
    lines.append(f"{row}: the kernels of crop 0 of the first call, in launch order (name; grid; block; demangled name, 160 characters):")
    for k in ks_all[0][:per_crop]:
        lines.append(f"  {k[3]}; grid {k[5]}x{k[6]}x{k[7]}; block {k[8]}x{k[9]}x{k[10]}; {k[4][:160]}")
    same = all([k[3] for k in ks] == [k[3] for k in ks_all[0][:per_crop]] * 30 for ks in ks_all)
    lines.append(f"  every crop of every call launches the same kernels in this order: {'yes' if same else 'NO'}")
    dur = collections.defaultdict(list)
    for ks in ks_all:
        for k in ks:
            dur[k[3]].append((k[2] - k[1]) / 1000.0)
    lines.append(f"  GPU duration per kernel, us (over the {len(calls)} calls):")
    for n, v in dur.items():
        lines.append(f"    {n}: {len(v)} kernels, median {statistics.median(v):.2f}, min {min(v):.2f}, max {max(v):.2f}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
