#!/usr/bin/env python3
"""The GPU duration of every kernel name in each NVTX range whole_call:<row> and kernel_only:<row> of the nsys run of an E19 timing window
(exported to SQLite; nothing is run). A kernel belongs to a range when the runtime API call that enqueued it starts inside the range; gpuSpin is
left out. Per range and kernel name: the count, and the median, min and max duration in us, under the CUDA trace (not a timing).

  kernel_durations.py <run.sqlite> [--out report.txt]
"""
import argparse, sqlite3, statistics
ap = argparse.ArgumentParser(); ap.add_argument("db"); ap.add_argument("--out", default=None); args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
names = dict(c.execute("select id, value from StringIds"))
ranges = sorted((s, e, t) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null")
                if t.startswith(("whole_call:", "kernel_only:")))
api = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
ks = [(s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL")]
lines = []
for s, e, t in ranges:
    by = {}
    for k in ks:
        if k[3] in api and s <= api[k[3]] <= e and k[2] != "gpuSpin":
            by.setdefault(k[2], []).append((k[1] - k[0]) / 1000.0)
    lines.append(f"{t}:")
    for name, d in by.items():
        lines.append(f"  {name}: {len(d)} kernels, median {statistics.median(d):.2f}, min {min(d):.2f}, max {max(d):.2f} us")
text = "\n".join(lines); print(text)
if args.out: open(args.out, "w").write(text + "\n")
