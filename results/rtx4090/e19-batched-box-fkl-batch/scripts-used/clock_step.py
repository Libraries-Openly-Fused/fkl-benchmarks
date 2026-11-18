#!/usr/bin/env python3
"""Where the kernels of one E19 row change duration within an NVTX range of the nsys run (nothing is run).

  clock_step.py <run.sqlite> <range> [--threshold us] [--out report.txt]

The kernels of the range (gpuSpin left out) in launch order, grouped in calls of the row's kernels per call (taken from the whole_call range:
its kernels / 101 calls). For every call, the median kernel duration. Prints the calls whose median is above the threshold and the first call
below it, with the call's GPU start on the wall clock (TARGET_INFO_SESSION_START_TIME + the trace time).
"""
import argparse
import datetime
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("range"); ap.add_argument("--threshold", type=float, default=2.03); ap.add_argument("--out", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
TZ = datetime.timezone(datetime.timedelta(hours=1))
t0 = list(c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME"))[0][0] / 1e9
names = dict(c.execute("select id, value from StringIds"))
ranges = {t: (s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null")}
api = dict(c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME"))
kernels = sorted((s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))
row = args.range.split(":", 1)[1]
ws, we = ranges["whole_call:" + row]
per_call = sum(1 for k in kernels if k[3] in api and ws <= api[k[3]] <= we and k[2] != "gpuSpin") // 101
s, e = ranges[args.range]
ks = [k for k in kernels if k[3] in api and s <= api[k[3]] <= e and k[2] != "gpuSpin"]
calls = [ks[i:i + per_call] for i in range(0, len(ks), per_call)]
meds = [statistics.median((k[1] - k[0]) / 1000.0 for k in call) for call in calls]
local = lambda t: datetime.datetime.fromtimestamp(t0 + t / 1e9, TZ).strftime("%H:%M:%S.%f")[:-3]
lines = [f"{args.range}: {len(calls)} calls of {per_call} kernels (the first is the warm-up call); median kernel duration per call, us"]
first_below = next((i for i, m in enumerate(meds) if m < args.threshold), None)
above = [i for i, m in enumerate(meds) if m >= args.threshold]
lines.append(f"  calls with a median >= {args.threshold}: {len(above)}; their medians {min(meds[i] for i in above):.2f}-{max(meds[i] for i in above):.2f}" if above else
             f"  no call with a median >= {args.threshold}")
if first_below is not None:
    lines.append(f"  the first call below {args.threshold}: call {first_below} (0 = the warm-up), GPU start {local(calls[first_below][0][0])}, median {meds[first_below]:.2f}")
    below = meds[first_below:]
    lines.append(f"  from it on: medians {min(below):.2f}-{max(below):.2f}, median of medians {statistics.median(below):.2f}")
if above:
    lines.append(f"  before it: median of medians {statistics.median(meds[i] for i in above):.2f}; last call above at {local(calls[above[-1]][0][0])}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "a").write(text + "\n")
