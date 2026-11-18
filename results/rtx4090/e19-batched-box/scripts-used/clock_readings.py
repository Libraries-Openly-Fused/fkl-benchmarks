#!/usr/bin/env python3
"""The nvidia-smi readings of an E18 window, as changes, with the runs of the window (nothing is run).

  clock_readings.py <window folder> [--runs warmup,quiet-run1,...] [--out report.txt]

Reads <window folder>/gpu_clocks.csv (every 100 ms: timestamp, clocks.sm, clocks.mem, pstate, utilization.gpu, power.draw, temperature.gpu) and
each run's load.txt (its start and end, to the second). Prints every reading whose SM clock, utilisation or pstate differs from the previous
reading, with the run it falls in. nvidia-smi repeats a reading until the driver updates it, about every 0.5 s, so a reading at time t states
the values at some moment of (t - 0.5 s, t], and utilization.gpu is the share of the time before that moment during which a kernel ran.
Then, for each run, the 2,550 MHz readings that fall in it.
"""
import argparse
import csv
import datetime
import os
import re

ap = argparse.ArgumentParser()
ap.add_argument("folder"); ap.add_argument("--runs", default="warmup,quiet-run1,quiet-run2,quiet-run3,nsys,ncu"); ap.add_argument("--out", default=None)
args = ap.parse_args()
TZ = datetime.timezone(datetime.timedelta(hours=1))
runs = []
for r in args.runs.split(","):
    p = os.path.join(args.folder, r, "load.txt")
    if not os.path.exists(p):
        continue
    text = open(p).read()
    s = re.search(r"start (\S+?)[;\s]", text + " ").group(1)
    e = re.search(r"end (\S+)", text).group(1)
    runs.append((r, datetime.datetime.fromisoformat(s).timestamp(), datetime.datetime.fromisoformat(e).timestamp() + 1.0))   # the end to the second


def run_of(t):
    for r, s, e in runs:
        if s <= t < e:
            return r
    return "between runs"


lines = ["runs (start and end to the second): " + "; ".join(f"{r} {datetime.datetime.fromtimestamp(s, TZ):%H:%M:%S}-"
                                                          f"{datetime.datetime.fromtimestamp(e - 1.0, TZ):%H:%M:%S}" for r, s, e in runs),
         "time | SM MHz | utilisation % | pstate | power | run (the readings where the SM clock, the utilisation or the pstate changes)"]
last, low = None, {}
for row in csv.reader(open(os.path.join(args.folder, "gpu_clocks.csv"))):
    if len(row) < 7:
        continue
    t = datetime.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=TZ)
    sm, util, ps = int(row[1].split()[0]), int(row[4].split()[0]), row[3].strip()
    key = (sm, util, ps)
    r = run_of(t.timestamp())
    if key != last:
        lines.append(f"{t:%H:%M:%S.%f}"[:-3] + f" | {sm} | {util} | {ps} | {row[5].strip()} | {r}")
        last = key
    if sm == 2550:
        low.setdefault(r, []).append((f"{t:%H:%M:%S.%f}"[:-3], util))
lines.append("")
lines.append("2,550 MHz readings per run: first-last (count; the utilisation readings among them)")
for r in [x[0] for x in runs] + ["between runs"]:
    if r in low:
        v = low[r]
        lines.append(f"{r}: {v[0][0]}-{v[-1][0]} ({len(v)}; utilisation {sorted(set(u for _, u in v))} %)")
    else:
        lines.append(f"{r}: none")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
