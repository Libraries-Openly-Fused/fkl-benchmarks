#!/usr/bin/env python3
"""The SM clock that nvidia-smi read while each row's kernels ran in an nsys run of an E18 window (nothing is run).

  clock_by_row.py <run.sqlite> <gpu_clocks.csv> [--out report.txt] [--steps row,row,...]

gpu_clocks.csv: the window's nvidia-smi samples (timestamp, clocks.sm, clocks.mem, pstate, utilization.gpu, power.draw, temperature.gpu),
every 100 ms. The values change only every ~0.5 s (nvidia-smi repeats a reading until the driver updates it), so a reading at time t states the
clock at some moment of (t - 0.5 s, t]. The kernels' GPU start times are put on the wall clock with the trace's session start
(TARGET_INFO_SESSION_START_TIME.utcEpochNs + the trace time). For every NVTX range whole_call:<row> and kernel_only:<row>, each kernel (gpuSpin
left out) is counted as
  at_2550 / at_2685   every reading within 0.5 s before and after its start is that clock
  transition         the readings within 0.5 s of its start differ: the clock changed within a second of it
The 2,550 MHz segments of the window (first and last reading) are printed first. --steps: for these rows, the GPU durations of their kernels in
launch order, in groups of 10 (the median of each group), to show whether a row's kernels change duration within the range.
"""
import argparse
import bisect
import csv
import datetime
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("clocks"); ap.add_argument("--out", default=None); ap.add_argument("--steps", default="")
args = ap.parse_args()
TZ = datetime.timezone(datetime.timedelta(hours=1))   # the window's local time, +01:00


def epoch(ts):
    d = datetime.datetime.strptime(ts.strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=TZ)
    return d.timestamp()


samples = []
for r in csv.reader(open(args.clocks)):
    if len(r) >= 7:
        samples.append((epoch(r[0]), int(r[1].split()[0]), int(r[4].split()[0])))
times = [s[0] for s in samples]
c = sqlite3.connect(args.db).cursor()
t0 = list(c.execute("select utcEpochNs from TARGET_INFO_SESSION_START_TIME"))[0][0] / 1e9
names = dict(c.execute("select id, value from StringIds"))
ranges = sorted((s, e, t) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null and end is not null"))
api = {corr: s for corr, s in c.execute("select correlationId, start from CUPTI_ACTIVITY_KIND_RUNTIME")}
kernels = sorted((s, e, names.get(n, "?"), corr) for s, e, n, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL"))


def local(t):
    return datetime.datetime.fromtimestamp(t, TZ).strftime("%H:%M:%S.%f")[:-3]


def clocks_near(t):
    lo, hi = bisect.bisect_left(times, t - 0.5), bisect.bisect_right(times, t + 0.5)
    return {samples[i][1] for i in range(lo, hi)}


lines = [f"trace session start {local(t0)}; {len(samples)} clock readings {local(times[0])}-{local(times[-1])}"]
seg, segs = None, []
for t, sm, util in samples:
    if sm == 2550 and seg is None:
        seg = [t, t, []]
    if sm == 2550:
        seg[1] = t
        seg[2].append(util)
    elif seg is not None:
        segs.append(seg)
        seg = None
if seg is not None:
    segs.append(seg)
for a, b, u in segs:
    lines.append(f"2,550 MHz readings {local(a)}-{local(b)} ({len(u)} readings; utilisation readings {sorted(set(u))} %)")
lines.append("range: kernels (gpuSpin left out) at 2,550 / at 2,685 / at a transition; range start-end (local time)")
for s, e, text in ranges:
    if not (text.startswith("whole_call:") or text.startswith("kernel_only:")):
        continue
    ks = [k for k in kernels if k[3] in api and s <= api[k[3]] <= e and k[2] != "gpuSpin"]
    cls = {"at_2550": 0, "at_2685": 0, "transition": 0}
    for k in ks:
        near = clocks_near(t0 + k[0] / 1e9)
        cls["at_2550" if near == {2550} else "at_2685" if near == {2685} else "transition"] += 1
    if cls["at_2550"] or cls["transition"]:
        lines.append(f"{text}: {cls['at_2550']} / {cls['at_2685']} / {cls['transition']}; {local(t0 + s / 1e9)}-{local(t0 + e / 1e9)}")
    else:
        lines.append(f"{text}: 0 / {cls['at_2685']} / 0")
for row in [r for r in args.steps.split(",") if r]:
    for interval in ("whole_call", "kernel_only"):
        rr = [(s, e) for s, e, t in ranges if t == f"{interval}:{row}"]
        if not rr:
            continue
        s, e = rr[0]
        ks = [k for k in kernels if k[3] in api and s <= api[k[3]] <= e and k[2] != "gpuSpin"]
        groups = [ks[i:i + 10] for i in range(0, len(ks), 10)]
        meds = " ".join(f"{statistics.median((k[1] - k[0]) / 1000.0 for k in g):.2f}" for g in groups)
        lines.append(f"{interval}:{row}: {len(ks)} kernels (the first is the warm-up call), medians of 10 in launch order (us): {meds}; first kernel at "
                     f"{local(t0 + ks[0][0] / 1e9)}, last at {local(t0 + ks[-1][0] / 1e9)}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
