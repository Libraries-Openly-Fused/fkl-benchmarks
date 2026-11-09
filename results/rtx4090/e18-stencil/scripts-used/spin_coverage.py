#!/usr/bin/env python3
"""From the nsys trace of a complete E18 run (exported to SQLite): does the spin cover the host's enqueue of every timed call?

  spin_coverage.py <e18-stencil.sqlite> [--out report.txt]

For every NVTX range kernel_only:<version> (fklb::timePreparedLaunch behind gpuSpin: one warm-up call without a spin, then 100 iterations of spin + call):
  - the calls: the kernels launched after each gpuSpin, up to the next one (the host's launch API calls are joined to the GPU kernels by correlation id)
  - covered: the host's launch call of the call's LAST kernel returned before the spin ended on the GPU (so the GPU never waited for the host
    within the timed interval); the margin is spin end - that return
  - the gaps on the GPU: from the end of the spin to the start of the first kernel, and between consecutive kernels of a call
For every NVTX range whole_call:<version>: the number of kernels per call (all kernels of the range / 101 calls).
In the whole trace: every API call whose name holds 'Tex', 'Array' or 'Surf' (texture, array or surface objects).
"""
import argparse
import collections
import sqlite3
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("db"); ap.add_argument("--out", default=None)
args = ap.parse_args()
c = sqlite3.connect(args.db).cursor()
names = dict(c.execute("select id, value from StringIds"))
ranges = [(t, s, e) for t, s, e in c.execute("select text, start, end from NVTX_EVENTS where text is not null order by start")]
api = {}   # correlation id -> (api name, start, end)
for corr, nid, s, e in c.execute("select correlationId, nameId, start, end from CUPTI_ACTIVITY_KIND_RUNTIME"):
    api[corr] = (names.get(nid, "?"), s, e)
kernels = []   # (gpu start, gpu end, name, correlation id)
for s, e, nid, corr in c.execute("select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start"):
    kernels.append((s, e, names.get(nid, "?"), corr))
lines = []


def launched_in(rs, re_):
    """the kernels whose host launch call started inside [rs, re_], in GPU order"""
    return [k for k in kernels if k[3] in api and rs <= api[k[3]][1] <= re_]


for text, rs, re_ in ranges:
    if text.startswith("whole_call:"):
        ks = launched_in(rs, re_)
        lines.append(f"{text}: {len(ks)} kernels in the range; per call (101 calls) {len(ks) / 101:.2f}; "
                     f"names {sorted(collections.Counter(k[2] for k in ks).items())}")
    if not text.startswith("kernel_only:"):
        continue
    ks = launched_in(rs, re_)
    calls, cur, spinEnd = [], None, None
    for k in ks:
        if k[2] == "gpuSpin":
            if cur is not None:
                calls.append(cur)
            cur = {"spinEnd": k[1], "kernels": []}
        elif cur is not None:
            cur["kernels"].append(k)
    if cur is not None:
        calls.append(cur)
    covered, margins, firstGaps, innerGaps, perCall = 0, [], [], [], []
    for call in calls:
        kk = call["kernels"]
        perCall.append(len(kk))
        if not kk:
            continue
        lastReturn = max(api[k[3]][2] for k in kk)
        margins.append(call["spinEnd"] - lastReturn)
        if lastReturn <= call["spinEnd"]:
            covered += 1
        firstGaps.append(kk[0][0] - call["spinEnd"])
        innerGaps += [kk[j + 1][0] - kk[j][1] for j in range(len(kk) - 1)]
    us = lambda ns: ns / 1000.0
    lines.append(f"{text}: {len(calls)} timed calls behind a spin; kernels per call {sorted(set(perCall))}; covered {covered} of {len(calls)}; "
                 f"margin (spin end - return of the last launch) min {us(min(margins)):.1f} us, median {us(statistics.median(margins)):.1f} us; "
                 f"GPU gap spin end -> first kernel max {us(max(firstGaps)):.2f} us, median {us(statistics.median(firstGaps)):.2f} us; "
                 + (f"between the kernels of a call max {us(max(innerGaps)):.2f} us, median {us(statistics.median(innerGaps)):.2f} us"
                    if innerGaps else "one kernel per call"))
tex = collections.Counter(n for n, s, e in api.values() if any(w in n for w in ("Tex", "Array", "Surf")))
lines.append(f"texture, array or surface API calls in the whole trace: {sum(tex.values())} {dict(tex)}")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
