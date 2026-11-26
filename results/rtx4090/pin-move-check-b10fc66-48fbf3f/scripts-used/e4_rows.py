#!/usr/bin/env python3
"""C-PM-CUDA: E4's timed intervals by row, from host_compare435.py's interval CSVs (CPU only, reads files).

  e4_rows.py <intervals.csv>... [--out e4-rows.txt]

E4 times each row with fklb::timeIt<L> (common/cuda_timing.h), L the row's launch lambda in runCase<N>; the lambdas of runCase are numbered by
GCC in source order, the same at 836a98e (#1 to #4), 469aa6d (#1 to #8) and main 976e42b (#1 to #12, the build-once rows added after #8):
  #1 launchFklAsym      fkl_fused, fkl_fused_late, fkl_fused_after      #7  launchHandRoi       hand_roi
  #2 launchFklHalf      fkl_fused_half_pixel                            #8  launch (cvGS)       cvgs_fused
  #3 launch (OpenCV)    opencv_percrop                                  #9  launchOnce          cvgs_fused_once
  #4 launch (NPP)       npp                                             #10 launchFklOnce       fkl_fused_once, fkl_fused_once_after
  #5 launchHand         hand_batch                                      #11 launchFklHalfOnce   fkl_fused_half_pixel_once
  #6 launchHandPadded   hand_padded                                     #12 launchHandRoiOnce   hand_roi_once
For every lambda: over the values of N, the result of the event interval of timeIt<L> (the instructions between its two cudaEventRecord calls,
with L's body where GCC inlined it) and of the functions the interval calls; and the functions that are not identical, with how many N have them.
"""
import argparse
import collections
import csv
import re
import sys

ROWS = {1: "fkl_fused (+ _late, _after)", 2: "fkl_fused_half_pixel", 3: "opencv_percrop", 4: "npp", 5: "hand_batch", 6: "hand_padded",
        7: "hand_roi", 8: "cvgs_fused", 9: "cvgs_fused_once", 10: "fkl_fused_once (+ _after)", 11: "fkl_fused_half_pixel_once", 12: "hand_roi_once"}
ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+"); ap.add_argument("--out")
a = ap.parse_args()
csv.field_size_limit(sys.maxsize)


def short(k):
    k = re.sub(r"runCase<\d+>", "runCase<N>", k)
    for _ in range(6):
        k = re.sub(r"<[^<>]*>", "<>", k)
    return k[:150]


lines = []
for p in a.csvs:
    lines.append(f"== {p.split('/')[-1]}")
    by = collections.defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r["kind"] != "event":
            continue
        m = re.search(r"timeIt<\(anonymousnamespace\)::runCase<(\d+)>.*?\{lambda\(\)#(\d+)\}", r["function"])
        if not m:
            lines.append(f"  (not a runCase lambda) {r['interval']} {r['function'][:120]}"); continue
        by[int(m.group(2))].append((int(m.group(1)), r))
    for lam in sorted(by):
        v = by[lam]
        st = collections.Counter(r["interval"] + (f" / callees {r['callees']}" if r["callees"] else "") for _, r in v)
        lines.append(f"  #{lam} {ROWS.get(lam, '?')}: {len(v)} N: " + "; ".join(f"{k}: {n}" for k, n in sorted(st.items())))
        odd = sorted(n for n, r in v if r["interval"] in ("different", "registers"))
        if odd:
            lines.append(f"      interval not identical at N = {', '.join(map(str, odd))}")
        bad = collections.defaultdict(set)
        for n_, r in v:
            for part in filter(None, r["callees_not_identical"].split(" | ")):
                w, k = part.split(" ", 1)
                bad[f"{w} {short(k)}"].add(n_)
        for k, ns in sorted(bad.items(), key=lambda x: (-len(x[1]), x[0])):
            lines.append(f"      {len(ns)} N: {k}")
        ext = collections.Counter(r["external_calls"] for _, r in v if r["external_calls"] and r["external_calls"] != "same")
        for k, n in ext.items():
            lines.append(f"      external calls at {n} N: {k[:300]}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
