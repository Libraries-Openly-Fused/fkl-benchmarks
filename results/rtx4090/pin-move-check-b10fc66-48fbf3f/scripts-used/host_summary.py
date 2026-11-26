#!/usr/bin/env python3
"""C-PM-CUDA: one table of the host-code comparisons (host_runs.sh's reports and interval CSVs; CPU only, reads files).

  host_summary.py <out/host> [--out summary.txt]

Per comparison (main-vs-head, measured-vs-main) and program: the functions of the two builds (paired, identical, differing only in __LINE__ or
the build path, differing otherwise, unpaired), and the timed intervals by kind (event: between cudaEventRecord calls, the GPU times; clock:
between steady_clock::now calls, host times), counted by the result of the interval and of the functions it calls.
"""
import argparse
import collections
import csv
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("dir"); ap.add_argument("--out")
a = ap.parse_args()
csv.field_size_limit(sys.maxsize)
TOT = re.compile(r"TOTAL: functions A (\d+), B (\d+); pairs (\d+); identical (\d+), different (\d+); only in A (\d+), only in B (\d+)")
LINE = re.compile(r"of the different: (\d+) differ only .*?, (\d+) otherwise")
lines = []
for cmp_ in ("main-vs-head", "measured-vs-main"):
    d = os.path.join(a.dir, cmp_)
    lines.append(f"== {cmp_}")
    lines.append("  program | functions A / B | paired | identical | only __LINE__ / path | other | only A / only B | event intervals | clock intervals")
    for f in sorted(x for x in os.listdir(d) if x.endswith(".txt") and not x.endswith("-diffs.txt")):
        name = f[:-4]
        text = open(os.path.join(d, f)).read()
        t = TOT.search(text); l = LINE.search(text)
        if not t or not l:
            lines.append(f"  {name}: no totals"); continue
        na, nb, pairs, ident, diff, oa, ob = map(int, t.groups())
        lo, other = map(int, l.groups())
        kinds = {"event": collections.Counter(), "clock": collections.Counter()}
        ip = os.path.join(d, name + "-intervals.csv")
        if os.path.exists(ip):
            for r in csv.DictReader(open(ip)):
                key = r["interval"] if not r["callees"] else f"{r['interval']}/{r['callees']}"
                kinds[r["kind"]][key] += 1
        fmt = lambda c: ", ".join(f"{k} {n}" for k, n in sorted(c.items())) or "-"
        lines.append(f"  {name} | {na} / {nb} | {pairs} | {ident} | {lo} | {other} | {oa} / {ob} | {fmt(kinds['event'])} | {fmt(kinds['clock'])}")
    lines.append("")
lines.append("intervals: <interval>/<the functions it calls>; identical, line/path (only __LINE__ or the build path), registers (the same instructions "
             "under one consistent renaming of the registers), different; unpaired: the function is in one build only")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
