#!/usr/bin/env python3
"""The required check rows (ok = 0 or 1) of every N of the E19 crop sweep checks window, one line per N and row, and the count of failures.

  checks_summary.py <window folder> [<E19 phase-1 checks csv>]

With the E19 phase-1 checks CSV (results/rx9060/e19-batched-box-phase1/check/), N = 30 is compared with it row by row, without the column
"crops": every field equal or not.
"""
import csv
import os
import sys

win = sys.argv[1]


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(line for line in f if not line.startswith("#")))


fails = 0
print("crops,row,stage,against,values,values_differing_or_beyond,max_abs_diff,non_finite,ok")
for d in sorted(os.listdir(os.path.join(win, "check"))):
    for r in rows(os.path.join(win, "check", d, "e19_batched_box_checks_GPU_AMD.csv")):
        if r["ok"] == "-1":
            continue
        fails += r["ok"] != "1"
        against = r["against"].replace('"', "")
        print(f'{r["crops"]},{r["row"]},{r["stage"]},"{against}",{r["values"]},{r["values_differing_or_beyond"]},{r["max_abs_diff"]},{r["non_finite"]},{r["ok"]}')
print(f"# required rows failing: {fails}")
if len(sys.argv) > 2:
    old = rows(sys.argv[2])
    new = rows(os.path.join(win, "check", "n030", "e19_batched_box_checks_GPU_AMD.csv"))
    same = len(old) == len(new) and all({k: v for k, v in a.items()} == {k: v for k, v in b.items() if k != "crops"} for a, b in zip(old, new))
    print(f"# N = 30 against E19 phase 1 ({len(old)} rows, every field but crops): {'equal' if same else 'DIFFERENT'}")
