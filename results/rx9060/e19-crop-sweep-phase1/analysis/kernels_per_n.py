#!/usr/bin/env python3
"""Kernels per call of every row at every N, from the profile runs of the E19 crop sweep checks window.

  kernels_per_n.py <window folder>

From each rocprofv3/nNNN/calls.txt (rocprof_calls.py's summary: kernels per call of each of the 3 profiled calls, and whether they are the same
in every call) and from the kernel trace (the grids of every dispatch of the run: one warm-up call and three profiled calls of every row, so
every count is divided by 4). The copies of rpp are __amd_rocclr_copyBufferRectAligned; their grids are counted per form.
"""
import collections
import csv
import os
import re
import sys

win = sys.argv[1]
print("crops\trow\tkernels_per_call\tsame_in_every_call\tcalls_alloc_copy_memset\tgrids_per_call (kernel grid x count)")
for d in sorted(os.listdir(os.path.join(win, "rocprofv3"))):
    n = int(d[1:])
    text = open(os.path.join(win, "rocprofv3", d, "calls.txt")).read()
    summary = text[text.index("summary per range name"):]
    grids = collections.defaultdict(collections.Counter)
    row_of = {"fk::linearFilterDPPKernel": "fkl_loop", "fk::morphologyDPPKernel": "fkl_batch"}
    with open(os.path.join(win, "rocprofv3", d, "trace", "e19_kernel_trace.csv"), newline="") as f:
        for k in csv.DictReader(f):
            name = re.sub(r"^void ", "", k["Kernel_Name"])
            short = re.match(r"[\w:]+", name).group(0)
            if short == "__amd_rocclr_copyBuffer":   # the uploads of the set-up, before the first call
                continue
            row = row_of.get(short, "rpp")
            grids[row][f'{short} {k["Grid_Size_X"]}x{k["Grid_Size_Y"]}x{k["Grid_Size_Z"]}'] += 1
    for row in ("fkl_loop", "fkl_batch", "rpp"):
        m = re.search(rf"  {row}: 3 calls; kernels per call \[([\d, ]+)\]; the same kernels in every call: (\w+)\n.*\n    inside the calls: (.*)\n", summary)
        per_call = m.group(1)
        inside = m.group(3)
        g = "; ".join(f"{key} x {c / 4:g}" for key, c in sorted(grids[row].items()))
        print(f"{n}\t{row}\t{per_call}\t{m.group(2)}\t{inside}\t{g}")
