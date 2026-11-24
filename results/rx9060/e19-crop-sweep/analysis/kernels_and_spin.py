#!/usr/bin/env python3
"""Kernels per timed range and the spin coverage at every N, from the rocprofv3 sweep of the E19 crop sweep timing window (no GPU).

  kernels_and_spin.py <rocprofv3 sweep folder> > kernels-and-spin.tsv

Reads each nNNN/calls.txt (rocprof_calls.py --spin) and nNNN/rpp-spin-with-copies.txt (rpp_spin_with_copies.py). For every N and row
(fkl_batch, rpp): the kernels in whole_call (101 calls) and in kernel_only (101 calls and 100 spins), the expected counts (fkl_batch 1 per
call, rpp N + 3 per call), and for kernel_only the iterations covered by the spin, the smallest and the median margin. rpp's coverage is the
one with its N copies counted (rpp_spin_with_copies.py). fkl_loop is left out (Oscar, 2026-11-20).
"""
import os
import re
import sys

root = sys.argv[1]
print("crops\trow\twhole_call_kernels\twhole_call_expected\tkernel_only_kernels\tkernel_only_expected\tcovered\tmargin_min_us\tmargin_median_us")
for d in sorted(os.listdir(root)):
    n = int(d[1:])
    text = open(os.path.join(root, d, "calls.txt")).read()
    spin = open(os.path.join(root, d, "rpp-spin-with-copies.txt")).read()
    for row, per_call in (("fkl_batch", 1), ("rpp", n + 3)):
        w = int(re.search(rf"  whole_call:{row}: 1 calls; kernels per call \[(\d+)\]", text).group(1))
        k = int(re.search(rf"  kernel_only:{row}: 1 calls; kernels per call \[(\d+)\]", text).group(1))
        if row == "rpp":
            m = re.search(r"(\d+) iterations, kernels per iteration \[(\d+)\], covered (\d+) of (\d+); margin min ([\d.]+) us, median ([\d.]+) us", spin)
            covered, mmin, mmed = f"{m.group(3)} of {m.group(4)}", m.group(5), m.group(6)
        else:
            m = re.search(rf"  kernel_only:{row}: (\d+) iterations, .*?covered (\d+) of (\d+); margin min ([\d.]+) us, median ([\d.]+) us", text)
            covered, mmin, mmed = f"{m.group(2)} of {m.group(3)}", m.group(4), m.group(5)
        print(f"{n}\t{row}\t{w}\t{101 * per_call}\t{k}\t{101 * per_call + 100}\t{covered}\t{mmin}\t{mmed}")
