#!/usr/bin/env python3
"""The "between" rows of the E19 crop sweep: fkl_batch's tensor against rpp's at every N, from the E19_DUMP files of the check runs.

  between.py <dump dir> > between.csv

For every N (the folders nNNN): the values, the number of values that differ (a != b, bit for bit as floats), the largest absolute difference
in output units, and the same in grey levels (x 57.375: out = v / 57.375 - 2, so one grey level is 1 / 57.375 in output units). Also the same
for fkl_loop against fkl_batch. A non-finite value in any tensor is counted and stops the comparison of that pair from being reported as ok.
"""
import array
import math
import os
import sys

dump = sys.argv[1]
print("crops,row,against,values,values_differing,max_abs_diff,max_abs_diff_grey_levels,non_finite")
for d in sorted(os.listdir(dump)):
    n = int(d[1:])
    t = {}
    for name in ("rpp", "fkl_batch", "fkl_loop"):
        a = array.array("f")
        with open(os.path.join(dump, d, f"{name}_tensor.f32"), "rb") as f:
            a.frombytes(f.read())
        assert len(a) == n * 65536, (d, name, len(a))
        t[name] = a
    for row, against in (("fkl_batch", "rpp"), ("fkl_loop", "fkl_batch")):
        a, b = t[row], t[against]
        differ = nonfinite = 0
        most = 0.0
        for x, y in zip(a, b):
            if not (math.isfinite(x) and math.isfinite(y)):
                nonfinite += 1
                continue
            if x != y:
                differ += 1
                most = max(most, abs(x - y))
        print(f"{n},{row},{against},{len(a)},{differ},{most:.9g},{most * 57.375:.6g},{nonfinite}")
