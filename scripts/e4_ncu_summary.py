#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E4d: kernels launched and DRAM bytes moved by each implementation of the E4 pipeline, from Nsight Compute runs of the E4 program with E4_PROFILE=1 (one launch of every
kernel of the implementation, E4_ONLY=<version>, one batch or one frame).

  python scripts/e4_ncu_summary.py <dir with ncu_<point>_<version>.csv> --out e4_dram_bytes.csv

One row per (point, version): the number of kernels, the sum of dram__bytes.sum (read + write, bytes) and of gpu__time_duration.sum of those kernels (us; isolated cold launches, clocks locked,
caches flushed by ncu between kernels, so DRAM bytes are the traffic of each kernel on its own: what an implementation moves between its kernels through DRAM is counted, which is the point).
"""
import csv
import glob
import io
import os
import re
import sys


def read_ncu(path):
    text = open(path, errors="replace").read()
    m = re.search(r'^"?ID"?,', text, re.M)
    if not m:
        return []
    rows = list(csv.reader(io.StringIO(text[m.start():])))
    h = rows[0]
    return [dict(zip(h, r)) for r in rows[1:] if len(r) == len(h) and r[0].strip().isdigit()]


def num(s):
    return float(s.replace(",", ""))


def main():
    d = sys.argv[1]
    out = sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv else os.path.join(d, "e4_dram_bytes.csv")
    rows = []
    for path in sorted(glob.glob(os.path.join(d, "ncu_*.csv"))):
        m = re.match(r"ncu_(.+)_(opencv_percrop|opencv_batch|npp|fkl_fused_half_pixel|fkl_fused)\.csv$", os.path.basename(path))
        if not m:
            continue
        ks = read_ncu(path)
        if not ks:
            continue
        rows.append({"point": m.group(1), "version": m.group(2), "kernels": len(ks), "dram_bytes": f"{sum(num(k['dram__bytes.sum']) for k in ks):.0f}",
                     "dram_MB": f"{sum(num(k['dram__bytes.sum']) for k in ks) / 1e6:.2f}", "kernel_time_us": f"{sum(num(k['gpu__time_duration.sum']) for k in ks) / 1000:.1f}"})
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["point", "version", "kernels", "dram_bytes", "dram_MB", "kernel_time_us"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
