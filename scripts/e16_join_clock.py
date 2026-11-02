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
"""E16: put the SM clock of every row into the CSV.

  python scripts/e16_join_clock.py results/rtx4090/e16-attention

The window folder holds gpu_clocks.csv (nvidia-smi, one sample about every 0.1 s: timestamp, SM clock, memory clock, pstate, utilisation, power, temperature) and run1, run2, ... each with e16_attention.csv and
e16_baselines.csv. Every row of those files has t_start_ms and t_end_ms, the wall clock of its timed blocks (ms since the epoch). For each row the samples inside [t_start_ms, t_end_ms] are taken (the nearest sample
when there is none) and the row gets: sm_mhz_median, sm_mhz_min, sm_mhz_samples, power_w_median, util_pct_median. The result is written next to the original as e16_attention_clock.csv and e16_baselines_clock.csv
(the originals are not changed); scripts/e16_summary.py and scripts/plot_e16.py read the _clock files when they exist. A row measured at a low clock is thus visible in the CSV, not hidden in a median.
"""
import bisect
import csv
import datetime
import glob
import os
import statistics as stats
import sys

NEW = ["sm_mhz_median", "sm_mhz_min", "sm_mhz_samples", "power_w_median", "util_pct_median"]


def number(text):
    return float(text.strip().split()[0])


def samples(path):
    """[(epoch ms, sm MHz, power W, utilisation %)] sorted by time, from the nvidia-smi sampler file."""
    out = []
    with open(path, newline="") as f:
        for r in csv.reader(f):
            if len(r) < 7:
                continue
            try:
                t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp() * 1000.0   # local time, as the program's wall clock
                out.append((t, number(r[1]), number(r[5]), number(r[4])))
            except (ValueError, IndexError):
                continue
    out.sort()
    return out


def join_file(path, smp, times):
    with open(path, newline="") as f:
        raw = f.readlines()
    comments = [l for l in raw if l.startswith("#")]
    rows = list(csv.DictReader([l for l in raw if not l.startswith("#")]))
    if not rows:
        return 0
    for r in rows:
        for c in NEW:
            r[c] = ""
        try:
            t0, t1 = float(r["t_start_ms"]), float(r["t_end_ms"])
        except (KeyError, ValueError):
            continue
        if t1 <= 0:
            continue
        lo, hi = bisect.bisect_left(times, t0), bisect.bisect_right(times, t1)
        sel = smp[lo:hi]
        if not sel and smp:
            sel = [smp[min(max(lo, 0), len(smp) - 1)]]
        if sel:
            r["sm_mhz_median"] = f"{stats.median(x[1] for x in sel):.0f}"
            r["sm_mhz_min"] = f"{min(x[1] for x in sel):.0f}"
            r["sm_mhz_samples"] = str(len(sel))
            r["power_w_median"] = f"{stats.median(x[2] for x in sel):.1f}"
            r["util_pct_median"] = f"{stats.median(x[3] for x in sel):.0f}"
    out = path[:-4] + "_clock.csv"
    with open(out, "w", newline="") as f:
        for l in comments:
            f.write(l)
        f.write("# sm_mhz_*, power_w_median, util_pct_median: the samples of the 100 ms nvidia-smi sampler (gpu_clocks.csv of the window) inside [t_start_ms, t_end_ms] of the row, joined by scripts/e16_join_clock.py\n")
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def main():
    folder = sys.argv[1]
    smp = samples(os.path.join(folder, "gpu_clocks.csv"))
    times = [x[0] for x in smp]
    print(f"{len(smp)} clock samples from {os.path.join(folder, 'gpu_clocks.csv')}")
    for run in sorted(glob.glob(os.path.join(folder, "run*"))):
        for name in ("e16_attention.csv", "e16_baselines.csv"):
            path = os.path.join(run, name)
            if os.path.exists(path):
                print(f"{path}: {join_file(path, smp, times)} rows -> {path[:-4]}_clock.csv")


if __name__ == "__main__":
    main()
