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
"""The graphics clock of the GPU while a program measured its variants.

  python scripts/clock_trace.py --clocks gpu_clocks.csv --stdout stdout_ts.txt --out clock_summary.csv [--boost 2550] [--match 'ms']

--clocks: `nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 500` (a sample every 0.5 s).
--stdout: the console output of the program, every line prefixed with the wall-clock time (HH:MM:SS) at which it was printed (the window scripts do that with `printf '%(%H:%M:%S)T'`).
A program prints a line when a variant has been measured; the window of a variant is therefore the time between the previous line and its own line (it contains the host-only work that precedes the
variant, its warm-up and its timed iterations). For every such line: the number of samples in the window, the minimum / median / maximum SM clock and the P-states in it, and the clock of the sample
just before the window (the state in which the variant starts). Windows shorter than the sampling interval have no sample of their own: they report the samples around them (before and after).
"""
import argparse
import csv
import datetime as dt
import re
import statistics as st


def parse_clocks(path):
    out = []
    for row in csv.reader(open(path)):
        if len(row) < 7:
            continue
        try:
            t = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
            mhz = int(row[1].strip().split()[0])
            out.append((t, mhz, row[3].strip(), int(row[4].strip().split()[0]), float(row[5].strip().split()[0])))
        except (ValueError, IndexError):
            continue
    return out


def parse_stdout(path):
    out = []
    rx = re.compile(r"^(\d\d:\d\d:\d\d) (.*)$")
    for line in open(path, errors="replace"):
        m = rx.match(line.rstrip("\n"))
        if m:
            out.append((m.group(1), m.group(2)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clocks", required=True)
    ap.add_argument("--stdout", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boost", type=int, default=2550, help="maximum SM clock of the GPU in MHz (RTX 4090: 2550)")
    ap.add_argument("--match", default="", help="only the lines of the program that contain this text end a window")
    args = ap.parse_args()
    clocks = parse_clocks(args.clocks)
    lines = parse_stdout(args.stdout)
    if not clocks or not lines:
        raise SystemExit("no clock samples or no time-stamped lines")
    day = clocks[0][0].date()
    stamps = []
    for hms, text in lines:
        t = dt.datetime.combine(day, dt.datetime.strptime(hms, "%H:%M:%S").time())
        if t < clocks[0][0] - dt.timedelta(hours=12):
            t += dt.timedelta(days=1)
        stamps.append((t, text))
    rows = []
    prev_t = None
    for t, text in stamps:
        if args.match and args.match not in text:
            continue
        start = prev_t if prev_t else t - dt.timedelta(seconds=2)
        window = [c for c in clocks if start <= c[0] <= t + dt.timedelta(seconds=0.999)]
        before = [c for c in clocks if c[0] < start]
        entering = before[-1] if before else None
        after = [c for c in clocks if c[0] > t]
        if not window:
            window = ([before[-1]] if before else []) + ([after[0]] if after else [])
        mhz = [c[1] for c in window]
        rows.append(dict(line_time=t.strftime("%H:%M:%S"), line=text[:90], window_s=int((t - start).total_seconds()), samples=len(window), clock_min=min(mhz) if mhz else "", clock_median=int(st.median(mhz)) if mhz else "",
                         clock_max=max(mhz) if mhz else "", pstates="/".join(sorted({c[2] for c in window})), utilisation_median=int(st.median(c[3] for c in window)) if window else "",
                         clock_before_window=entering[1] if entering else "", pstate_before_window=entering[2] if entering else ""))
        prev_t = t
    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    low = [r for r in rows if r["clock_median"] != "" and r["clock_median"] < 0.95 * args.boost]
    first_low = [r for r in rows if r["clock_min"] != "" and r["clock_min"] < 0.95 * args.boost]
    print(f"{len(rows)} windows; clocks: {len(clocks)} samples, {min(c[1] for c in clocks)} to {max(c[1] for c in clocks)} MHz overall; windows with a median clock below 95 % of {args.boost} MHz: {len(low)}; with any sample below: {len(first_low)}")
    for r in first_low[:12]:
        print(f"  {r['line_time']} {r['line'][:70]:70s} min {r['clock_min']} median {r['clock_median']} max {r['clock_max']} MHz, entering at {r['clock_before_window']} MHz ({r['pstate_before_window']}), {r['samples']} samples")


if __name__ == "__main__":
    main()
