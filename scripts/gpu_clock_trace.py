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
"""The graphics clock of a GPU while a benchmark runs, aligned with the lines that the benchmark prints.

A GPU that was at rest runs at a low clock until it has had work for a while: an RX 9060 stays at about 2.0 GHz for the first 0.7 s of continuous
work and is at 3.0 GHz after 1 s; a fused kernel that is bound by the core clock is 20 percent slower until then (see results/rx9060/README.md).
This script runs a command, stamps every line of its output with the time of arrival, and samples the clock at about 200 Hz.

  gpu_clock_trace.py run <outdir> -- <command> [args...]      writes <outdir>/lines.txt (epoch seconds, tab, line) and <outdir>/clk.txt (epoch seconds, MHz, busy %)
  gpu_clock_trace.py report <outdir> --regex <regex> [--first K] [--look-back S]
        for every output line that matches the regex, the clock in the window that ends at the line (--look-back seconds, default 0.52) and in its last
        0.05 s; with --interval the window is the interval from the line to the next matching line instead (benchmarks that print a line when a point starts)

Clock source: --clock-file (default: freq1_input of the hwmon of the first AMD GPU, /sys/class/drm/card*/device/hwmon/hwmon*/freq1_input, in Hz).
On an NVIDIA GPU give a file that holds the clock, or adapt sample() to `nvidia-smi --query-gpu=clocks.sm --format=csv,noheader,nounits`.
"""
import argparse
import glob
import os
import re
import statistics
import subprocess
import sys
import threading
import time


def find_clock_file():
    for card in sorted(glob.glob('/sys/class/drm/card*/device')):
        try:
            if open(card + '/vendor').read().strip() != '0x1002':
                continue
        except OSError:
            continue
        files = glob.glob(card + '/hwmon/hwmon*/freq1_input')
        if files:
            return files[0], card + '/gpu_busy_percent'
    return None, None


def run(args):
    clock_file, busy_file = (args.clock_file, None) if args.clock_file else find_clock_file()
    if clock_file is None:
        sys.exit('no GPU clock file found (use --clock-file)')
    os.makedirs(args.outdir, exist_ok=True)
    stop = threading.Event()
    samples = []

    def sample():
        while not stop.is_set():
            t = time.time()
            try:
                mhz = int(open(clock_file).read()) / 1e6
                busy = int(open(busy_file).read()) if busy_file else -1
            except (OSError, ValueError):
                mhz, busy = 0.0, -1
            samples.append((t, mhz, busy))
            time.sleep(0.005)

    thread = threading.Thread(target=sample)
    thread.start()
    lines = []
    p = subprocess.Popen(args.command, cwd=args.outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        lines.append((time.time(), line.rstrip('\n')))
    p.wait()
    stop.set()
    thread.join()
    with open(args.outdir + '/lines.txt', 'w') as f:
        for t, l in lines:
            f.write(f'{t:.4f}\t{l}\n')
    with open(args.outdir + '/clk.txt', 'w') as f:
        for t, c, b in samples:
            f.write(f'{t:.4f}\t{c:.0f}\t{b}\n')
    print(f'exit {p.returncode}, {len(lines)} lines, {len(samples)} clock samples, clock file {clock_file}')
    return p.returncode


def stats(samples, a, b):
    sel = [c for (t, c, _) in samples if a < t <= b]
    return (statistics.mean(sel), min(sel), max(sel)) if sel else (0.0, 0.0, 0.0)


def report(args):
    lines = [(float(l.split('\t')[0]), l.split('\t', 1)[1]) for l in open(args.outdir + '/lines.txt')]
    clk = [tuple(map(float, l.split('\t'))) for l in open(args.outdir + '/clk.txt')]
    hits = [(t, l) for (t, l) in lines if re.search(args.regex, l)]
    if args.first:
        hits = hits[:args.first + (1 if args.interval else 0)]
    print('line | clock MHz mean/min/max over the window | clock MHz mean/min/max over the last 0.05 s of the window')
    for i, (t, l) in enumerate(hits):
        if args.interval:
            if i + 1 >= len(hits):
                break
            a, b = t, hits[i + 1][0]
        else:
            a, b = t - args.look_back, t
        m, lo, hi = stats(clk, a, b)
        m2, lo2, hi2 = stats(clk, b - 0.05, b)
        print(f'{l.strip()[:70]:70s} | {m:5.0f} {lo:5.0f} {hi:5.0f} | {m2:5.0f} {lo2:5.0f} {hi2:5.0f}  (window {b - a:.2f} s)')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run')
    r.add_argument('outdir')
    r.add_argument('--clock-file')
    r.add_argument('command', nargs=argparse.REMAINDER)
    p = sub.add_parser('report')
    p.add_argument('outdir')
    p.add_argument('--regex', required=True)
    p.add_argument('--first', type=int, default=0)
    p.add_argument('--look-back', type=float, default=0.52)
    p.add_argument('--interval', action='store_true')
    args = ap.parse_args()
    if args.cmd == 'run':
        if args.command and args.command[0] == '--':
            args.command = args.command[1:]
        sys.exit(run(args))
    report(args)


if __name__ == '__main__':
    main()
