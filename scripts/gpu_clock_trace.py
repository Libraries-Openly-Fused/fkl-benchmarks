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
This script runs a command, stamps every line of its output with the time of arrival, and samples the clock about every 5 ms plus the time of the read (176 Hz in results/rx9060/clock-ramp/clock-trace-0.5s-warmup/). The command runs with <outdir> as its working directory.

  gpu_clock_trace.py run [--clock-file F] <outdir> -- <command> [args...]      writes <outdir>/lines.txt (epoch seconds, tab, line), <outdir>/clk.txt (epoch seconds, MHz, busy %)
        and <outdir>/meta.txt (# os, # kernel, # date, # command, # clock file: read when the trace starts; the report prints them first). --clock-file goes BEFORE <outdir>.
        A read of the clock that fails is not stored (the count is printed at the end). The files are written also when the command fails to start or the run is interrupted.
  gpu_clock_trace.py report <outdir> --regex <regex> [--first K] [--look-back S]
        for every output line that matches the regex, the clock in the window that ends at the line (--look-back seconds, default 0.52) and in its last
        0.05 s; with --interval the window is the interval from the line to the next matching line instead, and it is labelled with the text of the line that OPENS it. That is right only
        for a line printed when the work of a point STARTS and nothing else happens in between: for a line printed at the END of the work (a timing line) the window holds the NEXT item, and
        for a line printed after a spin and before the timed runs ("Executing N" of the latency hiding benchmark) it holds the timed runs of N and the spin of the next point
        (see results/rx9060/fkl-self/latency_hiding/clock-per-point.txt). The last matching line gets no row. A window without a clock sample prints n/a, not 0.

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


def shell_line(command):
    try:
        out = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        return 'unavailable'
    return out[0] if out else 'unavailable'


def machine_header(command, clock_file):
    return ['# os: ' + shell_line('. /etc/os-release && echo "$PRETTY_NAME"'), '# kernel: ' + shell_line('uname -r -v'), '# date: ' + shell_line('date -u +%Y-%m-%dT%H:%M:%SZ'),
            '# command: ' + ' '.join(command), '# clock file: ' + str(clock_file)]


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
    try:  # the clock must be readable before the command is started
        int(open(clock_file).read())
    except (OSError, ValueError) as e:
        sys.exit(f'cannot read the clock {clock_file}: {e}')
    os.makedirs(args.outdir, exist_ok=True)
    header = machine_header(args.command, clock_file)  # read when the trace starts
    stop = threading.Event()
    samples = []
    failed = [0]

    def sample():
        while not stop.is_set():
            t = time.time()
            try:
                mhz = int(open(clock_file).read()) / 1e6
                busy = int(open(busy_file).read()) if busy_file else -1
                samples.append((t, mhz, busy))
            except (OSError, ValueError):
                failed[0] += 1
            time.sleep(0.005)

    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    lines = []
    p = None
    interrupted = False
    try:
        try:
            p = subprocess.Popen(args.command, cwd=args.outdir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
        except OSError as e:
            sys.exit(f'cannot run {args.command[0]!r}: {e}')
        for line in p.stdout:
            lines.append((time.time(), line.rstrip('\n')))
        p.wait()
    except KeyboardInterrupt:
        interrupted = True
    finally:
        stop.set()
        thread.join(timeout=5)
        if p is not None and p.poll() is None:  # interrupted: do not leave the command running
            p.terminate()
        with open(args.outdir + '/lines.txt', 'w') as f:
            for t, l in lines:
                f.write(f'{t:.4f}\t{l}\n')
        with open(args.outdir + '/clk.txt', 'w') as f:
            for t, c, b in samples:
                f.write(f'{t:.4f}\t{c:.0f}\t{b}\n')
        with open(args.outdir + '/meta.txt', 'w') as f:
            f.write('\n'.join(header) + '\n')
    if interrupted:
        print(f'interrupted: the trace up to now ({len(lines)} lines, {len(samples)} clock samples) is in {args.outdir}')
        return 130
    print(f'exit {p.returncode}, {len(lines)} lines, {len(samples)} clock samples ({failed[0]} failed reads), clock file {clock_file}')
    if not samples:
        print('no clock sample was stored: every read of the clock failed', file=sys.stderr)
        return p.returncode or 1
    return p.returncode


def stats(samples, a, b):
    sel = [c for (t, c, _) in samples if a < t <= b]
    return (statistics.mean(sel), min(sel), max(sel)) if sel else None


def three(s):
    return f'{"n/a":>5} {"n/a":>5} {"n/a":>5}' if s is None else f'{s[0]:5.0f} {s[1]:5.0f} {s[2]:5.0f}'


def report(args):
    lines = [(float(l.split('\t')[0]), l.split('\t', 1)[1]) for l in open(args.outdir + '/lines.txt')]
    clk = [tuple(map(float, l.split('\t'))) for l in open(args.outdir + '/clk.txt')]
    hits = [(t, l) for (t, l) in lines if re.search(args.regex, l)]
    if os.path.exists(args.outdir + '/meta.txt'):  # traces taken before meta.txt existed have none
        print(open(args.outdir + '/meta.txt').read().rstrip('\n'))
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
        print(f'{l.strip():70s} | {three(stats(clk, a, b))} | {three(stats(clk, b - 0.05, b))}  (window {b - a:.2f} s)')


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
        if args.command and args.command[0].startswith('--clock-file'):
            sys.exit('--clock-file must come before <outdir>: run [--clock-file F] <outdir> -- <command>')
        if args.command and args.command[0] == '--':
            args.command = args.command[1:]
        if not args.command:
            sys.exit('no command given: run [--clock-file F] <outdir> -- <command> [args...]')
        sys.exit(run(args))
    report(args)


if __name__ == '__main__':
    main()
