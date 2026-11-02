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
"""Load logger: what else was running on the machine while an experiment was measured.

  python scripts/load_logger.py --log <dir>/load.log [--interval 20] [--bench-cores 0-3,16-19] [--ccd0 0-7,16-23]      (runs until it is killed; scripts/run_e0.sh and the window scripts start and stop it)

Every INTERVAL seconds (and once at the start) it appends one block to the log:
  <time>  load <1 min> <5 min> <15 min>  runnable <n>  busy% of the benchmark cores | of the other cores of their CCD | of the other CCD    (from /proc/stat, since the previous sample)
  and one line per process that used more than 3 % of a core since the previous sample: pid, % of a core, the cores of its affinity mask, the command line.
The benchmark itself is one of the processes listed (it is the one on the benchmark cores). It uses a few milliseconds of one core per sample.
"""
import argparse
import datetime as dt
import os
import platform
import signal
import sys
import time

CLK = os.sysconf("SC_CLK_TCK")


def os_kernel_lines():
    """The two comment lines of every log of a run (as scripts/os_kernel_lines.sh and common/system_info.h write them), read at run time."""
    try:
        name = platform.freedesktop_os_release()["PRETTY_NAME"]
    except (OSError, KeyError):
        name = "unavailable"
    u = os.uname()
    return f"# os: {name}\n# kernel: {u.sysname} {u.release}\n"


def parse_cores(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out |= set(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def read_cpu():
    busy, total = {}, {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3].isdigit():
                p = line.split()
                n = int(p[0][3:])
                v = list(map(int, p[1:9]))
                idle = v[3] + v[4]
                total[n] = sum(v)
                busy[n] = sum(v) - idle
    return busy, total


def read_procs():
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            with open(f"/proc/{pid}/stat") as f:
                s = f.read()
            rp = s.rindex(")")
            fields = s[rp + 2:].split()
            out[int(pid)] = (int(fields[11]) + int(fields[12]), s[s.index("(") + 1:rp])
        except (OSError, ValueError, IndexError):
            continue
    return out


def affinity(pid):
    try:
        cpus = sorted(os.sched_getaffinity(pid))
    except OSError:
        return "?"
    if len(cpus) == os.cpu_count():
        return "any"
    return ",".join(map(str, cpus))


def cmdline(pid, comm):
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            c = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        return (c or comm)[:100]
    except OSError:
        return comm


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=float, default=20.0)
    ap.add_argument("--bench-cores", default="0-3,16-19")
    ap.add_argument("--ccd0", default="0-7,16-23")
    args = ap.parse_args()
    bench, ccd0 = parse_cores(args.bench_cores), parse_cores(args.ccd0)
    stop = []
    signal.signal(signal.SIGTERM, lambda *_: stop.append(1))
    signal.signal(signal.SIGINT, lambda *_: stop.append(1))
    prev_b, prev_t = read_cpu()
    prev_p = read_procs()
    me = os.getpid()
    with open(args.log, "a", buffering=1) as log:
        log.write(f"# load logger started {dt.datetime.now().astimezone().isoformat(timespec='seconds')}, interval {args.interval} s, benchmark cores {args.bench_cores}, first CCD {args.ccd0}\n")
        log.write(os_kernel_lines())
        first = True
        while not stop:
            time.sleep(1.0 if first else args.interval)
            first = False
            b, t = read_cpu()
            procs = read_procs()
            groups = {"bench": bench, "ccd0_other": ccd0 - bench, "other_ccd": set(b) - ccd0}
            pct = {}
            for name, cores in groups.items():
                db = sum(b[c] - prev_b[c] for c in cores if c in b and c in prev_b)
                dtot = sum(t[c] - prev_t[c] for c in cores if c in t and c in prev_t)
                pct[name] = 100.0 * db / dtot if dtot else 0.0
            with open("/proc/loadavg") as f:
                la = f.read().split()
            interval_ticks = max(1, (sum(t.values()) - sum(prev_t.values())) // max(1, len(t)))
            lines = []
            for pid, (ticks, comm) in procs.items():
                if pid == me or pid not in prev_p:
                    continue
                d = ticks - prev_p[pid][0]
                share = 100.0 * d / interval_ticks
                if share >= 3.0:
                    lines.append((share, pid, comm))
            lines.sort(reverse=True)
            stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            log.write(f"{stamp} load {la[0]} {la[1]} {la[2]} runnable {la[3].split('/')[0]} busy% bench {pct['bench']:.0f} | ccd0-other {pct['ccd0_other']:.0f} | other-ccd {pct['other_ccd']:.0f}\n")
            for share, pid, comm in lines[:10]:
                log.write(f"    pid {pid} {share:.0f}% cores {affinity(pid)} {cmdline(pid, comm)}\n")
            prev_b, prev_t, prev_p = b, t, procs
        log.write(f"# load logger stopped {dt.datetime.now().astimezone().isoformat(timespec='seconds')}\n")


if __name__ == "__main__":
    main()
