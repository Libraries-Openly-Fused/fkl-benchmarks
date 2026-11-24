#!/usr/bin/env python3
"""Per run of the window e18-retime-b10fc66: the wait before it, the start of its timing, the SM clock then, and its first row's times
(CPU only: reads the window's files; nothing is run).

  first_row_clock.py <window folder> [--out first-row-clock.txt]

- wait: wait_idle's note in the run's load.txt ("idle after N s").
- timing start: the creation time of the run's timing CSV (<program>.csv), which the program creates when its timing begins (statx birth
  time, to the millisecond; it exists only on this machine's file system, so it is written out here).
- clock: the SM clock readings of gpu_clocks.csv (nvidia-smi every 100 ms; a reading states the clock of some moment of the previous
  0.5 s, when the driver updates it) from the timing start to 1 s after it, as MHz:count in time order.
- first row: the first row of the timing CSV (the first row the program times), its whole_call and kernel_only mean, in us.
"""
import argparse, csv, datetime, os, re, subprocess

ap = argparse.ArgumentParser(); ap.add_argument("window"); ap.add_argument("--out")
args = ap.parse_args()
rd = []
for r in csv.reader(open(os.path.join(args.window, "gpu_clocks.csv"))):
    try:
        rd.append((datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f"), int(r[1].split()[0])))
    except (ValueError, IndexError):
        continue
lines = [__doc__.strip().splitlines()[0], ""]
for prog in ("e18_box", "e18_box_split", "e18_stencil"):
    lines.append(f"{prog}:")
    for run in ("warmup-A", "warmup-B", "quiet-run1-A", "quiet-run1-B", "quiet-run2-A", "quiet-run2-B", "quiet-run3-A", "quiet-run3-B"):
        d = os.path.join(args.window, prog, run)
        f = os.path.join(d, prog + ".csv")
        born_s = subprocess.run(["stat", "-c", "%w", f], capture_output=True, text=True).stdout.strip()
        born = datetime.datetime.strptime(born_s[:26], "%Y-%m-%d %H:%M:%S.%f")
        wait = re.search(r"idle after (\d+) s", open(os.path.join(d, "load.txt")).read())
        seq = []
        for t, c in rd:
            if born <= t < born + datetime.timedelta(seconds=1):
                if seq and seq[-1][0] == c:
                    seq[-1][1] += 1
                else:
                    seq.append([c, 1])
        rows = [r for r in csv.reader(l for l in open(f) if not l.startswith("#"))]
        h = rows[0]; first = rows[1][h.index("version")]
        vals = {r[h.index("interval")]: float(r[h.index("mean_ms")]) * 1e3 for r in rows[1:] if r[h.index("version")] == first}
        lines.append(f"  {run:13s} wait {wait.group(1) if wait else '-':>3s} s; timing start {born.strftime('%H:%M:%S.%f')[:-3]}; clock in the next 1 s: "
                     f"{', '.join(f'{c}:{n}' for c, n in seq):24s}; {first}: whole_call {vals.get('whole_call', float('nan')):.2f} us, "
                     f"kernel_only {vals.get('kernel_only', float('nan')):.2f} us")
text = "\n".join(lines)
print(text)
if args.out:
    open(args.out, "w").write(text + "\n")
