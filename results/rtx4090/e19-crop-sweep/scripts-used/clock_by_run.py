#!/usr/bin/env python3
"""The SM clock of every run of an E19 crop-sweep window, from gpu_clocks.csv and each run's load.txt (nothing is run).

  clock_by_run.py <window folder> [--sweeps warmup,quiet-run1,quiet-run2,quiet-run3,nsys] [--out report.txt] [--csv clocks.csv]

nvidia-smi read the GPU every 100 ms (gpu_clocks.csv: timestamp, clocks.sm, ...). A run's readings are those from its start to its end (load.txt,
to the second; the first reading counted is the first at or after the start second, the last the last before the end second + 1). nvidia-smi
repeats a reading until the driver updates it (about every 0.5 s), so a reading states the clock at some moment of the previous 0.5 s. Per run:
the number of readings, the distinct SM clocks with their counts, the clock of the first reading, and whether every reading is 2,685 MHz. Per
sweep: the runs that are not all at 2,685 MHz.
"""
import argparse, csv, datetime, glob, os, re
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--sweeps", default="warmup,quiet-run1,quiet-run2,quiet-run3,nsys")
ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None); args = ap.parse_args()
readings = []
for r in csv.reader(open(os.path.join(args.folder, "gpu_clocks.csv"))):
    if len(r) < 2: continue
    t = datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f")
    readings.append((t, int(r[1].strip().split()[0])))
lines, out = [], []
for sweep in args.sweeps.split(","):
    off = []
    for d in sorted(glob.glob(os.path.join(args.folder, sweep, "N*"))):
        n = int(re.search(r"N(\d+)$", d).group(1)); txt = open(os.path.join(d, "load.txt")).read()
        st = datetime.datetime.fromisoformat(re.search(r"start (\S+?)(;|$)", txt, re.M).group(1)).replace(tzinfo=None)
        en = datetime.datetime.fromisoformat(re.search(r"end (\S+)", txt).group(1)).replace(tzinfo=None) + datetime.timedelta(seconds=1)
        rs = [c for t, c in readings if st <= t < en]
        counts = {c: rs.count(c) for c in sorted(set(rs))}
        all2685 = bool(rs) and set(rs) == {2685}
        if not all2685: off.append(n)
        lines.append(f"{sweep}/N{n:03d}: {len(rs)} readings; SM MHz {counts}; first {rs[0] if rs else '-'}; all at 2,685: {'yes' if all2685 else 'NO'}")
        out.append({"sweep": sweep, "crops": n, "readings": len(rs), "first_mhz": rs[0] if rs else "", "all_2685": "yes" if all2685 else "no",
                    "clocks": " ".join(f"{c}:{k}" for c, k in counts.items())})
    lines.append(f"{sweep}: runs not all at 2,685 MHz: {off if off else 'none'}")
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
