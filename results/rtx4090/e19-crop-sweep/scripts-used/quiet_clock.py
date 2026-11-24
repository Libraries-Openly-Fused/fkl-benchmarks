#!/usr/bin/env python3
"""The SM clock while each row was timed in the runs of an E19 crop-sweep window without a trace (the quiet sweeps), estimated (nothing is run).

  quiet_clock.py <window folder> [--sweeps quiet-run1,quiet-run2,quiet-run3,warmup] [--out report.txt] [--csv rows.csv]

A run has no timestamps of its rows, so each row's timed interval is placed backwards from the run's end:
  - the timing is the last part of a run, in the order fkl_loop, fkl_batch, opencv_cuda, npp, each whole_call then kernel_only;
  - each interval's length comes from its own CSV: whole_call = 101 calls x its mean (the warm-up call and the 100 timed calls); kernel_only =
    its mean + 100 x (the spin of spin_us + its mean) (fklb::timePreparedLaunch: a warm-up call, then 100 spins and calls);
  - the run's end is known to the second (load.txt, "end"): the process exited within [end, end + 1 s). Between the end of the timing and the exit
    (cudaFree, the context's teardown) up to 0.5 s is allowed. So the timing ended within [end - 0.5 s, end + 1 s), and each row's interval lies
    within [end - 0.5 s - (that row and the rows after it), end + 1 s - (the rows after it)].
nvidia-smi read the clock every 100 ms; a reading at t states the clock at some moment of (t - 0.5 s, t]. Per run and row (both intervals
together), the readings that can fall in that span are counted by SM clock: "2685 only" when every one of them is 2,685 MHz. Per row: the runs
that are not 2685 only. fkl_loop is listed for the clock only (it is in no table of the report).
"""
import argparse, csv, datetime, glob, os, re
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--sweeps", default="quiet-run1,quiet-run2,quiet-run3,warmup")
ap.add_argument("--out", default=None); ap.add_argument("--csv", default=None); args = ap.parse_args()
TZ = datetime.timezone(datetime.timedelta(hours=1))
ORDER = ["fkl_loop", "fkl_batch", "opencv_cuda", "npp"]
readings = []
for r in csv.reader(open(os.path.join(args.folder, "gpu_clocks.csv"))):
    if len(r) < 2: continue
    readings.append((datetime.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=TZ).timestamp(), int(r[1].strip().split()[0])))
lines, out, notonly = [], [], {}
for sweep in args.sweeps.split(","):
    for d in sorted(glob.glob(os.path.join(args.folder, sweep, "N*"))):
        n = int(re.search(r"N(\d+)$", d).group(1))
        end = datetime.datetime.fromisoformat(re.search(r"end (\S+)", open(os.path.join(d, "load.txt")).read()).group(1)).timestamp()
        rows = {(r["version"], r["interval"]): r for r in csv.DictReader(l for l in open(os.path.join(d, "e19_batched_box.csv")) if not l.startswith("#"))}
        dur = {}
        for v in ORDER:
            w, k = rows[(v, "whole_call")], rows[(v, "kernel_only")]
            dur[v] = 101 * float(w["mean_ms"]) / 1000 + (float(k["mean_ms"]) + 100 * (float(k["spin_us"]) / 1000 + float(k["mean_ms"]))) / 1000
        parts = []
        for i, v in enumerate(ORDER):
            after = sum(dur[x] for x in ORDER[i + 1:])
            a, b = end - 0.5 - after - dur[v], end + 1.0 - after
            rs = [c for t, c in readings if t > a and t - 0.5 < b]
            counts = {c: rs.count(c) for c in sorted(set(rs))}
            only = bool(rs) and set(rs) == {2685}
            if not only: notonly.setdefault(v, []).append(f"{sweep}/N{n:03d}")
            parts.append(f"{v} {'2685 only' if only else counts}")
            out.append({"sweep": sweep, "crops": n, "row": v, "span_s": f"{b - a:.2f}", "readings": len(rs), "only_2685": "yes" if only else "no",
                        "clocks": " ".join(f"{c}:{k}" for c, k in counts.items())})
        lines.append(f"{sweep}/N{n:03d} (timing {sum(dur.values()):.2f} s): " + "; ".join(parts))
lines.append("runs whose span holds a reading other than 2,685 MHz, per row:")
for v in ORDER:
    lines.append(f"  {v}: {len(notonly.get(v, []))} runs {notonly.get(v, [])}")
t = "\n".join(lines); print(t)
if args.out: open(args.out, "w").write(t + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
