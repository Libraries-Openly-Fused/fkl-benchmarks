#!/usr/bin/env python3
"""The E19 crop sweep's timing window, summarised from its CSVs (nothing is run).

  sweep_report.py <window folder> [--old <summary.csv of results/rtx4090/e19-batched-box-fkl-batch>] [--out report.txt] [--csv summary.csv]

Reads <window folder>/<sweep>/N<NNN>/e19_batched_box.csv (mean_ms of 100 calls per row and interval) of the quiet sweeps quiet-run1..3, and of
the unused sweeps warmup and nsys for reference. fkl_loop is in the raw CSVs only: it is in no table and no ratio of this report.
  times    per N, row (fkl_batch, opencv_cuda, npp) and interval: the 3 quiet means in us, their median, min and max, and the spread
           (max - min) / median
  ratios   per N and interval: opencv_cuda / fkl_batch and npp / fkl_batch in each quiet sweep (from that sweep's means), and their median,
           min and max
  bound    per N and row: whole_call / kernel_only of the medians (within 3 % of 1: GPU-bound; above: the host's work shows in whole_call)
  repeat   --old: N = 30 against the earlier window for fkl_batch, opencv_cuda and npp in both intervals: old median, new median, the change in %,
           both spreads, whether the change exceeds the larger spread, whether the 3-run ranges overlap, the flag above 3 %, and the bound (for a
           row whose whole_call is not within 3 % of its kernel_only, also the flag above 5 % of a host-bound number)
  unused   the warmup and nsys sweeps' means, and their change against the quiet median (not used)
"""
import argparse, csv, glob, os, re, statistics
ap = argparse.ArgumentParser(); ap.add_argument("folder"); ap.add_argument("--old", default=None); ap.add_argument("--out", default=None)
ap.add_argument("--csv", default=None); args = ap.parse_args()
ROWS, INTERVALS, QUIET = ["fkl_batch", "opencv_cuda", "npp"], ["whole_call", "kernel_only"], ["quiet-run1", "quiet-run2", "quiet-run3"]

def read(sweep):
    out = {}
    for d in sorted(glob.glob(os.path.join(args.folder, sweep, "N*"))):
        p = os.path.join(d, "e19_batched_box.csv")
        if not os.path.exists(p): continue
        n = int(re.search(r"N(\d+)$", d).group(1))
        for r in csv.DictReader(l for l in open(p) if not l.startswith("#")):
            if int(r["crops"]) != n: raise SystemExit(f"{p}: crops {r['crops']} is not {n}")
            out[(n, r["version"], r["interval"])] = float(r["mean_ms"]) * 1000.0
    return out

data = {s: read(s) for s in QUIET}
Ns = sorted({k[0] for k in data["quiet-run1"]})
for s in QUIET:
    if sorted({k[0] for k in data[s]}) != Ns: raise SystemExit(f"{s} does not have the same N as quiet-run1")
lines, out, med = [], [], {}
lines.append(f"quiet sweeps: {', '.join(QUIET)}; N = {', '.join(map(str, Ns))}; every value is the mean of 100 calls of one frame (N crops), in us")
for interval in INTERVALS:
    lines.append(f"{interval}: median (min-max) of the 3 quiet means, spread; then the per-sweep ratios over fkl_batch, median (min-max)")
    for n in Ns:
        parts = []
        for row in ROWS:
            v = [data[s][(n, row, interval)] for s in QUIET]
            m = statistics.median(v); sp = 100 * (max(v) - min(v)) / m
            med[(n, row, interval)] = (m, min(v), max(v), sp)
            parts.append(f"{row} {m:.2f} ({min(v):.2f}-{max(v):.2f}, {sp:.2f} %)")
            out.append({"kind": "time_us", "crops": n, "row": row, "interval": interval, "median": m, "min": min(v), "max": max(v), "spread_pct": sp,
                        **{s: x for s, x in zip(QUIET, v)}})
        for num in ("opencv_cuda", "npp"):
            r = [data[s][(n, num, interval)] / data[s][(n, "fkl_batch", interval)] for s in QUIET]
            m = statistics.median(r)
            parts.append(f"{num}/fkl_batch {m:.3f} ({min(r):.3f}-{max(r):.3f})")
            out.append({"kind": "ratio", "crops": n, "row": f"{num}/fkl_batch", "interval": interval, "median": m, "min": min(r), "max": max(r),
                        "spread_pct": "", **{s: x for s, x in zip(QUIET, r)}})
        lines.append(f"  N = {n:3d}: " + "; ".join(parts))
lines.append("whole_call / kernel_only of the medians (within 3 % of 1 = GPU-bound):")
for n in Ns:
    lines.append(f"  N = {n:3d}: " + "; ".join(f"{row} {med[(n, row, 'whole_call')][0] / med[(n, row, 'kernel_only')][0]:.4f}" for row in ROWS))
if args.old:
    old = {(r["row"], r["interval"]): r for r in csv.DictReader(open(args.old)) if r["kind"] == "time_us"}
    lines.append(f"repeat, N = 30 against {args.old}: old median / new median / change / spread old, new / exceeds the larger spread? / ranges overlap? / "
                 f"above 3 %? / bound")
    for interval in INTERVALS:
        for row in ROWS:
            o = old[(row, interval)]
            om, omin, omax, osp = float(o["median"]), float(o["min"]), float(o["max"]), float(o["spread_pct"])
            nm, nmin, nmax, nsp = med[(30, row, interval)]
            ch = 100 * (nm - om) / om
            ratio = med[(30, row, "whole_call")][0] / med[(30, row, "kernel_only")][0]
            bound = "GPU-bound" if abs(ratio - 1) < 0.03 else f"host share in whole_call (whole_call / kernel_only {ratio:.3f})"
            lines.append(f"  {row} {interval}: {om:.2f} / {nm:.2f} us / {ch:+.2f} % / {osp:.2f} %, {nsp:.2f} % / "
                         f"{'EXCEEDS' if abs(ch) > max(osp, nsp) else 'within the spread'} / {'overlap' if nmin <= omax and omin <= nmax else 'NO overlap'} / "
                         f"{'ABOVE 3 %' if abs(ch) > 3 else 'below 3 %'} / {bound}"
                         + ("" if abs(ratio - 1) < 0.03 else f" ({'ABOVE' if abs(ch) > 5 else 'below'} 5 %, the flag of a host-bound number)"))
            out.append({"kind": "repeat_change_pct", "crops": 30, "row": row, "interval": interval, "median": nm, "min": nmin, "max": nmax,
                        "spread_pct": nsp, "old_median": om, "old_spread_pct": osp, "change_pct": ch})
for extra in ("warmup", "nsys"):
    d = read(extra)
    if not d: continue
    lines.append(f"{extra} (not used): the change of its mean against the quiet median, in %, per N:")
    for n in Ns:
        lines.append(f"  N = {n:3d}: " + "; ".join(f"{row} {interval} {d[(n, row, interval)]:.2f} ({100 * (d[(n, row, interval)] / med[(n, row, interval)][0] - 1):+.1f} %)"
                                               for interval in INTERVALS for row in ROWS if (n, row, interval) in d))
text = "\n".join(lines); print(text)
if args.out: open(args.out, "w").write(text + "\n")
if args.csv:
    with open(args.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["kind", "crops", "row", "interval", "median", "min", "max", "spread_pct"] + QUIET + ["old_median", "old_spread_pct", "change_pct"])
        w.writeheader(); w.writerows(out)
