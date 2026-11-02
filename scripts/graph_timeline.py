#!/usr/bin/env python3
"""The kernel timeline of one launch of a graph structure, from an Nsight Systems report exported to SQLite.

    nsys profile --trace=cuda,nvtx --cuda-graph-trace=node -o REPORT  PROBE      (FKLB_GRAPH_STRUCTURES=child FKLB_GRAPH_CROPS=301 ... )
    nsys export --type sqlite --force-overwrite true REPORT.nsys-rep
    graph_timeline.py REPORT.sqlite --range 'child@301' --kernels-per-launch 301 [--csv OUT.csv --label child]

The probes (common/graph_structures.h) wrap the timed launches of a structure in an NVTX range "<structure>@<point>". The kernels that run inside that range are cut into launches of K = --kernels-per-launch
kernels in order of start time, and for each launch the script reports: the makespan (first start to last end), the kernel duration (median), the idle gap between a kernel and the next one in order of
start (median, mean, 90th percentile; for a serial chain that gap is what the structure costs per node), the largest number of kernels running at the same time, and the fraction of the makespan
with at least one kernel running. The numbers are those of the GPU timeline: nothing here comes from the documentation of CUDA.
"""
import argparse
import csv
import sqlite3
import statistics


def columns(db, table):
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]


def percentile(sorted_values, q):
    if not sorted_values:
        return float("nan")
    k = min(len(sorted_values) - 1, max(0, int(round(q * (len(sorted_values) - 1)))))
    return sorted_values[k]


def max_concurrency(intervals):
    events = []
    for s, e in intervals:
        events.append((s, 1))
        events.append((e, -1))
    events.sort(key=lambda t: (t[0], t[1]))
    cur = best = 0
    for _, d in events:
        cur += d
        best = max(best, cur)
    return best


def busy_ns(intervals):
    total = 0
    cur_s = cur_e = None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--range", required=True, help="text of the NVTX range, e.g. child@301")
    ap.add_argument("--kernels-per-launch", type=int, required=True)
    ap.add_argument("--csv", help="append one row to this CSV")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    db = sqlite3.connect(args.sqlite)
    ncols = columns(db, "NVTX_EVENTS")
    text_col = "text" if "text" in ncols else "textId"
    rng = db.execute(f"SELECT start, end FROM NVTX_EVENTS WHERE {text_col} = ? AND end IS NOT NULL ORDER BY start", (args.range,)).fetchall()
    if not rng:
        raise SystemExit(f"no NVTX range '{args.range}' in {args.sqlite}")
    start, end = rng[0]
    kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    kernels = db.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ? ORDER BY start", (start, end)).fetchall()
    K = args.kernels_per_launch
    launches = len(kernels) // K
    if launches == 0:
        raise SystemExit(f"{len(kernels)} kernels in the range, fewer than {K} per launch")
    leftover = len(kernels) - launches * K

    makespans, concurrency, busy_frac = [], [], []
    gaps = []
    durations = []
    for i in range(launches):
        chunk = kernels[i * K:(i + 1) * K]
        s0 = min(s for s, _ in chunk)
        e0 = max(e for _, e in chunk)
        makespans.append((e0 - s0) / 1000.0)
        concurrency.append(max_concurrency(chunk))
        busy_frac.append(busy_ns(chunk) / max(1, e0 - s0))
        durations.extend((e - s) / 1000.0 for s, e in chunk)
        for (s1, e1), (s2, _) in zip(chunk, chunk[1:]):
            gaps.append((s2 - e1) / 1000.0)
    gaps_sorted = sorted(gaps)
    row = {
        "label": args.label or args.range,
        "kernels_per_launch": K,
        "launches": launches,
        "leftover_kernels": leftover,
        "makespan_us_median": statistics.median(makespans),
        "makespan_us_min": min(makespans),
        "makespan_us_max": max(makespans),
        "makespan_per_kernel_us": statistics.median(makespans) / K,
        "kernel_duration_us_median": statistics.median(durations),
        "gap_us_median": statistics.median(gaps) if gaps else float("nan"),
        "gap_us_mean": statistics.fmean(gaps) if gaps else float("nan"),
        "gap_us_p90": percentile(gaps_sorted, 0.90),
        "max_concurrent_kernels": max(concurrency),
        "busy_fraction_median": statistics.median(busy_frac),
    }
    for k, v in row.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")
    if args.csv:
        new = False
        try:
            open(args.csv).close()
        except FileNotFoundError:
            new = True
        with open(args.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new:
                w.writeheader()
            w.writerow(row)


if __name__ == "__main__":
    main()
