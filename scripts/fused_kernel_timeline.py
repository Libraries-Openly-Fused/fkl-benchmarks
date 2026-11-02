#!/usr/bin/env python3
"""The GPU duration of the fused kernel in every timed loop of hf_fused_time_probe, from an Nsight Systems report exported to SQLite.

    nsys profile --trace=cuda,nvtx -o REPORT  hf_fused_time_probe_cu         (FKLB_FUSED_CROPS=1191 FKLB_FUSED_ITERS=50 FKLB_FUSED_WARMUP=10 ...)
    nsys export --type sqlite --force-overwrite true REPORT.nsys-rep
    fused_kernel_timeline.py REPORT.sqlite [--out fused_kernel_timeline.csv]

The probe wraps every timed loop in an NVTX range "<mode>@<crops>". In each range the kernels whose grid depth is the number of crops are the launches of the fused kernel (one block layer per
crop); for each range the script gives their number, the GPU duration (end minus start of the kernel, from CUPTI: it contains no launch latency), and the time from the entry of the CUDA runtime
call that launched the kernel to the start of the kernel on the GPU. Nsight Systems slows the host side, so the second number is larger than without a tool; the first is not affected.
"""
import argparse
import csv
import sqlite3
import statistics


def columns(db, table):
    return [r[1] for r in db.execute(f"PRAGMA table_info({table})")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sqlite")
    ap.add_argument("--out")
    ap.add_argument("--grid-z", type=int, help="grid depth of the fused kernel when it is not the number after @ in the range name (vf_fused_time_probe: 50 crops)")
    args = ap.parse_args()
    db = sqlite3.connect(args.sqlite)
    ncols = columns(db, "NVTX_EVENTS")
    text_col = "text" if "text" in ncols else "textId"
    if text_col == "textId":
        names = {i: s for i, s in db.execute("SELECT id, value FROM StringIds")}
    ranges = []
    for row in db.execute(f"SELECT start, end, {text_col} FROM NVTX_EVENTS WHERE end IS NOT NULL ORDER BY start"):
        text = names.get(row[2], "") if text_col == "textId" else row[2]
        if text and "@" in text:
            ranges.append((row[0], row[1], text))
    kcols = columns(db, "CUPTI_ACTIVITY_KIND_KERNEL")
    need = {"gridZ", "correlationId"} - set(kcols)
    if need:
        raise SystemExit(f"CUPTI_ACTIVITY_KIND_KERNEL has no column {need}: {kcols}")
    runtime = {c: (s, e) for c, s, e in db.execute("SELECT correlationId, start, end FROM CUPTI_ACTIVITY_KIND_RUNTIME")}
    rows = []
    for start, end, text in ranges:
        mode, point = text.rsplit("@", 1)
        crops = int(point)
        gridz = args.grid_z or crops
        fused = db.execute("SELECT start, end, correlationId FROM CUPTI_ACTIVITY_KIND_KERNEL WHERE start >= ? AND end <= ? AND gridZ = ? ORDER BY start", (start, end, gridz)).fetchall()
        if not fused:
            continue
        durations = [(e - s) / 1000.0 for s, e, _ in fused]
        latency = [(s - runtime[c][0]) / 1000.0 for s, _, c in fused if c in runtime]
        row = {"crops": crops, "mode": mode, "fused_kernels": len(fused), "duration_us_median": statistics.median(durations), "duration_us_mean": statistics.fmean(durations),
               "duration_us_min": min(durations), "duration_us_max": max(durations), "launch_call_to_kernel_start_us_median": statistics.median(latency) if latency else float("nan")}
        rows.append(row)
        print(f"{mode}@{crops}: {len(fused)} fused kernels, duration median {row['duration_us_median']:.2f} us (mean {row['duration_us_mean']:.2f}, min {row['duration_us_min']:.2f}, max {row['duration_us_max']:.2f}), "
              f"runtime call entry to kernel start {row['launch_call_to_kernel_start_us_median']:.2f} us")
    if args.out and rows:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
