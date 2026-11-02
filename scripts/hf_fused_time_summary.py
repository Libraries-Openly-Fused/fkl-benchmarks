#!/usr/bin/env python3
"""The time of the fused kernel in each loop of hf_fused_time_probe, over several runs of the probe.

    hf_fused_time_summary.py --runs RUN1 RUN2 RUN3 --out DIR [--nsys fused_kernel_timeline.csv]

Each RUN is a directory with hf_fused_time_probe.csv. For every number of crops and mode: the mean of the "second" interval (microseconds; what the experiment reports for the fused kernel) in
each run, the median of the runs, the host time of the call that launches the fused kernel, and the difference to alone_synced (the stream idle, then the launch). With --nsys (from
fused_kernel_timeline.py) the GPU duration of the fused kernel from Nsight Systems is added for the modes it covers. Writes hf_fused_time_summary.csv and hf_fused_time_summary.txt in --out.
"""
import argparse
import csv
import os
import statistics as st

MODES = ["loop_plain", "loop_child", "loop_child_chained", "loop_flat_serial", "loop_flat_parallel", "loop_flat_serial_pollute2MB", "loop_flat_serial_pollute64MB", "loop_plain_touch",
         "loop_child_touch", "alone_synced", "alone_synced_pollute2MB", "alone_synced_pollute64MB", "alone_pipelined10", "alone_pipelined50", "alone_queued_behind_spin"]


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    return list(csv.DictReader(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--nsys")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = {}
    for i, d in enumerate(args.runs):
        name = "hf_fused_time_probe.csv" if os.path.exists(os.path.join(d, "hf_fused_time_probe.csv")) else "vf_fused_time_probe.csv"
        for r in read(os.path.join(d, name)):
            data.setdefault((int(r.get("crops") or r["ops"]), r["mode"]), []).append(r)
    nsys = {}
    if args.nsys:
        for r in read(args.nsys):
            nsys[(int(r["crops"]), r["mode"])] = r
    crops_list = sorted({c for c, _ in data})
    header = ["crops", "mode", "runs", "second_us_mean_run1", "second_us_mean_run2", "second_us_mean_run3", "second_us_median_of_runs", "second_us_range_of_runs", "diff_to_alone_synced_us",
              "first_ms_mean_median_of_runs", "fused_launch_call_us_median", "nsys_kernel_duration_us_median"]
    rows, lines = [], []
    for c in crops_list:
        base = data.get((c, "alone_synced"))
        base_us = st.median(float(r["second_ms_mean"]) * 1000 for r in base) if base else None
        lines.append(f"== {c} crops ==")
        lines.append("mode | second per run (us) | median of runs | diff to alone_synced | fused launch call (us) | first (ms) | nsys kernel (us)")
        for m in MODES:
            rs = data.get((c, m))
            if not rs:
                continue
            seconds = [float(r["second_ms_mean"]) * 1000 for r in rs]
            med = st.median(seconds)
            call = st.median(float(r["fused_launch_call_us_median"]) for r in rs)
            first = st.median(float(r["first_ms_mean"]) for r in rs)
            ns = nsys.get((c, m))
            ns_text = f"{float(ns['duration_us_median']):.2f}" if ns else ""
            diff = f"{med - base_us:+.2f}" if base_us is not None else ""
            padded = (seconds + ["", "", ""])[:3]
            rows.append([c, m, len(rs)] + [f"{x:.3f}" if x != "" else "" for x in padded] + [f"{med:.3f}", f"{min(seconds):.3f} to {max(seconds):.3f}", diff, f"{first:.4f}", f"{call:.2f}", ns_text])
            lines.append(f"{m} | " + " / ".join(f"{x:.2f}" for x in seconds) + f" | {med:.2f} | {diff} | {call:.2f} | {first:.4f} | {ns_text}")
        lines.append("")
    with open(os.path.join(args.out, "hf_fused_time_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(args.out, "hf_fused_time_summary.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
