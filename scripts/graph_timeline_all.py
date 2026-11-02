#!/usr/bin/env python3
"""Runs scripts/graph_timeline.py on every Nsight Systems report of results/.../e0-graph-structure/nsys/ and collects the rows in timeline_summary.csv.

    graph_timeline_all.py results/rtx4090/e0-graph-structure/nsys

Each sub-directory is named <probe>_<structure>_<point> (hf_child_301, vf_flat_serial_502) and holds report.nsys-rep, recorded by the window script with
  nsys profile --trace=cuda,nvtx --cuda-graph-trace=node  FKLB_GRAPH_STRUCTURES=<structure> FKLB_GRAPH_CROPS=301 (hf) | FKLB_GRAPH_OPS=502 (vf)
The report is exported to SQLite (report.sqlite) if that has not been done, the NVTX range of the probe "<structure>@<point>" selects the timed launches, and the kernels per launch are the crops (hf)
or 50 x (operations + 1) (vf). The report must be exported with the nsys that recorded it (NSYS=<path>, the one of the toolkit of the window: cuda-13.4.3/bin/nsys). Nsight Systems slows the launches: the timeline gives the gaps between kernels and the concurrency, not the times of the experiments.
"""
import csv
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
STRUCTURES = ["plain", "child", "child_chained", "flat_serial", "flat_parallel"]


def main():
    root = sys.argv[1]
    rows = []
    for name in sorted(os.listdir(root)):
        d = os.path.join(root, name)
        rep = os.path.join(d, "report.nsys-rep")
        if not os.path.isfile(rep):
            continue
        probe, rest = name.split("_", 1)
        structure = next(s for s in sorted(STRUCTURES, key=len, reverse=True) if rest.startswith(s + "_"))
        point = int(rest[len(structure) + 1:])
        kernels = point if probe == "hf" else 50 * (point + 1)
        sql = os.path.join(d, "report.sqlite")
        if not os.path.isfile(sql):
            subprocess.run([os.environ.get("NSYS", "nsys"), "export", "--type", "sqlite", "-o", sql, "--force-overwrite", "true", rep], check=True, stdout=subprocess.DEVNULL)
        out = subprocess.run([sys.executable, os.path.join(HERE, "graph_timeline.py"), sql, "--range", f"{structure}@{point}", "--kernels-per-launch", str(kernels), "--label", name],
                             capture_output=True, text=True)
        if out.returncode != 0:
            print(f"{name}: {out.stderr.strip() or out.stdout.strip()}")
            continue
        row = {"probe": probe, "structure": structure, "point": point}
        for line in out.stdout.splitlines():
            k, v = line.split(": ", 1)
            row[k] = v
        rows.append(row)
        print(f"{name}: makespan {float(row['makespan_us_median']):.1f} us for {row['kernels_per_launch']} kernels, gap median {float(row['gap_us_median']):.3f} us, kernel {float(row['kernel_duration_us_median']):.3f} us, "
              f"max concurrent {row['max_concurrent_kernels']}, busy {float(row['busy_fraction_median']):.2f}")
    if rows:
        with open(os.path.join(root, "timeline_summary.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
