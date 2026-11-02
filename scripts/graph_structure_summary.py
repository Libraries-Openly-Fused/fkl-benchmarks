#!/usr/bin/env python3
"""Tables of the CUDA-Graphs structure probes (opencv-cuda/graphs/graph_structure_probe.cu, fkl-self/graphs/hf_graph_structure_probe.cu).

    graph_structure_summary.py graph_structure_probe.csv [more.csv ...] [--out DIR] [--tag NAME]

For every point (operations per crop, or crops) and structure: the nodes of the graph, the GPU time (median of the events around one launch: what the experiments report), the GPU time per kernel
node, the ratio to plain launches and to the child-node graph of the experiments, the host submit time, the total time with the synchronisation, the capture and instantiate times, and the spread
(largest over smallest GPU time). Several CSVs with the same columns (a quick run and a long run) are joined; a later file replaces the rows of the same point and structure of an earlier one.
Writes structure_summary.csv and structure_summary.md in --out (default: the directory of the first CSV).
"""
import argparse
import csv
import os

STRUCTURES = ["plain", "child", "child_chained", "flat_serial", "flat_parallel"]


def read(path):
    rows = []
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    for r in csv.DictReader(lines):
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = args.out or os.path.dirname(os.path.abspath(args.csv[0]))
    os.makedirs(out, exist_ok=True)

    table = {}
    point_name = None
    for path in args.csv:
        for r in read(path):
            point_name = point_name or list(r.keys())[0]
            table[(int(r[point_name]), r["structure"])] = r
    points = sorted({p for p, _ in table})

    header = [point_name, "structure", "kernel_nodes", "top_level_nodes", "gpu_ms_median", "gpu_us_per_kernel_node", "vs_plain", "vs_child", "submit_us_median", "total_ms_median", "capture_s",
              "instantiate_s", "instantiate_us_per_kernel_node", "first_launch_ms", "spread_pct", "reps", "warmup_s", "warmup_steady", "bit_exact_vs_plain"]
    rows = []
    for p in points:
        plain = table.get((p, "plain"))
        child = table.get((p, "child"))
        for s in STRUCTURES:
            r = table.get((p, s))
            if r is None:
                continue
            gpu = float(r["gpu_ms_median"])
            nodes = int(r["kernel_nodes"])
            vs_plain = gpu / float(plain["gpu_ms_median"]) if plain else float("nan")
            vs_child = gpu / float(child["gpu_ms_median"]) if child else float("nan")
            spread = (float(r["gpu_ms_max"]) / float(r["gpu_ms_min"]) - 1.0) * 100.0
            inst = float(r["instantiate_s"])
            rows.append([p, s, nodes, r["top_level_nodes"], f"{gpu:.5f}", f"{gpu * 1000.0 / nodes:.4f}", f"{vs_plain:.3f}", f"{vs_child:.3f}", f"{float(r['submit_us_median']):.1f}",
                         f"{float(r['total_ms_median']):.5f}", f"{float(r['capture_s']):.4f}", f"{inst:.4f}", f"{inst * 1e6 / nodes:.3f}" if s != "plain" else "", f"{float(r['first_launch_ms']):.4f}",
                         f"{spread:.1f}", r["reps"], f"{float(r['warmup_s']):.1f}", r["warmup_steady"], r["bit_exact_vs_plain"]])

    stem = "structure_summary" + (f"_{args.tag}" if args.tag else "")
    with open(os.path.join(out, stem + ".csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(out, stem + ".md"), "w") as f:
        f.write("| " + " | ".join(header) + " |\n|" + "---|" * len(header) + "\n")
        for row in rows:
            f.write("| " + " | ".join(str(x) for x in row) + " |\n")
    for row in rows:
        print("  ".join(f"{h}={x}" for h, x in zip(header[:10], row[:10])))
    print(f"wrote {os.path.join(out, stem)}.csv/.md")


if __name__ == "__main__":
    main()
