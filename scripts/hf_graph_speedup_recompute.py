#!/usr/bin/env python3
"""The speedup of the fused kernel against each graph structure, for the HF experiment of E0 (benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs).

    hf_graph_speedup_recompute.py --e0 DIR_RUN1 DIR_RUN2 DIR_RUN3 --probe hf_graph_structure_probe.csv [more.csv] --out DIR

--e0 are the directories of runs of the HF experiment on the RTX 4090 (each holds benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv). In that file "Iterated Batch" is the time of
the experiment's Graphs baseline (a graph of one child graph node per crop, launched with cudaGraphLaunch) and "Fused Batch" the time of the one fused kernel for the whole batch. The probe
(fkl-self/graphs/hf_graph_structure_probe.cu) times the same per-crop kernels in plain launches and in four graph structures at the same batch sizes. For every batch size the script gives:
  * the time of the fused kernel: the median over the E0 runs of "Fused Batch MeanTime";
  * the time of each structure from the probe (mean GPU time of one launch, the same kind of number as the E0 mean) and the speedup of the fused kernel against it (ratio of means);
  * the E0 value of the child-node graph (median over the runs) and the ratio of the probe's child to it, which checks that the probe reproduces the experiment;
  * the fastest structure that a user would write without changing the kernels (plain, child, child_chained, flat_serial, flat_parallel) and the speedup against it.
Writes hf_graph_speedups.csv and hf_graph_speedups.txt (with the maximum of each speedup series and the values at 661, 831 and 1191 crops) in --out.
"""
import argparse
import csv
import os
import statistics as st

STRUCTURES = ["plain", "child", "child_chained", "flat_serial", "flat_parallel"]
E0_FILE = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv"


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    return [dict(zip(header, row)) for row in reader]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0", nargs="+", required=True)
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fused, child = {}, {}
    for d in args.e0:
        for r in read(os.path.join(d, E0_FILE)):
            b = int(r["Batch size"])
            fused.setdefault(b, []).append(float(r["Fused Batch MeanTime"]))
            child.setdefault(b, []).append(float(r["Iterated Batch MeanTime"]))
    probe = {}
    for path in args.probe:
        for r in read(path):
            probe[(int(r["crops"]), r["structure"])] = r
    batches = sorted({b for b, _ in probe if b in fused})

    header = ["crops", "fused_ms", "e0_child_ms", "probe_child_over_e0_child"] + [f"{s}_ms" for s in STRUCTURES] + [f"speedup_vs_{s}" for s in STRUCTURES] + ["fastest_structure", "speedup_vs_fastest"]
    rows = []
    for b in batches:
        f_ms = st.median(fused[b])
        e0c = st.median(child[b])
        times = {s: float(probe[(b, s)]["gpu_ms_mean"]) for s in STRUCTURES if (b, s) in probe}
        if "child" not in times:
            continue
        best = min(times, key=times.get)
        rows.append([b, f"{f_ms:.6f}", f"{e0c:.6f}", f"{times['child'] / e0c:.3f}"] + [f"{times.get(s, float('nan')):.6f}" for s in STRUCTURES] +
                    [f"{times.get(s, float('nan')) / f_ms:.2f}" for s in STRUCTURES] + [best, f"{times[best] / f_ms:.2f}"])
    with open(os.path.join(args.out, "hf_graph_speedups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)

    idx = {h: i for i, h in enumerate(header)}
    lines = [f"batch sizes: {len(rows)} (the fused time is the median of {len(args.e0)} E0 runs)"]
    for label, col in (("child graph of the experiment (probe)", "speedup_vs_child"), ("flat serial graph", "speedup_vs_flat_serial"), ("fastest structure", "speedup_vs_fastest"),
                       ("plain launches", "speedup_vs_plain")):
        vals = [(int(r[0]), float(r[idx[col]])) for r in rows]
        top = max(vals, key=lambda t: t[1])
        lines.append(f"{label}: maximum {top[1]:.2f}x at {top[0]} crops; median over the sweep {st.median(v for _, v in vals):.2f}x")
    for target in (661, 831, 1191):
        for r in rows:
            if int(r[0]) == target:
                lines.append(f"{target} crops: fused {float(r[idx['fused_ms']]) * 1000:.2f} us; child {float(r[idx['child_ms']]):.4f} ms ({r[idx['speedup_vs_child']]}x), flat_serial {float(r[idx['flat_serial_ms']]):.4f} ms "
                             f"({r[idx['speedup_vs_flat_serial']]}x), plain {float(r[idx['plain_ms']]):.4f} ms ({r[idx['speedup_vs_plain']]}x); fastest {r[idx['fastest_structure']]} ({r[idx['speedup_vs_fastest']]}x)")
    worst = max(abs(float(r[idx["probe_child_over_e0_child"]]) - 1.0) for r in rows)
    lines.append(f"check of the probe against the experiment: the probe's child graph over the E0 child graph differs by at most {worst * 100:.1f} % over the sweep")
    with open(os.path.join(args.out, "hf_graph_speedups.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
