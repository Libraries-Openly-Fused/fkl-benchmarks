#!/usr/bin/env python3
"""Repeatability of the HF graph-structure probe and the interleave test (window "graphs-hf120").

    hf_graph_repeats.py --runs RUN1.csv RUN2.csv RUN3.csv --interleave INTERLEAVE.csv --e0 DIR_RUN1 DIR_RUN2 DIR_RUN3 --out DIR

--runs are three sweeps of fkl-self/graphs/hf_graph_structure_probe.cu over the same batch sizes (the first with all five structures, the others with child and flat_serial). For every batch size and
for the structures present in all three: the mean GPU time of one launch in each run, and the spread (largest over smallest, minus 1). --interleave is the same probe run with
FKLB_GRAPH_INTERLEAVE=1 (the work of crop 0 is launched and synchronised between two timed launches, as the loop of the E0 experiments does with the fused kernel); --e0 are runs of the HF experiment
(benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv: "Iterated Batch MeanTime" is the child-node graph of the experiment). For the batch sizes of the interleave run the script prints
the child graph of the experiment, of the three sweeps and of the interleaved run, with the ratios. Writes hf_graph_repeats.csv and hf_graph_repeats.txt in --out.
"""
import argparse
import csv
import os
import statistics as st

E0_FILE = "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv"
STRUCTURES = ["plain", "child", "child_chained", "flat_serial", "flat_parallel"]


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    return [dict(zip(header, row)) for row in reader]


def probe(path):
    return {(int(r["crops"]), r["structure"]): float(r["gpu_ms_mean"]) for r in read(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs=3, required=True)
    ap.add_argument("--interleave", required=True)
    ap.add_argument("--e0", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [probe(p) for p in args.runs]
    inter = probe(args.interleave)
    e0_child, e0_fused = {}, {}
    for d in args.e0:
        for r in read(os.path.join(d, E0_FILE)):
            b = int(r["Batch size"])
            e0_child.setdefault(b, []).append(float(r["Iterated Batch MeanTime"]))
            e0_fused.setdefault(b, []).append(float(r["Fused Batch MeanTime"]))
    batches = sorted({b for b, _ in runs[0]})

    lines = []
    rows = []
    lines.append(f"sweeps of {len(batches)} batch sizes, three runs: {', '.join(os.path.relpath(p) for p in args.runs)}")
    for s in ("child", "flat_serial"):
        spreads = []
        for b in batches:
            v = [r[(b, s)] for r in runs if (b, s) in r]
            if len(v) != 3:
                continue
            sp = max(v) / min(v) - 1
            spreads.append((sp, b))
            rows.append([b, s, f"{v[0]:.6f}", f"{v[1]:.6f}", f"{v[2]:.6f}", f"{st.median(v):.6f}", f"{sp * 100:.2f}"])
        vals = sorted(x for x, _ in spreads)
        worst = max(spreads)
        lines.append(f"{s}: spread between the three runs (largest / smallest - 1) over {len(spreads)} batch sizes: median {st.median(vals) * 100:.2f} %, 90th percentile "
                     f"{vals[int(0.9 * (len(vals) - 1))] * 100:.2f} %, largest {worst[0] * 100:.2f} % at {worst[1]} crops; batch sizes above 2 %: {sum(1 for x in vals if x > 0.02)}, above 5 %: {sum(1 for x in vals if x > 0.05)}")
    lines.append("")

    lines.append("child-node graph at the interleave batch sizes (mean GPU time of one launch, ms)")
    lines.append("crops | E0 child (median of runs) | sweep 1 | sweep 2 | sweep 3 | interleaved | sweep1/E0 | sweep2/E0 | sweep3/E0 | interleaved/E0")
    inter_batches = sorted({b for b, _ in inter})
    for b in inter_batches:
        e0c = st.median(e0_child[b])
        v = [r[(b, "child")] for r in runs]
        i = inter[(b, "child")]
        lines.append(f"{b} | {e0c:.4f} | " + " | ".join(f"{x:.4f}" for x in v) + f" | {i:.4f} | " + " | ".join(f"{x / e0c:.3f}" for x in v) + f" | {i / e0c:.3f}")
    lines.append("")
    lines.append("all structures at the interleave batch sizes: sweep 1 (no work in between) -> interleaved (ms, ratio interleaved / sweep 1)")
    for b in inter_batches:
        cells = []
        for s in STRUCTURES:
            if (b, s) in inter and (b, s) in runs[0]:
                cells.append(f"{s} {runs[0][(b, s)]:.4f} -> {inter[(b, s)]:.4f} ({inter[(b, s)] / runs[0][(b, s)]:.3f})")
        lines.append(f"{b}: " + "; ".join(cells))
    lines.append("")

    ratios = []
    for b in batches:
        e0c = st.median(e0_child[b])
        ratios.append((runs[0][(b, "child")] / e0c, b))
    lines.append(f"sweep 1 child / E0 child over {len(batches)} batch sizes: median {st.median(x for x, _ in ratios):.3f}, smallest {min(ratios)[0]:.3f} at {min(ratios)[1]} crops, largest {max(ratios)[0]:.3f} at {max(ratios)[1]} crops; "
                 f"batch sizes with ratio above 1.05: {sum(1 for x, _ in ratios if x > 1.05)}")
    med3 = []
    for b in batches:
        e0c = st.median(e0_child[b])
        med3.append((st.median(r[(b, "child")] for r in runs) / e0c, b))
    lines.append(f"median of the three sweeps child / E0 child: median {st.median(x for x, _ in med3):.3f}, smallest {min(med3)[0]:.3f} at {min(med3)[1]} crops, largest {max(med3)[0]:.3f} at {max(med3)[1]} crops; "
                 f"batch sizes above 1.05: {sum(1 for x, _ in med3 if x > 1.05)}")

    with open(os.path.join(args.out, "hf_graph_repeats.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["crops", "structure", "run1_ms", "run2_ms", "run3_ms", "median_ms", "spread_percent"])
        w.writerows(rows)
    with open(os.path.join(args.out, "hf_graph_repeats.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
