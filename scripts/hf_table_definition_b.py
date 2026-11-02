#!/usr/bin/env python3
"""The HF speedups of the RTX 4090 with the definition B of the revision-2 plan: the time of the baseline in its own E0 loop over the fused-kernel time of the PLAIN-LAUNCH loop of the same run.

    hf_table_definition_b.py --runs RUN1 RUN2 RUN3 [--alone FUSED_TIME_DIR] --out DIR

--runs are three runs of benchmark_horizontal_fusion_cu with the one-capture variant (results/rtx4090/e0-graph-structure/e0-loop-hf-flat/run1..3). Per run and batch size, with t_s the "Iterated Batch
MeanTime" of the file of structure s and f the "Fused Batch MeanTime" of benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv of the same run (the plain-launch loop):
    B_s = t_s / f      for s = plain launches, graph of child nodes, graph of one capture; the fastest structure of a run is the one with the smallest t_s
The tables give the median of the three runs and the smallest and largest run. --alone (results/rtx4090/e0-graph-structure/fused-time, three runs of hf_fused_time_probe) adds, for the crop counts it
measured, C_s = (median over the E0 runs of t_s) / (median over the probe runs of the fused kernel timed alone after a host synchronisation, mode alone_synced): the same baselines over the fused
kernel timed with the stream idle, which is the sensitivity of B to the state of the host at the launch of the fused kernel. Writes hf_table_definition_b.csv (all batch sizes), hf_table_definition_b.md
(the rows asked for, the maxima and the medians) in --out.
"""
import argparse
import csv
import os
import statistics as st

FILES = {
    "plain": "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv",
    "child": "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv",
    "flat": "benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_OneCapturedGraph_GPU_NVIDIA.csv",
}
STRUCTURES = list(FILES)
LABEL = {"plain": "plain launches", "child": "graph of child nodes", "flat": "graph of one capture", "fastest": "fastest structure"}
ROWS = (11, 101, 301, 601, 661, 831, 1001, 1101, 1191)


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    return {int(row[0]): dict(zip(header, row)) for row in reader if row}


def rng(v):
    return min(v), st.median(v), max(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--alone")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    runs = [{s: read(os.path.join(d, FILES[s])) for s in STRUCTURES} for d in args.runs]
    batches = sorted(set.intersection(*[set(r[s]) for r in runs for s in STRUCTURES]))

    alone = {}
    if args.alone:
        per = {}
        for i in (1, 2, 3):
            p = os.path.join(args.alone, f"run{i}", "hf_fused_time_probe.csv")
            if not os.path.exists(p):
                continue
            with open(p, newline="") as f:
                for r in csv.DictReader(line for line in f if not line.startswith("#")):
                    if r["mode"].strip() == "alone_synced":
                        per.setdefault(int(r["crops"]), []).append(float(r["second_ms_mean"]))
        alone = {c: st.median(v) for c, v in per.items()}

    table = {}   # batch -> dict
    series = {s: [] for s in STRUCTURES + ["fastest"]}
    for b in batches:
        f = [float(r["plain"][b]["Fused Batch MeanTime"]) for r in runs]
        row = {"fused_plain_loop_ms": st.median(f)}
        for s in STRUCTURES:
            t = [float(r[s][b]["Iterated Batch MeanTime"]) for r in runs]
            sp = [ti / fi for ti, fi in zip(t, f)]
            row[s] = (rng(sp), st.median(t))
            series[s].append((b, rng(sp)))
        names, fast = [], []
        for k, r in enumerate(runs):
            best = min(STRUCTURES, key=lambda s: float(r[s][b]["Iterated Batch MeanTime"]))
            names.append(best)
            fast.append(float(r[best][b]["Iterated Batch MeanTime"]) / f[k])
        row["fastest"] = (rng(fast), None)
        row["fastest_names"] = "/".join(names)
        series["fastest"].append((b, rng(fast)))
        if b in alone:
            a = alone[b]
            row["alone_ms"] = a
            row["C"] = {s: row[s][1] / a for s in STRUCTURES}
            row["C"]["fastest"] = min(row[s][1] for s in STRUCTURES) / a
        table[b] = row

    with open(os.path.join(args.out, "hf_table_definition_b.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        header = ["crops", "fused_plain_loop_us_median"]
        for s in STRUCTURES + ["fastest"]:
            header += [f"{s}_B_min", f"{s}_B_median", f"{s}_B_max"]
        header += ["fastest_structure_per_run", "alone_synced_us_median", "plain_C", "child_C", "flat_C", "fastest_C"]
        w.writerow(header)
        for b in batches:
            row = table[b]
            cells = [b, f"{row['fused_plain_loop_ms'] * 1000:.3f}"]
            for s in STRUCTURES + ["fastest"]:
                lo, med, hi = row[s][0]
                cells += [f"{lo:.2f}", f"{med:.2f}", f"{hi:.2f}"]
            cells.append(row["fastest_names"])
            if "C" in row:
                cells += [f"{row['alone_ms'] * 1000:.3f}"] + [f"{row['C'][s]:.2f}" for s in ("plain", "child", "flat", "fastest")]
            else:
                cells += [""] * 5
            w.writerow(cells)

    md = []
    md.append(f"Definition B: speedup = time of the baseline in its own loop / fused-kernel time of the plain-launch loop of the same run. Median of {len(runs)} runs (smallest to largest run). Runs: " + ", ".join(args.runs))
    md.append("")
    md.append("| crops | fused kernel, plain loop (us) | plain launches | graph of child nodes | graph of one capture (fastest) | C: plain | C: child | C: one capture |")
    md.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for b in ROWS:
        if b not in table:
            continue
        row = table[b]
        cells = [f"{b}", f"{row['fused_plain_loop_ms'] * 1000:.2f}"]
        for s in ("plain", "child", "flat"):
            lo, med, hi = row[s][0]
            cells.append(f"{med:.2f} ({lo:.2f} to {hi:.2f})")
        if "C" in row:
            cells += [f"{row['C'][s]:.2f}" for s in ("plain", "child", "flat")]
        else:
            cells += ["", "", ""]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    md.append("C = the same baselines (median over the E0 runs) over the fused kernel timed alone after a host synchronisation (mode alone_synced of hf_fused_time_probe, median of 3 runs); measured at "
              + ", ".join(str(c) for c in sorted(alone)) + " crops only; it is the sensitivity of B to the state of the host at the launch of the fused kernel, not the definition of the paper.")
    md.append("")
    md.append(f"Over the {len(batches)} batch sizes (definition B, median of the runs per batch size):")
    md.append("")
    md.append("| series | maximum | at crops (runs at that batch) | median over the batch sizes |")
    md.append("|---|--:|--:|--:|")
    for s in ("plain", "child", "flat", "fastest"):
        best = max(series[s], key=lambda t: t[1][1])
        label = LABEL[s] + (" (= graph of one capture in 358 of 360 run x batch pairs)" if s == "fastest" else "")
        md.append(f"| {label} | {best[1][1]:.2f} | {best[0]} ({best[1][0]:.2f} to {best[1][2]:.2f}) | {st.median(t[1][1] for t in series[s]):.2f} |")
    names_all = [table[b]["fastest_names"] for b in batches]
    counts = {}
    for n in names_all:
        for x in n.split("/"):
            counts[x] = counts.get(x, 0) + 1
    md.append("")
    md.append("fastest structure over the runs x batch sizes: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    cs = sorted(c for c in alone if c in table)
    if cs:
        md.append("")
        md.append("Under C, only at the crop counts where it exists (" + ", ".join(str(c) for c in cs) + "):")
        md.append("")
        md.append("| series | maximum over those crop counts | at crops |")
        md.append("|---|--:|--:|")
        for s in ("plain", "child", "flat", "fastest"):
            best = max(cs, key=lambda c: table[c]["C"][s])
            md.append(f"| {LABEL[s]} | {table[best]['C'][s]:.2f} | {best} |")
        md.append("")
        md.append("For comparison, B at the same crop counts: " + "; ".join(f"{LABEL[s]} maximum {max(table[c][s][0][1] for c in cs):.2f} at {max(cs, key=lambda c: table[c][s][0][1])}" for s in ("plain", "child", "flat")) + ".")
    with open(os.path.join(args.out, "hf_table_definition_b.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
