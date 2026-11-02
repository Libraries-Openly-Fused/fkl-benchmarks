#!/usr/bin/env python3
"""The horizontal-fusion experiment of E0 against every graph structure of its baseline, from runs of the E0 loop itself.

    hf_e0_structure_table.py --runs RUN1 RUN2 RUN3 [--probe hf_graph_speedups.csv] --out DIR

--runs are directories of full runs of benchmark_horizontal_fusion_cu built with the second graph variant (fkl-self/fusion/benchmark_horizontal_fusion.h: plain launches, a CUDA graph of one child graph node
per crop, a CUDA graph from one stream capture of all the launches). Each holds
  benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_GPU_NVIDIA.csv                              "Iterated Batch" = plain launches
  benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_GPU_NVIDIA.csv                    "Iterated Batch" = the graph of child nodes (the 2025 structure)
  benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs_OneCapturedGraph_GPU_NVIDIA.csv   "Iterated Batch" = the graph of one capture
and in each of them "Fused Batch" is the fused kernel, timed in the same loop as the baseline of that file. The speedup of a structure is the column "Mean Speedup" of its own file (the mean over the
iterations of the iterated time over the fused time of the same iteration); the ratio of the means of the two columns is given next to it. For every batch size: the three structures with
the median of the runs and the smallest and largest of the three runs; the fastest structure of each run (smallest "Iterated Batch MeanTime") and its speedup; the fused time of the three files of
a run (they are three measurements of the same kernel). With --probe (hf_graph_speedups.csv of hf_graph_speedup_recompute.py) the time of the graph of one capture is also given against the flat_serial
structure of the probe, and the child graph against the child structure of the probe. Writes hf_e0_structure_table.csv and hf_e0_structure_table.txt in --out.
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


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    return {int(row[0]): dict(zip(header, row)) for row in reader if row}


def rng(values):
    return min(values), st.median(values), max(values)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--probe")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    runs = [{s: read(os.path.join(d, FILES[s])) for s in STRUCTURES} for d in args.runs]
    batches = sorted(set.intersection(*[set(r[s]) for r in runs for s in STRUCTURES]))
    probe = {}
    if args.probe:
        for r in csv.DictReader(open(args.probe, newline="")):
            probe[int(r["crops"])] = r

    header = ["crops", "fused_ms_median"]
    for s in STRUCTURES:
        header += [f"{s}_ms_median", f"{s}_speedup_min", f"{s}_speedup_median", f"{s}_speedup_max", f"{s}_ratio_of_means_median"]
    header += ["fastest_structure_per_run", "fastest_speedup_min", "fastest_speedup_median", "fastest_speedup_max", "fused_ms_spread_between_files_percent"]
    # the fused kernel is timed in each of the three files; the same ratios with the fused time of the plain-launch file of the same run in the denominator, for the two graph structures
    for s in ("child", "flat"):
        header += [f"{s}_speedup_fused_of_plain_loop_min", f"{s}_speedup_fused_of_plain_loop_median", f"{s}_speedup_fused_of_plain_loop_max"]
    if probe:
        header += ["flat_over_probe_flat_serial", "child_over_probe_child"]
    rows = []
    series = {s: [] for s in STRUCTURES}
    series["fastest"] = []
    alt = {"child": [], "flat": []}
    for b in batches:
        row = [b]
        fused_all = [float(r[s][b]["Fused Batch MeanTime"]) for r in runs for s in STRUCTURES]
        row.append(f"{st.median(fused_all):.6f}")
        for s in STRUCTURES:
            t = [float(r[s][b]["Iterated Batch MeanTime"]) for r in runs]
            sp = [float(r[s][b]["Mean Speedup"]) for r in runs]
            rm = [float(r[s][b]["Iterated Batch MeanTime"]) / float(r[s][b]["Fused Batch MeanTime"]) for r in runs]
            lo, med, hi = rng(sp)
            row += [f"{st.median(t):.6f}", f"{lo:.2f}", f"{med:.2f}", f"{hi:.2f}", f"{st.median(rm):.2f}"]
            series[s].append((b, med, lo, hi))
        fastest_names, fastest_sp = [], []
        for r in runs:
            best = min(STRUCTURES, key=lambda s: float(r[s][b]["Iterated Batch MeanTime"]))
            fastest_names.append(best)
            fastest_sp.append(float(r[best][b]["Mean Speedup"]))
        lo, med, hi = rng(fastest_sp)
        series["fastest"].append((b, med, lo, hi))
        # the fused kernel is timed in each of the three files of a run: largest over smallest of those times, worst run
        spread = max(max(float(r[s][b]["Fused Batch MeanTime"]) for s in STRUCTURES) / min(float(r[s][b]["Fused Batch MeanTime"]) for s in STRUCTURES) - 1 for r in runs)
        row += ["/".join(fastest_names), f"{lo:.2f}", f"{med:.2f}", f"{hi:.2f}", f"{spread * 100:.1f}"]
        for s in ("child", "flat"):
            v = [float(r[s][b]["Iterated Batch MeanTime"]) / float(r["plain"][b]["Fused Batch MeanTime"]) for r in runs]
            lo2, med2, hi2 = rng(v)
            alt[s].append((b, med2, lo2, hi2))
            row += [f"{lo2:.2f}", f"{med2:.2f}", f"{hi2:.2f}"]
        if probe:
            p = probe.get(b)
            if p:
                row += [f"{st.median(float(r['flat'][b]['Iterated Batch MeanTime']) for r in runs) / float(p['flat_serial_ms']):.3f}",
                        f"{st.median(float(r['child'][b]['Iterated Batch MeanTime']) for r in runs) / float(p['child_ms']):.3f}"]
            else:
                row += ["", ""]
        rows.append(row)

    lines = [f"{len(batches)} batch sizes, {len(runs)} runs: " + ", ".join(args.runs),
             "speedup = column Mean Speedup of the file of the structure (same loop as the fused kernel); median of the runs, in brackets the smallest and the largest run"]
    for s in STRUCTURES + ["fastest"]:
        best = max(series[s], key=lambda t: t[1])
        med_all = st.median(t[1] for t in series[s])
        lines.append(f"{s}: maximum of the medians {best[1]:.2f}x at {best[0]} crops (runs {best[2]:.2f} to {best[3]:.2f}); median over the sweep {med_all:.2f}x")
    for s in ("child", "flat"):
        best = max(alt[s], key=lambda t: t[1])
        lines.append(f"{s}, fused time of the plain-launch file of the run: maximum of the medians {best[1]:.2f}x at {best[0]} crops (runs {best[2]:.2f} to {best[3]:.2f}); median over the sweep {st.median(t[1] for t in alt[s]):.2f}x")
    lines.append("")
    lines.append("crops | fused ms | plain | child nodes | one capture | fastest of each run (speedup) | child, fused of the plain file | one capture, fused of the plain file")
    for b in (11, 101, 301, 601, 661, 831, 1191):
        if b not in batches:
            continue
        i = batches.index(b)
        cells = []
        for s in STRUCTURES:
            _, med, lo, hi = series[s][i]
            cells.append(f"{med:.2f} ({lo:.2f} to {hi:.2f})")
        _, med, lo, hi = series["fastest"][i]
        names = rows[i][header.index("fastest_structure_per_run")]
        extra = []
        for s in ("child", "flat"):
            _, med2, lo2, hi2 = alt[s][i]
            extra.append(f"{med2:.2f} ({lo2:.2f} to {hi2:.2f})")
        lines.append(f"{b} | {float(rows[i][1]) * 1000:.2f} us | " + " | ".join(cells) + f" | {names}: {med:.2f} ({lo:.2f} to {hi:.2f}) | " + " | ".join(extra))
    names_all = [r[header.index("fastest_structure_per_run")] for r in rows]
    counts = {}
    for n in names_all:
        for x in n.split("/"):
            counts[x] = counts.get(x, 0) + 1
    lines.append("fastest structure over the 3 runs x %d batch sizes: " % len(batches) + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    spreads = [float(r[header.index("fused_ms_spread_between_files_percent")]) for r in rows]
    lines.append(f"fused kernel, largest over smallest of its three measurements in a run (worst run): median {st.median(spreads):.1f} %, largest {max(spreads):.1f} %")
    if probe:
        r1 = [float(r[header.index("flat_over_probe_flat_serial")]) for r in rows if r[header.index("flat_over_probe_flat_serial")]]
        r2 = [float(r[header.index("child_over_probe_child")]) for r in rows if r[header.index("child_over_probe_child")]]
        lines.append(f"E0 loop graph of one capture / probe flat_serial: median {st.median(r1):.3f}, range {min(r1):.3f} to {max(r1):.3f}; E0 loop child graph / probe child: median {st.median(r2):.3f}, range {min(r2):.3f} to {max(r2):.3f}")

    with open(os.path.join(args.out, "hf_e0_structure_table.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    with open(os.path.join(args.out, "hf_e0_structure_table.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
