#!/usr/bin/env python3
"""The speedup of the fused kernel against each graph structure, for the VF + HF experiment (mul_add) of E0.

    graph_speedup_recompute.py --e0 e0-vf-mul-add/benchmark_vertical_fusion_loopMulAdd_CUDAGraphs.csv --probe graph_structure_probe.csv [more.csv] --out DIR

The E0 CSV of the CUDA-Graphs pass holds, for every number of operations, the time of the OpenCV-CUDA graph of the experiment (child graph nodes, "OpenCV MeanTime") and of the fused kernel
("cvGS MeanTime"). The probe (opencv-cuda/graphs/graph_structure_probe.cu) times the same OpenCV-CUDA work in other structures at some of those numbers of operations. For each structure:
  * at the operations the probe measured: speedup = mean GPU time of the structure / cvGS MeanTime of the E0 run (ratio of means), next to the E0 value of the same kind for the child graph;
  * over the whole sweep: the GPU time of the structure is a straight line of the operations (checked by the residuals), so a least-squares line over the measured points of 502 operations or more gives
    the time at every operation count of the sweep; the plateau (mean over the sweep points from 2,502 operations) and the maximum come from that line, and are marked "fitted".
The check of the probe against the experiment is the child structure: its mean GPU time against OpenCV MeanTime of the E0 run at the same operations.
"""
import argparse
import csv
import os

STRUCTURES = ["plain", "child", "child_chained", "flat_serial", "flat_parallel"]


def read(path):
    with open(path, newline="") as f:
        lines = [line for line in f if not line.startswith("#")]
    reader = csv.reader(lines)
    header = [h.strip() for h in next(reader)]
    return [dict(zip(header, row)) for row in reader]


def fit(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e0", required=True)
    ap.add_argument("--probe", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--plateau-from", type=int, default=2502)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    e0 = {}
    for r in read(args.e0):
        keys = list(r.keys())
        e0[int(r[keys[0]])] = r
    probe = {}
    for path in args.probe:
        for r in read(path):
            probe[(int(r["ops"]), r["structure"])] = r
    ops_measured = sorted({o for o, _ in probe})

    lines = []
    out_rows = []
    lines.append("measured points: speedup = mean GPU time of the structure / cvGS MeanTime of the E0 run")
    lines.append("ops | cvGS ms | " + " | ".join(STRUCTURES) + " | E0 child (OpenCV MeanTime) | probe child / E0 child")
    for o in ops_measured:
        if o not in e0:
            continue
        cvgs = float(e0[o]["cvGS MeanTime"])
        e0child = float(e0[o]["OpenCV MeanTime"])
        cells = []
        for s in STRUCTURES:
            r = probe.get((o, s))
            if r is None:
                cells.append("-")
                continue
            t = float(r["gpu_ms_mean"])
            cells.append(f"{t:.4f} ms ({t / cvgs:,.0f}x)")
            out_rows.append([o, s, "measured", f"{t:.6f}", f"{cvgs:.6f}", f"{t / cvgs:.1f}"])
        c = probe.get((o, "child"))
        ratio = f"{float(c['gpu_ms_mean']) / e0child:.3f}" if c else "-"
        lines.append(f"{o} | {cvgs:.5f} | " + " | ".join(cells) + f" | {e0child:.4f} ms ({e0child / cvgs:,.0f}x) | {ratio}")

    lines.append("")
    lines.append(f"whole sweep (points of the E0 run from {args.plateau_from} operations), straight line fitted to the measured points of 502 operations or more")
    sweep = sorted(o for o in e0 if o >= args.plateau_from)
    for s in STRUCTURES:
        pts = [(o, float(probe[(o, s)]["gpu_ms_mean"])) for o in ops_measured if (o, s) in probe and o >= 502]
        if len(pts) < 3:
            continue
        a, b = fit([p[0] for p in pts], [p[1] for p in pts])
        worst = max(abs(a + b * o - t) / t for o, t in pts)
        speedups = []
        for o in sweep:
            cvgs = float(e0[o]["cvGS MeanTime"])
            speedups.append((o, (a + b * o) / cvgs))
        plateau = sum(v for _, v in speedups) / len(speedups)
        best = max(speedups, key=lambda t: t[1])
        lines.append(f"{s}: time = {a:.5f} ms + {b * 1000:.4f} us x ops (largest residual {worst * 100:.1f} % of the measured time); fitted plateau {plateau:,.0f}x, fitted maximum {best[1]:,.0f}x at {best[0]} operations")
        for o, v in speedups:
            out_rows.append([o, s, "fitted", f"{a + b * o:.6f}", f"{float(e0[o]['cvGS MeanTime']):.6f}", f"{v:.1f}"])
    child_e0 = [(o, float(e0[o]["OpenCV MeanTime"]) / float(e0[o]["cvGS MeanTime"])) for o in sweep]
    lines.append(f"E0 run, child-node graph, ratio of means: plateau {sum(v for _, v in child_e0) / len(child_e0):,.0f}x, maximum {max(child_e0, key=lambda t: t[1])[1]:,.0f}x at {max(child_e0, key=lambda t: t[1])[0]} operations")
    child_mean_of_ratios = [(o, float(e0[o]["Mean Speedup"])) for o in sweep]
    lines.append(f"E0 run, child-node graph, mean of ratios (the column Mean Speedup): plateau {sum(v for _, v in child_mean_of_ratios) / len(child_mean_of_ratios):,.0f}x, maximum "
                 f"{max(child_mean_of_ratios, key=lambda t: t[1])[1]:,.0f}x at {max(child_mean_of_ratios, key=lambda t: t[1])[0]} operations")

    with open(os.path.join(args.out, "graph_speedups.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ops", "structure", "kind", "gpu_ms_mean", "cvgs_ms_mean", "speedup_ratio_of_means"])
        w.writerows(out_rows)
    with open(os.path.join(args.out, "graph_speedups.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
