#!/usr/bin/env python3
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E16, the attention core with nothing fused around it (question Q1): TFLOP/s of the FKL tensor-core attention DPP (mma_raw) and of the baselines that are plausible best kernels, against the sequence length.

  python scripts/plot_e16.py --csv results/rtx4090/e16-attention [--causal 0] [--tag ""] [--gpu-label "RTX 4090"] [--out-dir figures] [--width 3.4] [--all]

Two panels stacked (head dimension 64 above, 128 below), x = S on a log2 axis (512 to 8192), y = TFLOP/s = 4 BH S^2 D (halved for causal) / time from the GPU time per call (blocks of back-to-back calls). One point = the median over the
runs (run*/ folders) of the run medians; the smallest and largest run median of every point are in the .txt sidecar. Series: FKL mma_raw (the library as shipped, its own tile choice, no tuning), SDPA flash (FlashAttention-2 of PyTorch),
cuDNN SDPA, flash-attn 2.8.3 and Triton's fused-attention forward (autotuner as shipped); with --all also the memory-efficient SDPA kernel. A baseline without a result on this GPU (no kernel for the architecture, not importable) is
not drawn and the sidecar says why. Output: e16_core_r2 (non-causal, --causal 0), e16_core_causal_r2 (--causal 1; a separate file that can be dropped); with --tag 5050 the names get the tag (e16_core_5050_r2).
"""
import csv
import math
import os
import statistics as stats
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator

import e16_summary as es
import plot_common as pc

SERIES = [("FKL (mma_raw)", "mma_raw"), ("SDPA flash (FA2)", "flash"), ("cuDNN SDPA", "cudnn"), ("flash-attn 2.8.3", "flash_attn"), ("Triton", "triton"), ("SDPA efficient", "efficient")]
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#56B4E9", "#E69F00"]   # Okabe-Ito in fixed order: FKL blue, then the baselines
MARKERS = ["o", "s", "^", "D", "v", "P"]
BEST_OF = ["flash", "cudnn", "efficient", "flash_attn", "triton"]


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--csv", required=True, help="folder of the runs (run1, run2, ...), e.g. results/rtx4090/e16-attention")
    ap.add_argument("--causal", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--gpu-label", default="")
    ap.add_argument("--out-dir", default=pc.FIGURES)
    ap.add_argument("--width", type=float, default=3.4)
    ap.add_argument("--height", type=float, default=3.0)
    ap.add_argument("--all", action="store_true", help="also draw the memory-efficient SDPA kernel")
    ap.add_argument("--y0", action="store_true", help="y axes from 0 (no cut); the files get the suffix _y0")
    ap.add_argument("--paper", action="store_true", help="the version for the paper: height 2.4 in (unless --height), the legend names of the paper, the legend in two rows")
    args = ap.parse_args()
    runs = es.load(args.csv)
    names = list(runs)
    series = list(SERIES if args.all else SERIES[:-1])
    if args.paper:
        paper_names = {"mma_raw": "FKL", "flash": "SDPA, FlashAttention-2", "cudnn": "SDPA, cuDNN", "flash_attn": "flash-attn 2.8.3", "triton": "Triton", "efficient": "SDPA, memory-efficient"}
        series = [(paper_names[v], v) for _, v in series]
        if args.height == 3.0:
            args.height = 2.4
    dims = sorted({d for rows in runs.values() for (_, d, _, c) in rows if c == args.causal})
    seqs = sorted({s for rows in runs.values() for (_, _, s, c) in rows if c == args.causal})
    gpu = next((row["gpu"] for rows in runs.values() for row in rows.values() if row.get("gpu")), "?")

    def values(variant, d, s):
        return [float(rows[(variant, d, s, args.causal)]["tflops_median"]) for rows in runs.values() if (variant, d, s, args.causal) in rows and es.tflops(rows[(variant, d, s, args.causal)])]

    notes, missing = [], {}
    for rows in runs.values():
        for (v, d, s, c), row in rows.items():
            if c == args.causal and not es.tflops(row) and row.get("note"):
                missing.setdefault(v, row["note"])
    pc.apply_style()
    fig, axes = plt.subplots(len(dims), 1, figsize=(args.width, args.height), sharex=True, constrained_layout=True, squeeze=False)
    handles = {}
    for ax, d in zip(axes[:, 0], dims):
        for i, (label, variant) in enumerate(series):
            pts = [(s, values(variant, d, s)) for s in seqs]
            pts = [(s, v) for s, v in pts if v]
            if not pts:
                continue
            xs, ys = [p[0] for p in pts], [stats.median(p[1]) for p in pts]
            (line,) = ax.plot(xs, ys, color=COLORS[i], marker=MARKERS[i], markersize=3.2, linewidth=1.0 if variant != "mma_raw" else 1.4, label=label, clip_on=False, zorder=3 if variant == "mma_raw" else 2)
            handles[label] = line
            for s, v in pts:
                notes.append(f"D={d} S={s} {label}: {stats.median(v):.1f} TFLOP/s [{min(v):.1f}-{max(v):.1f}] over {len(v)} runs")
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_locator(FixedLocator(seqs))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x)}"))
        allv = [stats.median(vs) for _, variant in series for sv in seqs for vs in [values(variant, d, sv)] if vs]   # the plotted points: medians over the runs
        lo, hi = (min(allv), max(allv)) if allv else (0.0, 1.0)
        if args.y0:
            ymin, ymax = 0.0, math.ceil(1.06 * hi / 10.0) * 10.0
            notes.append(f"D={d}: y axis from 0 to {ymax:.0f} TFLOP/s")
        else:
            ymin = max(0.0, math.floor(0.9 * lo / 10.0) * 10.0)   # the series are within about 10 % of each other: an axis from 0 would hide every difference; the figure says that the axis is cut
            ymax = math.ceil(1.04 * hi / 10.0) * 10.0
            ax.text(0.02, 0.05, f"y axis starts at {ymin:.0f}", transform=ax.transAxes, ha="left", va="bottom", color="#555555")   # in the figure itself, not only in the sidecar
            notes.append(f"D={d}: y axis from {ymin:.0f} to {ymax:.0f} TFLOP/s (does not start at 0: the series are within about 10 % of each other; the figure says so in each panel)")
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel("TFLOP/s")
        ax.text(0.98, 0.05, f"D = {d}", transform=ax.transAxes, ha="right", va="bottom")   # below the curves, which sit in the upper part of the panel
    axes[-1, 0].set_xlabel("Sequence length S")
    fig.legend(list(handles.values()), list(handles.keys()), loc="outside upper center", ncol=3, frameon=False, handlelength=1.6, columnspacing=1.0)

    # the best baseline and the fraction of it, per point, for the sidecar
    for d in dims:
        for s in seqs:
            fr, best, cd, eqf = [], [], [], []
            for rows in runs.values():
                cand = {b: es.tflops(rows[(b, d, s, args.causal)]) for b in BEST_OF if (b, d, s, args.causal) in rows and es.tflops(rows[(b, d, s, args.causal)])}
                raw = rows.get(("mma_raw", d, s, args.causal))
                if cand and raw and es.tflops(raw):
                    b = max(cand, key=cand.get)
                    best.append(b)
                    fr.append(es.tflops(raw) / cand[b])
                    if es.clock(raw) and es.clock(rows[(b, d, s, args.causal)]):
                        cd.append(100.0 * (es.clock(raw) / es.clock(rows[(b, d, s, args.causal)]) - 1.0))
                        eqf.append(es.tflops(raw) / cand[b] * es.clock(rows[(b, d, s, args.causal)]) / es.clock(raw))
            if fr:
                flag = f"; DIFFERENT CLOCKS (SM clock of FKL against the best baseline up to {max(cd, key=abs):+.1f} %)" if cd and max(abs(x) for x in cd) > es.CLOCK_TOLERANCE else ""
                eq = f"; at equal SM clock {stats.median(eqf):.3f} [{min(eqf):.3f}-{max(eqf):.3f}]" if eqf else ""
                notes.append(f"D={d} S={s}: FKL mma_raw / best baseline = {stats.median(fr):.3f} [{min(fr):.3f}-{max(fr):.3f}], best baseline {max(set(best), key=best.count)}{eq}{flag}")
    worst_below, worst_above, widest = (0.0, None), (0.0, None), (0.0, None)
    plotted = [variant for _, variant in series]
    for d in dims:
        for s_ in seqs:
            fracs, bests = [], []
            for rows in runs.values():
                cand = {b: es.tflops(rows[(b, d, s_, args.causal)]) for b in plotted if b != "mma_raw" and (b, d, s_, args.causal) in rows and es.tflops(rows[(b, d, s_, args.causal)])}
                raw = rows.get(("mma_raw", d, s_, args.causal))
                if cand and raw and es.tflops(raw):
                    b = max(cand, key=cand.get)
                    bests.append(b)
                    fracs.append(es.tflops(raw) / cand[b])
            if fracs:
                below = 100.0 * (1.0 - stats.median(fracs))   # statistic: the median over the runs of the per-run fraction FKL / best baseline (the 'FKL / best' column of the summaries); positive: FKL below the best baseline
                b = max(set(bests), key=bests.count)
                if below > worst_below[0]:
                    worst_below = (below, (d, s_, b))
                if -below > worst_above[0]:
                    worst_above = (-below, (d, s_, b))
            med = {v: stats.median(values(v, d, s_)) for v in plotted if values(v, d, s_)}
            if len(med) > 1:
                hi_v, lo_v = max(med, key=med.get), min(med, key=med.get)
                spread = 100.0 * (med[hi_v] - med[lo_v]) / med[hi_v]
                if spread > widest[0]:
                    widest = (spread, (d, s_, hi_v, lo_v))
    if worst_below[1]:
        d_, s_, b_ = worst_below[1]
        notes.append(f"largest difference on the plot, FKL below the best baseline: {worst_below[0]:.1f} % of the best baseline (D={d_}, S={s_}, best {b_}); FKL above the best baseline by up to {worst_above[0]:.1f} % of it"
                     + (f" (D={worst_above[1][0]}, S={worst_above[1][1]}, best {worst_above[1][2]})" if worst_above[1] else "") + " (statistic: 100 x (1 - median over the runs of the per-run fraction FKL / best baseline))")
    if widest[1]:
        d_, s_, hv, lv = widest[1]
        notes.append(f"largest spread between the highest and the lowest plotted series at one point: {widest[0]:.1f} % of the highest (D={d_}, S={s_}, highest {hv}, lowest {lv}; statistic: medians over the runs of TFLOP/s)")
    for v, why in sorted(missing.items()):
        notes.append(f"no result for {v} on {gpu}: {why}")
    drawn = {l for l in handles}
    for label, variant in series:
        if label not in drawn:
            notes.append(f"{label}: not drawn (no result on this GPU)")
    if not missing and all(l in drawn for l, _ in series):
        notes.append(f"baselines without a result on {gpu}: none (SDPA flash, cuDNN SDPA, flash-attn and Triton all ran; no baseline lacks a kernel for this architecture)")
    # the SM clock and the power of the rows of this figure, per series, and of the whole window (numbers to be checked against the *_clock.csv files and gpu_clocks.csv)
    for label, variant in series:
        rws = [r for rr in runs.values() for (v, dd, ss, cc), r in rr.items() if v == variant and cc == args.causal and es.clock(r)]
        if rws:
            clk, pwr = [float(r["sm_mhz_median"]) for r in rws], [float(r["power_w_median"]) for r in rws]
            notes.append(f"SM clock of the {len(rws)} rows of {label} in this figure (both head sizes, 5 sequence lengths, {len(runs)} runs): median {stats.median(clk):.0f} MHz (smallest row {min(clk):.0f}, largest {max(clk):.0f}); power median {stats.median(pwr):.0f} W "
                         f"(largest {max(pwr):.0f} W); rows with median power >= 440 W: {sum(1 for x in pwr if x >= 440)} of {len(pwr)}")
    info = os.path.join(args.csv, "gpu-info.csv")
    if os.path.exists(info):
        with open(info, newline="") as f:
            rec = list(csv.DictReader(f, skipinitialspace=True))[0]
        notes.append("GPU limits (gpu-info.csv, nvidia-smi): " + ", ".join(f"{k} = {v}" for k, v in rec.items() if k.startswith(("clocks.max.sm", "power.limit"))))
    clocks_file = os.path.join(args.csv, "gpu_clocks.csv")
    if os.path.exists(clocks_file):
        smv, pwv, tot = [], [], 0
        with open(clocks_file, newline="") as f:
            for r in csv.reader(f):
                if len(r) < 7:
                    continue
                tot += 1
                try:
                    if float(r[4].split()[0]) >= 50:
                        smv.append(float(r[1].split()[0]))
                        pwv.append(float(r[5].split()[0]))
                except (ValueError, IndexError):
                    pass
        if smv:
            notes.append(f"whole window (gpu_clocks.csv, 100 ms sampler, {len(smv)} of {tot} samples with GPU utilisation >= 50 %): SM clock median {stats.median(smv):.0f} MHz, 5th percentile {sorted(smv)[int(0.05 * len(smv))]:.0f} MHz; power median {stats.median(pwv):.0f} W, "
                         f"{100.0 * sum(1 for x in pwv if x >= 440) / len(pwv):.0f} % of the samples at >= 440 W")
    notes.append("palette (Okabe-Ito, fixed order: FKL blue, SDPA flash vermillion, cuDNN bluish green, flash-attn reddish purple, Triton sky blue) checked with the dataviz validator: lightness, chroma and normal-vision separation pass; the colour-blind separation of "
                 "reddish purple against bluish green is 7.6 (deutan) and the contrast against the white surface of reddish purple (2.98) and sky blue (2.25) is below 3:1, so the identity of a series is carried by its marker and the legend, and every plotted value is in this file")
    notes.append(f"GPU: {gpu}; {len(runs)} runs ({', '.join(names)}); causal = {args.causal}; BH = tokens / S so that BH x S is the same for every S; the library chooses its own tile sizes (no tuning), the baselines are used as shipped "
                 "(PyTorch SDPA backends, flash-attn 2.8.3, the Triton autotuner over 24 configurations that FKL has no counterpart of)")
    stem = f"e16_core{'_causal' if args.causal else ''}{'_' + args.tag if args.tag else ''}_r2{'_y0' if args.y0 else ''}.pdf"
    sources = [os.path.join(args.csv, n, f) for n in names for f in ("e16_attention.csv", "e16_baselines.csv") if os.path.exists(os.path.join(args.csv, n, f))]
    pc.save(fig, os.path.join(args.out_dir, stem), sources, "plot_e16.py", notes)


if __name__ == "__main__":
    main()
