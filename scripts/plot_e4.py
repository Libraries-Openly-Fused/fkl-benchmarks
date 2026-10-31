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
"""E4a figure and the compact CSVs of E4c and E4d, from the three quiet runs of the pre-processing program (results/rtx4090/e4-preprocessing).

  python scripts/plot_e4.py --csv results/rtx4090/e4-preprocessing [--out figures/e4_pipeline_r2.pdf]

Figure (3.4 x 2.0 in, one column): speedup of the fused pre-processing kernel (FKL, one launch) against OpenCV-CUDA with one batched call per operation, OpenCV-CUDA per crop and NPP, against the
number of crops (10 to 300); the median over the three runs of the per-run ratio (`e4_speedups.csv`). FKL uses the sampling convention of its baseline: ASYMMETRIC against OpenCV-CUDA, HALF_PIXEL against NPP.
CSVs (next to the figure's sidecar and in the results folder): e4c_compact.csv (the three whole frames: GPU time of every version, speedups, host time) and e4d_compact.csv (DRAM bytes of the whole call).
"""
import csv
import glob
import os
import shutil

import plot_common as pc
import matplotlib.pyplot as plt

SERIES = [("opencv_batch", "vs OpenCV-CUDA (batched ops)"), ("opencv_percrop", "vs OpenCV-CUDA (per crop)"), ("npp", "vs NPP")]
FRAMES = {720: "720p (720 x 480)", 1920: "1080p (1920 x 1080)", 3840: "4K (3840 x 2160)"}
OUT_SIZES = {720: "768 x 512", 1920: "2048 x 1152", 3840: "4096 x 2304"}


def rd(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def main():
    ap = pc.arg_parser(__doc__, "e4_pipeline_r2.pdf")
    args = ap.parse_args()
    d = args.csv if os.path.isdir(args.csv) else os.path.dirname(args.csv)
    runs = sorted(glob.glob(os.path.join(d, "quiet-run*", "e4_preprocessing.csv")))
    speed = rd(os.path.join(d, "e4_speedups.csv"))
    summ = {(r["experiment"], int(r["batch"]), int(r["src_w"]), r["version"]): r for r in rd(os.path.join(d, "e4_summary.csv"))}

    pc.apply_style()
    fig, ax = plt.subplots(figsize=(3.4, 2.0), constrained_layout=True)
    notes = []
    for i, (base, label) in enumerate(SERIES):
        pts = sorted((int(r["batch"]), float(r["speedup_median"]), float(r["speedup_smallest_run"]), float(r["speedup_largest_run"])) for r in speed if r["experiment"] == "E4a" and r["baseline"] == base)
        x, y = [p[0] for p in pts], [p[1] for p in pts]
        mk = pc.SERIES_MARKERS[i]
        ax.plot(x, y, color=pc.SERIES_COLORS[i], marker=mk, markersize=2.6, markevery=3, linewidth=1.0, label=label, clip_on=False,
                markerfacecolor="white" if base == "opencv_percrop" else pc.SERIES_COLORS[i], markeredgewidth=0.7)
        notes.append(f"{label}: {len(x)} points, batch {x[0]} to {x[-1]}; speedup {y[0]:.1f}x at batch {x[0]}, {y[-1]:.1f}x at batch {x[-1]}; largest run-to-run range of a point (smallest to largest of the three runs) "
                     f"{max(100 * (p[3] - p[2]) / p[1] for p in pts):.1f} % of the median")
    ax.set_xlim(0, 310)
    ax.set_ylim(bottom=0)
    ax.set_xticks(range(0, 301, 50))
    ax.set_xlabel("Number of crops (batch)")
    ax.set_ylabel("Speedup of the fused\nkernel")
    ax.legend(loc="upper left")
    notes.append("FKL sampling: ASYMMETRIC against OpenCV-CUDA (the convention of cv::cuda::resize), HALF_PIXEL against NPP (its convention)")
    notes.append("the three quiet runs read: " + ", ".join(os.path.relpath(p, pc.REPO) for p in runs) + " (median over the runs of the per-run ratio; no other process was running)")
    pc.save(fig, args.out, runs, "plot_e4.py", notes)

    # E4c compact CSV: the three whole frames
    rows = []
    for w, name in FRAMES.items():
        g = lambda v, col="gpu_mean_ms_median": float(summ[("E4c", 1, w, v)][col])
        sp = lambda base: next(float(r["speedup_median"]) for r in speed if r["experiment"] == "E4c" and int(r["src_w"]) == w and r["baseline"] == base)
        rows.append({"frame": name, "output": OUT_SIZES[w], "opencv_ms": f"{g('opencv_percrop'):.4g}", "npp_ms": f"{g('npp'):.4g}", "fkl_ms": f"{g('fkl_fused'):.4g}", "fkl_half_pixel_ms": f"{g('fkl_fused_half_pixel'):.4g}",
                     "speedup_vs_opencv": f"{sp('opencv_percrop'):.1f}", "speedup_vs_npp": f"{sp('npp'):.1f}", "opencv_kernels": 7, "npp_kernels": 9, "fkl_kernels": 1,
                     "host_us_opencv": f"{g('opencv_percrop', 'host_enqueue_us_median'):.1f}", "host_us_npp": f"{g('npp', 'host_enqueue_us_median'):.1f}", "host_us_fkl": f"{g('fkl_fused', 'host_enqueue_us_median'):.1f}",
                     "run_to_run_spread_pct_max": f"{max(float(summ[('E4c', 1, w, v)]['spread_pct']) for v in ('opencv_percrop', 'npp', 'fkl_fused', 'fkl_fused_half_pixel')):.1f}"})
    stem = os.path.splitext(args.out)[0]
    with open(stem + "_e4c_compact.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    # E4d compact CSV: DRAM bytes of the whole call (Nsight Compute), the caches flushed before every kernel (cold) and not flushed (warm)
    cold = {(r["point"], r["version"]): r for r in rd(os.path.join(d, "ncu-cold", "e4_dram_bytes.csv"))}
    warm = {(r["point"], r["version"]): r for r in rd(os.path.join(d, "ncu-warm", "e4_dram_bytes.csv"))}
    out = []
    for point, label in (("E4a-100", "100 crops of 60 x 120 -> 64 x 128"), ("E4c-3840", "one 4K frame -> 4096 x 2304")):
        for v in ("opencv_percrop", "opencv_batch", "npp", "fkl_fused", "fkl_fused_half_pixel"):
            if (point, v) not in cold:
                continue
            ref_c, ref_w = float(cold[(point, "fkl_fused")]["dram_MB"]), float(warm[(point, "fkl_fused")]["dram_MB"])
            out.append({"case": label, "version": v, "kernels": cold[(point, v)]["kernels"], "dram_MB_flushed_caches": cold[(point, v)]["dram_MB"], "dram_MB_not_flushed": warm[(point, v)]["dram_MB"],
                        "ratio_to_fkl_flushed": f"{float(cold[(point, v)]['dram_MB']) / ref_c:.1f}", "ratio_to_fkl_not_flushed": f"{float(warm[(point, v)]['dram_MB']) / ref_w:.1f}"})
    with open(stem + "_e4d_compact.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(out[0]))
        wr.writeheader()
        wr.writerows(out)
    print(f"wrote {stem}_e4c_compact.csv ({len(rows)} rows) and {stem}_e4d_compact.csv ({len(out)} rows)")


if __name__ == "__main__":
    main()
