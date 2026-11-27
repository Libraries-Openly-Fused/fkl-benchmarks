# E19 crop sweep: the timing window e19-crop-sweep-timing (RTX 4090)

> **Dated note, 2026-11-27.** The paper no longer prints numbers from this record. Its RTX 4090 crop-sweep factors and chart come from side B of `../e19-crop-sweep-retime-304492a/` (`window/quiet-run{1,2,3}-B/` and `differences-B/`; Oscar's answers of 2026-11-27). This record is unchanged.

> **Timed again (dated note, 2026-11-27).** These times were measured at fkl-benchmarks `b4fb594` (CSV repo `246b1c8`) with FKL `24478f0`. The sweep was timed again at FKL 304492a in `../e19-crop-sweep-retime-304492a/` (#70, 8c419c7), with the builds of the pins b10fc66 and 304492a alternating, as Oscar ordered on 2026-11-26.

This is the timing of the E19 crop sweep: E19's pipeline with N crops, N = 1, 10, 20, ..., 150. The program is `e19_crop_sweep_cu` at 246b1c8 (sha256 `2b800f1d...`), the binary of the checks window `results/rtx4090/e19-crop-sweep-checks/`, where every check passed at every N. The script is `results/rtx4090/e19-crop-sweep-phase2/window-timing.sh`; `window-script.sh` here is the copy that ran, and it is the same file.

- **When:** 2026-11-24, from 08:34:54 to 08:51:22 (the GPU steps), on the RTX 4090 (nvidia-smi index 1). The desktop session kept running. The window was granted by the coordinator for #56 at b0b744d.
- **The sweeps**, each made of 16 processes, one per N, in ascending order. Each process is a complete run (the checks, then the timing of every row) in `<sweep>/N<NNN>/`. The rows run in the order fkl_loop, fkl_batch, opencv_cuda, npp, each whole_call then kernel_only.
  - `warmup` (not used): 08:34:54-08:36:39.
  - `quiet-run1`: until 08:38:38.
  - `quiet-run2`: until 08:40:42.
  - `quiet-run3`: until 08:43:06.
  - `nsys`: under the CUDA API and NVTX trace, until 08:51:18. Its times are not used.
- **Before every run** (80 in all), the script waited for an idle machine: no measurement, build or compile job, a 1-minute load below 0.5, and no other program on the RTX 4090.
- Every run was pinned to cores 0-3,16-19, exited 0, passed its checks and wrote complete checks and timing CSVs. nvidia-smi read the GPU every 100 ms (`gpu_clocks.csv`).
- **The intervals** are E19's. whole_call is CUDA events around one call (100 calls after one warm-up call). kernel_only is the call behind a 5 ms GPU spin.
- **fkl_loop** ran first in every run only so that the row order is E19's. It is in the raw CSVs only: no table or ratio here uses it.

## Times: the median (min-max) of the three quiet sweeps, in us per frame (N crops)

Each value is the mean of 100 calls. The ratios are computed in each quiet sweep from that sweep's means, and shown as median (min-max). Source: `analysis/sweep-report.txt`, `analysis/e19_crop_sweep_summary.csv`, `scripts-used/sweep_report.py`.

#### whole_call

| N | fkl_batch, us | opencv_cuda, us | npp, us | opencv_cuda / fkl_batch | npp / fkl_batch |
|---|---|---|---|---|---|
| 1 | 4.71 (4.69-4.76) | 9.74 (9.71-10.51) | 8.48 (8.45-8.55) | 2.060 (2.045-2.242) | 1.799 (1.775-1.823) |
| 10 | 9.59 (9.56-9.66) | 80.75 (80.69-80.76) | 72.49 (72.45-72.49) | 8.422 (8.352-8.443) | 7.560 (7.500-7.579) |
| 20 | 15.81 (15.74-15.84) | 159.35 (159.09-159.94) | 143.46 (143.36-143.59) | 10.095 (10.061-10.127) | 9.066 (9.055-9.125) |
| 30 | 22.04 (21.93-22.06) | 238.84 (238.66-239.09) | 214.64 (214.50-214.69) | 10.848 (10.821-10.889) | 9.741 (9.726-9.786) |
| 40 | 28.22 (28.20-28.48) | 317.22 (316.83-317.27) | 285.63 (285.54-285.75) | 11.244 (11.125-11.250) | 10.126 (10.030-10.127) |
| 50 | 34.33 (34.31-34.38) | 396.26 (396.18-396.76) | 356.90 (356.60-357.05) | 11.550 (11.523-11.556) | 10.395 (10.372-10.407) |
| 60 | 40.61 (40.58-40.64) | 473.28 (472.20-473.29) | 427.89 (427.74-428.23) | 11.647 (11.638-11.655) | 10.538 (10.537-10.542) |
| 70 | 46.77 (46.72-46.80) | 553.34 (552.98-554.15) | 499.02 (498.99-499.16) | 11.841 (11.824-11.844) | 10.670 (10.663-10.684) |
| 80 | 52.58 (52.57-52.70) | 632.73 (632.36-633.43) | 569.89 (569.82-570.46) | 12.026 (12.018-12.036) | 10.837 (10.824-10.841) |
| 90 | 59.02 (58.88-59.28) | 711.16 (710.64-711.52) | 641.02 (640.91-641.24) | 12.049 (12.002-12.069) | 10.861 (10.811-10.891) |
| 100 | 65.35 (65.27-65.40) | 803.48 (803.21-806.86) | 713.00 (712.73-715.80) | 12.306 (12.295-12.337) | 10.920 (10.911-10.945) |
| 110 | 71.52 (71.41-71.55) | 890.58 (889.33-891.31) | 792.16 (791.97-793.25) | 12.454 (12.453-12.457) | 11.086 (11.074-11.093) |
| 120 | 77.67 (77.64-77.68) | 985.64 (983.95-987.11) | 874.80 (874.06-876.12) | 12.689 (12.668-12.714) | 11.262 (11.253-11.285) |
| 130 | 83.84 (83.78-83.84) | 1080.27 (1079.41-1080.69) | 948.94 (948.34-950.85) | 12.884 (12.875-12.899) | 11.318 (11.311-11.349) |
| 140 | 89.92 (89.81-89.92) | 1168.11 (1167.72-1170.99) | 1025.35 (1025.03-1027.01) | 13.003 (12.990-13.022) | 11.414 (11.402-11.421) |
| 150 | 96.08 (96.04-96.56) | 1254.84 (1254.13-1255.55) | 1101.08 (1100.57-1104.16) | 13.059 (13.003-13.061) | 11.460 (11.435-11.460) |

#### kernel_only

| N | fkl_batch, us | opencv_cuda, us | npp, us | opencv_cuda / fkl_batch | npp / fkl_batch |
|---|---|---|---|---|---|
| 1 | 4.13 (4.11-4.15) | 8.65 (8.61-8.91) | 8.21 (8.19-8.21) | 2.092 (2.083-2.159) | 1.991 (1.978-1.991) |
| 10 | 9.22 (9.22-9.22) | 78.90 (78.90-78.90) | 72.60 (72.46-72.60) | 8.557 (8.553-8.561) | 7.871 (7.859-7.877) |
| 20 | 15.38 (15.38-15.40) | 156.91 (156.84-156.92) | 143.54 (143.53-143.56) | 10.197 (10.188-10.203) | 9.332 (9.320-9.335) |
| 30 | 21.70 (21.69-21.71) | 236.35 (236.31-236.46) | 215.05 (215.03-215.06) | 10.889 (10.883-10.903) | 9.907 (9.904-9.916) |
| 40 | 28.01 (27.98-28.05) | 315.07 (315.01-315.14) | 286.38 (286.35-286.50) | 11.247 (11.233-11.260) | 10.223 (10.212-10.235) |
| 50 | 34.22 (34.21-34.38) | 392.85 (392.79-392.86) | 357.64 (357.64-357.75) | 11.480 (11.425-11.481) | 10.452 (10.401-10.456) |
| 60 | 40.46 (40.35-40.47) | 466.99 (466.97-467.01) | 428.97 (428.96-429.03) | 11.541 (11.541-11.572) | 10.603 (10.601-10.630) |
| 70 | 46.79 (46.73-46.86) | 548.40 (548.35-548.46) | 500.46 (500.46-500.50) | 11.721 (11.705-11.735) | 10.697 (10.681-10.711) |
| 80 | 52.71 (52.62-52.77) | 628.17 (628.17-628.39) | 571.56 (571.51-571.79) | 11.922 (11.905-11.938) | 10.848 (10.832-10.861) |
| 90 | 58.75 (58.74-58.77) | 704.94 (704.78-705.05) | 643.00 (642.81-643.14) | 11.997 (11.996-12.002) | 10.944 (10.943-10.944) |
| 100 | 64.98 (64.95-65.17) | 797.20 (796.93-800.07) | 714.76 (714.70-718.61) | 12.274 (12.263-12.277) | 11.005 (10.998-11.027) |
| 110 | 71.00 (70.98-71.06) | 882.89 (882.65-885.52) | 791.87 (791.40-793.61) | 12.435 (12.434-12.462) | 11.152 (11.149-11.169) |
| 120 | 77.06 (77.01-77.12) | 978.02 (977.64-978.73) | 875.81 (875.71-878.19) | 12.691 (12.691-12.695) | 11.371 (11.365-11.388) |
| 130 | 83.27 (83.24-83.41) | 1075.15 (1074.98-1076.46) | 949.32 (949.04-951.74) | 12.909 (12.906-12.916) | 11.404 (11.396-11.410) |
| 140 | 89.33 (89.32-89.49) | 1166.46 (1166.02-1166.74) | 1025.32 (1025.26-1028.01) | 13.053 (13.037-13.060) | 11.479 (11.478-11.487) |
| 150 | 95.51 (95.50-95.72) | 1249.23 (1248.70-1249.26) | 1100.73 (1100.66-1103.56) | 13.075 (13.051-13.080) | 11.525 (11.524-11.529) |

- **Spreads**, (max - min) / median of the three quiet means: at most 1.00 % for N ≥ 10. At N = 1: opencv_cuda whole_call 8.23 % (9.71-10.51; one sweep high), opencv_cuda kernel_only 3.47 %, fkl_batch whole_call 1.61 %, npp whole_call 1.09 %.
- **whole_call / kernel_only** of the medians: within 3 % of 1 for every row at N ≥ 20, so these are GPU-bound numbers.
  - The exceptions are N = 1 (fkl_batch 1.142, opencv_cuda 1.126, npp 1.032) and N = 10 (fkl_batch 1.040).
  - At those N, whole_call holds a share of the host's work (see `analysis/sweep-report.txt`).

## Kernels per call and the spin's coverage at every N (the nsys sweep; `analysis/spin-coverage.txt`, `scripts-used/spin_coverage.py`)

- **Kernels per call:** the kernels of each whole_call range divided by its 101 calls. At every N this gives fkl_batch 1, opencv_cuda 4N and npp 3N; fkl_loop gives N.
- **Coverage:** a kernel_only call is covered when the host's launch of its last kernel returns before the spin ends on the GPU.
  - Every row at every N: 100 of 100 calls covered.
  - fkl_loop's smallest margin is 4,292.1 us, at N = 150.
  - The trace has no texture, array or surface API call.

| N | kernels per call: fkl_batch / opencv_cuda / npp | kernel_only calls covered by the spin | smallest margin, us: fkl_batch / opencv_cuda / npp |
|---|---|---|---|
| 1 | 1 / 4 / 3 | 100/100 in every row | 4,991.6 / 4,977.3 / 4,985.0 |
| 10 | 1 / 40 / 30 | 100/100 in every row | 4,994.5 / 4,812.2 / 4,900.8 |
| 20 | 1 / 80 / 60 | 100/100 in every row | 4,990.1 / 4,685.6 / 4,773.7 |
| 30 | 1 / 120 / 90 | 100/100 in every row | 4,991.4 / 4,597.0 / 4,689.6 |
| 40 | 1 / 160 / 120 | 100/100 in every row | 4,992.5 / 4,414.4 / 4,587.6 |
| 50 | 1 / 200 / 150 | 100/100 in every row | 4,991.6 / 4,283.2 / 4,503.7 |
| 60 | 1 / 240 / 180 | 100/100 in every row | 4,992.5 / 4,016.9 / 4,418.0 |
| 70 | 1 / 280 / 210 | 100/100 in every row | 4,989.0 / 3,999.5 / 4,326.1 |
| 80 | 1 / 320 / 240 | 100/100 in every row | 4,992.0 / 3,873.5 / 4,191.3 |
| 90 | 1 / 360 / 270 | 100/100 in every row | 4,990.9 / 3,532.5 / 4,151.4 |
| 100 | 1 / 400 / 300 | 100/100 in every row | 4,992.0 / 3,492.2 / 3,944.6 |
| 110 | 1 / 440 / 330 | 100/100 in every row | 4,990.4 / 3,189.9 / 3,799.9 |
| 120 | 1 / 480 / 360 | 100/100 in every row | 4,991.4 / 3,055.8 / 3,778.9 |
| 130 | 1 / 520 / 390 | 100/100 in every row | 4,990.8 / 3,165.2 / 3,658.4 |
| 140 | 1 / 560 / 420 | 100/100 in every row | 4,989.7 / 2,710.0 / 3,407.4 |
| 150 | 1 / 600 / 450 | 100/100 in every row | 4,990.5 / 2,839.5 / 3,303.3 |

The smallest margin over all rows and N is 2,710.0 us, opencv_cuda at N = 140.

## The SM clock (flagged; no cause is claimed)

- **The quiet runs are not all at 2,685 MHz.** Each run starts after the idle wait, with the GPU idle at 210 MHz (P8). In each run, the clock reads 2,550 MHz once the GPU starts working, then 2,685. In quiet-run1/N150, for example, it read 2,550 from 08:38:34.7 to 08:38:36.1 and 2,685 from 08:38:36.2. Per run: `analysis/clock-by-run.txt`, `analysis/clock-by-run.csv`, `scripts-used/clock_by_run.py`.
- **The quiet runs have no trace, so the rows' timed spans are estimated.** The estimate comes from `analysis/quiet-clock.txt`, `analysis/quiet-clock.csv` and `scripts-used/quiet_clock.py`:
  - Each row's span is placed backwards from the run's end, with the row lengths taken from the run's own CSV.
  - It allows for the end being known only to the second, plus 0.5 s of teardown, so each span is about 1.5 s wider than its row.
  - In the span of fkl_batch, opencv_cuda or npp, many runs hold readings below 2,685 MHz:

| N | runs (of 3 quiet) whose span holds a reading below 2,685 MHz: fkl_batch / opencv_cuda / npp | the readings below 2,685 in those spans |
|---|---|---|
| 1 | 3 / 3 / 2 | 210, 240, 2,550 |
| 10 | 1 / 1 / 0 | 210, 2,550 |
| 20 | 3 / 3 / 2 | 210, 2,550 |
| 30 | 0 / 0 / 0 | - |
| 40 | 3 / 2 / 1 | 210, 2,550 |
| 50 | 2 / 2 / 0 | 225, 2,550 |
| 60 | 1 / 1 / 0 | 2,550 |
| 70 | 3 / 2 / 0 | 2,550 |
| 80 | 3 / 2 / 0 | 210, 2,550 |
| 90 | 1 / 0 / 0 | 2,550 |
| 100 | 2 / 1 / 1 | 2,550, 2,595, 2,670 |
| 110 | 3 / 1 / 0 | 210, 2,550 |
| 120 | 3 / 3 / 0 | 2,550 |
| 130 | 3 / 2 / 1 | 2,550 |
| 140 | 2 / 2 / 0 | 2,550 |
| 150 | 2 / 2 / 0 | 2,550 |

- **In the times**, the runs whose span holds such a reading do not differ from the others.
  - At N ≥ 10, every quiet mean is within -0.48 % to +0.92 % of its N's median, in both groups of runs.
  - A row timed entirely at 2,550 MHz would be about 5.3 % slower (2,685 / 2,550).
  - At N ≥ 40, fkl_batch's whole_call / kernel_only is between 0.998 and 1.008.
- **The nsys sweep has exact times per row.** It places each kernel on the wall clock and compares it with the readings within 0.5 s (`analysis/clock-by-row-nsys.txt`, `scripts-used/clock_by_row.py`):
  - No kernel of fkl_batch, opencv_cuda or npp is at 2,550 MHz throughout its ±0.5 s.
  - fkl_batch's whole_call falls inside the 2,550-to-2,685 transition at every N. Its kernel_only and opencv_cuda's whole_call are partly inside the transition at some N.
  - Every later range is at 2,685.
  - fkl_loop, the first timed row, ran at 2,550 or inside the transition at every N.

## The repeat: N = 30 against `results/rtx4090/e19-batched-box-fkl-batch/` (E19's timing window of 2026-11-18)

| row, interval | old, us | new, us | change | spread old, new | exceeds the larger spread? | above 3 %? | cause | GPU- or host-bound |
|---|---|---|---|---|---|---|---|---|
| fkl_batch whole_call | 22.11 | 22.04 | -0.31 % | 0.19 %, 0.55 % | no | no | not shown | GPU-bound |
| opencv_cuda whole_call | 239.12 | 238.84 | -0.12 % | 0.97 %, 0.18 % | no | no | not shown | GPU-bound |
| npp whole_call | 214.17 | 214.64 | +0.22 % | 0.32 %, 0.08 % | no | no | not shown | GPU-bound |
| fkl_batch kernel_only | 21.83 | 21.70 | -0.59 % | 0.09 %, 0.13 % | yes (no overlap) | no | not shown | GPU-bound |
| opencv_cuda kernel_only | 237.64 | 236.35 | -0.55 % | 0.03 %, 0.06 % | yes (no overlap) | no | not shown | GPU-bound |
| npp kernel_only | 212.17 | 215.05 | +1.36 % | 0.13 %, 0.02 % | yes (no overlap) | no | not shown | GPU-bound |

- Every change is below the 3 % flag of a GPU-bound number.
- The three kernel_only changes exceed both windows' spreads, which are at most 0.13 %, and their 3-run ranges do not overlap. No cause is named.
- At N = 30, the estimated spans of the three rows hold only 2,685 MHz readings in all three quiet runs.

## The unused sweeps (flags)

- **warmup:**
  - Every run passed. Against the quiet medians, at N ≥ 10: -0.9 % to +0.4 % in every row and interval.
  - At N = 1: fkl_batch kernel_only +4.5 %, opencv_cuda kernel_only +3.6 %, npp kernel_only +2.2 %, opencv_cuda whole_call +2.1 %, npp whole_call +1.4 %, fkl_batch whole_call +0.1 %.
- **nsys:** every run passed. Under the trace every number is higher than the quiet median, so its times are not used:
  - whole_call: fkl_batch +4.9 % to +21.3 %, opencv_cuda +61.1 % to +72.5 %, npp +24.3 % to +30.2 % at N ≥ 10 (+34.8 %, +63.4 % and +35.6 % at N = 1);
  - kernel_only: fkl_batch +4.2 % to +9.8 %, opencv_cuda +7.2 % to +9.0 %, npp +7.2 % to +8.2 % at N ≥ 10 (+19.2 %, +14.2 % and +10.1 % at N = 1).

## Files

| Path | Contents |
|---|---|
| `window-script.sh`, `window-console.txt` | the script that ran and its console |
| `environment-before.txt`, `environment-after.txt`, `gpu-info.csv` | the machine before and after the window |
| `gpu_clocks.csv`, `gpu_clocks.err` | the nvidia-smi sampler |
| `load.log` | the 5-second load log |
| `warmup/`, `quiet-run1/`, `quiet-run2/`, `quiet-run3/`, `nsys/` | one folder per N in each, holding stdout.txt, stderr.txt, exit_code.txt, load.txt, gpu-at-end.csv, e19_batched_box_checks.csv and e19_batched_box.csv |
| `analysis/` | the reports above, `nsys-reports.sha256` and `nsys-reports-sizes.txt` |
| `scripts-used/` | the scripts that wrote them, run on the CSVs, the readings and the SQLite exports of the nsys reports |

**Left out:**
- **The 16 nsys reports** of the nsys sweep, `nsys/N<NNN>/e19-crop-sweep-timing-N<NNN>.nsys-rep`: 119.6 MiB in all, from 0.4 MiB at N = 1 to 14.7 MiB at N = 150, left out at the coordinator's request. Each report's size and sha256 are in `analysis/nsys-reports-sizes.txt` and `analysis/nsys-reports.sha256`. The files are kept on the measuring machine until the paper is accepted.
- **Their SQLite exports** (made with `nsys export --type sqlite`, derived from the reports). They are in the scratch folder.
