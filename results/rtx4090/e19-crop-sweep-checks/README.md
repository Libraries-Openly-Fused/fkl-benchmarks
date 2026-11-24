# E19 crop sweep, the checks window e19-crop-sweep-checks (RTX 4090)

This is the checks window of the E19 crop sweep: E19's pipeline with N crops, N = 1, 10, 20, ..., 150, one process per N. The program is `e19_crop_sweep_cu` at 246b1c8, sha256 `2b800f1d...`. Its build record and the window's script, as committed before the window, are in `results/rtx4090/e19-crop-sweep-phase1/`; `window-script.sh` here is the copy that ran and is the same file. No time from this window is used.

- **When:** 2026-11-24, from 08:25:47 to 08:28:09 (the GPU steps), on the RTX 4090 (nvidia-smi index 1). The desktop session kept running.
- **Granted by** the coordinator for #56 at 6218e62. The guards passed and the window ran to its end. Every run exited 0.
- **Steps**, all pinned to cores 0-3,16-19, with the nvidia-smi sampler reading every 100 ms (`gpu_clocks.csv`):
  1. `checks/N<NNN>/`: E19_NO_TIMING=1 E19_CROPS=N, for the 16 N in ascending order.
  2. `profile/N<NNN>/`: E19_PROFILE=1 E19_PROFILE_CALLS=10 E19_CROPS=N under nsys (CUDA API and NVTX trace), for the 16 N in ascending order. The reports are `nsys/profile-N<NNN>.nsys-rep`.
  3. `timing-N150/`: after an idle wait (55 s, 1-minute load 0.48), one complete run, checks then timing, at E19_CROPS=150 under nsys (`nsys/timing-N150.nsys-rep`). It was run for the spin's coverage only; its times are not used.

## The checks at every N (`analysis/checks-summary.txt`, `scripts-used/checks_summary.py`)

Every check row passed at every N, and every gated row passed. Every row's column `crops` holds the run's N, and every comparison covers 65,536 N values.

| N | gated rows passed | fkl_batch vs its reference | values left from the fill | fkl_loop vs its reference | opencv_cuda float vs own box | npp float vs own box | NPP grey rule, differing | OpenCV buffers moved |
|---|---|---|---|---|---|---|---|---|
| 1 | 12 of 12 | 2.31e-07 | 0 | 6.12e-07 | 1.19e-07 | 3.22e-07 | 0 | 0 of 3 |
| 10 | 12 of 12 | 2.62e-07 | 0 | 6.16e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 21 |
| 20 | 12 of 12 | 2.62e-07 | 0 | 6.37e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 41 |
| 30 | 12 of 12 | 2.62e-07 | 0 | 6.37e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 61 |
| 40 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 81 |
| 50 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 101 |
| 60 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 121 |
| 70 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 141 |
| 80 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 161 |
| 90 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.28e-07 | 3.49e-07 | 0 | 0 of 181 |
| 100 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 201 |
| 110 | 12 of 12 | 2.62e-07 | 0 | 7.01e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 221 |
| 120 | 12 of 12 | 2.62e-07 | 0 | 7.12e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 241 |
| 130 | 12 of 12 | 2.62e-07 | 0 | 7.12e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 261 |
| 140 | 12 of 12 | 2.62e-07 | 0 | 7.12e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 281 |
| 150 | 12 of 12 | 2.62e-07 | 0 | 7.12e-07 | 1.31e-07 | 3.49e-07 | 0 | 0 of 301 |

- At every N, NPP's grey images fit the rule found in E19, with 0 differing values. That rule is t = 0.587f G; t = fmaf(0.299f, R, t); t = fmaf(0.114f, B, t); (int)(t + 0.5f).
- OpenCV's grey images fit its fixed point with 0 differing. Both libraries' box images are floor(sum / 9) with 0 differing.
- The OpenCV buffer check compares 2N + 1 data pointers.
- fkl_batch's tensor differs from fkl_loop's by at most 7.75e-7 for N ≥ 10 and 5.36e-7 at N = 1 (reported only).
- **N = 30 against E19** (`analysis/n30-vs-e19.txt`): all 28 rows of E19's checks window (results/rtx4090/e19-batched-box-fkl-batch-checks) are equal in every value column at N = 30. The only new rows are the 4 between rows below.

## The between rows: fkl_batch against opencv_cuda and npp (reported only; `analysis/between.csv`)

"Differing" counts the values that are not equal. Grey levels are the output units times 57.375.

| N | values | vs opencv_cuda: differing | largest, output units | largest, grey levels | vs npp: differing | largest, output units | largest, grey levels |
|---|---|---|---|---|---|---|---|
| 1 | 65,536 | 65,536 (100.00 %) | 0.017429 | 1.00002 | 63,366 (96.69 %) | 0.015493 | 0.88889 |
| 10 | 655,360 | 655,360 (100.00 %) | 0.019366 | 1.11112 | 632,928 (96.58 %) | 0.015493 | 0.88889 |
| 20 | 1,310,720 | 1,310,719 (100.00 %) | 0.019366 | 1.11112 | 1,265,648 (96.56 %) | 0.015493 | 0.88889 |
| 30 | 1,966,080 | 1,966,079 (100.00 %) | 0.019366 | 1.11112 | 1,898,472 (96.56 %) | 0.015493 | 0.88889 |
| 40 | 2,621,440 | 2,621,438 (100.00 %) | 0.019366 | 1.11112 | 2,531,567 (96.57 %) | 0.015493 | 0.88889 |
| 50 | 3,276,800 | 3,276,798 (100.00 %) | 0.019366 | 1.11112 | 3,164,634 (96.58 %) | 0.015493 | 0.88889 |
| 60 | 3,932,160 | 3,932,158 (100.00 %) | 0.019366 | 1.11112 | 3,797,685 (96.58 %) | 0.015493 | 0.88889 |
| 70 | 4,587,520 | 4,587,518 (100.00 %) | 0.019366 | 1.11112 | 4,430,642 (96.58 %) | 0.015493 | 0.88889 |
| 80 | 5,242,880 | 5,242,878 (100.00 %) | 0.019366 | 1.11112 | 5,063,297 (96.57 %) | 0.015493 | 0.88889 |
| 90 | 5,898,240 | 5,898,238 (100.00 %) | 0.019366 | 1.11112 | 5,696,202 (96.57 %) | 0.015493 | 0.88889 |
| 100 | 6,553,600 | 6,553,597 (100.00 %) | 0.019366 | 1.11112 | 6,329,018 (96.57 %) | 0.015493 | 0.88889 |
| 110 | 7,208,960 | 7,208,956 (100.00 %) | 0.019366 | 1.11112 | 6,962,053 (96.57 %) | 0.015493 | 0.88889 |
| 120 | 7,864,320 | 7,864,316 (100.00 %) | 0.019366 | 1.11112 | 7,595,132 (96.58 %) | 0.015493 | 0.88889 |
| 130 | 8,519,680 | 8,519,676 (100.00 %) | 0.019366 | 1.11112 | 8,228,151 (96.58 %) | 0.015493 | 0.88889 |
| 140 | 9,175,040 | 9,175,034 (100.00 %) | 0.019366 | 1.11112 | 8,861,092 (96.58 %) | 0.015493 | 0.88889 |
| 150 | 9,830,400 | 9,830,393 (100.00 %) | 0.019366 | 1.11112 | 9,494,190 (96.58 %) | 0.015493 | 0.88889 |

## Kernels per call at every N (the profile runs; `analysis/kernel-counts.txt`, `analysis/kernel-counts.csv`, `scripts-used/kernel_counts_sweep.py`)

The counts are taken over 10 calls per row, each in the NVTX range "<row>:call". Every call has the expected number of kernels: fkl_loop N, fkl_batch 1, opencv_cuda 4N, npp 3N.
- fkl_batch's one kernel is morphologyDPPKernel, with a grid of 16 x 32 x N and blocks of 16 x 8 x 1.
- The kernel names per crop are E19's.
- In parentheses is the median host enqueue of one call under nsys, in us.

| N | fkl_loop kernels (enqueue, us) | fkl_batch kernels, grid (enqueue, us) | opencv_cuda kernels (enqueue, us) | npp kernels (enqueue, us) | all as expected |
|---|---|---|---|---|---|
| 1 | 1 (3.6) | 1, 16 x 32 x 1 (3.3) | 4 (15.3) | 3 (9.7) | yes |
| 10 | 10 (30.6) | 1, 16 x 32 x 10 (3.6) | 40 (135.0) | 30 (91.8) | yes |
| 20 | 20 (63.4) | 1, 16 x 32 x 20 (3.9) | 80 (267.6) | 60 (179.9) | yes |
| 30 | 30 (94.1) | 1, 16 x 32 x 30 (4.0) | 120 (405.2) | 90 (271.5) | yes |
| 40 | 40 (122.5) | 1, 16 x 32 x 40 (4.2) | 160 (531.0) | 120 (360.5) | yes |
| 50 | 50 (151.9) | 1, 16 x 32 x 50 (4.3) | 200 (666.5) | 150 (453.1) | yes |
| 60 | 60 (183.7) | 1, 16 x 32 x 60 (4.5) | 240 (798.1) | 180 (543.1) | yes |
| 70 | 70 (213.4) | 1, 16 x 32 x 70 (4.6) | 280 (929.6) | 210 (630.0) | yes |
| 80 | 80 (243.3) | 1, 16 x 32 x 80 (4.5) | 320 (1067.0) | 240 (723.1) | yes |
| 90 | 90 (275.2) | 1, 16 x 32 x 90 (4.8) | 360 (1200.8) | 270 (809.0) | yes |
| 100 | 100 (304.0) | 1, 16 x 32 x 100 (4.9) | 400 (1324.9) | 300 (899.5) | yes |
| 110 | 110 (335.2) | 1, 16 x 32 x 110 (5.1) | 440 (1473.1) | 330 (983.6) | yes |
| 120 | 120 (367.5) | 1, 16 x 32 x 120 (5.2) | 480 (1624.6) | 360 (1082.3) | yes |
| 130 | 130 (396.5) | 1, 16 x 32 x 130 (5.3) | 520 (1732.3) | 390 (1163.3) | yes |
| 140 | 140 (426.4) | 1, 16 x 32 x 140 (5.5) | 560 (1878.3) | 420 (1258.2) | yes |
| 150 | 150 (456.9) | 1, 16 x 32 x 150 (5.7) | 600 (2014.5) | 450 (1349.0) | yes |

## The spin's coverage at N = 150 (the run timing-N150; `analysis/spin-coverage-N150.txt`, `scripts-used/spin_coverage.py`)

In every row, the 5 ms spin covers the host's enqueue of all 100 kernel_only calls. A call is covered when the host's launch of its last kernel returns before the spin ends on the GPU.

| row | kernels per call | covered | smallest margin, us | median margin, us |
|---|---|---|---|---|
| fkl_loop | 150 | 100 of 100 | 4,499.6 | 4,554.7 |
| fkl_batch | 1 | 100 of 100 | 4,990.8 | 4,994.8 |
| opencv_cuda | 600 | 100 of 100 | 2,921.7 | 3,037.8 |
| npp | 450 | 100 of 100 | 3,570.7 | 3,677.1 |

- The smallest margin is 2,921.7 us, in opencv_cuda.
- On the GPU, the gap from the spin's end to a call's first kernel is at most 2.78 us. The gap between the kernels of a call is at most 1.28 us.
- The run passed its checks and wrote its timing CSV (`timing-N150/e19_batched_box.csv`). Those times are not used: they were taken under nsys.

## The SM clock (`analysis/clock-readings.txt`, `scripts-used/clock_readings.py`)

- In the checks and profile runs, the SM clock read 210 MHz at the start of checks/N001, then 2,550 MHz. It read 2,685 MHz once, in profile/N050.
- timing-N150 began at 2,550 MHz and read 2,685 MHz from 08:28:01.6 on, during its timing.
- No time from this window is used.

## The N = 1 kernel (an observation; no cause is claimed and nothing was changed)

At N = 1, fkl_batch's kernel compiles differently from N ≥ 10 (see `results/rtx4090/e19-crop-sweep-phase1/README.md`):
- 568 instructions, 35 registers and a 40-byte stack frame;
- a stack copy of the 40-byte BatchRead parameter (5 `STL.64`) and `LDL` of the crop's pointer;
- generic `LD.E.U8` loads of the pixels, where N ≥ 10 uses `LDG.E.U8`.

Its checks at N = 1 pass like every other N's.

## Files

| Path | Contents |
|---|---|
| `window-script.sh`, `window-console.txt` | the script that ran and its console |
| `environment-before.txt`, `environment-after.txt`, `gpu-info.csv` | the machine before and after the window |
| `gpu_clocks.csv`, `gpu_clocks.err` | the nvidia-smi sampler |
| `load.log` | the 5-second load log |
| `checks/N<NNN>/` | each checks run: stdout.txt, stderr.txt, exit_code.txt, load.txt, gpu-at-end.csv and e19_batched_box_checks.csv |
| `profile/N<NNN>/` | each profile run: its stdout and the other run files |
| `timing-N150/` | the N = 150 run under nsys: the run files, the checks CSV and the timing CSV |
| `nsys/` | the 17 nsys reports: 0.2-1.0 MiB for each profile, 14.6 MiB for timing-N150, 24 MiB in all |
| `analysis/` | the reports above |
| `scripts-used/` | the scripts that wrote them, run on the SQLite exports and the CSVs |

Left out: the SQLite exports of the 17 nsys reports (70 MiB, derived from the reports, made with `nsys export --type sqlite`). They are in the scratch folder.
