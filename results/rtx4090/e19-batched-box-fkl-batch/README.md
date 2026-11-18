# E19 batched box, phase 3: the timing window e19-fkl-batch-timing (RTX 4090, 2026-11-18 17:33:13-17:33:39 +01:00)

**The pipeline, per frame:** 30 crops of 256 x 256 of the 3840 x 2160 seeded frame. Each crop is converted RGB to grey, box filtered (3 x 3, REPLICATE at the crop's edges) and converted to float as (v / 255 - 0.45) / 0.225, into plane i of a 30 x 1 x 256 x 256 tensor. The program is `opencv-cuda/stencil/e19_batched_box.cu`.

**The rows**, in this order in every run:

| row | calls per frame | kernels per frame |
|---|---|--:|
| `fkl_loop` | FKL at 24478f0: one `fk::executeLinearFilter` per crop | 30 |
| `fkl_batch` | FKL at 24478f0: one `fk::executeBoxBlur<1>` per frame (MorphologyDPP; a BatchRead of the 30 crop views, grid z = 30) | 1 |
| `opencv_cuda` | `cvtColor`, `box->apply`, `convertTo` per crop | 120 |
| `npp` | `nppiRGBToGray_8u_C3C1R_Ctx`, `nppiFilterBoxBorder_8u_C1R_Ctx`, `nppiScale_8u32f_C1R_Ctx` per crop | 90 |

**The binary:** 8ff14e5, the build of 17:24 (`../e19-batched-box-fkl-batch-checks/build.log`: 25 warnings, all of them CUDA 13 deprecations inside FKL's headers; the build info is 8ff14e5 without -dirty). sha256 `e989a41e…` (`binary-sha256.txt`). It is the binary of the window e19-fkl-batch-checks. HEAD (00d4d85) descends from 8ff14e5, and 8ff14e5..HEAD changes only `results/rtx4090/e19-batched-box-fkl-batch-checks/`.

**The window** (`window-script.sh`, sha256 `44a834b3…`, and `window-console.txt`). It is the script of the window e19-timing (`../e19-batched-box/`) with the protocol unchanged; only the paths, the commit, the guards' source paths and a check of the output folder's name differ. The script stops at a failed run or a failed idle wait; neither happened.
1. **warmup/**: one complete run (the checks, then the timing), not used.
2. **quiet-run1..3/**: each after `wait_idle`, which passed after 0 s at 1-minute loads of 0.29-0.35. Each is one complete run pinned to cores 0-3,16-19.
3. **nsys/**: its own step, one complete run under the CUDA API and NVTX trace (`nsys/e19-fkl-batch-timing.nsys-rep`).

The sampler read nvidia-smi every 100 ms throughout (`gpu_clocks.csv`). There was no ncu pass, no dump and no clock warm-up.

**Every run** exited 0 and reported "E19 batched box checks passed". Its 28 check rows are identical to those of the window e19-fkl-batch-checks (`checks-comparison.txt`).

**Per row and run, two intervals over one whole frame:**
- **whole_call:** `fklb::timeIt`, 100 calls after one warm-up call.
- **kernel_only:** `fklb::timePreparedLaunch` behind one 5 ms `gpuSpin` per call, with nothing to prepare.

Left out of this record: the SQLite export of the nsys report (9.3 MiB, on the measuring machine in `scratch/e19-fkl-batch/nsys-export-timing/`).

## The times (`e19-report.txt`, `e19_batched_box_summary.csv`, `scripts-used/e19_report.py`)
The median and the min-max of the 3 quiet-run means, in µs per frame:

| row | whole_call | kernel_only |
|---|--:|--:|
| `fkl_loop` | 74.32 (74.23-74.34) | 73.61 (73.58-73.86) |
| `fkl_batch` | 22.11 (22.07-22.11) | 21.83 (21.83-21.85) |
| `opencv_cuda` | 239.12 (238.54-240.86) | 237.64 (237.63-237.70) |
| `npp` | 214.17 (214.15-214.84) | 212.17 (212.14-212.41) |

The ratios, each the median and the min-max of the 3 per-run ratios:

| ratio | whole_call | kernel_only |
|---|--:|--:|
| `fkl_batch` / `fkl_loop` | 0.297 (0.297-0.298) | 0.297 (0.296-0.297) |
| `opencv_cuda` / `fkl_loop` | 3.218 (3.213-3.240) | 3.229 (3.217-3.230) |
| `npp` / `fkl_loop` | 2.885 (2.882-2.890) | 2.882 (2.872-2.887) |
| `fkl_loop` / `fkl_batch` | 3.362 (3.361-3.363) | 3.373 (3.368-3.383) |
| `opencv_cuda` / `fkl_batch` | 10.813 (10.808-10.894) | 10.884 (10.877-10.890) |
| `npp` / `fkl_batch` | 9.704 (9.684-9.717) | 9.721 (9.716-9.723) |

- **The run-to-run spread** ((max - min) / median): at most 0.97 %, for `opencv_cuda` whole_call. The others are 0.03-0.39 %.
- **whole_call against kernel_only:** whole_call is 0.6-1.3 % longer in every row (`e19-report.txt`), so the host's enqueue does not set whole_call in the quiet runs. Every interval is GPU-bound.
- **The warmup and nsys runs** are not used. Their means are listed in `e19-report.txt`; the nsys run's are longer, because of the trace.

## The repeats of the old rows, against the window e19-timing (`../e19-batched-box/`, 10:17Z)
Each line gives the old median, the new median, the change, the spread of both windows, and whether the change exceeds the larger of the two spreads. All six intervals are GPU-bound, and a GPU-bound change is flagged above 3 %.

| row, interval | old (µs) | new (µs) | change | spread old / new | exceeds the spread? | flag (above 3 %) |
|---|--:|--:|--:|--:|---|---|
| `fkl_loop` whole_call | 74.29 | 74.32 | +0.03 % | 0.22 % / 0.14 % | no, within the spread | no |
| `opencv_cuda` whole_call | 239.48 | 239.12 | -0.15 % | 0.21 % / 0.97 % | no, within the spread | no |
| `npp` whole_call | 214.84 | 214.17 | -0.31 % | 1.43 % / 0.32 % | no, within the spread | no |
| `fkl_loop` kernel_only | 73.57 | 73.61 | +0.06 % | 0.46 % / 0.39 % | no, within the spread | no |
| `opencv_cuda` kernel_only | 237.64 | 237.64 | +0.00 % | 0.11 % / 0.03 % | no, within the spread | no |
| `npp` kernel_only | 212.10 | 212.17 | +0.03 % | 0.11 % / 0.13 % | no, within the spread | no |

- In every line the new 3-run range overlaps the old one.
- The quiet runs of both windows read 2,685 MHz throughout (below), so no repeat is flagged for the clock.

## The nsys run (`nsys/spin-coverage.txt`, `nsys/kernel-time-by-interval.txt`, `nsys/kernel-durations.txt`)
- **Spin coverage:** 100 of 100 timed calls in every row.
  - The margins (spin end minus the return of the last launch), min / median: `fkl_loop` 4,900.3 / 4,909.8 µs; `fkl_batch` 4,987.7 / 4,996.5 µs; `opencv_cuda` 4,568.9 / 4,602.8 µs; `npp` 4,465.6 / 4,734.2 µs. The smallest margin is 4,465.6 µs (`npp`).
  - The GPU gap from the spin's end to the first kernel is at most 5.89 µs (`fkl_loop`); for `fkl_batch` it is at most 2.75 µs.
- **The kernels per call:** 30, 1, 120 and 90. `fkl_batch`'s one kernel is `morphologyDPPKernel`, grid 16 x 32 x 30, block 16 x 8 (`../e19-batched-box-fkl-batch-checks/fkl-batch-grid.txt`). No texture, array or surface API call appears in the trace.
- **The kernel durations,** median over the 100 timed calls, under the trace. Each cell is the sum of the kernels of one frame / the first-kernel-start to last-kernel-end span:

| row | whole_call | kernel_only |
|---|--:|--:|
| `fkl_loop` | 62.40 / 92.64 µs | 59.74 / 79.52 µs |
| `fkl_batch` | 20.35 / 20.35 µs | 21.09 / 21.09 µs |
| `opencv_cuda` | 169.54 / 398.95 µs | 169.59 / 260.29 µs |
| `npp` | 169.19 / 266.63 µs | 169.12 / 229.09 µs |

- **`fkl_loop`'s values mix two clocks;** see below.
- **`fkl_batch`'s kernel** is 21.09 µs (20.38-21.22) in kernel_only, where each call follows a 5 ms spin. In whole_call, where the calls run back to back, it is 20.35 µs (20.19-21.12). That is 3.6 % longer after a spin. kernel_only's first call, the warm-up without a spin, takes 20.38 µs. The cause is not known; this record does not name one.

## The SM clock (`clocks/`; `scripts-used/clock_readings.py`, `clock_by_row.py`, `clock_step.py`)
nvidia-smi updates its values about every 0.5 s. A reading at time t states the clock at some moment of (t - 0.5 s, t].
- **The quiet runs** (17:33:16-17:33:25): every reading is 2,685 MHz. There is no 2,550 MHz reading in any of them, so no interval of a quiet run started at 2,550 MHz. Each run ended at 2,685 MHz (`gpu-at-end.csv`).
  - The quiet runs have no per-row timestamps: the per-row statement rests on the readings of the whole run.
- **FLAG, warmup (not used):** 210 MHz at 17:33:13.607, then 2,550 MHz from 17:33:14.414 to 17:33:14.814 (5 readings, at 0 % utilisation), then 2,685 MHz from 17:33:14.914 (at 100 %). Its `fkl_loop` whole_call mean is 77.91 µs, 4.8 % above the quiet-run median.
- **FLAG, the nsys run:** 2,550 MHz from 17:33:27.932 to 17:33:30.836 (30 readings), then 2,685 MHz from 17:33:30.936.
  - The trace maps every `whole_call:fkl_loop` kernel to 2,550 MHz readings only. Of the `kernel_only:fkl_loop` kernels, 960 map to 2,550 only and 2,070 to readings of both clocks.
  - Every `fkl_batch` kernel, and every `whole_call:opencv_cuda` kernel, maps to readings of both clocks. The `kernel_only:opencv_cuda` kernels map to 2,685 only (12,000) or to both (120). Every `npp` kernel maps to 2,685 only.
  - The kernel durations show where `fkl_loop`'s duration changed:
    - `whole_call:fkl_loop` (17:33:30.265-17:33:30.276): all 101 calls have a median kernel duration of 2.08-2.21 µs.
    - `kernel_only:fkl_loop` (17:33:30.276-17:33:30.786): calls 0-47 have medians of 2.08 µs. From call 48, at 17:33:30.521, they have 1.98 µs.
    - 2.08 / 1.98 = 1.051, against 2,685 / 2,550 = 1.053. The step falls within the 0.5 s before the first 2,685 MHz reading (17:33:30.936). It is the same step as in the nsys run of the window e19-timing (call 48, 2.08 to 1.98 µs).
  - `fkl_batch`'s ranges (17:33:30.786-17:33:31.293) come after that step. Its whole_call kernel durations are 20.19-20.67 µs from call 1 on.
  - The nsys run ended at 570 MHz (`nsys/gpu-at-end.csv`), after the trace.
