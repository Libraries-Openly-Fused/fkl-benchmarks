# E19 batched box: the timing window e19-timing (RTX 4090, 2026-11-18 11:17:13-11:17:34 +01:00)

> **Superseded (dated note, 2026-11-27).** These times were measured at fkl-benchmarks `237074f` (CSV repo `69df252`) with FKL `24478f0`. They were not timed again at the later pins b10fc66 (#60) or 304492a (#66). The paper takes its 30-crop values from the E19 crop sweep, which was timed again at FKL 304492a in `../e19-crop-sweep-retime-304492a/` (#70, 8c419c7). Oscar's decision, 2026-11-26.

**The pipeline, per frame:** 30 crops of 256 x 256 of the 3840 x 2160 seeded frame. Each crop is converted RGB to grey, box filtered (3 x 3, REPLICATE at the crop's edges), and converted to float as (v / 255 - 0.45) / 0.225, into plane i of a 30 x 1 x 256 x 256 tensor. The program is `opencv-cuda/stencil/e19_batched_box.cu`.

**The rows**, in this order in every run:

| row | calls per frame | kernels per frame |
|---|---|--:|
| `fkl_loop` | FKL at 24478f0: one `fk::executeLinearFilter` per crop | 30 |
| `opencv_cuda` | `cvtColor`, `box->apply`, `convertTo` per crop | 120 |
| `npp` | `nppiRGBToGray_8u_C3C1R_Ctx`, `nppiFilterBoxBorder_8u_C1R_Ctx`, `nppiScale_8u32f_C1R_Ctx` per crop | 90 |

**The binary:** 69df252, the build of 11:12 (`../e19-batched-box-fkl-recheck/build.log`: 25 warnings, all of them CUDA 13 deprecations inside FKL's headers; the build info is 69df252 without -dirty). sha256 `2aa1486c…` (`binary-sha256.txt`). It is the binary of the window e19-fkl-recheck. HEAD descends from 69df252 with no source change since.

**The window** (`window-script.sh`, `window-console.txt`). The script stops at a failed run or a failed idle wait; neither happened.
1. **warmup/**: one complete run (the checks, then the timing), not used.
2. **quiet-run1..3/**: each after `wait_idle`, which passed after 0 s at 1-minute loads of 0.21-0.28. Each is one complete run pinned to cores 0-3,16-19.
3. **nsys/**: its own step, one complete run under the CUDA API and NVTX trace (`nsys/e19-timing.nsys-rep`).

The sampler read nvidia-smi every 100 ms throughout (`gpu_clocks.csv`). There was no ncu pass, no dump and no clock warm-up.

**Every run** exited 0 and reported "E19 batched box checks passed". Its 25 check rows are identical to those of the window e19-fkl-recheck.

**Per row and run, two intervals over one whole frame:**
- **whole_call:** `fklb::timeIt`, 100 calls after one warm-up call.
- **kernel_only:** `fklb::timePreparedLaunch` behind one 5 ms `gpuSpin` per call, with nothing to prepare.

Left out of this record: the SQLite export of the nsys report (9.2 MiB, on the measuring machine in `scratch/e19-batched-box/nsys-export-timing/`).

## The times (`e19-report.txt`, `e19_batched_box_summary.csv`, `scripts-used/e19_report.py`)
The median and the min-max of the 3 quiet-run means, in µs per frame:

| row | whole_call | kernel_only |
|---|--:|--:|
| `fkl_loop` | 74.29 (74.22-74.38) | 73.57 (73.52-73.86) |
| `opencv_cuda` | 239.48 (239.14-239.66) | 237.64 (237.64-237.91) |
| `npp` | 214.84 (213.92-216.98) | 212.10 (212.04-212.28) |

The ratios, each the median and the min-max of the 3 per-run ratios:

| ratio | whole_call | kernel_only |
|---|--:|--:|
| `opencv_cuda` / `fkl_loop` | 3.222 (3.219-3.227) | 3.232 (3.217-3.234) |
| `npp` / `fkl_loop` | 2.895 (2.879-2.917) | 2.882 (2.872-2.887) |

- **The run-to-run spread** ((max - min) / median): at most 1.43 %, for `npp` whole_call. The others are 0.11-0.46 %.
- **The warmup and nsys runs** are not used. Their means are listed in `e19-report.txt`; the nsys run's are longer, because of the trace.

## The nsys run (`nsys/spin-coverage.txt`, `nsys/kernel-time-by-interval.txt`)
- **Spin coverage:** 100 of 100 timed calls in every row.
  - The margins (spin end minus the return of the last launch), min / median: `fkl_loop` 4,901.7 / 4,910.0 µs; `opencv_cuda` 4,546.5 / 4,605.7 µs; `npp` 4,706.8 / 4,736.5 µs. The smallest margin is 4,546.5 µs (`opencv_cuda`).
  - The GPU gap from the spin's end to the first kernel is at most 2.53 µs.
- **The kernels per call:** 30, 120 and 90, as in the profile runs. No texture, array or surface API call appears in the trace.
- **The kernel durations,** median over the 100 timed calls, under the trace. Each cell is the sum of the kernels of one frame / the first-kernel-start to last-kernel-end span:

| row | whole_call | kernel_only |
|---|--:|--:|
| `fkl_loop` | 62.53 / 91.34 µs | 59.78 / 79.54 µs |
| `opencv_cuda` | 169.70 / 397.88 µs | 169.60 / 260.27 µs |
| `npp` | 169.09 / 267.71 µs | 169.12 / 229.12 µs |

- The values of `fkl_loop` mix two clocks; see below.

## The SM clock (`clocks/`; `scripts-used/clock_readings.py`, `clock_by_row.py`, `clock_step.py`)
nvidia-smi updates its values about every 0.5 s. A reading at time t states the clock at some moment of (t - 0.5 s, t].
- **The quiet runs** (11:17:15-11:17:23): every reading is 2,685 MHz. There is no 2,550 MHz reading in any of them, so no interval of a quiet run started at 2,550 MHz.
  - The quiet runs have no per-row timestamps: the per-row statement rests on the readings of the whole run.
- **FLAG, warmup (not used):** 2,550 MHz from 11:17:14.209 to 11:17:14.609 (5 readings at 100 % utilisation), then 2,685 MHz from 11:17:14.709. Its first row, `fkl_loop`, very likely started at 2,550 MHz. Its `fkl_loop` whole_call mean is 78.05 µs, 5.1 % above the quiet-run median.
- **FLAG, the nsys run:** 2,550 MHz from 11:17:25.624 to 11:17:28.027 (25 readings), then 2,685 MHz.
  - The trace maps every `fkl_loop` kernel to readings within 0.5 s of both clocks. Every `opencv_cuda` whole_call kernel maps the same way, and so do 7,080 of the 12,120 `opencv_cuda` kernel_only kernels; the other 5,040 map to 2,685 only.
  - Every `npp` kernel maps to 2,685 only.
  - The kernel durations show where the clock changed:
    - `whole_call:fkl_loop` (11:17:27.653-11:17:27.664): all 101 calls have a median kernel duration of 2.08-2.21 µs.
    - `kernel_only:fkl_loop` (11:17:27.664-11:17:28.174): calls 0-47 have medians of 2.08 µs. From call 48, at 11:17:27.908, they have 1.98 µs.
    - 2.08 / 1.98 = 1.051, against 2,685 / 2,550 = 1.053.
  - So in the nsys run, `fkl_loop`'s whole_call and its kernel_only interval started at 2,550 MHz, and the clock went to 2,685 MHz during its kernel_only interval.
  - The `opencv_cuda` and `npp` intervals came after 11:17:28.174. Their kernel durations are constant at the 2,685 MHz values.
