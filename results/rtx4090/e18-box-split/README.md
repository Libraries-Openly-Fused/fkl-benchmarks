# E18 box split, the window: the pipeline's resize and box each alone, in FKL and in NPP (RTX 4090, CUDA 13.4.3)

This is the timing of `e18_box_split_cu` (`opencv-cuda/stencil/e18_box_split.cu`): the window e18-box-split, 2026-11-17 16:32:59-16:33:35 (+01:00). The rows, the checks and the kernels are described in `../e18-box-split-phase1/README.md`.

**The question.** In the window e18-box (`../e18-box/`), `npp` (two calls) took less time than `fkl_trunc_halfpixel` (one fused kernel) for the same output: 0.85x in the whole call. Is that FKL's box filter kernel itself, or the fusion? The fused kernel computes the resized value of every position of each tile's 18 x 10 halo. The split times each step alone in both libraries.

The numbers below are measurements. Why a kernel takes the time it takes is not examined here.

## Results
- **The box alone** (the same 1920 x 1080 input, the same output floor(sum / 9)): `npp_box` takes 0.739x the time of `fkl_box` (whole call, min-max 0.738-0.739). Their kernels take 21.34 µs and 29.44 µs.
- **The resize alone** (the same crop, HALF_PIXEL): `npp_resize` takes 0.839x the time of `fkl_resize` (0.834-0.845). Their kernels take 10.88 µs and 13.28 µs.
- **The pipeline:** `npp` takes 0.846x the time of `fkl_trunc_halfpixel` (0.846-0.848). In the window e18-box it was 0.85x.
- **The fusion:** FKL's two steps alone take 1.166x the time of the fused call (whole call, 1.165-1.166).
  - In kernels, `fkl_resize`'s kernel plus `fkl_box`'s kernel take 42.72 µs, against 37.98 µs for the fused kernel (1.125x).
  - NPP's two calls alone take 1.061x the `npp` row (whole call). Their kernels take 32.23 µs against `npp`'s two kernels at 32.00 µs (1.007x).

**The kernels in one table** (nsys, the whole-call interval, the median of 100 calls):

| | FKL | NPP | FKL - NPP |
|---|--:|--:|--:|
| the resize alone | 13.28 µs | 10.88 µs | +2.40 µs |
| the box alone | 29.44 µs | 21.34 µs | +8.10 µs |
| the two alone, added | 42.72 µs | 32.23 µs | +10.49 µs |
| the pipeline (fused kernel; NPP's two kernels) | 37.98 µs | 32.00 µs | +5.98 µs |
| the pipeline - the two alone | -4.74 µs | -0.23 µs | |

## How the window ran
- **The binary:** step 0's build at 417a106 (`step0/`), sha256 `7f5ef327…`. It was compiled before the window, with SASS identical to the phase-1 binary's.
- **The steps:**
  1. **`warmup/`:** one complete run (the checks and the timing), not used. 5 s.
  2. **`quiet-run1/` to `quiet-run3/`:** three complete runs, each after an idle check (1-minute load < 0.5, at most 300 s). Each was idle at once (load 0.30, 0.36, 0.41). 5 s each.
  3. **`nsys/`:** one complete run under nsys (`--trace=cuda,nvtx --sample=none --cpuctxsw=none`), its own step after the timed runs. 11 s.
  - No ncu pass, and no clock warm-up inside the program.
- **The setup:**
  - Every run was pinned to cores 0-3,16-19 with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
  - The load logger and the nvidia-smi sampler (`gpu_clocks.csv`, every 100 ms) ran throughout.
  - The blocked-programs guard checked the binary before the start.
- **The script:** `window-script.sh` is the window e18-box's script with these changes only: the paths, the commit, the program, the check string, and no ncu pass.
- **Every run** exited with 0 and printed "E18 box split checks passed". Its agreement and check CSVs equal phase 1's in every field, its stderr is empty, and its CSVs hold no INCOMPLETE and no -nan.
- **The clocks, by segment** (`gpu_clocks.csv`; `clocks/`):
  - **The three quiet runs** read 2,685 MHz SM and 10,251 MHz memory (P2) throughout, at 45-48 °C.
  - **The nsys run** read 2,550 MHz from 16:33:21.5 to 16:33:25.9. Its timing began inside that segment, so `fkl_box_asym`'s kernels were measured partly at 2,550 MHz (see "The SM clock and the first row" below).
  - **The warm-up run (not used)** read 2,550 MHz from 16:33:00.1 to 16:33:01.5, at its start.

## The times (CUDA events; the median of the three runs' means, each of 100 calls)
From `e18_box_split_summary.csv` and `e18_box_split_report.md` (`scripts-used/e18_box_split_report.py`).

| # | row | whole_call (µs) | min-max | kernel_only (µs) | min-max | host preparation (µs) |
|--:|---|--:|--:|--:|--:|--:|
| 1 | `fkl_box_asym` | 40.031 | 39.951-40.052 | 41.019 | 40.991-41.029 | 0.143 |
| 2 | `fkl_trunc_halfpixel` | 39.195 | 39.159-39.234 | 40.134 | 40.129-40.191 | 0.146 |
| 3 | `npp` | 33.190 | 33.172-33.196 | 33.929 | 33.864-33.970 | 0.039 |
| 4 | `npp_resize` | 12.214 | 12.206-12.278 | 12.291 | 12.288-12.300 | 0.039 |
| 5 | `fkl_resize` | 14.561 | 14.529-14.644 | 14.414 | 14.394-14.493 | 0.105 |
| 6 | `npp_box` | 22.996 | 22.966-23.007 | 23.552 | 23.552-23.552 | 0.038 |
| 7 | `fkl_box` | 31.141 | 31.060-31.142 | 31.756 | 31.751-31.772 | 0.056 |

- **Spread:** the largest is 0.8 % (`fkl_resize`, whole_call).
- **Host preparation:** the kernel_only interval's host time (the view, the IOps and the details; the ROI pointer; nothing for `npp_box`).
- **Which rows are GPU-bound:** all of them. The host preparation is 0.04-0.15 µs, against 12-41 µs of GPU time.

**The ratios** (`e18_box_split_ratios.csv`): the time of the first row over the time of the second, as the median of the three per-run ratios.

| ratio | whole_call | min-max | kernel_only | min-max |
|---|--:|--:|--:|--:|
| `npp_box` / `fkl_box` | 0.739 | 0.738-0.739 | 0.742 | 0.741-0.742 |
| `npp_resize` / `fkl_resize` | 0.839 | 0.834-0.845 | 0.853 | 0.848-0.854 |
| `npp` / `fkl_trunc_halfpixel` | 0.846 | 0.846-0.848 | 0.844 | 0.844-0.847 |
| (`fkl_resize` + `fkl_box`) / `fkl_trunc_halfpixel` | 1.166 | 1.165-1.166 | 1.150 | 1.149-1.153 |
| (`npp_resize` + `npp_box`) / `npp` | 1.061 | 1.060-1.063 | 1.057 | 1.055-1.058 |

## The kernels (nsys)
From `nsys/kernel-durations.txt` and `.csv` (`scripts-used/kernel_durations.py`). Each value is the GPU duration of the call's kernel: the median of the 100 timed calls, with their min-max.
- whole_call leaves out `fklb::timeIt`'s warm-up call.
- kernel_only takes the calls behind the spin.

| row | kernels per call | kernel | whole_call (µs) | kernel_only (µs) |
|---|--:|---|--:|--:|
| `fkl_box_asym` | 1 | `linearFilterDPPKernel` | 40.77 (40.51-41.22) | 40.45 (40.29-42.85) |
| `fkl_trunc_halfpixel` | 1 | `linearFilterDPPKernel` (the fused kernel) | 37.98 (37.76-38.24) | 39.39 (39.26-39.84) |
| `npp` | 2 | `resizeNaive` | 10.82 (10.72-10.91) | 11.26 (11.23-11.36) |
| | | `ForEachPixelNaiveInLargeImage` (the box) | 21.18 (21.09-21.34) | 21.73 (21.60-21.82) |
| `npp_resize` | 1 | `resizeNaive` | 10.88 (10.78-10.95) | 11.26 (11.20-11.30) |
| `fkl_resize` | 1 | `launchTransformDPP_Kernel` | 13.28 (13.25-13.34) | 13.76 (13.73-13.82) |
| `npp_box` | 1 | `ForEachPixelNaiveInLargeImage` | 21.34 (21.25-21.50) | 21.98 (21.92-22.11) |
| `fkl_box` | 1 | `linearFilterDPPKernel` | 29.44 (29.25-29.82) | 30.40 (30.34-30.59) |

- **Sums of the medians** (kernel_only in brackets):
  - `fkl_resize` + `fkl_box` = 42.72 µs (44.16) against the fused kernel's 37.98 µs (39.39): 1.125x (1.121).
  - `npp_resize` + `npp_box` = 32.23 µs (33.25) against `npp`'s two kernels, 32.00 µs (32.99): 1.007x (1.008).
- **Kernel ratios, whole_call (kernel_only):** `fkl_box` / `npp_box` 1.379 (1.383); `fkl_resize` / `npp_resize` 1.221 (1.222); the fused kernel / `npp`'s two kernels 1.187 (1.194).
- **Kernels per call:** rows 4-7 launch one kernel per call, `npp` two, and the FKL pipeline rows one (`nsys/spin-coverage.txt`).
- **Other API calls:** no texture, array or surface API call in the whole trace.
- **Spin coverage** (`nsys/spin-coverage.txt`, #43's `spin_coverage.py`, unchanged):
  - every row: 100 of 100 calls covered;
  - the smallest margin (spin end - the return of the call's last launch): 4,988.5 µs (`npp`);
  - the GPU gap from the spin's end to the first kernel: at most 2.59 µs.
- **`nsys/kernel-time-by-interval.txt`** is #48's `kernel_time_by_interval.py`, unchanged: the per-call sums and spans, in the format of `../e18-box/`.

## The repeated rows under the rule for repeats
**Old** is the window e18-box (2026-11-17 12:09 +01:00, `../e18-box/e18_box_summary.csv`, binary of 7dca510); **new** is this window. The kernels of the two binaries are SASS-identical.
- **Spread:** max - min of the three runs' means.
- **"Exceeds the spread":** |new - old| is larger than the larger of the old and the new spread.
- **All three rows are GPU-bound,** so the flag threshold is 3 %.

From `e18_box_split_repeats.csv`:

| row | interval | old (µs) | new (µs) | old / new spread (µs) | change | exceeds the spread | above 3 % |
|---|---|--:|--:|--:|--:|---|---|
| `fkl_box_asym` | whole_call | 42.006 | 40.031 | 0.119 / 0.101 | -1.975 µs (-4.7 %) | yes | **yes** |
| `fkl_box_asym` | kernel_only | 41.580 | 41.019 | 0.522 / 0.038 | -0.561 µs (-1.3 %) | yes | no |
| `fkl_trunc_halfpixel` | whole_call | 39.595 | 39.195 | 0.084 / 0.075 | -0.400 µs (-1.0 %) | yes | no |
| `fkl_trunc_halfpixel` | kernel_only | 40.950 | 40.134 | 0.031 / 0.061 | -0.816 µs (-2.0 %) | yes | no |
| `npp` | whole_call | 33.567 | 33.190 | 0.040 / 0.024 | -0.377 µs (-1.1 %) | yes | no |
| `npp` | kernel_only | 34.080 | 33.929 | 0.058 / 0.106 | -0.151 µs (-0.4 %) | yes | no |

- **Every repeat is lower than before,** by 0.4-4.7 %. Every change exceeds the spreads, which are 0.02-0.52 µs.
- **One change is above 3 %:** `fkl_box_asym`'s whole call, -4.7 %. Its cause is not determined.
  - It is the first row of every run, and the row of the window e18-box's observation 2: its whole-call kernel was 5 % slower than the same kernel in `fkl_asym_once`.
  - In that window, its kernel_only / whole_call ratio was 0.991 (`../e18-box/e18_box_kernel_vs_whole.csv`). In the three quiet runs here it is 1.024 (1.024-1.027), the same as `fkl_trunc_halfpixel`'s 1.024 (`e18_box_split_kernel_vs_whole.csv`).
  - Under nsys, here as there, its whole-call kernel is the longer of its two: 40.77 µs against 40.45 µs (ratio 0.992). The other rows' kernel ratios, kernel_only over whole_call, are 1.030-1.037. In both nsys runs, its whole-call kernels ran at 2,550 MHz (next section).
  - **A candidate, not a cause: the SM clock** (next section). In the window e18-box, every quiet run's timing began right after 2,550 MHz readings, and in quiet-run2 they continued into it. `fkl_box_asym`'s whole call is the timing's first interval. Here the quiet runs read 2,685 MHz throughout. 2,685 / 2,550 = 1.053; old / new for this interval is 1.049.
- **The kernels under nsys, old against new** (one nsys run each; the whole-call medians, then the kernel-only medians, from both windows' `kernel-time-by-interval.txt`):
  - `fkl_box_asym`: 41.09 against 40.77 µs (-0.8 %), and 40.51 against 40.45 µs;
  - `fkl_trunc_halfpixel`: 38.05 against 37.98 µs, and 39.42 against 39.39 µs;
  - `npp`'s two kernels: 32.03 against 32.00 µs, and 32.96 against 32.99 µs.
  - Each of these changes is 0.8 % or less. A single nsys run has no run-to-run spread; the min-max over its 100 calls is in the kernel table above.
- **The pipeline ratio** `npp` / `fkl_trunc_halfpixel` is 0.846 here against 0.848 there in the whole call, and 0.844 against 0.832 in kernel_only (`../e18-box/e18_box_speedups.csv`).

## The SM clock and the first row (both windows)
Files: `clocks/`, from `scripts-used/clock_readings.py` and `clock_by_row.py`; the second also reads the nsys traces exported to SQLite.

**What the readings can resolve:**
- **The sampler's resolution.** nvidia-smi was sampled every 100 ms, but it repeats a reading until the driver updates it, about every 0.5 s. A reading at time t therefore states the clock at some moment of the 0.5 s before t. `utilization.gpu` is the share of time a kernel ran before that moment.
- **The quiet runs log no per-row times.** What bounds them:
  - each run's start and end, to the second (`load.txt`);
  - the program's fixed order: the checks, whose last part is host work with the GPU idle, then the timing, whose first interval is `fkl_box_asym`'s whole call (about 5 ms), then its kernel_only (about 0.5 s, 100 spins of 5 ms);
  - the utilisation step from 0 to about 100 % that the spins cause.
- **This window's CSVs** still carry their creation times (`clocks/csv-creation-times.txt`). `e18_box_split.csv` is created when the timing begins, which gives a lower bound. The window e18-box's files exist only as committed copies, so those times are lost.
- **The nsys runs** put every kernel on the wall clock: the trace's session start plus the kernel's GPU time.

**(a) The quiet runs of the window e18-box** (`clocks/clock-readings-window-e18-box.txt`). In each run, the 2,550 MHz readings fall at the end of the checks, with the GPU idle, directly before the timing:

| run | readings at 2,550 MHz | utilisation in them | the next reading |
|---|---|---|---|
| quiet-run1 | 12:09:24.233-24.634 | 0 % | 24.734: 2,685 MHz, 100 % (the timing has begun) |
| quiet-run2 | 12:09:31.241-31.641 and 31.742-32.142 | 0 %, then **95 %** | 32.242: 2,685 MHz, 100 % |
| quiet-run3 | 12:09:38.749-39.150 | 0 % | 39.250: 2,685 MHz, 100 % |

- **quiet-run2:** the reading of 95 % at 2,550 MHz states the clock at a moment after the timing had begun. nvidia-smi's utilisation sample period is 1/6 s to 1 s, so the 95 % covers at least 0.16 s before that moment. So the timing began at 2,550 MHz, unless the clock rose and fell again within those 0.5 s: `fkl_box_asym`'s whole call, the first 5 ms, and at least the first part of its kernel_only.
- **quiet-run1 and quiet-run3:** the clock changed from 2,550 to 2,685 MHz within the same 0.5 s as the timing's start. The sampler cannot show at which clock the 5 ms whole call ran.
- **The next row,** `fkl_box_halfpixel`, starts about 0.5 s after the timing:
  - in quiet-run1 and quiet-run3, a 2,685 MHz reading falls before it can start;
  - in quiet-run2, the readings do not exclude 2,550 MHz at its start. Its times in quiet-run2 (whole call 40.602 µs, kernel_only 42.001 µs) equal those of the other two runs (40.629 and 40.604 µs; 41.991 and 41.990 µs).
- **The per-run times:**
  - `fkl_box_asym`'s kernel_only is 42.065 µs in quiet-run2 against 41.580 and 41.543 µs in the others. It is that row's only outlier, +1.2 %.
  - Its whole call is 41.970, 42.089 and 42.006 µs, 4.8-5.1 % above this window's 40.031 µs.

**(b) The nsys runs: which kernels ran at 2,550 MHz** (`clocks/clock-by-row-nsys-*.txt`).
- **The two counts:**
  - by the readings: a kernel is at one clock when every reading within 0.5 s of its start shows that clock;
  - by the durations: the medians of each group of 10 kernels in launch order.
- **This window** (the readings at 2,550 MHz from 16:33:21.520 to 25.925; the timing began at 16:33:25.419):
  - `whole_call:fkl_box_asym`, 16:33:25.419-25.424: all 101 kernels at 2,550 MHz by the readings. Their durations are flat, 40.74-40.80 µs.
  - `kernel_only:fkl_box_asym`, 25.424-25.929: by the readings, 20 kernels at 2,550 MHz and 81 within 0.5 s of the change. By the durations, the first 30 take 42.66-42.69 µs and the rest 40.38-40.46 µs.
  - `fkl_trunc_halfpixel`, 25.930-26.439: within 0.5 s of the change by the readings, but its durations are flat (whole call 37.92-38.02 µs, kernel_only 39.33-39.52 µs), so it ran at one clock. Its whole-call kernel median, 37.98 µs, is within 0.2 % of the window e18-box's 38.05 µs, which ran at 2,685 MHz.
  - Every other row: 2,685 MHz.
- **The window e18-box** (2,550 MHz from 12:09:46.160 to 52.069, then 2,685 from 52.169; the timing began at 12:09:51.877):
  - `whole_call:fkl_box_asym`, 51.877-51.882: flat, 40.91-41.15 µs (median 41.09 µs). The same kernel in `fkl_asym_once`'s whole call, at 2,685 MHz, takes 38.97 µs. 41.09 / 38.97 = 1.054, and 2,685 / 2,550 = 1.053.
  - `kernel_only:fkl_box_asym`: the first 40 kernels take 42.96-43.01 µs, the rest 40.35-40.51 µs. The change falls at about 52.08, between the last 2,550 reading (52.069) and the first 2,685 one (52.169).
  - `fkl_box_halfpixel`, 52.388-52.898: within 0.5 s of the change by the readings. Its durations are flat: 39.62 µs (whole call) and 41.12 µs (kernel_only), against 39.65 and 41.12 µs for `fkl_half_once`.
  - Every other row: 2,685 MHz.
- **So in both nsys runs**, the kernels measured at 2,550 MHz are:
  - all of `fkl_box_asym`'s whole-call kernels;
  - the first 30 (this window) or 40 (the window e18-box) of its 100 kernel-only kernels.

  Its kernel-only medians, 40.45 and 40.51 µs, come from the later kernels. The kernel table above shows the effect in its min-max (40.29-42.85 µs).

**The candidate.** The timing's first interval is `fkl_box_asym`'s whole call. It follows the GPU's idle end of the checks, where the clock fell to 2,550 MHz:
- in every quiet run of the window e18-box;
- in the nsys runs of both windows.

In this window's quiet runs, that idle part lasted about 0.5 s (0 % in one reading) and the clock stayed at 2,685 MHz. In the window e18-box it lasted about 1.5 s. This is a candidate for observation 2 and for this window's -4.7 % flag, not a cause: the sampler does not resolve the 5 ms whole call in quiet-run1 and quiet-run3.

## Open observation 1 of the window e18-box, here
In the CUDA-event times (`e18_box_split_kernel_vs_whole.csv`), the kernel_only interval is above the whole_call interval in six of the seven rows:
- `fkl_box_asym`, `fkl_trunc_halfpixel`, `npp`, `npp_box` and `fkl_box` by 2.0-2.4 %;
- `npp_resize` by 0.6 %;
- `fkl_resize` is the exception, below it: 0.991 (0.984-0.995).

Under nsys, the kernels themselves run 3.0-3.7 % longer after the spin in every row except `fkl_box_asym`, whose kernel_only kernel is the shorter (0.992). Its whole-call kernels ran at 2,550 MHz (the section above). The cause of the 3.0-3.7 % is not determined. The readings show 2,685 MHz for those rows, but at their 0.5 s resolution they cannot show a change within one 5 ms spin.

## Files
- **The window:**
  - `window-script.sh` and `window-console.txt`;
  - `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`;
  - `gpu_clocks.csv`, `gpu_clocks.err` (empty), `load.log`.
- **`step0/`:** `step0.sh`, `console.txt`, `binary-sha256.txt`, `build.log`.
- **`warmup/`, `quiet-run1/`, `quiet-run2/`, `quiet-run3/`:** each run's
  - `e18_box_split.csv`, `e18_box_split_agreement.csv`, `e18_box_split_checks.csv`;
  - `stdout.txt`, `stderr.txt` (empty), `exit_code.txt`, `load.txt`, `gpu-at-end.csv`.
- **`nsys/`:** the same files, plus
  - `e18-box-split.nsys-rep`;
  - `spin-coverage.txt`;
  - `kernel-time-by-interval.txt`;
  - `kernel-durations.txt`, `kernel-durations.csv`.
- **The summary:** `e18_box_split_summary.csv`, `e18_box_split_ratios.csv`, `e18_box_split_kernel_vs_whole.csv`, `e18_box_split_repeats.csv`, `e18_box_split_report.md`.
- **`clocks/`:**
  - `clock-readings-this-window.txt` and `clock-readings-window-e18-box.txt`, from `clock_readings.py`: every change of the SM clock, utilisation or pstate, with its run;
  - `clock-by-row-nsys-this-window.txt` and `clock-by-row-nsys-window-e18-box.txt`, from `clock_by_row.py`: the kernels of each nsys range by clock, and the first rows' durations in groups of 10;
  - `csv-creation-times.txt`: the creation times of this window's CSVs (`stat`).
- **`scripts-used/`:**
  - `e18_box_split_report.py`;
  - `kernel_durations.py`;
  - `clock_readings.py`, `clock_by_row.py`;
  - `spin_coverage.py` and `kernel_time_by_interval.py`, both #48's, unchanged.
- **Left out:**
  - the SQLite exports of the two nsys runs (1.3 MiB for this window, 1.9 MiB for the window e18-box; both derived from the committed `.nsys-rep` files) and this window's export log;
  - step 0's SASS dump.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program or script named; nothing was filtered.
