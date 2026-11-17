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
- **The clocks:** during the timing, `gpu_clocks.csv` shows the SM at 2,685 MHz and the memory at 10,251 MHz (P2), at 45-48 °C. These are the clocks and temperatures of the window e18-box.

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
  - Under nsys, here as there, its whole-call kernel is the longer of its two: 40.77 µs against 40.45 µs (ratio 0.992). The other rows' kernel ratios, kernel_only over whole_call, are 1.030-1.037.
- **The kernels under nsys, old against new** (one nsys run each; the whole-call medians, then the kernel-only medians, from both windows' `kernel-time-by-interval.txt`):
  - `fkl_box_asym`: 41.09 against 40.77 µs (-0.8 %), and 40.51 against 40.45 µs;
  - `fkl_trunc_halfpixel`: 38.05 against 37.98 µs, and 39.42 against 39.39 µs;
  - `npp`'s two kernels: 32.03 against 32.00 µs, and 32.96 against 32.99 µs.
  - Each of these changes is 0.8 % or less. A single nsys run has no run-to-run spread; the min-max over its 100 calls is in the kernel table above.
- **The pipeline ratio** `npp` / `fkl_trunc_halfpixel` is 0.846 here against 0.848 there in the whole call, and 0.844 against 0.832 in kernel_only (`../e18-box/e18_box_speedups.csv`).

## Open observation 1 of the window e18-box, here
In the CUDA-event times (`e18_box_split_kernel_vs_whole.csv`), the kernel_only interval is above the whole_call interval in six of the seven rows:
- `fkl_box_asym`, `fkl_trunc_halfpixel`, `npp`, `npp_box` and `fkl_box` by 2.0-2.4 %;
- `npp_resize` by 0.6 %;
- `fkl_resize` is the exception, below it: 0.991 (0.984-0.995).

Under nsys, the kernels themselves run 3.0-3.7 % longer after the spin in every row except `fkl_box_asym`, whose kernel_only kernel is the shorter (0.992). The cause is not determined.

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
- **`scripts-used/`:**
  - `e18_box_split_report.py`;
  - `kernel_durations.py`;
  - `spin_coverage.py` and `kernel_time_by_interval.py`, both #48's, unchanged.
- **Left out:**
  - the SQLite export of the nsys run (1.3 MiB, derived from the `.nsys-rep`) and its export log;
  - step 0's SASS dump.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program or script named; nothing was filtered.
