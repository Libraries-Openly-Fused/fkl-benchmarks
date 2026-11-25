# E18 box window (e18-box), 2026-11-17, 12:09:14-12:10:01: the box filter after a resize, timed (RTX 4090)

**Note, 2026-11-25:** `e18_box_cu` was timed again at FKL 24478f0 (A) and at the pin b10fc66 (B), the merge of FKL #433 that changes `linearFilterDPPKernel`, in `../e18-retime-b10fc66/` (24 runs, the binaries of #60's identity check). This record's numbers are unchanged. B against A, same window: the 16 FKL numbers -0.33 % to +1.78 % (B slower in most), `opencv_cuda` and `npp` -0.12 % to +0.26 %; no number crosses the 3 % flag. A and B against this record: no number crosses it either (at most +1.56 %).

This window timed E18 with the 3 x 3 box filter (`opencv-cuda/stencil/e18_box.cu` at 7dca510, branch `e18-box`).

**The pipeline:** a 3840 x 2160 uchar3 frame (E4's seeded source) is processed in three steps, in uchar from start to end:
1. cropped to 2560 x 1440 at (640, 360);
2. resized bilinearly to 1920 x 1080;
3. filtered with a 3 x 3 box filter, border REPLICATE.

**The 10 variants:** they are listed in the header comment of the program.
- The record of phase 1 is `../e18-box-phase1/`: the checks against exact references, the kernel and API counts and the static kernel metrics.
- OpenCV-CUDA's and NPP's 8-bit box filters truncate (floor(sum / 9)). So FKL has rounding variants (`fkl_box_*`, as RPP rounds) and truncating twins (`fkl_trunc_*`, as OpenCV-CUDA and NPP).
  - `fkl_trunc_asym` gives bitwise the same output as `opencv_cuda`.
  - `fkl_trunc_halfpixel` differs from `npp` only where their resizes round the exact ties of HALF_PIXEL differently: 115,927 values.

**Setup.**
- **Machine:** RTX 4090 = nvidia-smi index 1 (`gpu-info.csv`). CUDA 13.4.3, driver 610.43.02, Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic. The desktop session kept running.
- **The coordinator** held the paper agent's compiles and ran nothing on this machine during the window.
- **The code:** branch `e18-box` at 7dca510, the head of #48. Its base is main fa76331, which was main during the window, with FKL 24478f0, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245.

**The build (step 0, `step0/`).** Compile only, 12:04:33-12:04:42, before the window.
- Only the target `e18_box_cu`, preset `local-rtx4090`: nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, OpenCV 4.14.0. Build exit 0, with the 25 usual vector-type deprecation warnings.
- The binary carries 7dca510 without -dirty. sha256: `af354faf683eee64f6147c46bedfd7ec3c0389993d5cab8cadb5624987ff371a`.
- Its SASS census (`kernel-census.tsv`) is identical to that of phase 1's binary (36233ec) in every column compared: the instructions, constant-bank operands, LDL/STL, registers, stack, local memory and CONSTANT[0] of all 9 kernels.

## What ran (`window-script.sh`, `window-console.txt`)
**Guards before the start.** The script stops before starting unless all of these hold:
- the frozen-program guard passes;
- the binary is the one step 0 checked (its sha256 and its commit);
- no measurement, build or compile job is running (`scripts/quiet_check.sh`);
- no program outside /usr has a context on the RTX 4090;
- the output folder does not exist yet.

The load before the start was 0.03.

**The steps.** Every run is pinned to cores 0-3,16-19 with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, and has a 180 s timeout. The load logger (`load.log`) and the nvidia-smi sampler (`gpu_clocks.csv`, every 100 ms, on core 14) ran throughout.
1. **`warmup/`:** one complete run, the checks and the timing, not used. 8 s.
2. **`quiet-run1..3/`:** three complete runs, 7, 7 and 8 s.
   - Before each one the script waited for an idle machine: no measurement, build or compile job, a 1-minute load below 0.5, and no other program on the RTX 4090. Each was idle at once, at loads 0.11, 0.25 and 0.31.
   - Every run checked first, then timed every variant in the fixed order: the per-call versions, then the build-once variants.
3. **`nsys/`:** after the three timed runs, as its own step, one complete run under `nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none`, 14 s.
4. **`ncu/`:** the four timed fused kernels, 2 s.
   - The mode: `E18_PROFILE=1 E18_PROFILE_CALLS=1`, with `E18_ONLY` set to the four per-call FKL variants, so each kernel is launched twice: the warm-up and the call in its NVTX range.
   - The metrics: #43's set and the LaunchStats and Occupancy sections, with ncu's base clocks locked (its default `--clock-control base`).

**The checks of every run.**
- Every run exited 0, with an empty stderr.
- Every complete run (the warm-up, the three quiet runs and the nsys run) printed "E18 box checks passed".
- No CSV is INCOMPLETE or holds -nan.
- The three check CSVs of each of these five runs are identical to phase 1's, row by row (`e18_box_agreement.csv`, `e18_box_checks.csv`, `e18_box_kernel.csv`).
- The clocks: during the timing, `gpu_clocks.csv` shows the SM at 2,685 MHz and the memory at 10,251 MHz (P2). At the end of each run the memory clock briefly shows 10,501 MHz (P0); between the runs the SM clock falls to 2,550 MHz at 0 % utilisation. The GPU stayed at 43-48 °C.
  - **Correction (2026-11-17, #50).** The 2,550 MHz readings fall inside the runs, not between them: at the idle end of each quiet run's checks, directly before its timing, and from 12:09:46.2 to 52.1 in the nsys run.
    - In quiet-run2 they continue into the timing (12:09:31.7-32.1, at 95 % utilisation). The nsys run's timing began at 2,550 MHz: all of `fkl_box_asym`'s whole-call kernels and the first 40 of its kernel-only kernels ran at that clock.
    - The timing's first interval is `fkl_box_asym`'s whole call. That makes the clock a candidate, not a cause, for "`fkl_box_asym`, the first variant timed in each run" below.
    - The nsys run reached 49 °C.
    - The analysis: `../e18-box-split/README.md`, "The SM clock and the first row".

## The times (`e18_box_summary.csv`, `e18_box_report.md`)
Each value is the median of the three quiet runs' means (100 iterations each), with the min-max of the three means. The CSVs are in ms; this table is in µs.
- **whole_call:** `fklb::timeIt`, E4's interval (Figure 15).
- **kernel_only:** `fklb::timePreparedLaunch` behind the 5 ms `gpuSpin`, E1's interval (Table III). The host preparation is timed separately.

| version | box | whole_call (µs) | min-max | kernel_only (µs) | min-max | kernel_only host preparation (µs) |
|---|---|--:|--:|--:|--:|--:|
| `fkl_box_asym` | rounds | 42.0 | 41.97-42.09 | 41.6 | 41.54-42.06 | 0.14 |
| `fkl_box_halfpixel` | rounds | 40.6 | 40.60-40.63 | 42.0 | 41.99-42.00 | 0.14 |
| `fkl_trunc_asym` | truncates | 39.3 | 39.27-39.38 | 40.3 | 40.21-40.33 | 0.15 |
| `fkl_trunc_halfpixel` | truncates | 39.6 | 39.53-39.62 | 40.9 | 40.93-40.96 | 0.15 |
| `opencv_cuda` | truncates | 48.4 | 48.44-48.52 | 49.2 | 49.16-49.18 | 0.06 |
| `npp` | truncates | 33.6 | 33.53-33.57 | 34.1 | 34.03-34.08 | 0.04 |
| `fkl_asym_once` | rounds | 40.3 | 40.24-40.29 | 41.6 | 41.51-41.58 | 0.04 |
| `fkl_half_once` | rounds | 40.9 | 40.93-40.95 | 42.0 | 42.00-42.04 | 0.04 |
| `fkl_trunc_asym_once` | truncates | 39.3 | 39.30-39.33 | 40.3 | 40.27-40.32 | 0.04 |
| `fkl_trunc_halfpixel_once` | truncates | 39.5 | 39.51-39.56 | 40.9 | 40.95-40.96 | 0.04 |

**Spread.** The largest spread of the three means is 1.3 % (`fkl_box_asym`, kernel_only: one run at 42.06 µs, two at 41.54 and 41.55 µs). Every other spread is at most 0.3 %. No value exceeds the 3 % flag for GPU times.

## The speedups (`e18_box_speedups.csv`)
Each value is the library's time over FKL's: the median of the three per-run ratios, with their min-max.

| baseline / FKL | outputs | whole_call | min-max | kernel_only | min-max |
|---|---|--:|--:|--:|--:|
| opencv_cuda / `fkl_trunc_asym` | equal | 1.23 | 1.23-1.23 | 1.22 | 1.22-1.22 |
| npp / `fkl_trunc_halfpixel` | equal but for the resize ties | 0.85 | 0.85-0.85 | 0.83 | 0.83-0.83 |
| opencv_cuda / `fkl_box_asym` | FKL rounds, the library truncates | 1.15 | 1.15-1.16 | 1.18 | 1.17-1.18 |
| npp / `fkl_box_halfpixel` | FKL rounds, the library truncates | 0.83 | 0.83-0.83 | 0.81 | 0.81-0.81 |
| opencv_cuda / `fkl_trunc_asym_once` | equal | 1.23 | 1.23-1.23 | 1.22 | 1.22-1.22 |
| npp / `fkl_trunc_halfpixel_once` | equal but for the resize ties | 0.85 | 0.85-0.85 | 0.83 | 0.83-0.83 |
| opencv_cuda / `fkl_asym_once` | FKL rounds, the library truncates | 1.20 | 1.20-1.20 | 1.18 | 1.18-1.18 |
| npp / `fkl_half_once` | FKL rounds, the library truncates | 0.82 | 0.82-0.82 | 0.81 | 0.81-0.81 |

- **FKL against OpenCV-CUDA:** with equal outputs, FKL's one kernel is 1.23 times as fast for the whole call and 1.22 times for the kernels.
- **FKL against NPP: NPP is faster.** Its two kernels take 0.85 times FKL's whole call, and 0.83 times FKL's kernel time.
- **Building every call costs nothing measurable here.** The `_once` forms give the speedups of the per-call versions within 0.01, except `fkl_box_asym`'s whole call (see below). The host preparation of the per-call versions is 0.14-0.15 µs.

**Where the time goes (`nsys/`, the kernels' own GPU durations, medians over the 100 whole_call calls).**

| version | its kernels (µs) | sum (µs) |
|---|---|--:|
| `fkl_trunc_halfpixel` | `linearFilterDPPKernel` 38.1 | 38.1 |
| `npp` | `resizeNaive` 10.8, `ForEachPixelNaiveInLargeImage` (the box) 21.2 | 32.0 |
| `opencv_cuda` | `resize_linear` 9.3, `transformSimple` 6.1 (two of them), `copy` (`copyMakeBorder`) 6.0, `ForEachPixelNaiveInLargeImage` (NPP's 4-channel box) 18.8 | 46.2 |

- FKL's one kernel is slower than the sum of NPP's resize and box kernels. These numbers do not show why.
- One fact about its structure: for a 3 x 3 kernel, each 16 x 8 tile stages 18 x 10 resized values in shared memory (`neighborhood.h:85`). So it computes the resize for 1.4 times as many pixels as it writes.
- OpenCV-CUDA's call holds three kernels more than NPP's: the two colour conversions and the padding copy, 18.1 µs together.

## The spin covers every kernel-only call (`nsys/spin-coverage.txt`)
- **The method:** `spin_coverage.py` (#43's script, unchanged), on the SQLite export of the nsys run. It joins every kernel to its host launch call by correlation id.
- **The coverage:** for every variant, all 100 timed calls are covered. The host's launch call of a call's last kernel returned before the spin ended on the GPU, so the GPU never waited for the host inside the interval.
- **The smallest margin**, spin end minus that return: 4,972.1 µs for `opencv_cuda`, whose calls hold the most launches (5). That includes its `cudaGetDeviceProperties` in every call (from OpenCV's `NppStreamHandler`; 0.17 µs median in phase 1's trace). `npp` has 4,987.0 µs, and every FKL variant at least 4,990.8 µs.
- **The gaps on the GPU:**
  - from the end of the spin to the first kernel: at most 2.62 µs (median 2.0-2.2 µs), with the start event inside it;
  - between the kernels of one baseline call: at most 0.86 µs (`opencv_cuda`, median 0.77 µs) and 0.77 µs (`npp`, median 0.74 µs).
- **Kernels per call:** FKL 1, `opencv_cuda` 5, `npp` 2, in both intervals, as in phase 1.
- **No texture, array or surface API calls** appear in the whole trace.

## The open observation of #43: FKL's kernel_only is above its whole_call (`e18_box_kernel_vs_whole.csv`)

| version | kernel_only / whole_call | min-max |
|---|--:|--:|
| `fkl_box_asym` | 0.991 | 0.989-0.999 |
| `fkl_box_halfpixel` | 1.034 | 1.034-1.034 |
| `fkl_trunc_asym` | 1.024 | 1.023-1.027 |
| `fkl_trunc_halfpixel` | 1.034 | 1.034-1.036 |
| `opencv_cuda` | 1.015 | 1.014-1.015 |
| `npp` | 1.015 | 1.015-1.015 |
| `fkl_asym_once` | 1.032 | 1.032-1.032 |
| `fkl_half_once` | 1.026 | 1.026-1.027 |
| `fkl_trunc_asym_once` | 1.025 | 1.024-1.026 |
| `fkl_trunc_halfpixel_once` | 1.036 | 1.035-1.036 |

- **Yes, again.** Every FKL variant but `fkl_box_asym` is 2.4-3.6 % slower in kernel_only than in whole_call. #43 saw 2.1-3.4 %.
- **This time the libraries show it too**, by 1.5 %. In #43 their kernel_only was below their whole_call (OpenCV 0.1687 against 0.1737 ms, NPP 0.1199 against 0.1213 ms).
- **The excess is inside the kernels** (`nsys/kernel-time-by-interval.txt`, from `scripts-used/kernel_time_by_interval.py`). Under nsys, the kernels' own GPU durations are 2.3-3.8 % longer after the spin than in the back-to-back calls of whole_call:
  - FKL: 1.4-1.5 µs longer, for example 38.05 against 39.42 µs for `fkl_trunc_halfpixel`;
  - `npp`: 2.9 %;
  - `opencv_cuda`: 2.3 %.
  - The gaps cannot explain it: at most 2.6 µs before the first kernel, and below 1 µs between kernels.
- **The cause is not determined.** The 100 ms clock samples show no change during the timing, but they cannot resolve a 5 ms spin.

**`fkl_box_asym`, the first variant timed in each run.**
- Its whole call is 42.0 µs, against 40.3 µs for `fkl_asym_once`, which launches the same kernel with the same arguments. Under nsys the kernel itself takes 41.09 µs there, against 38.98 µs for `fkl_asym_once`.
- This holds in all three runs (41.97-42.09 µs), and makes its kernel_only / whole_call ratio 0.99.
- Its kernel_only time (41.6 µs) equals `fkl_asym_once`'s (41.6 µs). So for `fkl_box_asym` and `opencv_cuda`, the kernel_only speedup (1.18) is the one that does not depend on the position in the pass order.
- The cause is not determined. The pass order is fixed in the program, so the position effect was not tested.

## The fused kernels under ncu (`ncu/stdout.txt`)
ncu locked the clocks to base. The table gives the launch in the NVTX range; the CSV also holds the warm-up launch, which is within 0.9 µs of it.

| | fkl_box_asym | fkl_box_halfpixel | fkl_trunc_asym | fkl_trunc_halfpixel |
|---|--:|--:|--:|--:|
| duration under ncu (base clocks) | 42.98 µs | 43.65 µs | 41.89 µs | 42.14 µs |
| achieved occupancy | 92.66 % | 92.74 % | 92.60 % | 92.63 % |
| theoretical occupancy | 100 % | 100 % | 100 % | 100 % |
| blocks per SM allowed by registers / shared memory / warps / blocks | 12 / 12 / 12 / 24 | the same | the same | the same |
| registers per thread | 40 | 40 | 40 | 40 |
| block, grid | 128 threads, 16,200 blocks (120 x 135) | the same | the same | the same |
| static shared memory | 4,284 B | 4,284 B | 4,284 B | 4,284 B |

The static metrics are in phase 1's README (SASS instructions 800 / 808 / 864 / 872, no local memory, CONSTANT[0] 472 / 472 / 488 / 488 B, arguments 114 / 114 / 130 / 130 B).

## Files
- **The window:**
  - `window-script.sh` and `window-console.txt`: the script and its complete output;
  - `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`;
  - `gpu_clocks.csv`, `gpu_clocks.err` (empty), `load.log`.
- **`step0/`:** `step0.sh`, `console.txt`, `binary-sha256.txt`, `build.log`.
- **`warmup/`, `quiet-run1/`, `quiet-run2/`, `quiet-run3/`:** each run's
  - `e18_box.csv`, `e18_box_agreement.csv`, `e18_box_checks.csv`, `e18_box_kernel.csv`;
  - `stdout.txt`, `stderr.txt` (empty), `exit_code.txt`, `load.txt`, `gpu-at-end.csv`.
- **`nsys/`:** the same files, plus
  - `e18-box.nsys-rep`;
  - `spin-coverage.txt`;
  - `kernel-time-by-interval.txt`.
- **`ncu/`:** `stdout.txt` (the ncu CSV), `stderr.txt` (empty), `exit_code.txt`, `load.txt`, `gpu-at-end.csv`, and `e18_box_kernel.csv` (the profile mode writes it too).
- **The summary:**
  - `e18_box_summary.csv`, `e18_box_speedups.csv`, `e18_box_kernel_vs_whole.csv`, `e18_box_report.md`, from `scripts-used/e18_box_report.py`;
  - `kernel-census.tsv`, from `../e18-stencil-phase1/kernel_census.py`.
- **`scripts-used/`:** `e18_box_report.py`, `spin_coverage.py` (#43's, unchanged), `kernel_time_by_interval.py`.
- **Left out:**
  - the SQLite export of the nsys run (1.9 MiB, derived from the `.nsys-rep`) and its export log;
  - the SASS dump of the census.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program or script named; nothing was filtered.
