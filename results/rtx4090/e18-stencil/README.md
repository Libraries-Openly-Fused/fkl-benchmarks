# E18 window (e18-stencil), 2026-11-09, 13:58:00-13:58:31: the stencil after a resize, timed (RTX 4090)

This window timed E18 (`opencv-cuda/stencil/e18_stencil.cu` at bd5900d, branch `stencil-experiment`): FKL's LinearFilterDPP in one kernel against OpenCV-CUDA and NPP, one call per step.
- **The chain:** a 3840 x 2160 8UC3 frame (E4's seeded source) is cropped to 2560 x 1440 at (640, 360), converted to float, resized bilinearly to 1920 x 1080, sharpened with the 3 x 3 kernel 0 -1 0 / -1 5 -1 / 0 -1 0 (border REPLICATE), and converted to 8UC3 with saturation, ties to even.
- **The implementations:** they are listed in the header comment of the program. The record of phase 1 (the agreement, the checks of FKL's path, the kernel counts and metrics) is `../e18-stencil-phase1/`.

**Setup.**
- Machine: RTX 4090 = nvidia-smi index 1 (00000000:09:00.0), CUDA 13.4.3, driver 610.43.02, Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic. The desktop session kept running.
- The coordinator held the paper agent's compiles and ran nothing on this machine during the window.

**The build (step 0, `step0/`).** Compile only, 13:49:04-13:49:11.
- Preset `local-rtx4090`: nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, OpenCV 4.14.
- Submodules: FKL 1da77899, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- The binary carries bd5900d without -dirty.
- sha256: `55e1d59a53ad185bc650e0d1ef675fb338084e1664ca4590efc063b6c19c132d`.

## What ran (`window-script.sh`, `window-console.txt`)
**Guards before the start.** The script stops before starting unless all of these hold:
- the blocked-programs guard passes;
- the binary is the one step 0 checked (its sha256 and its commit);
- no measurement, build or compile job is running (`scripts/quiet_check.sh`);
- no other program has a context on the RTX 4090;
- the output folder does not exist yet.

The load before the start was 0.03.

**The steps.** Every step is pinned to cores 0-3,16-19 with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
1. **`warmup/`:** one complete run, the checks and the timing, not used. 4 s.
2. **`quiet-run1..3/`:** three complete runs, back to back, 4-5 s each.
   - Every run checked first, then timed. Each passed its checks ("E18 checks passed").
   - Its agreement and check rows equal those of phase 1.
   - Exit 0, empty stderr, 13 rows each, no INCOMPLETE line and no -nan.
3. **`nsys/`:** after the three timed runs, as its own step, one complete run under `nsys profile --trace=cuda,nvtx` (11 s, exit 0). Its times are not used.
4. **`ncu/`:** the two fused kernels, one launch each (`E18_PROFILE=1 E18_ONLY=fkl_stencil_asym,fkl_stencil_halfpixel`, 1 s, exit 0).
   - The metrics are E1's set plus the LaunchStats and Occupancy sections.
   - Its output CSV is `ncu/stdout.txt`.

**Recorded throughout the window:**
- `load.log`: the load logger, every 5 s. Only the program was busy on the benchmark cores.
- `gpu_clocks.csv`: nvidia-smi every 100 ms. During the three runs the SM clock was 2,685-2,700 MHz and the temperature 43-46 °C.
- `environment-before.txt` and `environment-after.txt`. The only other compute context on the RTX 4090 was the desktop terminal, `/usr/bin/ptyxis`.

## The timing protocol (`e18_stencil.csv` of each run)
The implementations are timed in this order, the same in every run:
1. `fkl_stencil_asym`
2. `fkl_stencil_halfpixel`
3. `opencv_cuda`
4. `npp`
5. `fkl_stencil_asym_once`
6. `fkl_stencil_halfpixel_once`

Each implementation is timed in two intervals, first the whole call, then kernel-only.
- **whole_call, E4's interval (Figure 15):** `fklb::timeIt`, CUDA events around one call, 100 calls after one warm-up call.
  - The per-call FKL variants build the crop view, every IOp and the details inside the interval.
  - The `_once` variants hold only the launch path and the kernel.
  - The baselines are called as users call them: OpenCV-CUDA's six calls and NPP's four.
  - When the GPU waits for the host, the interval also holds the host's dispatch.
- **kernel_only, E1's interval (Table III):** `fklb::timePreparedLaunch` with E1's `gpuSpin` (5,000 µs) as `beforeLaunch`.
  - The preparation is timed on the host: the crop view, the IOps and the details for FKL; the ROI for OpenCV; the ROI pointer for NPP; nothing for the `_once` variants.
  - Then come the spin, the start event, the whole call (every kernel of it) and the stop event, 100 iterations after one warm-up.
  - The interval runs from the end of the spin to the end of the call's last kernel.

## Results: the median of the three runs' means, with their min-max (`e18_summary.csv`, `e18_speedups.csv`, `e18_report.md`)

| version | whole_call (ms) | kernel_only (ms) |
|---|--:|--:|
| fkl_stencil_asym | 0.0385 (0.0382-0.0386) | 0.0393 (0.0392-0.0393) |
| fkl_stencil_halfpixel | 0.0385 (0.0384-0.0386) | 0.0398 (0.0395-0.0398) |
| opencv_cuda | 0.1737 (0.1737-0.1739) | 0.1687 (0.1686-0.1691) |
| npp | 0.1213 (0.1206-0.1215) | 0.1199 (0.1199-0.1212) |
| fkl_stencil_asym_once | 0.0387 (0.0386-0.0387) | 0.0399 (0.0399-0.0399) |
| fkl_stencil_halfpixel_once | 0.0388 (0.0387-0.0388) | 0.0399 (0.0399-0.0399) |

The spread between the three runs is at most 1.1 % (npp, kernel_only). The host preparation of the kernel-only interval is 0.1 µs or less (median) for every version.

**FKL's speedup over the baseline of its convention**, the median of the three per-run ratios, with their min-max:

| baseline / FKL | whole_call | kernel_only |
|---|--:|--:|
| opencv_cuda / fkl_stencil_asym | 4.52 (4.51-4.54) | 4.30 (4.29-4.32) |
| npp / fkl_stencil_halfpixel | 3.15 (3.13-3.16) | 3.03 (3.01-3.05) |
| opencv_cuda / fkl_stencil_asym_once | 4.49 (4.49-4.50) | 4.23 (4.22-4.23) |
| npp / fkl_stencil_halfpixel_once | 3.13 (3.10-3.14) | 3.00 (3.00-3.04) |

**Observation: for FKL, kernel_only is 2.1-3.4 % above whole_call.** For the baselines it is below, because their whole call also holds the host's dispatch between their kernels.
- In the nsys pass, the fused kernel itself runs 37.1-38.8 µs (median, per range) back to back in the whole_call ranges, and 38.3-39.1 µs after the 5 ms spin in the kernel_only ranges.
- So the difference is in the kernel, not in the intervals.
- Its cause was not determined.

## Kernels per call, and whether the spin covers the enqueue (`nsys/spin-coverage.txt`)
The nsys trace was analysed by `scripts-used/spin_coverage.py`. It was exported to SQLite with `nsys export --type sqlite`; the export is not kept.

**Kernels per call**, in the whole_call ranges (101 calls each):
- every FKL variant: **1** (`linearFilterDPPKernel`);
- opencv_cuda: **6** (`transformSmart` x 2, `resize_linear`, `transformSimple` x 2, `filter2D`);
- npp: **4** (`ForEachPixelNaiveInLargeImage` x 3, `resizeNaive`).

**Spin coverage, in the kernel_only ranges:** for every implementation, all 100 timed calls are **covered**. The host's launch call of the call's last kernel returned before the spin ended on the GPU.
- The smallest margin is 4,974.6 µs (opencv_cuda), 4,981.1 µs (npp) and 4,992.2 µs (FKL).
- On the GPU, the gap from the end of the spin to the first kernel is at most 3.0 µs (median 2.2 µs). The start event is in it.
- Between the kernels of one baseline call the gaps are at most 1.47 µs (opencv_cuda, median 0.70 µs) and 1.38 µs (npp, median 0.67 µs).
- So no host gap is inside the kernel-only interval.

**Texture objects:** the trace has no texture, array or surface API call, runtime or driver, in any call, the creation of OpenCV's filter included.
- OpenCV's `filter2D` kernel takes a reader named `tex_filter2D_float4_reader`, but its `apply()` creates and destroys no texture object.
- This holds for all 202 timed calls of the trace and in phase 1's profile.

## The fused kernel, Table III's metrics
Static data come from cuobjdump of the measured binary (`kernel-census.tsv`, counted as `scripts/sass_tools.py` counts); occupancy comes from ncu (`ncu/stdout.txt`).

| | fkl_stencil_asym | fkl_stencil_halfpixel |
|---|--:|--:|
| SASS instructions | 840 | 848 |
| registers | 40 | 40 |
| stack, local memory | 0, 0 | 0, 0 |
| LDL / STL | 0 / 0 | 0 / 0 |
| shared memory | 4,284 B (static) | 4,284 B |
| CONSTANT[0] | 480 B | 480 B |
| kernel arguments | 122 B | 122 B |
| block, grid | 16 x 8 (128 threads), 120 x 135 | the same |
| theoretical occupancy | 100 % (48 warps per SM) | 100 % |
| achieved occupancy | 92.53 % | 92.34 % |
| blocks per SM allowed by registers / shared memory / warps / blocks | 12 / 12 / 12 / 24 | the same |
| duration under ncu (clocks locked by ncu) | 42.0 µs | 41.6 µs |

- **Arguments:** 122 B is the sum of `sizeof` of the five arguments: the details 24, the two Reads 72 (image 48, weights 24), multiply 1, accumulate 1, the epilogue write 24.
- **Shared memory:** the halo of the default tile (22 x 14 float3) plus the 7 x 7 coefficients. These are the defaults of `LinearFilterDPPDetails<float3>`: a 16 x 8 tile and a largest kernel of 7 x 7, the call a user writes.
- **No `__grid_constant__`:** `linearFilterDPPKernel` takes its parameters by value without it (FKL 1da77899, `linear_filter.h:285`). Its parameters are read only as constant-bank operands (76 instructions with a c[0x0] operand). There is no local memory, because everything is inlined.

## Files
- `window-script.sh`, `window-console.txt`: the script and its complete output.
- The window's records:
  - `environment-before.txt`, `environment-after.txt`;
  - `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err` (empty);
  - `load.log`.
- `step0/`: `console.txt`, `binary-sha256.txt`, `build.log`.
- `warmup/`, `quiet-run1/`, `quiet-run2/`, `quiet-run3/`: each run's
  - `e18_stencil.csv`, `e18_stencil_agreement.csv`, `e18_stencil_checks.csv`;
  - `stdout.txt`, `stderr.txt`, `exit_code.txt`;
  - `load.txt`, `gpu-at-end.csv`.
- `nsys/`: the same files, plus `e18-stencil.nsys-rep` and `spin-coverage.txt`.
- `ncu/`: `stdout.txt` (the ncu CSV), `stderr.txt`, `exit_code.txt`, `load.txt`, `gpu-at-end.csv`.
- The summary:
  - `e18_summary.csv`, `e18_speedups.csv`, `e18_report.md`, from `scripts-used/e18_report.py`;
  - `kernel-census.tsv`, from `../e18-stencil-phase1/kernel_census.py`.
- `scripts-used/`: `spin_coverage.py`, `e18_report.py`.
- **Left out:**
  - the SQLite export of the nsys trace (1.5 MiB, derived from the `.nsys-rep`);
  - the SASS dump of the binary.

  They stay in `scratch/` on this machine. Every file here is the complete output of the program named; nothing was filtered.
- **Outside this folder:** the start and end lines of the window are also in the reference checkout's `results/rtx4090/e0-run.log`, which is not committed.
