# E18 timed again at the pin b10fc66 (window e18-retime-b10fc66), 2026-11-25, 00:37:00-00:41:52 (+01:00): RTX 4090

**Why.** The pin move to FKL b10fc66 (#60, the merge of FKL #433) changes `linearFilterDPPKernel`: its details lose width and height.
- **The device code** (`../fkl-pr433-pin-check/`): every FKL kernel of E18's three programs that uses the filter DPP differs by 8 or 16 SASS instructions, with constant bank 0 8 B smaller.
- **The outputs** (`../task-b-sm89-check/`): bitwise those of 24478f0.
- **This window** times E18 again, so that the E18 records match the code at the new pin. The paper reports no E18 number.

**The binaries.** No new build: A and B are the binaries of #60's compile-only identity check (`binaries.sha256`):
- **A:** fkl-benchmarks main a8f9daa with FKL 24478f0;
- **B:** 2160c70 (the pin and the details port) with FKL b10fc66.
- **Options:** both builds used `scripts/check_fkl_tree.sh --only bench` with the options of the reference build of preset `local-rtx4090`. Every build-information field of the CSV headers equals the records': nvcc 13.4.92, sm_89, `-fmad` on, `-lineinfo` on, GNU 15.2.0, Release, `-march=native`, `-ftemplate-depth=1000`, OpenCV 4.14.0, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- **What differs from the records' binaries:**
  1. The repository commit reads `unknown` in the CSV headers. The build's source is a link farm without `.git`; the binaries are identified by their sha256 and carry their FKL commit.
  2. The records' binaries were built in their branch worktrees, one target each; A and B come from a full bench-group build in scratch folders.
  3. `../e18-stencil/` was measured at FKL 1da77899, whose device code equals 24478f0's (`../fkl-pr432-check/`). The other two records were measured at 24478f0, as A.
  4. Since the records, the CMake files changed only in `cmake/fklb_write_build_info.cmake` (the paths of the dirty check) and in new targets. The lines of the E18 targets are unchanged. `common/` changed only in `hip_timing.h`, which the CUDA programs do not include.
- **The machine:** the OS kernel is now Linux 7.0.0-43-generic; the records ran on 7.0.0-38-generic. The driver is the same, 610.43.02.

## What ran (`window-script.sh`; output in `window/`)
**Guards before the start** (all passed; `GUARD_ONLY=1` passed at 00:28:41 and again at 00:35:56). The script stops unless all of these hold:
- the frozen-program guard passes;
- the manifest is the committed one (its sha256 is in the script), and all six binaries match it;
- A carries FKL 24478f0 and B carries b10fc66, with no `-dirty` commit;
- HEAD differs from main a8f9daa only under `results/`;
- the script is the committed one and runs from its place;
- the output folder is the requested one and new;
- no measurement, build or compile job runs, and no program outside /usr uses the RTX 4090.

The coordinator held the paper agent's compiles and ran nothing on this machine during the window.

**The runs.**
- **Order:** the programs ran in the order `e18_box`, `e18_box_split`, `e18_stencil`. Per program, one process per run: `warmup-A`, `warmup-B` (not used), then `quiet-run1-A`, `quiet-run1-B`, `quiet-run2-A`, `quiet-run2-B`, `quiet-run3-A`, `quiet-run3-B`. That makes 24 runs.
- **Each run:** pinned to cores 0-3,16-19, with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` and a 180 s timeout. No environment variable is set for the program.
- **Each program kept its own protocol:** the checks, then `whole_call` (`fklb::timeIt`, 100 calls after one warm-up call) and `kernel_only` (`fklb::timePreparedLaunch` behind the 5 ms `gpuSpin`) of every row, in its own order.
- **The sampler:** nvidia-smi read the GPU every 100 ms throughout (`gpu_clocks.csv`), as in the crop sweep. `gpu-at-end.csv` was written after every run.
- **No nsys or ncu pass.**

**The checks of every run.**
- All 24 runs exited 0 with an empty stderr and printed their checks line ("E18 box checks passed", "E18 box split checks passed", "E18 checks passed").
- No CSV is INCOMPLETE or holds -nan.
- The check CSVs of every run were compared with the record's `quiet-run1`, row by row (`analysis/checks-vs-records.txt`):
  - 52 of 56 are equal: every `*_agreement.csv` and `*_checks.csv` of A and of B, and A's `e18_box_kernel.csv`.
  - The 4 that differ are B's `e18_box_kernel.csv`, the kernel-attribute table. In the 4 FKL rows its details argument is 16 B instead of 24 B (the two ints #433 removes), and the arguments' total is 8 B smaller. This is the difference `../task-b-sm89-check/` recorded.

**The clocks and the load.**
- The GPU stayed at 40-51 °C.
- The 1-minute load was 0.29 at the start. The window's own runs raised it towards the 0.5 of `wait_idle`.
- The SM clock at the start of each run's timing is in `analysis/first-row-clock.txt`: the creation time of the timing CSV, and the readings of the next second. In 23 of the 24 runs the clock read 2,550 MHz during the first second of the timing, and 2,685 (or 2,700) for the rest of it. Only `e18_stencil` quiet-run3-A read 2,685 MHz throughout. Between runs the GPU idled at 210 MHz (`analysis/clock-by-run.txt`).

## How the runs were spaced, per program, against each record
`wait_idle` ran before every run here, for A and B alike. The waits are in each run's `load.txt`.

| program | this window: wait before each run | the record |
|---|---|---|
| `e18_box` | 0 s before the warm-ups and quiet-run1-A/B; then 30, 5, 10 and 5 s before quiet-run2-A, -2B, -3A, -3B | `../e18-box/`: the warm-up, then three quiet runs back to back (`wait_idle` returned at once: "idle after 0 s") |
| `e18_box_split` | 5-15 s before every run | `../e18-box-split/`: the warm-up, then three quiet runs back to back ("idle after 0 s") |
| `e18_stencil` | 5-15 s before every run except quiet-run3-A (0 s) | `../e18-stencil/`: the warm-up and three quiet runs back to back, with no `wait_idle` |

- **A and B were spaced alike,** so this does not touch B against A.
- **It qualifies A and B against old.** In the records every run followed the previous one directly. Here most runs started after the GPU had idled.

## The times (`analysis/retime-summary.csv`, `analysis/retime-report.txt`)
- **Each value:** the median of the three quiet runs' means (100 iterations each), with the min-max of the three means, in µs.
- **old:** the record's three quiet runs, recomputed here. The script checked that they equal the records' summary CSVs.
- **The changes:** in % of the medians.
  - **exceeds:** larger than the larger spread of the two sides;
  - **apart:** the two min-max ranges do not overlap;
  - **FLAG:** above 3 %. All 46 numbers are GPU-bound by the records' rule: `kernel_only` within 10 % of `whole_call`, and its host preparation below 1 % of its GPU time.

| program | version | interval | old (us) | A (us) | B (us) | B / A | B / old | A / old |
|---|---|---|--:|--:|--:|--:|--:|--:|
| `e18_box` | `fkl_box_asym` | whole_call | 42.01 (41.97-42.09) | 42.35 (41.97-42.95) | 42.39 (42.14-42.53) | +0.10 % | +0.92 % **exceeds** (apart) | +0.82 % |
| `e18_box` | `fkl_box_asym` | kernel_only | 41.58 (41.54-42.06) | 42.21 (41.52-42.40) | 42.08 (41.93-42.45) | -0.33 % | +1.19 % | +1.53 % |
| `e18_box` | `fkl_box_halfpixel` | whole_call | 40.60 (40.60-40.63) | 40.57 (40.48-40.72) | 40.97 (40.96-41.10) | +0.99 % **exceeds** (apart) | +0.90 % **exceeds** (apart) | -0.09 % |
| `e18_box` | `fkl_box_halfpixel` | kernel_only | 41.99 (41.99-42.00) | 41.99 (41.98-41.99) | 42.16 (42.14-42.23) | +0.42 % **exceeds** (apart) | +0.41 % **exceeds** (apart) | -0.01 % (apart) |
| `e18_box` | `fkl_trunc_asym` | whole_call | 39.29 (39.27-39.38) | 39.29 (39.11-39.30) | 39.49 (39.48-39.49) | +0.50 % **exceeds** (apart) | +0.49 % **exceeds** (apart) | -0.01 % |
| `e18_box` | `fkl_trunc_asym` | kernel_only | 40.31 (40.21-40.33) | 40.19 (40.11-40.25) | 40.90 (40.90-40.92) | +1.78 % **exceeds** (apart) | +1.47 % **exceeds** (apart) | -0.30 % |
| `e18_box` | `fkl_trunc_halfpixel` | whole_call | 39.59 (39.53-39.62) | 39.52 (39.34-39.53) | 39.85 (39.83-39.90) | +0.82 % **exceeds** (apart) | +0.63 % **exceeds** (apart) | -0.18 % (apart) |
| `e18_box` | `fkl_trunc_halfpixel` | kernel_only | 40.95 (40.93-40.96) | 40.96 (40.86-40.96) | 40.96 (40.96-40.98) | +0.00 % (apart) | +0.03 % (apart) | +0.03 % |
| `e18_box` | `opencv_cuda` | whole_call | 48.45 (48.44-48.52) | 48.30 (48.17-48.39) | 48.42 (48.32-48.53) | +0.26 % | -0.06 % | -0.32 % (apart) |
| `e18_box` | `opencv_cuda` | kernel_only | 49.17 (49.16-49.18) | 49.15 (49.15-49.15) | 49.15 (49.14-49.15) | +0.00 % | -0.03 % (apart) | -0.03 % (apart) |
| `e18_box` | `npp` | whole_call | 33.57 (33.53-33.57) | 33.46 (33.42-33.51) | 33.48 (33.24-33.49) | +0.06 % | -0.25 % (apart) | -0.31 % **exceeds** (apart) |
| `e18_box` | `npp` | kernel_only | 34.08 (34.03-34.08) | 33.97 (33.89-34.00) | 33.93 (33.80-33.99) | -0.12 % | -0.43 % (apart) | -0.31 % (apart) |
| `e18_box` | `fkl_asym_once` | whole_call | 40.27 (40.24-40.29) | 40.17 (40.04-40.23) | 40.50 (40.28-40.59) | +0.84 % **exceeds** (apart) | +0.58 % | -0.26 % (apart) |
| `e18_box` | `fkl_asym_once` | kernel_only | 41.57 (41.51-41.58) | 41.47 (41.24-41.47) | 41.98 (41.47-41.98) | +1.25 % **exceeds** (apart) | +1.00 % | -0.25 % (apart) |
| `e18_box` | `fkl_half_once` | whole_call | 40.94 (40.93-40.95) | 40.90 (40.66-40.93) | 41.15 (40.99-41.18) | +0.59 % (apart) | +0.50 % **exceeds** (apart) | -0.09 % |
| `e18_box` | `fkl_half_once` | kernel_only | 42.01 (42.00-42.04) | 42.00 (41.99-42.01) | 42.28 (42.12-42.30) | +0.67 % **exceeds** (apart) | +0.64 % **exceeds** (apart) | -0.03 % |
| `e18_box` | `fkl_trunc_asym_once` | whole_call | 39.31 (39.30-39.33) | 39.23 (39.18-39.36) | 39.53 (39.42-39.57) | +0.77 % **exceeds** (apart) | +0.56 % **exceeds** (apart) | -0.21 % |
| `e18_box` | `fkl_trunc_asym_once` | kernel_only | 40.28 (40.27-40.32) | 40.24 (40.19-40.24) | 40.91 (40.90-40.92) | +1.67 % **exceeds** (apart) | +1.56 % **exceeds** (apart) | -0.10 % (apart) |
| `e18_box` | `fkl_trunc_halfpixel_once` | whole_call | 39.54 (39.51-39.56) | 39.48 (39.46-39.48) | 39.76 (39.73-39.77) | +0.71 % **exceeds** (apart) | +0.55 % **exceeds** (apart) | -0.16 % **exceeds** (apart) |
| `e18_box` | `fkl_trunc_halfpixel_once` | kernel_only | 40.95 (40.95-40.96) | 40.95 (40.95-40.96) | 40.96 (40.96-40.96) | +0.02 % (apart) | +0.03 % **exceeds** (apart) | +0.01 % |
| `e18_box_split` | `fkl_box_asym` | whole_call | 40.03 (39.95-40.05) | 43.29 (41.97-43.49) | 43.41 (42.33-43.68) | +0.28 % | +8.44 % **exceeds** (apart) **FLAG** | +8.13 % **exceeds** (apart) **FLAG** |
| `e18_box_split` | `fkl_box_asym` | kernel_only | 41.02 (40.99-41.03) | 41.89 (41.61-42.29) | 42.44 (42.37-42.58) | +1.30 % (apart) | +3.46 % **exceeds** (apart) **FLAG** | +2.13 % **exceeds** (apart) |
| `e18_box_split` | `fkl_trunc_halfpixel` | whole_call | 39.19 (39.16-39.23) | 39.29 (39.27-39.31) | 39.53 (39.50-39.57) | +0.61 % **exceeds** (apart) | +0.85 % **exceeds** (apart) | +0.24 % **exceeds** (apart) |
| `e18_box_split` | `fkl_trunc_halfpixel` | kernel_only | 40.13 (40.13-40.19) | 40.22 (40.18-40.27) | 40.84 (40.84-40.88) | +1.55 % **exceeds** (apart) | +1.77 % **exceeds** (apart) | +0.21 % **exceeds** |
| `e18_box_split` | `npp` | whole_call | 33.19 (33.17-33.20) | 33.23 (33.22-33.28) | 33.27 (33.25-33.29) | +0.13 % | +0.25 % **exceeds** (apart) | +0.12 % (apart) |
| `e18_box_split` | `npp` | kernel_only | 33.93 (33.86-33.97) | 33.96 (33.83-33.99) | 33.93 (33.92-33.97) | -0.11 % | -0.01 % | +0.10 % |
| `e18_box_split` | `npp_resize` | whole_call | 12.21 (12.21-12.28) | 12.24 (12.23-12.25) | 12.32 (12.20-12.37) | +0.69 % | +0.90 % | +0.21 % |
| `e18_box_split` | `npp_resize` | kernel_only | 12.29 (12.29-12.30) | 12.29 (12.29-12.29) | 12.29 (12.29-12.29) | -0.00 % | -0.00 % | +0.00 % |
| `e18_box_split` | `fkl_resize` | whole_call | 14.56 (14.53-14.64) | 14.55 (14.51-14.57) | 14.53 (14.49-14.57) | -0.15 % | -0.22 % | -0.07 % |
| `e18_box_split` | `fkl_resize` | kernel_only | 14.41 (14.39-14.49) | 14.59 (14.55-14.60) | 14.59 (14.53-14.66) | +0.01 % | +1.25 % **exceeds** (apart) | +1.25 % **exceeds** (apart) |
| `e18_box_split` | `npp_box` | whole_call | 23.00 (22.97-23.01) | 23.02 (22.96-23.05) | 22.99 (22.90-23.03) | -0.14 % | -0.02 % | +0.12 % |
| `e18_box_split` | `npp_box` | kernel_only | 23.55 (23.55-23.55) | 23.55 (23.55-23.55) | 23.55 (23.55-23.55) | +0.00 % | +0.00 % | +0.00 % |
| `e18_box_split` | `fkl_box` | whole_call | 31.14 (31.06-31.14) | 31.18 (31.12-31.18) | 31.17 (31.16-31.19) | -0.03 % | +0.09 % (apart) | +0.13 % |
| `e18_box_split` | `fkl_box` | kernel_only | 31.76 (31.75-31.77) | 31.75 (31.74-31.76) | 31.75 (31.75-31.77) | +0.01 % | -0.01 % | -0.02 % |
| `e18_stencil` | `fkl_stencil_asym` | whole_call | 38.45 (38.25-38.56) | 40.14 (38.49-40.18) | 41.05 (40.82-41.73) | +2.25 % (apart) | +6.75 % **exceeds** (apart) **FLAG** | +4.40 % **exceeds** **FLAG** |
| `e18_stencil` | `fkl_stencil_asym` | kernel_only | 39.26 (39.16-39.26) | 40.58 (39.45-40.77) | 40.51 (40.39-41.03) | -0.16 % | +3.19 % **exceeds** (apart) **FLAG** | +3.36 % **exceeds** (apart) **FLAG** |
| `e18_stencil` | `fkl_stencil_halfpixel` | whole_call | 38.52 (38.39-38.58) | 38.57 (38.55-38.57) | 38.77 (38.72-38.85) | +0.52 % **exceeds** (apart) | +0.64 % **exceeds** (apart) | +0.12 % |
| `e18_stencil` | `fkl_stencil_halfpixel` | kernel_only | 39.79 (39.52-39.79) | 39.84 (39.81-39.86) | 39.94 (39.94-39.94) | +0.23 % **exceeds** (apart) | +0.38 % (apart) | +0.15 % (apart) |
| `e18_stencil` | `opencv_cuda` | whole_call | 173.72 (173.69-173.86) | 173.87 (173.77-173.93) | 173.88 (173.64-174.71) | +0.00 % | +0.09 % | +0.09 % |
| `e18_stencil` | `opencv_cuda` | kernel_only | 168.75 (168.56-169.10) | 168.73 (168.64-169.01) | 169.23 (168.69-169.27) | +0.30 % | +0.28 % | -0.01 % |
| `e18_stencil` | `npp` | whole_call | 121.31 (120.58-121.47) | 120.68 (120.57-120.87) | 121.24 (120.71-121.45) | +0.46 % | -0.06 % | -0.53 % |
| `e18_stencil` | `npp` | kernel_only | 119.87 (119.87-121.23) | 119.26 (119.25-120.52) | 119.63 (119.30-120.67) | +0.32 % | -0.20 % | -0.51 % |
| `e18_stencil` | `fkl_stencil_asym_once` | whole_call | 38.66 (38.65-38.67) | 38.66 (38.61-38.71) | 38.90 (38.90-38.98) | +0.63 % **exceeds** (apart) | +0.61 % **exceeds** (apart) | -0.02 % |
| `e18_stencil` | `fkl_stencil_asym_once` | kernel_only | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | +0.00 % (apart) | +0.00 % (apart) | +0.00 % |
| `e18_stencil` | `fkl_stencil_halfpixel_once` | whole_call | 38.80 (38.73-38.84) | 38.82 (38.78-38.83) | 39.02 (39.01-39.08) | +0.52 % **exceeds** (apart) | +0.57 % **exceeds** (apart) | +0.05 % |
| `e18_stencil` | `fkl_stencil_halfpixel_once` | kernel_only | 39.94 (39.94-39.94) | 39.94 (39.94-39.94) | 39.97 (39.96-39.99) | +0.08 % **exceeds** (apart) | +0.08 % **exceeds** (apart) | +0.00 % |

**Summary.**
- **B against A** (only the FKL pin differs; same window, same spacing): no number is flagged.
  - The rows with an unchanged kernel (`opencv_cuda`, `npp`, `npp_resize`, `npp_box`, `fkl_resize`): -0.15 % to +0.69 %, all within the spread.
  - The FKL rows that launch `linearFilterDPPKernel`: -0.33 % to +2.25 %. B is the slower side in 27 of the 30, and 18 exceed the spread, all of those with B slower:
    - `e18_box`: up to +1.78 % (`fkl_trunc_asym`, kernel_only);
    - `e18_box_split`: up to +1.55 % (`fkl_trunc_halfpixel`, kernel_only);
    - `e18_stencil`: up to +2.25 % (`fkl_stencil_asym`, whole_call, within the spread).
  - B's kernel is a different kernel (8 or 16 more SASS instructions). That this makes the difference is not shown.
- **A and B against old:** 4 numbers are flagged. Each is the first row its program times, on both sides:
  - `e18_box_split` `fkl_box_asym`: whole_call +8.13 % (A) and +8.44 % (B); kernel_only +3.46 % (B).
  - `e18_stencil` `fkl_stencil_asym`: whole_call +4.40 % (A) and +6.75 % (B); kernel_only +3.36 % (A) and +3.19 % (B).
  - Their spreads here reach 4.2 % (A's `e18_stencil` whole_call), against 0.1-0.8 % in the records.
- **What goes with the flags** (`analysis/first-row-clock.txt`):
  - In these runs the clock read 2,550 MHz in the first second of the timing, after the GPU had idled. The records' runs followed one another directly.
  - The one run that read 2,685 MHz throughout that second, `e18_stencil` quiet-run3-A (no wait before it), gave the record's values: 38.49 µs whole_call and 39.45 µs kernel_only, against the record's 38.45 and 39.26. Every other `e18_stencil` run gave 40.1-41.7 µs whole_call.
  - The first row of `e18_box` (`fkl_box_asym`) also read 2,550 MHz in that second here, and is within 1.6 % of old. Its record has 2,550 MHz readings before its timing too (`../e18-box/README.md`).
  - So the clock at the start of the timing is a candidate for the flagged first rows, as in `../e18-box-split/README.md`. It is not shown to be the cause.
- **The other numbers against old:** A/old -0.53 % to +2.13 % and B/old -0.43 % to +1.77 %; none is flagged.

## What this says and what not
- **What it says:** at the pin b10fc66, E18's FKL rows are within 2.3 % of their times at 24478f0 (B against A, same window), and none of them crosses the 3 % flag. The rows whose kernels did not change are within 0.7 %.
- **What it does not show:**
  - why B is slower in some FKL rows;
  - why the first rows are slower than in the records.
- **Not measured here:** the times with the records' spacing (runs back to back).
- **The records are unchanged.** Each has a dated note that points here.

## Files
- `window-script.sh`: the script that ran (sha256 359e1346…). `window/window-script.sh` is its copy.
- `binaries.sha256`: the six binaries (sha256 b20a729a…).
- `window/`: the complete output of the window:
  - `window-console.txt`, `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err` (empty), `load.log`;
  - per program and run, `window/<program>/<run>/`: the program's CSVs, `stdout.txt`, `stderr.txt` (empty), `exit_code.txt`, `load.txt` and `gpu-at-end.csv`.
- `analysis/` (CPU only, after the window):
  - `retime-report.txt`, `retime-summary.csv`, `retime-table.md` (the table above), `clock-by-run.txt` and `checks-vs-records.txt`, from `scripts-used/retime_report.py <window> ../ --out ... --csv ... --clocks ... --checks ... --md ...`;
  - `first-row-clock.txt`, from `scripts-used/first_row_clock.py <window>`. It holds the creation times of the timing CSVs, which exist only on this machine's file system.
- Every file is complete; nothing was filtered or left out.
