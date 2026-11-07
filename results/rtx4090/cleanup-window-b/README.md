# Cleanup window B, 2026-11-07, 23:14:25-23:19:22: E4, reference against cleaned

This window followed window A (`../cleanup-window-a/`, which also holds the sequence, the step-0 record and the report). RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.

**The two builds** (both built with nvcc 13.4.92 for sm_89, GNU 15.2.0, Release, preset `local-rtx4090`, against FKL 64ad9195, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245):
- **reference:** `build/local-rtx4090` of the reference checkout, built on 2026-11-07 at 00:45 from fkl-benchmarks a5dd9fa, the sources before the cleanup. It was not rebuilt.
- **cleaned:** main 836a98e, the cleanup PRs #18 to #24. Step 0 built it on 2026-11-07, 23:08:10-23:09:59, compile only:
  - a clean detached worktree with the submodules at the reference checkout's commits;
  - the build info shows 836a98e without -dirty, and its other fields equal the reference's;
  - the cache options and compile flags are equal (E4 without the include directory handwritten-cuda, by design);
  - kernel SASS against the reference: NPP and CPU-time programs 47 of 47 identical, E9 5 of 5, E4 60 identical plus the 2 frame kernels renamed by design.
  - Its record is in `../cleanup-window-a/step0/`. `build-info-*.txt` and `binary-sha256.txt` in this folder are copies of it.

**Binaries (sha256):**

| program | reference (build/local-rtx4090, a5dd9fa) | cleaned (main 836a98e) |
|---|---|---|
| `e4_preprocessing_cu` | `06fe9e3408ef7e15bf17055a8175bcf21165a5a82d5f9dc5cae7084bf7fe0934` | `6e64d7e9c36fbfec9b43bad329a2c1c94ccda23c487da218ce3a2647a6f475c8` |

**What ran:**
1. **Three complete quiet sweeps** (E4a crops and E4c frames, every implementation of the build) per build, pinned to cores 0-3,16-19. Run 1 reference then cleaned, run 2 cleaned then reference, run 3 reference then cleaned.
   - Folders: `{reference,cleaned}/e4-preprocessing/quiet-run1..3`.
   - The reference sweep has 5 versions and took about 50 s. The cleaned sweep has 4 (no opencv_batch, removed in #23) and took about 43 s.
   - `e4_summary.csv` and `e4_speedups.csv` in each build folder come from `scripts/e4_summary.py`, in the published format: the median over the runs of the per-run ratio, with the smallest and largest run; npp is compared against fkl_fused_half_pixel.
2. **Nsight Compute at E4c 3840 x 2160, cleaned build only**, for fkl_fused, opencv_percrop and npp.
   - Settings: one implementation per process; E4_PROFILE=1 E4_ONLY=<v> E4_EXPERIMENTS=E4c E4_FRAMES_ONLY=3840; `--csv --page raw` with dram__bytes, dram__bytes_read, dram__bytes_write and gpu__time_duration (avg, max, min, sum); clock control base.
   - `cleaned/e4-preprocessing/ncu-cold-4k`: the ncu default, caches flushed before every kernel.
   - `cleaned/e4-preprocessing/ncu-warm-4k`: `--cache-control none`.
   - `e4_dram_bytes.csv` in each comes from `scripts/e4_ncu_summary.py`.
   - `e4_4k_dram_all_passes.csv` (this folder and the diagnostic's) lists read, write and total DRAM bytes and the kernel time of every 4K pass of the day: this window and the diagnostic, both builds.

All runs exited 0, with no INCOMPLETE line and no -nan. The kernel counts are 1, 7 and 9, as published.

**Checks before the start** (the window refuses to start if one fails):
- the frozen-program guard (`refuse_blocked_binaries`) on every binary of both builds;
- step 0's record and the sha256 of every binary;
- the expected commit strings, without -dirty, inside the binaries;
- `busy_processes` (scripts/quiet_check.sh): no compiler, profiler, benchmark binary, window or runner script;
- no compute context on the RTX 4090 from a program outside /usr (the desktop's /usr/bin/ptyxis holds one);
- a new output folder.

**Compared with** `../e4-preprocessing/` (quiet-run1..3, ncu-cold, ncu-warm), made with FKL c8fcbe8. Its ncu passes were taken with the first build of the program.

In short (details in `../cleanup-window-a/cleanup-windows-ab-report.md`):
- **Drift:** within, except three host-time points of about 2.3 us.
- **Cleanup effect by design:** E4a fkl_fused gpu_mean_ms is +8 % (batch 10) to +50 % (batch 300). FKL's crop views, read and operations are now built in the timed call, about 30 ns per crop. Speedup against opencv_percrop: batch 100 152.5x -> 111.7x, batch 300 207.3x -> 138.1x (reference now -> cleaned).
- **Unchanged:** opencv_percrop, npp and every E4c frame are within.
- **DRAM bytes at 4K:**
  - Cold is within 2 % of published.
  - Warm changed, but by drift, not by the cleanup: the reference build gives the same today, as the diagnostic shows (`../cleanup-window-b-warm-diagnostic/`).
  - fkl_fused's value is not repeatable between passes, because its 108 MiB output is larger than the L2 and ncu counts only what is written back during the kernel. It must move at least 131.7 MiB (frame read 23.7 plus output 108.0) sooner or later.
- **Agreement:** the 132 agreement rows that both builds have are identical to the published runs.

**Files:**
- `binary-sha256.txt` and `build-info-{reference,cleaned}.txt`: from step 0.
- `cleaned/e4-preprocessing/ncu-version.txt`.
- `window-script.sh` and `window-console.txt`: the script and its console.
- `environment-before.txt` and `environment-after.txt`: load and top processes, plus the git state of both checkouts.
- `gpu-info.csv`, and `load.log` (load every 5 s, logger on core 12 at nice 19).
- Every run folder also has `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` and `gpu-at-end.csv`.
