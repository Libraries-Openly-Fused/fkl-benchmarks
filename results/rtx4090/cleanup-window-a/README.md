# Cleanup window A, 2026-11-07, 23:11:59-23:14:25: NPP, CPU time and E9, reference against cleaned

Part of the re-measurement after the baseline cleanup of the CUDA programs (Oscar's decisions of 2026-11-07, coordinator session).
- The sequence: step 0 (23:08:10-23:09:59, compile only), 120 s of idle, this window, then window B (`../cleanup-window-b/`). Files: `sequence-script.sh`, `sequence-console.txt`.
- RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.

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
| `benchmark_npp_batchresize_x_split3D_cu` | `3c2e543ac6122c10203e6dada9ede5a4aa13adee38594ba0b18694cbb6c70b6e` | `44b06dcf92ffcd7b70ced3ed50458b5a3401f0e5bb3792a8bda4c834433df32b` |
| `benchmark_CPU_NPP_vs_FastNPP_cu` | `c8f420f603022becbc59515eebeeee402b63c356e4a13e75edaf6864afc81355` | `50bf1ed262d39d66310e317ea31348cc37a8fe4357483bde2bf7ee678faad1fa` |
| `benchmark_CPU_OpenCV_vs_cvGS_cu` | `6dd16841a7ba9c02224228e3db9b35c6cdc7ee6e0dd37eff5807319f5bf15ba8` | `90aa59d9e94257d9027df3366f8fd439cdca36fcf11b38c7de79af1313f85e00` |
| `e9_cross_library_cu` | `2342b32882e7c14762be50b54948e33326f4a8dda66be5f09db11cbebdb423c0` | `36f2a5b4ccfc8c5c5fc5de1092cb73e19cecd11c693ba3eeb07c572f0f26dfc4` |

**What ran:** three runs of each program on each build, the two builds alternating.
- The build order flipped every run: run 1 reference then cleaned, run 2 cleaned then reference, run 3 reference then cleaned. Each program's pair of runs was back to back.
- `{reference,cleaned}/e9-cross-library/run0..2`: `e9_cross_library_cu`, with E9_RUN = 0, 1, 2 rotating the order of the variants. Pinned to cores 0-3,16-19, as in the published window.
- `{reference,cleaned}/e0-npp/run1..3`: `benchmark_npp_batchresize_x_split3D_cu`, both tests in one process.
- `{reference,cleaned}/e0-cpu-time-npp/run1..3`: `benchmark_CPU_NPP_vs_FastNPP_cu`.
- `{reference,cleaned}/e0-cpu-time-opencv/run1..3`: `benchmark_CPU_OpenCV_vs_cvGS_cu`.
- The three NPP and CPU-time programs ran unpinned, each in its own working directory, as `scripts/run_e0.sh` runs them.
- Then `{reference,cleaned}/e9-cross-library/nsys`: one nsys pass per build, one call of each variant (E9_NSYS=1 E9_SEEDS=1), with the kernel count from the sqlite export in `kernels.txt`. Both builds give 6 launches in the range, as published.
  - `e9.sqlite`, the export of `e9.nsys-rep`, is not committed, as in the published folder.

All 24 runs exited 0, with no INCOMPLETE line, no -nan and no failed check.

**Checks before the start** (the window refuses to start if one fails):
- the frozen-program guard (`refuse_blocked_binaries`) on every binary of both builds;
- step 0's record and the sha256 of every binary;
- the expected commit strings, without -dirty, inside the binaries;
- `busy_processes` (scripts/quiet_check.sh): no compiler, profiler, benchmark binary, window or runner script;
- no compute context on the RTX 4090 from a program outside /usr (the desktop's /usr/bin/ptyxis holds one);
- a new output folder.

**Compared with** the published folders:
- `../e0-npp-after45/` and `../e0-cpu-time-npp-after45/`: 3 runs each, window of 2026-11-06, FKL c8fcbe8;
- `../e0-cpu-time-opencv/`: one run, 2026-10-30;
- `../e9-cross-library/`: 3 runs, 2026-11-06, FKL c8fcbe8.

**Report:** `cleanup-windows-ab-report.md` covers windows A and B and the warm-pass diagnostic. `cleanup-windows-ab-report.csv` has every number. `cleanup_ab_report.py` wrote both and only reads the result folders.
- For each number: published, reference now and cleaned, each the median over the runs with its run-to-run spread.
- drift = reference now against published (pins, time, environment); cleanup effect = cleaned against reference now.
- Each change is judged against the spreads and the 3 % (GPU-bound) or 5 % (host-bound) threshold.

In short, for this window:
- **Drift:** within the thresholds, except the single-run noise of `e0-cpu-time-opencv`.
- **Cleanup effect by design:** the CPU-time cvGS host time is +5 % (batch 2) to +84 % (batch 152), because its GpuMat ROIs are now made in every iteration (about 25 ns per crop). Its speedup at batch 152 goes from 431.3x to 233.8x.
- **Not by design:** FastNPP is 1 to 4 % faster. The probable cause, not isolated: its descriptor arrays are now stack arrays written in the call, instead of pinned host arrays filled once.
- **Unchanged:** NPP, E9 and the No_CPU_Overhead test are within the spread.
- **Agreement:** every agreement row (NPP, CPU-time NPP and E9) is identical to the published run of the same index.

**Files:**
- `binary-sha256.txt` and `build-info-{reference,cleaned}.txt`: from step 0.
- `step0/`: step 0's console, its checks (`sass-*.txt`, the empty diffs `build-info.diff`, `cache.diff`, `flags-*.diff`), the configure and build logs, and its script.
- `window-script.sh` and `window-console.txt`: the script and its console.
- `environment-before.txt` and `environment-after.txt`: load and top processes, plus the git state of both checkouts.
- `gpu-info.csv`, and `load.log` (load every 5 s, logger on core 12 at nice 19).
- Every run folder also has `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` and `gpu-at-end.csv`.
