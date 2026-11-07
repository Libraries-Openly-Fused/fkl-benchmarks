# Cleanup window B, warm-pass diagnostic, 2026-11-07, 23:33:08-23:34:09: E4 at 4K under Nsight Compute, both builds

**Why it ran:** in window B, the warm 4K pass (`--cache-control none`) of the cleaned build moved 114.14 MiB for fkl_fused, against 83.72 MiB published. The diagnostic asks whether the reference build reproduces the published value, and how repeatable the warm single launch is. RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02, Nsight Compute 2026.3.1.0 (`ncu-version.txt`). The desktop session kept running.

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

**What ran:** the settings of window B: E4c 3840 x 2160; fkl_fused, opencv_percrop and npp; one implementation per process; E4_PROFILE=1; the same metrics; clock control base; pinned to cores 0-3,16-19.
- Both builds, warm (`--cache-control none`) and cold (the ncu default), two repetitions.
- Repetition 1: reference then cleaned. Repetition 2: cleaned then reference. Within a build: warm then cold.
- Folders: `{reference,cleaned}/ncu-{warm,cold}-4k-rep{1,2}`, each with its `e4_dram_bytes.csv` (`scripts/e4_ncu_summary.py`).
- `e4_4k_dram_all_passes.csv` lists every 4K pass of the day: window B and this diagnostic.
All 24 passes exited 0, with kernel counts 1, 7 and 9.

**Checks before the start** (the window refuses to start if one fails):
- the frozen-program guard (`refuse_blocked_binaries`) on every binary of both builds;
- step 0's record and the sha256 of every binary;
- the expected commit strings, without -dirty, inside the binaries;
- `busy_processes` (scripts/quiet_check.sh): no compiler, profiler, benchmark binary, window or runner script;
- no compute context on the RTX 4090 from a program outside /usr (the desktop's /usr/bin/ptyxis holds one);
- a new output folder.

**Findings** (also in `../cleanup-window-a/cleanup-windows-ab-report.md`, section "Warm-pass diagnostic"):
- **The reference build does not reproduce the published warm value.**
  - fkl_fused: reference 110.19 and 110.36 MiB, cleaned 110.14 and 109.53.
  - The cleanup does not change the warm numbers. The difference from the published 83.72 MiB is drift since the published passes of 2026-10-31.
  - Those passes were made with the first build of the program at FKL c8fcbe8. Its binary no longer exists.
- **Repeatability today** (five passes: window B plus these four):
  - warm: fkl_fused 109.53-114.14 MiB (4.2 %), opencv_percrop 0.9 %, npp 1.4 %;
  - cold: fkl_fused 86.47-107.11 MiB (20.5 %), opencv_percrop 1.7 %, npp 3.6 %.
  - Window B's 114.14 MiB has the same DRAM writes as the other warm passes (86.5 MiB) and 4 MiB more reads.
- **Why fkl_fused varies:**
  - It reads its frame (23.7 MiB) and writes a 108 MiB output, which is larger than the 72 MB (68.7 MiB) L2.
  - ncu counts only what is written back during the kernel: 60 to 87 MiB of the output, varying between passes, and the kernel time with it. The rest is still dirty in L2 when the kernel ends.
  - What it must move sooner or later is 131.7 MiB.
  - For opencv_percrop and npp, which move 10 to 15 times more, the same effect is a few percent.
- **Ratios against fkl_fused today:**
  - warm: OpenCV 11.4x to 11.8x, NPP 13.5x to 14.3x (published 14.1x and 16.8x);
  - cold: OpenCV 10.2x to 12.4x, NPP 12.0x to 15.4x (published 12.6x and 14.9x).

**Compared with** `../e4-preprocessing/ncu-cold/` and `../e4-preprocessing/ncu-warm/`.

**Files:**
- `binary-sha256.txt` and `build-info-{reference,cleaned}.txt`: from step 0.
- `window-script.sh` and `window-console.txt`: the script and its console.
- `environment-before.txt` and `environment-after.txt`: load and top processes, plus the git state of both checkouts.
- `gpu-info.csv`, and `load.log` (load every 5 s, logger on core 12 at nice 19).
- Every pass folder also has the `.err` and `.exit` file of each ncu process, and `load.txt`.
