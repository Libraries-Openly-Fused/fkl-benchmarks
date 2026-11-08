# E4 window G (e4-hand-roi), 2026-11-08, 12:40:20-12:42:56: the final RTX 4090 data of Figure 15

This is the E4 program (`opencv-cuda/batched-preprocessing/e4_preprocessing.cu`) at main 469aa6d, the merge of #35. Against the window e4-hand-cvgs (`../e4-hand-cvgs/`) it adds `hand_roi`, the hand-written kernel for crop ROIs that change on every iteration, sizes included. **hand_roi / FKL is the hand line of Figure 15**; `hand_batch` and `hand_padded` stay as repository diagnostics.

RTX 4090 = nvidia-smi index 1 (00000000:09:00.0), CUDA 13.4.3, driver 610.43.02, Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic. The desktop session kept running.

**The build.** Main 469aa6d. Step 0 built it on 2026-11-08, 12:36:54-12:38:21, compile only; its record is in `step0/`.
- It used a clean detached worktree (`worktrees/fkl-benchmarks-window-g`), with the submodules at the gitlinks, which equal the reference checkout's: FKL 64ad9195, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- Preset `local-rtx4090`, nvcc 13.4.92 for sm_89, GNU 15.2.0, Release.
- The build info shows 469aa6d without -dirty. Its other fields, the cache options, the compile flags and the CUDA include directories equal those of the e4-hand-cvgs build (2dac1c8).
- SASS by demangled name:
  - against the head of #35 (0f96c4f): 182 of 182 kernels identical;
  - against 2dac1c8: its 152 kernels identical, plus the 30 `handPreprocRoi<N>` of hand_roi.
- The cvGPUSpeedup kernels are among the identical ones, so the thread-fusion check of e4-hand-cvgs (`../e4-hand-cvgs/tf-check/`) applies unchanged.
- `e4_preprocessing_cu`: sha256 `121a088fd6ad888103246a267f48814b384a6254dae706958c577e7aead0588f` (`step0/binary-sha256.txt`).

**The versions.** E4c (whole frames) has fkl_fused, fkl_fused_half_pixel, opencv_percrop and npp. E4a (crops) has those four, then, in this order:
- **`fkl_fused_late`**: exactly the fkl_fused call, timed immediately before hand_batch.
- **`hand_batch`**: E1c's `handPreproc<BATCH>`, copied unchanged, with 8 bytes of crop pointers per crop and compile-time constants.
- **`hand_padded`**: E1c's diagnostic, with its parameter block padded to FKL's read IOp. hand_padded - hand_batch is the larger launch-parameter block, not FKL's build cost.
- **`hand_roi`**: the hand-written kernel with its CPU function, `handRoiParameters`.
  - The CPU function is plain C++ and the CUDA runtime, with no library type or call. In every call it takes the crop rectangles (x, y, width, height).
  - For every crop it computes the pointer into the frame, the width, height and pitch, and the resize factors, written out from FKL's per-crop formula: `float(1.0 / (double(64) / width))`, and the same for y with 128 and the height.
  - It writes them into the parameter block, 32 bytes per crop: 344, 4,824 and 9,624 bytes at batch 10, 150 and 300.
  - The kernel, `handPreprocRoi<BATCH>`, reads these values per crop and does E1c's arithmetic. The output size and the constants of the chain stay compile-time constants.
  - It has its own `cudaMallocPitch` frame and `cudaMalloc` output (those of hand_batch), checks the launch with `cudaGetLastError`, and uses no FKL object.
- **`fkl_fused_after`**: the fkl_fused call again, timed immediately after hand_roi. With fkl_fused_late it brackets the three hand-written kernels.
- **`cvgs_fused`**: cvGPUSpeedup in the form of the call in `opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu`.
  - `cvGS::executeOperations` is called with its default arguments, which request thread fusion. Thread fusion cannot take effect in this chain: the SASS is the same as with `<false>`.

**What the ratios measure.** Every time is E4's: CUDA events around one call of the whole implementation, 100 calls after one warm-up, and every side builds its batch inside the interval.
- **hand_roi / FKL is a whole-call ratio:** build + dispatch + kernel. For claims about the kernels themselves, E1c (Table III) remains the source.
- **The FKL time** for the hand ratios is the mean of fkl_fused_late and fkl_fused_after; the ratio against fkl_fused is given too.

**What ran** (`window-script.sh`, `window-console.txt`), with the settings of the window e4-hand-cvgs:
1. A warm-up, untimed, not used: E4a batch 300 only, 4 s, in `warmup/`. The GPU started at 210 MHz (P8).
2. Three complete quiet sweeps, back to back, in `quiet-run1..3`: E4a (30 batch sizes, 10 versions) and E4c (3 frames, 4 versions).
   - Pinned to cores 0-3,16-19, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
   - 50-51 s each, all exit 0, with no INCOMPLETE line and no -nan.
3. Throughout the window:
   - `gpu_clocks.csv`: nvidia-smi every 100 ms on core 14, 1,559 samples; `gpu_clocks.err` is empty. The runs ran at 2,685-2,700 MHz.
   - `load.log`: the load logger every 5 s on core 12.

**Checks before the start:** the frozen-program guard; step 0's record and the sha256 of the binary; the commit string inside it, without -dirty; `busy_processes` (nothing running); no compute context on the RTX 4090 from a program outside /usr; a new output folder.

**Report:** `e4-hand-roi-report.md`, with every number in `e4-hand-roi-report.csv`. `e4_hand_roi_report.py` wrote both and only reads the result folders. Its findings section is `e4-hand-roi-findings.md`.
- **hand_roi / FKL** (adjacent FKL time) is 0.960 at batch 10, 0.734 at 50, 0.692 at 100, 0.674 at 150 and 0.659 at 300.
  - hand_roi's call needs 1.6 to 2.8 us of host time, FKL's 2.0 to 12.3 us: FKL builds its crop views and operations on the host, inside the interval.
  - hand_roi / hand_batch is 0.99 to 1.09: the CPU function over every crop, and the larger parameter block.
- **The position has no visible effect:** from batch 20 on, the two controls are within 0.8 % of fkl_fused.
- **cvGPUSpeedup / OpenCV-CUDA per crop:** 33.4x at batch 10, 79.5x at 50, 105.3x at 100, 113.8x at 150 and 131.0x at 300.
- **Agreement:** all 936 rows pass their own reference. hand_roi, hand_batch, hand_padded, cvgs_fused and both FKL controls are bitwise equal to fkl_fused at every batch in every run.
- **Every version against e4-hand-cvgs:** the medians are within +-0.7 %. FKL's host times are unchanged, so hand_roi's kernels did not move FKL's host code.
  - The 11 flags are all cvgs_fused, at scattered batches and in both directions. They are a run-level effect: runs 1 and 3 are 5-6 % above run 2, which equals e4-hand-cvgs. The probable cause is not isolated; see the findings.

`e4_summary.csv` and `e4_speedups.csv` come from `scripts/e4_summary.py`, in the published format. e4_speedups.csv has only the published baselines.

**Files:**
- `step0/`: step 0's record.
  - its console and script, the configure and build logs, `main-commit.txt`, `binary-sha256.txt`;
  - the build info of both builds (`build-info-window-{f,g}.txt`) and their diff;
  - `cache.diff`;
  - the flags and CUDA include directories of both builds, with their diffs;
  - `sass-vs-0f96c4f.txt`, `sass-vs-2dac1c8.txt`.
  - The copies of the binaries it compared are not committed.
- `warmup/`, `quiet-run1..3/`. Each holds:
  - `e4_preprocessing.csv`, `e4_preprocessing_agreement.csv`, `e4_preprocessing_pairs.csv`;
  - `stdout.txt`, `stderr.txt`, `exit_code.txt`;
  - `load.txt` (its start and end), `gpu-at-end.csv`.
- `window-script.sh`, `window-console.txt`, `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log`.
- `e4-hand-roi-report.md`, `e4-hand-roi-report.csv`, `e4-hand-roi-findings.md`, `e4_hand_roi_report.py`, `e4_summary.csv`, `e4_speedups.csv`.
