# E4 window H (e4-build-once), 2026-11-08, 14:18:43-14:21:19: the build-once series of Figure 15 (RTX 4090)

This is the E4 program (`opencv-cuda/batched-preprocessing/e4_preprocessing.cu`) at main f534a38, the merge of #37. Against the window e4-hand-roi (`../e4-hand-roi/`) it adds the **build-once variants**.
- Each variant builds its crop views (or its ROIs, or its parameter block) and every operation **once, before the timing**. Every timed call is then only the launch path, so its interval holds the launch path and the kernel.
- The build is excluded on purpose: Figure 15 leaves out the host-side cost, because the paper does not describe the builder API.
- **The build-once variants are valid only while the crops (their positions and sizes) do not change.** What is built once holds every crop's position and size, and for hand_roi_once also its pitch and resize factors.
- This is the condition of FastNPP's variant without CPU overhead (`npp/batchresize/benchmark_npp_batchresize_x_split3D.cu`, `test_npp_batchresize_x_split3D_No_CPU_Overhead`), whose crop size is a compile-time constant.
- The baselines (opencv_percrop, npp) are timed as a user calls them.
- Every existing version runs unchanged. The whole-call record stays `../e4-hand-roi/`.

RTX 4090 = nvidia-smi index 1 (00000000:09:00.0), CUDA 13.4.3, driver 610.43.02, Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic. The desktop session kept running.

**The build.** Step 0 built main f534a38 on 2026-11-08, 14:14:38-14:16:35, compile only; its record is in `step0/`.
- It used a clean detached worktree (`worktrees/fkl-benchmarks-window-h`), with the submodules at the gitlinks, which equal the reference checkout's: FKL 64ad9195, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- Preset `local-rtx4090`, nvcc 13.4.92 for sm_89, GNU 15.2.0, Release.
- The build info shows f534a38 without -dirty. Its other fields, the cache options, the compile flags and the CUDA include directories equal those of the e4-hand-roi build (469aa6d). The 27 warning lines are the same set.
- SASS by demangled name: 182 of 182 kernels are identical against the head of #37 (8dc722d) and 182 of 182 against 469aa6d. **No new kernel:** the build-once variants launch the kernels of the existing versions.
- `e4_preprocessing_cu`: sha256 `1387fcb9633403085e128e0287f6f47c8d8b03275dbc17c576fb89e5562fb5a3` (`step0/binary-sha256.txt`).

**The versions.**
- E4c (whole frames) has fkl_fused, fkl_fused_half_pixel, opencv_percrop and npp.
- E4a (crops) first has the ten versions of e4-hand-roi in its order: fkl_fused, fkl_fused_half_pixel, opencv_percrop, npp, fkl_fused_late, hand_batch, hand_padded, hand_roi, fkl_fused_after, cvgs_fused.
- The build-once variants come after them, in this order:
  - **`cvgs_fused_once`**: cvgs_fused with the ROIs and the cvGS operations built once. Every call is `cvGS::executeOperations` with them, with its default arguments as in cvgs_fused.
  - **`fkl_fused_half_pixel_once`**: fkl_fused_half_pixel with the crop views and every operation built once. Every call is `fk::executeOperations` with them.
  - **`fkl_fused_once`**: the same for fkl_fused, timed immediately before hand_roi_once.
  - **`hand_roi_once`**: hand_roi with its CPU function, `handRoiParameters`, run once. Every call is the launch with that parameter block and `cudaGetLastError`. It has its own copy of the frame and its own output, made as hand_roi's.
  - **`fkl_fused_once_after`**: the fkl_fused_once call again, timed immediately after hand_roi_once. The two FKL calls bracket hand_roi_once.

**What the ratios measure.** Every time is E4's: CUDA events around one call, 100 calls after one warm-up.
- For the build-once variants the call is the launch path alone.
- When the GPU waits for a launch, the interval also holds the host's dispatch of that launch.
- The FKL time of the hand line is the mean of fkl_fused_once and fkl_fused_once_after.

**What ran** (`window-script.sh`, `window-console.txt`), with the settings of the window e4-hand-roi:
1. A warm-up, untimed, not used: E4a batch 300 only, 4 s, in `warmup/`. The GPU started at 210 MHz (P8).
2. Three complete quiet sweeps, back to back, in `quiet-run1..3`: E4a (30 batch sizes, 15 versions) and E4c (3 frames, 4 versions).
   - Pinned to cores 0-3,16-19, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
   - 49-50 s each, all exit 0, empty stderr, 462 rows each, with no INCOMPLETE line and no -nan.
3. Throughout the window:
   - `gpu_clocks.csv`: nvidia-smi every 100 ms on core 14, 1,526 samples; `gpu_clocks.err` is empty. The runs ran at 2,685 MHz.
   - `load.log`: the load logger every 5 s on core 12.

**Checks before the start:** the frozen-program guard; step 0's record and the sha256 of the binary; the commit string inside it, without -dirty; `busy_processes` (nothing running); no compute context on the RTX 4090 from a program outside /usr; a new output folder. The same checks also ran as a dry check (`GUARD_ONLY=1`) after step 0.

**Report:** `e4-build-once-report.md`, with every number in `e4-build-once-report.csv`. `e4_build_once_report.py` wrote both and only reads the result folders. Its findings section is `e4-build-once-findings.md`.
- **The build-once series**, at batch 10, 150 and 300:
  - opencv_percrop / fkl_fused_once: 37.7x, 176.4x, 206.4x;
  - npp / fkl_fused_half_pixel_once: 37.4x, 179.5x, 210.8x;
  - opencv_percrop / cvgs_fused_once: 37.4x, 175.2x, 205.4x.
- **hand_roi_once / FKL-once** is 1.003 at batch 10, then 0.941 to 0.977 from batch 20 on (0.954 at 300). With the build excluded, FKL's call is within 2 to 6 % of the hand-written kernel's. With the build inside the call (e4-hand-roi), the ratio was 0.96 to 0.66.
- **The build** is 9 to 32 % of FKL's whole-call interval: fkl_fused_once / fkl_fused is 0.907 at batch 10 and 0.680 at 300.
- **Agreement:** all 1,386 rows pass their own reference. The five build-once pairs are bitwise equal in 90 of 90.
- **The existing versions against e4-hand-roi:**
  - The versions that are not FKL stay within +-0.5 %.
  - FKL's four whole-call versions are faster: gpu_mean_ms -1.5 to -1.8 %, host times -3.8 to -4.7 % (medians over batches 50 to 300). That gives 62 of the 63 flags.
  - Probable cause, not isolated: the binary composition, as in the diagnostic F2. The FKL call's executeOperations is inlined in 469aa6d's binary and out of line in this one, where the whole-call and the build-once calls share it. See the findings.

`e4_summary.csv`, `e4_speedups.csv` and `e4_speedups_once.csv` come from `scripts/e4_summary.py` at f534a38. e4_speedups.csv has the published baselines, and e4_speedups_once.csv the build-once variants against the same baselines.

**Files:**
- `step0/`: step 0's record.
  - its console and script, the configure and build logs, `main-commit.txt`, `binary-sha256.txt`;
  - the build info of both builds (`build-info-window-{g,h}.txt`) and their diff;
  - `cache.diff`;
  - the flags and CUDA include directories of both builds, with their diffs;
  - `sass-vs-8dc722d.txt`, `sass-vs-469aa6d.txt`.
  - The copies of the binaries it compared are not committed.
- `warmup/`, `quiet-run1..3/`. Each holds:
  - `e4_preprocessing.csv`, `e4_preprocessing_agreement.csv`, `e4_preprocessing_pairs.csv`;
  - `stdout.txt`, `stderr.txt`, `exit_code.txt`;
  - `load.txt` (its start and end), `gpu-at-end.csv`.
- `window-script.sh`, `window-console.txt`, `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log`.
- `e4-build-once-report.md`, `e4-build-once-report.csv`, `e4-build-once-findings.md`, `e4_build_once_report.py`, `e4_summary.csv`, `e4_speedups.csv`, `e4_speedups_once.csv`.
