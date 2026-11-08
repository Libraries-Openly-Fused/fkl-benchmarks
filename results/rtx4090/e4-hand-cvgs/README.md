# E4 window e4-hand-cvgs, 2026-11-08, 12:13:16-12:15:53: the hand-written kernel of E1c and cvGPUSpeedup (RTX 4090)

These are two new series for Figure 15, from the E4 program (`opencv-cuda/batched-preprocessing/e4_preprocessing.cu`, #34):
- FKL against the hand-written fused kernel of E1c;
- cvGPUSpeedup against OpenCV-CUDA called per crop.

The window also repeats E4's existing series. RTX 4090 = nvidia-smi index 1 (00000000:09:00.0), CUDA 13.4.3, driver 610.43.02, Ubuntu 26.04.1 LTS, Linux 7.0.0-38-generic. The desktop session kept running.

**The build.** Main 2dac1c8, the merge of #34. Step 0 built it on 2026-11-08, 12:08:28-12:09:48, compile only; its record is in `step0/`.
- It used a clean detached worktree (`worktrees/fkl-benchmarks-window-f`), with the submodules at the gitlinks, which equal the reference checkout's: FKL 64ad9195, cvGPUSpeedup 6d26c65, FastNPP a1b5e245.
- Preset `local-rtx4090`, nvcc 13.4.92 for sm_89, GNU 15.2.0, Release.
- The build info shows 2dac1c8 without -dirty; its other fields, and the cache options, equal window B's (`../cleanup-window-b/`, main 836a98e).
- The compile flags equal window B's except the two include directories of cvGPUSpeedup, which the target links since #34.
- SASS by demangled name:
  - against the head of #34 (52ae33a): 152 of 152 kernels identical;
  - against window B's binary: its 62 kernels identical, plus 90 new ones (30 `handPreproc<N>`, 30 `handPreprocPadded<N, PAD>`, 30 cvGPUSpeedup kernels).
- `e4_preprocessing_cu`: sha256 `8b56ffc282bd3d03f0b3014e0cfff6c4c49ca51ce6e59d010d7fbb21db19f01b` (`step0/binary-sha256.txt`).

**The versions.** E4c (whole frames) has the four existing versions: fkl_fused, fkl_fused_half_pixel, opencv_percrop, npp. E4a (crops) has those four, then five new ones in this order:
- **`fkl_fused_late`**: exactly the fkl_fused call, timed immediately before hand_batch.
- **`hand_batch`**: E1c's `handPreproc<BATCH>` (`handwritten-cuda/e1_preproc.cu`), copied unchanged.
  - Its parameter block holds the crop pointers, 8 bytes per crop. The constants of the chain are compile-time constants.
  - It has its own `cudaMallocPitch` frame and its own `cudaMalloc` output, and uses no FKL object.
  - Every call builds the parameter block from the crop positions, launches, and checks the launch.
  - Its SASS equals E1c's kernel's (the compile checks of #34).
- **`hand_padded`**: E1c's diagnostic, the same arithmetic with its parameter block padded to the size of FKL's read IOp.
  - hand_padded - hand_batch therefore measures the larger launch-parameter block.
  - It is not FKL's build cost: the hand build writes only the descriptors, not the padding.
- **`fkl_fused_after`**: the fkl_fused call again, timed immediately after hand_padded. With fkl_fused_late it brackets the hand-written kernels.
- **`cvgs_fused`**: cvGPUSpeedup in the form of the call in `opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu`, with E4's input and constants.
  - The chain: an array of GpuMat ROIs -> `cvGS::resize` -> `cvtColor` -> `multiply` -> `subtract` -> `divide` -> `split`.
  - `cvGS::executeOperations` is called with its default arguments, which request thread fusion. Thread fusion cannot take effect in this chain, because the batched resize read and `TensorSplit` do not support it. The SASS is the same as with `executeOperations<false>` (`tf-check/`).

**What the ratios measure.** Every time is E4's: CUDA events around one call of the whole implementation, 100 calls after one warm-up. Every side builds its batch inside the interval, because the crop locations may change on every image.
- **hand_batch / FKL is a whole-call ratio:** build + dispatch + kernel. For claims about the kernels themselves, E1c (Table III) remains the source.
- **The FKL time** for the hand ratios is the mean of fkl_fused_late and fkl_fused_after, the two FKL calls next to the hand-written kernels; the ratio against fkl_fused is given too.

**What ran** (`window-script.sh`, `window-console.txt`).
1. A warm-up, untimed, not used: one sweep of batch 300 only (`E4_EXPERIMENTS=E4a E4_BATCHES_ONLY=300`), 4 s, in `warmup/`. The GPU started at 210 MHz (P8). Window B's runs followed other GPU work.
2. Three complete quiet sweeps, back to back, in `quiet-run1..3`: E4a (30 batch sizes, 10 to 300, 9 versions) and E4c (3 frames, 4 versions).
   - Pinned to cores 0-3,16-19, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, as in window B.
   - 50 s each, all exit 0, with no INCOMPLETE line and no -nan.
3. Throughout the window:
   - `gpu_clocks.csv`: nvidia-smi every 100 ms on core 14, 1,534 samples; `gpu_clocks.err` is empty.
   - `load.log`: the load logger every 5 s on core 12.

No reference build ran, because the new versions have no older one. The existing series are compared with window B's cleaned runs (`../cleanup-window-b/cleaned/e4-preprocessing/`). Every batch now ends with the five new versions, so their neighbours changed:
- the next batch's fkl_fused follows cvgs_fused (in window B it followed npp);
- E4c's first frame follows batch 300's cvgs_fused.

**Checks before the start:** the frozen-program guard; step 0's record and the sha256 of the binary; the commit string inside it, without -dirty; `busy_processes` (nothing running); no compute context on the RTX 4090 from a program outside /usr; a new output folder.

**Report:** `e4-hand-cvgs-report.md`, with every number in `e4-hand-cvgs-report.csv`. `e4_hand_cvgs_report.py` wrote both and only reads the result folders. Its findings section is `e4-hand-cvgs-findings.md`.
- **hand_batch / FKL** (adjacent FKL time) is 0.92 at batch 10, 0.74 at 50, 0.67 at 100, 0.64 at 150, 0.60 at 300.
  - FKL's call builds its crop views and operations on the host (host_enqueue_us 2.1 to 12.3 us), and the GPU waits for it inside the interval. The hand-written call needs 1.6 to 1.8 us.
  - The position has no visible effect: from batch 20 on, fkl_fused_late and fkl_fused_after are within 1.4 % of fkl_fused.
- **hand_padded / hand_batch** is 1.01 to 1.10: the larger parameter block, +1.6 us at batch 300.
- **cvGPUSpeedup / OpenCV-CUDA per crop:** 33x at batch 10, 81x at 50, 104x at 100, 116x at 150, 128x at 300. cvgs_fused is 1.6 to 8.2 % slower than fkl_fused.
- **Agreement:** all 846 rows pass their own reference. hand_batch, hand_padded, cvgs_fused and both FKL controls are bitwise equal to fkl_fused at every batch in every run.
- **The existing series against window B:**
  - opencv_percrop, npp and every E4c frame are within.
  - FKL's host times are 5 % higher (median) from batch 50 on; its GPU times move by about +2 % with them.
  - Probable cause, not isolated: FKL's host code compiled differently in the changed translation unit. Its device code is identical. A diagnostic that would isolate it is in the findings.

`e4_summary.csv` and `e4_speedups.csv` come from `scripts/e4_summary.py`, in the published format: the median over the runs, with the smallest and largest run. e4_speedups.csv has only the published baselines: opencv_percrop against fkl_fused, npp against fkl_fused_half_pixel.

**Files:**
- `step0/`: step 0's record.
  - its console and script, the configure and build logs, `main-commit.txt`, `binary-sha256.txt`;
  - the build info of both builds (`build-info-window-{b,f}.txt`) and their diff;
  - `cache.diff`;
  - the flags and CUDA include directories of both builds, with their diffs, also without cvGPUSpeedup's directories;
  - `sass-vs-52ae33a.txt` and `sass-vs-window-b.txt`.
  - The copies of the binaries it compared are not committed.
- `tf-check/`: the thread-fusion check, with its own README.
- `warmup/`, `quiet-run1..3/`. Each holds:
  - `e4_preprocessing.csv`, `e4_preprocessing_agreement.csv`, `e4_preprocessing_pairs.csv`;
  - `stdout.txt`, `stderr.txt`, `exit_code.txt`;
  - `load.txt` (its start and end), `gpu-at-end.csv`.
- `window-script.sh`, `window-console.txt`, `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log`.
- `e4-hand-cvgs-report.md`, `e4-hand-cvgs-report.csv`, `e4-hand-cvgs-findings.md`, `e4_hand_cvgs_report.py`, `e4_summary.csv`, `e4_speedups.csv`.
