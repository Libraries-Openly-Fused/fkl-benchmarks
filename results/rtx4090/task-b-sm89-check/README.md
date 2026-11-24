# Task B: FKL PR #433 on the RTX 4090 (sm_89)

FKL PR #433 is AMD's draft, which only Oscar merges. It batches LinearFilterDPP, MedianFilterDPP and the four Quad DPPs, with each image's size taken from the read. It also:
- removes width and height from their details;
- adds a floatN accumulator;
- gives the median per-channel min/max;
- adds num_elems_y/z to the write operations.

This folder holds the CUDA check, which compares the head with its base:
- **head:** 4b11bc6892a79d9481406da7903138ca567e2c6a (branch LTS-C++17-batched-filter-dpps, 6 commits, 19 files);
- **base:** 24478f0aa11dc3f12feea83178f7fde828322fb0, the FKL pin of this repository.

**Note, 2026-11-25:** #433 was merged by Oscar on 2026-11-24 at 22:02:02Z as `b10fc6609426bd18020b9213c7196b528fc677a8`, whose tree (`62024145`) is the tree of 4b11bc6, the head checked here. The pin move is fkl-benchmarks #60 (`../fkl-pr433-pin-check/`).

Nothing is timed in this task. This repository's pin does not change.

**State.** Everything is done:
- the compile step, 2026-11-24 21:23:37-21:41:29 +01:00;
- the device-code comparison;
- the GPU window, granted by the coordinator for 724cdfc and run 21:50:01-21:51:02 +01:00 (`window/`).

**Results.**
- Everything compiles for sm_89.
- Only the kernels of the DPPs that #433 changes have different SASS.
- Every FKL filter test passes: 7 of 7 at 24478f0, 10 of 10 at 4b11bc6.
- The outputs of the five programs are the same before and after, bit for bit: every check and agreement row, and every E19 dump. The one CSV that differs is the kernel-attribute table of e18_box, whose details argument is 8 B smaller, the two ints #433 removes.

## Step 2: the builds (compile only)

No GPU is visible to any process of this step (`CUDA_VISIBLE_DEVICES` is empty). Every build names sm_89 explicitly.
- Toolchain: nvcc 13.4.92 (toolchains/cuda-13.4.3), cmake 4.2.3, g++ 15.2.0.
- The compilers run on cores 8-15,24-31 at nice 10, with TMPDIR on disk.
- The script is `scripts-used/compile-step2.sh`, and its console is `compile/step2-console.txt`.

### A. FKL's test matrix (tests/ and utests/), at 24478f0 and at 4b11bc6

FKL's CMake runs out of source, in Release, with `-DCUDA_ARCH=89`. FKL's CMake sets `CMAKE_CUDA_ARCHITECTURES` OFF and gives every target its own `CUDA_ARCH`, whose default is `native`. A `-DCMAKE_CUDA_ARCHITECTURES=89`, which is what `scripts/check_fkl_tree.sh` passes, therefore has no effect on FKL's own targets, so FKL is configured here directly.

The targets come from CMake's code model, as in check_fkl_tree.sh, and the per-target report is written by `scripts/check_fkl_tree_report.py`. The expected rejections are judged by their ctest tests (`<target>_rejected`), which only build.

| | targets | compiled | failed | expected rejections held | build |
|---|---|---|---|---|---|
| 24478f0 (before) | 156 | 144 | 0 | 12 of 12 | 419 s |
| 4b11bc6 (after) | 162 | 150 | 0 | 12 of 12 | 488 s |

- The 6 new targets are #433's 3 new tests, each as `_cu` and `_cpp`: test_linear_filter_batch, test_median_filter_batch and test_quad_dpps_batch.
- `utest_flash_attention_mma_cu` compiles at sm_89 at both commits.
- The warning sites are the same at both commits: 30 sites, in vector_utils.h (20), thread_fusion.h (5) and tests/thread_fusion/test_thread_fusion.h (5). The after build has more warning lines (1,495 against 1,420) only because more translation units include those headers.

### B. The five fkl-benchmarks programs that use these DPPs, at main b4fb594

- **Before:** worktree `fkl-benchmarks-task-b`, branch task-b-sm89-check, with external/fkl at 24478f0, unchanged.
- **After:** worktree `fkl-benchmarks-task-b-after`, detached at b4fb594, with external/fkl at 4b11bc6 and the details port applied.
- Both are built with `cmake --preset local-rtx4090` (sm_89) and `cmake --build` of the five targets. Each build took 55 s, and every target built on both sides.

**The details port (`details-port.diff`, sha256 ba519e3b51a357f452c1b556de9450dda646123b546806a71d154fe62d36e108).**
- #433 removes `width` and `height` from `LinearFilterDPPDetails`. So each of its 27 initialisers in the five programs loses its first two fields, and nothing else changes: e18_box.cu 12, e18_box_split.cu 7, e18_stencil.cu 6, e19_batched_box.cu 1, e19_crop_sweep.cu 1.
- #433 does not change morphology.h, so `MorphologyDPPDetails`, the details of E19's fkl_batch (fk::executeBoxBlur), keeps its sizes. Its initialisers stay as they are.
- The port and the moved gitlink are not committed, so the after binaries carry the commit `b4fb594...-dirty` and FKL 4b11bc6. The before binaries carry b4fb594 and FKL 24478f0. Both sides carry cvGPUSpeedup 6d26c65 and FastNPP a1b5e24.

The warnings of the two builds are the same 25 sites, all in FKL's headers.

The sha256 of every binary that the window runs is in `window-binaries.sha256`: the 17 FKL test executables and the 10 program executables.

### C. AMD's failure of utest_flash_attention_mma_cu with -arch=native

AMD saw this target fail with -arch=native on a machine without a GPU. Here the target was built at 4b11bc6 with `CUDA_ARCH=native` and no GPU visible:
- nvcc prints `nvcc warning : Cannot find valid GPU for '-arch=native', default arch is used`. The default is sm_75 (`nvcc -arch=native --dryrun` gives `-arch=sm_75`).
- ptxas then fails: `Feature 'cp.async' requires .target sm_80 or higher`. The 4,807 error lines are all of this kind.

#433 changes neither the attention files nor any CMake file, so this failure does not come from #433. On the RTX 4090, `native` is sm_89, and the sm_89 build compiles (A).

## The device code, before against after (`devcode/`)

`scripts/devcode_compare.py` reads HIP's offload bundles only. For sm_89, `scripts-used/devcode_sm89.py` does the same comparison on CUDA executables, with cuobjdump 13.4.3, as in the E19 SASS comparisons and `scripts/sass_compare_builds.py`:
- **SASS:** `cuobjdump -sass` is split per kernel and normalised by `scripts/sass_tools.py`, which removes the addresses and encodings and does not rename registers.
- **Pairing:** kernels are paired by mangled name, with the hash of the anonymous namespace removed. Nothing else of the name is changed.
- **Resources:** `cuobjdump -res-usage` gives registers, stack, shared memory, local memory and constant bank 0, the bank of the kernel parameters.

Nothing is renamed: every kernel of an executable found on both sides has a pair.

### FKL's CUDA tests (`devcode-fkl-tests.txt`, `.csv`)

- 56 `_cu` executables exist at both commits. The 3 new tests exist only at 4b11bc6.
- 225 kernels on each side: 190 identical, 35 different, 0 without a pair.
- The 35 different kernels are every kernel of the 6 tests whose files #433 changes, and no other:

| test | kernels | different |
|---|---|---|
| test_box_filter_quad_dpp_cu | 6 | 6 (launchBoxFilterQuadDPP_Kernel) |
| test_convolution_fast_dpp_cu | 5 | 5 (launchConvQuadDPP_Kernel) |
| test_linear_filter_dpp_cu | 10 | 10 (linearFilterDPPKernel) |
| test_median_fast_dpp_cu | 5 | 5 (launchMedianQuadDPP_Kernel) |
| test_median_filter_dpp_cu | 4 | 4 (medianFilterDPPKernel) |
| test_morphology_fast_dpp_cu | 5 | 5 (launchMorphQuadDPP_Kernel) |

test_morphology_dpp_cu (MorphologyDPP) is identical, like every other test. In the 35 different kernels:
- the stack, the shared memory and the local memory are the same before and after;
- constant bank 0 is 8 B smaller in each;
- the instruction counts and the registers change per kernel, from -320 to +200 instructions. Some registers change by much: launchMorphQuadDPP_Kernel 54 → 128, 96 → 62 and 72 → 96; launchMedianQuadDPP_Kernel 167 → 130.

Every one of them is listed in the .txt, with all its numbers.

### The five programs (`devcode-programs.txt`, `.csv`)

44 kernels on each side: 25 identical, 19 different, 0 without a pair.
- **Different:** the 19 kernels are every `linearFilterDPPKernel` of the programs, and only those.
- **Identical:** gpuSpin, the resize-alone `launchTransformDPP_Kernel` of E18 (3), and all 17 `morphologyDPPKernel` instances of E19's fkl_batch (1 in e19_batched_box_cu, 16 in e19_crop_sweep_cu).

| program | different / kernels | instructions | registers | stack | shared | constant bank 0 |
|---|---|---|---|---|---|---|
| e18_box_cu | 8 / 9 | +8 (7 kernels), +16 (1) | 38 → 39 in 2, otherwise 40 | 0 | 4,284 B | -8 B |
| e18_box_split_cu | 5 / 7 | +8 (4), +16 (1) | 38 → 39 in 1, otherwise 40 | 0 | 4,284 B | -8 B |
| e18_stencil_cu | 4 / 7 | +8 (2), +16 (2) | 40 | 0 | 4,284 B | -8 B |
| e19_batched_box_cu | 1 / 3 | 688 → 696 | 39 | 0 | 1,428 B | 464 → 456 B |
| e19_crop_sweep_cu | 1 / 18 | 688 → 696 | 39 | 0 | 1,428 B | 464 → 456 B |

The 8 B of constant bank 0 are the two ints that the details no longer hold.

## The GPU window (`window-checks.sh`), 2026-11-24 21:50:01-21:51:02 +01:00

The coordinator granted it for 724cdfc, with the script's sha256 241a0ba5, the manifest's 8afc521d and the port diff's ba519e3b. The times come from `window/window-console.txt`.

### Results

**FKL's CUDA filter tests: every test passed.**
- At 24478f0: 7 of 7 (`window/fkl-tests-before-console.txt`).
- At 4b11bc6: 10 of 10 (`window/fkl-tests-after-console.txt`). These are the same 7, plus test_linear_filter_batch_cu, test_median_filter_batch_cu and test_quad_dpps_batch_cu.

**The checks of the five programs: 40 runs, 20 on each side, and every one passed.** Each run exited 0 with an empty standard error, printed its pass line and wrote complete CSVs.
- before: b4fb594, FKL 24478f0;
- after: FKL 4b11bc6, with the port.

**Before against after** (`window/comparison.txt`, `window/comparison.csv`), on the CPU after the GPU steps:

| | compared | equal | different |
|---|---|---|---|
| CSV data rows | 24 CSVs on each side | 23 | 1: e18_box_kernel.csv |
| E19 dump files, byte for byte | 136 on each side | 136 | 0 |

- **The 23 equal CSVs** are:
  - the checks CSVs of all five programs, at all 16 N of the crop sweep;
  - the agreement CSVs of the three E18 programs (51, 22 and 13 rows).
- **The 136 equal dumps** are, at e19_batched_box and at each of the 16 N of the crop sweep:
  - the tensors of fkl_loop, fkl_batch, opencv_cuda and npp;
  - the grey and box images of opencv_cuda and npp.

  fkl_loop's LinearFilterDPP kernel has different SASS (above), but its output is unchanged.
- **The one CSV that differs, e18_box_kernel.csv**, is not an output. It is the table of the attributes of e18_box's four fused kernels (cudaFuncGetAttributes and the size of each argument). Its first difference is at data row 2:
  - before: `fkl_box_asym,40,4284,0,0,128,12,48,48,100,24,64,48,12,1,1,24,114`
  - after: `fkl_box_asym,40,4284,0,0,128,12,48,48,100,16,64,48,12,1,1,24,106`

  `arg_details_bytes` goes from 24 to 16 B in all four rows, and `arg_total_bytes` drops by the same 8 B (114 → 106, 130 → 122): these are the two ints the details no longer hold. Registers (40), static shared memory (4,284 B), local memory and occupancy are the same.
- **stdout** (reported only): the 4 lines that differ in e18_box and the 1 in e18_stencil print the same argument sizes (details 24 → 16 B). The other stdout files are equal line for line.

The clock samples (`window/gpu_clocks.csv`, every 100 ms) read 2,550 MHz throughout, and 210 MHz while the GPU was idle. Nothing was timed.

### The steps

Its steps stop at the first failure:
1. FKL's 7 CUDA filter tests of tests/image_processing at 24478f0, with `ctest -j1 --stop-on-failure`:
   - box_filter_quad_dpp, convolution_fast_dpp, linear_filter_dpp, median_fast_dpp, median_filter_dpp, morphology_fast_dpp: the 6 that #433 changes;
   - morphology_dpp: unchanged, and the DPP of E19's fkl_batch.
2. The same 7 at 4b11bc6, plus the 3 new tests: 10 tests.
3. The checks only (no timing) of the five programs, before:
   - e18_box, e18_box_split, e18_stencil (E18_NO_TIMING=1);
   - e19_batched_box (E19_NO_TIMING=1, E19_DUMP);
   - e19_crop_sweep at N = 1, 10, 20, ..., 150 (E19_NO_TIMING=1, E19_CROPS, E19_DUMP).
4. The same runs after.

Then, on the CPU only, `scripts-used/compare_outputs.py` compares the outputs before against after, bit for bit:
- the data rows of every CSV, with their sha256. The `#` header lines hold the commits, the clocks and the temperature, so they are left out;
- every E19 dump file, byte for byte, with its sha256.

The dumps, 1,539 MiB per side (136 files each), stay in the scratch folder (scratch/task-b/dumps) and are not committed. Their sha256 are in `window/dumps-sha256.txt`.

The guards are those of the crop sweep's checks script:
- no blocked program, in the binaries and in the ctest lists;
- every binary's sha256 matches `window-binaries.sha256`, whose own sha256 is in the script;
- the commits carried by the programs, and the state of the four worktrees (the after worktree's change is exactly `details-port.diff`);
- the FKL builds are configured from the right trees, and the ctest lists have exactly 7 and 10 tests;
- the script is the committed one;
- the output folder is new, and so is the dump folder;
- no busy process and no foreign GPU context.

## Files

| File | Contents |
|---|---|
| `details-port.diff` | the port of the five programs to #433's details (the diff of the after worktree's sources) |
| `window-binaries.sha256` | the sha256 of the 27 binaries the window runs |
| `window-checks.sh` | the window's script |
| `compile/step2-console.txt` | the console of the compile step |
| `compile/fkl-{before,after}-*` | A: configure and build logs, target lists, the rejection tests' log, the per-target report (CSV), the build time |
| `compile/bench-{before,after}-*` | B: configure log, build output, build standard error (the warnings), build time |
| `compile/fkl-native-nogpu-*` | C: configure and build logs |
| `devcode/devcode-fkl-tests.*` | the device-code comparison of FKL's CUDA tests: report (.txt) and every kernel with its numbers (.csv) |
| `devcode/devcode-programs.*` | the same for the five programs |
| `scripts-used/` | compile-step2.sh, devcode_sm89.py, compare_outputs.py |
| `window/window-console.txt`, `window/window-script.sh` | the window's console and the copy of the script it ran |
| `window/fkl-tests-{before,after}-console.txt`, `.log` | ctest's console and its output log (`-V --output-log`) |
| `window/runs/{before,after}/<program>[/N<NNN>]/` | each run: stdout, stderr, exit code, its CSVs, the load and the clock at its end |
| `window/comparison.txt`, `.csv` | the comparison before against after |
| `window/dumps-sha256.txt` | the sha256 of the 272 dump files |
| `window/environment-{before,after}.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log` | the machine's state and the samplers |

Left out: the E19 dumps (1,539 MiB per side, in scratch/task-b/dumps; their sha256 are in `window/dumps-sha256.txt`), the build trees and the binaries. The FKL builds are in the scratch folder (scratch/task-b/fkl-{before,after}-build), and the program builds are in the two worktrees' build/local-rtx4090.
