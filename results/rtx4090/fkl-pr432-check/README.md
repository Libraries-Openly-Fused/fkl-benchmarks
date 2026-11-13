# FKL PR #432 before the pin move: device-code identity of the CUDA programs (RTX 4090 machine, CUDA 13.4.3; compile only, nothing ran)

**The library change.** FusedKernelLibrary PR #432 adds the HIP versions (`ParArch::GPU_AMD`) of the 7 image-processing DPPs of #431.
- **Base and head:** base LTS-C++17 at 1da77899, the current pin; branch `LTS-C++17-image-processing-dpps-hip`; head `2c42a80e12168f4183ba835cdde353cc1e33be07`.
- **The 2 commits:**
  - ed122be: the HIP versions of the 7 DPPs;
  - 2c42a80: the GPU parts of the 7 DPP tests run on HIP too.
- **Headers:** the 7 DPP headers in `algorithms/image_processing/` (box_filter_fast, convolution_fast, linear_filter, median_fast, median_filter, morphology, morphology_fast) only gain lines, 1,075 in all, inside `#if defined(FK_HIP_BACKEND)` blocks.
- **neighborhood.h:** one line changes, `#if defined(__NVCC__)` becomes `#if defined(__NVCC__) || defined(FK_HIP_BACKEND)`.
- **Tests:** the remaining 27 lines change the 7 tests.
- **The CUDA build:** nvcc never defines `FK_HIP_BACKEND`, so every addition is preprocessed away there.
- **Why a check is needed:** this repository's E18 program (`opencv-cuda/stencil/e18_stencil.cu`) includes `linear_filter.h`, so a CUDA identity check is due before the pin moves.
- **The merge:** #432 was merged into LTS-C++17 on 2026-11-13 at 12:18:24Z as `24478f0aa11dc3f12feea83178f7fde828322fb0`, with parents 1da77899 and 2c42a80. Its tree, `153ebf2b`, is the tree of 2c42a80. No file differs, so the builds of this check, which used 2c42a80, hold for 24478f0. The gitlink points at 24478f0.

## Result
- **The 35 CUDA programs of fkl-benchmarks 8c1fa98** (E18's `e18_stencil_cu` included) were built once against FKL 1da77899 and once against 2c42a80, then compared kernel by kernel (`sass-identity-1da7789-vs-2c42a80.txt`).
- **Kernels:** 1,797 in each build, and all 1,797 pair up.
- **Renames:** 0. No kernel's mangled name changes (`rename-causes-1da7789-vs-2c42a80.txt`: 1,797 of 1,797 names the same).
- **SASS:** identical for all 1,797. None differs, no kernel is left without a pair, and there are no key collisions.
- **e18_stencil_cu:** 7 of 7 kernels identical, the fused `linearFilterDPPKernel` of E18 included.
- **Compile:** both builds compiled all 40 targets of the bench group (35 `_cu`, 5 `_cpp`), in 2,811 s and 2,805 s (`identity-builds/*/report.txt`).
- **The binaries' bytes** (`binary-sha256-*.txt`): 39 of the 40 differ between the builds. Every program except `batch_build_compilation_time_cpp` embeds the FKL commit in its build information (checked with `strings` on `e18_stencil_cu` and `benchmark_latency_hiding_cpp`). `batch_build_compilation_time_cpp` is byte-identical.

## Method
The method is that of `../fkl-pr431-check/`.
- **Compile only.** No program was run, for any purpose. The frozen 2025 programs were compiled and their SASS dumped, nothing more.
- **Builds** (`scripts-used/identity_builds.sh`, output in `identity-builds/console.txt`):
  - Run: 2026-11-13, 12:40:30 to 13:27:24 (+01:00). The two builds ran at the same time on separate cores.
  - fkl-benchmarks 8c1fa98 (main), from the clean worktree `worktrees/fkl-benchmarks-pin-432` at that commit, with cvGPUSpeedup 6d26c65 and FastNPP a1b5e245.
  - Base tree: the worktree's own `external/fkl` at its gitlink 1da77899.
  - Head tree: a clean detached worktree of FKL at 2c42a80, fetched from `refs/pull/432/head`.
  - Both trees have the thread-pool submodule at 27e733f.
  - Toolchain: nvcc 13.4.92 for sm_89, g++ 15.2.0, Release, OpenCV 4.14 with CUDA.
- **The build line, once per tree:**
  ```
  scripts/check_fkl_tree.sh <FKL tree> --out <dir> --only bench --ref-build build/local-rtx4090 --jobs 12 --cores 8-15,24-31
  ```
  - The head build used `--cores 0-7,16-23`.
  - `--ref-build` takes the `FKLB_*` options, `OPENCV_DIR` and nvcc from the CMakeCache of the reference build of preset `local-rtx4090`. That build was only read.
- **The comparison,** run in the scratch folder of the builds:
  ```
  python3 -I scripts-used/sass_identity.py base-1da7789/bench-build/bin head-2c42a80/bench-build/bin --tools scripts --arch sm_89 --glob '*_cu' --out sass-identity-1da7789-vs-2c42a80.txt
  python3 -I scripts-used/rename_causes.py base-1da7789/bench-build/bin head-2c42a80/bench-build/bin > rename-causes-1da7789-vs-2c42a80.txt
  ```
  - The dump tool is the cuobjdump of CUDA 13.4.3.
  - SASS is normalised by `scripts/sass_tools.py`: addresses and encodings are removed, and registers are not renamed.
  - Pairing is by demangled name, with `drop_default`. `drop_default` removes `,fk::BinaryType` from both sides alike; #432 changes no template parameter.
  - `sass_identity.py` is byte-identical to the one of `../fkl-pr431-check/scripts-used/`, and `rename_causes.py` is copied from there.

## What this says and what not
- **What it says:** at fkl-benchmarks 8c1fa98, the device code of every CUDA program of this repository, E18 included, is the same instruction for instruction at FKL 1da77899 and at the head of #432, with the same kernel names. So the GPU results these programs gave at 1da77899 stand for 2c42a80 as far as the device code goes.
- **What it does not cover:**
  - The host code was not compared.
  - The HIP device code of #432 is AMD's to check.
  - The CPU builds were not compared; the 5 `_cpp` programs only compiled.
  - Nothing here measures time.
- **The merge commit:** its tree equals 2c42a80's (153ebf2b), so the result applies to 24478f0 unchanged.

## Files
- `sass-identity-1da7789-vs-2c42a80.txt`: the identity report.
- `rename-causes-1da7789-vs-2c42a80.txt`: the names, by cause.
- `binary-sha256-base-1da7789.txt`, `binary-sha256-head-2c42a80.txt`: sha256 of the 40 binaries of each build.
- `scripts-used/identity_builds.sh`, `scripts-used/sass_identity.py`, `scripts-used/rename_causes.py`.
- `identity-builds/console.txt`: the console of `identity_builds.sh`.
- `identity-builds/base-1da7789/` and `identity-builds/head-2c42a80/`: per build, the files that `check_fkl_tree.sh` wrote:
  - `report.txt` (the same text as that build's console);
  - `bench-report.csv`;
  - `bench-targets.txt`;
  - `bench-targets-built.txt`;
  - `bench-time.txt`;
  - `bench-configure.log`.
- **Left out:**
  - the build logs, `bench-build.log` (17.10 MiB each);
  - the build directories;
  - the two empty `bench-rejected-targets.txt` (the bench group has no expected rejections);
  - the two console logs, which are byte-identical to `report.txt`.

  They stay in `scratch/pin-432/` on this machine. Every file here is the complete output of the program named; nothing was filtered.
