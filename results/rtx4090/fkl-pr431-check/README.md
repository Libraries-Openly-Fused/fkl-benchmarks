# FKL PR #431 and the pin move to 1da77899: device-code identity of the CUDA programs (RTX 4090 machine, CUDA 13.4.3; compile only, nothing ran)

**The library change.** FusedKernelLibrary PR #431 ports the image-processing DPPs of FKL main to LTS-C++17.
- **Base and head:** base LTS-C++17 at 64ad9195; branch `LTS-C++17-image-processing-dpps`; head `c6f7473ed8092adf708359c1e4b2e205ddadbc9d`.
- **The 4 commits:**
  - 988dcd1: the Unary forms of Sub, Mul and Div;
  - 0097b07: two lines of `fused_operation.h`, and their reproducer test;
  - 733b5fa: `FK_DEVICE_STATIC`;
  - c6f7473: the DPPs.
- **The merge:** #431 was merged into LTS-C++17 on 2026-11-09 at 09:22:33Z as `1da778992756e0fa08ec289225fe18dca134412c`, with parents 64ad9195 and c6f7473. The merged tree, `ccdb367d`, is the tree of c6f7473. No file differs, so the results of this folder apply unchanged to 1da77899.

**What #431 changes:**
- **New headers.** It adds 8 headers in `algorithms/image_processing/` and the tests of the DPPs:
  - `neighborhood.h`;
  - `linear_filter.h` and `median_filter.h`;
  - `morphology.h` and `morphology_fast.h`;
  - `convolution_fast.h`, `median_fast.h` and `box_filter_fast.h`.
- **Changed files.** It changes 4 existing files:
  - `algorithms/basic_ops/arithmetic.h`, +61/-12. Sub, Mul and Div get a fourth template parameter, `IT = BinaryType`, and a `UnaryType` specialization on a `Tuple<I1, I2>`.
  - `core/execution_model/operation_model/fused_operation.h`: 2 lines.
  - `core/utils/utils.h`, +3: the macro `FK_DEVICE_STATIC`.
  - `core/execution_model/cpu_multithread.h`: a comment.
- **Unchanged.** `image_processing.h` does not include the new headers. A program of this repository therefore sees only the changes to these 4 files.
- **Kernel names.** A kernel instantiated with `fk::Sub<T>`, `fk::Mul<T>` or `fk::Div<T>` now has the default argument `fk::BinaryType` in its mangled name.

## Result
- **The 34 CUDA programs of fkl-benchmarks 8f4a95b** were built once against FKL 64ad9195 and once against c6f7473, then compared kernel by kernel (`sass-identity-64ad919-vs-c6f7473.txt`).
- **Kernels:** 1,790 in each build, and all 1,790 pair up. 1,221 of the pairs have a different mangled name because of the default argument. The other 569 have no Sub, Mul or Div in their names.
- **SASS:** identical for all 1,790. None differs, no kernel is left without a pair, and there are no key collisions.
- **Names:** the pairing removed 15,704 `,fk::BinaryType` arguments from the names of the c6f7473 build, and none from the 64ad9195 build.
- **Cause of the renames:** all 1,221 come from the default argument. Masking the module id that nvcc puts in the names of anonymous namespaces (`_GLOBAL__N__<8 hex>_`) leaves every one of them still different, and no other pair renamed (`rename-causes-64ad919-vs-c6f7473.txt`, from `scripts-used/rename_causes.py`).
- **The coordinator's re-check:** byte for byte, instruction encodings included. It agrees.
- **Compile:** both builds compiled all 39 targets of the bench group, the 34 `_cu` programs and 5 `_cpp` programs. They took 2,808 s and 2,802 s (`identity-builds/*/report.txt`, `bench-time.txt`).

## Method
- **Compile only.** No program was run, for any purpose.
- **Builds** (`scripts-used/identity_builds.sh`, output in `identity-builds/console.txt`):
  - Run: 2026-11-08, 21:40:33 to 22:27:25 (+01:00). The two builds ran at the same time on separate cores.
  - fkl-benchmarks 8f4a95b, from a clean detached worktree with cvGPUSpeedup 6d26c65 and FastNPP a1b5e245 (the pins of this repository).
  - The two FKL trees were clean worktrees of 64ad9195 and of c6f7473, each with its thread-pool submodule at 27e733f.
  - Toolchain: nvcc 13.4.92 for sm_89, g++ 15.2.0, Release, OpenCV 4.14 with CUDA.
- **The build line, once per tree:**
  ```
  scripts/check_fkl_tree.sh <FKL tree> --out <dir> --only bench --ref-build build/local-rtx4090 --jobs 12 --cores 8-15,24-31
  ```
  - The c6f7473 build used `--cores 0-7,16-23`.
  - `--ref-build` takes the `FKLB_*` options, `OPENCV_DIR` and nvcc from the CMakeCache of the reference build of preset `local-rtx4090`. That build was only read.
  - The bench group builds every `*_cu` and `*_cpp` target of this repository.
- **The comparison,** run in the scratch folder of the builds:
  ```
  python3 -I scripts-used/sass_identity.py base-64ad919/bench-build/bin branch-c6f7473/bench-build/bin --tools scripts --arch sm_89 --glob '*_cu' --out sass-identity-64ad919-vs-c6f7473.txt
  ```
  - `--tools` is the `scripts/` of 8f4a95b, for `sass_tools.py`. The dump tool is the cuobjdump of CUDA 13.4.3.
  - Each program's SASS is split per kernel and normalised by `scripts/sass_tools.py`: addresses and encodings are removed, and registers are not renamed. This is the same normalisation as `scripts/sass_compare_builds.py`.
  - Kernels are paired by demangled name with the whitespace removed (`drop_default` in the script). `,fk::BinaryType` is removed only where it is the fourth top-level template argument of `fk::Sub`, `fk::Mul` or `fk::Div`. Nothing else in a name changes, so two kernels pair only when their names are otherwise identical.
  - The script exits 0 only if every kernel has a pair and every pair has identical SASS.
- **Check of the script** (`sass-identity-script-check-pr423-vs-pr427.txt`):
  - The same script, run on the bench builds of the #423 and #427 checks, between which no name changes.
  - It gives 1,685 of 1,685 kernels identical and 0 renamed. That is the result of `scripts/sass_compare_builds.py` in `../fkl-pr427-check/`.
- **Program set.** It is 34 programs, not the 35 of `../fkl-pr427-check/`:
  - this repository has since removed the two `_ctxfix` NPP programs;
  - it added `benchmark_vfhf_max_speedup_cu`;
  - `e4_preprocessing_cu` now has 182 kernels.

## Compile check at the pin (`compile-check/`)
- **What was built:** the branch of the pin move at fkl-benchmarks 434dada.
  - 434dada is 8f4a95b plus the gitlink and the files in `results/`.
  - Its `external/fkl` was checked out at `1da778992756e0fa08ec289225fe18dca134412c`, with thread-pool 27e733f, cvGPUSpeedup 6d26c65 and FastNPP a1b5e245.
  - `scripts-used/compile_check.sh` stops unless the checkout and the gitlink of HEAD are both at 1da77899 and the worktree is clean.
- **When:** compile only, 2026-11-09, 11:28:04 to 12:05:50 (+01:00).
- **The build line:**
  ```
  scripts/check_fkl_tree.sh <worktree>/external/fkl --out <dir> --only bench --ref-build build/local-rtx4090
  ```
  This used the defaults (12 jobs on cores 8-15,24-31) and the same toolchain as above.
- **Result:** all 39 targets compiled, 0 failed and 0 were not built, in 2,263 s (`report.txt`).
- **Against the c6f7473 build of the identity check:** the same 39 targets and 0 regressions (`compare-with-c6f7473-identity-build.txt`, from `scripts/check_fkl_tree_compare.py`).
- **SASS against the c6f7473 build:** 1,790 of 1,790 kernels identical, paired by demangled name as above (`sass-identity-head-vs-c6f7473-identity-build.txt`).
  - 146 of them are in anonymous namespaces and carry a different module id in their mangled names. Nothing else in their names differs (`rename-causes-head-vs-c6f7473-identity-build.txt`).
  - By mangled name alone, `scripts/sass_compare_builds.py` therefore reports 1,644 identical kernels and 146 in each build only (`sass-head-vs-c6f7473-identity-build.txt`).
  - The two builds differ in the fkl-benchmarks commit (8f4a95b and 434dada) and in their build folders. Which of the two sets the module id was not determined.
  - Between the 64ad9195 and c6f7473 builds, which share the commit, the module ids are the same.
- **Left out:**
  - `bench-build.log` (17.02 MiB);
  - the empty `bench-rejected-targets.txt`;
  - the build directory.

  They stay in `scratch/pin-1da7789/check/` on this machine.

## What this says and what not
- **What it says:** at fkl-benchmarks 8f4a95b, the device code of every CUDA program of this repository is the same, instruction for instruction, at FKL 64ad9195 and at 1da77899. Only kernel names differ, and only by the default argument. So the GPU results these programs gave at 64ad9195 stand for 1da77899 as far as the device code goes.
- **What it does not cover:**
  - The host code was not compared. Its Sub, Mul and Div instantiations carry the same default argument.
  - The HIP device code at 1da77899 is AMD's record, in `results/rx9060/`.
  - The CPU builds were not compared; the 5 `_cpp` programs only compiled.
  - Nothing here measures time.
- **The library's own checks of #431** are in the description of FusedKernelLibrary #431: its CI (5 workflows, 9 jobs, all passed) and the local builds and tests on this machine.

## Files
- `sass-identity-64ad919-vs-c6f7473.txt`: the identity report.
- `sass-identity-script-check-pr423-vs-pr427.txt`: the check of the script.
- `scripts-used/identity_builds.sh` and `scripts-used/sass_identity.py`: the scripts that ran.
- `scripts-used/compile_check.sh` and `scripts-used/rename_causes.py`: the scripts of the compile check at the pin and of the rename count.
- `rename-causes-64ad919-vs-c6f7473.txt`: the renamed pairs of the identity check, by cause.
- `compile-check/`: the compile check at the pin.
  - The files that `check_fkl_tree.sh` wrote: `report.txt`, `bench-report.csv`, `bench-targets.txt`, `bench-targets-built.txt`, `bench-time.txt` and `bench-configure.log`.
  - `console.txt`: `report.txt` plus the start and end lines of `compile_check.sh`.
  - The comparisons with the c6f7473 build: `compare-with-c6f7473-identity-build.txt`, `sass-identity-head-vs-c6f7473-identity-build.txt`, `sass-head-vs-c6f7473-identity-build.txt` and `rename-causes-head-vs-c6f7473-identity-build.txt`.
- `identity-builds/console.txt`: the console of `identity_builds.sh`.
- `identity-builds/base-64ad919/` and `identity-builds/branch-c6f7473/`: per build, the files that `check_fkl_tree.sh` wrote:
  - `report.txt` (the same text as that build's console);
  - `bench-report.csv`;
  - `bench-targets.txt`;
  - `bench-targets-built.txt`;
  - `bench-time.txt`;
  - `bench-configure.log`.
- **Left out:**
  - the build logs, `bench-build.log` (17.24 MiB and 17.27 MiB);
  - the build directories;
  - the two empty `bench-rejected-targets.txt` (the bench group has no expected rejections);
  - the two console logs, which are byte-identical to `report.txt`.

  They stay in `scratch/dpp-port/identity/` on this machine. Every file here is the complete output of the program named; nothing was filtered.
