# Final checks of the library pin (RTX 4090, CUDA 13.4.3) and of the E16 test

## The pin since 2026-11-13: 24478f0
`external/fkl` of fkl-benchmarks points at 24478f0aa11dc3f12feea83178f7fde828322fb0 (the merge of #432 into LTS-C++17, 2026-11-13). Its tree, 153ebf2b, is the tree of the PR head 2c42a80. It is 1da77899 plus #432, the HIP (`ParArch::GPU_AMD`) versions of the 7 image-processing DPPs: 1,075 added lines in 7 headers, all inside `#if defined(FK_HIP_BACKEND)` blocks, one guard of `neighborhood.h` that becomes `__NVCC__ || FK_HIP_BACKEND`, and the DPP tests. The thread-pool submodule is the same (27e733f).
The device code of the 35 CUDA programs of this repository (at 8c1fa98) is the same at 1da77899 and at 2c42a80: 1,797 of 1,797 kernels identical, none renamed; compile only, sm_89 (`../fkl-pr432-check/`). The records of this machine that were measured at 1da77899 (`../e18-stencil-phase1/`, `../e18-stencil/`) keep that commit in their headers; their GPU results stand for 24478f0 as far as the device code goes. The host code was not compared, and the HIP and CPU builds were not part of this comparison (the HIP device code is AMD's to check).

## The pin from 2026-11-09 to 2026-11-13: 1da77899
`external/fkl` of fkl-benchmarks pointed at 1da778992756e0fa08ec289225fe18dca134412c (`Beta-0.1.14-LTS-223-g1da7789`, the merge of #431 into LTS-C++17). It is 64ad9195 plus #431, which ports the image-processing DPPs of FKL main in 8 new headers (`image_processing.h` does not include them) and changes 4 existing files: `arithmetic.h` (Sub, Mul and Div get a defaulted fourth template parameter and a Unary form), `fused_operation.h` (2 lines), `utils.h` (`FK_DEVICE_STATIC`) and a comment in `cpu_multithread.h`. The thread-pool submodule is the same (27e733f).
The device code of the 34 CUDA programs of this repository (at 8f4a95b) is the same at 64ad9195 and at c6f7473, the head of #431, whose tree is the tree of 1da77899: 1,790 of 1,790 kernels identical, 1,221 of them under a new mangled name (the default argument); compile only, sm_89 (`../fkl-pr431-check/`). The coordinator re-checked it byte for byte, encodings included. The records of this machine that were measured at 64ad9195 (`../cleanup-window-a/` to `../cleanup-window-e/`, `../cleanup-window-b-warm-diagnostic/`, `../e4-hand-cvgs/`, `../e4-hand-cvgs-host-diagnostic/`, `../e4-hand-roi/`, `../e4-build-once/`) keep that commit in their headers; their GPU results stand for 1da77899 as far as the device code goes. The host code was not compared, and the HIP and CPU builds were not part of this comparison (the HIP device code is AMD's record, in `../../rx9060/`). The compile check of the bench group at the pin is in `../fkl-pr431-check/compile-check/`.

## The pin from 2026-11-07 to 2026-11-09: 64ad9195
`external/fkl` of fkl-benchmarks pointed at 64ad9195cd785d0c93a70b35e02d5a6c7c4a5048 (`Beta-0.1.14-LTS-218-g64ad919`, the merge of #427 into LTS-C++17) from fkl-benchmarks bdf53a8 until 2026-11-09. It is c8fcbe8 plus four merges: #422 (the E16 utest), #426 (a fix of the ROCm CI), #423 (the chain check of the executors) and #427 (the crop fix of issue #19). 14 files differ; the thread-pool submodule is the same (27e733f).
The device code of the 35 CUDA programs of this repository is the same at c8fcbe8, at 9e20bd4e (the head of #423) and at 14512602 (the head of #427, the tree of 64ad9195): 1,685 of 1,685 kernels identical in each of the two comparisons (`../fkl-pr423-check/`, `../fkl-pr427-check/`; sm_89). So the GPU results these programs gave at c8fcbe8 or at 9e20bd4e stand for 64ad9195 as far as the device code goes. The host code was not compared, and the HIP and CPU builds were not part of these comparisons. The checks of 64ad9195 (compile check, CPU tests, FKL's CUDA tests, the cvGPUSpeedup and FastNPP suites) are in `../fkl-pr427-check/` and `../fkl-ctest-pr427-14512602/`.

## The pin until 2026-11-07: LTS-C++17 c8fcbe8 (merge of PR #417)
`external/fkl` of fkl-benchmarks pointed at c8fcbe816d629377886cbdd19c83527f37ccfc04 from fkl-benchmarks e9e599d until 2026-11-07: 18 commits over the pin 38c4f32 that carried the numbers of E0 to E2 (#411, #412, #414, #405, #416 and #417). The attention headers are unchanged since d776713.

**Full ctest at c8fcbe8** (`ctest-c8fcbe8-rtx4090.log`, `ctest-c8fcbe8-list.txt`): a clean checkout of c8fcbe8 (no modified file; the only untracked entries are the two build directories; the `thread-pool` submodule of the worktree is a link to a checkout at the same commit, 27e733f),
configured with `-DENABLE_CUDA=ON -DENABLE_CPU=ON -DENABLE_CPU_MULTITHREAD=ON -DCUDA_ARCH=89 -DCMAKE_BUILD_TYPE=Release`, nvcc 13.4.92, g++ 15.2, `ctest -j4` on the RTX 4090 (CUDA tests) and the Ryzen 9 5950X (CPU tests):
**89 tests, 89 passed, 0 failed, 6.4 s.** Of them 46 are CUDA targets (`*_cu`) and 43 are CPU targets (`*_cpp`). The CUDA-only configuration of the same commit builds the same 46 CUDA executables (`build-tip`, used as the SASS baseline below).
`utest_get_arg_large_pack` (the 1,100-operation chain of PR #417) is among the CUDA targets and passes; it is compiled and run on Linux only.

**What the 89 tests are.** Every target comes from one of the two test options of the library's CMake, `BUILD_TEST` (directory `tests/`) and `BUILD_UTEST` (directory `utests/`), both ON by default. Each source file is built once with the host compiler (`_cpp`, unless
the file says `ONLY_CU` or `ONLY_GPU`) and once with nvcc (`_cu`, unless it says `ONLY_CPU`); the HIP variant (`_hip`) is not configured here (no HIP compiler). Counts:

| class | directory | CUDA (`_cu`) | CPU (`_cpp`) | total |
|---|---|---|---|---|
| tests | `tests/` without `tests/examples/` (algorithm 12, data 11, operation 12, string 8, thread_fusion 6, buildAPI 2, cudabug 4) | 29 | 26 | 55 |
| examples | `tests/examples/` (compiler_explorer_example, readme_test_code, inlining_and_LDL_STL, test_divergent_hf_executor) | 4 | 2 | 6 |
| utests | `utests/` | 13 | 15 | 28 |
| benchmarks | - | 0 | 0 | 0 |
| **all** | | **46** | **43** | **89** |

No benchmark is excluded: the LTS-C++17 tree at c8fcbe8 has no benchmark directory and no benchmark option in its CMake (the benchmarks of the paper are in the fkl-benchmarks repository). The `tests/` and `tests/examples/` counts are by source directory
(`ctest --show-only=json-v1`, test names matched to the `.h` sources); the 3 utests whose sources are generated from `.in` templates (`utest_saturate_cu/_cpp`, `utest_cuda_vector_utils_cpp`) are counted by their name (`utest_`) and sit under `utests/`.

## E16: `utest_flash_attention_mma` rebased onto c8fcbe8 (branch LTS-C++17-flash-mma-utest, pull request #422 against LTS-C++17)
Head ed547a5 (two commits on c8fcbe8: cdd6464 the test with a double-precision oracle and the SIMT FlashAttention as second reference, ed547a5 the tolerance of 3e-3), one added file, `utests/algorithm/attention/utest_flash_attention_mma.h` (160 lines).
- Test on the RTX 4090 (sm_89) and on the RTX 5050 (sm_120, built for 120): exit 0, 21 checks passed, 0 failed on each (`e16-utest-ed547a5-rtx4090.txt`, `e16-utest-ed547a5-rtx5050.txt`); the largest error of the MMA kernel against the oracle is 1.21e-3 on both (tolerance 3e-3).
- Full ctest of the branch, CUDA targets, on the 4090 (`ctest-e16-ed547a5-rtx4090.log`): 47 tests, 47 passed, 0 failed (the 46 of the tip and the new one).
- SASS of the library kernels, branch build against the build of c8fcbe8 (`e16-sass-vs-c8fcbe8.txt`, `scripts/sass_compare_builds.py`, sm_89): 46 executables in both (the branch has one more, the new test), 153 kernels, identical 153, different 0. The branch only adds a test file.
