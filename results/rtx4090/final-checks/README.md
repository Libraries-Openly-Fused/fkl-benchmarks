# Final checks of the library pin (RTX 4090, CUDA 13.4.3) and of the E16 test

## The final pin: LTS-C++17 c8fcbe8 (merge of PR #417)
`external/fkl` of fkl-benchmarks points at c8fcbe816d629377886cbdd19c83527f37ccfc04 (since fkl-benchmarks e9e599d): 18 commits over the pin 38c4f32 that carried the numbers of E0 to E2 (#411, #412, #414, #405, #416 and #417). The attention headers are unchanged since d776713.

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
