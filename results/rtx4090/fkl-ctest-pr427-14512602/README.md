# FKL PR #427 (issue #19, the crop fix) at its head 14512602 on the RTX 4090: FKL's 48 CUDA tests, the cvGPUSpeedup suite (25 tests) and the FastNPP suite (2 tests), run of 2026-11-07 00:20:28 to 00:24:35

Window approved by the coordinator after the step-1 report (10 minutes reserved; nothing else of mine running, and the coordinator made no tool calls on this machine until it ended). Library tree under test: the head of FusedKernelLibrary PR #427, `14512602de7f703c4d010f5f348069404e5d87b5` (4 commits on LTS-C++17 7b2deb66; the compile check, the CPU tests and the SASS comparison of this head are in `../fkl-pr427-check/README.md`). PR #427 was merged into LTS-C++17 on 2026-11-06 as `64ad9195cd785d0c93a70b35e02d5a6c7c4a5048` (parents 7b2deb66 and 14512602); the merged tree (`7c35e151`) is the tree of 14512602, no file differs, so these results apply unchanged to the merged commit.

Builds, all from step 1 (compile only, `../fkl-pr427-check/`), sm_89, nvcc 13.4.92, g++ 15.2, Release; the sha256 of every test executable is in `<suite>/binary-sha256.txt`:
- FKL: `scratch/pr427/check/fkltests-build`, the fkltests build of `scripts/check_fkl_tree.sh` on that head (141 targets, 129 compiled).
- cvGPUSpeedup 6d26c65 and FastNPP a1b5e24, the pins of this repository: configured a second time from the clones of the compile check (their `fkl/` is a link to the tree under test) with `ENABLE_BENCHMARK=ON`, as for the runs of FKL PRs #409 to #416; `scratch/pr427/check/cvgs-bench-build` and `fastnpp-bench-build`.

Command (`window-script.sh`, the whole script under `timeout -k 10 590`): the three suites one after the other, each `ctest <filter> --timeout 240 -j1 --output-on-failure --no-compress-output -T Test` under `timeout -k 10 <what is left of 540 s counted from the start of the first suite>`, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (the RTX 4090, with the desktop session on it: 449 MiB used, P8, 3 % before; 449 MiB, 0 %, 480 MHz, 39 C after; `gpu-info.csv`, `gpu-after.csv`), pinned to cores 0-3,16-19, one test at a time. Filters: FKL `-R '_cu$'`; cvGPUSpeedup `-E '^benchmark_image_resolution_MAD_loop$'`; FastNPP none.
- cvGPUSpeedup registers 26 tests with `ENABLE_BENCHMARK=ON`. The 26th, `benchmark_image_resolution_MAD_loop`, is the full loop of 60 experiments; the runs of #409 to #416 left it out of the suite (`ctest -E benchmark_image_resolution_MAD_loop`) and ran a 3-experiment variant apart, which the coordinator dropped for #427. So the 25 tests here are the 25 of the #416 run by name.
- The planned single runs of this repository's two NPP programs were dropped by the coordinator before the window (these programs will be rewritten without FKL objects).

The tests are functional tests, not measurements: pass / fail per test, no timing claim (the times below are ctest's wall times of this one run). Load before 0.14 0.94 3.13, after 1.19 1.07 2.66 (`environment-before.txt`, `environment-after.txt`, `load.log` every 5 s, 50 samples). Above 3 % of a core in the log: the tests on their cores; the desktop's file indexer `localsearch-3` (up to 96 % of one core, in 16 of the 50 samples); the Claude sessions (at most 7 %); and the terminal (4 %, once).

**Result: 75 of 75 passed, 0 failed, 0 timeouts, 0 not run; every ctest exited with 0.**

| suite | tests | passed | total test time (real) | the suite ran |
|---|--:|--:|--:|---|
| FKL, CUDA tests (`_cu$`) | 48 | 48 | 7.25 s | 00:20:28 to 00:20:36 |
| cvGPUSpeedup | 25 | 25 | 236.42 s | 00:20:36 to 00:24:32 |
| FastNPP | 2 | 2 | 0.33 s | 00:24:32 to 00:24:33 |

Against the earlier runs:
- The 48 FKL tests are, by name, the 48 of the #423 window (`../fkl-ctest-pr423-9e20bd4e/`: 48 of 48, 7.43 s). The tests that #427 adds or changes are inside two of them: `test_ptr_nd_cu` (commit 36dcb86; 0.46 s here, 0.45 s in the #423 window) and `basic_test_cu` (commit 123dd65; 0.26 s here, 0.53 s there).
- The 25 cvGPUSpeedup tests are the 25 of the #416 run of 2026-10-30 (FKL 231dec8 then: 25 of 25, 251.91 s).
- FastNPP's suite is not the one of #416. FastNPP #44 moved its two benchmarks into this repository (`npp/`), and #45 added `fastNPP_resize_vs_npp_test`, so at the pin a1b5e24 the suite has these 2 tests.

What this does not cover: the 12 compile-fail tests (`*_rejected`) and the 45 CPU tests are in `../fkl-pr427-check/` (12 of 12 and 45 of 45 passed). The programs of this repository were compiled there (40 of 40, device code identical to the head of #423) but not run here. Nothing here says anything about speed.

## FKL, the 48 CUDA tests (`fkl-cuda/`)

| # | test | result | ctest time (s) |
|--:|---|---|--:|
| 2 | `test_batchresize_build_cu` | Passed | 0.00 |
| 4 | `test_border_reader_cu` | Passed | 0.00 |
| 6 | `test_crop_cu` | Passed | 0.00 |
| 8 | `test_deinterlace_cu` | Passed | 0.00 |
| 10 | `test_raw_image_types_cu` | Passed | 0.00 |
| 12 | `test_warp_cu` | Passed | 0.00 |
| 14 | `batch_build_compilation_time_cu` | Passed | 0.28 |
| 15 | `test_nvcc124_126_BugReproducer_cu` | Passed | 0.00 |
| 16 | `test_nvcc1281_129_BugReproducer_cu` | Passed | 0.00 |
| 18 | `test_nvcc131_BugReproducer_cu` | Passed | 0.00 |
| 20 | `basic_test_cu` | Passed | 0.26 |
| 21 | `test_circular_tensor_cu` | Passed | 0.55 |
| 23 | `test_ptr_nd_cu` | Passed | 0.46 |
| 25 | `test_rect_cu` | Passed | 0.00 |
| 27 | `test_tuple_cu` | Passed | 0.00 |
| 29 | `test_typelist_cu` | Passed | 0.00 |
| 31 | `compiler_explorer_example_cu` | Passed | 0.19 |
| 32 | `inlining_and_LDL_STL_cu` | Passed | 0.28 |
| 34 | `readme_test_code_cu` | Passed | 0.20 |
| 35 | `test_divergent_hf_executor_cu` | Passed | 0.18 |
| 37 | `test_filtered_index_sequence_cu` | Passed | 0.00 |
| 39 | `test_fused_operation_cu` | Passed | 0.00 |
| 41 | `test_instantiable_operations_cu` | Passed | 0.19 |
| 43 | `test_operation_fuser_cu` | Passed | 0.00 |
| 45 | `test_operation_tuple_cu` | Passed | 0.18 |
| 47 | `test_operation_types_cu` | Passed | 0.00 |
| 49 | `test_IOps_to_string_cu` | Passed | 0.19 |
| 51 | `test_fk_string_cu` | Passed | 0.00 |
| 53 | `test_literal_to_string_cu` | Passed | 0.00 |
| 55 | `test_struct_to_string_cu` | Passed | 0.00 |
| 57 | `test_large_grids_cu` | Passed | 0.32 |
| 59 | `test_thread_fusion_cu` | Passed | 0.00 |
| 61 | `test_thread_fusion_alignment_cu` | Passed | 0.49 |
| 62 | `utest_flash_attention_cu` | Passed | 0.19 |
| 63 | `utest_flash_attention_mma_cu` | Passed | 0.20 |
| 64 | `utest_softmax_cu` | Passed | 0.19 |
| 66 | `utest_color_conversion_cu` | Passed | 0.17 |
| 68 | `utest_deinterlace_cu` | Passed | 0.20 |
| 70 | `utest_resize_sampling_cu` | Passed | 0.23 |
| 72 | `utest_yuv_to_rgb_matrices_cu` | Passed | 0.20 |
| 74 | `utest_constexpr_cmath_cu` | Passed | 0.00 |
| 78 | `utest_chain_check_cu` | Passed | 0.20 |
| 82 | `utest_executors_cu` | Passed | 0.20 |
| 83 | `utest_stream_ownership_cu` | Passed | 0.20 |
| 85 | `utest_stream_ref_counts_cu` | Passed | 0.00 |
| 87 | `utest_VBase_cu` | Passed | 0.00 |
| 89 | `utest_get_arg_large_pack_cu` | Passed | 0.21 |
| 91 | `utest_saturate_cu` | Passed | 1.23 |

## cvGPUSpeedup, 25 tests (`cvGPUSpeedup/`)

| # | test | result | ctest time (s) |
|--:|---|---|--:|
| 1 | `test_batchread_x_write3D` | Passed | 1.72 |
| 2 | `test_circularbatchread_x_write3D` | Passed | 0.48 |
| 3 | `test_batchresize_aspectratio_x_split3D` | Passed | 49.38 |
| 4 | `test_batchresize_x_split3D` | Passed | 85.86 |
| 5 | `test_cvtColor` | Passed | 1.71 |
| 6 | `testOpenCVInstallation` | Passed | 0.29 |
| 7 | `test_read_x_split` | Passed | 12.77 |
| 8 | `test_read_x_write` | Passed | 9.49 |
| 9 | `test_fused_resize` | Passed | 0.33 |
| 10 | `test_resize_CPUvsGPUresults` | Passed | 0.35 |
| 11 | `test_resize_write` | Passed | 2.25 |
| 12 | `test_resize_x_split` | Passed | 0.31 |
| 13 | `test_convertTo` | Passed | 0.20 |
| 14 | `test_saturate_cast` | Passed | 0.20 |
| 15 | `test_split` | Passed | 0.20 |
| 16 | `test_warping_opencv` | Passed | 0.33 |
| 17 | `benchmark_CPU_OpenCV_vs_cvGS` | Passed | 2.72 |
| 18 | `benchmark_CPUandGPU_cvGS_vs_fk` | Passed | 0.19 |
| 19 | `benchmark_vertical_fusionCompareAdd` | Passed | 10.38 |
| 20 | `benchmark_vertical_fusionCompareDiv` | Passed | 10.41 |
| 21 | `benchmark_vertical_fusionCompareSub` | Passed | 10.36 |
| 22 | `benchmark_vertical_fusionCompareMul` | Passed | 10.36 |
| 23 | `benchmark_vertical_fusionCompareMul1` | Passed | 7.04 |
| 24 | `benchmark_vertical_fusionCompareMulAdd` | Passed | 12.05 |
| 25 | `benchmark_vertical_fusionCompareMulAdd1` | Passed | 7.06 |

## FastNPP, 2 tests (`FastNPP/`)

| # | test | result | ctest time (s) |
|--:|---|---|--:|
| 1 | `fastNPP_basic_test` | Passed | 0.00 |
| 2 | `fastNPP_resize_vs_npp_test` | Passed | 0.33 |

## Files
- `window-script.sh` (the script that ran), `window-console.txt` (its complete output), `environment-before.txt`, `environment-after.txt`, `load.log`, `gpu-info.csv`, `gpu-after.csv`.
- Per suite, in `fkl-cuda/`, `cvGPUSpeedup/` and `FastNPP/`:
  - `test-list.txt`: `ctest -N` with the suite's filter.
  - `binary-sha256.txt`.
  - `ctest-stdout.txt`: the console output of ctest.
  - `ctest-output.log`: written by `--output-log`; the same text as `ctest-stdout.txt`.
  - `ctest-exit-code.txt`.
  - `Test.xml`: the record of `-T Test`.
  - `LastTest_<date>.log`: ctest's own log of the run, with the output of every test. The date in the name is in UTC.
- How the `LastTest_<date>.log` files got here: the window script looked for `Testing/Temporary/LastTest.log`. ctest does not write that file when it runs with `-T Test`; it writes `LastTest_<date>.log` instead. So the script copied nothing, and I copied the three files from the build directories after the window. Their times are the window's (00:20:36, 00:24:32 and 00:24:33), and they hold 48, 25 and 2 tests, all passed.
- There is no `LastTestsFailed.log`: ctest writes it only when a test fails.
- Every file is the complete output of the program named; nothing was filtered.
