# FKL ctest matrix, the 48 CUDA tests, at the head of PR #423 (9e20bd4e): run of 2026-11-06 19:09:00 to 19:09:11 (window B)

Window B approved by the coordinator (start announced for 19:09:00, 5 minutes reserved, nothing else running on the machine from then until it ended). Library tree under test: the head of FusedKernelLibrary PR #423, `9e20bd4e5286be70fbeb0f79a19b96f00c81a06f` (details, compile check and SASS comparison: `../fkl-pr423-check/README.md`). Build: `scratch/pr423/check-fkltests-rerun/fkltests-build`, the build of the final run of `scripts/check_fkl_tree.sh --only fkltests` on that head (141 targets, 129 compiled, sm_89, nvcc 13.4.92, g++ 15.2; the build to which `../fkl-pr423-check/ctest-cpu.log` and `fkltests-final-report.txt` belong; the coordinator accepted it in place of the first build, `check-fast/fkltests-build`, which has the same tree and flags); the 48 test executables' sha256 are in `binary-sha256.txt`. Command (`window-script.sh`, the script of window A with the build, the output folder and a shorter hard timeout changed): `ctest -R '_cu$' --timeout 600 -j1 --output-on-failure --no-compress-output -T Test` under `timeout -k 10 290`, `CUDA_VISIBLE_DEVICES=1` (the RTX 4090, the desktop session running on it: 449 MiB used, P8, 0 % before; 449 MiB, 0 %, 300 MHz, 41 C after, `gpu-info.csv`, `gpu-after.csv`), pinned to cores 0-3,16-19, one test at a time. The tests are functional tests, not measurements: pass / fail per test, no timing claim (the times below are ctest's wall times of this one run). Load before 0.07 0.22 2.59, after 0.20 0.24 2.57 (`environment-before.txt`, `environment-after.txt`, `load.log` every 5 s; the only process above 3 % in the log is the desktop's file indexer).

**Result: 48 of 48 passed, 0 failed, 0 timeouts, 0 not run; ctest exit 0; total test time (real) 7.43 s; the ctest process ran 19:09:00 to 19:09:08.** The 46 tests of window A (`../fkl-ctest-baseline-c8fcbe8/`, FKL c8fcbe8: 46 of 46, 7.11 s) passed again, and the two tests that the head adds passed: `utest_chain_check_cu` (0.18 s: per its CMake file, the predicate on the types of the chains and the correct chains executed, on the CUDA back-end) and `utest_flash_attention_mma_cu` (0.21 s: the E16 utest of #422, which is in the base 509d987). The slowest test is `utest_saturate_cu` (1.21 s), the next ones are below 0.6 s; 23 tests report 0.00 s (ctest rounds to 10 ms; I did not look at what each of them does at run time).
What this does not cover: the 12 compile-fail tests (`*_rejected`) and the 45 CPU tests are in `../fkl-pr423-check/` (12 of 12 and 45 of 45 passed); nothing here says anything about speed.

| # | test | result | ctest time (s) |
|--:|---|---|--:|
| 2 | `test_batchresize_build_cu` | Passed | 0.00 |
| 4 | `test_border_reader_cu` | Passed | 0.00 |
| 6 | `test_crop_cu` | Passed | 0.00 |
| 8 | `test_deinterlace_cu` | Passed | 0.00 |
| 10 | `test_raw_image_types_cu` | Passed | 0.00 |
| 12 | `test_warp_cu` | Passed | 0.00 |
| 14 | `batch_build_compilation_time_cu` | Passed | 0.27 |
| 15 | `test_nvcc124_126_BugReproducer_cu` | Passed | 0.00 |
| 16 | `test_nvcc1281_129_BugReproducer_cu` | Passed | 0.00 |
| 18 | `test_nvcc131_BugReproducer_cu` | Passed | 0.00 |
| 20 | `basic_test_cu` | Passed | 0.53 |
| 21 | `test_circular_tensor_cu` | Passed | 0.54 |
| 23 | `test_ptr_nd_cu` | Passed | 0.45 |
| 25 | `test_rect_cu` | Passed | 0.00 |
| 27 | `test_tuple_cu` | Passed | 0.00 |
| 29 | `test_typelist_cu` | Passed | 0.00 |
| 31 | `compiler_explorer_example_cu` | Passed | 0.19 |
| 32 | `inlining_and_LDL_STL_cu` | Passed | 0.28 |
| 34 | `readme_test_code_cu` | Passed | 0.21 |
| 35 | `test_divergent_hf_executor_cu` | Passed | 0.19 |
| 37 | `test_filtered_index_sequence_cu` | Passed | 0.00 |
| 39 | `test_fused_operation_cu` | Passed | 0.00 |
| 41 | `test_instantiable_operations_cu` | Passed | 0.19 |
| 43 | `test_operation_fuser_cu` | Passed | 0.00 |
| 45 | `test_operation_tuple_cu` | Passed | 0.18 |
| 47 | `test_operation_types_cu` | Passed | 0.00 |
| 49 | `test_IOps_to_string_cu` | Passed | 0.20 |
| 51 | `test_fk_string_cu` | Passed | 0.00 |
| 53 | `test_literal_to_string_cu` | Passed | 0.00 |
| 55 | `test_struct_to_string_cu` | Passed | 0.00 |
| 57 | `test_large_grids_cu` | Passed | 0.33 |
| 59 | `test_thread_fusion_cu` | Passed | 0.00 |
| 61 | `test_thread_fusion_alignment_cu` | Passed | 0.48 |
| 62 | `utest_flash_attention_cu` | Passed | 0.19 |
| 63 | `utest_flash_attention_mma_cu` | Passed | 0.21 |
| 64 | `utest_softmax_cu` | Passed | 0.20 |
| 66 | `utest_color_conversion_cu` | Passed | 0.18 |
| 68 | `utest_deinterlace_cu` | Passed | 0.18 |
| 70 | `utest_resize_sampling_cu` | Passed | 0.22 |
| 72 | `utest_yuv_to_rgb_matrices_cu` | Passed | 0.18 |
| 74 | `utest_constexpr_cmath_cu` | Passed | 0.00 |
| 78 | `utest_chain_check_cu` | Passed | 0.18 |
| 82 | `utest_executors_cu` | Passed | 0.19 |
| 83 | `utest_stream_ownership_cu` | Passed | 0.19 |
| 85 | `utest_stream_ref_counts_cu` | Passed | 0.00 |
| 87 | `utest_VBase_cu` | Passed | 0.00 |
| 89 | `utest_get_arg_large_pack_cu` | Passed | 0.21 |
| 91 | `utest_saturate_cu` | Passed | 1.21 |

## Files
`ctest-stdout.txt` (the ctest output of the run), `ctest-output.log` (ctest's own log with the output of every test), `LastTest.log`, `Test.xml`, `ctest-exit-code.txt`, `test-list.txt` (`ctest -N -R '_cu$'`, 48 tests), `binary-sha256.txt`, `environment-before.txt`, `environment-after.txt`, `load.log`, `gpu-info.csv`, `gpu-after.csv`, `window-console.txt`, `window-script.sh`. Every file is the complete output of the program named; nothing was filtered.
