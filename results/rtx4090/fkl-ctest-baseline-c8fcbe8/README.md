# FKL ctest matrix, the 46 CUDA tests, baseline build (FKL c8fcbe8): control run of 2026-11-06 16:50:00 to 16:50:12

Window A approved by the coordinator as a control before the head of the library-checks PR exists. Build: `scratch/fkl-check-baseline-fkltests-c8fcbe8/fkltests-build` (FKL c8fcbe8, compiled by `scripts/check_fkl_tree.sh --only fkltests`, 125 of 125 executables, sm_89, nvcc 13.4.92, g++ 15.2; the binaries' sha256 are in `binary-sha256.txt`). Command: `ctest -R '_cu$' --timeout 600 -j1 --output-on-failure --no-compress-output -T Test`, `CUDA_VISIBLE_DEVICES=1` (the RTX 4090, the desktop session running on it), pinned to cores 0-3,16-19, one test at a time. The tests are functional tests, not measurements: pass / fail per test, no timing claim (the times below are ctest's wall times of one run). Load before 0.00 0.25 1.99, after 0.23 0.29 1.98 (`environment-before.txt`, `environment-after.txt`, `load.log` every 5 s).

**Result: 46 of 46 passed, 0 failed, 0 timeouts, 0 not run; ctest exit 0; total test time (real) 7.11 s; the ctest process ran 16:50:01 to 16:50:08.** The slowest test is `utest_saturate_cu` (1.21 s), the next ones are below 0.6 s; 23 tests report 0.00 s (ctest rounds to 10 ms; I did not look at what each of them does at run time). Nothing at the baseline fails, so there is no pre-existing failure of c8fcbe8 to explain.

| # | test | result | ctest time (s) |
|--:|---|---|--:|
| 2 | `test_batchresize_build_cu` | Passed | 0.00 |
| 4 | `test_border_reader_cu` | Passed | 0.00 |
| 6 | `test_crop_cu` | Passed | 0.00 |
| 8 | `test_deinterlace_cu` | Passed | 0.00 |
| 10 | `test_raw_image_types_cu` | Passed | 0.00 |
| 12 | `test_warp_cu` | Passed | 0.00 |
| 14 | `batch_build_compilation_time_cu` | Passed | 0.26 |
| 15 | `test_nvcc124_126_BugReproducer_cu` | Passed | 0.00 |
| 16 | `test_nvcc1281_129_BugReproducer_cu` | Passed | 0.00 |
| 18 | `test_nvcc131_BugReproducer_cu` | Passed | 0.00 |
| 20 | `basic_test_cu` | Passed | 0.55 |
| 21 | `test_circular_tensor_cu` | Passed | 0.54 |
| 23 | `test_ptr_nd_cu` | Passed | 0.45 |
| 25 | `test_rect_cu` | Passed | 0.00 |
| 27 | `test_tuple_cu` | Passed | 0.00 |
| 29 | `test_typelist_cu` | Passed | 0.00 |
| 31 | `compiler_explorer_example_cu` | Passed | 0.18 |
| 32 | `inlining_and_LDL_STL_cu` | Passed | 0.28 |
| 34 | `readme_test_code_cu` | Passed | 0.20 |
| 35 | `test_divergent_hf_executor_cu` | Passed | 0.19 |
| 37 | `test_filtered_index_sequence_cu` | Passed | 0.00 |
| 39 | `test_fused_operation_cu` | Passed | 0.00 |
| 41 | `test_instantiable_operations_cu` | Passed | 0.20 |
| 43 | `test_operation_fuser_cu` | Passed | 0.00 |
| 45 | `test_operation_tuple_cu` | Passed | 0.20 |
| 47 | `test_operation_types_cu` | Passed | 0.00 |
| 49 | `test_IOps_to_string_cu` | Passed | 0.19 |
| 51 | `test_fk_string_cu` | Passed | 0.00 |
| 53 | `test_literal_to_string_cu` | Passed | 0.00 |
| 55 | `test_struct_to_string_cu` | Passed | 0.00 |
| 57 | `test_large_grids_cu` | Passed | 0.33 |
| 59 | `test_thread_fusion_cu` | Passed | 0.00 |
| 61 | `test_thread_fusion_alignment_cu` | Passed | 0.49 |
| 62 | `utest_flash_attention_cu` | Passed | 0.18 |
| 63 | `utest_softmax_cu` | Passed | 0.20 |
| 65 | `utest_color_conversion_cu` | Passed | 0.18 |
| 67 | `utest_deinterlace_cu` | Passed | 0.19 |
| 69 | `utest_resize_sampling_cu` | Passed | 0.23 |
| 71 | `utest_yuv_to_rgb_matrices_cu` | Passed | 0.20 |
| 73 | `utest_constexpr_cmath_cu` | Passed | 0.00 |
| 79 | `utest_executors_cu` | Passed | 0.20 |
| 80 | `utest_stream_ownership_cu` | Passed | 0.19 |
| 82 | `utest_stream_ref_counts_cu` | Passed | 0.00 |
| 84 | `utest_VBase_cu` | Passed | 0.00 |
| 86 | `utest_get_arg_large_pack_cu` | Passed | 0.21 |
| 88 | `utest_saturate_cu` | Passed | 1.21 |

Files: `ctest-stdout.txt` (the console of ctest), `ctest-output.log`, `LastTest.log`, `Test.xml` (the output of every test), `test-list.txt`, `binary-sha256.txt`, `gpu-info.csv`, `gpu-after.csv`, `window-console.txt`, `window-script.sh`.
