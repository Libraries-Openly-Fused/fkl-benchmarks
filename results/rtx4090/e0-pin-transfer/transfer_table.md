| group | benchmarks commit | baseline (library) | binary | results it produced | kernels in baseline | in c8fcbe8 build | identical | different | only on one side | matched by |
|---|---|---|---|---|--:|--:|--:|--:|--:|---|
| a334424 | a334424 | frozen binary of the measurement (fb9ac3a) | `benchmark_latency_hiding_cu` | e0-latency-hiding | 61 | 61 | 61 | 0 | 0 | mangled name |
| a334424 | a334424 | frozen binary of the measurement (fb9ac3a) | `benchmark_numinstructions_x_operation_cu` | e0-numinstructions | 185 | 185 | 185 | 0 | 0 | mangled name |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPUandGPU_cvGS_vs_fk_cu` | e0-wrapper-overhead | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPU_OpenCV_vs_cvGS_cu` | e0-cpu-time-opencv | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 269d229 | 269d229 | frozen binary of the measurement (fb9ac3a) | `benchmark_data_type_cu` | e0-data-type | 26 | 26 | 26 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 5aac244 | 5aac244 | frozen binary of the measurement (fb9ac3a) | `benchmark_CPU_NPP_vs_FastNPP_cu` | e0-cpu-time-npp | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 5aac244 | 5aac244 | frozen binary of the measurement (fb9ac3a) | `benchmark_npp_batchresize_x_split3D_cu` | e0-npp | 15 | 15 | 15 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_image_resolution_MAD_loop_cu` | e0-data-size | 6 | 6 | 6 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_horizontal_fusion_cu` | e0-hf | 122 | 122 | 122 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `benchmark_CPUandGPU_cvGS_vs_fk_cu` | e0-wrapper-overhead-order | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| 1a70cab | 1a70cab | frozen binary of the measurement (fb9ac3a) | `resize_convention_cu` | e6-resize-convention | 2 | 2 | 2 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `graph_instantiate_time_cu` | e0-graph-instantiate | 0 | 0 | 0 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMulAdd1_cu` | e0-vf-mul-add1, e0-vf-mul-add1-run1 | 200 | 200 | 200 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMul1_cu` | e0-vf-mul1 | 200 | 200 | 200 | 0 | 0 | mangled name |
| a266bc9 | a266bc9 | frozen binary of the measurement (fb9ac3a) | `benchmark_vertical_fusionCompareMulAdd_cu` | e0-vf-mul-add | 200 | 200 | 200 | 0 | 0 | mangled name |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_vf_chain_cu` | e1-vf-chain-v2 | 18 | 18 | 18 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_hf_batch_cu` | e1-hf-batch-v2 | 16 | 16 | 16 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e1 | 74f47c6 | frozen binary of the measurement (38c4f32) | `e1_preproc_cu` | e1-preproc-v3 | 20 | 20 | 20 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e2-old | e9e599d | rebuild at 38c4f32 against the frozen binary (source check) (38c4f32) | `e2_fkl_chain_cu` | e2-chain (same binary rebuilt at the old library) | 7 | 7 | 7 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| e2-new | e9e599d | frozen binary of the measurement (38c4f32) | `e2_fkl_chain_cu` | e2-chain | 7 | 7 | 7 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| mul4-old | e9e599d | rebuild at 38c4f32 against the frozen binary (source check) (38c4f32) | `benchmark_image_resolution_MAD_loop_mul4_cu` | e0-data-size-mul4 (same binary rebuilt at the old library) | 2 | 2 | 2 | 0 | 0 | mangled name |
| mul4-new | e9e599d | frozen binary of the measurement (38c4f32) | `benchmark_image_resolution_MAD_loop_mul4_cu` | e0-data-size-mul4, e0-data-size-parity | 2 | 2 | 2 | 0 | 0 | mangled name |
| nan_saturate | f3afadc | frozen binary of the measurement (fb9ac3a) | `nan_saturate_cu` | e6-nan-saturate | 1 | 1 | 1 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| nv12 | f81306d | rebuild at fb9ac3a (no frozen binary exists) (fb9ac3a) | `benchmark_nv12_pipeline_cu` | e4b-nv12 | 15 | 15 | 15 | 0 | 0 | demangled name, ResizeSampling argument and anonymous-namespace hash ignored |
| **total (new tree against the baseline)** | | | 22 binaries | | **1132** | **1132** | **1132** | **0** | **0** | |
