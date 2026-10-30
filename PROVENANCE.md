# Provenance of the moved and copied files

Every file below was taken verbatim (first commit of this repository) from the submodule commit pinned in `external/`:

- `external/fkl` = FusedKernelLibrary @ `25057d8`
- `external/cvGPUSpeedup` = cvGPUSpeedup-FastOpenCV @ `6d26c65`
- `external/FastNPP` = FastNPP @ `9dfc4f4`

`moved` = the file also lives in the library's `benchmarks/` directory, which is removed from the library by a PR (the move). `copied (shared harness file)` = lives in the library's `tests/` directory and stays there; the copy here is the benchmark-side harness. Later commits change these files (include paths, harness fixes); `git log -p -- <file>` shows every change against the verbatim first copy. `last change` = the last commit that touched the source path in the source repository.

| kind | source repository | source path | pinned commit | last change | file here | sha256 (12) of the verbatim copy |
|---|---|---|---|---|---|---|
| moved | FastNPP | `benchmarks/CPU/benchmark_CPU_NPP_vs_FastNPP.cu` | 9dfc4f4 | 328573d | `npp/CPU/benchmark_CPU_NPP_vs_FastNPP.cu` | fe77588ce986 |
| moved | FastNPP | `benchmarks/batchresize/benchmark_npp_batchresize_x_split3D.cu` | 9dfc4f4 | 9796a91 | `npp/batchresize/benchmark_npp_batchresize_x_split3D.cu` | 94ac54cac69b |
| moved | FastNPP | `benchmarks/nppbenchmark.h` | 9dfc4f4 | 0f2e5c2 | `npp/nppbenchmark.h` | 2fd6279fa3e5 |
| moved | FusedKernelLibrary | `benchmarks/fkBenchmarksCommon.h` | 25057d8 | 7309118 | `fkl-self/fkBenchmarksCommon.h` | 897512de628f |
| moved | FusedKernelLibrary | `benchmarks/fusion/benchmark_horizontal_fusion.h` | 25057d8 | 7309118 | `fkl-self/fusion/benchmark_horizontal_fusion.h` | d1dd182abd98 |
| moved | FusedKernelLibrary | `benchmarks/fusion/benchmark_thread_fusion.h` | 25057d8 | fcb22f2 | `fkl-self/fusion/benchmark_thread_fusion.h` | 4d2de5ead987 |
| moved | FusedKernelLibrary | `benchmarks/latencyhiding/benchmark_latency_hiding.h` | 25057d8 | 4ca7f32 | `fkl-self/latencyhiding/benchmark_latency_hiding.h` | e04acc98164b |
| moved | FusedKernelLibrary | `benchmarks/launcher.in` | 25057d8 | b985ee0 | `common/launcher.in` | 8ee236a2b556 |
| moved | FusedKernelLibrary | `benchmarks/numinstructions/benchmark_numinstructions_x_operation.h` | 25057d8 | 4ca7f32 | `fkl-self/numinstructions/benchmark_numinstructions_x_operation.h` | 5f0d74889730 |
| moved | FusedKernelLibrary | `benchmarks/oneExecutionBenchmark.h` | 25057d8 | e32d3c5 | `fkl-self/oneExecutionBenchmark.h` | 971cda4940b5 |
| moved | FusedKernelLibrary | `benchmarks/twoExecutionsBenchmark.h` | 25057d8 | d0519a9 | `fkl-self/twoExecutionsBenchmark.h` | 726be170c593 |
| copied (shared harness file) | FusedKernelLibrary | `tests/main.cpp` | 25057d8 | 2392e4d | `common/main.cpp` | 84953890f6f7 |
| copied (shared harness file) | FusedKernelLibrary | `tests/main.h` | 25057d8 | acb8fe1 | `common/main.h` | b5ae0cd9b3d2 |
| copied (shared harness file) | FusedKernelLibrary | `tests/nvtx.h` | 25057d8 | 8fe6784 | `common/nvtx.h` | 67e8bb66ce8f |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/benchmark_CPU_OpenCV_vs_cvGS.cu` | 6d26c65 | ece01c9 | `opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu` | 9c3b309b4d88 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/benchmark_CPUandGPU_cvGS_vs_fk.cu` | 6d26c65 | f20a94e | `opencv-cuda/benchmark_CPUandGPU_cvGS_vs_fk.cu` | 36e188c93615 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/benchmark_image_resolution_MAD_loop.cu` | 6d26c65 | ece01c9 | `opencv-cuda/benchmark_image_resolution_MAD_loop.cu` | c814ee9dc1e6 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/add/CMakeLists.txt` | 6d26c65 | fdc0dd4 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/add/CMakeLists.txt` | 396ee897e613 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/add/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/add/launcher.cu` | cba8971da95a |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/add/opType.cuh` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/add/opType.cuh` | 38ea821f31a9 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/add/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/add/realBatch.h` | 91547f919515 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/div/CMakeLists.txt` | 6d26c65 | 2d420d8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/div/CMakeLists.txt` | 6c8c77368682 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/div/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/div/launcher.cu` | 4ee0871acc3b |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/div/opType.cuh` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/div/opType.cuh` | e71363f0f68d |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/div/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/div/realBatch.h` | 91547f919515 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul/CMakeLists.txt` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul/CMakeLists.txt` | 4dc646222216 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul/launcher.cu` | 6e27f9b2a84e |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul/opType.cuh` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul/opType.cuh` | 38b01415ba6a |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul/realBatch.h` | 91547f919515 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul1/CMakeLists.txt` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul1/CMakeLists.txt` | bbb44fa8c397 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul1/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul1/launcher.cu` | d7794ebe142d |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul1/opType.cuh` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul1/opType.cuh` | 38b01415ba6a |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul1/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul1/realBatch.h` | 1567ff8b07ed |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add/CMakeLists.txt` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/CMakeLists.txt` | f8cbbadc33ef |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/launcher.cu` | 32014c6c8359 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add/opType.cuh` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/opType.cuh` | 2fe2f71fb3d3 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add/realBatch.h` | 91547f919515 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add1/CMakeLists.txt` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add1/CMakeLists.txt` | 7db51d518882 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add1/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add1/launcher.cu` | 4416bc8f57ac |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add1/opType.cuh` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add1/opType.cuh` | 2fe2f71fb3d3 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/mul_add1/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/mul_add1/realBatch.h` | 1567ff8b07ed |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/sub/CMakeLists.txt` | 6d26c65 | ee0253d | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/sub/CMakeLists.txt` | 63aebdc17ee7 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/sub/launcher.cu` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/sub/launcher.cu` | 627cc3bcdeb2 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/sub/opType.cuh` | 6d26c65 | e52a1f8 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/sub/opType.cuh` | 6891affb9d62 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/sub/realBatch.h` | 6d26c65 | ece01c9 | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/sub/realBatch.h` | 91547f919515 |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_kernel.cu.in` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_kernel.cu.in` | 32a84c5a5f4d |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_kernel.h.in` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_kernel.h.in` | 68319b51e56f |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_launcher.h.in` | 6d26c65 | d4e654a | `opencv-cuda/verticalfusion/vertical_fusion_kernel_instances/templates/experiment_launcher.h.in` | 52bd96848d5a |
| moved | cvGPUSpeedup-FastOpenCV | `benchmarks/verticalfusion/vertical_fusion_static_loop.cuh` | 6d26c65 | fdce43a | `opencv-cuda/verticalfusion/vertical_fusion_static_loop.cuh` | cd5a6452ea53 |
| copied (shared harness file) | cvGPUSpeedup-FastOpenCV | `tests/testsCommon.cuh` | 6d26c65 | acf69a9 | `opencv-cuda/testsCommon.cuh` | e0c8581b7182 |
