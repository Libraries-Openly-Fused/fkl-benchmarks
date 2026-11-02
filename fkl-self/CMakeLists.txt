# FKL against itself. Each benchmark is one header that defines launch(); it is built for every back-end listed.
# CSV files: <function>_<ParArch>.csv (see fkl-self/oneExecutionBenchmark.h and twoExecutionsBenchmark.h).

fklb_add_benchmark(benchmark_horizontal_fusion
    BACKENDS cu hip cpp LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/fusion/benchmark_horizontal_fusion.h")

fklb_add_benchmark(benchmark_thread_fusion
    BACKENDS cu hip cpp LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/fusion/benchmark_thread_fusion.h")

fklb_add_benchmark(benchmark_latency_hiding
    BACKENDS cu hip cpp LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/latencyhiding/benchmark_latency_hiding.h")

fklb_add_benchmark(benchmark_numinstructions_x_operation
    BACKENDS cu hip cpp LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/numinstructions/benchmark_numinstructions_x_operation.h")

# E0 baseline fairness check of the CUDA-Graphs series of the HF experiment: the same kernels and parameters as plain launches, as the experiment's child-node graph, as chained child nodes, and as flat graphs
# (results/rtx4090/e0-graph-structure). CUDA only: the structures are CUDA graph API calls.
fklb_add_benchmark(hf_graph_structure_probe
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graphs/hf_graph_structure_probe.cu")

# E0: the time of the fused kernel of the HF experiment in each loop of the experiment and on its own (results/rtx4090/e0-graph-structure/fused-time). CUDA only.
fklb_add_benchmark(hf_fused_time_probe
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/graphs/hf_fused_time_probe.cu")

# E16: forward attention with FKL (the tensor-core and the SIMT DPP; K and V read raw, or dequantised in the prologue from int8 / fp8 e4m3 with one scale per token; an epilogue chain
# in registers) against the same work as separate kernels (results/rtx4090/e16-attention, results/rtx5050/e16-attention). CUDA only. Baselines: pytorch-attention/e16_baselines.py.
fklb_add_benchmark(e16_attention
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/attention/e16_attention.cu")

# Link check (results/rtx4090/pcie-check): host to device and device to host copy bandwidth of pinned and pageable memory while the link state is sampled. CUDA only.
fklb_add_benchmark(pcie_bandwidth
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/pcie/pcie_bandwidth.cu")

# Seed of the compile-time generator (E3): copied from FKL tests/buildAPI/. Host-side build() of one NV12 pipeline.
fklb_add_benchmark(batch_build_compilation_time
    BACKENDS cu hip cpp LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/compiletime/batch_build_compilation_time.h")

# FKL pipelines (e.g. the NV12 preprocessing, E4b): pipelines/CMakeLists.txt, when it exists
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/pipelines/CMakeLists.txt")
    add_subdirectory(pipelines)
endif()

# E6: the value of fk::SaturateCast for NaN on the device (cu only: it reads the hardware conversions through inline PTX)
fklb_add_benchmark(nan_saturate
    BACKENDS cu LIBS fkl
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/numerics/nan_saturate.cu")

# E6: fk::SaturateCast<float, T> over all 2^32 float bit patterns against cv::cuda::device::saturate_cast, the PTX conversion and the host functions
fklb_add_benchmark(e6_saturate_exhaustive
    BACKENDS cu LIBS fkl opencv
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/numerics/saturate_exhaustive.cu")

# E6 (AMD): fk::SaturateCast<float, T> over all 2^32 float bit patterns on the device (gfx1200) against the host, the conversion of HIP and an independent reference
fklb_add_benchmark(e6_saturate_exhaustive_amd
    BACKENDS hip LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/numerics/e6_saturate_exhaustive_amd.h")
