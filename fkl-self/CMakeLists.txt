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
