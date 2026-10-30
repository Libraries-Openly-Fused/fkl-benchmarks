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
