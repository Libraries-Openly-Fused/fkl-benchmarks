# FKL pipelines. E4b: the NV12 video pre-processing pipeline for every back-end (cu, hip, cpp). The cpp target also runs
# ParArch::CPU_MULTITHREAD when external/fkl/thread-pool is checked out (LIBS fkl_mt).
fklb_add_benchmark(benchmark_nv12_pipeline
    BACKENDS cu hip cpp LIBS fkl_mt
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_nv12_pipeline.h")
