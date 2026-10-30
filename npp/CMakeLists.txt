# FastNPP against NPP. CUDA only (NPP).

fklb_add_benchmark(benchmark_npp_batchresize_x_split3D
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batchresize/benchmark_npp_batchresize_x_split3D.cu")

fklb_add_benchmark(benchmark_CPU_NPP_vs_FastNPP
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CPU/benchmark_CPU_NPP_vs_FastNPP.cu")
