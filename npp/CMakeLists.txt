# FastNPP against NPP. CUDA only (NPP).

fklb_add_benchmark(benchmark_npp_batchresize_x_split3D
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batchresize/benchmark_npp_batchresize_x_split3D.cu")

# E9: operations of cvGPUSpeedup, of FastNPP and of the user in one kernel (needs OpenCV for the cvGS helpers).
if(FKLB_BUILD_OPENCV_CUDA)
    fklb_add_benchmark(e9_cross_library
        BACKENDS cu LIBS fastnpp npp cvgs opencv
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/e9_cross_library.cu")
endif()

fklb_add_benchmark(benchmark_CPU_NPP_vs_FastNPP
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CPU/benchmark_CPU_NPP_vs_FastNPP.cu")
