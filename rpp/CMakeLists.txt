# FKL on HIP against AMD's RPP (Radeon Performance Primitives, the rppt tensor API with the HIP back-end) on the pre-processing
# pipeline (a) of the paper. Needs a HIP compiler, an AMD GPU and the RPP package of ROCm (find_package(rpp)); -DFKLB_BUILD_RPP=ON.
fklb_add_benchmark(benchmark_rpp_preprocessing
    BACKENDS hip LIBS fkl rpp
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rpp_preprocessing.h"
    INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external/FastRPP/include")

# E18 (AMD): a 3 x 3 box filter after a crop and a resize, in 8 bits from start to end: RPP (rppt_resize, then rppt_box_filter) against FKL's
# LinearFilterDPP in one kernel. ROCTx (rocprofiler-sdk-roctx, part of ROCm) names the calls for rocprofv3, as NVTX does in the CUDA programs.
fklb_add_benchmark(e18_box
    BACKENDS hip LIBS fkl rpp
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/e18_box.h")
if(TARGET e18_box_hip)
    find_package(rocprofiler-sdk-roctx CONFIG REQUIRED)
    target_link_libraries(e18_box_hip PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
endif()

# E19 (AMD): the 3 x 3 box filter inside a batched pipeline (grey before it, a float conversion after it) on 30 crops of one frame: FKL in a
# loop of fused calls (fkl_loop) and in one batched call (fkl_batch), against RPP (rpp). The AMD side of opencv-cuda/stencil/e19_batched_box.cu.
fklb_add_benchmark(e19_batched_box
    BACKENDS hip LIBS fkl rpp
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/e19_batched_box.h")
if(TARGET e19_batched_box_hip)
    find_package(rocprofiler-sdk-roctx CONFIG REQUIRED)
    target_link_libraries(e19_batched_box_hip PRIVATE rocprofiler-sdk-roctx::rocprofiler-sdk-roctx)
endif()
