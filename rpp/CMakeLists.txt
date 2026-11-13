# FKL on HIP against AMD's RPP (Radeon Performance Primitives, the rppt tensor API with the HIP back-end) on the pre-processing
# pipeline (a) of the paper. Needs a HIP compiler, an AMD GPU and the RPP package of ROCm (find_package(rpp)); -DFKLB_BUILD_RPP=ON.
fklb_add_benchmark(benchmark_rpp_preprocessing
    BACKENDS hip LIBS fkl rpp
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rpp_preprocessing.h"
    INCLUDE_DIRS "${CMAKE_SOURCE_DIR}/external/FastRPP/include")

# E18 (AMD): the FKL side of opencv-cuda/stencil/e18_stencil.cu, LinearFilterDPP after a resize in one kernel (RPP has no filter with user weights)
fklb_add_benchmark(e18_stencil
    BACKENDS hip LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/e18_stencil.h")
