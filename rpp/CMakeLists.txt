# FKL on HIP against AMD's RPP (Radeon Performance Primitives, the rppt tensor API with the HIP back-end) on the two pre-processing
# pipelines of the paper. Needs a HIP compiler, an AMD GPU and the RPP package of ROCm (find_package(rpp)); -DFKLB_BUILD_RPP=ON.
fklb_add_benchmark(benchmark_rpp_preprocessing
    BACKENDS hip LIBS fkl rpp
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_rpp_preprocessing.h")
