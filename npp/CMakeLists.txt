# FastNPP against NPP. CUDA only (NPP).

fklb_add_benchmark(benchmark_npp_batchresize_x_split3D
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batchresize/benchmark_npp_batchresize_x_split3D.cu")

fklb_add_benchmark(benchmark_CPU_NPP_vs_FastNPP
    BACKENDS cu LIBS fastnpp npp
    SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CPU/benchmark_CPU_NPP_vs_FastNPP.cu")

# Diagnostic (run only in a paused window): the same two benchmarks with the NPP stream context built correctly. initNppStreamContext of FastNPP's
# tests/testsNppCommon.h stores the MINOR compute capability in the "major" field and the major in the "minor" one (a 4090, 8.9, is passed to NPP as 9.8).
# The corrected copy of that header is generated here from the pinned one and placed first in the include path, so FastNPP itself is not touched; if FastNPP
# is fixed upstream the pattern below is not found and the targets are not created.
file(READ "${CMAKE_SOURCE_DIR}/external/FastNPP/tests/testsNppCommon.h" FKLB_NPP_COMMON)
string(REPLACE "&ccmajor, cudaDevAttrComputeCapabilityMinor" "&ccmajor, cudaDevAttrComputeCapabilityMajor" FKLB_NPP_COMMON_FIXED "${FKLB_NPP_COMMON}")
string(REPLACE "&ccminor, cudaDevAttrComputeCapabilityMajor" "&ccminor, cudaDevAttrComputeCapabilityMinor" FKLB_NPP_COMMON_FIXED "${FKLB_NPP_COMMON_FIXED}")
if(NOT FKLB_NPP_COMMON_FIXED STREQUAL FKLB_NPP_COMMON)
    file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/context-diagnostic/tests/testsNppCommon.h" "${FKLB_NPP_COMMON_FIXED}")
    fklb_add_benchmark(benchmark_npp_batchresize_x_split3D_ctxfix
        BACKENDS cu LIBS fastnpp npp
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/batchresize/benchmark_npp_batchresize_x_split3D.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/context-diagnostic")
    fklb_add_benchmark(benchmark_CPU_NPP_vs_FastNPP_ctxfix
        BACKENDS cu LIBS fastnpp npp
        SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/CPU/benchmark_CPU_NPP_vs_FastNPP.cu"
        INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/context-diagnostic")
else()
    message(STATUS "fkl-benchmarks: the NPP stream-context diagnostic is skipped (FastNPP's tests/testsNppCommon.h no longer has the swapped attributes)")
endif()
