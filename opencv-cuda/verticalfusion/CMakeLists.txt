# Vertical fusion (VF) against OpenCV-CUDA: one executable per operation chain, with one generated translation unit per
# experiment (kernelN.cu fuses FIRST_VALUE + INCREMENT * (N - 1) operations: 2, 102, ... 19902 with 200 experiments).
# nvcc writes its phase times of every kernelN.cu to <generated dir>/kernelN.csv (-time, from cvGS commit 4766bf3).

set(NUM_EXPERIMENTS 200 CACHE STRING "Experiments (translation units) per VF benchmark; 200 = the paper's 19902 operations")
math(EXPR NUM_EXPERIMENTS_AS_INT "${NUM_EXPERIMENTS}")
if(NUM_EXPERIMENTS_AS_INT LESS_EQUAL 0)
    message(FATAL_ERROR "NUM_EXPERIMENTS must be a positive integer, got ${NUM_EXPERIMENTS}")
endif()
option(FKLB_BUILD_VF_EXTRA "Also build the VF chains that are not in the paper (mul, add, sub, div)" OFF)

set(VF_DIR "${CMAKE_CURRENT_SOURCE_DIR}/vertical_fusion_kernel_instances")

function(fklb_add_vf_benchmark TARGET_NAME INSTANCE)
    set(OPTYPE ${INSTANCE})   # substituted into the templates
    set(GEN "${CMAKE_CURRENT_BINARY_DIR}/generated/${INSTANCE}/${NUM_EXPERIMENTS_AS_INT}_experiments")
    file(MAKE_DIRECTORY "${GEN}")
    set(KERNEL_SOURCES "")
    set(LAUNCHER_INCLUDES_BLOCK "")
    set(LAUNCHER_DISPATCH_BLOCK "")
    foreach(N RANGE 1 ${NUM_EXPERIMENTS_AS_INT})
        configure_file("${VF_DIR}/templates/experiment_kernel.h.in" "${GEN}/kernel${N}.h" @ONLY)
        configure_file("${VF_DIR}/templates/experiment_kernel.cu.in" "${GEN}/kernel${N}.cu" @ONLY)
        set_source_files_properties("${GEN}/kernel${N}.cu" PROPERTIES COMPILE_OPTIONS "-time=${GEN}/kernel${N}.csv")
        list(APPEND KERNEL_SOURCES "${GEN}/kernel${N}.cu")
        string(APPEND LAUNCHER_INCLUDES_BLOCK "#include \"kernel${N}.h\"\n")
        string(APPEND LAUNCHER_DISPATCH_BLOCK "DISPATCH_INSTANCE(${N})\n")
    endforeach()
    set(GENERATED_INCLUDES "${LAUNCHER_INCLUDES_BLOCK}")
    set(GENERATED_DISPATCH_CALLS "${LAUNCHER_DISPATCH_BLOCK}")
    set(NUM_EXPERIMENTS_TOTAL ${NUM_EXPERIMENTS_AS_INT})
    configure_file("${VF_DIR}/templates/experiment_launcher.h.in" "${GEN}/launcher.h" @ONLY)

    fklb_add_benchmark(${TARGET_NAME}
        BACKENDS cu LIBS cvgs opencv
        SOURCES "${VF_DIR}/${INSTANCE}/launcher.cu" ${KERNEL_SOURCES}
        INCLUDE_DIRS "${GEN}"
        DEFINITIONS CPP_NUM_EXPERIMENTS=${NUM_EXPERIMENTS_AS_INT})
endfunction()

# In the paper: Fig. VerticalOnly (mul_add1, mul1: batch 1) and Fig. VerticalHorizontalFusion (mul_add: batch 50)
fklb_add_vf_benchmark(benchmark_vertical_fusionCompareMulAdd1 mul_add1)
fklb_add_vf_benchmark(benchmark_vertical_fusionCompareMul1    mul1)
fklb_add_vf_benchmark(benchmark_vertical_fusionCompareMulAdd  mul_add)
if(FKLB_BUILD_VF_EXTRA)
    fklb_add_vf_benchmark(benchmark_vertical_fusionCompareMul mul)
    fklb_add_vf_benchmark(benchmark_vertical_fusionCompareAdd add)
    fklb_add_vf_benchmark(benchmark_vertical_fusionCompareSub sub)
    fklb_add_vf_benchmark(benchmark_vertical_fusionCompareDiv div)
endif()
