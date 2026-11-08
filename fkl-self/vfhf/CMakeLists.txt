# benchmark_vfhf_max_speedup: the maximum speedup of vertical and horizontal fusion on one GPU, FKL against itself (System 6 of the GPU-size
# figure; see benchmark_vfhf_max_speedup.h). One generated translation unit per number of operations instantiates the fused side
# (vfhf::fused<NUM_OPS>, vfhf_fused.h), as opencv-cuda/verticalfusion/ does with one kernelN.cu per experiment. Back-ends cu and hip.

set(FKLB_VFHF_POINTS "2;102;502;1002;2002;3002;4002;5002;6002;7002;8002;9002;10002;12002;14002;16002;18002;19902" CACHE STRING
    "benchmark_vfhf_max_speedup: the numbers of operations, each 2 + 100 (k - 1), semicolon separated; one generated translation unit each")

set(VFHF_GEN "${CMAKE_CURRENT_BINARY_DIR}/generated")
file(MAKE_DIRECTORY "${VFHF_GEN}")
set(VFHF_POINTS_LIST "")
set(VFHF_EXTERN_DECLARATIONS "")
set(VFHF_SOURCES_cu "")
set(VFHF_SOURCES_hip "")
foreach(NUM_OPS IN LISTS FKLB_VFHF_POINTS)
    if(NOT NUM_OPS MATCHES "^[0-9]+$" OR NUM_OPS LESS 2)
        message(FATAL_ERROR "FKLB_VFHF_POINTS: '${NUM_OPS}' is not a number of operations 2 + 100 (k - 1)")
    endif()
    math(EXPR VFHF_K_MINUS_1 "(${NUM_OPS} - 2) / 100")
    math(EXPR VFHF_EXPECTED "2 + 100 * ${VFHF_K_MINUS_1}")
    if(NOT NUM_OPS EQUAL VFHF_EXPECTED)
        message(FATAL_ERROR "FKLB_VFHF_POINTS: ${NUM_OPS} is not 2 + 100 (k - 1)")
    endif()
    foreach(BACKEND cu hip)
        configure_file("${CMAKE_CURRENT_SOURCE_DIR}/templates/vfhf_point.in" "${VFHF_GEN}/vfhf_point_${NUM_OPS}.${BACKEND}" @ONLY)
        list(APPEND VFHF_SOURCES_${BACKEND} "${VFHF_GEN}/vfhf_point_${NUM_OPS}.${BACKEND}")
    endforeach()
    if(VFHF_POINTS_LIST STREQUAL "")
        set(VFHF_POINTS_LIST "${NUM_OPS}")
    else()
        string(APPEND VFHF_POINTS_LIST ", ${NUM_OPS}")
    endif()
    string(APPEND VFHF_EXTERN_DECLARATIONS "extern template void vfhf::fused<${NUM_OPS}>(fk::Stream_<fk::defaultParArch>&, "
                                           "const std::array<fk::Ptr2D<uchar>, vfhf::BATCH>&, const fk::Tensor<float>&);\n")
endforeach()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/templates/vfhf_points.h.in" "${VFHF_GEN}/vfhf_points.h" @ONLY)

fklb_add_benchmark(benchmark_vfhf_max_speedup
    BACKENDS cu hip LIBS fkl
    LAUNCHER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/benchmark_vfhf_max_speedup.h")
foreach(BACKEND cu hip)
    if(TARGET benchmark_vfhf_max_speedup_${BACKEND})
        target_sources(benchmark_vfhf_max_speedup_${BACKEND} PRIVATE ${VFHF_SOURCES_${BACKEND}})
        target_include_directories(benchmark_vfhf_max_speedup_${BACKEND} PRIVATE "${VFHF_GEN}")
    endif()
endforeach()
