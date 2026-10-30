# fklb_add_benchmark(<name>
#     BACKENDS <cu|hip|cpp>...     required: the back-ends this benchmark supports. A back-end that is not listed is never
#                                  built, so nvcc-only code cannot reach the hip or cpp builds.
#     [LAUNCHER_HEADER <header>]   FKL style: a header that defines launch(); a launcher.<ext> that includes it is generated
#                                  from common/launcher.in for each back-end (so the same header is compiled by nvcc,
#                                  amdclang++ and the host compiler)
#     [SOURCES <files>...]         plain sources (used instead of LAUNCHER_HEADER)
#     [LIBS fkl|cvgs|fastnpp|npp|opencv...]
#     [INCLUDE_DIRS <dirs>...] [DEFINITIONS <defs>...])
# One executable <name>_<backend> per listed and available back-end, run by ctest from build/run/<target>/ (the CSV
# files are written to the working directory). Test label: benchmark.

set(FKLB_BACKENDS cu hip cpp)
set(FKLB_EXT_cu cu)
set(FKLB_EXT_hip hip)
set(FKLB_EXT_cpp cpp)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_library(fklb_common INTERFACE)
target_include_directories(fklb_common INTERFACE "${CMAKE_SOURCE_DIR}" "${CMAKE_BINARY_DIR}/generated")
target_compile_definitions(fklb_common INTERFACE ENABLE_BENCHMARK)

function(fklb_setup_cuda_target TGT)
    set_target_properties(${TGT} PROPERTIES CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON CUDA_EXTENSIONS OFF
                                            CUDA_ARCHITECTURES "${FKLB_CUDA_ARCH}")
    if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 13)
        set_target_properties(${TGT} PROPERTIES CUDA_RUNTIME_LIBRARY Hybrid)
    else()
        set_target_properties(${TGT} PROPERTIES CUDA_RUNTIME_LIBRARY Shared)
    endif()
    # The flags of the library builds
    target_compile_options(${TGT} PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--threads 0 -rdc=false -split-compile=0 -ftemplate-depth=${FKLB_TEMPLATE_DEPTH}>)
    if(FKLB_LINEINFO)
        target_compile_options(${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    endif()
    if(FKLB_FMAD_OFF)
        target_compile_options(${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-fmad=false>)
    endif()
    target_link_libraries(${TGT} PRIVATE CUDA::cudart)
endfunction()

function(fklb_setup_hip_target TGT)
    set_target_properties(${TGT} PROPERTIES HIP_STANDARD 17 HIP_STANDARD_REQUIRED ON HIP_EXTENSIONS OFF
                                            HIP_RUNTIME_LIBRARY Shared HIP_ARCHITECTURES "${CMAKE_HIP_ARCHITECTURES}")
    target_compile_options(${TGT} PRIVATE
        $<$<COMPILE_LANGUAGE:HIP>:-Werror=c++20-extensions -ftemplate-depth=${FKLB_TEMPLATE_DEPTH}>)
    target_link_libraries(${TGT} PRIVATE hip::host)
endfunction()

function(fklb_add_benchmark NAME)
    cmake_parse_arguments(ARG "" "LAUNCHER_HEADER" "BACKENDS;SOURCES;LIBS;INCLUDE_DIRS;DEFINITIONS" ${ARGN})
    if(NOT ARG_BACKENDS)
        message(FATAL_ERROR "fklb_add_benchmark(${NAME}): BACKENDS must list the supported back-ends (cu, hip, cpp)")
    endif()
    if(NOT ARG_LAUNCHER_HEADER AND NOT ARG_SOURCES)
        message(FATAL_ERROR "fklb_add_benchmark(${NAME}): give LAUNCHER_HEADER or SOURCES")
    endif()

    foreach(BACKEND IN LISTS ARG_BACKENDS)
        if(NOT BACKEND IN_LIST FKLB_BACKENDS)
            message(FATAL_ERROR "fklb_add_benchmark(${NAME}): unknown back-end '${BACKEND}' (cu, hip, cpp)")
        endif()
        if(NOT FKLB_HAVE_${BACKEND})
            message(STATUS "fkl-benchmarks: ${NAME}_${BACKEND} skipped (back-end not available)")
            continue()
        endif()

        set(TGT ${NAME}_${BACKEND})
        set(SRCS "${CMAKE_SOURCE_DIR}/common/main.cpp")
        if(ARG_LAUNCHER_HEADER)
            get_filename_component(HDR_DIR "${ARG_LAUNCHER_HEADER}" DIRECTORY)
            set(TARGET_NAME ${NAME})   # substituted into common/launcher.in
            set(LAUNCHER "${CMAKE_CURRENT_BINARY_DIR}/${TGT}/launcher.${FKLB_EXT_${BACKEND}}")
            configure_file("${CMAKE_SOURCE_DIR}/common/launcher.in" "${LAUNCHER}" @ONLY)
            list(APPEND SRCS "${LAUNCHER}" "${ARG_LAUNCHER_HEADER}")
        else()
            set(HDR_DIR "")
            list(APPEND SRCS ${ARG_SOURCES})
        endif()

        add_executable(${TGT} ${SRCS})
        add_dependencies(${TGT} fklb_build_info)
        set_target_properties(${TGT} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
        target_link_libraries(${TGT} PRIVATE fklb_common)
        target_include_directories(${TGT} PRIVATE ${HDR_DIR} ${ARG_INCLUDE_DIRS})
        target_compile_definitions(${TGT} PRIVATE ${ARG_DEFINITIONS})
        foreach(LIB IN LISTS ARG_LIBS)
            target_link_libraries(${TGT} PRIVATE fklb_${LIB})
        endforeach()
        if(NOT "${FKLB_CPU_ARCH}" STREQUAL "disabled")
            target_compile_options(${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-march=${FKLB_CPU_ARCH} -mtune=${FKLB_CPU_ARCH}>)
        endif()
        target_compile_options(${TGT} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-ftemplate-depth=${FKLB_TEMPLATE_DEPTH}>)
        if(BACKEND STREQUAL "cu")
            fklb_setup_cuda_target(${TGT})
        elseif(BACKEND STREQUAL "hip")
            fklb_setup_hip_target(${TGT})
        endif()

        set(RUNDIR "${CMAKE_BINARY_DIR}/run/${TGT}")
        file(MAKE_DIRECTORY "${RUNDIR}")
        add_test(NAME ${TGT} COMMAND ${TGT} WORKING_DIRECTORY "${RUNDIR}")
        set_tests_properties(${TGT} PROPERTIES LABELS benchmark)
    endforeach()
endfunction()
