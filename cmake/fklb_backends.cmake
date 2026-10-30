# Back-ends: cu (CUDA, nvcc), hip (HIP, amdclang++, AMD GPUs only) and cpp (CPU, the C++ compiler).
# FKLB_HAVE_cu / FKLB_HAVE_hip / FKLB_HAVE_cpp tell fklb_add_benchmark() which back-ends can be built here.
include(CheckLanguage)

set(FKLB_HAVE_cpp ${FKLB_ENABLE_CPU})
set(FKLB_HAVE_cu OFF)
set(FKLB_HAVE_hip OFF)

if(FKLB_ENABLE_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "${FKLB_CUDA_ARCH}")
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        if(NOT "${FKLB_REQUIRED_NVCC}" STREQUAL "")
            string(REPLACE "." "\\." FKLB_REQUIRED_NVCC_RE "${FKLB_REQUIRED_NVCC}")
            if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^${FKLB_REQUIRED_NVCC_RE}\\.")
                message(FATAL_ERROR
                    "nvcc ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}) is not CUDA ${FKLB_REQUIRED_NVCC}. "
                    "Use a preset (it pins the 13.4.3 nvcc) or -DCMAKE_CUDA_COMPILER=<path>/nvcc, "
                    "and source toolchains/env-cuda13.4.sh. -DFKLB_REQUIRED_NVCC= disables this check.")
            endif()
        endif()
        find_package(CUDAToolkit REQUIRED)
        set(FKLB_HAVE_cu ON)
        message(STATUS "fkl-benchmarks: CUDA ${CMAKE_CUDA_COMPILER_VERSION} (${CMAKE_CUDA_COMPILER}), architectures ${FKLB_CUDA_ARCH}")
    else()
        message(FATAL_ERROR "No CUDA compiler found. Source toolchains/env-cuda13.4.sh and use a preset (they pin the "
                            "CUDA 13.4 nvcc through FKLB_CUDA_HOME), or configure with -DFKLB_ENABLE_CUDA=OFF for a "
                            "build without the cu back-end.")
    endif()
endif()

if(FKLB_ENABLE_HIP)
    # Only the amd HIP platform: HIP on NVIDIA GPUs is not an FKL back-end (CUDA is).
    if(NOT DEFINED CMAKE_HIP_PLATFORM)
        set(CMAKE_HIP_PLATFORM "amd")
    endif()
    check_language(HIP)
    if(CMAKE_HIP_COMPILER AND "${CMAKE_HIP_PLATFORM}" STREQUAL "amd")
        enable_language(HIP)
        foreach(ROCM_DIR "${CMAKE_HIP_COMPILER_ROCM_ROOT}" "$ENV{HIP_PATH}" "$ENV{ROCM_PATH}")
            if(NOT "${ROCM_DIR}" STREQUAL "")
                list(APPEND CMAKE_PREFIX_PATH "${ROCM_DIR}")
            endif()
        endforeach()
        find_package(hip CONFIG REQUIRED)
        set(FKLB_HAVE_hip ON)
        message(STATUS "fkl-benchmarks: HIP ${CMAKE_HIP_COMPILER_VERSION}, architectures ${CMAKE_HIP_ARCHITECTURES}")
    else()
        message(STATUS "fkl-benchmarks: no HIP compiler for AMD GPUs found, the hip back-end is disabled")
    endif()
endif()
