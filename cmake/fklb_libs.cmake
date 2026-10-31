# Interface targets for the pinned libraries (all header only) and the system OpenCV.
# The libraries are consumed read-only from external/, never through their own CMake.

add_library(fklb_fkl INTERFACE)
target_include_directories(fklb_fkl INTERFACE "${CMAKE_SOURCE_DIR}/external/fkl/include")

# FKL with the multithreaded CPU back-end (ParArch::CPU_MULTITHREAD): the thread-pool submodule of external/fkl, the define
# of the library and the threads library. Targets ask for it with LIBS fkl_mt; a benchmark that does not, and every cu and
# hip binary built without it, are the same as before. Without the submodule (git submodule update --init --recursive)
# fkl_mt is fkl, and the benchmarks compile without ParArch::CPU_MULTITHREAD (they test FK_CPU_MULTITHREAD_ENABLED).
add_library(fklb_fkl_mt INTERFACE)
target_link_libraries(fklb_fkl_mt INTERFACE fklb_fkl)
if(EXISTS "${CMAKE_SOURCE_DIR}/external/fkl/thread-pool/include/ThreadPool.h")
    find_package(Threads REQUIRED)
    target_include_directories(fklb_fkl_mt INTERFACE "${CMAKE_SOURCE_DIR}/external/fkl/thread-pool/include")
    target_compile_definitions(fklb_fkl_mt INTERFACE FK_CPU_MULTITHREAD_ENABLED)
    target_link_libraries(fklb_fkl_mt INTERFACE Threads::Threads)
    set(FKLB_HAVE_CPU_MULTITHREAD ON)
else()
    message(STATUS "fkl-benchmarks: external/fkl/thread-pool is not checked out, ParArch::CPU_MULTITHREAD is not built")
    set(FKLB_HAVE_CPU_MULTITHREAD OFF)
endif()

# cvGPUSpeedup: its include/ and its root (the benchmarks use tests/testUtils.cuh)
add_library(fklb_cvgs INTERFACE)
target_include_directories(fklb_cvgs INTERFACE
    "${CMAKE_SOURCE_DIR}/external/cvGPUSpeedup/include" "${CMAKE_SOURCE_DIR}/external/cvGPUSpeedup")
target_link_libraries(fklb_cvgs INTERFACE fklb_fkl)

# FastNPP: its include/ and its root (the benchmarks use include/fast_npp.h and tests/testsNppCommon.h)
add_library(fklb_fastnpp INTERFACE)
target_include_directories(fklb_fastnpp INTERFACE
    "${CMAKE_SOURCE_DIR}/external/FastNPP/include" "${CMAKE_SOURCE_DIR}/external/FastNPP")
target_link_libraries(fklb_fastnpp INTERFACE fklb_fkl)

if(FKLB_HAVE_cu)
    add_library(fklb_npp INTERFACE)
    target_link_libraries(fklb_npp INTERFACE CUDA::nppc CUDA::nppial CUDA::nppidei CUDA::nppig)
endif()

# OpenCV (system build). opencv-cuda/ needs the CUDA modules; opencv-cpu/ only core and imgproc (imgcodecs when it is built).
if(FKLB_BUILD_OPENCV_CPU AND NOT (FKLB_BUILD_OPENCV_CUDA AND FKLB_HAVE_cu))
    if(OPENCV_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${OPENCV_DIR}")
    endif()
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc OPTIONAL_COMPONENTS imgcodecs)
    message(STATUS "fkl-benchmarks: OpenCV ${OpenCV_VERSION} (${OpenCV_DIR}), CPU modules only")
endif()
if(FKLB_BUILD_OPENCV_CPU OR (FKLB_BUILD_OPENCV_CUDA AND FKLB_HAVE_cu))
    add_library(fklb_opencv_cpu INTERFACE)
endif()

if(FKLB_BUILD_OPENCV_CUDA AND FKLB_HAVE_cu)
    # OpenCVConfig.cmake calls the legacy FindCUDA module: keep it (CMP0146 OLD) and point it at the pinned toolkit, so
    # that OpenCV's CUDA dependencies resolve to the same CUDA as nvcc and never to another one on this machine.
    cmake_policy(SET CMP0146 OLD)
    set(CUDA_TOOLKIT_ROOT_DIR "${CUDAToolkit_LIBRARY_ROOT}")
    if(OPENCV_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${OPENCV_DIR}")
    endif()
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core cudaarithm imgproc cudafilters cudaimgproc cudawarping imgcodecs)
    message(STATUS "fkl-benchmarks: OpenCV ${OpenCV_VERSION} (${OpenCV_DIR})")
    add_library(fklb_opencv INTERFACE)
    target_link_libraries(fklb_opencv INTERFACE
        opencv_core opencv_cudaarithm opencv_imgproc opencv_cudafilters opencv_cudaimgproc opencv_cudawarping opencv_imgcodecs)
endif()

if(TARGET fklb_opencv_cpu)
    target_link_libraries(fklb_opencv_cpu INTERFACE opencv_core opencv_imgproc)
    if(TARGET opencv_imgcodecs)
        target_link_libraries(fklb_opencv_cpu INTERFACE opencv_imgcodecs)
    endif()
endif()

# RPP, the Radeon Performance Primitives of ROCm (the tensor API, rppt_*, with the HIP back-end), for rpp/
if(FKLB_BUILD_RPP AND FKLB_HAVE_hip)
    find_package(rpp CONFIG REQUIRED)
    get_target_property(FKLB_RPP_INCLUDE rpp::rpp INTERFACE_INCLUDE_DIRECTORIES)
    add_library(fklb_rpp INTERFACE)
    target_link_libraries(fklb_rpp INTERFACE fklb_fkl rpp::rpp)
    # the headers include each other as <rppdefs.h>: <dir>/rpp for each include directory of the package
    if(NOT FKLB_RPP_INCLUDE)
        message(FATAL_ERROR "fkl-benchmarks: rpp::rpp has no INTERFACE_INCLUDE_DIRECTORIES")
    endif()
    foreach(FKLB_RPP_DIR IN LISTS FKLB_RPP_INCLUDE)
        target_include_directories(fklb_rpp INTERFACE "${FKLB_RPP_DIR}/rpp")
    endforeach()
    message(STATUS "fkl-benchmarks: RPP ${rpp_VERSION} (${rpp_DIR})")
endif()
