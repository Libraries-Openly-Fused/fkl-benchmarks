# Interface targets for the pinned libraries (all header only) and the system OpenCV.
# The libraries are consumed read-only from external/, never through their own CMake.

add_library(fklb_fkl INTERFACE)
target_include_directories(fklb_fkl INTERFACE "${CMAKE_SOURCE_DIR}/external/fkl/include")

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

# OpenCV (system build). opencv-cuda/ needs the CUDA modules; opencv-cpu/ only core, imgproc and imgcodecs.
if(FKLB_BUILD_OPENCV_CPU AND NOT (FKLB_BUILD_OPENCV_CUDA AND FKLB_HAVE_cu))
    if(OPENCV_DIR)
        list(APPEND CMAKE_PREFIX_PATH "${OPENCV_DIR}")
    endif()
    find_package(OpenCV CONFIG REQUIRED COMPONENTS core imgproc imgcodecs)
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
    target_link_libraries(fklb_opencv_cpu INTERFACE opencv_core opencv_imgproc opencv_imgcodecs)
endif()
