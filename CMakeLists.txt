cmake_minimum_required(VERSION 3.28 FATAL_ERROR)

project(fkl-benchmarks VERSION 0.1.0 LANGUAGES CXX
        DESCRIPTION "Benchmarks of the Fused Kernel Library and its wrappers (TPDS paper, revision 2)")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(fklb_options)
include(fklb_backends)
include(fklb_libs)
include(fklb_build_info)
include(fklb_targets)

enable_testing()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(FKLB_BUILD_FKL_SELF)
    add_subdirectory(fkl-self)
endif()
if(FKLB_BUILD_OPENCV_CUDA)
    add_subdirectory(opencv-cuda)
endif()
if(FKLB_BUILD_NPP)
    add_subdirectory(npp)
endif()
if(FKLB_BUILD_OPENCV_CPU)
    add_subdirectory(opencv-cpu)
endif()
if(FKLB_BUILD_HANDWRITTEN)
    add_subdirectory(handwritten-cuda)
endif()
