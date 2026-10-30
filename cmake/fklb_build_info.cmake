# What was built, with what, from which commits, written to build/generated/fklb_build_info.h on EVERY build (a custom
# target), not only at configure time: a commit made after the configure must show in the header of the next CSV.
# common/system_info.h writes this header, plus the run-time GPU, driver and clocks, at the top of every CSV.
if(FKLB_HAVE_cu)
    set(FKLB_NVCC_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")
    set(FKLB_NVCC_PATH "${CMAKE_CUDA_COMPILER}")
else()
    set(FKLB_NVCC_VERSION "none")
    set(FKLB_NVCC_PATH "none")
endif()
set(FKLB_OPENCV_VERSION_STR "none")
if(OpenCV_VERSION)
    set(FKLB_OPENCV_VERSION_STR "${OpenCV_VERSION}")
endif()
set(FKLB_FMAD_STR "on")
if(FKLB_FMAD_OFF)
    set(FKLB_FMAD_STR "off")
endif()
set(FKLB_LINEINFO_STR "off")
if(FKLB_LINEINFO)
    set(FKLB_LINEINFO_STR "on")
endif()

file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/generated")
file(WRITE "${CMAKE_BINARY_DIR}/generated/fklb_build_info_vars.cmake"
"set(FKLB_NVCC_VERSION \"${FKLB_NVCC_VERSION}\")
set(FKLB_NVCC_PATH \"${FKLB_NVCC_PATH}\")
set(FKLB_CUDA_ARCH \"${FKLB_CUDA_ARCH}\")
set(FKLB_CXX_COMPILER_STR \"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\")
set(FKLB_BUILD_TYPE \"${CMAKE_BUILD_TYPE}\")
set(FKLB_CPU_ARCH \"${FKLB_CPU_ARCH}\")
set(FKLB_FMAD_STR \"${FKLB_FMAD_STR}\")
set(FKLB_LINEINFO_STR \"${FKLB_LINEINFO_STR}\")
set(FKLB_TEMPLATE_DEPTH \"${FKLB_TEMPLATE_DEPTH}\")
set(FKLB_OPENCV_VERSION_STR \"${FKLB_OPENCV_VERSION_STR}\")
")

add_custom_target(fklb_build_info ALL
    COMMAND "${CMAKE_COMMAND}" "-DFKLB_SRC=${CMAKE_SOURCE_DIR}" "-DFKLB_BIN=${CMAKE_BINARY_DIR}"
            -P "${CMAKE_SOURCE_DIR}/cmake/fklb_write_build_info.cmake"
    BYPRODUCTS "${CMAKE_BINARY_DIR}/generated/fklb_build_info.h"
    COMMENT "Writing build/generated/fklb_build_info.h"
    VERBATIM)
