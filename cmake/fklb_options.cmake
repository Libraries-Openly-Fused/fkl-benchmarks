# Options of the fkl-benchmarks build. The defaults reproduce the build configuration of the paper (the library CMake
# defaults: C++17, native CPU architecture, -lineinfo, template depth 1000, nvcc default -fmad=true).

set(FKLB_CUDA_ARCH "native" CACHE STRING
    "CUDA architectures, CMAKE_CUDA_ARCHITECTURES syntax (e.g. 89 for the RTX 4090, 120 for the RTX 5050, native)")
set(FKLB_CPU_ARCH "native" CACHE STRING "-march/-mtune of the CPU code (native, haswell... or disabled)")
set(FKLB_TEMPLATE_DEPTH "1000" CACHE STRING "-ftemplate-depth of every compiler (default as in the library builds)")
option(FKLB_FMAD_OFF "Compile the CUDA code with -fmad=false (experiment E6: no FMA contraction)" OFF)
option(FKLB_LINEINFO "Compile the CUDA code with -lineinfo (the library default; it does not change the SASS)" ON)
set(FKLB_REQUIRED_NVCC "13.4" CACHE STRING
    "Required nvcc major.minor. The configure fails with any other toolkit (empty disables the check)")

option(FKLB_ENABLE_CPU  "Build the cpp back-end targets" ON)
option(FKLB_ENABLE_CUDA "Build the cu back-end targets (needs nvcc)" ON)
option(FKLB_ENABLE_HIP  "Build the hip back-end targets (needs amdclang++)" ON)

option(FKLB_BUILD_FKL_SELF    "Build fkl-self/ (FKL against itself)" ON)
option(FKLB_BUILD_OPENCV_CUDA "Build opencv-cuda/ (needs OpenCV with CUDA, -DOPENCV_DIR=<install prefix>)" ON)
option(FKLB_BUILD_NPP         "Build npp/ (FastNPP against NPP)" ON)
option(FKLB_BUILD_RPP         "Build rpp/ (FKL against AMD RPP on the HIP back-end; needs the rpp package of ROCm and an AMD GPU)" OFF)
option(FKLB_BUILD_HANDWRITTEN "Build handwritten-cuda/ (E1: hand-written fused kernels against FKL; CUDA only)" ON)
option(FKLB_BUILD_OPENCV_CPU  "Build opencv-cpu/ (FKL CPU back-end against OpenCV on the CPU; any OpenCV 4.x with core, imgproc, imgcodecs)" OFF)

set(OPENCV_DIR "" CACHE PATH "Install prefix of the system OpenCV build (4.14, with CUDA modules)")
