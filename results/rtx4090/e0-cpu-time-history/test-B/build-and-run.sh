#!/bin/bash
# TEST B of results/rtx4090/e0-cpu-time-history (2026-11-06): the host-only program hostbench.cu built with six host flag sets, run pinned to cores 0-3,16-19 with CUDA_VISIBLE_DEVICES empty (a CUDA call would fail).
# Run from a scratch folder that holds hostbench.cu; the pins are those of this repository (cvGS 6d26c65, FastNPP a1b5e245, FKL c8fcbe8). The flag sets: -O3/-O2/-O1/-O0 with -DNDEBUG, -O3 without NDEBUG, and the flags a CMake build
# with an EMPTY CMAKE_BUILD_TYPE gives (none: nvcc passes no -O and no -DNDEBUG to g++ for the host code; checked from the nvcc --dryrun of a CMake build of cvGPUSpeedup with an empty build type).
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
R=/home/oscar/tfkl-bench/fkl-benchmarks; O=/home/oscar/tfkl-bench/deps/opencv-4.14-cuda13.4/install
INC="-I$R/external/fkl/include -I$R/external/cvGPUSpeedup/include -I$R/external/cvGPUSpeedup -I$R/external/FastNPP/include -I$O/include/opencv4"
LIBS="-L$O/lib -Xlinker -rpath=$O/lib -lopencv_core -lopencv_cudaarithm -lopencv_imgproc -lopencv_cudaimgproc -lopencv_cudawarping -lopencv_cudafilters"
for v in "o3ndebug:-O3 -DNDEBUG" "o2ndebug:-O2 -DNDEBUG" "o1ndebug:-O1 -DNDEBUG" "o0ndebug:-O0 -DNDEBUG" "o3debugassert:-O3" "emptybuildtype:"; do
  n=${v%%:*}; f=${v#*:}
  (taskset -c 8-15,24-31 nvcc -std=c++17 -arch=sm_89 -x cu $f $INC hostbench.cu -o hostbench_$n $LIBS > build_$n.log 2>&1) &
done; wait
echo "# load before: $(cut -d' ' -f1-3 /proc/loadavg) at $(date -Iseconds)"; echo "flagset,round,batch,cvgs_read_us,cvgs_all_us,fastnpp_read_us,fastnpp_all_us"
for r in 1 2 3; do for n in o3ndebug o2ndebug o1ndebug o0ndebug o3debugassert emptybuildtype; do CUDA_VISIBLE_DEVICES= taskset -c 0-3,16-19 ./hostbench_$n $n | sed "s/^$n,/$n,$r,/"; done; done
echo "# load after: $(cut -d' ' -f1-3 /proc/loadavg) at $(date -Iseconds)"
