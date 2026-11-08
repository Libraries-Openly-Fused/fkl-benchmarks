#!/usr/bin/env bash
# vfhf_run_build_extra.sh <source worktree> <build dir>: the build of benchmark_vfhf_max_speedup_hip for the extra points 15002 and 17002: the flags of the RPP runs
# (rpp_run_build.sh) and of ../../step0/vfhf_run_build.sh, with FKLB_VFHF_POINTS="15002;17002" the only difference, every compile timed by time_tu.sh into <build dir>.tu.log
# (seconds, peak resident KB, exit code). Both paths absolute.
set -u
SRC=$1; B=$2
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
export TMPDIR=$NS/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
: > $B.tu.log
( ulimit -v 16000000
  cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_OPENCV_CPU=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=OFF "-DFKLB_VFHF_POINTS=15002;17002" \
    "-DCMAKE_HIP_COMPILER_LAUNCHER=$NS/scripts/time_tu.sh;$B.tu.log" "-DCMAKE_CXX_COMPILER_LAUNCHER=$NS/scripts/time_tu.sh;$B.tu.log" \
    > $B.configure.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 benchmark_vfhf_max_speedup_hip > $B.build.log 2>&1; echo "build exit $?" )
grep -cE "warning:" $B.build.log | sed 's/^/warnings: /'
