#!/usr/bin/env bash
# rpp_run_build.sh <source worktree> <build dir>: the build of benchmark_rpp_preprocessing_hip for the RX 9060 runs (the flags of bench_vs_fkl.sh's hip part)
set -u
SRC=$1; B=$2
export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
( ulimit -v 16000000
  cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_OPENCV_CPU=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON > $B.configure.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 benchmark_rpp_preprocessing_hip > $B.build.log 2>&1; echo "build exit $?" )
grep -cE "warning:" $B.build.log | sed 's/^/warnings: /'
