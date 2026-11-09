#!/usr/bin/env bash
# run_build.sh <commit>: the build of benchmark_rpp_preprocessing_hip for the RX 9060 timing runs, from a clean source tree of fkl-benchmarks
# <commit> (git archive), with external/fkl -> the clean FKL checkout at 1da77899 and external/FastRPP -> a clean FastRPP checkout at efbf009;
# the flags of the previous runs (results/rx9060/rpp-preprocessing-built-once/step0/rpp_run_build.sh). Compile and link only.
set -u
C=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; REPO=/home/cudeiro/rocm-fkl-benchmarks/fkl-benchmarks
SRC=$NS/vsrc/fastrpp-run; B=$NS/vbuild/fastrpp-run-hip; FR=$NS/fastrpp/FastRPP-efbf009; FKL=$NS/fkl-1da7789
export TMPDIR=$NS/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
# FastRPP: a clean detached checkout of efbf009 (only include/ is used), from the worktree's submodule clone
if [ ! -d $FR ]; then git clone -q --no-checkout $NS/wt-fastrpp/external/FastRPP $FR && (cd $FR && git checkout -q --detach efbf00912e0a0c35cc3b0bd633948696b07385ce); fi
[ "$(cd $FR && git rev-parse HEAD)" = efbf00912e0a0c35cc3b0bd633948696b07385ce ] && [ -z "$(cd $FR && git status --porcelain)" ] || { echo "FastRPP checkout not clean at efbf009"; exit 1; }
[ "$(cd $FKL && git rev-parse HEAD)" = 1da778992756e0fa08ec289225fe18dca134412c ] && [ -z "$(cd $FKL && git status --porcelain)" ] || { echo "FKL checkout not clean at 1da77899"; exit 1; }
[ "$(cd $REPO && git ls-tree $C external/FastRPP | awk '{print $3}')" = efbf00912e0a0c35cc3b0bd633948696b07385ce ] || { echo "commit $C does not pin FastRPP efbf009"; exit 1; }
rm -rf $SRC; mkdir -p $SRC/external/cvGPUSpeedup $SRC/external/FastNPP
(cd $REPO && git --no-optional-locks archive $C -- CMakeLists.txt CMakePresets.json cmake common fkl-self rpp opencv-cpu opencv-cuda npp handwritten-cuda scripts) | tar -x -C $SRC
ln -s $FKL $SRC/external/fkl; ln -s $FR $SRC/external/FastRPP
echo "sources: fkl-benchmarks $(cd $REPO && git rev-parse $C) (git archive); FKL $(cd $FKL && git rev-parse HEAD) (clean); FastRPP $(cd $FR && git rev-parse HEAD) (clean)" | tee $NS/fastrpp/run-build-ref.txt
rm -rf $B
( ulimit -v 16000000
  cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_OPENCV_CPU=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON > $B.configure.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 benchmark_rpp_preprocessing_hip > $B.build.log 2>&1; echo "build exit $?" )
echo "warnings: $(grep -cE 'warning:' $B.build.log); errors: $(grep -cE 'error:' $B.build.log)"
sha256sum $B/bin/benchmark_rpp_preprocessing_hip | tee -a $NS/fastrpp/run-build-ref.txt
