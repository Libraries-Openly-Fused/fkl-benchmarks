#!/usr/bin/env bash
# ocv_build.sh <stage> <fkl checkout> <commit>: the build check of opencv-cpu/ (FKLB_BUILD_OPENCV_CPU, the E5 CPU programs) of fkl-benchmarks
# <commit>, from a git archive as pin_build.sh, with external/fkl -> <fkl checkout> (its thread-pool submodule checked out, which the two
# targets need) and the OpenCV 4.14.0 CPU build of this machine. Only the cpp back-end and only opencv-cpu/. Compile and link only.
set -u
STAGE=$1; FKLDIR=$2; C=$3
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; REPO=/home/cudeiro/rocm-fkl-benchmarks/fkl-benchmarks; FR=$NS/fastrpp/FastRPP-efbf009
OCV=/home/cudeiro/bvf-paper-bench-deps/opencv-4.14.0-cpu-linux-x86_64-g++-13
SRC=$NS/vsrc/$STAGE; B=$NS/vbuild/$STAGE-cpp; OUT=$NS/vout/$STAGE
export TMPDIR=$NS/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
mkdir -p $OUT; rm -rf $SRC $B; mkdir -p $SRC/external/cvGPUSpeedup $SRC/external/FastNPP
(cd $REPO && git --no-optional-locks archive $C -- CMakeLists.txt CMakePresets.json cmake common fkl-self rpp opencv-cpu opencv-cuda npp handwritten-cuda scripts) | tar -x -C $SRC
ln -s $FKLDIR $SRC/external/fkl; ln -s $FR $SRC/external/FastRPP
echo "benchmarks $C, FKL $(cd $FKLDIR && git rev-parse HEAD) dirty=$(cd $FKLDIR && git status --porcelain | wc -l), thread-pool $(cd $FKLDIR/thread-pool && git rev-parse HEAD) dirty=$(cd $FKLDIR/thread-pool && git status --porcelain | wc -l), OpenCV $OCV" | tee $OUT/ref.txt
( ulimit -v 16000000
  cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DFKLB_ENABLE_CUDA=OFF -DFKLB_ENABLE_HIP=OFF -DFKLB_BUILD_FKL_SELF=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_RPP=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_OPENCV_CPU=ON -DOPENCV_DIR=$OCV > $OUT/configure-cpp.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 -k 0 > $OUT/build-cpp.log 2>&1; echo "build exit $?" )
echo "$(ls $B/bin 2>/dev/null | wc -l) binaries: $(ls $B/bin 2>/dev/null | tr '\n' ' '); FAILED $(grep -c FAILED $OUT/build-cpp.log); warnings $(grep -c 'warning:' $OUT/build-cpp.log)"
