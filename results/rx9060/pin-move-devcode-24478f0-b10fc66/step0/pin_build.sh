#!/usr/bin/env bash
# pin_build.sh <stage> <fkl checkout> [commit]: the HIP build of fkl-benchmarks <commit> (default 8c1fa98) as bench_vs_fkl.sh's hip part, from a git
# archive, with external/fkl -> <fkl checkout> and external/FastRPP -> the clean FastRPP checkout at efbf009. Compile and link only.
set -u
STAGE=$1; FKLDIR=$2; C=${3:-8c1fa98311268277ec6e81d7b91439a031239bbe}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; REPO=/home/cudeiro/rocm-fkl-benchmarks/fkl-benchmarks; FR=$NS/fastrpp/FastRPP-efbf009
SRC=$NS/vsrc/$STAGE; B=$NS/vbuild/$STAGE-hip; OUT=$NS/vout/$STAGE
export TMPDIR=$NS/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
mkdir -p $OUT; rm -rf $SRC $B; mkdir -p $SRC/external/cvGPUSpeedup $SRC/external/FastNPP
(cd $REPO && git --no-optional-locks archive $C -- CMakeLists.txt CMakePresets.json cmake common fkl-self rpp opencv-cpu opencv-cuda npp handwritten-cuda scripts) | tar -x -C $SRC
ln -s $FKLDIR $SRC/external/fkl; ln -s $FR $SRC/external/FastRPP
echo "benchmarks $C, FKL $(cd $FKLDIR && git rev-parse HEAD) dirty=$(cd $FKLDIR && git status --porcelain | wc -l), FastRPP $(cd $FR && git rev-parse HEAD)" | tee $OUT/ref.txt
( ulimit -v 16000000
  cmake -S $SRC -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_OPENCV_CPU=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON > $OUT/configure-hip.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 -k 0 > $OUT/build-hip.log 2>&1; echo "build exit $?" )
echo "$(ls $B/bin | wc -l) binaries; FAILED $(grep -c FAILED $OUT/build-hip.log); warnings $(grep -c 'warning:' $OUT/build-hip.log)"
