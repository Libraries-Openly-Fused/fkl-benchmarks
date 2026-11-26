#!/usr/bin/env bash
# rt_build.sh: B of the RX 9060 re-times RT-A1 (E19 crop sweep) and RT-A2 (E18), the HIP build of fkl-benchmarks main 73a9650 (FKL 304492a,
# the merge of FKL #435) from a clean git worktree, so that the binaries record 73a9650 and 304492a without -dirty, as the E19 crop sweep's
# binary recorded c8eb575 (results/rx9060/e19-crop-sweep/step0/). The worktree's external/fkl (304492a) and external/FastRPP (efbf009) are
# checked out from the local clones; cvGPUSpeedup, FastNPP and FKL's thread-pool are not checked out, as in that build. The CMake options are
# pin_build.sh's (results/rx9060/pin-move-devcode-b10fc66-48fbf3f/step0/), the build of A. Builds e19_crop_sweep_hip and e18_box_hip only.
# Compile and link only.
set -u
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; WT=$NS/wt-rt-73a9650; B=$NS/vbuild/rt-73a9650-hip; OUT=$NS/rt/build-b
export TMPDIR=$NS/sys ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1
mkdir -p $OUT; rm -rf $B
{ echo "fkl-benchmarks $(cd $WT && git rev-parse HEAD); git status --porcelain: $(cd $WT && git status --porcelain | wc -l) lines"
  echo "external/fkl $(cd $WT/external/fkl && git rev-parse HEAD) dirty=$(cd $WT/external/fkl && git status --porcelain | wc -l)"
  echo "external/FastRPP $(cd $WT/external/FastRPP && git rev-parse HEAD) dirty=$(cd $WT/external/FastRPP && git status --porcelain | wc -l)"; } | tee $OUT/source-state.txt
( ulimit -v 16000000
  cmake -S $WT -B $B -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=$ROCM_PATH/bin/amdclang++ -DCMAKE_HIP_COMPILER=$ROCM_PATH/bin/amdclang++ \
    -DCMAKE_HIP_ARCHITECTURES=gfx1200 -DFKLB_ENABLE_CUDA=OFF -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF -DFKLB_BUILD_OPENCV_CPU=OFF \
    -DFKLB_BUILD_HANDWRITTEN=OFF -DFKLB_BUILD_RPP=ON > $OUT/configure-hip.log 2>&1; echo "configure exit $?"
  nice -n 10 ninja -C $B -j2 -k 0 e19_crop_sweep_hip e18_box_hip > $OUT/build-hip.log 2>&1; echo "build exit $?" )
echo "FAILED $(grep -c FAILED $OUT/build-hip.log); warnings $(grep -c 'warning:' $OUT/build-hip.log)"
(cd $B && sha256sum bin/e19_crop_sweep_hip bin/e18_box_hip) | tee $OUT/binary-sha256.txt
