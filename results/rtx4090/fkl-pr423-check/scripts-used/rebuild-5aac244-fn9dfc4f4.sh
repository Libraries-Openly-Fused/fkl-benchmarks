#!/usr/bin/env bash
# Group 5aac244 of the pin transfer at the head of FKL PR #423 with FastNPP at 9dfc4f4 (the FastNPP pin of the c8fcbe8 transfer table): pin_transfer.sh links the CURRENT FastNPP pin (a1b5e24, resize
# convention changed by its #45), whose kernels differ from the frozen binaries by that change alone. Compile only. A local clone of FastNPP at 9dfc4f4 (fkl/ linked to the tree under test, as the check farm
# does) in a new worktree of fkl-benchmarks at 5aac244; the pinned external/FastNPP is not touched.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh; export CUDA_VISIBLE_DEVICES=
ROOT=/home/oscar/tfkl-bench/fkl-benchmarks; S=/home/oscar/tfkl-bench/scratch/pr423; CORES=${PIN_CORES:-4-7,20-23}; OUT=$S/pin-transfer-out
echo "nvcc: $(nvcc --version | sed -n 4p)  start $(date -Iseconds)"
[ -d $S/fastnpp-9dfc4f4 ] || { git clone -q --local --no-checkout $ROOT/external/FastNPP $S/fastnpp-9dfc4f4 && git -C $S/fastnpp-9dfc4f4 checkout -q --detach 9dfc4f4 && rmdir $S/fastnpp-9dfc4f4/fkl && ln -s $S/check-fast/fkl-overlay $S/fastnpp-9dfc4f4/fkl; } || exit 1
echo "FastNPP: $(git -C $S/fastnpp-9dfc4f4 log --oneline -1)"
W=$S/pin-transfer-work/src-5aac244-fn9dfc4f4
[ -d $W ] || { git -C $ROOT worktree add -q --detach $W 5aac244 || exit 1
  rm -rf $W/external/fkl $W/external/cvGPUSpeedup $W/external/FastNPP
  ln -s $S/check-fast/fkl-overlay $W/external/fkl; ln -s $ROOT/external/cvGPUSpeedup $W/external/cvGPUSpeedup; ln -s $S/fastnpp-9dfc4f4 $W/external/FastNPP
  cp $ROOT/CMakeUserPresets.json $W/; }
taskset -c $CORES cmake --preset local-rtx4090 -S $W -B $W/build > $OUT/configure-5aac244-fn9dfc4f4.log 2>&1 || { echo "configure failed"; exit 2; }
taskset -c $CORES nice -n 5 cmake --build $W/build --target benchmark_CPU_NPP_vs_FastNPP_cu benchmark_npp_batchresize_x_split3D_cu -j 12 > $OUT/build-5aac244-fn9dfc4f4.log 2>&1 || { echo "build failed: $(grep -m2 error $OUT/build-5aac244-fn9dfc4f4.log | cut -c1-200)"; exit 3; }
IGN=',\(fk::ResizeSampling\)0'
echo "== 5aac244 + FastNPP 9dfc4f4 + FKL head, compared with the FROZEN binaries (e0-frozen-5aac244)"
python3 $ROOT/scripts/sass_compare_builds.py $ROOT/build/e0-frozen-5aac244/bin $W/build/bin --glob "*_cu" --out $OUT/sass_5aac244_fn9dfc4f4.txt | grep "^kernels identical"
python3 $ROOT/scripts/sass_compare_builds.py $ROOT/build/e0-frozen-5aac244/bin $W/build/bin --glob "*_cu" --match-demangled --ignore "$IGN" --out $OUT/sass_5aac244_fn9dfc4f4_demangled.txt | grep "^kernels identical"
echo "== the same build compared with the c8fcbe8 build of the group (FastNPP 9dfc4f4, /home/oscar/tfkl-bench/pin-transfer-work/src-5aac244/build)"
python3 $ROOT/scripts/sass_compare_builds.py /home/oscar/tfkl-bench/pin-transfer-work/src-5aac244/build/bin $W/build/bin --glob "*_cu" --out $OUT/sass_5aac244_fn9dfc4f4_vs_c8fcbe8.txt | grep "^kernels identical"
echo "done $(date -Iseconds)"
