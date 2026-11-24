#!/bin/bash
# Task B, step 2 (COMPILE ONLY; no GPU is used: every nvcc gets an explicit sm_89, nothing is run except the compile-fail ctest tests,
# which only build): FKL PR #433 (head 4b11bc6) against its base 24478f0, for CUDA sm_89.
#   A. FKL's own test matrix (tests/ and utests/), every target that compiles something, at 24478f0 (before) and 4b11bc6 (after): FKL's CMake
#      out of source, CUDA_ARCH=89 (FKL sets CMAKE_CUDA_ARCHITECTURES OFF and gives every target CUDA_ARCH, whose default is native), Release,
#      nvcc 13.4.3. The targets come from CMake's code model, as in scripts/check_fkl_tree.sh; the expected rejections (ctest <target>_rejected)
#      are judged by their ctest tests. Per-target report with scripts/check_fkl_tree_report.py.
#   B. The five fkl-benchmarks CUDA programs that use the DPPs (e18_box, e18_box_split, e18_stencil, e19_batched_box, e19_crop_sweep) at main
#      b4fb594: before = worktree fkl-benchmarks-task-b (branch task-b-sm89-check, external/fkl 24478f0, unchanged), after = worktree
#      fkl-benchmarks-task-b-after (detached at b4fb594, external/fkl 4b11bc6, the details port applied: details-port.diff). cmake --preset
#      local-rtx4090 (sm_89) in each worktree's own build/local-rtx4090.
#   C. AMD's failure case, one target: utest_flash_attention_mma_cu at 4b11bc6 with CUDA_ARCH=native and no GPU visible (CUDA_VISIBLE_DEVICES
#      empty), as on a machine without a GPU: what native becomes there, and whether the target builds.
# Compilers pinned to cores 8-15,24-31, nice 10, TMPDIR on disk. Every step is logged; a failed build is recorded, not fatal (the report says
# what failed), except a configure failure of A or B.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
# no GPU is visible to any process of this step (CMake's compiler checks included); A and B name sm_89 explicitly and need none
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/task-b; WB=$T/worktrees/fkl-benchmarks-task-b; WA=$T/worktrees/fkl-benchmarks-task-b-after
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
CORES=8-15,24-31; JOBS=12
REPORT_PY=$WB/scripts/check_fkl_tree_report.py
LOG=$S/step2-console.txt
exec > >(tee -a $LOG) 2>&1
echo "# $(date -Iseconds) task B step 2 (compile only) starts; load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "nvcc: $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//'); cmake: $(cmake --version | head -1); c++: $(c++ --version | head -1)"
# the pins, checked before anything is built
[ "$(git -C $T/worktrees/fkl-pr433-base-24478f0 rev-parse HEAD)" = 24478f0aa11dc3f12feea83178f7fde828322fb0 ] || { echo "FKL before is not 24478f0"; exit 1; }
[ "$(git -C $T/worktrees/fkl-pr433-4b11bc6 rev-parse HEAD)" = 4b11bc6892a79d9481406da7903138ca567e2c6a ] || { echo "FKL after is not 4b11bc6"; exit 1; }
[ "$(git -C $WB rev-parse HEAD)" = b4fb59487e8f3cc7077bc9576e413080dddd2512 ] || { echo "before worktree is not at b4fb594"; exit 1; }
[ "$(git -C $WA rev-parse HEAD)" = b4fb59487e8f3cc7077bc9576e413080dddd2512 ] || { echo "after worktree is not at b4fb594"; exit 1; }
[ "$(git -C $WB/external/fkl rev-parse HEAD)" = 24478f0aa11dc3f12feea83178f7fde828322fb0 ] || { echo "before external/fkl is not 24478f0"; exit 1; }
[ "$(git -C $WA/external/fkl rev-parse HEAD)" = 4b11bc6892a79d9481406da7903138ca567e2c6a ] || { echo "after external/fkl is not 4b11bc6"; exit 1; }
[ -z "$(git -C $WB status --porcelain --untracked-files=no)" ] || { echo "the before worktree has local changes"; exit 1; }
git -C $WA diff -- opencv-cuda | cmp -s - $S/details-port.diff || { echo "the after worktree's change is not details-port.diff"; exit 1; }
[ "$(git -C $WA status --porcelain --untracked-files=no | sort | tr '\n' ' ')" = " M external/fkl  M opencv-cuda/stencil/e18_box.cu  M opencv-cuda/stencil/e18_box_split.cu  M opencv-cuda/stencil/e18_stencil.cu  M opencv-cuda/stencil/e19_batched_box.cu  M opencv-cuda/stencil/e19_crop_sweep.cu " ] \
  || { echo "the after worktree has other changes: $(git -C $WA status --porcelain --untracked-files=no | tr '\n' ' ')"; exit 1; }
STATUS=0

# A. FKL's test matrix
fkl_tests() {  # tag tree
  local tag=$1 src=$2 b=$S/fkl-$1-build
  echo; echo "# $(date -Iseconds) A $tag: configure $src ($(git -C $src rev-parse --short HEAD)), CUDA_ARCH=89"
  mkdir -p $b/.cmake/api/v1/query && touch $b/.cmake/api/v1/query/codemodel-v2
  taskset -c $CORES cmake -S $src -B $b -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc \
    -DCUDA_ARCH=89 > $S/fkl-$tag-configure.log 2>&1 || { echo "A $tag: CONFIGURE FAILED"; grep -m3 -A3 'CMake Error' $S/fkl-$tag-configure.log; exit 1; }
  python3 $REPORT_PY targets $b > $S/fkl-$tag-targets.txt || { echo "A $tag: the code model could not be read"; exit 1; }
  ctest --test-dir $b -N 2> /dev/null | sed -n 's/^ *Test *#[0-9]*: \(.*\)_rejected$/\1/p' > $S/fkl-$tag-rejected-targets.txt
  grep -vxFf $S/fkl-$tag-rejected-targets.txt $S/fkl-$tag-targets.txt > $S/fkl-$tag-targets-built.txt
  echo "A $tag: $(wc -l < $S/fkl-$tag-targets.txt) targets, $(wc -l < $S/fkl-$tag-rejected-targets.txt) of them expected rejections; building $(wc -l < $S/fkl-$tag-targets-built.txt)"
  local t0=$(date +%s)
  # shellcheck disable=SC2046
  /usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $S/fkl-$tag-time.txt \
    taskset -c $CORES nice -n 10 make -C $b -k -j$JOBS --output-sync=target $(tr '\n' ' ' < $S/fkl-$tag-targets-built.txt) > $S/fkl-$tag-build.log 2>&1
  echo "A $tag: build exit $? after $(( $(date +%s) - t0 )) s ($(tail -1 $S/fkl-$tag-time.txt))"
  local t1=$(date +%s)
  taskset -c $CORES nice -n 10 ctest --test-dir $b -V -R '_rejected$' -j1 --timeout 900 > $S/fkl-$tag-rejected.log 2>&1
  echo "A $tag: rejection tests after $(( $(date +%s) - t1 )) s: $(grep -E 'tests passed|tests failed' $S/fkl-$tag-rejected.log | tail -1)"
  python3 $REPORT_PY report fkltests-$tag $S/fkl-$tag-build.log $S/fkl-$tag-targets.txt $S/fkl-$tag-report.csv $S/fkl-$tag-rejected.log
  [ $? -eq 0 ] || STATUS=1
}
fkl_tests before $T/worktrees/fkl-pr433-base-24478f0
fkl_tests after $T/worktrees/fkl-pr433-4b11bc6

# B. the five fkl-benchmarks programs
TARGETS="e18_box_cu e18_box_split_cu e18_stencil_cu e19_batched_box_cu e19_crop_sweep_cu"
bench() {  # tag worktree
  local tag=$1 w=$2
  echo; echo "# $(date -Iseconds) B $tag: configure $w (HEAD $(git -C $w rev-parse --short HEAD), external/fkl $(git -C $w/external/fkl rev-parse --short HEAD))"
  ( cd $w && taskset -c $CORES cmake --preset local-rtx4090 > $S/bench-$tag-configure.log 2>&1 ) || { echo "B $tag: CONFIGURE FAILED"; tail -5 $S/bench-$tag-configure.log; exit 1; }
  local t0=$(date +%s) rc
  /usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $S/bench-$tag-time.txt \
    taskset -c $CORES nice -n 10 cmake --build $w/build/local-rtx4090 -j5 -- -k --output-sync=target $TARGETS > $S/bench-$tag-build.log 2> $S/bench-$tag-build-stderr.log
  rc=$?
  echo "B $tag: build exit $rc after $(( $(date +%s) - t0 )) s ($(tail -1 $S/bench-$tag-time.txt))"
  [ $rc -eq 0 ] || STATUS=1
  for t in $TARGETS; do
    if [ -x $w/build/local-rtx4090/bin/$t ]; then echo "B $tag $t: BUILT $(sha256sum < $w/build/local-rtx4090/bin/$t | cut -c1-64)"; else echo "B $tag $t: NOT BUILT"; fi
  done
}
bench before $WB
bench after $WA

# C. utest_flash_attention_mma_cu with CUDA_ARCH=native and no GPU visible
echo; echo "# $(date -Iseconds) C: utest_flash_attention_mma_cu at 4b11bc6, CUDA_ARCH=native, CUDA_VISIBLE_DEVICES empty"
b=$S/fkl-native-nogpu-build
CUDA_VISIBLE_DEVICES= taskset -c $CORES cmake -S $T/worktrees/fkl-pr433-4b11bc6 -B $b -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc > $S/fkl-native-nogpu-configure.log 2>&1
echo "C: configure exit $?; CUDA_ARCH in the cache: $(grep '^CUDA_ARCH:' $b/CMakeCache.txt 2>/dev/null); native detected by CMake: $(grep '^CMAKE_CUDA_ARCHITECTURES_NATIVE' $b/CMakeFiles/*/CMakeCUDACompiler.cmake 2>/dev/null | head -1)"
CUDA_VISIBLE_DEVICES= taskset -c $CORES nice -n 10 make -C $b VERBOSE=1 utest_flash_attention_mma_cu > $S/fkl-native-nogpu-build.log 2>&1
echo "C: build exit $?; first error: $(grep -m1 -E 'error|fatal' $S/fkl-native-nogpu-build.log | cut -c1-300)"
echo "C: the nvcc arch flags used: $(grep -m1 -oE -- '(-arch[= ][^ ]+|--generate-code=[^ ]+|-gencode[= ][^ ]+)( [^ ]*arch[^ ]*)*' $S/fkl-native-nogpu-build.log)"

echo; echo "# $(date -Iseconds) task B step 2 (compile only) ends, status $STATUS (0 = everything that should compile compiled); load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $STATUS
