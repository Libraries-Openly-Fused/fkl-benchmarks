#!/bin/bash
# Task C-MAIN-NV, step 1 (COMPILE ONLY; no GPU is visible to any process of this step, every nvcc gets sm_89 through CUDA_ARCH=89, and
# nothing is run except the compile-fail ctest tests (<target>_rejected), which only build): FKL PR #436 (head 1df2738, task C on FKL main)
# against its base, main ebb3891 (C++20), CUDA. A copy of task C-NV's step 1 (scratch/pr435/compile.sh) with these commits and, because
# another session (fkl-reduce) has priority on this machine, nice 19 and at most 8 make jobs in all (4 per build).
#   FKL's own test matrix (tests/ and utests/), every target that compiles something, at ebb3891 (base) and 1df2738 (head), each from its own
#   clean detached worktree of the FusedKernelLibrary clone (thread-pool submodule 27e733f): FKL's CMake out of source, Release,
#   -DCUDA_ARCH=89 (FKL sets CMAKE_CUDA_ARCHITECTURES OFF and gives every target CUDA_ARCH, whose default is native), nvcc 13.4.3.
#   These are the flags of the #433 check (results/rtx4090/task-b-sm89-check/scripts-used/compile-step2.sh, part A).
#   The targets come from CMake's code model, as in scripts/check_fkl_tree.sh; the expected rejections are judged by their ctest tests.
#   Per-target report with fkl-benchmarks' scripts/check_fkl_tree_report.py (copied from main 976e42b).
# The two builds run at the same time on separate cores (base 8-11,24-27, head 12-15,28-31), 4 jobs each, nice 19, TMPDIR on disk.
# A failed target is recorded, not fatal (the report says what failed); a configure failure or a tree that is not clean is fatal.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/pr436
BASE_TREE=$T/worktrees/fkl-pr436-base-ebb3891; HEAD_TREE=$T/worktrees/fkl-pr436-head-1df2738
BASE_SHA=ebb3891bea18a152d92dfd5092e117160490e34b; HEAD_SHA=1df2738e7bd8eba1bfb1a0baf9ad215062d014ca
TP_SHA=27e733f50f7f40f171fd84655e05d5acf1fdfbee
export TMPDIR=$S/tmp; mkdir -p $TMPDIR $S/compile
REPORT_PY=$S/check_fkl_tree_report.py
LOG=$S/compile/console.txt
[ -e $LOG ] && { echo "$LOG exists: not overwriting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
echo "# $(date -Iseconds) task C-MAIN-NV step 1 (compile only) starts; load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "nvcc: $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//')"
echo "nvcc path: $FKLB_CUDA_HOME/bin/nvcc; cmake: $(cmake --version | head -1); c++: $(c++ --version | head -1); make: $(make --version | head -1)"
echo "check_fkl_tree_report.py sha256: $(sha256sum < $REPORT_PY | cut -c1-64)"

check_tree() {  # tag tree sha
  local tag=$1 tree=$2 sha=$3
  [ "$(git -C $tree rev-parse HEAD)" = $sha ] || { echo "$tag: $tree is not at $sha"; exit 1; }
  [ "$(git -C $tree/thread-pool rev-parse HEAD)" = $TP_SHA ] || { echo "$tag: thread-pool is not at $TP_SHA"; exit 1; }
  [ -z "$(git -C $tree status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "$tag: $tree is not clean: $(git -C $tree status --porcelain | head -5 | tr '\n' ' ')"; exit 1; }
  echo "$tag: $tree at $(git -C $tree rev-parse HEAD) ($(git -C $tree describe --always --dirty --abbrev=7)), thread-pool $(git -C $tree/thread-pool rev-parse --short HEAD), clean"
}
check_tree base $BASE_TREE $BASE_SHA
check_tree head $HEAD_TREE $HEAD_SHA

fkl_tests() {  # tag tree cores
  local tag=$1 src=$2 cores=$3 b=$S/fkl-$1-build p=$S/compile/fkl-$1
  echo "# $(date -Iseconds) $tag: configure $src ($(git -C $src rev-parse --short HEAD)), CUDA_ARCH=89, cores $cores"
  mkdir -p $b/.cmake/api/v1/query && touch $b/.cmake/api/v1/query/codemodel-v2
  echo "cmake -S $src -B $b -G \"Unix Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc -DCUDA_ARCH=89" > $p-commands.txt
  taskset -c $cores nice -n 19 cmake -S $src -B $b -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc \
    -DCUDA_ARCH=89 > $p-configure.log 2>&1 || { echo "$tag: CONFIGURE FAILED"; grep -m3 -A3 'CMake Error' $p-configure.log; return 2; }
  python3 -I $REPORT_PY targets $b > $p-targets.txt || { echo "$tag: the code model could not be read"; return 2; }
  ctest --test-dir $b -N 2> /dev/null | sed -n 's/^ *Test *#[0-9]*: \(.*\)_rejected$/\1/p' > $p-rejected-targets.txt
  grep -vxFf $p-rejected-targets.txt $p-targets.txt > $p-targets-built.txt
  echo "$tag: $(wc -l < $p-targets.txt) targets, $(wc -l < $p-rejected-targets.txt) of them expected rejections; building $(wc -l < $p-targets-built.txt)"
  echo "make -C $b -k -j4 --output-sync=target <the targets of $(basename $p)-targets-built.txt>  (taskset -c $cores, nice -n 19)" >> $p-commands.txt
  local t0=$(date +%s)
  # shellcheck disable=SC2046
  /usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $p-time.txt \
    taskset -c $cores nice -n 19 make -C $b -k -j4 --output-sync=target $(tr '\n' ' ' < $p-targets-built.txt) > $p-build.log 2>&1
  echo "$tag: build exit $? after $(( $(date +%s) - t0 )) s ($(tail -1 $p-time.txt))"
  echo "ctest --test-dir $b -V -R '_rejected\$' -j1 --timeout 900  (taskset -c $cores, nice -n 19)" >> $p-commands.txt
  local t1=$(date +%s)
  taskset -c $cores nice -n 19 ctest --test-dir $b -V -R '_rejected$' -j1 --timeout 900 > $p-rejected.log 2>&1
  echo "$tag: rejection tests after $(( $(date +%s) - t1 )) s: $(grep -E 'tests passed|tests failed' $p-rejected.log | tail -1)"
  python3 -I $REPORT_PY report fkltests-$tag $p-build.log $p-targets.txt $p-report.csv $p-rejected.log
}
fkl_tests base $BASE_TREE 8-11,24-27 > $S/compile/fkl-base-console.txt 2>&1 &
PB=$!
fkl_tests head $HEAD_TREE 12-15,28-31 > $S/compile/fkl-head-console.txt 2>&1 &
PH=$!
wait $PB; RB=$?
wait $PH; RH=$?
echo "--- base console"; cat $S/compile/fkl-base-console.txt
echo "--- head console"; cat $S/compile/fkl-head-console.txt
check_tree base-after-build $BASE_TREE $BASE_SHA
check_tree head-after-build $HEAD_TREE $HEAD_SHA
echo "# $(date -Iseconds) task C-MAIN-NV step 1 (compile only) ends: report exit base $RB, head $RH (0 = every target that should compile compiled and every rejection held); load $(cut -d' ' -f1-3 /proc/loadavg)"
[ $RB -eq 0 ] && [ $RH -eq 0 ]
