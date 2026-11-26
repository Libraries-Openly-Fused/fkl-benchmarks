#!/bin/bash
# Task C-NV-A, step 1 (COMPILE ONLY; no GPU is visible): #435's head moved from 48fbf3f to d1cff06 (one commit, which changes only
# include/fused_kernel/algorithms/attention/flash_attention_mma.h: fix A, the attribute call of flashAttentionMmaSplits moves into its static
# initializer). The head worktree (worktrees/fkl-pr435-e322cad, its path kept so that the step 1 build tree stays valid) is moved to d1cff06
# as a clean detached checkout; the head build tree is configured again with the same command (its version message then names d1cff06)
# and rebuilt with the same make command and target list: make rebuilds only what the new commit changes. The targets that recompiled are
# listed from the build log. The rejection tests run again. The 48fbf3f executables were copied to scratch/pr435/fkl-head48-bins first.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/pr435
TREE=$T/worktrees/fkl-pr435-e322cad; B=$S/fkl-head-build; P=$S/compile/fkl-head-d1cff06
OLD=48fbf3f08e7551c3eb70ac2302d8ebc15ecad645; NEW=d1cff068dd2063ec9d4ff0206aef1836e98ba036
TP_SHA=27e733f50f7f40f171fd84655e05d5acf1fdfbee
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
REPORT_PY=$S/check_fkl_tree_report.py
LOG=$P-console.txt
[ -e $LOG ] && { echo "$LOG exists: not overwriting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
echo "# $(date -Iseconds) task C-NV-A step 1 (compile only) starts; load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "nvcc: $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//')"
[ "$(git -C $TREE rev-parse HEAD)" = $OLD ] || { echo "the head worktree is not at 48fbf3f"; exit 1; }
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "the head worktree is not clean"; exit 1; }
echo "files changed 48fbf3f..d1cff06: $(git -C $TREE diff --name-only $OLD $NEW | tr '\n' ' ')"
git -C $TREE checkout --quiet --detach $NEW || { echo "checkout failed"; exit 1; }
[ "$(git -C $TREE rev-parse HEAD)" = $NEW ] || { echo "the head worktree is not at d1cff06"; exit 1; }
[ "$(git -C $TREE/thread-pool rev-parse HEAD)" = $TP_SHA ] || { echo "thread-pool is not at $TP_SHA"; exit 1; }
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "the head worktree is not clean after the checkout"; exit 1; }
echo "head: $TREE at $(git -C $TREE rev-parse HEAD) ($(git -C $TREE describe --always --dirty --abbrev=7)), thread-pool $(git -C $TREE/thread-pool rev-parse --short HEAD), clean"
echo "cmake -S $TREE -B $B -G \"Unix Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc -DCUDA_ARCH=89" > $P-commands.txt
taskset -c 0-7,16-23 cmake -S $TREE -B $B -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc \
  -DCUDA_ARCH=89 > $P-configure.log 2>&1 || { echo "CONFIGURE FAILED"; tail -5 $P-configure.log; exit 1; }
echo "configure: $(grep 'version:' $P-configure.log | sed 's/^-- *//')"
python3 -I $REPORT_PY targets $B > $P-targets.txt || { echo "the code model could not be read"; exit 1; }
ctest --test-dir $B -N 2> /dev/null | sed -n 's/^ *Test *#[0-9]*: \(.*\)_rejected$/\1/p' > $P-rejected-targets.txt
grep -vxFf $P-rejected-targets.txt $P-targets.txt > $P-targets-built.txt
cmp -s $P-targets-built.txt $S/compile/fkl-head-targets-built.txt && echo "the target list is the one of step 1 ($(wc -l < $P-targets-built.txt) targets)" \
  || { echo "the target list differs from step 1"; diff $S/compile/fkl-head-targets-built.txt $P-targets-built.txt; exit 1; }
echo "make -C $B -k -j12 --output-sync=target <the targets of $(basename $P)-targets-built.txt>  (taskset -c 0-7,16-23, nice -n 10)" >> $P-commands.txt
t0=$(date +%s)
# shellcheck disable=SC2046
/usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $P-time.txt \
  taskset -c 0-7,16-23 nice -n 10 make -C $B -k -j12 --output-sync=target $(tr '\n' ' ' < $P-targets-built.txt) > $P-build.log 2>&1
echo "build exit $? after $(( $(date +%s) - t0 )) s ($(tail -1 $P-time.txt))"
grep -oE '(Building (CUDA|CXX|C) object|Linking (CUDA|CXX|C) (executable|shared library)) \S+' $P-build.log > $P-recompiled.txt
echo "what make rebuilt ($(wc -l < $P-recompiled.txt) lines, also in $(basename $P)-recompiled.txt):"; sed 's/^/  /' $P-recompiled.txt
echo "ctest --test-dir $B -V -R '_rejected\$' -j1 --timeout 900  (taskset -c 0-7,16-23, nice -n 10)" >> $P-commands.txt
taskset -c 0-7,16-23 nice -n 10 ctest --test-dir $B -V -R '_rejected$' -j1 --timeout 900 > $P-rejected.log 2>&1
echo "rejection tests: $(grep -E 'tests passed|tests failed' $P-rejected.log | tail -1)"
python3 -I $REPORT_PY report fkltests-head-d1cff06 $P-build.log $P-targets.txt $P-report.csv $P-rejected.log | tail -1
RC=${PIPESTATUS[0]}
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] && [ "$(git -C $TREE rev-parse HEAD)" = $NEW ] \
  && echo "head-after-build: at $NEW, clean" || { echo "the head worktree changed during the build"; RC=1; }
echo "# $(date -Iseconds) task C-NV-A step 1 (compile only) ends, report exit $RC; load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $RC
