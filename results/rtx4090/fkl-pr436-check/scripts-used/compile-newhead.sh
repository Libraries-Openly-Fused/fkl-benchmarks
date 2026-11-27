#!/bin/bash
# Task C-MAIN-NV, step 1b (COMPILE ONLY; no GPU is visible; nothing is run but configure, make and the listing of the tests): #436's head
# moves from 1df2738 to NEW (one commit by AMD, Oscar's decision on the FlashDecode finding: low_precision::attnToF32 in flash_decode.h and a
# new CUDA utest, utest_flash_decode.h). The head worktree (worktrees/fkl-pr436-head-1df2738, its path kept so that the step 1 build tree
# stays valid) is moved to NEW as a clean detached checkout; the head build tree is configured again with the same command and rebuilt with
# the same make command, over the targets of the code model (step 1's plus the new ones, which are listed); make rebuilds only what the new
# commit changes, and the targets that recompiled are listed from the build log. Before that, the 1df2738 test executables are copied to
# scratch/pr436/fkl-head-1df2738-bins (for the comparison of the two heads). nice 19, 8 jobs (fkl-reduce has priority on this machine).
# Usage: NEW=<full sha> compile-newhead.sh
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/pr436
TREE=$T/worktrees/fkl-pr436-head-1df2738; B=$S/fkl-head-build
OLD=1df2738e7bd8eba1bfb1a0baf9ad215062d014ca; NEW=${NEW:?the new head, a full sha}
[[ $NEW =~ ^[0-9a-f]{40}$ ]] || { echo "NEW is not a full sha"; exit 1; }
P=$S/compile/fkl-head-${NEW:0:7}
TP_SHA=27e733f50f7f40f171fd84655e05d5acf1fdfbee
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
REPORT_PY=$S/check_fkl_tree_report.py
LOG=$P-console.txt
[ -e $LOG ] && { echo "$LOG exists: not overwriting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
echo "# $(date -Iseconds) task C-MAIN-NV step 1b (compile only) starts; load $(cut -d' ' -f1-3 /proc/loadavg)"
echo "nvcc: $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//')"
[ "$(git -C $TREE rev-parse HEAD)" = $OLD ] || { echo "the head worktree is not at 1df2738"; exit 1; }
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "the head worktree is not clean"; exit 1; }
git -C $TREE fetch -q origin $NEW || { echo "fetch of $NEW failed"; exit 1; }
echo "commits 1df2738..${NEW:0:7}: $(git -C $TREE log --oneline $OLD..$NEW | tr '\n' ';')"
git -C $TREE merge-base --is-ancestor $OLD $NEW || { echo "$NEW does not descend from 1df2738"; exit 1; }
echo "files changed 1df2738..${NEW:0:7}: $(git -C $TREE diff --name-only $OLD $NEW | tr '\n' ' ')"
# the 1df2738 executables, before the rebuild replaces any
D=$S/fkl-head-1df2738-bins
[ -e $D ] && { echo "$D exists: not overwriting"; exit 1; }
mkdir -p $D && cp -p $B/bin/Release/*_cu $D/ && (cd $B/bin/Release && sha256sum $(ls | grep '_cu$')) > $D/sha256-in-build.txt \
  && (cd $D && sha256sum -c --quiet sha256-in-build.txt) || { echo "the copy of the 1df2738 executables failed"; exit 1; }
echo "the $(ls $D | grep -c '_cu$') 1df2738 executables copied to $D"
git -C $TREE checkout --quiet --detach $NEW || { echo "checkout failed"; exit 1; }
[ "$(git -C $TREE rev-parse HEAD)" = $NEW ] || { echo "the head worktree is not at $NEW"; exit 1; }
git -C $TREE submodule update --init -q thread-pool
[ "$(git -C $TREE/thread-pool rev-parse HEAD)" = $TP_SHA ] || { echo "thread-pool is not at $TP_SHA"; exit 1; }
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "the head worktree is not clean after the checkout"; exit 1; }
echo "head: $TREE at $(git -C $TREE rev-parse HEAD) ($(git -C $TREE describe --always --dirty --abbrev=7)), thread-pool $(git -C $TREE/thread-pool rev-parse --short HEAD), clean"
echo "cmake -S $TREE -B $B -G \"Unix Makefiles\" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc -DCUDA_ARCH=89  (nice -n 19)" > $P-commands.txt
taskset -c 8-15,24-31 nice -n 19 cmake -S $TREE -B $B -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc \
  -DCUDA_ARCH=89 > $P-configure.log 2>&1 || { echo "CONFIGURE FAILED"; tail -5 $P-configure.log; exit 1; }
echo "configure: $(grep 'version:' $P-configure.log | sed 's/^-- *//')"
python3 -I $REPORT_PY targets $B > $P-targets.txt || { echo "the code model could not be read"; exit 1; }
ctest --test-dir $B -N 2> /dev/null | sed -n 's/^ *Test *#[0-9]*: \(.*\)_rejected$/\1/p' > $P-rejected-targets.txt
grep -vxFf $P-rejected-targets.txt $P-targets.txt > $P-targets-built.txt
removed=$(comm -23 <(sort $S/compile/fkl-head-targets-built.txt) <(sort $P-targets-built.txt))
added=$(comm -13 <(sort $S/compile/fkl-head-targets-built.txt) <(sort $P-targets-built.txt))
echo "targets: $(wc -l < $P-targets-built.txt); new against step 1: ${added:-none}; gone: ${removed:-none}"
[ -z "$removed" ] || { echo "a target of step 1 is gone: stopping"; exit 1; }
echo "make -C $B -k -j8 --output-sync=target <the targets of $(basename $P)-targets-built.txt>  (taskset -c 8-15,24-31, nice -n 19)" >> $P-commands.txt
t0=$(date +%s)
# shellcheck disable=SC2046
/usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $P-time.txt \
  taskset -c 8-15,24-31 nice -n 19 make -C $B -k -j8 --output-sync=target $(tr '\n' ' ' < $P-targets-built.txt) > $P-build.log 2>&1
echo "build exit $? after $(( $(date +%s) - t0 )) s ($(tail -1 $P-time.txt); not a result)"
grep -oE '(Building (CUDA|CXX|C) object|Linking (CUDA|CXX|C) (executable|shared library)) \S+' $P-build.log > $P-recompiled.txt
echo "what make rebuilt ($(wc -l < $P-recompiled.txt) lines, also in $(basename $P)-recompiled.txt):"; sed 's/^/  /' $P-recompiled.txt
python3 -I $REPORT_PY report fkltests-head-${NEW:0:7} $P-build.log $P-targets.txt $P-report.csv | tail -1
RC=${PIPESTATUS[0]}
echo "CUDA tests now: $(ctest --test-dir $B -N -R '_cu$' 2>/dev/null | grep -c 'Test *#')"
[ -z "$(git -C $TREE status --porcelain --ignore-submodules=none --untracked-files=all)" ] && [ "$(git -C $TREE rev-parse HEAD)" = $NEW ] \
  && echo "head-after-build: at $NEW, clean" || { echo "the head worktree changed during the build"; RC=1; }
echo "# $(date -Iseconds) task C-MAIN-NV step 1b (compile only) ends, report exit $RC; load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $RC
