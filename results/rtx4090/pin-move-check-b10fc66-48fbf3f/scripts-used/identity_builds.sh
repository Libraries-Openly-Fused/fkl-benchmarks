#!/bin/bash
# C-PM-CUDA: the pin move of fkl-benchmarks to FKL 48fbf3f (FKL #435's head: every DPP through executeOperations<DPP>, one kernel
# launchDPP_Kernel): the CUDA device-code identity check, COMPILE ONLY. No program runs, for any purpose; no GPU is visible
# (CUDA_VISIBLE_DEVICES is empty). The frozen 2025 programs are compiled and their SASS dumped with cuobjdump, nothing more.
#   base = fkl-benchmarks 976e42b (main before #64/#65, the same code as main bd019f6) with FKL b10fc66 (its gitlink):
#          the clean detached worktree worktrees/fkl-benchmarks-pin-435-base
#   head = branch pin-fkl-435 at 2518c75 (AMD's ede9e5e + a4b493c, and the four CUDA port commits) with FKL 48fbf3f (its gitlink):
#          worktrees/fkl-benchmarks-pin-435
# Both: the bench group of scripts/check_fkl_tree.sh of their own checkout (every *_cu and *_cpp target), built at the same time on separate
# cores, with the FKLB_* options, OPENCV_DIR and nvcc of the reference build build/local-rtx4090 (only read), as in
# results/rtx4090/fkl-pr433-pin-check/. Then:
#   1. pr435_sass_identity.py base vs head over every *_cu program: the kernels paired by their demangled names with FKL #435's rename rule
#      (the script of results/rtx4090/fkl-pr435-check/, sha256 checked), with registers, stack, LDL / STL and instruction counts;
#   2. diff_summary.py of the differing kernels;
#   3. sha256 of every binary of both builds.
# Every failed step stops the script. The identity report is written whatever it finds (its exit status is recorded, not acted on).
# Output: scratch/pin-435/{base-b10fc66,head-48fbf3f}/ (the check_fkl_tree.sh folders), scratch/pin-435/out/ and scratch/pin-435/console.txt.
set -u
T=/home/oscar/tfkl-bench; REF=$T/fkl-benchmarks/build/local-rtx4090; O=$T/scratch/pin-435
B=$T/worktrees/fkl-benchmarks-pin-435-base; W=$T/worktrees/fkl-benchmarks-pin-435
B_SHA=976e42bfad0ba61c3bb22a555cf72f4da6896aeb; B_FKL=b10fc6609426bd18020b9213c7196b528fc677a8
W_SHA=2518c75219a4af8039f9a8d062323bcbb4181487; W_FKL=48fbf3f08e7551c3eb70ac2302d8ebc15ecad645; TP=27e733f50f7f40f171fd84655e05d5acf1fdfbee
SU=$O/scripts-used
say() { echo "# $(date -Iseconds) $*"; }
fail() { say "STOP: $*"; exit 1; }
for d in $O/base-b10fc66 $O/head-48fbf3f $O/out $O/console.txt; do [ -e $d ] && { echo "$d exists: not starting"; exit 1; }; done
exec > >(tee -a $O/console.txt) 2>&1
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$O/tmp; mkdir -p $TMPDIR $O/out
say "pin-435 identity builds start (compile only, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'); script sha256 $(sha256sum "$0" | cut -d' ' -f1)"
# the checkouts
[ "$(git -C $B rev-parse HEAD)" = $B_SHA ] || fail "base worktree is not at 976e42b"
[ "$(git -C $B/external/fkl rev-parse HEAD)" = $B_FKL ] || fail "base external/fkl is not at b10fc66"
[ "$(git -C $B ls-tree HEAD external/fkl | awk '{print $3}')" = $B_FKL ] || fail "base gitlink is not b10fc66"
[ "$(git -C $W rev-parse HEAD)" = $W_SHA ] || fail "head worktree is not at 2518c75"
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] || fail "head external/fkl is not at 48fbf3f"
[ "$(git -C $W ls-tree HEAD external/fkl | awk '{print $3}')" = $W_FKL ] || fail "head gitlink is not 48fbf3f"
for d in $B $W; do
  [ "$(git -C $d/external/fkl/thread-pool rev-parse HEAD)" = $TP ] || fail "$d: thread-pool is not at 27e733f"
  [ -z "$(git -C $d status --porcelain --untracked-files=no)" ] || fail "$d has tracked changes"
  [ -z "$(git -C $d/external/fkl status --porcelain --untracked-files=no)" ] || fail "$d/external/fkl has tracked changes"
  for s in external/cvGPUSpeedup external/FastNPP; do
    [ "$(git -C $d/$s rev-parse HEAD)" = "$(git -C $d ls-tree HEAD $s | awk '{print $3}')" ] || fail "$d/$s is not at its gitlink"
  done
done
git -C $W merge-base --is-ancestor $B_SHA HEAD || fail "the head does not descend from 976e42b"
# the comparison scripts
mkdir -p $SU; cp $T/scratch/pr435/analysis/pr435_sass_identity.py $T/scratch/pr435/analysis/diff_summary.py $SU/ || fail "copy of the scripts"
[ "$(sha256sum $SU/pr435_sass_identity.py | cut -d' ' -f1)" = c9cedd90cf9e42b9beaccd9ad1966bf26f9117eb3bd3c28cd48941cdc4a44b1e ] || fail "pr435_sass_identity.py is not the recorded one"
cp "$0" $SU/identity_builds.sh
say "base: fkl-benchmarks $(git -C $B rev-parse --short HEAD), FKL $(git -C $B/external/fkl rev-parse --short HEAD); head: fkl-benchmarks $(git -C $W rev-parse --short HEAD), FKL $(git -C $W/external/fkl rev-parse --short HEAD); cvGS $(git -C $W/external/cvGPUSpeedup rev-parse --short HEAD), FastNPP $(git -C $W/external/FastNPP rev-parse --short HEAD); nvcc $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//'); load $(cut -d' ' -f1-3 /proc/loadavg)"
# the two builds
$B/scripts/check_fkl_tree.sh $B/external/fkl --out $O/base-b10fc66 --only bench --ref-build $REF --jobs 12 --cores 8-15,24-31 > $O/base-run.log 2>&1 &
PA=$!
$W/scripts/check_fkl_tree.sh $W/external/fkl --out $O/head-48fbf3f --only bench --ref-build $REF --jobs 12 --cores 0-7,16-23 > $O/head-run.log 2>&1 &
PB=$!
wait $PA; RA=$?; say "base build ended, exit $RA"
wait $PB; RB=$?; say "head build ended, exit $RB"
for r in base-b10fc66 head-48fbf3f; do echo "== $r"; grep -E '^bench: ' $O/$r/report.txt | cut -c1-200; awk -F, 'NR > 1 {n[$3]++} END {for (k in n) printf "  %s %d\n", k, n[k]}' $O/$r/bench-report.csv; awk -F, 'NR > 1 && $3 != "OK"' $O/$r/bench-report.csv | head -20; done
[ "$(git -C $B/external/fkl rev-parse HEAD)" = $B_FKL ] && [ "$(git -C $B rev-parse HEAD)" = $B_SHA ] || fail "the base checkout moved during the build"
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] && [ "$(git -C $W rev-parse HEAD)" = $W_SHA ] || fail "the head checkout moved during the build"
[ $RA = 0 ] || fail "the base build failed (exit $RA)"
[ $RB = 0 ] || fail "the head build failed (exit $RB)"
BA=$O/base-b10fc66/bench-build/bin; BB=$O/head-48fbf3f/bench-build/bin
[ -d $BA ] && [ -d $BB ] || fail "a bin folder is missing"
# 1. base vs head, every *_cu program
say "1. pr435_sass_identity base vs head"
python3 -I $SU/pr435_sass_identity.py --tools $W/scripts --a b10fc66=$BA --b 48fbf3f=$BB --out $O/out/sass-identity-b10fc66-vs-48fbf3f.txt \
  --csv $O/out/sass-identity-b10fc66-vs-48fbf3f.csv > /dev/null; R1=$?
say "pr435_sass_identity exit $R1"; sed -n 3,9p $O/out/sass-identity-b10fc66-vs-48fbf3f.txt
# 2. the differing kernels by DPP
python3 -I $SU/diff_summary.py $O/out/sass-identity-b10fc66-vs-48fbf3f.csv --out $O/out/differences-b10fc66-vs-48fbf3f.txt > /dev/null || fail "diff_summary.py"
head -40 $O/out/differences-b10fc66-vs-48fbf3f.txt
# 3. sha256 of the binaries
(cd $BA && sha256sum $(ls | sort)) > $O/out/binary-sha256-base-b10fc66.txt || fail "sha256 base"
(cd $BB && sha256sum $(ls | sort)) > $O/out/binary-sha256-head-48fbf3f.txt || fail "sha256 head"
say "binaries: base $(wc -l < $O/out/binary-sha256-base-b10fc66.txt), head $(wc -l < $O/out/binary-sha256-head-48fbf3f.txt); byte-identical pairs $(join <(awk '{print $2, $1}' $O/out/binary-sha256-base-b10fc66.txt) <(awk '{print $2, $1}' $O/out/binary-sha256-head-48fbf3f.txt) | awk '$2 == $3' | wc -l)"
say "pin-435 identity builds ended (identity exit $R1); load $(cut -d' ' -f1-3 /proc/loadavg)"
