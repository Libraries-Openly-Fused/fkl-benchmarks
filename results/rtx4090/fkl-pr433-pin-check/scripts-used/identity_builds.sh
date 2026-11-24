#!/bin/bash
# The pin move of fkl-benchmarks to FKL b10fc66 (the merge of FKL #433, batched filter DPPs): the CUDA device-code identity check, COMPILE ONLY.
# No program runs, for any purpose; no GPU is visible (CUDA_VISIBLE_DEVICES is empty). The frozen 2025 programs are compiled and their SASS
# dumped with cuobjdump, nothing more.
#   base = fkl-benchmarks main a8f9daa with FKL 24478f0 (its gitlink): the clean detached worktree worktrees/fkl-benchmarks-pin-433-base
#   head = branch pin-fkl-433 at 2160c70 with FKL b10fc66 (its gitlink) and the two details ports: worktrees/fkl-benchmarks-pin-433
# Both: the bench group of scripts/check_fkl_tree.sh of their own checkout (every *_cu and *_cpp target), built at the same time on separate
# cores, with the FKLB_* options, OPENCV_DIR and nvcc of the reference build build/local-rtx4090 (only read), as in results/rtx4090/fkl-pr432-check/.
# Then:
#   1. sass_identity.py base vs head over every *_cu program, and rename_causes.py (the scripts of fkl-pr432-check, unchanged);
#   2. sass_identity.py of the 5 programs of task B (e18_box, e18_box_split, e18_stencil, e19_batched_box, e19_crop_sweep): the "after" binaries
#      of task B (worktrees/fkl-benchmarks-task-b-after/build/local-rtx4090/bin: b4fb594-dirty + FKL 4b11bc6, the binaries whose outputs the task B
#      window checked; their sha256 are checked against results/rtx4090/task-b-sm89-check/window-binaries.sha256 first) vs the head build;
#   3. sha256 of every binary of both builds.
# Every failed step stops the script. The identity reports are written whatever they find (their exit status is recorded, not acted on).
# Output: scratch/pin-433/{base-24478f0,head-b10fc66}/ (the check_fkl_tree.sh folders), scratch/pin-433/out/ and scratch/pin-433/console.txt.
set -u
T=/home/oscar/tfkl-bench; REF=$T/fkl-benchmarks/build/local-rtx4090; O=$T/scratch/pin-433
B=$T/worktrees/fkl-benchmarks-pin-433-base; W=$T/worktrees/fkl-benchmarks-pin-433; A=$T/worktrees/fkl-benchmarks-task-b-after/build/local-rtx4090/bin
B_SHA=a8f9daa3cc529e8a13c88ae63fcbb227e89adf61; B_FKL=24478f0aa11dc3f12feea83178f7fde828322fb0
W_SHA=2160c70d34f980b87601bebf7ad16658184a019f; W_FKL=b10fc6609426bd18020b9213c7196b528fc677a8; TP=27e733f50f7f40f171fd84655e05d5acf1fdfbee
MANIFEST=$W/results/rtx4090/task-b-sm89-check/window-binaries.sha256
SU=$O/scripts-used; PROGS="e18_box_cu e18_box_split_cu e18_stencil_cu e19_batched_box_cu e19_crop_sweep_cu"
say() { echo "# $(date -Iseconds) $*"; }
fail() { say "STOP: $*"; exit 1; }
for d in $O/base-24478f0 $O/head-b10fc66 $O/out; do [ -e $d ] && { echo "$d exists: not starting"; exit 1; }; done
exec > >(tee -a $O/console.txt) 2>&1
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$O/tmp; mkdir -p $TMPDIR $O/out
say "pin-433 identity builds start (compile only, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'); script sha256 $(sha256sum "$0" | cut -d' ' -f1)"
# the checkouts
[ "$(git -C $B rev-parse HEAD)" = $B_SHA ] || fail "base worktree is not at a8f9daa"
[ "$(git -C $B/external/fkl rev-parse HEAD)" = $B_FKL ] || fail "base external/fkl is not at 24478f0"
[ "$(git -C $B ls-tree HEAD external/fkl | awk '{print $3}')" = $B_FKL ] || fail "base gitlink is not 24478f0"
[ "$(git -C $W rev-parse HEAD)" = $W_SHA ] || fail "head worktree is not at 2160c70"
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] || fail "head external/fkl is not at b10fc66"
[ "$(git -C $W ls-tree HEAD external/fkl | awk '{print $3}')" = $W_FKL ] || fail "head gitlink is not b10fc66"
for d in $B $W; do
  [ "$(git -C $d/external/fkl/thread-pool rev-parse HEAD)" = $TP ] || fail "$d: thread-pool is not at 27e733f"
  [ -z "$(git -C $d status --porcelain --untracked-files=no)" ] || fail "$d has tracked changes"
  [ -z "$(git -C $d/external/fkl status --porcelain --untracked-files=no)" ] || fail "$d/external/fkl has tracked changes"
  for s in external/cvGPUSpeedup external/FastNPP; do
    [ "$(git -C $d/$s rev-parse HEAD)" = "$(git -C $d ls-tree HEAD $s | awk '{print $3}')" ] || fail "$d/$s is not at its gitlink"
  done
done
[ "$(git -C $B rev-parse 'HEAD^{tree}')" = "$(git -C $W rev-parse 'HEAD~1^{tree}')" ] || fail "the head's parent is not main's tree"
# the task B "after" binaries are the recorded ones
grep "fkl-benchmarks-task-b-after/build/local-rtx4090/bin/" $MANIFEST > $O/out/taskb-after-binaries.sha256
[ "$(wc -l < $O/out/taskb-after-binaries.sha256)" = 5 ] || fail "the manifest has not 5 after binaries"
sha256sum -c $O/out/taskb-after-binaries.sha256 || fail "the task B after binaries are not the recorded ones"
# the comparison scripts: the ones of fkl-pr432-check, unchanged
mkdir -p $SU; cp $W/results/rtx4090/fkl-pr432-check/scripts-used/{sass_identity.py,rename_causes.py} $SU/ || fail "copy of the scripts"
[ "$(sha256sum $SU/sass_identity.py | cut -d' ' -f1)" = c8ab1c23168b8026c0d3d6dc07fa90383e2f6ebb59c711b932be11324aaf375f ] || fail "sass_identity.py is not the recorded one"
cp "$0" $SU/identity_builds.sh
say "base: fkl-benchmarks $(git -C $B rev-parse --short HEAD), FKL $(git -C $B/external/fkl rev-parse --short HEAD); head: fkl-benchmarks $(git -C $W rev-parse --short HEAD), FKL $(git -C $W/external/fkl rev-parse --short HEAD); cvGS $(git -C $W/external/cvGPUSpeedup rev-parse --short HEAD), FastNPP $(git -C $W/external/FastNPP rev-parse --short HEAD); load $(cut -d' ' -f1-3 /proc/loadavg)"
# the two builds
$B/scripts/check_fkl_tree.sh $B/external/fkl --out $O/base-24478f0 --only bench --ref-build $REF --jobs 12 --cores 8-15,24-31 > $O/base-run.log 2>&1 &
PA=$!
$W/scripts/check_fkl_tree.sh $W/external/fkl --out $O/head-b10fc66 --only bench --ref-build $REF --jobs 12 --cores 0-7,16-23 > $O/head-run.log 2>&1 &
PB=$!
wait $PA; RA=$?; say "base build ended, exit $RA"
wait $PB; RB=$?; say "head build ended, exit $RB"
for r in base-24478f0 head-b10fc66; do echo "== $r"; grep -E '^bench: ' $O/$r/report.txt | cut -c1-200; awk -F, 'NR > 1 {n[$3]++} END {for (k in n) printf "  %s %d\n", k, n[k]}' $O/$r/bench-report.csv; awk -F, 'NR > 1 && $3 != "OK"' $O/$r/bench-report.csv | head -20; done
[ "$(git -C $B/external/fkl rev-parse HEAD)" = $B_FKL ] && [ "$(git -C $B rev-parse HEAD)" = $B_SHA ] || fail "the base checkout moved during the build"
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] && [ "$(git -C $W rev-parse HEAD)" = $W_SHA ] || fail "the head checkout moved during the build"
[ $RA = 0 ] || fail "the base build failed (exit $RA)"
[ $RB = 0 ] || fail "the head build failed (exit $RB)"
BA=$O/base-24478f0/bench-build/bin; BB=$O/head-b10fc66/bench-build/bin
[ -d $BA ] && [ -d $BB ] || fail "a bin folder is missing"
# 1. base vs head, every *_cu program
say "1. sass_identity base vs head"
python3 -I $SU/sass_identity.py $BA $BB --tools $W/scripts --arch sm_89 --glob '*_cu' --out $O/out/sass-identity-24478f0-vs-b10fc66.txt > /dev/null; R1=$?
say "sass_identity exit $R1"; sed -n 3p $O/out/sass-identity-24478f0-vs-b10fc66.txt; grep '^TOTAL' $O/out/sass-identity-24478f0-vs-b10fc66.txt
python3 -I $SU/rename_causes.py $BA $BB --arch sm_89 --glob '*_cu' > $O/out/rename-causes-24478f0-vs-b10fc66.txt || fail "rename_causes.py base vs head"
tail -n 3 $O/out/rename-causes-24478f0-vs-b10fc66.txt
# 2. the 5 programs: task B after vs head
say "2. sass_identity task B after vs head, the 5 programs"
mkdir -p $O/out/head-5 || fail "mkdir head-5"
for p in $PROGS; do ln -s $BB/$p $O/out/head-5/$p || fail "link $p"; done
python3 -I $SU/sass_identity.py $A $O/out/head-5 --tools $W/scripts --arch sm_89 --glob '*_cu' --out $O/out/sass-identity-taskb-after-vs-b10fc66.txt > /dev/null; R2=$?
say "sass_identity exit $R2"; sed -n 3p $O/out/sass-identity-taskb-after-vs-b10fc66.txt; grep '^TOTAL' $O/out/sass-identity-taskb-after-vs-b10fc66.txt
python3 -I $SU/rename_causes.py $A $O/out/head-5 --arch sm_89 --glob '*_cu' > $O/out/rename-causes-taskb-after-vs-b10fc66.txt || fail "rename_causes.py task B after vs head"
tail -n 3 $O/out/rename-causes-taskb-after-vs-b10fc66.txt
# 3. sha256 of the binaries
(cd $BA && sha256sum $(ls | sort)) > $O/out/binary-sha256-base-24478f0.txt || fail "sha256 base"
(cd $BB && sha256sum $(ls | sort)) > $O/out/binary-sha256-head-b10fc66.txt || fail "sha256 head"
say "binaries: base $(wc -l < $O/out/binary-sha256-base-24478f0.txt), head $(wc -l < $O/out/binary-sha256-head-b10fc66.txt); byte-identical pairs $(join <(awk '{print $2, $1}' $O/out/binary-sha256-base-24478f0.txt) <(awk '{print $2, $1}' $O/out/binary-sha256-head-b10fc66.txt) | awk '$2 == $3' | wc -l)"
say "pin-433 identity builds ended (exit codes: identity 1 = $R1, identity 2 = $R2); load $(cut -d' ' -f1-3 /proc/loadavg)"
