#!/bin/bash
# Device-code identity check of the DPP port, step 1 (COMPILE ONLY, no program runs): the bench group of
# scripts/check_fkl_tree.sh at fkl-benchmarks 8f4a95b, built twice at the same time on separate cores:
#   base   = FKL 64ad9195 (the pin), worktree worktrees/fkl-base-64ad9195
#   branch = FKL LTS-C++17-image-processing-dpps, worktree worktrees/fkl-image-dpps (its HEAD is recorded)
# The FKLB_* options, OPENCV_DIR and nvcc come from the CMakeCache of the reference build build/local-rtx4090, which is
# only read. Output: scratch/dpp-port/identity/{base,branch}-<sha>/ and identity/console.txt.
set -u
T=/home/oscar/tfkl-bench; FBW=$T/worktrees/fkl-benchmarks-8f4a95b; REF=$T/fkl-benchmarks/build/local-rtx4090
BASE=$T/worktrees/fkl-base-64ad9195; BRANCH=$T/worktrees/fkl-image-dpps; O=$T/scratch/dpp-port/identity
mkdir -p $O
exec > >(tee -a $O/console.txt) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
[ "$(git -C $FBW rev-parse HEAD)" = 8f4a95be2d3d6db8646e061d228d4b694fb21e9d ] || { say "fkl-benchmarks worktree is not at 8f4a95b: stop"; exit 1; }
[ "$(git -C $BASE rev-parse HEAD)" = 64ad9195cd785d0c93a70b35e02d5a6c7c4a5048 ] || { say "base worktree is not at 64ad9195: stop"; exit 1; }
for d in $BASE $BRANCH $FBW; do [ -z "$(git -C $d status --porcelain --untracked-files=no)" ] || { say "$d has local changes: stop"; exit 1; }; done
BR=$(git -C $BRANCH rev-parse HEAD)
say "identity builds start: fkl-benchmarks $(git -C $FBW rev-parse --short HEAD), cvGS $(git -C $FBW/external/cvGPUSpeedup rev-parse --short HEAD), FastNPP $(git -C $FBW/external/FastNPP rev-parse --short HEAD); base $(git -C $BASE rev-parse --short HEAD), branch ${BR:0:7}; load $(cut -d' ' -f1-3 /proc/loadavg)"
$FBW/scripts/check_fkl_tree.sh $BASE --out $O/base-64ad919 --only bench --ref-build $REF --jobs 12 --cores 8-15,24-31 > $O/base-run.log 2>&1 &
PA=$!
$FBW/scripts/check_fkl_tree.sh $BRANCH --out $O/branch-${BR:0:7} --only bench --ref-build $REF --jobs 12 --cores 0-7,16-23 > $O/branch-run.log 2>&1 &
PB=$!
wait $PA; RA=$?; say "base build ended, exit $RA"
wait $PB; RB=$?; say "branch build ended, exit $RB"
[ "$(git -C $BRANCH rev-parse HEAD)" = "$BR" ] || say "WARNING: the branch HEAD moved during the build"
for r in base-64ad919 branch-${BR:0:7}; do echo "== $r"; tail -5 $O/$r/report.txt 2>/dev/null; done
say "identity builds ended; load $(cut -d' ' -f1-3 /proc/loadavg)"
