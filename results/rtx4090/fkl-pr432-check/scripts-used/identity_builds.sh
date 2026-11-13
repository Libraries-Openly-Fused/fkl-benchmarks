#!/bin/bash
# Device-code identity check of FKL #432 (the HIP versions of the 7 image-processing DPPs), step 1 (COMPILE ONLY, no program runs): the bench
# group of scripts/check_fkl_tree.sh at fkl-benchmarks 8c1fa98 (every *_cu and *_cpp target, e18_stencil_cu included), built twice at the same time
# on separate cores:
#   base = FKL 1da77899 (the pin), the external/fkl of the worktree worktrees/fkl-benchmarks-pin-432 (at its gitlink)
#   head = FKL #432 at 2c42a80, the detached worktree worktrees/fkl-pr432-2c42a80
# The FKLB_* options, OPENCV_DIR and nvcc come from the CMakeCache of the reference build build/local-rtx4090, which is only read. Output:
# scratch/pin-432/{base-1da7789,head-2c42a80}/ and scratch/pin-432/console.txt.
set -u
T=/home/oscar/tfkl-bench; FBW=$T/worktrees/fkl-benchmarks-pin-432; REF=$T/fkl-benchmarks/build/local-rtx4090
BASE=$FBW/external/fkl; HEAD=$T/worktrees/fkl-pr432-2c42a80; O=$T/scratch/pin-432
mkdir -p $O
exec > >(tee -a $O/console.txt) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
[ "$(git -C $FBW rev-parse HEAD)" = 8c1fa98311268277ec6e81d7b91439a031239bbe ] || { say "fkl-benchmarks worktree is not at 8c1fa98: stop"; exit 1; }
[ "$(git -C $BASE rev-parse HEAD)" = 1da778992756e0fa08ec289225fe18dca134412c ] || { say "base tree is not at 1da77899: stop"; exit 1; }
[ "$(git -C $HEAD rev-parse HEAD)" = 2c42a80e12168f4183ba835cdde353cc1e33be07 ] || { say "head tree is not at 2c42a80: stop"; exit 1; }
for d in $BASE $HEAD $FBW; do [ -z "$(git -C $d status --porcelain --untracked-files=no)" ] || { say "$d has local changes: stop"; exit 1; }; done
say "identity builds start: fkl-benchmarks $(git -C $FBW rev-parse --short HEAD), cvGS $(git -C $FBW/external/cvGPUSpeedup rev-parse --short HEAD), FastNPP $(git -C $FBW/external/FastNPP rev-parse --short HEAD); base $(git -C $BASE rev-parse --short HEAD), head $(git -C $HEAD rev-parse --short HEAD); load $(cut -d' ' -f1-3 /proc/loadavg)"
$FBW/scripts/check_fkl_tree.sh $BASE --out $O/base-1da7789 --only bench --ref-build $REF --jobs 12 --cores 8-15,24-31 > $O/base-run.log 2>&1 &
PA=$!
$FBW/scripts/check_fkl_tree.sh $HEAD --out $O/head-2c42a80 --only bench --ref-build $REF --jobs 12 --cores 0-7,16-23 > $O/head-run.log 2>&1 &
PB=$!
wait $PA; RA=$?; say "base build ended, exit $RA"
wait $PB; RB=$?; say "head build ended, exit $RB"
[ "$(git -C $BASE rev-parse HEAD)" = 1da778992756e0fa08ec289225fe18dca134412c ] || say "WARNING: the base tree moved during the build"
[ "$(git -C $HEAD rev-parse HEAD)" = 2c42a80e12168f4183ba835cdde353cc1e33be07 ] || say "WARNING: the head tree moved during the build"
for r in base-1da7789 head-2c42a80; do echo "== $r"; tail -n 5 $O/$r/report.txt 2>/dev/null; done
say "identity builds ended; load $(cut -d' ' -f1-3 /proc/loadavg)"
