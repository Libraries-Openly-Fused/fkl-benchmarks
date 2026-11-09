#!/bin/bash
# Compile check of the pin branch (COMPILE ONLY, no program runs): the bench group of scripts/check_fkl_tree.sh at the branch head,
# with external/fkl of the branch worktree (1da77899) as the FKL tree and the options of the reference build of preset local-rtx4090
# (only read). Output: scratch/pin-1da7789/check/ and console.txt.
set -u
W=/home/oscar/tfkl-bench/worktrees/fkl-benchmarks-pin-1da7789; REF=/home/oscar/tfkl-bench/fkl-benchmarks/build/local-rtx4090
O=/home/oscar/tfkl-bench/scratch/pin-1da7789
exec > >(tee -a $O/console.txt) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
[ "$(git -C $W/external/fkl rev-parse HEAD)" = 1da778992756e0fa08ec289225fe18dca134412c ] || { say "external/fkl is not at 1da77899: stop"; exit 1; }
[ "$(git -C $W ls-tree HEAD external/fkl | awk '{print $3}')" = 1da778992756e0fa08ec289225fe18dca134412c ] || { say "the gitlink of HEAD is not 1da77899: stop"; exit 1; }
[ -z "$(git -C $W status --porcelain)" ] || { say "the worktree has local changes: stop"; exit 1; }
H=$(git -C $W rev-parse HEAD)
say "compile check start: fkl-benchmarks ${H:0:7}, $(git -C $W submodule status --recursive | tr '\n' ';'); load $(cut -d' ' -f1-3 /proc/loadavg)"
$W/scripts/check_fkl_tree.sh $W/external/fkl --out $O/check --only bench --ref-build $REF
RC=$?
[ "$(git -C $W rev-parse HEAD)" = "$H" ] || say "WARNING: the branch HEAD moved during the build"
say "compile check ended, exit $RC; load $(cut -d' ' -f1-3 /proc/loadavg)"
exit $RC
