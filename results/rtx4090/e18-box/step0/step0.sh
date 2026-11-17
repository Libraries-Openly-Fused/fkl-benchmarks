#!/bin/bash
# E18 box phase 2, step 0 (COMPILE ONLY, no program runs): e18_box_cu at the branch head, preset local-rtx4090, in the worktree's own build.
# Records the submodules, the build log, the commit the binary carries and its sha256 (the window script checks both before it starts).
set -u
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e18-box; O=$T/scratch/e18-box-step0; BIN=$W/build/local-rtx4090/bin/e18_box_cu
source $T/toolchains/env-cuda13.4.sh >/dev/null 2>&1
export TMPDIR=$T/scratch/e18-box/tmp
HEADC=$(git -C $W rev-parse HEAD)
exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) step 0 (compile only): e18_box_cu at $HEADC"
[ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || { echo "the worktree has local changes: stop"; exit 1; }
git -C $W submodule status --recursive
cmake --build $W/build/local-rtx4090 --target e18_box_cu > $O/build.log 2>&1; rc=$?
echo "build exit $rc"
[ $rc = 0 ] || exit 1
strings $BIN | grep -q -x $HEADC && echo "binary carries $HEADC" || { echo "the binary does not carry $HEADC: stop"; exit 1; }
strings $BIN | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "a -dirty commit: stop"; exit 1; }
sha256sum $BIN > $O/binary-sha256.txt
echo "warnings: $(grep -c warning $O/build.log)"
echo "# $(date -Iseconds) step 0 ended: all checks passed"
