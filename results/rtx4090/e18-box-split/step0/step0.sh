#!/bin/bash
# E18 box split phase 2, step 0 (COMPILE ONLY, no program runs): e18_box_split_cu at the approved head, preset local-rtx4090, in the worktree's
# own build. Records the submodules, the build log, the commit the binary carries and its sha256 (the window script checks both before it
# starts), and compares the binary's SASS with the dump of the phase-1 binary (62202d1).
set -u
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e18-box-split; O=$T/scratch/e18-box-split-step0; BIN=$W/build/local-rtx4090/bin/e18_box_split_cu
EXPECT=417a106bcbe1627f9894fef83a6bc7a7e1f88b86
source $T/toolchains/env-cuda13.4.sh >/dev/null 2>&1
export TMPDIR=$T/scratch/e18-box-split/tmp
HEADC=$(git -C $W rev-parse HEAD)
exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) step 0 (compile only): e18_box_split_cu at $HEADC"
[ "$HEADC" = "$EXPECT" ] || { echo "the head is not the approved $EXPECT: stop"; exit 1; }
[ "$(git -C $W rev-parse origin/e18-box-split)" = "$EXPECT" ] || { echo "origin/e18-box-split is not $EXPECT: stop"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || { echo "the worktree has local changes: stop"; exit 1; }
git -C $W submodule status --recursive
cmake --build $W/build/local-rtx4090 --target e18_box_split_cu > $O/build.log 2>&1; rc=$?
echo "build exit $rc"
[ $rc = 0 ] || exit 1
strings $BIN | grep -q -x $HEADC && echo "binary carries $HEADC" || { echo "the binary does not carry $HEADC: stop"; exit 1; }
strings $BIN | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "a -dirty commit: stop"; exit 1; }
sha256sum $BIN > $O/binary-sha256.txt
echo "warnings: $(grep -c warning $O/build.log)"
mkdir -p $O/census
cuobjdump -arch sm_89 -sass $BIN > $O/census/e18_box_split_cu.sass
python3 - $W/scripts $O/census/e18_box_split_cu.sass $T/scratch/e18-box-split/census-62202d1/e18_box_split_cu.sass <<'PY' || { echo "the SASS differs from the phase-1 binary's: stop"; exit 1; }
import sys
sys.path.insert(0, sys.argv[1])
import sass_tools as st
a = st.split_functions(open(sys.argv[2]).read()); b = st.split_functions(open(sys.argv[3]).read())
same = sorted(a) == sorted(b) and all(st.normalise_sass(a[k], rename=False) == st.normalise_sass(b[k], rename=False) for k in a)
print(f"SASS: {len(a)} kernels, {'every kernel identical to' if same else 'DIFFERENT from'} the phase-1 binary (62202d1)")
sys.exit(0 if same else 1)
PY
cat $O/binary-sha256.txt
echo "# $(date -Iseconds) step 0 ended: all checks passed"
