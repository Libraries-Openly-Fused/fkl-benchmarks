#!/bin/bash
# E18 box, phase 1: the correctness run of e18_box_cu, checks only (E18_NO_TIMING=1: nothing is timed). Output in scratch/e18-box/checks/.
set -u
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e18-box; BIN=$W/build/local-rtx4090/bin/e18_box_cu; O=$T/scratch/e18-box/checks
source $T/scratch/blocked_programs.sh
refuse_blocked_binaries "$BIN" || exit 1
source $T/toolchains/env-cuda13.4.sh >/dev/null 2>&1
export TMPDIR=$T/scratch/e18-box/tmp
mkdir -p "$O"; cd "$O" || exit 1
echo "# $(date -Iseconds) checks run: $(sha256sum $BIN | cut -c1-16) fkl-benchmarks $(git -C $W rev-parse --short HEAD)$(git -C $W diff --quiet HEAD || echo -dirty) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" | tee console-header.txt
E18_NO_TIMING=1 "$BIN" > stdout.txt 2> stderr.txt
echo $? > exit_code.txt
echo "# exit $(cat exit_code.txt); $(date -Iseconds)"
