#!/bin/bash
# E18 box, phase 1: the nsys pass of e18_box_cu in its profile mode (E18_PROFILE=1: a warm-up call, then E18_PROFILE_CALLS=10 calls per
# implementation in its NVTX range; nothing compared, nothing timed). CUDA API and NVTX trace. Output in scratch/e18-box/profile/.
set -u
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e18-box; BIN=$W/build/local-rtx4090/bin/e18_box_cu; O=$T/scratch/e18-box/profile
NSYS=$T/toolchains/cuda-13.4.3/bin/nsys
source $T/scratch/blocked_programs.sh
refuse_blocked_binaries "$BIN" || exit 1
source $T/toolchains/env-cuda13.4.sh >/dev/null 2>&1
export TMPDIR=$T/scratch/e18-box/tmp
mkdir -p "$O"; cd "$O" || exit 1
[ -e e18-box-profile.nsys-rep ] && { echo "e18-box-profile.nsys-rep exists: stop"; exit 1; }
echo "# $(date -Iseconds) profile run: $(sha256sum $BIN | cut -c1-16) fkl-benchmarks $(git -C $W rev-parse --short HEAD)$(git -C $W diff --quiet HEAD || echo -dirty)"
E18_PROFILE=1 E18_PROFILE_CALLS=10 $NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false -o e18-box-profile "$BIN" > profile-console.txt 2>&1
echo $? > exit_code.txt
$NSYS export --type sqlite --force-overwrite=true -o e18-box-profile.sqlite e18-box-profile.nsys-rep > export-console.txt 2>&1
python3 $W/results/rtx4090/e18-box-phase1/profile_census.py e18-box-profile.sqlite --calls 10 --out profile-census.txt > /dev/null
echo "# exit $(cat exit_code.txt); $(date -Iseconds)"
