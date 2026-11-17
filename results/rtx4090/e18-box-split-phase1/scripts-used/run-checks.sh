#!/bin/bash
# E18 box split, phase 1: the one correctness run of e18_box_split_cu, checks only (E18_NO_TIMING=1: nothing is timed, no profiler). The
# coordinator's grant of 2026-11-17 (option 1): one run, after the compile. Output in scratch/e18-box-split/checks/.
set -u
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e18-box-split; BIN=$W/build/local-rtx4090/bin/e18_box_split_cu; O=$T/scratch/e18-box-split/checks
source $T/scratch/blocked_programs.sh
refuse_blocked_binaries "$BIN" || exit 1
source $T/toolchains/env-cuda13.4.sh >/dev/null 2>&1
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1
export TMPDIR=$T/scratch/e18-box-split/tmp
[ -e "$O" ] && { echo "$O exists: this run happens once; stop"; exit 1; }
mkdir -p "$O"; cd "$O" || exit 1
nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid --format=csv > gpu-compute-apps-before.csv 2>&1
echo "# $(date -Iseconds) checks run: sha256 $(sha256sum $BIN | cut -c1-16) fkl-benchmarks $(git -C $W rev-parse --short HEAD)$(git -C $W diff --quiet HEAD || echo -dirty) CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES CUDA_DEVICE_ORDER=$CUDA_DEVICE_ORDER" | tee console-header.txt
E18_NO_TIMING=1 "$BIN" > stdout.txt 2> stderr.txt
echo $? > exit_code.txt
echo "# exit $(cat exit_code.txt); $(date -Iseconds)" | tee -a console-header.txt
