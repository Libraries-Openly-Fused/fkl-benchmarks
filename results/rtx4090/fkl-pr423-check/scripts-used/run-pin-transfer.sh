#!/usr/bin/env bash
# SASS comparison of the pin-transfer binaries built against the head of FKL PR #423 (9e20bd4e) with the frozen binaries that produced the numbers. Compile only. New work dir: the worktrees link external/fkl to the head.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
cd /home/oscar/tfkl-bench/fkl-benchmarks
echo "nvcc: $(nvcc --version | sed -n 4p)  start $(date -Iseconds)  fkl tree: $(git -C /home/oscar/tfkl-bench/scratch/pr423/check-fast/fkl-overlay log --oneline -1)"
export ONLY="a334424 269d229 5aac244 1a70cab a266bc9 e1 mul4-new e2-new"
export PIN_CORES=4-7,20-23
exec scripts/pin_transfer.sh /home/oscar/tfkl-bench/scratch/pr423/check-fast/fkl-overlay /home/oscar/tfkl-bench/scratch/pr423/pin-transfer-work /home/oscar/tfkl-bench/scratch/pr423/pin-transfer-out /home/oscar/tfkl-bench/worktrees/fkl-pin-38c4f32
