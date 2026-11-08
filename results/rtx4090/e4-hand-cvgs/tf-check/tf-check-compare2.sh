#!/bin/bash
# Third step of the thread-fusion check: the demangled names of the two builds differ in two places, (fk::TF)1/0 and the thread-fusion flag of
# fk::TransformDPPDetails_<void,(bool)1/0,...>, so the pairing ignores both (as the E9 comparison did). Nothing is rebuilt.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e4-series; O=$T/scratch/tf-check/out
exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) default vs <false> again (tf-check-compare2.sh): both TF arguments ignored in the pairing"
cp "$0" $O/
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/default $O/tf-false --arch sm_89 --glob 'tfcheck_cu' --match-demangled --ignore '\(fk::TF\)[01]' --ignore '(?<=TransformDPPDetails_<void,)\(bool\)[01]' \
    --out $O/sass-default-vs-tf-false.txt | grep '^kernels identical')
echo "default vs <false>, without the two TF arguments: $r"
echo "# $(date -Iseconds) comparison ended"
