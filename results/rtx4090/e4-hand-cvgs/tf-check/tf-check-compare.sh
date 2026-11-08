#!/bin/bash
# Second step of the thread-fusion check: scripts/sass_compare_builds.py takes only executable files, so the two objects of tf-check.sh (nvcc -c output) were
# skipped by its two comparisons. Here they are marked executable and the two comparisons are run again; nothing is rebuilt.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e4-series; O=$T/scratch/tf-check/out
exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) comparisons again (tf-check-compare.sh): the objects marked executable"
cp "$0" $O/
chmod +x $O/default/tfcheck_cu $O/tf-false/tfcheck_cu
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/default $O/tf-false --arch sm_89 --glob 'tfcheck_cu' --match-demangled --ignore '\(fk::TF\)[01]' --out $O/sass-default-vs-tf-false.txt | grep '^kernels identical')
echo "default vs <false>, without the TF argument: $r"
r2=$(python3 $S $O/e4 $O/default --arch sm_89 --glob 'tfcheck_cu' --match-demangled --out $O/sass-e4-vs-default.txt | grep '^kernels identical')
echo "e4_preprocessing_cu vs default: $r2"
echo "# $(date -Iseconds) comparisons ended"
