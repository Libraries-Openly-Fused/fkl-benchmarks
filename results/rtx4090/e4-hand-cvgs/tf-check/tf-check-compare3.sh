#!/bin/bash
# Fourth step of the thread-fusion check: sass_compare_builds.py demangles with c++filt, which writes the thread-fusion flag of fk::TransformDPPDetails_ as
# true/false (not (bool)1/0), so the pairing ignores '(fk::TF)0/1,' and that flag with the regexes of the E9 comparison. Nothing is rebuilt.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; W=$T/worktrees/fkl-benchmarks-e4-series; O=$T/scratch/tf-check/out
exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) default vs <false> again (tf-check-compare3.sh): the E9 regexes for the two TF arguments"
cp "$0" $O/
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/default $O/tf-false --arch sm_89 --glob 'tfcheck_cu' --match-demangled --ignore '\(fk::TF\)[01],' --ignore '(?<=TransformDPPDetails_<void,)(?:true|false),' \
    --out $O/sass-default-vs-tf-false.txt | grep '^kernels identical')
echo "default vs <false>, without the two TF arguments: $r"
echo "# $(date -Iseconds) comparison ended"
