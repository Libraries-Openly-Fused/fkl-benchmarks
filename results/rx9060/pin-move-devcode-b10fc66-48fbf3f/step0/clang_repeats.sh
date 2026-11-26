#!/usr/bin/env bash
# clang_repeats.sh <out dir>: clang++-23 at the 8 MiB stack limit, at FKL c8fcbe8 and at 48fbf3f with the same compiler and environment, to
# separate the FKL change from the compiler change (23.1.5 in ../e3-compile-limits/, 23.1.6 now): N = 1,516 and 1,517 with address
# randomisation off, then 5 repeats of N = 1,515 and 1,516 with it on (../e3-compile-limits/: 5 of 5 and 3 of 5 compile). Compile only.
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; FKLB=$NS/wt-pin435
export TMPDIR=$NS/sys
ulimit -S -s 8192
L="python3 -I $FKLB/scripts/e3_clang_limits.py"
for tree in fkl-c8fcbe8 fkl-48fbf3f; do
    O="--out $OUT --fkl-include $NS/$tree/include"
    $L run --backend clang --ns 1516,1517 --no-aslr --series repeat-noaslr-$tree $O
    $L run --backend clang --ns 1515,1516 --reps 5 --series repeat-aslr-$tree $O
done
echo "clang repeats done"
