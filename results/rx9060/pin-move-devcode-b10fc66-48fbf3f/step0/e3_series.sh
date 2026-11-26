#!/usr/bin/env bash
# e3_series.sh <out dir> <FKL include dir>: VI-L at a tree, amdclang++ (HIP, gfx1200) only, with scripts/e3_clang_limits.py of fkl-benchmarks
# (pin-fkl-435): the commands of results/rx9060/e3-compile-limits/run_series.sh sections 2 and 3 for the hip back-end (bisection of the
# 8 MiB stack limit with address randomisation off, then 5 repeats of N-1, N, N+1 with it on; N = 2,046 and 2,047 with a 16 MiB stack),
# with the bisection between 1,300 and 1,450 instead of 1,100 and 2,046 (the boundary at FKL c8fcbe8 is 1,367 / 1,368), and N = 2,047 with
# -fbracket-depth = N + 2. Compile only: e3_clang_limits.py's run and bisect modes never execute a program.
set -u
OUT=$1; INC=$2
FKLB=/home/cudeiro/rocm-fkl-benchmarks/tmp/wt-pin435
export ROCM_PATH=/opt/rocm/core-10.1 HIP_PATH=/opt/rocm/core-10.1 LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys
ulimit -S -s 8192   # the soft limit only: e3_clang_limits.py raises it for --stack-kb (bash's plain ulimit -s also lowers the hard limit)
L="python3 -I $FKLB/scripts/e3_clang_limits.py"; O="--out $OUT --fkl-include $INC"
t=$OUT/bisect.txt
$L bisect --backend hip --lo 1300 --hi 1450 --no-aslr --series stack-bisect-noaslr $O | tee "$t"
hi=$(tail -1 "$t" | sed -E 's/.* not ([0-9]+)$/\1/')
echo "first failing N (no ASLR): $hi"
$L run --backend hip --ns $((hi - 1)),$hi,$((hi + 1)) --reps 5 --series stack-repeats-aslr $O
$L run --backend hip --ns 2046,2047 --stack-kb 16384 --series stack-16MiB $O
$L run --backend hip --ns 2047 --stack-kb 16384 --bracket-depth auto --series bracket-depth $O
echo "e3 series done"
