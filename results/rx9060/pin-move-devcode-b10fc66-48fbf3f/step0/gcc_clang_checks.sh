#!/usr/bin/env bash
# gcc_clang_checks.sh <out dir>: the g++ 13 and clang++ 23 columns of VI-P (E11) and VI-L (E3) at FKL 48fbf3f, compile only, with the
# scripts of fkl-benchmarks a4b493c (pin-fkl-435). Both compilers build FKL's CPU back-end; nothing is run.
# - E11: scripts/e11_diagnostics.py with g++-13 and clang++-23, at c8fcbe8 (no chain checks), b10fc66 (the pin) and 48fbf3f (the head).
# - E3, clang++-23: bisection of the 8 MiB stack limit (address randomisation off) between 1,450 and 1,600 (1,516 / 1,517 at c8fcbe8), then
#   5 repeats of N-1, N, N+1 with it on; N = 2,046 and 2,047 with a 16 MiB stack; N = 2,047 with -fbracket-depth = N + 2.
# - E3, g++-13: N = 2,047, 2,100 and 4,000 with the defaults (no nesting limit at c8fcbe8). The run and bisect modes never execute a program.
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; FKLB=$NS/wt-pin435
export TMPDIR=$NS/sys
ulimit -S -s 8192   # the soft limit only: e3_clang_limits.py raises it for --stack-kb
# no address-space limit: a compile stopped by it would crash like a stack overflow (the largest compile here, g++ at N = 4,000, used
# 6.5 GiB of 60 at c8fcbe8)
mkdir -p $OUT/e11 $OUT/e3
for spec in unchanged-c8fcbe8:fkl-c8fcbe8 pin-b10fc66:fkl-b10fc66 head-48fbf3f:fkl-48fbf3f; do
    label=${spec%%:*}; tree=${spec##*:}; mkdir -p $OUT/e11/$label
    python3 -I $FKLB/scripts/e11_diagnostics.py --compilers g++-13,clang++-23 --label $label --include $NS/$tree/include --out $OUT/e11/$label \
        > $OUT/e11/$label.log 2>&1; echo "e11 $label exit $?"
done
L="python3 -I $FKLB/scripts/e3_clang_limits.py"; O="--out $OUT/e3 --fkl-include $NS/fkl-48fbf3f/include"
t=$OUT/e3/bisect-clang.txt
$L bisect --backend clang --lo 1450 --hi 1600 --no-aslr --series stack-bisect-noaslr $O | tee "$t"
hi=$(tail -1 "$t" | sed -E 's/.* not ([0-9]+)$/\1/')
echo "clang first failing N (no ASLR): $hi"
$L run --backend clang --ns $((hi - 1)),$hi,$((hi + 1)) --reps 5 --series stack-repeats-aslr $O
$L run --backend clang --ns 2046,2047 --stack-kb 16384 --series stack-16MiB $O
$L run --backend clang --ns 2047 --stack-kb 16384 --bracket-depth auto --series bracket-depth $O
$L run --backend gcc --ns 2047,2100,4000 --series default $O
echo "gcc clang checks done"
