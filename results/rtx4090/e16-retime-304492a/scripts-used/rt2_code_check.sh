#!/bin/bash
# RT-2, the code checks of E16 (CPU ONLY: the binaries are only read with cuobjdump and objdump; no program runs; no GPU is visible).
#   old = window D's cleaned e16_attention_cu: worktrees/fkl-benchmarks-window-d/build/local-rtx4090/bin/e16_attention_cu (main f27c605,
#         FKL 64ad9195; sha256 268c78ad..., results/rtx4090/cleanup-window-d/binary-sha256-e16.txt), the build behind the paper's
#         data/e16-attention-rtx4090-cleanup (cleanup-window-d/cleaned/e16-summary/run1..3)
#   A   = scratch/pin-435/base-b10fc66/bench-build/bin/e16_attention_cu (fkl-benchmarks 976e42b, FKL b10fc66; RT-1's and RT-4's A)
#   B   = scratch/pin-435/main-304492a/bench-build/bin/e16_attention_cu (main 73a9650, FKL 304492a; RT-1's B)
# A against old:
#   1. the device code, every kernel: #60's sass_identity.py (results/rtx4090/fkl-pr433-pin-check/scripts-used/), which pairs the kernels
#      by demangled name with FKL #432's defaulted ',fk::BinaryType' of fk::Sub/Mul/Div removed (FKL 64ad9195 is before #432);
#   2. the host code, every function, with the timed intervals: host_compare435.py --no-rename --drop-binarytype-default --intervals
#      --path-delta 5 (A's source path, scratch/pin-435/base-b10fc66/bench-src/, is 5 characters longer than old's,
#      worktrees/fkl-benchmarks-window-d/).
# B against A:
#   3. the device code with FKL #435's rename rule (pr435_sass_identity.py), and cuobjdump -res-usage of both.
set -u
T=/home/oscar/tfkl-bench; R=$T/scratch/rt2; O=$R/out-v2; LOG=$R/code-check-console-v2.txt
OLDD=$T/worktrees/fkl-benchmarks-window-d/build/local-rtx4090/bin; A=$T/scratch/pin-435/base-b10fc66/bench-build/bin; B=$T/scratch/pin-435/main-304492a/bench-build/bin
W=$T/worktrees/fkl-benchmarks-rt2; REC60=$W/results/rtx4090/fkl-pr433-pin-check; P435=$T/scratch/pr435/analysis; HOSTC=$T/scratch/pin-435/host_compare435.py
P=e16_attention_cu
[ -e $LOG ] && { echo "$LOG exists: not starting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
mkdir -p $O
say "RT-2 code checks start (CPU only); script sha256 $(sha256sum "$0" | cut -d' ' -f1)"
for f in $REC60/scripts-used/sass_identity.py $P435/pr435_sass_identity.py $HOSTC $W/scripts/sass_tools.py; do echo "tool sha256 $(sha256sum $f)"; done
[ "$(sha256sum $OLDD/$P | cut -d' ' -f1)" = 268c78adf599b94b519c88cb49f91031ca7de486a21358d59600e6062ab18ccc ] || { say "old is not window D's cleaned binary"; exit 1; }
for b in $OLDD $A $B; do echo "binary $(sha256sum $b/$P); commits $(strings $b/$P | grep -E '^[0-9a-f]{40}(-dirty)?$|^unknown$' | tr '\n' ' ')"; done
say "1. the device code of old against A (#60's sass_identity.py, BinaryType default dropped)"
nice -n 19 python3 -I $REC60/scripts-used/sass_identity.py $OLDD $A --tools $W/scripts --arch sm_89 --glob $P --out $O/sass-old-vs-A.txt > /dev/null
echo "exit $?: $(grep -m1 '^TOTAL' $O/sass-old-vs-A.txt)"
say "2. the host code of old against A (no rename, BinaryType default dropped, path delta 5, the timed intervals)"
nice -n 19 python3 -I $HOSTC $OLDD/$P $A/$P --tools $P435 --sass-tools $W/scripts --no-rename --drop-binarytype-default --path-delta 5 --intervals \
  --out $O/host-old-vs-A.txt --csv $O/host-old-vs-A.csv --diffs $O/host-old-vs-A-diffs.txt --intervals-csv $O/host-old-vs-A-intervals.csv > /dev/null
echo "exit $?: $(grep -m1 '^TOTAL' $O/host-old-vs-A.txt | cut -c1-220)"
say "3. the device code of B against A (FKL #435's rename rule) and the resources"
nice -n 19 python3 -I $P435/pr435_sass_identity.py --tools $W/scripts --glob $P --a b10fc66=$A --b 304492a=$B --out $O/sass-A-vs-B.txt --csv $O/sass-A-vs-B.csv > /dev/null
echo "exit $?: $(sed -n 6p $O/sass-A-vs-B.txt)"
for side in old=$OLDD A=$A B=$B; do n=${side%%=*}; d=${side#*=}; cuobjdump -arch sm_89 -res-usage $d/$P > $O/res-usage-$n.txt 2>&1; done
say "RT-2 code checks end"
