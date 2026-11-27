#!/bin/bash
# RT-2b, the code checks of the E16 query tile (CPU ONLY: the binaries are only read with cuobjdump and objdump; no program runs; no GPU is
# visible).
#   old = the record's binary: fkl-benchmarks/build/rtx4090/bin/e16_query_tile_cu (sha256 763e7e3c..., results/rtx4090/e16-query-tile/
#         binary-sha256.txt; built 2026-11-04 12:06 from 35ebfd8-dirty with FKL c8fcbe8, preset rtx4090; only read here)
#   A   = scratch/pin-435/base-b10fc66/bench-build/bin/e16_query_tile_cu (fkl-benchmarks 976e42b, FKL b10fc66; RT-1's, RT-2's and RT-4's A)
#   B   = scratch/pin-435/main-304492a/bench-build/bin/e16_query_tile_cu (main 73a9650, FKL 304492a; RT-1's B)
# A against old:
#   1. the device code, every kernel: #60's sass_identity.py, which pairs the kernels by demangled name with FKL #432's defaulted
#      ',fk::BinaryType' removed (c8fcbe8 is before #432);
#   2. the host code, every function, with the timed intervals: host_compare435.py --no-rename --drop-binarytype-default --intervals
#      --path-delta 24 (A's source root, scratch/pin-435/base-b10fc66/bench-src/, is 24 characters longer than old's,
#      /home/oscar/tfkl-bench/fkl-benchmarks/; computed and printed before use).
# B against A:
#   3. the device code with FKL #435's rename rule (pr435_sass_identity.py), and cuobjdump -res-usage of all three;
#   4. the host code with the timed intervals (host_compare435.py, its rename rule; the coordinator: the timed host code changed with #435).
set -u
T=/home/oscar/tfkl-bench; R=$T/scratch/rt2b; O=$R/out; LOG=$R/code-check-console.txt
OLDD=$T/fkl-benchmarks/build/rtx4090/bin; A=$T/scratch/pin-435/base-b10fc66/bench-build/bin; B=$T/scratch/pin-435/main-304492a/bench-build/bin
W=$T/worktrees/fkl-benchmarks-rt2b; REC60=$W/results/rtx4090/fkl-pr433-pin-check; P435=$T/scratch/pr435/analysis; HOSTC=$T/scratch/pin-435/host_compare435.py
P=e16_query_tile_cu
[ -e $LOG ] && { echo "$LOG exists: not starting"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
mkdir -p $O
say "RT-2b code checks start (CPU only); script sha256 $(sha256sum "$0" | cut -d' ' -f1)"
for f in $REC60/scripts-used/sass_identity.py $P435/pr435_sass_identity.py $HOSTC $W/scripts/sass_tools.py; do echo "tool sha256 $(sha256sum $f)"; done
[ "$(sha256sum $OLDD/$P | cut -d' ' -f1)" = 763e7e3c83e80078ef9a1f213c50466159675ddec0aca430dc0446ad02032d79 ] || { say "old is not the record's binary"; exit 1; }
for b in $OLDD $A $B; do echo "binary $(sha256sum $b/$P); commits $(strings $b/$P | grep -E '^[0-9a-f]{40}(-dirty)?$|^unknown$' | tr '\n' ' ')"; done
say "1. the device code of old against A (#60's sass_identity.py, BinaryType default dropped)"
nice -n 19 python3 -I $REC60/scripts-used/sass_identity.py $OLDD $A --tools $W/scripts --arch sm_89 --glob $P --out $O/sass-old-vs-A.txt > /dev/null
echo "exit $?: $(grep -m1 '^TOTAL' $O/sass-old-vs-A.txt)"
say "2. the host code of old against A (no rename, BinaryType default dropped, path delta 24, the timed intervals)"
nice -n 19 python3 -I $HOSTC $OLDD/$P $A/$P --tools $P435 --sass-tools $W/scripts --no-rename --drop-binarytype-default --path-delta 24 --intervals \
  --out $O/host-old-vs-A.txt --csv $O/host-old-vs-A.csv --diffs $O/host-old-vs-A-diffs.txt --intervals-csv $O/host-old-vs-A-intervals.csv > /dev/null
echo "exit $?: $(grep -m1 '^TOTAL' $O/host-old-vs-A.txt | cut -c1-220)"
say "3. the device code of B against A (FKL #435's rename rule) and the resources"
nice -n 19 python3 -I $P435/pr435_sass_identity.py --tools $W/scripts --glob $P --a b10fc66=$A --b 304492a=$B --out $O/sass-A-vs-B.txt --csv $O/sass-A-vs-B.csv > /dev/null
echo "exit $?: $(sed -n 6p $O/sass-A-vs-B.txt)"
for side in old=$OLDD A=$A B=$B; do n=${side%%=*}; d=${side#*=}; cuobjdump -arch sm_89 -res-usage $d/$P > $O/res-usage-$n.txt 2>&1; done
say "4. the host code of B against A (#435's rename rule, the timed intervals)"
nice -n 19 python3 -I $HOSTC $A/$P $B/$P --tools $P435 --sass-tools $W/scripts --intervals \
  --out $O/host-A-vs-B.txt --csv $O/host-A-vs-B.csv --diffs $O/host-A-vs-B-diffs.txt --intervals-csv $O/host-A-vs-B-intervals.csv > /dev/null
echo "exit $?: $(grep -m1 '^TOTAL' $O/host-A-vs-B.txt | cut -c1-220)"
say "RT-2b code checks end"
