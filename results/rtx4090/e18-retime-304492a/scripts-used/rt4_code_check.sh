#!/bin/bash
# RT-4, the code checks (COMPILE ONLY / CPU ONLY: the binaries are only read with cuobjdump and objdump; no program runs; no GPU is visible).
#   A   = scratch/pin-435/base-b10fc66/bench-build/bin: fkl-benchmarks 976e42b, FKL b10fc66 (#68's identity build; RT-1's A)
#   B   = scratch/pin-435/main-304492a/bench-build/bin: fkl-benchmarks main 73a9650, FKL 304492a (RT-1's B)
#   old = #63's B (fkl-benchmarks 2160c70, FKL b10fc66, scratch/pin-433/head-b10fc66), removed after #63; rebuilt by rebuild_old.sh in
#         its own folder from A's checkout (976e42b, whose tree outside results/ is 2160c70's): "rebuild". Its sha256 differ from #63's (the
#         module ids of nvcc's anonymous namespaces differ, and the binaries with them).
# A against old:
#   1. the configure log of the rebuild against #63's B's, from #60's record (identity-builds/head-b10fc66/bench-configure.log);
#   2. the device code of A, and of the rebuild, against what #60's record keeps of #63's B: the SASS line counts of its 17 E18 kernels that
#      differed from 24478f0 (sass_linecounts_vs_60.py: #60's own sass_identity.py functions and cuobjdump 13.3);
#   3. the device code of A against the rebuild, every kernel (pr435_sass_identity.py without the rename rule: both are FKL b10fc66);
#   4. the host code of A against the rebuild, every function, with the timed intervals (host_compare435.py --no-rename --intervals).
# B against A:
#   5. the device code (pr435_sass_identity.py with FKL #435's rename rule) and cuobjdump -res-usage of both.
# Output: scratch/rt4/out/ and scratch/rt4/code-check-console.txt. Every step's exit status is recorded; the script stops only when a tool
# cannot run.
set -u
T=/home/oscar/tfkl-bench; R=$T/scratch/rt4; O=$R/out; LOG=$R/code-check-console.txt
A=$T/scratch/pin-435/base-b10fc66/bench-build/bin; B=$T/scratch/pin-435/main-304492a/bench-build/bin; OLD=$T/scratch/pin-433/head-b10fc66
W=$T/worktrees/fkl-benchmarks-rt4; REC60=$W/results/rtx4090/fkl-pr433-pin-check
P435=$T/scratch/pr435/analysis; HOSTC=$T/scratch/pin-435/host_compare435.py
PROGS="e18_box e18_box_split e18_stencil"
[ -e $LOG ] && { echo "$LOG exists: not starting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
mkdir -p $O
say "RT-4 code checks start (CPU only); script sha256 $(sha256sum "$0" | cut -d' ' -f1)"
for f in $R/sass_linecounts_vs_60.py $P435/pr435_sass_identity.py $P435/sass_identity_norename.py $HOSTC $REC60/scripts-used/sass_identity.py $R/tools-2160c70/sass_tools.py; do
  echo "tool sha256 $(sha256sum $f)"; done
[ "$(sha256sum $P435/pr435_sass_identity.py | cut -d' ' -f1)" = c9cedd90cf9e42b9beaccd9ad1966bf26f9117eb3bd3c28cd48941cdc4a44b1e ] || { say "pr435_sass_identity.py is not the recorded one"; exit 1; }
[ "$(sha256sum $HOSTC | cut -d' ' -f1)" = 79b8317c5998fba4143a69a4d9eb6b1b1372fdf4d7275d3a79e889f7d46c175f ] || { say "host_compare435.py is not the recorded one"; exit 1; }
cmp -s $R/tools-2160c70/sass_tools.py $W/scripts/sass_tools.py && echo "scripts/sass_tools.py is the same at 2160c70 and at 8c419c7"
for b in $A $B $OLD/bench-build/bin; do for p in $PROGS; do echo "binary $(sha256sum $b/${p}_cu)"; done; done
# 1. the configure logs
say "1. the rebuild's configure log against #63's B's (#60's record)"
diff <(grep -v -E '^-- (Configuring|Generating) done' $REC60/identity-builds/head-b10fc66/bench-configure.log) \
     <(grep -v -E '^-- (Configuring|Generating) done' $OLD/bench-configure.log) > $O/configure-old-vs-rebuild.diff
echo "configure logs: exit $? ($(wc -l < $O/configure-old-vs-rebuild.diff) diff lines, the two CMake timing lines left out; $(grep -c . $OLD/bench-configure.log) lines each)"
# 2. the line counts against #60's record
say "2. SASS line counts against #60's record of #63's B (cuobjdump 13.3)"
for side in A=$A rebuild=$OLD/bench-build/bin; do n=${side%%=*}; d=${side#*=}
  PATH=/usr/local/cuda-13.3/bin:$PATH nice -n 19 python3 -I $R/sass_linecounts_vs_60.py $REC60/scripts-used/sass_identity.py \
    $REC60/sass-identity-24478f0-vs-b10fc66.txt $R/tools-2160c70 $d $PROGS > $O/linecounts-vs-60-$n.txt
  echo "$n: exit $?: $(tail -1 $O/linecounts-vs-60-$n.txt)"; done
# 3. A against the rebuild, device code
say "3. the device code of A against the rebuild (no rename)"
nice -n 19 python3 -I $P435/sass_identity_norename.py $P435/pr435_sass_identity.py --tools $W/scripts --glob 'e18_*_cu' --a A=$A \
  --b rebuild=$OLD/bench-build/bin --out $O/sass-A-vs-rebuild.txt --csv $O/sass-A-vs-rebuild.csv > /dev/null
echo "exit $?: $(sed -n 6p $O/sass-A-vs-rebuild.txt)"
# 4. A against the rebuild, host code
say "4. the host code of A against the rebuild (no rename, with the timed intervals)"
for p in $PROGS; do
  nice -n 19 python3 -I $HOSTC $A/${p}_cu $OLD/bench-build/bin/${p}_cu --tools $P435 --sass-tools $W/scripts --no-rename --intervals \
    --out $O/host-A-vs-rebuild-$p.txt --csv $O/host-A-vs-rebuild-$p.csv --diffs $O/host-A-vs-rebuild-$p-diffs.txt \
    --intervals-csv $O/host-A-vs-rebuild-$p-intervals.csv > /dev/null
  echo "$p: exit $?: $(grep -m1 -E '^functions|TOTAL|^paired' $O/host-A-vs-rebuild-$p.txt | cut -c1-200)"; done
# 5. B against A, device code and resources
say "5. the device code of B against A (FKL #435's rename rule) and the resources"
nice -n 19 python3 -I $P435/pr435_sass_identity.py --tools $W/scripts --glob 'e18_*_cu' --a b10fc66=$A --b 304492a=$B \
  --out $O/sass-A-vs-B.txt --csv $O/sass-A-vs-B.csv > /dev/null
echo "exit $?: $(sed -n 6p $O/sass-A-vs-B.txt)"
for side in A=$A B=$B; do n=${side%%=*}; d=${side#*=}
  for p in $PROGS; do echo "== ${p}_cu"; cuobjdump -arch sm_89 -res-usage $d/${p}_cu; done > $O/res-usage-$n.txt 2>&1; done
say "RT-4 code checks end"
