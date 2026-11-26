#!/bin/bash
# GPU window "fkl-pr435-check-d1cff06" (FKL's tests and the attention harness only; nothing is timed; about a minute; the desktop session keeps
# running): FKL PR #435's new head d1cff06 (48fbf3f + fix A, which changes only flash_attention_mma.h) on the RTX 4090 (sm_89). Built in
# C-NV-A's steps 1 and 1b (compile only); every binary is listed with its sha256 in window-binaries.sha256, whose own sha256 is MANIFEST_SHA256
# below. Steps, in this order; the window stops at the first failure:
#   1. FKL's CUDA tests at d1cff06: the 59 ctest tests named *_cu (the same 59 names as at 48fbf3f and b10fc66), ctest -j1 --stop-on-failure
#      --timeout 300 in scratch/pr435/fkl-head-build, rebuilt at d1cff06 (only utest_flash_attention_mma_cu recompiled). Every test must pass.
#   2. The attention harness at d1cff06: pr435_attention (scratch/pr435/harness-build-d1cff06/head-d1cff06), into
#      scratch/pr435/harness-out-d1cff06/head-d1cff06. It must exit 0 with an empty standard error and print its PASS line.
#   No nsys, no ncu, no clock warm-up. SAMPLER: nvidia-smi every 100 ms through the window (gpu_clocks.csv), on core 14. The tests and the
#   harness are pinned to cores 0-3,16-19. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (the RTX 4090).
# Guards before anything starts: no blocked program (the binaries and the ctest list); every binary's sha256 is the manifest's; the ctest list
# has exactly the 59 tests, the names of the C-NV window; the FKL head worktree is clean at d1cff06 (thread-pool 27e733f) and the build is
# configured from it; the 48fbf3f attention outputs of the C-NV window are still as that window recorded them (its harness-outputs-sha256.txt);
# the record's worktree is at main bd019f6 or a descendant that differs from it only under results/rtx4090/fkl-pr435-check/, with no local
# change outside results/; this script and compare_attention_bitwise.py are the versions committed at HEAD and the script runs from its place
# there; the output folder is the requested one and new, and so is the harness output folder; no measurement, build or compile job runs and no
# other program uses the RTX 4090. RTX 4090 = nvidia-smi index 1. Nothing is overwritten.
# After the GPU steps, CPU only: harness/source/compare_attention_bitwise.py compares the 32 attention outputs at d1cff06 with the C-NV
# window's at 48fbf3f, bit for bit, and the outputs' sha256 are listed. The harness outputs stay in the scratch folder, not committed.
# Output: results/rtx4090/fkl-pr435-check/d1cff06/window/ of the record's worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-pr435-d1cff06
S=$T/scratch/pr435; source $W/scripts/os_kernel_lines.sh
REC=$W/results/rtx4090/fkl-pr435-check; SUB=$REC/d1cff06
OUT=$SUB/window; HOUT=$S/harness-out-d1cff06; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/fkl-pr435-check/d1cff06/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MAIN=bd019f639fbb532b8c726b6b020314c7b056ff65
FKL_B=d1cff068dd2063ec9d4ff0206aef1836e98ba036; TP=27e733f50f7f40f171fd84655e05d5acf1fdfbee
FKLT_B=$T/worktrees/fkl-pr435-e322cad   # the head worktree keeps its C-NV step 1 path; it is at d1cff06
BUILD_B=$S/fkl-head-build
HB_B=$S/harness-build-d1cff06/head-d1cff06
H48=$S/harness-out/head-48fbf3f; H48_STDOUT=$REC/window/harness/head-48fbf3f/pr435_attention-stdout.txt; H48_SHA=$REC/window/harness-outputs-sha256.txt
RE='_cu$'; NTESTS=59
MANIFEST=$SUB/window-binaries.sha256
MANIFEST_SHA256=a1d245c803c7f6499192feb7790253c56c0a733061601127d6483861a18d28e9
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
# the binaries: the manifest is the committed one, every binary in it has its sha256, and none of them is blocked
echo "$MANIFEST_SHA256  $MANIFEST" | sha256sum -c --quiet || { echo "the binaries' manifest differs from the committed one: not starting"; exit 1; }
sha256sum -c --quiet $MANIFEST || { echo "a binary differs from the build record: not starting"; exit 1; }
refuse_blocked_binaries $(awk '{print $2}' $MANIFEST) || exit 1
refuse_blocked_tests $BUILD_B -R "$RE" || exit 1
LIST_B=$(ctest --test-dir $BUILD_B -N -R "$RE" 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: //p')
LIST_CNV=$(sed -n 's/^ *Start *[0-9]*: \([^ ]*\).*/\1/p' $REC/window/fkl-tests-head-48fbf3f-console.txt)
[ "$(echo "$LIST_B" | grep -c .)" = $NTESTS ] && [ "$(echo "$LIST_B" | sort)" = "$(echo "$LIST_CNV" | sort)" ] \
  || { echo "the build does not list the $NTESTS tests of the C-NV window: not starting"; exit 1; }
# every test the window runs is an executable of the manifest
for t in $LIST_B; do
  grep -q " $BUILD_B/bin/Release/$t\$" $MANIFEST || { echo "$t is not in the manifest: not starting"; exit 1; }
done
# the tree: the FKL head worktree and the build's source, the record's worktree
[ "$(git -C $FKLT_B rev-parse HEAD)" = $FKL_B ] && [ -z "$(git -C $FKLT_B status --porcelain --ignore-submodules=none --untracked-files=all)" ] \
  || { echo "$FKLT_B is not a clean $FKL_B: not starting"; exit 1; }
[ "$(git -C $FKLT_B/thread-pool rev-parse HEAD)" = $TP ] || { echo "$FKLT_B/thread-pool is not $TP: not starting"; exit 1; }
grep -q -x "CMAKE_HOME_DIRECTORY:INTERNAL=$FKLT_B" $BUILD_B/CMakeCache.txt || { echo "$BUILD_B is not configured from $FKLT_B: not starting"; exit 1; }
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results/rtx4090/fkl-pr435-check' || { echo "HEAD differs from $MAIN outside the record: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
# this script and the comparison are the committed ones, run from their place in the worktree
SELF=results/rtx4090/fkl-pr435-check/d1cff06/window-script.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
git -C $W ls-files --error-unmatch $SELF > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $SELF || { echo "$SELF is not the version committed at HEAD: not starting"; exit 1; }
COMPARE=$SUB/harness/source/compare_attention_bitwise.py
git -C $W ls-files --error-unmatch ${COMPARE#$W/} > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- ${COMPARE#$W/} || { echo "compare_attention_bitwise.py is not the committed version: not starting"; exit 1; }
# the 48fbf3f attention outputs that step 2 is compared with: still as the C-NV window recorded them (CPU only)
python3 -I $COMPARE $H48_STDOUT $H48 $H48_STDOUT $H48 --sha256 $H48_SHA > /dev/null || { echo "the 48fbf3f attention outputs are not those the C-NV window recorded: not starting"; exit 1; }
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
source $W/scripts/quiet_check.sh
# foreign_gpu: a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session)
foreign_gpu() { nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/'; }
busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(foreign_gpu)
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -e $HOUT ] && { echo "$HOUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT/harness $HOUT/head-d1cff06
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window fkl-pr435-check-d1cff06 starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $FKLT_B $FKLT_B/thread-pool; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "libraries of utest_flash_attention_mma_cu (d1cff06):"; ldd $BUILD_B/bin/Release/utest_flash_attention_mma_cu | grep -E 'cudart|stdc'
  echo "libraries of pr435_attention (d1cff06):"; ldd $HB_B/pr435_attention | grep -E 'cudart|stdc'; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $W/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
taskset -c 14 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 \
  > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
SAMPLER_PID=$!
stop_helpers() {
  kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null
  kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
}
finish() {   # finish <message>: stops the helpers, writes the environment after, logs the end
  stop_helpers
  { date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
  echo "# $(date -Iseconds) GPU window fkl-pr435-check-d1cff06 $1" | tee -a $LOG
  echo "fkl-pr435-check-d1cff06 window end $(date -Iseconds)"
}
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# 1. FKL's 59 CUDA tests at d1cff06, one after the other, stopping at the first failure
start=$(date +%s)
echo "# $(date -Iseconds) fkl-tests-head-d1cff06: ctest -R '$RE' -j1 --stop-on-failure --timeout 300 in $BUILD_B"
env $GENV taskset -c 0-3,16-19 ctest --test-dir $BUILD_B -R "$RE" -j1 --stop-on-failure --timeout 300 -V --output-log $OUT/fkl-tests-head-d1cff06.log \
  > $OUT/fkl-tests-head-d1cff06-console.txt 2>&1
rc=$?
echo "# $(date -Iseconds) fkl-tests-head-d1cff06 exit $rc after $(( $(date +%s) - start )) s: $(grep -E 'tests passed|tests failed' $OUT/fkl-tests-head-d1cff06-console.txt | tail -1)"
if [ $rc -ne 0 ] || ! grep -q "100% tests passed, 0 tests failed out of $NTESTS" $OUT/fkl-tests-head-d1cff06-console.txt; then
  finish "STOPPED after fkl-tests-head-d1cff06 (exit $rc)"; exit 1
fi
# 2. the attention harness at d1cff06: it must exit 0 with an empty stderr and print its PASS line
d=$OUT/harness/head-d1cff06; mkdir -p $d; p=pr435_attention
start=$(date +%s)
echo "# $(date -Iseconds) harness head-d1cff06 $p"
env $GENV timeout 300 taskset -c 0-3,16-19 $HB_B/$p $HOUT/head-d1cff06 > $d/$p-stdout.txt 2> $d/$p-stderr.txt
rc=$?; echo $rc > $d/$p-exit_code.txt
nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $d/$p-gpu-at-end.csv
echo "# $(date -Iseconds) harness head-d1cff06 $p exit $rc after $(( $(date +%s) - start )) s; $(grep -c '^OUT ' $d/$p-stdout.txt) output files; stderr $(wc -c < $d/$p-stderr.txt) B"
if [ $rc -ne 0 ] || [ -s $d/$p-stderr.txt ] || ! grep -q "^$p: .* PASS " $d/$p-stdout.txt; then
  finish "STOPPED after harness head-d1cff06 $p (exit $rc)"; exit 1
fi
finish "ended (GPU steps)"
# CPU only, after the GPU steps: the 32 attention outputs at d1cff06 against the C-NV window's at 48fbf3f, and their sha256
python3 -I $COMPARE $H48_STDOUT $H48 $d/$p-stdout.txt $HOUT/head-d1cff06 --sha256 $H48_SHA --out $OUT/harness-comparison.txt > /dev/null
echo "compare_attention_bitwise exit $? ($(tail -1 $OUT/harness-comparison.txt))"
( cd $HOUT && find . -type f | sort | xargs sha256sum ) > $OUT/harness-outputs-sha256.txt
echo "harness outputs: $(wc -l < $OUT/harness-outputs-sha256.txt) files, $(du -sm $HOUT | cut -f1) MiB (not committed)"
