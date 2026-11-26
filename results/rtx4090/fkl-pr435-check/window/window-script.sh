#!/bin/bash
# GPU window "fkl-pr435-check" (FKL's tests and the output harness only; nothing is timed; a few minutes; the desktop session keeps running):
# FKL PR #435 (head 48fbf3f) against its base b10fc66 on the RTX 4090 (sm_89). Built in steps 1 and 1b (compile only); every binary is listed
# with its sha256 in window-binaries.sha256, whose own sha256 is MANIFEST_SHA256 below. Steps, in this order; the window stops at the first failure:
#   1. FKL's CUDA tests at 48fbf3f: the 59 ctest tests named *_cu (every CUDA target of tests/ and utests/; the same 59 names at both commits),
#      ctest -j1 --stop-on-failure --timeout 300 in scratch/pr435/fkl-head-build. They include utest_flash_attention_mma_cu with #435's
#      split-KV + combine test. Every test must pass.
#   2. The same 59 tests at b10fc66 (scratch/pr435/fkl-base-build): the baseline. Every test must pass.
#   3. The harness of the base build (b10fc66): pr435_image, then pr435_attention, into scratch/pr435/harness-out/base-b10fc66.
#   4. The harness of the head build (48fbf3f), the same, into scratch/pr435/harness-out/head-48fbf3f.
#      Each harness run must exit 0 with an empty standard error and print its PASS line.
#   No nsys, no ncu, no clock warm-up. SAMPLER: nvidia-smi every 100 ms through the window (gpu_clocks.csv), on core 14. The tests and the
#   harness are pinned to cores 0-3,16-19. CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 (the RTX 4090).
# Guards before anything starts: no blocked program (the binaries and the two ctest lists); every binary's sha256 is the manifest's; each ctest
# list has exactly the 59 tests, with the same names in both builds; the two FKL worktrees are clean at b10fc66 and 48fbf3f (thread-pool
# 27e733f) and the two FKL builds are configured from them; the record's worktree is at main 976e42b or a descendant that differs from it
# only under results/rtx4090/fkl-pr435-check/, with no local change outside results/; this script and compare_harness.py are the versions
# committed at HEAD and the script runs from its place there; the output folder is the requested one and new, and so is the harness output
# folder; no measurement, build or compile job runs and no other program uses the RTX 4090. RTX 4090 = nvidia-smi index 1. Nothing is
# overwritten.
# After the GPU steps, CPU only: harness/source/compare_harness.py compares the two harness output folders (bit for bit across the commits,
# and the split-path checks), and the outputs' sha256 are listed. The harness outputs stay in the scratch folder, not committed.
# Output: results/rtx4090/fkl-pr435-check/window/ of the record's worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-pr435
S=$T/scratch/pr435; source $W/scripts/os_kernel_lines.sh
REC=$W/results/rtx4090/fkl-pr435-check
OUT=$REC/window; HOUT=$S/harness-out; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/fkl-pr435-check/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MAIN=976e42bfad0ba61c3bb22a555cf72f4da6896aeb
FKL_A=b10fc6609426bd18020b9213c7196b528fc677a8; FKL_B=48fbf3f08e7551c3eb70ac2302d8ebc15ecad645; TP=27e733f50f7f40f171fd84655e05d5acf1fdfbee
FKLT_A=$T/worktrees/fkl-pr435-base-b10fc66; FKLT_B=$T/worktrees/fkl-pr435-e322cad   # the head worktree keeps its step 1 path; it is at 48fbf3f
BUILD_A=$S/fkl-base-build; BUILD_B=$S/fkl-head-build
HB_A=$S/harness-build/base-b10fc66; HB_B=$S/harness-build/head-48fbf3f
RE='_cu$'; NTESTS=59
MANIFEST=$REC/window-binaries.sha256
MANIFEST_SHA256=1f193c5d8800eb12fc51f4b3dac7a955f898914ba03c30ff5c1081af9d564634
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
# the binaries: the manifest is the committed one, every binary in it has its sha256, and none of them is blocked
echo "$MANIFEST_SHA256  $MANIFEST" | sha256sum -c --quiet || { echo "the binaries' manifest differs from the committed one: not starting"; exit 1; }
sha256sum -c --quiet $MANIFEST || { echo "a binary differs from the build record: not starting"; exit 1; }
refuse_blocked_binaries $(awk '{print $2}' $MANIFEST) || exit 1
refuse_blocked_tests $BUILD_A -R "$RE" || exit 1
refuse_blocked_tests $BUILD_B -R "$RE" || exit 1
LIST_A=$(ctest --test-dir $BUILD_A -N -R "$RE" 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: //p')
LIST_B=$(ctest --test-dir $BUILD_B -N -R "$RE" 2>/dev/null | sed -n 's/^ *Test *#[0-9]*: //p')
[ "$(echo "$LIST_A" | grep -c .)" = $NTESTS ] && [ "$LIST_A" = "$LIST_B" ] || { echo "the two builds do not list the same $NTESTS tests: not starting"; exit 1; }
# every test the window runs is an executable of the manifest
for t in $LIST_A; do
  grep -q " $BUILD_A/bin/Release/$t\$" $MANIFEST && grep -q " $BUILD_B/bin/Release/$t\$" $MANIFEST || { echo "$t is not in the manifest: not starting"; exit 1; }
done
# the trees: the FKL worktrees and the FKL builds' sources, the record's worktree
for x in "$FKLT_A $FKL_A $BUILD_A" "$FKLT_B $FKL_B $BUILD_B"; do
  set -- $x
  [ "$(git -C $1 rev-parse HEAD)" = $2 ] && [ -z "$(git -C $1 status --porcelain --ignore-submodules=none --untracked-files=all)" ] \
    || { echo "$1 is not a clean $2: not starting"; exit 1; }
  [ "$(git -C $1/thread-pool rev-parse HEAD)" = $TP ] || { echo "$1/thread-pool is not $TP: not starting"; exit 1; }
  grep -q -x "CMAKE_HOME_DIRECTORY:INTERNAL=$1" $3/CMakeCache.txt || { echo "$3 is not configured from $1: not starting"; exit 1; }
done
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results/rtx4090/fkl-pr435-check' || { echo "HEAD differs from $MAIN outside the record: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
# this script and the comparison are the committed ones, run from their place in the worktree
SELF=results/rtx4090/fkl-pr435-check/window-script.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
git -C $W ls-files --error-unmatch $SELF > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $SELF || { echo "$SELF is not the version committed at HEAD: not starting"; exit 1; }
COMPARE=$REC/harness/source/compare_harness.py
git -C $W ls-files --error-unmatch ${COMPARE#$W/} > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- ${COMPARE#$W/} || { echo "compare_harness.py is not the committed version: not starting"; exit 1; }
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
mkdir -p $OUT/harness $HOUT/base-b10fc66 $HOUT/head-48fbf3f
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window fkl-pr435-check starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $FKLT_A $FKLT_A/thread-pool $FKLT_B $FKLT_B/thread-pool; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "libraries of utest_flash_attention_mma_cu (head):"; ldd $BUILD_B/bin/Release/utest_flash_attention_mma_cu | grep -E 'cudart|stdc'
  echo "libraries of pr435_attention (head):"; ldd $HB_B/pr435_attention | grep -E 'cudart|stdc'; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window fkl-pr435-check $1" | tee -a $LOG
  echo "fkl-pr435-check window end $(date -Iseconds)"
}
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# fkl_tests <tag> <build>: the 59 CUDA tests of one FKL commit, one after the other, stopping at the first failure
fkl_tests() {
  local tag=$1 b=$2 start=$(date +%s) rc
  echo "# $(date -Iseconds) fkl-tests-$tag: ctest -R '$RE' -j1 --stop-on-failure --timeout 300 in $b"
  env $GENV taskset -c 0-3,16-19 ctest --test-dir $b -R "$RE" -j1 --stop-on-failure --timeout 300 -V --output-log $OUT/fkl-tests-$tag.log \
    > $OUT/fkl-tests-$tag-console.txt 2>&1
  rc=$?
  echo "# $(date -Iseconds) fkl-tests-$tag exit $rc after $(( $(date +%s) - start )) s: $(grep -E 'tests passed|tests failed' $OUT/fkl-tests-$tag-console.txt | tail -1)"
  if [ $rc -ne 0 ] || ! grep -q "100% tests passed, 0 tests failed out of $NTESTS" $OUT/fkl-tests-$tag-console.txt; then
    finish "STOPPED after fkl-tests-$tag (exit $rc)"; exit 1
  fi
}
# harness <side> <build dir>: pr435_image, then pr435_attention; each must exit 0 with an empty stderr and print its PASS line
harness() {
  local side=$1 hb=$2 p start rc d=$OUT/harness/$1
  mkdir -p $d
  for p in pr435_image pr435_attention; do
    start=$(date +%s)
    echo "# $(date -Iseconds) harness $side $p"
    env $GENV timeout 300 taskset -c 0-3,16-19 $hb/$p $HOUT/$side > $d/$p-stdout.txt 2> $d/$p-stderr.txt
    rc=$?; echo $rc > $d/$p-exit_code.txt
    nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $d/$p-gpu-at-end.csv
    echo "# $(date -Iseconds) harness $side $p exit $rc after $(( $(date +%s) - start )) s; $(grep -c '^OUT ' $d/$p-stdout.txt) output files; stderr $(wc -c < $d/$p-stderr.txt) B"
    if [ $rc -ne 0 ] || [ -s $d/$p-stderr.txt ] || ! grep -q "^$p: .* PASS " $d/$p-stdout.txt; then
      finish "STOPPED after harness $side $p (exit $rc)"; exit 1
    fi
  done
}
# 1-2. FKL's CUDA tests at 48fbf3f, then at b10fc66
fkl_tests head-48fbf3f $BUILD_B
fkl_tests base-b10fc66 $BUILD_A
# 3-4. the harness, base, then head
harness base-b10fc66 $HB_A
harness head-48fbf3f $HB_B
finish "ended (GPU steps)"
# CPU only, after the GPU steps: the harness outputs base against head, and their sha256
python3 -I $COMPARE $HOUT/base-b10fc66 $HOUT/head-48fbf3f --out $OUT/harness-comparison.txt --csv $OUT/harness-comparison.csv > /dev/null
echo "compare_harness exit $? ($(tail -1 $OUT/harness-comparison.txt))"
( cd $HOUT && find . -type f | sort | xargs sha256sum ) > $OUT/harness-outputs-sha256.txt
echo "harness outputs: $(wc -l < $OUT/harness-outputs-sha256.txt) files, $(du -sm $HOUT | cut -f1) MiB (not committed)"
