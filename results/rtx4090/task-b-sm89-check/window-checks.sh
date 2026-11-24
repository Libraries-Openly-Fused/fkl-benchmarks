#!/bin/bash
# GPU window "task-b-sm89-check" (tests and checks only; nothing is timed; a few minutes; the desktop session keeps running): FKL PR #433
# (head 4b11bc6) against its base 24478f0 on the RTX 4090 (sm_89). Built in step 2 (compile only), every binary listed with its sha256 in
# window-binaries.sha256 (whose own sha256 is MANIFEST_SHA256 below). Steps, in this order; the window stops at the first failure:
#   1. FKL's CUDA filter tests at 24478f0: the 7 tests of tests/image_processing that exist at both commits (box_filter_quad_dpp,
#      convolution_fast_dpp, linear_filter_dpp, median_fast_dpp, median_filter_dpp, morphology_dpp, morphology_fast_dpp; #433 changes all of
#      them but morphology_dpp), ctest -j1 --stop-on-failure in the build fkl-before-build. Every test must pass.
#   2. The same at 4b11bc6, plus its 3 new tests (linear_filter_batch, median_filter_batch, quad_dpps_batch): 10 tests, build fkl-after-build.
#   3. The checks only (no timing) of the five fkl-benchmarks programs, BEFORE = main b4fb594 with FKL 24478f0: e18_box_cu, e18_box_split_cu,
#      e18_stencil_cu (E18_NO_TIMING=1), e19_batched_box_cu (E19_NO_TIMING=1, E19_DUMP), e19_crop_sweep_cu at N = 1, 10, 20, ..., 150
#      (E19_NO_TIMING=1, E19_CROPS=N, E19_DUMP). Every run must exit 0, report its checks passed and write a complete checks CSV.
#   4. The same with AFTER = b4fb594 with FKL 4b11bc6 and the details port (details-port.diff), the same order.
#   No nsys, no ncu, no clock warm-up. SAMPLER: nvidia-smi every 100 ms through the window (gpu_clocks.csv), on core 14. Runs pinned to cores
#   0-3,16-19.
# Guards before anything starts: no blocked program (the binaries and the ctest lists); every binary's sha256 is the manifest's; the programs
# carry the expected commits (before: b4fb594 and FKL 24478f0, no -dirty; after: b4fb594-dirty and FKL 4b11bc6); the record's worktree is at
# b4fb594 or a descendant that differs from it only under results/, with no local change outside results/; the after worktree is b4fb594 with
# exactly the committed details-port.diff and external/fkl at 4b11bc6; the two FKL worktrees are at 24478f0 and 4b11bc6 with no local change,
# and the FKL builds are configured from them; this script is the one committed at HEAD and runs from there; the output folder is the requested
# one and new, and so is the dump folder; the ctest lists have exactly the 7 and 10 tests; no measurement, build or compile job runs and no
# other program uses the RTX 4090. RTX 4090 = nvidia-smi index 1. Nothing is overwritten.
# After the GPU steps, CPU only: the outputs before against after, bit for bit (compare_outputs.py): the data rows of every CSV and every dump
# file; the dumps' sha256 list. The dumps (about 1.8 GiB per side) stay in the scratch folder, not committed.
# Output: results/rtx4090/task-b-sm89-check/window/ of the record's worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-task-b; WA=$T/worktrees/fkl-benchmarks-task-b-after
S=$T/scratch/task-b; source $W/scripts/os_kernel_lines.sh
REC=$W/results/rtx4090/task-b-sm89-check
OUT=$REC/window; DUMPS=$S/dumps; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/task-b-sm89-check/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MAIN=b4fb59487e8f3cc7077bc9576e413080dddd2512; FKL_A=24478f0aa11dc3f12feea83178f7fde828322fb0; FKL_B=4b11bc6892a79d9481406da7903138ca567e2c6a
FKLT_A=$T/worktrees/fkl-pr433-base-24478f0; FKLT_B=$T/worktrees/fkl-pr433-4b11bc6
BUILD_A=$S/fkl-before-build; BUILD_B=$S/fkl-after-build
BIN_A=$W/build/local-rtx4090/bin; BIN_B=$WA/build/local-rtx4090/bin
PROGS="e18_box_cu e18_box_split_cu e18_stencil_cu e19_batched_box_cu e19_crop_sweep_cu"
CROPS_LIST="1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
EXISTING='test_box_filter_quad_dpp|test_convolution_fast_dpp|test_linear_filter_dpp|test_median_fast_dpp|test_median_filter_dpp|test_morphology_dpp|test_morphology_fast_dpp'
NEWTESTS='test_linear_filter_batch|test_median_filter_batch|test_quad_dpps_batch'
RE_A="^($EXISTING)_cu\$"; RE_B="^($EXISTING|$NEWTESTS)_cu\$"
MANIFEST=$REC/window-binaries.sha256
MANIFEST_SHA256=8afc521de0af11a13e87400d36c172ae19aaafdda452ad578140f7b8362171c5
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
# the binaries: the manifest is the committed one, every binary in it has its sha256, and none of them is blocked
echo "$MANIFEST_SHA256  $MANIFEST" | sha256sum -c --quiet || { echo "the binaries' manifest differs from the committed one: not starting"; exit 1; }
sha256sum -c --quiet $MANIFEST || { echo "a binary differs from the build record: not starting"; exit 1; }
refuse_blocked_binaries $(awk '{print $2}' $MANIFEST) || exit 1
refuse_blocked_tests $BUILD_A -R "$RE_A" || exit 1
refuse_blocked_tests $BUILD_B -R "$RE_B" || exit 1
[ "$(ctest --test-dir $BUILD_A -N -R "$RE_A" 2>/dev/null | grep -c '^ *Test *#')" = 7 ] || { echo "the before build does not list exactly the 7 tests: not starting"; exit 1; }
[ "$(ctest --test-dir $BUILD_B -N -R "$RE_B" 2>/dev/null | grep -c '^ *Test *#')" = 10 ] || { echo "the after build does not list exactly the 10 tests: not starting"; exit 1; }
# the programs carry the expected commits
for p in $PROGS; do
  strings $BIN_A/$p | grep -q -x $MAIN || { echo "before $p: not $MAIN: not starting"; exit 1; }
  strings $BIN_A/$p | grep -q -x $FKL_A || { echo "before $p: not FKL $FKL_A: not starting"; exit 1; }
  strings $BIN_A/$p | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "before $p: a -dirty commit: not starting"; exit 1; }
  strings $BIN_B/$p | grep -q -x $MAIN-dirty || { echo "after $p: not $MAIN-dirty: not starting"; exit 1; }
  strings $BIN_B/$p | grep -q -x $FKL_B || { echo "after $p: not FKL $FKL_B: not starting"; exit 1; }
done
# the trees: the record's worktree, the after worktree, the FKL worktrees and the FKL builds' sources
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results' || { echo "HEAD differs from $MAIN outside results/: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $FKL_A ] || { echo "the before worktree's FKL is not $FKL_A: not starting"; exit 1; }
[ "$(git -C $WA rev-parse HEAD)" = $MAIN ] || { echo "the after worktree is not at $MAIN: not starting"; exit 1; }
[ "$(git -C $WA/external/fkl rev-parse HEAD)" = $FKL_B ] && [ -z "$(git -C $WA/external/fkl status --porcelain --untracked-files=no)" ] \
  || { echo "the after worktree's FKL is not a clean $FKL_B: not starting"; exit 1; }
git -C $WA diff -- opencv-cuda | cmp -s - $REC/details-port.diff || { echo "the after worktree's change is not details-port.diff: not starting"; exit 1; }
[ "$(git -C $WA status --porcelain --untracked-files=no | grep -v -E '^ M (external/fkl|opencv-cuda/stencil/e1[89]_[a-z_]+\.cu)$')" = "" ] \
  || { echo "the after worktree has other changes: not starting"; exit 1; }
for x in "$FKLT_A $FKL_A $BUILD_A" "$FKLT_B $FKL_B $BUILD_B"; do
  set -- $x
  [ "$(git -C $1 rev-parse HEAD)" = $2 ] && [ -z "$(git -C $1 status --porcelain --untracked-files=no)" ] || { echo "$1 is not a clean $2: not starting"; exit 1; }
  grep -q -x "CMAKE_HOME_DIRECTORY:INTERNAL=$1" $3/CMakeCache.txt || { echo "$3 is not configured from $1: not starting"; exit 1; }
done
# this script is the committed one, run from its place in the worktree
SELF=results/rtx4090/task-b-sm89-check/window-checks.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
git -C $W ls-files --error-unmatch $SELF > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $SELF || { echo "$SELF is not the version committed at HEAD: not starting"; exit 1; }
COMPARE=$REC/scripts-used/compare_outputs.py
git -C $W ls-files --error-unmatch ${COMPARE#$W/} > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- ${COMPARE#$W/} || { echo "compare_outputs.py is not the committed version: not starting"; exit 1; }
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
[ -e $DUMPS ] && { echo "$DUMPS exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT $DUMPS
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window task-b-sm89-check starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $W/external/fkl $W/external/FastNPP $W/external/cvGPUSpeedup $W/external/fkl/thread-pool $WA $WA/external/fkl $FKLT_A $FKLT_B; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "libraries of e19_crop_sweep_cu (before):"; ldd $BIN_A/e19_crop_sweep_cu | grep -E 'npp|cudart|opencv'
  echo "libraries of e19_crop_sweep_cu (after):"; ldd $BIN_B/e19_crop_sweep_cu | grep -E 'npp|cudart|opencv'; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window task-b-sm89-check $1" | tee -a $LOG
  echo "task-b-sm89-check window end $(date -Iseconds)"
}
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# fkl_tests <tag> <build> <regex> <count>: the CUDA filter tests of one FKL commit, one after the other, stopping at the first failure
fkl_tests() {
  local tag=$1 b=$2 re=$3 count=$4 start=$(date +%s) rc
  echo "# $(date -Iseconds) fkl-tests-$tag: ctest -R '$re' -j1 in $b"
  env $GENV taskset -c 0-3,16-19 ctest --test-dir $b -R "$re" -j1 --stop-on-failure --timeout 300 -V --output-log $OUT/fkl-tests-$tag.log \
    > $OUT/fkl-tests-$tag-console.txt 2>&1
  rc=$?
  echo "# $(date -Iseconds) fkl-tests-$tag exit $rc after $(( $(date +%s) - start )) s: $(grep -E 'tests passed|tests failed' $OUT/fkl-tests-$tag-console.txt | tail -1)"
  if [ $rc -ne 0 ] || ! grep -q "100% tests passed, 0 tests failed out of $count" $OUT/fkl-tests-$tag-console.txt; then
    finish "STOPPED after fkl-tests-$tag (exit $rc)"; exit 1
  fi
}
# run <side> <run folder> <binary dir> <program> [VAR=value ...]: pinned to cores 0-3,16-19; the run folder is runs/<side>/<run folder>
run() {
  local side=$1 name=$2 bin=$3 prog=$4; shift 4
  local dir=$OUT/runs/$side/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  echo "# $(date -Iseconds) $side/$name $prog $*"
  ( cd $dir && env $GENV "$@" timeout 300 taskset -c 0-3,16-19 $bin/$prog > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $side/$name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks passed: $(grep -c -E '^E1[89] .*checks passed$|^E18 checks passed$' $dir/stdout.txt)"
}
# stop_at_failure <side> <run folder> <pass line> <checks CSV>: a run that did not exit 0, did not print its pass line or did not write a complete
# checks CSV ends the window here
stop_at_failure() {
  local dir=$OUT/runs/$1/$2 rc=$(cat $OUT/runs/$1/$2/exit_code.txt 2>/dev/null) ok=1
  [ "$rc" = 0 ] || ok=0
  grep -q -x "$3" $dir/stdout.txt || ok=0
  [ -s $dir/$4 ] && ! grep -q '^# INCOMPLETE' $dir/$4 || ok=0
  [ $ok = 1 ] && return 0
  finish "STOPPED after $1/$2 (exit ${rc:-missing})"
  exit 1
}
programs() {  # programs <side> <binary dir>
  local side=$1 bin=$2 n d
  run $side e18_box $bin e18_box_cu E18_NO_TIMING=1;             stop_at_failure $side e18_box "E18 box checks passed" e18_box_checks.csv
  run $side e18_box_split $bin e18_box_split_cu E18_NO_TIMING=1; stop_at_failure $side e18_box_split "E18 box split checks passed" e18_box_split_checks.csv
  run $side e18_stencil $bin e18_stencil_cu E18_NO_TIMING=1;     stop_at_failure $side e18_stencil "E18 checks passed" e18_stencil_checks.csv
  mkdir -p $DUMPS/$side/e19_batched_box
  run $side e19_batched_box $bin e19_batched_box_cu E19_NO_TIMING=1 E19_DUMP=$DUMPS/$side/e19_batched_box
  stop_at_failure $side e19_batched_box "E19 batched box checks passed" e19_batched_box_checks.csv
  for n in $CROPS_LIST; do
    d=e19_crop_sweep/N$(printf %03d $n); mkdir -p $DUMPS/$side/$d
    run $side $d $bin e19_crop_sweep_cu E19_NO_TIMING=1 E19_CROPS=$n E19_DUMP=$DUMPS/$side/$d
    stop_at_failure $side $d "E19 batched box checks passed" e19_batched_box_checks.csv
  done
}
# 1-2. FKL's CUDA filter tests at 24478f0, then at 4b11bc6
fkl_tests before $BUILD_A "$RE_A" 7
fkl_tests after $BUILD_B "$RE_B" 10
# 3-4. the checks of the five programs, before, then after
programs before $BIN_A
programs after $BIN_B
finish "ended (GPU steps)"
# CPU only, after the GPU steps: the outputs before against after, and the dumps' sha256
python3 -I $COMPARE --runs $OUT/runs --dumps $DUMPS --out $OUT/comparison.txt --csv $OUT/comparison.csv > /dev/null
echo "compare_outputs exit $? ($(sed -n 2p $OUT/comparison.txt))"
( cd $DUMPS && find . -type f | sort | xargs sha256sum ) > $OUT/dumps-sha256.txt
echo "dumps: $(wc -l < $OUT/dumps-sha256.txt) files, $(du -sm $DUMPS | cut -f1) MiB (not committed)"
