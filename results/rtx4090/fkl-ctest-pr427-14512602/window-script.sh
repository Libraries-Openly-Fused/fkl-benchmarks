#!/bin/bash
# GPU window "ctest-pr427" (10 minutes reserved by the coordinator, hard timeout BUDGET = 540 s counted from the start of the first suite; the desktop session keeps running).
# Everything here was built against the head of FKL PR #427 (issue #19, the crop fix) in step 1, compile only, without the GPU:
#   1. fkl-cuda:      FKL's CUDA tests, `ctest -R '_cu$'`, in the fkltests build of scripts/check_fkl_tree.sh;
#   2. cvGPUSpeedup:  its ctest suite (its pin), configured with ENABLE_BENCHMARK=ON as for #416, without benchmark_image_resolution_MAD_loop as for #416 (25 tests);
#   3. FastNPP:       its ctest suite (its pin a1b5e24: 2 tests), configured with ENABLE_BENCHMARK=ON.
# (The two NPP programs of fkl-benchmarks are not run: the coordinator dropped them, 2026-11-06, since they will be rewritten without FKL objects.)
# One test at a time (-j1) on the RTX 4090 (nvidia-smi index 1, CUDA_DEVICE_ORDER=PCI_BUS_ID), pinned to cores 0-3,16-19. Not a measurement: pass / fail / timeout
# per test, logs kept, no timing claim.
# Environment: HEAD (required: the full SHA under test), CHECK (the output folder of the compile check), CLONE (the FKL clone at HEAD), OUT,
# START_DELAY (seconds to wait first), GUARD_ONLY (only the guard), DRY_RUN (lists the tests with ctest -N; runs nothing on the GPU, writes nothing to
# e0-run.log; for testing this script).
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
HEAD=${HEAD:?set HEAD to the full SHA of the PR #427 head under test}
CHECK=${CHECK:-$T/scratch/pr427/check}; CLONE=${CLONE:-$T/scratch/pr427/fkl-head}
OUT=${OUT:-$FB/results/rtx4090/fkl-ctest-pr427-${HEAD:0:8}}
LOG=$FB/results/rtx4090/e0-run.log; [ -n "${DRY_RUN:-}" ] && LOG=/dev/null
BUDGET=${BUDGET:-540}; PER_TEST=${PER_TEST:-240}; CORES_RUN=0-3,16-19
SUITE_NAME=(fkl-cuda cvGPUSpeedup FastNPP)
SUITE_DIR=("${FKLTESTS_BUILD:-$CHECK/fkltests-build}" "${CVGS_BUILD:-$CHECK/cvgs-bench-build}" "${FASTNPP_BUILD:-$CHECK/fastnpp-bench-build}")
SUITE_FILTER=('_cu$' '' '')
# cvGPUSpeedup: the 25 tests of #409 to #416, which ran `ctest -E benchmark_image_resolution_MAD_loop` (the full MAD loop, 60 experiments, was never part of
# the suite run; its 3-experiment smoke variant was run apart, and the coordinator dropped it for #427)
SUITE_EXCLUDE=('' '^benchmark_image_resolution_MAD_loop$' '')
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
ctest_filter() { [ -n "${SUITE_FILTER[$1]}" ] && printf -- '-R\n%s\n' "${SUITE_FILTER[$1]}"; [ -n "${SUITE_EXCLUDE[$1]}" ] && printf -- '-E\n%s\n' "${SUITE_EXCLUDE[$1]}"; return 0; }

# Checks that need no GPU, before anything starts: the clone is at HEAD without local changes, every suite lists tests
[ "$(git -C $CLONE rev-parse HEAD 2>/dev/null)" = "$HEAD" ] || { echo "the clone $CLONE is not at $HEAD: not starting"; exit 1; }
[ -z "$(git -C $CLONE status --short --untracked-files=no)" ] || { echo "the clone $CLONE has local changes: not starting"; exit 1; }
for k in 0 1 2; do
  mapfile -t f < <(ctest_filter $k)
  n=$(cd "${SUITE_DIR[$k]}" 2>/dev/null && ctest -N "${f[@]}" | sed -n 's/^Total Tests: //p')
  [ "${n:-0}" -gt 0 ] || { echo "${SUITE_NAME[$k]}: no tests in ${SUITE_DIR[$k]}: not starting"; exit 1; }
done

sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then
  if [ -z "${DRY_RUN:-}" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
  echo "(dry run) the guard would refuse now:"; echo "$busy" | cut -c1-200
fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window ctest-pr427 starts${DRY_RUN:+ (DRY RUN: nothing runs on the GPU)}; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement of mine running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "fkl-benchmarks $(git -C $FB rev-parse HEAD)"
  echo "FKL tree under test (PR #427 head) $(git -C $CLONE rev-parse HEAD), local changes: $(git -C $CLONE status --short --untracked-files=no | wc -l)"
  for s in cvgs-src fastnpp-src; do [ -e $CHECK/$s ] && echo "$s $(git -C $CHECK/$s rev-parse HEAD 2>/dev/null) fkl -> $(readlink -f $CHECK/$s/fkl 2>/dev/null)"; done
  for k in 0 1 2; do echo "${SUITE_NAME[$k]} build ${SUITE_DIR[$k]}: $(grep -E '^(ENABLE_BENCHMARK|CMAKE_BUILD_TYPE|CMAKE_CUDA_ARCHITECTURES):' ${SUITE_DIR[$k]}/CMakeCache.txt | tr '\n' ' ')"; done
  nvcc --version | tail -2; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,memory.used,utilization.gpu --format=csv > $OUT/gpu-info.csv
for k in 0 1 2; do
  o=$OUT/${SUITE_NAME[$k]}; mkdir -p $o; mapfile -t f < <(ctest_filter $k)
  ( cd ${SUITE_DIR[$k]} && ctest -N "${f[@]}" ) > $o/test-list.txt 2>&1
  ( cd ${SUITE_DIR[$k]} && ctest --show-only=json-v1 "${f[@]}" ) | python3 -c 'import json,sys; [print(t["command"][0]) for t in json.load(sys.stdin)["tests"] if t.get("command")]' | sort -u | xargs -r sha256sum > $o/binary-sha256.txt 2>&1
done
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
start=$(date +%s); deadline=$((start + BUDGET))
for k in 0 1 2; do
  o=$OUT/${SUITE_NAME[$k]}; d=${SUITE_DIR[$k]}; mapfile -t f < <(ctest_filter $k); rem=$((deadline - $(date +%s)))
  if [ $rem -lt 10 ]; then echo "# $(date -Iseconds) ${SUITE_NAME[$k]}: NOT RUN, the budget of $BUDGET s is used up" | tee $o/not-run.txt; continue; fi
  echo "# $(date -Iseconds) ${SUITE_NAME[$k]} starts (hard timeout $rem s, per test $PER_TEST s)"
  t0=$(date +%s)
  if [ -n "${DRY_RUN:-}" ]; then
    ( cd $d && ctest -N "${f[@]}" > $o/ctest-stdout.txt 2>&1; echo $? > $o/ctest-exit-code.txt )
  else
    rm -f $d/Testing/Temporary/LastTest.log $d/Testing/Temporary/LastTestsFailed.log
    ( cd $d && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c $CORES_RUN timeout -k 10 $rem ctest "${f[@]}" --timeout $PER_TEST -j1 --output-on-failure --no-compress-output -T Test --output-log $o/ctest-output.log > $o/ctest-stdout.txt 2>&1; echo $? > $o/ctest-exit-code.txt )
    cp -p $d/Testing/Temporary/LastTest.log $o/LastTest.log 2>/dev/null; cp -p $d/Testing/Temporary/LastTestsFailed.log $o/LastTestsFailed.log 2>/dev/null
    cp -p $(ls -d $d/Testing/2*/ 2>/dev/null | tail -1)Test.xml $o/Test.xml 2>/dev/null
  fi
  echo "# $(date -Iseconds) ${SUITE_NAME[$k]} ended, exit $(cat $o/ctest-exit-code.txt) after $(( $(date +%s) - t0 )) s"
  grep -E 'tests passed|tests failed|Total Test time|Total Tests' $o/ctest-stdout.txt | cut -c1-200
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv > $OUT/gpu-after.csv
echo "# $(date -Iseconds) GPU window ctest-pr427 ended${DRY_RUN:+ (DRY RUN)}; load after: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
echo "ctest-pr427 window end $(date -Iseconds)"
