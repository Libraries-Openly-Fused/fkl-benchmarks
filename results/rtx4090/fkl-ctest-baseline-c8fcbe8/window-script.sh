#!/bin/bash
# GPU window "ctest-a" (about 5 to 10 minutes, the desktop session keeps running): the 46 CUDA tests of FKL's ctest matrix, BASELINE build (FKL c8fcbe8 built by scripts/check_fkl_tree.sh, compile-only run of 2026-11-06),
# `ctest -R '_cu$' --timeout 600 -j1` one test at a time on the RTX 4090 (nvidia-smi index 1), pinned to cores 0-3,16-19. Not a measurement: pass / fail / timeout per test, logs kept per test, no timing claim.
# The load limit of the quiet guard is waived (the coordinator, 2026-11-06); the load is recorded before and after and every 5 s. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
BUILD=$T/scratch/fkl-check-baseline-fkltests-c8fcbe8/fkltests-build
OUT=$FB/results/rtx4090/fkl-ctest-baseline-c8fcbe8; LOG=$FB/results/rtx4090/e0-run.log
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window ctest-a starts; load before: $(cut -d' ' -f1-3 /proc/loadavg) (the load limit of the guard is waived for this window: the tests are not a measurement); no compile job, transfer or measurement of mine running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; echo "fkl-benchmarks $(git -C $FB rev-parse HEAD)"; echo "FKL tree under test $(git -C $FB/external/fkl rev-parse HEAD)"; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,memory.used,utilization.gpu --format=csv > $OUT/gpu-info.csv
( cd $BUILD && ctest -N -R '_cu$' ) > $OUT/test-list.txt 2>&1
find $BUILD/bin -type f -name '*_cu' -exec sha256sum {} + 2>/dev/null | sort -k2 > $OUT/binary-sha256.txt
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
start=$(date +%s)
echo "# $(date -Iseconds) ctest starts"
( cd $BUILD && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 taskset -c 0-3,16-19 timeout -k 20 840 ctest -R '_cu$' --timeout 600 -j1 --output-on-failure --no-compress-output -T Test --output-log $OUT/ctest-output.log > $OUT/ctest-stdout.txt 2>&1; echo $? > $OUT/ctest-exit-code.txt )
echo "# $(date -Iseconds) ctest ended, exit $(cat $OUT/ctest-exit-code.txt) after $(( $(date +%s) - start )) s"
cp -p $BUILD/Testing/Temporary/LastTest.log $OUT/LastTest.log 2>/dev/null; cp -p $BUILD/Testing/Temporary/LastTestsFailed.log $OUT/LastTestsFailed.log 2>/dev/null
cp -p $(ls -d $BUILD/Testing/2*/ 2>/dev/null | tail -1)Test.xml $OUT/Test.xml 2>/dev/null
tail -12 $OUT/ctest-stdout.txt | cut -c1-200
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
nvidia-smi -i 1 --query-gpu=memory.used,utilization.gpu,clocks.sm,temperature.gpu --format=csv > $OUT/gpu-after.csv
echo "# $(date -Iseconds) GPU window ctest-a ended; load after: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
echo "ctest-a window end $(date -Iseconds)"
