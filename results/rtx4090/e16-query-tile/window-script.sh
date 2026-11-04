#!/bin/bash
# GPU window "qt" (quiet, about 8 to 12 minutes, the desktop session keeps running): the E16 query-tile experiment e16_query_tile_cu, D = 128, non-causal, S = 512 .. 8192, three runs (E16_ROTATE 0, 1, 2) one after the other.
# RTX 4090 = nvidia-smi index 1, pinned to cores 0-3,16-19. Results: results/rtx4090/e16-query-tile/run{0,1,2}/e16_query_tile.csv. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
OUT=${QT_OUTDIR:-$FB/results/rtx4090/e16-query-tile}; LOG=${QT_LOG:-$FB/results/rtx4090/e0-run.log}
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window qt starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; git -C $FB rev-parse HEAD; git -C $FB status --porcelain --untracked-files=no; git -C $FB/external/fkl rev-parse HEAD; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv > $OUT/gpu-info.csv
sha256sum $FB/build/rtx4090/bin/e16_query_tile_cu > $OUT/binary-sha256.txt
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
for rot in 0 1 2; do
  mkdir -p $OUT/run$rot; start=$(date +%s)
  echo "# $(date -Iseconds) run $rot (E16_ROTATE=$rot)"
  ( cd $OUT/run$rot && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 E16_DIMS=128 E16_CAUSAL=0 E16_ROTATE=$rot timeout 1500 taskset -c 0-3,16-19 $FB/build/rtx4090/bin/e16_query_tile_cu > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) run $rot exit $(cat $OUT/run$rot/exit_code.txt) after $(( $(date +%s) - start )) s; csv rows $(grep -vc '^#' $OUT/run$rot/e16_query_tile.csv 2>/dev/null)"
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window qt ended" | tee -a $LOG
echo "qt window end $(date -Iseconds)"
