#!/bin/bash
# GPU window "e6" (quiet, about 6 to 8 minutes, the desktop session keeps running): the E6 program e6_fused_vs_unfused_cu, built twice (rtx4090 = -fmad on, rtx4090-fmad-off), one run each, RTX 4090 = nvidia-smi index 1,
# pinned to cores 0-3,16-19. The raw outputs (bit patterns) go to results/cuda13.4/e6-raw (outside the repository), the CSVs and logs to results/rtx4090/e6-fused-vs-unfused/<tag>/. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
OUT=${E6_OUTDIR:-$FB/results/rtx4090/e6-fused-vs-unfused}; RAW=${E6_RAWDIR:-$T/results/cuda13.4/e6-raw}; LOG=${E6_LOG:-$FB/results/rtx4090/e0-run.log}
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ] || [ -n "$(ls -A $RAW 2>/dev/null)" ]; then echo "$OUT or $RAW is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT $RAW
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e6 starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; git -C $FB rev-parse HEAD; git -C $FB status --porcelain --untracked-files=no; git -C $FB/external/fkl rev-parse HEAD; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv > $OUT/gpu-info.csv
sha256sum $FB/build/rtx4090/bin/e6_fused_vs_unfused_cu $FB/build/rtx4090-fmad-off/bin/e6_fused_vs_unfused_cu > $OUT/binary-sha256.txt
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
for pair in "fmad_on rtx4090" "fmad_off rtx4090-fmad-off"; do
  set -- $pair; tag=$1; preset=$2
  mkdir -p $OUT/$tag; start=$(date +%s)
  echo "# $(date -Iseconds) run $tag ($preset)"
  ( cd $OUT/$tag && E6_TAG=$tag E6_OUT=$RAW timeout 900 taskset -c 0-3,16-19 $FB/build/$preset/bin/e6_fused_vs_unfused_cu > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) $tag exit $(cat $OUT/$tag/exit_code.txt) after $(( $(date +%s) - start )) s; $(tail -n 2 $OUT/$tag/stdout.txt | tr '\n' '|' | cut -c1-200)"
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e6 ended" | tee -a $LOG
echo "e6 window end $(date -Iseconds)"
