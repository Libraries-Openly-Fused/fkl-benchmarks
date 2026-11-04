#!/bin/bash
# GPU window "qt-ncu" (quiet, about 3 minutes, the desktop session keeps running): Nsight Compute on the four attention kernels of the E16 query-tile program at D = 128, S = 2048, non-causal, BH = 128:
# raw_bq64_kv32_w4, int8_bq64_w4, int8_bq128_w4, int8nodq_bq64_w4 (E16_PROFILE=1: one call per variant). ncu --set full, clock control base (the ncu default), pinned to cores 0-3,16-19.
# The report stays out of the repository (results/cuda13.4/e16-qt-ncu); the CSV exports are made after the window. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
OUT=${QT_NCU_OUTDIR:-$FB/results/rtx4090/e16-query-tile/ncu}; REP=${QT_NCU_REPDIR:-$T/results/cuda13.4/e16-qt-ncu}; LOG=${QT_LOG:-$FB/results/rtx4090/e0-run.log}
NCU=$T/toolchains/cuda-13.4.3/bin/ncu; BIN=$FB/build/rtx4090/bin/e16_query_tile_cu
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ] || [ -n "$(ls -A $REP 2>/dev/null)" ]; then echo "$OUT or $REP is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT $REP
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window qt-ncu starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/processes-before.txt
{ sha256sum $BIN; $NCU --version | tail -2; echo "fkl-benchmarks $(git -C $FB rev-parse HEAD)"; echo "external/fkl $(git -C $FB/external/fkl rev-parse HEAD)"; } > $OUT/binary-sha256.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv > $OUT/gpu-info.csv
start=$(date +%s)
( cd $OUT && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 E16_PROFILE=1 E16_DIMS=128 E16_SEQS=2048 E16_CAUSAL=0 E16_ROTATE=0 E16_VARIANTS=raw_bq64_kv32_w4,int8_bq64_w4,int8_bq128_w4,int8nodq_bq64_w4 \
  timeout 900 taskset -c 0-3,16-19 $NCU --set full --kernel-name "regex:FlashAttentionMma" --launch-count 12 --rename-kernels 0 --export $REP/e16_qt_d128_s2048 --force-overwrite $BIN > ncu_stdout.txt 2> ncu_stderr.txt; echo $? > ncu_exit.txt )
echo "ncu exit $(cat $OUT/ncu_exit.txt) after $(( $(date +%s) - start )) s; $(grep -c '==PROF==' $OUT/ncu_stdout.txt) ncu lines; report $(du -h $REP/e16_qt_d128_s2048.ncu-rep* 2> /dev/null | cut -f1)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/processes-after.txt
echo "# $(date -Iseconds) GPU window qt-ncu ended" | tee -a $LOG
echo "qt-ncu window end $(date -Iseconds)"
