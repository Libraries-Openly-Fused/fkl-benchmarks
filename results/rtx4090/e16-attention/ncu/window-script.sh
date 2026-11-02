#!/bin/bash
# GPU window "e16-ncu" (quiet, about 3 to 4 minutes): Nsight Compute on the attention kernels of the E16 program at D = 128, S = 2048, non-causal, BH = 128 (RTX 4090 = nvidia-smi index 1, the card that also drives the display):
# one call per variant (E16_PROFILE=1): mma_raw, mma_int8kv_fused, mma_int8kv_unfused (= the dequantKV kernel + the bf16 kernel), mma_epilogue_fused, mma_epilogue_separate (= the bf16 kernel + the affineBf16 kernel), mma_fp8kv_fused: 8 kernels.
# ncu --set full, --launch-count 20 (8 kernels are expected; any extra launch is kept and filtered afterwards), --rename-kernels 0 (full names with the template arguments), clock control base (the ncu default), pinned to cores 0-3,16-19. The .ncu-rep stays out of the repository (results/cuda13.4/e16-ncu); the CSV export is made after the window. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh; OUT=$FB/results/rtx4090/e16-attention/ncu; REP=$T/results/cuda13.4/e16-ncu; LOG=$FB/results/rtx4090/e0-run.log
NCU=$T/toolchains/cuda-13.4.3/bin/ncu; E16=$FB/build/local-rtx4090/bin/e16_attention_cu
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
mkdir -p $OUT $REP
echo "# $(date -Iseconds) GPU window e16-ncu starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked)" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/processes-before.txt
{ sha256sum $E16; $NCU --version | tail -2; echo "fkl-benchmarks $(git -C $FB rev-parse HEAD)-dirty (E16 sources are not committed)"; echo "external/fkl $(git -C $FB/external/fkl rev-parse HEAD)"; } > $OUT/binary-sha256.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current --format=csv > $OUT/gpu-info.csv
start=$(date +%s)
( cd $OUT && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 E16_PROFILE=1 E16_DIMS=128 E16_SEQS=2048 E16_CAUSAL=0 E16_VARIANTS=mma_raw,mma_int8kv,mma_epilogue,mma_fp8kv_fused \
  timeout 900 taskset -c 0-3,16-19 $NCU --set full --kernel-name "regex:FlashAttentionMma|dequantKV|affineBf16" --launch-count 20 --rename-kernels 0 --export $REP/e16_d128_s2048 --force-overwrite $E16 > ncu_stdout.txt 2> ncu_stderr.txt; echo $? > ncu_exit.txt )
echo "ncu exit $(cat $OUT/ncu_exit.txt) after $(( $(date +%s) - start )) s; $(grep -c '==PROF==' $OUT/ncu_stdout.txt) ncu lines; report $(du -h $REP/e16_d128_s2048.ncu-rep* 2> /dev/null | cut -f1)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/processes-after.txt
echo "# $(date -Iseconds) GPU window e16-ncu ended" >> $LOG
echo "e16-ncu window end $(date -Iseconds)"
