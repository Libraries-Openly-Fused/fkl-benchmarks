#!/bin/bash
# GPU window "e4-hand-cvgs" (quiet, about 8 minutes, the desktop session keeps running): the E4 program with the hand-written kernels of E1c, the FKL controls
# and cvGPUSpeedup (main 2dac1c8, the merge of #34), step 0's build. No reference build: the new versions have no older one.
#   WARMUP=1: before run 1, one untimed sweep of batch 300 only (E4_EXPERIMENTS=E4a E4_BATCHES_ONLY=300), its files in warmup/, not used.
#   1. Three complete quiet sweeps (E4a crops and E4c frames, every version), back to back, pinned to cores 0-3,16-19 as in window B.
#      Folders: quiet-run1..3.
#   SAMPLER=1: nvidia-smi every 100 ms through the whole window (gpu_clocks.csv), on core 14.
# Each folder is new; nothing is overwritten. RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first.
# Output: results/rtx4090/e4-hand-cvgs/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
W=$T/worktrees/fkl-benchmarks-window-f; NEW=$W/build/local-rtx4090/bin; STEP0=$T/scratch/e4-window-build
NEWC=2dac1c8ac467a45ff3076b4e25a0fb8160e5cca4
R=$FB/results/rtx4090; OUT=$R/e4-hand-cvgs; LOG=$R/e0-run.log
PROG=e4_preprocessing_cu
WARMUP=${WARMUP:?set WARMUP=0 or 1}; SAMPLER=${SAMPLER:?set SAMPLER=0 or 1}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW/$PROG || exit 1
# the binary is the one step 0 checked (sha256), and it carries the expected commit without -dirty
grep -q 'step 0 ended: all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
grep -E "fkl-benchmarks-window-f/build/local-rtx4090/bin/$PROG\$" $STEP0/binary-sha256.txt | sha256sum -c --quiet || { echo "the binary differs from step 0's record: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -x $NEWC || { echo "$PROG: not the expected commit: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$PROG: a -dirty commit: not starting"; exit 1; }
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
# a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session) refuses the window
gpuapps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/')
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e4-hand-cvgs starts (WARMUP=$WARMUP SAMPLER=$SAMPLER); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $FB $W $W/external/fkl $W/external/FastNPP $W/external/cvGPUSpeedup $W/external/fkl/thread-pool; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
SAMPLER_PID=""
if [ "$SAMPLER" = 1 ]; then
  taskset -c 14 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 \
    > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
  SAMPLER_PID=$!
fi
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <run folder> [VAR=value ...]: pinned to cores 0-3,16-19
run() {
  local name=$1; shift
  local dir=$OUT/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  echo "# $(date -Iseconds) $name $*"
  ( cd $dir && env $GENV "$@" timeout 900 taskset -c 0-3,16-19 $NEW/$PROG > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l)"
}
[ "$WARMUP" = 1 ] && run warmup E4_EXPERIMENTS=E4a E4_BATCHES_ONLY=300
for n in 1 2 3; do run quiet-run$n; done
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e4-hand-cvgs ended" | tee -a $LOG
echo "e4-hand-cvgs window end $(date -Iseconds)"
