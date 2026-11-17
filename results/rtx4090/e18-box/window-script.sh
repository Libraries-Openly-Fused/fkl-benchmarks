#!/bin/bash
# GPU window "e18-box" (quiet, about 2 minutes, the desktop session keeps running): the timing of E18 with the 3 x 3 box filter (FKL's
# LinearFilterDPP in one kernel against OpenCV-CUDA and NPP), branch e18-box at 7dca510, step 0's build.
#   1. WARMUP=1: one complete run (the checks and the timing), its files in warmup/, not used.
#   2. Three complete quiet runs, quiet-run1..3, pinned to cores 0-3,16-19 as in #43's window. Before each, the script waits for an idle machine
#      (wait_idle: no measurement, build or compile job, a 1-minute load below IDLE_LOAD, no other program on the RTX 4090), up to IDLE_MAX
#      seconds; if the machine is not idle by then, the window stops there. Every run checks first, then times every variant in the same order.
#   3. After the three timed runs, its own step: one complete run under nsys (CUDA API and NVTX trace), in nsys/: whether the spin covers the
#      host's enqueue of every call of the kernel-only interval.
#   4. ncu: the four timed fused kernels (E18_PROFILE=1 E18_PROFILE_CALLS=1, E18_ONLY = the four per-call FKL variants: a warm-up call and one call
#      in its NVTX range each, so two launches per kernel), #43's metric set with ncu's locked clocks: the duration and the achieved and
#      theoretical occupancy (ncu/).
#   SAMPLER=1: nvidia-smi every 100 ms through the whole window (gpu_clocks.csv), on core 14.
# Each folder is new; nothing is overwritten. RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first.
# Output: results/rtx4090/e18-box/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-e18-box; source $W/scripts/os_kernel_lines.sh
NEW=$W/build/local-rtx4090/bin; STEP0=$T/scratch/e18-box-step0
NEWC=7dca5106d2f6b48ea030a1f3f04b70872b28b86e
OUT=$W/results/rtx4090/e18-box; LOG=$FB/results/rtx4090/e0-run.log
PROG=e18_box_cu
NSYS=$T/toolchains/cuda-13.4.3/bin/nsys; NCU=$T/toolchains/cuda-13.4.3/bin/ncu
WARMUP=${WARMUP:?set WARMUP=0 or 1}; SAMPLER=${SAMPLER:?set SAMPLER=0 or 1}
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW/$PROG || exit 1
# the binary is the one step 0 checked (sha256), and it carries the expected commit without -dirty
grep -q 'step 0 ended: all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
grep -E "fkl-benchmarks-e18-box/build/local-rtx4090/bin/$PROG\$" $STEP0/binary-sha256.txt | sha256sum -c --quiet || { echo "the binary differs from step 0's record: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -x $NEWC || { echo "$PROG: not the expected commit: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$PROG: a -dirty commit: not starting"; exit 1; }
export TMPDIR=$T/scratch/e18-box/tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $W/scripts/quiet_check.sh
# foreign_gpu: a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session)
foreign_gpu() { nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/'; }
busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(foreign_gpu)
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e18-box starts (WARMUP=$WARMUP SAMPLER=$SAMPLER IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $W/external/fkl $W/external/FastNPP $W/external/cvGPUSpeedup $W/external/fkl/thread-pool; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $W/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
SAMPLER_PID=""
if [ "$SAMPLER" = 1 ]; then
  taskset -c 14 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 \
    > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
  SAMPLER_PID=$!
fi
stop_helpers() {
  [ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
  kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
}
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# wait_idle <name>: waits (5 s steps) for no measurement, build or compile job, a 1-minute load below IDLE_LOAD and no other program on the RTX
# 4090; returns 1 if that does not happen within IDLE_MAX seconds
IDLE_NOTE=""
wait_idle() {
  local name=$1 t0=$(date +%s) load busy gpu
  while :; do
    load=$(cut -d' ' -f1 /proc/loadavg); busy=$(busy_processes); gpu=$(foreign_gpu)
    if [ -z "$busy" ] && [ -z "$gpu" ] && awk -v l=$load -v m=$IDLE_LOAD 'BEGIN { exit !(l < m) }'; then
      IDLE_NOTE="idle after $(( $(date +%s) - t0 )) s (1-minute load $load)"; return 0
    fi
    if [ $(( $(date +%s) - t0 )) -ge $IDLE_MAX ]; then
      echo "# $(date -Iseconds) $name: the machine is not idle after $IDLE_MAX s (1-minute load $load; busy: $(echo $busy | cut -c1-150); GPU: $gpu)"
      return 1
    fi
    sleep 5
  done
}
# run <run folder> <prefix command, or -> [VAR=value ...]: pinned to cores 0-3,16-19
run() {
  local name=$1 prefix=$2; shift 2
  [ "$prefix" = "-" ] && prefix=""
  local dir=$OUT/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)${IDLE_NOTE:+; $IDLE_NOTE}" > $dir/load.txt
  echo "# $(date -Iseconds) $name $prefix $*${IDLE_NOTE:+ ($IDLE_NOTE)}"
  IDLE_NOTE=""
  ( cd $dir && env $GENV "$@" timeout 180 taskset -c 0-3,16-19 $prefix $NEW/$PROG > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks: $(grep -c 'E18 box checks passed' $dir/stdout.txt)"
}
[ "$WARMUP" = 1 ] && run warmup -
for n in 1 2 3; do
  wait_idle quiet-run$n || { stop_helpers; echo "# $(date -Iseconds) GPU window e18-box STOPPED before quiet-run$n (not idle)" | tee -a $LOG; exit 1; }
  run quiet-run$n -
done
# 3. the nsys pass, after the three timed runs: a complete run (checks and timing) under the CUDA API and NVTX trace
run nsys "$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false -o $OUT/nsys/e18-box"
# 4. the ncu pass: the four timed fused kernels, two launches each (the profile mode's warm-up call and its one call in the NVTX range), #43's
#    metric set (E1's) and the LaunchStats and Occupancy sections; ncu locks the clocks to base (its default --clock-control base)
run ncu "$NCU --csv --page raw --print-kernel-base mangled --kernel-name-base mangled --kernel-name regex:linearFilterDPPKernel --metrics gpu__time_duration.avg,gpu__time_duration.max,gpu__time_duration.min,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_active.max.pct_of_peak_sustained_active,sm__warps_active.min.pct_of_peak_sustained_active,sm__warps_active.sum.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,sm__maximum_warps_avg_per_active_cycle,smsp__maximum_warps_avg_per_active_cycle --section LaunchStats --section Occupancy" E18_PROFILE=1 E18_PROFILE_CALLS=1 E18_ONLY=fkl_box_asym,fkl_box_halfpixel,fkl_trunc_asym,fkl_trunc_halfpixel
stop_helpers
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e18-box ended" | tee -a $LOG
echo "e18-box window end $(date -Iseconds)"
