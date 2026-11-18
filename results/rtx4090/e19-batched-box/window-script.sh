#!/bin/bash
# GPU window "e19-timing" (quiet, about 1 minute plus the idle waits; the desktop session keeps running): the timing of E19, the batched grey
# box pipeline on 30 crops of one frame, e19_batched_box_cu at 69df252 (the build of 2026-11-18 11:12, sha256 2aa1486c..., the binary of the
# window e19-fkl-recheck), with the rows fkl_loop, opencv_cuda, npp in this order in every run. The protocol of the windows e18-box and
# e18-box-split:
#   1. WARMUP=1: one complete run (the checks, then the timing), its files in warmup/, not used.
#   2. Three complete quiet runs, quiet-run1..3, pinned to cores 0-3,16-19. Before each, the script waits for an idle machine (wait_idle: no
#      measurement, build or compile job, a 1-minute load below IDLE_LOAD, no other program on the RTX 4090), up to IDLE_MAX seconds; if the
#      machine is not idle by then, the window stops there. Every run checks first, then times every row in the same order.
#   3. After the three timed runs, its own step: one complete run under nsys (CUDA API and NVTX trace), in nsys/: the kernels of every call,
#      their durations, whether the spin covers the host's enqueue of every call of the kernel-only interval, and the clock per row.
#   No ncu pass. No clock warm-up inside the program. No dump (E19_DUMP is not set).
#   SAMPLER: nvidia-smi every 100 ms through the whole window (gpu_clocks.csv), on core 14.
# Every run's exit code is checked, and the run must report "E19 batched box checks passed" and write its timing CSV: the window stops at a
# failed run. Each folder is new; nothing is overwritten. RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first.
# Output: results/rtx4090/e19-batched-box/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-e19-batched-box; source $W/scripts/os_kernel_lines.sh
NEW=$W/build/local-rtx4090/bin; S=$T/scratch/e19-batched-box
NEWC=69df252aadbd727a28d83663c3c10012412a05c4
OUT=$W/results/rtx4090/e19-batched-box; LOG=$FB/results/rtx4090/e0-run.log
PROG=e19_batched_box_cu
NSYS=$T/toolchains/cuda-13.4.3/bin/nsys
WARMUP=${WARMUP:?set WARMUP=0 or 1}
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW/$PROG || exit 1
# the binary is the one of the build record (sha256), it carries the expected commit without -dirty, and no source changed since that commit
grep -E "build/local-rtx4090/bin/$PROG\$" $S/binary-sha256-ab.txt | (cd $W && sha256sum -c --quiet) || { echo "the binary differs from the build record: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -x $NEWC || { echo "$PROG: not the expected commit: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$PROG: a -dirty commit: not starting"; exit 1; }
git -C $W merge-base --is-ancestor $NEWC HEAD || { echo "the worktree is not at $NEWC or after it: not starting"; exit 1; }
git -C $W diff --quiet $NEWC HEAD -- common cmake fkl-self opencv-cuda npp external CMakeLists.txt CMakePresets.json || { echo "a source changed after $NEWC: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- common cmake fkl-self opencv-cuda npp external CMakeLists.txt CMakePresets.json)" ] || { echo "the worktree has local source changes: not starting"; exit 1; }
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
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
EXPORT=$S/nsys-export-timing
[ -e $EXPORT ] && { echo "$EXPORT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT $EXPORT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e19-timing starts (WARMUP=$WARMUP IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $W/external/fkl $W/external/FastNPP $W/external/cvGPUSpeedup $W/external/fkl/thread-pool; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "libraries of $PROG:"; ldd $NEW/$PROG | grep -E 'npp|cudart|opencv'; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window e19-timing $1" | tee -a $LOG
  echo "e19-timing window end $(date -Iseconds)"
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
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks passed: $(grep -c 'E19 batched box checks passed' $dir/stdout.txt)"
}
# stop_at_failure <run folder>: a run that did not exit 0, did not pass its checks, or wrote no complete timing CSV ends the window here
stop_at_failure() {
  local dir=$OUT/$1 rc=$(cat $OUT/$1/exit_code.txt 2>/dev/null)
  if [ "$rc" = 0 ] && grep -q 'E19 batched box checks passed' $dir/stdout.txt && [ -s $dir/e19_batched_box.csv ] \
     && ! grep -q '^# INCOMPLETE' $dir/e19_batched_box.csv; then
    return 0
  fi
  finish "STOPPED after $1 (exit ${rc:-missing})"
  exit 1
}
if [ "$WARMUP" = 1 ]; then run warmup -; stop_at_failure warmup; fi
for n in 1 2 3; do
  wait_idle quiet-run$n || { finish "STOPPED before quiet-run$n (not idle)"; exit 1; }
  run quiet-run$n -
  stop_at_failure quiet-run$n
done
# 3. the nsys pass, after the three timed runs: a complete run (checks and timing) under the CUDA API and NVTX trace
run nsys "$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false -o $OUT/nsys/e19-timing"
stop_at_failure nsys
finish "ended (GPU steps)"
# CPU only, after the GPU steps: the SQLite export of the trace, into the scratch folder (not committed)
$NSYS export --type sqlite --output $EXPORT/e19-timing.sqlite $OUT/nsys/e19-timing.nsys-rep > $EXPORT/export.log 2>&1
echo "nsys export exit $?"
