#!/bin/bash
# GPU window "e19-crop-sweep-timing" (quiet; about 15-30 minutes with the idle waits; the desktop session keeps running): the timing of the E19
# crop sweep, e19_crop_sweep_cu of branch e19-crop-sweep at 246b1c8 (the build of 2026-11-24 08:00, sha256 2b800f1d..., in BIN_SHA256; the
# binary of the window e19-crop-sweep-checks, results/rtx4090/e19-crop-sweep-checks): E19's pipeline with N crops, N = 1, 10, 20, ..., 150
# (E19_CROPS), the rows fkl_loop, fkl_batch, opencv_cuda, npp in this order in every run. The protocol of the window e19-fkl-batch-timing
# (results/rtx4090/e19-batched-box-fkl-batch/window-script.sh), with a sweep in place of a run. A sweep is 16 processes, one per N, in
# ascending order (1, 10, ..., 150); every process is a complete run of the program (the checks, then the timing of every row), in its own
# folder <sweep>/N<NNN>/. The sweeps, in this order:
#   1. warmup: one complete sweep, not used.
#   2. quiet-run1, quiet-run2, quiet-run3: three complete sweeps, the times of the report.
#   3. nsys: one complete sweep under nsys (CUDA API and NVTX trace; reports <sweep>/N<NNN>/e19-crop-sweep-timing-N<NNN>.nsys-rep): the
#      kernels of every call, whether the spin covers the host's enqueue of every kernel_only call, and the clock per row. Its times are not used.
# Before EVERY run (all 80), wait_idle: no measurement, build or compile job, a 1-minute load below IDLE_LOAD and no other program on the RTX
# 4090, up to IDLE_MAX seconds; if the machine is not idle by then, the window stops there. Every run is pinned to cores 0-3,16-19.
# Every run must exit 0, report "E19 batched box checks passed" and write a complete timing CSV: the window stops at the first failed run.
# No ncu. No clock warm-up inside the program. No dump (E19_DUMP is not set). SAMPLER: nvidia-smi every 100 ms through the whole window
# (gpu_clocks.csv), on core 14. RTX 4090 = nvidia-smi index 1. Each folder is new; nothing is overwritten.
# Guards before anything starts: no blocked program; the binary's sha256 is BIN_SHA256; it carries the commit 246b1c8 and no -dirty commit;
# HEAD is 246b1c8 or a descendant that differs from it only under results/, with no local change outside results/; this script is the one
# committed at HEAD (results/rtx4090/e19-crop-sweep-phase2/window-timing.sh) and runs from there; the output folder is the requested one
# (results/rtx4090/e19-crop-sweep) and new; no measurement, build or compile job runs and no other program uses the RTX 4090.
# After the GPU steps, CPU only: the 16 nsys reports exported to SQLite into the scratch folder (not committed).
# Output: results/rtx4090/e19-crop-sweep/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-e19-crop-sweep; source $W/scripts/os_kernel_lines.sh
NEW=$W/build/local-rtx4090/bin; S=$T/scratch/e19-crop-sweep
NEWC=246b1c8c9267b850bb3aa19c904172eb8f6ffad8
OUT=$W/results/rtx4090/e19-crop-sweep; LOG=$FB/results/rtx4090/e0-run.log
# the record's folder is the one named in the grant request, so that no rename is needed after the window
REQUESTED_OUT=results/rtx4090/e19-crop-sweep
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
PROG=e19_crop_sweep_cu
NSYS=$T/toolchains/cuda-13.4.3/bin/nsys
CROPS_LIST="1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW/$PROG || exit 1
# the binary is the one of the build record (sha256), it carries the expected commit without -dirty, and nothing outside results/ changed since
# that commit (HEAD differs from it only under results/, and the worktree has no local change outside results/)
BIN_SHA256=2b800f1d5626a6a32523cbfe50b960668cc3f15386f03b7be442df1cdbb26c8c
echo "$BIN_SHA256  $NEW/$PROG" | sha256sum -c --quiet || { echo "the binary differs from the build record: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -x $NEWC || { echo "$PROG: not the expected commit: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$PROG: a -dirty commit: not starting"; exit 1; }
git -C $W merge-base --is-ancestor $NEWC HEAD || { echo "HEAD is not $NEWC or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $NEWC HEAD -- . ':!results' || { echo "HEAD differs from $NEWC outside results/: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
# this script is the committed one, run from its place in the worktree
SELF=results/rtx4090/e19-crop-sweep-phase2/window-timing.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
git -C $W ls-files --error-unmatch $SELF > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $SELF || { echo "$SELF is not the version committed at HEAD: not starting"; exit 1; }
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
EXPORT=$S/nsys-export-timing
[ -e $EXPORT ] && { echo "$EXPORT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT $EXPORT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e19-crop-sweep-timing starts (IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
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
  echo "# $(date -Iseconds) GPU window e19-crop-sweep-timing $1" | tee -a $LOG
  echo "e19-crop-sweep-timing window end $(date -Iseconds)"
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
  ( cd $dir && env $GENV "$@" timeout 300 taskset -c 0-3,16-19 $prefix $NEW/$PROG > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks passed: $(grep -c 'E19 batched box checks passed' $dir/stdout.txt)"
}
# stop_at_failure <run folder>: a run that did not exit 0, did not pass its checks, or wrote no complete checks or timing CSV ends the window here
stop_at_failure() {
  local dir=$OUT/$1 rc=$(cat $OUT/$1/exit_code.txt 2>/dev/null)
  if [ "$rc" = 0 ] && grep -q 'E19 batched box checks passed' $dir/stdout.txt \
     && [ -s $dir/e19_batched_box_checks.csv ] && ! grep -q '^# INCOMPLETE' $dir/e19_batched_box_checks.csv \
     && [ -s $dir/e19_batched_box.csv ] && ! grep -q '^# INCOMPLETE' $dir/e19_batched_box.csv; then
    return 0
  fi
  finish "STOPPED after $1 (exit ${rc:-missing})"
  exit 1
}
# sweep <sweep folder> <nsys or ->: the 16 processes in ascending N, each after wait_idle, each stopping the window if it fails
sweep() {
  local name=$1 mode=$2 n d prefix
  for n in $CROPS_LIST; do
    d=$name/N$(printf %03d $n)
    wait_idle $d || { finish "STOPPED before $d (not idle)"; exit 1; }
    prefix=-
    [ "$mode" = nsys ] && prefix="$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false -o $OUT/$d/e19-crop-sweep-timing-N$(printf %03d $n)"
    run $d "$prefix" E19_CROPS=$n
    stop_at_failure $d
  done
  echo "# $(date -Iseconds) sweep $name done"
}
# 1. the warm-up sweep (not used)
sweep warmup -
# 2. the three quiet sweeps
for k in 1 2 3; do sweep quiet-run$k -; done
# 3. the nsys sweep, after the three timed sweeps (its times are not used)
sweep nsys nsys
finish "ended (GPU steps)"
# CPU only, after the GPU steps: the SQLite export of the 16 traces, into the scratch folder (not committed)
for f in $OUT/nsys/N*/e19-crop-sweep-timing-N*.nsys-rep; do
  b=$(basename $f .nsys-rep)
  $NSYS export --type sqlite --output $EXPORT/$b.sqlite $f >> $EXPORT/export.log 2>&1
  rc=$?
  echo "nsys export $b exit $rc"
  [ "$rc" = 0 ] || exit 1
done
