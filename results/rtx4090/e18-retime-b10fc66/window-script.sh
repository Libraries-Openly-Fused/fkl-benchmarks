#!/bin/bash
# GPU window "e18-retime-b10fc66" (quiet; about 10-15 minutes with the idle waits; the desktop session keeps running): the E18 programs
# e18_box_cu, e18_box_split_cu and e18_stencil_cu timed again, at FKL 24478f0 (A) and at FKL b10fc66 (B), the merge of FKL #433 that changes
# the LinearFilterDPP kernel (its details lose width and height). No new build: the binaries are those of the compile-only identity check of
# fkl-benchmarks #60 (scratch/pin-433, scripts/check_fkl_tree.sh --only bench, the options of the reference build build/local-rtx4090):
#   A = scratch/pin-433/base-24478f0/bench-build/bin/<program>: fkl-benchmarks main a8f9daa, FKL 24478f0;
#   B = scratch/pin-433/head-b10fc66/bench-build/bin/<program>: fkl-benchmarks 2160c70 (#60's first commit: the pin and the details port), FKL b10fc66.
# Their sha256 are in binaries.sha256 next to this script, whose own sha256 is MANIFEST_SHA256 below. The source of these builds is a link
# farm without .git, so their build information names the repository commit "unknown"; they carry the FKL commit (A 24478f0, B b10fc66).
# Every program keeps its own timing protocol, rows and row order (the program is unchanged; no environment variable is set): each run is
# a complete run of the program, the checks first, then whole_call (fklb::timeIt) and kernel_only (fklb::timePreparedLaunch behind gpuSpin)
# of every row in the program's order. Per program, in this order, one process per run, each in its own folder <program>/<run>/:
#   warmup-A, warmup-B (not used), then quiet-run1-A, quiet-run1-B, quiet-run2-A, quiet-run2-B, quiet-run3-A, quiet-run3-B.
# The programs in this order: e18_box, e18_box_split, e18_stencil (24 runs). No nsys or ncu pass.
# Before EVERY run, wait_idle: no measurement, build or compile job, a 1-minute load below IDLE_LOAD and no other program on the RTX 4090, up
# to IDLE_MAX seconds; if the machine is not idle by then, the window stops there. Every run is pinned to cores 0-3,16-19, with a 180 s
# timeout (as in the three E18 windows). Every run must exit 0, print its "checks passed" line and write every CSV of the program complete
# (no "# INCOMPLETE", no -nan): the window stops at the first failed run. SAMPLER: nvidia-smi every 100 ms through the whole window
# (gpu_clocks.csv), on core 14, as in the crop sweep; gpu-at-end.csv after every run. RTX 4090 = nvidia-smi index 1. Nothing is overwritten.
# Guards before anything starts: no blocked program; the manifest is the committed one (MANIFEST_SHA256) and every binary matches it; A
# carries FKL 24478f0 and B FKL b10fc66, and no binary a -dirty commit; this script is the one committed at HEAD and runs from there; HEAD
# differs from main a8f9daa only under results/; the output folder is the requested one (results/rtx4090/e18-retime-b10fc66/window) and
# new; no measurement, build or compile job runs and no other program uses the RTX 4090. GUARD_ONLY=1 runs the guards only.
# Output: results/rtx4090/e18-retime-b10fc66/window/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-e18-retime; source $W/scripts/os_kernel_lines.sh
S=$T/scratch/e18-retime; P=$T/scratch/pin-433
BINA=$P/base-24478f0/bench-build/bin; BINB=$P/head-b10fc66/bench-build/bin
FKLA=24478f0aa11dc3f12feea83178f7fde828322fb0; FKLB=b10fc6609426bd18020b9213c7196b528fc677a8
MAIN=a8f9daa3cc529e8a13c88ae63fcbb227e89adf61
REC=results/rtx4090/e18-retime-b10fc66
OUT=$W/$REC/window; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/e18-retime-b10fc66/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MANIFEST=$W/$REC/binaries.sha256; MANIFEST_SHA256=b20a729a7d6aab35dedf21e4563b519de767517025820fcbadfe223b2df0f420
PROGS="e18_box e18_box_split e18_stencil"
declare -A CHECKS=([e18_box]="E18 box checks passed" [e18_box_split]="E18 box split checks passed" [e18_stencil]="E18 checks passed")
declare -A CSVS=([e18_box]="e18_box.csv e18_box_agreement.csv e18_box_checks.csv e18_box_kernel.csv"
                 [e18_box_split]="e18_box_split.csv e18_box_split_agreement.csv e18_box_split_checks.csv"
                 [e18_stencil]="e18_stencil.csv e18_stencil_agreement.csv e18_stencil_checks.csv")
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
for p in $PROGS; do refuse_blocked_binaries $BINA/${p}_cu $BINB/${p}_cu || exit 1; done
# the manifest is the committed one, and the six binaries are the ones it names
[ "$(sha256sum $MANIFEST | cut -d' ' -f1)" = "$MANIFEST_SHA256" ] || { echo "the manifest is not the committed one: not starting"; exit 1; }
[ "$(grep -c . $MANIFEST)" = 6 ] || { echo "the manifest does not name 6 binaries: not starting"; exit 1; }
for p in $PROGS; do grep -q -E "  $BINA/${p}_cu\$" $MANIFEST && grep -q -E "  $BINB/${p}_cu\$" $MANIFEST || { echo "the manifest does not name both ${p}_cu: not starting"; exit 1; }; done
sha256sum -c --quiet $MANIFEST || { echo "a binary differs from the manifest: not starting"; exit 1; }
for p in $PROGS; do
  strings $BINA/${p}_cu | grep -q -x $FKLA || { echo "A ${p}_cu does not carry FKL $FKLA: not starting"; exit 1; }
  strings $BINB/${p}_cu | grep -q -x $FKLB || { echo "B ${p}_cu does not carry FKL $FKLB: not starting"; exit 1; }
  strings $BINA/${p}_cu $BINB/${p}_cu | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "${p}_cu: a -dirty commit: not starting"; exit 1; }
done
# HEAD differs from main only under results/; no local change outside results/
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not main $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results' || { echo "HEAD differs from main outside results/: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
# this script is the committed one, run from its place in the worktree, and the manifest is committed too
SELF=$REC/window-script.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
for f in $SELF $REC/binaries.sha256; do
  git -C $W ls-files --error-unmatch $f > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $f || { echo "$f is not the version committed at HEAD: not starting"; exit 1; }
done
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
source $W/scripts/quiet_check.sh
# foreign_gpu: a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session)
foreign_gpu() { nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/'; }
busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(foreign_gpu)
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e18-retime-b10fc66 starts (IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $T/worktrees/fkl-benchmarks-pin-433-base $T/worktrees/fkl-benchmarks-pin-433-base/external/fkl $T/worktrees/fkl-benchmarks-pin-433 $T/worktrees/fkl-benchmarks-pin-433/external/fkl; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "binaries:"; sha256sum -c $MANIFEST
  for p in $PROGS; do for b in $BINA/${p}_cu $BINB/${p}_cu; do
    echo "$b: commits $(strings $b | grep -E '^[0-9a-f]{40}(-dirty)?$|^unknown$' | tr '\n' ' ')"; ldd $b | grep -E 'npp|cudart|opencv'; done; done; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window e18-retime-b10fc66 $1" | tee -a $LOG
  echo "e18-retime-b10fc66 window end $(date -Iseconds)"
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
# run <program> <run name> <binary>: one complete run of the binary, pinned to cores 0-3,16-19, in <program>/<run name>/
run() {
  local prog=$1 name=$2 bin=$3
  local dir=$OUT/$prog/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)${IDLE_NOTE:+; $IDLE_NOTE}; binary $bin" > $dir/load.txt
  echo "# $(date -Iseconds) $prog/$name $bin${IDLE_NOTE:+ ($IDLE_NOTE)}"
  IDLE_NOTE=""
  ( cd $dir && env $GENV timeout 180 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $prog/$name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks passed: $(grep -c -x "${CHECKS[$prog]}" $dir/stdout.txt)"
}
# stop_at_failure <program> <run name>: a run that did not exit 0, did not print its checks line, or lacks a complete CSV ends the window here
stop_at_failure() {
  local prog=$1 dir=$OUT/$1/$2 rc=$(cat $OUT/$1/$2/exit_code.txt 2>/dev/null) f ok=1
  [ "$rc" = 0 ] && grep -q -x "${CHECKS[$prog]}" $dir/stdout.txt || ok=0
  for f in ${CSVS[$prog]}; do
    [ -s $dir/$f ] && ! grep -q '^# INCOMPLETE' $dir/$f && ! grep -q -- '-nan' $dir/$f || ok=0
  done
  [ $ok = 1 ] && return 0
  finish "STOPPED after $prog/$2 (exit ${rc:-missing})"
  exit 1
}
# one_run <program> <run name> <binary>: wait_idle, the run, the failure check
one_run() {
  wait_idle $1/$2 || { finish "STOPPED before $1/$2 (not idle)"; exit 1; }
  run $1 $2 $3
  stop_at_failure $1 $2
}
for p in $PROGS; do
  one_run $p warmup-A $BINA/${p}_cu
  one_run $p warmup-B $BINB/${p}_cu
  for k in 1 2 3; do
    one_run $p quiet-run$k-A $BINA/${p}_cu
    one_run $p quiet-run$k-B $BINB/${p}_cu
  done
  echo "# $(date -Iseconds) $p done"
done
finish "ended"
