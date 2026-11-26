#!/bin/bash
# GPU window "e19-crop-sweep-retime-304492a" (quiet; about 20-30 minutes with the idle waits; the desktop session keeps running): the E19
# crop sweep of results/rtx4090/e19-crop-sweep/ timed again at the new pin of fkl-benchmarks, FKL 304492a (#435 as merged; #66), against the
# pin before it, FKL b10fc66. Task RT-1 (the coordinator, Oscar's order with Q3). No new build in the window: both binaries are
# e19_crop_sweep_cu of a compile-only build of scripts/check_fkl_tree.sh (a link farm of the checkout, the FKLB_* options, OpenCV and nvcc of
# the reference build build/local-rtx4090, which are those of the preset local-rtx4090 of the record's build):
#   A = scratch/pin-435/base-b10fc66/bench-build/bin/e19_crop_sweep_cu: fkl-benchmarks 976e42b, FKL b10fc66 (#68's identity build; its
#       sha256 is #68's record's);
#   B = scratch/pin-435/main-304492a/bench-build/bin/e19_crop_sweep_cu: fkl-benchmarks main 73a9650 (= 658a896 and #67, which changes only
#       results/rx9060/), FKL 304492a.
# Their sha256 are in binaries.sha256 next to this script, whose own sha256 is MANIFEST_SHA256 below. The source of these builds is a link
# farm without .git, so their build information names the repository commit "unknown"; they carry the FKL commit (A b10fc66, B 304492a).
# The record's protocol (results/rtx4090/e19-crop-sweep/window-script.sh), with A and B alternating as in #63 (results/rtx4090/e18-retime-
# b10fc66): a sweep is 16 processes, one per N (E19_CROPS = 1, 10, 20, ..., 150), in ascending order; every process is a complete run of the
# program (the checks, then whole_call and kernel_only of every row, in the order fkl_loop, fkl_batch, opencv_cuda, npp), in its own folder
# <sweep>/N<NNN>/. The sweeps, in this order (128 runs):
#   warmup-A, warmup-B (not used), then quiet-run1-A, quiet-run1-B, quiet-run2-A, quiet-run2-B, quiet-run3-A, quiet-run3-B.
# No nsys sweep and no ncu.
# Before EVERY run, wait_idle: no measurement, build or compile job, a 1-minute load below IDLE_LOAD and no other program on the RTX 4090, up
# to IDLE_MAX seconds; if the machine is not idle by then, the window stops there. AFTER every run, the same check without the load: if a build,
# a compile job or another program on the RTX 4090 appeared during the run (fkl-reduce has priority on this machine), the window stops there
# and says so, and that run is the affected one (Oscar 2026-11-26: the affected runs are redone). Every run is pinned to cores 0-3,16-19, with
# a 300 s timeout. Every run must exit 0, report "E19 batched box checks passed" and write complete checks and timing CSVs (no
# "# INCOMPLETE"): the window stops at the first failed run. No clock warm-up inside the program. No dump (E19_DUMP is not set). SAMPLER:
# nvidia-smi every 100 ms through the whole window (gpu_clocks.csv), on core 14; gpu-at-end.csv after every run. RTX 4090 = nvidia-smi index
# 1. Each folder is new; nothing is overwritten.
# Guards before anything starts: no blocked program; the manifest is the committed one (MANIFEST_SHA256), names the two binaries and both
# match it; A carries FKL b10fc66 and B FKL 304492a, and neither a -dirty commit; HEAD is main 73a9650 or a descendant that differs from it
# only under results/, with no local change outside results/; this script is the one committed at HEAD and runs from there; the output folder
# is the requested one (results/rtx4090/e19-crop-sweep-retime-304492a/window) and new; no measurement, build or compile job runs and no other
# program uses the RTX 4090. GUARD_ONLY=1 runs the guards only.
# Output: results/rtx4090/e19-crop-sweep-retime-304492a/window/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-rt1; source $W/scripts/os_kernel_lines.sh
S=$T/scratch/rt1; P=$T/scratch/pin-435
PROG=e19_crop_sweep_cu
BINA=$P/base-b10fc66/bench-build/bin/$PROG; BINB=$P/main-304492a/bench-build/bin/$PROG
FKLA=b10fc6609426bd18020b9213c7196b528fc677a8; FKLB=304492af4d777165bd556fdd6113439635724d02
MAIN=73a9650ea1a65daa92af97641d3bf0719682101e
REC=results/rtx4090/e19-crop-sweep-retime-304492a
OUT=$W/$REC/window; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/e19-crop-sweep-retime-304492a/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MANIFEST=$W/$REC/binaries.sha256; MANIFEST_SHA256=dfb6f9810506c48e19a4ddead549b8b923e2924dc0df09f17d581e17251b90fa
CROPS_LIST="1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $BINA $BINB || exit 1
# the manifest is the committed one, and the two binaries are the ones it names
[ "$(sha256sum $MANIFEST | cut -d' ' -f1)" = "$MANIFEST_SHA256" ] || { echo "the manifest is not the committed one: not starting"; exit 1; }
[ "$(grep -c . $MANIFEST)" = 2 ] || { echo "the manifest does not name 2 binaries: not starting"; exit 1; }
grep -q -E "  $BINA\$" $MANIFEST && grep -q -E "  $BINB\$" $MANIFEST || { echo "the manifest does not name both binaries: not starting"; exit 1; }
sha256sum -c --quiet $MANIFEST || { echo "a binary differs from the manifest: not starting"; exit 1; }
strings $BINA | grep -q -x $FKLA || { echo "A does not carry FKL $FKLA: not starting"; exit 1; }
strings $BINB | grep -q -x $FKLB || { echo "B does not carry FKL $FKLB: not starting"; exit 1; }
strings $BINA $BINB | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "a -dirty commit: not starting"; exit 1; }
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
echo "# $(date -Iseconds) GPU window e19-crop-sweep-retime-304492a starts (IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  echo "$W $(git -C $W rev-parse HEAD)"; git -C $W status --porcelain --untracked-files=no
  for b in $BINA $BINB; do echo "$b: $(sha256sum < $b | cut -c1-64); commits $(strings $b | grep -E '^[0-9a-f]{40}$|^unknown$' | sort -u | tr '\n' ' ')"
    echo "libraries:"; ldd $b | grep -E 'npp|cudart|opencv'; done; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window e19-crop-sweep-retime-304492a $1" | tee -a $LOG
  echo "e19-crop-sweep-retime-304492a window end $(date -Iseconds)"
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
# still_quiet <name>: after a run, no build or compile job and no other program on the RTX 4090 appeared (the load is not tested here: the run
# itself raises it); returns 1 and says what appeared otherwise
still_quiet() {
  local busy gpu
  busy=$(busy_processes); gpu=$(foreign_gpu)
  [ -z "$busy" ] && [ -z "$gpu" ] && return 0
  echo "# $(date -Iseconds) $1: after the run, another job is running (busy: $(echo $busy | cut -c1-150); GPU: $gpu): this run is affected"
  return 1
}
# run <run folder> <binary> [VAR=value ...]: pinned to cores 0-3,16-19
run() {
  local name=$1 bin=$2; shift 2
  local dir=$OUT/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)${IDLE_NOTE:+; $IDLE_NOTE}" > $dir/load.txt
  echo "# $(date -Iseconds) $name $bin $*${IDLE_NOTE:+ ($IDLE_NOTE)}"
  IDLE_NOTE=""
  ( cd $dir && env $GENV "$@" timeout 300 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
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
# sweep <sweep folder> <binary>: the 16 processes in ascending N, each after wait_idle and followed by still_quiet; each stops the window if
# it fails or another job appeared during it
sweep() {
  local name=$1 bin=$2 n d
  for n in $CROPS_LIST; do
    d=$name/N$(printf %03d $n)
    wait_idle $d || { finish "STOPPED before $d (not idle)"; exit 1; }
    run $d $bin E19_CROPS=$n
    stop_at_failure $d
    still_quiet $d || { finish "STOPPED after $d (another job appeared during it; $d is affected)"; exit 1; }
  done
  echo "# $(date -Iseconds) sweep $name done"
}
# 1. the warm-up sweeps (not used), A then B
sweep warmup-A $BINA
sweep warmup-B $BINB
# 2. the three quiet pairs, A then B
for k in 1 2 3; do
  sweep quiet-run$k-A $BINA
  sweep quiet-run$k-B $BINB
done
finish "ended (GPU steps)"
