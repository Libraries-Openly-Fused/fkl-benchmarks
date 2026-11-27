#!/bin/bash
# GPU window "e16-retime-304492a" (quiet; about 40-45 minutes with the idle waits; the desktop session keeps running): E16 timed again at the
# new pin of fkl-benchmarks, FKL 304492a (#435 as merged; #66), against the pin before it, FKL b10fc66. Task RT-2 (the coordinator, Oscar's
# order of 2026-11-26 for the pin move). The protocol of E16 in window D (results/rtx4090/cleanup-window-d/window-script.sh, step 1), the
# run behind the paper's E16 data (cleanup-window-d/cleaned/e16-summary/run1..3), with FKL A and FKL B where window D had the reference and
# the cleaned build, and with RT-1's checks around every run. No new build in the window: the binaries are those of RT-1
# (results/rtx4090/e19-crop-sweep-retime-304492a/), compile-only builds of scripts/check_fkl_tree.sh:
#   A = scratch/pin-435/base-b10fc66/bench-build/bin/e16_attention_cu: fkl-benchmarks 976e42b, FKL b10fc66 (#68's identity build);
#   B = scratch/pin-435/main-304492a/bench-build/bin/e16_attention_cu: fkl-benchmarks main 73a9650, FKL 304492a (RT-1's compile-B).
# The Python baselines are pytorch-attention/e16_baselines.py of this worktree (equal to window D's cleaned script, f27c605's), with
# venv-torch and scripts/triton_env.sh, as in window D. The sha256 of the two binaries and the two scripts are in manifest.sha256 next to
# this script, whose own sha256 is MANIFEST_SHA256 below. The source of the builds is a link farm without .git, so their build information
# names the repository commit "unknown"; they carry the FKL commit (A b10fc66, B 304492a).
# Three rounds r = 0, 1, 2 (E16_ROTATE = r for the FKL program, --rotate r for the Python baselines), each run in its own folder:
#   round 0: FKL A, FKL B, Python A, Python B; round 1: the reverse order (Python B, Python A, FKL B, FKL A); round 2: as round 0.
#   Folders: {A,B}/e16-attention/run<r+1> (e16_attention.csv) and {A,B}/e16-baselines/run<r+1> (e16_baselines.csv). The Python baselines run
#   once per side and round, at the place of window D's reference and cleaned script; both sides run the same script.
# Every run reads the same input bytes: the 10 files of e16-work/inputs (one per configuration, written in window D's pre-window step), the
# FKL program through E16_INPUT_DIR, the Python script through --input-dir; their sha256 are checked against window D's record
# (results/rtx4090/cleanup-window-d/inputs-sha256.txt, committed) before the start. The query tile is not run.
# Before EVERY run, wait_idle: no measurement, build or compile job, a 1-minute load below IDLE_LOAD and no other program on the RTX 4090, up
# to IDLE_MAX seconds; if the machine is not idle by then, the window stops there. AFTER every run, still_quiet: if a build, a compile job
# or another program on the RTX 4090 appeared during the run (fkl-reduce has priority on this machine), the window stops there and says so,
# and that run is the affected one (Oscar 2026-11-26: the affected runs are redone). Every run is pinned to cores 0-3,16-19, with window D's
# timeouts (1800 s for the FKL program, 3600 s for Python). Every run must exit 0 and write its CSV with data rows and no "# INCOMPLETE":
# the window stops at the first failed run. TMPDIR is on disk (scratch/rt2/tmp), for the Python runs too (the RAM disk /tmp is not used).
# SAMPLER: nvidia-smi every 100 ms through the whole window (gpu_clocks.csv, the fields of window D's, for scripts/e16_join_clock.py), on
# core 14; gpu-at-end.csv after every run. RTX 4090 = nvidia-smi index 1. Nothing is overwritten.
# Guards before anything starts: no blocked program; the manifest is the committed one (MANIFEST_SHA256), names the four files and all
# match it; A carries FKL b10fc66 and B FKL 304492a, and neither a -dirty commit; the Python script is f27c605's; the 10 input files match
# window D's record and the folder holds nothing else; HEAD is main 3f641d9 or a descendant that differs from it only under results/, with
# no local change outside results/; this script and the manifest are the ones committed at HEAD and the script runs from there; the output
# folder is the requested one (results/rtx4090/e16-retime-304492a/window) and new; no measurement, build or compile job runs and no other
# program uses the RTX 4090. GUARD_ONLY=1 runs the guards only.
# Output: results/rtx4090/e16-retime-304492a/window/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-rt2; source $W/scripts/os_kernel_lines.sh
S=$T/scratch/rt2; P=$T/scratch/pin-435; INPUTS=$T/e16-work/inputs; PY=$T/venv-torch/bin/python
BINA=$P/base-b10fc66/bench-build/bin/e16_attention_cu; BINB=$P/main-304492a/bench-build/bin/e16_attention_cu
PY16=$W/pytorch-attention/e16_baselines.py
FKLA=b10fc6609426bd18020b9213c7196b528fc677a8; FKLB=304492af4d777165bd556fdd6113439635724d02
MAIN=3f641d9fbd22bb1607452b7d766abf94d67efe07; PYC=f27c6051c6d2a94e9f7f6e32307abcc1b3c43ec1
REC=results/rtx4090/e16-retime-304492a
OUT=$W/$REC/window; LOG=$FB/results/rtx4090/e0-run.log
REQUESTED_OUT=results/rtx4090/e16-retime-304492a/window
[ "$OUT" = "$W/$REQUESTED_OUT" ] || { echo "the output folder $OUT is not the requested $W/$REQUESTED_OUT: not starting"; exit 1; }
MANIFEST=$W/$REC/manifest.sha256; MANIFEST_SHA256=880b5afc848e86ce98f789ce2a658546c6803845105ee45c042e48c38d1860de
INPUTS_SHA=$W/results/rtx4090/cleanup-window-d/inputs-sha256.txt; INPUTS_SHA_SHA256=5158c8f812d79eaeb2110832e0d3698a9181b2e1641cf55caee4fae0d012d8de
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $BINA $BINB || exit 1
# the manifest is the committed one, and the four files are the ones it names
[ "$(sha256sum $MANIFEST | cut -d' ' -f1)" = "$MANIFEST_SHA256" ] || { echo "the manifest is not the committed one: not starting"; exit 1; }
[ "$(grep -c . $MANIFEST)" = 4 ] || { echo "the manifest does not name 4 files: not starting"; exit 1; }
for f in $BINA $BINB $PY16 $W/scripts/triton_env.sh; do grep -q -E "  $f\$" $MANIFEST || { echo "the manifest does not name $f: not starting"; exit 1; }; done
sha256sum -c --quiet $MANIFEST || { echo "a file differs from the manifest: not starting"; exit 1; }
strings $BINA | grep -q -x $FKLA || { echo "A does not carry FKL $FKLA: not starting"; exit 1; }
strings $BINB | grep -q -x $FKLB || { echo "B does not carry FKL $FKLB: not starting"; exit 1; }
strings $BINA $BINB | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "a -dirty commit: not starting"; exit 1; }
git -C $W diff --quiet $PYC HEAD -- pytorch-attention/e16_baselines.py scripts/triton_env.sh || { echo "the Python script or triton_env.sh differs from f27c605's: not starting"; exit 1; }
# the input files: window D's, and nothing else in the folder
[ "$(sha256sum $INPUTS_SHA | cut -d' ' -f1)" = "$INPUTS_SHA_SHA256" ] || { echo "window D's inputs-sha256.txt is not the committed one: not starting"; exit 1; }
( cd $INPUTS && sha256sum -c --quiet $INPUTS_SHA ) || { echo "an E16 input file differs from window D's record: not starting"; exit 1; }
[ "$(ls $INPUTS | wc -l)" = "$(grep -c . $INPUTS_SHA)" ] || { echo "e16-work/inputs holds other files: not starting"; exit 1; }
# HEAD differs from main only under results/; no local change outside results/
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not main $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results' || { echo "HEAD differs from main outside results/: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
# this script is the committed one, run from its place in the worktree, and the manifest is committed too
SELF=$REC/window-script.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
for f in $SELF $REC/manifest.sha256; do
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
echo "# $(date -Iseconds) GPU window e16-retime-304492a starts (IDLE_LOAD=$IDLE_LOAD IDLE_MAX=$IDLE_MAX); load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $W $T/worktrees/fkl-benchmarks-pin-435-base $T/worktrees/fkl-benchmarks-pin-435-base/external/fkl; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done
  echo "manifest:"; sha256sum -c $MANIFEST
  echo "inputs ($INPUTS):"; ( cd $INPUTS && sha256sum -c $INPUTS_SHA )
  for b in $BINA $BINB; do echo "$b: commits $(strings $b | grep -E '^[0-9a-f]{40}(-dirty)?$|^unknown$' | tr '\n' ' ')"; ldd $b | grep -E 'cudart|cublas|cudnn'; done
  echo "python: $($PY --version 2>&1); $($PY -c 'import torch, triton; print("torch", torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(), "triton", triton.__version__)' 2>&1)"
  $PY -m pip freeze 2>/dev/null | grep -i -E '^(torch|triton|flash|nvidia-cudnn|numpy)' ; } > $OUT/environment-before.txt 2>&1
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
  echo "# $(date -Iseconds) GPU window e16-retime-304492a $1" | tee -a $LOG
  echo "e16-retime-304492a window end $(date -Iseconds)"
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
data_rows() { grep -v -c -E '^#|^$' $1 2>/dev/null; }
# run <name> <folder> <csv> <timeout> <command...>: one pinned run in its own folder, with window D's environment
run() {
  local name=$1 dir=$2 csvf=$3 tmo=$4; shift 4
  mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)${IDLE_NOTE:+; $IDLE_NOTE}; $*" > $dir/load.txt
  echo "# $(date -Iseconds) $name $*${IDLE_NOTE:+ ($IDLE_NOTE)}"
  IDLE_NOTE=""
  # the Python runs with scripts/triton_env.sh, as window D's pyrun; the FKL runs without it, as window D's run
  ( cd $dir && if [ "$1" = "$PY" ]; then source $W/scripts/triton_env.sh > /dev/null 2>&1 || exit 1; fi
    env $GENV TMPDIR=$TMPDIR timeout $tmo taskset -c 0-3,16-19 "$@" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; $csvf data rows: $(data_rows $dir/$csvf); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); stderr $(stat -c %s $dir/stderr.txt) B"
}
# stop_at_failure <name> <folder> <csv>: a run that did not exit 0, or wrote no data row, or an INCOMPLETE CSV, ends the window here
stop_at_failure() {
  local rc=$(cat $2/exit_code.txt 2>/dev/null)
  if [ "$rc" = 0 ] && [ "$(data_rows $2/$3)" -gt 0 ] 2>/dev/null && ! grep -q '^# INCOMPLETE' $2/$3; then return 0; fi
  finish "STOPPED after $1 (exit ${rc:-missing})"
  exit 1
}
# one_run <name> <folder> <csv> <timeout> <command...>: wait_idle, the run, the failure check, still_quiet
one_run() {
  local name=$1 dir=$2 csvf=$3
  wait_idle $name || { finish "STOPPED before $name (not idle)"; exit 1; }
  run "$@"
  stop_at_failure $name $dir $csvf
  still_quiet $name || { finish "STOPPED after $name (another job appeared during it; $name is affected)"; exit 1; }
}
e16_fkl() {   # e16_fkl <side> <round>
  local bin=$BINA; [ $1 = B ] && bin=$BINB
  one_run $1/e16-attention/run$(( $2 + 1 )) $OUT/$1/e16-attention/run$(( $2 + 1 )) e16_attention.csv 1800 env E16_ROTATE=$2 E16_INPUT_DIR=$INPUTS $bin
}
e16_py() {    # e16_py <side> <round>
  one_run $1/e16-baselines/run$(( $2 + 1 )) $OUT/$1/e16-baselines/run$(( $2 + 1 )) e16_baselines.csv 3600 $PY $PY16 --out e16_baselines.csv --rotate $2 --input-dir $INPUTS
}
for r in 0 1 2; do
  echo "# $(date -Iseconds) E16 round $r (E16_ROTATE=$r, --rotate $r)"
  if [ $r = 1 ]; then e16_py B $r; e16_py A $r; e16_fkl B $r; e16_fkl A $r
  else e16_fkl A $r; e16_fkl B $r; e16_py A $r; e16_py B $r; fi
done
finish "ended (GPU steps)"
