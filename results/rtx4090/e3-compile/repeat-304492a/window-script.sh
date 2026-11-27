#!/bin/bash
# Window "e3-repeat-304492a" (quiet, compile only, no GPU work, about 55-60 minutes): the controlled compile repeat at the pin of
# fkl-benchmarks, FKL 304492a (#435 as merged; #66). Task RT-3 (the coordinator, Oscar's order of 2026-11-26 for the pin move): the protocol
# of results/rtx4090/e3-compile/repeat-pr423-9e20bd4e/ (the source of the paper's compile-time table), exactly, with scripts/e3_sweep.py
# unchanged: the distinct-operations chain (variant A) at N = 1, 100, 500, 1,000, 1,500, 2,000, 3,000 and 4,000, three compiles each in a
# row, ascending, one warm-up compile at N = 1 first (own CSV). One compile at a time, pinned to core 8 (second CCD, cores 8-15 and 24-31),
# nvcc --threads 1, the flags of the E3 sweeps; GNU time for the peak memory, nvcc --time for the phases, the size of the cubin. Everything
# else of mine (the python parent of the sweep, cuobjdump, the watcher) runs on cores 4-7,20-23 (first CCD), the load logger on core 5.
# What differs from that repeat (README):
#   - the library: this branch's external/fkl, the pin 304492a (a clean submodule checkout), given to the sweep as --include and named in
#     the CSV headers through FKLB_FKL_TREE, as the repeat named its tree;
#   - the generated translation units go to scratch/rt3/work (the repeat: the reference checkout's build/e3-repeat-pr423);
#   - TMPDIR is on disk, scratch/rt3/tmp, for nvcc and the sweep (the repeat set none: /tmp, the RAM disk; Oscar's rule of 2026-11-03);
#   - the checks around the compiles: wait_idle before the warm-up and before the sweep (no measurement, build or compile job, no other
#     program on the RTX 4090, a 1-minute load below IDLE_LOAD, up to IDLE_MAX s); still_quiet after each; and a WATCHER during both, every
#     5 s: a build or compile job that is not one of the sweep's own processes, or a program on the RTX 4090, stops the sweep at once (the
#     window ends there, says what appeared and which compile was running: that compile and the ones after it are to be redone; fkl-reduce
#     has priority on this machine, Oscar 2026-11-26).
# Guards before anything starts: this script is the one committed at HEAD and runs from its place; HEAD is main d66e854 or a descendant that
# differs from it only under results/, with no local change outside results/; the scripts the window runs equal their manifest
# (scripts-manifest.sha256, sha256 MANIFEST_SHA256); external/fkl is at 304492a, clean; nvcc is 13.4.92; the output folder
# (results/rtx4090/e3-compile/repeat-304492a/window) and the work folder are new; no measurement, build or compile job runs and no other
# program uses the RTX 4090. GUARD_ONLY=1 runs the guards only. No GPU program runs in this window, and no program is executed: the
# sweep only compiles.
# E3_* variables: only for testing the script on tiny inputs (E3_NS, E3_REPS, E3_OUT, E3_WORK, E3_LOG); the window uses the defaults.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-rt3; source $W/scripts/os_kernel_lines.sh
S=$T/scratch/rt3; PY=$T/venv-torch/bin/python
MAIN=d66e854828aed8187277673e43e8c3239115b0dc; FKLC=304492af4d777165bd556fdd6113439635724d02
REC=results/rtx4090/e3-compile/repeat-304492a
OUT=${E3_OUT:-$W/$REC/window}; WORK=${E3_WORK:-$S/work}; LOG=${E3_LOG:-$FB/results/rtx4090/e0-run.log}
NS=${E3_NS:-1,100,500,1000,1500,2000,3000,4000}; REPS=${E3_REPS:-3}; FKLTREE=$W/external/fkl
CORE=8; AUX=4-7,20-23
MANIFEST=$W/$REC/scripts-manifest.sha256; MANIFEST_SHA256=f9e5188af3fc6d706c7745b0f7d2f074c049e3fef79b998ac5c541bc8efd446d
IDLE_LOAD=${IDLE_LOAD:-0.5}; IDLE_MAX=${IDLE_MAX:-300}
# this script is the committed one, run from its place in the worktree, and the manifest is committed too
SELF=$REC/window-script.sh
[ "$(readlink -f "$0")" = "$(readlink -f $W/$SELF)" ] || { echo "this script is not $W/$SELF: not starting"; exit 1; }
for f in $SELF $REC/scripts-manifest.sha256; do
  git -C $W ls-files --error-unmatch $f > /dev/null 2>&1 && git -C $W diff --quiet HEAD -- $f || { echo "$f is not the version committed at HEAD: not starting"; exit 1; }
done
git -C $W merge-base --is-ancestor $MAIN HEAD || { echo "HEAD is not main $MAIN or a descendant of it: not starting"; exit 1; }
git -C $W diff --quiet $MAIN HEAD -- . ':!results' || { echo "HEAD differs from main outside results/: not starting"; exit 1; }
[ -z "$(git -C $W status --porcelain --untracked-files=no -- . ':!results')" ] || { echo "the worktree has local changes outside results/: not starting"; exit 1; }
[ "$(sha256sum $MANIFEST | cut -d' ' -f1)" = "$MANIFEST_SHA256" ] || { echo "the scripts manifest is not the committed one: not starting"; exit 1; }
( cd $W && sha256sum -c --quiet $MANIFEST ) || { echo "a script differs from the manifest: not starting"; exit 1; }
[ "$(git -C $FKLTREE rev-parse HEAD)" = $FKLC ] && [ "$(git -C $W ls-tree HEAD external/fkl | awk '{print $3}')" = $FKLC ] || { echo "external/fkl is not at 304492a: not starting"; exit 1; }
[ -z "$(git -C $FKLTREE status --porcelain --untracked-files=all)" ] || { echo "external/fkl has local changes: not starting"; exit 1; }
nvcc --version | grep -q 'V13.4.92' || { echo "nvcc is not 13.4.92: not starting"; exit 1; }
source $W/scripts/quiet_check.sh
# foreign_gpu: a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session)
foreign_gpu() { nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/'; }
busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(foreign_gpu)
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -e $WORK ] && { echo "$WORK exists: not starting"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
mkdir -p $OUT $WORK
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) window e3-repeat-304492a starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked)" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ echo "# $(date -Iseconds) before the window"; os_kernel_lines; uptime; cat /proc/loadavg; free -m | head -2
  echo "scaling governor, core $CORE: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unavailable); current frequency: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo unavailable) kHz; boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
  echo "TMPDIR: $TMPDIR ($(df -h --output=fstype,target $TMPDIR | tail -1))"
  ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "fkl-benchmarks: $W $(git -C $W rev-parse HEAD); tracked files that differ from it:"; git -C $W status --porcelain --untracked-files=no
  echo "FKL tree compiled against (the pin, external/fkl): $FKLTREE $(git -C $FKLTREE rev-parse HEAD); files that differ: $(git -C $FKLTREE status --porcelain --untracked-files=all | wc -l)"
  echo "scripts:"; ( cd $W && sha256sum -c $MANIFEST )
  nvcc --version | tail -2; g++ --version | head -1; echo "python: $($PY --version)"; } > $OUT/environment-before.txt 2>&1
taskset -c 5 nice -n 19 python3 $W/scripts/load_logger.py --log $OUT/load.log --interval 20 --bench-cores $CORE,$((CORE + 16)) --ccd0 8-15,24-31 > /dev/null 2>&1 &
LOGGER_PID=$!
# descendants <pid>: the pid and every process below it
descendants() { ps -eo pid=,ppid= | awk -v root=$1 '{ p[$1] = $2 } END { n = 0; s[root] = 1; do { c = 0; for (x in p) if (!(x in s) && (p[x] in s)) { s[x] = 1; c = 1 } } while (c); for (x in s) print x }'; }
# foreign_busy <pid>: busy_processes without the processes of the sweep (pid and below)
foreign_busy() { local own=" $(descendants $1 | tr '\n' ' ') "; busy_processes | while read -r line; do case "$own" in *" ${line%% *} "*) ;; *) echo "$line";; esac; done; }
# watch <pid> <name>: every 5 s, a build or compile job that is not the sweep's, or a program on the RTX 4090: kills the sweep and says so
watch_sweep() {
  local pid=$1 name=$2 b g
  while kill -0 $pid 2> /dev/null; do
    b=$(foreign_busy $pid); g=$(foreign_gpu)
    if [ -n "$b$g" ]; then
      { echo "# $(date -Iseconds) WATCHER: during $name another job is running (busy: $(echo $b | cut -c1-200); GPU: $g)"
        echo "# the sweep's compile then: $(ps -o args= -p $(descendants $pid | tr '\n' ',' | sed 's/,$//') 2>/dev/null | grep -m1 -E '(^| )nvcc ' | cut -c1-200)"; } | tee -a $OUT/watcher.log
      touch $OUT/DISTURBED
      kill -- $(descendants $pid) 2> /dev/null
      return
    fi
    sleep 5
  done
}
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
still_quiet() {
  local busy gpu
  busy=$(busy_processes); gpu=$(foreign_gpu)
  [ -z "$busy" ] && [ -z "$gpu" ] && return 0
  echo "# $(date -Iseconds) $1: after it, another job is running (busy: $(echo $busy | cut -c1-150); GPU: $gpu): it is affected"
  return 1
}
finish() {
  kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
  { echo "# $(date -Iseconds) after the window"; uptime; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
  { echo "# sha256 of the translation units that were compiled"; sha256sum $WORK/*.cu; } > $OUT/tu-sha256.txt 2>&1
  echo "# $(date -Iseconds) window e3-repeat-304492a $1" | tee -a $LOG
  echo "e3-repeat-304492a window end $(date -Iseconds)"
}
cd $W
export FKLB_FKL_TREE=$FKLTREE      # the header of the CSVs names the tree that is compiled against
SWEEP="$PY scripts/e3_sweep.py --variants A --core $CORE --tree 304492a --include $FKLTREE/include --out $OUT --work $WORK --keep-time-files"
# step <name> <sweep arguments...>: wait_idle, the sweep in the background with the watcher, still_quiet; stops the window on a failure
step() {
  local name=$1; shift
  wait_idle $name || { finish "STOPPED before $name (not idle)"; exit 1; }
  echo "# $(date -Iseconds) start $name: $SWEEP $* ($IDLE_NOTE)" | cut -c1-300; IDLE_NOTE=""
  taskset -c $AUX $SWEEP "$@" &
  local pid=$!
  watch_sweep $pid $name &
  local wpid=$!
  taskset -cp $AUX $wpid > /dev/null 2>&1   # the watcher on the first CCD, with the sweep's parent
  wait $pid; local rc=$?
  kill $wpid 2> /dev/null; wait $wpid 2> /dev/null
  echo "# $(date -Iseconds) end $name (exit $rc)"
  [ -e $OUT/DISTURBED ] && { finish "STOPPED during $name (another job appeared; see watcher.log)"; exit 1; }
  [ $rc = 0 ] || { finish "STOPPED after $name (exit $rc)"; exit 1; }
  still_quiet $name || { finish "STOPPED after $name (another job appeared during it)"; exit 1; }
}
echo "# warm-up: one compile of N = 1 (fills the file cache; recorded in its own CSV)"
step warm-up --ns 1 --reps 1 --csv e3_repeat_warmup_A.csv
echo "# distinct operations (variant A) at N = $NS, $REPS compiles each"
step sweep --ns $NS --reps $REPS --rep-threshold 1e9 --csv e3_repeat_A.csv
echo "# $(date -Iseconds) the set is done"
finish "ended"
