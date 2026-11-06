#!/bin/bash
# Window "e3-repeat-pr423" (quiet, compile only, no GPU work, about 55 minutes): the controlled compile repeat at the head of FusedKernelLibrary PR #423 (9e20bd4e5286be70fbeb0f79a19b96f00c81a06f, a clone in scratch/pr423/fkl-head), the library as it will be after the merge.
# The set of results/rtx4090/e3-compile/repeat-c8fcbe8 without its StaticLoop part: the distinct-operations chain (variant A) at N = 1, 100, 500, 1,000, 1,500, 2,000, 3,000 and 4,000, three compiles each, one warm-up compile first (own CSV).
# One compile at a time, pinned to core 8 (second CCD, cores 8-15 and 24-31), nvcc --threads 1, the flags of the E3 sweeps (scripts/e3_sweep.py, unchanged); GNU time for the peak memory, nvcc --time for the phases, the size of the cubin.
# Everything else of mine (the python parent of the sweep, cuobjdump, the watcher) runs on cores 4-7,20-23 (first CCD), the load logger on core 5. START_DELAY seconds are waited first (the start time announced is computed from the launch).
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
# E3_* variables: only for testing the script on tiny inputs; the window uses the defaults
OUT=${E3_OUT:-$FB/results/rtx4090/e3-compile/repeat-pr423-9e20bd4e}; LOG=${E3_LOG:-$FB/results/rtx4090/e0-run.log}; PY=$T/venv-torch/bin/python
NS=${E3_NS:-1,100,500,1000,1500,2000,3000,4000}; REPS=${E3_REPS:-3}; FKLTREE=${E3_FKL:-$T/scratch/pr423/fkl-head}
CORE=${E3_CORE:-8}; AUX=4-7,20-23
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) window e3-repeat-pr423 starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked)" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ echo "# $(date -Iseconds) before the window"; os_kernel_lines; uptime; cat /proc/loadavg; free -m | head -2
  echo "scaling governor, core $CORE: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unavailable); current frequency: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo unavailable) kHz; boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
  ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "fkl-benchmarks: $(git -C $FB rev-parse HEAD); tracked files that differ from it:"; git -C $FB status --porcelain --untracked-files=no
  echo "FKL tree compiled against (PR #423 head): $FKLTREE $(git -C $FKLTREE rev-parse HEAD); tracked files that differ: $(git -C $FKLTREE status --porcelain --untracked-files=no | wc -l); the pin external/fkl of the repository: $(git -C $FB/external/fkl rev-parse HEAD) (not used here)"
  nvcc --version | tail -2; g++ --version | head -1; echo "python: $($PY --version)"; } > $OUT/environment-before.txt 2>&1
taskset -c 5 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 20 --bench-cores $CORE,$((CORE + 16)) --ccd0 8-15,24-31 > /dev/null 2>&1 &
LOGGER_PID=$!
cd $FB
run() { echo "# $(date -Iseconds) start: $*" | cut -c1-260; "$@"; local rc=$?; echo "# $(date -Iseconds) end (exit $rc)"; return $rc; }
export FKLB_FKL_TREE=$FKLTREE      # the header of the CSVs names the tree that is compiled against (not the pin external/fkl)
SWEEP="taskset -c $AUX $PY scripts/e3_sweep.py --variants A --core $CORE --tree 9e20bd4e --include $FKLTREE/include --out $OUT --work ${E3_WORK:-$FB/build/e3-repeat-pr423} --keep-time-files"
echo "# warm-up: one compile of N = 1 (fills the file cache; recorded in its own CSV)"
run $SWEEP --ns 1 --reps 1 --csv e3_repeat_warmup_A.csv
echo "# distinct operations (variant A) at N = $NS, $REPS compiles each"
run $SWEEP --ns $NS --reps $REPS --rep-threshold 1e9 --csv e3_repeat_A.csv
echo "# $(date -Iseconds) the set is done"
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ echo "# $(date -Iseconds) after the window"; uptime; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
{ echo "# sha256 of the translation units that were compiled"; sha256sum ${E3_WORK:-$FB/build/e3-repeat-pr423}/*.cu; } > $OUT/tu-sha256.txt 2>&1
echo "# $(date -Iseconds) window e3-repeat-pr423 ended" | tee -a $LOG
echo "e3-repeat-pr423 window end $(date -Iseconds)"
