#!/bin/bash
# Window "e3-repeat" (quiet, compile only, no GPU work, about 75 minutes): the controlled compile repeat at the final pin of fkl-benchmarks (external/fkl at c8fcbe8, the library as the paper uses it).
# One compile at a time, pinned to core 8 (second CCD, cores 8-15 and 24-31), nvcc --threads 1, the flags of the E3 sweeps; GNU time for the peak memory, nvcc --time for the phases, the size of the cubin.
# Everything else of mine (the python parents of the sweeps, cuobjdump, the watcher) runs on cores 4-7,20-23 (first CCD, which the benchmarks do not use here), the load logger on core 5.
# Order: one warm-up compile (N = 1, own CSV, kept); the REQUESTED set, three compiles each: StaticLoop at 2, 10,002 and 19,902 operations (k = 1, 101, 200), then the distinct-operations chain (variant A) at
# N = 1, 1,000, 2,000, 3,000, 4,000; then the EXTRA lengths for the figure (own CSVs): distinct N = 100, 500, 1,500 and the other StaticLoop lengths of the old figure, three compiles each.
# START_DELAY seconds are waited first (the start time announced is computed from the launch).
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
# E3_* variables: only for testing the script on tiny inputs; the window uses the defaults
OUT=${E3_OUT:-$FB/results/rtx4090/e3-compile/repeat-c8fcbe8}; LOG=${E3_LOG:-$FB/results/rtx4090/e0-run.log}; PY=$T/venv-torch/bin/python
NS_REQ=${E3_NS_REQ:-1,1000,2000,3000,4000}; KS_REQ=${E3_KS_REQ:-1,101,200}; NS_EXTRA=${E3_NS_EXTRA:-100,500,1500}; KS_EXTRA=${E3_KS_EXTRA:-2,3,6,11,21,41,61,81,126,151,176}; REPS=${E3_REPS:-3}
CORE=${E3_CORE:-8}; AUX=4-7,20-23
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) window e3-repeat starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked)" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ echo "# $(date -Iseconds) before the window"; os_kernel_lines; uptime; cat /proc/loadavg; free -m | head -2
  echo "scaling governor, core $CORE: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unavailable); current frequency: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo unavailable) kHz; boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
  ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "fkl-benchmarks: $(git -C $FB rev-parse HEAD); tracked files that differ from it:"; git -C $FB status --porcelain --untracked-files=no
  echo "external/fkl: $(git -C $FB/external/fkl rev-parse HEAD); tracked files that differ: $(git -C $FB/external/fkl status --porcelain --untracked-files=no | wc -l)"
  nvcc --version | tail -2; g++ --version | head -1; echo "python: $($PY --version)"; } > $OUT/environment-before.txt 2>&1
taskset -c 5 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 20 --bench-cores $CORE,$((CORE + 16)) --ccd0 8-15,24-31 > /dev/null 2>&1 &
LOGGER_PID=$!
cd $FB
run() { echo "# $(date -Iseconds) start: $*" | cut -c1-260; "$@"; local rc=$?; echo "# $(date -Iseconds) end (exit $rc)"; return $rc; }
SWEEP="taskset -c $AUX $PY scripts/e3_sweep.py --variants A --core $CORE --tree c8fcbe8 --out $OUT --work ${E3_WORK:-$FB/build/e3-repeat} --keep-time-files"
LOOP="taskset -c $AUX $PY scripts/e3_staticloop.py --build build/merged --instance mul_add1 --core $CORE --out $OUT --keep-time-files"
echo "# warm-up: one compile of N = 1 (fills the file cache; recorded in its own CSV)"
run $SWEEP --ns 1 --reps 1 --csv e3_repeat_warmup_A.csv
echo "# REQUESTED: StaticLoop at 2, 10,002 and 19,902 operations, three compiles each"
run $LOOP --ks $KS_REQ --reps $REPS --rep-threshold 1e9 --csv e3_repeat_staticloop.csv
echo "# REQUESTED: distinct operations (variant A) at N = 1, 1000, 2000, 3000, 4000, three compiles each"
run $SWEEP --ns $NS_REQ --reps $REPS --rep-threshold 1e9 --csv e3_repeat_A.csv
echo "# $(date -Iseconds) the requested set is done"
echo "# EXTRA: distinct operations at N = 100, 500, 1500, three compiles each"
run $SWEEP --ns $NS_EXTRA --reps $REPS --rep-threshold 1e9 --csv e3_repeat_extra_A.csv
echo "# EXTRA: StaticLoop at the other lengths of the old figure, three compiles each"
run $LOOP --ks $KS_EXTRA --reps $REPS --rep-threshold 1e9 --csv e3_repeat_extra_staticloop.csv
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ echo "# $(date -Iseconds) after the window"; uptime; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
{ echo "# sha256 of the translation units that were compiled"; sha256sum ${E3_WORK:-$FB/build/e3-repeat}/*.cu; sha256sum $FB/build/merged/opencv-cuda/verticalfusion/generated/mul_add1/200_experiments/kernel{1,2,3,6,11,21,41,61,81,101,126,151,176,200}.cu; } > $OUT/tu-sha256.txt 2>&1
echo "# $(date -Iseconds) window e3-repeat ended" | tee -a $LOG
echo "e3-repeat window end $(date -Iseconds)"
