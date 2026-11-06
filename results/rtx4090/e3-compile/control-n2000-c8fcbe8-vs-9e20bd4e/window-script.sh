#!/bin/bash
# Window "e3-control-n2000" (quiet, compile only, no GPU work, about 10 minutes; approved by the coordinator for 2026-11-06 20:45:00, latest end 21:00): a same-day control of the compile time of the
# distinct-operations chain (variant A) at N = 2,000: FKL c8fcbe8 (the pin external/fkl) and the head of PR #423 (9e20bd4e, scratch/pr423/fkl-head) ALTERNATING, three pairs, the order swapped between pairs:
# (c8fcbe8, head), (head, c8fcbe8), (c8fcbe8, head). Same translation unit (scripts/e3_sweep.py generates the same text; the file names are those of the two repeats), same flags, one compile at a time on
# core 8, nvcc --threads 1; one warm-up compile at N = 1 per tree first (own CSVs). The python parents on cores 4-7,20-23, the load logger on core 5. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
OUT=${E3_OUT:-$FB/results/rtx4090/e3-compile/control-n2000-c8fcbe8-vs-9e20bd4e}; LOG=${E3_LOG:-$FB/results/rtx4090/e0-run.log}; PY=$T/venv-torch/bin/python
N=${E3_N:-2000}; HEADTREE=${E3_FKL:-$T/scratch/pr423/fkl-head}; PINTREE=$FB/external/fkl; WORK=${E3_WORK:-$FB/build/e3-control-n2000}
CORE=${E3_CORE:-8}; AUX=4-7,20-23
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null)" ]; then echo "$OUT is not empty: not starting (nothing is overwritten)"; exit 1; fi
mkdir -p $OUT/nvcc-time
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) window e3-control-n2000 starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked)" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ echo "# $(date -Iseconds) before the window"; os_kernel_lines; uptime; cat /proc/loadavg; free -m | head -2
  echo "scaling governor, core $CORE: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_governor 2>/dev/null || echo unavailable); current frequency: $(cat /sys/devices/system/cpu/cpu$CORE/cpufreq/scaling_cur_freq 2>/dev/null || echo unavailable) kHz; boost: $(cat /sys/devices/system/cpu/cpufreq/boost 2>/dev/null || echo unavailable)"
  ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "fkl-benchmarks: $(git -C $FB rev-parse HEAD); tracked files that differ from it:"; git -C $FB status --porcelain --untracked-files=no
  echo "tree c8fcbe8 (the pin): $PINTREE $(git -C $PINTREE rev-parse HEAD); tracked files that differ: $(git -C $PINTREE status --porcelain --untracked-files=no | wc -l)"
  echo "tree 9e20bd4e (PR #423 head): $HEADTREE $(git -C $HEADTREE rev-parse HEAD); tracked files that differ: $(git -C $HEADTREE status --porcelain --untracked-files=no | wc -l)"
  nvcc --version | tail -2; g++ --version | head -1; echo "python: $($PY --version)"; } > $OUT/environment-before.txt 2>&1
taskset -c 5 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 20 --bench-cores $CORE,$((CORE + 16)) --ccd0 8-15,24-31 > /dev/null 2>&1 &
LOGGER_PID=$!
cd $FB
run() { echo "# $(date -Iseconds) start: $*" | cut -c1-260; "$@"; local rc=$?; echo "# $(date -Iseconds) end (exit $rc)"; return $rc; }
# one compile: <tree label> <n> <csv name>; the time file of nvcc is renamed after the compile so that the next one does not overwrite it
one() {
  local label=$1 n=$2 csv=$3
  if [ "$label" = c8fcbe8 ]; then
    run taskset -c $AUX $PY scripts/e3_sweep.py --variants A --core $CORE --tree c8fcbe8 --include $PINTREE/include --out $OUT --work $WORK --keep-time-files --ns $n --reps 1 --rep-threshold 1e9 --csv $csv
  else
    run env FKLB_FKL_TREE=$HEADTREE taskset -c $AUX $PY scripts/e3_sweep.py --variants A --core $CORE --tree 9e20bd4e --include $HEADTREE/include --out $OUT --work $WORK --keep-time-files --ns $n --reps 1 --rep-threshold 1e9 --csv $csv
  fi
  mv $OUT/nvcc-time/${label}_A_${n}_rep0.time.csv $OUT/nvcc-time/${csv%.csv}.time.csv 2>/dev/null
}
echo "# warm-up: one compile of N = 1 per tree (fills the file cache with both trees' headers; own CSVs, not used)"
one c8fcbe8 1 warmup_1_c8fcbe8.csv
one 9e20bd4e 1 warmup_2_9e20bd4e.csv
echo "# pairs at N = $N: (c8fcbe8, head), (head, c8fcbe8), (c8fcbe8, head)"
one c8fcbe8 $N pair1_1_c8fcbe8.csv
one 9e20bd4e $N pair1_2_9e20bd4e.csv
one 9e20bd4e $N pair2_1_9e20bd4e.csv
one c8fcbe8 $N pair2_2_c8fcbe8.csv
one c8fcbe8 $N pair3_1_c8fcbe8.csv
one 9e20bd4e $N pair3_2_9e20bd4e.csv
echo "# $(date -Iseconds) the set is done"
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ echo "# $(date -Iseconds) after the window"; uptime; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
{ echo "# sha256 of the translation units that were compiled"; sha256sum $WORK/*.cu; } > $OUT/tu-sha256.txt 2>&1
echo "# $(date -Iseconds) window e3-control-n2000 ended" | tee -a $LOG
echo "e3-control-n2000 window end $(date -Iseconds)"
