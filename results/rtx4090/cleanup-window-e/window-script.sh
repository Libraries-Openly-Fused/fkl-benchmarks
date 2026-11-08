#!/bin/bash
# GPU window "cleanup-e" (quiet, about 2 minutes, the desktop session keeps running): the cleaned E2 FKL program with its passes in the order of the Python driver.
#   cleaned: build/local-rtx4090 of worktrees/fkl-benchmarks-window-e (step 0 of window E, scratch/cleanup-window-e-build/main-commit.txt): e2_fkl_chain_cu,
#            which times every k warm, L2-flushed, kernel-only (PR #28). The reference is not re-run: its program has no kernel-only pass, so its flushed
#            pass already follows the warm one.
# An nvidia-smi sampler (every 100 ms: timestamp, SM clock, memory clock, pstate, utilisation, power, temperature) runs from before the first run to after the
# last: gpu_clocks.csv.
# Three runs of the program (all six chain lengths) on the published input e2-work/e2_input.u8: cleaned/e2-fkl/run1..3; its outputs go to
# e2-work-cleanup-e/cleaned-run<n>, their sha256 to the run folder.
# Everything is pinned to cores 0-3,16-19 (the load logger on core 12, the sampler on core 13, both at nice 19). RTX 4090 = nvidia-smi index 1.
# START_DELAY seconds are waited first. Output: results/rtx4090/cleanup-window-e/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
WE=$T/worktrees/fkl-benchmarks-window-e; STEP0=$T/scratch/cleanup-window-e-build
NEW_E2=$WE/build/local-rtx4090/bin/e2_fkl_chain_cu; NEWC=$(cat $STEP0/main-commit.txt 2>/dev/null)
R=$FB/results/rtx4090; OUT=$R/cleanup-window-e; LOG=$R/e0-run.log; E2OUT=$T/e2-work-cleanup-e; INPUT=$T/e2-work/e2_input.u8
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW_E2 || exit 1
grep -q 'all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
sha256sum -c --quiet $STEP0/binary-sha256.txt || { echo "a binary differs from step 0's record: not starting"; exit 1; }
strings $NEW_E2 | grep -q -x "$NEWC" || { echo "$NEW_E2: not $NEWC: not starting"; exit 1; }
[ "$(sha256sum < $INPUT | cut -d' ' -f1)" = "726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995" ] || { echo "the E2 input is not the published one: not starting"; exit 1; }
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/')
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -e $E2OUT ] && { echo "$E2OUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT $E2OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh; cp $STEP0/binary-sha256.txt $STEP0/main-commit.txt $STEP0/build-info-reference.txt $STEP0/build-info-cleaned.txt $OUT/
taskset -c 13 nice -n 19 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
SAMPLER_PID=$!
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
sleep 2
echo "# $(date -Iseconds) GPU window cleanup-e starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs; clock sampler pid $SAMPLER_PID" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $FB $FB/external/fkl $WE $WE/external/fkl; do echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
run() {
  local dir=$1 bin=$2; shift 2
  mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  ( cd $dir && env $GENV "$@" timeout 1800 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) ${dir#$OUT/} $*: exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l)"
}
for n in 1 2 3; do
  mkdir -p $E2OUT/cleaned-run$n
  run $OUT/cleaned/e2-fkl/run$n $NEW_E2 E2_INPUT=$INPUT E2_OUT_DIR=$E2OUT/cleaned-run$n
  ( cd $E2OUT/cleaned-run$n && sha256sum fkl_out_k*.f32 ) > $OUT/cleaned/e2-fkl/run$n/outputs-sha256.txt
  sleep 5
done
sleep 2
kill $SAMPLER_PID $LOGGER_PID 2> /dev/null; wait $SAMPLER_PID $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window cleanup-e ended; clock samples: $(wc -l < $OUT/gpu_clocks.csv)" | tee -a $LOG
echo "cleanup-e window end $(date -Iseconds)"
