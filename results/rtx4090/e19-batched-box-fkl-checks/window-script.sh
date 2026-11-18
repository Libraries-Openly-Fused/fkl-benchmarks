#!/bin/bash
# GPU window "e19-fkl-checks" (checks only, about half a minute; the desktop session keeps running): E19 phase 2, the row fkl_loop added,
# e19_batched_box_cu of branch e19-batched-box at f487b3f (the build of 2026-11-18 11:03, sha256 2b59cd1c...). Nothing is timed. Two steps, each
# one run:
#   1. checks: E19_NO_TIMING=1, with E19_DUMP (the rows' grey and box images and the tensors as raw bytes, into the scratch folder, not
#      committed). The checks of every row (opencv_cuda, npp, fkl_loop), then the program exits; no timing.
#   2. nsys, its own step after the checks run, and only if they passed: E19_PROFILE=1 (a warm-up call per row outside any range, then 10
#      calls, each in the NVTX range "<row>:call"; nothing compared or timed), under the CUDA API and NVTX trace: the kernels of one call.
#   No ncu. SAMPLER: nvidia-smi every 100 ms through the window (gpu_clocks.csv), on core 14.
# Every step's exit code is checked: the window stops at a failed step. Each folder is new; nothing is overwritten. RTX 4090 = nvidia-smi index 1.
# After the GPU steps, the nsys report is exported to SQLite into the scratch folder (CPU only).
# Output: results/rtx4090/e19-batched-box-fkl-checks/ of the branch worktree.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-e19-batched-box; source $W/scripts/os_kernel_lines.sh
NEW=$W/build/local-rtx4090/bin; S=$T/scratch/e19-batched-box
NEWC=f487b3f559ec0be8fb1d7942512c12beba8b117d
OUT=$W/results/rtx4090/e19-batched-box-fkl-checks; LOG=$FB/results/rtx4090/e0-run.log
PROG=e19_batched_box_cu
NSYS=$T/toolchains/cuda-13.4.3/bin/nsys
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $NEW/$PROG || exit 1
# the binary is the one of the build record (sha256), and it carries the expected commit without -dirty
grep -E "build/local-rtx4090/bin/$PROG\$" $S/binary-sha256-p2.txt | (cd $W && sha256sum -c --quiet) || { echo "the binary differs from the build record: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -x $NEWC || { echo "$PROG: not the expected commit: not starting"; exit 1; }
strings $NEW/$PROG | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$PROG: a -dirty commit: not starting"; exit 1; }
[ "$(git -C $W rev-parse HEAD)" = "$NEWC" ] || { echo "the worktree is not at $NEWC: not starting"; exit 1; }
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
DUMP=$S/dump-p2; EXPORT=$S/nsys-export-p2
[ -e $DUMP ] && { echo "$DUMP exists: not starting (nothing is overwritten)"; exit 1; }
[ -e $EXPORT ] && { echo "$EXPORT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT $DUMP $EXPORT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e19-fkl-checks starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs"
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
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <run folder> <prefix command, or -> [VAR=value ...]: pinned to cores 0-3,16-19
run() {
  local name=$1 prefix=$2; shift 2
  [ "$prefix" = "-" ] && prefix=""
  local dir=$OUT/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  echo "# $(date -Iseconds) $name $prefix $*"
  ( cd $dir && env $GENV "$@" timeout 180 taskset -c 0-3,16-19 $prefix $NEW/$PROG > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l); checks passed: $(grep -c 'E19 batched box checks passed' $dir/stdout.txt)"
}
# stop_at_failure <run folder>: a step that did not exit 0 ends the window here (the helpers are stopped and the end is logged)
stop_at_failure() {
  local rc=$(cat $OUT/$1/exit_code.txt 2>/dev/null)
  [ "$rc" = 0 ] && return 0
  stop_helpers
  { date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
  echo "# $(date -Iseconds) GPU window e19-fkl-checks STOPPED after $1 (exit ${rc:-missing})" | tee -a $LOG
  echo "e19-fkl-checks window end $(date -Iseconds)"
  exit 1
}
# 1. the checks run (no timing)
run checks - E19_NO_TIMING=1 E19_DUMP=$DUMP
stop_at_failure checks
# 2. the nsys pass, its own step after the checks run: the profile mode (nothing compared or timed) under the CUDA API and NVTX trace
run nsys "$NSYS profile --trace=cuda,nvtx --sample=none --cpuctxsw=none --force-overwrite=false -o $OUT/nsys/e19-fkl-profile" E19_PROFILE=1 E19_PROFILE_CALLS=10
stop_at_failure nsys
stop_helpers
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e19-fkl-checks ended (GPU steps)" | tee -a $LOG
# CPU only, after the GPU steps: the SQLite export of the trace, into the scratch folder (not committed)
$NSYS export --type sqlite --output $EXPORT/e19-fkl-profile.sqlite $OUT/nsys/e19-fkl-profile.nsys-rep > $EXPORT/export.log 2>&1
echo "nsys export exit $?"
echo "e19-fkl-checks window end $(date -Iseconds)"
