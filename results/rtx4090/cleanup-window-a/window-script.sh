#!/bin/bash
# GPU window "cleanup-a" (quiet, about 4 minutes, the desktop session keeps running). Two builds, alternating:
#   reference: build/local-rtx4090 of the reference checkout (a5dd9fa, built 2026-11-07 00:45);
#   cleaned:   step 0's build of main 836a98e.
# Three runs of each program on each build. Each run's folder is new; nothing is overwritten.
#   e9-cross-library/run0..2  e9_cross_library_cu (E9_RUN = 0, 1, 2 on both builds), pinned to cores 0-3,16-19 as in the published window;
#   e0-npp/run1..3            benchmark_npp_batchresize_x_split3D_cu (both tests, one process);
#   e0-cpu-time-npp/run1..3   benchmark_CPU_NPP_vs_FastNPP_cu;
#   e0-cpu-time-opencv/run1..3 benchmark_CPU_OpenCV_vs_cvGS_cu.
# The three NPP and CPU-time programs run unpinned, each in its own working directory, as scripts/run_e0.sh runs them.
# The build order flips every run: 1 reference then cleaned, 2 cleaned then reference, 3 reference then cleaned. Each program's pair of runs is
# back to back. Then one nsys pass of E9 per build: one call of each variant, E9_NSYS=1 E9_SEEDS=1, as in the published window.
# RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first. Output: results/rtx4090/cleanup-window-a/{reference,cleaned}/<experiment>/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
REF=$FB/build/local-rtx4090/bin; NEW=$T/worktrees/fkl-benchmarks-window/build/local-rtx4090/bin; STEP0=$T/scratch/cleanup-window-build
REFC=a5dd9fa54210f60d0b631ce34508e95c86c387ce; NEWC=836a98e025bb4182528cbf47f082d548a943fbac
R=$FB/results/rtx4090; OUT=$R/cleanup-window-a; LOG=$R/e0-run.log
PROGS="e9_cross_library_cu benchmark_npp_batchresize_x_split3D_cu benchmark_CPU_NPP_vs_FastNPP_cu benchmark_CPU_OpenCV_vs_cvGS_cu"
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
for p in $PROGS; do refuse_blocked_binaries $REF/$p $NEW/$p || exit 1; done
# the binaries are the ones step 0 checked (sha256), and they carry the expected commits without -dirty
grep -q 'all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
grep -E "/($(echo $PROGS | tr ' ' '|'))\$" $STEP0/binary-sha256.txt | sha256sum -c --quiet || { echo "a binary differs from step 0's record: not starting"; exit 1; }
for p in $PROGS; do
  strings $REF/$p | grep -q -x $REFC && strings $NEW/$p | grep -q -x $NEWC || { echo "$p: not the expected commits: not starting"; exit 1; }
  strings $REF/$p $NEW/$p | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$p: a -dirty commit: not starting"; exit 1; }
done
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
# a process outside /usr with a context on the RTX 4090 (a program of this machine's work, not the desktop session) refuses the window
gpuapps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/')
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh; cp $STEP0/binary-sha256.txt $STEP0/build-info-reference.txt $STEP0/build-info-cleaned.txt $OUT/
echo "# $(date -Iseconds) GPU window cleanup-a starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $FB $FB/external/fkl $FB/external/FastNPP $FB/external/cvGPUSpeedup $T/worktrees/fkl-benchmarks-window \
           $T/worktrees/fkl-benchmarks-window/external/fkl $T/worktrees/fkl-benchmarks-window/external/FastNPP $T/worktrees/fkl-benchmarks-window/external/cvGPUSpeedup; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <build> <experiment> <run folder> <program> <pin 0|1> [VAR=value ...]
run() {
  local build=$1 exp=$2 name=$3 prog=$4 pin=$5; shift 5
  local bin=$REF/$prog; [ $build = cleaned ] && bin=$NEW/$prog
  local dir=$OUT/$build/$exp/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)" > $dir/load.txt
  echo "# $(date -Iseconds) $build $exp $name $*"
  if [ $pin = 1 ]; then
    ( cd $dir && env $GENV "$@" timeout 600 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  else
    ( cd $dir && env $GENV "$@" timeout 600 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  fi
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $build $exp $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -e '-nan' $dir/*.csv 2>/dev/null | wc -l) files; $(grep -c -E 'failed!!' $dir/stdout.txt) failed"
}
for n in 1 2 3; do
  order="reference cleaned"; [ $n = 2 ] && order="cleaned reference"
  for b in $order; do run $b e9-cross-library run$((n - 1)) e9_cross_library_cu 1 E9_RUN=$((n - 1)); done
  for b in $order; do run $b e0-npp run$n benchmark_npp_batchresize_x_split3D_cu 0; done
  for b in $order; do run $b e0-cpu-time-npp run$n benchmark_CPU_NPP_vs_FastNPP_cu 0; done
  for b in $order; do run $b e0-cpu-time-opencv run$n benchmark_CPU_OpenCV_vs_cvGS_cu 0; done
done
for b in reference cleaned; do
  bin=$REF/e9_cross_library_cu; [ $b = cleaned ] && bin=$NEW/e9_cross_library_cu
  d=$OUT/$b/e9-cross-library/nsys; mkdir -p $d
  ( cd $d && env $GENV E9_NSYS=1 E9_SEEDS=1 timeout 300 taskset -c 0-3,16-19 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false -o e9 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt
    nsys export --type sqlite --force-overwrite true -o e9.sqlite e9.nsys-rep > export.txt 2>&1
    python3 - <<'PY' > kernels.txt 2>&1
import sqlite3
c=sqlite3.connect("e9.sqlite")
for r in c.execute("select s.value, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value"):
    print(r[1], r[0][:300])
print("total kernel launches in the range:", c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0])
PY
  )
  echo "# $(date -Iseconds) $b nsys pass exit $(cat $d/exit_code.txt); $(tail -1 $d/kernels.txt)"
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window cleanup-a ended" | tee -a $LOG
echo "cleanup-a window end $(date -Iseconds)"
