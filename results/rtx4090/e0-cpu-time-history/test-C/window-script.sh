#!/bin/bash
# GPU window "testc" (about 3 to 5 minutes, the desktop session keeps running): TEST C of results/rtx4090/e0-cpu-time-history: the two real CPU-time programs benchmark_CPU_OpenCV_vs_cvGS_cu and
# benchmark_CPU_NPP_vs_FastNPP_cu built at the CURRENT pins (cvGS 6d26c65, FastNPP a1b5e245, FKL c8fcbe8, fkl-benchmarks working tree of HEAD) three ways: CMAKE_BUILD_TYPE Release (-O3 -DNDEBUG, what the repository uses),
# None (the flags of an empty build type: no -O, no -DNDEBUG: what the 2025 cvGS CMake gave unless a build type was passed) and Debug (-g, no -O, no -DNDEBUG); 3 rounds, the order of the builds rotated per round,
# both programs per build, as scripts/run_e0.sh runs them (their own working directory, no arguments), pinned to cores 0-3,16-19 like the host-check windows. RTX 4090 = nvidia-smi index 1.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
C=$T/scratch/testC
OUT=$FB/results/rtx4090/e0-cpu-time-history/test-C; LOG=$FB/results/rtx4090/e0-run.log
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running; load $(cut -d' ' -f1-3 /proc/loadavg)"; exit 0; }
if [ -n "$(ls -A $OUT 2>/dev/null | grep -v "^README.md$")" ]; then echo "$OUT is not empty: not starting"; exit 1; fi
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window testc starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; echo "fkl-benchmarks $(git -C $FB rev-parse HEAD)"; for r in external/fkl external/cvGPUSpeedup external/FastNPP; do echo "$r $(git -C $FB/$r rev-parse HEAD)"; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,utilization.gpu --format=csv > $OUT/gpu-info.csv
for bt in Release None Debug; do grep -m1 "^CUDA_FLAGS" $C/build-$bt/opencv-cuda/CMakeFiles/benchmark_CPU_OpenCV_vs_cvGS_cu.dir/flags.make; grep -m1 "^CXX_FLAGS" $C/build-$bt/opencv-cuda/CMakeFiles/benchmark_CPU_OpenCV_vs_cvGS_cu.dir/flags.make | sed "s/^/$bt /"; done > $OUT/flags.txt
sha256sum $C/build-*/bin/benchmark_CPU_OpenCV_vs_cvGS_cu $C/build-*/bin/benchmark_CPU_NPP_vs_FastNPP_cu > $OUT/binary-sha256.txt
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
ORDERS=("Release None Debug" "None Debug Release" "Debug Release None")
for r in 1 2 3; do
  for bt in ${ORDERS[$((r-1))]}; do
    for prog in benchmark_CPU_OpenCV_vs_cvGS_cu benchmark_CPU_NPP_vs_FastNPP_cu; do
      d=$OUT/$bt/round$r/$prog; mkdir -p $d; start=$(date +%s)
      echo "# $(date -Iseconds) round $r $bt $prog"
      ( cd $d && env CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 timeout 300 taskset -c 0-3,16-19 $C/build-$bt/bin/$prog > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
      echo "# $(date -Iseconds) exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $d | grep -c '\.csv$')"
    done
  done
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window testc ended; load after: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a $LOG
echo "testc window end $(date -Iseconds)"
