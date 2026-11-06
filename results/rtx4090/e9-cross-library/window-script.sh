#!/bin/bash
# GPU window "e9" (quiet, about 6 minutes, the desktop session keeps running), FastNPP at a1b5e245 (#44, #45 merged), FKL c8fcbe8, cvGS 6d26c65.
# (i) E9: 3 runs (E9_RUN 0,1,2) of e9_cross_library_cu pinned to cores 0-3,16-19, then one nsys pass of one call of each variant; (ii) e0-npp and e0-cpu-time-npp, 3 runs each, alternating, run as scripts/run_e0.sh runs them
# (no pinning, the benchmark's own working directory). RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
B=$FB/build/local-rtx4090/bin; R=$FB/results/rtx4090
OUT=$R/e9-cross-library; LOG=$R/e0-run.log
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
sleep ${START_DELAY:-0}
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
for d in $OUT $R/e0-npp-after45 $R/e0-cpu-time-npp-after45; do if [ -n "$(ls -A $d 2>/dev/null)" ]; then echo "$d is not empty: not starting"; exit 1; fi; done
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
echo "# $(date -Iseconds) GPU window e9 starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; for r in $FB $FB/external/fkl $FB/external/FastNPP $FB/external/cvGPUSpeedup; do echo "$r $(git -C $r rev-parse HEAD)"; done; git -C $FB status --porcelain --untracked-files=no; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu --format=csv > $OUT/gpu-info.csv
sha256sum $B/e9_cross_library_cu $B/benchmark_CPU_NPP_vs_FastNPP_cu $B/benchmark_npp_batchresize_x_split3D_cu > $OUT/binary-sha256.txt
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
for rot in 0 1 2; do
  mkdir -p $OUT/run$rot; start=$(date +%s)
  echo "# $(date -Iseconds) E9 run $rot"
  ( cd $OUT/run$rot && env $GENV E9_RUN=$rot timeout 600 taskset -c 0-3,16-19 $B/e9_cross_library_cu > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) E9 run $rot exit $(cat $OUT/run$rot/exit_code.txt) after $(( $(date +%s) - start )) s"
done
mkdir -p $OUT/nsys; ( cd $OUT/nsys && env $GENV E9_NSYS=1 E9_SEEDS=1 timeout 300 taskset -c 0-3,16-19 nsys profile --capture-range=cudaProfilerApi --capture-range-end=stop --stats=false -o e9 $B/e9_cross_library_cu > stdout.txt 2> stderr.txt; echo $? > exit_code.txt
  nsys export --type sqlite --force-overwrite true -o e9.sqlite e9.nsys-rep > export.txt 2>&1
  python3 - <<'PY' > kernels.txt 2>&1
import sqlite3
c=sqlite3.connect("e9.sqlite")
for r in c.execute("select s.value, count(*) from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.demangledName group by s.value"):
    print(r[1], r[0][:300])
print("total kernel launches in the range:", c.execute("select count(*) from CUPTI_ACTIVITY_KIND_KERNEL").fetchone()[0])
PY
)
echo "# $(date -Iseconds) nsys pass exit $(cat $OUT/nsys/exit_code.txt)"; tail -1 $OUT/nsys/kernels.txt
for rep in 1 2 3; do
  for pair in "e0-npp-after45|benchmark_npp_batchresize_x_split3D_cu" "e0-cpu-time-npp-after45|benchmark_CPU_NPP_vs_FastNPP_cu"; do
    IFS='|' read -r name bin <<<"$pair"; dir=$R/$name/run$rep; mkdir -p $dir; start=$(date +%s)
    echo "# $(date -Iseconds) $name run $rep"
    ( cd $dir && env $GENV $B/$bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
    nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
    echo "# $(date -Iseconds) $name run $rep exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); -nan in: $(grep -l -e '-nan' $dir/*.csv 2>/dev/null | wc -l) files"
  done
done
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e9 ended" | tee -a $LOG
echo "e9 window end $(date -Iseconds)"
