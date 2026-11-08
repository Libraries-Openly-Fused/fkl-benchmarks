#!/bin/bash
# GPU window "e4-hand-cvgs-host-diagnostic" (F2; quiet, about 1 minute, the desktop session keeps running). Is FKL's higher host time in the window
# e4-hand-cvgs (host_enqueue_us +5 % median against window B from batch 50 on) the binary or drift? Two builds of e4_preprocessing_cu, alternating:
#   836a98e: window B's binary (worktrees/fkl-benchmarks-window, sha256 6e64d7e9...);
#   2dac1c8: the binary of the window e4-hand-cvgs (worktrees/fkl-benchmarks-window-f, sha256 8b56ffc2...).
# Only FKL's two versions, on the batches of crops: E4_EXPERIMENTS=E4a E4_ONLY=fkl_fused,fkl_fused_half_pixel, under the settings of the window e4-hand-cvgs:
#   a warm-up first (the 2dac1c8 binary, E4_EXPERIMENTS=E4a E4_BATCHES_ONLY=300, untimed, warmup/, not used);
#   three rounds, the build order flipping every round (1: 836a98e then 2dac1c8, 2: 2dac1c8 then 836a98e, 3: 836a98e then 2dac1c8), back to back, pinned to
#   cores 0-3,16-19; nvidia-smi every 100 ms on core 14 through the whole window (gpu_clocks.csv); the load logger on core 12.
# Folders: binary-836a98e/run1..3, binary-2dac1c8/run1..3. Each folder is new; nothing is overwritten. RTX 4090 = nvidia-smi index 1.
# Output: results/rtx4090/e4-hand-cvgs-host-diagnostic/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
OLDB=$T/worktrees/fkl-benchmarks-window/build/local-rtx4090/bin/e4_preprocessing_cu; OLDC=836a98e025bb4182528cbf47f082d548a943fbac
NEWB=$T/worktrees/fkl-benchmarks-window-f/build/local-rtx4090/bin/e4_preprocessing_cu; NEWC=2dac1c8ac467a45ff3076b4e25a0fb8160e5cca4
R=$FB/results/rtx4090; OUT=$R/e4-hand-cvgs-host-diagnostic; LOG=$R/e0-run.log
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $OLDB $NEWB || exit 1
# the two binaries are the recorded ones (window B's README and step 0 of the window e4-hand-cvgs) and carry their commits without -dirty
[ "$(sha256sum $OLDB | cut -d' ' -f1)" = 6e64d7e9c36fbfec9b43bad329a2c1c94ccda23c487da218ce3a2647a6f475c8 ] || { echo "836a98e's binary is not the recorded one: not starting"; exit 1; }
[ "$(sha256sum $NEWB | cut -d' ' -f1)" = 8b56ffc282bd3d03f0b3014e0cfff6c4c49ca51ce6e59d010d7fbb21db19f01b ] || { echo "2dac1c8's binary is not the recorded one: not starting"; exit 1; }
strings $OLDB | grep -q -x $OLDC && strings $NEWB | grep -q -x $NEWC || { echo "not the expected commits: not starting"; exit 1; }
strings $OLDB $NEWB | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "a -dirty commit: not starting"; exit 1; }
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
source $FB/scripts/quiet_check.sh; busy=$(busy_processes)
if [ -n "$busy" ]; then echo "a measurement, a build or a compile job is running: not starting"; echo "$busy" | cut -c1-200; exit 1; fi
gpuapps=$(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | grep -v -E ', /usr/')
[ -n "$gpuapps" ] && { echo "a program uses the RTX 4090: not starting"; echo "$gpuapps"; exit 1; }
[ -n "${GUARD_ONLY:-}" ] && { echo "guard ok at $(date +%H:%M:%S): nothing else is running"; exit 0; }
[ -e $OUT ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
mkdir -p $OUT
exec > >(tee -a $OUT/window-console.txt) 2>&1
cp "$0" $OUT/window-script.sh
( sha256sum $OLDB; sha256sum $NEWB ) > $OUT/binary-sha256.txt
echo "# $(date -Iseconds) GPU window e4-hand-cvgs-host-diagnostic starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $T/worktrees/fkl-benchmarks-window $T/worktrees/fkl-benchmarks-window-f; do
    echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
taskset -c 14 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 \
  > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
SAMPLER_PID=$!
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <binary> <run folder> [VAR=value ...]: pinned to cores 0-3,16-19
run() {
  local bin=$1 name=$2; shift 2
  local dir=$OUT/$name; mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  echo "# $(date -Iseconds) $name $*"
  ( cd $dir && env $GENV "$@" timeout 900 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) $name exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; csv $(ls $dir | grep -c '\.csv$'); INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l); -nan in: $(grep -l -- '-nan' $dir/*.csv 2>/dev/null | wc -l)"
}
run $NEWB warmup E4_EXPERIMENTS=E4a E4_BATCHES_ONLY=300
for n in 1 2 3; do
  order="836a98e 2dac1c8"; [ $n = 2 ] && order="2dac1c8 836a98e"
  for b in $order; do
    bin=$OLDB; [ $b = 2dac1c8 ] && bin=$NEWB
    run $bin binary-$b/run$n E4_EXPERIMENTS=E4a E4_ONLY=fkl_fused,fkl_fused_half_pixel
  done
done
kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window e4-hand-cvgs-host-diagnostic ended" | tee -a $LOG
echo "e4-hand-cvgs-host-diagnostic window end $(date -Iseconds)"
