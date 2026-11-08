#!/bin/bash
# GPU window "cleanup-c" (quiet, about 20 minutes, the desktop session keeps running): E1 and E2, the reference build against the cleaned build (step 0 of window C).
#   reference: E1 build/ref-rtx4090-e1 (a5dd9fa, preset rtx4090-e1: the flags of the published E1 builds), E2 build/local-rtx4090 (a5dd9fa)
#   cleaned:   the builds of step 0 (main at step 0, scratch/cleanup-window-c-build/main-commit.txt): build/rtx4090-e1 and build/local-rtx4090 of worktrees/fkl-benchmarks-window-c
# 1. E1: three runs of e1_vf_chain, e1_hf_batch and e1_preproc on each build; the build order flips every run (1 reference then cleaned, 2 cleaned then reference,
#    3 reference then cleaned); each program's pair of runs is back to back. The cleaned build also writes the kernel-only columns (e1::gpuSpin, E1_SPIN_US = 5000).
# 2. The spin check: each cleaned E1 program once more with E1_SPIN_US = 10000 (twice the spin).
# 3. E1's Nsight Compute pass of the cleaned build (E1_PROFILE=1: one launch of every kernel; occupancy and duration, for scripts/e1_table.py, which joins the ncu data
#    by mangled name, so the cleaned build needs its own).
# 4. E2, the FKL program: three runs per build, the order flipping as in 1 (warm, L2-flushed and, cleaned only, kernel-only), on the published input
#    e2-work/e2_input.u8 (sha256 checked); the outputs fkl_out_k*.f32 go to e2-work-cleanup/<build>-run<n> (outside the repository, sha256 recorded in the run folder).
#    Then the cleaned program once more with E1_SPIN_US = 10000.
# 5. E2, the Python driver (triton/e2_chain.py of the cleaned checkout, venv-torch, scripts/triton_env.sh): three runs, run n with the outputs and the CSV of the cleaned
#    FKL run n (--fkl-dir, --fkl-csv); default chain lengths (2, 10, 50, 100, 500) and versions; no --fkl-compile (no compile-time measurement) and no large-k run.
#    Then the spin check of the Python side: one more run with E1_SPIN_US = 10000, with the outputs and the CSV of the cleaned FKL spin2x run.
# Everything is pinned to cores 0-3,16-19. RTX 4090 = nvidia-smi index 1. START_DELAY seconds are waited first. Output: results/rtx4090/cleanup-window-c/{reference,cleaned}/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
WC=$T/worktrees/fkl-benchmarks-window-c; STEP0=$T/scratch/cleanup-window-c-build
REF_E1=$FB/build/ref-rtx4090-e1/bin; REF_E2=$FB/build/local-rtx4090/bin; NEW_E1=$WC/build/rtx4090-e1/bin; NEW_E2=$WC/build/local-rtx4090/bin
REFC=a5dd9fa54210f60d0b631ce34508e95c86c387ce; NEWC=$(cat $STEP0/main-commit.txt 2>/dev/null)
R=$FB/results/rtx4090; OUT=$R/cleanup-window-c; LOG=$R/e0-run.log; E2OUT=$T/e2-work-cleanup; INPUT=$T/e2-work/e2_input.u8
E1P="e1_vf_chain_cu e1_hf_batch_cu e1_preproc_cu"
NCU=$T/toolchains/cuda-13.4.3/bin/ncu; PY=$T/venv-torch/bin/python
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
for p in $E1P; do refuse_blocked_binaries $REF_E1/$p $NEW_E1/$p || exit 1; done
refuse_blocked_binaries $REF_E2/e2_fkl_chain_cu $NEW_E2/e2_fkl_chain_cu || exit 1
grep -q 'all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
sha256sum -c --quiet $STEP0/binary-sha256.txt || { echo "a binary differs from step 0's record: not starting"; exit 1; }
for b in $REF_E1/e1_vf_chain_cu $REF_E1/e1_hf_batch_cu $REF_E1/e1_preproc_cu $REF_E2/e2_fkl_chain_cu; do strings $b | grep -q -x $REFC || { echo "$b: not $REFC: not starting"; exit 1; }; done
for b in $NEW_E1/e1_vf_chain_cu $NEW_E1/e1_hf_batch_cu $NEW_E1/e1_preproc_cu $NEW_E2/e2_fkl_chain_cu; do strings $b | grep -q -x "$NEWC" || { echo "$b: not $NEWC: not starting"; exit 1; }; done
[ "$(sha256sum < $INPUT | cut -d' ' -f1)" = "726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995" ] || { echo "the E2 input is not the published one: not starting"; exit 1; }
export TMPDIR=$T/scratch/e6tmp; mkdir -p $TMPDIR
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
cp "$0" $OUT/window-script.sh; cp $STEP0/binary-sha256.txt $STEP0/main-commit.txt $STEP0/build-info-*.txt $OUT/
echo "# $(date -Iseconds) GPU window cleanup-c starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $FB $FB/external/fkl $WC $WC/external/fkl; do echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <folder> <binary> [VAR=value ...]: one pinned run in its own folder
run() {
  local dir=$1 bin=$2; shift 2
  mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)" > $dir/load.txt
  ( cd $dir && env $GENV "$@" timeout 1800 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) ${dir#$OUT/} $*: exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l)"
}
declare -A EXP=([e1_vf_chain_cu]=e1-vf-chain [e1_hf_batch_cu]=e1-hf-batch [e1_preproc_cu]=e1-preproc)
# 1. E1
for n in 1 2 3; do
  order="reference cleaned"; [ $n = 2 ] && order="cleaned reference"
  for p in $E1P; do
    for b in $order; do bin=$REF_E1/$p; [ $b = cleaned ] && bin=$NEW_E1/$p; run $OUT/$b/${EXP[$p]}/run$n $bin; done
  done
done
# 2. the spin check
for p in $E1P; do run $OUT/cleaned/${EXP[$p]}/spin2x $NEW_E1/$p E1_SPIN_US=10000; done
# 3. E1's Nsight Compute pass of the cleaned build
$NCU --version > $OUT/cleaned/ncu-version.txt 2>&1
for p in $E1P; do
  d=$OUT/cleaned/${EXP[$p]}/profile; mkdir -p $d; start=$(date +%s)
  ( cd $d && env $GENV E1_PROFILE=1 timeout 1800 taskset -c 0-3,16-19 $NCU --csv --page raw \
      --metrics gpu__time_duration.avg,gpu__time_duration.max,gpu__time_duration.min,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_active.max.pct_of_peak_sustained_active,sm__warps_active.min.pct_of_peak_sustained_active,sm__warps_active.sum.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,sm__maximum_warps_avg_per_active_cycle,smsp__maximum_warps_avg_per_active_cycle \
      $NEW_E1/$p > ncu.csv 2> ncu.err; echo $? > ncu.exit )
  echo "# $(date -Iseconds) ncu ${EXP[$p]}: exit $(cat $d/ncu.exit) after $(( $(date +%s) - start )) s; kernel rows $(grep -c -E '^\"[0-9]+\",' $d/ncu.csv)"
done
# 4. E2, the FKL program
for n in 1 2 3; do
  order="reference cleaned"; [ $n = 2 ] && order="cleaned reference"
  for b in $order; do
    bin=$REF_E2/e2_fkl_chain_cu; [ $b = cleaned ] && bin=$NEW_E2/e2_fkl_chain_cu
    mkdir -p $E2OUT/$b-run$n
    run $OUT/$b/e2-fkl/run$n $bin E2_INPUT=$INPUT E2_OUT_DIR=$E2OUT/$b-run$n
    ( cd $E2OUT/$b-run$n && sha256sum fkl_out_k*.f32 ) > $OUT/$b/e2-fkl/run$n/outputs-sha256.txt
  done
done
mkdir -p $E2OUT/cleaned-spin2x
run $OUT/cleaned/e2-fkl/spin2x $NEW_E2/e2_fkl_chain_cu E2_INPUT=$INPUT E2_OUT_DIR=$E2OUT/cleaned-spin2x E1_SPIN_US=10000
# 5. E2, the Python driver
for n in 1 2 3; do
  d=$OUT/cleaned/e2-python/run$n; mkdir -p $d; start=$(date +%s)
  ( cd $d && source $FB/scripts/triton_env.sh > /dev/null 2>&1 && env $GENV timeout 3600 taskset -c 0-3,16-19 $PY $WC/triton/e2_chain.py --out e2_chain.csv --input $INPUT \
      --fkl-dir $E2OUT/cleaned-run$n --fkl-csv $OUT/cleaned/e2-fkl/run$n/e2_fkl_chain.csv > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $d/gpu-at-end.csv
  echo "# $(date -Iseconds) cleaned/e2-python/run$n: exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s; rows ok $(grep -c ',ok,' $d/e2_chain.csv 2>/dev/null)"
done
d=$OUT/cleaned/e2-python/spin2x; mkdir -p $d; start=$(date +%s)
( cd $d && source $FB/scripts/triton_env.sh > /dev/null 2>&1 && env $GENV E1_SPIN_US=10000 timeout 3600 taskset -c 0-3,16-19 $PY $WC/triton/e2_chain.py --out e2_chain.csv --input $INPUT \
    --fkl-dir $E2OUT/cleaned-spin2x --fkl-csv $OUT/cleaned/e2-fkl/spin2x/e2_fkl_chain.csv > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $d/gpu-at-end.csv
echo "# $(date -Iseconds) cleaned/e2-python/spin2x: exit $(cat $d/exit_code.txt) after $(( $(date +%s) - start )) s; rows ok $(grep -c ',ok,' $d/e2_chain.csv 2>/dev/null)"
kill $LOGGER_PID 2> /dev/null; wait $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window cleanup-c ended" | tee -a $LOG
echo "cleanup-c window end $(date -Iseconds)"
