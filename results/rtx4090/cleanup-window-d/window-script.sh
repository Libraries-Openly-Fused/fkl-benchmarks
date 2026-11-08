#!/bin/bash
# GPU window "cleanup-d" (quiet, about 45 minutes, the desktop session keeps running): E16 reference against cleaned, E2 at k = 5002, and E1's ncu pass again.
#   reference: build/local-rtx4090 of the reference checkout (a5dd9fa): e16_attention_cu and e2_fkl_chain_cu; pytorch-attention/e16_baselines.py of the reference checkout
#   cleaned:   E16: build/local-rtx4090 of worktrees/fkl-benchmarks-window-d (step 0 of window D, scratch/cleanup-window-d-build/main-commit.txt) and that worktree's
#              pytorch-attention/e16_baselines.py; E1 and E2: window C's cleaned builds and triton/e2_chain.py (worktrees/fkl-benchmarks-window-c, main 1a2535c;
#              their sources are unchanged since; sha256 checked against window C's record)
# An nvidia-smi sampler (every 100 ms: timestamp, SM clock, memory clock, pstate, utilisation, power, temperature, as in the published E16 window) runs from
# before the first run to after the last: gpu_clocks.csv, for scripts/e16_join_clock.py and for the clock of every other run.
# 1. E16, three rounds r = 0, 1, 2 (E16_ROTATE = r for the FKL program, --rotate r for the Python baselines): the FKL program on both builds and the Python
#    baselines on both scripts, each run in its own folder. Round 0: FKL reference, FKL cleaned, Python reference, Python cleaned. Round 1: the reverse order
#    (Python cleaned, Python reference, FKL cleaned, FKL reference). Round 2: as round 0. The cleaned program and the cleaned baselines read the input files of
#    e16-work/inputs (written and hashed before the window; sha256 checked before the start); the reference program and the reference baselines make their own
#    inputs (the device generator, which gives the same bytes; seeded torch.randn). The query tile is not run.
# 2. E2 at k = 5002. The FKL program, three runs per build, the build order flipping every run (1 reference then cleaned, 2 cleaned then reference, 3 reference
#    then cleaned; the program always runs its six chain lengths), on the published input e2-work/e2_input.u8; its outputs go to e2-work-cleanup-d/<build>-run<n>,
#    their sha256 to the run folder. Then the cleaned Python driver with --ks 5002, three runs, run n with the outputs and the CSV of the cleaned FKL run n
#    (warm, L2-flushed and kernel-only).
# 3. E1's Nsight Compute pass of window C's cleaned build again, with --print-kernel-base mangled (the names scripts/e1_table.py joins with the SASS of the binary).
# Everything is pinned to cores 0-3,16-19 (the load logger on core 12, the sampler on core 13, both at nice 19). RTX 4090 = nvidia-smi index 1.
# START_DELAY seconds are waited first. Output: results/rtx4090/cleanup-window-d/{reference,cleaned}/.
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; source $FB/scripts/os_kernel_lines.sh
WD=$T/worktrees/fkl-benchmarks-window-d; WC=$T/worktrees/fkl-benchmarks-window-c
STEP0=$T/scratch/cleanup-window-d-build; PRE=$T/scratch/cleanup-window-d-pre; INPUTS=$T/e16-work/inputs
REF_E16=$FB/build/local-rtx4090/bin/e16_attention_cu; NEW_E16=$WD/build/local-rtx4090/bin/e16_attention_cu
REF_E2=$FB/build/local-rtx4090/bin/e2_fkl_chain_cu; NEW_E2=$WC/build/local-rtx4090/bin/e2_fkl_chain_cu; NEW_E1=$WC/build/rtx4090-e1/bin
REF_PY16=$FB/pytorch-attention/e16_baselines.py; NEW_PY16=$WD/pytorch-attention/e16_baselines.py; NEW_PY2=$WC/triton/e2_chain.py
REFC=a5dd9fa54210f60d0b631ce34508e95c86c387ce; NEWC_D=$(cat $STEP0/main-commit.txt 2>/dev/null); NEWC_C=1a2535cf090893bebc82714a410be8c300dde916
R=$FB/results/rtx4090; RC=$R/cleanup-window-c; OUT=$R/cleanup-window-d; LOG=$R/e0-run.log; E2OUT=$T/e2-work-cleanup-d; INPUT=$T/e2-work/e2_input.u8
E1P="e1_vf_chain_cu e1_hf_batch_cu e1_preproc_cu"
NCU=$T/toolchains/cuda-13.4.3/bin/ncu; PY=$T/venv-torch/bin/python
source $T/scratch/blocked_programs.sh   # no frozen 2025 program runs: refuses before anything starts
refuse_blocked_binaries $REF_E16 $NEW_E16 $REF_E2 $NEW_E2 || exit 1
for p in $E1P; do refuse_blocked_binaries $NEW_E1/$p || exit 1; done
grep -q 'all checks passed' $STEP0/console.txt 2>/dev/null || { echo "step 0 has not passed: not starting"; exit 1; }
grep -q 'pre-window steps ended: all checks passed' $PRE/console.txt 2>/dev/null || { echo "the pre-window steps have not passed: not starting"; exit 1; }
sha256sum -c --quiet $STEP0/binary-sha256.txt || { echo "an E16 binary differs from step 0's record: not starting"; exit 1; }
sha256sum -c --quiet $RC/binary-sha256.txt || { echo "an E1 or E2 binary differs from window C's record: not starting"; exit 1; }
for b in $REF_E16 $REF_E2; do strings $b | grep -q -x $REFC || { echo "$b: not $REFC: not starting"; exit 1; }; done
strings $NEW_E16 | grep -q -x "$NEWC_D" || { echo "$NEW_E16: not $NEWC_D: not starting"; exit 1; }
for b in $NEW_E2 $NEW_E1/e1_vf_chain_cu $NEW_E1/e1_hf_batch_cu $NEW_E1/e1_preproc_cu; do strings $b | grep -q -x $NEWC_C || { echo "$b: not $NEWC_C: not starting"; exit 1; }; done
[ "$(sha256sum < $INPUT | cut -d' ' -f1)" = "726b9ef38a8277d6021009d344446bf5915fe75f746d07950e608f63614a8995" ] || { echo "the E2 input is not the published one: not starting"; exit 1; }
( cd $INPUTS && sha256sum -c --quiet $PRE/inputs-sha256.txt ) || { echo "an E16 input file differs from the pre-window record: not starting"; exit 1; }
[ "$(ls $INPUTS | wc -l)" = "$(wc -l < $PRE/inputs-sha256.txt)" ] || { echo "e16-work/inputs holds other files: not starting"; exit 1; }
for f in $REF_PY16 $NEW_PY16 $NEW_PY2; do [ -f $f ] || { echo "$f missing: not starting"; exit 1; }; done
git -C $WD diff --quiet HEAD -- pytorch-attention/e16_baselines.py && git -C $WC diff --quiet HEAD -- triton/e2_chain.py || { echo "a cleaned Python script differs from its commit: not starting"; exit 1; }
git -C $FB diff --quiet HEAD -- pytorch-attention/e16_baselines.py && git -C $FB diff --quiet $REFC HEAD -- pytorch-attention/e16_baselines.py || { echo "the reference Python script is not a5dd9fa's: not starting"; exit 1; }
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
cp "$0" $OUT/window-script.sh
cp $STEP0/binary-sha256.txt $OUT/binary-sha256-e16.txt; cp $RC/binary-sha256.txt $OUT/binary-sha256-e1-e2-window-c.txt
cp $STEP0/main-commit.txt $STEP0/build-info-reference.txt $STEP0/build-info-cleaned.txt $OUT/; cp $PRE/inputs-sha256.txt $PRE/inputs-sizes.txt $OUT/
taskset -c 13 nice -n 19 nvidia-smi -i 1 --query-gpu=timestamp,clocks.sm,clocks.mem,pstate,utilization.gpu,power.draw,temperature.gpu --format=csv,noheader -lms 100 > $OUT/gpu_clocks.csv 2> $OUT/gpu_clocks.err &
SAMPLER_PID=$!
taskset -c 12 nice -n 19 python3 $FB/scripts/load_logger.py --log $OUT/load.log --interval 5 > /dev/null 2>&1 &
LOGGER_PID=$!
sleep 2
echo "# $(date -Iseconds) GPU window cleanup-d starts; load before: $(cut -d' ' -f1-3 /proc/loadavg); no compile job, transfer or measurement running (checked); the desktop session runs; clock sampler pid $SAMPLER_PID" | tee -a $LOG
os_kernel_lines | tee -a $LOG
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "compute contexts on the RTX 4090: $(nvidia-smi -i 1 --query-compute-apps=pid,process_name --format=csv,noheader | tr '\n' ';')"
  for r in $FB $FB/external/fkl $WC $WC/external/fkl $WD $WD/external/fkl; do echo "$r $(git -C $r rev-parse HEAD)"; git -C $r status --porcelain --untracked-files=no; done; } > $OUT/environment-before.txt 2>&1
nvidia-smi -i 1 --query-gpu=name,pci.bus_id,driver_version,pstate,clocks.sm,clocks.max.sm,power.limit,pcie.link.gen.current,pcie.link.width.current,utilization.gpu,temperature.gpu --format=csv > $OUT/gpu-info.csv
GENV="CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1"
# run <folder> <binary> [VAR=value ...]: one pinned run of a program in its own folder
run() {
  local dir=$1 bin=$2; shift 2
  mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  ( cd $dir && env $GENV "$@" timeout 1800 taskset -c 0-3,16-19 $bin > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) ${dir#$OUT/} $*: exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; INCOMPLETE in: $(grep -l '^# INCOMPLETE' $dir/*.csv 2>/dev/null | wc -l)"
}
# data_lines <folder>: the data lines of the CSV files of a run folder (comment lines and gpu-at-end.csv left out)
data_lines() { for f in $1/*.csv; do case $f in *gpu-at-end.csv) ;; *) printf '%s %s ' $(basename $f) $(grep -v -c -E '^#|^$' $f);; esac; done; }
# pyrun <folder> <script> [args ...]: one pinned run of a Python script (venv-torch, scripts/triton_env.sh) in its own folder
pyrun() {
  local dir=$1 script=$2; shift 2
  mkdir -p $dir; local start=$(date +%s)
  echo "load before: $(cut -d' ' -f1-3 /proc/loadavg); start $(date -Iseconds)" > $dir/load.txt
  ( cd $dir && source $FB/scripts/triton_env.sh > /dev/null 2>&1 && env $GENV timeout 3600 taskset -c 0-3,16-19 $PY $script "$@" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg); end $(date -Iseconds)" >> $dir/load.txt
  nvidia-smi -i 1 --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > $dir/gpu-at-end.csv
  echo "# $(date -Iseconds) ${dir#$OUT/} $(basename $script) $*: exit $(cat $dir/exit_code.txt) after $(( $(date +%s) - start )) s; data lines: $(data_lines $dir)"
}
e16_fkl() {   # e16_fkl <build> <round>
  if [ $1 = reference ]; then run $OUT/reference/e16-attention/run$(( $2 + 1 )) $REF_E16 E16_ROTATE=$2
  else run $OUT/cleaned/e16-attention/run$(( $2 + 1 )) $NEW_E16 E16_ROTATE=$2 E16_INPUT_DIR=$INPUTS; fi
}
e16_py() {    # e16_py <build> <round>
  if [ $1 = reference ]; then pyrun $OUT/reference/e16-baselines/run$(( $2 + 1 )) $REF_PY16 --out e16_baselines.csv --rotate $2
  else pyrun $OUT/cleaned/e16-baselines/run$(( $2 + 1 )) $NEW_PY16 --out e16_baselines.csv --rotate $2 --input-dir $INPUTS; fi
}
# 1. E16
for r in 0 1 2; do
  echo "# $(date -Iseconds) E16 round $r (E16_ROTATE=$r, --rotate $r)"
  if [ $r = 1 ]; then e16_py cleaned $r; e16_py reference $r; e16_fkl cleaned $r; e16_fkl reference $r
  else e16_fkl reference $r; e16_fkl cleaned $r; e16_py reference $r; e16_py cleaned $r; fi
done
# 2. E2 at k = 5002: the FKL program on both builds, then the cleaned Python driver
for n in 1 2 3; do
  order="reference cleaned"; [ $n = 2 ] && order="cleaned reference"
  for b in $order; do
    bin=$REF_E2; [ $b = cleaned ] && bin=$NEW_E2
    mkdir -p $E2OUT/$b-run$n
    run $OUT/$b/e2-fkl/run$n $bin E2_INPUT=$INPUT E2_OUT_DIR=$E2OUT/$b-run$n
    ( cd $E2OUT/$b-run$n && sha256sum fkl_out_k*.f32 ) > $OUT/$b/e2-fkl/run$n/outputs-sha256.txt
  done
done
for n in 1 2 3; do
  pyrun $OUT/cleaned/e2-python-k5002/run$n $NEW_PY2 --ks 5002 --out e2_chain.csv --input $INPUT --fkl-dir $E2OUT/cleaned-run$n --fkl-csv $OUT/cleaned/e2-fkl/run$n/e2_fkl_chain.csv
done
# 3. E1's Nsight Compute pass of window C's cleaned build, with mangled kernel names
declare -A EXP=([e1_vf_chain_cu]=e1-vf-chain [e1_hf_batch_cu]=e1-hf-batch [e1_preproc_cu]=e1-preproc)
$NCU --version > $OUT/cleaned/ncu-version.txt 2>&1
for p in $E1P; do
  d=$OUT/cleaned/e1-ncu/${EXP[$p]}; mkdir -p $d; start=$(date +%s)
  ( cd $d && env $GENV E1_PROFILE=1 timeout 1800 taskset -c 0-3,16-19 $NCU --csv --page raw --print-kernel-base mangled \
      --metrics gpu__time_duration.avg,gpu__time_duration.max,gpu__time_duration.min,gpu__time_duration.sum,sm__warps_active.avg.pct_of_peak_sustained_active,sm__warps_active.max.pct_of_peak_sustained_active,sm__warps_active.min.pct_of_peak_sustained_active,sm__warps_active.sum.pct_of_peak_sustained_active,sm__maximum_warps_per_active_cycle_pct,sm__maximum_warps_avg_per_active_cycle,smsp__maximum_warps_avg_per_active_cycle \
      $NEW_E1/$p > ncu.csv 2> ncu.err; echo $? > ncu.exit )
  echo "# $(date -Iseconds) ncu ${EXP[$p]} (mangled names): exit $(cat $d/ncu.exit) after $(( $(date +%s) - start )) s; kernel rows $(grep -c -E '^\"[0-9]+\",' $d/ncu.csv); mangled: $(grep -c -E '^\"[0-9]+\",.*\"_Z' $d/ncu.csv)"
done
sleep 2
kill $SAMPLER_PID $LOGGER_PID 2> /dev/null; wait $SAMPLER_PID $LOGGER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > $OUT/environment-after.txt 2>&1
echo "# $(date -Iseconds) GPU window cleanup-d ended; clock samples: $(wc -l < $OUT/gpu_clocks.csv)" | tee -a $LOG
echo "cleanup-d window end $(date -Iseconds)"
