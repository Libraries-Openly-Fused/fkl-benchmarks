#!/bin/bash
# C-PM-CUDA, the VI-L facts at the head of the pin move (fkl-benchmarks 2518c75, FKL 48fbf3f): COMPILE ONLY, nothing is executed and no GPU
# is visible (CUDA_VISIBLE_DEVICES is empty). scripts/e3_sweep.py of the head checkout, one compile at a time per variant, each variant pinned
# to its own core (8 and 9; their SMT siblings 24 and 25 stay idle), one repetition (the compile times are not a result of this check), run
# through scratch/pin-435/e3_sweep_nogpu.py (e3_sweep.py unchanged, with a CSV header that does not query the GPU):
#   variant A at N = 1, 100, 500, 1000, 1500, 2000, 3000, 4000: the registers, spills and stack of the fused kernel (the paper: at most 17
#             registers and no local memory);
#   variant B at N = 3847 and 3848: 3847 compiles, 3848 fails in ptxas with the parameter-space message (the paper's N = 3,848 limit).
# Output: scratch/pin-435/e3/{A,B}/ (e3_compile.csv and the logs of e3_sweep.py), scratch/pin-435/e3/console.txt.
set -u
T=/home/oscar/tfkl-bench; O=$T/scratch/pin-435/e3; W=$T/worktrees/fkl-benchmarks-pin-435
W_SHA=2518c75219a4af8039f9a8d062323bcbb4181487; W_FKL=48fbf3f08e7551c3eb70ac2302d8ebc15ecad645
say() { echo "# $(date -Iseconds) $*"; }
fail() { say "STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
mkdir -p $O/tmp $O/A $O/B $O/work-A $O/work-B
exec > >(tee -a $O/console.txt) 2>&1
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
export TMPDIR=$O/tmp
[ "$(git -C $W rev-parse HEAD)" = $W_SHA ] || fail "the head worktree is not at 2518c75"
[ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] || fail "the head external/fkl is not at 48fbf3f"
[ -z "$(git -C $W status --porcelain --untracked-files=no)" ] || fail "the head worktree has tracked changes"
[ -z "$(git -C $W/external/fkl status --porcelain --untracked-files=no)" ] || fail "the head external/fkl has tracked changes"
cp "$0" $O/e3_runs.sh; cp $T/scratch/pin-435/e3_sweep_nogpu.py $O/
say "E3 compiles start (compile only, CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'); script sha256 $(sha256sum "$0" | cut -d' ' -f1); e3_sweep.py sha256 $(sha256sum $W/scripts/e3_sweep.py | cut -d' ' -f1); e3_sweep_nogpu.py sha256 $(sha256sum $T/scratch/pin-435/e3_sweep_nogpu.py | cut -d' ' -f1); nvcc $(nvcc --version | grep release | sed 's/^ *//'); load $(cut -d' ' -f1-3 /proc/loadavg)"
cd $W || fail "cd $W"
python3 -I $T/scratch/pin-435/e3_sweep_nogpu.py $W --variants A --ns 1,100,500,1000,1500,2000,3000,4000 --core 8 --reps 1 --out $O/A --work $O/work-A > $O/A/sweep.log 2>&1 &
PA=$!
python3 -I $T/scratch/pin-435/e3_sweep_nogpu.py $W --variants B --ns 3847,3848 --core 9 --reps 1 --out $O/B --work $O/work-B > $O/B/sweep.log 2>&1 &
PB=$!
wait $PA; RA=$?; say "variant A ended, exit $RA"
wait $PB; RB=$?; say "variant B ended, exit $RB"
[ "$(git -C $W rev-parse HEAD)" = $W_SHA ] && [ "$(git -C $W/external/fkl rev-parse HEAD)" = $W_FKL ] || fail "the head checkout moved during the compiles"
for v in A B; do echo "== $v"; tail -5 $O/$v/sweep.log; ls $O/$v; done
say "E3 compiles ended (A exit $RA, B exit $RB); load $(cut -d' ' -f1-3 /proc/loadavg)"
