#!/bin/bash
# Task C-MAIN-NV: compiles the before/after harness (pr436_image.cu, pr436_attention.cu) against FKL main ebb3891 (base) and FKL #436's
# head 1df2738 (-DPR436_HEAD). COMPILE ONLY: no GPU is visible. The nvcc flags are those of FKL main's CUDA tests in the step 1 builds
# (CUDA_FLAGS of their flags.make: -O3 -DNDEBUG -std=c++20 --generate-code=arch=compute_89,code=[compute_89,sm_89] -ftemplate-depth=1000
# -lineinfo; main is C++20), with FKL's include folder; no CPU_MULTITHREAD (the harness uses only the GPU). Another session (fkl-reduce)
# has priority on this machine: nice 19, the 4 compiles at the same time (at most 8 jobs), on cores 8-15,24-31.
# Usage: build_harness.sh <output folder, new>
set -eu
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/pr436; H=$S/harness
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
OUT=${1:?output folder}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/build-console.txt") 2>&1
echo "# $(date -Iseconds) harness build starts; nvcc $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//')"
for f in pr436_common.h pr436_image.cu pr436_attention.cu build_harness.sh compare_harness.py; do echo "source sha256 $(sha256sum $H/$f)"; done
FLAGS=(-O3 -DNDEBUG -std=c++20 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -ftemplate-depth=1000 -lineinfo)
build() {  # side tree sha define program
  local side=$1 tree=$2 sha=$3 def=$4 p=$5 d=$OUT/$1
  mkdir -p $d
  [ "$(git -C $tree rev-parse HEAD)" = $sha ] || { echo "$tree is not at $sha"; return 1; }
  [ -z "$(git -C $tree status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "$tree is not clean"; return 1; }
  local cmd=($FKLB_CUDA_HOME/bin/nvcc "${FLAGS[@]}" $def -I$H -I$tree/include $H/$p.cu -o $d/$p)
  echo "$side $p: FKL tree $tree at $(git -C $tree rev-parse HEAD) ($(git -C $tree describe --always --dirty --abbrev=7)), clean; ${cmd[*]}"
  local t0=$(date +%s)
  taskset -c 8-15,24-31 nice -n 19 "${cmd[@]}" > $d/$p-build.log 2>&1 || { echo "$side $p: BUILD FAILED"; head -40 $d/$p-build.log; return 1; }
  echo "$side $p: built in $(( $(date +%s) - t0 )) s, sha256 $(sha256sum < $d/$p | cut -c1-64), $(grep -c warning $d/$p-build.log) warning lines"
}
B=$T/worktrees/fkl-pr436-base-ebb3891; BS=ebb3891bea18a152d92dfd5092e117160490e34b
HD=$T/worktrees/fkl-pr436-head-1df2738; HS=1df2738e7bd8eba1bfb1a0baf9ad215062d014ca
pids=()
for p in pr436_image pr436_attention; do
  build base-ebb3891 $B $BS "" $p & pids+=($!)
  build head-1df2738 $HD $HS "-DPR436_HEAD" $p & pids+=($!)
done
rc=0; for pid in "${pids[@]}"; do wait $pid || rc=1; done
echo "# $(date -Iseconds) harness build ends, exit $rc"
exit $rc
