#!/bin/bash
# Task C-NV: compiles the before/after harness (pr435_image.cu, pr435_attention.cu) against FKL b10fc66 (base) and #435's head
# 48fbf3f (-DPR435_HEAD; the head since 2026-11-26 14:50, e322cad + a commit that changes only a test file; its worktree keeps
# the path worktrees/fkl-pr435-e322cad). COMPILE ONLY: no GPU is visible. The nvcc flags are those of FKL's CUDA tests in the step 1 builds
# (CUDA_FLAGS of their flags.make: -O3 -DNDEBUG -std=c++17 --generate-code=arch=compute_89,code=[compute_89,sm_89]
# -ftemplate-depth=1000 -lineinfo), with FKL's include folder; no CPU_MULTITHREAD (the harness uses only the GPU).
# Usage: build_harness.sh <output folder, new>
set -eu
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
T=/home/oscar/tfkl-bench; S=$T/scratch/pr435; H=$S/harness
export TMPDIR=$S/tmp; mkdir -p $TMPDIR
OUT=${1:?output folder}
[ -e "$OUT" ] && { echo "$OUT exists"; exit 1; }
mkdir -p "$OUT"
exec > >(tee "$OUT/build-console.txt") 2>&1
echo "# $(date -Iseconds) harness build starts; nvcc $($FKLB_CUDA_HOME/bin/nvcc --version | grep release | sed 's/^ *//')"
for f in pr435_common.h pr435_image.cu pr435_attention.cu build_harness.sh compare_harness.py; do echo "source sha256 $(sha256sum $H/$f)"; done
FLAGS=(-O3 -DNDEBUG -std=c++17 "--generate-code=arch=compute_89,code=[compute_89,sm_89]" -ftemplate-depth=1000 -lineinfo)
build() {  # side tree sha define
  local side=$1 tree=$2 sha=$3 def=$4 d=$OUT/$1
  mkdir -p $d
  [ "$(git -C $tree rev-parse HEAD)" = $sha ] || { echo "$tree is not at $sha"; exit 1; }
  [ -z "$(git -C $tree status --porcelain --ignore-submodules=none --untracked-files=all)" ] || { echo "$tree is not clean"; exit 1; }
  echo "$side: FKL tree $tree at $(git -C $tree rev-parse HEAD) ($(git -C $tree describe --always --dirty --abbrev=7)), clean"
  for p in pr435_image pr435_attention; do
    local cmd=($FKLB_CUDA_HOME/bin/nvcc "${FLAGS[@]}" $def -I$H -I$tree/include $H/$p.cu -o $d/$p)
    echo "$side $p: ${cmd[*]}"
    local t0=$(date +%s)
    taskset -c 8-15,24-31 nice -n 10 "${cmd[@]}" > $d/$p-build.log 2>&1 || { echo "$side $p: BUILD FAILED"; head -40 $d/$p-build.log; exit 1; }
    echo "$side $p: built in $(( $(date +%s) - t0 )) s, sha256 $(sha256sum < $d/$p | cut -c1-64), $(grep -c warning $d/$p-build.log) warning lines"
  done
}
build base-b10fc66 $T/worktrees/fkl-pr435-base-b10fc66 b10fc6609426bd18020b9213c7196b528fc677a8 ""
build head-48fbf3f $T/worktrees/fkl-pr435-e322cad 48fbf3f08e7551c3eb70ac2302d8ebc15ecad645 "-DPR435_HEAD"
echo "# $(date -Iseconds) harness build ends"
