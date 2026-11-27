#!/bin/bash
# RT-4, the code check of A against "old" (COMPILE ONLY; no GPU is visible; no program runs). "old" = #63's B
# (results/rtx4090/e18-retime-b10fc66/binaries.sha256: fkl-benchmarks 2160c70 with FKL b10fc66, built by scripts/check_fkl_tree.sh --only bench
# into scratch/pin-433/head-b10fc66; that build was removed after #63). It is rebuilt here, in the same folder, so that every path the
# compilers see is the one #63's B saw:
#   - the source: the clean worktree worktrees/fkl-benchmarks-pin-435-base at 976e42b, A's own checkout. Its tree outside results/ is
#     2160c70's (git diff --name-only 2160c70 976e42b lists only results/), with the same FKL b10fc66, cvGPUSpeedup 6d26c65, FastNPP a1b5e24;
#   - the script: check_fkl_tree_rt4.sh = RT-1's check_fkl_tree_rt.sh (scripts/check_fkl_tree.sh with the checkout from CHECK_ROOT) with the
#     targets cut to the three E18 programs;
#   - the options: the reference build build/local-rtx4090 (only read), as #63's B and A.
# Then the sha256 of the three binaries against #63's manifest. nice 19, 8 jobs, cores 8-15,24-31 (fkl-reduce has priority).
set -u
T=/home/oscar/tfkl-bench; R=$T/scratch/rt4; OUT=$T/scratch/pin-433/head-b10fc66; LOG=$R/rebuild-old-console.txt
WT=$T/worktrees/fkl-benchmarks-pin-435-base; REF=$T/fkl-benchmarks/build/local-rtx4090
[ -e $OUT ] && { echo "$OUT exists: not starting"; exit 1; }
[ -e $LOG ] && { echo "$LOG exists: not starting"; exit 1; }
exec > >(tee -a $LOG) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
fail() { say "STOP: $*"; exit 1; }
source $T/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES=
say "RT-4 rebuild of #63's B starts (compile only, nice 19, -j8); script sha256 $(sha256sum "$0" | cut -d' ' -f1), build script $(sha256sum $R/check_fkl_tree_rt4.sh | cut -d' ' -f1); load $(cut -d' ' -f1-3 /proc/loadavg)"
[ "$(git -C $WT rev-parse HEAD)" = 976e42bfad0ba61c3bb22a555cf72f4da6896aeb ] || fail "the worktree is not at 976e42b"
[ -z "$(git -C $WT status --porcelain --untracked-files=all)" ] || fail "the worktree is not clean"
[ "$(git -C $WT/external/fkl rev-parse HEAD)" = b10fc6609426bd18020b9213c7196b528fc677a8 ] || fail "external/fkl is not at b10fc66"
[ -z "$(git -C $T/fkl-benchmarks diff --name-only 2160c70d34f980b87601bebf7ad16658184a019f 976e42bfad0ba61c3bb22a555cf72f4da6896aeb | grep -v '^results/')" ] \
  || fail "2160c70 and 976e42b differ outside results/"
say "2160c70 and 976e42b differ only under results/; worktree clean at 976e42b, FKL b10fc66"
mkdir -p $(dirname $OUT)
CHECK_ROOT=$WT nice -n 19 bash $R/check_fkl_tree_rt4.sh $WT/external/fkl --out $OUT --only bench --ref-build $REF --jobs 8 --cores 8-15,24-31
RC=$?; say "build exit $RC"
[ $RC = 0 ] || fail "the build failed"
[ "$(git -C $WT rev-parse HEAD)" = 976e42bfad0ba61c3bb22a555cf72f4da6896aeb ] && [ -z "$(git -C $WT status --porcelain --untracked-files=all)" ] || fail "the worktree changed"
# #63's manifest, from main
git -C $T/fkl-benchmarks show 8c419c772425fcb8191bf0b3a97dc33d82ca5a8f:results/rtx4090/e18-retime-b10fc66/binaries.sha256 | grep 'pin-433/head-b10fc66' > $R/old-manifest-63B.txt
(cd $OUT/bench-build/bin && sha256sum e18_box_cu e18_box_split_cu e18_stencil_cu | sed "s#  #  $OUT/bench-build/bin/#") > $R/rebuilt-old-sha256.txt
if cmp -s <(sort -k2 $R/old-manifest-63B.txt) <(sort -k2 $R/rebuilt-old-sha256.txt); then say "the rebuild REPRODUCES #63's B: all three sha256 equal"
else say "the rebuild does NOT reproduce #63's B byte for byte:"; diff <(sort -k2 $R/old-manifest-63B.txt) <(sort -k2 $R/rebuilt-old-sha256.txt); fi
say "RT-4 rebuild of #63's B ends; load $(cut -d' ' -f1-3 /proc/loadavg)"
