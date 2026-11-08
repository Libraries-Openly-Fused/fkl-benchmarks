#!/bin/bash
# Step 0 of the cleanup window C: compile only (no measurement, no GPU work). Builds the cleaned E1 and E2 binaries from a clean checkout of fkl-benchmarks main
# (the commit origin/main points at when this starts, recorded): a detached worktree of its own, the submodules at the commits main pins (cloned from the reference
# checkout, which pins the same commits), and two builds, each with the preset of its reference:
#   build/rtx4090-e1   preset rtx4090-e1 with OpenCV and NPP off (as build/ref-rtx4090-e1, the flags of the published E1 builds): e1_vf_chain, e1_hf_batch, e1_preproc
#   build/local-rtx4090  preset local-rtx4090 (as build/local-rtx4090 of the reference checkout): e2_fkl_chain
# Every check stops the script:
#   - no tracked change in the checkout or its submodules, before and after the builds;
#   - the build info carries the main commit without -dirty and is otherwise equal to its reference's;
#   - the cache options and the compile flags of the targets equal the reference's;
#   - the binaries carry the commit;
#   - SASS against the reference (by demangled name): E1 54 of 54 and E2 7 of 7 identical, plus e1::gpuSpin in each binary (only in the cleaned build).
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-window-c; O=$T/scratch/cleanup-window-c-build
SUBS="external/fkl external/FastNPP external/cvGPUSpeedup external/fkl/thread-pool"
E1T="e1_vf_chain_cu e1_hf_batch_cu e1_preproc_cu"; E2T="e2_fkl_chain_cu"
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
[ -e $W ] && { echo "$W exists: not starting"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/console.txt) 2>&1
git -C $FB fetch -q origin || fail "git fetch"
MAIN=$(git -C $FB rev-parse origin/main)
echo "# $(date -Iseconds) step 0 of window C starts (compile only); main = $MAIN"
echo "$MAIN" > $O/main-commit.txt
git -C $FB worktree add --detach $W $MAIN || fail "git worktree add"
cd $W || fail "cd $W"
git -c protocol.file.allow=always -c submodule.external/fkl.url=$FB/external/fkl -c submodule.external/FastNPP.url=$FB/external/FastNPP \
    -c submodule.external/cvGPUSpeedup.url=$FB/external/cvGPUSpeedup submodule update --init external/fkl external/FastNPP external/cvGPUSpeedup || fail "submodules"
git -C external/fkl -c protocol.file.allow=always -c submodule.thread-pool.url=$FB/external/fkl/thread-pool submodule update --init thread-pool || fail "thread-pool"
cp $FB/CMakeUserPresets.json $W/ || fail "CMakeUserPresets.json"
for s in $SUBS; do
  a=$(git -C $W/$s rev-parse HEAD); b=$(git -C $FB/$s rev-parse HEAD); echo "$s $a (reference checkout $b)"
  [ "$a" = "$b" ] || fail "$s is not at the commit of the reference checkout"
done
clean_check() {
  for r in $W $(for s in $SUBS; do echo $W/$s; done); do
    d=$(git -C $r status --porcelain --untracked-files=no); [ -z "$d" ] || fail "tracked changes in $r ($1): $d"
  done
  echo "no tracked change in the checkout or its submodules ($1)"
}
clean_check "before the builds"
cmake --preset rtx4090-e1 -B $W/build/rtx4090-e1 -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF > $O/configure-e1.log 2>&1 || fail "configure E1 (configure-e1.log)"
cmake --preset local-rtx4090 > $O/configure-e2.log 2>&1 || fail "configure E2 (configure-e2.log)"
echo "# $(date -Iseconds) builds start"
taskset -c 8-15,24-31 nice -n 10 cmake --build $W/build/rtx4090-e1 --target $E1T -j 12 > $O/build-e1.log 2>&1 || fail "build E1 (build-e1.log)"
taskset -c 8-15,24-31 nice -n 10 cmake --build $W/build/local-rtx4090 --target $E2T -j 12 > $O/build-e2.log 2>&1 || fail "build E2 (build-e2.log)"
echo "# $(date -Iseconds) builds ended"
clean_check "after the builds"

keys='^(CMAKE_BUILD_TYPE|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_HOST_COMPILER|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELEASE|OpenCV_DIR|OPENCV_DIR|FKLB_[A-Z0-9_]+|E1_[A-Z_]+|E2_[A-Z_]+)(:[A-Z]+)?='
for pair in "e1 $FB/build/ref-rtx4090-e1 $W/build/rtx4090-e1 $E1T" "e2 $FB/build/local-rtx4090 $W/build/local-rtx4090 $E2T"; do
  set -- $pair; tag=$1; REFB=$2; NEWB=$3; shift 3; targets="$*"
  grep -E '^#define FKLB_' $REFB/generated/fklb_build_info.h > $O/build-info-reference-$tag.txt
  grep -E '^#define FKLB_' $NEWB/generated/fklb_build_info.h > $O/build-info-cleaned-$tag.txt
  grep -q "FKLB_COMMIT_REPO \"$MAIN\"" $O/build-info-cleaned-$tag.txt || fail "build info $tag: the repository commit is not $MAIN"
  grep -q -e '-dirty' $O/build-info-cleaned-$tag.txt && fail "build info $tag: -dirty"
  diff <(grep -v FKLB_COMMIT_REPO $O/build-info-reference-$tag.txt) <(grep -v FKLB_COMMIT_REPO $O/build-info-cleaned-$tag.txt) > $O/build-info-$tag.diff \
    || fail "build info $tag differs from its reference beyond the repository commit (build-info-$tag.diff)"
  diff <(grep -E "$keys" $REFB/CMakeCache.txt | sort) <(grep -E "$keys" $NEWB/CMakeCache.txt | sort) > $O/cache-$tag.diff || fail "cache options $tag differ (cache-$tag.diff)"
  for t in $targets; do
    fr=$(find $REFB -path "*/$t.dir/flags.make" | head -1); fn=$(find $NEWB -path "*/$t.dir/flags.make" | head -1)
    [ -n "$fr" ] && [ -n "$fn" ] || fail "flags.make of $t not found"
    # the build folders have different names (build/ref-rtx4090-e1 against build/rtx4090-e1): both become BUILD before the checkouts become SRC
    diff <(sed -e "s#$REFB#BUILD#g" -e "s#$FB#SRC#g" $fr) <(sed -e "s#$NEWB#BUILD#g" -e "s#$W#SRC#g" $fn) > $O/flags-$t.diff || fail "compile flags of $t differ (flags-$t.diff)"
    strings $NEWB/bin/$t | grep -q -x $MAIN || fail "$t does not carry $MAIN"
    strings $NEWB/bin/$t | grep -q -E '^[0-9a-f]{40}-dirty$' && fail "$t carries a -dirty commit"
  done
  echo "$tag: build info $MAIN without -dirty, otherwise as $REFB; cache options and compile flags as the reference; binaries carry the commit"
done
mkdir -p $O/sass/reference $O/sass/cleaned
for t in $E1T; do cp $FB/build/ref-rtx4090-e1/bin/$t $O/sass/reference/; cp $W/build/rtx4090-e1/bin/$t $O/sass/cleaned/; done
for t in $E2T; do cp $FB/build/local-rtx4090/bin/$t $O/sass/reference/; cp $W/build/local-rtx4090/bin/$t $O/sass/cleaned/; done
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'e1_*' --match-demangled --out $O/sass-e1.txt | grep '^kernels identical'); echo "E1: $r"
[ "$r" = "kernels identical: 54, different: 0, only in A: 0, only in B: 3" ] || fail "SASS of E1 (sass-e1.txt)"
r=$(python3 $S $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'e2_*' --match-demangled --out $O/sass-e2.txt | grep '^kernels identical'); echo "E2: $r"
[ "$r" = "kernels identical: 7, different: 0, only in A: 0, only in B: 1" ] || fail "SASS of E2 (sass-e2.txt)"
( for t in $E1T; do sha256sum $FB/build/ref-rtx4090-e1/bin/$t; done; for t in $E2T; do sha256sum $FB/build/local-rtx4090/bin/$t; done
  for t in $E1T; do sha256sum $W/build/rtx4090-e1/bin/$t; done; for t in $E2T; do sha256sum $W/build/local-rtx4090/bin/$t; done ) > $O/binary-sha256.txt
cat $O/binary-sha256.txt
echo "# $(date -Iseconds) step 0 ended: all checks passed"
