#!/bin/bash
# Step 0 of the cleanup window D: compile only (no measurement, no GPU work). Builds the cleaned E16 program from a clean checkout of fkl-benchmarks main
# (the commit origin/main points at when this starts, recorded): a detached worktree of its own, the submodules at the commits main pins (cloned from the reference
# checkout, which pins the same commits), and one build with the preset of its reference:
#   build/local-rtx4090  preset local-rtx4090 (as build/local-rtx4090 of the reference checkout): e16_attention_cu
# The E1 and E2 binaries of window D are window C's cleaned builds (their sources are unchanged since 1a2535c; the window checks their sha256 against window C's record).
# Every check stops the script:
#   - no tracked change in the checkout or its submodules, before and after the build;
#   - the build info carries the main commit without -dirty and is otherwise equal to its reference's;
#   - the cache options and the compile flags of the target equal the reference's;
#   - the binary carries the commit;
#   - SASS against the reference (by demangled name): 41 kernels identical, and only in the reference the 5 device input generators that the cleanup replaced
#     with the host generator (bit-identical bytes, checked before PR #26).
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-window-d; O=$T/scratch/cleanup-window-d-build
SUBS="external/fkl external/FastNPP external/cvGPUSpeedup external/fkl/thread-pool"
E16T="e16_attention_cu"
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
[ -e $W ] && { echo "$W exists: not starting"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/console.txt) 2>&1
git -C $FB fetch -q origin || fail "git fetch"
MAIN=$(git -C $FB rev-parse origin/main)
echo "# $(date -Iseconds) step 0 of window D starts (compile only); main = $MAIN"
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
clean_check "before the build"
cmake --preset local-rtx4090 > $O/configure.log 2>&1 || fail "configure (configure.log)"
echo "# $(date -Iseconds) build starts"
taskset -c 8-15,24-31 nice -n 10 cmake --build $W/build/local-rtx4090 --target $E16T -j 12 > $O/build.log 2>&1 || fail "build (build.log)"
echo "# $(date -Iseconds) build ended"
clean_check "after the build"

keys='^(CMAKE_BUILD_TYPE|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_HOST_COMPILER|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELEASE|OpenCV_DIR|OPENCV_DIR|FKLB_[A-Z0-9_]+|E1_[A-Z_]+|E2_[A-Z_]+|E16_[A-Z_]+)(:[A-Z]+)?='
REFB=$FB/build/local-rtx4090; NEWB=$W/build/local-rtx4090
grep -E '^#define FKLB_' $REFB/generated/fklb_build_info.h > $O/build-info-reference.txt
grep -E '^#define FKLB_' $NEWB/generated/fklb_build_info.h > $O/build-info-cleaned.txt
grep -q "FKLB_COMMIT_REPO \"$MAIN\"" $O/build-info-cleaned.txt || fail "build info: the repository commit is not $MAIN"
grep -q -e '-dirty' $O/build-info-cleaned.txt && fail "build info: -dirty"
diff <(grep -v FKLB_COMMIT_REPO $O/build-info-reference.txt) <(grep -v FKLB_COMMIT_REPO $O/build-info-cleaned.txt) > $O/build-info.diff \
  || fail "build info differs from its reference beyond the repository commit (build-info.diff)"
diff <(grep -E "$keys" $REFB/CMakeCache.txt | sort) <(grep -E "$keys" $NEWB/CMakeCache.txt | sort) > $O/cache.diff || fail "cache options differ (cache.diff)"
for t in $E16T; do
  fr=$(find $REFB -path "*/$t.dir/flags.make" | head -1); fn=$(find $NEWB -path "*/$t.dir/flags.make" | head -1)
  [ -n "$fr" ] && [ -n "$fn" ] || fail "flags.make of $t not found"
  # both build folders become BUILD before the checkouts become SRC (the folders have the same name here, the order matters only when they differ)
  diff <(sed -e "s#$REFB#BUILD#g" -e "s#$FB#SRC#g" $fr) <(sed -e "s#$NEWB#BUILD#g" -e "s#$W#SRC#g" $fn) > $O/flags-$t.diff || fail "compile flags of $t differ (flags-$t.diff)"
  strings $NEWB/bin/$t | grep -q -x $MAIN || fail "$t does not carry $MAIN"
  strings $NEWB/bin/$t | grep -q -E '^[0-9a-f]{40}-dirty$' && fail "$t carries a -dirty commit"
done
echo "build info $MAIN without -dirty, otherwise as $REFB; cache options and compile flags as the reference; the binary carries the commit"
mkdir -p $O/sass/reference $O/sass/cleaned
for t in $E16T; do cp $REFB/bin/$t $O/sass/reference/; cp $NEWB/bin/$t $O/sass/cleaned/; done
r=$(python3 $W/scripts/sass_compare_builds.py $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'e16_*' --match-demangled --out $O/sass-e16.txt | grep '^kernels identical'); echo "E16: $r"
[ "$r" = "kernels identical: 41, different: 0, only in A: 5, only in B: 0" ] || fail "SASS of E16 (sass-e16.txt)"
( for t in $E16T; do sha256sum $REFB/bin/$t; done; for t in $E16T; do sha256sum $NEWB/bin/$t; done ) > $O/binary-sha256.txt
cat $O/binary-sha256.txt
echo "# $(date -Iseconds) step 0 ended: all checks passed"
