#!/bin/bash
# Step 0 of the cleanup windows A and B: compile only (no measurement, no GPU work). Builds the cleaned binaries from a clean checkout of
# fkl-benchmarks main 836a98e:
#   - a detached worktree of its own;
#   - the submodules at the commits main pins, cloned from the reference checkout, which pins the same commits;
#   - the reference's preset, local-rtx4090;
#   - the five programs of windows A and B.
# Every check below stops the script:
#   - no tracked change in the checkout or its submodules, before and after the build;
#   - the build info carries 836a98e without -dirty, with the same pins, compiler and flags as build/local-rtx4090 (the reference, read only);
#   - the cache options and the compile flags of the five targets equal the reference's;
#   - the binaries carry the commit;
#   - SASS against the reference: 47 of 47 for the NPP and CPU-time programs, 5 of 5 for E9 (R1's thread-fusion template arguments ignored),
#     60 identical for E4 with the 2 frame kernels renamed by design.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-window; O=$T/scratch/cleanup-window-build
MAIN=836a98e025bb4182528cbf47f082d548a943fbac
REFB=$FB/build/local-rtx4090; NEWB=$W/build/local-rtx4090
TARGETS="benchmark_npp_batchresize_x_split3D_cu benchmark_CPU_NPP_vs_FastNPP_cu benchmark_CPU_OpenCV_vs_cvGS_cu e9_cross_library_cu e4_preprocessing_cu"
SUBS="external/fkl external/FastNPP external/cvGPUSpeedup external/fkl/thread-pool"
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
[ -e $W ] && { echo "$W exists: not starting"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/console.txt) 2>&1
echo "# $(date -Iseconds) step 0 starts (compile only)"
git -C $FB cat-file -e "$MAIN^{commit}" || fail "$MAIN is not in the repository"
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
[ -f $NEWB/CMakeCache.txt ] || fail "no $NEWB/CMakeCache.txt"
echo "# $(date -Iseconds) build starts: $TARGETS"
taskset -c 8-15,24-31 nice -n 10 cmake --build $NEWB --target $TARGETS -j 12 > $O/build.log 2>&1 || fail "build (build.log)"
echo "# $(date -Iseconds) build ended; warnings $(grep -c -i warning $O/build.log)"
clean_check "after the build"

# build info: the same as the reference's except the commit of this repository
grep -E '^#define FKLB_' $REFB/generated/fklb_build_info.h > $O/build-info-reference.txt
grep -E '^#define FKLB_' $NEWB/generated/fklb_build_info.h > $O/build-info-cleaned.txt
grep -q "FKLB_COMMIT_REPO \"$MAIN\"" $O/build-info-cleaned.txt || fail "build info: the repository commit is not $MAIN"
grep -q -e '-dirty' $O/build-info-cleaned.txt && fail "build info: -dirty"
diff <(grep -v FKLB_COMMIT_REPO $O/build-info-reference.txt) <(grep -v FKLB_COMMIT_REPO $O/build-info-cleaned.txt) > $O/build-info.diff \
  || fail "build info differs from the reference beyond the repository commit (build-info.diff)"
echo "build info: $MAIN, no -dirty; pins, nvcc, arch, host compiler, build type, flags as the reference"
# cache options and the compile flags of the five targets (source paths replaced)
keys='^(CMAKE_BUILD_TYPE|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_HOST_COMPILER|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELEASE|OpenCV_DIR|OPENCV_DIR|FKLB_[A-Z0-9_]+)(:[A-Z]+)?='
diff <(grep -E "$keys" $REFB/CMakeCache.txt | sort) <(grep -E "$keys" $NEWB/CMakeCache.txt | sort) > $O/cache.diff || fail "cache options differ (cache.diff)"
for t in $TARGETS; do
  fr=$(find $REFB -path "*/$t.dir/flags.make" | head -1); fn=$(find $NEWB -path "*/$t.dir/flags.make" | head -1)
  [ -n "$fr" ] && [ -n "$fn" ] || fail "flags.make of $t not found"
  # the one difference by design: E4 no longer has the include directory handwritten-cuda (PR #23)
  x=""; [ $t = e4_preprocessing_cu ] && x='s# -ISRC/handwritten-cuda##'
  diff <(sed -e "s#$FB#SRC#g" -e "$x" $fr) <(sed -e "s#$W#SRC#g" $fn) > $O/flags-$t.diff || fail "compile flags of $t differ (flags-$t.diff)"
done
echo "cache options and compile flags: as the reference (E4: without the include directory handwritten-cuda, by design)"
for t in $TARGETS; do
  strings $NEWB/bin/$t | grep -q -x $MAIN || fail "$t does not carry $MAIN"
  strings $NEWB/bin/$t | grep -q -E '^[0-9a-f]{40}-dirty$' && fail "$t carries a -dirty commit"
done
echo "binaries: all carry $MAIN, none -dirty"
# SASS against the reference (copies, so the reference is only read)
mkdir -p $O/sass/reference $O/sass/cleaned
for t in $TARGETS; do cp $REFB/bin/$t $O/sass/reference/; cp $NEWB/bin/$t $O/sass/cleaned/; done
S=$W/scripts/sass_compare_builds.py
r=$(python3 $S $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'benchmark_*' --out $O/sass-npp.txt | grep '^kernels identical'); echo "NPP and CPU time: $r"
[ "$r" = "kernels identical: 47, different: 0, only in A: 0, only in B: 0" ] || fail "SASS of the NPP and CPU-time programs (sass-npp.txt)"
r=$(python3 $S $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'e9_cross_library_cu' --match-demangled \
      --ignore '\(fk::TF\)[01],|(?<=TransformDPPDetails_<void,)(?:true|false),' --out $O/sass-e9.txt | grep '^kernels identical'); echo "E9: $r"
[ "$r" = "kernels identical: 5, different: 0, only in A: 0, only in B: 0" ] || fail "SASS of E9 (sass-e9.txt)"
r=$(python3 $S $O/sass/reference $O/sass/cleaned --arch sm_89 --glob 'e4_preprocessing_cu' --out $O/sass-e4.txt | grep '^kernels identical'); echo "E4: $r"
[ "$r" = "kernels identical: 60, different: 0, only in A: 2, only in B: 2" ] || fail "SASS of E4 (sass-e4.txt)"
( for t in $TARGETS; do sha256sum $REFB/bin/$t; done; for t in $TARGETS; do sha256sum $NEWB/bin/$t; done ) > $O/binary-sha256.txt
cat $O/binary-sha256.txt
echo "# $(date -Iseconds) step 0 ended: all checks passed"
