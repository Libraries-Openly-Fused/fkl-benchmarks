#!/bin/bash
# Step 0 of the E4 window G, the final RTX 4090 data of Figure 15 (compile only: no measurement, no GPU work). Builds e4_preprocessing_cu from a clean
# checkout of fkl-benchmarks main (the merge of #35, hand_roi; recorded): a detached worktree of its own, the submodules at the commits main pins (cloned from the
# reference checkout, which pins the same commits), preset local-rtx4090, one fresh configure. Every check stops the script:
#   - no tracked change in the checkout or its submodules, before and after the build;
#   - the build info carries the main commit without -dirty and is otherwise equal to that of the window e4-hand-cvgs (worktrees/fkl-benchmarks-window-f,
#     2dac1c8; only read);
#   - the cache options, the compile flags and the CUDA include directories equal 2dac1c8's;
#   - the binary carries the commit;
#   - SASS by demangled name: identical to the binary of #35's head 0f96c4f (worktrees/fkl-benchmarks-e4-roi, 7bfd2c5a...; only read), all 182 kernels;
#     against 2dac1c8's binary (8b56ffc2...): its 152 kernels identical, the 30 of hand_roi new.
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=$T/worktrees/fkl-benchmarks-window-g; O=$T/scratch/e4-window-g-build
WT=$T/worktrees/fkl-benchmarks-window-f; WB=$WT/build/local-rtx4090
PR=$T/worktrees/fkl-benchmarks-e4-roi/build/local-rtx4090
MAIN_EXPECTED=469aa6df965f4728101076121d33b9b04d0def6b
SUBS="external/fkl external/FastNPP external/cvGPUSpeedup external/fkl/thread-pool"
TG=e4_preprocessing_cu
fail() { echo "# $(date -Iseconds) STOP: $*"; exit 1; }
[ -e $O ] && { echo "$O exists: not starting"; exit 1; }
[ -e $W ] && { echo "$W exists: not starting"; exit 1; }
mkdir -p $O; exec > >(tee -a $O/console.txt) 2>&1
export TMPDIR=$O/tmp; mkdir -p $TMPDIR   # the temporaries of nvcc and the host compiler stay on disk (/tmp is a RAM disk here)
cp "$0" $O/step0-script.sh
git -C $FB fetch -q origin || fail "git fetch"
MAIN=$(git -C $FB rev-parse origin/main)
echo "# $(date -Iseconds) step 0 of the E4 window G starts (compile only); main = $MAIN"
[ "$MAIN" = "$MAIN_EXPECTED" ] || fail "main is not $MAIN_EXPECTED"
echo "$MAIN" > $O/main-commit.txt
[ "$(sha256sum $WB/bin/$TG | cut -d' ' -f1)" = 8b56ffc282bd3d03f0b3014e0cfff6c4c49ca51ce6e59d010d7fbb21db19f01b ] || fail "2dac1c8's binary is not the recorded one"
[ "$(sha256sum $PR/bin/$TG | cut -d' ' -f1)" = 7bfd2c5a4844fccfb7942cd1eefe2b0051b3d51f811ba4ed5a15cdff926889e1 ] || fail "0f96c4f's binary is not the recorded one"
git -C $FB worktree add --detach $W $MAIN || fail "git worktree add"
git -c protocol.file.allow=always -c submodule.external/fkl.url=$FB/external/fkl -c submodule.external/FastNPP.url=$FB/external/FastNPP \
    -c submodule.external/cvGPUSpeedup.url=$FB/external/cvGPUSpeedup -C $W submodule update --init external/fkl external/FastNPP external/cvGPUSpeedup || fail "submodules"
git -C $W/external/fkl -c protocol.file.allow=always -c submodule.thread-pool.url=$FB/external/fkl/thread-pool submodule update --init thread-pool || fail "thread-pool"
cp $FB/CMakeUserPresets.json $W/ || fail "CMakeUserPresets.json"
for s in $SUBS; do
  a=$(git -C $W/$s rev-parse HEAD); b=$(git -C $FB/$s rev-parse HEAD)
  if [ "$s" = external/fkl/thread-pool ]; then g=$(git -C $W/external/fkl ls-tree HEAD thread-pool | awk '{print $3}'); else g=$(git -C $W ls-tree HEAD $s | awk '{print $3}'); fi
  echo "$s $a (gitlink $g, reference checkout $b)"
  [ "$a" = "$g" ] && [ "$a" = "$b" ] || fail "$s is not at the gitlink and the reference checkout's commit"
done
clean_check() {
  for r in $W $(for s in $SUBS; do echo $W/$s; done); do
    d=$(git -C $r status --porcelain --untracked-files=no); [ -z "$d" ] || fail "tracked changes in $r ($1): $d"
  done
  echo "no tracked change in the checkout or its submodules ($1)"
}
clean_check "before the build"
NEWB=$W/build/local-rtx4090
( cd $W && cmake --preset local-rtx4090 ) > $O/configure.log 2>&1 || fail "configure (configure.log)"
grep -q "^CMAKE_CACHEFILE_DIR:INTERNAL=$NEWB\$" $NEWB/CMakeCache.txt || fail "the build folder is not $NEWB"
echo "# $(date -Iseconds) build starts ($NEWB: $TG)"
taskset -c 8-15,24-31 nice -n 10 cmake --build $NEWB --target $TG -j 12 > $O/build.log 2>&1 || fail "build (build.log)"
echo "# $(date -Iseconds) build ended ($(grep -c 'warning' $O/build.log) warning lines)"
clean_check "after the build"

grep -E '^#define FKLB_' $WB/generated/fklb_build_info.h > $O/build-info-window-f.txt
grep -E '^#define FKLB_' $NEWB/generated/fklb_build_info.h > $O/build-info-window-g.txt
grep -q "FKLB_COMMIT_REPO \"$MAIN\"" $O/build-info-window-g.txt || fail "build info: the repository commit is not $MAIN"
grep -q -e '-dirty' $O/build-info-window-g.txt && fail "build info: -dirty"
diff <(grep -v FKLB_COMMIT_REPO $O/build-info-window-f.txt) <(grep -v FKLB_COMMIT_REPO $O/build-info-window-g.txt) > $O/build-info.diff \
  || fail "build info differs from 2dac1c8's beyond the repository commit (build-info.diff)"
keys='^(CMAKE_BUILD_TYPE|CMAKE_CUDA_COMPILER|CMAKE_CXX_COMPILER|CMAKE_CUDA_HOST_COMPILER|CMAKE_CUDA_FLAGS|CMAKE_CXX_FLAGS|CMAKE_CUDA_FLAGS_RELEASE|CMAKE_CXX_FLAGS_RELEASE|OpenCV_DIR|OPENCV_DIR|FKLB_[A-Z0-9_]+|E1_[A-Z_]+|E2_[A-Z_]+|E4A_[A-Z_]+|E16_[A-Z_]+)(:[A-Z]+)?='
diff <(grep -E "$keys" $WB/CMakeCache.txt | sort) <(grep -E "$keys" $NEWB/CMakeCache.txt | sort) > $O/cache.diff || fail "cache options differ from 2dac1c8's (cache.diff)"
fr=$(find $WB -path "*/$TG.dir/flags.make" | head -1); fn=$(find $NEWB -path "*/$TG.dir/flags.make" | head -1)
[ -n "$fr" ] && [ -n "$fn" ] || fail "flags.make of $TG not found"
sed -e "s#$WB#BUILD#g" -e "s#$WT#SRC#g" $fr > $O/flags-window-f.txt
sed -e "s#$NEWB#BUILD#g" -e "s#$W#SRC#g" $fn > $O/flags-window-g.txt
diff $O/flags-window-f.txt $O/flags-window-g.txt > $O/flags-$TG.diff || fail "compile flags differ from 2dac1c8's (flags-$TG.diff)"
rr=$(dirname $fr)/includes_CUDA.rsp; rn=$(dirname $fn)/includes_CUDA.rsp
sed -e "s#$WB#BUILD#g" -e "s#$WT#SRC#g" $rr > $O/includes-cuda-window-f.txt
sed -e "s#$NEWB#BUILD#g" -e "s#$W#SRC#g" $rn > $O/includes-cuda-window-g.txt
diff $O/includes-cuda-window-f.txt $O/includes-cuda-window-g.txt > $O/includes-cuda.diff || fail "CUDA include directories differ from 2dac1c8's (includes-cuda.diff)"
strings $NEWB/bin/$TG | grep -q -x $MAIN || fail "$TG does not carry $MAIN"
strings $NEWB/bin/$TG | grep -q -E '^[0-9a-f]{40}-dirty$' && fail "$TG carries a -dirty commit"
echo "build info $MAIN without -dirty, otherwise as 2dac1c8's; cache options, compile flags and CUDA include directories as 2dac1c8's; the binary carries the commit"
S=$W/scripts/sass_compare_builds.py
mkdir -p $O/sass/window-f $O/sass/pr-head $O/sass/window-g
cp $WB/bin/$TG $O/sass/window-f/; cp $PR/bin/$TG $O/sass/pr-head/; cp $NEWB/bin/$TG $O/sass/window-g/
r=$(python3 $S $O/sass/pr-head $O/sass/window-g --arch sm_89 --glob 'e4_*' --match-demangled --out $O/sass-vs-0f96c4f.txt | grep '^kernels identical'); echo "against 0f96c4f: $r"
[ "$r" = "kernels identical: 182, different: 0, only in A: 0, only in B: 0" ] || fail "SASS against 0f96c4f (sass-vs-0f96c4f.txt)"
r=$(python3 $S $O/sass/window-f $O/sass/window-g --arch sm_89 --glob 'e4_*' --match-demangled --out $O/sass-vs-2dac1c8.txt | grep '^kernels identical'); echo "against 2dac1c8: $r"
[ "$r" = "kernels identical: 152, different: 0, only in A: 0, only in B: 30" ] || fail "SASS against 2dac1c8 (sass-vs-2dac1c8.txt)"
( sha256sum $WB/bin/$TG $PR/bin/$TG $NEWB/bin/$TG ) > $O/binary-sha256.txt
cat $O/binary-sha256.txt
echo "temporaries left in $TMPDIR: $(ls -A $TMPDIR | wc -l)"; rmdir $TMPDIR 2>/dev/null
echo "# $(date -Iseconds) step 0 ended: all checks passed"
