#!/bin/bash
# Step 1 for FKL PR #427 (issue #19, the crop fix): compile only, no GPU program, no measurement, no quiet window (the coordinator, 2026-11-06; it is told
# when this starts and ends, so that no measurement is scheduled on this machine meanwhile).
#  a. a local clone of the pinned external/fkl with refs/pull/427/head fetched, checked out detached at HEAD (refused if the PR head is not HEAD);
#  b. scripts/check_fkl_tree.sh <clone> --out $CHECK --only bench,cvgs,fastnpp,fkltests (expected rejections judged by their ctest tests);
#  c. cvGPUSpeedup and FastNPP configured a second time from the check's own clones (their fkl/ is the tree under test) with ENABLE_BENCHMARK=ON, as
#     for #416, every target built (the suites of the GPU window);
#  d. FKL's CPU tests, ctest -R '_cpp$' -j8, on the fkltests build;
#  e. the test lists of the three suites of the GPU window (ctest -N);
#  f. comparison with the run at the PR #423 head (the base of #427: 7b2deb6 has the include/ and tests/ of 9e20bd4e) and with the c8fcbe8 baseline;
#  g. SASS of the 35 CUDA programs of the bench group against the 9e20bd4e build.
set -u
HEAD=${HEAD:?set HEAD to the full SHA of the final PR #427 head}
T=/home/oscar/tfkl-bench; FB=$T/fkl-benchmarks; W=${W:-$T/scratch/pr427}; CLONE=$W/fkl-head; CHECK=$W/check
P423=$T/scratch/pr423; BASE=7b2deb6660921106e524a12cd5fca376feb0d544; CORES=8-15,24-31; JOBS=12
mkdir -p $W
exec > >(tee -a $W/step1-console.txt) 2>&1
say() { echo "# $(date -Iseconds) $*"; }
say "step 1 for FKL PR #427, head $HEAD, starts; load $(cut -d' ' -f1-3 /proc/loadavg)"
# a. the clone
if [ ! -d $CLONE ]; then git clone -q --local --no-checkout $FB/external/fkl $CLONE || { say "clone failed"; exit 1; }; fi
git -C $CLONE fetch -q https://github.com/Libraries-Openly-Fused/FusedKernelLibrary.git refs/pull/427/head || { say "fetch failed"; exit 1; }
[ "$(git -C $CLONE rev-parse FETCH_HEAD)" = "$HEAD" ] || { say "refs/pull/427/head is $(git -C $CLONE rev-parse FETCH_HEAD), not $HEAD: stop"; exit 1; }
git -C $CLONE checkout -q --detach $HEAD || { say "checkout failed"; exit 1; }
echo "clone at $(git -C $CLONE rev-parse HEAD), local changes $(git -C $CLONE status --short --untracked-files=no | wc -l); merge-base with $BASE: $(git -C $CLONE merge-base $HEAD $BASE)"
git -C $CLONE log --format='%h %an %cI %s' $BASE..$HEAD
git -C $CLONE diff --stat $BASE $HEAD
# b. the compile check (it refuses a non-empty --out)
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
$FB/scripts/check_fkl_tree.sh $CLONE --out $CHECK --only bench,cvgs,fastnpp,fkltests; say "check_fkl_tree.sh exit $?"
# c. cvGPUSpeedup and FastNPP with ENABLE_BENCHMARK=ON, from the same clones, all targets
OPENCV=$(sed -n 's/^OPENCV_DIR:[A-Z]*=//p' $FB/build/local-rtx4090/CMakeCache.txt)
for lib in cvgs fastnpp; do
  extra=(-DCMAKE_CUDA_ARCHITECTURES=89 -DENABLE_BENCHMARK=ON); [ $lib = cvgs ] && extra+=("-DOPENCV_DIR=$OPENCV")
  b=$CHECK/$lib-bench-build
  taskset -c $CORES cmake -S $CHECK/$lib-src -B $b -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc" "${extra[@]}" > $CHECK/$lib-bench-configure.log 2>&1
  say "$lib with ENABLE_BENCHMARK=ON: configure exit $?"
  /usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o $CHECK/$lib-bench-time.txt \
    taskset -c $CORES nice -n 10 make -C $b -k -j$JOBS --output-sync=target > $CHECK/$lib-bench-build.log 2>&1
  say "$lib with ENABLE_BENCHMARK=ON: build exit $? ($(tail -1 $CHECK/$lib-bench-time.txt)); errors: $(grep -c -E 'error:|Error [0-9]' $CHECK/$lib-bench-build.log)"
done
# d. FKL's CPU tests
taskset -c $CORES ctest --test-dir $CHECK/fkltests-build -R '_cpp$' -j8 --output-on-failure > $CHECK/ctest-cpu.log 2>&1; say "CPU ctest exit $?"
grep -E 'tests passed|Total Test time' $CHECK/ctest-cpu.log
# e. the test lists of the window
ctest --test-dir $CHECK/fkltests-build -N -R '_cu$' > $CHECK/window-list-fkl-cuda.txt 2>&1
ctest --test-dir $CHECK/cvgs-bench-build -N > $CHECK/window-list-cvGPUSpeedup.txt 2>&1
ctest --test-dir $CHECK/fastnpp-bench-build -N > $CHECK/window-list-FastNPP.txt 2>&1
for f in fkl-cuda cvGPUSpeedup FastNPP; do echo "window suite $f: $(tail -1 $CHECK/window-list-$f.txt)"; done
ls -la $CHECK/bench-build/bin/benchmark_CPU_NPP_vs_FastNPP_cu $CHECK/bench-build/bin/benchmark_npp_batchresize_x_split3D_cu
# f. comparisons (later folders override earlier ones)
python3 $FB/scripts/check_fkl_tree_compare.py $P423/check-fast,$P423/check-bench,$P423/check-fkltests-rerun $CHECK > $CHECK/compare-with-pr423-head.txt 2>&1; say "compare with the #423 head run: exit $? (1 = a regression)"
python3 $FB/scripts/check_fkl_tree_compare.py $T/scratch/fkl-check-baseline-c8fcbe8,$T/scratch/fkl-check-baseline-fkltests-c8fcbe8 $CHECK > $CHECK/compare-with-c8fcbe8-baseline.txt 2>&1; say "compare with the c8fcbe8 baseline: exit $?"
grep -E '^==|REGRESSION' $CHECK/compare-with-pr423-head.txt | head -12
# g. SASS of the bench group's CUDA programs against the 9e20bd4e build
python3 $FB/scripts/sass_compare_builds.py $P423/check-bench/bench-build/bin $CHECK/bench-build/bin --arch sm_89 --glob '*_cu' --out $CHECK/sass-bench-pr427-vs-9e20bd4e.txt > /dev/null 2>&1; say "SASS comparison exit $?"
sed -n 1,5p $CHECK/sass-bench-pr427-vs-9e20bd4e.txt
say "step 1 ended; load $(cut -d' ' -f1-3 /proc/loadavg)"
