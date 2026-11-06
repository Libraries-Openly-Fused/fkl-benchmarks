#!/bin/bash
# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Compile check of an FKL tree against everything fkl-benchmarks builds and against the tests of the two downstream libraries, at their pinned commits. COMPILE ONLY, no GPU is used, no pin of this
# repository is changed and nothing is written under the repository: the FKL tree under test is symlinked into scratch copies.
#
#   scripts/check_fkl_tree.sh FKL_TREE [--out DIR] [--jobs N] [--cores LIST] [--only bench,cvgs,fastnpp] [--ref-build DIR]
#
#   FKL_TREE     a checkout of FusedKernelLibrary (it must contain include/fused_kernel and lib/export); e.g. a `git worktree` of the PR head, or external/fkl for the baseline
#   --out        scratch folder of this run (default /home/oscar/tfkl-bench/scratch/fkl-check-<FKL sha>-<time>); must not exist or be empty; TMPDIR of the compilers is OUT/tmp (never /tmp, a RAM disk)
#   --jobs       parallel compile jobs (default 12; the peak is a few GiB per nvcc, the machine has 121 GiB)
#   --cores      cores the compilers are pinned to (default 8-15,24-31: the cores that are not the measurement cores 0-3,16-19)
#   --only       groups to run, comma separated (default bench,cvgs,fastnpp; fkltests is opt-in):
#                  bench    fkl-benchmarks (this repository at HEAD, its pinned cvGPUSpeedup and FastNPP headers, external/fkl = FKL_TREE): every `*_cu` target (CUDA) and every `*_cpp` target (CPU)
#                  cvgs     the 15 test targets of external/cvGPUSpeedup at its pin, built with ITS OWN CMake against FKL_TREE (its fkl/ replaced by a link; OpenCV 4.14 with CUDA)
#                  fastnpp  the 2 test targets of external/FastNPP at its pin, built with ITS OWN CMake against FKL_TREE
#                  fkltests the test matrix of FKL itself (tests/ and utests/), configured from FKL_TREE out of source; compile only (running the CUDA ones needs the GPU). The targets are EVERY target of the
#                           configured tree that compiles something (executable or library: the CMake file API types EXECUTABLE, SHARED_LIBRARY, STATIC_LIBRARY, MODULE_LIBRARY, OBJECT_LIBRARY; at c8fcbe8 these are
#                           the same 125 targets that the name filter `_(cu|cpp)$` found, for 46 CUDA and 43 CPU ctest tests), whatever their names. A target that has a ctest test
#                           `<target>_rejected` (a compile-fail test: it builds the target and passes only if the build FAILS, with the message of the check and with no other error) is an EXPECTED REJECTION: it is not
#                           built by the group; its result is that of the ctest test (`ctest -R '_rejected$' -j1`, run after the build): REJECTED_OK when it passed, REJECTED_FAIL when it did not.
#   --ref-build  the configured fkl-benchmarks build whose FKLB_* options, OPENCV_DIR and nvcc are reused (default build/local-rtx4090)
#
# Output (in OUT): <group>-configure.log, <group>-build.log, <group>-targets.txt (all targets of the group), <group>-rejected-targets.txt and <group>-rejected.log (the expected rejections and the ctest output of their tests,
# when the group has any), <group>-report.csv (group, target, status OK / FAIL / NOT_BUILT / REJECTED_OK / REJECTED_FAIL, first error line or the message of the rejection test), report.txt (the same, readable, with the
# wall time of every group and the largest memory use of one compiler process) and the exit status: 0 when every target of every selected group compiled and every expected rejection held, 1 otherwise.
# scripts/check_fkl_tree_compare.py compares the reports of two runs (REJECTED_OK counts as OK there).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
FKL=""; OUT=""; JOBS=12; CORES="8-15,24-31"; ONLY="bench,cvgs,fastnpp"; REF="$ROOT/build/local-rtx4090"
while [ $# -gt 0 ]; do
  case $1 in
    --out) OUT=$2; shift 2;; --jobs) JOBS=$2; shift 2;; --cores) CORES=$2; shift 2;; --only) ONLY=$2; shift 2;; --ref-build) REF=$2; shift 2;;
    -h|--help) sed -n 15,39p "$0"; exit 0;;
    -*) echo "unknown option $1"; exit 2;;
    *) FKL=$1; shift;;
  esac
done
[ -n "$FKL" ] || { echo "usage: $0 FKL_TREE [--out DIR] [--jobs N] [--cores LIST] [--only bench,cvgs,fastnpp] [--ref-build DIR]"; exit 2; }
FKL=$(cd "$FKL" && pwd) || exit 2
[ -d "$FKL/include/fused_kernel" ] && [ -f "$FKL/lib/export/FKLConfig.cmake" ] || { echo "$FKL is not an FKL tree (include/fused_kernel or lib/export/FKLConfig.cmake missing)"; exit 2; }
[ -f "$REF/CMakeCache.txt" ] || { echo "reference build $REF has no CMakeCache.txt (use --ref-build)"; exit 2; }
FKL_SHA=$(git -C "$FKL" rev-parse HEAD 2>/dev/null || echo "not-a-git-tree")
FKL_ORIG=$FKL
[ -n "$OUT" ] || OUT=/home/oscar/tfkl-bench/scratch/fkl-check-${FKL_SHA:0:7}-$(date +%H%M%S)
if [ -n "$(ls -A "$OUT" 2>/dev/null)" ]; then echo "$OUT is not empty: not starting"; exit 2; fi
mkdir -p "$OUT/tmp" || exit 2
export TMPDIR=$OUT/tmp
# A fresh clone or worktree of FKL has no thread-pool submodule: the CPU_MULTITHREAD back-end (and the CPU benchmarks that use it) would silently be left out, and the result would not be comparable with the baseline.
# Then the pinned thread-pool of this repository's external/fkl is linked in, through a link overlay of the tree (the tree itself is not touched).
if [ ! -f "$FKL/thread-pool/include/ThreadPool.h" ] && [ -f "$ROOT/external/fkl/thread-pool/include/ThreadPool.h" ]; then
  mkdir -p "$OUT/fkl-overlay" || exit 2
  for e in "$FKL"/* "$FKL"/.[!.]*; do b=$(basename "$e"); case $b in thread-pool) continue;; esac; [ -e "$e" ] && ln -s "$e" "$OUT/fkl-overlay/$b"; done
  ln -s "$ROOT/external/fkl/thread-pool" "$OUT/fkl-overlay/thread-pool"
  FKL=$OUT/fkl-overlay
  OVERLAY_NOTE="thread-pool of the tree under test is missing: the pinned thread-pool of external/fkl ($(git -C "$ROOT/external/fkl/thread-pool" rev-parse --short HEAD)) is linked in (overlay $FKL)"
fi

# the CUDA 13.4 toolchain: the one of this machine when no FKLB_CUDA_HOME is set
if [ -z "${FKLB_CUDA_HOME:-}" ]; then
  if [ -f /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh ]; then source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh; else source "$ROOT/scripts/env-cuda13.4.sh"; fi
fi
command -v nvcc > /dev/null || { echo "nvcc not found"; exit 2; }
nvcc --version | grep -q 'release 13\.4' || { echo "nvcc is not CUDA 13.4"; exit 2; }

cache() { grep -E "^$1:" "$REF/CMakeCache.txt" | head -1 | sed -E 's/^[^=]*=//'; }
OPENCV=$(cache OPENCV_DIR)
# every FKLB_* option of the reference build, as -D flags
FKLB_ARGS=()
while IFS= read -r l; do n=${l%%:*}; v=${l#*=}; FKLB_ARGS+=("-D$n=$v"); done < <(grep -E '^FKLB_[A-Z0-9_]+:(BOOL|STRING)=' "$REF/CMakeCache.txt")
COMMON=(-G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release "-DCMAKE_CUDA_COMPILER=$FKLB_CUDA_HOME/bin/nvcc")

REPORT=$OUT/report.txt
say() { echo "$@" | tee -a "$REPORT"; }
say "check_fkl_tree: FKL tree $FKL_ORIG ($FKL_SHA), $(date -Iseconds), jobs $JOBS, cores $CORES, groups $ONLY"
[ -n "${OVERLAY_NOTE:-}" ] && say "note: $OVERLAY_NOTE"
say "nvcc: $(nvcc --version | grep release | sed 's/^ *//')  c++: $(c++ --version | head -1)  OpenCV: $OPENCV"
say "fkl-benchmarks HEAD $(git -C "$ROOT" rev-parse --short HEAD); cvGPUSpeedup $(git -C "$ROOT/external/cvGPUSpeedup" rev-parse --short HEAD); FastNPP $(git -C "$ROOT/external/FastNPP" rev-parse --short HEAD); the repository itself is not changed"
STATUS=0

# a clone of a pinned library with its fkl/ replaced by a link to the tree under test (its CMake needs a git repository: the version suffix is read from git)
clone_lib() {  # name -> $OUT/<name>-src
  local lib=$ROOT/external/$1 dst=$OUT/$2-src sha
  sha=$(git -C "$lib" rev-parse HEAD) || return 1
  git clone -q --local --no-checkout "$lib" "$dst" && git -C "$dst" checkout -q --detach "$sha" || return 1
  rmdir "$dst/fkl" && ln -s "$FKL" "$dst/fkl"
}

run_group() {  # group src_dir extra_cmake_args... ; the targets are chosen by $TARGET_FILTER: a regular expression on the target names of `make help`, or the word "targets" for every target of the configured tree that compiles something (executable or library)
  local g=$1 src=$2; shift 2
  local b=$OUT/$g-build
  say ""; say "== $g: configure ($src)"
  # the query that makes CMake write its code model (the file API) at the configure step
  if [ "$TARGET_FILTER" = targets ]; then mkdir -p "$b/.cmake/api/v1/query" && touch "$b/.cmake/api/v1/query/codemodel-v2"; fi
  if ! taskset -c "$CORES" cmake -S "$src" -B "$b" "${COMMON[@]}" "$@" > "$OUT/$g-configure.log" 2>&1; then
    say "CONFIGURE FAILED: $(grep -m1 -A3 'CMake Error' "$OUT/$g-configure.log" | tr '\n' ' ' | cut -c1-400)"; STATUS=1; return
  fi
  if [ "$TARGET_FILTER" = targets ]; then
    python3 "$ROOT/scripts/check_fkl_tree_report.py" targets "$b" > "$OUT/$g-targets.txt" || { say "$g: the code model of CMake could not be read"; STATUS=1; return; }
  else
    make -C "$b" help | sed -n 's/^\.\.\. //p' | grep -E "$TARGET_FILTER" > "$OUT/$g-targets.txt"
  fi
  # The targets that must NOT compile: those with a ctest test <target>_rejected. They are not built here; the test builds them (see below).
  ctest --test-dir "$b" -N 2> /dev/null | sed -n 's/^ *Test *#[0-9]*: \(.*\)_rejected$/\1/p' > "$OUT/$g-rejected-targets.txt"
  grep -vxFf "$OUT/$g-rejected-targets.txt" "$OUT/$g-targets.txt" > "$OUT/$g-targets-built.txt"
  say "$g: $(wc -l < "$OUT/$g-targets.txt") targets: $(tr '\n' ' ' < "$OUT/$g-targets.txt" | cut -c1-600)"
  [ -s "$OUT/$g-rejected-targets.txt" ] && say "$g: $(wc -l < "$OUT/$g-rejected-targets.txt") of them are expected rejections (a ctest test <target>_rejected builds the target and passes only if the build fails with the message of the check and no other error): not built by make, judged by that test"
  local t0=$(date +%s)
  # shellcheck disable=SC2046
  /usr/bin/time -f "%e s wall, largest resident set of one process %M KiB" -o "$OUT/$g-time.txt" \
    taskset -c "$CORES" nice -n 10 make -C "$b" -k -j"$JOBS" --output-sync=target $(tr '\n' ' ' < "$OUT/$g-targets-built.txt") > "$OUT/$g-build.log" 2>&1
  say "$g: build finished after $(( $(date +%s) - t0 )) s ($(cat "$OUT/$g-time.txt" | tail -1))"
  local rejected_log=()
  if [ -s "$OUT/$g-rejected-targets.txt" ]; then
    local t1=$(date +%s)
    taskset -c "$CORES" nice -n 10 ctest --test-dir "$b" -V -R '_rejected$' -j1 --timeout 900 > "$OUT/$g-rejected.log" 2>&1
    say "$g: the rejection tests finished after $(( $(date +%s) - t1 )) s ($(grep -E 'tests passed|tests failed' "$OUT/$g-rejected.log" | tail -1))"
    rejected_log=("$OUT/$g-rejected.log")
  fi
  python3 "$ROOT/scripts/check_fkl_tree_report.py" report "$g" "$OUT/$g-build.log" "$OUT/$g-targets.txt" "$OUT/$g-report.csv" "${rejected_log[@]}" | tee -a "$REPORT"
  [ "${PIPESTATUS[0]}" -eq 0 ] || STATUS=1
}

IFS=',' read -ra GROUPS_TO_RUN <<< "$ONLY"
for g in "${GROUPS_TO_RUN[@]}"; do
  case $g in
    bench)
      # a farm of links to the sources of this repository; external/fkl is the tree under test, the two libraries are the pinned checkouts
      mkdir -p "$OUT/bench-src/external"
      for e in cmake common fkl-self handwritten-cuda npp opencv-cpu opencv-cuda pytorch-attention rpp torch-inductor triton CMakeLists.txt CMakePresets.json LICENSE NOTICE; do
        [ -e "$ROOT/$e" ] && ln -s "$ROOT/$e" "$OUT/bench-src/$e"
      done
      ln -s "$FKL" "$OUT/bench-src/external/fkl"; ln -s "$ROOT/external/cvGPUSpeedup" "$OUT/bench-src/external/cvGPUSpeedup"; ln -s "$ROOT/external/FastNPP" "$OUT/bench-src/external/FastNPP"
      TARGET_FILTER='_(cu|cpp)$' run_group bench "$OUT/bench-src" "${FKLB_ARGS[@]}" "-DOPENCV_DIR=$OPENCV";;
    cvgs)
      clone_lib cvGPUSpeedup cvgs || { say "cvgs: clone failed"; STATUS=1; continue; }
      TARGET_FILTER='^test.+' run_group cvgs "$OUT/cvgs-src" "-DOPENCV_DIR=$OPENCV" -DCMAKE_CUDA_ARCHITECTURES=89;;
    fastnpp)
      clone_lib FastNPP fastnpp || { say "fastnpp: clone failed"; STATUS=1; continue; }
      TARGET_FILTER='^fastNPP_' run_group fastnpp "$OUT/fastnpp-src" -DCMAKE_CUDA_ARCHITECTURES=89;;
    fkltests)
      # FKL's own CMake, out of source (nothing is written into FKL_TREE): every target of the tree that compiles something (the test matrix and the compile-fail targets)
      TARGET_FILTER=targets run_group fkltests "$FKL" -DCMAKE_CUDA_ARCHITECTURES=89;;
    *) say "unknown group $g";;
  esac
done
say ""; say "check_fkl_tree: done $(date -Iseconds), exit status $STATUS (0 = every target of every group compiled and every expected rejection held)"
exit $STATUS
