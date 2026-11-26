#!/usr/bin/env bash
# window.sh <out dir>: the GPU window of FKL #435 (task C) on the RX 9060. Nothing is timed. The first failure ends the window.
# 1. tests: FKL's HIP tests at the #435 head, `ctest -R "_hip$" -E "^benchmark_" --stop-on-failure --timeout 300` in the head's build tree
#    (TEST_BUILD, configured from a clean checkout of AFTER_COMMIT): FKL's tests only, no benchmark target, nothing from fkl-benchmarks or
#    cvGPUSpeedup. It must exit 0 with "100% tests passed".
# 2. harness: harness/dpps_before_after.cpp built for HIP against BEFORE_COMMIT (A) and AFTER_COMMIT (B) by harness/build.sh, each run once
#    with HARNESS_GPU=1 (ParArch::CPU, CPU_MULTITHREAD and GPU_AMD) in its own folder; each must exit 0 with "outputs written".
# After the GPU steps (no GPU): harness/compare.sh A B, byte for byte (its exit code is recorded; it does not stop anything, the window is over).
# Guards, before anything runs (GUARD_ONLY=1: the guards only):
#   - TEST_MANIFEST (sha256sum of every *_hip executable of TEST_BUILD/bin, paths relative to TEST_BUILD/bin) and HARNESS_MANIFEST (sha256sum of
#     hip/A/dpps_before_after_hip and hip/B/dpps_before_after_hip, relative to its folder) check;
#   - every test ctest selects has its executable in TEST_MANIFEST, and ctest selects at least one test;
#   - TEST_BUILD's source tree (CMAKE_HOME_DIRECTORY) is at AFTER_COMMIT with no changes, and its configure log line ends in the short hash of
#     AFTER_COMMIT (not -dirty);
#   - harness A carries BEFORE_COMMIT and B AFTER_COMMIT (build.sh builds only from clean checkouts);
#   - the output folder does not exist.
# Before every step the runner is idle: no Runner.Worker and a 1-minute load average below 0.8; if it is not within IDLE_MAX seconds (default
# 1800, so the window can wait for a CI job to finish), the window stops there.
#   TEST_BUILD=<dir> TEST_MANIFEST=<file> HARNESS_MANIFEST=<file> CONFIGURE_LOG=<file> BEFORE_COMMIT=<40 hex> AFTER_COMMIT=<40 hex>
#   [IDLE_MAX=1800] [GUARD_ONLY=1] window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
TEST_BUILD=${TEST_BUILD:?}; TEST_MANIFEST=${TEST_MANIFEST:?}; HARNESS_MANIFEST=${HARNESS_MANIFEST:?}; CONFIGURE_LOG=${CONFIGURE_LOG:?}
BEFORE_COMMIT=${BEFORE_COMMIT:?}; AFTER_COMMIT=${AFTER_COMMIT:?}; IDLE_MAX=${IDLE_MAX:-1800}
HERE=$(cd "$(dirname "$0")" && pwd)
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset HARNESS_GPU
fail() { echo "$1: not starting"; exit 1; }
# the binaries
( cd "$TEST_BUILD/bin" && sha256sum -c --quiet "$TEST_MANIFEST" ) || fail "a test executable differs from $TEST_MANIFEST"
HBASE=$(cd "$(dirname "$HARNESS_MANIFEST")" && pwd)
( cd "$HBASE" && sha256sum -c --quiet "$(basename "$HARNESS_MANIFEST")" ) || fail "a harness binary differs from $HARNESS_MANIFEST"
BIN_A=$HBASE/hip/A/dpps_before_after_hip; BIN_B=$HBASE/hip/B/dpps_before_after_hip
grep -q " hip/A/dpps_before_after_hip\$" "$HARNESS_MANIFEST" && grep -q " hip/B/dpps_before_after_hip\$" "$HARNESS_MANIFEST" || fail "the harness manifest does not list A and B"
strings "$BIN_A" | grep -q -x "$BEFORE_COMMIT" || fail "harness A does not carry $BEFORE_COMMIT"
strings "$BIN_B" | grep -q -x "$AFTER_COMMIT" || fail "harness B does not carry $AFTER_COMMIT"
# the tests ctest selects, each with its executable in the manifest
TESTS=$(cd "$TEST_BUILD" && ctest -N -R "_hip\$" -E "^benchmark_" | awk '/Test +#/ {print $3}')
[ -n "$TESTS" ] || fail "ctest selects no test"
for t in $TESTS; do grep -q "  $t\$" "$TEST_MANIFEST" || fail "test $t has no executable in $TEST_MANIFEST"; done
# the test build's source tree: AFTER_COMMIT, clean, and so at configure time
SRC=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$TEST_BUILD/CMakeCache.txt")
[ "$(cd "$SRC" && git rev-parse HEAD)" = "$AFTER_COMMIT" ] || fail "$SRC is not at $AFTER_COMMIT"
[ -z "$(cd "$SRC" && git status --porcelain --untracked-files=no)" ] || fail "$SRC has changes"
grep -q -E "version: .*-${AFTER_COMMIT:0:7}\$" "$CONFIGURE_LOG" || fail "$CONFIGURE_LOG does not end its version line in ${AFTER_COMMIT:0:7}"
[ -e "$OUT" ] && fail "$OUT exists (nothing is overwritten)"
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): $(wc -l < "$TEST_MANIFEST") test executables, $(echo $TESTS | wc -w) tests selected, harness A $BEFORE_COMMIT," \
       "B $AFTER_COMMIT, test build of $SRC at $AFTER_COMMIT (clean), $OUT is new;" \
       "Runner.Worker: $(pgrep -x Runner.Worker > /dev/null && echo running || echo none); load $(cut -d' ' -f1-3 /proc/loadavg)"
  exit 0
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window.sh"; cp "$TEST_MANIFEST" "$OUT/test-manifest.sha256"; cp "$HARNESS_MANIFEST" "$OUT/harness-manifest.sha256"
idle() {   # no Runner.Worker and a load average below 0.8 (1 minute); the window stops after IDLE_MAX s of waiting
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do
    [ $waited -ge $IDLE_MAX ] && { echo "# $(date -Iseconds) not idle after ${waited} s (loadavg $(cut -d' ' -f1-3 /proc/loadavg)): the window stops here"; exit 1; }
    sleep 15; waited=$((waited + 15))
  done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
echo "# $(date -Iseconds) GPU window fkl-pr435 starts; before $BEFORE_COMMIT, after $AFTER_COMMIT; load $(cut -d' ' -f1-3 /proc/loadavg)"
idle
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  /opt/rocm/core-10.1/bin/rocminfo 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
# 1. the tests
mkdir -p "$OUT/tests"
idle
echo "# $(date -Iseconds) tests: ctest -R \"_hip\$\" -E \"^benchmark_\" --stop-on-failure --timeout 300"
( cd "$TEST_BUILD" && ctest -R "_hip\$" -E "^benchmark_" --stop-on-failure --timeout 300 --output-on-failure -j1 > "$OUT/tests/ctest.txt" 2>&1; echo $? > "$OUT/tests/exit_code.txt" )
echo "# $(date -Iseconds) tests: exit $(cat "$OUT/tests/exit_code.txt"); $(grep -E 'tests passed' "$OUT/tests/ctest.txt")"
if [ "$(cat "$OUT/tests/exit_code.txt")" != 0 ] || ! grep -q "100% tests passed" "$OUT/tests/ctest.txt"; then
  echo "# $(date -Iseconds) tests FAILED: the window stops here"; exit 1
fi
# 2. the harness, A then B
for which in A B; do
  bin=$BIN_A; [ $which = B ] && bin=$BIN_B
  dir="$OUT/harness/$which"; mkdir -p "$dir"
  idle
  echo "# $(date -Iseconds) harness $which"
  ( cd "$dir" && HARNESS_GPU=1 timeout 600 "$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) harness $which: exit $(cat "$dir/exit_code.txt"); $(grep 'outputs written' "$dir/stdout.txt")"
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q "outputs written" "$dir/stdout.txt"; then
    echo "# $(date -Iseconds) harness $which FAILED: the window stops here"; exit 1
  fi
done
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window fkl-pr435 ended"
# after the window, without the GPU: the comparison
"$HERE/harness/compare.sh" "$OUT/harness/A" "$OUT/harness/B" > "$OUT/harness/compare.txt" 2>&1
echo "# compare: exit $?; $(tail -1 "$OUT/harness/compare.txt")"
