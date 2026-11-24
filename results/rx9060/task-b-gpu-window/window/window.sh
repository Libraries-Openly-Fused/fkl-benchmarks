#!/usr/bin/env bash
# window.sh <out dir>: the task B GPU window on the RX 9060, written from results/rx9060/e19-repeat-check/window.sh with all its guards.
# Nothing in it is timed except step 4. Every step stops the window at its first failure.
# 1. tests-after: every filter test of FKL's tests/image_processing at the PR head (AFTER_COMMIT), the 6 existing and the 3 new ones (HIP
#    builds, which run the CPU, CPU_MULTITHREAD and GPU cases). Each must exit 0 with a last line ending in PASS.
# 2. tests-before: the 6 existing filter tests built at 24478f0 (BEFORE_COMMIT), the same rule. With step 1 this shows the uchar Quads'
#    exact checks (== to the host reference) passing before and after.
# 3. checks-before / checks-after: the checks only of fkl-benchmarks' E18 box (E18_NO_TIMING=1, output digests), E19 (E19_NO_TIMING=1,
#    E19_DUMP) and the E19 crop sweep at N = 30 (E19_CROPS=30, E19_NO_TIMING=1, E19_DUMP), built against 24478f0 and against the PR head
#    (their LinearFilterDPP details ported, nothing else). Each must exit 0 with its "checks passed" line.
# 4. The harness (harness/filters_before_after.hip), A = built against 24478f0, B = against the PR head: every run first checks and dumps the
#    outputs of the existing float filter instantiations (filters_outputs.bin), then times the four median cases (E19's protocol: 2 s clock
#    warm-up, then whole_call and kernel_only behind a 5 ms spin, 100 calls after one warm-up call). The runs, in this order:
#      unused-A, unused-B: not used; then quiet-run1-A, quiet-run1-B, quiet-run2-A, quiet-run2-B, quiet-run3-A, quiet-run3-B.
#    Each must exit 0 with "filters_before_after checks passed".
# The comparisons and the summary are made after the window, without the GPU (analysis/).
# Guards, before anything runs: MANIFEST (a sha256sum file of every binary the window runs, with paths relative to it) checks; the harness
#   binaries carry BEFORE_COMMIT and AFTER_COMMIT; the output folder is new. Before every run the runner is idle (no Runner.Worker, a load
#   average below 0.8). Throughout the window the GPU's graphics and memory clocks, its average power and its edge, junction and memory
#   temperatures are read in the same sampler iteration every 100 ms into gpu_clocks.csv.
# If the runner is not idle within IDLE_MAX seconds (default 600) before a run, the window stops there. GUARD_ONLY=1: the guards only.
#   MANIFEST=<sha256sum file> BEFORE_COMMIT=<40 hex> AFTER_COMMIT=<40 hex> [IDLE_MAX=600] [GUARD_ONLY=1] window.sh <out dir>
# The manifest's paths: tests-after/test_*_hip, tests-before/test_*_hip, fklb-before/{e18_box,e19_batched_box,e19_crop_sweep}_hip,
#   fklb-after/..., harness/A/filters_before_after_hip, harness/B/filters_before_after_hip.
set -u
OUT=${1:?usage: window.sh <out dir>}
MANIFEST=${MANIFEST:?}; BEFORE_COMMIT=${BEFORE_COMMIT:?}; AFTER_COMMIT=${AFTER_COMMIT:?}; IDLE_MAX=${IDLE_MAX:-600}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset FILTERS_NO_TIMING FILTERS_SPIN_US FILTERS_CLOCK_WARMUP_MS E18_ONLY E18_PROFILE E18_SPIN_US E18_CLOCK_WARMUP_MS E19_ONLY E19_PROFILE \
      E19_SPIN_US E19_CLOCK_WARMUP_MS E19_DUMP E19_CROPS E19_NO_TIMING E18_NO_TIMING
BASE=$(cd "$(dirname "$MANIFEST")" && pwd)
( cd "$BASE" && sha256sum -c --quiet "$(basename "$MANIFEST")" ) || { echo "a binary differs from $MANIFEST: not starting"; exit 1; }
listed() { awk '{print $2}' "$MANIFEST" | grep -E "$1" | sed "s|^|$BASE/|"; }
TESTS_AFTER=$(listed '^tests-after/test_[a-z0-9_]+_hip$'); TESTS_BEFORE=$(listed '^tests-before/test_[a-z0-9_]+_hip$')
[ "$(echo $TESTS_AFTER | wc -w)" = 9 ] && [ "$(echo $TESTS_BEFORE | wc -w)" = 6 ] || { echo "the manifest does not list 9 tests after and 6 before: not starting"; exit 1; }
for side in before after; do
  for p in e18_box_hip e19_batched_box_hip e19_crop_sweep_hip; do
    [ -x "$BASE/fklb-$side/$p" ] && grep -q " fklb-$side/$p\$" "$MANIFEST" || { echo "fklb-$side/$p is not in the manifest: not starting"; exit 1; }
  done
done
for which in A B; do
  grep -q " harness/$which/filters_before_after_hip\$" "$MANIFEST" || { echo "harness/$which is not in the manifest: not starting"; exit 1; }
done
BIN_A=$BASE/harness/A/filters_before_after_hip; BIN_B=$BASE/harness/B/filters_before_after_hip
strings "$BIN_A" | grep -q -x "$BEFORE_COMMIT" || { echo "A does not carry $BEFORE_COMMIT: not starting"; exit 1; }
strings "$BIN_B" | grep -q -x "$AFTER_COMMIT" || { echo "B does not carry $AFTER_COMMIT: not starting"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): $(wc -l < "$MANIFEST") binaries, before $BEFORE_COMMIT, after $AFTER_COMMIT, $OUT is new;" \
       "Runner.Worker: $(pgrep -x Runner.Worker > /dev/null && echo running || echo none); load $(cut -d' ' -f1-3 /proc/loadavg)"
  exit 0
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window.sh"
cp "$MANIFEST" "$OUT/manifest.sha256"
SAMPLER_PID=""
trap '[ -n "$SAMPLER_PID" ] && kill $SAMPLER_PID 2> /dev/null' EXIT
idle() {   # the runner is idle: no Runner.Worker and a load average below 0.8 (1 minute); the window stops after IDLE_MAX s of waiting
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do
    [ $waited -ge $IDLE_MAX ] && { echo "# $(date -Iseconds) not idle after ${waited} s (loadavg $(cut -d' ' -f1-3 /proc/loadavg)): the window stops here"; exit 1; }
    sleep 15; waited=$((waited + 15))
  done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
hwmon=""
for card in /sys/class/drm/card*/device; do
  [ "$(cat $card/vendor 2> /dev/null)" = 0x1002 ] || continue
  for h in $card/hwmon/hwmon*; do [ -e $h/freq1_input ] && { hwmon=$h; break 2; }; done
done
echo "# $(date -Iseconds) GPU window task-b starts; before $BEFORE_COMMIT, after $AFTER_COMMIT; hwmon ${hwmon:-none}; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  /opt/rocm/core-10.1/bin/rocminfo 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null)"; sleep 0.1; done ) \
    > "$OUT/gpu_clocks.csv" 2> /dev/null &
  SAMPLER_PID=$!
fi
# <folder> <name> <pass pattern> <env...> -- <binary>: one run in <folder>/<name>, which must exit 0 with a line matching the pattern
step() {
  local folder=$1 name=$2 pattern=$3; shift 3
  local vars=(); while [ "$1" != "--" ]; do vars+=("$1"); shift; done; shift
  local bin=$1 dir="$OUT/$folder/$name"
  mkdir -p "$dir"
  idle
  local start=$(date +%s)
  echo "# $(date -Iseconds) $folder/$name ${vars[*]:-}"
  ( cd "$dir" && env "${vars[@]}" timeout 600 "$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  local code=$(cat "$dir/exit_code.txt")
  echo "# $(date -Iseconds) $folder/$name exit $code after $(( $(date +%s) - start )) s; FAIL lines $(grep -c 'FAIL' "$dir/stdout.txt");" \
       "matching lines $(grep -c -E "$pattern" "$dir/stdout.txt")"
  if [ "$code" != 0 ] || ! grep -q -E "$pattern" "$dir/stdout.txt"; then
    echo "# $(date -Iseconds) $folder/$name FAILED: the window stops here"
    exit 1
  fi
}
# 1, 2. the tests: the last line must end in PASS (checked with tail after the run, as the pattern sees every line)
for side in after before; do
  if [ $side = after ]; then list=$TESTS_AFTER; else list=$TESTS_BEFORE; fi
  for t in $list; do
    step tests-$side "$(basename "$t")" 'PASS$' DUMMY=1 -- "$t"
    tail -1 "$OUT/tests-$side/$(basename "$t")/stdout.txt" | grep -q 'PASS$' || { echo "# $(date -Iseconds) $(basename "$t"): the last line is not a PASS: the window stops here"; exit 1; }
  done
done
# 3. the fkl-benchmarks checks, before and after
for side in before after; do
  step checks-$side e18_box 'E18 checks passed' E18_NO_TIMING=1 -- "$BASE/fklb-$side/e18_box_hip"
  mkdir -p "$OUT/checks-$side/e19_batched_box/dumps" "$OUT/checks-$side/e19_crop_sweep_n030/dumps"
  step checks-$side e19_batched_box 'E19 checks passed' E19_NO_TIMING=1 "E19_DUMP=$OUT/checks-$side/e19_batched_box/dumps" -- "$BASE/fklb-$side/e19_batched_box_hip"
  step checks-$side e19_crop_sweep_n030 'E19 checks passed' E19_CROPS=30 E19_NO_TIMING=1 "E19_DUMP=$OUT/checks-$side/e19_crop_sweep_n030/dumps" -- "$BASE/fklb-$side/e19_crop_sweep_hip"
done
# 4. the harness: the outputs, then the median speed
step harness unused-A 'filters_before_after checks passed' DUMMY=1 -- "$BIN_A"
step harness unused-B 'filters_before_after checks passed' DUMMY=1 -- "$BIN_B"
for k in 1 2 3; do
  step harness quiet-run$k-A 'filters_before_after checks passed' DUMMY=1 -- "$BIN_A"
  step harness quiet-run$k-B 'filters_before_after checks passed' DUMMY=1 -- "$BIN_B"
done
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window task-b ended"
