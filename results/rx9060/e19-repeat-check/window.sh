#!/usr/bin/env bash
# window.sh <out dir>: the E19 RX 9060 repeat check, written from results/rx9060/e19-crop-sweep-phase1/phase2/window.sh with all its guards.
# It asks whether the change of rpp's kernel_only at N = 30 between E19's window (results/rx9060/e19-batched-box/) and the crop sweep's
# (results/rx9060/e19-crop-sweep/) follows the program or the window, by timing both programs alternately in one window:
#   A = E19's e19_batched_box_hip (the binary E19 timed), with no environment variable;
#   B = the crop sweep's e19_crop_sweep_hip with E19_CROPS=30.
# Every run is complete: the checks, then the 2 s clock warm-up spin (E19_CLOCK_WARMUP_MS at its default), then the timing of every row in
# E19's order (fkl_loop, fkl_batch, rpp; whole_call, then kernel_only). The runs, in this order:
#   1. unused-A, unused-B: not used;
#   2. quiet-run1-A, quiet-run1-B, quiet-run2-A, quiet-run2-B, quiet-run3-A, quiet-run3-B;
#   3. rocprofv3-A, rocprofv3-B: under rocprofv3 (kernel, HIP runtime, ROCTx, memory copy and allocation traces), their times not used.
# The summary is made after the window, without the GPU. The window stops at a failed step: a run that does not exit 0 with "E19 checks passed"
# ends it, and nothing after it runs.
# Guards: BIN_A is e19_batched_box_hip with the sha256 SHA_A and carries COMMIT_A; BIN_B is e19_crop_sweep_hip with the sha256 SHA_B and
# carries COMMIT_B; neither carries a -dirty commit; the output folder is new; before every run the runner is idle (no Runner.Worker, a load
# average below 0.8). Throughout the window the GPU's graphics and memory clocks (hwmon freq1_input, freq2_input), its average power
# (power1_average, microwatts) and its edge, junction and memory temperatures (temp1-3_input, millidegrees) are read in the same sampler
# iteration every 100 ms into gpu_clocks.csv.
# If the runner is not idle within IDLE_MAX seconds (default 600) before a run, the window stops there. GUARD_ONLY=1: the guards only (the
# binaries, their commits, the new output folder, and the runner's state now); nothing is created and nothing runs.
#   BIN_A=<e19_batched_box_hip> SHA_A=<sha256> COMMIT_A=<40 hex> BIN_B=<e19_crop_sweep_hip> SHA_B=<sha256> COMMIT_B=<40 hex> [IDLE_MAX=600]
#   [GUARD_ONLY=1] window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
BIN_A=${BIN_A:?}; SHA_A=${SHA_A:?}; COMMIT_A=${COMMIT_A:?}; BIN_B=${BIN_B:?}; SHA_B=${SHA_B:?}; COMMIT_B=${COMMIT_B:?}; IDLE_MAX=${IDLE_MAX:-600}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E19_CROPS E19_ONLY E19_NO_TIMING E19_PROFILE E19_SPIN_US E19_CLOCK_WARMUP_MS E19_DUMP
guard() {   # <binary> <expected name> <sha256> <commit>
  [ "$(basename "$1")" = "$2" ] || { echo "$1 is not $2: not starting"; exit 1; }
  echo "$3  $1" | sha256sum -c --quiet || { echo "$1 differs from the record: not starting"; exit 1; }
  strings "$1" | grep -q -x "$4" || { echo "$1 does not carry $4: not starting"; exit 1; }
  strings "$1" | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$1 carries a -dirty commit: not starting"; exit 1; }
}
guard "$BIN_A" e19_batched_box_hip "$SHA_A" "$COMMIT_A"
guard "$BIN_B" e19_crop_sweep_hip "$SHA_B" "$COMMIT_B"
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): A $SHA_A $COMMIT_A, B $SHA_B $COMMIT_B, $OUT is new; Runner.Worker:" \
       "$(pgrep -x Runner.Worker > /dev/null && echo running || echo none); load $(cut -d' ' -f1-3 /proc/loadavg)"
  exit 0
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window.sh"
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
echo "# $(date -Iseconds) GPU window e19-repeat-check starts; A $SHA_A $COMMIT_A; B $SHA_B $COMMIT_B; hwmon ${hwmon:-none}; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  /opt/rocm/core-10.1/bin/rocminfo 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null)"; sleep 0.1; done ) \
    > "$OUT/gpu_clocks.csv" 2> /dev/null &
  SAMPLER_PID=$!
fi
run() {   # <run folder> <A or B> <prefix command, or ->: A runs BIN_A with no variable, B runs BIN_B with E19_CROPS=30
  local name=$1 which=$2 prefix=$3 bin vars
  [ "$prefix" = "-" ] && prefix=""
  if [ "$which" = A ]; then bin=$BIN_A; vars=""; else bin=$BIN_B; vars="E19_CROPS=30"; fi
  local dir="$OUT/$name"; mkdir -p "$dir"
  idle
  local start=$(date +%s)
  echo "# $(date -Iseconds) $name ($which: $(basename "$bin") $vars) $prefix"
  ( cd "$dir" && env $vars timeout 300 $prefix "$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) $name exit $(cat "$dir/exit_code.txt") after $(( $(date +%s) - start )) s; csv $(ls "$dir" | grep -c '\.csv$'); INCOMPLETE in:" \
       "$(grep -l '^# INCOMPLETE' "$dir"/*.csv 2> /dev/null | wc -l); -nan in: $(grep -l -- '-nan' "$dir"/*.csv 2> /dev/null | wc -l);" \
       "checks: $(grep -c 'E19 checks passed' "$dir/stdout.txt")"
  # a failed step ends the window: a run that does not exit 0 with "E19 checks passed"
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q 'E19 checks passed' "$dir/stdout.txt"; then
    echo "# $(date -Iseconds) $name FAILED: the window stops here"
    exit 1
  fi
}
TRACE="$ROCPROF --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace --output-format csv -d trace -o e19 --"
run unused-A A -
run unused-B B -
for k in 1 2 3; do run quiet-run$k-A A -; run quiet-run$k-B B -; done
run rocprofv3-A A "$TRACE"
run rocprofv3-B B "$TRACE"
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window e19-repeat-check ended"
