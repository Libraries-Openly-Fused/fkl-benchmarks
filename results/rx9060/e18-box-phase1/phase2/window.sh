#!/usr/bin/env bash
# window.sh <out dir>: the GPU window of E18 box on the RX 9060 (phase 2). Written in phase 1 and NOT RUN: it runs only in a window scheduled with
# the coordinator. The protocol is the RTX 4090's (results/rtx4090/e18-stencil/window-script.sh):
#   1. WARMUP=1: one complete run (the checks and the timing), in warmup/, not used.
#   2. Three complete quiet runs, back to back: quiet-run1..3. Every run checks first, then times every variant in the same order (rpp,
#      fkl_box_halfpixel, fkl_half_once; whole_call, then kernel_only; rpp/e18_box.h).
#   3. After the three timed runs, as its own step: one complete run under rocprofv3 (kernel, HIP runtime, ROCTx, memory copy and allocation
#      traces), in rocprofv3/. Its times are not used: rocprof_calls.py --spin gives the kernels of every call and whether the spin covers the
#      host's enqueue of every call of the kernel-only interval.
#   4. The summary: e18_box_report.py (the median of the three runs' means with their min-max, and the speedups).
# Guards: the binary is BIN with the sha256 SHA256 and carries the commit COMMIT without -dirty; the output folder is new; before every run the
# runner is idle (no Runner.Worker, a load average below 0.8). Throughout the window the GPU's graphics and memory clocks (hwmon freq1_input,
# freq2_input) are sampled every 100 ms into gpu_clocks.csv.
#   BIN=<e18_box_hip> COMMIT=<40 hex> SHA256=<sha256> WARMUP=0|1 window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
BIN=${BIN:?}; COMMIT=${COMMIT:?}; SHA256=${SHA256:?}; WARMUP=${WARMUP:?set WARMUP=0 or 1}
HERE=$(cd "$(dirname "$0")" && pwd)
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
PY="$NS/scripts/safepy.sh"
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E18_ONLY E18_NO_TIMING E18_PROFILE E18_SPIN_US
[ "$(basename "$BIN")" = e18_box_hip ] || { echo "$BIN is not e18_box_hip: not starting"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
echo "$SHA256  $BIN" | sha256sum -c --quiet || { echo "the binary differs from the record: not starting"; exit 1; }
strings "$BIN" | grep -q -x "$COMMIT" || { echo "the binary does not carry $COMMIT: not starting"; exit 1; }
strings "$BIN" | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "the binary carries a -dirty commit: not starting"; exit 1; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window.sh"
idle() {   # the runner is idle: no Runner.Worker and a load average below 0.8 (1 minute)
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do sleep 15; waited=$((waited + 15)); done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
hwmon=""
for card in /sys/class/drm/card*/device; do
  [ "$(cat $card/vendor 2> /dev/null)" = 0x1002 ] || continue
  for h in $card/hwmon/hwmon*; do [ -e $h/freq1_input ] && { hwmon=$h; break 2; }; done
done
echo "# $(date -Iseconds) GPU window e18-box starts (WARMUP=$WARMUP); binary $SHA256 $COMMIT; hwmon ${hwmon:-none}; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  /opt/rocm/core-10.1/bin/rocminfo 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
SAMPLER_PID=""
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null)"; sleep 0.1; done ) \
    > "$OUT/gpu_clocks.csv" 2> /dev/null &
  SAMPLER_PID=$!
fi
run() {   # <run folder> <prefix command, or -> [VAR=value ...]
  local name=$1 prefix=$2; shift 2
  [ "$prefix" = "-" ] && prefix=""
  local dir="$OUT/$name"; mkdir -p "$dir"
  idle
  local start=$(date +%s)
  echo "# $(date -Iseconds) $name $prefix $*"
  ( cd "$dir" && env "$@" timeout 300 $prefix "$BIN" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  echo "# $(date -Iseconds) $name exit $(cat "$dir/exit_code.txt") after $(( $(date +%s) - start )) s; csv $(ls "$dir" | grep -c '\.csv$'); INCOMPLETE in:" \
       "$(grep -l '^# INCOMPLETE' "$dir"/*.csv 2> /dev/null | wc -l); -nan in: $(grep -l -- '-nan' "$dir"/*.csv 2> /dev/null | wc -l);" \
       "checks: $(grep -c 'E18 checks passed' "$dir/stdout.txt")"
}
[ "$WARMUP" = 1 ] && run warmup -
for n in 1 2 3; do run quiet-run$n -; done
run rocprofv3 "$ROCPROF --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace --output-format csv -d trace -o e18box --"
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
$PY "$HERE/e18_box_report.py" "$OUT"
$PY "$HERE/../rocprofv3/rocprof_calls.py" "$OUT/rocprofv3/trace" e18box --spin --out "$OUT/rocprofv3/calls.txt" --tsv "$OUT/rocprofv3/calls.tsv" | tail -20
echo "# $(date -Iseconds) GPU window e18-box ended"
