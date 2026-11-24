#!/usr/bin/env bash
# window.sh <out dir>: the timing window of the E19 crop sweep on the RX 9060 (phase 2), written from
# results/rx9060/e19-batched-box-phase1/phase2/window.sh (the E18 RX 9060 protocol) with all its guards. It runs only in a window granted by
# the coordinator. A sweep is 16 processes, one per N, in ascending order (E19_CROPS = 1, 10, 20, ..., 150), each a complete run: the checks,
# then the 2 s clock warm-up spin (E19_CLOCK_WARMUP_MS at its default), then the timing of every row in E19's order (fkl_loop, fkl_batch, rpp;
# whole_call, then kernel_only; rpp/e19_crop_sweep.h). The sweeps:
#   1. warmup/nNNN: one complete sweep, not used.
#   2. quiet-run1..3/nNNN: three complete quiet sweeps, back to back.
#   3. rocprofv3/nNNN: after them, as its own step, one complete sweep under rocprofv3 (kernel, HIP runtime, ROCTx, memory copy and allocation
#      traces). Its times are not used: it gives the kernels of every call and the spin coverage of the kernel_only interval at every N.
# The summary is made after the window, without the GPU. The window stops at a failed step: a run that does not exit 0 with "E19 checks
# passed" ends it, and nothing after it runs.
# Guards: the binary is BIN (e19_crop_sweep_hip) with the sha256 SHA256 and carries the commit COMMIT without -dirty; the output folder is new;
# before every run (all 80) the runner is idle (no Runner.Worker, a load average below 0.8). Throughout the window the GPU's graphics and
# memory clocks (hwmon freq1_input, freq2_input), its average power (power1_average, microwatts) and its edge, junction and memory
# temperatures (temp1-3_input, millidegrees) are read in the same sampler iteration every 100 ms into gpu_clocks.csv.
# If the runner is not idle within IDLE_MAX seconds (default 600) before a run, the window stops there. GUARD_ONLY=1: the guards only (the
# binary, its commit, the new output folder, and the runner's state now); nothing is created and nothing runs.
#   BIN=<e19_crop_sweep_hip> COMMIT=<40 hex> SHA256=<sha256> [IDLE_MAX=600] [GUARD_ONLY=1] window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
BIN=${BIN:?}; COMMIT=${COMMIT:?}; SHA256=${SHA256:?}; IDLE_MAX=${IDLE_MAX:-600}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
ROCPROF=/opt/rocm/core-10.1/bin/rocprofv3
SWEEP="1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E19_CROPS E19_ONLY E19_NO_TIMING E19_PROFILE E19_SPIN_US E19_CLOCK_WARMUP_MS E19_DUMP
[ "$(basename "$BIN")" = e19_crop_sweep_hip ] || { echo "$BIN is not e19_crop_sweep_hip: not starting"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
echo "$SHA256  $BIN" | sha256sum -c --quiet || { echo "the binary differs from the record: not starting"; exit 1; }
strings "$BIN" | grep -q -x "$COMMIT" || { echo "the binary does not carry $COMMIT: not starting"; exit 1; }
strings "$BIN" | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "the binary carries a -dirty commit: not starting"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): binary $SHA256, commit $COMMIT, $OUT is new; Runner.Worker:" \
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
echo "# $(date -Iseconds) GPU window e19-crop-sweep starts; binary $SHA256 $COMMIT; hwmon ${hwmon:-none}; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  /opt/rocm/core-10.1/bin/rocminfo 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null)"; sleep 0.1; done ) \
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
       "checks: $(grep -c 'E19 checks passed' "$dir/stdout.txt")"
  # a failed step ends the window: a run that does not exit 0 with "E19 checks passed"
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q 'E19 checks passed' "$dir/stdout.txt"; then
    echo "# $(date -Iseconds) $name FAILED: the window stops here"
    exit 1
  fi
}
sweep() {   # <sweep folder> <prefix command, or ->: the 16 N in ascending order
  local name=$1 prefix=$2
  echo "# $(date -Iseconds) sweep $name starts"
  for n in $SWEEP; do run "$name/$(printf 'n%03d' $n)" "$prefix" E19_CROPS=$n; done
  echo "# $(date -Iseconds) sweep $name ended"
}
sweep warmup -
for k in 1 2 3; do sweep quiet-run$k -; done
sweep rocprofv3 "$ROCPROF --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace --output-format csv -d trace -o e19 --"
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window e19-crop-sweep ended"
