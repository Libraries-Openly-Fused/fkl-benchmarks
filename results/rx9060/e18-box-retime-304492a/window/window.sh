#!/usr/bin/env bash
# window.sh <out dir>: RT-A2, the E18 box re-time on the RX 9060 after the pin move to FKL 304492a (the merge of FKL #435; Oscar's order of
# 2026-11-26, relayed by the coordinator session). It times e18_box_hip at both pins in one window, alternately:
#   A = e18_box_hip of fkl-benchmarks 976e42b with FKL b10fc66 (the base build of results/rx9060/pin-move-devcode-b10fc66-48fbf3f/, #67);
#   B = e18_box_hip of fkl-benchmarks main 73a9650 with FKL 304492a, built from a clean worktree
#       (results/rx9060/e19-crop-sweep-retime-304492a/step0/).
# A's code is the code of B in results/rx9060/e18-box-retime-b10fc66/ (#62; device code 6 of 6 functions and host code 128 of 128 identical),
# so A against that record's B repeats it, and B against A shows the move.
# The protocol is #62's, the record's own (results/rx9060/e18-box/window.sh), unchanged: every run is complete, one process per run: the checks,
# then the 2 s clock warm-up spin (E18_CLOCK_WARMUP_MS at its default), then the timing of every row in the program's order (rpp, rpp_ltrb,
# fkl_box_halfpixel, fkl_half_once; whole_call, then kernel_only), with sclk before and after every interval. The runs, in this order:
#   1. unused-A, unused-B: not used;
#   2. quiet-run1-A, quiet-run1-B, quiet-run2-A, quiet-run2-B, quiet-run3-A, quiet-run3-B;
#   3. rocprofv3-A, rocprofv3-B: one complete run of each under rocprofv3, as the record's (kernel, HIP runtime, ROCTx, memory copy and
#      allocation traces, the same command); their times are not used, only the kernels and their durations.
# The summary is made after the window, without the GPU (retime_report.py).
# Guards, before anything is created:
#   - every file of MANIFEST (sha256sum format: the two binaries and this script) has its sha256;
#   - BIN_A and BIN_B are e18_box_hip with the sha256 SHA_A and SHA_B; A carries the FKL commit FKL_A (its archive build records
#     fkl-benchmarks as "unknown"), B carries the FKL commit FKL_B and the fkl-benchmarks commit REPO_B; neither carries a -dirty commit;
#   - the output folder is new.
# Before every run the machine is idle: no Runner.Worker, a load average below 0.8 (1 minute), and no process holding the GPU (the amdkfd
# process list, /sys/class/kfd/kfd/proc, is empty: no other ROCm program, FastRPP's included, is running); if it is not idle within IDLE_MAX
# seconds (default 600) the window stops there. A failed step ends the window: a run that does not exit 0 with "E18 checks passed", or whose CSVs
# hold "# INCOMPLETE" or "-nan", stops it and nothing after it runs.
# Throughout the window the GPU's graphics and memory clocks (hwmon freq1_input, freq2_input), its average power (power1_average,
# microwatts) and its edge, junction and memory temperatures (temp1-3_input, millidegrees) are read every 100 ms into gpu_clocks.csv.
# GUARD_ONLY=1: the guards only (and the runner's state now); nothing is created and nothing runs.
# ROCMINFO and ROCPROF name rocminfo and rocprofv3 (default: ROCm 10.1's); only a mock of this script, which must not open the GPU, sets
# them to stubs. launch.sh unsets both, and the first line of the console names the two that ran.
#   MANIFEST=<file> BIN_A=<e18_box_hip> SHA_A=<sha256> FKL_A=<40 hex> BIN_B=<e18_box_hip> SHA_B=<sha256> FKL_B=<40 hex> REPO_B=<40 hex>
#   [IDLE_MAX=600] [GUARD_ONLY=1] window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
MANIFEST=${MANIFEST:?}; BIN_A=${BIN_A:?}; SHA_A=${SHA_A:?}; FKL_A=${FKL_A:?}; BIN_B=${BIN_B:?}; SHA_B=${SHA_B:?}; FKL_B=${FKL_B:?}
REPO_B=${REPO_B:?}; IDLE_MAX=${IDLE_MAX:-600}
ROCMINFO=${ROCMINFO:-/opt/rocm/core-10.1/bin/rocminfo}; ROCPROF=${ROCPROF:-/opt/rocm/core-10.1/bin/rocprofv3}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
KFD=/sys/class/kfd/kfd/proc
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E18_ONLY E18_NO_TIMING E18_PROFILE E18_SPIN_US E18_CLOCK_WARMUP_MS
( cd "$(dirname "$MANIFEST")" && sha256sum -c --quiet "$(basename "$MANIFEST")" ) || { echo "the manifest does not match: not starting"; exit 1; }
guard() {   # <binary> <sha256> <FKL commit> [<fkl-benchmarks commit>]
  [ "$(basename "$1")" = e18_box_hip ] || { echo "$1 is not e18_box_hip: not starting"; exit 1; }
  echo "$2  $1" | sha256sum -c --quiet || { echo "$1 differs from the record: not starting"; exit 1; }
  strings "$1" | grep -q -x "$3" || { echo "$1 does not carry FKL $3: not starting"; exit 1; }
  [ -z "${4:-}" ] || strings "$1" | grep -q -x "$4" || { echo "$1 does not carry fkl-benchmarks $4: not starting"; exit 1; }
  strings "$1" | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$1 carries a -dirty commit: not starting"; exit 1; }
  return 0
}
guard "$BIN_A" "$SHA_A" "$FKL_A"
guard "$BIN_B" "$SHA_B" "$FKL_B" "$REPO_B"
[ "$SHA_A" != "$SHA_B" ] || { echo "A and B are the same binary: not starting"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -d "$KFD" ] || { echo "$KFD is missing: the GPU-process guard cannot run: not starting"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): manifest, A $SHA_A FKL $FKL_A, B $SHA_B FKL $FKL_B repo $REPO_B, $OUT is new; Runner.Worker:" \
       "$(pgrep -x Runner.Worker > /dev/null && echo running || echo none); load $(cut -d' ' -f1-3 /proc/loadavg);" \
       "GPU processes: $(ls "$KFD" | wc -l)"
  exit 0
fi
mkdir -p "$OUT"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window.sh"
cp "$MANIFEST" "$OUT/manifest.sha256"
SAMPLER_PID=""
trap '[ -n "$SAMPLER_PID" ] && kill $SAMPLER_PID 2> /dev/null' EXIT
busy() {   # a reason the machine is not idle, or nothing
  pgrep -x Runner.Worker > /dev/null && { echo "Runner.Worker"; return; }
  awk '{exit !($1 >= 0.8)}' /proc/loadavg && { echo "load $(cut -d' ' -f1 /proc/loadavg)"; return; }
  [ -n "$(ls "$KFD")" ] && { echo "GPU processes $(ls "$KFD" | tr '\n' ' ')"; return; }
  return 0
}
idle() {   # no Runner.Worker, a load average below 0.8 (1 minute), no process holding the GPU; the window stops after IDLE_MAX s of waiting
  local waited=0 why
  while why=$(busy); [ -n "$why" ]; do
    [ $waited -ge $IDLE_MAX ] && { echo "# $(date -Iseconds) not idle after ${waited} s ($why; loadavg $(cut -d' ' -f1-3 /proc/loadavg)): the window stops here"; exit 1; }
    sleep 15; waited=$((waited + 15))
  done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
}
hwmon=""
for card in /sys/class/drm/card*/device; do
  [ "$(cat $card/vendor 2> /dev/null)" = 0x1002 ] || continue
  for h in $card/hwmon/hwmon*; do [ -e $h/freq1_input ] && { hwmon=$h; break 2; }; done
done
echo "# $(date -Iseconds) GPU window e18-box-retime-304492a starts; A $SHA_A FKL $FKL_A; B $SHA_B FKL $FKL_B repo $REPO_B; hwmon ${hwmon:-none}; rocminfo $ROCMINFO; rocprofv3 $ROCPROF; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  echo "GPU processes (amdkfd): $(ls "$KFD" | wc -l)"
  $ROCMINFO 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null)"; sleep 0.1; done ) \
    > "$OUT/gpu_clocks.csv" 2> /dev/null &
  SAMPLER_PID=$!
fi
run() {   # <run folder> <A or B> <prefix command, or ->: one complete run of BIN_A or BIN_B, no environment variable
  local name=$1 which=$2 prefix=$3 bin
  [ "$prefix" = "-" ] && prefix=""
  if [ "$which" = A ]; then bin=$BIN_A; else bin=$BIN_B; fi
  local dir="$OUT/$name"; mkdir -p "$dir"
  idle
  local start=$(date +%s)
  echo "# $(date -Iseconds) $name ($which: $bin) $prefix"
  ( cd "$dir" && timeout 300 $prefix "$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  local incomplete=$(grep -l '^# INCOMPLETE' "$dir"/*.csv 2> /dev/null | wc -l) nan=$(grep -l -- '-nan' "$dir"/*.csv 2> /dev/null | wc -l)
  echo "# $(date -Iseconds) $name exit $(cat "$dir/exit_code.txt") after $(( $(date +%s) - start )) s; csv $(ls "$dir" | grep -c '\.csv$'); INCOMPLETE in:" \
       "$incomplete; -nan in: $nan; checks: $(grep -c 'E18 checks passed' "$dir/stdout.txt")"
  # a failed step ends the window
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q 'E18 checks passed' "$dir/stdout.txt" || [ "$incomplete" != 0 ] || [ "$nan" != 0 ]; then
    echo "# $(date -Iseconds) $name FAILED: the window stops here"
    exit 1
  fi
}
TRACE="$ROCPROF --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace --output-format csv -d trace -o e18box --"
run unused-A A -
run unused-B B -
for k in 1 2 3; do run quiet-run$k-A A -; run quiet-run$k-B B -; done
run rocprofv3-A A "$TRACE"
run rocprofv3-B B "$TRACE"
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window e18-box-retime-304492a ended"
