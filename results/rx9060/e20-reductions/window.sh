#!/usr/bin/env bash
# window.sh <out dir>: the timed GPU window of E20 reductions on the RX 9060 (task E20-RPP, Oscar's "Yes both" of 2026-11-27 in the
# coordinator session): rpp/e20_reductions.h (e20_reductions_hip), the rows fkl and rpp, at the 16 N of the crop sweep. It runs only when
# the coordinator has granted it, as one detached call.
# A run is one complete process of the program, as in the NVIDIA program: for every N in ascending order (1, 10, 20, ..., 150; E20_CROPS
# unset), the checks on the probe and on the frame that is timed, then the timing of every row in the program's order (fkl, rpp;
# whole_call, then kernel_only), with the 2 s clock warm-up spin before the first interval and sclk before and after every interval. The
# runs, in this order, each in its own folder:
#   1. unused: not used (the first run after an idle GPU);
#   2. quiet-run1, quiet-run2, quiet-run3: the three runs whose times are used;
#   3. rocprofv3: one run under rocprofv3 (kernel, HIP runtime, ROCTx, memory copy, memory set and allocation traces): its times are not
#      used, only the GPU operations of every call and the spin coverage of the kernel_only interval.
# The summary is made after the window, without the GPU.
# Guards, before anything is created:
#   - every file of MANIFEST (sha256sum format: the binary and this script) has its sha256;
#   - BIN is e20_reductions_hip with the sha256 SHA, carries the FKL commit FKL and the fkl-benchmarks commit REPO (its build info), and no
#     -dirty commit;
#   - the output folder is new.
# Before every run the machine is idle: no Runner.Worker (no CI job on this runner), a load average below 0.8 (1 minute), and no process
# holding the GPU (the amdkfd process list, /sys/class/kfd/kfd/proc, is empty: no other ROCm program, FastRPP's included, is running). If it
# is not idle within IDLE_MAX seconds (default 600) the window stops there. After every run the same is checked again, and the sampler
# below records Runner.Worker and the GPU processes every 100 ms: a run during which a CI job or another GPU program ran, or that does not
# exit 0 with "E20 reductions (AMD) checks passed", or whose CSVs hold "# INCOMPLETE" or "-nan", stops the window, and nothing after it
# runs (the first failure ends it).
# Throughout the window the GPU's graphics and memory clocks (hwmon freq1_input, freq2_input), its average power (power1_average,
# microwatts), its edge, junction and memory temperatures (temp1-3_input, millidegrees), whether a Runner.Worker runs and the number of GPU
# processes are read every 100 ms into gpu_clocks.csv.
# GUARD_ONLY=1: the guards only (and the machine's state now); nothing is created and nothing runs.
# ROCMINFO and ROCPROF name rocminfo and rocprofv3 (default: ROCm 10.1's); only a mock of this script, which must not open the GPU, sets
# them to stubs. launch.sh unsets both.
#   MANIFEST=<file> BIN=<e20_reductions_hip> SHA=<sha256> FKL=<40 hex> REPO=<40 hex> [IDLE_MAX=600] [GUARD_ONLY=1] window.sh <out dir>
set -u
OUT=${1:?usage: window.sh <out dir>}
MANIFEST=${MANIFEST:?}; BIN=${BIN:?}; SHA=${SHA:?}; FKL=${FKL:?}; REPO=${REPO:?}; IDLE_MAX=${IDLE_MAX:-600}
ROCMINFO=${ROCMINFO:-/opt/rocm/core-10.1/bin/rocminfo}; ROCPROF=${ROCPROF:-/opt/rocm/core-10.1/bin/rocprofv3}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
KFD=/sys/class/kfd/kfd/proc
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E20_CROPS E20_ONLY E20_NO_TIMING E20_PROFILE E20_PROFILE_CALLS E20_SPIN_US E20_CLOCK_WARMUP_MS
( cd "$(dirname "$MANIFEST")" && sha256sum -c --quiet "$(basename "$MANIFEST")" ) || { echo "the manifest does not match: not starting"; exit 1; }
[ "$(basename "$BIN")" = e20_reductions_hip ] || { echo "$BIN is not e20_reductions_hip: not starting"; exit 1; }
echo "$SHA  $BIN" | sha256sum -c --quiet || { echo "$BIN does not have the sha256 $SHA: not starting"; exit 1; }
strings "$BIN" | grep -q -x "$FKL" || { echo "$BIN does not carry FKL $FKL: not starting"; exit 1; }
strings "$BIN" | grep -q -x "$REPO" || { echo "$BIN does not carry fkl-benchmarks $REPO: not starting"; exit 1; }
strings "$BIN" | grep -q -E '^[0-9a-f]{40}-dirty$' && { echo "$BIN carries a -dirty commit: not starting"; exit 1; }
[ -e "$OUT" ] && { echo "$OUT exists: not starting (nothing is overwritten)"; exit 1; }
[ -d "$KFD" ] || { echo "$KFD is missing: the GPU-process guard cannot run: not starting"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): manifest, $BIN $SHA, FKL $FKL, fkl-benchmarks $REPO, $OUT is new; Runner.Worker:" \
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
echo "# $(date -Iseconds) GPU window e20-reductions starts; $BIN $SHA; FKL $FKL; fkl-benchmarks $REPO; hwmon ${hwmon:-none}; rocminfo $ROCMINFO; rocprofv3 $ROCPROF; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  echo "GPU processes (amdkfd): $(ls "$KFD" | wc -l)"
  $ROCMINFO 2> /dev/null | grep -E 'Marketing Name|Name: +gfx|Compute Unit' | head -6; } > "$OUT/environment-before.txt" 2>&1
( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc,runner_worker,gpu_processes"
  while true; do
    r=0; pgrep -x Runner.Worker > /dev/null && r=1
    if [ -n "$hwmon" ]; then
      echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null),$r,$(ls "$KFD" | wc -l)"
    else
      echo "$(date +%s%N),,,,,,,$r,$(ls "$KFD" | wc -l)"
    fi
    sleep 0.1
  done ) > "$OUT/gpu_clocks.csv" 2> /dev/null &
SAMPLER_PID=$!
run() {   # <run folder> <prefix command, or ->: one complete run of BIN
  local name=$1 prefix=$2
  [ "$prefix" = "-" ] && prefix=""
  local dir="$OUT/$name"; mkdir -p "$dir"
  idle
  local start=$(date +%s%N) runners0 gpus0
  echo "# $(date -Iseconds) $name $prefix"
  ( cd "$dir" && timeout 600 $prefix "$BIN" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  local end=$(date +%s%N)
  local incomplete=$(grep -l '^# INCOMPLETE' "$dir"/*.csv 2> /dev/null | wc -l) nan=$(grep -l -- '-nan' "$dir"/*.csv 2> /dev/null | wc -l)
  # what the sampler saw during the run: a CI job, or a GPU process other than the run's own (the run holds one; rocprofv3's run as well)
  local runners=$(awk -F, -v a=$start -v b=$end 'NR > 1 && $1 >= a && $1 <= b && $8 == 1' "$OUT/gpu_clocks.csv" | wc -l)
  local others=$(awk -F, -v a=$start -v b=$end 'NR > 1 && $1 >= a && $1 <= b && $9 > 1' "$OUT/gpu_clocks.csv" | wc -l)
  local after=$(busy)
  echo "# $(date -Iseconds) $name exit $(cat "$dir/exit_code.txt") after $(( (end - start) / 1000000000 )) s; csv $(ls "$dir" | grep -c '\.csv$');" \
       "INCOMPLETE in: $incomplete; -nan in: $nan; checks: $(grep -c 'E20 reductions (AMD) checks passed' "$dir/stdout.txt");" \
       "samples with a Runner.Worker: $runners, with another GPU process: $others; after the run: ${after:-idle}"
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q 'E20 reductions (AMD) checks passed' "$dir/stdout.txt" || [ "$incomplete" != 0 ] ||
     [ "$nan" != 0 ] || [ "$runners" != 0 ] || [ "$others" != 0 ] || [ -n "$(echo "$after" | grep -v '^load')" ]; then
    echo "# $(date -Iseconds) $name FAILED: the window stops here"
    exit 1
  fi
}
TRACE="$ROCPROF --kernel-trace --memory-copy-trace --memory-allocation-trace --hip-runtime-trace --marker-trace --output-format csv -d trace -o e20 --"
run unused -
for k in 1 2 3; do run quiet-run$k -; done
run rocprofv3 "$TRACE"
kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window e20-reductions ended"
