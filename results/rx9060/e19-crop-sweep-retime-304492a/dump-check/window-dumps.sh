#!/usr/bin/env bash
# window-dumps.sh <out dir> <dump dir>: the dump check of RT-A1 (the coordinator's option (b), 2026-11-26): the output tensors of
# e19_crop_sweep_hip at both pins, for e19_crop_sweep_differences.csv (fkl_batch's tensor against rpp's). It is the checks step of
# results/rx9060/e19-crop-sweep-phase1/window-checks.sh, unchanged: at each of the 16 N, in ascending order, one run with the checks only
# (E19_NO_TIMING=1: nothing is timed) and E19_DUMP=<dump dir>/<A or B>/nNNN, which writes the run's five tensors (rpp_grey.u8, rpp_box.u8,
# rpp_tensor.f32, fkl_loop_tensor.f32, fkl_batch_tensor.f32). The runs, in this order:
#   1. check-A/nNNN: A = fkl-benchmarks 976e42b with FKL b10fc66 (the binary of quiet-run*-A);
#   2. check-B/nNNN: B = fkl-benchmarks main 73a9650 with FKL 304492a (the binary of quiet-run*-B).
# The dumps (about 1.1 GB per binary) stay outside the repository; after the window, ../analysis/after_dumps.sh writes their sha256 and the
# comparisons into this folder.
# Guards, before anything is created: those of ../window.sh (the manifest: the two binaries and this script; the names and sha256; A carries
# FKL_A, B carries FKL_B and REPO_B; no -dirty commit), and the output folder and the dump folder are new. Before every run (all 32) the
# machine is idle: no Runner.Worker, a load average below 0.8 (1 minute), and an empty amdkfd process list (/sys/class/kfd/kfd/proc); if it
# is not idle within IDLE_MAX seconds (default 600) the window stops there. A failed step ends the window: a run that does not exit 0 with
# "E19 checks passed", or whose CSVs hold "# INCOMPLETE" or "-nan", or that wrote fewer than the five tensors, stops it.
# The hwmon sampler of ../window.sh runs throughout (gpu_clocks.csv). GUARD_ONLY=1: the guards only; nothing is created and nothing runs.
# ROCMINFO names rocminfo (default: ROCm 10.1's); only a mock, which must not open the GPU, sets it to a stub. launch-dumps.sh unsets it.
#   MANIFEST=<file> BIN_A=<e19_crop_sweep_hip> SHA_A=<sha256> FKL_A=<40 hex> BIN_B=<e19_crop_sweep_hip> SHA_B=<sha256> FKL_B=<40 hex>
#   REPO_B=<40 hex> [IDLE_MAX=600] [GUARD_ONLY=1] window-dumps.sh <out dir> <dump dir>
set -u
OUT=${1:?usage: window-dumps.sh <out dir> <dump dir>}; DUMPS=${2:?usage: window-dumps.sh <out dir> <dump dir>}
MANIFEST=${MANIFEST:?}; BIN_A=${BIN_A:?}; SHA_A=${SHA_A:?}; FKL_A=${FKL_A:?}; BIN_B=${BIN_B:?}; SHA_B=${SHA_B:?}; FKL_B=${FKL_B:?}
REPO_B=${REPO_B:?}; IDLE_MAX=${IDLE_MAX:-600}
ROCMINFO=${ROCMINFO:-/opt/rocm/core-10.1/bin/rocminfo}
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
SWEEP="1 10 20 30 40 50 60 70 80 90 100 110 120 130 140 150"
KFD=/sys/class/kfd/kfd/proc
export TMPDIR=$NS/sys LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E19_CROPS E19_ONLY E19_NO_TIMING E19_PROFILE E19_SPIN_US E19_CLOCK_WARMUP_MS E19_DUMP
( cd "$(dirname "$MANIFEST")" && sha256sum -c --quiet "$(basename "$MANIFEST")" ) || { echo "the manifest does not match: not starting"; exit 1; }
guard() {   # <binary> <sha256> <FKL commit> [<fkl-benchmarks commit>]
  [ "$(basename "$1")" = e19_crop_sweep_hip ] || { echo "$1 is not e19_crop_sweep_hip: not starting"; exit 1; }
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
[ -e "$DUMPS" ] && { echo "$DUMPS exists: not starting (nothing is overwritten)"; exit 1; }
[ -d "$KFD" ] || { echo "$KFD is missing: the GPU-process guard cannot run: not starting"; exit 1; }
if [ -n "${GUARD_ONLY:-}" ]; then
  echo "guards ok at $(date -Iseconds): manifest, A $SHA_A FKL $FKL_A, B $SHA_B FKL $FKL_B repo $REPO_B, $OUT and $DUMPS are new; Runner.Worker:" \
       "$(pgrep -x Runner.Worker > /dev/null && echo running || echo none); load $(cut -d' ' -f1-3 /proc/loadavg);" \
       "GPU processes: $(ls "$KFD" | wc -l)"
  exit 0
fi
mkdir -p "$OUT" "$DUMPS"
exec > >(tee -a "$OUT/window-console.txt") 2>&1
cp "$0" "$OUT/window-dumps.sh"
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
echo "# $(date -Iseconds) GPU window e19-crop-sweep-retime-304492a dump check starts; A $SHA_A FKL $FKL_A; B $SHA_B FKL $FKL_B repo $REPO_B; hwmon ${hwmon:-none}; rocminfo $ROCMINFO; load $(cut -d' ' -f1-3 /proc/loadavg)"
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150
  echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  echo "GPU processes (amdkfd): $(ls "$KFD" | wc -l)"
  $ROCMINFO 2> /dev/null | grep -E 'Marketing Name|Name: +gfx' | head -4; } > "$OUT/environment-before.txt" 2>&1
if [ -n "$hwmon" ]; then
  ( echo "time_ns,sclk_hz,mclk_hz,power_uw,temp_edge_mc,temp_junction_mc,temp_mem_mc"; while true; do echo "$(date +%s%N),$(cat $hwmon/freq1_input),$(cat $hwmon/freq2_input 2> /dev/null),$(cat $hwmon/power1_average 2> /dev/null),$(cat $hwmon/temp1_input 2> /dev/null),$(cat $hwmon/temp2_input 2> /dev/null),$(cat $hwmon/temp3_input 2> /dev/null)"; sleep 0.1; done ) \
    > "$OUT/gpu_clocks.csv" 2> /dev/null &
  SAMPLER_PID=$!
fi
run() {   # <run folder> <A or B> <N>: one check run of BIN_A or BIN_B at N crops, dumping its tensors into $DUMPS/<A or B>/nNNN
  local name=$1 which=$2 n=$3 bin
  if [ "$which" = A ]; then bin=$BIN_A; else bin=$BIN_B; fi
  local dir="$OUT/$name"; mkdir -p "$dir"
  DUMPDIR="$DUMPS/$which/$(printf 'n%03d' $n)"; mkdir -p "$DUMPDIR"
  idle
  local start=$(date +%s)
  echo "# $(date -Iseconds) $name ($which) E19_CROPS=$n E19_NO_TIMING=1 E19_DUMP=$DUMPDIR"
  ( cd "$dir" && env E19_CROPS=$n E19_NO_TIMING=1 E19_DUMP="$DUMPDIR" timeout 300 "$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt )
  local incomplete=$(grep -l '^# INCOMPLETE' "$dir"/*.csv 2> /dev/null | wc -l) nan=$(grep -l -- '-nan' "$dir"/*.csv 2> /dev/null | wc -l)
  local dumped=$(ls "$DUMPDIR" 2> /dev/null | grep -c -E '^(rpp_grey\.u8|rpp_box\.u8|rpp_tensor\.f32|fkl_loop_tensor\.f32|fkl_batch_tensor\.f32)$')
  echo "# $(date -Iseconds) $name exit $(cat "$dir/exit_code.txt") after $(( $(date +%s) - start )) s; csv $(ls "$dir" | grep -c '\.csv$'); INCOMPLETE in:" \
       "$incomplete; -nan in: $nan; checks: $(grep -c 'E19 checks passed' "$dir/stdout.txt"); tensors dumped: $dumped"
  # a failed step ends the window
  if [ "$(cat "$dir/exit_code.txt")" != 0 ] || ! grep -q 'E19 checks passed' "$dir/stdout.txt" || [ "$incomplete" != 0 ] || [ "$nan" != 0 ] ||
     [ "$dumped" != 5 ]; then
    echo "# $(date -Iseconds) $name FAILED: the window stops here"
    exit 1
  fi
}
for x in A B; do
  echo "# $(date -Iseconds) checks of $x start"
  for n in $SWEEP; do run "check-$x/$(printf 'n%03d' $n)" $x $n; done
  echo "# $(date -Iseconds) checks of $x ended"
done
[ -n "$SAMPLER_PID" ] && { kill $SAMPLER_PID 2> /dev/null; wait $SAMPLER_PID 2> /dev/null; }
{ date -Iseconds; cat /proc/loadavg; ps -eo pid,pcpu,psr,comm,args --sort=-pcpu | head -16 | cut -c1-150; } > "$OUT/environment-after.txt" 2>&1
echo "# $(date -Iseconds) GPU window e19-crop-sweep-retime-304492a dump check ended"
