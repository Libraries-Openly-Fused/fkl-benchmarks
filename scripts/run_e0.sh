#!/usr/bin/env bash
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
# E0: re-run every existing experiment of the paper on the RTX 4090, one at a time, from a build of this repository.
#
#   scripts/run_e0.sh --list
#   scripts/run_e0.sh [--only NAME]...          (default: all, in the order of --list; short ones first)
#
# Environment: source scripts/env-cuda13.4.sh with CUDA_VISIBLE_DEVICES set to the RTX 4090; BUILD=<build dir> (default
# build/local-rtx4090). Each experiment runs in results/rtx4090/<name>/ (stdout.txt, stderr.txt, exit_code.txt, the CSVs the
# benchmark writes in its working directory). Refuses to run when
#   - the tracked files have uncommitted changes (the CSV header would say -dirty; set FKLB_ALLOW_DIRTY=1 to override),
#   - another process uses the GPU (the measurement would be contended; the desktop session's own processes are ignored),
#   - results/rtx4090/<name>/ already holds files (move them away; nothing is overwritten).
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BUILD=${BUILD:-$ROOT/build/local-rtx4090}
OUT=$ROOT/results/rtx4090

# name | binary (in $BUILD/bin) | what it is
EXPERIMENTS=(
  "e0-latency-hiding|benchmark_latency_hiding_cu|Fig. 1 latency hiding (60 points, 1..1181 Muls)"
  "e0-numinstructions|benchmark_numinstructions_x_operation_cu|instructions per operation (100 points, 500 in total)"
  "e0-wrapper-overhead|benchmark_CPUandGPU_cvGS_vs_fk_cu|cvGS wrapper overhead (bit-exact check, GPU and CPU enqueue time)"
  "e0-cpu-time-opencv|benchmark_CPU_OpenCV_vs_cvGS_cu|CPU execution time, OpenCV-CUDA vs cvGS"
  "e0-cpu-time-npp|benchmark_CPU_NPP_vs_FastNPP_cu|CPU execution time, NPP vs FastNPP"
  "e0-npp|benchmark_npp_batchresize_x_split3D_cu|comparison with NPP (both variants, fixed ROI)"
  "e0-data-type|benchmark_data_type_cu|data type (26 type combinations, batch 50)"
  "e0-data-size|benchmark_image_resolution_MAD_loop_cu|data size (60 points, 6 type combinations; hours)"
  "e0-hf|benchmark_horizontal_fusion_cu|horizontal fusion + CUDA Graphs (120 points, batch 1..1191)"
  "e0-graph-instantiate|graph_instantiate_time_cu|CUDA Graphs capture + instantiate time of the OpenCV baseline"
  "e0-vf-mul-add1|benchmark_vertical_fusionCompareMulAdd1_cu|VF mul_add1 (batch 1) + CUDA Graphs (200 points to 19,902 ops)"
  "e0-vf-mul1|benchmark_vertical_fusionCompareMul1_cu|VF mul1 (batch 1) + CUDA Graphs"
  "e0-vf-mul-add|benchmark_vertical_fusionCompareMulAdd_cu|VF + HF mul_add (batch 50) + CUDA Graphs (hours)"
)

if [ "${1:-}" = "--list" ]; then
  printf '%-22s %-52s %s\n' name binary description
  for e in "${EXPERIMENTS[@]}"; do IFS='|' read -r n b d <<<"$e"; printf '%-22s %-52s %s\n' "$n" "$b" "$d"; done
  exit 0
fi

ONLY=()
while [ $# -gt 0 ]; do
  case $1 in --only) ONLY+=("$2"); shift 2;; *) echo "unknown argument $1"; exit 2;; esac
done

command -v nvcc >/dev/null || { echo "nvcc not found: source scripts/env-cuda13.4.sh"; exit 1; }
nvcc --version | grep -q 'release 13\.4' || { echo "nvcc is not CUDA 13.4"; exit 1; }
[ "${CUDA_DEVICE_ORDER:-}" = PCI_BUS_ID ] && [ -n "${CUDA_VISIBLE_DEVICES:-}" ] || { echo "set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES (source scripts/env-cuda13.4.sh)"; exit 1; }
GPU_NAME=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name --format=csv,noheader)
case $GPU_NAME in *4090*) ;; *) echo "GPU $CUDA_VISIBLE_DEVICES is '$GPU_NAME', not the RTX 4090"; exit 1;; esac
SRC_PATHS="common cmake fkl-self opencv-cuda npp external CMakeLists.txt CMakePresets.json"   # what the binaries are built from
if [ -n "$(git -C "$ROOT" status --porcelain --untracked-files=no -- $SRC_PATHS)" ] && [ "${FKLB_ALLOW_DIRTY:-0}" != 1 ]; then
  echo "uncommitted changes in the sources of the binaries ($SRC_PATHS): commit them, rebuild, and run again (or FKLB_ALLOW_DIRTY=1)"; exit 1
fi
# The binaries must have been built from a clean commit, and nothing that affects them may have changed since (scripts/, results/
# and the docs may). The build info is rewritten by every build; BUILD can be a frozen copy of bin/ and generated/.
BUILT=$(grep -o 'FKLB_COMMIT_REPO "[^"]*"' "$BUILD/generated/fklb_build_info.h" | cut -d'"' -f2)
case $BUILT in *-dirty|unknown) [ "${FKLB_ALLOW_DIRTY:-0}" = 1 ] || { echo "the build is of an uncommitted tree ($BUILT): commit, rebuild, run again"; exit 1; };; esac
if [ "${FKLB_ALLOW_DIRTY:-0}" != 1 ] && ! git -C "$ROOT" diff --quiet "${BUILT%-dirty}" HEAD -- $SRC_PATHS; then
  echo "sources changed since the build ($BUILT): rebuild first"; exit 1
fi
echo "binaries: $BUILD/bin, built from fkl-benchmarks $BUILT"

# GPU processes of the desktop session (the 4090 drives the display) do not count; anything else does
DESKTOP='/usr/bin/(ptyxis|nautilus|gnome-shell|Xwayland|Xorg|gnome-terminal|firefox)'
# True when a non-desktop process uses the GPU, or when the utilization does not settle at <= 10 % in three consecutive
# one-second samples within 60 s (the utilization is averaged over the last second, so it is still high right after the
# previous experiment ended)
gpu_busy() {
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=process_name --format=csv,noheader | grep -v -E "$DESKTOP" | grep -q . && return 0
  local u idle=0
  for _ in $(seq 1 60); do
    u=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    if [ "$u" -le 10 ]; then idle=$((idle + 1)); [ $idle -ge 3 ] && return 1; else idle=0; fi
    sleep 1
  done
  return 0
}

mkdir -p "$OUT"
for e in "${EXPERIMENTS[@]}"; do
  IFS='|' read -r name bin desc <<<"$e"
  if [ ${#ONLY[@]} -gt 0 ]; then
    keep=0; for o in "${ONLY[@]}"; do [ "$o" = "$name" ] && keep=1; done; [ $keep = 1 ] || continue
  fi
  dir=$OUT/$name
  if [ -d "$dir" ] && [ -n "$(ls -A "$dir" 2>/dev/null)" ]; then echo "$name: $dir is not empty, skipped"; continue; fi
  [ -x "$BUILD/bin/$bin" ] || { echo "$name: $BUILD/bin/$bin is not built, skipped"; continue; }
  if gpu_busy; then echo "$name: another process is using the GPU, stopping"; exit 1; fi
  # A pause file holds the runner between experiments (for GPU work that must not overlap a measurement): touch it to pause,
  # remove it to continue. The experiment in flight is never interrupted.
  if [ -e "$OUT/.pause" ]; then echo "$(date -Iseconds) paused before $name (remove $OUT/.pause to continue)"; while [ -e "$OUT/.pause" ]; do sleep 5; done; fi
  mkdir -p "$dir"
  echo "$(date -Iseconds) $name: $desc"
  start=$(date +%s)
  (cd "$dir" && "$BUILD/bin/$bin" > stdout.txt 2> stderr.txt; echo $? > exit_code.txt)
  rc=$(cat "$dir/exit_code.txt")
  printf '%s %s: exit %s after %s s; csv: %s\n' "$(date -Iseconds)" "$name" "$rc" "$(( $(date +%s) - start ))" "$(ls "$dir" | grep -c '\.csv$')"
  # a first block of junk (-nan variances) means the warm-up sweep wrote to the CSV
  if grep -l -e '-nan' "$dir"/*.csv >/dev/null 2>&1; then echo "  WARNING: -nan in $(grep -l -e '-nan' "$dir"/*.csv | xargs -n1 basename | tr '\n' ' ')"; fi
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=clocks.sm,clocks.mem,temperature.gpu,power.draw --format=csv > "$dir/gpu-at-end.csv"
done
