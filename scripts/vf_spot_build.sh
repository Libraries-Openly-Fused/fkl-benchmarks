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
# Build the vertical-fusion benchmarks for a few experiment numbers only (-DVF_POINTS), against one or more trees of the library, to repeat some points of the long runs.
# Compile only. The benchmarks are those of this repository at <commit>; with VF_POINTS empty they are the programs of the paper's runs.
#   source scripts/env-cuda13.4.sh; scripts/vf_spot_build.sh <commit> <work dir> "<experiment numbers, ;-separated>" <tag>=<fkl tree> [...]
set -u
SHA=$1; WORK=$2; POINTS=$3; shift 3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORES=${VF_CORES:-8-15,24-31}
mkdir -p "$WORK"
for spec in "$@"; do
  tag=${spec%%=*}; tree=$(cd "${spec#*=}" && pwd); W=$WORK/src-$tag
  if [ ! -d "$W" ]; then
    git -C "$ROOT" worktree add -q --detach "$W" "$SHA" || exit 1
    rm -rf "$W/external/fkl" "$W/external/cvGPUSpeedup" "$W/external/FastNPP"
    ln -s "$tree" "$W/external/fkl"; ln -s "$ROOT/external/cvGPUSpeedup" "$W/external/cvGPUSpeedup"; ln -s "$ROOT/external/FastNPP" "$W/external/FastNPP"
    cp "$ROOT/CMakeUserPresets.json" "$W/"
  fi
  taskset -c "$CORES" cmake --preset local-rtx4090 -S "$W" -B "$W/build" -DVF_POINTS="$POINTS" > "$WORK/configure-$tag.log" 2>&1 || { echo "$tag: configure failed"; exit 1; }
  taskset -c "$CORES" nice -n 5 cmake --build "$W/build" --target benchmark_vertical_fusionCompareMulAdd_cu benchmark_vertical_fusionCompareMulAdd1_cu benchmark_vertical_fusionCompareMul1_cu -j 12 > "$WORK/build-$tag.log" 2>&1 \
    && echo "$tag: built, library $(git -C "$tree" rev-parse --short HEAD), points $POINTS" || echo "$tag: build failed ($(grep -m1 error "$WORK/build-$tag.log" | cut -c1-160))"
done
echo "vf spot build done $(date -Iseconds)"
