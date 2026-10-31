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
# Build the batched pre-processing programs (E1c: e1_preproc_cu, E4: e4_preprocessing_cu) of ONE commit of this repository against several trees of the library, each in its own build
# directory, to compare their kernels and their times (compile only: nothing is executed).
#   source scripts/env-cuda13.4.sh; scripts/regression_build.sh <benchmarks commit> <work dir> <tag>=<fkl tree> [<tag>=<fkl tree> ...]
set -u
SHA=$1; WORK=$2; shift 2
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORES=${REG_CORES:-4-7,20-23}
mkdir -p "$WORK"
one() {
  local tag=${1%%=*} tree; tree=$(cd "${1#*=}" && pwd)
  local W=$WORK/src-$tag
  if [ ! -d "$W" ]; then
    git -C "$ROOT" worktree add -q --detach "$W" "$SHA" || return 1
    rm -rf "$W/external/fkl" "$W/external/cvGPUSpeedup" "$W/external/FastNPP"
    ln -s "$tree" "$W/external/fkl"; ln -s "$ROOT/external/cvGPUSpeedup" "$W/external/cvGPUSpeedup"; ln -s "$ROOT/external/FastNPP" "$W/external/FastNPP"
    cp "$ROOT/CMakeUserPresets.json" "$W/"
  fi
  taskset -c "$CORES" cmake --preset local-rtx4090 -S "$W" -B "$W/build" > "$WORK/configure-$tag.log" 2>&1 || { echo "$tag: configure failed"; return 1; }
  taskset -c "$CORES" nice -n 5 cmake --build "$W/build" --target e4_preprocessing_cu e1_preproc_cu -j 2 > "$WORK/build-$tag.log" 2>&1 && echo "$tag: built, library $(git -C "$tree" rev-parse --short HEAD)" || echo "$tag: build failed ($(grep -m1 error "$WORK/build-$tag.log" | cut -c1-160))"
}
for spec in "$@"; do one "$spec" & done
wait
echo "regression build done $(date -Iseconds)"
