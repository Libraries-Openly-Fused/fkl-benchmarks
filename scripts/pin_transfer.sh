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
# Do the results transfer to a newer pin of the library? For every experiment: rebuild its program from the commit of fkl-benchmarks that produced the numbers (the "# repo" line of its CSV), with
# external/fkl replaced by the NEW tree (everything else as it was), and compare every kernel of the new binary with the kernel of the frozen binary that produced the numbers (cuobjdump -sass,
# normalised, per mangled name, or per demangled name without the ResizeSampling template argument when the names differ: scripts/sass_compare_builds.py). Compile only: nothing is executed.
#   source scripts/env-cuda13.4.sh; scripts/pin_transfer.sh <new fkl tree (the checkout of external/fkl)> <work dir> <result dir> <tree of 38c4f32> <tree of fb9ac3a>
set -u
FKL=$(cd "$1" && pwd); WORK=$2; OUT=$3; OLD=$(cd "${4:-$1}" && pwd); OLD0=$(cd "${5:-$1}" && pwd)
ROOT=$(cd "$(dirname "$0")/.." && pwd)
CORES=${PIN_CORES:-8-15,24-31}
IGNORE=',\(fk::ResizeSampling\)0'
mkdir -p "$WORK" "$OUT"
{ echo "new tree: $(git -C "$FKL" log --oneline -1)"; echo "old tree (source check of the E2 program): $(git -C "$OLD" log --oneline -1)"; echo "oldest tree (source check of the NV12 program): $(git -C "$OLD0" log --oneline -1)"; } | tee "$OUT/trees.txt"
# build_group <tag> <benchmarks commit> <fkl tree> <frozen build dir under build/> <preset and cmake options> <targets...>     (ONLY="tag tag" runs just these groups)
build_group() {
  local tag=$1 sha=$2 tree=$3 frozen=$4 preset=$5; shift 5
  [ -n "${ONLY:-}" ] && [[ " $ONLY " != *" $tag "* ]] && return 0
  local W=$WORK/src-$tag
  if [ ! -d "$W" ]; then
    git -C "$ROOT" worktree add -q --detach "$W" "$sha" || return 1
    rm -rf "$W/external/fkl" "$W/external/cvGPUSpeedup" "$W/external/FastNPP"
    ln -s "$tree" "$W/external/fkl"; ln -s "$ROOT/external/cvGPUSpeedup" "$W/external/cvGPUSpeedup"; ln -s "$ROOT/external/FastNPP" "$W/external/FastNPP"
    cp "$ROOT/CMakeUserPresets.json" "$W/"
  fi
  taskset -c "$CORES" cmake --preset $preset -S "$W" -B "$W/build" > "$OUT/configure-$tag.log" 2>&1 || { echo "$tag: configure failed (see configure-$tag.log)"; return 1; }
  taskset -c "$CORES" nice -n 5 cmake --build "$W/build" --target "$@" -j 12 > "$OUT/build-$tag.log" 2>&1 || { echo "$tag: build failed (see build-$tag.log): $(grep -m2 'error' "$OUT/build-$tag.log" | cut -c1-200)"; return 1; }
  [ "$frozen" = "-" ] && { echo "== $tag (benchmarks $sha, library $(git -C "$tree" rev-parse --short HEAD)): built, the baseline of the group that follows"; return 0; }
  local base="$ROOT/build/$frozen/bin"; [[ "$frozen" == @* ]] && base="$WORK/src-${frozen#@}/build/bin"
  echo "== $tag (benchmarks $sha, library $(git -C "$tree" rev-parse --short HEAD), compared with $frozen)"
  # by mangled kernel name; when some names exist on one side only, once more by demangled name without the template argument that the newer library added with its default value
  taskset -c "$CORES" python3 "$ROOT/scripts/sass_compare_builds.py" "$base" "$W/build/bin" --glob "*_cu" --out "$OUT/sass_$tag.txt" | grep "^kernels identical"
  if ! grep -q 'only in A: 0, only in B: 0' "$OUT/sass_$tag.txt"; then
    taskset -c "$CORES" python3 "$ROOT/scripts/sass_compare_builds.py" "$base" "$W/build/bin" --glob "*_cu" --match-demangled --ignore "$IGNORE" --out "$OUT/sass_${tag}_demangled.txt" | grep "^kernels identical"
  fi
}
build_group a334424 a334424 "$FKL" e0-frozen-a334424 local-rtx4090 benchmark_latency_hiding_cu benchmark_numinstructions_x_operation_cu
build_group 269d229 269d229 "$FKL" e0-frozen-269d229 local-rtx4090 benchmark_CPUandGPU_cvGS_vs_fk_cu benchmark_CPU_OpenCV_vs_cvGS_cu benchmark_data_type_cu
build_group 5aac244 5aac244 "$FKL" e0-frozen-5aac244 local-rtx4090 benchmark_CPU_NPP_vs_FastNPP_cu benchmark_npp_batchresize_x_split3D_cu
build_group 1a70cab 1a70cab "$FKL" e0-frozen-1a70cab local-rtx4090 benchmark_image_resolution_MAD_loop_cu benchmark_horizontal_fusion_cu benchmark_CPUandGPU_cvGS_vs_fk_cu resize_convention_cu
build_group a266bc9 a266bc9 "$FKL" e0-frozen-a266bc9 local-rtx4090 graph_instantiate_time_cu benchmark_vertical_fusionCompareMulAdd1_cu benchmark_vertical_fusionCompareMul1_cu benchmark_vertical_fusionCompareMulAdd_cu
build_group e1 74f47c6 "$FKL" rtx4090-e1 "rtx4090-e1 -DFKLB_BUILD_OPENCV_CUDA=OFF -DFKLB_BUILD_NPP=OFF" e1_vf_chain_cu e1_hf_batch_cu e1_preproc_cu
build_group mul4-old e9e599d "$OLD" e4 local-rtx4090 benchmark_image_resolution_MAD_loop_mul4_cu
build_group mul4-new e9e599d "$FKL" e4 local-rtx4090 benchmark_image_resolution_MAD_loop_mul4_cu
build_group nan_saturate f3afadc "$FKL" local-rtx4090 local-rtx4090 nan_saturate_cu
build_group nv12-old f81306d "$OLD0" - local-rtx4090 benchmark_nv12_pipeline_cu
build_group nv12 f81306d "$FKL" @nv12-old local-rtx4090 benchmark_nv12_pipeline_cu
build_group e2-old e9e599d "$OLD" e4 local-rtx4090 e2_fkl_chain_cu
build_group e2-new e9e599d "$FKL" e4 local-rtx4090 e2_fkl_chain_cu
echo "pin transfer done $(date -Iseconds)"
