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
# Builds the exhaustive saturate_cast program (fkl-self/numerics/saturate_exhaustive.cu) against ANOTHER FKL include tree than the pinned submodule, with exactly the flags and the
# link line of the CMake target e6_saturate_exhaustive_cu: the same source, only the -I of fused_kernel changes. Compile only.
#   source scripts/env-cuda13.4.sh; scripts/e6_build_against.sh <build dir with e6_saturate_exhaustive_cu already built> <path to include/ of the FKL tree> <name of the binary>
set -eu
B=$(cd "$1" && pwd); INC=$2; NAME=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
D=$B/fkl-self/CMakeFiles/e6_saturate_exhaustive_cu.dir
PIN=$ROOT/external/fkl/include
RSP=$(sed "s#-I\"$PIN\"#-I\"$INC\"#" "$D/includes_CUDA.rsp")
grep -q "$INC" <<<"$RSP" || { echo "the pinned include path was not found in $D/includes_CUDA.rsp"; exit 1; }
OBJ=$B/fkl-self/CMakeFiles/e6_saturate_exhaustive_cu.dir/numerics/saturate_exhaustive_$NAME.cu.o
FLAGS=$(grep '^CUDA_FLAGS' "$D/flags.make" | sed 's/^CUDA_FLAGS = //')
eval "$FKLB_CUDA_HOME/bin/nvcc -forward-unknown-to-host-compiler -DENABLE_BENCHMARK $RSP $FLAGS -c $ROOT/fkl-self/numerics/saturate_exhaustive.cu -o $OBJ"
LINK=$(cat "$D/link.txt" | sed "s#CMakeFiles/e6_saturate_exhaustive_cu.dir/numerics/saturate_exhaustive.cu.o#$OBJ#; s#-o ../bin/e6_saturate_exhaustive_cu#-o ../bin/e6_saturate_exhaustive_${NAME}_cu#")
(cd "$B/fkl-self" && eval "$LINK")
echo "built $B/bin/e6_saturate_exhaustive_${NAME}_cu against $INC"
