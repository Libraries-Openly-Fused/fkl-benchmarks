#!/usr/bin/env bash
# build.sh <FKL tree> <output binary> cpu|hip: builds harness/dpps_before_after.cpp against the FKL tree's include/ and thread-pool/include
# (CPU_MULTITHREAD on, as FKL's tests: -DFK_CPU_MULTITHREAD_ENABLED), with this repository's common/ (sha256.h). FKL main is C++20, so the
# flags are those of main's tests (its build commands, ROCm CI configuration):
#   cpu: amdclang++ as C++ (no HIP: the binary never opens the GPU): -O3 -DNDEBUG -std=c++20 -march=native -mtune=native -pthread;
#   hip: amdclang++ -x hip --offload-arch=gfx1200 -O3 -DNDEBUG -std=c++20 -ftemplate-depth=1000 -pthread, --hip-link.
# The FKL tree must be a clean git checkout; its HEAD is compiled into the binary (HARNESS_FKL_COMMIT) and printed by every run.
# HARNESS_EXECUTORS is 1 when the tree's executor_kernels.h defines launchDPP_Kernel (FKL #436), 0 otherwise (main ebb3891): it picks the calls.
# Compile and link only; nothing runs.
set -eu
FKL=$(cd "${1:?usage: build.sh <FKL tree> <output binary> cpu|hip}" && pwd)
OUT=${2:?usage: build.sh <FKL tree> <output binary> cpu|hip}
KIND=${3:?usage: build.sh <FKL tree> <output binary> cpu|hip}
HERE=$(cd "$(dirname "$0")" && pwd)
FKLB=$(cd "$HERE/../../../.." && pwd)
ROCM=/opt/rocm/core-10.1
[ -z "$(cd "$FKL" && git status --porcelain --untracked-files=no)" ] || { echo "$FKL has changes: not building"; exit 1; }
COMMIT=$(cd "$FKL" && git rev-parse HEAD)
if grep -q "launchDPP_Kernel" "$FKL/include/fused_kernel/core/execution_model/executor_details/executor_kernels.h"; then EXECUTORS=1; else EXECUTORS=0; fi
COMMON=(-O3 -DNDEBUG -std=c++20 -pthread -DFK_CPU_MULTITHREAD_ENABLED
        "-DHARNESS_FKL_COMMIT=\"$COMMIT\"" "-DHARNESS_EXECUTORS=$EXECUTORS" -I"$FKLB" -I"$FKL/include" -I"$FKL/thread-pool/include")
mkdir -p "$(dirname "$OUT")"
( ulimit -v 16000000
  export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys
  case "$KIND" in
    cpu) "$ROCM/bin/amdclang++" -x c++ "${COMMON[@]}" -march=native -mtune=native "$HERE/dpps_before_after.cpp" -o "$OUT" ;;
    hip) "$ROCM/bin/amdclang++" -x hip --offload-arch=gfx1200 "${COMMON[@]}" -ftemplate-depth=1000 "$HERE/dpps_before_after.cpp" -o "$OUT" --hip-link ;;
    *) echo "the third argument is cpu or hip"; exit 1 ;;
  esac )
echo "built $OUT ($KIND) against FKL $COMMIT (HARNESS_EXECUTORS=$EXECUTORS): sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
