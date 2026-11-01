#!/bin/bash
# E11, mistake 3, run: builds fkl-self/diagnostics/e11_batch_mismatch_run.cu (a batch of 50 images read, 60 written) for the CPU, CPU_MULTITHREAD, HIP and CUDA back-ends
# and runs each three times. The CUDA program is built only if NVCC is set, and run only if there is an NVIDIA GPU (otherwise it is compiled and not run).
# usage, from the root of fkl-benchmarks (external/fkl and its thread-pool submodule checked out): results/rx9060/e11-diagnostics/batch-run/run.sh > output.txt
#   BACKENDS="cpu cpu_mt hip cuda"   (default) the back-ends to build and run; on an NVIDIA machine: BACKENDS="cpu cpu_mt cuda" NVCC=nvcc run.sh
#   FKL=external/fkl  ARCH=gfx1200  CUDA_ARCH=sm_89  CXX=g++-13  HIPCXX=/opt/rocm/core-10.1/bin/amdclang++  CCBIN=<host compiler of nvcc, default $CXX>  OUT=build/e11-batch-run
set -u
FKL=${FKL:-external/fkl}; ARCH=${ARCH:-gfx1200}; CUDA_ARCH=${CUDA_ARCH:-sm_89}; NVCC=${NVCC:-}; CXX=${CXX:-g++-13}
HIPCXX=${HIPCXX:-/opt/rocm/core-10.1/bin/amdclang++}; BACKENDS=${BACKENDS:-cpu cpu_mt hip cuda}
SRC=fkl-self/diagnostics/e11_batch_mismatch_run.cu
OUT=${OUT:-build/e11-batch-run}; mkdir -p "$OUT"; CCBIN=${CCBIN:-$(command -v $CXX)}
B="-O3 -DNDEBUG -std=c++17 -ftemplate-depth=1000"
has() { case " $BACKENDS " in *" $1 "*) return 0;; esac; return 1; }
echo "# E11 batch-run: $(date -u +%Y-%m-%dT%H:%M:%SZ), FKL $(git -C "$FKL" rev-parse --short=7 HEAD) ($FKL), source $SRC at fkl-benchmarks $(git rev-parse --short=7 HEAD), back-ends: $BACKENDS"
{ has cpu || has cpu_mt; } && echo "# $($CXX --version | head -1)"
has hip && echo "# $($HIPCXX --version | head -1)"
has cuda && [ -n "$NVCC" ] && echo "# $($NVCC --version | tail -2 | head -1)"
build() { echo "\$ $*"; "$@" 2>&1 | grep -E "error|Error" | head -5; }
runs() { for i in 1 2 3; do "$1"; echo "exit code $?"; done; }
if has cpu; then
  echo; echo "## CPU ($CXX)"
  build $CXX -x c++ -march=native $B -I$FKL/include $SRC -o $OUT/e11_run_cpu; runs $OUT/e11_run_cpu
fi
if has cpu_mt; then
  echo; echo "## CPU_MULTITHREAD ($CXX)"
  build $CXX -x c++ -march=native $B -DFK_CPU_MULTITHREAD_ENABLED -DE11_RUN_CPU_MT -I$FKL/include -I$FKL/thread-pool/include -pthread $SRC -o $OUT/e11_run_cpu_mt; runs $OUT/e11_run_cpu_mt
fi
if has hip; then
  echo; echo "## HIP (amdclang++, $ARCH)"
  build $HIPCXX -x hip --offload-arch=$ARCH $B -I$FKL/include $SRC -o $OUT/e11_run_hip; runs $OUT/e11_run_hip
fi
if has cuda && [ -n "$NVCC" ]; then
  echo; echo "## CUDA (nvcc, $CUDA_ARCH)"
  if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    build $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include $SRC -o $OUT/e11_run_cuda; runs $OUT/e11_run_cuda
  else
    echo "no NVIDIA GPU on this machine: compiled only, not run"
    echo "\$ $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include -c $SRC -o $OUT/e11_run_cuda.o"
    $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include -c $SRC -o $OUT/e11_run_cuda.o 2>&1 | grep -E "error" | head -5; echo "nvcc exit code ${PIPESTATUS[0]}"
  fi
fi
