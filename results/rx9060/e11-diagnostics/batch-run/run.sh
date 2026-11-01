#!/bin/bash
# E11, mistake 3, run: builds fkl-self/diagnostics/e11_batch_mismatch_run.cu (a batch of 50 images read, 60 written) for the CPU, CPU_MULTITHREAD and HIP back-ends,
# runs each three times, and, if NVCC is set, compiles it for CUDA (the CUDA program is not run here: this machine has no NVIDIA GPU).
# usage (from the root of fkl-benchmarks, with external/fkl and its thread-pool submodule checked out): [FKL=<FusedKernelLibrary checkout, default external/fkl>] [NVCC=<nvcc>] [ARCH=gfx1200] [CUDA_ARCH=sm_89] [CCBIN=<nvcc host compiler, default $CXX>] results/rx9060/e11-diagnostics/batch-run/run.sh > output.txt
# The binaries go to build/e11-batch-run (OUT).
# On an NVIDIA machine: NVCC=nvcc FKL=... run.sh builds the CUDA program as ./e11_run_cuda and runs it three times.
set -u
FKL=${FKL:-external/fkl}; ARCH=${ARCH:-gfx1200}; CUDA_ARCH=${CUDA_ARCH:-sm_89}; NVCC=${NVCC:-}; CXX=${CXX:-g++-13}
HIPCXX=${HIPCXX:-/opt/rocm/core-10.1/bin/amdclang++}
SRC=fkl-self/diagnostics/e11_batch_mismatch_run.cu
OUT=${OUT:-build/e11-batch-run}; mkdir -p "$OUT"; CCBIN=${CCBIN:-$(command -v $CXX)}
B="-O3 -DNDEBUG -std=c++17 -ftemplate-depth=1000"
echo "# E11 batch-run: $(date -u +%Y-%m-%dT%H:%M:%SZ), FKL $(git -C "$FKL" rev-parse --short=7 HEAD) ($FKL), source $SRC at fkl-benchmarks $(git rev-parse --short=7 HEAD)"
echo "# $($CXX --version | head -1); $($HIPCXX --version | head -1); $( [ -n "$NVCC" ] && $NVCC --version | tail -2 | head -1 )"
build() { echo "\$ $*"; "$@" 2>&1 | grep -E "error|Error" | head -5; }
runs() { for i in 1 2 3; do "$1"; echo "exit code $?"; done; }
echo; echo "## CPU (g++)"
build $CXX -x c++ -march=native $B -I$FKL/include $SRC -o $OUT/e11_run_cpu; runs $OUT/e11_run_cpu
echo; echo "## CPU_MULTITHREAD (g++)"
build $CXX -x c++ -march=native $B -DFK_CPU_MULTITHREAD_ENABLED -DE11_RUN_CPU_MT -I$FKL/include -I$FKL/thread-pool/include -pthread $SRC -o $OUT/e11_run_cpu_mt; runs $OUT/e11_run_cpu_mt
echo; echo "## HIP (amdclang++, $ARCH)"
build $HIPCXX -x hip --offload-arch=$ARCH $B -I$FKL/include $SRC -o $OUT/e11_run_hip; runs $OUT/e11_run_hip
if [ -n "$NVCC" ]; then
  echo; echo "## CUDA (nvcc, $CUDA_ARCH)"
  if command -v nvidia-smi > /dev/null 2>&1 && nvidia-smi -L > /dev/null 2>&1; then
    build $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include $SRC -o $OUT/e11_run_cuda; runs $OUT/e11_run_cuda
  else
    echo "no NVIDIA GPU on this machine: compiled only, not run"
    echo "\$ $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include -c $SRC -o $OUT/e11_run_cuda.o"; $NVCC -x cu -arch $CUDA_ARCH -ccbin $CCBIN $B -I$FKL/include -c $SRC -o $OUT/e11_run_cuda.o 2>&1 | grep -E "error" | head -5; echo "nvcc exit code ${PIPESTATUS[0]}"
  fi
fi
