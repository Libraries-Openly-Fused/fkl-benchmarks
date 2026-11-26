#!/bin/bash
# C-PM-CUDA: variant B at N = 3848 compiled once more with scripts/e3_sweep.py's nvcc command (line 143), to keep ptxas's full message: the
# sweep's CSV keeps a shortened first line. COMPILE ONLY, no GPU visible; the TU is the sweep's own (work-B/current_B_3848.cu).
set -u
source /home/oscar/tfkl-bench/toolchains/env-cuda13.4.sh
export CUDA_VISIBLE_DEVICES= TMPDIR=/home/oscar/tfkl-bench/scratch/pin-435/e3/tmp
cd /home/oscar/tfkl-bench/scratch/pin-435/e3/ptxas-B-3848
echo "# $(date -Iseconds) start; TU sha256 $(sha256sum /home/oscar/tfkl-bench/scratch/pin-435/e3/work-B/current_B_3848.cu | cut -d' ' -f1)"
taskset -c 9 nvcc -std=c++17 -O3 -DNDEBUG "--generate-code=arch=compute_89,code=[compute_89,sm_89]" --threads 1 -ftemplate-depth=1000 -lineinfo --resource-usage -I /home/oscar/tfkl-bench/worktrees/fkl-benchmarks-pin-435/external/fkl/include /home/oscar/tfkl-bench/scratch/pin-435/e3/work-B/current_B_3848.cu -o /home/oscar/tfkl-bench/scratch/pin-435/e3/ptxas-B-3848/current_B_3848 > nvcc-stdout.txt 2> nvcc-stderr.txt
echo "# $(date -Iseconds) nvcc exit $?"
