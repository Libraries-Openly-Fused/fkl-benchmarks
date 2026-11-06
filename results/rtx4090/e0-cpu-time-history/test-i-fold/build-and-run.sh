#!/bin/bash
# Test (i): SYNTHETIC: one function of FKL 1f33224 (cxp::max) copied as text into fold_synth.cpp; NOT a build of the 2025 library. Host only, g++ 15.2, no CUDA.
# Five host flag sets ("none" = what nvcc passes to g++ with an empty CMAKE_BUILD_TYPE: no -O, no -DNDEBUG); run pinned to cores 0-3,16-19, 3 rounds, load recorded.
for v in "none:" "o0ndebug:-O0 -DNDEBUG" "o1ndebug:-O1 -DNDEBUG" "o2ndebug:-O2 -DNDEBUG" "o3ndebug:-O3 -DNDEBUG"; do n=${v%%:*}; f=${v#*:}; g++ -std=c++17 $f fold_synth.cpp -o fold_$n; done
echo "# load before: $(cut -d' ' -f1-3 /proc/loadavg) at $(date -Iseconds)"; echo "flagset,round,B,paper_fold_us,loop_now_us"
for r in 1 2 3; do for n in none o0ndebug o1ndebug o2ndebug o3ndebug; do taskset -c 0-3,16-19 ./fold_$n $n | sed "s/^$n,/$n,$r,/"; done; done
echo "# load after: $(cut -d' ' -f1-3 /proc/loadavg) at $(date -Iseconds)"
