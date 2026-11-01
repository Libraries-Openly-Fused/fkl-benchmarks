#!/bin/bash
# The series of the README of this folder, one compile at a time (scripts/e3_clang_limits.py waits while a CI job is running on the machine).
#   ./run_series.sh [SECTION...]      default: 1 2 3 4 5 6 7 8 9
# Needs ROCm (ROCM_PATH, default /opt/rocm/core-10.1), clang++-23 and g++-13; the FKL tree is external/fkl of this repository (the pinned c8fcbe8).
# Sections 1 to 5 were run first, in one go; 6 to 9 afterwards, with the commands below. Sections 6 and 7 read the result of the bisection of section 2 from e3_limits.csv.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); ROOT=$(cd "$HERE/../../.." && pwd)
export ROCM_PATH=${ROCM_PATH:-/opt/rocm/core-10.1}
export HIP_PATH=$ROCM_PATH LD_LIBRARY_PATH=$ROCM_PATH/lib:${LD_LIBRARY_PATH:-}
L="python3 $ROOT/scripts/e3_clang_limits.py"; O="--out $HERE"

# the largest N of the bisection with address randomisation off that compiled: lo_ok BACKEND
lo_ok() {
  python3 - "$HERE/e3_limits.csv" "$1" <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r['series'] == 'stack-bisect-noaslr' and r['backend'] == sys.argv[2] and r['outcome'] == 'ok']
print(max(int(r['n']) for r in rows))
PY
}

s1() {
  echo "== 1. default flags (stack limit 8192 kB, address randomisation on, default -fbracket-depth)"
  for b in hip clang; do $L run --backend $b --ns 1100,2046,2047 --series default $O; done
  $L run --backend gcc --ns 1100,2046,2047,2100 --series default $O
}
s2() {
  echo "== 2. where the 8 MiB stack runs out: bisection with address randomisation off, then 5 repeats of N-1, N, N+1 (N = first failing) with it on"
  local t; t=$(mktemp)
  for b in hip clang; do
    $L bisect --backend $b --lo 1100 --hi 2046 --no-aslr --series stack-bisect-noaslr $O > "$t"; cat "$t"
    local hi; hi=$(tail -1 "$t" | sed -E 's/.* not ([0-9]+)$/\1/')
    $L run --backend $b --ns $((hi - 1)),$hi,$((hi + 1)) --reps 5 --series stack-repeats-aslr $O
  done
  rm -f "$t"
}
s3() {
  echo "== 3. stack limit raised to 16 MiB: N = 2046 (below the fold expression limit) and 2047"
  for b in hip clang; do $L run --backend $b --ns 2046,2047 --stack-kb 16384 --series stack-16MiB $O; done
}
s4() {
  echo "== 4. -fbracket-depth at N = 2100: 2100 and 2101 are not enough, 2102 = N + 2 is"
  for b in hip clang; do
    for d in 2100 2101 auto; do $L run --backend $b --ns 2100 --bracket-depth $d --stack-kb 32768 --series bracket-depth $O; done
  done
}
s5() {
  echo "== 5. N = 4000 (bracket depth N + 2, stack limit 32 MiB; g++ with its defaults)"
  for b in clang hip; do $L run --backend $b --ns 4000 --bracket-depth auto --stack-kb 32768 --series n4000 $O; done
  $L run --backend gcc --ns 4000 --series n4000 $O
}
s6() {
  echo "== 6. N-1 of the boundary (largest N that compiled without randomisation, minus one), 5 repeats with randomisation on"
  for b in hip clang; do $L run --backend $b --ns $(( $(lo_ok $b) - 1 )) --reps 5 --series stack-below-aslr $O; done
}
s7() {
  echo "== 7. the stack that N = 2046 needs: 10, 12 and 14 MiB with address randomisation off"
  for b in hip clang; do
    for kb in 10240 12288 14336; do $L run --backend $b --ns 2046 --stack-kb $kb --no-aslr --series stack-needed-2046 $O; done
  done
}
s8() {
  echo "== 8. does the kernel launch and compute the same as the CPU back-end (g++)? kernel argument block of 712 B (N = 100) to 16,312 B (N = 4000)"
  $L check --backend hip --ns 100,1100 --series check $O
  $L check --backend hip --ns 2046,3000,4000 --stack-kb 32768 --bracket-depth auto --series check $O
}
s9() {
  echo "== 9. amdclang++ without the resource usage remark (-Rpass-analysis=kernel-resource-usage), which costs time and memory"
  $L run --backend hip --ns 1100 --no-remarks --series hip-no-remarks $O
  $L run --backend hip --ns 2046 --no-remarks --stack-kb 16384 --series hip-no-remarks $O
  $L run --backend hip --ns 4000 --no-remarks --bracket-depth auto --stack-kb 32768 --series hip-no-remarks $O
}
for s in ${*:-1 2 3 4 5 6 7 8 9}; do s$s; done
echo "== done"
