#!/usr/bin/env bash
# vfhf_runs_extra.sh <out dir>: the extra points 15002 and 17002 of benchmark_vfhf_max_speedup_hip on the RX 9060 (main 7b96c47, FKL 64ad9195; the binary of vfhf_run_build_extra.sh), each run after
# the idle wait, the defaults of the program otherwise (warm-up 2 s, budget 30 s):
#   run0 fused first, run1 baseline first (VFHF_BASELINE_FIRST=1), run2 fused first, both points (made from vfhf_runs.sh: only the
#   binary, the comments and the fixed mode differ)
set -u
OUT=$1; WHAT=runs
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
BIN=$NS/vbuild/run-vfhf-extra-hip/bin/benchmark_vfhf_max_speedup_hip
MACHINE=$NS/vfhfw/step0/vfhf_machine.sh
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset VFHF_BASELINE_FIRST VFHF_WARMUP_SECONDS VFHF_BUDGET_SECONDS VFHF_MAX_OPS
mkdir -p $OUT
LOG=$OUT/runs.log; : > $LOG
idle() {   # the runner is idle: no Runner.Worker and a load average below 0.8 (1 minute)
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do sleep 15; waited=$((waited + 15)); done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $LOG
}
run() {   # <name> <baseline first: 0 or 1> <max ops: 0 for every point>
  local name=$1 bf=$2 mo=$3
  mkdir -p $OUT/$name
  echo "== $name: sha256 $(sha256sum $BIN | cut -c1-16) VFHF_BASELINE_FIRST=$bf VFHF_MAX_OPS=$mo" >> $LOG
  idle
  bash $MACHINE > $OUT/$name/machine-before.txt 2>&1
  echo "  start $(date -u +%FT%TZ)" >> $LOG
  ( cd $OUT/$name && ulimit -v 40000000 && VFHF_BASELINE_FIRST=$bf VFHF_MAX_OPS=$mo $BIN > stdout.log 2>&1; echo "  exit $?" >> $LOG )
  echo "  end $(date -u +%FT%TZ); $(tail -1 $OUT/$name/stdout.log)" >> $LOG
  bash $MACHINE > $OUT/$name/machine-after.txt 2>&1
}
case $WHAT in
  check) run check 0 1002 ;;
  runs)  run run0 0 0; run run1 1 0; run run2 0 0 ;;
  *)     echo "usage: vfhf_runs.sh <out dir> check|runs" >&2; exit 2 ;;
esac
echo "ALL DONE $(date -u +%FT%TZ)" >> $LOG
