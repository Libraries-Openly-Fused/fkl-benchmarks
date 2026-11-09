#!/usr/bin/env bash
# runs.sh <out dir>: the RX 9060 timing runs of benchmark_rpp_preprocessing_hip with the FastRPP variants (fkl-benchmarks 84d0bfa, FKL 1da77899,
# FastRPP efbf009), the protocol of results/rx9060/rpp-preprocessing-built-once/step0/rpp_runs_once.sh: three noise runs (E5_ROTATE_OFFSET 0, 1,
# 2) and the natural-image run (kodim23, offset 0), each after the idle wait (no Runner.Worker, load average below 0.8), the program's defaults
# otherwise (E5_ITERS 30, E5_WARMUP 3, E5_WARMUP_SECONDS 2.0), ulimit -v 40 GB as before.
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
BIN=$NS/vbuild/fastrpp-run-hip/bin/benchmark_rpp_preprocessing_hip
IMG=/home/cudeiro/rocm-fkl-benchmarks/e5-cpu/data/kodim23.ppm
IMGINFO="Kodak Lossless True Color Image Suite kodim23.png https://r0k.us/graphics/kodak/kodak/kodim23.png sha256 e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf (released by Kodak for unrestricted usage)"
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E5_ITERS E5_WARMUP E5_WARMUP_SECONDS E5_IMAGE E5_IMAGE_INFO E5_ROTATE_OFFSET
mkdir -p $OUT
LOG=$OUT/runs.log; : > $LOG
{ echo "binary $(sha256sum $BIN)"; echo "environment variables naming HIP/HSA/ROC/AMD/kernarg: $(env | grep -c -i -E '^(HIP|HSA|ROC|AMD)_|kernarg')"
  echo "kodim23 sha256 $(sha256sum $IMG | cut -d' ' -f1)"; } >> $LOG
idle() {   # the runner is idle: no Runner.Worker and a load average below 0.8 (1 minute)
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do sleep 15; waited=$((waited + 15)); done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $LOG
}
run() {   # <name> <rotate offset> [natural]
  local name=$1 rot=$2 nat=${3:-}
  mkdir -p $OUT/$name
  echo "== $name: sha256 $(sha256sum $BIN | cut -c1-16) E5_ROTATE_OFFSET=$rot ${nat:+E5_IMAGE=$IMG}" >> $LOG
  idle
  echo "  start $(date -u +%FT%TZ)" >> $LOG
  ( cd $OUT/$name && ulimit -v 40000000 && if [ -n "$nat" ]; then E5_ROTATE_OFFSET=$rot E5_IMAGE=$IMG E5_IMAGE_INFO="$IMGINFO" $BIN > stdout.log 2>&1; else E5_ROTATE_OFFSET=$rot $BIN > stdout.log 2>&1; fi; echo "  exit $?" >> $LOG )
  echo "  end $(date -u +%FT%TZ); $(tail -1 $OUT/$name/stdout.log)" >> $LOG
}
run noise/run0 0
run noise/run1 1
run noise/run2 2
run natural-image 0 natural
echo "ALL DONE $(date -u +%FT%TZ)" >> $LOG
