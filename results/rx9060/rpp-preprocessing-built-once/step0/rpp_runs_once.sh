#!/usr/bin/env bash
# rpp_runs_once.sh <out dir>: the built-once runs of benchmark_rpp_preprocessing_hip on the RX 9060 (fkl-benchmarks main bdc7d19, FKL 64ad9195):
# three noise runs (E5_ROTATE_OFFSET 0, 1, 2) and the natural-image run, each after the idle wait; the defaults of the program otherwise
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp
BIN=$NS/vbuild/run-once-hip/bin/benchmark_rpp_preprocessing_hip
IMG=/home/cudeiro/rocm-fkl-benchmarks/e5-cpu/data/kodim23.ppm
IMGINFO="Kodak Lossless True Color Image Suite kodim23.png https://r0k.us/graphics/kodak/kodak/kodim23.png sha256 e3111a2fd4da24af15d6459ef9eacfe54106b38e27b4a21821b75c3f5d2d5baf (released by Kodak for unrestricted usage)"
export LD_LIBRARY_PATH=/opt/rocm/core-10.1/lib:${LD_LIBRARY_PATH:-}
unset E5_ITERS E5_WARMUP E5_WARMUP_SECONDS E5_IMAGE E5_IMAGE_INFO E5_ROTATE_OFFSET
mkdir -p $OUT
LOG=$OUT/runs.log; : > $LOG
idle() {   # the runner is idle: no Runner.Worker and a load average below 0.8 (1 minute)
  local waited=0
  while pgrep -x Runner.Worker > /dev/null || awk '{exit !($1 >= 0.8)}' /proc/loadavg; do sleep 15; waited=$((waited + 15)); done
  echo "  idle after ${waited} s wait; loadavg $(cut -d' ' -f1-3 /proc/loadavg)" >> $LOG
}
run() {   # <name> <binary> <rotate offset> [natural]
  local name=$1 bin=$2 rot=$3 nat=${4:-}
  mkdir -p $OUT/$name
  echo "== $name: $(basename $(dirname $(dirname $bin))) sha256 $(sha256sum $bin | cut -c1-16) E5_ROTATE_OFFSET=$rot ${nat:+E5_IMAGE=$IMG}" >> $LOG
  idle
  echo "  start $(date -u +%FT%TZ)" >> $LOG
  ( cd $OUT/$name && ulimit -v 40000000 && if [ -n "$nat" ]; then E5_ROTATE_OFFSET=$rot E5_IMAGE=$IMG E5_IMAGE_INFO="$IMGINFO" $bin > stdout.log 2>&1; else E5_ROTATE_OFFSET=$rot $bin > stdout.log 2>&1; fi; echo "  exit $?" >> $LOG )
  echo "  end $(date -u +%FT%TZ); $(tail -1 $OUT/$name/stdout.log)" >> $LOG
}
run noise/run0 $BIN 0
run noise/run1 $BIN 1
run noise/run2 $BIN 2
run natural-image $BIN 0 natural
echo "ALL DONE $(date -u +%FT%TZ)" >> $LOG
