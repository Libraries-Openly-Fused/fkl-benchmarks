#!/usr/bin/env bash
# run_checks.sh <out dir>: the comparisons of this record, from the two pin_build.sh builds (stages pin433-base and pin433-head) and the
# binaries of the task B GPU window (../task-b-gpu-window/, staged in tmp/taskb/window-bins). Compile products only: nothing is run on a GPU.
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; HERE=$(cd "$(dirname "$0")" && pwd)
DC=$NS/wt-pin433/scripts/devcode_compare_renamed.py   # scripts/devcode_compare_renamed.py of fkl-benchmarks a8f9daa
A=$NS/vbuild/pin433-base-hip/bin; B=$NS/vbuild/pin433-head-hip/bin; W=$NS/taskb/window-bins
PY="bash $NS/scripts/safepy.sh"   # python3 -I with ulimit -v 2 GB and TMPDIR in tmp/sys
export ROCM_PATH=/opt/rocm/core-10.1
mkdir -p $OUT/functions $OUT/functions-labels $OUT/e19-host-code/offsets $OUT/e19-host-code/rodata
# 1. Device code (gfx1200) of every HIP program, a8f9daa + FKL 24478f0 (A) against 2160c70 + FKL b10fc66 (B), both modes
$PY $DC $A $B --table $OUT/functions > $OUT/devcode-compare.txt 2>&1; echo "devcode default exit $?"
$PY $DC $A $B --labels-as-symbols --table $OUT/functions-labels > $OUT/devcode-compare-labels-as-symbols.txt 2>&1; echo "devcode labels exit $?"
# 2. The function that holds each symbol that differs with --labels-as-symbols
$PY $HERE/label_owners.py $DC $A $B $OUT/functions-labels e18_box_hip e19_batched_box_hip e19_crop_sweep_hip > $OUT/labels-as-symbols-owners.txt 2>&1
echo "label owners exit $?"
# 3. The three rpp programs against the task B GPU window's binaries: head against fklb-after (B), base against fklb-before (A)
ONLY=e18_box_hip,e19_batched_box_hip,e19_crop_sweep_hip
$PY $DC $W/fklb-after $B --only $ONLY > $OUT/devcode-head-vs-task-b-window-after.txt 2>&1; echo "head vs window after exit $?"
$PY $DC $W/fklb-before $A --only $ONLY > $OUT/devcode-base-vs-task-b-window-before.txt 2>&1; echo "base vs window before exit $?"
# 4. Host code (x86-64) of the two E19 programs, A against B, with the reach of each row's timed call
ROOTS=(--root 'fkl_loop=::\{lambda\(\)#1\}>::_M_invoke\(' --root 'fkl_batch=::\{lambda\(\)#3\}>::_M_invoke\(' --root 'rpp=::\{lambda\(\)#5\}>::_M_invoke\('
       --harness '^void fklb::timeIt<' --harness '^void fklb::timePreparedLaunch<')
for P in e19_batched_box_hip e19_crop_sweep_hip; do
    $PY $HERE/host_compare.py $A/$P $B/$P --out $OUT/e19-host-code/offsets "${ROOTS[@]}" > /dev/null 2>$OUT/e19-host-code/offsets/$P.stderr
    echo "host $P offsets exit $?"
    $PY $HERE/host_compare.py $A/$P $B/$P --out $OUT/e19-host-code/rodata --rodata-bytes 16 \
        --same-path $NS/vsrc/pin433-base=$NS/vsrc/pin433-head --same-path $NS/vbuild/pin433-base-hip=$NS/vbuild/pin433-head-hip \
        "${ROOTS[@]}" > /dev/null 2>$OUT/e19-host-code/rodata/$P.stderr
    echo "host $P rodata exit $?"
done
find $OUT -name '*.stderr' -empty -delete
# 5. sha256 of every executable of the four builds
( cd $NS/vbuild && for d in pin433-base-hip pin433-head-hip pin433-base-ocv-cpp pin433-head-ocv-cpp; do [ -d $d/bin ] && (cd $d/bin && sha256sum * | sed "s|  |  $d/|"); done ) > $OUT/binaries-sha256.txt
echo "binaries $(wc -l < $OUT/binaries-sha256.txt)"
