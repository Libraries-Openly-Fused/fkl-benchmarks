#!/usr/bin/env bash
# run_checks.sh <out dir>: the comparisons of this record, from the two pin_build.sh builds (stages pin435-base: fkl-benchmarks 976e42b with
# FKL b10fc66, and pin435-head: a4b493c with FKL 48fbf3f). Compile products only: nothing runs on a GPU. E3 (VI-L) is step0/e3_series.sh.
set -u
OUT=$1
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; HERE=$(cd "$(dirname "$0")" && pwd)
A=$NS/vbuild/pin435-base-hip/bin; B=$NS/vbuild/pin435-head-hip/bin
PY="bash $NS/scripts/safepy.sh"   # python3 -I with ulimit -v 2 GB and TMPDIR in tmp/sys
FKLB=$NS/wt-pin435                # fkl-benchmarks a4b493c (pin-fkl-435): scripts/e11_diagnostics.py and fkl-self/diagnostics
export ROCM_PATH=/opt/rocm/core-10.1 TMPDIR=$NS/sys
mkdir -p $OUT/functions $OUT/host-code/raw $OUT/host-code/rename $OUT/host-code/default $OUT/host-code/strict $OUT/e11
# 1. (a) Device code (gfx1200) of every HIP program, A against B, names paired by the task C rename rule, both modes
$PY $HERE/devcode_compare_taskc.py $A $B --table $OUT/functions > $OUT/devcode-compare.txt 2>&1; echo "devcode default exit $?"
$PY $HERE/devcode_compare_taskc.py $A $B --labels-as-symbols > $OUT/devcode-compare-labels-as-symbols.txt 2>&1; echo "devcode labels exit $?"
# 2. (b) Host code (x86-64) of the four programs behind RX 9060 paper rows. default = --rename-taskc --line-args, with the roots of the rows;
#    strict = default with --rodata-bytes 16; raw (the #61 script's form) and rename (--rename-taskc only) give the levels of the README.
CROP=(--root 'fkl_loop=::\{lambda\(\)#1\}>::_M_invoke\(' --root 'fkl_batch=::\{lambda\(\)#3\}>::_M_invoke\(' --root 'rpp=::\{lambda\(\)#5\}>::_M_invoke\('
      --harness '^voidfklb::timeIt<' --harness '^voidfklb::timePreparedLaunch<')
RPPROWS=(fkl_fused fkl_fused_halfpixel fkl_perop_batch rpp_crop_batch rpp_crop_batch_unsplit hand_batch hand_padded hand_roi fkl_fused_halfpixel_once
         hand_roi_halfpixel_once fastrpp_shared fastrpp_crop_batch fastrpp_shared_once)   # the row lambdas #1..#13, in the order of the source
PRE=(); i=1
for row in "${RPPROWS[@]}"; do PRE+=(--root "$row=rppbench::runA<[0-9]+>\(.*\)::\{lambda\(std::chrono::time_point<[^#]*&\)#$i\}>::_M_invoke\("); i=$((i + 1)); done
PRE+=(--harness '^fklb::measureWall\(' --harness '^rppbench::timeVariant\(')
SAME=(--same-path $NS/vsrc/pin435-base=$NS/vsrc/pin435-head --same-path $NS/vbuild/pin435-base-hip=$NS/vbuild/pin435-head-hip)
for P in e19_crop_sweep_hip benchmark_rpp_preprocessing_hip benchmark_vfhf_max_speedup_hip benchmark_latency_hiding_hip; do
    R=(); [ $P = e19_crop_sweep_hip ] && R=("${CROP[@]}"); [ $P = benchmark_rpp_preprocessing_hip ] && R=("${PRE[@]}")
    $PY $HERE/host_compare_taskc.py $A/$P $B/$P --out $OUT/host-code/raw > /dev/null 2> $OUT/host-code/raw/$P.stderr; echo "host $P raw exit $?"
    $PY $HERE/host_compare_taskc.py $A/$P $B/$P --rename-taskc --out $OUT/host-code/rename > /dev/null 2> $OUT/host-code/rename/$P.stderr
    echo "host $P rename exit $?"
    $PY $HERE/host_compare_taskc.py $A/$P $B/$P --rename-taskc --line-args --out $OUT/host-code/default "${R[@]}" > /dev/null \
        2> $OUT/host-code/default/$P.stderr; echo "host $P default exit $?"
    $PY $HERE/host_compare_taskc.py $A/$P $B/$P --rename-taskc --line-args --rodata-bytes 16 "${SAME[@]}" --out $OUT/host-code/strict "${R[@]}" \
        > /dev/null 2> $OUT/host-code/strict/$P.stderr; echo "host $P strict exit $?"
done
# raw and rename: the summaries only, gzipped; default and strict: the summaries and diffs as they are, the function tables and reach lists
# gzipped
rm -f $OUT/host-code/raw/*.reach-* $OUT/host-code/rename/*.reach-* $OUT/host-code/raw/*.functions.tsv $OUT/host-code/rename/*.functions.tsv \
      $OUT/host-code/raw/*.diffs.txt $OUT/host-code/rename/*.diffs.txt
gzip -9 -n $OUT/host-code/raw/*.summary.txt $OUT/host-code/rename/*.summary.txt
gzip -9 -n $OUT/host-code/default/*.functions.tsv $OUT/host-code/default/*.reach-* $OUT/host-code/strict/*.functions.tsv $OUT/host-code/strict/*.reach-*
find $OUT/host-code -name "*.stderr" -size 0 -delete   # the empty error outputs
# 3. (b) VI-K: each measured row's source at its measured commit against main 976e42b (the program is not changed by this pin move)
{ for c in 84d0bfa 6dc0541 151d517 bdc7d19; do
      rows=$(cd $FKLB && git show $c:rpp/benchmark_rpp_preprocessing.h | python3 -I -c "
import re, sys
t = sys.stdin.read(); m = re.search(r'timeRotated\(out, N, [^,]+,\s*\{(.*?)\}\);', t, re.S)
print(' '.join(re.findall(r'\{\"([a-z_]+)\", ', m.group(1))))")
      echo "=== measured commit $c against 976e42b; rows: $rows"
      (cd $FKLB && $PY $HERE/row_sources.py . $c 976e42b $rows)
      echo "--- the timed loop and the warm-up loop of fklb::measureWall (common/hip_wallclock.h), whitespace collapsed, sha256:"
      for k in $c 976e42b; do
          t=$(cd $FKLB && git show $k:common/hip_wallclock.h | sed -n '/^inline WallStats measureWall/,/^}/p')
          echo "  $k timed loop $(echo "$t" | sed -n '/std::vector<double> total/,/^    WallStats s;/p' | tr -s ' \n' ' ' | sha256sum | cut -c1-16)," \
               "warm-up loop $(echo "$t" | sed -n '/while (!(runs/,/const double warmupSeconds/p' | tr -s ' \n' ' ' | sha256sum | cut -c1-16)"
      done
  done } > $OUT/vi-k-sources.txt 2>&1; echo "vi-k sources exit $?"
# 4. (c) VI-P: E11, amdclang++ only, the library without the checks (c8fcbe8), the pin (b10fc66) and the head (48fbf3f)
( ulimit -v 16000000
  for spec in unchanged-c8fcbe8:fkl-c8fcbe8 pin-b10fc66:fkl-b10fc66 head-48fbf3f:wt-fkl-taskc; do
      label=${spec%%:*}; tree=${spec##*:}; mkdir -p $OUT/e11/$label
      python3 -I $FKLB/scripts/e11_diagnostics.py --compilers $ROCM_PATH/bin/amdclang++ --label $label --include $NS/$tree/include --out $OUT/e11/$label \
          > $OUT/e11/$label.log 2>&1; echo "e11 $label exit $?"
  done )
# 5. (c) VI-N: FKL's HIP tests and targets, in the ROCm CI configuration builds of the two FKL commits
{ for d in $NS/vbuild/pr435-head-hip $NS/taskc/phase2/build-hip-base; do
      src=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' $d/CMakeCache.txt); c=$(cd $src && git rev-parse HEAD)
      echo "FKL $c ($d):"
      echo "  ctest tests $(cd $d && ctest -N | grep -c 'Test *#'); with _hip at the end: $(cd $d && ctest -N -R '_hip$' -E '^benchmark_' | grep -c 'Test *#');" \
           "HIP compile-rejection tests: $(cd $d && ctest -N -R '_hip_rejected$' | grep -c 'Test *#')"
      echo "  _hip build targets: $(cd $d && ninja -t targets all | grep -oE '^[A-Za-z0-9_]+_hip:' | grep -vc '^cmake_object_order')"
  done } > $OUT/vi-n.txt 2>&1; echo "vi-n exit $?"
# 6. sha256 of every executable of the two builds
( cd $NS/vbuild && for d in pin435-base-hip pin435-head-hip; do (cd $d/bin && sha256sum * | sed "s|  |  $d/|"); done ) > $OUT/binaries-sha256.txt
echo "binaries $(wc -l < $OUT/binaries-sha256.txt)"
