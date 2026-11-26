#!/usr/bin/env bash
# post_window.sh: the analysis of RT-A1's window (window/), without the GPU, with the record's own scripts:
#   1. every rocprofv3-A/nNNN and rocprofv3-B/nNNN: rocprof_calls.py --spin (../../e18-box-phase1/rocprofv3/) -> calls.txt, calls.tsv, calls.err;
#      rpp_spin_with_copies.py (../../e19-batched-box/analysis/) -> rpp-spin-with-copies.txt; then the sha256 of the kernel and HIP API trace
#      CSVs (trace-csv-sha256.txt) and the CSVs gzipped, as in ../../e19-crop-sweep/rocprofv3/;
#   2. a view of the window for A and for B (views/A, views/B: quiet-run1..3, warmup = unused-X, rocprofv3 = rocprofv3-X, gpu_clocks.csv, and the
#      console lines of that binary's runs with the names mapped), and the record's e19_crop_sweep_report.py on each -> summary/A, summary/B;
#   3. the record's kernels_and_spin.py on each -> analysis/kernels-and-spin-A.tsv, -B.tsv;
#   4. retime_report.py -> summary/retime.csv, summary/retime-report.md, summary/checks-identity.txt, paper-data/{A,B}/.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd); R=$(cd "$HERE/.." && pwd); W=$HERE/window
export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys
PY="python3 -I"; ulimit -v 8000000
for x in A B; do
  for d in $W/rocprofv3-$x/n*; do
    ( cd $d && [ -e trace-csv-sha256.txt ] || {
        $PY $R/e18-box-phase1/rocprofv3/rocprof_calls.py trace e19 --spin --out calls.txt --tsv calls.tsv 2> calls.err
        $PY $R/e19-batched-box/analysis/rpp_spin_with_copies.py trace e19 > rpp-spin-with-copies.txt
        (cd trace && sha256sum e19_kernel_trace.csv e19_hip_api_trace.csv) > trace-csv-sha256.txt
        gzip -9 -n trace/e19_kernel_trace.csv trace/e19_hip_api_trace.csv; } )
  done
  V=$HERE/views/$x; rm -rf $V; mkdir -p $V
  for k in 1 2 3; do ln -s ../../window/quiet-run$k-$x $V/quiet-run$k; done
  ln -s ../../window/unused-$x $V/warmup; ln -s ../../window/rocprofv3-$x $V/rocprofv3; ln -s ../../window/gpu_clocks.csv $V/gpu_clocks.csv
  # the console lines of this binary's runs (each run's idle line, start line and exit line), with the run names of the record
  awk -v x=$x '/^  idle after/ {idle = $0; next}
               match($0, /^# [^ ]+ (unused|quiet-run[123]|rocprofv3)-[AB]\/n[0-9]+ /) {
                 split($3, p, "/"); n = split(p[1], q, "-"); if (q[n] != x) next
                 name = p[1]; sub("-" x "$", "", name); if (name == "unused") name = "warmup"
                 line = $0; sub(p[1] "/", name "/", line); if (idle != "" && line !~ / exit /) print idle; idle = ""; print line }' \
      $W/window-console.txt > $V/window-console.txt
  $PY $R/e19-crop-sweep/analysis/e19_crop_sweep_report.py $V $R/e19-batched-box $HERE/summary/$x > /dev/null
  $PY $R/e19-crop-sweep/analysis/kernels_and_spin.py $V/rocprofv3 > $HERE/analysis/kernels-and-spin-$x.tsv
done
$PY $HERE/analysis/retime_report.py $HERE
echo "post-window analysis done"
