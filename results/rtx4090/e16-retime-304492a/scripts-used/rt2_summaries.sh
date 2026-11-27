#!/bin/bash
# RT-2, after the window (CPU only: reads CSVs; nothing is run): each side's E16 summary in the layout of the paper's folder
# (results/rtx4090/cleanup-window-d/cleaned/e16-summary/), by the repository's own scripts, unchanged:
#   scripts/e16_join_clock.py <folder>   the SM clock of every row from the window's sampler -> run<k>/*_clock.csv, join.log
#   scripts/e16_summary.py <folder>      -> e16_summary_core.csv, e16_summary_fusion.csv, e16_summary_kernels.csv, e16_q2_compact.csv,
#                                           e16_text_numbers.txt, summary.log
# For each side, a scratch folder holds copies of the window's run CSVs (run<k>/e16_attention.csv from <side>/e16-attention/run<k>, and
# run<k>/e16_baselines.csv from <side>/e16-baselines/run<k>) and a link to the window's gpu_clocks.csv; the scripts run there, as window D's
# did (so join.log names the scratch paths), and the folder, without the link, is copied to <dest>/<side>/e16-summary/. The copies are
# checked byte for byte against the run folders'. Then rt2_report.py (old = window D's cleaned e16-summary) -> <dest>/analysis/.
# Usage: rt2_summaries.sh <window folder> <scripts folder of the repository> <scratch folder, new> <dest folder (the record)> <old e16-summary>
set -eu
WIN=$(readlink -f $1); SCR=$(readlink -f $2); SC=$3; DEST=$(readlink -f $4); OLD=$(readlink -f $5); HERE=$(dirname $(readlink -f $0))
[ -e $SC ] && { echo "$SC exists"; exit 1; }
for side in A B; do [ -e $DEST/$side/e16-summary ] && { echo "$DEST/$side/e16-summary exists"; exit 1; }; done
[ -e $DEST/analysis ] && { echo "$DEST/analysis exists"; exit 1; }
for side in A B; do
  d=$SC/$side; mkdir -p $d
  ln -s $WIN/gpu_clocks.csv $d/gpu_clocks.csv
  for k in 1 2 3; do
    mkdir -p $d/run$k
    cp -p $WIN/$side/e16-attention/run$k/e16_attention.csv $d/run$k/
    cp -p $WIN/$side/e16-baselines/run$k/e16_baselines.csv $d/run$k/
    cmp -s $WIN/$side/e16-attention/run$k/e16_attention.csv $d/run$k/e16_attention.csv && cmp -s $WIN/$side/e16-baselines/run$k/e16_baselines.csv $d/run$k/e16_baselines.csv \
      || { echo "$side run$k: a copy differs"; exit 1; }
  done
  python3 -I $SCR/e16_join_clock.py $d > $d/join.log 2>&1
  python3 -I $SCR/e16_summary.py $d > $d/summary.log 2>&1
  mkdir -p $DEST/$side/e16-summary
  (cd $d && find . -type f | sort) | while read f; do mkdir -p $DEST/$side/e16-summary/$(dirname $f); cp -p $d/$f $DEST/$side/e16-summary/$f; done
  echo "$side: $(ls $DEST/$side/e16-summary | tr '\n' ' ')"
done
mkdir -p $DEST/analysis
python3 -I $HERE/rt2_report.py $OLD $DEST/A/e16-summary $DEST/B/e16-summary --out $DEST/analysis/retime-report.md --csv $DEST/analysis/retime-numbers.csv
for side in A B; do diff $OLD/e16_text_numbers.txt $DEST/$side/e16-summary/e16_text_numbers.txt > $DEST/analysis/text-numbers-old-vs-$side.diff || true; done
diff $DEST/A/e16-summary/e16_text_numbers.txt $DEST/B/e16-summary/e16_text_numbers.txt > $DEST/analysis/text-numbers-A-vs-B.diff || true
echo "rt2_summaries: done"
