#!/bin/bash
# RT-2, after the window (CPU only: reads the window's CSVs; nothing is run): rt2_summaries.sh, i.e. each side's e16-summary in the paper's
# layout by scripts/e16_join_clock.py and scripts/e16_summary.py (unchanged), and rt2_report.py against window D's cleaned e16-summary.
set -eu
REC=$(readlink -f $(dirname $0)/..); RES=$(dirname $REC); REPO=$(dirname $(dirname $RES))
[ -d $REC/window ] || { echo "window/ missing"; exit 1; }
bash $REC/scripts-used/rt2_summaries.sh $REC/window $REPO/scripts /home/oscar/tfkl-bench/scratch/rt2/summary-work $REC $RES/cleanup-window-d/cleaned/e16-summary
