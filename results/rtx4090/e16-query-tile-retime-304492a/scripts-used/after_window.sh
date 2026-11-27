#!/bin/bash
# RT-2b, after the window (CPU only: reads the window's CSVs; nothing is run): rt2b_after.sh, i.e. each side in the record's layout with
# scripts/e16_query_tile_analyse.py (unchanged), and rt2b_report.py against the record results/rtx4090/e16-query-tile/.
set -eu
REC=$(readlink -f $(dirname $0)/..); RES=$(dirname $REC); REPO=$(dirname $(dirname $RES))
[ -d $REC/window ] || { echo "window/ missing"; exit 1; }
bash $REC/scripts-used/rt2b_after.sh $REC/window $REPO/scripts $REC $RES/e16-query-tile
