#!/bin/bash
# RT-2b, after the window (CPU only: reads the window's CSVs; nothing is run).
#   1. Each side in the record's layout: <dest>/<side>/run0..2/e16_query_tile.csv (copies of <window>/<side>/run0..2/, checked byte for
#      byte) and scripts/e16_query_tile_analyse.py (unchanged) on that folder: e16_query_tile_derived.csv and .md, the record's format.
#   2. rt2b_report.py: old (the record) / A / B -> <dest>/analysis/.
# Usage: rt2b_after.sh <window folder> <scripts folder of the repository> <dest folder (the record)> <old record folder>
set -eu
WIN=$(readlink -f $1); SCR=$(readlink -f $2); DEST=$(readlink -f $3); OLD=$(readlink -f $4); HERE=$(dirname $(readlink -f $0))
for side in A B; do [ -e $DEST/$side ] && { echo "$DEST/$side exists"; exit 1; }; done
[ -e $DEST/analysis ] && { echo "$DEST/analysis exists"; exit 1; }
for side in A B; do
  for k in 0 1 2; do
    mkdir -p $DEST/$side/run$k
    cp -p $WIN/$side/run$k/e16_query_tile.csv $DEST/$side/run$k/
    cmp -s $WIN/$side/run$k/e16_query_tile.csv $DEST/$side/run$k/e16_query_tile.csv || { echo "$side run$k: the copy differs"; exit 1; }
  done
  python3 -I $SCR/e16_query_tile_analyse.py $DEST/$side > $DEST/$side/analyse-stdout.txt
  echo "$side: $(ls $DEST/$side | tr '\n' ' ')"
done
mkdir -p $DEST/analysis
python3 -I $HERE/rt2b_report.py $OLD $DEST/A $DEST/B --out $DEST/analysis/retime-report.md --csv $DEST/analysis/retime-numbers.csv
echo "rt2b_after: done"
