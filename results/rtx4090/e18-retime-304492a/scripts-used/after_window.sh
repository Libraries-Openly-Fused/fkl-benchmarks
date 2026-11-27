#!/bin/bash
# RT-4, after the window (CPU only: reads the window's files; nothing is run). From the record folder's parent (results/rtx4090):
#   1. rt4_report.py: old (#63's B) / A / B, spreads, changes, exceeds, overlap, bound, flags; the clocks per run; the check CSVs against
#      #63's B quiet-run1-B -> analysis/;
#   2. first_row_clock.py (#63's): the SM clock at the start of each run's timing -> analysis/first-row-clock.txt;
#   3. rt4_formats.sh: A's and B's outputs in the three E18 records' formats, by the records' own scripts -> formats/{A,B}/<record>/.
set -eu
REC=$(readlink -f $(dirname $0)/..); RES=$(dirname $REC); SU=$REC/scripts-used
[ -d $REC/window ] && [ ! -e $REC/analysis ] && [ ! -e $REC/formats ] || { echo "window/ missing, or analysis/ or formats/ exists"; exit 1; }
mkdir -p $REC/analysis
python3 -I $SU/rt4_report.py $REC/window $RES --out $REC/analysis/retime-report.txt --csv $REC/analysis/retime-summary.csv \
  --clocks $REC/analysis/clock-by-run.txt --checks $REC/analysis/checks-vs-old.txt --md $REC/analysis/retime-table.md > /dev/null
python3 -I $SU/first_row_clock.py $REC/window --out $REC/analysis/first-row-clock.txt > /dev/null
bash $SU/rt4_formats.sh $REC/window $RES /home/oscar/tfkl-bench/scratch/rt4/formats-work $REC/formats
echo "after_window: done; $(grep -c . $REC/analysis/retime-summary.csv) summary lines; $(tail -1 $REC/analysis/retime-report.txt | cut -c1-200)"
