#!/bin/bash
# RT-4, after the window (CPU only: reads CSVs; nothing is run): the outputs of A and of B in the three E18 records' own formats, by the
# records' own report scripts, unchanged:
#   ../e18-box/scripts-used/e18_box_report.py              -> e18_box_summary.csv, e18_box_speedups.csv, e18_box_kernel_vs_whole.csv, e18_box_report.md
#   ../e18-box-split/scripts-used/e18_box_split_report.py  -> e18_box_split_summary.csv, _ratios.csv, _kernel_vs_whole.csv, _repeats.csv, _report.md
#                                                             (--old: the same side's e18_box_summary.csv, as the record took the window e18-box's)
#   ../e18-stencil/scripts-used/e18_report.py              -> e18_summary.csv, e18_speedups.csv, e18_report.md
# The scripts read <folder>/quiet-run1..3/<program>.csv and write into <folder>. For each side, a scratch folder per record holds links
# quiet-run<k> -> <window>/<program>/quiet-run<k>-<side>; the scripts run there and their outputs are copied to <dest>/<side>/<record>/.
# Usage: rt4_formats.sh <window folder> <results/rtx4090 folder (the records' scripts)> <scratch folder, new> <dest folder, new>
set -eu
WIN=$(readlink -f $1); RECS=$(readlink -f $2); SC=$3; DEST=$4
[ -e $SC ] && { echo "$SC exists"; exit 1; }
[ -e $DEST ] && { echo "$DEST exists"; exit 1; }
declare -A PROG=([e18-box]=e18_box [e18-box-split]=e18_box_split [e18-stencil]=e18_stencil)
for side in A B; do
  for rec in e18-box e18-box-split e18-stencil; do
    d=$SC/$side/$rec; mkdir -p $d
    for k in 1 2 3; do
      src=$WIN/${PROG[$rec]}/quiet-run$k-$side
      [ -s $src/${PROG[$rec]}.csv ] || { echo "$src/${PROG[$rec]}.csv is missing"; exit 1; }
      ln -s $src $d/quiet-run$k
    done
  done
  python3 -I $RECS/e18-box/scripts-used/e18_box_report.py $SC/$side/e18-box > $SC/$side/e18-box-stdout.txt
  python3 -I $RECS/e18-box-split/scripts-used/e18_box_split_report.py $SC/$side/e18-box-split --old $SC/$side/e18-box/e18_box_summary.csv > $SC/$side/e18-box-split-stdout.txt
  python3 -I $RECS/e18-stencil/scripts-used/e18_report.py $SC/$side/e18-stencil > $SC/$side/e18-stencil-stdout.txt
  for rec in e18-box e18-box-split e18-stencil; do
    mkdir -p $DEST/$side/$rec
    find $SC/$side/$rec -maxdepth 1 -type f -exec cp {} $DEST/$side/$rec/ \;
    echo "$side $rec: $(ls $DEST/$side/$rec | tr '\n' ' ')"
  done
done
