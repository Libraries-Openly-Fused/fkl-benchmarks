#!/bin/bash
# RT-3, after the window (CPU only: reads the window's CSVs; nothing is compiled or run). The repeat's own scripts, unchanged:
#   1. scripts/e3_repeat_table.py window/      -> window/e3_repeat_summary.csv / .md, window/e3_table_chains.csv (the repeat's tables)
#   2. scripts/e3_repeat_pair.py ../repeat-pr423-9e20bd4e window --old-label 9e20bd4e --new-label 304492a --out window/e3_repeat_vs_9e20bd4e
#      (old / new / the values in the order they ran / spread / change / exceeds / the 5 % flag of a host-bound compile, per length and quantity)
#   3. the sha256 of the eight translation units against the repeat's tu-sha256.txt (by length) -> window/tu-sha256-vs-9e20bd4e.txt
set -eu
REC=$(readlink -f $(dirname $0)/..); E3=$(dirname $REC); REPO=$(dirname $(dirname $(dirname $E3))); WIN=$REC/window; OLD=$E3/repeat-pr423-9e20bd4e
[ -s $WIN/e3_repeat_A.csv ] || { echo "window/e3_repeat_A.csv missing"; exit 1; }
for f in e3_repeat_summary.csv e3_repeat_vs_9e20bd4e.csv tu-sha256-vs-9e20bd4e.txt; do [ -e $WIN/$f ] && { echo "$WIN/$f exists"; exit 1; }; done
cd $REPO
python3 -I scripts/e3_repeat_table.py $WIN > $WIN/e3_repeat_table-stdout.txt
python3 -I scripts/e3_repeat_pair.py $OLD $WIN --old-label 9e20bd4e --new-label 304492a --out $WIN/e3_repeat_vs_9e20bd4e > $WIN/e3_repeat_pair-stdout.txt
python3 -I - $OLD/tu-sha256.txt $WIN/tu-sha256.txt > $WIN/tu-sha256-vs-9e20bd4e.txt <<'PY'
import re, sys
def read(p):
    out = {}
    for l in open(p):
        m = re.match(r"([0-9a-f]{64})\s+\S*_A_(\d+)\.cu$", l.strip())
        if m: out[int(m.group(2))] = m.group(1)
    return out
o, n = read(sys.argv[1]), read(sys.argv[2])
print("the translation units by length N: sha256 at 9e20bd4e (the repeat) and here")
for k in sorted(set(o) | set(n)):
    print(f"N = {k}: {'equal' if o.get(k) == n.get(k) else 'DIFFERENT'} ({o.get(k, '-')[:16]} / {n.get(k, '-')[:16]})")
PY
cat $WIN/tu-sha256-vs-9e20bd4e.txt
echo "after_window: done"
