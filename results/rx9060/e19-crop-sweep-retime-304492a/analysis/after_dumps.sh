#!/usr/bin/env bash
# after_dumps.sh: the analysis of the dump check (dump-check/window/ and the dumps in tmp/rt/dumps-304492a), without the GPU:
#   1. the sha256 and size of every dump file, per binary (dump-check/dump-sha256-{A,B}.txt, the format of
#      ../e19-crop-sweep-phase1/analysis/dump-sha256.txt), and how many lines equal that file's (the record's binary's dumps);
#   2. phase 1's dump_prefix.sh on each binary's dumps (dump-check/dump-prefix-{A,B}.txt);
#   3. phase 1's between.py on each (dump-check/between-{A,B}.csv): fkl_batch's tensor against rpp's, and fkl_loop's against fkl_batch's;
#   4. the check rows of every check run against phase 1's check runs at the same N (dump-check/checks-identity.txt);
#   5. paper-data/{A,B}/e19_crop_sweep_differences.csv in the coordinator's layout:
#      gpu,crops,baseline,values,values_not_equal,max_abs_diff_output_units,max_abs_diff_grey_levels (fkl_batch against rpp; grey levels =
#      output units x 57.375).
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd); P1=$(cd "$HERE/../e19-crop-sweep-phase1" && pwd); C=$HERE/dump-check
DUMPS=/home/cudeiro/rocm-fkl-benchmarks/tmp/rt/dumps-304492a
export TMPDIR=/home/cudeiro/rocm-fkl-benchmarks/tmp/sys
for x in A B; do
  ( cd $DUMPS/$x && for f in $(find . -type f | sed 's|^\./||' | sort); do echo "$(sha256sum $f | cut -c1-64)  $(stat -c %s $f)  $f"; done ) > $C/dump-sha256-$x.txt
  echo "$x: $(wc -l < $C/dump-sha256-$x.txt) dump files; equal to the record's (phase 1) line for line: $(grep -c -x -F -f $P1/analysis/dump-sha256.txt $C/dump-sha256-$x.txt) of $(wc -l < $P1/analysis/dump-sha256.txt)"
  bash $P1/analysis/dump_prefix.sh $DUMPS/$x > $C/dump-prefix-$x.txt
  echo "$x: N whose five files are all the first N planes of N = 150's: $(grep -c -v DIFFERENT $C/dump-prefix-$x.txt) of 16"
  ( ulimit -v 4000000; python3 -I $P1/analysis/between.py $DUMPS/$x > $C/between-$x.csv )
  python3 -I - $C/between-$x.csv $HERE/paper-data/$x/e19_crop_sweep_differences.csv <<'PY'
import csv, sys
rows = [r for r in csv.DictReader(open(sys.argv[1])) if r["row"] == "fkl_batch" and r["against"] == "rpp"]
assert len(rows) == 16 and all(r["non_finite"] == "0" for r in rows), rows
with open(sys.argv[2], "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["gpu", "crops", "baseline", "values", "values_not_equal", "max_abs_diff_output_units", "max_abs_diff_grey_levels"])
    for r in sorted(rows, key=lambda r: int(r["crops"])):
        w.writerow(["rx9060", r["crops"], "rpp", r["values"], r["values_differing"], r["max_abs_diff"], r["max_abs_diff_grey_levels"]])
PY
  diff -q <(cut -d, -f1-8 $C/between-$x.csv) <(cut -d, -f1-8 $P1/analysis/between.csv) > /dev/null && echo "$x: between.csv equals phase 1's" \
    || echo "$x: between.csv DIFFERS from phase 1's"
done
{ for x in A B; do s=0; t=0
    for d in $C/window/check-$x/n*; do n=$(basename $d); t=$((t + 1))
      cmp -s <(grep -v '^#' $d/e19_batched_box_checks_GPU_AMD.csv) <(grep -v '^#' $P1/window/check/$n/e19_batched_box_checks_GPU_AMD.csv) && s=$((s + 1)) \
        || echo "$x $n: the check rows differ from phase 1's"
    done
    echo "$x: $s of $t check runs have phase 1's check rows (window/check/nNNN, the data rows)"
  done; } > $C/checks-identity.txt
cat $C/checks-identity.txt
echo "after_dumps done"
