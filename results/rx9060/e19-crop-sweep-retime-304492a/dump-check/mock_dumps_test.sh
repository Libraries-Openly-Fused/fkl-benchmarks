#!/usr/bin/env bash
# mock_dumps_test.sh: window-dumps.sh against stand-in binaries named e19_crop_sweep_hip (rocminfo a stub), so that nothing opens the GPU. A
# stand-in logs each call (which binary, E19_CROPS, E19_NO_TIMING, whether E19_DUMP is set, the number of E19_ variables), writes the five
# tensors into E19_DUMP (or four with MOCK_SHORT=<A or B>-<N>) and its checks CSV, and prints the checks line. Checks:
#   1. 32 runs, A's 16 N then B's, each with E19_CROPS, E19_NO_TIMING=1 and E19_DUMP only, the dumps in <dump dir>/<A or B>/nNNN;
#   2. a run that writes four tensors stops the window there;
#   3. an existing dump folder is refused, and nothing is created;
#   4. a process in the GPU-process list (a copy of the script whose KFD folder is a test folder with one entry) blocks the first run.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
M=$(mktemp -d -p /home/cudeiro/rocm-fkl-benchmarks/tmp/sys); trap 'rm -rf "$M"' EXIT
FA=b10fc6609426bd18020b9213c7196b528fc677a8; FB=304492af4d777165bd556fdd6113439635724d02; RB=73a9650ea1a65daa92af97641d3bf0719682101e
for w in A B; do mkdir -p $M/$w; { echo '#!/usr/bin/env bash'; echo ": <<'STRINGS'"; [ $w = A ] && echo $FA || printf '%s\n%s\n' $FB $RB; echo STRINGS
  echo "echo \"$w \${E19_CROPS:-unset} \${E19_NO_TIMING:-unset} \${E19_DUMP:+dump} \$(env | grep -c '^E19_')\" >> $M/calls.log"
  echo 'for f in rpp_grey.u8 rpp_box.u8 rpp_tensor.f32 fkl_loop_tensor.f32 fkl_batch_tensor.f32; do'
  echo "  [ \"\${MOCK_SHORT:-}\" = \"$w-\$E19_CROPS\" ] && [ \$f = fkl_batch_tensor.f32 ] && continue; echo x > \"\$E19_DUMP/\$f\"; done"
  echo 'echo x > e19_batched_box_checks_GPU_AMD.csv; echo "E19 checks passed"'; } > $M/$w/e19_crop_sweep_hip; chmod +x $M/$w/e19_crop_sweep_hip; done
cp $HERE/window-dumps.sh $M/w.sh; sed 's|^KFD=/sys/class/kfd/kfd/proc$|KFD='$M'/kfd|' $HERE/window-dumps.sh > $M/wk.sh; mkdir -p $M/kfd/4242
(cd $M && sha256sum $M/A/e19_crop_sweep_hip $M/B/e19_crop_sweep_hip w.sh wk.sh > m)
s() { sha256sum $1 | cut -c1-64; }
w() {   # <script> <out> <dumps> [VAR=value ...]
  local sc=$1 out=$2 dumps=$3; shift 3
  env "$@" ROCMINFO=/bin/true MANIFEST=$M/m BIN_A=$M/A/e19_crop_sweep_hip SHA_A=$(s $M/A/e19_crop_sweep_hip) FKL_A=$FA BIN_B=$M/B/e19_crop_sweep_hip \
    SHA_B=$(s $M/B/e19_crop_sweep_hip) FKL_B=$FB REPO_B=$RB bash $M/$sc $out $dumps
}
w w.sh $M/o1 $M/d1 > $M/t1.txt 2>&1; e=$?
echo "1. exit $e; runs $(wc -l < $M/calls.log); binaries in order: $(awk '{print $1}' $M/calls.log | uniq -c | awk '{printf "%s%s ", $1, $2}')"
echo "   N of A: $(awk '$1 == "A" {print $2}' $M/calls.log | tr '\n' ' ')"
echo "   runs with E19_NO_TIMING=1, E19_DUMP set and 3 E19_ variables: $(awk '$3 == 1 && $4 == "dump" && $5 == 3' $M/calls.log | wc -l)"
echo "   dump folders: A $(ls $M/d1/A | wc -l), B $(ls $M/d1/B | wc -l); files $(find $M/d1 -type f | wc -l)"
rm -f $M/calls.log
w w.sh $M/o2 $M/d2 MOCK_SHORT=B-40 > $M/t2.txt 2>&1; e=$?
echo "2. MOCK_SHORT=B-40: exit $e; runs $(wc -l < $M/calls.log); last line: $(tail -1 $M/o2/window-console.txt | cut -d' ' -f3-)"
rm -f $M/calls.log
echo "3. existing dump folder: $(w w.sh $M/o3 $M/d1 2>&1 | tail -1 | sed "s|$M|M|g"); created: $(ls -d $M/o3 2> /dev/null | wc -l)"
w wk.sh $M/o4 $M/d4 IDLE_MAX=15 > $M/t4.txt 2>&1; e=$?
echo "4. a GPU process listed: exit $e; runs $(cat $M/calls.log 2> /dev/null | wc -l); last line: $(tail -1 $M/o4/window-console.txt | cut -d' ' -f3- | sed 's|loadavg [0-9. ]*|loadavg ...|')"
