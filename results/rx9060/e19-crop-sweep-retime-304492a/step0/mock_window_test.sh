#!/usr/bin/env bash
# mock_window_test.sh: the two window scripts run against stand-in binaries, with rocminfo and rocprofv3 replaced by stubs, so that
# nothing opens the GPU. The stand-ins are shell scripts named e19_crop_sweep_hip and e18_box_hip that carry the commit strings, log each call
# (which binary, E19_CROPS, the run folder, the number of E19_/E18_ variables), write the run's two CSVs and print the checks line. Checks:
#   1. both windows run to the end, with the runs in the order and number of the protocol (E19: 10 sweeps of 16, E18: 10 runs);
#   2. a failing run (MOCK_FAIL=B-40) stops the E19 window there, and nothing after it runs;
#   3. the guards refuse an existing output folder, a wrong sha256 and a wrong fkl-benchmarks commit, and create nothing;
#   4. a process in the GPU-process list (here a copy of window.sh whose KFD folder is a test folder with one entry) blocks the first run, and
#      the window stops after IDLE_MAX = 15 s.
set -u
HERE=$(cd "$(dirname "$0")/.." && pwd); E18=$(cd "$HERE/../e18-box-retime-304492a" && pwd)
M=$(mktemp -d -p /home/cudeiro/rocm-fkl-benchmarks/tmp/sys); trap 'rm -rf "$M"' EXIT
FA=b10fc6609426bd18020b9213c7196b528fc677a8; FB=304492af4d777165bd556fdd6113439635724d02; RB=73a9650ea1a65daa92af97641d3bf0719682101e
stand_in() {   # <dir> <name> <A or B> <checks line> <crops variable>
  mkdir -p $1; { echo '#!/usr/bin/env bash'; echo ": <<'STRINGS'"; [ $3 = A ] && echo $FA || printf '%s\n%s\n' $FB $RB; echo STRINGS
    echo "echo \"$3 \${$5:-unset} \$(basename \$(pwd)) \$(env | grep -c -E '^E1[89]_')\" >> $M/calls.log"
    echo "[ \"\${MOCK_FAIL:-}\" = \"$3-\${$5:-}\" ] && exit 3"
    echo 'echo x > run_GPU_AMD.csv; echo x > run_checks_GPU_AMD.csv'; echo "echo \"$4\""; } > $1/$2; chmod +x $1/$2; }
printf '#!/usr/bin/env bash\nwhile [ $# -gt 0 ] && [ "$1" != "--" ]; do shift; done; shift; exec "$@"\n' > $M/rocprof; chmod +x $M/rocprof
for w in A B; do stand_in $M/19$w e19_crop_sweep_hip $w "E19 checks passed" E19_CROPS; stand_in $M/18$w e18_box_hip $w "E18 checks passed" E18_ONLY; done
cp $HERE/window.sh $M/w19.sh; cp $E18/window.sh $M/w18.sh
sed 's|^KFD=/sys/class/kfd/kfd/proc$|KFD='$M'/kfd|' $HERE/window.sh > $M/w19kfd.sh; mkdir -p $M/kfd/4242
(cd $M && sha256sum $M/19A/e19_crop_sweep_hip $M/19B/e19_crop_sweep_hip w19.sh w19kfd.sh > m19 && sha256sum $M/18A/e18_box_hip $M/18B/e18_box_hip w18.sh > m18)
s() { sha256sum $1 | cut -c1-64; }
w19() {   # <script> <out> [VAR=value ...]
  local sc=$1 out=$2; shift 2
  env "$@" ROCMINFO=/bin/true ROCPROF=$M/rocprof MANIFEST=$M/m19 BIN_A=$M/19A/e19_crop_sweep_hip SHA_A=$(s $M/19A/e19_crop_sweep_hip) FKL_A=$FA \
    BIN_B=$M/19B/e19_crop_sweep_hip SHA_B=${SHA_B_OVERRIDE:-$(s $M/19B/e19_crop_sweep_hip)} FKL_B=$FB REPO_B=${REPO_B_OVERRIDE:-$RB} bash $M/$sc $out
}
w19 w19.sh $M/o1 > $M/t1.txt 2>&1; e=$?
echo "1. E19 window: exit $e; runs $(wc -l < $M/calls.log); sweeps: $(grep 'sweep .* starts' $M/o1/window-console.txt | awk '{print $4}' | tr '\n' ' ')"
echo "   binaries in order: $(awk '{print $1}' $M/calls.log | uniq -c | awk '{printf "%s%s ", $1, $2}'); runs with one E19_ variable: $(awk '$4 == 1' $M/calls.log | wc -l)"
echo "   N of the first sweep: $(head -16 $M/calls.log | awk '{print $2}' | tr '\n' ' ')"
rm -f $M/calls.log
env ROCMINFO=/bin/true ROCPROF=$M/rocprof MANIFEST=$M/m18 BIN_A=$M/18A/e18_box_hip SHA_A=$(s $M/18A/e18_box_hip) FKL_A=$FA BIN_B=$M/18B/e18_box_hip \
  SHA_B=$(s $M/18B/e18_box_hip) FKL_B=$FB REPO_B=$RB bash $M/w18.sh $M/o18 > $M/t18.txt 2>&1; e=$?
echo "   E18 window: exit $e; runs: $(awk '{printf "%s:%s ", $1, $3}' $M/calls.log); E18_ variables set: $(awk '{s += $4} END {print s}' $M/calls.log)"
rm -f $M/calls.log
w19 w19.sh $M/o2 MOCK_FAIL=B-40 > $M/t2.txt 2>&1; e=$?
echo "2. MOCK_FAIL=B-40: exit $e; runs $(wc -l < $M/calls.log); last line: $(tail -1 $M/o2/window-console.txt | cut -d' ' -f3-)"
rm -f $M/calls.log
echo "3. existing folder: $(w19 w19.sh $M/o1 2>&1 | tail -1 | sed "s|$M|M|g")"
echo "   wrong sha256 of B: $(SHA_B_OVERRIDE=$(s $M/19A/e19_crop_sweep_hip) w19 w19.sh $M/o3 2>&1 | tail -1 | sed "s|$M|M|g"); created: $(ls -d $M/o3 2> /dev/null | wc -l)"
echo "   wrong fkl-benchmarks commit: $(REPO_B_OVERRIDE=976e42b000000000000000000000000000000000 w19 w19.sh $M/o4 2>&1 | tail -1 | sed "s|$M|M|g"); created: $(ls -d $M/o4 2> /dev/null | wc -l)"
w19 w19kfd.sh $M/o5 IDLE_MAX=15 > $M/t5.txt 2>&1; e=$?
echo "4. a GPU process listed: exit $e; runs $(cat $M/calls.log 2> /dev/null | wc -l); last line: $(tail -1 $M/o5/window-console.txt | cut -d' ' -f3- | sed "s|loadavg [0-9. ]*|loadavg ...|")"
