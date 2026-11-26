#!/usr/bin/env bash
# code_checks.sh: the compile-only checks of RT-A1 and RT-A2's binaries (code-checks/ of this folder and of ../e18-box-retime-304492a/, and
# resources/), with the scripts of ../pin-move-devcode-b10fc66-48fbf3f/step0/ (#67) and step0/e19_resources.py. Nothing runs on the GPU.
#   record19 = e19_crop_sweep_hip that ../e19-crop-sweep/ timed (c8eb575, FKL 24478f0, f9fcb2a4...);
#   record18 = e18_box_hip that ../e18-box-retime-b10fc66/ timed as B (2160c70, FKL b10fc66, 3007b72e...);
#   A = #67's base build (976e42b, FKL b10fc66); H = #67's head build (a4b493c, FKL 48fbf3f); B = step0/rt_build.sh (73a9650, FKL 304492a).
set -u
NS=/home/cudeiro/rocm-fkl-benchmarks/tmp; V=$NS/vbuild; HERE=$(cd "$(dirname "$0")/.." && pwd); E18=$(cd "$HERE/../e18-box-retime-304492a" && pwd)
S67=$(cd "$HERE/../pin-move-devcode-b10fc66-48fbf3f/step0" && pwd); SCRIPTS=$(cd "$HERE/../../../scripts" && pwd)
PY="bash $NS/scripts/safepy.sh"; export ROCM_PATH=/opt/rocm/core-10.1 TMPDIR=$NS/sys
W=$(mktemp -d -p $NS/sys); trap 'rm -rf "$W"' EXIT
mk() { mkdir -p $W/$1; ln -s $2 $W/$1/$(basename $2); }
mk rec19 $V/e19sweep-clean/bin/e19_crop_sweep_hip; mk rec18 $V/pin433-head-hip/bin/e18_box_hip
for p in e19_crop_sweep_hip e18_box_hip; do mk A $V/pin435-base-hip/bin/$p; mk H $V/pin435-head-hip/bin/$p; mk B $V/rt-73a9650-hip/bin/$p; done
mk A19 $V/pin435-base-hip/bin/e19_crop_sweep_hip; mk A18 $V/pin435-base-hip/bin/e18_box_hip
C=$HERE/code-checks; C18=$E18/code-checks; mkdir -p $C $C18 $HERE/resources
$PY $S67/devcode_compare_taskc.py $W/rec19 $W/A19 > $C/devcode-old19-vs-A.txt 2>&1
$PY $S67/devcode_compare_taskc.py $W/H $W/B > $C/devcode-head-vs-B.txt 2>&1
$PY $S67/devcode_compare_taskc.py $W/H $W/B --labels-as-symbols > $C/devcode-head-vs-B-labels.txt 2>&1
$PY $S67/devcode_compare_taskc.py $W/A $W/B > $C/devcode-A-vs-B.txt 2>&1
$PY $S67/devcode_compare_taskc.py $W/rec18 $W/A18 > $C18/devcode-62B-vs-A.txt 2>&1
ROOTS=(--root 'fkl_loop=::\{lambda\(\)#1\}>::_M_invoke\(' --root 'fkl_batch=::\{lambda\(\)#3\}>::_M_invoke\(' --root 'rpp=::\{lambda\(\)#5\}>::_M_invoke\('
       --harness '^voidfklb::timeIt<' --harness '^voidfklb::timePreparedLaunch<')
h() {   # <A binary> <B binary> <summary file> [options]
  local a=$1 b=$2 out=$3; shift 3
  $PY $S67/host_compare_taskc.py $a $b --out $W/h "$@" > /dev/null 2>&1; cp $W/h/$(basename $b).summary.txt $out; rm -rf $W/h
}
h $W/rec19/e19_crop_sweep_hip $W/A19/e19_crop_sweep_hip $C/host-record-vs-A.summary.txt --line-args "${ROOTS[@]}"
h $W/H/e19_crop_sweep_hip $W/B/e19_crop_sweep_hip $C/host-pin435head-vs-B.summary.txt --rename-taskc --line-args
h $W/rec18/e18_box_hip $W/A18/e18_box_hip $C18/host-62B-vs-A.summary.txt --line-args
h $W/A18/e18_box_hip $W/B/e18_box_hip $C18/host-A-vs-B.summary.txt --rename-taskc --line-args
h $W/H/e18_box_hip $W/B/e18_box_hip $C18/host-pin435head-vs-B.summary.txt --rename-taskc --line-args
$PY $HERE/step0/e19_resources.py $HERE/resources/e19-fkl-batch-resources.csv old=$W/rec19/e19_crop_sweep_hip A=$W/A19/e19_crop_sweep_hip \
    B=$W/B/e19_crop_sweep_hip --scripts $SCRIPTS > $HERE/resources/e19-fkl-batch-resources.md 2>&1
echo "code checks done"
