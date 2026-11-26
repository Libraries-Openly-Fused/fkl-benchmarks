#!/usr/bin/env bash
# launch.sh [GUARD_ONLY=1]: the one GPU window of RT-A1 and RT-A2 (Oscar's re-times of 2026-11-26), run as one detached call once the
# coordinator has granted it. RT-A1 (the E19 crop sweep, window.sh of this folder) runs first; RT-A2 (E18, ../e18-box-retime-304492a/window.sh)
# runs after it only if RT-A1 ended with exit 0. Each writes its own window/ folder, which must not exist yet. GUARD_ONLY=1 runs the guards
# of both scripts only: nothing is created and nothing runs on the GPU.
set -u
HERE=$(cd "$(dirname "$0")" && pwd); E18=$(cd "$HERE/../e18-box-retime-304492a" && pwd)
V=/home/cudeiro/rocm-fkl-benchmarks/tmp/vbuild
unset ROCMINFO ROCPROF
FKL_A=b10fc6609426bd18020b9213c7196b528fc677a8; FKL_B=304492af4d777165bd556fdd6113439635724d02; REPO_B=73a9650ea1a65daa92af97641d3bf0719682101e
MANIFEST=$HERE/manifest.sha256 FKL_A=$FKL_A FKL_B=$FKL_B REPO_B=$REPO_B \
  BIN_A=$V/pin435-base-hip/bin/e19_crop_sweep_hip SHA_A=9e7a4e0f758ea70365ed6382c1b16f0e9fcf08475972a6d9e7eba86cc82ddfa7 \
  BIN_B=$V/rt-73a9650-hip/bin/e19_crop_sweep_hip SHA_B=842ca53052ef19e0ae99eec01f405f26ddb590a226cdb4e384c4b8932ed2c5e9 \
  bash "$HERE/window.sh" "$HERE/window"
e19=$?
echo "# RT-A1 (e19-crop-sweep-retime-304492a) exit $e19"
[ $e19 = 0 ] || { echo "# RT-A2 does not run"; exit $e19; }
MANIFEST=$E18/manifest.sha256 FKL_A=$FKL_A FKL_B=$FKL_B REPO_B=$REPO_B \
  BIN_A=$V/pin435-base-hip/bin/e18_box_hip SHA_A=f32f98e2f84b22fc599bdbe89cabcc957c3ed6df94948e2bf93a053bb6d584ff \
  BIN_B=$V/rt-73a9650-hip/bin/e18_box_hip SHA_B=93dd323e7debcf7d7d8a56d12987b00d28c8936714b4f5c6798e50c774864097 \
  bash "$E18/window.sh" "$E18/window"
e18=$?
echo "# RT-A2 (e18-box-retime-304492a) exit $e18"
exit $e18
