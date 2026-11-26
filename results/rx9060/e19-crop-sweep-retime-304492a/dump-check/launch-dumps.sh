#!/usr/bin/env bash
# launch-dumps.sh [GUARD_ONLY=1]: the dump-check GPU window of RT-A1, run as one detached call once the coordinator has granted it. It runs
# window-dumps.sh of this folder with the binaries of ../launch.sh, writing window/ here and the dumps to
# /home/cudeiro/rocm-fkl-benchmarks/tmp/rt/dumps-304492a (outside the repository); both must not exist yet. GUARD_ONLY=1 runs the guards only.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
V=/home/cudeiro/rocm-fkl-benchmarks/tmp/vbuild
unset ROCMINFO
MANIFEST=$HERE/manifest.sha256 FKL_A=b10fc6609426bd18020b9213c7196b528fc677a8 FKL_B=304492af4d777165bd556fdd6113439635724d02 \
  REPO_B=73a9650ea1a65daa92af97641d3bf0719682101e \
  BIN_A=$V/pin435-base-hip/bin/e19_crop_sweep_hip SHA_A=9e7a4e0f758ea70365ed6382c1b16f0e9fcf08475972a6d9e7eba86cc82ddfa7 \
  BIN_B=$V/rt-73a9650-hip/bin/e19_crop_sweep_hip SHA_B=842ca53052ef19e0ae99eec01f405f26ddb590a226cdb4e384c4b8932ed2c5e9 \
  bash "$HERE/window-dumps.sh" "$HERE/window" /home/cudeiro/rocm-fkl-benchmarks/tmp/rt/dumps-304492a
e=$?
echo "# dump check (window-dumps.sh) exit $e"
exit $e
