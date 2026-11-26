#!/usr/bin/env bash
# launch.sh [GUARD_ONLY=1]: the GPU window of FKL #436 on the RX 9060, run as one detached call once the coordinator has granted it:
# window.sh of this folder with the HIP test build of the #436 head (tmp/cmain/build-hip, main's ROCm CI configuration, configured from the
# clean tree at AFTER_COMMIT) and the harness binaries of tmp/cmain/harness-bins, writing window/ here (it must not exist yet).
set -u
HERE=$(cd "$(dirname "$0")" && pwd); C=/home/cudeiro/rocm-fkl-benchmarks/tmp/cmain
TEST_BUILD=$C/build-hip TEST_MANIFEST=$HERE/manifests/test-manifest.sha256 HARNESS_MANIFEST=$C/harness-bins/manifest.sha256 \
  CONFIGURE_LOG=$C/head-hip.configure.log BEFORE_COMMIT=ebb3891bea18a152d92dfd5092e117160490e34b \
  AFTER_COMMIT=1df2738e7bd8eba1bfb1a0baf9ad215062d014ca bash "$HERE/window.sh" "$HERE/window"
