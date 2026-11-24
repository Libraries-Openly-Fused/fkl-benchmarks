#!/usr/bin/env bash
# launch.sh [GUARD_ONLY=1]: the E18 re-time window, with the binaries and commits of the brief. One call; the output folder is window-run.
H=/home/cudeiro/rocm-fkl-benchmarks/tmp/e18retime
exec env -u ROCMINFO -u ROCPROF MANIFEST=$H/manifest.sha256 \
  BIN_A=$H/bins/A/e18_box_hip SHA_A=987b8a2ec372fec0bc483cb56c78ced73a7c0a4e62f7c9e83cddbd1c6b165bc7 FKL_A=24478f0aa11dc3f12feea83178f7fde828322fb0 \
  BIN_B=$H/bins/B/e18_box_hip SHA_B=3007b72e92eae1783ee6c4d5478066a31fa1413a5ccc3bee72cd88af4729e1f2 FKL_B=b10fc6609426bd18020b9213c7196b528fc677a8 \
  IDLE_MAX=600 "$@" bash $H/window.sh $H/window-run
