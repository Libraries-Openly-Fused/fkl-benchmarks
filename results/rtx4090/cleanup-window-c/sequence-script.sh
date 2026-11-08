#!/bin/bash
# The cleanup window C in one sequence, each step only if the one before ended normally: step 0 (compile only, cleanup-window-c-build.sh), 120 s of idle so that
# nothing of the build remains, window C (gpu-window-cleanup-c.sh, which makes its own checks and refuses to start if one fails).
S=/home/oscar/tfkl-bench/scratch
echo "# $(date -Iseconds) sequence starts"
bash $S/cleanup-window-c-build.sh || { echo "# $(date -Iseconds) step 0 stopped: no window"; exit 1; }
START_DELAY=120 bash $S/gpu-window-cleanup-c.sh || { echo "# $(date -Iseconds) window C refused or failed"; exit 1; }
echo "# $(date -Iseconds) sequence ended"
