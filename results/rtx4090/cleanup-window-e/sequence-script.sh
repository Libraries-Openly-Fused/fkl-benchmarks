#!/bin/bash
# The cleanup window E in one sequence, each step only if the one before ended normally: step 0 (compile only, cleanup-window-e-build.sh), 120 s of idle so that
# nothing of the build remains, window E (gpu-window-cleanup-e.sh, which makes its own checks and refuses to start if one fails).
S=/home/oscar/tfkl-bench/scratch
echo "# $(date -Iseconds) sequence starts"
bash $S/cleanup-window-e-build.sh || { echo "# $(date -Iseconds) step 0 stopped: no window"; exit 1; }
START_DELAY=120 bash $S/gpu-window-cleanup-e.sh || { echo "# $(date -Iseconds) window E refused or failed"; exit 1; }
echo "# $(date -Iseconds) sequence ended"
