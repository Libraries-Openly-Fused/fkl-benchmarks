#!/bin/bash
# The cleanup windows A and B in one sequence, each step only if the one before ended normally:
#   step 0 (compile only, cleanup-window-build.sh), 120 s of idle so that nothing of the build remains,
#   window A (gpu-window-cleanup-a.sh), window B (gpu-window-cleanup-b.sh).
# Each window script makes its own checks (frozen programs, step 0's record, quiet machine, idle GPU, new output folder) and refuses to start
# if one fails.
S=/home/oscar/tfkl-bench/scratch
echo "# $(date -Iseconds) sequence starts"
bash $S/cleanup-window-build.sh || { echo "# $(date -Iseconds) step 0 stopped: no window"; exit 1; }
START_DELAY=120 bash $S/gpu-window-cleanup-a.sh || { echo "# $(date -Iseconds) window A refused or failed: window B not started"; exit 1; }
bash $S/gpu-window-cleanup-b.sh || { echo "# $(date -Iseconds) window B refused or failed"; exit 1; }
echo "# $(date -Iseconds) sequence ended"
