#!/bin/bash
# The cleanup window D in one sequence, each step only if the one before ended normally: step 0 (compile only, cleanup-window-d-build.sh), 30 s of idle, the
# untimed pre-window steps (cleanup-window-d-pre.sh: the E16 input files with their sizes and SHA-256, the reference/cleaned error check, the k = 5002 warm-up),
# then window D (gpu-window-cleanup-d.sh after 120 s of idle; it makes its own checks and refuses to start if one fails).
S=/home/oscar/tfkl-bench/scratch
echo "# $(date -Iseconds) sequence starts"
bash $S/cleanup-window-d-build.sh || { echo "# $(date -Iseconds) step 0 stopped: no window"; exit 1; }
sleep 30
bash $S/cleanup-window-d-pre.sh || { echo "# $(date -Iseconds) the pre-window steps stopped: no window"; exit 1; }
START_DELAY=120 bash $S/gpu-window-cleanup-d.sh || { echo "# $(date -Iseconds) window D refused or failed"; exit 1; }
echo "# $(date -Iseconds) sequence ended"
