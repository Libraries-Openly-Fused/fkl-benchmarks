# E19 batched box, phase 1: the re-check window e19-recheck (RTX 4090, 2026-11-18 10:54:43-10:54:44 +01:00)

`e19_batched_box_cu` at b5d99e0, the build of 10:53 (`build.log`: 0 warnings; build info b5d99e0 without -dirty), sha256 `5c1b1562…` (`binary-sha256.txt`).

The fix since the window e19-checks (`../e19-batched-box-phase1/`) adds one thing: NPP's grey rule, found from that window's dump, as NPP's 7th candidate. That rule is t = 0.587f G; t = fmaf(0.299f, R, t); t = fmaf(0.114f, B, t); (int)(t + 0.5f). No library call changed.

The window had one step (`window-script.sh`, which stops at a failed step):
- **checks/**: one checks run, `E19_NO_TIMING=1`, with `E19_DUMP`. Exit 0: "E19 batched box checks passed".
- No nsys, no ncu, no timing.

The dump (22.5 MiB) is left out of this record. It is on the measuring machine, in `scratch/e19-batched-box/dump-recheck/`.

## The checks (checks/e19_batched_box_checks.csv), against phase 1
- **Every row of phase 1 has the same numbers.** The only changes are these:
  - **NPP's new 7th grey row**, the found rule: 0 of 1,966,080 differ. It is now NPP's reference, and it passes.
  - **The phase-1 reference** (the float rule, left to right, + 0.5f, truncated) still differs at 135 values. It is now a reported row.
  - **NPP's box against the box of the reference grey:** 143 values differed in phase 1; now 0.
  - **NPP's float against the conversion of the reference box:** 0.0174295 in phase 1; now 3.485e-7, the same as against its own box.
- **opencv_cuda:**
  - grey 0 differ;
  - box floor(sum / 9): 0 differ (nearest: 873,678);
  - against the reference pipeline: 0;
  - float 1.283e-7, all finite;
  - buffers: 61 of 61 unchanged.
- **npp:**
  - grey 0 differ under the found rule;
  - box floor(sum / 9): 0 differ (nearest: 873,666);
  - against the reference pipeline: 0;
  - float 3.485e-7, all finite.
- **Between the libraries** (reported): grey 2,541 values differ, box 2,497, float at most 0.0174296.
- **The outputs** of both rows (grey, box, tensor) are byte-identical to phase 1's (`dump-comparison.txt`).

## The SM clock (`gpu_clocks.csv`, every 100 ms; `clock-readings.txt`)
- **All 8 readings** (10:54:43.700-10:54:44.407) are 210 MHz SM and 405 MHz memory: P8 first, then P2 from 10:54:44.106.
- **The reading at the end of the run** (`checks/gpu-at-end.csv`) is also 210 MHz.
- **What this does and does not show:** nvidia-smi updates its values only about every 0.5 s, and the run lasted about a second, so the readings do not show the clock while the kernels ran. The checks do not depend on it: the outputs are byte-identical to phase 1's, which read 2,550 MHz.
