# E19 crop sweep, phase 2: the timing window's script (RTX 4090)

`window-timing.sh` is the script of the timing window e19-crop-sweep-timing. It was committed before the window, so that it can be verified on GitHub. Its record will be `results/rtx4090/e19-crop-sweep/`, and the script sits outside that folder so that the window's new-folder guard keeps working.

The script follows the protocol of the window e19-fkl-batch-timing (`results/rtx4090/e19-batched-box-fkl-batch/window-script.sh`), with one change: a sweep takes the place of a run.
- **A sweep** is 16 processes of `e19_crop_sweep_cu`, one per N, in ascending order (N = 1, 10, ..., 150). Each process is a complete run of the program, the checks and then the timing of every row, in its own folder `<sweep>/N<NNN>/`.
- **The sweeps, in order:**
  1. `warmup`: not used.
  2. `quiet-run1`, `quiet-run2`, `quiet-run3`: the times of the report.
  3. `nsys`: under the CUDA API and NVTX trace. Its times are not used.
- **Before every run** (80 in all), the script waits for an idle machine: no measurement, build or compile job, a 1-minute load below 0.5, and no other program on the RTX 4090. If that takes longer than 300 s, the window stops.
- **Every run** is pinned to cores 0-3,16-19. It must exit 0, report its checks passed, and write complete checks and timing CSVs. The window stops at the first run that fails.
- nvidia-smi records the clock every 100 ms through the window.

**The guards** are those of `results/rtx4090/e19-crop-sweep-phase1/window-checks.sh`:
- no blocked program;
- the binary's sha256 is `2b800f1d...` and it carries the commit 246b1c8, with no -dirty commit;
- HEAD is 246b1c8 or a descendant that differs from it only under results/, with no local change outside results/;
- the script runs as the copy committed here;
- the output folder is `results/rtx4090/e19-crop-sweep` and is new;
- no busy process and no foreign GPU context.

The main differences from the e19-fkl-batch-timing script:
- the per-run timeout is 300 s (it was 180 s);
- a run must also write a complete checks CSV;
- the warm-up sweep is always run (no WARMUP switch);
- the idle wait comes before every run, not only before the quiet runs.
