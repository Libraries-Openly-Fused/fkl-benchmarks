# Diagnostic window F2 (e4-hand-cvgs-host-diagnostic), 2026-11-08, 12:24:55-12:25:26: FKL's host time, binary or drift (RTX 4090)

**The question.** In the window e4-hand-cvgs (`../e4-hand-cvgs/`), FKL's host times were higher than in window B: fkl_fused and fkl_fused_half_pixel, host_enqueue_us and host_launch_us, a median of +5 % from batch 50 on, with 54 flags. Is that the binary or drift?

**The method.** The two binaries, in one window, alternating:
- **old** = 836a98e, window B's binary: `worktrees/fkl-benchmarks-window`, sha256 `6e64d7e9...`.
- **new** = 2dac1c8, the binary of the window e4-hand-cvgs: `worktrees/fkl-benchmarks-window-f`, sha256 `8b56ffc2...` (its step 0 is `../e4-hand-cvgs/step0/`).
- Both are recorded in `binary-sha256.txt`. They have the same kernels: SASS identical for every kernel of the old one.

RTX 4090 = nvidia-smi index 1, CUDA 13.4.3, driver 610.43.02. The desktop session kept running.

**What ran** (`window-script.sh`, `window-console.txt`), under the settings of the window e4-hand-cvgs:
1. The warm-up: the new binary, E4a batch 300 only, untimed, in `warmup/` and not used. 4 s; the GPU started at 210 MHz.
2. Three rounds, the build order flipping every round as in window B: old then new, new then old, old then new, back to back.
   - Each run: `E4_EXPERIMENTS=E4a E4_ONLY=fkl_fused,fkl_fused_half_pixel` (30 batch sizes), pinned to cores 0-3,16-19, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`.
   - Folders: `binary-836a98e/run1..3` and `binary-2dac1c8/run1..3`. 3 to 5 s each; all exit 0, with no INCOMPLETE line and no -nan.
3. Throughout the window:
   - `gpu_clocks.csv`: nvidia-smi every 100 ms on core 14, 295 samples; `gpu_clocks.err` is empty.
   - `load.log`: the load logger every 5 s on core 12.

**Checks before the start:** the frozen-program guard on both binaries; their sha256 and commit strings, without -dirty; `busy_processes` (nothing running); no compute context on the RTX 4090 from a program outside /usr; a new output folder.

**Report:** `e4-hand-cvgs-host-diagnostic-report.md`, with every number in `.csv`. `e4_host_diagnostic_report.py` wrote both and only reads the result folders. Its findings section is `e4-hand-cvgs-host-diagnostic-findings.md`. The report gives, per batch and version, in the standing form:
- old against new in this window: old, new, spread, change, exceeds or within, the flags;
- the drift columns: the old binary today against window B, and the new binary today against e4-hand-cvgs.

**Verdict: binary composition, not drift.**
- New against old in this window, batches 50 to 300 (median):
  - host_enqueue_us +4.9 % (fkl_fused) and +4.6 % (half pixel);
  - host_launch_us +5.0 % and +5.3 %;
  - gpu_mean_ms +2.0 %, which follows the host time.
- Each binary keeps its own host time: the old one is within -1.4 % of window B, and the new one within -1.7 % of e4-hand-cvgs.
- The GPU-time drift columns (+3 %) come from this window's lower SM clock (2,550 MHz in five of the six runs, against 2,685-2,700 in the full sweeps). Both binaries share that clock, so their comparison here is not affected.

**Files:**
- `binary-836a98e/run1..3`, `binary-2dac1c8/run1..3`, `warmup/`. Each holds:
  - `e4_preprocessing.csv`, `e4_preprocessing_agreement.csv`;
  - `e4_preprocessing_pairs.csv` in the new binary's folders. It has only its header, because the versions it compares did not run.
  - `stdout.txt`, `stderr.txt`, `exit_code.txt`, `load.txt` (its start and end), `gpu-at-end.csv`.
- `binary-sha256.txt`, `window-script.sh`, `window-console.txt`, `environment-before.txt`, `environment-after.txt`, `gpu-info.csv`, `gpu_clocks.csv`, `gpu_clocks.err`, `load.log`.
- `e4-hand-cvgs-host-diagnostic-report.md`, `e4-hand-cvgs-host-diagnostic-report.csv`, `e4-hand-cvgs-host-diagnostic-findings.md`, `e4_host_diagnostic_report.py`.
