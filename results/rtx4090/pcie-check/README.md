# PCIe link check of the RTX 4090 (2026-11-01 19:18:30 to 19:18:57)

Question (coordinator, from Oscar): does the link of the RTX 4090 (0000:09:00.0, root port 0000:00:03.1, both x16, maximum 16 GT/s = PCIe 4.0) run at generation 4 and width 16 while the benchmarks are timed? The sysfs reading outside a run shows 2.5 GT/s, which is the idle power-saving state.
None of the earlier windows (E0, E4, E16, host-time check, graph probes) logged the link, so this check was made with the same machine, driver and pinning, in a quiet window of 27 seconds (`scripts/pcie_check_summary.py` reads the files here).

Method: `sampler.csv` = nvidia-smi every 100 ms (timestamp, pstate, pcie.link.gen.current and max, pcie.link.width.current and max, SM clock, utilisation, power; core 14); `phases.txt` = epoch times of the phases: idle 5 s, load (the E16 program, mma_raw, D = 128, S = 4096, non-causal: `load_stdout.txt`, `e16_attention.csv`), copies (`pcie_bandwidth_cu`, `fkl-self/pcie/pcie_bandwidth.cu`: 256 MB, host to device and device to host,
from pinned and from pageable memory, `pcie_bandwidth.csv`), idle 3 s. Pinned to cores 0-3,16-19. Binaries and their sha256: `binary-sha256.txt`; `window-run.txt` is the output of the window script: its first line is the start line of the window, then the OS and kernel lines, the exit codes, the bandwidth lines, and the last line is the end marker. Disclosure: during the copy phase, at about 19:18:50 (the command printed 19:18:53), a read-only `date`, `ls`, `pgrep` and one `nvidia-smi` query ran in the session that launched the window (to answer a message). They are not in `sampler.csv` (a separate process writes it), and the sampler shows no extra line and no delay from them: its intervals between 19:18:48 and 19:18:56 are 0.100 to 0.101 s, the largest interval of the whole file is 0.110 s (19:18:43.7, the change from the load phase to the copy phase). The copy bandwidths come from CUDA events around each copy, not from the sampler.

## Result

```
PCIe link check, 265 samples of results/rtx4090/pcie-check/sampler.csv; link maximum seen: generation 4, width 16

idle     5.0 s,  50 samples: pstate P8 (50); link generation current 1 (50) (max 4); link width current 16 (50) (max 16); SM clock median 210 MHz; utilisation median 1 %; power median 29 W
load     7.1 s,  71 samples: pstate P2 (66), P8 (5); link generation current 4 (66), 1 (5) (max 4); link width current 16 (71) (max 16); SM clock median 2580 MHz; utilisation median 100 %; power median 449 W
copy    10.5 s, 104 samples: pstate P2 (104); link generation current 4 (104) (max 4); link width current 16 (104) (max 16); SM clock median 2685 MHz; utilisation median 100 %; power median 84 W
idle2    3.0 s,  30 samples: pstate P0 (15), P5 (10), P2 (5); link generation current 4 (20), 2 (10) (max 4); link width current 16 (30) (max 16); SM clock median 2685 MHz; utilisation median 0 %; power median 79 W

copies H2D pinned  :  286 copies of 256 MB in 3.0 s, median 25.59 GB/s (smallest 24.92, largest 25.65)
copies D2H pinned  :  277 copies of 256 MB in 3.0 s, median 24.79 GB/s (smallest 24.53, largest 24.81)
copies H2D pageable:   77 copies of 256 MB in 1.5 s, median 13.81 GB/s (smallest 12.99, largest 13.97)
copies D2H pageable:   70 copies of 256 MB in 1.5 s, median 12.65 GB/s (smallest 8.62, largest 12.73)
```

Reading, in the standing form (a measurement that was not logged before; no old value): the link of the RTX 4090 is **generation 4, width 16** whenever the GPU is out of the idle state: 104 of 104 samples of the copy phase (P2) and 66 of 71 of the load phase (P2; the other 5 samples are the first half second before the clock ramps, still P8 with generation 1), and the pinned copies reach
25.6 GB/s (host to device) and 24.8 GB/s (device to host), what a PCIe 4.0 x16 link gives (a generation 3 x1 link would give about 0.9 GB/s). At idle (P8) the generation is 1 (2.5 GT/s) and the width 16, as the sysfs reading showed; after the load it steps down through generation 4 and 2 within a second. Spread: pinned copies 24.9 to 25.7 and 24.5 to 24.8 GB/s over 286 and 277 copies.
Bound: link state and copy bandwidth are device and link numbers (GPU-bound for the load; the copies are link-bound). What it does not show: the link state during the earlier windows themselves; indirect evidence is the pstate column of their samplers (the E16 4090 window: P2 in 9,381 of 9,510 samples, P0 80, P3 15, P8 34), and the link follows the pstate in this check.
