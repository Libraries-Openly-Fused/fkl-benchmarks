# E16, Nsight Compute pass on the attention kernels (RTX 4090, CUDA 13.4.3)

Question (Q2, the prologue pair): at D = 128, S = 2048, non-causal (BH = 128), the fused int8 attention kernel of FKL is slower than the unfused pair (the dequantisation kernel, then the bf16 attention kernel), although it moves about 4.9 times fewer computed bytes of K and V (`../e16_summary_fusion.csv`, `../README.md`).
The pass measures the metrics below for the kernels involved. **It states no cause**: the tables are what Nsight Compute reported, and the section "Controls and ratios" says which differences are larger than the noise of the pass.
(a) issue-slot, pipe and scheduler utilisation; (b) the warp stall reasons; (c) the executed instructions, by SASS opcode; (d) the memory traffic (DRAM bytes against the computed 69 / 338 MB of K and V, L1 and L2 hit rates, shared-memory wavefronts and bank conflicts); (e) the occupancy; (f) the dequantisation kernel of the unfused pair; plus the epilogue pair as a control (the same attention kernel in both forms).

## What ran

One window on 2026-11-01, **19:37:00 to 19:37:25** (25 s; the first two attempts, announced for 19:25:15 and 19:33:00, did not start: the start guard of the window script refused, first because it matched the command line of a watcher of mine and then because it listed the script's own subshell; nothing ran on the GPU in either).
The guard passed: no compile job, benchmark or transfer was running, load before 0.14 / 0.13 / 0.25, and `processes-before.txt` / `processes-after.txt` list the desktop and the Claude sessions (each below 3 % of a core). **The card also drives the display (both monitors are connected to it), as in every other window of this revision.**
Command (`window-script.sh` is the script, kept for the record, and `window-run.txt` its output: the first line is the start line of the window, the last line the end marker): `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 E16_PROFILE=1 E16_DIMS=128 E16_SEQS=2048 E16_CAUSAL=0 E16_VARIANTS=mma_raw,mma_int8kv,mma_epilogue,mma_fp8kv_fused taskset -c 0-3,16-19 ncu --set full --kernel-name "regex:FlashAttentionMma|dequantKV|affineBf16" --launch-count 20 --rename-kernels 0 --export <report> build/local-rtx4090/bin/e16_attention_cu`
(`E16_PROFILE=1` makes `fkl-self/attention/e16_attention.cu` do one call per variant, with no warm-up and no timing). Nsight Compute 2026.3.1.0 of the CUDA 13.4.3 toolkit, driver 610.43.02, `--clock-control base` and `--cache-control all` (the defaults), kernel replay with 45 passes per kernel; the binary is the one of `binary-sha256.txt`
(sha256 bc050359…, built from the working tree on top of fkl-benchmarks c588536, FKL c8fcbe8; the same binary ran the pcie-check window). Eight kernels were profiled, in the order of the table (`ncu_stdout.txt`). The report (`e16_d128_s2048.ncu-repz`, 38.7 MB) stays outside the repository; it was exported with
`ncu --import <report> --page details --csv --print-units base` and `--page raw` (`ncu_details.csv`, `ncu_raw.csv`) and, per kernel, `--page source --print-source sass --csv` (summed by `scripts/e16_ncu_sass.py`: `ncu_opcodes.csv`, `ncu_opcodes_full.csv`, `ncu_opcodes.md`); `scripts/e16_ncu_summary.py` writes `ncu_summary.csv`, `ncu_summary.md` and
`ncu_details_tidy.csv`, and `scripts/e16_ncu_readme.py` writes this file from them.

## How to read it

- Nsight Compute locks the SM clock to its base clock (2.54 GHz in this pass) and flushes the caches before every replay pass: **no time of this pass is comparable with the benchmark windows** (boost clock, power-limited, warm caches, back-to-back calls). Only ratios and counts are used. Everything in the pass is GPU-bound.
- One launch per kernel, so there is no run-to-run spread. The spread of a metric is estimated from the three launches of the same instantiation (ids 0, 3 and 6: the attention kernel of `mma_raw`, of the unfused int8 pair and of the separate epilogue pair have the same template arguments and shapes, and different input values);
  a difference between kernels is only called a difference when it is larger than the spread of those three.
- Stall data come from PC sampling (statistical) and from the warp-state counters; the "estimated speedup" values quoted at the end are heuristics of the tool.
- The DRAM counters count the bytes that went to or came from DRAM during the kernel: lines still dirty in the 72 MB L2 when the kernel ends are not counted, so the writes are lower than the output sizes.
