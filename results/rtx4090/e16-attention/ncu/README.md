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

## The eight kernels (launch order of the pass)

| id | kernel | tile BQ x BKV x warps | Q, K, V reads | epilogue | registers | block | grid | dynamic smem B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mma_raw attention (bf16 path) | 64 x 64 x 4 | Bf16, Bf16, Bf16 | identity | 254 | 128 | 4,096 | 49,152 |
| 1 | mma_int8kv_fused attention | 64 x 32 x 4 | Bf16, Int8, Int8 | identity | 174 | 128 | 4,096 | 49,152 |
| 2 | dequantKV (int8 unfused) | - | - | - | 16 | 256 | 262,144 | 0 |
| 3 | mma_int8kv_unfused attention (bf16 path) | 64 x 64 x 4 | Bf16, Bf16, Bf16 | identity | 254 | 128 | 4,096 | 49,152 |
| 4 | mma_fp8kv_fused attention | 64 x 32 x 4 | Bf16, Fp8, Fp8 | identity | 174 | 128 | 4,096 | 49,152 |
| 5 | mma_epilogue_fused attention | 64 x 64 x 4 | Bf16, Bf16, Bf16 | Mul+Add | 254 | 128 | 4,096 | 49,152 |
| 6 | mma_epilogue_separate attention (bf16 path) | 64 x 64 x 4 | Bf16, Bf16, Bf16 | identity | 254 | 128 | 4,096 | 49,152 |
| 7 | affineBf16 (epilogue separate) | - | - | - | 16 | 256 | 131,072 | 0 |

## Time, instructions, issue and occupancy

| id | kernel | duration us | warp instr. M | HMMA M | issue slots busy % | IPC | no eligible % | tensor pipe active % | tensor busy us | warp cycles / issued instr. | active warps / scheduler | theor. occ. % | achieved occ. % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mma_raw attention (bf16 path) | 1803.5 | 357.4 | 67.1 | 15.24 | 0.61 | 84.67 | 46.0 | 830.0 | 12.85 | 1.97 | 16.67 | 16.41 |
| 1 | mma_int8kv_fused attention | 2413.2 | 764.5 | 67.1 | 24.34 | 0.98 | 75.59 | 34.4 | 829.9 | 8.11 | 1.98 | 16.67 | 16.59 |
| 2 | dequantKV (int8 unfused) | 247.9 | 92.3 | 0.0 | 28.67 | 1.16 | 70.56 | 0.0 | - | 30.84 | 9.08 | 100 | 75.44 |
| 3 | mma_int8kv_unfused attention (bf16 path) | 1799.6 | 357.4 | 67.1 | 15.25 | 0.61 | 84.68 | 46.0 | 827.8 | 12.85 | 1.97 | 16.67 | 16.40 |
| 4 | mma_fp8kv_fused attention | 2414.4 | 883.0 | 67.1 | 28.11 | 1.13 | 71.77 | 34.4 | 829.6 | 7.04 | 1.99 | 16.67 | 16.57 |
| 5 | mma_epilogue_fused attention | 1807.6 | 359.0 | 67.1 | 15.28 | 0.62 | 84.61 | 46.0 | 831.7 | 12.80 | 1.97 | 16.67 | 16.41 |
| 6 | mma_epilogue_separate attention (bf16 path) | 1806.9 | 357.4 | 67.1 | 15.22 | 0.61 | 84.68 | 46.0 | 831.4 | 12.85 | 1.97 | 16.67 | 16.41 |
| 7 | affineBf16 (epilogue separate) | 118.0 | 18.9 | 0.0 | 12.35 | 0.50 | 87.02 | 0.0 | - | 67.33 | 8.74 | 100 | 71.64 |

## Stalls (Nsight Compute warp-state statistics and PC sampling)

| id | kernel | stall reasons, warp cycles per issued instruction (top 5) | stall samples by opcode (top 6) | samples |
|---|---:|---:|---:|---:|
| 0 | mma_raw attention (bf16 path) | math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25 | HMMA 71.6%, BRA 6.9%, LDSM 3.3%, MUFU 2.0%, LOP3 1.9%, FADD 1.5% | 284,822 |
| 1 | mma_int8kv_fused attention | math_pipe_throttle 2.33, wait 2.30, selected 1.00, long_scoreboard 0.86, short_scoreboard 0.74 | HMMA 35.4%, FMUL 11.2%, I2F 7.7%, BRA 6.8%, SHF 5.4%, PRMT 3.5% | 380,201 |
| 2 | dequantKV (int8 unfused) | long_scoreboard 24.72, wait 1.60, short_scoreboard 1.20, selected 1.00, not_selected 0.92 | I2F 67.9%, FMUL 12.9%, BRA 4.2%, UISETP 3.1%, IMAD 2.3%, ULDC 1.3% | 38,594 |
| 3 | mma_int8kv_unfused attention (bf16 path) | math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25 | HMMA 71.5%, BRA 7.0%, LDSM 3.4%, MUFU 2.0%, LOP3 1.9%, FADD 1.5% | 284,903 |
| 4 | mma_fp8kv_fused attention | math_pipe_throttle 2.12, wait 1.71, selected 1.00, long_scoreboard 0.81, short_scoreboard 0.56 | HMMA 36.1%, FMUL 11.6%, LOP3 8.8%, BRA 7.3%, SHF 6.0%, F2FP 3.4% | 383,197 |
| 5 | mma_epilogue_fused attention | math_pipe_throttle 7.75, wait 2.13, selected 1.00, barrier 0.75, not_selected 0.26 | HMMA 71.7%, BRA 7.0%, LDSM 3.3%, MUFU 2.0%, LOP3 1.9%, FMNMX 1.5% | 284,826 |
| 6 | mma_epilogue_separate attention (bf16 path) | math_pipe_throttle 7.78, wait 2.15, selected 1.00, barrier 0.75, not_selected 0.25 | HMMA 71.5%, BRA 7.0%, LDSM 3.4%, MUFU 2.0%, LOP3 1.9%, FMNMX 1.5% | 284,876 |
| 7 | affineBf16 (epilogue separate) | long_scoreboard 59.11, wait 2.87, short_scoreboard 1.43, drain 1.25, selected 1.00 | PRMT 86.7%, BRA 4.5%, IMAD 2.8%, ISETP 1.3%, EXIT 1.3%, STG 0.6% | 18,215 |

## Executed warp instructions by SASS opcode (top 16 over all kernels; every opcode is in `ncu_opcodes.csv`, with its modifiers in `ncu_opcodes_full.csv`)

| opcode | mma_raw attention | mma_int8kv_fused attention | dequantKV | mma_int8kv_unfused attention | mma_fp8kv_fused attention | mma_epilogue_fused attention | mma_epilogue_separate attention | affineBf16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HMMA | 67,108,864 | 67,108,864 | 0 | 67,108,864 | 67,108,864 | 67,108,864 | 67,108,864 | 0 |
| FMUL | 24,697,920 | 102,620,928 | 2,097,152 | 24,699,200 | 102,614,528 | 24,697,920 | 24,697,920 | 0 |
| LOP3 | 26,411,008 | 39,436,288 | 4,194,304 | 26,411,008 | 98,156,544 | 26,411,008 | 26,411,008 | 0 |
| LDSM | 33,685,504 | 33,685,504 | 0 | 33,685,504 | 33,685,504 | 33,685,504 | 33,685,504 | 0 |
| LEA | 29,081,600 | 39,698,432 | 4,194,304 | 29,081,600 | 39,698,432 | 29,081,600 | 29,081,600 | 2,097,152 |
| IMAD | 22,200,320 | 48,201,728 | 6,291,456 | 22,200,320 | 50,298,880 | 22,724,608 | 22,200,320 | 4,194,304 |
| F2FP | 8,912,896 | 42,467,328 | 2,097,152 | 8,912,896 | 109,576,192 | 8,912,896 | 8,912,896 | 1,048,576 |
| SHF | 2,768,896 | 60,342,272 | 4,194,304 | 2,768,896 | 60,342,272 | 2,768,896 | 2,768,896 | 0 |
| FADD | 19,955,712 | 23,101,440 | 0 | 19,955,712 | 23,101,440 | 19,955,712 | 19,955,712 | 0 |
| FMNMX | 19,922,944 | 23,068,672 | 0 | 19,922,944 | 23,068,672 | 19,922,944 | 19,922,944 | 0 |
| IADD3 | 16,515,072 | 24,870,912 | 8,388,608 | 16,515,072 | 24,870,912 | 16,515,072 | 16,515,072 | 0 |
| FFMA | 17,891,328 | 18,939,904 | 0 | 17,891,328 | 18,939,904 | 18,939,904 | 17,891,328 | 1,048,576 |
| MUFU | 17,858,560 | 18,907,136 | 0 | 17,858,560 | 18,907,136 | 17,858,560 | 17,858,560 | 0 |
| ISETP | 6,995,968 | 28,491,776 | 4,194,304 | 6,995,968 | 28,491,776 | 6,995,968 | 6,995,968 | 2,097,152 |
| I2F | 0 | 67,108,864 | 2,097,152 | 0 | 0 | 0 | 0 | 0 |
| HADD2 | 0 | 0 | 0 | 0 | 67,108,864 | 0 | 0 | 0 |
| all | 357,447,575 | 764,481,084 | 92,274,688 | 357,448,882 | 882,963,635 | 359,020,439 | 357,447,575 | 18,874,368 |

## Memory (DRAM bytes as the DRAM counters saw them, L1 and L2 hit rates, shared-memory wavefronts and bank conflicts)

| id | kernel | DRAM read MB | DRAM write MB | read + write MB | DRAM throughput % | L1 hit % | L2 hit % | shared wavefronts M | bank conflicts M | conflicts % of wavefronts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | mma_raw attention (bf16 path) | 201.5 | 50.8 | 252.3 | 14.23 | 2.56 | 95.52 | 142.7 | 0.17 | 0.12 |
| 1 | mma_int8kv_fused attention | 136.4 | 45.4 | 181.7 | 7.66 | 15.20 | 94.84 | 222.8 | 4.97 | 2.23 |
| 2 | dequantKV (int8 unfused) | 71.0 | 101.2 | 172.3 | 70.68 | 45.76 | 66.03 | 2.1 | 0.00 | 0.00 |
| 3 | mma_int8kv_unfused attention (bf16 path) | 201.4 | 50.7 | 252.1 | 14.25 | 2.56 | 95.52 | 142.7 | 0.17 | 0.12 |
| 4 | mma_fp8kv_fused attention | 137.8 | 45.1 | 182.9 | 7.70 | 13.93 | 94.22 | 222.9 | 4.98 | 2.24 |
| 5 | mma_epilogue_fused attention | 201.4 | 50.9 | 252.2 | 14.19 | 2.56 | 95.52 | 142.7 | 0.16 | 0.11 |
| 6 | mma_epilogue_separate attention (bf16 path) | 202.9 | 51.0 | 253.9 | 14.29 | 2.56 | 95.49 | 142.7 | 0.19 | 0.13 |
| 7 | affineBf16 (epilogue separate) | 67.1 | 32.8 | 99.9 | 86.14 | 47.17 | 50.08 | 1.0 | 0.00 | 0.00 |

## Controls and ratios

Three launches of the same instantiation (ids 0, 3, 6): duration 1803.5 / 1799.6 / 1806.9 us, spread 0.41 %; executed instructions 0.0004 %; DRAM bytes 0.72 %; shared wavefronts 0.011 %; shared bank conflicts 8.7 %; issue slots busy 15.22 to 15.25 %.

- bf16 attention kernel (mean of ids 0, 3, 6): 1803.3 us. Fused int8 2413.2 us = x1.338; fused fp8 2414.4 us = x1.339; epilogue-fused 1807.6 us = +0.24 % (within the 0.41 % spread of the identical launches).
- int8 pair: dequantKV 247.9 + attention 1799.6 = 2047.5 us against the fused kernel 2413.2 us: separate / fused = 0.8484 (the fused kernel is 17.9 % slower). The benchmark window at the same configuration (D = 128, S = 2048, non-causal, boost clock, warm caches): 0.8665 [0.8595-0.8791].
- epilogue pair: attention 1806.9 + affineBf16 118.0 = 1925.0 us against the fused kernel 1807.6 us: separate / fused = 1.0649; benchmark window 1.0585 [1.0470-1.0681].
- tensor work: HMMA executed 67,108,864 (mma_raw) and 67,108,864 (fused int8) and 67,108,864 (fused fp8); LDSM 33,685,504 in all three.
- warp instructions executed: mma_raw 357.4 M, fused int8 764.5 M (x2.139, +407.0 M), fused fp8 883.0 M (x2.470, +525.5 M), dequantKV 92.3 M, epilogue-fused 359.0 M (+1.6 M).
- I2F executed: fused int8 67,108,864, dequantKV 2,097,152: ratio 32.0; the number of query tiles per head is S / BLOCK_Q = 2048 / 64 = 32.
- DRAM (read + write): fused int8 181.7 MB; unfused pair 424.4 MB. Reads: mma_raw 201.5 MB (Q + K + V in bf16 are 3 x 67.1 MB); fused int8 136.4 MB, that is Q (67.1 MB) plus 69.3 MB of K and V as int8 with their scales (computed in `../e16_summary_fusion.csv`: 69 MB). The writes are lower than the O (67.1 MB) and the bf16 K,V of dequantKV (134.2 MB) because lines still dirty in the 72 MB L2 at the end of a kernel are not counted by the DRAM counters.

## Scaling of the extra time of the fused int8 kernel with S (COMPUTED FROM THE BENCHMARK RUNS, not from Nsight Compute)

From the three complete runs of the E16 harness (`../run*/e16_attention.csv`: GPU time per call at the boost clock, back-to-back calls, BH x S = 262,144 tokens at every S, non-causal): per run the time of `mma_int8kv_fused` minus that of `mma_raw`, and of `mma_int8kv_unfused` minus that of `mma_raw` (the separate dequantisation kernel and the effect of its bf16 output), then the median over the runs of the per-run difference with [smallest-largest]; `scripts/e16_prologue_scaling.py` writes the table for both head sizes, both masks and fp8 as well (`../e16_prologue_scaling.csv`, `../e16_prologue_scaling.md`).

| D | S | BH | BLOCK_Q of the fused kernel | query tiles per head | mma_raw ms | fused - raw ms | (fused - raw) / S us | unfused - raw ms | (fused - raw) / (unfused - raw) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 512 | 512 | 128 | 4 | 0.257 | 0.060 [0.059-0.068] | 0.118 [0.114-0.133] | 0.117 [0.115-0.125] | 0.517 [0.508-0.544] |
| 64 | 1024 | 256 | 128 | 8 | 0.466 | 0.120 [0.120-0.122] | 0.117 [0.117-0.119] | 0.116 [0.116-0.118] | 1.031 [1.028-1.034] |
| 64 | 2048 | 128 | 128 | 16 | 0.884 | 0.255 [0.251-0.261] | 0.125 [0.123-0.127] | 0.116 [0.112-0.122] | 2.194 [2.142-2.244] |
| 64 | 4096 | 64 | 128 | 32 | 1.709 | 0.540 [0.535-0.541] | 0.132 [0.131-0.132] | 0.124 [0.118-0.125] | 4.346 [4.337-4.531] |
| 64 | 8192 | 32 | 128 | 64 | 3.379 | 1.087 [1.086-1.087] | 0.133 [0.133-0.133] | 0.125 [0.125-0.125] | 8.698 [8.666-8.713] |
| 128 | 512 | 512 | 64 | 8 | 0.509 | 0.163 [0.158-0.178] | 0.318 [0.309-0.348] | 0.227 [0.224-0.247] | 0.722 [0.697-0.726] |
| 128 | 1024 | 256 | 64 | 16 | 0.938 | 0.305 [0.276-0.321] | 0.298 [0.269-0.314] | 0.224 [0.220-0.263] | 1.230 [1.224-1.389] |
| 128 | 2048 | 128 | 64 | 32 | 1.823 | 0.533 [0.526-0.569] | 0.260 [0.257-0.278] | 0.235 [0.217-0.242] | 2.418 [2.173-2.452] |
| 128 | 4096 | 64 | 64 | 64 | 3.550 | 1.066 [1.055-1.076] | 0.260 [0.257-0.263] | 0.237 [0.234-0.277] | 4.497 [3.881-4.507] |
| 128 | 8192 | 32 | 64 | 128 | 6.971 | 2.084 [2.080-2.091] | 0.254 [0.254-0.255] | 0.297 [0.219-0.301] | 7.041 [6.908-9.526] |

### What the scaling shows, and what it does not

- At constant tokens the grid of each kernel is constant (4,096 blocks at D = 128 with 64-row query blocks, 2,048 at D = 64 with 128-row ones) and the loop of a block has S / BLOCK_KV iterations, so any cost that is paid in every iteration of the loop grows in proportion to S, like the time of `mma_raw`. The extra time of the fused kernel grows with S (D = 128: 0.254 to 0.318 us per unit of S, falling by 20 % over the range; D = 64: 0.118 to 0.133 us, rising by 13 %).
  The extra time of the separate dequantisation (unfused minus raw) does not grow with S: 0.22 to 0.24 ms at D = 128 up to S = 4096 (0.30 ms at S = 8192, spread of the runs 0.22 to 0.30), 0.12 ms at D = 64. The two cross between S = 512 and S = 1024 at both head sizes, which is where the sign of the fused-against-unfused comparison changes (`../README.md`).
- The growth is therefore what a per-iteration cost of the fused kernel predicts, the repeated conversion being one of them. It is not specific to it: the quantised path also uses BLOCK_KV = 32 against 64 for the bf16 path (twice the iterations and barriers per block), loads the per-token scales and stores the converted tiles to shared memory.
- The conversions do not explain the extra time alone across head sizes: at the same tokens and S, D = 64 (BLOCK_Q = 128) does 4 times fewer conversions than D = 128 (BLOCK_Q = 64) (conversions grow with tokens x S x D / BLOCK_Q), but its extra time per unit of S is 0.37 to 0.52 of the D = 128 value, not 0.25.

## Where the conversion happens in the library kernel

`external/fkl` at c8fcbe8, `include/fused_kernel/algorithms/attention/flash_attention_mma.h`:

- lines 332-336: `QUANT_KV` is the schedule for int8 or fp8 per-token K and V: "cp.async the RAW QUANTIZED bytes + smem->smem dequant pass";
- lines 1335 and 1346-1348: one block per (query block, head): `bh = blockIdx.y`, `qBlockIdx = blockIdx.x` (reversed for causal), `qBlockBase = qBlockIdx * BLOCK_Q`; the profiled kernel has grid (32, 128, 1) (`ncu_details.csv`): 2048 / 64 = 32 query blocks per head, 128 heads;
- lines 1408-1409 and 1432: without causal mask `kvEnd = seq_k`, `numIter = ceil(kvEnd / BLOCK_KV)`, `itBegin = 0, itEnd = numIter` (the trimming of the range applies only with a block-sparse mask, which these runs do not use);
- lines 1667-1672: the K and V planes and their scales are addressed by `bh` and `seqK` only, so every query block of a head reads the same K and V;
- line 1690: `for (int kv = itBegin; kv < itEnd; ++kv)` in the `QUANT_KV` branch: every block walks over all the K and V tiles of its head; in each iteration `cpasyncQuantTile` copies the raw bytes (lines 1696 and 1727), `dequantTile` converts the K tile (line 1705, when `FP8_QK` is not set; for int8 it is not) and the V tile (line 1745) from the staging bytes to bf16 in shared memory.

So a K or V element is converted once per query block of its head, S / BLOCK_Q times (32 in the profiled kernel), and the measured count of `I2F` instructions of the fused int8 kernel is 32.0 times that of `dequantKV`. (For fp8 the loop and the V conversion are the same; whether the K conversion pass is present depends on `FK_ENABLE_FP8_QK`, lines 340-341, which I have not checked for this build.)

## Status of the claim

The Nsight Compute metrics (the same tensor work, 2.14 times the instructions, the extra instructions in the conversion opcode classes, an `I2F` count equal to the number of query blocks times that of the standalone kernel) and the growth of the extra time with S are **consistent with the repeated conversion being a large part of the cost of the fused kernel; no experiment removed it, so neither is a measured cause.**
An experiment that would test it, not run: the quantised kernel with a larger query block (BLOCK_Q = 128 at D = 128 instead of the library's 64, if the library's template accepts it, which halves the conversion passes per head), or with the conversion left out (wrong results, timing only). It needs a window; the paper agent and Oscar decide whether it is wanted.

## What Nsight Compute's own rules said (verbatim, optimisation hints of the tool for ids 0 and 1; they are heuristics of the tool, not results of this study)

- id 0 `SOLBottleneck` (the tool's estimated speedup - %): This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
- id 0 `MemoryCacheAccessPattern` (the tool's estimated speedup 12.69 %): The memory access pattern for global stores to L1TEX might not be optimal. On average, only 16.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global stores.
- id 0 `IssueSlotUtilization` (the tool's estimated speedup 50.12 %): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 6.5 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.97 active warps per scheduler, but only an average of 0.19 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.
- id 0 `CPIStall` (the tool's estimated speedup 50.12 %): On average, each warp of this workload spends 7.8 cycles being stalled waiting for the execution pipe to be available. This stall occurs when all active warps execute their next instruction on a specific, oversubscribed math pipeline. Try to increase the number of active warps to hide the existent latency or try changing the instruction mix to utilize all available pipelines in a more balanced way. When this reason dominates, the kernel is usually compute-pipeline-bound; otherwise, increasing active warps can help hide latency. This stall type represents about 60.5% of the total average of 12.9 cycles between issuing two instructions.
- id 0 `FPInstructions` (the tool's estimated speedup 1.33 %): This workload executes 17891328 fused and 44653632 non-fused FP32 instructions. By converting pairs of non-fused instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point), higher-throughput equivalent, the achieved FP32 performance could be increased by up to 36% (relative to its current performance).
- id 0 `TheoreticalOccupancy` (the tool's estimated speedup 50.12 %): The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory.
- id 0 `UncoalescedGlobalAccess` (the tool's estimated speedup 1.475 %): This workload has uncoalesced global accesses resulting in a total of 2097152 excessive sectors (1% of the total 140509184 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional information on reducing uncoalesced device memory accesses.
- id 1 `SOLBottleneck` (the tool's estimated speedup - %): This workload exhibits low compute throughput and memory bandwidth utilization relative to the peak performance of this device. Achieved compute throughput and/or memory bandwidth below 60.0% of peak typically indicate latency issues. Look at Scheduler Statistics and Warp State Statistics for potential reasons.
- id 1 `MemoryCacheAccessPattern` (the tool's estimated speedup 22.94 %): The memory access pattern for global loads from L1TEX might not be optimal. On average, only 8.0 of the 32 bytes transmitted per sector are utilized by each thread. This could possibly be caused by a stride between threads. Check the Source Counters section for uncoalesced global loads.
- id 1 `IssueSlotUtilization` (the tool's estimated speedup 65.81 %): Every scheduler is capable of issuing one instruction per cycle, but for this workload each scheduler only issues an instruction every 4.1 cycles. This might leave hardware resources underutilized and may lead to less optimal performance. Out of the maximum of 12 warps per scheduler, this workload allocates an average of 1.98 active warps per scheduler, but only an average of 0.28 warps were eligible per cycle. Eligible warps are the subset of active warps that are ready to issue their next instruction. Every cycle with no eligible warp results in no instruction being issued and the issue slot remains unused. To increase the number of eligible warps, avoid possible load imbalances due to highly different execution durations per warp. Reducing stalls indicated on the Warp State Statistics and Source Counters sections can help, too.
- id 1 `FPInstructions` (the tool's estimated speedup 2.909 %): This workload executes 18939904 fused and 125722368 non-fused FP32 instructions. By converting pairs of non-fused instructions to their fused (https://docs.nvidia.com/cuda/floating-point/#cuda-and-floating-point), higher-throughput equivalent, the achieved FP32 performance could be increased by up to 43% (relative to its current performance).
- id 1 `TheoreticalOccupancy` (the tool's estimated speedup 65.81 %): The 2.00 theoretical warps per scheduler this kernel can issue according to its occupancy are below the hardware maximum of 12. This kernel's theoretical occupancy (16.7%) is limited by the number of required registers, and the required amount of shared memory.
- id 1 `UncoalescedGlobalAccess` (the tool's estimated speedup 1.551 %): This workload has uncoalesced global accesses resulting in a total of 2097152 excessive sectors (3% of the total 81788928 sectors). Check the L2 Theoretical Sectors Global Excessive table for the primary source locations. The CUDA Programming Guide (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-memory-accesses) has additional information on reducing uncoalesced device memory accesses.
