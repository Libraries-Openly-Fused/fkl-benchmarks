# E16 query-tile experiment: Nsight Compute pass (RTX 4090, 2026-11-04 12:45:00 to 12:45:15)

One window (the desktop session running, cores 0-3,16-19), Nsight Compute 2026.3.1 `--set full`, clock control base and caches flushed (the defaults: **no time of this pass is comparable with the timing runs**; only the counts and the relations between the four kernels), `E16_PROFILE=1 E16_DIMS=128 E16_SEQS=2048 E16_CAUSAL=0` (BH = 128), four kernels in this launch order: `raw_bq64_kv32_w4`, `int8_bq64_w4`, `int8_bq128_w4`, `int8nodq_bq64_w4` (TIMING ONLY: wrong output, the kernel of the patched header). The report (`e16_qt_d128_s2048.ncu-repz`, 21 MB) stays outside the repository (`results/cuda13.4/e16-qt-ncu`). Tables made by `scripts/e16_qt_ncu_summary.py` and `scripts/e16_ncu_sass.py --kernels`: `ncu_selected.csv/.md` (metrics), `ncu_opcodes.csv`, `ncu_opcodes_full.csv`, `ncu_opcodes.md` (executed warp instructions by SASS opcode; the sums equal smsp__inst_executed.sum for all four). One launch per kernel, no spread. **Prediction written before the run: I2F halves from BQ64 to BQ128 and is zero in no-dequant: confirmed exactly.**

| | raw_bq64_kv32 | int8_bq64 | int8_bq128 | int8nodq_bq64 |
|---|---:|---:|---:|---:|
| I2F (warp instructions) | 0 | 67,108,864 (2^26) | 33,554,432 (2^25) | 0 |
| warp instructions executed | 434,649,109 | 764,481,084 | 521,842,236 | 599,883,536 |
| HMMA | 67,108,864 | 67,108,864 | 67,108,864 | 67,108,864 |
| FMUL / F2FP | 35.5 M / 8.9 M | 102.6 M / 42.5 M | 69.1 M / 25.7 M | 35.5 M / 8.9 M |
| shared-memory wavefronts (ld / st) | 150.3 M (134.7 M / 0) | 222.6 M (151.7 M / 34.7 M) | 118.4 M (76.6 M / 16.8 M) | 226.5 M (151.9 M / 36.5 M) |
| shared-memory bank conflicts | 0 | 4.69 M | 1.96 M | 8.63 M |
| issue slots active (% of peak) | 18.2 | 24.5 | 19.1 | 21.0 |
| xu pipe active (%) | 6.3 | 22.0 | 15.4 | 5.3 |
| stall (warps per issue): long scoreboard / short scoreboard / barrier / math pipe throttle / wait | 0.20 / 0.46 / 0.54 / 5.73 / 2.51 | 0.84 / 0.75 / 0.42 / 2.34 / 2.30 | 0.86 / 0.90 / 0.54 / 4.31 / 2.41 | 0.13 / 0.85 / 0.67 / 4.18 / 2.08 |
| registers / dynamic smem B / DRAM bytes read | 171 / 24,576 / 201.4 M | 174 / 49,152 / 137.0 M | 255 / 49,152 / 136.5 M | 174 / 49,152 / 135.6 M |
| duration at base clock (ns, not comparable with the timings) | 1,853,728 | 2,407,392 | 2,123,456 | 2,203,200 |

What the counts show (no cause beyond them): (1) the conversion count is exactly S / BLOCK_Q x the K and V elements (2^31 elements / 32 lanes per warp instruction): 32 tiles at BQ64, 16 at BQ128, so it halves; the matrix instructions (HMMA) are the same in all four. (2) The int8 kernel executes 329.8 M more warp instructions than the bf16 kernel at BQ64 (+75.9 %); removing the arithmetic (nodq) removes 164.6 M of them (I2F 67.1 M, FMUL 67.1 M, F2FP 33.6 M), i.e. half of the extra instructions; the other half (shift, logic and integer operations of the byte handling, the extra shared-memory stores and loads) stays in nodq. (3) The shared-memory pass is unchanged by removing the arithmetic (226.5 M wavefronts in nodq against 222.6 M), and halves with the query tiles (118.4 M at BQ128). (4) The scale-load wait is also gone in nodq: long-scoreboard stalls are 0.84 in the int8 kernel and 0.13 in nodq (the load is still issued, but nothing waits for its value): **the 'share of the conversion' of the timing runs therefore includes the wait for the scale, not only the arithmetic**. (5) The xu pipe (conversions) is busy 22 % in the int8 kernel and 15 % at BQ128. Caveat of the reading: the BQ128 kernels use 255 registers and spill; nodq may be scheduled differently by the compiler.
