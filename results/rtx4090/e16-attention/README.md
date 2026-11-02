# E16, forward attention with FKL: the attention core and the work around it (RTX 4090 (sm_89), CUDA 13.4.3)

Question Q1 (nothing fused around the pattern): how close is FKL's tensor-core attention DPP, as shipped, to the best attention kernels? Question Q2 (the work before and after the pattern): what does fusing it buy, on FKL's own kernel and against the same work as separate PyTorch kernels? Every number below is the median over three complete runs of the run median, with [smallest-largest] of the run medians; no result is tuned, and a ratio below 1 is reported as it is.

## Q1: the attention core (TFLOP/s, GPU-bound)

| D | mask | S | BH | FKL mma_raw TFLOP/s | tile BQxBKVxwarps | best baseline | FKL / best baseline | clock check |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 64 | non-causal | 512 | 512 | 133.9 | 128x64x4 | triton | 0.930 [0.924-0.960] | DIFFERENT CLOCKS |
| 64 | non-causal | 1024 | 256 | 147.4 | 128x64x4 | triton | 0.958 [0.958-0.963] |  |
| 64 | non-causal | 2048 | 128 | 155.4 | 128x64x4 | triton | 0.990 [0.987-1.001] |  |
| 64 | non-causal | 4096 | 64 | 160.9 | 128x64x4 | triton | 1.013 [1.012-1.016] |  |
| 64 | non-causal | 8192 | 32 | 162.7 | 128x64x4 | triton | 1.012 [1.012-1.016] |  |
| 64 | causal | 512 | 512 | 105.1 | 64x64x4 | triton | 0.997 [0.996-1.017] | DIFFERENT CLOCKS |
| 64 | causal | 1024 | 256 | 116.9 | 64x128x4 | triton | 0.978 [0.977-0.985] |  |
| 64 | causal | 2048 | 128 | 131.6 | 64x128x4 | flash_attn | 1.015 [1.015-1.031] | DIFFERENT CLOCKS |
| 64 | causal | 4096 | 64 | 140.1 | 128x64x4 | flash_attn | 1.030 [1.029-1.032] |  |
| 64 | causal | 8192 | 32 | 144.4 | 128x64x4 | flash_attn | 1.036 [1.036-1.038] |  |
| 128 | non-causal | 512 | 512 | 135.1 | 64x64x4 | flash_attn | 0.916 [0.913-0.943] | DIFFERENT CLOCKS |
| 128 | non-causal | 1024 | 256 | 146.5 | 64x64x4 | flash_attn | 0.940 [0.925-0.955] | DIFFERENT CLOCKS |
| 128 | non-causal | 2048 | 128 | 150.8 | 64x64x4 | triton | 0.953 [0.946-0.959] | DIFFERENT CLOCKS |
| 128 | non-causal | 4096 | 64 | 154.9 | 64x64x4 | triton | 0.962 [0.962-0.967] | DIFFERENT CLOCKS |
| 128 | non-causal | 8192 | 32 | 157.7 | 64x64x4 | triton | 0.976 [0.975-0.976] | DIFFERENT CLOCKS |
| 128 | causal | 512 | 512 | 108.0 | 64x64x4 | flash_attn | 0.988 [0.977-0.997] | DIFFERENT CLOCKS |
| 128 | causal | 1024 | 256 | 125.2 | 64x64x4 | flash_attn | 0.938 [0.932-0.945] | DIFFERENT CLOCKS |
| 128 | causal | 2048 | 128 | 137.1 | 64x64x4 | flash_attn | 0.958 [0.951-1.001] | DIFFERENT CLOCKS |
| 128 | causal | 4096 | 64 | 144.0 | 64x64x4 | flash_attn | 0.975 [0.975-0.979] | DIFFERENT CLOCKS |
| 128 | causal | 8192 | 32 | 148.9 | 64x64x4 | flash_attn | 1.005 [1.001-1.014] |  |

All baselines per point (TFLOP/s, flash = FlashAttention-2 of PyTorch, cuDNN, efficient, flash-attn 2.8.3, Triton) are in `e16_summary_core.csv`; the time per call in the run CSVs.

## Q2: the work around the pattern (fused against separate)

| work | D | S | mask | fused ms | separate ms | separate / fused | verdict | kernels fused / separate | PyTorch composition ms | regs / local B (fused) | tile fused | tile separate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| int8 K,V prologue | 128 | 2048 | non-causal | 2.364 | 2.048 | 0.867 [0.860-0.879] | fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +4.2 %) | 1 / 2 | 2.024 (3 kernels) | 174 / 0 | 64x32x4 | 64x64x4 |
| int8 K,V prologue | 128 | 8192 | non-causal | 9.055 | 7.269 | 0.802 [0.794-0.803] | fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +3.5 %) | 1 / 2 | 7.201 (3 kernels) | 174 / 0 | 64x32x4 | 64x64x4 |
| fp8 e4m3 K,V prologue | 128 | 2048 | non-causal | 2.382 | 2.044 | 0.862 [0.853-0.868] | fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +4.1 %) | 1 / 2 | - | 174 / 0 | 64x32x4 | 64x64x4 |
| fp8 e4m3 K,V prologue | 128 | 8192 | non-causal | 9.229 | 7.225 | 0.780 [0.778-0.788] | fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %) | 1 / 2 | - | 174 / 0 | 64x32x4 | 64x64x4 |
| epilogue o*a+b | 128 | 2048 | non-causal | 1.830 | 1.921 | 1.058 [1.047-1.068] | fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -3.5 %) | 1 / 2 | 1.893 (2 kernels) | 254 / 0 | 64x64x4 | 64x64x4 |
| epilogue o*a+b | 128 | 8192 | non-causal | 6.970 | 7.098 | 1.018 [1.013-1.023] | fused faster, exceeds the spread | 1 / 2 | 7.072 (2 kernels) | 254 / 0 | 64x64x4 | 64x64x4 |

All configurations are in `e16_summary_fusion.csv`. The numbers for the text:

```
E16 numbers for the text, NVIDIA GeForce RTX 4090, 3 runs (run1, run2, run3); a value is the median over the runs of the per-run value, with [smallest-largest] over the runs
Q1, the attention core with nothing fused around it: fraction = FKL mma_raw TFLOP/s / TFLOP/s of the best of {flash, cudnn, efficient, flash_attn, triton} at the same configuration (GPU-bound numbers)
  D=64, non-causal: fraction of the best baseline from 0.93 (S=512) to 1.01 (S=4096); by S: 512: 0.93 [0.92-0.96], best triton, 1024: 0.96 [0.96-0.96], best triton, 2048: 0.99 [0.99-1.00], best triton, 4096: 1.01 [1.01-1.02], best triton, 8192: 1.01 [1.01-1.02], best triton
  D=64, causal: fraction of the best baseline from 0.98 (S=1024) to 1.04 (S=8192); by S: 512: 1.00 [1.00-1.02], best triton, 1024: 0.98 [0.98-0.98], best triton, 2048: 1.02 [1.02-1.03], best flash_attn, 4096: 1.03 [1.03-1.03], best flash_attn, 8192: 1.04 [1.04-1.04], best flash_attn
  D=128, non-causal: fraction of the best baseline from 0.92 (S=512) to 0.98 (S=8192); by S: 512: 0.92 [0.91-0.94], best flash_attn, 1024: 0.94 [0.92-0.96], best flash_attn, 2048: 0.95 [0.95-0.96], best triton, 4096: 0.96 [0.96-0.97], best triton, 8192: 0.98 [0.97-0.98], best triton
  D=128, causal: fraction of the best baseline from 0.94 (S=1024) to 1.01 (S=8192); by S: 512: 0.99 [0.98-1.00], best flash_attn, 1024: 0.94 [0.93-0.94], best flash_attn, 2048: 0.96 [0.95-1.00], best flash_attn, 4096: 0.97 [0.97-0.98], best flash_attn, 8192: 1.01 [1.00-1.01], best flash_attn
FKL mma_raw against the best baseline over all 20 points: fraction from 0.916 to 1.036; ABOVE the best baseline by the median of the runs in 6 of 20 points: D=64 non-causal S=4096 (1.013); D=64 non-causal S=8192 (1.012); D=64 causal S=2048 (1.015); D=64 causal S=4096 (1.030); D=64 causal S=8192 (1.036); D=128 causal S=8192 (1.005). Within 1 % below it (0.990 to 1.000) in 2 points: D=64 non-causal S=2048 (0.990); D=64 causal S=512 (0.997). Points where one run is above 1 but the median is not: 3: D=64 non-causal S=2048 (0.990) run maximum 1.001; D=64 causal S=512 (0.997) run maximum 1.017; D=128 causal S=2048 (0.958) run maximum 1.001.
Q2, the work around the pattern: ratio separate / fused (> 1: the fused form is faster), range of the median ratio over the configurations of the group (GPU-bound numbers)
  int8 K,V prologue, FKL separate kernels / FKL fused, D=64, non-causal: 0.784 (S=8192) to 1.179 (S=512)
  int8 K,V prologue, FKL separate kernels / FKL fused, D=128, non-causal: 0.802 (S=8192) to 1.102 (S=512)
  int8 K,V prologue, FKL separate kernels / FKL fused, D=64, causal: 0.822 (S=8192) to 1.269 (S=512)
  int8 K,V prologue, FKL separate kernels / FKL fused, D=128, causal: 0.849 (S=8192) to 1.371 (S=512)
  int8 K,V prologue, PyTorch composition / FKL fused, D=64, non-causal: 0.816 (S=8192) to 1.202 (S=512)
  int8 K,V prologue, PyTorch composition / FKL fused, D=128, non-causal: 0.795 (S=8192) to 1.117 (S=512)
  int8 K,V prologue, PyTorch composition / FKL fused, D=64, causal: 0.856 (S=8192) to 1.386 (S=512)
  int8 K,V prologue, PyTorch composition / FKL fused, D=128, causal: 0.863 (S=8192) to 1.399 (S=512)
  int8 K,V prologue, torch.compile dequant + flash / FKL fused, D=64, non-causal: 0.794 (S=8192) to 1.025 (S=512)
  int8 K,V prologue, torch.compile dequant + flash / FKL fused, D=128, non-causal: 0.788 (S=8192) to 0.991 (S=512)
  int8 K,V prologue, torch.compile dequant + flash / FKL fused, D=64, causal: 0.838 (S=8192) to 1.216 (S=512)
  int8 K,V prologue, torch.compile dequant + flash / FKL fused, D=128, causal: 0.851 (S=8192) to 1.270 (S=512)
  fp8 e4m3 K,V prologue, FKL separate kernels / FKL fused, D=64, non-causal: 0.791 (S=8192) to 1.187 (S=512)
  fp8 e4m3 K,V prologue, FKL separate kernels / FKL fused, D=128, non-causal: 0.780 (S=8192) to 1.126 (S=512)
  fp8 e4m3 K,V prologue, FKL separate kernels / FKL fused, D=64, causal: 0.831 (S=8192) to 1.282 (S=512)
  fp8 e4m3 K,V prologue, FKL separate kernels / FKL fused, D=128, causal: 0.815 (S=8192) to 1.382 (S=512)
  epilogue o*a+b, FKL separate kernels / FKL fused, D=64, non-causal: 1.010 (S=8192) to 1.150 (S=512)
  epilogue o*a+b, FKL separate kernels / FKL fused, D=128, non-causal: 1.018 (S=8192) to 1.200 (S=512)
  epilogue o*a+b, FKL separate kernels / FKL fused, D=64, causal: 1.021 (S=8192) to 1.265 (S=512)
  epilogue o*a+b, FKL separate kernels / FKL fused, D=128, causal: 1.031 (S=8192) to 1.387 (S=512)
  epilogue o*a+b, PyTorch composition / FKL fused, D=64, non-causal: 1.062 (S=4096) to 1.160 (S=512)
  epilogue o*a+b, PyTorch composition / FKL fused, D=128, non-causal: 1.014 (S=8192) to 1.199 (S=512)
  epilogue o*a+b, PyTorch composition / FKL fused, D=64, causal: 1.068 (S=8192) to 1.428 (S=512)
  epilogue o*a+b, PyTorch composition / FKL fused, D=128, causal: 1.048 (S=8192) to 1.408 (S=512)
Q2, sign and size of the fused-against-separate difference at every S (separate time / fused time, median [smallest-largest] over the runs; verdict: which form is faster, and whether the run medians of the two forms do not overlap = exceeds the spread)
  int8 K,V prologue, D=64, non-causal: S=512: 1.179 [1.178-1.180] fused faster, exceeds the spread; S=1024: 0.994 [0.993-0.994] fused slower, exceeds the spread; S=2048: 0.878 [0.878-0.878] fused slower, exceeds the spread; S=4096: 0.815 [0.815-0.815] fused slower, exceeds the spread; S=8192: 0.784 [0.784-0.785] fused slower, exceeds the spread
  int8 K,V prologue, D=128, non-causal: S=512: 1.102 [1.091-1.103] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to +6.3 %); S=1024: 0.953 [0.931-0.958] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +3.6 %); S=2048: 0.867 [0.860-0.879] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +4.2 %); S=4096: 0.822 [0.820-0.827] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %); S=8192: 0.802 [0.794-0.803] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +3.5 %)
  int8 K,V prologue, D=64, causal: S=512: 1.269 [1.269-1.271] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -10.7 %); S=1024: 1.086 [1.082-1.088] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -7.5 %); S=2048: 0.944 [0.931-0.947] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +4.1 %); S=4096: 0.866 [0.865-0.866] fused slower, exceeds the spread; S=8192: 0.822 [0.822-0.824] fused slower, exceeds the spread
  int8 K,V prologue, D=128, causal: S=512: 1.371 [1.365-1.373] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -2.3 %); S=1024: 1.088 [1.082-1.109] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -2.8 %); S=2048: 0.946 [0.945-0.950] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to -2.3 %); S=4096: 0.883 [0.879-0.883] fused slower, exceeds the spread; S=8192: 0.849 [0.839-0.851] fused slower, exceeds the spread
  fp8 e4m3 K,V prologue, D=64, non-causal: S=512: 1.187 [1.187-1.187] fused faster, exceeds the spread; S=1024: 1.004 [1.004-1.005] fused faster, exceeds the spread; S=2048: 0.889 [0.889-0.889] fused slower, exceeds the spread; S=4096: 0.824 [0.824-0.824] fused slower, exceeds the spread; S=8192: 0.791 [0.791-0.793] fused slower, exceeds the spread
  fp8 e4m3 K,V prologue, D=128, non-causal: S=512: 1.126 [1.112-1.127] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to +3.0 %); S=1024: 0.972 [0.970-0.980] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +5.4 %); S=2048: 0.862 [0.853-0.868] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +4.1 %); S=4096: 0.804 [0.802-0.808] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %); S=8192: 0.780 [0.778-0.788] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %)
  fp8 e4m3 K,V prologue, D=64, causal: S=512: 1.282 [1.281-1.284] fused faster, exceeds the spread; S=1024: 1.099 [1.096-1.101] fused faster, exceeds the spread; S=2048: 0.946 [0.944-0.946] fused slower, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %); S=4096: 0.878 [0.878-0.879] fused slower, exceeds the spread; S=8192: 0.831 [0.831-0.832] fused slower, exceeds the spread
  fp8 e4m3 K,V prologue, D=128, causal: S=512: 1.382 [1.381-1.382] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -5.6 %); S=1024: 1.091 [1.090-1.131] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to +4.7 %); S=2048: 0.962 [0.947-0.962] fused slower, exceeds the spread; S=4096: 0.866 [0.862-0.870] fused slower, exceeds the spread; S=8192: 0.815 [0.814-0.823] fused slower, exceeds the spread
  epilogue o*a+b, D=64, non-causal: S=512: 1.150 [1.138-1.159] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -4.0 %); S=1024: 1.072 [1.067-1.085] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -2.8 %); S=2048: 1.034 [1.030-1.044] fused faster, exceeds the spread; S=4096: 1.021 [1.021-1.025] fused faster, exceeds the spread; S=8192: 1.010 [1.010-1.010] fused faster, exceeds the spread
  epilogue o*a+b, D=128, non-causal: S=512: 1.200 [1.197-1.231] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -7.0 %); S=1024: 1.112 [1.106-1.130] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to +3.0 %); S=2048: 1.058 [1.047-1.068] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -3.5 %); S=4096: 1.031 [1.030-1.035] fused faster, exceeds the spread; S=8192: 1.018 [1.013-1.023] fused faster, exceeds the spread
  epilogue o*a+b, D=64, causal: S=512: 1.265 [1.243-1.274] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -11.8 %); S=1024: 1.134 [1.121-1.142] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -8.5 %); S=2048: 1.075 [1.039-1.078] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -4.7 %); S=4096: 1.042 [1.042-1.043] fused faster, exceeds the spread; S=8192: 1.021 [1.021-1.021] fused faster, exceeds the spread
  epilogue o*a+b, D=128, causal: S=512: 1.387 [1.379-1.392] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -11.3 %); S=1024: 1.199 [1.175-1.206] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -3.5 %); S=2048: 1.111 [1.081-1.114] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to +2.3 %); S=4096: 1.055 [1.049-1.061] fused faster, exceeds the spread; S=8192: 1.031 [1.031-1.037] fused faster, exceeds the spread, DIFFERENT CLOCKS (up to -3.7 %)
Global bytes of one call, COMPUTED from the sizes and not measured (compulsory traffic, every element once; re-reads across query tiles are not counted); t = BH x S tokens, d = head dimension: prologue fused (int8 or e4m3 K and V read once, fp32 scales) 2 t d + 8 t; prologue unfused (dequantisation kernel reads them, writes bf16; attention reads bf16) 10 t d + 8 t; PyTorch composition (int8 and bf16 scales read, bf16 written, flash reads bf16) 10 t d + 4 t; raw bf16 K and V 4 t d; epilogue (output O): fused 2 t d (written once), separate 6 t d (written, read, written). Per configuration in the columns *_global_bytes_computed of e16_summary_fusion.csv.
SM clock check: 60 fused/separate comparisons have the SM clock of both members from the sampler; 31 of them differ by more than 2 % and are marked 'DIFFERENT CLOCKS' in the table and the CSV: int8 K,V prologue D=64 S=512 causal=1 (-10.73 %); int8 K,V prologue D=64 S=1024 causal=1 (-7.47 %); int8 K,V prologue D=64 S=2048 causal=1 (4.12 %); int8 K,V prologue D=128 S=512 causal=0 (6.29 %); int8 K,V prologue D=128 S=1024 causal=0 (3.57 %); int8 K,V prologue D=128 S=2048 causal=0 (4.17 %); int8 K,V prologue D=128 S=4096 causal=0 (2.34 %); int8 K,V prologue D=128 S=8192 causal=0 (3.51 %); int8 K,V prologue D=128 S=512 causal=1 (-2.26 %); int8 K,V prologue D=128 S=1024 causal=1 (-2.84 %); int8 K,V prologue D=128 S=2048 causal=1 (-2.29 %); fp8 e4m3 K,V prologue D=64 S=2048 causal=1 (2.31 %) ...
SM clock check of the core: 12 of 20 configurations have an SM clock of mma_raw that differs by more than 2 % from the one of the best baseline.
Fused-prologue pairs: in-kernel shared-memory dequantisation (cp.async of the quantised bytes) against a separate dequantisation kernel that writes bf16 to global memory; the two attention kernels differ in schedule details and tile sizes (block_q, block_kv), so these pairs are not a pure fusion effect. Only the epilogue pair (same kernel and schedule) is.
dequant_flash, dequant_compiled_flash and epilogue_flash are compositions of off-the-shelf pieces that a user has today, not a hand-fused quantised-cache attention kernel and not the best that could be written for these cases.
```

## What this folder holds
Three complete runs (`run1`..`run3`) of the E16 harness on one GPU: forward attention, bf16 Q, K, V of shape (BH, S, D), fp32 accumulation, bf16 output; D = 64 and 128, S = 512 to 8192, causal and not; BH = tokens / S so that BH x S is the same for every S
(tokens: 262,144 on the RTX 4090: BH from 512 to 32; the RTX 5050 runs in results/rtx5050/e16-attention are material only and are not used in the paper: the card sits in a PCIe 3.0 x1 slot, which would bias the comparison of many kernels against one fused kernel). Per run: `e16_attention.csv` (FKL, fkl-self/attention/e16_attention.cu) and `e16_baselines.csv` (PyTorch SDPA backends, flash-attn, Triton,
the dequantisation and epilogue compositions; pytorch-attention/e16_baselines.py), their `_clock.csv` versions (the same rows plus the SM clock of the row, see Clock), the time-stamped output, the exit codes, the GPU state at the end. Window-wide: `load.log` (every 20 s),
`gpu_clocks.csv` (SM clock, power, temperature every 0.1 s), `processes-before.txt`, `processes-after.txt`, `binary-sha256.txt`, `gpu-info.csv`, `e16_sass_counts.csv` (the attention kernels of the binary: template arguments and SASS instruction counts). `precheck/` is the tiny
configuration run at the start of the window to check the programs; `smoke/` and `smoke-py/` are the two short functional tests that came before (3 blocks of about 5 ms; not measurements).
`python scripts/e16_join_clock.py <this folder>` adds the clock, then `python scripts/e16_summary.py <this folder>` writes `e16_summary_core.csv`, `e16_summary_fusion.csv`, `e16_summary_kernels.csv`, `e16_q2_compact.csv` and `e16_text_numbers.txt`;
the figures come from `scripts/plot_e16.py`.

## Method
- Time per call: `blocks` (15) blocks of back-to-back calls between two CUDA events, each block about 20 ms long: the GPU time of the call, with no gap for the host between calls (PyTorch's dispatch costs 50 to 100 us per call and would otherwise be charged to the baselines in the smallest
  configurations). Before the blocks, a warm-up of at least 10 calls and at least 300 ms of GPU time (a GPU that has just started the process, or has run a light kernel, is not at its boost clock). `single_ms_median` is the median of 15 single calls, each between two events and synchronised:
  the launch latency of the call is in it, so that column is launch-sensitive at small S. TFLOP/s = 4 BH S^2 D (halved for causal) / time. The numbers of the figures and tables are GPU-bound (TFLOP/s of the blocks).
- Each number in the summaries is the median over the runs of the median of the run, with the smallest and the largest of the run medians as the run-to-run spread. "Exceeds the spread" means that the run medians of the two forms do not overlap.
- Order: the order of the variants must not decide a comparison. In the FKL program the variants are grouped as {raw}, {int8 fused, int8 unfused}, {fp8 fused, fp8 unfused}, {epilogue fused, epilogue separate}; for configuration k of run r the groups are rotated by (k + r - 1) mod 4 and the two members
  of a pair are swapped when that shift is odd, so the members of a pair are always adjacent and their order flips between configurations and between runs; the SIMT variant is always last (column `position` = place in the order). The Python harness does the same with the groups
  {flash, epilogue_flash, dequant_flash, dequant_compiled_flash} (one group, members swapped on odd shifts), {cuDNN}, {efficient}, {math}, {flash-attn}, {Triton}. The order of the two programs alternates between the runs (odd runs: FKL first). Because the order changes with the run,
  the run-to-run range contains the order effect.
- Clock: every row has the wall-clock span of its timed blocks (`t_start_ms`, `t_end_ms`); `scripts/e16_join_clock.py` joins it with the 100 ms nvidia-smi sampler of the window (`gpu_clocks.csv`) and writes `sm_mhz_median`, `sm_mhz_min`, `sm_mhz_samples`, `power_w_median` and `util_pct_median`
  into the `_clock.csv` files. The summaries mark every comparison whose two members differ in SM clock by more than 2 % as DIFFERENT CLOCKS and give, next to the raw fraction or ratio, the same number at equal SM clock (TFLOP/s per MHz, or time x clock).
  The GPUs run at their default clocks and power limits. The RTX 4090 reaches its 450 W power limit in the long runs (about 40 % of the samples at 440 W or more), and kernels that draw more power run at a lower SM clock (the raw FKL kernel about 2 to 4 % below the best baselines on median,
  up to 9 % in single rows): the raw numbers are what the GPU does, the equal-clock numbers show what remains if the clocks were equal; neither is hidden.
- Library as shipped: the attention headers are those of the pin (identical to the ones of pull request #422, which only adds the unit test); the tile sizes are the library's own compile-time choice (BLOCK_Q = BLOCK_KV = 0), no per-shape tuning, no extra flags.
  Baselines as shipped: PyTorch 2.14 SDPA backends (flash = FlashAttention-2, cuDNN, memory-efficient, math), flash-attn 2.8.3, the forward kernel of Triton's fused-attention tutorial with its autotuner (24 configurations, which FKL has no counterpart of).
  Neither side is tuned per shape. Every baseline is imported and called once on a tiny input before the main loop (`# probe` lines in the header of `e16_baselines.csv`); a baseline that fails writes its own error text in its rows and the others go on.
- Error: largest and mean absolute error of the output against an fp32 attention of the same (dequantised) inputs, for two heads. The quantisation error itself is excluded: it measures the arithmetic of the kernel. The epilogue variants are compared with ref * a + b.
- Kernels of one call: for the FKL program from capturing the call into a CUDA graph and reading its kernel nodes (count, registers, local bytes, static and dynamic shared memory, blocks per SM, occupancy per kernel); for the baselines from CUPTI through torch.profiler (count and names).
- Global bytes: `*_global_bytes_computed` in `e16_summary_fusion.csv` are COMPUTED from the sizes, not measured: the compulsory traffic of one call with every element counted once (formulas in `e16_text_numbers.txt`); what the kernels re-read across query tiles is not in them.

## Read with
- Schedules and tiles, measured and not inferred: the columns `schedule`, `block_q`, `block_kv`, `num_warps` and `split_kv` of `e16_attention.csv` come from the template arguments of the attention kernel of the call, and `e16_sass_counts.csv` counts the SASS instructions of every
  instantiation in the binary (LDGSTS = cp.async, LDG, LDS, LDSM, STS, HMMA). `mma_raw`, the epilogue variants and the bf16 buffers of the unfused variants run the library's bf16 cp.async schedule (16-byte LDGSTS copies of K and V tiles). The fused int8 and fp8
  variants run the library's quantised schedule: the raw quantised bytes are copied with cp.async (8-byte LDGSTS) into a staging area of shared memory and dequantised from shared memory to shared memory inside the kernel (STS instructions of the dequantisation pass appear
  in their SASS; the per-token scales are read from global memory with LDG). No attention kernel of the series uses a register-prefetch schedule. Each path has its own tile sizes, the library's own choice (`flash_attention_mma.h`): D = 128: bf16 path BLOCK_Q 64 and BLOCK_KV 64,
  quantised path BLOCK_Q 64 and BLOCK_KV 32; D = 64: quantised path BLOCK_Q 128 and BLOCK_KV 32; bf16 path BLOCK_Q 128 and BLOCK_KV 64 when non-causal (BH x S >= 65,536, always here), and when causal at S >= 8192 or at S >= 4096 with BH >= 64, otherwise BLOCK_Q 64 and BLOCK_KV 128
  when causal at S >= 1024, otherwise 64 and 64: the tile of every row is in its columns, so the points where it changes with S and causality are visible. A fused-against-unfused prologue pair therefore compares an in-kernel shared-memory dequantisation pass (half the global bytes for K and V)
  with a separate dequantisation kernel that writes bf16 to global memory, and the two attention kernels differ in schedule details and tile sizes as well as in fusion: it is not a pure fusion effect. The registers, local-memory bytes, blocks per SM and tile of both kernels are printed
  next to every pair; these are measured facts and no cause of a difference is stated. The epilogue pair (`mma_epilogue_fused` against `mma_epilogue_separate`) runs the same attention kernel and schedule and differs only in the epilogue.
- The binary also contains kernels that no configuration of the series calls: the D = 128 raw-path instantiations with BLOCK_Q 128, BLOCK_KV 128 and 8 warps (255 registers and 104 bytes of stack on sm_89, 16 bytes on sm_120; the library selects them only for S >= 16384), and the SPLIT_KV flavour of every
  instantiation (the library splits the key range only when the grid fills less than about 70 % (causal) or 45 % (dense) of one wave of the GPU, which does not happen with the BH used here: every FKL row shows 1 kernel, and its own local bytes).
- `dequant_flash`, `dequant_compiled_flash` and `epilogue_flash` are compositions of off-the-shelf pieces that a user has today (elementwise kernels around the flash kernel). They are not a hand-fused quantised-cache attention kernel such as the ones in serving systems,
  and not the best that could be written for these cases. The kernel counts are measured (CUPTI): dequant_flash 3 (two elementwise kernels and flash); dequant_compiled_flash 3 (TorchInductor emitted two Triton kernels, one for K and one for V, then flash); epilogue_flash 2.
- PCIe link of the RTX 4090: not logged during these runs. A link check under load made afterwards (`../pcie-check/`: sampler every 100 ms, an attention kernel, pinned and pageable copies of 256 MB) shows generation 4, width 16 whenever the GPU is out of the idle state (pinned copies 25.6 and 24.8 GB/s); the pstate of this window was P2 in 9,381 of its 9,510 samples (P8 in 34).
- `ncu/` is the Nsight Compute pass on the attention kernels at D = 128, S = 2048, non-causal (eight kernels, one launch each, base-clock lock, cold caches; its own README): executed instructions by SASS opcode, issue, stall, memory and occupancy metrics of the fused int8 and fp8 kernels, the dequantisation kernel and the epilogue pair as a control. `e16_prologue_scaling.csv` and `.md` (`scripts/e16_prologue_scaling.py`, computed from the runs of this folder, not from Nsight Compute) give the extra time of the fused quantised kernel over `mma_raw` against S next to the cost of the separate dequantisation. Neither states a cause of the prologue result: the metrics and the scaling are consistent with the repeated conversion per query block being a large part of the fused kernel's cost, and no experiment removed it.
- Nothing here says anything about Hopper or data-centre Blackwell: the RTX 4090 is sm_89.


## Repaired files (the first version of the FKL program wrote a comma into a field)

The `schedule` string of the quantised variants of the first version of `e16_attention.cu` was `cp.async quantised bytes, smem dequant`, with a comma, and the CSV writer does not quote: 40 rows of each `run*/e16_attention.csv` (the fused int8 and fp8 rows, 120 rows in all) had 27 fields instead of 26. No value was lost: the columns before `schedule` were intact and the shifted columns held the same text. The three files were repaired by one text replacement, the field `cp.async quantised bytes, smem dequant` wrapped in double quotes (the text is unchanged, a standard CSV reader now gives 26 fields in every row), and the `_clock.csv` files and the summaries were regenerated from the repaired files (the summary tables are identical cell by cell to the ones made from the originals with a tolerant reader). The source is fixed (the string has no comma any more); the binary that made this run is the one in `binary-sha256.txt`. The originals are kept in `original-csv/` (small); their sha256 (and those of the repaired files) are below, so the repair can be checked: `sed 's/cp.async quantised bytes, smem dequant/"cp.async quantised bytes, smem dequant"/'` applied to an original must give the repaired file byte for byte.

```
originals (before the repair), as written by the program:
a2734a7706513c19c9a9da3f24ab87eab9c16767ea215559c4ba3315abdf1e8f  run1/e16_attention.csv (original)
096a94f027b9fd7fe23974798edaa594965120467928fd22772cbab831cf4f9a  run2/e16_attention.csv (original)
7e835077a3fbc1e2382278bb202ff7af0d5c9152b568843de43467083d36c8d1  run3/e16_attention.csv (original)
repaired (committed):
069b71e9ea372c0d3260584ba6e93914b02a930e3fc9c4264cc2dedb6f8d3b17  run1/e16_attention.csv (repaired)
a24fe31525c677e8d92452b8e43165469f6cc2a5aaf00241e05cb323485385a2  run2/e16_attention.csv (repaired)
f12f3bb7101ae504e8fb3244f3265c48e9f24a6e83deaf9abfb930c01002e506  run3/e16_attention.csv (repaired)
```
