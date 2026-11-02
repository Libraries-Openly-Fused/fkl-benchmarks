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
