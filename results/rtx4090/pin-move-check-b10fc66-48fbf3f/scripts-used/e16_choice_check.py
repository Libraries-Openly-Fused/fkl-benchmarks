#!/usr/bin/env python3
"""C-PM-CUDA: at every configuration that E16 measured, does the ported automatic choice name the kernel that was measured? (CPU only: reads
the record's CSVs; nothing is built or run.)

  e16_choice_check.py <e16_attention.csv>... [--query-tile <e16_query_tile.csv>...] [--sms 128] [--out report.txt]

The tiles. Before FKL #435, executeFlashAttentionMma<D> (BLOCK_Q = BLOCK_KV = 0, NUM_WARPS = 4) chose them inside; since #435,
fk::flashAttentionMmaTiles<D, K, V>(bh, seqQ, causal, {}, 4) returns them and the ported launch code names that FlashAttentionMmaDPP. The two
are the same conditions (b10fc66 flash_attention_mma.h executeFlashAttentionMma; 48fbf3f flash_attention_mma.h flashAttentionMmaTiles), with
no block-sparse mask:
  raw bf16 K and V (mma_raw, the *_unfused variants, the epilogue pair):
    D <= 64:  causal ? (seqQ >= 8192 or (seqQ >= 4096 and bh >= 64)) : bh * seqQ >= 65536  -> 128 x 64 x 4;
              else causal and seqQ >= 1024 -> 64 x 128 x 4; else 64 x 64 x 4
    D == 128: seqQ >= 16384 -> 128 x 128 x 8; else 64 x 64 x 4
  other K and V reads (the int8 and e4m3 fused variants): (D <= 64 ? 128 : 64) x 32 x 4.
Each MMA row's block_q, block_kv and num_warps (from the measured kernel's template arguments) must be this choice.
The split. Both sides decide it with the same formula (executeFlashAttentionMma's split block at b10fc66; flashAttentionMmaSplits at
48fbf3f), from wave = blocks per SM of the unsplit kernel x SMs. blocks per SM: the measured kernel's (the kernel_facts column,
cudaOccupancyMaxActiveBlocksPerMultiprocessor of that kernel in the run); SMs: --sms (the RTX 4090 has 128). Every row must give 1 split, and
its split_kv column must be false.
--query-tile: e16_query_tile's explicit tiles. At b10fc66 executeFlashAttentionMma<D, BQ, BKV, NW> also split explicit tiles by the same
formula; at 48fbf3f FlashAttentionMmaDPP's executor never splits. Each row's split count by the formula must be 1.
"""
import argparse
import csv
import re

ap = argparse.ArgumentParser()
ap.add_argument("csvs", nargs="+"); ap.add_argument("--query-tile", nargs="*", default=[]); ap.add_argument("--sms", type=int, default=128)
ap.add_argument("--out")
a = ap.parse_args()
RAW = {"mma_raw", "mma_int8kv_unfused", "mma_fp8kv_unfused", "mma_epilogue_fused", "mma_epilogue_separate"}
QUANT = {"mma_int8kv_fused", "mma_fp8kv_fused"}


def tiles(variant, d, s, bh, causal):
    if variant in RAW:
        if d <= 64:
            want128 = (s >= 8192 or (s >= 4096 and bh >= 64)) if causal else bh * s >= 65536
            if want128:
                return (128, 64, 4)
            if causal and s >= 1024:
                return (64, 128, 4)
            return (64, 64, 4)
        if d == 128:
            return (128, 128, 8) if s >= 16384 else (64, 64, 4)
        return (64, 64, 4)
    return (128 if d <= 64 else 64, 32, 4)


def splits(bpm, bq, bkv, bh, s, causal, sms):
    wave = (bpm if bpm > 0 else 1) * sms
    numq = (s + bq - 1) // bq
    grid = numq * bh
    avg = max(1, (s // 2 if causal else s) // bkv)
    num = grid * 10
    under = num < wave * 7 if causal else num < wave * 9 // 2
    n = 1
    if under and avg >= 8:
        bysat = (wave + grid - 1) // grid
        bywork = avg // 4
        n = max(min(min(bysat, bywork), 16), 1)
    return n


def rows(path):
    return list(csv.DictReader(l for l in open(path) if not l.startswith("#")))


def bpm_of(facts):
    for seg in facts.split(" ; "):
        if "FlashAttention" in seg.split(" grid ")[0]:
            m = re.search(r"blocks_per_SM (\d+)", seg)
            return int(m.group(1)) if m else None
    return None


lines, bad, n, kernels = [], 0, 0, set()
for p in a.csvs:
    for r in rows(p):
        v = r["variant"]
        if v not in RAW | QUANT:
            continue
        n += 1
        d, s, bh, causal = int(r["d"]), int(r["seq"]), int(r["bh"]), r["causal"] == "1"
        got = (int(r["block_q"]), int(r["block_kv"]), int(r["num_warps"]))
        want = tiles(v, d, s, bh, causal)
        bpm = bpm_of(r["kernel_facts"])
        sp = splits(bpm or 0, want[0], want[1], bh, s, causal, a.sms) if bpm else None
        ok = got == want and sp == 1 and r["split_kv"] == "false"
        bad += not ok
        kv = "raw" if v in RAW else v.split("_")[1]
        epi = "epilogue" if v == "mma_epilogue_fused" else "identity"
        kernels.add((d, kv, epi, got))
        if not ok:
            lines.append(f"  MISMATCH {p}: {v} D {d} S {s} BH {bh} causal {int(causal)}: measured {got} split_kv {r['split_kv']}, choice {want}, splits {sp} (blocks per SM {bpm})")
lines.insert(0, f"e16_attention: {n} MMA rows in {len(a.csvs)} files; the choice names the measured tiles and gives 1 split in {n - bad}, not in {bad}")
lines.insert(1, f"  distinct MMA kernels measured (D, K/V read, epilogue, tiles): {len(kernels)}")
for k in sorted(kernels):
    lines.insert(2, f"    {k}")
qn = qbad = 0
for p in a.query_tile:
    for r in rows(p):
        if r["variant"].startswith("int8nodq"):
            pass
        if "block_q" not in r or not r["block_q"]:
            continue
        qn += 1
        d, s, bh, causal = int(r["d"]), int(r["seq"]), int(r["bh"]), r["causal"] == "1"
        bq, bkv = int(r["block_q"]), int(r["block_kv"])
        bpm = bpm_of(r["kernel_facts"])
        sp = splits(bpm or 0, bq, bkv, bh, s, causal, a.sms) if bpm else None
        if sp != 1 or r.get("split_kv", "false") != "false":
            qbad += 1
            lines.append(f"  QUERY-TILE SPLIT {p}: {r['variant']} D {d} S {s} BH {bh}: {sp} splits (blocks per SM {bpm}), split_kv {r.get('split_kv')}")
if a.query_tile:
    lines.append(f"e16_query_tile: {qn} rows in {len(a.query_tile)} files; the base's split formula gives 1 split in {qn - qbad}, not in {qbad}")
text = "\n".join(lines)
print(text)
if a.out:
    open(a.out, "w").write(text + "\n")
