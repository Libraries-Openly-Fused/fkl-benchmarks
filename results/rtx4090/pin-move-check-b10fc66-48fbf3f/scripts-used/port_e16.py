#!/usr/bin/env python3
"""C-PM-CUDA: ports fkl-self/attention/e16_attention.cu and e16_query_tile.cu to FKL #435 (Oscar's QA3, as the coordinator passed it on).

  port_e16.py <worktree>

- Every automatic call fk::executeFlashAttentionMma<D>(qr, kr, vr, o, ..., stream[, -1.f, epilogue]) becomes, in place in its launch
  lambda (no helper), what that function did inside: fk::flashAttentionMmaTiles names the FlashAttentionMmaDPP (the branches are the tiles
  the old function's run-time choice could take: 128 x 64, 64 x 128 and 64 x 64 for raw bf16 K and V at D <= 64; 128 x 128 x 8 and
  64 x 64 above; D <= 64 ? 128 : 64 by 32 for the other reads), fk::flashAttentionMmaSplits decides the split-KV, and a split runs
  FlashAttentionMmaSplitKVDPP into a partial buffer from cudaMallocAsync, then its CombineDPP, then cudaFreeAsync, as the old function did.
- Explicit tiles fk::executeFlashAttentionMma<D, BQ, BKV, NW>(...) become fk::executeOperations<fk::FlashAttentionMmaDPP<GPU_NVIDIA,
  __nv_bfloat16, D, Q, K, V, AttentionIdentityEpilogue, BQ, BKV, NW>>(stream, ...).
- fk::executeFlashAttention<D>(...) becomes fk::executeOperations<fk::FlashAttentionDPP<GPU_NVIDIA, __nv_bfloat16, D, Q, K, V>>(stream, ...).
- parseAttentionKernel reads the new kernel names: launchDPP_Kernel<fk::FlashAttentionMmaDPP<PA, OT, D, Q, K, V, E, BQ, BKV, NW, SM>, ...>,
  launchDPP_Kernel<fk::FlashAttentionMmaSplitKVDPP<the same>, ...> (split_kv true) and launchDPP_Kernel<fk::FlashAttentionDPP<PA, OT, D, Q,
  K, V, E, BLOCK_N, WARPS_PER_BLOCK>, ...>: the same fields, one position later (the leading ParArch), SPLIT_KV from the DPP's name.
- The comments and the CSV note strings that name the old calls.
Every replacement must match the expected number of times. Prints what it changed.
"""
import re
import sys

W = sys.argv[1] + "/fkl-self/attention/"


def branch(ind, bq, bkv, nw, epi):
    e = ", -1.f, epilogue" if epi else ""
    ec = ", epilogue" if epi else ""
    E = "EpiIOp" if epi else "fk::AttentionIdentityEpilogue"
    i = ind
    return [
        f"{i}using MmaDPP = fk::FlashAttentionMmaDPP<fk::ParArch::GPU_NVIDIA, __nv_bfloat16, D, QIOp, KIOp, VIOp, {E}, {bq}, {bkv}, {nw}>;",
        f"{i}const int splits = fk::flashAttentionMmaSplits<MmaDPP>(pt.bh, pt.seq, pt.seq, pt.causal);",
        f"{i}if (splits == 1) {{",
        f"{i}    fk::executeOperations<MmaDPP>(stream, qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal{e});",
        f"{i}}} else {{",
        f"{i}    using SplitDPP = fk::FlashAttentionMmaSplitKVDPP<fk::ParArch::GPU_NVIDIA, __nv_bfloat16, D, QIOp, KIOp, VIOp, {E}, {bq}, {bkv}, {nw}>;",
        f"{i}    float* partial = nullptr;",
        f"{i}    FKLB_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&partial),",
        f"{i}                                    fk::flashAttentionMmaPartialFloats(pt.bh, pt.seq, splits, D) * sizeof(float), stream.getCUDAStream()));",
        f"{i}    fk::executeOperations<SplitDPP>(stream, qr, kr, vr, partial, splits, pt.bh, pt.seq, pt.seq, pt.causal{e});",
        f"{i}    fk::executeOperations<typename SplitDPP::CombineDPP>(stream, partial, o, pt.bh, pt.seq, splits{ec});",
        f"{i}    FKLB_CUDA_CHECK(cudaFreeAsync(partial, stream.getCUDAStream()));",
        f"{i}}}",
    ]


def automatic(ind, epi):
    i = ind
    out = [
        f"{i}// the library's automatic choice (FKL #435: the caller makes it): fk::flashAttentionMmaTiles names the FlashAttentionMmaDPP, and",
        f"{i}// fk::flashAttentionMmaSplits decides the split-KV (then FlashAttentionMmaSplitKVDPP and its CombineDPP, two kernels)",
        f"{i}using QIOp = std::decay_t<decltype(qr)>;",
        f"{i}using KIOp = std::decay_t<decltype(kr)>;",
        f"{i}using VIOp = std::decay_t<decltype(vr)>;",
    ]
    if epi:
        out.append(f"{i}using EpiIOp = std::decay_t<decltype(epilogue)>;")
    out += [
        f"{i}const fk::FlashAttentionMmaTiles tiles = fk::flashAttentionMmaTiles<D, KIOp, VIOp>(pt.bh, pt.seq, pt.causal);",
        f"{i}if constexpr (fk::isRawBf16Read<KIOp> && fk::isRawBf16Read<VIOp> && D <= 64) {{",
        f"{i}    if (tiles.blockQ == 128 && tiles.blockKV == 64 && tiles.numWarps == 4) {{",
        *branch(i + "        ", 128, 64, 4, epi),
        f"{i}    }} else if (tiles.blockQ == 64 && tiles.blockKV == 128 && tiles.numWarps == 4) {{",
        *branch(i + "        ", 64, 128, 4, epi),
        f"{i}    }} else if (tiles.blockQ == 64 && tiles.blockKV == 64 && tiles.numWarps == 4) {{",
        *branch(i + "        ", 64, 64, 4, epi),
        f"{i}    }} else {{",
        f"{i}        throw std::logic_error(\"flashAttentionMmaTiles returned tiles this variant does not name\");",
        f"{i}    }}",
        f"{i}}} else if constexpr (fk::isRawBf16Read<KIOp> && fk::isRawBf16Read<VIOp>) {{",
        f"{i}    if (tiles.blockQ == 128 && tiles.blockKV == 128 && tiles.numWarps == 8) {{",
        *branch(i + "        ", 128, 128, 8, epi),
        f"{i}    }} else if (tiles.blockQ == 64 && tiles.blockKV == 64 && tiles.numWarps == 4) {{",
        *branch(i + "        ", 64, 64, 4, epi),
        f"{i}    }} else {{",
        f"{i}        throw std::logic_error(\"flashAttentionMmaTiles returned tiles this variant does not name\");",
        f"{i}    }}",
        f"{i}}} else {{",
        f"{i}    if (tiles.blockQ == (D <= 64 ? 128 : 64) && tiles.blockKV == 32 && tiles.numWarps == 4) {{",
        *branch(i + "        ", "(D <= 64 ? 128 : 64)", 32, 4, epi),
        f"{i}    }} else {{",
        f"{i}        throw std::logic_error(\"flashAttentionMmaTiles returned tiles this variant does not name\");",
        f"{i}    }}",
        f"{i}}}",
    ]
    return "\n".join(out)


def explicit(ind, bq, bkv, nw):
    i = ind
    return "\n".join([
        f"{i}fk::executeOperations<fk::FlashAttentionMmaDPP<fk::ParArch::GPU_NVIDIA, __nv_bfloat16, D, std::decay_t<decltype(qr)>, std::decay_t<decltype(kr)>,",
        f"{i}                                                std::decay_t<decltype(vr)>, fk::AttentionIdentityEpilogue, {bq}, {bkv}, {nw}>>(",
        f"{i}    stream, qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal);",
    ])


PARSE_OLD = '''    const std::vector<std::string> mma = templateArgs(dem, "launchFlashAttentionMmaDPP_Kernel");
    if (mma.size() >= 9) {   // OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp, SPLIT_KV
        f.blockQ = std::atoi(mma[6].c_str());
        f.blockKV = std::atoi(mma[7].c_str());
        f.numWarps = std::atoi(mma[8].c_str());
        f.splitKV = mma.size() > 10 ? mma[10] : "false";
        const bool kRaw = has(mma[3], "Bf16AttentionRead") && has(mma[4], "Bf16AttentionRead");
        const bool kQuant = (has(mma[3], "Int8TokenDequantRead") && has(mma[4], "Int8TokenDequantRead")) || (has(mma[3], "Fp8TokenDequantRead") && has(mma[4], "Fp8TokenDequantRead"));
        f.schedule = kRaw ? "cp.async bf16" : (kQuant ? "cp.async quantised bytes + smem dequant" : "register prefetch");
        return;
    }
    const std::vector<std::string> simt = templateArgs(dem, "launchFlashAttentionDPP_Kernel");
    if (simt.size() >= 8) {   // OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK
        f.blockQ = std::atoi(simt[7].c_str());
        f.blockKV = std::atoi(simt[6].c_str());
        f.numWarps = std::atoi(simt[7].c_str());'''
PARSE_NEW = '''    // FKL #435: every kernel is fk::launchDPP_Kernel<DPP, ...>; its DPP's template arguments start with the ParArch. The split-KV producer is
    // the DPP fk::FlashAttentionMmaSplitKVDPP (the same arguments as fk::FlashAttentionMmaDPP), so SPLIT_KV is the name of the DPP
    const std::vector<std::string> split = templateArgs(dem, "fk::FlashAttentionMmaSplitKVDPP");
    const std::vector<std::string> mma = split.empty() ? templateArgs(dem, "fk::FlashAttentionMmaDPP") : split;
    if (mma.size() >= 10) {   // ParArch, OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_Q, BLOCK_KV, NUM_WARPS, ScoreModOp
        f.blockQ = std::atoi(mma[7].c_str());
        f.blockKV = std::atoi(mma[8].c_str());
        f.numWarps = std::atoi(mma[9].c_str());
        f.splitKV = split.empty() ? "false" : "true";
        const bool kRaw = has(mma[4], "Bf16AttentionRead") && has(mma[5], "Bf16AttentionRead");
        const bool kQuant = (has(mma[4], "Int8TokenDequantRead") && has(mma[5], "Int8TokenDequantRead")) || (has(mma[4], "Fp8TokenDequantRead") && has(mma[5], "Fp8TokenDequantRead"));
        f.schedule = kRaw ? "cp.async bf16" : (kQuant ? "cp.async quantised bytes + smem dequant" : "register prefetch");
        return;
    }
    const std::vector<std::string> simt = templateArgs(dem, "fk::FlashAttentionDPP");
    if (simt.size() >= 9) {   // ParArch, OT, HEAD_DIM, QIOp, KIOp, VIOp, EpilogueIOp, BLOCK_N, WARPS_PER_BLOCK
        f.blockQ = std::atoi(simt[8].c_str());
        f.blockKV = std::atoi(simt[7].c_str());
        f.numWarps = std::atoi(simt[8].c_str());'''
FACTS_OLD = "    // the first attention kernel of the call, from its demangled template arguments (BLOCK_Q, BLOCK_KV, NUM_WARPS, SPLIT_KV) and from"
FACTS_NEW = "    // the first attention kernel of the call, from its demangled template arguments (BLOCK_Q, BLOCK_KV, NUM_WARPS; SPLIT_KV: the split DPP) and from"


def rep(s, a, b, n, what):
    c = s.count(a)
    if c != n:
        sys.exit(f"{what}: {c} matches, expected {n}: {a[:100]}")
    return s.replace(a, b)


# ---- e16_attention.cu
p = W + "e16_attention.cu"
s = open(p).read()
IND = " " * 16
s = rep(s, IND + "fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);", automatic(IND, False), 6, "attention identity")
s = rep(s, IND + "fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream, -1.f, epilogue);", automatic(IND, True), 1, "attention epilogue")
s = rep(s, " " * 20 + "fk::executeFlashAttention<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);",
        "\n".join([" " * 20 + "fk::executeOperations<fk::FlashAttentionDPP<fk::ParArch::GPU_NVIDIA, __nv_bfloat16, D, std::decay_t<decltype(qr)>,",
                   " " * 20 + "                                          std::decay_t<decltype(kr)>, std::decay_t<decltype(vr)>>>(",
                   " " * 20 + "    stream, qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal);"]), 1, "attention simt")
s = rep(s, PARSE_OLD, PARSE_NEW, 1, "attention parse")
s = rep(s, FACTS_OLD, FACTS_NEW, 1, "attention facts comment")
s = rep(s, '"into a CUDA graph; the library chooses the tiles (no tuning: <D, 0, 0, 4> and <D, 32, 4> are the library\'s defaults); ',
        '"into a CUDA graph; the library chooses the tiles (no tuning: fk::flashAttentionMmaTiles and fk::flashAttentionMmaSplits, the choice of '
        'executeFlashAttentionMma<D, 0, 0, 4> before FKL #435, and FlashAttentionDPP\'s default <32, 4>); ', 1, "attention note")
open(p, "w").write(s)
print("e16_attention.cu: 6 automatic + 1 automatic with the epilogue + 1 SIMT call, parseAttentionKernel, 1 comment, 1 CSV note")

# ---- e16_query_tile.cu
p = W + "e16_query_tile.cu"
s = open(p).read()
s = rep(s, IND + "fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);", automatic(IND, False), 1, "query tile automatic")
for bq, bkv, nw, n in ((32, 32, 2, 3), (64, 32, 4, 3), (128, 32, 4, 3)):
    s = rep(s, IND + f"fk::executeFlashAttentionMma<D, {bq}, {bkv}, {nw}>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);",
            explicit(IND, bq, bkv, nw), n, f"query tile {bq}x{bkv}x{nw}")
s = rep(s, PARSE_OLD, PARSE_NEW, 1, "query tile parse")
s = rep(s, FACTS_OLD, FACTS_NEW, 1, "query tile facts comment")
s = rep(s, "EXPLICIT tile arguments (executeFlashAttentionMma<D, BLOCK_Q, BLOCK_KV, NUM_WARPS, bf16>, which bypasses the library's heuristic):",
        "EXPLICIT tile arguments (fk::executeOperations<fk::FlashAttentionMmaDPP<GPU_NVIDIA, bf16, D, Q, K, V, identity, BLOCK_Q, BLOCK_KV, NUM_WARPS>>, "
        "which bypasses the library's heuristic; executeFlashAttentionMma<D, BLOCK_Q, BLOCK_KV, NUM_WARPS, bf16> before FKL #435):", 1, "query tile header")
s = rep(s, "Every variant is executeFlashAttentionMma with EXPLICIT tiles (variant name: bq = BLOCK_Q, kv = BLOCK_KV, w = warps; raw_default is ",
        "Every variant is FlashAttentionMmaDPP with EXPLICIT tiles (variant name: bq = BLOCK_Q, kv = BLOCK_KV, w = warps; raw_default is ", 1, "query tile note")
open(p, "w").write(s)
print("e16_query_tile.cu: 1 automatic + 9 explicit calls, parseAttentionKernel, 1 comment, the header line, 1 CSV note")
