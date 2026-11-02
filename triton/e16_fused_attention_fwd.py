# Copyright 2026 Oscar Amoros Huguet
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""E16 baseline: the fused-attention forward kernel of Triton, bf16 in and out, fp32 accumulation.

The algorithm and the structure of the loop are those of the fused-attention tutorial of Triton (python/tutorials/06-fused-attention.py: one program per
(block of queries, head), the K and V blocks loop with the online softmax in the exp2 domain, tl.dot on the tensor cores, the causal case stops at the diagonal and
masks the diagonal block); this file is the forward part only, with plain pointer arithmetic instead of the block pointers of the tutorial (the tensor descriptors and the warp
specialisation of the Hopper and Blackwell data-centre variants of the tutorial are not used: neither the RTX 4090 nor the RTX 5050 has them) and the tile sizes, number of stages
and number of warps chosen by triton.autotune among a small list, as the tutorial does. The autotune runs on the first call of a configuration (outside the timed region of
pytorch-attention/e16_baselines.py).
"""
import torch
import triton
import triton.language as tl


def _configs():
    out = []
    for bm in (64, 128):
        for bn in (32, 64):
            for stages in (2, 3, 4):
                for warps in (4, 8):
                    out.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_stages=stages, num_warps=warps))
    return out


@triton.autotune(configs=_configs(), key=["N_CTX", "HEAD_DIM", "CAUSAL"])
@triton.jit
def _attn_fwd(Q, K, V, Out, sm_scale,  #
              stride_qh, stride_qm, stride_kh, stride_kn, stride_vh, stride_vn, stride_oh, stride_om,  #
              N_CTX,  #
              HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, CAUSAL: tl.constexpr):
    start_m = tl.program_id(0)
    off_h = tl.program_id(1).to(tl.int64)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    q_ptrs = Q + off_h * stride_qh + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_ptrs = K + off_h * stride_kh + offs_n[None, :] * stride_kn + offs_d[:, None]
    v_ptrs = V + off_h * stride_vh + offs_n[:, None] * stride_vn + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < N_CTX, other=0.0)
    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.full([BLOCK_M], 1.0, dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    qk_scale = sm_scale * 1.44269504   # the softmax in the exp2 domain: scale / ln 2
    hi = N_CTX
    if CAUSAL:
        hi = tl.minimum((start_m + 1) * BLOCK_M, N_CTX)
    for start_n in range(0, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        cols = start_n + offs_n
        k = tl.load(k_ptrs + start_n * stride_kn, mask=cols[None, :] < N_CTX, other=0.0)
        qk = tl.dot(q, k) * qk_scale
        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, -1.0e6)
        qk = tl.where(cols[None, :] < N_CTX, qk, -1.0e6)
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp2(qk - m_ij[:, None])
        alpha = tl.math.exp2(m_i - m_ij)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]
        v = tl.load(v_ptrs + start_n * stride_vn, mask=cols[:, None] < N_CTX, other=0.0)
        acc = tl.dot(p.to(v.dtype), v, acc)
        m_i = m_ij
    acc = acc / l_i[:, None]
    o_ptrs = Out + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(o_ptrs, acc.to(Out.dtype.element_ty), mask=offs_m[:, None] < N_CTX)


def attention_forward(q, k, v, causal):
    """q, k, v: (1, BH, S, D) bf16 CUDA tensors, last dimension contiguous; returns the output of the same shape."""
    assert q.dim() == 4 and q.shape == k.shape == v.shape and q.shape[0] == 1
    _, bh, seq, d = q.shape
    out = torch.empty_like(q)
    grid = lambda meta: (triton.cdiv(seq, meta["BLOCK_M"]), bh)  # noqa: E731
    _attn_fwd[grid](q, k, v, out, d ** -0.5,  #
                    q.stride(1), q.stride(2), k.stride(1), k.stride(2), v.stride(1), v.stride(2), out.stride(1), out.stride(2),  #
                    seq, HEAD_DIM=d, CAUSAL=bool(causal))
    return out
