/* Task C-NV harness, attention DPPs on ParArch::GPU_NVIDIA: SoftmaxDPP, FlashAttentionDPP (SIMT), FlashAttentionMmaDPP
   (explicit tiles, the automatic choice, int8 K/V), the split-KV path, and FlashDecodeDPP.
   Base build (b10fc66): executeSoftmax, executeFlashAttention<D, ...>, executeFlashAttentionMma<D, BQ, BKV, NW>, which
   chooses the tiles when BQ = BKV = 0 and decides the split-KV itself (then it runs the split kernel and
   flashFwdSplitCombineKernel), and executeFlashDecode<D>.
   Head build (e322cad): executeOperations<DPP>(stream, ...). For the automatic choice the head harness does what the PR
   says the caller does: flashAttentionMmaTiles names the FlashAttentionMmaDPP, flashAttentionMmaSplits decides the split,
   and with more than 1 split it runs FlashAttentionMmaSplitKVDPP into a partial buffer and then its CombineDPP.
   The split cases ("mma_split_*"), on the three shapes of the PR's testMmaSplit:
   - identity epilogue: the base's automatic split against the head's split + combine with flashAttentionMmaSplits' count;
     compared across the commits bit for bit;
   - epilogue o * 2 + 0.5: the base's split path drops it (the bug the PR fixes), so its file is compared with the base's
     identity file. At the head, split + combine and the unsplit FlashAttentionMmaDPP ("mma_unsplit_*", head only) are
     checked as the PR's test does at 48fbf3f (utest_flash_attention_mma.h testMmaSplit): against the double oracle on the
     bf16-rounded inputs ("mma_split_*_oracle_f64", computed here on the host as the test's cpuAttention does), within
     TOL = 3e-3 without the epilogue and 2 * TOL of 2 * oracle + 0.5 with it. compare_harness.py does the checks.
   The split count is printed: at the head flashAttentionMmaSplits; at both commits also the count of the base's formula
   (executeFlashAttentionMma's code, recomputed here with the occupancy of the unsplit kernel of that commit).
   See pr435_common.h. */
#include "pr435_common.h"

#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>
#include <fused_kernel/algorithms/attention/flash_decode.h>
#include <fused_kernel/algorithms/attention/softmax.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/core/data/ptr_nd.h>

#include <cuda_bf16.h>

#include <algorithm>
#include <cmath>
#include <type_traits>

using namespace fk;
using namespace h435;

namespace {

constexpr ParArch GPU = ParArch::GPU_NVIDIA;
using GpuStream = Stream_<ParArch::GPU_NVIDIA>;
using Identity = AttentionIdentityEpilogue;

std::vector<float> floats(const size_t n, const int salt, const float lo = -1.f, const float hi = 1.f) {
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = fval(static_cast<int>(i), 0, salt, lo, hi);
    return v;
}
std::vector<__nv_bfloat16> bf16s(const size_t n, const int salt) {
    std::vector<__nv_bfloat16> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = __float2bfloat16(fval(static_cast<int>(i), 0, salt, -1.f, 1.f));
    return v;
}
std::vector<int8_t> int8s(const size_t n, const int salt) {
    std::vector<int8_t> v(n);
    for (size_t i = 0; i < n; ++i) v[i] = static_cast<int8_t>(static_cast<int>(uval(static_cast<int>(i), 0, salt)) - 128);
    return v;
}

// the epilogue of the split cases and of the SIMT epilogue case: o * 2 + 0.5
inline auto epilogue() { return Mul<float>::build(2.f).then(Add<float>::build(0.5f)); }
using Epi = std::decay_t<decltype(epilogue())>;

// ---- softmax ----
void softmaxCases(GpuStream& s) {
    {
        const int rows = 37, width = 301;
        Ptr2D<float> in(width, rows, 0, MemType::DeviceAndPinned);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < width; ++x) in.at(Point{x, y, 0}) = fval(x, y, 501);
        in.upload(s);
        Ptr2D<float> out(width, rows, 0, MemType::Device);
        cudaCheck(cudaMemset2D(out.ptr().data, out.dims().pitch, 0xff, width * sizeof(float), rows), "cudaMemset2D");
        const auto inRead = PerThreadRead<ND::_2D, float>::build(in.ptr());
#ifdef PR435_HEAD
        executeOperations<SoftmaxDPP<GPU, std::decay_t<decltype(inRead)>, float>>(s, inRead, out, rows, width);
#else
        executeSoftmax(inRead, out, rows, width, s);
#endif
        s.sync();
        std::vector<float> h(static_cast<size_t>(rows) * width);
        cudaCheck(cudaMemcpy2D(h.data(), width * sizeof(float), out.ptr().data, out.dims().pitch, width * sizeof(float), rows,
                               cudaMemcpyDeviceToHost), "cudaMemcpy2D");
        writeBin("softmax_f32_37x301", h.data(), h.size() * sizeof(float));
    }
    {   // a fused prologue: the input * 0.7
        const int rows = 8, width = 1024;
        Ptr2D<float> in(width, rows, 0, MemType::DeviceAndPinned);
        for (int y = 0; y < rows; ++y)
            for (int x = 0; x < width; ++x) in.at(Point{x, y, 0}) = fval(x, y, 502);
        in.upload(s);
        Ptr2D<float> out(width, rows, 0, MemType::Device);
        cudaCheck(cudaMemset2D(out.ptr().data, out.dims().pitch, 0xff, width * sizeof(float), rows), "cudaMemset2D");
        const auto inRead = PerThreadRead<ND::_2D, float>::build(in.ptr()).then(Mul<float>::build(0.7f));
#ifdef PR435_HEAD
        executeOperations<SoftmaxDPP<GPU, std::decay_t<decltype(inRead)>, float>>(s, inRead, out, rows, width);
#else
        executeSoftmax(inRead, out, rows, width, s);
#endif
        s.sync();
        std::vector<float> h(static_cast<size_t>(rows) * width);
        cudaCheck(cudaMemcpy2D(h.data(), width * sizeof(float), out.ptr().data, out.dims().pitch, width * sizeof(float), rows,
                               cudaMemcpyDeviceToHost), "cudaMemcpy2D");
        writeBin("softmax_f32_fused_8x1024", h.data(), h.size() * sizeof(float));
    }
}

// ---- FlashAttentionDPP (SIMT) ----
template <int D, typename Q, typename K, typename V, typename E = Identity>
void runSimt(GpuStream& s, const Q& q, const K& k, const V& v, float* o, const int bh, const int sq, const int sk,
             const bool causal, const E& epi = {}) {
#ifdef PR435_HEAD
    executeOperations<FlashAttentionDPP<GPU, float, D, Q, K, V, E, 32, 4>>(s, q, k, v, o, bh, sq, sk, causal, -1.f, epi);
#else
    executeFlashAttention<D, 32, 4>(q, k, v, o, bh, sq, sk, causal, s, -1.f, epi);
#endif
}
template <int D>
void simtCase(GpuStream& s, const char* name, const int bh, const int sq, const int sk, const bool causal,
              const bool int8kv, const bool withEpilogue, const int salt) {
    const size_t nQ = static_cast<size_t>(bh) * sq * D, nK = static_cast<size_t>(bh) * sk * D;
    float* q = deviceCopy(floats(nQ, salt));
    float* o = deviceSentinel<float>(nQ);
    const auto qR = makeAttentionRead(q, bh, sq, D);
    if (int8kv) {
        int8_t* k8 = deviceCopy(int8s(nK, salt + 1));
        int8_t* v8 = deviceCopy(int8s(nK, salt + 2));
        float* ks = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 3, 0.002f, 0.02f));
        float* vs = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 4, 0.002f, 0.02f));
        runSimt<D>(s, qR, makeInt8KVRead(k8, ks, bh, sk, D), makeInt8KVRead(v8, vs, bh, sk, D), o, bh, sq, sk, causal);
        dumpDevice(name, o, nQ);
        cudaFree(k8); cudaFree(v8); cudaFree(ks); cudaFree(vs);
    } else {
        float* k = deviceCopy(floats(nK, salt + 1));
        float* v = deviceCopy(floats(nK, salt + 2));
        const auto kR = makeAttentionRead(k, bh, sk, D);
        const auto vR = makeAttentionRead(v, bh, sk, D);
        if (withEpilogue) runSimt<D>(s, qR, kR, vR, o, bh, sq, sk, causal, epilogue());
        else runSimt<D>(s, qR, kR, vR, o, bh, sq, sk, causal);
        dumpDevice(name, o, nQ);
        cudaFree(k); cudaFree(v);
    }
    cudaFree(q); cudaFree(o);
}

// ---- FlashAttentionMmaDPP ----
// the split count of the base's executeFlashAttentionMma for these tiles, recomputed from its code with the occupancy of
// the unsplit kernel of this commit (bpm)
int baseFormulaSplits(const int bpm, const int bh, const int sq, const int sk, const bool causal, const int BQ,
                      const int BKV) {
    int dev = 0, sms = 0;
    cudaCheck(cudaGetDevice(&dev), "cudaGetDevice");
    cudaCheck(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev), "cudaDeviceGetAttribute");
    const int wave = (bpm > 0 ? bpm : 1) * sms;
    const int numQ = (sq + BQ - 1) / BQ;
    const long gridBlocks = (long)numQ * bh;
    const int avgIter = std::max(1, (int)(((long)(causal ? sk / 2 : sk)) / BKV));
    const long num = gridBlocks * 10;
    const bool under = causal ? (num < (long)wave * 7) : (num < (long)wave * 9 / 2);
    int splits = 1;
    if (under && avgIter >= 8) {
        const int bySat = (int)((wave + gridBlocks - 1) / gridBlocks);
        const int byWork = avgIter / 4;
        splits = std::max(std::min(std::min(bySat, byWork), 16), 1);
    }
    return splits;
}
template <int D, int BQ, int BKV, int NW, typename Q, typename K, typename V, typename E>
int unsplitOccupancy() {
#ifdef PR435_HEAD
    using DPP = FlashAttentionMmaDPP<GPU, float, D, Q, K, V, E, BQ, BKV, NW>;
    const auto kernel = launchDPP_Kernel<DPP, typename DPP::Params>;
#else
    using DPP = FlashAttentionMmaDPP<float, D, Q, K, V, E, BQ, BKV, NW, NoScoreMod>;
    const auto kernel = launchFlashAttentionMmaDPP_Kernel<float, D, Q, K, V, E, BQ, BKV, NW, NoScoreMod>;
#endif
    if (DPP::SMEM_BYTES > 48000)
        cudaCheck(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, DPP::SMEM_BYTES),
                  "cudaFuncSetAttribute");
    int bpm = 0;
    cudaCheck(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&bpm, kernel, DPP::THREADS, DPP::SMEM_BYTES), "occupancy");
    return bpm;
}

// explicit tiles. Base: executeFlashAttentionMma<D, BQ, BKV, NW> (which splits by itself when its formula says so).
// Head: unsplit -> FlashAttentionMmaDPP; split -> the split count of flashAttentionMmaSplits, SplitKVDPP + CombineDPP.
// Returns the split count used (head) or the base formula's count (base).
template <int D, int BQ, int BKV, int NW, typename Q, typename K, typename V, typename E = Identity>
int runMma(GpuStream& s, const bool split, const Q& q, const K& k, const V& v, float* o, const int bh, const int sq,
           const int sk, const bool causal, const E& epi = {}) {
    const int formula = baseFormulaSplits(unsplitOccupancy<D, BQ, BKV, NW, Q, K, V, E>(), bh, sq, sk, causal, BQ, BKV);
#ifdef PR435_HEAD
    using MmaDPP = FlashAttentionMmaDPP<GPU, float, D, Q, K, V, E, BQ, BKV, NW>;
    if (!split) {
        executeOperations<MmaDPP>(s, q, k, v, o, bh, sq, sk, causal, -1.f, epi);
        return 1;
    }
    using SplitDPP = FlashAttentionMmaSplitKVDPP<GPU, float, D, Q, K, V, E, BQ, BKV, NW>;
    const int splits = flashAttentionMmaSplits<MmaDPP>(bh, sq, sk, causal);
    std::printf("  flashAttentionMmaSplits = %d, the base formula = %d\n", splits, formula);
    require(splits == formula, "flashAttentionMmaSplits differs from the base's formula");
    float* partial = nullptr;
    cudaCheck(cudaMalloc(&partial, flashAttentionMmaPartialFloats(bh, sq, splits, D) * sizeof(float)), "cudaMalloc partial");
    executeOperations<SplitDPP>(s, q, k, v, partial, splits, bh, sq, sk, causal, -1.f, epi);
    executeOperations<typename SplitDPP::CombineDPP>(s, partial, o, bh, sq, splits, epi);
    s.sync();
    cudaFree(partial);
    return splits;
#else
    (void)split;
    std::printf("  the base formula = %d splits (executeFlashAttentionMma decides)\n", formula);
    executeFlashAttentionMma<D, BQ, BKV, NW>(q, k, v, o, bh, sq, sk, causal, s, -1.f, epi);
    return formula;
#endif
}

struct Bf16Inputs {
    __nv_bfloat16 *q, *k, *v;
    size_t nQ;
    std::vector<__nv_bfloat16> hq, hk, hv;
    Bf16Inputs(const int D, const int bh, const int sq, const int sk, const int salt)
        : nQ(static_cast<size_t>(bh) * sq * D) {
        const size_t nK = static_cast<size_t>(bh) * sk * D;
        hq = bf16s(nQ, salt);
        hk = bf16s(nK, salt + 1);
        hv = bf16s(nK, salt + 2);
        q = deviceCopy(hq);
        k = deviceCopy(hk);
        v = deviceCopy(hv);
    }
    ~Bf16Inputs() { cudaFree(q); cudaFree(k); cudaFree(v); }
};

// explicit tiles 64 x 64 x 4 on shapes the base does not split (fewer than 8 KV tiles per q-block)
template <int D>
void mmaExplicitCase(GpuStream& s, const char* name, const int bh, const int sq, const int sk, const bool causal, const int salt) {
    Bf16Inputs in(D, bh, sq, sk, salt);
    float* o = deviceSentinel<float>(in.nQ);
    std::printf("%s:\n", name);
    const int n = runMma<D, 64, 64, 4>(s, false, makeAttentionRead(in.q, bh, sq, D), makeAttentionRead(in.k, bh, sk, D),
                                       makeAttentionRead(in.v, bh, sk, D), o, bh, sq, sk, causal);
#ifndef PR435_HEAD
    require(n == 1, std::string(name) + ": the base would split this shape");
#endif
    (void)n;
    dumpDevice(name, o, in.nQ);
    cudaFree(o);
}

// the automatic choice: base executeFlashAttentionMma<D> (BQ = BKV = 0); head flashAttentionMmaTiles + Splits
template <int D, int BQ, int BKV, int NW, typename Q, typename K, typename V>
void mmaAutoRun(GpuStream& s, const Q& q, const K& k, const V& v, float* o, const int bh, const int sq, const int sk,
                const bool causal) {
#ifdef PR435_HEAD
    using MmaDPP = FlashAttentionMmaDPP<GPU, float, D, Q, K, V, Identity, BQ, BKV, NW>;
    const int splits = flashAttentionMmaSplits<MmaDPP>(bh, sq, sk, causal);
    std::printf("  flashAttentionMmaTiles = %d x %d x %d, flashAttentionMmaSplits = %d\n", BQ, BKV, NW, splits);
    if (splits == 1) {
        executeOperations<MmaDPP>(s, q, k, v, o, bh, sq, sk, causal);
    } else {
        if constexpr (BQ == 64 && BKV == 64 && NW == 4) {
            using SplitDPP = FlashAttentionMmaSplitKVDPP<GPU, float, D, Q, K, V, Identity, BQ, BKV, NW>;
            float* partial = nullptr;
            cudaCheck(cudaMalloc(&partial, flashAttentionMmaPartialFloats(bh, sq, splits, D) * sizeof(float)), "cudaMalloc");
            executeOperations<SplitDPP>(s, q, k, v, partial, splits, bh, sq, sk, causal);
            executeOperations<typename SplitDPP::CombineDPP>(s, partial, o, bh, sq, splits);
            s.sync();
            cudaFree(partial);
        } else {
            fail("the harness has the split path of the automatic choice only for 64 x 64 x 4");
        }
    }
#else
    (void)q; (void)k; (void)v; (void)o; (void)bh; (void)sq; (void)sk; (void)causal; (void)s;
#endif
}
template <int D, typename Q, typename K, typename V>
void mmaAuto(GpuStream& s, const Q& q, const K& k, const V& v, float* o, const int bh, const int sq, const int sk,
             const bool causal) {
#ifdef PR435_HEAD
    constexpr bool RAW = isRawBf16Read<K> && isRawBf16Read<V>;
    const FlashAttentionMmaTiles t = flashAttentionMmaTiles<D, K, V>(bh, sq, causal);
    const auto is = [&](const int bq, const int bkv, const int nw) {
        return t.blockQ == bq && t.blockKV == bkv && t.numWarps == nw;
    };
    if constexpr (RAW && D <= 64) {
        if (is(128, 64, 4)) return mmaAutoRun<D, 128, 64, 4>(s, q, k, v, o, bh, sq, sk, causal);
        if (is(64, 128, 4)) return mmaAutoRun<D, 64, 128, 4>(s, q, k, v, o, bh, sq, sk, causal);
        if (is(64, 64, 4)) return mmaAutoRun<D, 64, 64, 4>(s, q, k, v, o, bh, sq, sk, causal);
    } else if constexpr (RAW && D == 128) {
        if (is(128, 128, 8)) return mmaAutoRun<D, 128, 128, 8>(s, q, k, v, o, bh, sq, sk, causal);
        if (is(64, 64, 4)) return mmaAutoRun<D, 64, 64, 4>(s, q, k, v, o, bh, sq, sk, causal);
    } else if constexpr (!RAW) {
        if (is(D <= 64 ? 128 : 64, 32, 4)) return mmaAutoRun<D, (D <= 64 ? 128 : 64), 32, 4>(s, q, k, v, o, bh, sq, sk, causal);
    }
    fail("flashAttentionMmaTiles gave tiles the harness does not instantiate: " + std::to_string(t.blockQ) + " x " +
         std::to_string(t.blockKV) + " x " + std::to_string(t.numWarps));
#else
    executeFlashAttentionMma<D>(q, k, v, o, bh, sq, sk, causal, s);
#endif
}
template <int D>
void mmaAutoCase(GpuStream& s, const char* name, const int bh, const int sq, const int sk, const bool causal, const int salt) {
    Bf16Inputs in(D, bh, sq, sk, salt);
    float* o = deviceSentinel<float>(in.nQ);
    std::printf("%s:\n", name);
    mmaAuto<D>(s, makeAttentionRead(in.q, bh, sq, D), makeAttentionRead(in.k, bh, sk, D), makeAttentionRead(in.v, bh, sk, D), o,
               bh, sq, sk, causal);
    dumpDevice(name, o, in.nQ);
    cudaFree(o);
}
// the automatic choice with int8 K / V (dequantised by the read: the fused-prologue path)
template <int D>
void mmaAutoInt8Case(GpuStream& s, const char* name, const int bh, const int sq, const int sk, const bool causal, const int salt) {
    const size_t nQ = static_cast<size_t>(bh) * sq * D, nK = static_cast<size_t>(bh) * sk * D;
    __nv_bfloat16* q = deviceCopy(bf16s(nQ, salt));
    int8_t* k8 = deviceCopy(int8s(nK, salt + 1));
    int8_t* v8 = deviceCopy(int8s(nK, salt + 2));
    float* ks = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 3, 0.002f, 0.02f));
    float* vs = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 4, 0.002f, 0.02f));
    float* o = deviceSentinel<float>(nQ);
    std::printf("%s:\n", name);
    mmaAuto<D>(s, makeAttentionRead(q, bh, sq, D), makeInt8KVRead(k8, ks, bh, sk, D), makeInt8KVRead(v8, vs, bh, sk, D), o, bh,
               sq, sk, causal);
    dumpDevice(name, o, nQ);
    cudaFree(q); cudaFree(k8); cudaFree(v8); cudaFree(ks); cudaFree(vs); cudaFree(o);
}

// the double oracle of utest_flash_attention_mma.h (cpuAttention), on the bf16-rounded inputs: O = softmax(scale Q K^T
// [causal: key j <= query i]) V, scale = 1 / sqrt(D)
std::vector<double> oracle(const Bf16Inputs& in, const int D, const int bh, const int sq, const int sk, const bool causal) {
    std::vector<double> o(static_cast<size_t>(bh) * sq * D, 0.0), s(sk);
    const double scale = 1.0 / std::sqrt(static_cast<double>(D));
    for (int b = 0; b < bh; ++b) {
        for (int i = 0; i < sq; ++i) {
            const int kEnd = causal ? std::min(sk, i + 1) : sk;
            double m = -1e300;
            for (int j = 0; j < kEnd; ++j) {
                double dot = 0.0;
                for (int c = 0; c < D; ++c)
                    dot += static_cast<double>(__bfloat162float(in.hq[(static_cast<size_t>(b) * sq + i) * D + c])) *
                           static_cast<double>(__bfloat162float(in.hk[(static_cast<size_t>(b) * sk + j) * D + c]));
                s[j] = dot * scale;
                m = std::max(m, s[j]);
            }
            double l = 0.0;
            for (int j = 0; j < kEnd; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int j = 0; j < kEnd; ++j) {
                const double pj = s[j] / l;
                for (int c = 0; c < D; ++c)
                    o[(static_cast<size_t>(b) * sq + i) * D + c] +=
                        pj * static_cast<double>(__bfloat162float(in.hv[(static_cast<size_t>(b) * sk + j) * D + c]));
            }
        }
    }
    return o;
}

// the split path, tiles 64 x 64 x 4, on the three shapes of the PR's testMmaSplit
template <int D>
void splitCase(GpuStream& s, const char* tag, const int bh, const int sq, const int sk, const bool causal, const int salt) {
    Bf16Inputs in(D, bh, sq, sk, salt);
    const auto qR = makeAttentionRead(in.q, bh, sq, D);
    const auto kR = makeAttentionRead(in.k, bh, sk, D);
    const auto vR = makeAttentionRead(in.v, bh, sk, D);
    const std::string base = std::string("mma_split_") + tag;
    {
        const std::vector<double> o = oracle(in, D, bh, sq, sk, causal);
        writeBin(base + "_oracle_f64", o.data(), o.size() * sizeof(double));
    }
    {
        float* o = deviceSentinel<float>(in.nQ);
        std::printf("%s_identity:\n", base.c_str());
        const int n = runMma<D, 64, 64, 4>(s, true, qR, kR, vR, o, bh, sq, sk, causal);
        require(n >= 2, base + ": the shape is not split");
        dumpDevice(base + "_identity", o, in.nQ);
        cudaFree(o);
    }
    {
        float* o = deviceSentinel<float>(in.nQ);
        std::printf("%s_epilogue:\n", base.c_str());
        const int n = runMma<D, 64, 64, 4>(s, true, qR, kR, vR, o, bh, sq, sk, causal, epilogue());
        require(n >= 2, base + ": the shape is not split");
        dumpDevice(base + "_epilogue", o, in.nQ);
        cudaFree(o);
    }
#ifdef PR435_HEAD
    {   // the unsplit FlashAttentionMmaDPP on the same shape (head only: the base has no unsplit call for a split shape)
        float* o = deviceSentinel<float>(in.nQ);
        std::printf("mma_unsplit_%s_identity / _epilogue (head only):\n", tag);
        runMma<D, 64, 64, 4>(s, false, qR, kR, vR, o, bh, sq, sk, causal);
        dumpDevice(std::string("mma_unsplit_") + tag + "_identity", o, in.nQ);
        cudaCheck(cudaMemset(o, 0xff, in.nQ * sizeof(float)), "cudaMemset");
        runMma<D, 64, 64, 4>(s, false, qR, kR, vR, o, bh, sq, sk, causal, epilogue());
        dumpDevice(std::string("mma_unsplit_") + tag + "_epilogue", o, in.nQ);
        cudaFree(o);
    }
#endif
}

// ---- FlashDecodeDPP ----
template <int D, typename Q, typename K, typename V>
void runDecode(GpuStream& s, const Q& q, const K& k, const V& v, float* o, float* ws, const int bh, const int sk) {
#ifdef PR435_HEAD
    executeOperations<FlashDecodeDPP<GPU, float, D, Q, K, V>>(s, q, k, v, o, ws, bh, sk);
#else
    executeFlashDecode<D>(q, k, v, o, ws, bh, sk, s);
#endif
}
template <int D>
void decodeCase(GpuStream& s, const char* name, const int bh, const int sk, const bool int8kv, const int salt) {
    const int splits = flashDecodeSplits(bh, sk);
    const size_t nQ = static_cast<size_t>(bh) * D, nK = static_cast<size_t>(bh) * sk * D;
    float* ws = nullptr;
    if (splits > 1) {
        const size_t n = flashDecodeWorkspaceFloats(bh, splits, D);
        cudaCheck(cudaMalloc(&ws, n * sizeof(float)), "cudaMalloc workspace");
        cudaCheck(cudaMemset(ws, 0, n * sizeof(float)), "cudaMemset workspace");
    }
    std::printf("%s: flashDecodeSplits = %d\n", name, splits);
    __nv_bfloat16* q = deviceCopy(bf16s(nQ, salt));
    float* o = deviceSentinel<float>(nQ);
    const auto qR = makeAttentionRead(q, bh, 1, D);
    if (int8kv) {
        int8_t* k8 = deviceCopy(int8s(nK, salt + 1));
        int8_t* v8 = deviceCopy(int8s(nK, salt + 2));
        float* ks = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 3, 0.002f, 0.02f));
        float* vs = deviceCopy(floats(static_cast<size_t>(bh) * sk, salt + 4, 0.002f, 0.02f));
        runDecode<D>(s, qR, makeInt8KVRead(k8, ks, bh, sk, D), makeInt8KVRead(v8, vs, bh, sk, D), o, ws, bh, sk);
        dumpDevice(name, o, nQ);
        cudaFree(k8); cudaFree(v8); cudaFree(ks); cudaFree(vs);
    } else {
        __nv_bfloat16* k = deviceCopy(bf16s(nK, salt + 1));
        __nv_bfloat16* v = deviceCopy(bf16s(nK, salt + 2));
        runDecode<D>(s, qR, makeAttentionRead(k, bh, sk, D), makeAttentionRead(v, bh, sk, D), o, ws, bh, sk);
        dumpDevice(name, o, nQ);
        cudaFree(k); cudaFree(v);
    }
    cudaFree(q); cudaFree(o);
    if (ws != nullptr) cudaFree(ws);
}

} // namespace

int main(int argc, char** argv) {
    start(argc, argv, "pr435_attention");
    try {
        GpuStream s;
        softmaxCases(s);
        simtCase<64>(s, "fa_simt_d64_dense_b3_s45_s77", 3, 45, 77, false, false, false, 601);
        simtCase<128>(s, "fa_simt_d128_causal_b2_s96", 2, 96, 96, true, false, false, 611);
        simtCase<64>(s, "fa_simt_d64_int8kv_b2_s40_s130", 2, 40, 130, false, true, false, 621);
        simtCase<64>(s, "fa_simt_d64_epilogue_b2_s33_s65", 2, 33, 65, false, false, true, 631);
        mmaExplicitCase<64>(s, "mma_64x64x4_d64_dense_b2_s128_s300", 2, 128, 300, false, 701);
        mmaExplicitCase<128>(s, "mma_64x64x4_d128_causal_b2_s200", 2, 200, 200, true, 711);
        mmaExplicitCase<128>(s, "mma_64x64x4_d128_dense_b2_s70_s300", 2, 70, 300, false, 721);
        mmaAutoCase<64>(s, "mma_auto_d64_causal_b4_s1024", 4, 1024, 1024, true, 731);
        mmaAutoCase<64>(s, "mma_auto_d64_dense_b32_s2048", 32, 2048, 2048, false, 741);
        mmaAutoCase<128>(s, "mma_auto_d128_dense_b1_s16384", 1, 16384, 16384, false, 751);
        mmaAutoCase<64>(s, "mma_auto_d64_dense_b2_s64_s1024", 2, 64, 1024, false, 761);
        mmaAutoInt8Case<64>(s, "mma_auto_d64_int8kv_b2_s128_s200", 2, 128, 200, false, 771);
        splitCase<64>(s, "d64_dense_b2_s64_s1024", 2, 64, 1024, false, 801);
        splitCase<64>(s, "d64_causal_b1_s1024", 1, 1024, 1024, true, 811);
        splitCase<128>(s, "d128_dense_b2_s70_s1500", 2, 70, 1500, false, 821);
        decodeCase<128>(s, "decode_d128_bf16_b8_s4096", 8, 4096, false, 901);
        decodeCase<64>(s, "decode_d64_int8kv_b4_s1000", 4, 1000, true, 911);
        decodeCase<64>(s, "decode_d64_bf16_b2_s200_one_split", 2, 200, false, 921);
        s.sync();
    } catch (const std::exception& e) {
        fail(std::string("exception: ") + e.what());
    }
    return finish("pr435_attention");
}
