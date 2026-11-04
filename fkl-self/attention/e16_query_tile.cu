/* Copyright 2026 Oscar Amoros Huguet

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

// E16 query-tile experiment: does the extra time of the fused int8 attention kernel fall in proportion to the number of query tiles?
// Derived from e16_attention.cu (same data, timing, error check and kernel-facts code); only the variants, their order and one CSV column differ. D = 128 (D = 64 compiles but is not the question), bf16 Q, K, V
// (BH, S, D), BH = max(4, tokens / S), non-causal. Every variant is the tensor-core kernel with EXPLICIT tile arguments (executeFlashAttentionMma<D, BLOCK_Q, BLOCK_KV, NUM_WARPS, bf16>, which bypasses the library's heuristic):
//   raw_default              the library's own pick for raw bf16 (BQ64 / BKV64 / 4 warps at D = 128 below S = 16384): the bf16 control of e16_attention.cu
//   raw_bq32_kv32_w2, raw_bq64_kv32_w4, raw_bq128_kv32_w4        bf16 raw, BLOCK_KV = 32 as the quantised schedule uses, 2 / 4 / 4 warps (BLOCK_Q / NUM_WARPS = 16 / 16 / 32)
//   int8_bq32_w2, int8_bq64_w4, int8_bq128_w4                    K and V int8 with one fp32 scale per token (the library's quantised schedule), BLOCK_KV = 32, the same tiles
//   int8nodq_*               TIMING ONLY, THE OUTPUT IS WRONG, NEVER A RESULT: the same quantised schedule with a patched copy of the library header (fkl-self/attention/nodq/, diff in flash_attention_mma_nodq.diff,
//                            found first by the include path; the library is not changed). dequantTile does NOT convert or scale: it REMOVES the int-to-float conversion, the multiply by the scale and the float-to-bf16 rounding;
//                            it KEEPS the cp.async copy of the raw bytes, the staging area and the shared-memory pass over the same tile, the barriers, the load of the per-token scale from global memory (kept alive with an
//                            empty asm volatile that uses it) and the same 16-byte shared-memory store of the same size: the two int8 bytes of a pair are packed into the 32-bit slot. Everything after the pass sees garbage
//                            bf16 values (it can be NaN or Inf, which does not change the time of the arithmetic). The column timing_only is 1 for these rows and the file must never be used as a result of those rows.
// DECISION RULE (fixed before the first run): excess(BQ, S) = time(int8, BQ) - time(raw bf16 with the same BQ, BKV = 32, same warps) and ratio(S) = excess(128) / excess(64) for each S. Halving the number of query tiles
// halves everything the quantised schedule repeats per query tile (the conversion, but also the scale loads, the shared-memory staging pass and its extra barriers), so a ratio near 0.5 means 'a cost repeated per
// query tile', not 'the conversion'. Read: ratio <= 0.65 at S >= 2048: proportional to the number of query tiles; ratio >= 0.85: not proportional; in between: partly. The share of the conversion is
// share(BQ, S) = (time(int8) - time(int8nodq)) / (time(int8) - time(raw)): the arithmetic only, because nodq keeps the same copies, passes, barriers and loads. The tile sizes change occupancy and registers
// for both kernels, which is why the excess is always against the bf16 kernel with the same BQ, and ratio and share are the only quantities interpreted. The kernel facts (registers, local memory, shared
// memory, blocks per SM, tile and warp counts) are columns of the CSV; DEEP_Q_SMEM is a constant false in the library and DUAL_Q_SMEM needs FK_FA_DUAL_Q (default 0) and the raw schedule, so neither is active.
// Environment as e16_attention.cu: E16_SEQS (512..8192), E16_TOKENS (262144), E16_ITERS (15 blocks), E16_BLOCK_MS (20), E16_WARMUP (10) and E16_WARMUP_MS (300), E16_ROTATE (0, 1, 2 for the three runs: the order of the
// variants is rotated by (configuration index + E16_ROTATE) modulo the number of variants), E16_PROFILE (1: one call per variant, no timing, for Nsight Compute), E16_VARIANTS (prefixes).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/nan_safe.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cxxabi.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define E16_CUDA(call)                                                                                                                                                                  \
    do {                                                                                                                                                                                 \
        const cudaError_t e16Err = (call);                                                                                                                                               \
        if (e16Err != cudaSuccess) {                                                                                                                                                     \
            throw std::runtime_error(std::string(#call) + " failed: " + cudaGetErrorString(e16Err) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + ")");                           \
        }                                                                                                                                                                                \
    } while (0)

namespace {
using bf16 = __nv_bfloat16;

// ------------------------------------------------------------------------------------------------------------------------------------------------------ data
__host__ __device__ inline uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

// approximately normal(0, 1): the sum of four uniforms, centred and scaled
__global__ void fillNormal(bf16* p, const size_t n, const uint64_t seed) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) { return; }
    const uint64_t h = splitmix(seed + i);
    float s = 0.f;
    for (int k = 0; k < 4; ++k) { s += static_cast<float>((h >> (16 * k)) & 0xffff) * (1.f / 65536.f); }
    p[i] = __float2bfloat16((s - 2.f) * 1.7320508f);
}

// one warp per token: scale = max|x| / (127 for int8, 448 for e4m3), data = round(x / scale) or the e4m3 byte of x / scale
template <int D, bool FP8>
__global__ void quantizeTokens(const bf16* x, int8_t* q, float* scales, const long tokens) {
    const long t = static_cast<long>(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    if (t >= tokens) { return; }
    float mx = 0.f;
    for (int e = lane; e < D; e += 32) { mx = fmaxf(mx, fabsf(__bfloat162float(x[t * D + e]))); }
    for (int o = 16; o > 0; o >>= 1) { mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o)); }
    const float sc = mx > 0.f ? mx / (FP8 ? 448.f : 127.f) : 1.f;
    if (lane == 0) { scales[t] = sc; }
    for (int e = lane; e < D; e += 32) {
        const float v = __bfloat162float(x[t * D + e]) / sc;
        if constexpr (FP8) {
            const __nv_fp8_e4m3 f8(v);
            q[t * D + e] = static_cast<int8_t>(f8.__x);
        } else {
            q[t * D + e] = static_cast<int8_t>(__float2int_rn(v));
        }
    }
}

// the unfused prologue: K and V (z = 0, 1) dequantised to bf16 buffers, one kernel
template <int D, bool FP8>
__global__ void dequantKV(const int8_t* kq, const float* ks, const int8_t* vq, const float* vs, bf16* ko, bf16* vo, const long tokens) {
    const int8_t* q = blockIdx.z == 0 ? kq : vq;
    const float* s = blockIdx.z == 0 ? ks : vs;
    bf16* o = blockIdx.z == 0 ? ko : vo;
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= tokens * D) { return; }
    const long t = i / D;
    float v;
    if constexpr (FP8) {
        __nv_fp8_e4m3 f8;
        f8.__x = static_cast<__nv_fp8_storage_t>(q[i]);
        v = static_cast<float>(f8);
    } else {
        v = static_cast<float>(q[i]);
    }
    o[i] = __float2bfloat16(v * s[t]);
}

// the separate epilogue kernel: o = o * a + b, in place
__global__ void affineBf16(bf16* o, const size_t n, const float a, const float b) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { o[i] = __float2bfloat16(__bfloat162float(o[i]) * a + b); }
}

// the references, fp32, for two heads: float copies of Q, K, V of those heads (dequantised when the variant is quantised)
template <bool FP8>
__global__ void gatherHeadsF32(const bf16* x, float* out, const long headStride, const long head0, const long head1, const long perHead) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 2 * perHead) { return; }
    const long which = i / perHead, r = i % perHead;
    out[i] = __bfloat162float(x[(which == 0 ? head0 : head1) * headStride + r]);
}
template <bool FP8>
__global__ void gatherHeadsQuant(const int8_t* q, const float* scales, float* out, const long headStride, const long head0, const long head1, const long perHead, const int d) {
    const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= 2 * perHead) { return; }
    const long which = i / perHead, r = i % perHead;
    const long head = which == 0 ? head0 : head1;
    float v;
    if constexpr (FP8) {
        __nv_fp8_e4m3 f8;
        f8.__x = static_cast<__nv_fp8_storage_t>(q[head * headStride + r]);
        v = static_cast<float>(f8);
    } else {
        v = static_cast<float>(q[head * headStride + r]);
    }
    out[i] = v * scales[head * (headStride / d) + r / d];
}

// one block per (query row, checked head); scores of the whole row in shared memory; plain fp32
__global__ void referenceAttention(const float* q, const float* k, const float* v, float* out, const int seq, const int d, const bool causal, const float scale) {
    extern __shared__ float s[];
    __shared__ float red[32];
    const int row = blockIdx.x, h = blockIdx.y;
    const float* qi = q + (static_cast<size_t>(h) * seq + row) * d;
    const float* kh = k + static_cast<size_t>(h) * seq * d;
    const float* vh = v + static_cast<size_t>(h) * seq * d;
    const int kvEnd = causal ? row + 1 : seq;
    float m = -FLT_MAX;
    for (int j = threadIdx.x; j < kvEnd; j += blockDim.x) {
        float acc = 0.f;
        for (int e = 0; e < d; ++e) { acc += qi[e] * kh[static_cast<size_t>(j) * d + e]; }
        s[j] = acc * scale;
        m = fmaxf(m, s[j]);
    }
    for (int o = 16; o > 0; o >>= 1) { m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = m; }
    __syncthreads();
    m = -FLT_MAX;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) { m = fmaxf(m, red[w]); }
    __syncthreads();
    float l = 0.f;
    for (int j = threadIdx.x; j < kvEnd; j += blockDim.x) {
        s[j] = expf(s[j] - m);
        l += s[j];
    }
    for (int o = 16; o > 0; o >>= 1) { l += __shfl_xor_sync(0xffffffffu, l, o); }
    if ((threadIdx.x & 31) == 0) { red[threadIdx.x >> 5] = l; }
    __syncthreads();
    l = 0.f;
    for (int w = 0; w < static_cast<int>(blockDim.x >> 5); ++w) { l += red[w]; }
    __syncthreads();
    for (int e = threadIdx.x; e < d; e += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < kvEnd; ++j) { acc += s[j] * vh[static_cast<size_t>(j) * d + e]; }
        out[(static_cast<size_t>(h) * seq + row) * d + e] = acc / l;
    }
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------ helpers
template <class T>
struct Dev {
    T* p{nullptr};
    explicit Dev(const size_t n = 0) { if (n) { E16_CUDA(cudaMalloc(&p, n * sizeof(T))); } }
    Dev(const Dev&) = delete;
    Dev& operator=(const Dev&) = delete;
    ~Dev() { if (p) { cudaFree(p); } }
};

std::string envString(const char* name, const std::string& fallback) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : fallback;
}
std::vector<int> envList(const char* name, const std::string& fallback) {
    std::vector<int> out;
    std::stringstream ss(envString(name, fallback));
    std::string item;
    while (std::getline(ss, item, ',')) { out.push_back(std::stoi(item)); }
    return out;
}
bool variantWanted(const std::string& name) {
    const std::string list = envString("E16_VARIANTS", "");
    if (list.empty()) { return true; }
    std::stringstream ss(list);
    std::string prefix;
    while (std::getline(ss, prefix, ',')) { if (name.rfind(prefix, 0) == 0) { return true; } }
    return false;
}

struct KernelFacts {
    int kernels{-1};
    std::string text;
    // the first attention kernel of the call, from its demangled template arguments (BLOCK_Q, BLOCK_KV, NUM_WARPS, SPLIT_KV) and from the types of its K and V reads (the schedule the library selects for them);
    // scripts/e16_sass_counts.py counts the LDGSTS (cp.async) instructions of every instantiation in the binary as the measured check of the schedule
    std::string schedule, splitKV;
    int blockQ{0}, blockKV{0}, numWarps{0};
};

std::vector<std::string> templateArgs(const std::string& demangled, const std::string& key) {   // top-level template arguments of key<...>
    const size_t p = demangled.find(key + "<");
    if (p == std::string::npos) { return {}; }
    std::vector<std::string> out;
    std::string cur;
    int depth = 1;
    for (size_t i = p + key.size() + 1; i < demangled.size(); ++i) {
        const char c = demangled[i];
        if (c == '<') { ++depth; }
        if (c == '>') { if (--depth == 0) { break; } }
        if (c == ',' && depth == 1) { out.push_back(cur); cur.clear(); continue; }
        if (!(cur.empty() && c == ' ')) { cur += c; }
    }
    out.push_back(cur);
    return out;
}

void parseAttentionKernel(const std::string& mangled, KernelFacts& f) {
    int status = 0;
    char* d = abi::__cxa_demangle(mangled.c_str(), nullptr, nullptr, &status);
    const std::string dem = (status == 0 && d) ? d : mangled;
    std::free(d);
    const auto has = [](const std::string& s, const char* what) { return s.find(what) != std::string::npos; };
    const std::vector<std::string> mma = templateArgs(dem, "launchFlashAttentionMmaDPP_Kernel");
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
        f.numWarps = std::atoi(simt[7].c_str());
        f.splitKV = "false";
        f.schedule = "simt (thread-staged shared tiles)";
    }
}

// capture one call into a graph and read its kernel nodes: the number of kernels, and for each its name, registers, local memory, shared memory and blocks per SM
KernelFacts captureFacts(fk::Stream& stream, const std::function<void()>& run) {
    KernelFacts facts;
    cudaGraph_t graph = nullptr;
    try {
        E16_CUDA(cudaStreamBeginCapture(stream.getCUDAStream(), cudaStreamCaptureModeRelaxed));
        run();
        E16_CUDA(cudaStreamEndCapture(stream.getCUDAStream(), &graph));
        size_t n = 0;
        E16_CUDA(cudaGraphGetNodes(graph, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        E16_CUDA(cudaGraphGetNodes(graph, nodes.data(), &n));
        int dev = 0, sms = 0, maxThreadsSM = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        int count = 0;
        std::ostringstream text;
        for (const cudaGraphNode_t node : nodes) {
            cudaGraphNodeType type;
            E16_CUDA(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) { continue; }
            cudaKernelNodeParams kp;
            E16_CUDA(cudaGraphKernelNodeGetParams(node, &kp));
            cudaFuncAttributes attr;
            E16_CUDA(cudaFuncGetAttributes(&attr, kp.func));
            const int threads = static_cast<int>(kp.blockDim.x * kp.blockDim.y * kp.blockDim.z);
            int blocksPerSM = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kp.func, threads, kp.sharedMemBytes);
            const char* name = nullptr;
            cudaFuncGetName(&name, kp.func);
            std::string shortName = name ? name : "?";
            const size_t paren = shortName.find('(');
            if (shortName.size() > 70) { shortName = shortName.substr(0, 70) + "..."; }
            (void)paren;
            if (facts.schedule.empty() && name && std::string(name).find("FlashAttention") != std::string::npos) { parseAttentionKernel(name, facts); }
            if (count++) { text << " ; "; }
            text << shortName << " grid " << kp.gridDim.x << "x" << kp.gridDim.y << "x" << kp.gridDim.z << " block " << threads << " regs " << attr.numRegs << " local_B " << attr.localSizeBytes
                 << " static_smem_B " << attr.sharedSizeBytes << " dynamic_smem_B " << kp.sharedMemBytes << " blocks_per_SM " << blocksPerSM << " occupancy "
                 << (maxThreadsSM > 0 ? 100.0 * blocksPerSM * threads / maxThreadsSM : 0.0) << "%";
        }
        facts.kernels = count;
        facts.text = text.str();
    } catch (const std::exception& e) {
        facts.kernels = -1;
        facts.text = std::string("capture failed: ") + e.what();
        cudaGetLastError();
        if (graph == nullptr) {
            cudaGraph_t dummy = nullptr;
            cudaStreamEndCapture(stream.getCUDAStream(), &dummy);
            if (dummy) { cudaGraphDestroy(dummy); }
        }
    }
    if (graph) { cudaGraphDestroy(graph); }
    return facts;
}

struct Timing {
    double median{0}, min{0}, max{0};   // per call, from blocks of back-to-back calls
    double single{0};                    // median of single calls, each synchronised (the launch latency of the call is in it)
    int reps{0};                         // calls per block
    long long tStartMs{0}, tEndMs{0};    // wall clock (ms since the epoch) at the start of the first block and at the end of the last block: joined with the SM clock sampler by scripts/e16_join_clock.py
};
int& configIndex() { static int index = 0; return index; }
int rotateOffset() { static const int offset = std::atoi(std::getenv("E16_ROTATE") ? std::getenv("E16_ROTATE") : "0"); return offset; }
long long nowMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
// iters blocks of `reps` calls enqueued back to back between two events (reps chosen from a first call so that a block lasts about blockMs): the time per call is the GPU time of the call, with no gap
// for the host between calls; then iters single calls, each between two events and synchronised.
// The warm-up lasts at least `warmup` calls and at least warmupMs of GPU time: the clock of the GPU must be back at its boost, after a light kernel (the SIMT variant) or the start of the process, before the first block.
Timing timeCall(fk::Stream& stream, const std::function<void()>& run, const int warmup, const int iters, const double blockMs, const double warmupMs) {
    cudaEvent_t a, b;
    E16_CUDA(cudaEventCreate(&a));
    E16_CUDA(cudaEventCreate(&b));
    const auto elapsed = [&](const int calls) {
        E16_CUDA(cudaEventRecord(a, stream.getCUDAStream()));
        for (int i = 0; i < calls; ++i) { run(); }
        E16_CUDA(cudaEventRecord(b, stream.getCUDAStream()));
        E16_CUDA(cudaEventSynchronize(b));
        float f = 0.f;
        E16_CUDA(cudaEventElapsedTime(&f, a, b));
        return static_cast<double>(f);
    };
    for (int i = 0; i < warmup; ++i) { run(); }
    E16_CUDA(cudaStreamSynchronize(stream.getCUDAStream()));
    for (double spent = 0.0; spent < warmupMs;) { spent += elapsed(4); }
    const double first = std::max(elapsed(1), 1e-3);
    const int reps = std::max(1, std::min(200, static_cast<int>(std::ceil(blockMs / first))));
    std::vector<double> blocks, singles;
    const long long tStart = nowMs();
    for (int i = 0; i < iters; ++i) { blocks.push_back(elapsed(reps) / reps); }
    const long long tEnd = nowMs();
    for (int i = 0; i < iters; ++i) { singles.push_back(elapsed(1)); }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    std::sort(blocks.begin(), blocks.end());
    std::sort(singles.begin(), singles.end());
    return {blocks[blocks.size() / 2], blocks.front(), blocks.back(), singles[singles.size() / 2], reps, tStart, tEnd};
}

// ------------------------------------------------------------------------------------------------------------------------------------------------------ one (D, S) point
struct Point {
    int d, seq, bh;
    bool causal;
};

template <int D>
struct Runner {
    fklb::CsvWriter& csv;
    fk::Stream stream;
    const int warmup, iters, simtMaxSeq;
    const double blockMs, warmupMs;
    const float epiA, epiB;
    std::string gpu;

    Runner(fklb::CsvWriter& c, const int w, const int i, const int simt, const double block, const double warmMs, const float a, const float b, std::string g)
        : csv(c), warmup(w), iters(i), simtMaxSeq(simt), blockMs(block), warmupMs(warmMs), epiA(a), epiB(b), gpu(std::move(g)) {}

    // error of the output o (bf16, BH x S x D) against the reference of two heads
    void errorAgainst(const Point& pt, const bf16* o, const float* qRef, const float* kRef, const float* vRef, double& maxErr, double& meanErr, const float affA, const float affB) {
        const size_t perHead = static_cast<size_t>(pt.seq) * D;
        Dev<float> ref(2 * perHead);
        const dim3 grid(pt.seq, 2);
        const size_t smem = static_cast<size_t>(pt.seq) * sizeof(float);
        if (smem > 48 * 1024) { E16_CUDA(cudaFuncSetAttribute(referenceAttention, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem))); }
        referenceAttention<<<grid, 128, smem>>>(qRef, kRef, vRef, ref.p, pt.seq, D, pt.causal, 1.f / std::sqrt(static_cast<float>(D)));
        E16_CUDA(cudaGetLastError());
        std::vector<float> hRef(2 * perHead);
        E16_CUDA(cudaMemcpy(hRef.data(), ref.p, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
        std::vector<bf16> hOut(2 * perHead);
        E16_CUDA(cudaMemcpy(hOut.data(), o, perHead * sizeof(bf16), cudaMemcpyDeviceToHost));
        E16_CUDA(cudaMemcpy(hOut.data() + perHead, o + static_cast<size_t>(pt.bh - 1) * perHead, perHead * sizeof(bf16), cudaMemcpyDeviceToHost));
        maxErr = 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < hRef.size(); ++i) {
            const double r = static_cast<double>(hRef[i]) * affA + affB;
            const double e = std::fabs(static_cast<double>(__bfloat162float(hOut[i])) - r);
            maxErr = fklb::maxKeepNaN(maxErr, e);
            sum += e;
        }
        meanErr = sum / static_cast<double>(hRef.size());
    }

    static std::string clean(std::string text) {   // the CSV has no quoting
        for (char& c : text) { if (c == ',' || c == '\n' || c == '\r') { c = ';'; } }
        return text;
    }

    void row(const Point& pt, const std::string& variant, const int position, const int iters_, const Timing& t, const double maxErr, const double meanErr, const KernelFacts& kf, const std::string& note) {
        const double flops = 4.0 * pt.bh * static_cast<double>(pt.seq) * pt.seq * pt.d * (pt.causal ? 0.5 : 1.0);
        const double tflops = t.median > 0 ? flops / (t.median * 1e-3) / 1e12 : 0.0;
        csv.row(gpu, variant, pt.d, pt.seq, pt.bh, pt.causal ? 1 : 0, position, iters_, t.reps, t.tStartMs, t.tEndMs, t.median, t.min, t.max, t.single, tflops, maxErr, meanErr, kf.kernels, kf.schedule, kf.blockQ, kf.blockKV, kf.numWarps, kf.splitKV, clean(kf.text), clean(note), variant.find("nodq") != std::string::npos ? 1 : 0);
        csv.flush();
        std::cout << variant << " D=" << pt.d << " S=" << pt.seq << " BH=" << pt.bh << " causal=" << pt.causal << ": " << t.median << " ms (" << t.reps << " calls per block), single " << t.single << " ms, " << tflops
                  << " TFLOP/s, max err " << maxErr << ", kernels " << kf.kernels << (note.empty() ? "" : " | " + note) << std::endl;
    }

    void run(const Point& pt) {
        const size_t n = static_cast<size_t>(pt.bh) * pt.seq * D;
        const long tokens = static_cast<long>(pt.bh) * pt.seq;
        Dev<bf16> q(n), k(n), v(n), o(n), kd(n), vd(n);
        fillNormal<<<(n + 255) / 256, 256>>>(q.p, n, 11 + 1000 * D + pt.seq);
        fillNormal<<<(n + 255) / 256, 256>>>(k.p, n, 22 + 1000 * D + pt.seq);
        fillNormal<<<(n + 255) / 256, 256>>>(v.p, n, 33 + 1000 * D + pt.seq);
        Dev<int8_t> k8(n), v8(n), kf8(n), vf8(n);
        Dev<float> k8s(tokens), v8s(tokens), kf8s(tokens), vf8s(tokens);
        quantizeTokens<D, false><<<(tokens + 7) / 8, 256>>>(k.p, k8.p, k8s.p, tokens);
        quantizeTokens<D, false><<<(tokens + 7) / 8, 256>>>(v.p, v8.p, v8s.p, tokens);
        quantizeTokens<D, true><<<(tokens + 7) / 8, 256>>>(k.p, kf8.p, kf8s.p, tokens);
        quantizeTokens<D, true><<<(tokens + 7) / 8, 256>>>(v.p, vf8.p, vf8s.p, tokens);
        E16_CUDA(cudaGetLastError());
        E16_CUDA(cudaDeviceSynchronize());
        // fp32 copies of the two checked heads (first and last): bf16 inputs as they are, and the dequantised int8 and e4m3 K and V
        const size_t perHead = static_cast<size_t>(pt.seq) * D;
        const long headStride = static_cast<long>(perHead);
        const long head0 = 0, head1 = pt.bh - 1;
        const int gathers = static_cast<int>((2 * perHead + 255) / 256);
        Dev<float> qRef(2 * perHead), kRef(2 * perHead), vRef(2 * perHead), kRefI8(2 * perHead), vRefI8(2 * perHead), kRefF8(2 * perHead), vRefF8(2 * perHead);
        gatherHeadsF32<false><<<gathers, 256>>>(q.p, qRef.p, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsF32<false><<<gathers, 256>>>(k.p, kRef.p, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsF32<false><<<gathers, 256>>>(v.p, vRef.p, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsQuant<false><<<gathers, 256>>>(k8.p, k8s.p, kRefI8.p, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<false><<<gathers, 256>>>(v8.p, v8s.p, vRefI8.p, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<true><<<gathers, 256>>>(kf8.p, kf8s.p, kRefF8.p, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<true><<<gathers, 256>>>(vf8.p, vf8s.p, vRefF8.p, headStride, head0, head1, static_cast<long>(perHead), D);
        E16_CUDA(cudaGetLastError());
        E16_CUDA(cudaDeviceSynchronize());

        const auto qIOp = fk::makeAttentionRead(static_cast<const bf16*>(q.p), pt.bh, pt.seq, D);
        const auto kIOp = fk::makeAttentionRead(static_cast<const bf16*>(k.p), pt.bh, pt.seq, D);
        const auto vIOp = fk::makeAttentionRead(static_cast<const bf16*>(v.p), pt.bh, pt.seq, D);
        const auto kdIOp = fk::makeAttentionRead(static_cast<const bf16*>(kd.p), pt.bh, pt.seq, D);
        const auto vdIOp = fk::makeAttentionRead(static_cast<const bf16*>(vd.p), pt.bh, pt.seq, D);
        const auto k8IOp = fk::makeInt8KVRead(k8.p, k8s.p, pt.bh, pt.seq, D);
        const auto v8IOp = fk::makeInt8KVRead(v8.p, v8s.p, pt.bh, pt.seq, D);
        const auto k8nIOp = fk::makeInt8KVNoDequantRead(k8.p, k8s.p, pt.bh, pt.seq, D);
        const auto v8nIOp = fk::makeInt8KVNoDequantRead(v8.p, v8s.p, pt.bh, pt.seq, D);
        const auto epilogue = fk::Mul<float>::build(epiA).then(fk::Add<float>::build(epiB));
        const dim3 dqGrid((tokens * D + 255) / 256, 1, 2);

        const auto attnRaw = [&](const auto& kk, const auto& vv) { fk::executeFlashAttentionMma<D, 0, 0, 4, bf16>(qIOp, kk, vv, o.p, pt.bh, pt.seq, pt.seq, pt.causal, stream); };

        struct V {
            std::string name;
            std::function<void()> fn;
            const float* kRefP;
            const float* vRefP;
            float a, b;
            int iters;
        };
        std::vector<V> variants;
        const auto attn = [&](auto bq, auto bkv, auto nw, const auto& kk, const auto& vv) {
            fk::executeFlashAttentionMma<D, decltype(bq)::value, decltype(bkv)::value, decltype(nw)::value, bf16>(qIOp, kk, vv, o.p, pt.bh, pt.seq, pt.seq, pt.causal, stream);
        };
        using I32 = std::integral_constant<int, 32>; using I64 = std::integral_constant<int, 64>; using I128 = std::integral_constant<int, 128>; using W2 = std::integral_constant<int, 2>; using W4 = std::integral_constant<int, 4>;
        variants.push_back({"raw_default", [&] { fk::executeFlashAttentionMma<D, 0, 0, 4, bf16>(qIOp, kIOp, vIOp, o.p, pt.bh, pt.seq, pt.seq, pt.causal, stream); }, kRef.p, vRef.p, 1.f, 0.f, iters});
        variants.push_back({"raw_bq32_kv32_w2", [&] { attn(I32{}, I32{}, W2{}, kIOp, vIOp); }, kRef.p, vRef.p, 1.f, 0.f, iters});
        variants.push_back({"raw_bq64_kv32_w4", [&] { attn(I64{}, I32{}, W4{}, kIOp, vIOp); }, kRef.p, vRef.p, 1.f, 0.f, iters});
        variants.push_back({"raw_bq128_kv32_w4", [&] { attn(I128{}, I32{}, W4{}, kIOp, vIOp); }, kRef.p, vRef.p, 1.f, 0.f, iters});
        variants.push_back({"int8_bq32_w2", [&] { attn(I32{}, I32{}, W2{}, k8IOp, v8IOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        variants.push_back({"int8_bq64_w4", [&] { attn(I64{}, I32{}, W4{}, k8IOp, v8IOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        variants.push_back({"int8_bq128_w4", [&] { attn(I128{}, I32{}, W4{}, k8IOp, v8IOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        variants.push_back({"int8nodq_bq32_w2", [&] { attn(I32{}, I32{}, W2{}, k8nIOp, v8nIOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        variants.push_back({"int8nodq_bq64_w4", [&] { attn(I64{}, I32{}, W4{}, k8nIOp, v8nIOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        variants.push_back({"int8nodq_bq128_w4", [&] { attn(I128{}, I32{}, W4{}, k8nIOp, v8nIOp); }, kRefI8.p, vRefI8.p, 1.f, 0.f, iters});
        // ORDER: the variants are rotated by (configuration index + E16_ROTATE) modulo their number, so that over the configurations and the three runs every variant is measured at several positions
        const int shift = (configIndex()++ + rotateOffset()) % static_cast<int>(variants.size());
        std::rotate(variants.begin(), variants.begin() + shift, variants.end());
        const std::vector<V>& ordered = variants;
        int position = 0;
        for (const V& variant : ordered) {
            if (!variantWanted(variant.name)) { continue; }
            ++position;
            std::string note;
            try {
                E16_CUDA(cudaMemsetAsync(o.p, 0, n * sizeof(bf16), stream.getCUDAStream()));
                variant.fn();
                E16_CUDA(cudaStreamSynchronize(stream.getCUDAStream()));
                double maxErr = 0.0, meanErr = 0.0;
                errorAgainst(pt, o.p, qRef.p, variant.kRefP, variant.vRefP, maxErr, meanErr, variant.a, variant.b);
                if (envString("E16_PROFILE", "0") == "1") {   // one call per variant and no timing: for Nsight Compute (ncu --kernel-name regex:"FlashAttention|dequantKV")
                    row(pt, variant.name, position, 0, Timing{}, maxErr, meanErr, KernelFacts{}, "profile run: one call, not timed");
                    continue;
                }
                const KernelFacts facts = captureFacts(stream, variant.fn);
                const Timing t = timeCall(stream, variant.fn, warmup, variant.iters, blockMs, warmupMs);
                row(pt, variant.name, position, variant.iters, t, maxErr, meanErr, facts, note);
            } catch (const std::exception& e) {
                cudaGetLastError();
                row(pt, variant.name, position, 0, Timing{}, 0.0, 0.0, KernelFacts{}, std::string("failed: ") + e.what());
            }
        }
    }
};

template <int D>
int runDim(fklb::CsvWriter& csv, const std::vector<int>& seqs, const std::vector<int>& causals, const int tokens, const int minBh, const int fixedBh, const std::string& gpu) {
    Runner<D> runner(csv, std::atoi(envString("E16_WARMUP", "10").c_str()), std::atoi(envString("E16_ITERS", "15").c_str()), std::atoi(envString("E16_SIMT_MAX_SEQ", "2048").c_str()),
                     std::atof(envString("E16_BLOCK_MS", "20").c_str()), std::atof(envString("E16_WARMUP_MS", "300").c_str()), static_cast<float>(std::atof(envString("E16_EPI_A", "1.25").c_str())), static_cast<float>(std::atof(envString("E16_EPI_B", "0.5").c_str())), gpu);
    int failures = 0;
    for (const int seq : seqs) {
        for (const int causal : causals) {
            try {
                runner.run(Point{D, seq, fixedBh > 0 ? fixedBh : std::max(minBh, tokens / seq), causal != 0});
            } catch (const std::exception& e) {
                ++failures;
                cudaGetLastError();
                std::cerr << "D=" << D << " S=" << seq << " causal=" << causal << " failed: " << e.what() << std::endl;
            }
        }
    }
    return failures;
}
}  // namespace

int launch() {
    int failures = 0;
    try {
        cudaDeviceProp prop;
        E16_CUDA(cudaGetDeviceProperties(&prop, 0));
        fklb::CsvWriter csv("e16_query_tile.csv", "E16 query-tile experiment: fused int8 attention against bf16 and the no-dequantise timing variant",
                            {"gpu", "variant", "d", "seq", "bh", "causal", "position", "blocks", "calls_per_block", "t_start_ms", "t_end_ms", "time_ms_median", "time_ms_min", "time_ms_max", "single_ms_median", "tflops_median", "max_abs_err", "mean_abs_err", "kernels", "schedule", "block_q", "block_kv", "num_warps", "split_kv", "kernel_facts", "note", "timing_only"},
                            "bf16 Q, K, V (BH, S, D) contiguous, normal(0, 1), fp32 accumulation, bf16 output; D = 128, non-causal; time_ms_* = per call, from `blocks` blocks of `calls_per_block` calls enqueued back to back between two CUDA events "
                            "(the GPU time of the call; the block lasts about E16_BLOCK_MS), after a warm-up of at least E16_WARMUP_MS of GPU time; position = place of the variant in the order of its configuration (the variants are rotated by "
                            "configuration index + E16_ROTATE); TFLOP/s = 4 BH S^2 D / time_ms_median. Every variant is executeFlashAttentionMma with EXPLICIT tiles (variant name: bq = BLOCK_Q, kv = BLOCK_KV, w = warps; raw_default is "
                            "the library's own pick). int8_* = K and V int8 with one fp32 scale per token (quantised schedule). int8nodq_* = TIMING ONLY (timing_only = 1): THE OUTPUT IS WRONG (max_abs_err and mean_abs_err of those "
                            "rows are meaningless) and the rows are never a result: the same schedule with a patched private copy of the library header whose dequantTile removes the int-to-float conversion, the scale multiply and the "
                            "float-to-bf16 rounding and keeps the copies, the shared-memory pass, the barriers, the scale loads and the 16-byte store. The decision rule and the definitions of excess, ratio and share are in the "
                            "header comment of fkl-self/attention/e16_query_tile.cu, fixed before the first run. kernel_facts = registers, local memory, shared memory and blocks per SM of each kernel of the call.");
        const std::vector<int> dims = envList("E16_DIMS", "64,128"), seqs = envList("E16_SEQS", "512,1024,2048,4096,8192"), causals = envList("E16_CAUSAL", "0,1");
        const int tokens = std::atoi(envString("E16_TOKENS", "262144").c_str()), minBh = std::atoi(envString("E16_MIN_BH", "4").c_str()), fixedBh = std::atoi(envString("E16_BH", "0").c_str());
        for (const int d : dims) {
            if (d == 64) { failures += runDim<64>(csv, seqs, causals, tokens, minBh, fixedBh, prop.name); }
            else if (d == 128) { failures += runDim<128>(csv, seqs, causals, tokens, minBh, fixedBh, prop.name); }
            else { std::cerr << "head dimension " << d << " is not compiled" << std::endl; ++failures; }
        }
    } catch (const std::exception& e) {
        std::cerr << "e16_attention: " << e.what() << std::endl;
        return -1;
    }
    return failures == 0 ? 0 : -1;
}
