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

// E16, forward attention with FKL: the tensor-core DPP (FlashAttentionMmaDPP, fused/raw prologue, fused epilogue) and the SIMT DPP (FlashAttentionDPP), against their unfused alternatives.
// bf16 Q, K, V of shape (BH, S, D) contiguous, fp32 accumulation, bf16 output; D = 64 and 128; BH = max(minBH, tokens / S), so BH x S is the same for every S; causal and not causal.
// Inputs: the file e16_input_d<D>_s<S>_bh<BH>.bin of E16_INPUT_DIR (see readInputs), written by whichever FKL program runs first and read by pytorch-attention/e16_baselines.py,
// so that both compute on the same bytes: Q, K, V and the int8 and e4m3 K and V with their fp32 per-token scales, generated on the host. The notes of the CSV record
// the size and SHA-256 of every input file.
// Variants (all with the library as it is: no tuning, the tile sizes are the ones the library picks):
//   mma_raw                  FlashAttentionMmaDPP, Q, K, V as raw bf16 reads (the bf16 cp.async schedule)
//   mma_int8kv_fused         K and V as int8 with one fp32 scale per token, dequantised inside the kernel: the library's quantised schedule copies the raw bytes with cp.async and dequantises shared memory to shared memory
//   mma_int8kv_unfused       the same data: a separate kernel dequantises K and V to bf16 buffers, then mma_raw (the bf16 cp.async schedule). Fused against unfused therefore compares an in-kernel dequantisation pass
//                            with a separate kernel, and the two attention kernels differ in schedule details and tile sizes (columns block_q, block_kv, num_warps); only the epilogue pair (same kernel) is a pure fusion effect
//   mma_fp8kv_fused / _unfused   the same with e4m3 K and V (one fp32 scale per token)
//   mma_epilogue_fused       mma_raw with the epilogue o * a + b as an IOp chain (Mul then Add) in registers before the single write
//   mma_epilogue_separate    mma_raw, then a separate kernel that reads the bf16 output, applies o * a + b and writes it back
//   simt_raw                 FlashAttentionDPP (one warp per query row), the reference point; run up to E16_SIMT_MAX_SEQ only
// Per variant and configuration: the time per call (E16_ITERS blocks of back-to-back calls between two CUDA events, each block about E16_BLOCK_MS long, after the warm-up: median, smallest, largest) and the time of a single synchronised call,
// every call launching its own argument set, built before the events (see timeCall; its host time is prep_us_per_call), TFLOP/s = 4 BH S^2 D (halved for causal) / time, the
// largest and mean absolute error of the output against an fp32 attention of the same (dequantised) inputs for two heads (the first and the last), the kernels of one call and, for each,
// registers, local memory per thread, static and dynamic shared memory and the blocks per SM, from capturing the call into a CUDA graph and reading its kernel nodes.
// Environment: E16_INPUT_DIR (the folder of the input files, default the working directory), E16_DIMS (64,128), E16_SEQS (512,1024,2048,4096,8192), E16_TOKENS (262144; BH = max(E16_MIN_BH, tokens / S)), E16_BH (fixed BH for every S, default 0 = off), E16_MIN_BH (4), E16_CAUSAL (0,1),
// E16_VARIANTS (prefixes, default all), E16_ITERS (15 blocks), E16_BLOCK_MS (20), E16_WARMUP (10 calls) and E16_WARMUP_MS (300: and at least this much GPU time, so that the clock is back at its boost), E16_SIMT_MAX_SEQ (2048),
// E16_ROTATE (0: the offset of the order of the variants, see ORDER in Runner::run; the long runs use 0, 1, 2), E16_PROFILE (1: one call per variant, no timing, for Nsight Compute), E16_EPI_A (1.25), E16_EPI_B (0.5).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/nan_safe.h>
#include <common/sha256.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/attention/flash_attention.h>
#include <fused_kernel/algorithms/attention/flash_attention_mma.h>

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cxxabi.h>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {

// ------------------------------------------------------------------------------------------------------------------------------------------------------ data
// Q, K and V: bf16 values approximately normal(0, 1), the sum of four uniforms taken from a splitmix hash of (seed + index), centred and scaled. Seeds: 11 (Q), 22 (K), 33 (V), each + 1000 D + S.
// K and V are also quantised per token (one fp32 scale per token): int8 with scale = max|x| / 127 and data = round(x / scale), e4m3 with scale = max|x| / 448 and data = the e4m3 byte of x / scale.
// This is the generator of the program's earlier device kernels, run on the host: it gives the same bytes (checked against the device kernels for D = 64 and 128).
uint64_t splitmix(uint64_t x) {
    x += 0x9e3779b97f4a7c15ull;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

void fillNormal(std::vector<__nv_bfloat16>& p, const uint64_t seed) {
    for (size_t i = 0; i < p.size(); ++i) {
        const uint64_t h = splitmix(seed + i);
        float s = 0.f;
        for (int k = 0; k < 4; ++k) { s += static_cast<float>((h >> (16 * k)) & 0xffff) * (1.f / 65536.f); }
        p[i] = __float2bfloat16((s - 2.f) * 1.7320508f);
    }
}

void quantizeTokens(const std::vector<__nv_bfloat16>& x, const int d, const bool fp8, std::vector<int8_t>& q, std::vector<float>& scales) {
    const size_t tokens = x.size() / d;
    q.resize(x.size());
    scales.resize(tokens);
    for (size_t t = 0; t < tokens; ++t) {
        float mx = 0.f;
        for (int e = 0; e < d; ++e) { mx = std::fmax(mx, std::fabs(__bfloat162float(x[t * d + e]))); }
        const float sc = mx > 0.f ? mx / (fp8 ? 448.f : 127.f) : 1.f;
        scales[t] = sc;
        for (int e = 0; e < d; ++e) {
            const float v = __bfloat162float(x[t * d + e]) / sc;
            q[t * d + e] = fp8 ? static_cast<int8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3)) : static_cast<int8_t>(std::nearbyint(v));
        }
    }
}

struct Inputs {
    std::vector<__nv_bfloat16> q, k, v;
    std::vector<int8_t> k8, v8, kf8, vf8;     // int8 and e4m3 (stored as bytes) K and V
    std::vector<float> k8s, v8s, kf8s, vf8s;  // their per-token scales
};

// The input file of a configuration, in this order: Q, K, V (bf16, BH x S x D each), then K and V in int8, each followed by its fp32 scales (BH x S), then K and V in e4m3, each
// followed by its scales. Raw bytes in the native little-endian order, no header; its size is 10 n + 16 BH S bytes for n = BH S D. Whichever program runs first
// (e16_attention.cu or e16_query_tile.cu) writes it, with the generator above; both read it, and so does pytorch-attention/e16_baselines.py.
std::string inputDir() { return std::getenv("E16_INPUT_DIR") ? std::getenv("E16_INPUT_DIR") : "."; }
std::string inputName(const int d, const int seq, const int bh) {
    return "e16_input_d" + std::to_string(d) + "_s" + std::to_string(seq) + "_bh" + std::to_string(bh) + ".bin";
}

// the vectors of the inputs, sized for the configuration, and the byte ranges of the file in its order
std::vector<std::pair<char*, std::streamsize>> sizedParts(Inputs& in, const int d, const int seq, const int bh) {
    const size_t n = static_cast<size_t>(bh) * seq * d, tokens = static_cast<size_t>(bh) * seq;
    in.q.resize(n); in.k.resize(n); in.v.resize(n);
    in.k8.resize(n); in.v8.resize(n); in.kf8.resize(n); in.vf8.resize(n);
    in.k8s.resize(tokens); in.v8s.resize(tokens); in.kf8s.resize(tokens); in.vf8s.resize(tokens);
    const auto bytesOf = [](auto& vec) { return std::make_pair(reinterpret_cast<char*>(vec.data()), static_cast<std::streamsize>(vec.size() * sizeof(vec[0]))); };
    return {bytesOf(in.q), bytesOf(in.k), bytesOf(in.v), bytesOf(in.k8), bytesOf(in.k8s), bytesOf(in.v8), bytesOf(in.v8s),
            bytesOf(in.kf8), bytesOf(in.kf8s), bytesOf(in.vf8), bytesOf(in.vf8s)};
}

void writeInputs(const int d, const int seq, const int bh, const std::string& path) {
    Inputs in;
    const auto parts = sizedParts(in, d, seq, bh);
    const uint64_t base = 1000ull * d + seq;
    fillNormal(in.q, 11 + base);
    fillNormal(in.k, 22 + base);
    fillNormal(in.v, 33 + base);
    quantizeTokens(in.k, d, false, in.k8, in.k8s);
    quantizeTokens(in.v, d, false, in.v8, in.v8s);
    quantizeTokens(in.k, d, true, in.kf8, in.kf8s);
    quantizeTokens(in.v, d, true, in.vf8, in.vf8s);
    std::ofstream out(path, std::ios::binary);
    for (const auto& part : parts) out.write(part.first, part.second);
    out.close();
    if (!out) throw std::runtime_error("cannot write " + path);
}

// a file of another size stops the program
Inputs readInputs(const int d, const int seq, const int bh, const std::string& path) {
    Inputs in;
    const auto parts = sizedParts(in, d, seq, bh);
    const size_t expected = 10 * static_cast<size_t>(bh) * seq * d + 16 * static_cast<size_t>(bh) * seq;
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw std::runtime_error("cannot read " + path);
    const size_t size = static_cast<size_t>(f.tellg());
    if (size != expected) throw std::runtime_error(path + " has " + std::to_string(size) + " bytes, " + std::to_string(expected) + " expected");
    f.seekg(0);
    for (const auto& part : parts) f.read(part.first, part.second);
    if (!f) throw std::runtime_error("cannot read " + path);
    return in;
}

// The input file of every configuration of this run, written when it does not exist, then hashed: the notes of the CSV get the folder and the size and
// SHA-256 of every file, because the files are deleted after a window.
std::string prepareInputFiles(const std::vector<int>& dims, const std::vector<int>& seqs, const int tokens, const int minBh, const int fixedBh) {
    std::string notes = "; input files in " + inputDir() + ":";
    for (const int d : dims) {
        for (const int seq : seqs) {
            const int bh = fixedBh > 0 ? fixedBh : std::max(minBh, tokens / seq);
            const std::string path = inputDir() + "/" + inputName(d, seq, bh);
            bool exists = false;
            {
                std::ifstream probe(path, std::ios::binary);
                exists = static_cast<bool>(probe);
            }
            if (!exists) writeInputs(d, seq, bh, path);
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            const long long bytes = f ? static_cast<long long>(f.tellg()) : -1;
            const std::string hash = fklb::sha256File(path);
            std::cout << "input " << path << (exists ? " (read)" : " (written)") << ": " << bytes << " bytes, sha256 " << hash << std::endl;
            notes += " " + inputName(d, seq, bh) + " " + std::to_string(bytes) + " bytes sha256 " + hash + (exists ? "" : " (written by this run)") + ";";
        }
    }
    return notes;
}

// the unfused prologue: K and V (z = 0, 1) dequantised to bf16 buffers, one kernel
template <int D, bool FP8>
__global__ void dequantKV(const int8_t* kq, const float* ks, const int8_t* vq, const float* vs, __nv_bfloat16* ko, __nv_bfloat16* vo, const long tokens) {
    const int8_t* q = blockIdx.z == 0 ? kq : vq;
    const float* s = blockIdx.z == 0 ? ks : vs;
    __nv_bfloat16* o = blockIdx.z == 0 ? ko : vo;
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
__global__ void affineBf16(__nv_bfloat16* o, const size_t n, const float a, const float b) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) { o[i] = __float2bfloat16(__bfloat162float(o[i]) * a + b); }
}

// the references, fp32, for two heads: float copies of Q, K, V of those heads (dequantised when the variant is quantised)
template <bool FP8>
__global__ void gatherHeadsF32(const __nv_bfloat16* x, float* out, const long headStride, const long head0, const long head1, const long perHead) {
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

// ------------------------------------------------------------------------------------------------------------------------------------------------------ infrastructure
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
        FKLB_CUDA_CHECK(cudaStreamBeginCapture(stream.getCUDAStream(), cudaStreamCaptureModeRelaxed));
        run();
        FKLB_CUDA_CHECK(cudaStreamEndCapture(stream.getCUDAStream(), &graph));
        size_t n = 0;
        FKLB_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
        std::vector<cudaGraphNode_t> nodes(n);
        FKLB_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
        int dev = 0, sms = 0, maxThreadsSM = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
        cudaDeviceGetAttribute(&maxThreadsSM, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        int count = 0;
        std::ostringstream text;
        for (const cudaGraphNode_t node : nodes) {
            cudaGraphNodeType type;
            FKLB_CUDA_CHECK(cudaGraphNodeGetType(node, &type));
            if (type != cudaGraphNodeTypeKernel) { continue; }
            cudaKernelNodeParams kp;
            FKLB_CUDA_CHECK(cudaGraphKernelNodeGetParams(node, &kp));
            cudaFuncAttributes attr;
            FKLB_CUDA_CHECK(cudaFuncGetAttributes(&attr, kp.func));
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
    double prepUs{0};                    // host time of building the argument set of one call (and releasing the one it replaces), mean over the calls of the blocks and of the single calls
};
int& configIndex() { static int index = 0; return index; }
int rotateOffset() { static const int offset = std::atoi(std::getenv("E16_ROTATE") ? std::getenv("E16_ROTATE") : "0"); return offset; }
long long nowMs() { return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(); }
// iters blocks of `reps` calls enqueued back to back between two events (reps chosen from a first call so that a block lasts about blockMs): the time per call is the GPU time of the call, with no gap
// for the host between calls; then iters single calls, each between two events and synchronised.
// Every call launches its own argument set (prepare(): the operations of the call), built before the events. The sets of a block are the elements of a container made once, before timing; before every
// block (and every single call) each element it uses is rebuilt by assignment, and the host time of that, building the new set and releasing the one it replaces, is prepUs (steady_clock, per call).
// The warm-up lasts at least `warmup` calls and at least warmupMs of GPU time: the clock of the GPU must be back at its boost, after a light kernel (the SIMT variant) or the start of the process, before the first block.
template <typename Prepare, typename Launch>
Timing timeCall(fk::Stream& stream, const Prepare& prepare, const Launch& launch, const int warmup, const int iters, const double blockMs, const double warmupMs) {
    using Clock = std::chrono::steady_clock;
    constexpr int MAX_REPS = 200;
    cudaEvent_t a, b;
    FKLB_CUDA_CHECK(cudaEventCreate(&a));
    FKLB_CUDA_CHECK(cudaEventCreate(&b));
    std::vector<decltype(prepare())> sets(MAX_REPS, prepare());
    double prepNs = 0.0;
    long prepCalls = 0;
    const auto elapsed = [&](const int calls) {
        const auto h0 = Clock::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        for (int i = 0; i < calls; ++i) { sets[i] = prepare(); }
        fklb::preparedBarrier(sets);
        std::atomic_signal_fence(std::memory_order_seq_cst);
        prepNs += std::chrono::duration<double, std::nano>(Clock::now() - h0).count();
        prepCalls += calls;
        FKLB_CUDA_CHECK(cudaEventRecord(a, stream.getCUDAStream()));
        for (int i = 0; i < calls; ++i) { launch(sets[i]); }
        FKLB_CUDA_CHECK(cudaEventRecord(b, stream.getCUDAStream()));
        FKLB_CUDA_CHECK(cudaEventSynchronize(b));
        float f = 0.f;
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&f, a, b));
        return static_cast<double>(f);
    };
    for (int i = 0; i < warmup; ++i) { launch(prepare()); }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(stream.getCUDAStream()));
    for (double spent = 0.0; spent < warmupMs;) { spent += elapsed(4); }
    const double first = std::max(elapsed(1), 1e-3);
    const int reps = std::max(1, std::min(MAX_REPS, static_cast<int>(std::ceil(blockMs / first))));
    prepNs = 0.0;   // the preparation is reported for the measured blocks and single calls
    prepCalls = 0;
    std::vector<double> blocks, singles;
    const long long tStart = nowMs();
    for (int i = 0; i < iters; ++i) { blocks.push_back(elapsed(reps) / reps); }
    const long long tEnd = nowMs();
    for (int i = 0; i < iters; ++i) { singles.push_back(elapsed(1)); }
    FKLB_CUDA_CHECK(cudaGetLastError());
    FKLB_CUDA_CHECK(cudaEventDestroy(a));
    FKLB_CUDA_CHECK(cudaEventDestroy(b));
    std::sort(blocks.begin(), blocks.end());
    std::sort(singles.begin(), singles.end());
    return {blocks[blocks.size() / 2], blocks.front(), blocks.back(), singles[singles.size() / 2], reps, tStart, tEnd, prepNs / static_cast<double>(prepCalls) / 1000.0};
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
    void errorAgainst(const Point& pt, const __nv_bfloat16* o, const float* qRef, const float* kRef, const float* vRef, double& maxErr, double& meanErr, const float affA, const float affB) {
        const size_t perHead = static_cast<size_t>(pt.seq) * D;
        float* ref = nullptr;
        FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ref), 2 * perHead * sizeof(float)));
        const dim3 grid(pt.seq, 2);
        const size_t smem = static_cast<size_t>(pt.seq) * sizeof(float);
        if (smem > 48 * 1024) { FKLB_CUDA_CHECK(cudaFuncSetAttribute(referenceAttention, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem))); }
        referenceAttention<<<grid, 128, smem>>>(qRef, kRef, vRef, ref, pt.seq, D, pt.causal, 1.f / std::sqrt(static_cast<float>(D)));
        FKLB_CUDA_CHECK(cudaGetLastError());
        std::vector<float> hRef(2 * perHead);
        FKLB_CUDA_CHECK(cudaMemcpy(hRef.data(), ref, hRef.size() * sizeof(float), cudaMemcpyDeviceToHost));
        FKLB_CUDA_CHECK(cudaFree(ref));
        std::vector<__nv_bfloat16> hOut(2 * perHead);
        FKLB_CUDA_CHECK(cudaMemcpy(hOut.data(), o, perHead * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
        FKLB_CUDA_CHECK(cudaMemcpy(hOut.data() + perHead, o + static_cast<size_t>(pt.bh - 1) * perHead, perHead * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost));
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

    static std::string clean(std::string text) {   // commas and line breaks become ';', so that the text stays one field for readers that split lines on commas
        for (char& c : text) { if (c == ',' || c == '\n' || c == '\r') { c = ';'; } }
        return text;
    }

    void row(const Point& pt, const std::string& variant, const int position, const int iters_, const Timing& t, const double maxErr, const double meanErr, const KernelFacts& kf, const std::string& note) {
        const double flops = 4.0 * pt.bh * static_cast<double>(pt.seq) * pt.seq * pt.d * (pt.causal ? 0.5 : 1.0);
        const double tflops = t.median > 0 ? flops / (t.median * 1e-3) / 1e12 : 0.0;
        std::ostringstream prep;   // empty in profile mode (reps = 0), where nothing is timed
        if (t.reps > 0) prep << std::setprecision(9) << t.prepUs;
        csv.row(gpu, variant, pt.d, pt.seq, pt.bh, pt.causal ? 1 : 0, position, iters_, t.reps, t.tStartMs, t.tEndMs, t.median, t.min, t.max, t.single, tflops, maxErr, meanErr, kf.kernels, kf.schedule, kf.blockQ, kf.blockKV, kf.numWarps, kf.splitKV, clean(kf.text), clean(note), prep.str());
        csv.flush();
        std::cout << variant << " D=" << pt.d << " S=" << pt.seq << " BH=" << pt.bh << " causal=" << pt.causal << ": " << t.median << " ms (" << t.reps << " calls per block), single " << t.single << " ms, " << tflops
                  << " TFLOP/s, preparation " << t.prepUs << " us per call, max err " << maxErr << ", kernels " << kf.kernels << (note.empty() ? "" : " | " + note) << std::endl;
    }

    void run(const Point& pt) {
        const size_t n = static_cast<size_t>(pt.bh) * pt.seq * D;
        const long tokens = static_cast<long>(pt.bh) * pt.seq;
        const Inputs in = readInputs(D, pt.seq, pt.bh, inputDir() + "/" + inputName(D, pt.seq, pt.bh));
        // the device copies of the inputs, the output and the bf16 K and V of the unfused prologues
        __nv_bfloat16 *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *kd = nullptr, *vd = nullptr;
        int8_t *k8 = nullptr, *v8 = nullptr, *kf8 = nullptr, *vf8 = nullptr;
        float *k8s = nullptr, *v8s = nullptr, *kf8s = nullptr, *vf8s = nullptr;
        for (__nv_bfloat16** p : {&q, &k, &v, &o, &kd, &vd}) { FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p), n * sizeof(__nv_bfloat16))); }
        for (int8_t** p : {&k8, &v8, &kf8, &vf8}) { FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p), n)); }
        for (float** p : {&k8s, &v8s, &kf8s, &vf8s}) { FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p), tokens * sizeof(float))); }
        FKLB_CUDA_CHECK(cudaMemcpy(q, in.q.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(k, in.k.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(v, in.v.data(), n * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(k8, in.k8.data(), n, cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(v8, in.v8.data(), n, cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(kf8, in.kf8.data(), n, cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(vf8, in.vf8.data(), n, cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(k8s, in.k8s.data(), tokens * sizeof(float), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(v8s, in.v8s.data(), tokens * sizeof(float), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(kf8s, in.kf8s.data(), tokens * sizeof(float), cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy(vf8s, in.vf8s.data(), tokens * sizeof(float), cudaMemcpyHostToDevice));
        // fp32 copies of the two checked heads (first and last): bf16 inputs as they are, and the dequantised int8 and e4m3 K and V
        const size_t perHead = static_cast<size_t>(pt.seq) * D;
        const long headStride = static_cast<long>(perHead);
        const long head0 = 0, head1 = pt.bh - 1;
        const int gathers = static_cast<int>((2 * perHead + 255) / 256);
        float *qRef = nullptr, *kRef = nullptr, *vRef = nullptr, *kRefI8 = nullptr, *vRefI8 = nullptr, *kRefF8 = nullptr, *vRefF8 = nullptr;
        for (float** p : {&qRef, &kRef, &vRef, &kRefI8, &vRefI8, &kRefF8, &vRefF8}) { FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(p), 2 * perHead * sizeof(float))); }
        gatherHeadsF32<false><<<gathers, 256>>>(q, qRef, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsF32<false><<<gathers, 256>>>(k, kRef, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsF32<false><<<gathers, 256>>>(v, vRef, headStride, head0, head1, static_cast<long>(perHead));
        gatherHeadsQuant<false><<<gathers, 256>>>(k8, k8s, kRefI8, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<false><<<gathers, 256>>>(v8, v8s, vRefI8, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<true><<<gathers, 256>>>(kf8, kf8s, kRefF8, headStride, head0, head1, static_cast<long>(perHead), D);
        gatherHeadsQuant<true><<<gathers, 256>>>(vf8, vf8s, vRefF8, headStride, head0, head1, static_cast<long>(perHead), D);
        FKLB_CUDA_CHECK(cudaGetLastError());
        FKLB_CUDA_CHECK(cudaDeviceSynchronize());

        const dim3 dqGrid((tokens * D + 255) / 256, 1, 2);

        // The variants: each builds the argument set of one call (prepare: the operations it launches) and launches it (launch: the library call written out).
        // timeCall gives every call its own set; once is one call, for the error check, the graph capture and the profile run.
        struct V {
            std::string name;
            std::function<void()> once;
            std::function<Timing()> timed;
            const float* kRefP;
            const float* vRefP;
            float a, b;
            int iters;
            int group;    // 0 raw, 1 int8 pair, 2 fp8 pair, 3 epilogue pair, 4 SIMT (always last)
        };
        std::vector<V> variants;
        const auto add = [&](const std::string& name, const auto& prepare, const auto& launch, const float* kr, const float* vr, const float a, const float b, const int its, const int group) {
            variants.push_back({name, [=] { launch(prepare()); },
                                [=] { return timeCall(stream, prepare, launch, warmup, its, blockMs, warmupMs); }, kr, vr, a, b, its, group});
        };
        const int simtIters = std::max(3, iters / 6);
        add("mma_raw",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(k, pt.bh, pt.seq, D), fk::makeAttentionRead(v, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
            }, kRef, vRef, 1.f, 0.f, iters, 0);
        add("mma_int8kv_fused",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeInt8KVRead(k8, k8s, pt.bh, pt.seq, D), fk::makeInt8KVRead(v8, v8s, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
            }, kRefI8, vRefI8, 1.f, 0.f, iters, 1);
        add("mma_int8kv_unfused",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(kd, pt.bh, pt.seq, D), fk::makeAttentionRead(vd, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                dequantKV<D, false><<<dqGrid, 256, 0, stream.getCUDAStream()>>>(k8, k8s, v8, v8s, kd, vd, tokens);
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
            }, kRefI8, vRefI8, 1.f, 0.f, iters, 1);
        add("mma_fp8kv_fused",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeFp8KVRead(kf8, kf8s, pt.bh, pt.seq, D), fk::makeFp8KVRead(vf8, vf8s, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
            }, kRefF8, vRefF8, 1.f, 0.f, iters, 2);
        add("mma_fp8kv_unfused",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(kd, pt.bh, pt.seq, D), fk::makeAttentionRead(vd, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                dequantKV<D, true><<<dqGrid, 256, 0, stream.getCUDAStream()>>>(kf8, kf8s, vf8, vf8s, kd, vd, tokens);
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
            }, kRefF8, vRefF8, 1.f, 0.f, iters, 2);
        add("mma_epilogue_fused",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(k, pt.bh, pt.seq, D), fk::makeAttentionRead(v, pt.bh, pt.seq, D),
                                         fk::Mul<float>::build(epiA).then(fk::Add<float>::build(epiB))); },
            [&](const auto& set) {
                const auto& [qr, kr, vr, epilogue] = set;
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream, -1.f, epilogue);
            }, kRef, vRef, epiA, epiB, iters, 3);
        add("mma_epilogue_separate",
            [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(k, pt.bh, pt.seq, D), fk::makeAttentionRead(v, pt.bh, pt.seq, D)); },
            [&](const auto& set) {
                const auto& [qr, kr, vr] = set;
                fk::executeFlashAttentionMma<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
                affineBf16<<<(n + 255) / 256, 256, 0, stream.getCUDAStream()>>>(o, n, epiA, epiB);
            }, kRef, vRef, epiA, epiB, iters, 3);
        if (pt.seq <= simtMaxSeq) {
            add("simt_raw",
                [&] { return std::make_tuple(fk::makeAttentionRead(q, pt.bh, pt.seq, D), fk::makeAttentionRead(k, pt.bh, pt.seq, D), fk::makeAttentionRead(v, pt.bh, pt.seq, D)); },
                [&](const auto& set) {
                    const auto& [qr, kr, vr] = set;
                    fk::executeFlashAttention<D>(qr, kr, vr, o, pt.bh, pt.seq, pt.seq, pt.causal, stream);
                }, kRef, vRef, 1.f, 0.f, simtIters, 4);
        }
        // ORDER: the groups {raw}, {int8 pair}, {fp8 pair}, {epilogue pair} are rotated by (configuration index + E16_ROTATE) mod 4 and the two members of a pair are swapped when that shift is odd, so that over the
        // configurations and the runs every variant is measured at several positions and the members of a pair are always adjacent in both orders; the SIMT variant is always last.
        const int shift = (configIndex()++ + rotateOffset()) % 4;
        std::vector<V> ordered;
        for (int gi = 0; gi < 4; ++gi) {
            std::vector<const V*> members;
            for (const V& v : variants) { if (v.group == (gi + shift) % 4) { members.push_back(&v); } }
            if (shift % 2 == 1) { std::reverse(members.begin(), members.end()); }
            for (const V* m : members) { ordered.push_back(*m); }
        }
        for (const V& v : variants) { if (v.group == 4) { ordered.push_back(v); } }
        const bool profile = envString("E16_PROFILE", "0") == "1";   // one call per variant and no timing: for Nsight Compute (ncu --kernel-name regex:"FlashAttention|dequantKV")
        int position = 0;
        for (const V& variant : ordered) {
            if (!variantWanted(variant.name)) { continue; }
            ++position;
            FKLB_CUDA_CHECK(cudaMemsetAsync(o, 0, n * sizeof(__nv_bfloat16), stream.getCUDAStream()));
            variant.once();
            FKLB_CUDA_CHECK(cudaGetLastError());
            FKLB_CUDA_CHECK(cudaStreamSynchronize(stream.getCUDAStream()));
            double maxErr = 0.0, meanErr = 0.0;
            errorAgainst(pt, o, qRef, variant.kRefP, variant.vRefP, maxErr, meanErr, variant.a, variant.b);
            if (profile) {
                row(pt, variant.name, position, 0, Timing{}, maxErr, meanErr, KernelFacts{}, "profile run: one call, not timed");
                continue;
            }
            const KernelFacts facts = captureFacts(stream, variant.once);
            const Timing t = variant.timed();
            row(pt, variant.name, position, variant.iters, t, maxErr, meanErr, facts, "");
        }
        for (void* p : {static_cast<void*>(q), static_cast<void*>(k), static_cast<void*>(v), static_cast<void*>(o), static_cast<void*>(kd), static_cast<void*>(vd),
                        static_cast<void*>(k8), static_cast<void*>(v8), static_cast<void*>(kf8), static_cast<void*>(vf8), static_cast<void*>(k8s), static_cast<void*>(v8s),
                        static_cast<void*>(kf8s), static_cast<void*>(vf8s), static_cast<void*>(qRef), static_cast<void*>(kRef), static_cast<void*>(vRef),
                        static_cast<void*>(kRefI8), static_cast<void*>(vRefI8), static_cast<void*>(kRefF8), static_cast<void*>(vRefF8)}) {
            FKLB_CUDA_CHECK(cudaFree(p));
        }
    }
};

template <int D>
void runDim(fklb::CsvWriter& csv, const std::vector<int>& seqs, const std::vector<int>& causals, const int tokens, const int minBh, const int fixedBh, const std::string& gpu) {
    Runner<D> runner(csv, std::atoi(envString("E16_WARMUP", "10").c_str()), std::atoi(envString("E16_ITERS", "15").c_str()), std::atoi(envString("E16_SIMT_MAX_SEQ", "2048").c_str()),
                     std::atof(envString("E16_BLOCK_MS", "20").c_str()), std::atof(envString("E16_WARMUP_MS", "300").c_str()), static_cast<float>(std::atof(envString("E16_EPI_A", "1.25").c_str())), static_cast<float>(std::atof(envString("E16_EPI_B", "0.5").c_str())), gpu);
    for (const int seq : seqs) {
        for (const int causal : causals) {
            runner.run(Point{D, seq, fixedBh > 0 ? fixedBh : std::max(minBh, tokens / seq), causal != 0});
        }
    }
}
}  // namespace

int launch() {
    cudaDeviceProp prop;
    FKLB_CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    const std::vector<int> dims = envList("E16_DIMS", "64,128"), seqs = envList("E16_SEQS", "512,1024,2048,4096,8192"), causals = envList("E16_CAUSAL", "0,1");
    const int tokens = std::atoi(envString("E16_TOKENS", "262144").c_str()), minBh = std::atoi(envString("E16_MIN_BH", "4").c_str()), fixedBh = std::atoi(envString("E16_BH", "0").c_str());
    const std::string inputNotes = prepareInputFiles(dims, seqs, tokens, minBh, fixedBh);
    fklb::CsvWriter csv("e16_attention.csv", "E16 forward attention with FKL: the tensor-core DPP and the SIMT DPP against their unfused alternatives",
                        {"gpu", "variant", "d", "seq", "bh", "causal", "position", "blocks", "calls_per_block", "t_start_ms", "t_end_ms", "time_ms_median", "time_ms_min", "time_ms_max", "single_ms_median", "tflops_median", "max_abs_err", "mean_abs_err", "kernels", "schedule", "block_q", "block_kv", "num_warps", "split_kv", "kernel_facts", "note", "prep_us_per_call"},
                        "bf16 Q, K, V (BH, S, D) contiguous, normal(0, 1), fp32 accumulation, bf16 output; inputs: the file e16_input_d<D>_s<S>_bh<BH>.bin of E16_INPUT_DIR, generated on the host (written by whichever FKL program runs first, read by pytorch-attention/e16_baselines.py; the size and SHA-256 of every file are at the end of these notes); "
                        "time_ms_* = per call, from `blocks` blocks of `calls_per_block` calls enqueued back to back between two CUDA events "
                        "(the GPU time of the call, no host gap between calls; the block lasts about E16_BLOCK_MS), after a warm-up of at least E16_WARMUP_MS of GPU time; every call launches its own argument set (the operations of the call), rebuilt before the events; "
                        "prep_us_per_call = host time (steady_clock) of building that set and releasing the one it replaces, per call; position = place of the variant in the order of its configuration (ORDER: groups rotated by configuration + E16_ROTATE, pair members swapped on odd shifts, SIMT last); t_start_ms and t_end_ms = wall clock of the blocks, for the join with the SM clock sampler; single_ms_median = median of `blocks` single calls, each between two events and synchronised (launch latency included); "
                        "TFLOP/s = 4 BH S^2 D (halved for causal) / time_ms_median; error against an fp32 attention of the same (dequantised) inputs for the first and the last head; kernels and per-kernel facts from capturing one call "
                        "into a CUDA graph; the library chooses the tiles (no tuning: <D, 0, 0, 4> and <D, 32, 4> are the library's defaults); mma_epilogue_*: o * " + envString("E16_EPI_A", "1.25") + " + " + envString("E16_EPI_B", "0.5") + "; SCHEDULES (columns schedule, block_q, block_kv, num_warps, split_kv: from the template arguments of the attention kernel; the LDGSTS count of every instantiation is in e16_sass_counts.csv): mma_raw, mma_epilogue_* and the bf16 buffers "
                        "of the *_unfused variants run the library's bf16 cp.async schedule (16-byte cp.async copies of K and V tiles); the fused int8 and fp8 variants run the library's quantised schedule: the raw quantised bytes are copied with cp.async (8 bytes) into a "
                        "staging area of shared memory and dequantised shared memory to shared memory inside the kernel, the per-token scales are read from global memory; each path has its own tile sizes (the library's choice). A fused-against-unfused prologue "
                        "pair therefore compares an in-kernel shared-memory dequantisation pass (half the global bytes for K and V) with a separate dequantisation kernel that writes bf16 to global memory, and the two kernels differ in schedule details and tile "
                        "sizes as well as in fusion: it is not a pure fusion effect. The epilogue pair runs the same kernel and schedule and differs only in the epilogue" + inputNotes);
    for (const int d : dims) {
        if (d == 64) { runDim<64>(csv, seqs, causals, tokens, minBh, fixedBh, prop.name); }
        else if (d == 128) { runDim<128>(csv, seqs, causals, tokens, minBh, fixedBh, prop.name); }
        else { throw std::runtime_error("head dimension " + std::to_string(d) + " is not compiled"); }
    }
    return 0;
}
