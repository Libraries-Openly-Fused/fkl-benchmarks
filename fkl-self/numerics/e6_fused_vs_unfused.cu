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

// E6 (reduced set): fused against kernel-per-op, special values, aliasing. The semantics and what counts as the specified result are in
// README_e6_fused_vs_unfused.md of this folder (written and committed before this program). The program is built twice from this source
// (-fmad on: preset rtx4090; -fmad off: preset rtx4090-fmad-off) and run once per build:
//   E6_TAG=fmad_on|fmad_off  E6_OUT=<directory for the raw outputs, outside the repository>  ./e6_fused_vs_unfused_cu
// The raw outputs of every case are written as bit patterns (<E6_OUT>/<tag>/<case>.bin) for the comparison across the two builds
// (scripts/e6_analyse.py); the CSVs written here are the within-build results. Host references: -ffp-contract=off (see the CMake target).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/nan_safe.h>
#include <common/resize_reference.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/static_loop.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {
#define E6_CHECK(call)                                                                                                        \
    {                                                                                                                         \
        const cudaError_t err_ = (call);                                                                                      \
        if (err_ != cudaSuccess) {                                                                                            \
            std::cerr << #call << " failed: " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            std::exit(2);                                                                                                     \
        }                                                                                                                     \
    }

std::string g_tag, g_out;
constexpr int SEEDS = 5;
constexpr int SEED0 = 1000;

// ---------- comparison of two float arrays by bit pattern ----------
inline uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline int64_t mono(float f) { const uint32_t u = bitsOf(f); return (u & 0x80000000u) ? -static_cast<int64_t>(u & 0x7fffffffu) : static_cast<int64_t>(u); }  // monotonic map; +0 and -0 both 0

struct Stats {
    size_t n = 0, differing = 0, nanA = 0, nanB = 0, infA = 0, infB = 0, nanMismatch = 0, signedZeroDiff = 0;
    double maxUlp = 0, sumUlp = 0;
    size_t finitePairs = 0;
    void add(const float a, const float b) {
        ++n;
        const bool na = std::isnan(a), nb = std::isnan(b);
        nanA += na; nanB += nb; infA += std::isinf(a); infB += std::isinf(b);
        if (bitsOf(a) != bitsOf(b)) ++differing;
        if (na != nb) { ++nanMismatch; return; }
        if (na) return;
        if (a == 0.f && b == 0.f && bitsOf(a) != bitsOf(b)) ++signedZeroDiff;
        if (std::isinf(a) || std::isinf(b)) { if (a != b) { maxUlp = std::numeric_limits<double>::infinity(); } return; }
        const double u = std::abs(static_cast<double>(mono(a) - mono(b)));
        maxUlp = fklb::maxKeepNaN(maxUlp, u);
        sumUlp += u; ++finitePairs;
    }
    bool identical() const { return differing == 0; }
};

void dump(const std::string& name, const void* data, const size_t bytes) {
    const std::string path = g_out + "/" + g_tag + "/" + name + ".bin";
    std::ofstream f(path, std::ios::binary);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
}

// ---------- P1: chain of PAIRS x (Mul, Add) on a float array ----------
constexpr int P1_W = 480, P1_H = 270;
constexpr float P1_A = 0.999f, P1_B = 0.003f;   // fixed point 3.0, inputs in [0.5, 2]: values stay in [0.5, 3] at every length (checked and printed)

template <int PAIRS>
void p1Fused(fk::Ptr2D<float>& in, fk::Ptr2D<float>& out, cudaStream_t s) {
    using Loop = fk::Binary<fk::StaticLoop<fk::FusedOperation<fk::Binary<fk::Mul<float>>, fk::Binary<fk::Add<float>>>, PAIRS>>;
    Loop loop;
    fk::get_opt<0>(loop.params).params = P1_A;
    fk::get_opt<1>(loop.params).params = P1_B;
    const auto readOp = fk::PerThreadRead<fk::ND::_2D, float>::build(in.ptr());
    const auto writeOp = fk::PerThreadWrite<fk::ND::_2D, float>::build(out.ptr());
    const auto tDetails = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>::build_details(readOp, loop, writeOp);
    const dim3 block(32, 8);
    const dim3 grid((P1_W + block.x - 1) / block.x, (P1_H + block.y - 1) / block.y);
    fk::launchTransformDPP_Kernel<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED, true><<<grid, block, 0, s>>>(tDetails, readOp, loop, writeOp);
    E6_CHECK(cudaGetLastError());
}

// one operation per launch: Mul then Add, ping-pong between two buffers, every intermediate is a float Ptr2D in global memory
void p1PerOp(fk::Ptr2D<float>& in, fk::Ptr2D<float>& tmp, fk::Ptr2D<float>& out, const int pairs, cudaStream_t s) {
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    const auto mulOp = fk::Mul<float>::build(P1_A);
    const auto addOp = fk::Add<float>::build(P1_B);
    fk::Ptr2D<float>* src = &in;
    for (int i = 0; i < pairs; ++i) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(src->ptr()), mulOp, fk::PerThreadWrite<fk::ND::_2D, float>::build(tmp.ptr()));
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(tmp.ptr()), addOp, fk::PerThreadWrite<fk::ND::_2D, float>::build(out.ptr()));
        src = &out;
    }
}

void p1CompareOne(const int pairs, const int seed, fklb::CsvWriter& csv, fk::Ptr2D<float>& dIn, fk::Ptr2D<float>& dTmp, fk::Ptr2D<float>& dF, fk::Ptr2D<float>& dP, cudaStream_t s,
                  const std::vector<float>& hIn, void (*fused)(fk::Ptr2D<float>&, fk::Ptr2D<float>&, cudaStream_t)) {
    fused(dIn, dF, s);
    p1PerOp(dIn, dTmp, dP, pairs, s);
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> f(P1_W * P1_H), p(P1_W * P1_H);
    E6_CHECK(cudaMemcpy2D(f.data(), P1_W * 4, dF.ptr().data, dF.ptr().dims.pitch, P1_W * 4, P1_H, cudaMemcpyDeviceToHost));
    E6_CHECK(cudaMemcpy2D(p.data(), P1_W * 4, dP.ptr().data, dP.ptr().dims.pitch, P1_W * 4, P1_H, cudaMemcpyDeviceToHost));
    Stats fp, f64f, f64p;
    double maxErrF = 0, maxErrP = 0, outMin = 1e300, outMax = -1e300;
    for (size_t i = 0; i < f.size(); ++i) {
        double r = hIn[i];
        for (int k = 0; k < pairs; ++k) { r = r * static_cast<double>(P1_A); r = r + static_cast<double>(P1_B); }   // fp64 reference, two separate roundings in double
        const float rf = static_cast<float>(r);
        fp.add(f[i], p[i]); f64f.add(f[i], rf); f64p.add(p[i], rf);
        const double ulpRef = std::ldexp(1.0, std::ilogb(static_cast<double>(rf)) - 23);
        maxErrF = fklb::maxKeepNaN(maxErrF, std::abs(static_cast<double>(f[i]) - r) / ulpRef);
        maxErrP = fklb::maxKeepNaN(maxErrP, std::abs(static_cast<double>(p[i]) - r) / ulpRef);
        outMin = std::min<double>(outMin, f[i]); outMax = std::max<double>(outMax, f[i]);
    }
    const std::string name = "p1_pairs" + std::to_string(pairs) + "_seed" + std::to_string(seed);
    dump(name + "_fused", f.data(), f.size() * 4);
    dump(name + "_perop", p.data(), p.size() * 4);
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "fused_vs_perop", fp.n, int(fp.identical()), fp.differing, fp.maxUlp, fp.nanA, fp.nanB, fp.infA, fp.infB, "all", outMin, outMax);
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "fused_vs_fp64_rounded", f64f.n, int(f64f.identical()), f64f.differing, f64f.maxUlp, f64f.nanA, f64f.nanB, f64f.infA, f64f.infB, "all", maxErrF, 0);
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "perop_vs_fp64_rounded", f64p.n, int(f64p.identical()), f64p.differing, f64p.maxUlp, f64p.nanA, f64p.nanB, f64p.infA, f64p.infB, "all", maxErrP, 0);
}

// ---------- P2 / P3: crops -> resize -> reorder -> Mul -> Sub -> Div -> split (the E4a chain) ----------
constexpr float ALPHA = 0.3f;
constexpr float SUBV[3] = {1.f, 4.f, 3.2f};
constexpr float DIVV[3] = {3.2f, 0.6f, 11.8f};

struct Geo { int srcW, srcH, cropW, cropH, outW, outH; };

template <int BATCH, fk::ResizeSampling SAMPLING>
void resizeNormalise(const Geo& g, const std::vector<std::array<int, 2>>& pos, const std::vector<int>& isEdge, const std::vector<unsigned char>& noise, const int seed, const char* pipeline, const char* conv,
                     fklb::CsvWriter& csv, cudaStream_t s) {
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    fk::Ptr2D<uchar3> dInput(g.srcW, g.srcH);
    const size_t inPitch = dInput.ptr().dims.pitch;
    E6_CHECK(cudaMemcpy2D(dInput.ptr().data, inPitch, noise.data(), static_cast<size_t>(g.srcW) * 3, static_cast<size_t>(g.srcW) * 3, g.srcH, cudaMemcpyHostToDevice));
    std::array<fk::Ptr2D<uchar3>, BATCH> crops;
    for (int i = 0; i < BATCH; ++i) crops[i] = dInput.crop(fk::Point{pos[i][0], pos[i][1], 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(g.cropW), static_cast<uint>(g.cropH), static_cast<uint>(inPitch)});
    const auto resizeOp = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, SAMPLING>::build(fk::Size(g.outW, g.outH));
    const auto reorder = fk::VectorReorder<float3, 2, 1, 0>::build();
    const auto mul = fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA));
    const auto sub = fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2]));
    const auto div = fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2]));
    using DPP = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>;
    const size_t plane = static_cast<size_t>(g.outW) * g.outH, total = static_cast<size_t>(BATCH) * 3 * plane;
    // fused: ONE kernel, the read is a BatchRead of the crop ROIs followed by the Resize (the code of the E4a program)
    fk::Tensor<float> dFused(g.outW, g.outH, BATCH, 3);
    const auto readOp = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(resizeOp);
    fk::executeOperations<DPP>(st, readOp, reorder, mul, sub, div, fk::Write<fk::TensorSplit<float3>>{dFused.ptr()});
    // per-op: one launch per operation and per crop; the intermediates are float3 Ptr2D (outW x outH) in global memory, rounded to fp32 after every operation:
    //   step 1 read+Resize (the resize is the read of the crop ROI: uchar3 -> float3) -> A; step 2 VectorReorder A -> B; step 3 Mul B -> A; step 4 Sub A -> B; step 5 Div B -> A; step 6 TensorSplit write A -> tensor
    fk::Tensor<float> dPer(g.outW, g.outH, BATCH, 3);
    fk::Ptr2D<float3> bufA(g.outW, g.outH), bufB(g.outW, g.outH);
    for (int i = 0; i < BATCH; ++i) {
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops[i]).then(resizeOp), fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), reorder, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufB.ptr()));
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufB.ptr()), mul, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), sub, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufB.ptr()));
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufB.ptr()), div, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        // the split of crop i into its three planes: the tensor of this crop alone (BATCH = 1) copied to the batch tensor
        fk::Tensor<float> one(g.outW, g.outH, 1, 3);
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), fk::Write<fk::TensorSplit<float3>>{one.ptr()});
        E6_CHECK(cudaMemcpyAsync(dPer.ptr().data + static_cast<size_t>(i) * 3 * plane, one.ptr().data, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> f(total), p(total);
    E6_CHECK(cudaMemcpy(f.data(), dFused.ptr().data, total * 4, cudaMemcpyDeviceToHost));
    E6_CHECK(cudaMemcpy(p.data(), dPer.ptr().data, total * 4, cudaMemcpyDeviceToHost));
    // fp64 reference (the chain in double with the float constants), in the tensor layout: [crop][channel][y][x], output channel k is the source channel 2 - k
    std::vector<float> ref(total);
    std::vector<double> refD(total), resized(plane * 3);
    for (int i = 0; i < BATCH; ++i) {
        const unsigned char* base = noise.data() + (static_cast<size_t>(pos[i][1]) * g.srcW + pos[i][0]) * 3;
        fklb::resizeBilinear<unsigned char>(base, static_cast<size_t>(g.srcW) * 3, g.cropW, g.cropH, 3, g.outW, g.outH, SAMPLING == fk::ResizeSampling::HALF_PIXEL ? fklb::ResizeSampling::HALF_PIXEL : fklb::ResizeSampling::ASYMMETRIC, resized.data());
        for (int y = 0; y < g.outH; ++y) for (int x = 0; x < g.outW; ++x) for (int k = 0; k < 3; ++k) {
            const double v = (resized[(static_cast<size_t>(y) * g.outW + x) * 3 + (2 - k)] * static_cast<double>(ALPHA) - static_cast<double>(SUBV[k])) / static_cast<double>(DIVV[k]);
            const size_t idx = ((static_cast<size_t>(i) * 3 + k) * g.outH + y) * g.outW + x;
            refD[idx] = v; ref[idx] = static_cast<float>(v);
        }
    }
    // regions: interior and border ("border" = the crop touches the image edge (P2), or the output pixel is in the first/last row or column (P3))
    const bool cropBased = BATCH > 1;
    Stats fpI, fpB, fdI, fdB;
    double maxErrI = 0, maxErrB = 0;
    for (int i = 0; i < BATCH; ++i) for (int k = 0; k < 3; ++k) for (int y = 0; y < g.outH; ++y) for (int x = 0; x < g.outW; ++x) {
        const size_t idx = ((static_cast<size_t>(i) * 3 + k) * g.outH + y) * g.outW + x;
        const bool border = cropBased ? isEdge[i] != 0 : (x == 0 || y == 0 || x == g.outW - 1 || y == g.outH - 1);
        const double ulpRef = std::ldexp(1.0, std::ilogb(static_cast<double>(ref[idx])) - 23);
        const double e = std::abs(static_cast<double>(f[idx]) - refD[idx]) / ulpRef;
        (border ? fpB : fpI).add(f[idx], p[idx]);
        (border ? fdB : fdI).add(f[idx], ref[idx]);
        (border ? maxErrB : maxErrI) = fklb::maxKeepNaN(border ? maxErrB : maxErrI, e);
    }
    const std::string name = std::string(pipeline) + "_" + conv + "_seed" + std::to_string(seed);
    dump(name + "_fused", f.data(), total * 4);
    dump(name + "_perop", p.data(), total * 4);
    auto row = [&](const char* cmp, const char* region, const Stats& st_, const double extra) {
        csv.row(pipeline, conv, seed, cmp, st_.n, int(st_.identical()), st_.differing, st_.maxUlp, st_.nanA, st_.nanB, st_.infA, st_.infB, region, extra, 0);
    };
    row("fused_vs_perop", "interior", fpI, 0); row("fused_vs_perop", "border", fpB, 0);
    row("fused_vs_fp64_rounded", "interior", fdI, maxErrI); row("fused_vs_fp64_rounded", "border", fdB, maxErrB);
}

// P2 crops: 50 crops of CROP_W x CROP_H of the source: the four corners, one ending on the last row, one ending on the last column, 44 random interior ones
std::vector<std::array<int, 2>> p2Positions(const Geo& g, std::mt19937& rng, std::vector<int>& edge) {
    std::vector<std::array<int, 2>> pos;
    const int mx = g.srcW - g.cropW, my = g.srcH - g.cropH;
    pos = {{0, 0}, {mx, 0}, {0, my}, {mx, my}, {static_cast<int>(rng() % (mx - 2)) + 1, my}, {mx, static_cast<int>(rng() % (my - 2)) + 1}};
    edge.assign(6, 1);
    std::uniform_int_distribution<int> dx(1, mx - 1), dy(1, my - 1);
    while (pos.size() < 50) { pos.push_back({dx(rng), dy(rng)}); edge.push_back(0); }
    return pos;
}


// ---------- special values per operation type (the specification is the table of README_e6_fused_vs_unfused.md) ----------
inline float fromBits(const uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
const std::vector<float>& specials() {
    static const std::vector<float> v = {
        fromBits(0x7fc00000u), fromBits(0xffc00000u), fromBits(0x7fa00000u), fromBits(0xffa00000u),   // +qNaN, -qNaN, +sNaN, -sNaN
        INFINITY, -INFINITY, 0.f, -0.f, fromBits(0x00000001u), fromBits(0x007fffffu), fromBits(0x80000001u),   // +-0, smallest and largest positive denormal, smallest negative denormal
        FLT_MIN, FLT_MAX, -FLT_MAX, 0.5f, -0.5f, 1.5f, -1.5f, 2.5f, -2.5f, 126.5f, 127.5f, 254.5f, 255.5f, 255.f, 256.f, 127.f, 128.f, -128.f, -129.f, -127.5f,
        32767.f, 32768.f, -32768.f, -32769.f, 65535.f, 65535.5f, 65536.f, 2147483520.f, 2147483648.f, -2147483648.f, -2147483904.f, 4294967296.f, -4294967296.f, 1.f, -1.f, 3.f};
    return v;
}
std::string hexf(const float f) { char b[32]; std::snprintf(b, sizeof(b), "0x%08x", bitsOf(f)); return b; }

// class-or-value criterion: both NaN (any payload and sign), or the same bits, or finite and within 1 ULP of the float rounding of the double evaluation `ref`
bool specifiedFloat(const float got, const double refDouble, const float expectedBitsRef, const bool useBitsRef) {
    if (std::isnan(refDouble)) return std::isnan(got);
    if (std::isnan(got)) return false;
    if (std::isinf(refDouble)) return std::isinf(got) && (got > 0) == (refDouble > 0);
    const float rf = static_cast<float>(refDouble);
    if (useBitsRef) return bitsOf(got) == bitsOf(expectedBitsRef);
    if (std::isinf(got)) return false;
    return std::abs(mono(got) - mono(rf)) <= 1;
}

struct SvCsv {
    fklb::CsvWriter rows, summary;
    size_t total = 0, failed = 0;
    SvCsv(const std::string& tag)
        : rows("e6_special_values_" + tag + ".csv", "E6 special values: every case", {"operation", "operand", "input", "input_bits", "expected", "observed", "observed_bits", "scored", "pass", "payload_differs"},
               "scored = 0: reported only (Cast of NaN, Inf and out-of-range values is undefined behaviour in C++); pass = the specified class or value of README_e6_fused_vs_unfused.md; payload_differs = both NaN with different bit patterns (not a failure)"),
          summary("e6_special_values_summary_" + tag + ".csv", "E6 special values: per operation", {"operation", "cases", "scored", "pass", "fail", "not_scored"}, "") {}
};

template <typename IOp>
std::vector<float> run1DFloat(const std::vector<float>& in, const IOp& op, cudaStream_t s) {
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    fk::Ptr1D<float> d(static_cast<int>(in.size())), o(static_cast<int>(in.size()));
    E6_CHECK(cudaMemcpy(d.ptr().data, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_1D, float>::build(d.ptr()), op, fk::PerThreadWrite<fk::ND::_1D, float>::build(o.ptr()));
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> out(in.size());
    E6_CHECK(cudaMemcpy(out.data(), o.ptr().data, in.size() * 4, cudaMemcpyDeviceToHost));
    return out;
}
template <typename... IOps>
std::vector<float> runChain(const std::vector<float>& in, cudaStream_t s, const IOps&... ops) {
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    fk::Ptr1D<float> d(static_cast<int>(in.size())), o(static_cast<int>(in.size()));
    E6_CHECK(cudaMemcpy(d.ptr().data, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_1D, float>::build(d.ptr()), ops..., fk::PerThreadWrite<fk::ND::_1D, float>::build(o.ptr()));
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> out(in.size());
    E6_CHECK(cudaMemcpy(out.data(), o.ptr().data, in.size() * 4, cudaMemcpyDeviceToHost));
    return out;
}
template <typename T, typename IOp>
std::vector<T> run1DInt(const std::vector<float>& in, const IOp& op, cudaStream_t s) {
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    fk::Ptr1D<float> d(static_cast<int>(in.size()));
    fk::Ptr1D<T> o(static_cast<int>(in.size()));
    E6_CHECK(cudaMemcpy(d.ptr().data, in.data(), in.size() * 4, cudaMemcpyHostToDevice));
    fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_1D, float>::build(d.ptr()), op, fk::PerThreadWrite<fk::ND::_1D, T>::build(o.ptr()));
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<T> out(in.size());
    E6_CHECK(cudaMemcpy(out.data(), o.ptr().data, in.size() * sizeof(T), cudaMemcpyDeviceToHost));
    return out;
}

void svFloat(SvCsv& c, const std::string& name, const std::string& operand, const std::vector<float>& in, const std::vector<float>& got, const std::vector<float>& expectedFloat, const std::vector<double>& expectedDouble, const bool bitsRef) {
    size_t pass = 0, fail = 0;
    for (size_t i = 0; i < in.size(); ++i) {
        const bool ok = specifiedFloat(got[i], expectedDouble[i], expectedFloat[i], bitsRef);
        const bool payload = std::isnan(got[i]) && std::isnan(expectedFloat[i]) && bitsOf(got[i]) != bitsOf(expectedFloat[i]);
        c.rows.row(name, operand, std::to_string(in[i]), hexf(in[i]), hexf(expectedFloat[i]), std::to_string(got[i]), hexf(got[i]), 1, int(ok), int(payload));
        ok ? ++pass : ++fail;
    }
    c.summary.row(name + " " + operand, in.size(), in.size(), pass, fail, 0);
    c.total += in.size(); c.failed += fail;
}

template <typename T>
void svInt(SvCsv& c, const std::string& name, const std::vector<float>& in, const std::vector<T>& got, const bool saturating) {
    size_t pass = 0, fail = 0, notScored = 0;
    const double lo = static_cast<double>(std::numeric_limits<T>::lowest()), hi = static_cast<double>(std::numeric_limits<T>::max());
    for (size_t i = 0; i < in.size(); ++i) {
        const float x = in[i];
        bool scored = true, ok = true;
        long long expected = 0;
        if (saturating) {   // NaN -> 0, round to nearest even, saturate
            if (std::isnan(x)) expected = 0;
            else { const double r = std::nearbyint(static_cast<double>(x)); expected = static_cast<long long>(std::min(std::max(r, lo), hi)); }
        } else {            // Cast = static_cast: scored only where the C++ conversion is defined (finite, truncation in range)
            if (std::isnan(x) || std::isinf(x) || std::trunc(static_cast<double>(x)) < lo || std::trunc(static_cast<double>(x)) > hi) scored = false;
            else expected = static_cast<long long>(static_cast<T>(x));
        }
        if (scored) { ok = static_cast<long long>(got[i]) == expected; ok ? ++pass : ++fail; } else ++notScored;
        c.rows.row(name, "", std::to_string(x), hexf(x), scored ? std::to_string(expected) : std::string("undefined"), std::to_string(static_cast<long long>(got[i])), "", int(scored), int(ok), 0);
    }
    c.summary.row(name, in.size(), in.size() - notScored, pass, fail, notScored);
    c.total += in.size() - notScored; c.failed += fail;
}

void specialValues(const std::string& tag, cudaStream_t s) {
    SvCsv c(tag);
    const auto& in = specials();
    // arithmetic with an operand set; the expected result is the IEEE float result computed on the host (-ffp-contract=off)
    const float operands[] = {0.f, -0.f, 1.f, 2.f, 0.5f, INFINITY, fromBits(0x7fc00000u)};
    for (const float op : operands) {
        const std::string opn = hexf(op);
        std::vector<float> eAdd(in.size()), eSub(in.size()), eMul(in.size()), eDiv(in.size());
        std::vector<double> dAdd(in.size()), dSub(in.size()), dMul(in.size()), dDiv(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            volatile float a = in[i], b = op;
            eAdd[i] = a + b; eSub[i] = a - b; eMul[i] = a * b; eDiv[i] = a / b;
            dAdd[i] = eAdd[i]; dSub[i] = eSub[i]; dMul[i] = eMul[i]; dDiv[i] = eDiv[i];   // the specified result of a float operation is the IEEE float result
        }
        svFloat(c, "Add", opn, in, run1DFloat(in, fk::Add<float>::build(op), s), eAdd, dAdd, true);
        svFloat(c, "Sub", opn, in, run1DFloat(in, fk::Sub<float>::build(op), s), eSub, dSub, true);
        svFloat(c, "Mul", opn, in, run1DFloat(in, fk::Mul<float>::build(op), s), eMul, dMul, true);
        svFloat(c, "Div", opn, in, run1DFloat(in, fk::Div<float>::build(op), s), eDiv, dDiv, true);
    }
    {   // Mul-add (x * 2 + 0.5) and Normalize (x * 0.3 - 1) / 3.2 as separate float operations: the specified result rounds after each
        std::vector<float> eMA(in.size()), eN(in.size());
        std::vector<double> dMA(in.size()), dN(in.size());
        for (size_t i = 0; i < in.size(); ++i) {
            volatile float a = in[i], two = 2.f, half = 0.5f, k = ALPHA, one = 1.f, d = 3.2f;
            volatile float t = a * two; eMA[i] = t + half;
            volatile float u = a * k; volatile float v = u - one; eN[i] = v / d;
            dMA[i] = eMA[i]; dN[i] = eN[i];
        }
        svFloat(c, "Mul-add", "x*2+0.5", in, runChain(in, s, fk::Mul<float>::build(2.f), fk::Add<float>::build(0.5f)), eMA, dMA, true);
        svFloat(c, "Normalize", "(x*0.3-1)/3.2", in, runChain(in, s, fk::Mul<float>::build(ALPHA), fk::Sub<float>::build(1.f), fk::Div<float>::build(3.2f)), eN, dN, true);
    }
    svInt<unsigned char>(c, "Cast float->uchar", in, run1DInt<unsigned char>(in, fk::Cast<float, unsigned char>::build(), s), false);
    svInt<signed char>(c, "Cast float->schar", in, run1DInt<signed char>(in, fk::Cast<float, signed char>::build(), s), false);
    svInt<short>(c, "Cast float->short", in, run1DInt<short>(in, fk::Cast<float, short>::build(), s), false);
    svInt<unsigned short>(c, "Cast float->ushort", in, run1DInt<unsigned short>(in, fk::Cast<float, unsigned short>::build(), s), false);
    svInt<int>(c, "Cast float->int", in, run1DInt<int>(in, fk::Cast<float, int>::build(), s), false);
    svInt<unsigned char>(c, "SaturateCast float->uchar", in, run1DInt<unsigned char>(in, fk::SaturateCast<float, unsigned char>::build(), s), true);
    svInt<signed char>(c, "SaturateCast float->schar", in, run1DInt<signed char>(in, fk::SaturateCast<float, signed char>::build(), s), true);
    svInt<short>(c, "SaturateCast float->short", in, run1DInt<short>(in, fk::SaturateCast<float, short>::build(), s), true);
    svInt<unsigned short>(c, "SaturateCast float->ushort", in, run1DInt<unsigned short>(in, fk::SaturateCast<float, unsigned short>::build(), s), true);
    svInt<int>(c, "SaturateCast float->int", in, run1DInt<int>(in, fk::SaturateCast<float, int>::build(), s), true);
    // ColorConversion: RGB2Gray with one special value in one channel and 0.5 in the others; channel reorder (bits must move unchanged)
    {
        std::vector<float3> in3;
        std::vector<float> flat;
        for (const float v : in) { in3.push_back(make_float3(v, 0.5f, 0.5f)); in3.push_back(make_float3(0.5f, v, 0.5f)); in3.push_back(make_float3(0.5f, 0.5f, v)); }
        fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
        fk::Ptr1D<float3> d(static_cast<int>(in3.size()));
        fk::Ptr1D<float> og(static_cast<int>(in3.size()));
        fk::Ptr1D<float3> orr(static_cast<int>(in3.size()));
        E6_CHECK(cudaMemcpy(d.ptr().data, in3.data(), in3.size() * sizeof(float3), cudaMemcpyHostToDevice));
        using DPP = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>;
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_1D, float3>::build(d.ptr()), fk::RGB2Gray<float3, float>::build(), fk::PerThreadWrite<fk::ND::_1D, float>::build(og.ptr()));
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_1D, float3>::build(d.ptr()), fk::VectorReorder<float3, 2, 1, 0>::build(), fk::PerThreadWrite<fk::ND::_1D, float3>::build(orr.ptr()));
        E6_CHECK(cudaStreamSynchronize(s));
        std::vector<float> gg(in3.size());
        std::vector<float3> rr(in3.size());
        E6_CHECK(cudaMemcpy(gg.data(), og.ptr().data, gg.size() * 4, cudaMemcpyDeviceToHost));
        E6_CHECK(cudaMemcpy(rr.data(), orr.ptr().data, rr.size() * sizeof(float3), cudaMemcpyDeviceToHost));
        std::vector<float> flatIn, eF(in3.size());
        std::vector<double> eD(in3.size());
        for (size_t i = 0; i < in3.size(); ++i) {
            flatIn.push_back(in3[i].x);
            volatile float x = in3[i].x, y = in3[i].y, z = in3[i].z, k0 = 0.299f, k1 = 0.587f, k2 = 0.114f;
            volatile float p0 = x * k0, p1 = y * k1, p2 = z * k2, s01 = p0 + p1;
            eF[i] = s01 + p2;
            eD[i] = (static_cast<double>(in3[i].x) * 0.299f + static_cast<double>(in3[i].y) * 0.587f) + static_cast<double>(in3[i].z) * 0.114f;   // double evaluation of the formula with the float constants
            if (std::isnan(in3[i].x) || std::isnan(in3[i].y) || std::isnan(in3[i].z)) eD[i] = std::numeric_limits<double>::quiet_NaN();
        }
        svFloat(c, "RGB2Gray", "0.299R+0.587G+0.114B", flatIn, gg, eF, eD, false);
        size_t pass = 0, fail = 0;
        for (size_t i = 0; i < in3.size(); ++i) {
            const bool ok = bitsOf(rr[i].x) == bitsOf(in3[i].z) && bitsOf(rr[i].y) == bitsOf(in3[i].y) && bitsOf(rr[i].z) == bitsOf(in3[i].x);
            c.rows.row("VectorReorder<float3,2,1,0>", "", hexf(in3[i].x) + "," + hexf(in3[i].y) + "," + hexf(in3[i].z), "", "bits moved", hexf(rr[i].x) + "," + hexf(rr[i].y) + "," + hexf(rr[i].z), "", 1, int(ok), 0);
            ok ? ++pass : ++fail;
        }
        c.summary.row("VectorReorder<float3,2,1,0>", in3.size(), in3.size(), pass, fail, 0);
        c.total += in3.size(); c.failed += fail;
    }
    // Resize on a float image with planted NaN and Inf: bilinear, both sampling conventions; expected = the sum of the four weighted neighbours in double with the clamped indices
    {
        constexpr int IW = 8, IH = 8, OW = 16, OH = 16;
        std::vector<float> img(IW * IH);
        std::mt19937 rng(SEED0);
        std::uniform_real_distribution<float> u(1.f, 100.f);
        for (auto& v : img) v = u(rng);
        img[2 * IW + 2] = fromBits(0x7fc00000u); img[5 * IW + 5] = INFINITY; img[5 * IW + 2] = -INFINITY; img[2 * IW + 5] = 0.f;
        fk::Ptr2D<float> d(IW, IH), o(OW, OH);
        E6_CHECK(cudaMemcpy2D(d.ptr().data, d.ptr().dims.pitch, img.data(), IW * 4, IW * 4, IH, cudaMemcpyHostToDevice));
        fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
        using DPP = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>;
        for (int conv = 0; conv < 2; ++conv) {
            if (conv == 0) fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(d.ptr()).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OW, OH))), fk::PerThreadWrite<fk::ND::_2D, float>::build(o.ptr()));
            else fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(d.ptr()).then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OW, OH))), fk::PerThreadWrite<fk::ND::_2D, float>::build(o.ptr()));
            E6_CHECK(cudaStreamSynchronize(s));
            std::vector<float> got(OW * OH), exp(OW * OH), idx(OW * OH);
            std::vector<double> expD(OW * OH);
            E6_CHECK(cudaMemcpy2D(got.data(), OW * 4, o.ptr().data, o.ptr().dims.pitch, OW * 4, OH, cudaMemcpyDeviceToHost));
            const double sc = static_cast<double>(IW) / OW;
            for (int y = 0; y < OH; ++y) for (int x = 0; x < OW; ++x) {
                const double sx = conv == 0 ? x * sc : (x + 0.5) * sc - 0.5, sy = conv == 0 ? y * sc : (y + 0.5) * sc - 0.5;
                const int x0 = static_cast<int>(std::floor(sx)), y0 = static_cast<int>(std::floor(sy));
                const double fx = sx - x0, fy = sy - y0;
                auto at = [&](int xx, int yy) { return static_cast<double>(img[std::min(std::max(yy, 0), IH - 1) * IW + std::min(std::max(xx, 0), IW - 1)]); };
                const double v = (1 - fx) * (1 - fy) * at(x0, y0) + fx * (1 - fy) * at(x0 + 1, y0) + (1 - fx) * fy * at(x0, y0 + 1) + fx * fy * at(x0 + 1, y0 + 1);
                expD[y * OW + x] = v; exp[y * OW + x] = static_cast<float>(v); idx[y * OW + x] = static_cast<float>(y * OW + x);
            }
            svFloat(c, conv == 0 ? "Resize ASYMMETRIC (float 8x8 -> 16x16, NaN/Inf planted)" : "Resize HALF_PIXEL (float 8x8 -> 16x16, NaN/Inf planted)", "output pixel index in 'input'", idx, got, exp, expD, false);
        }
    }
    std::cout << "special values: " << (c.total - c.failed) << " of " << c.total << " scored cases give the specified result (" << c.failed << " exceptions listed in e6_special_values_" << tag << ".csv)" << std::endl;
}

// ---------- aliasing ----------
void aliasing(fklb::CsvWriter& csv, cudaStream_t s) {
    constexpr int W = 1920, H = 1080;
    fk::Stream_<fk::ParArch::GPU_NVIDIA> st(s);
    std::mt19937 rng(SEED0);
    std::uniform_real_distribution<float> u(0.5f, 2.f);
    std::vector<float> h(static_cast<size_t>(W) * H);
    for (auto& v : h) v = u(rng);
    auto upload = [&](fk::Ptr2D<float>& d) { E6_CHECK(cudaMemcpy2D(d.ptr().data, d.ptr().dims.pitch, h.data(), W * 4, W * 4, H, cudaMemcpyHostToDevice)); };
    auto download = [&](fk::Ptr2D<float>& d, std::vector<float>& o, const int w, const int hh) { o.resize(static_cast<size_t>(w) * hh); E6_CHECK(cudaMemcpy2D(o.data(), w * 4, d.ptr().data, d.ptr().dims.pitch, w * 4, hh, cudaMemcpyDeviceToHost)); };
    using DPP = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>;
    // element-wise chain, 1,000 x (Mul, Add) fused: out of place, then in place
    using Loop = fk::Binary<fk::StaticLoop<fk::FusedOperation<fk::Binary<fk::Mul<float>>, fk::Binary<fk::Add<float>>>, 1000>>;
    Loop loop;
    fk::get_opt<0>(loop.params).params = P1_A;
    fk::get_opt<1>(loop.params).params = P1_B;
    fk::Ptr2D<float> in(W, H), out(W, H);
    upload(in);
    fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(in.ptr()), loop, fk::PerThreadWrite<fk::ND::_2D, float>::build(out.ptr()));
    fk::Ptr2D<float> inplace(W, H);
    upload(inplace);
    fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(inplace.ptr()), loop, fk::PerThreadWrite<fk::ND::_2D, float>::build(inplace.ptr()));
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> a, b;
    download(out, a, W, H); download(inplace, b, W, H);
    Stats e;
    for (size_t i = 0; i < a.size(); ++i) e.add(a[i], b[i]);
    csv.row("alias_elementwise_1000pairs", "in_place_vs_out_of_place", 0, "bitwise", e.n, int(e.identical()), e.differing, e.maxUlp, e.nanA, e.nanB, e.infA, e.infB, "all", 0, 0);
    // ReadBack gather in place: Resize to half size (INTER_LINEAR) reads many source pixels per output pixel; written into the same buffer (same pointer and pitch)
    const int OW = W / 2, OH = H / 2;
    const auto rs = fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OW, OH));
    fk::Ptr2D<float> ref(OW, OH);
    upload(in);
    fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(in.ptr()).then(rs), fk::PerThreadWrite<fk::ND::_2D, float>::build(ref.ptr()));
    E6_CHECK(cudaStreamSynchronize(s));
    std::vector<float> r;
    download(ref, r, OW, OH);
    for (int run = 0; run < 3; ++run) {
        fk::Ptr2D<float> buf(W, H);
        upload(buf);
        // the output is a RawPtr over the same memory with the output dimensions and the same pitch
        fk::RawPtr<fk::ND::_2D, float> alias{buf.ptr().data, {static_cast<uint>(OW), static_cast<uint>(OH), static_cast<uint>(buf.ptr().dims.pitch)}};
        fk::executeOperations<DPP>(st, fk::PerThreadRead<fk::ND::_2D, float>::build(buf.ptr()).then(rs), fk::PerThreadWrite<fk::ND::_2D, float>::build(alias));
        E6_CHECK(cudaStreamSynchronize(s));
        std::vector<float> g;
        download(buf, g, OW, OH);
        Stats d;
        for (size_t i = 0; i < r.size(); ++i) d.add(r[i], g[i]);
        csv.row("alias_gather_resize_half", "in_place_vs_out_of_place", run, "bitwise", d.n, int(d.identical()), d.differing, d.maxUlp, d.nanA, d.nanB, d.infA, d.infB, "all", 0, 0);
    }
}
}  // namespace

int launch() {
    const char* tag = std::getenv("E6_TAG");
    const char* out = std::getenv("E6_OUT");
    if (tag == nullptr || out == nullptr) { std::cerr << "set E6_TAG (fmad_on or fmad_off) and E6_OUT (a directory outside the repository)" << std::endl; return 1; }
    g_tag = tag; g_out = out;
    const std::string mk = "mkdir -p '" + g_out + "/" + g_tag + "'";
    if (std::system(mk.c_str()) != 0) return 1;
    cudaStream_t s;
    E6_CHECK(cudaStreamCreate(&s));
    fklb::CsvWriter csv("e6_fused_vs_unfused_" + g_tag + ".csv", "E6 fused against kernel-per-op", {"pipeline", "case", "seed", "comparison", "n", "identical", "differing", "max_ulp", "nan_a", "nan_b", "inf_a", "inf_b", "region", "x1", "x2"},
                        "build tag " + g_tag + "; identical = memcmp of the bit patterns; max_ulp in ULP (integer distance of the monotonic bit mapping; NaN pairs excluded and counted); for fused_vs_fp64_rounded x1 = max error against the unrounded double result in ULP; "
                        "for P1 fused_vs_perop x1,x2 = min and max of the output; the fp64 host reference is compiled with -ffp-contract=off");
    // ---- P1
    {
        fk::Ptr2D<float> dIn(P1_W, P1_H), dTmp(P1_W, P1_H), dF(P1_W, P1_H), dP(P1_W, P1_H);
        for (int seed = 0; seed < SEEDS; ++seed) {
            std::mt19937 rng(SEED0 + seed);
            std::uniform_real_distribution<float> u(0.5f, 2.f);
            std::vector<float> h(static_cast<size_t>(P1_W) * P1_H);
            for (auto& v : h) v = u(rng);
            E6_CHECK(cudaMemcpy2D(dIn.ptr().data, dIn.ptr().dims.pitch, h.data(), P1_W * 4, P1_W * 4, P1_H, cudaMemcpyHostToDevice));
            p1CompareOne(1, seed, csv, dIn, dTmp, dF, dP, s, h, &p1Fused<1>);
            p1CompareOne(10, seed, csv, dIn, dTmp, dF, dP, s, h, &p1Fused<10>);
            p1CompareOne(100, seed, csv, dIn, dTmp, dF, dP, s, h, &p1Fused<100>);
            p1CompareOne(1000, seed, csv, dIn, dTmp, dF, dP, s, h, &p1Fused<1000>);
        }
    }
    csv.flush();
    // ---- P2 (50 crops of 60 x 120 of a 3840 x 2160 noise image -> 64 x 128, six of them on the image edge) and P3 (one 1920 x 1080 image -> 224 x 224, ROI = the whole image)
    for (int seed = 0; seed < SEEDS; ++seed) {
        std::mt19937 rng(SEED0 + 100 + seed);
        {
            const Geo g{3840, 2160, 60, 120, 64, 128};
            std::vector<unsigned char> noise(static_cast<size_t>(g.srcW) * g.srcH * 3);
            for (auto& v : noise) v = static_cast<unsigned char>(rng());
            std::vector<int> edge;
            const auto pos = p2Positions(g, rng, edge);
            resizeNormalise<50, fk::ResizeSampling::ASYMMETRIC>(g, pos, edge, noise, seed, "P2", "ASYMMETRIC", csv, s);
            resizeNormalise<50, fk::ResizeSampling::HALF_PIXEL>(g, pos, edge, noise, seed, "P2", "HALF_PIXEL", csv, s);
        }
        {
            const Geo g{1920, 1080, 1920, 1080, 224, 224};
            std::vector<unsigned char> noise(static_cast<size_t>(g.srcW) * g.srcH * 3);
            for (auto& v : noise) v = static_cast<unsigned char>(rng());
            const std::vector<std::array<int, 2>> pos{{0, 0}};
            const std::vector<int> edge{0};
            resizeNormalise<1, fk::ResizeSampling::ASYMMETRIC>(g, pos, edge, noise, seed, "P3", "ASYMMETRIC", csv, s);
            resizeNormalise<1, fk::ResizeSampling::HALF_PIXEL>(g, pos, edge, noise, seed, "P3", "HALF_PIXEL", csv, s);
        }
        csv.flush();
    }
    specialValues(g_tag, s);
    aliasing(csv, s);
    csv.flush();
    std::cout << "E6 " << g_tag << ": done, raw outputs in " << g_out << "/" << g_tag << std::endl;
    return 0;
}
