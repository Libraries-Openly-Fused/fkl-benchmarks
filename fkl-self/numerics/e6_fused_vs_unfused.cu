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

// E6 (reduced set), what remains: fused against kernel-per-op (P1 chains of 1 to 1,000 Mul/Add pairs; P2 and P3, the resize -> reorder -> Mul -> Sub -> Div -> split image pipelines in both resize conventions) with
// -fmad on and off. The program is built twice from this source (-fmad on: preset rtx4090; -fmad off: preset rtx4090-fmad-off) and run once per build:
//   E6_TAG=fmad_on|fmad_off  E6_OUT=<directory for the raw outputs, outside the repository>  ./e6_fused_vs_unfused_cu
// The raw outputs of every case are written as bit patterns (<E6_OUT>/<tag>/<case>.bin) for the comparison across the two builds (scripts/e6_analyse.py); the CSV written here holds the within-build results.
// The results in results/rtx4090/e6-fused-vs-unfused and ...-run2 were produced by the FULL program, which also held the special-value study and the aliasing test: de47e88 (run 1) and 4203918 (run 2); those two
// studies were removed from this file afterwards (git history keeps them); P1, P2 and P3 are unchanged. Host references: -ffp-contract=off (see the CMake target).
#include <common/status_check.h>
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
std::string g_tag, g_out;
constexpr int SEEDS = 5;
constexpr int SEED0 = 1000;

// ---------- comparison of two float arrays by bit pattern ----------
inline uint32_t bitsOf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
#define E6_EXTRA(st) (st).maxAbs, (st).magAtMaxUlp, ((st).maxOut > 0 ? (st).maxAbs / (st).maxOut : 0.0), (st).nSmall, (st).maxUlpOutside
inline int64_t mono(float f) { const uint32_t u = bitsOf(f); return (u & 0x80000000u) ? -static_cast<int64_t>(u & 0x7fffffffu) : static_cast<int64_t>(u); }  // monotonic map; +0 and -0 both 0

struct Stats {
    size_t n = 0, differing = 0, nanA = 0, nanB = 0, infA = 0, infB = 0, nanMismatch = 0, signedZeroDiff = 0;
    double maxUlp = 0, sumUlp = 0;
    size_t finitePairs = 0;
    // run 2 (new columns): maximum absolute difference, |b| at the element of maximum ULP, the largest |value| of either side, elements whose both values are below 1e-3 in magnitude
    // (the cancelling zone), and the maximum ULP over the elements outside that zone
    double maxAbs = 0, magAtMaxUlp = 0, maxOut = 0, maxUlpOutside = 0;
    size_t nSmall = 0;
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
        if (u > maxUlp) magAtMaxUlp = std::abs(static_cast<double>(b));
        maxAbs = std::max(maxAbs, std::abs(static_cast<double>(a) - static_cast<double>(b)));
        maxOut = std::max(maxOut, std::max(std::abs(static_cast<double>(a)), std::abs(static_cast<double>(b))));
        if (std::max(std::abs(a), std::abs(b)) < 1e-3f) ++nSmall; else maxUlpOutside = std::max(maxUlpOutside, u);
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
    fk::Binary<fk::StaticLoop<fk::FusedOperation<fk::Binary<fk::Mul<float>>, fk::Binary<fk::Add<float>>>, PAIRS>> loop;
    fk::get_opt<0>(loop.params).params = P1_A;
    fk::get_opt<1>(loop.params).params = P1_B;
    const auto readOp = fk::PerThreadRead<fk::ND::_2D, float>::build(in.ptr());
    const auto writeOp = fk::PerThreadWrite<fk::ND::_2D, float>::build(out.ptr());
    const auto tDetails = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>::build_details(readOp, loop, writeOp);
    const dim3 block(32, 8);
    const dim3 grid((P1_W + block.x - 1) / block.x, (P1_H + block.y - 1) / block.y);
    fk::launchTransformDPP_Kernel<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED, true><<<grid, block, 0, s>>>(tDetails, readOp, loop, writeOp);
    FKLB_CUDA_CHECK(cudaGetLastError());
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
    FKLB_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> f(P1_W * P1_H), p(P1_W * P1_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(f.data(), P1_W * 4, dF.ptr().data, dF.ptr().dims.pitch, P1_W * 4, P1_H, cudaMemcpyDeviceToHost));
    FKLB_CUDA_CHECK(cudaMemcpy2D(p.data(), P1_W * 4, dP.ptr().data, dP.ptr().dims.pitch, P1_W * 4, P1_H, cudaMemcpyDeviceToHost));
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
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "fused_vs_perop", fp.n, int(fp.identical()), fp.differing, fp.maxUlp, fp.nanA, fp.nanB, fp.infA, fp.infB, "all", outMin, outMax, E6_EXTRA(fp));
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "fused_vs_fp64_rounded", f64f.n, int(f64f.identical()), f64f.differing, f64f.maxUlp, f64f.nanA, f64f.nanB, f64f.infA, f64f.infB, "all", maxErrF, 0, E6_EXTRA(f64f));
    csv.row("P1", "pairs=" + std::to_string(pairs), seed, "perop_vs_fp64_rounded", f64p.n, int(f64p.identical()), f64p.differing, f64p.maxUlp, f64p.nanA, f64p.nanB, f64p.infA, f64p.infB, "all", maxErrP, 0, E6_EXTRA(f64p));
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
    FKLB_CUDA_CHECK(cudaMemcpy2D(dInput.ptr().data, inPitch, noise.data(), static_cast<size_t>(g.srcW) * 3, static_cast<size_t>(g.srcW) * 3, g.srcH, cudaMemcpyHostToDevice));
    std::array<fk::Ptr2D<uchar3>, BATCH> crops;
    for (int i = 0; i < BATCH; ++i) crops[i] = dInput.crop2D(fk::Rect(pos[i][0], pos[i][1], g.cropW, g.cropH));
    const auto resizeOp = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, SAMPLING>::build(fk::Size(g.outW, g.outH));
    const auto reorder = fk::VectorReorder<float3, 2, 1, 0>::build();
    const auto mul = fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA));
    const auto sub = fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2]));
    const auto div = fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2]));
    const size_t plane = static_cast<size_t>(g.outW) * g.outH, total = static_cast<size_t>(BATCH) * 3 * plane;
    // fused: ONE kernel, the read is a BatchRead of the crop ROIs followed by the Resize (the code of the E4a program)
    fk::Tensor<float> dFused(g.outW, g.outH, BATCH, 3);
    const auto readOp = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(resizeOp);
    fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, readOp, reorder, mul, sub, div, fk::Write<fk::TensorSplit<float3>>{dFused.ptr()});
    // per-op: one launch per operation and per crop; the intermediates are float3 Ptr2D (outW x outH) in global memory, rounded to fp32 after every operation:
    //   step 1 read+Resize (the resize is the read of the crop ROI: uchar3 -> float3) -> A; step 2 VectorReorder A -> B; step 3 Mul B -> A; step 4 Sub A -> B; step 5 Div B -> A; step 6 TensorSplit write A -> tensor
    fk::Tensor<float> dPer(g.outW, g.outH, BATCH, 3);
    fk::Ptr2D<float3> bufA(g.outW, g.outH), bufB(g.outW, g.outH);
    for (int i = 0; i < BATCH; ++i) {
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops[i]).then(resizeOp), fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), reorder, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufB.ptr()));
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufB.ptr()), mul, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), sub, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufB.ptr()));
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufB.ptr()), div, fk::PerThreadWrite<fk::ND::_2D, float3>::build(bufA.ptr()));
        // the split of crop i into its three planes: the tensor of this crop alone (BATCH = 1) copied to the batch tensor
        fk::Tensor<float> one(g.outW, g.outH, 1, 3);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(st, fk::PerThreadRead<fk::ND::_2D, float3>::build(bufA.ptr()), fk::Write<fk::TensorSplit<float3>>{one.ptr()});
        FKLB_CUDA_CHECK(cudaMemcpyAsync(dPer.ptr().data + static_cast<size_t>(i) * 3 * plane, one.ptr().data, 3 * plane * sizeof(float), cudaMemcpyDeviceToDevice, s));
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(s));
    std::vector<float> f(total), p(total);
    FKLB_CUDA_CHECK(cudaMemcpy(f.data(), dFused.ptr().data, total * 4, cudaMemcpyDeviceToHost));
    FKLB_CUDA_CHECK(cudaMemcpy(p.data(), dPer.ptr().data, total * 4, cudaMemcpyDeviceToHost));
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
        csv.row(pipeline, conv, seed, cmp, st_.n, int(st_.identical()), st_.differing, st_.maxUlp, st_.nanA, st_.nanB, st_.infA, st_.infB, region, extra, 0, E6_EXTRA(st_));
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


}  // namespace

int launch() {
    const char* tag = std::getenv("E6_TAG");
    const char* out = std::getenv("E6_OUT");
    if (tag == nullptr || out == nullptr) { std::cerr << "set E6_TAG (fmad_on or fmad_off) and E6_OUT (a directory outside the repository)" << std::endl; return 1; }
    g_tag = tag; g_out = out;
    const std::string mk = "mkdir -p '" + g_out + "/" + g_tag + "'";
    if (std::system(mk.c_str()) != 0) return 1;
    cudaStream_t s;
    FKLB_CUDA_CHECK(cudaStreamCreate(&s));
    fklb::CsvWriter csv("e6_fused_vs_unfused_" + g_tag + ".csv", "E6 fused against kernel-per-op", {"pipeline", "case", "seed", "comparison", "n", "identical", "differing", "max_ulp", "nan_a", "nan_b", "inf_a", "inf_b", "region", "x1", "x2", "max_abs", "mag_at_max_ulp", "max_abs_over_max_out", "n_small_lt_1e-3", "max_ulp_outside_small"},
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
            FKLB_CUDA_CHECK(cudaMemcpy2D(dIn.ptr().data, dIn.ptr().dims.pitch, h.data(), P1_W * 4, P1_W * 4, P1_H, cudaMemcpyHostToDevice));
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
    csv.flush();
    std::cout << "E6 " << g_tag << ": done, raw outputs in " << g_out << "/" << g_tag << std::endl;
    return 0;
}
