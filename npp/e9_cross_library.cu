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

// E9, cross-library demonstration: operations of cvGPUSpeedup, operations of FastNPP and an operation of the user in ONE fk::executeOperations, one kernel launch.
// Input: a 3840 x 2160 uchar3 image of SEEDED NOISE (3 seeds, never constant, so the addressing and the sampling are tested), 50 crops of 60 x 120 (the E4a geometry: six on the edge of the image, 44 random inside),
// each resized to 64 x 128 float3, then: swap R and B, scale by 1/255, the user's gamma correction (per channel powf(x, g), g = 1/2.2 at run time), subtract the mean, divide by the standard deviation, and write
// a (crop, channel, y, x) float tensor.
//   M1  cvGS::resize | fastNPP::SwapChannels | cvGS::multiply | USER gamma | fastNPP::SubC | cvGS::divide | cvGS::split        (one kernel)
//   M2  fastNPP::ResizeBatch | cvGS::cvtColor | fastNPP::MulC | USER gamma | cvGS::subtract | fastNPP::DivC | fastNPP::CopyBatch   (one kernel)
//   R1  the same chain with cvGPUSpeedup operations only (+ the user's gamma), launched by cvGS::executeOperations
//   R2  the same chain with FastNPP operations only (+ the user's gamma), launched by fastNPP::executeOperations
// M1 shares its resize with R1 (cvGS's) and M2 with R2 (FastNPP's), which is why there are two mixed kernels: each mixed kernel is compared with the single-library kernel whose resize it uses, so the comparison does
// not depend on the sampling convention of the FastNPP resize (asymmetric before FastNPP #45, half-pixel after it). R1 against R2 and M1 against M2 are reported too: bit-identical when both resizes sample the same way
// (before #45), different by the sampling convention only after it.
// PREDICTIONS, written before the window: (1) M1 = R1 and M2 = R2 BIT-IDENTICAL (memcmp of the float output, all seeds, NaN count 0); (2) M1 and R1 have identical SASS, M2 and R2 likewise (reported either way);
// (3) one launch for each of the four (kernel nodes of the call captured in a CUDA graph = 1, and nsys in the window); (4) the time of M1 equals that of R1 and M2 that of R2 within the spread of the runs (3 runs,
// rotated order, 3 % GPU-bound flag). WHAT THIS SHOWS AND WHAT NOT: cvGPUSpeedup and FastNPP share authors and the operations of both are the same FKL operations under different names; the demonstration shows that the
// type system and the executor accept operations from two libraries and an operation written by a user in one kernel at no cost, not that third parties will write such libraries. A type-system friction that
// needed a wrapper change would be a finding to report; the libraries are not changed.
// NOTE 2026-11-06 (compile): prediction (2) was found FALSE at compile time. cuobjdump sm_89: M1 34 registers / 552 SASS lines, R1 32 / 544, M2 33 / 544, R2 32 / 560; stack, shared and local memory 0 for all four.
// Reason: the swap of FastNPP is VectorReorderRT (run-time index vector), that of cvGS is the compile-time RGB2BGR ColorConversion, and SubC / DivC of FastNPP are not the same FKL types as cvGS subtract / divide, so the mixed
// kernels contain different ops than their references (same results). The SASS comparison is reported as a size comparison, not as an identity test, and the time comparison M against R contains that difference too. CORRECTION 2026-11-06 (after the window, comment only, the binaries are of commit 9bf0ca9): the comparison of the demangled kernel types shows that the swap is the ONLY differing operation: FastNPP's SwapChannels is VectorReorderRT, the cvGS cvtColor RGB2BGR is VectorReorder<float3, 2, 1, 0>; SubC / DivC and subtract / divide are the SAME FKL types (the clause on them above is wrong).
// Each library reads its own copy of the frame (OpenCV: a GpuMat uploaded from the noise bytes; FastNPP: a cudaMallocPitch image of the same bytes) and
// R1, M2 and R2 write buffers of their own library (R1: a continuous GpuMat; M2 and R2: dense cudaMalloc tensors through plane pointers). Every call of a
// variant builds its batch and its operations: the crop locations may change on every image, so the ROIs and the descriptor arrays are made in the call.
// Environment: E9_NSYS (one call of each variant and of the two listings in a cudaProfilerStart/Stop range, then exit; for nsys), E9_RUN (0, 1, 2: the order of the four variants is rotated by it), E9_SEEDS (3), E9_BLOCK_MS (20), E9_BLOCKS (15).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/status_check.h>

#include <cvGPUSpeedup.cuh>
#include <fast_npp.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/image_processing/resize.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>

#include <npp.h>

#include <algorithm>
#include <array>
#include <cuda_profiler_api.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// ---- SHORT LISTING (for the paper): the user's operation in 7 lines, the same operation as ::Gamma below under the name used in the paper ----
namespace short_listing {
struct Gamma {
    using Parent = fk::BinaryOperation<float3, float, float3, Gamma>;
    DECLARE_BINARY_PARENT
    FK_HOST_DEVICE_FUSE float3 exec(float3 i, float g) {
        return {powf(i.x, g), powf(i.y, g), powf(i.z, g)};
    }
};
}  // namespace short_listing
// ---- END SHORT LISTING (user operation) ----

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128, BATCH = 50;
constexpr float GAMMA = 1.f / 2.2f;
constexpr float SCALE = 1.f / 255.f;
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f};
constexpr float STD[3] = {0.229f, 0.224f, 0.225f};

// The user's operation, defined here, in neither library: per-channel gamma correction with a run-time exponent.
struct Gamma {
    using Parent = fk::BinaryOperation<float3, float, float3, Gamma>;
    DECLARE_BINARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType in, const ParamsType& g) { return {powf(in.x, g), powf(in.y, g), powf(in.z, g)}; }
};

// The input of one seed: each library's own copy of the same noise bytes, and the crop positions
struct Setup {
    cv::cuda::GpuMat src;                         // OpenCV's copy of the frame (read by the cvGPUSpeedup operations)
    Npp8u* nppSrc = nullptr;                      // FastNPP's copy of the frame
    size_t nppStep = 0;
    std::vector<std::array<int, 2>> pos;          // crop origins (x, y)
    std::array<cv::cuda::GpuMat, BATCH> crops;    // ROIs of src, used only by the two listings, which are not timed
    std::vector<int> edge;
    ~Setup() { if (nppSrc) cudaFree(nppSrc); }
};

void makeSetup(Setup& s, const int seed) {
    std::mt19937 rng(1200 + seed);
    std::vector<unsigned char> noise(static_cast<size_t>(SRC_W) * SRC_H * 3);
    for (auto& v : noise) v = static_cast<unsigned char>(rng());
    s.src.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, noise.data()));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&s.nppSrc), &s.nppStep, static_cast<size_t>(SRC_W) * 3, SRC_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(s.nppSrc, s.nppStep, noise.data(), static_cast<size_t>(SRC_W) * 3, static_cast<size_t>(SRC_W) * 3, SRC_H, cudaMemcpyHostToDevice));
    const int mx = SRC_W - CROP_W, my = SRC_H - CROP_H;
    s.pos = {{0, 0}, {mx, 0}, {0, my}, {mx, my}, {static_cast<int>(rng() % (mx - 2)) + 1, my}, {mx, static_cast<int>(rng() % (my - 2)) + 1}};
    s.edge.assign(6, 1);
    std::uniform_int_distribution<int> dx(1, mx - 1), dy(1, my - 1);
    while (static_cast<int>(s.pos.size()) < BATCH) { s.pos.push_back({dx(rng), dy(rng)}); s.edge.push_back(0); }
    for (int i = 0; i < BATCH; ++i) {
        s.crops[i] = s.src(cv::Rect(s.pos[i][0], s.pos[i][1], CROP_W, CROP_H));
    }
}

struct Stats {
    size_t n = 0, differing = 0, nanA = 0, nanB = 0;
    double maxAbs = 0;
};
Stats compare(const std::vector<float>& a, const std::vector<float>& b) {
    Stats s;
    for (size_t i = 0; i < a.size(); ++i) {
        ++s.n;
        s.nanA += std::isnan(a[i]);
        s.nanB += std::isnan(b[i]);
        if (std::memcmp(&a[i], &b[i], 4) != 0) {
            ++s.differing;
            const double d = std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
            if (std::isfinite(d)) s.maxAbs = std::max(s.maxAbs, d);
        }
    }
    return s;
}

int kernelNodes(cudaStream_t stream, const std::function<void()>& call) {
    cudaGraph_t graph{};
    FKLB_CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    call();
    FKLB_CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
    size_t n = 0;
    FKLB_CUDA_CHECK(cudaGraphGetNodes(graph, nullptr, &n));
    std::vector<cudaGraphNode_t> nodes(n);
    FKLB_CUDA_CHECK(cudaGraphGetNodes(graph, nodes.data(), &n));
    int kernels = 0;
    for (auto node : nodes) { cudaGraphNodeType t; FKLB_CUDA_CHECK(cudaGraphNodeGetType(node, &t)); kernels += (t == cudaGraphNodeTypeKernel); }
    cudaGraphDestroy(graph);
    return kernels;
}

struct Timing { double median = 0, min = 0, max = 0; };
Timing timeCall(cudaStream_t stream, const std::function<void()>& call, const int blocks, const double blockMs) {
    cudaEvent_t e0, e1;
    FKLB_CUDA_CHECK(cudaEventCreate(&e0)); FKLB_CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 10; ++i) call();
    FKLB_CUDA_CHECK(cudaStreamSynchronize(stream));
    // warm-up of at least 300 ms of GPU time, then a block length from one timed call
    FKLB_CUDA_CHECK(cudaEventRecord(e0, stream));
    for (int i = 0; i < 20; ++i) call();
    FKLB_CUDA_CHECK(cudaEventRecord(e1, stream)); FKLB_CUDA_CHECK(cudaEventSynchronize(e1));
    float ms = 0; FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const int perBlock = std::max(1, static_cast<int>(blockMs / std::max(1e-3, ms / 20.0)));
    for (int i = 0; i < static_cast<int>(300.0 / blockMs) + 1; ++i) for (int j = 0; j < perBlock; ++j) call();
    std::vector<double> t;
    for (int b = 0; b < blocks; ++b) {
        FKLB_CUDA_CHECK(cudaEventRecord(e0, stream));
        for (int j = 0; j < perBlock; ++j) call();
        FKLB_CUDA_CHECK(cudaEventRecord(e1, stream)); FKLB_CUDA_CHECK(cudaEventSynchronize(e1));
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
        t.push_back(ms / perBlock);
    }
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    std::sort(t.begin(), t.end());
    return {t[t.size() / 2], t.front(), t.back()};
}

std::string envString(const char* name, const char* fallback) { const char* v = std::getenv(name); return v ? v : fallback; }
}  // namespace

int launch() {
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    // one plain stream, wrapped by each library's own stream type: FKL's for the mixed kernels and the listings, OpenCV's for R1 (cvGPUSpeedup's
    // executor), and the NPP stream context for R2 (FastNPP's executor)
    fk::Stream stream(cs);
    const cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(cs);
    // The NPP stream context, filled field by field from the device properties and the stream's flags (CUDA 13 has no nppGetStreamContext)
    int device = 0;
    FKLB_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    FKLB_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
    NppStreamContext nppCtx{};
    nppCtx.hStream = cs;
    nppCtx.nCudaDeviceId = device;
    nppCtx.nMultiProcessorCount = props.multiProcessorCount;
    nppCtx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
    nppCtx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
    nppCtx.nSharedMemPerBlock = props.sharedMemPerBlock;
    nppCtx.nCudaDevAttrComputeCapabilityMajor = props.major;
    nppCtx.nCudaDevAttrComputeCapabilityMinor = props.minor;
    FKLB_CUDA_CHECK(cudaStreamGetFlags(cs, &nppCtx.nStreamFlags));
    const int seeds = std::atoi(envString("E9_SEEDS", "3").c_str()), run = std::atoi(envString("E9_RUN", "0").c_str());
    const int blocks = std::atoi(envString("E9_BLOCKS", "15").c_str());
    const double blockMs = std::atof(envString("E9_BLOCK_MS", "20").c_str());
    fklb::CsvWriter agree("e9_agreement.csv", "E9 cross-library kernels: agreement of the outputs", {"seed", "comparison", "n", "identical", "differing", "max_abs_difference", "nan_a", "nan_b"},
                          "seeded noise 3840 x 2160 uchar3, 50 crops 60 x 120 -> 64 x 128 float3, tensor (crop, channel, y, x); identical = memcmp of the bit patterns; M1 = cvGS resize + FastNPP + cvGS + user gamma, M2 = FastNPP resize + cvGS + FastNPP + user gamma, "
                          "R1 = cvGS only (cvGS's executor), R2 = FastNPP only (FastNPP's executor), each with the same user gamma; each library reads its own copy of the frame");
    fklb::CsvWriter timing("e9_timing.csv", "E9 cross-library kernels: time per call and launches", {"run", "variant", "position", "kernels_per_call", "launch_ms_median", "launch_ms_min", "launch_ms_max", "graph_ms_median", "graph_ms_min", "graph_ms_max", "calls_per_graph"},
                           "kernels_per_call = kernel nodes of one call captured into a CUDA graph; launch_ms = time per call of plain back-to-back calls, each building its batch and its operations (host and GPU), graph_ms = GPU time per call from a CUDA graph of calls_per_graph calls (no host work; the standing GPU-bound time); blocks between two CUDA events, warm-up of at least 300 ms; the order of the variants is rotated by E9_RUN");
    for (int seed = 0; seed < seeds; ++seed) {
        Setup s;
        makeSetup(s, seed);
        if (seed == 0) {
            std::cout << "frame steps (bytes): OpenCV " << s.src.step << ", FastNPP " << s.nppStep << std::endl;
        }
        fk::Tensor<float> t1(OUT_W, OUT_H, BATCH, 3), t2(OUT_W, OUT_H, BATCH, 3);   // the outputs of M1 and of THE LISTING (FKL's executor)
        const size_t plane = static_cast<size_t>(OUT_W) * OUT_H, total = static_cast<size_t>(BATCH) * 3 * plane;
        if (t1.ptr().dims.pitch != static_cast<size_t>(OUT_W) * sizeof(float)) { std::cerr << "padded tensor rows" << std::endl; return 1; }
        const int order[3] = {2, 1, 0};
        const float mean[3] = {MEAN[0], MEAN[1], MEAN[2]}, stdv[3] = {STD[0], STD[1], STD[2]}, scaleArray[3] = {SCALE, SCALE, SCALE};
        const cv::Scalar sc(SCALE, SCALE, SCALE), mn(MEAN[0], MEAN[1], MEAN[2]), sd(STD[0], STD[1], STD[2]);
        // R1's output, OpenCV's: one continuous GpuMat, one row per crop holding its three planes
        cv::cuda::GpuMat outR1;
        cv::cuda::createContinuous(BATCH, 3 * OUT_W * OUT_H, CV_32F, outR1);
        if (!outR1.isContinuous()) { std::cerr << "R1's output is not continuous" << std::endl; return 1; }
        // M2's and R2's outputs, NPP style: dense tensors (cudaMalloc) written through plane pointers, which are set up once because they depend
        // only on the layout of the tensor, not on the crops
        float *outM2 = nullptr, *outR2 = nullptr;
        FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outM2), total * sizeof(float)));
        FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&outR2), total * sizeof(float)));
        std::array<Npp32f*, BATCH> dstM2[3], dstR2[3];
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < BATCH; ++i) {
                dstM2[c][i] = outM2 + (static_cast<size_t>(i) * 3 + c) * plane;
                dstR2[c][i] = outR2 + (static_cast<size_t>(i) * 3 + c) * plane;
            }
        }
        struct Variant { std::string name; std::function<void()> call; float* out; };
        // Every call builds its batch and its operations: the crop locations may change on every image, so the ROIs (cvGPUSpeedup) and the
        // descriptor and ROI arrays (FastNPP) are made in the call, on that library's own copy of the frame.
        std::vector<Variant> variants = {
            {"M1", [&] {
                 std::array<cv::cuda::GpuMat, BATCH> crops;
                 for (int i = 0; i < BATCH; ++i) crops[i] = s.src(cv::Rect(s.pos[i][0], s.pos[i][1], CROP_W, CROP_H));
                 fk::executeOperations<fk::TransformDPP<>>(stream, cvGS::resize<CV_8UC3, cv::INTER_LINEAR>(crops, cv::Size(OUT_W, OUT_H), BATCH),
                                                           fastNPP::SwapChannels_32f_C3R_Ctx(order), cvGS::multiply<CV_32FC3>(sc), Gamma::build(GAMMA),
                                                           fastNPP::SubC_32f_C3R_Ctx(mean), cvGS::divide<CV_32FC3>(sd), cvGS::split<CV_32FC3>(t1.ptr()));
             }, t1.ptr().data},
            {"M2", [&] {
                 std::array<NppiImageDescriptor, BATCH> descriptors{};
                 std::array<NppiResizeBatchROI_Advanced, BATCH> rois{};
                 for (int i = 0; i < BATCH; ++i) {
                     descriptors[i].pData = s.nppSrc + static_cast<size_t>(s.pos[i][1]) * s.nppStep + static_cast<size_t>(s.pos[i][0]) * 3;
                     descriptors[i].nStep = static_cast<int>(s.nppStep);
                     descriptors[i].oSize = NppiSize{CROP_W, CROP_H};
                     rois[i].oSrcRectROI = NppiRect{0, 0, CROP_W, CROP_H};
                     rois[i].oDstRectROI = NppiRect{0, 0, OUT_W, OUT_H};
                 }
                 fk::executeOperations<fk::TransformDPP<>>(stream,
                                                           fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(OUT_W, OUT_H, descriptors.data(), rois.data()),
                                                           cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(), fastNPP::MulC_32f_C3R_Ctx(scaleArray), Gamma::build(GAMMA),
                                                           cvGS::subtract<CV_32FC3>(mn), fastNPP::DivC_32f_C3R_Ctx(stdv),
                                                           fastNPP::CopyBatch_32f_C3P3R_Ctx<BATCH>(dstM2, OUT_W * 4, NppiSize{OUT_W, OUT_H}));
             }, outM2},
            {"R1", [&] {
                 std::array<cv::cuda::GpuMat, BATCH> crops;
                 for (int i = 0; i < BATCH; ++i) crops[i] = s.src(cv::Rect(s.pos[i][0], s.pos[i][1], CROP_W, CROP_H));
                 cvGS::executeOperations(cvStream, cvGS::resize<CV_8UC3, cv::INTER_LINEAR>(crops, cv::Size(OUT_W, OUT_H), BATCH),
                                         cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(), cvGS::multiply<CV_32FC3>(sc), Gamma::build(GAMMA),
                                         cvGS::subtract<CV_32FC3>(mn), cvGS::divide<CV_32FC3>(sd), cvGS::split<CV_32FC3>(outR1, cv::Size(OUT_W, OUT_H)));
             }, reinterpret_cast<float*>(outR1.data)},
            {"R2", [&] {
                 std::array<NppiImageDescriptor, BATCH> descriptors{};
                 std::array<NppiResizeBatchROI_Advanced, BATCH> rois{};
                 for (int i = 0; i < BATCH; ++i) {
                     descriptors[i].pData = s.nppSrc + static_cast<size_t>(s.pos[i][1]) * s.nppStep + static_cast<size_t>(s.pos[i][0]) * 3;
                     descriptors[i].nStep = static_cast<int>(s.nppStep);
                     descriptors[i].oSize = NppiSize{CROP_W, CROP_H};
                     rois[i].oSrcRectROI = NppiRect{0, 0, CROP_W, CROP_H};
                     rois[i].oDstRectROI = NppiRect{0, 0, OUT_W, OUT_H};
                 }
                 fastNPP::executeOperations(nppCtx,
                                            fastNPP::ResizeBatch_8u32f_C3R_Advanced_Ctx<NPPI_INTER_LINEAR, BATCH>(OUT_W, OUT_H, descriptors.data(), rois.data()),
                                            fastNPP::SwapChannels_32f_C3R_Ctx(order), fastNPP::MulC_32f_C3R_Ctx(scaleArray), Gamma::build(GAMMA),
                                            fastNPP::SubC_32f_C3R_Ctx(mean), fastNPP::DivC_32f_C3R_Ctx(stdv),
                                            fastNPP::CopyBatch_32f_C3P3R_Ctx<BATCH>(dstR2, OUT_W * 4, NppiSize{OUT_W, OUT_H}));
             }, outR2},
        };
        // THE LISTING: the mixed kernel M1 written out in one call (operations built in the call), one launch; its output (t2) must equal that of M1
        const auto listing = [&] {
            fk::executeOperations<fk::TransformDPP<>>(stream,
                cvGS::resize<CV_8UC3, cv::INTER_LINEAR>(s.crops, cv::Size(OUT_W, OUT_H), BATCH),   // cvGPUSpeedup
                fastNPP::SwapChannels_32f_C3R_Ctx(order),                                           // FastNPP
                cvGS::multiply<CV_32FC3>(sc),                                                       // cvGPUSpeedup
                Gamma::build(GAMMA),                                                                // the user's operation
                fastNPP::SubC_32f_C3R_Ctx(mean),                                                    // FastNPP
                cvGS::divide<CV_32FC3>(sd),                                                         // cvGPUSpeedup
                cvGS::split<CV_32FC3>(t2.ptr()));                                                   // cvGPUSpeedup
        };
        // ---- SHORT LISTING (for the paper): the same call as THE LISTING with plain names; compared with M1 in the window ----
        fk::Tensor<float> t6(OUT_W, OUT_H, BATCH, 3);
        using short_listing::Gamma;
        const auto shortListing = [&, crops = s.crops, size = cv::Size(OUT_W, OUT_H), scale = sc, stddev = sd, gamma = GAMMA, output = t6.ptr()] {
            fk::executeOperations<fk::TransformDPP<>>(stream,
                cvGS::resize<CV_8UC3, cv::INTER_LINEAR>(crops, size, BATCH), // cvGS
                fastNPP::SwapChannels_32f_C3R_Ctx(order),                    // FastNPP
                cvGS::multiply<CV_32FC3>(scale),                             // cvGS
                Gamma::build(gamma),                                         // user
                fastNPP::SubC_32f_C3R_Ctx(mean),                             // FastNPP
                cvGS::divide<CV_32FC3>(stddev),                              // cvGS
                cvGS::split<CV_32FC3>(output));                              // cvGS
        };
        // ---- END SHORT LISTING (call) ----
        std::vector<std::vector<float>> out(4);
        for (size_t v = 0; v < variants.size(); ++v) {
            FKLB_CUDA_CHECK(cudaMemsetAsync(variants[v].out, 0xff, total * 4, cs));   // NaN pattern: an unwritten element shows up as NaN
            variants[v].call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            out[v].resize(total);
            FKLB_CUDA_CHECK(cudaMemcpy(out[v].data(), variants[v].out, total * 4, cudaMemcpyDeviceToHost));
        }
        auto row = [&](const char* name, const std::vector<float>& a, const std::vector<float>& b) {
            const Stats st = compare(a, b);
            agree.row(seed, name, st.n, int(st.differing == 0), st.differing, st.maxAbs, st.nanA, st.nanB);
            std::cout << "seed " << seed << " " << name << ": " << (st.differing == 0 ? "identical" : std::to_string(st.differing) + " of " + std::to_string(st.n) + " differ, max abs " + std::to_string(st.maxAbs)) << " (NaN " << st.nanA << ", " << st.nanB << ")" << std::endl;
        };
        {
            FKLB_CUDA_CHECK(cudaMemsetAsync(t2.ptr().data, 0xff, total * 4, cs));
            listing();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            std::vector<float> l(total);
            FKLB_CUDA_CHECK(cudaMemcpy(l.data(), t2.ptr().data, total * 4, cudaMemcpyDeviceToHost));
            row("M1listing_vs_M1", l, out[0]);
            FKLB_CUDA_CHECK(cudaMemsetAsync(t6.ptr().data, 0xff, total * 4, cs));
            shortListing();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            FKLB_CUDA_CHECK(cudaMemcpy(l.data(), t6.ptr().data, total * 4, cudaMemcpyDeviceToHost));
            row("M1shortlisting_vs_M1", l, out[0]);
        }
        row("M1_vs_R1", out[0], out[2]); row("M2_vs_R2", out[1], out[3]); row("R1_vs_R2", out[2], out[3]); row("M1_vs_M2", out[0], out[1]);
        agree.flush();
        if (seed == 0 && std::getenv("E9_NSYS") != nullptr) {   // nsys cross-check of the launch count: ONE call of each variant inside the profiler range (nsys --capture-range=cudaProfilerApi); the kernels of the range are counted by nsys
            FKLB_CUDA_CHECK(cudaProfilerStart());
            for (size_t v = 0; v < variants.size(); ++v) { variants[v].call(); FKLB_CUDA_CHECK(cudaStreamSynchronize(cs)); }
            listing(); FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            shortListing(); FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            FKLB_CUDA_CHECK(cudaProfilerStop());
            return 0;
        }
        if (seed == 0) {   // launches and times once (seed 0)
            int kernels[4];
            for (size_t v = 0; v < variants.size(); ++v) kernels[v] = kernelNodes(cs, variants[v].call);
            std::vector<size_t> rot = {0, 1, 2, 3};
            std::rotate(rot.begin(), rot.begin() + (run % 4), rot.end());
            int position = 0;
            const int callsPerGraph = 100;
            // GPU time per call: callsPerGraph calls captured into one graph and replayed, so that the launch overhead of the host does not enter (a call is about 6 us of GPU time)
            auto timeGraph = [&](const std::function<void()>& call) -> Timing {
                cudaGraph_t graph{};
                cudaGraphExec_t exec{};
                FKLB_CUDA_CHECK(cudaStreamBeginCapture(cs, cudaStreamCaptureModeThreadLocal));
                for (int i = 0; i < callsPerGraph; ++i) call();
                FKLB_CUDA_CHECK(cudaStreamEndCapture(cs, &graph));
                FKLB_CUDA_CHECK(cudaGraphInstantiate(&exec, graph, 0));
                const std::function<void()> replay = [&] { FKLB_CUDA_CHECK(cudaGraphLaunch(exec, cs)); };
                Timing t = timeCall(cs, replay, blocks, blockMs);
                t.median /= callsPerGraph; t.min /= callsPerGraph; t.max /= callsPerGraph;
                FKLB_CUDA_CHECK(cudaGraphExecDestroy(exec)); FKLB_CUDA_CHECK(cudaGraphDestroy(graph));
                return t;
            };
            for (const size_t v : rot) {
                const Timing t = timeCall(cs, variants[v].call, blocks, blockMs);
                const Timing g = timeGraph(variants[v].call);
                timing.row(run, variants[v].name, ++position, kernels[v], t.median, t.min, t.max, g.median, g.min, g.max, callsPerGraph);
                timing.flush();
                std::cout << variants[v].name << ": " << kernels[v] << " kernel(s) per call, launch loop " << t.median << " ms (min " << t.min << ", max " << t.max << "), graph " << g.median << " ms (min " << g.min << ", max " << g.max << ")" << std::endl;
            }
        }
        FKLB_CUDA_CHECK(cudaFree(outM2));
        FKLB_CUDA_CHECK(cudaFree(outR2));
    }
    return 0;
}
