// Copyright 2026 Oscar Amoros Huguet
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// E4a and E4c (NVIDIA): the pre-processing chain of a DNN, on batches of crops (E4a) and on whole frames (E4c), FKL against OpenCV-CUDA and NPP.
//
// The chain (the same as pipeline (a) of the CPU rows, bench_preproc_a.cpp): 8UC3 crop -> float, bilinear resize, multiply by 0.3, RGB to BGR,
// subtract {1, 4, 3.2}, divide by {3.2, 0.6, 11.8}, split into one NCHW float tensor (BATCH x 3 x outH x outW, planes contiguous).
//   E4a  a seeded 3840 x 2160 8UC3 image, BATCH crops of 60 x 120 at (i, i), output 64 x 128; BATCH = E4A_BATCHES (default 10, 20, ... 300, as the CPU rows)
//   E4c  the same chain on one whole frame (batch 1): 720x480, 1920x1080 and 3840x2160 (the "480p", "1080p" and "4K" of the CPU rows, in that order),
//        output 64/60 x 128/120 of the frame (768x512, 2048x1152, 4096x2304; the factor 15/16 is exact in float): the regime where the working set
//        is far above the caches and the launch overhead is irrelevant
// Every implementation starts from the same bytes at the same pitch (the source is uploaded once to a device image that FKL, OpenCV-CUDA and NPP all read,
// the crops are ROIs of it: one RawPtr2D / GpuMat ROI / pointer per crop, never a Crop operation followed by a Resize):
//   opencv_percrop      OpenCV-CUDA, every operation per crop: convertTo, resize, multiply, cvtColor, subtract, divide, split (7 calls per crop)
//   opencv_batch        OpenCV-CUDA, the strongest baseline an expert writes (the "OCVBatch" of cvGPUSpeedup's test): convertTo and resize per crop into a stacked
//                       buffer, then ONE multiply, cvtColor, subtract and divide for the whole stack, then split per crop (not run for batch 1: same as opencv_percrop)
//   npp                 NPP: nppiConvert_8u32f per crop, ONE nppiResizeBatch_32f_C3R_Advanced for the batch (nppiResize_32f_C3R for a single frame: the batched call needs more than one image), then per crop SwapChannels, MulC, SubC, DivC and
//                       Copy_32f_C3P3R into the tensor; the intermediates ping-pong between two stacks (fewer than a buffer per step)
//   npp_ctx_as_tests    (only when named in E4_ONLY) the same NPP calls with the stream context built as FastNPP's tests do (major and minor swapped: see makeNppContext)
//   fkl_fused           FKL, one kernel: BatchRead of the crop ROIs -> Resize -> VectorReorder -> Mul -> Sub -> Div -> TensorSplit (thread fusion off)
//   fkl_fused_half_pixel  the same with fk::ResizeSampling::HALF_PIXEL, the convention of NPP and cv::resize
// Sampling convention of every implementation (the names are those of fk::ResizeSampling and of common/resize_reference.h):
//   ASYMMETRIC (source position x * scale): FKL by default (fkl_fused), cv::cuda::resize (opencv_percrop, opencv_batch)
//   HALF_PIXEL ((x + 0.5) * scale - 0.5):   NPP (npp), FKL with ResizeSampling::HALF_PIXEL (fkl_fused_half_pixel); cv::resize on the CPU
// The outputs of implementations with different conventions differ (the agreement between them is informational, written to the agreement CSV); each output is
// checked against the scalar double-precision reference of its OWN convention (fklb::resizeBilinear + the element-wise chain in double): that check is strict,
// max |output - reference| <= TOLERANCE (1e-4, as the CPU rows), and a version that exceeds it is reported (ok_own = 0), never hidden.
// Timing: CUDA events around one launch of the whole implementation (including the host calls that enqueue it), 100 launches after one warm-up, mean and min;
// intermediate bytes: what the implementation allocates besides the source and the output tensor (tallied) and the change of free device memory (cudaMemGetInfo).
// Host time of one call: host_enqueue_us (the launch function returns, stream idle before each of 100 calls) and host_launch_us (100 calls back to back without synchronisation, wall
// time per call: when the GPU is slower than the host this is the GPU's pace, because the launch queue fills up).
//
//   E4_BATCHES_ONLY=100,300 / E4_FRAMES_ONLY=3840 (frame widths): only these points of the sweeps (for the profiler)
//   E4_NPP_TESTS_FIRST=1 measures npp_ctx_as_tests before npp (order check of the NPP stream-context diagnostic; both run only when named in E4_ONLY)
//   E4_ONLY=a,b   run only these implementations (default all); E4_PROFILE=1: every implementation launches once and nothing is timed (for ncu, one
//   implementation per run with E4_ONLY, to count its kernels and its DRAM bytes). E4_EXPERIMENTS=E4a,E4c (default both).
//   Writes e4_preprocessing.csv and e4_preprocessing_agreement.csv.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/resize_reference.h>

#include "e1_common.h"

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <npp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "e4_batches.h"   // E4A_BATCHES, generated by CMake from the cache variable of the same name

namespace {
using e1::Timing;

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, UP_W = 64, UP_H = 128;
constexpr int BATCHES[] = {E4A_BATCHES};
constexpr float ALPHA = 0.3f;
constexpr float SUBV[3] = {1.f, 4.f, 3.2f};
constexpr float DIVV[3] = {3.2f, 0.6f, 11.8f};
constexpr double TOLERANCE = 1e-4;

#define E4_CHECK(call)                                                                                           \
    {                                                                                                            \
        const cudaError_t err_ = (call);                                                                         \
        if (err_ != cudaSuccess) {                                                                               \
            std::cerr << #call << " failed: " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ \
                      << ")" << std::endl;                                                                       \
            return false;                                                                                        \
        }                                                                                                        \
    }
#define E4_NPP(call)                                                                                             \
    {                                                                                                            \
        const NppStatus st_ = (call);                                                                            \
        if (st_ != NPP_SUCCESS) {                                                                                \
            std::cerr << #call << " failed: NppStatus " << static_cast<int>(st_) << " (" << __FILE__ << ":" << __LINE__ << ")" << std::endl; \
            return false;                                                                                        \
        }                                                                                                        \
    }

bool profileMode() { return std::getenv("E4_PROFILE") != nullptr && std::getenv("E4_PROFILE")[0] == '1'; }

bool selected(const std::string& name) {
    const char* only = std::getenv("E4_ONLY");
    if (only == nullptr) return name != "npp_ctx_as_tests";   // the diagnostic variant runs only when it is named
    const std::string list = std::string(",") + only + ",";
    return list.find("," + name + ",") != std::string::npos;
}

// E4_BATCHES_ONLY=100,300 and E4_FRAMES_ONLY=1920,3840 (widths) restrict a run to some points of the sweep, for the profiler (one launch of every kernel of a batch of 300 crops is thousands of kernels)
bool listed(const char* env, const int value) {
    const char* e = std::getenv(env);
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + std::to_string(value) + ",") != std::string::npos;
}

bool experimentSelected(const std::string& name) {
    const char* e = std::getenv("E4_EXPERIMENTS");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// The source, as in the CPU rows (bench_preproc_a.cpp): an LCG with seed 12345, bytes 31..24, 23..16, 15..8 of every state, row by row
std::vector<unsigned char> seededNoise(const int width, const int height) {
    std::vector<unsigned char> noise(static_cast<size_t>(width) * height * 3);
    uint32_t state = 12345u;
    for (size_t i = 0; i < noise.size(); i += 3) {
        state = state * 1664525u + 1013904223u;
        noise[i] = static_cast<unsigned char>(state >> 24);
        noise[i + 1] = static_cast<unsigned char>(state >> 16);
        noise[i + 2] = static_cast<unsigned char>(state >> 8);
    }
    return noise;
}

// Device memory the implementation allocates for itself, with the tally of its size
struct Pool {
    std::vector<void*> ptrs;
    size_t bytes = 0;
    template <typename T>
    T* alloc(const size_t count) {
        void* p = nullptr;
        if (cudaMalloc(&p, count * sizeof(T)) != cudaSuccess) return nullptr;
        ptrs.push_back(p);
        bytes += count * sizeof(T);
        return static_cast<T*>(p);
    }
    ~Pool() { for (void* p : ptrs) cudaFree(p); }
};

size_t freeBytes() {
    size_t f = 0, t = 0;
    cudaMemGetInfo(&f, &t);
    return f;
}

struct Geometry {
    const char* experiment;
    int batch, srcW, srcH, cropW, cropH, outW, outH;
    bool diagonal;   // crop i at (i, i) (E4a); otherwise one crop at (0, 0) (E4c: the whole frame)
    int x0(const int i) const { return diagonal ? i : 0; }
    int y0(const int i) const { return diagonal ? i : 0; }
    size_t planeElems() const { return static_cast<size_t>(outW) * outH; }
    size_t tensorElems() const { return static_cast<size_t>(batch) * 3 * planeElems(); }
};

// The scalar reference of the whole chain in double, for one sampling convention, in the layout of the tensor
std::vector<double> reference(const Geometry& g, const std::vector<unsigned char>& noise, const fklb::ResizeSampling sampling) {
    std::vector<double> ref(g.tensorElems());
    std::vector<double> resized(g.planeElems() * 3);
    for (int i = 0; i < g.batch; ++i) {
        const unsigned char* base = noise.data() + (static_cast<size_t>(g.y0(i)) * g.srcW + g.x0(i)) * 3;
        fklb::resizeBilinear<unsigned char>(base, static_cast<size_t>(g.srcW) * 3, g.cropW, g.cropH, 3, g.outW, g.outH, sampling, resized.data());
        for (int y = 0; y < g.outH; ++y) {
            for (int x = 0; x < g.outW; ++x) {
                for (int k = 0; k < 3; ++k) {   // output channel k is the source channel 2 - k (RGB to BGR); the constants are the float ones
                    const double v = (resized[(static_cast<size_t>(y) * g.outW + x) * 3 + (2 - k)] * static_cast<double>(ALPHA) - static_cast<double>(SUBV[k])) / static_cast<double>(DIVV[k]);
                    ref[((static_cast<size_t>(i) * 3 + k) * g.outH + y) * g.outW + x] = v;
                }
            }
        }
    }
    return ref;
}

double maxAbsDiff(const std::vector<float>& out, const std::vector<double>& ref) {
    double m = 0;
    for (size_t i = 0; i < out.size(); ++i) m = std::max(m, std::abs(static_cast<double>(out[i]) - ref[i]));
    return m;
}

struct Writers {
    fklb::CsvWriter& timing;
    fklb::CsvWriter& agreement;
};

// Runs one implementation: clears the output, launches it once and checks it, then times it. Returns false on a CUDA error.
template <typename Launch>
bool measure(const Geometry& g, const char* version, const char* convention, cudaStream_t cs, float* dOut, const Launch& launch, const size_t intermediateBytes,
             const size_t meminfoDelta, const std::vector<unsigned char>& noise, const std::vector<double>& refAsym, const std::vector<double>& refHalf, Writers& w, bool& allOk) {
    if (!selected(version)) return true;
    const size_t outBytes = g.tensorElems() * sizeof(float);
    E4_CHECK(cudaMemsetAsync(dOut, 0, outBytes, cs));
    launch();
    E4_CHECK(cudaStreamSynchronize(cs));
    std::vector<float> host(g.tensorElems());
    E4_CHECK(cudaMemcpy(host.data(), dOut, outBytes, cudaMemcpyDeviceToHost));
    const double eAsym = maxAbsDiff(host, refAsym), eHalf = maxAbsDiff(host, refHalf);
    const bool own = std::string(convention) == "ASYMMETRIC";
    const double eOwn = own ? eAsym : eHalf;
    const bool ok = eOwn <= TOLERANCE;
    allOk = allOk && ok;
    Timing t;
    if (!profileMode() && !e1::timeIt(cs, launch, t)) return false;
    // Host time of one call, in two ways (microseconds):
    //   host_enqueue_us  the time the launch function takes to return, with the stream idle before each call (mean of 100 calls, synchronised between them): what it costs the host to enqueue the call
    //   host_launch_us   100 calls back to back with no synchronisation between them, the wall time per call (as the host_launch_us of the E2 script): when the GPU is slower than the host, the
    //                    launch queue fills up and this is the time the GPU needs, not what the host spends
    double hostEnqueueUs = 0, hostLaunchUs = 0;
    if (!profileMode()) {
        using Clock = std::chrono::steady_clock;
        const auto us = [](const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        for (int i = 0; i < e1::ITERATIONS; ++i) {
            E4_CHECK(cudaStreamSynchronize(cs));
            const auto t0 = Clock::now();
            launch();
            hostEnqueueUs += us(t0, Clock::now()) / e1::ITERATIONS;
        }
        E4_CHECK(cudaStreamSynchronize(cs));
        const auto t0 = Clock::now();
        for (int i = 0; i < e1::ITERATIONS; ++i) launch();
        hostLaunchUs = us(t0, Clock::now()) / e1::ITERATIONS;
        E4_CHECK(cudaStreamSynchronize(cs));
    }
    w.timing.row(g.experiment, g.batch, g.srcW, g.srcH, g.cropW, g.cropH, g.outW, g.outH, version, convention, t.mean, t.min, intermediateBytes, meminfoDelta, outBytes, hostEnqueueUs, hostLaunchUs);
    w.timing.flush();
    w.agreement.row(g.experiment, g.batch, version, convention, eOwn, int(ok), eAsym, eHalf);
    w.agreement.flush();
    std::cout << g.experiment << " batch " << g.batch << " " << g.srcW << "x" << g.srcH << " -> " << g.outW << "x" << g.outH << "  " << version << " (" << convention << "): "
              << t.mean << " ms (min " << t.min << "), host enqueue " << hostEnqueueUs << " us, back to back " << hostLaunchUs << " us, intermediates " << intermediateBytes << " B, max |err| vs own reference " << eOwn << (ok ? "" : "  EXCEEDS " + std::to_string(TOLERANCE))
              << ", vs ASYMMETRIC " << eAsym << ", vs HALF_PIXEL " << eHalf << std::endl;
    return true;
}

// The NPP stream context. CUDA 13 has no nppGetStreamContext: it is filled field by field from the device properties. asInTests = true builds it the way
// initNppStreamContext of FastNPP's tests/testsNppCommon.h does, which stores the MINOR compute capability in the "major" field and the major in the "minor"
// one (8.9 is passed to NPP as 9.8): the diagnostic variant npp_ctx_as_tests measures whether NPP behaves differently with it.
NppStreamContext makeNppContext(cudaStream_t stream, const bool asInTests) {
    int device = 0;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    unsigned int flags = 0;
    cudaStreamGetFlags(stream, &flags);
    NppStreamContext ctx{};
    ctx.hStream = stream;
    ctx.nCudaDeviceId = device;
    ctx.nMultiProcessorCount = prop.multiProcessorCount;
    ctx.nMaxThreadsPerMultiProcessor = prop.maxThreadsPerMultiProcessor;
    ctx.nMaxThreadsPerBlock = prop.maxThreadsPerBlock;
    ctx.nSharedMemPerBlock = prop.sharedMemPerBlock;
    ctx.nCudaDevAttrComputeCapabilityMajor = asInTests ? prop.minor : prop.major;
    ctx.nCudaDevAttrComputeCapabilityMinor = asInTests ? prop.major : prop.minor;
    ctx.nStreamFlags = flags;
    return ctx;
}

template <int BATCH>
bool runCase(const Geometry& g, const std::vector<unsigned char>& noise, cudaStream_t cs, fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream, Writers& w, bool& allOk) {
    // the device source, shared by every implementation: the same bytes at the same pitch
    fk::Ptr2D<uchar3> dInput(g.srcW, g.srcH);
    const size_t inPitch = dInput.ptr().dims.pitch;
    E4_CHECK(cudaMemcpy2D(dInput.ptr().data, inPitch, noise.data(), static_cast<size_t>(g.srcW) * 3, static_cast<size_t>(g.srcW) * 3, g.srcH, cudaMemcpyHostToDevice));
    const std::vector<double> refAsym = reference(g, noise, fklb::ResizeSampling::ASYMMETRIC), refHalf = reference(g, noise, fklb::ResizeSampling::HALF_PIXEL);

    // the output tensor of FKL (fk::Tensor) and the ones of the other implementations (the same layout: BATCH x 3 x outH x outW, planes contiguous)
    fk::Tensor<float> dTensor(g.outW, g.outH, BATCH, 3);
    if (dTensor.ptr().dims.pitch != static_cast<size_t>(g.outW) * sizeof(float)) {
        std::cerr << "the tensor rows are padded (pitch " << dTensor.ptr().dims.pitch << "), fk::TensorSplit assumes they are not" << std::endl;
        return false;
    }
    Pool outputs;
    float* dOutCv = outputs.alloc<float>(g.tensorElems());
    float* dOutNpp = outputs.alloc<float>(g.tensorElems());
    if (dOutCv == nullptr || dOutNpp == nullptr) return false;

    // ---- FKL: one kernel; the crops are ROIs (Ptr2D crops of the device image), the read is a BatchRead of them followed by the Resize
    std::array<fk::Ptr2D<uchar3>, BATCH> crops;
    for (int i = 0; i < BATCH; ++i) {
        crops[i] = dInput.crop(fk::Point{g.x0(i), g.y0(i), 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(g.cropW), static_cast<uint>(g.cropH), static_cast<uint>(inPitch)});
    }
    const auto readAsym = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(g.outW, g.outH)));
    const auto readHalf = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(
        fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(g.outW, g.outH)));
    const auto reorder = fk::VectorReorder<float3, 2, 1, 0>::build();
    const auto mul = fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA));
    const auto sub = fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2]));
    const auto div = fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2]));
    const fk::Write<fk::TensorSplit<float3>> write{dTensor.ptr()};
    auto launchFklAsym = [&] { fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream, readAsym, reorder, mul, sub, div, write); };
    auto launchFklHalf = [&] { fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream, readHalf, reorder, mul, sub, div, write); };
    if (!measure(g, "fkl_fused", "ASYMMETRIC", cs, dTensor.ptr().data, launchFklAsym, 0, 0, noise, refAsym, refHalf, w, allOk)) return false;
    if (!measure(g, "fkl_fused_half_pixel", "HALF_PIXEL", cs, dTensor.ptr().data, launchFklHalf, 0, 0, noise, refAsym, refHalf, w, allOk)) return false;

    // ---- OpenCV-CUDA
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    const cv::cuda::GpuMat src(g.srcH, g.srcW, CV_8UC3, dInput.ptr().data, inPitch);
    const cv::Scalar alphaS(ALPHA, ALPHA, ALPHA), subS(SUBV[0], SUBV[1], SUBV[2]), divS(DIVV[0], DIVV[1], DIVV[2]);
    const cv::Size outSize(g.outW, g.outH);
    std::vector<std::array<cv::cuda::GpuMat, 3>> planes(BATCH);
    for (int i = 0; i < BATCH; ++i) {
        for (int c = 0; c < 3; ++c) planes[i][c] = cv::cuda::GpuMat(g.outH, g.outW, CV_32F, dOutCv + (static_cast<size_t>(i) * 3 + c) * g.planeElems(), static_cast<size_t>(g.outW) * sizeof(float));
    }
    {   // per crop, every operation
        const size_t before = freeBytes();
        cv::cuda::GpuMat f32(g.cropH, g.cropW, CV_32FC3), rs(outSize, CV_32FC3), a(outSize, CV_32FC3), b(outSize, CV_32FC3);
        E4_CHECK(cudaStreamSynchronize(cs));
        const size_t delta = before - std::min(before, freeBytes());
        const size_t tally = f32.step * f32.rows + (rs.step * rs.rows) * 3;
        auto launch = [&] {
            for (int i = 0; i < BATCH; ++i) {
                const cv::cuda::GpuMat roi(src, cv::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
                roi.convertTo(f32, CV_32FC3, 1.0, cvs);
                cv::cuda::resize(f32, rs, outSize, 0., 0., cv::INTER_LINEAR, cvs);
                cv::cuda::multiply(rs, alphaS, a, 1.0, -1, cvs);
                cv::cuda::cvtColor(a, b, cv::COLOR_RGB2BGR, 0, cvs);
                cv::cuda::subtract(b, subS, a, cv::noArray(), -1, cvs);
                cv::cuda::divide(a, divS, b, 1.0, -1, cvs);
                cv::cuda::split(b, planes[i].data(), cvs);
            }
        };
        if (!measure(g, "opencv_percrop", "ASYMMETRIC", cs, dOutCv, launch, tally, delta, noise, refAsym, refHalf, w, allOk)) return false;
    }
    if constexpr (BATCH > 1) {   // the batched OpenCV baseline: per-crop convertTo + resize into a stack, then one call per operation for the whole stack
        const size_t before = freeBytes();
        cv::cuda::GpuMat f32(g.cropH, g.cropW, CV_32FC3), stackRs(BATCH * g.outH, g.outW, CV_32FC3), stackA(BATCH * g.outH, g.outW, CV_32FC3), stackB(BATCH * g.outH, g.outW, CV_32FC3);
        E4_CHECK(cudaStreamSynchronize(cs));
        const size_t delta = before - std::min(before, freeBytes());
        const size_t tally = f32.step * f32.rows + (stackRs.step * stackRs.rows) * 3;
        auto launch = [&] {
            for (int i = 0; i < BATCH; ++i) {
                const cv::cuda::GpuMat roi(src, cv::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
                roi.convertTo(f32, CV_32FC3, 1.0, cvs);
                cv::cuda::GpuMat dst = stackRs.rowRange(i * g.outH, (i + 1) * g.outH);
                cv::cuda::resize(f32, dst, outSize, 0., 0., cv::INTER_LINEAR, cvs);
            }
            cv::cuda::multiply(stackRs, alphaS, stackA, 1.0, -1, cvs);
            cv::cuda::cvtColor(stackA, stackB, cv::COLOR_RGB2BGR, 0, cvs);
            cv::cuda::subtract(stackB, subS, stackA, cv::noArray(), -1, cvs);
            cv::cuda::divide(stackA, divS, stackB, 1.0, -1, cvs);
            for (int i = 0; i < BATCH; ++i) {
                const cv::cuda::GpuMat rows = stackB.rowRange(i * g.outH, (i + 1) * g.outH);
                cv::cuda::split(rows, planes[i].data(), cvs);
            }
        };
        if (!measure(g, "opencv_batch", "ASYMMETRIC", cs, dOutCv, launch, tally, delta, noise, refAsym, refHalf, w, allOk)) return false;
    }

    // ---- NPP: per-crop conversion, one batched resize, per-crop arithmetic and split (two ping-pong stacks)
    {
        const NppStreamContext ctxCorrect = makeNppContext(cs, false), ctxAsInTests = makeNppContext(cs, true);
        const size_t before = freeBytes();
        Pool pool;
        const size_t cropFloats = static_cast<size_t>(g.cropW) * g.cropH * 3, outFloats = g.planeElems() * 3;
        float* cropStack = pool.alloc<float>(BATCH * cropFloats);
        float* stackA = pool.alloc<float>(BATCH * outFloats);
        float* stackB = pool.alloc<float>(BATCH * outFloats);
        NppiImageDescriptor* dSrc = pool.alloc<NppiImageDescriptor>(BATCH);
        NppiImageDescriptor* dDst = pool.alloc<NppiImageDescriptor>(BATCH);
        NppiResizeBatchROI_Advanced* dRoi = pool.alloc<NppiResizeBatchROI_Advanced>(BATCH);
        if (cropStack == nullptr || stackA == nullptr || stackB == nullptr || dSrc == nullptr || dDst == nullptr || dRoi == nullptr) return false;
        std::vector<NppiImageDescriptor> hSrc(BATCH), hDst(BATCH);
        std::vector<NppiResizeBatchROI_Advanced> hRoi(BATCH);
        const NppiSize cropSize{g.cropW, g.cropH}, upSize{g.outW, g.outH};
        for (int i = 0; i < BATCH; ++i) {
            hSrc[i].pData = cropStack + static_cast<size_t>(i) * cropFloats;
            hSrc[i].nStep = g.cropW * 3 * static_cast<int>(sizeof(float));
            hSrc[i].oSize = cropSize;
            hDst[i].pData = stackA + static_cast<size_t>(i) * outFloats;
            hDst[i].nStep = g.outW * 3 * static_cast<int>(sizeof(float));
            hDst[i].oSize = upSize;
            hRoi[i].oSrcRectROI = NppiRect{0, 0, g.cropW, g.cropH};
            hRoi[i].oDstRectROI = NppiRect{0, 0, g.outW, g.outH};
        }
        E4_CHECK(cudaMemcpyAsync(dSrc, hSrc.data(), sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, cs));
        E4_CHECK(cudaMemcpyAsync(dDst, hDst.data(), sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, cs));
        E4_CHECK(cudaMemcpyAsync(dRoi, hRoi.data(), sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, cs));
        E4_CHECK(cudaStreamSynchronize(cs));
        const size_t delta = before - std::min(before, freeBytes());
        const int order[3] = {2, 1, 0};
        const Npp32f mulV[3] = {ALPHA, ALPHA, ALPHA}, subV[3] = {SUBV[0], SUBV[1], SUBV[2]}, divV[3] = {DIVV[0], DIVV[1], DIVV[2]};
        const int stepOut = g.outW * 3 * static_cast<int>(sizeof(float));
        bool launchOk = true;
        auto nppLaunch = [&](const NppStreamContext& ctx) {
            for (int i = 0; i < BATCH; ++i) {
                const Npp8u* srcCrop = reinterpret_cast<const Npp8u*>(dInput.ptr().data) + static_cast<size_t>(g.y0(i)) * inPitch + static_cast<size_t>(g.x0(i)) * 3;
                launchOk &= nppiConvert_8u32f_C3R_Ctx(srcCrop, static_cast<int>(inPitch), reinterpret_cast<Npp32f*>(hSrc[i].pData), hSrc[i].nStep, cropSize, ctx) == NPP_SUCCESS;
            }
            if constexpr (BATCH == 1) {   // NPP's batched resize needs more than one image (nBatchSize > 1): a single frame uses the single-image resize
                launchOk &= nppiResize_32f_C3R_Ctx(reinterpret_cast<const Npp32f*>(hSrc[0].pData), hSrc[0].nStep, cropSize, NppiRect{0, 0, g.cropW, g.cropH},
                                                   reinterpret_cast<Npp32f*>(hDst[0].pData), hDst[0].nStep, upSize, NppiRect{0, 0, g.outW, g.outH}, NPPI_INTER_LINEAR, ctx) == NPP_SUCCESS;
            } else {
                launchOk &= nppiResizeBatch_32f_C3R_Advanced_Ctx(g.outW, g.outH, dSrc, dDst, dRoi, BATCH, NPPI_INTER_LINEAR, ctx) == NPP_SUCCESS;
            }
            for (int i = 0; i < BATCH; ++i) {
                Npp32f* a = stackA + static_cast<size_t>(i) * outFloats;
                Npp32f* b = stackB + static_cast<size_t>(i) * outFloats;
                Npp32f* const planesOut[3] = {dOutNpp + (static_cast<size_t>(i) * 3 + 0) * g.planeElems(), dOutNpp + (static_cast<size_t>(i) * 3 + 1) * g.planeElems(),
                                              dOutNpp + (static_cast<size_t>(i) * 3 + 2) * g.planeElems()};
                launchOk &= nppiSwapChannels_32f_C3R_Ctx(a, stepOut, b, stepOut, upSize, order, ctx) == NPP_SUCCESS;
                launchOk &= nppiMulC_32f_C3R_Ctx(b, stepOut, mulV, a, stepOut, upSize, ctx) == NPP_SUCCESS;
                launchOk &= nppiSubC_32f_C3R_Ctx(a, stepOut, subV, b, stepOut, upSize, ctx) == NPP_SUCCESS;
                launchOk &= nppiDivC_32f_C3R_Ctx(b, stepOut, divV, a, stepOut, upSize, ctx) == NPP_SUCCESS;
                launchOk &= nppiCopy_32f_C3P3R_Ctx(a, stepOut, planesOut, g.outW * static_cast<int>(sizeof(float)), upSize, ctx) == NPP_SUCCESS;
            }
        };
        auto launch = [&] { nppLaunch(ctxCorrect); };
        auto launchAsInTests = [&] { nppLaunch(ctxAsInTests); };
        // E4_NPP_TESTS_FIRST=1 measures the diagnostic variant first (so that an order effect can be told from a context effect)
        const bool testsFirst = std::getenv("E4_NPP_TESTS_FIRST") != nullptr && std::getenv("E4_NPP_TESTS_FIRST")[0] == '1';
        for (int pass = 0; pass < 2; ++pass) {
            if ((pass == 0) != testsFirst) {
                if (!measure(g, "npp", "HALF_PIXEL", cs, dOutNpp, launch, pool.bytes, delta, noise, refAsym, refHalf, w, allOk)) return false;
            } else {
                if (!measure(g, "npp_ctx_as_tests", "HALF_PIXEL", cs, dOutNpp, launchAsInTests, pool.bytes, delta, noise, refAsym, refHalf, w, allOk)) return false;
            }
        }
        if (selected("npp") && selected("npp_ctx_as_tests")) {   // does the context change the NPP output at all? (bitwise, one launch of each)
            std::vector<float> outCorrect(g.tensorElems()), outTests(g.tensorElems());
            const size_t outBytes = g.tensorElems() * sizeof(float);
            E4_CHECK(cudaMemsetAsync(dOutNpp, 0, outBytes, cs));
            launch();
            E4_CHECK(cudaStreamSynchronize(cs));
            E4_CHECK(cudaMemcpy(outCorrect.data(), dOutNpp, outBytes, cudaMemcpyDeviceToHost));
            E4_CHECK(cudaMemsetAsync(dOutNpp, 0, outBytes, cs));
            launchAsInTests();
            E4_CHECK(cudaStreamSynchronize(cs));
            E4_CHECK(cudaMemcpy(outTests.data(), dOutNpp, outBytes, cudaMemcpyDeviceToHost));
            double maxDiff = 0;
            for (size_t i = 0; i < outCorrect.size(); ++i) maxDiff = std::max(maxDiff, std::abs(static_cast<double>(outCorrect[i]) - outTests[i]));
            const bool identical = e1::sameBits(outCorrect, outTests);
            w.agreement.row(g.experiment, g.batch, "npp_ctx_as_tests_vs_npp", "HALF_PIXEL", maxDiff, int(identical), "", "");
            w.agreement.flush();
            std::cout << g.experiment << " batch " << g.batch << ": NPP output with the context of the tests against the correct context: " << (identical ? "bit-identical" : "DIFFERENT, max |diff| " + std::to_string(maxDiff)) << std::endl;
        }
        if (!launchOk) {
            std::cerr << "an NPP call failed" << std::endl;
            return false;
        }
    }
    return true;
}

template <size_t... Is>
bool runBatches(std::index_sequence<Is...>, const std::vector<unsigned char>& noise, cudaStream_t cs, fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream, Writers& w, bool& allOk) {
    bool ok = true;
    ((ok = ok && (!listed("E4_BATCHES_ONLY", BATCHES[Is]) || runCase<BATCHES[Is]>(Geometry{"E4a", BATCHES[Is], SRC_W, SRC_H, CROP_W, CROP_H, UP_W, UP_H, true}, noise, cs, fkStream, w, allOk))), ...);
    return ok;
}
}  // namespace

int launch() {
    cudaStream_t cs;
    if (cudaStreamCreate(&cs) != cudaSuccess) return -1;
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    const std::string notes = std::string("thread fusion: off (fk::TF::DISABLED); source: seeded 8UC3 noise (LCG seed 12345, as the CPU rows); timing: CUDA events around one launch ") +
                              "(including the host calls that enqueue it), 100 launches after one warm-up; sampling: ASYMMETRIC = FKL default and cv::cuda::resize, HALF_PIXEL = NPP and fkl_fused_half_pixel; " +
                              "ok_own = max |output - scalar double reference of the version's own convention| <= 1e-4" + (profileMode() ? "; PROFILE MODE: nothing was timed, the times are zero" : "");
    fklb::CsvWriter timing("e4_preprocessing.csv", "E4a/E4c pre-processing: FKL against OpenCV-CUDA and NPP",
                           {"experiment", "batch", "src_w", "src_h", "crop_w", "crop_h", "out_w", "out_h", "version", "convention", "gpu_mean_ms", "gpu_min_ms", "intermediate_bytes", "meminfo_delta_bytes", "output_bytes", "host_enqueue_us", "host_launch_us"}, notes);
    fklb::CsvWriter agreement("e4_preprocessing_agreement.csv", "E4a/E4c pre-processing: agreement with the scalar references",
                              {"experiment", "batch", "version", "convention", "max_abs_err_own_reference", "ok_own", "max_abs_err_vs_asymmetric", "max_abs_err_vs_half_pixel"},
                              "informational where the conventions differ: the columns vs_asymmetric and vs_half_pixel compare the output with both references; ok_own is the strict check");
    Writers w{timing, agreement};
    bool allOk = true, ok = true;
    if (experimentSelected("E4a")) {
        const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
        ok = runBatches(std::make_index_sequence<sizeof(BATCHES) / sizeof(BATCHES[0])>{}, noise, cs, fkStream, w, allOk) && ok;
    }
    if (experimentSelected("E4c")) {
        struct Frame { int w, h; };
        const Frame frames[] = {{720, 480}, {1920, 1080}, {3840, 2160}};   // "480p", "1080p", "4K" of the CPU rows
        for (const Frame& f : frames) {
            if (!listed("E4_FRAMES_ONLY", f.w)) continue;
            const std::vector<unsigned char> noise = seededNoise(f.w, f.h);
            const Geometry g{"E4c", 1, f.w, f.h, f.w, f.h, f.w * 64 / 60, f.h * 128 / 120, false};
            ok = runCase<1>(g, noise, cs, fkStream, w, allOk) && ok;
        }
    }
    if (!allOk) std::cerr << "at least one version exceeds the tolerance against the reference of its own convention (ok_own = 0 in the agreement CSV)" << std::endl;
    return ok ? 0 : -1;
}
