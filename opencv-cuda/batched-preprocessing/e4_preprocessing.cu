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
//   E4c  the same chain on one whole frame: 720x480, 1920x1080 and 3840x2160 (the "480p", "1080p" and "4K" of the CPU rows, in that order),
//        output 64/60 x 128/120 of the frame (768x512, 2048x1152, 4096x2304; the factor 15/16 is exact in float): the regime where the working set
//        is far above the caches and the launch overhead is irrelevant. A single frame is not a batch on any side: FKL reads the frame, OpenCV uses
//        its frame GpuMat, NPP calls its single-image resize.
// Every implementation reads its own copy of the same host bytes and writes its own output, each made with its own library: FKL an fk::Ptr2D and an
// fk::Tensor, OpenCV-CUDA a GpuMat (upload) and a continuous GpuMat, NPP cudaMallocPitch images and a cudaMalloc tensor. Each crops its own way: FKL
// with Ptr2D::crop2D, OpenCV with a GpuMat ROI, NPP with a pointer into its frame.
//   opencv_percrop      OpenCV-CUDA, every operation per crop: convertTo, resize, multiply, cvtColor, subtract, divide, split (7 calls per crop)
//   npp                 NPP: nppiConvert_8u32f per crop, ONE nppiResizeBatch_32f_C3R_Advanced for the batch (nppiResize_32f_C3R for a single frame), then per crop
//                       SwapChannels, MulC, SubC, DivC and Copy_32f_C3P3R into the tensor; the intermediates ping-pong between two stacks (fewer than a buffer per step)
//   fkl_fused           FKL, one kernel: BatchRead of the crop views -> Resize -> VectorReorder -> Mul -> Sub -> Div -> TensorSplit (thread fusion off); on a frame,
//                       a read of the frame -> Resize -> the same operations
//   fkl_fused_half_pixel  the same with fk::ResizeSampling::HALF_PIXEL, the convention of NPP and cv::resize
// Sampling convention of every implementation (the names are those of fk::ResizeSampling and of common/resize_reference.h):
//   ASYMMETRIC (source position x * scale): FKL by default (fkl_fused), cv::cuda::resize (opencv_percrop)
//   HALF_PIXEL ((x + 0.5) * scale - 0.5):   NPP (npp), FKL with ResizeSampling::HALF_PIXEL (fkl_fused_half_pixel); cv::resize on the CPU
// The outputs of implementations with different conventions differ (the agreement between them is informational, written to the agreement CSV); each output is
// checked against the scalar double-precision reference of its OWN convention (fklb::resizeBilinear + the element-wise chain in double): that check is strict,
// max |output - reference| <= TOLERANCE (1e-4, as the CPU rows), and a version that exceeds it is reported (ok_own = 0), never hidden.
// Timing: CUDA events around one call of the whole implementation, 100 calls after one warm-up, mean and min (common/cuda_timing.h). The call is everything the
// implementation does for a frame: the crop locations may change on every image, so it builds its batch (FKL: the crop views, the read and every operation;
// OpenCV: the ROIs; NPP: the crop pointers) and enqueues its work, all inside the interval.
// Intermediate bytes: what the implementation allocates besides its frame and its output tensor (tallied from the allocations) and the change of free device memory
// (cudaMemGetInfo). Host time of one call: host_enqueue_us (the call returns, stream idle before each of 100 calls) and host_launch_us (100 calls back to back without
// synchronisation, wall time per call: when the GPU is slower than the host this is the GPU's pace, because the launch queue fills up).
//
//   E4_BATCHES_ONLY=100,300 / E4_FRAMES_ONLY=3840 (frame widths): only these points of the sweeps (for the profiler)
//   E4_ONLY=a,b   run only these implementations (default all); E4_PROFILE=1: every implementation launches once and nothing is timed (for ncu, one
//   implementation per run with E4_ONLY, to count its kernels and its DRAM bytes). E4_EXPERIMENTS=E4a,E4c (default both).
//   Writes e4_preprocessing.csv and e4_preprocessing_agreement.csv.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/nan_safe.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/data/rect.h>
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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "e4_batches.h"   // E4A_BATCHES, generated by CMake from the cache variable of the same name

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, UP_W = 64, UP_H = 128;
constexpr int BATCHES[] = {E4A_BATCHES};
constexpr float ALPHA = 0.3f;
constexpr float SUBV[3] = {1.f, 4.f, 3.2f};
constexpr float DIVV[3] = {3.2f, 0.6f, 11.8f};
constexpr double TOLERANCE = 1e-4;

bool profileMode() { return std::getenv("E4_PROFILE") != nullptr && std::getenv("E4_PROFILE")[0] == '1'; }

bool selected(const std::string& name) {
    const char* only = std::getenv("E4_ONLY");
    if (only == nullptr) return true;
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

size_t freeBytes() {
    size_t f = 0, t = 0;
    FKLB_CUDA_CHECK(cudaMemGetInfo(&f, &t));
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
    for (size_t i = 0; i < out.size(); ++i) m = fklb::maxKeepNaN(m, std::abs(static_cast<double>(out[i]) - ref[i]));   // a NaN in the output makes m NaN, and NaN <= TOLERANCE is false
    return m;
}

struct Writers {
    fklb::CsvWriter& timing;
    fklb::CsvWriter& agreement;
};

// Runs one implementation: clears its output, launches it once and checks it, then times it. A failed call throws (common/status_check.h).
template <typename Launch>
void measure(const Geometry& g, const char* version, const char* convention, cudaStream_t cs, float* dOut, const Launch& launch, const size_t intermediateBytes,
             const size_t meminfoDelta, const std::vector<double>& refAsym, const std::vector<double>& refHalf, Writers& w, bool& allOk) {
    if (!selected(version)) return;
    const size_t outBytes = g.tensorElems() * sizeof(float);
    FKLB_CUDA_CHECK(cudaMemsetAsync(dOut, 0, outBytes, cs));
    launch();
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    std::vector<float> host(g.tensorElems());
    FKLB_CUDA_CHECK(cudaMemcpy(host.data(), dOut, outBytes, cudaMemcpyDeviceToHost));
    const double eAsym = maxAbsDiff(host, refAsym), eHalf = maxAbsDiff(host, refHalf);
    const bool own = std::string(convention) == "ASYMMETRIC";
    const double eOwn = own ? eAsym : eHalf;
    const bool ok = eOwn <= TOLERANCE;
    allOk = allOk && ok;
    fklb::Timing t;
    if (!profileMode()) fklb::timeIt(cs, launch, t);
    // Host time of one call, in two ways (microseconds):
    //   host_enqueue_us  the time the call takes to return, with the stream idle before each call (mean of 100 calls, synchronised between them): what it costs the host to build and enqueue the call
    //   host_launch_us   100 calls back to back with no synchronisation between them, the wall time per call (as the host_launch_us of the E2 script): when the GPU is slower than the host, the
    //                    launch queue fills up and this is the time the GPU needs, not what the host spends
    double hostEnqueueUs = 0, hostLaunchUs = 0;
    if (!profileMode()) {
        using Clock = std::chrono::steady_clock;
        const auto us = [](const Clock::time_point a, const Clock::time_point b) { return std::chrono::duration<double, std::micro>(b - a).count(); };
        for (int i = 0; i < fklb::ITERATIONS; ++i) {
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            const auto t0 = Clock::now();
            launch();
            hostEnqueueUs += us(t0, Clock::now()) / fklb::ITERATIONS;
        }
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        const auto t0 = Clock::now();
        for (int i = 0; i < fklb::ITERATIONS; ++i) launch();
        hostLaunchUs = us(t0, Clock::now()) / fklb::ITERATIONS;
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    }
    w.timing.row(g.experiment, g.batch, g.srcW, g.srcH, g.cropW, g.cropH, g.outW, g.outH, version, convention, t.mean, t.min, intermediateBytes, meminfoDelta, outBytes, hostEnqueueUs, hostLaunchUs);
    w.timing.flush();
    w.agreement.row(g.experiment, g.batch, version, convention, eOwn, int(ok), eAsym, eHalf);
    w.agreement.flush();
    std::cout << g.experiment << " batch " << g.batch << " " << g.srcW << "x" << g.srcH << " -> " << g.outW << "x" << g.outH << "  " << version << " (" << convention << "): "
              << t.mean << " ms (min " << t.min << "), host enqueue " << hostEnqueueUs << " us, back to back " << hostLaunchUs << " us, intermediates " << intermediateBytes << " B, max |err| vs own reference " << eOwn << (ok ? "" : "  EXCEEDS " + std::to_string(TOLERANCE))
              << ", vs ASYMMETRIC " << eAsym << ", vs HALF_PIXEL " << eHalf << std::endl;
}

template <int BATCH>
void runCase(const Geometry& g, const std::vector<unsigned char>& noise, cudaStream_t cs, fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream, Writers& w, bool& allOk) {
    // a whole frame (E4c, batch 1) is one crop that covers the frame; a batch of crops (E4a) has more than one
    constexpr bool frame = BATCH == 1;
    if (frame != !g.diagonal || (frame && (g.cropW != g.srcW || g.cropH != g.srcH))) throw std::runtime_error("batch 1 must be a whole frame, and a whole frame batch 1");
    const std::vector<double> refAsym = reference(g, noise, fklb::ResizeSampling::ASYMMETRIC), refHalf = reference(g, noise, fklb::ResizeSampling::HALF_PIXEL);
    const size_t outBytes = g.tensorElems() * sizeof(float);
    const size_t hostRow = static_cast<size_t>(g.srcW) * 3;

    // ---- FKL: its own copy of the frame (fk::Ptr2D) and its own output (fk::Tensor, the layout of the other outputs)
    fk::Ptr2D<uchar3> dInput(g.srcW, g.srcH);
    FKLB_CUDA_CHECK(cudaMemcpy2D(dInput.ptr().data, dInput.ptr().dims.pitch, noise.data(), hostRow, hostRow, g.srcH, cudaMemcpyHostToDevice));
    fk::Tensor<float> dTensor(g.outW, g.outH, BATCH, 3);
    if (dTensor.ptr().dims.pitch != static_cast<size_t>(g.outW) * sizeof(float)) {
        throw std::runtime_error("the rows of the FKL tensor are padded (pitch " + std::to_string(dTensor.ptr().dims.pitch) + "), fk::TensorSplit assumes they are not");
    }
    // Every call builds its batch and every operation: the crop locations may change on every image, so the crop views, the read and the
    // operations are made in the call (and released at its end, inside the timed interval).
    auto launchFklAsym = [&] {
        if constexpr (frame) {   // a single frame is not a batch: the read is a read of the frame
            fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream,
                fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dInput).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(g.outW, g.outH))),
                fk::VectorReorder<float3, 2, 1, 0>::build(), fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA)),
                fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2])), fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2])),
                fk::TensorSplit<float3>::build(dTensor.ptr()));
        } else {
            std::array<fk::Ptr2D<uchar3>, BATCH> crops;
            for (int i = 0; i < BATCH; ++i) crops[i] = dInput.crop2D(fk::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
            fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream,
                fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(g.outW, g.outH))),
                fk::VectorReorder<float3, 2, 1, 0>::build(), fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA)),
                fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2])), fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2])),
                fk::TensorSplit<float3>::build(dTensor.ptr()));
        }
    };
    auto launchFklHalf = [&] {
        if constexpr (frame) {   // a single frame is not a batch: the read is a read of the frame
            fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream,
                fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dInput).then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(g.outW, g.outH))),
                fk::VectorReorder<float3, 2, 1, 0>::build(), fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA)),
                fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2])), fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2])),
                fk::TensorSplit<float3>::build(dTensor.ptr()));
        } else {
            std::array<fk::Ptr2D<uchar3>, BATCH> crops;
            for (int i = 0; i < BATCH; ++i) crops[i] = dInput.crop2D(fk::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
            fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream,
                fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(g.outW, g.outH))),
                fk::VectorReorder<float3, 2, 1, 0>::build(), fk::Mul<float3>::build(make_float3(ALPHA, ALPHA, ALPHA)),
                fk::Sub<float3>::build(make_float3(SUBV[0], SUBV[1], SUBV[2])), fk::Div<float3>::build(make_float3(DIVV[0], DIVV[1], DIVV[2])),
                fk::TensorSplit<float3>::build(dTensor.ptr()));
        }
    };
    measure(g, "fkl_fused", "ASYMMETRIC", cs, dTensor.ptr().data, launchFklAsym, 0, 0, refAsym, refHalf, w, allOk);
    measure(g, "fkl_fused_half_pixel", "HALF_PIXEL", cs, dTensor.ptr().data, launchFklHalf, 0, 0, refAsym, refHalf, w, allOk);

    // ---- OpenCV-CUDA: its own copy of the frame (GpuMat::upload) and its own output (one continuous GpuMat, BATCH x 3 planes of outH rows)
    {
        cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
        cv::cuda::GpuMat src;
        src.upload(cv::Mat(g.srcH, g.srcW, CV_8UC3, const_cast<unsigned char*>(noise.data())));
        cv::cuda::GpuMat out;
        cv::cuda::createContinuous(BATCH * 3 * g.outH, g.outW, CV_32F, out);
        // the plane views of the output are set up once: they depend only on the layout of the output, not on the crops
        std::vector<std::array<cv::cuda::GpuMat, 3>> planes(BATCH);
        for (int i = 0; i < BATCH; ++i) {
            for (int c = 0; c < 3; ++c) planes[i][c] = out.rowRange((i * 3 + c) * g.outH, (i * 3 + c + 1) * g.outH);
        }
        const cv::Scalar alphaS(ALPHA, ALPHA, ALPHA), subS(SUBV[0], SUBV[1], SUBV[2]), divS(DIVV[0], DIVV[1], DIVV[2]);
        const cv::Size outSize(g.outW, g.outH);
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        const size_t before = freeBytes();
        cv::cuda::GpuMat f32(g.cropH, g.cropW, CV_32FC3), rs(outSize, CV_32FC3), a(outSize, CV_32FC3), b(outSize, CV_32FC3);
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        const size_t delta = before - std::min(before, freeBytes());
        const size_t tally = f32.step * f32.rows + (rs.step * rs.rows) * 3;
        // Every call makes the ROI of every crop: the crop locations may change on every image. A frame is not a batch: its GpuMat is used as it is.
        auto launch = [&] {
            for (int i = 0; i < BATCH; ++i) {
                const cv::cuda::GpuMat roi = frame ? src : src(cv::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
                roi.convertTo(f32, CV_32FC3, 1.0, cvs);
                cv::cuda::resize(f32, rs, outSize, 0., 0., cv::INTER_LINEAR, cvs);
                cv::cuda::multiply(rs, alphaS, a, 1.0, -1, cvs);
                cv::cuda::cvtColor(a, b, cv::COLOR_RGB2BGR, 0, cvs);
                cv::cuda::subtract(b, subS, a, cv::noArray(), -1, cvs);
                cv::cuda::divide(a, divS, b, 1.0, -1, cvs);
                cv::cuda::split(b, planes[i].data(), cvs);
            }
        };
        if (g.batch == BATCHES[0] || frame) {
            std::cout << g.experiment << " batch " << g.batch << " " << g.srcW << "x" << g.srcH << ": frame steps (bytes) FKL " << dInput.ptr().dims.pitch << ", OpenCV " << src.step
                      << "; OpenCV output continuous " << out.isContinuous() << std::endl;
        }
        measure(g, "opencv_percrop", "ASYMMETRIC", cs, reinterpret_cast<float*>(out.data), launch, tally, delta, refAsym, refHalf, w, allOk);
    }

    // ---- NPP: its own copy of the frame (cudaMallocPitch) and its own output (cudaMalloc); per-crop conversion, one batched resize, per-crop arithmetic and
    // split, the intermediates ping-ponging between two stacks
    {
        // The NPP stream context, filled field by field from the device properties and the stream's flags (CUDA 13 has no nppGetStreamContext)
        int device = 0;
        FKLB_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        FKLB_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        NppStreamContext ctx{};
        ctx.hStream = cs;
        ctx.nCudaDeviceId = device;
        ctx.nMultiProcessorCount = props.multiProcessorCount;
        ctx.nMaxThreadsPerMultiProcessor = props.maxThreadsPerMultiProcessor;
        ctx.nMaxThreadsPerBlock = props.maxThreadsPerBlock;
        ctx.nSharedMemPerBlock = props.sharedMemPerBlock;
        ctx.nCudaDevAttrComputeCapabilityMajor = props.major;
        ctx.nCudaDevAttrComputeCapabilityMinor = props.minor;
        FKLB_CUDA_CHECK(cudaStreamGetFlags(cs, &ctx.nStreamFlags));
        ctx.hWorkspace = 0;

        Npp8u* nppSrc = nullptr;
        size_t nppSrcStep = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppSrc), &nppSrcStep, hostRow, g.srcH));
        FKLB_CUDA_CHECK(cudaMemcpy2D(nppSrc, nppSrcStep, noise.data(), hostRow, hostRow, g.srcH, cudaMemcpyHostToDevice));
        Npp32f* nppOut = nullptr;
        FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nppOut), outBytes));
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        const size_t before = freeBytes();
        // the crop stack and the two ping-pong stacks: one cudaMallocPitch image each, image i of a stack starting at row i * (image height)
        Npp8u *cropStack = nullptr, *stackA = nullptr, *stackB = nullptr;
        size_t cropStep = 0, upStep = 0, upStepB = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&cropStack), &cropStep, static_cast<size_t>(g.cropW) * 3 * sizeof(Npp32f), static_cast<size_t>(BATCH) * g.cropH));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&stackA), &upStep, static_cast<size_t>(g.outW) * 3 * sizeof(Npp32f), static_cast<size_t>(BATCH) * g.outH));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&stackB), &upStepB, static_cast<size_t>(g.outW) * 3 * sizeof(Npp32f), static_cast<size_t>(BATCH) * g.outH));
        if (upStepB != upStep) throw std::runtime_error("cudaMallocPitch returned different steps for the two stacks");
        const NppiSize cropSize{g.cropW, g.cropH}, upSize{g.outW, g.outH};
        // NPP's batched-resize descriptors are set up once: they point only at NPP's own crop stack and stack A, with full-image ROIs, so they do
        // not depend on where the crops are in the frame. A single frame does not use them (NPP's batched resize needs more than one image).
        NppiImageDescriptor *dSrc = nullptr, *dDst = nullptr;
        NppiResizeBatchROI_Advanced* dRoi = nullptr;
        size_t descriptorBytes = 0;
        if constexpr (BATCH > 1) {
            FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dSrc), sizeof(NppiImageDescriptor) * BATCH));
            FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dDst), sizeof(NppiImageDescriptor) * BATCH));
            FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dRoi), sizeof(NppiResizeBatchROI_Advanced) * BATCH));
            descriptorBytes = (2 * sizeof(NppiImageDescriptor) + sizeof(NppiResizeBatchROI_Advanced)) * BATCH;
            std::vector<NppiImageDescriptor> hSrc(BATCH), hDst(BATCH);
            std::vector<NppiResizeBatchROI_Advanced> hRoi(BATCH);
            for (int i = 0; i < BATCH; ++i) {
                hSrc[i].pData = cropStack + static_cast<size_t>(i) * g.cropH * cropStep;
                hSrc[i].nStep = static_cast<int>(cropStep);
                hSrc[i].oSize = cropSize;
                hDst[i].pData = stackA + static_cast<size_t>(i) * g.outH * upStep;
                hDst[i].nStep = static_cast<int>(upStep);
                hDst[i].oSize = upSize;
                hRoi[i].oSrcRectROI = NppiRect{0, 0, g.cropW, g.cropH};
                hRoi[i].oDstRectROI = NppiRect{0, 0, g.outW, g.outH};
            }
            FKLB_CUDA_CHECK(cudaMemcpyAsync(dSrc, hSrc.data(), sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, cs));
            FKLB_CUDA_CHECK(cudaMemcpyAsync(dDst, hDst.data(), sizeof(NppiImageDescriptor) * BATCH, cudaMemcpyHostToDevice, cs));
            FKLB_CUDA_CHECK(cudaMemcpyAsync(dRoi, hRoi.data(), sizeof(NppiResizeBatchROI_Advanced) * BATCH, cudaMemcpyHostToDevice, cs));
        }
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        const size_t delta = before - std::min(before, freeBytes());
        const size_t tally = cropStep * BATCH * g.cropH + 2 * upStep * BATCH * g.outH + descriptorBytes;
        const int order[3] = {2, 1, 0};
        const Npp32f mulV[3] = {ALPHA, ALPHA, ALPHA}, subV[3] = {SUBV[0], SUBV[1], SUBV[2]}, divV[3] = {DIVV[0], DIVV[1], DIVV[2]};
        const int stepUp = static_cast<int>(upStep);
        // Every call computes the source pointer of every crop on NPP's own frame: the crop locations may change on every image.
        auto launch = [&] {
            for (int i = 0; i < BATCH; ++i) {
                const Npp8u* srcCrop = nppSrc + static_cast<size_t>(g.y0(i)) * nppSrcStep + static_cast<size_t>(g.x0(i)) * 3;
                Npp32f* const crop = reinterpret_cast<Npp32f*>(cropStack + static_cast<size_t>(i) * g.cropH * cropStep);
                FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(srcCrop, static_cast<int>(nppSrcStep), crop, static_cast<int>(cropStep), cropSize, ctx));
            }
            if constexpr (BATCH == 1) {   // a single frame: NPP's single-image resize
                FKLB_NPP_CHECK(nppiResize_32f_C3R_Ctx(reinterpret_cast<const Npp32f*>(cropStack), static_cast<int>(cropStep), cropSize, NppiRect{0, 0, g.cropW, g.cropH},
                                                      reinterpret_cast<Npp32f*>(stackA), stepUp, upSize,
                                                      NppiRect{0, 0, g.outW, g.outH}, NPPI_INTER_LINEAR, ctx));
            } else {
                FKLB_NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(g.outW, g.outH, dSrc, dDst, dRoi, BATCH, NPPI_INTER_LINEAR, ctx));
            }
            for (int i = 0; i < BATCH; ++i) {
                Npp32f* const planesOut[3] = {nppOut + (static_cast<size_t>(i) * 3 + 0) * g.planeElems(), nppOut + (static_cast<size_t>(i) * 3 + 1) * g.planeElems(),
                                              nppOut + (static_cast<size_t>(i) * 3 + 2) * g.planeElems()};
                Npp32f* const a = reinterpret_cast<Npp32f*>(stackA + static_cast<size_t>(i) * g.outH * upStep);
                Npp32f* const b = reinterpret_cast<Npp32f*>(stackB + static_cast<size_t>(i) * g.outH * upStep);
                FKLB_NPP_CHECK(nppiSwapChannels_32f_C3R_Ctx(a, stepUp, b, stepUp, upSize, order, ctx));
                FKLB_NPP_CHECK(nppiMulC_32f_C3R_Ctx(b, stepUp, mulV, a, stepUp, upSize, ctx));
                FKLB_NPP_CHECK(nppiSubC_32f_C3R_Ctx(a, stepUp, subV, b, stepUp, upSize, ctx));
                FKLB_NPP_CHECK(nppiDivC_32f_C3R_Ctx(b, stepUp, divV, a, stepUp, upSize, ctx));
                FKLB_NPP_CHECK(nppiCopy_32f_C3P3R_Ctx(a, stepUp, planesOut, g.outW * static_cast<int>(sizeof(Npp32f)), upSize, ctx));
            }
        };
        if (g.batch == BATCHES[0] || frame) {
            std::cout << g.experiment << " batch " << g.batch << " " << g.srcW << "x" << g.srcH << ": NPP steps (bytes) frame " << nppSrcStep << ", crop stack " << cropStep
                      << ", stacks " << upStep << "; NPP output dense, " << g.outW * sizeof(Npp32f) << " bytes per row" << std::endl;
        }
        measure(g, "npp", "HALF_PIXEL", cs, nppOut, launch, tally, delta, refAsym, refHalf, w, allOk);
        FKLB_CUDA_CHECK(cudaFree(nppSrc));
        FKLB_CUDA_CHECK(cudaFree(nppOut));
        FKLB_CUDA_CHECK(cudaFree(cropStack));
        FKLB_CUDA_CHECK(cudaFree(stackA));
        FKLB_CUDA_CHECK(cudaFree(stackB));
        if constexpr (BATCH > 1) {
            FKLB_CUDA_CHECK(cudaFree(dSrc));
            FKLB_CUDA_CHECK(cudaFree(dDst));
            FKLB_CUDA_CHECK(cudaFree(dRoi));
        }
    }
}

template <size_t... Is>
void runBatches(std::index_sequence<Is...>, const std::vector<unsigned char>& noise, cudaStream_t cs, fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream, Writers& w, bool& allOk) {
    ((listed("E4_BATCHES_ONLY", BATCHES[Is]) ? runCase<BATCHES[Is]>(Geometry{"E4a", BATCHES[Is], SRC_W, SRC_H, CROP_W, CROP_H, UP_W, UP_H, true}, noise, cs, fkStream, w, allOk)
                                             : void()), ...);
}
}  // namespace

int launch() {
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    const std::string notes = std::string("thread fusion: off (fk::TF::DISABLED); source: seeded 8UC3 noise (LCG seed 12345, as the CPU rows), one copy per implementation; ") +
                              "timing: CUDA events around one call of the whole implementation, which builds its batch (FKL: crop views, read and operations; OpenCV: ROIs; NPP: crop pointers) "
                              "and enqueues its work, because the crop locations may change on every image; 100 calls after one warm-up; sampling: ASYMMETRIC = FKL default and "
                              "cv::cuda::resize, HALF_PIXEL = NPP and fkl_fused_half_pixel; ok_own = max |output - scalar double reference of the version's own convention| <= 1e-4" +
                              (profileMode() ? "; PROFILE MODE: nothing was timed, the times are zero" : "");
    fklb::CsvWriter timing("e4_preprocessing.csv", "E4a/E4c pre-processing: FKL against OpenCV-CUDA and NPP",
                           {"experiment", "batch", "src_w", "src_h", "crop_w", "crop_h", "out_w", "out_h", "version", "convention", "gpu_mean_ms", "gpu_min_ms", "intermediate_bytes", "meminfo_delta_bytes", "output_bytes", "host_enqueue_us", "host_launch_us"}, notes);
    fklb::CsvWriter agreement("e4_preprocessing_agreement.csv", "E4a/E4c pre-processing: agreement with the scalar references",
                              {"experiment", "batch", "version", "convention", "max_abs_err_own_reference", "ok_own", "max_abs_err_vs_asymmetric", "max_abs_err_vs_half_pixel"},
                              "informational where the conventions differ: the columns vs_asymmetric and vs_half_pixel compare the output with both references; ok_own is the strict check");
    Writers w{timing, agreement};
    bool allOk = true;
    if (experimentSelected("E4a")) {
        const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
        runBatches(std::make_index_sequence<sizeof(BATCHES) / sizeof(BATCHES[0])>{}, noise, cs, fkStream, w, allOk);
    }
    if (experimentSelected("E4c")) {
        struct Frame { int w, h; };
        const Frame frames[] = {{720, 480}, {1920, 1080}, {3840, 2160}};   // "480p", "1080p", "4K" of the CPU rows
        for (const Frame& f : frames) {
            if (!listed("E4_FRAMES_ONLY", f.w)) continue;
            const std::vector<unsigned char> noise = seededNoise(f.w, f.h);
            const Geometry g{"E4c", 1, f.w, f.h, f.w, f.h, f.w * 64 / 60, f.h * 128 / 120, false};
            runCase<1>(g, noise, cs, fkStream, w, allOk);
        }
    }
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    if (!allOk) std::cerr << "at least one version exceeds the tolerance against the reference of its own convention (ok_own = 0 in the agreement CSV)" << std::endl;
    return allOk ? 0 : -1;
}
