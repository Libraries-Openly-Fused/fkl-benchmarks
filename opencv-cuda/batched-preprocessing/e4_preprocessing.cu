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

// E4a and E4c (NVIDIA): the pre-processing chain of a DNN, on batches of crops (E4a) and on whole frames (E4c), FKL against OpenCV-CUDA and NPP; on the
// batches also FKL against the hand-written fused kernel of E1c and cvGPUSpeedup against OpenCV-CUDA.
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
// On the batches of crops only (E4a):
//   hand_batch          the hand-written fused kernel of E1c (handwritten-cuda/e1_preproc.cu, copied unchanged below), one launch for the batch: the crop pointers
//                       in a __grid_constant__ array of 8 bytes per crop; the crop size, the output size and the constants of the chain are compile-time constants.
//                       It reads its own copy of the frame (cudaMallocPitch, as in E1c) and writes its own output (cudaMalloc, the layout of the other outputs).
//   hand_padded         E1c's diagnostic: the same arithmetic on 24-byte crop descriptors, with the parameter block padded to the size of FKL's read IOp, so
//                       that the size of the parameter block is separated from the generated code
//   cvgs_fused          cvGPUSpeedup, one launch: an array of GpuMat ROIs -> cvGS::resize -> cvtColor (RGB to BGR) -> multiply -> subtract -> divide -> split into one
//                       continuous GpuMat, the call of opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu, on its own copy of the frame (GpuMat::upload); cvGS::executeOperations
//                       launches with its default, thread fusion enabled
// Sampling convention of every implementation (the names are those of fk::ResizeSampling and of common/resize_reference.h):
//   ASYMMETRIC (source position x * scale): FKL by default (fkl_fused), cv::cuda::resize (opencv_percrop), the hand-written kernels (hand_batch, hand_padded),
//                                           cvGPUSpeedup (cvgs_fused, whose resize is FKL's with the default sampling)
//   HALF_PIXEL ((x + 0.5) * scale - 0.5):   NPP (npp), FKL with ResizeSampling::HALF_PIXEL (fkl_fused_half_pixel); cv::resize on the CPU
// The outputs of implementations with different conventions differ (the agreement between them is informational, written to the agreement CSV); each output is
// checked against the scalar double-precision reference of its OWN convention (fklb::resizeBilinear + the element-wise chain in double): that check is strict,
// max |output - reference| <= TOLERANCE (1e-4, as the CPU rows), and a version that exceeds it is reported (ok_own = 0), never hidden.
// Timing: CUDA events around one call of the whole implementation, 100 calls after one warm-up, mean and min (common/cuda_timing.h). The call is everything the
// implementation does for a frame: the crop locations may change on every image, so it builds its batch (FKL: the crop views, the read and every operation;
// OpenCV: the ROIs; NPP: the crop pointers; the hand-written kernels: the parameter block of crop pointers or crop descriptors; cvGPUSpeedup: the ROIs and every
// operation) and enqueues its work, all inside the interval.
// Pairs of versions (E4a): the outputs of the checked calls compared element by element, with the bitwise equality and the largest difference: hand_batch and
// hand_padded against fkl_fused, cvgs_fused against fkl_fused and against opencv_percrop.
// Intermediate bytes: what the implementation allocates besides its frame and its output tensor (tallied from the allocations) and the change of free device memory
// (cudaMemGetInfo). Host time of one call: host_enqueue_us (the call returns, stream idle before each of 100 calls) and host_launch_us (100 calls back to back without
// synchronisation, wall time per call: when the GPU is slower than the host this is the GPU's pace, because the launch queue fills up).
//
//   E4_BATCHES_ONLY=100,300 / E4_FRAMES_ONLY=3840 (frame widths): only these points of the sweeps (for the profiler)
//   E4_ONLY=a,b   run only these implementations (default all); E4_PROFILE=1: every implementation launches once and nothing is timed (for ncu, one
//   implementation per run with E4_ONLY, to count its kernels and its DRAM bytes). E4_EXPERIMENTS=E4a,E4c (default both).
//   Writes e4_preprocessing.csv, e4_preprocessing_agreement.csv and e4_preprocessing_pairs.csv.
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

#include <cvGPUSpeedup.cuh>

#include <npp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
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

// ---- The hand-written fused kernels of E1c (handwritten-cuda/e1_preproc.cu), copied unchanged: the same names, types and arithmetic, so that they compile to
// the SASS of E1c's kernels. The constants of the chain are compile-time constants of the kernel, as in E1c, with the values of SUBV and DIVV.
constexpr float SUB_X = SUBV[0], SUB_Y = SUBV[1], SUB_Z = SUBV[2];
constexpr float DIV_X = DIVV[0], DIV_Y = DIVV[1], DIV_Z = DIVV[2];
static_assert(fk::getDefaultBlockSize(UP_W, UP_H).x == 32 && fk::getDefaultBlockSize(UP_W, UP_H).y == 8,
              "the hand-written kernels are launched with the block FKL launches for an output image");

template <int BATCH>
struct CropPointers {
    const unsigned char* data[BATCH];
};

template <int BATCH>
struct CropDescriptors {
    struct Crop { const unsigned char* data; unsigned int width, height, pitch; };
    Crop crop[BATCH];
};

__device__ __forceinline__ float3 loadPixel(const unsigned char* base, const size_t pitch, const int x, const int y) {
    const uchar3 p = *reinterpret_cast<const uchar3*>(base + y * pitch + x * sizeof(uchar3));
    return make_float3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
}

// resize (bilinear, ASYMMETRIC: the arithmetic of fk::InterpolateComplete) -> reorder -> * alpha -> - sub -> / div -> planar write.
// srcW and srcH are the size of the crop, imageStride the bytes between two images of the tensor, channelPixels the number of
// elements between two planes of an image (width * height), as fk::TensorSplit computes them.
__device__ __forceinline__ void preprocess(const unsigned char* base, const size_t inPitch, const int srcW, const int srcH, const float fx, const float fy,
                                           const int x, const int y, const int z, float* tensor, const size_t pitch, const size_t imageStride,
                                           const int channelPixels) {
    const float src_x = x * fx;
    const float src_y = y * fy;
    const int x1 = __float2int_rd(src_x);
    const int y1 = __float2int_rd(src_y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    const int x2_read = min(x2, srcW - 1);
    const int y2_read = min(y2, srcH - 1);
    const float3 r00 = loadPixel(base, inPitch, x1, y1);
    const float3 r10 = loadPixel(base, inPitch, x2_read, y1);
    const float3 r01 = loadPixel(base, inPitch, x1, y2_read);
    const float3 r11 = loadPixel(base, inPitch, x2_read, y2_read);
    const float3 v = make_float3(
        (r00.x * ((x2 - src_x) * (y2 - src_y))) + (r10.x * ((src_x - x1) * (y2 - src_y))) + (r01.x * ((x2 - src_x) * (src_y - y1))) + (r11.x * ((src_x - x1) * (src_y - y1))),
        (r00.y * ((x2 - src_x) * (y2 - src_y))) + (r10.y * ((src_x - x1) * (y2 - src_y))) + (r01.y * ((x2 - src_x) * (src_y - y1))) + (r11.y * ((src_x - x1) * (src_y - y1))),
        (r00.z * ((x2 - src_x) * (y2 - src_y))) + (r10.z * ((src_x - x1) * (y2 - src_y))) + (r01.z * ((x2 - src_x) * (src_y - y1))) + (r11.z * ((src_x - x1) * (src_y - y1))));
    // RGB to BGR, then the three operations, each on the three channels
    const float o0 = ((v.z * ALPHA) - SUB_X) / DIV_X;
    const float o1 = ((v.y * ALPHA) - SUB_Y) / DIV_Y;
    const float o2 = ((v.x * ALPHA) - SUB_Z) / DIV_Z;
    float* const plane = reinterpret_cast<float*>(reinterpret_cast<char*>(tensor) + z * imageStride + y * pitch) + x;
    plane[0] = o0;
    plane[channelPixels] = o1;
    plane[2 * channelPixels] = o2;
}

template <int BATCH>
__global__ void handPreproc(const __grid_constant__ CropPointers<BATCH> crops, const size_t inPitch, float* __restrict__ tensor, const size_t pitch,
                            const size_t imageStride, const float fx, const float fy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= UP_W || y >= UP_H) return;
    preprocess(crops.data[z], inPitch, CROP_W, CROP_H, fx, fy, x, y, z, tensor, pitch, imageStride, UP_W * UP_H);
}

template <int BATCH, size_t PAD>
struct PaddedCropDescriptors {
    CropDescriptors<BATCH> descriptors;
    unsigned char padding[PAD];
};

template <int BATCH, size_t PAD>
__global__ void handPreprocPadded(const __grid_constant__ PaddedCropDescriptors<BATCH, PAD> params, float* __restrict__ tensor, const size_t pitch,
                                  const size_t imageStride, const float fx, const float fy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= UP_W || y >= UP_H) return;
    const auto& c = params.descriptors.crop[z];
    preprocess(c.data, c.pitch, static_cast<int>(c.width), static_cast<int>(c.height), fx, fy, x, y, z, tensor, pitch, imageStride, UP_W * UP_H);
}

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
    fklb::CsvWriter& pairs;
};

// Runs one implementation: clears its output, launches it once and checks it, then times it. A failed call throws (common/status_check.h).
// Returns the output of the checked call (empty when the implementation is not selected), for the comparison of pairs of versions.
template <typename Launch>
std::vector<float> measure(const Geometry& g, const char* version, const char* convention, cudaStream_t cs, float* dOut, const Launch& launch,
                           const size_t intermediateBytes, const size_t meminfoDelta, const std::vector<double>& refAsym, const std::vector<double>& refHalf, Writers& w,
                           bool& allOk) {
    if (!selected(version)) return {};
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
    return host;
}

// One row of the pairs CSV: two outputs of the same chain, element by element (skipped when either version did not run)
void comparePair(const Geometry& g, const char* versionA, const std::vector<float>& a, const char* versionB, const std::vector<float>& b, Writers& w) {
    if (a.empty() || b.empty()) return;
    if (a.size() != b.size()) throw std::runtime_error(std::string("the outputs of ") + versionA + " and " + versionB + " differ in size");
    const bool bitwise = std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
    double m = 0;
    for (size_t i = 0; i < a.size(); ++i) m = fklb::maxKeepNaN(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    w.pairs.row(g.experiment, g.batch, versionA, versionB, int(bitwise), m, int(m <= TOLERANCE));
    w.pairs.flush();
    std::cout << g.experiment << " batch " << g.batch << "  " << versionA << " vs " << versionB << ": " << (bitwise ? "bitwise equal" : "not bitwise equal")
              << ", max |difference| " << m << std::endl;
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
    const std::vector<float> outFkl = measure(g, "fkl_fused", "ASYMMETRIC", cs, dTensor.ptr().data, launchFklAsym, 0, 0, refAsym, refHalf, w, allOk);
    measure(g, "fkl_fused_half_pixel", "HALF_PIXEL", cs, dTensor.ptr().data, launchFklHalf, 0, 0, refAsym, refHalf, w, allOk);

    // ---- OpenCV-CUDA: its own copy of the frame (GpuMat::upload) and its own output (one continuous GpuMat, BATCH x 3 planes of outH rows)
    std::vector<float> outOpenCV;
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
        outOpenCV = measure(g, "opencv_percrop", "ASYMMETRIC", cs, reinterpret_cast<float*>(out.data), launch, tally, delta, refAsym, refHalf, w, allOk);
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

    // ---- On the batches of crops (E4a) only: the hand-written kernels, whose crop size and output size are compile-time constants, and cvGPUSpeedup
    if constexpr (!frame) {
        if (g.cropW != CROP_W || g.cropH != CROP_H || g.outW != UP_W || g.outH != UP_H) {
            throw std::runtime_error("the hand-written kernels are compiled for 60x120 crops and 64x128 outputs");
        }

        // ---- The hand-written kernels: their own copy of the frame (cudaMallocPitch, uploaded from the same host bytes, as in E1c) and their own output
        // (cudaMalloc, BATCH x 3 planes of outH x outW floats, the layout of the other outputs); no FKL object
        unsigned char* hSrc = nullptr;
        size_t hPitch = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&hSrc), &hPitch, hostRow, g.srcH));
        FKLB_CUDA_CHECK(cudaMemcpy2D(hSrc, hPitch, noise.data(), hostRow, hostRow, g.srcH, cudaMemcpyHostToDevice));
        float* hOut = nullptr;
        FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&hOut), outBytes));
        // Set up once, because they depend only on the crop size and the output size, not on where the crops are: the strides of the output, the resize
        // factors (exactly as fk::Resize::build computes them) and the launch geometry (the block FKL launches for an output image, one z per crop).
        const size_t pitch = static_cast<size_t>(UP_W) * sizeof(float), imageStride = pitch * UP_H * 3;
        const double cfx = static_cast<double>(UP_W) / CROP_W, cfy = static_cast<double>(UP_H) / CROP_H;
        const float fx = static_cast<float>(1.0 / cfx), fy = static_cast<float>(1.0 / cfy);
        const dim3 block(32, 8);
        const dim3 grid((UP_W + block.x - 1) / block.x, (UP_H + block.y - 1) / block.y, BATCH);
        // the padding makes hand_padded's parameter block as large as FKL's read IOp (which holds the per-crop pointers, sizes and resize parameters)
        constexpr size_t PAD = 24 * BATCH + 16;
        static_assert(sizeof(CropDescriptors<BATCH>) + PAD ==
                          sizeof(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(std::declval<std::array<fk::Ptr2D<uchar3>, BATCH>&>())
                                     .then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(UP_W, UP_H)))),
                      "hand_padded's parameter block is as large as FKL's read IOp");
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));   // the upload of the frame is done before the first timed interval
        // Every call builds the parameter block from the crop locations, which may change on every image, then launches the kernel and checks the launch.
        auto launchHand = [&] {
            CropPointers<BATCH> crops;
            for (int i = 0; i < BATCH; ++i) crops.data[i] = hSrc + static_cast<size_t>(g.y0(i)) * hPitch + static_cast<size_t>(g.x0(i)) * sizeof(uchar3);
            handPreproc<BATCH><<<grid, block, 0, cs>>>(crops, hPitch, hOut, pitch, imageStride, fx, fy);
            FKLB_CUDA_CHECK(cudaGetLastError());
        };
        auto launchHandPadded = [&] {
            PaddedCropDescriptors<BATCH, PAD> params;
            for (int i = 0; i < BATCH; ++i) {
                params.descriptors.crop[i] = {hSrc + static_cast<size_t>(g.y0(i)) * hPitch + static_cast<size_t>(g.x0(i)) * sizeof(uchar3),
                                              static_cast<unsigned int>(CROP_W), static_cast<unsigned int>(CROP_H), static_cast<unsigned int>(hPitch)};
            }
            handPreprocPadded<BATCH, PAD><<<grid, block, 0, cs>>>(params, hOut, pitch, imageStride, fx, fy);
            FKLB_CUDA_CHECK(cudaGetLastError());
        };
        const std::vector<float> outHand = measure(g, "hand_batch", "ASYMMETRIC", cs, hOut, launchHand, 0, 0, refAsym, refHalf, w, allOk);
        const std::vector<float> outHandPadded = measure(g, "hand_padded", "ASYMMETRIC", cs, hOut, launchHandPadded, 0, 0, refAsym, refHalf, w, allOk);
        FKLB_CUDA_CHECK(cudaFree(hSrc));
        FKLB_CUDA_CHECK(cudaFree(hOut));

        // ---- cvGPUSpeedup: its own copy of the frame (GpuMat::upload) and its own output (one continuous GpuMat, one row per crop holding its three planes:
        // the layout of the other outputs, and the one cvGS::split writes)
        std::vector<float> outCvgs;
        {
            cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
            cv::cuda::GpuMat src;
            src.upload(cv::Mat(g.srcH, g.srcW, CV_8UC3, const_cast<unsigned char*>(noise.data())));
            cv::cuda::GpuMat out;
            cv::cuda::createContinuous(BATCH, g.outW * g.outH * 3, CV_32F, out);
            // the constants of the chain and the output size are set up once: they do not depend on the crops
            const cv::Scalar alphaS(ALPHA, ALPHA, ALPHA), subS(SUBV[0], SUBV[1], SUBV[2]), divS(DIVV[0], DIVV[1], DIVV[2]);
            const cv::Size outSize(g.outW, g.outH);
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));   // the upload of the frame is done before the first timed interval
            // Every call makes the ROI of every crop (the crop locations may change on every image) and builds every operation on them, in the form of the
            // cvGPUSpeedup call of opencv-cuda/benchmark_CPU_OpenCV_vs_cvGS.cu; the ROIs are released at the end of the call, inside the interval.
            auto launch = [&] {
                std::array<cv::cuda::GpuMat, BATCH> crops;
                for (int i = 0; i < BATCH; ++i) crops[i] = src(cv::Rect(g.x0(i), g.y0(i), g.cropW, g.cropH));
                cvGS::executeOperations(cvs, cvGS::resize<CV_8UC3, cv::INTER_LINEAR, BATCH>(crops, outSize, BATCH), cvGS::cvtColor<cv::COLOR_RGB2BGR, CV_32FC3>(),
                                        cvGS::multiply<CV_32FC3>(alphaS), cvGS::subtract<CV_32FC3>(subS), cvGS::divide<CV_32FC3>(divS),
                                        cvGS::split<CV_32FC3>(out, outSize));
            };
            outCvgs = measure(g, "cvgs_fused", "ASYMMETRIC", cs, reinterpret_cast<float*>(out.data), launch, 0, 0, refAsym, refHalf, w, allOk);
        }

        // ---- Pairs of versions: the outputs of their checked calls
        comparePair(g, "hand_batch", outHand, "fkl_fused", outFkl, w);
        comparePair(g, "hand_padded", outHandPadded, "fkl_fused", outFkl, w);
        comparePair(g, "cvgs_fused", outCvgs, "fkl_fused", outFkl, w);
        comparePair(g, "cvgs_fused", outCvgs, "opencv_percrop", outOpenCV, w);
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
                              "timing: CUDA events around one call of the whole implementation, which builds its batch (FKL: crop views, read and operations; OpenCV: ROIs; NPP: crop pointers; "
                              "hand_batch and hand_padded: the parameter block of crop pointers or crop descriptors; cvgs_fused: ROIs and operations) and enqueues its work, because the "
                              "crop locations may change on every image; 100 calls after one warm-up; hand_batch and hand_padded: the hand-written kernels of E1c "
                              "(handwritten-cuda/e1_preproc.cu), E4a only; cvgs_fused: cvGPUSpeedup (cvGS::executeOperations, its default thread fusion), E4a only; sampling: "
                              "ASYMMETRIC = FKL default, cv::cuda::resize, the hand-written kernels and cvGPUSpeedup, HALF_PIXEL = NPP and fkl_fused_half_pixel; ok_own = max |output - "
                              "scalar double reference of the version's own convention| <= 1e-4" +
                              (profileMode() ? "; PROFILE MODE: nothing was timed, the times are zero" : "");
    fklb::CsvWriter timing("e4_preprocessing.csv", "E4a/E4c pre-processing: FKL against OpenCV-CUDA and NPP; E4a also against the hand-written kernels of E1c, and cvGPUSpeedup",
                           {"experiment", "batch", "src_w", "src_h", "crop_w", "crop_h", "out_w", "out_h", "version", "convention", "gpu_mean_ms", "gpu_min_ms", "intermediate_bytes", "meminfo_delta_bytes", "output_bytes", "host_enqueue_us", "host_launch_us"}, notes);
    fklb::CsvWriter agreement("e4_preprocessing_agreement.csv", "E4a/E4c pre-processing: agreement with the scalar references",
                              {"experiment", "batch", "version", "convention", "max_abs_err_own_reference", "ok_own", "max_abs_err_vs_asymmetric", "max_abs_err_vs_half_pixel"},
                              "informational where the conventions differ: the columns vs_asymmetric and vs_half_pixel compare the output with both references; ok_own is the strict check");
    fklb::CsvWriter pairs("e4_preprocessing_pairs.csv", "E4a pre-processing: pairs of versions compared element by element",
                          {"experiment", "batch", "version_a", "version_b", "bitwise_equal", "max_abs_diff", "within_tolerance"},
                          "the outputs of the checked call of each version (before its timing); bitwise_equal = 1 when the two outputs have the same bytes; "
                          "within_tolerance = 1 when max_abs_diff <= 1e-4 (informational: each version's strict check is ok_own of the agreement CSV)");
    Writers w{timing, agreement, pairs};
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
