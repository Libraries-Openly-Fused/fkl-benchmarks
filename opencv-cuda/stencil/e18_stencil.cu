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

// E18 (NVIDIA): a stencil after a resize, FKL's LinearFilterDPP in one kernel against OpenCV-CUDA and NPP, one call per step.
//
// The chain, on one 3840 x 2160 8UC3 frame (the seeded source of E4, as in Figure 15):
//   crop       2560 x 1440 at (640, 360)
//   to float   before the resize, so that no intermediate is rounded
//   resize     2560 x 1440 -> 1920 x 1080, bilinear (a uniform scale of 0.75), each implementation with its sampling convention
//   sharpen    the 3 x 3 kernel  0 -1 0 / -1 5 -1 / 0 -1 0  (it is symmetric: correlation and convolution give the same result), border REPLICATE
//   to 8 bits  a saturating conversion to 8UC3: values at or above 255 give 255, at or below 0 give 0, the others round to the nearest
//              integer with ties to even (FKL's SaturateCast, OpenCV's saturate_cast in convertTo, NPP's NPP_RND_NEAR)
//   output     1920 x 1080 8UC3
// Every implementation reads its own copy of the same host bytes and writes its own output, each made with its own library. Each crops its own
// way: FKL with Ptr2D::crop2D, OpenCV with a GpuMat ROI, NPP with a pointer into its frame and the ROI size.
//   fkl_stencil_asym              FKL, one kernel: LinearFilterDPP (fk::executeLinearFilter). Its image Read IOp is the read of the crop view followed by
//                                 the resize (fk::Resize, ResizeSampling::ASYMMETRIC, the default), which returns float3: the DPP stages a tile of
//                                 resized pixels, with their replicated border, in shared memory. The weights are its second Read IOp (a read of a
//                                 3 x 3 fk::Ptr2D<float3> with each weight in the three channels). Multiply and accumulate are the Unary forms of
//                                 fk::Mul and fk::Add on float3. The epilogue is fk::SaturateCast<float3, uchar3> followed by the write of the output.
//                                 Every call builds the crop view, every IOp and the details of the DPP, and launches.
//   fkl_stencil_halfpixel         the same with fk::ResizeSampling::HALF_PIXEL
//   fkl_stencil_asym_once         fkl_stencil_asym with the crop view, the IOps and the details built once; every call is fk::executeLinearFilter
//   fkl_stencil_halfpixel_once    the same for fkl_stencil_halfpixel
//   opencv_cuda                   OpenCV-CUDA, one call per step: GpuMat ROI, convertTo (CV_32FC3), cv::cuda::resize (INTER_LINEAR: ASYMMETRIC),
//                                 cv::cuda::cvtColor (BGR to BGRA), the linear filter (cv::cuda::createLinearFilter, CV_32FC4, BORDER_REPLICATE),
//                                 convertTo (CV_8UC4), cv::cuda::cvtColor (BGRA to BGR). The linear filter of cv::cuda takes only one- or four-channel
//                                 images, so the three channels are given a fourth before the filter and lose it after the conversion to 8 bits (the
//                                 other way, split / three filters / merge, takes more calls). The filter object is created once, before the calls, as
//                                 a user creates it once for a stream of frames.
//   npp                           NPP, one call per step: nppiConvert_8u32f_C3R_Ctx (a pointer into the frame and the crop size), nppiResize_32f_C3R_Ctx
//                                 (NPPI_INTER_LINEAR: HALF_PIXEL), nppiFilterBorder_32f_C3R_Ctx (NPP_BORDER_REPLICATE, anchor (1, 1)),
//                                 nppiConvert_32f8u_C3R_Ctx (NPP_RND_NEAR)
// Sampling convention (the names of fk::ResizeSampling and common/resize_reference.h): ASYMMETRIC (source position x * scale): fkl_stencil_asym(_once),
// cv::cuda::resize. HALF_PIXEL ((x + 0.5) * scale - 0.5): fkl_stencil_halfpixel(_once), NPP.
//
// Every run checks first, then times (the timing is skipped when a check fails):
//   Agreement (e18_stencil_agreement.csv): every output against the scalar double-precision reference of its own convention (common/resize_reference.h,
//   the 3 x 3 kernel with the replicated border and the saturating conversion in double), each FKL variant against the baseline of its
//   convention, and the build-once variants against their per-call variants: the number of differing values and pixels, the largest difference,
//   and, over the differing values, the largest distance of the unrounded reference to the nearest half level (where the roundings of two
//   nearly equal values can differ by one level).
//   Checks of FKL's path (e18_stencil_checks.csv), which FKL's tests do not cover (a Read IOp with a resize as the image of the DPP, uchar3 to float3):
//     the resize alone (the same image Read IOp, written as float3 by fk::executeOperations) against the double-precision resize of its convention;
//     LinearFilterDPP with the identity kernel (1 at the anchor) and a float3 write against that resize alone, which must be bitwise identical: it
//     shows that the staged tile holds the resized pixels at the right places, the border included;
//     the resized float images of OpenCV and NPP (their own intermediates, read after the call) against FKL's resize of the same convention.
//   Timing (e18_stencil.csv), two intervals per implementation, in the order of the list above (the per-call versions, then the build-once
//   variants), the same in every run:
//     whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call, E4's interval (Figure 15)
//     kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch, E1's interval (Table III): the preparation (FKL: the crop view, the IOps
//                  and the details; OpenCV: the ROI; NPP: the ROI pointer; the build-once variants: nothing) is timed on the host; the whole call
//                  (every kernel of it) is enqueued behind the spin, so the GPU interval runs from the end of the spin to the end of its last kernel
//   The fused kernel's parameter block: the sizes of its five arguments.
//   E18_ONLY=a,b: only these implementations. E18_NO_TIMING=1: the checks only. E18_SPIN_US: the length of the spin (default 5000).
//   E18_PROFILE=1: every selected implementation is called once inside an NVTX range named after it, after a synchronised setup, and nothing is
//   compared or timed (to count the kernels of one call with nsys, or to profile the fused kernel with ncu).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/execution_model/executors.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <npp.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_X = 640, CROP_Y = 360, CROP_W = 2560, CROP_H = 1440;
constexpr int OUT_W = 1920, OUT_H = 1080;
constexpr int KERNEL_W = 3, KERNEL_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr float WEIGHTS[KERNEL_H][KERNEL_W] = {{0.f, -1.f, 0.f}, {-1.f, 5.f, -1.f}, {0.f, -1.f, 0.f}};

bool selected(const std::string& name) {
    const char* e = std::getenv("E18_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// The source of E4 (opencv-cuda/batched-preprocessing/e4_preprocessing.cu), as in the CPU rows (bench_preproc_a.cpp): an LCG with seed 12345,
// bytes 31..24, 23..16, 15..8 of every state, row by row
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

// The scalar reference in double: the crop of the frame, the bilinear resize of the convention, the 3 x 3 kernel with the replicated border of
// the resized image, then the saturating conversion (at or above 255: 255; at or below 0: 0; otherwise the nearest integer, ties to even).
// `resized` receives the resized image and `sharpened` the result of the kernel before the conversion (OUT_W * OUT_H * 3 doubles each, interleaved).
std::vector<unsigned char> reference(const std::vector<unsigned char>& noise, const fklb::ResizeSampling sampling, std::vector<double>& resized,
                                     std::vector<double>& sharpened) {
    resized.assign(static_cast<size_t>(OUT_W) * OUT_H * 3, 0.);
    sharpened.assign(static_cast<size_t>(OUT_W) * OUT_H * 3, 0.);
    const unsigned char* crop = noise.data() + (static_cast<size_t>(CROP_Y) * SRC_W + CROP_X) * 3;
    fklb::resizeBilinear(crop, static_cast<size_t>(SRC_W) * 3, CROP_W, CROP_H, 3, OUT_W, OUT_H, sampling, resized.data());
    std::vector<unsigned char> out(static_cast<size_t>(OUT_W) * OUT_H * 3);
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                double acc = 0.;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    for (int kx = 0; kx < KERNEL_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), OUT_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), OUT_H - 1);
                        acc += static_cast<double>(WEIGHTS[ky][kx]) * resized[(static_cast<size_t>(sy) * OUT_W + sx) * 3 + c];
                    }
                }
                const double v = acc >= 255. ? 255. : (acc <= 0. ? 0. : std::nearbyint(acc));
                out[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] = static_cast<unsigned char>(v);
                sharpened[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] = acc;
            }
        }
    }
    return out;
}

struct Difference {
    size_t values = 0;          // compared values (OUT_W * OUT_H * 3)
    size_t valuesDiffer = 0;    // values that differ
    size_t pixelsDiffer = 0;    // pixels with at least one differing channel
    int maxLevels = 0;          // largest |a - b| in 8-bit levels
    double maxToHalf = -1.;     // over the differing values: the largest distance of the unrounded reference to the nearest half level
                                // (k + 0.5), where two roundings of nearly the same value can differ; -1 when no value differs
};

// `sharpened`: the unrounded reference of the convention, for the distance of each differing value to a rounding boundary
Difference compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const std::vector<double>& sharpened) {
    if (a.size() != b.size()) throw std::runtime_error("compare8u: different sizes");
    Difference d;
    d.values = a.size();
    for (size_t p = 0; p < a.size(); p += 3) {
        bool pixel = false;
        for (int c = 0; c < 3; ++c) {
            const int diff = std::abs(static_cast<int>(a[p + c]) - static_cast<int>(b[p + c]));
            if (diff != 0) {
                ++d.valuesDiffer;
                pixel = true;
                d.maxLevels = std::max(d.maxLevels, diff);
                const double v = sharpened[p + c];
                d.maxToHalf = std::max(d.maxToHalf, std::abs(v - (std::floor(v) + 0.5)));
            }
        }
        if (pixel) ++d.pixelsDiffer;
    }
    return d;
}

// a 1920 x 1080 uchar3 image of the device, rows `pitch` bytes apart, as tightly packed host bytes
std::vector<unsigned char> download8u(const void* data, const size_t pitch) {
    std::vector<unsigned char> host(static_cast<size_t>(OUT_W) * OUT_H * 3);
    FKLB_CUDA_CHECK(cudaMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3, data, pitch, static_cast<size_t>(OUT_W) * 3, OUT_H, cudaMemcpyDeviceToHost));
    return host;
}

// a 1920 x 1080 float3 image of the device, rows `pitch` bytes apart, as tightly packed host floats
std::vector<float> download32f(const void* data, const size_t pitch) {
    std::vector<float> host(static_cast<size_t>(OUT_W) * OUT_H * 3);
    FKLB_CUDA_CHECK(cudaMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3 * sizeof(float), data, pitch, static_cast<size_t>(OUT_W) * 3 * sizeof(float),
                                 OUT_H, cudaMemcpyDeviceToHost));
    return host;
}

template <typename A, typename B>
double maxAbs(const std::vector<A>& a, const std::vector<B>& b) {
    double m = 0.;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

bool bitwiseEqual(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}


const char* name(const fklb::ResizeSampling s) { return s == fklb::ResizeSampling::ASYMMETRIC ? "ASYMMETRIC" : "HALF_PIXEL"; }

struct Output {
    std::string version;
    fklb::ResizeSampling sampling;
    std::vector<unsigned char> image;
};

// The GPU spin of the kernel-only interval, the one of E1 (handwritten-cuda/e1_common.h): one thread waits on the GPU's global timer and touches no
// memory, so the L2 stays as the previous call left it. Enqueued before the start event (the beforeLaunch hook of fklb::timePreparedLaunch), it keeps
// the GPU busy while the host records the start event and enqueues the whole call: the call is already queued when the GPU reaches the start event, and
// the interval holds its kernels, not the host's dispatch of them. It has to last longer than the slowest dispatch of one call.
__global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E18_SPIN_US: the length of gpuSpin in microseconds, default 5000
unsigned long long spinUs() {
    const char* e = std::getenv("E18_SPIN_US");
    return std::strtoull(e != nullptr ? e : "5000", nullptr, 10);
}

using Details = fk::LinearFilterDPPDetails<float3>;   // the default tile (16 x 8) and largest kernel (7 x 7): the call a user writes

// The arguments of one fk::executeLinearFilter, as the kernel-only interval of the per-call FKL variants prepares them (built in place, then launched)
template <typename Reads, typename Multiply, typename Accumulate, typename Write>
struct FilterArguments {
    Reads reads;
    Multiply multiply;
    Accumulate accumulate;
    Write output;
    Details details;
};

// One implementation: its whole call, its kernel-only measurement and the download of its output
struct Implementation {
    std::string version;
    fklb::ResizeSampling sampling;
    std::function<void()> call;
    std::function<void(fklb::Timing& gpu, fklb::Timing& host)> kernelOnly;
    std::function<std::vector<unsigned char>()> download;
};
} // namespace

int launch() {
    const bool profile = std::getenv("E18_PROFILE") != nullptr && std::string(std::getenv("E18_PROFILE")) == "1";
    const bool timing = !profile && !(std::getenv("E18_NO_TIMING") != nullptr && std::string(std::getenv("E18_NO_TIMING")) == "1");
    std::cout << "E18 stencil: " << (profile ? "profile mode (one call per implementation, nothing compared, nothing timed)"
                                             : (timing ? "agreement checks, then the timing" : "agreement checks only (E18_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };
    // The implementations, in the order in which they are checked and timed (the same in every run)
    std::vector<Implementation> impls;

    // ---- FKL: its own copy of the frame (fk::Ptr2D), the weights (a 3 x 3 fk::Ptr2D<float3>) and one output (fk::Ptr2D<uchar3>) per variant
    fk::Ptr2D<uchar3> dInput(SRC_W, SRC_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(dInput.ptr().data, dInput.ptr().dims.pitch, noise.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    fk::Ptr2D<float3> dWeights(KERNEL_W, KERNEL_H), dIdentity(KERNEL_W, KERNEL_H);
    {
        std::vector<float3> w(KERNEL_W * KERNEL_H), id(KERNEL_W * KERNEL_H);
        for (int ky = 0; ky < KERNEL_H; ++ky) {
            for (int kx = 0; kx < KERNEL_W; ++kx) {
                const float v = WEIGHTS[ky][kx];
                w[ky * KERNEL_W + kx] = make_float3(v, v, v);
                const float one = (kx == ANCHOR_X && ky == ANCHOR_Y) ? 1.f : 0.f;
                id[ky * KERNEL_W + kx] = make_float3(one, one, one);
            }
        }
        FKLB_CUDA_CHECK(cudaMemcpy2D(dWeights.ptr().data, dWeights.ptr().dims.pitch, w.data(), KERNEL_W * sizeof(float3), KERNEL_W * sizeof(float3), KERNEL_H,
                                     cudaMemcpyHostToDevice));
        FKLB_CUDA_CHECK(cudaMemcpy2D(dIdentity.ptr().data, dIdentity.ptr().dims.pitch, id.data(), KERNEL_W * sizeof(float3), KERNEL_W * sizeof(float3), KERNEL_H,
                                     cudaMemcpyHostToDevice));
    }
    fk::Ptr2D<uchar3> outAsym(OUT_W, OUT_H), outHalf(OUT_W, OUT_H), outAsymOnce(OUT_W, OUT_H), outHalfOnce(OUT_W, OUT_H);

    // fkl_stencil_asym: every call builds the crop view, every IOp and the details, and launches
    const auto callAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
        const auto weights = fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights);
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsym));
        const Details details{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), multiply, accumulate, output, fkStream)) {
            throw std::runtime_error("fkl_stencil_asym: the details are not valid");
        }
    };
    // its kernel-only measurement: the same building as the preparation (host), the launch alone behind the spin (GPU)
    const auto prepareAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
                                              fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H))),
                                          fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights));
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsym));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(multiply)>, std::decay_t<decltype(accumulate)>,
                               std::decay_t<decltype(output)>>{reads, multiply, accumulate, output,
                                                               Details{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y}};
    };
    // fkl_stencil_halfpixel: the same with ResizeSampling::HALF_PIXEL
    const auto callHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H)));
        const auto weights = fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights);
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalf));
        const Details details{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), multiply, accumulate, output, fkStream)) {
            throw std::runtime_error("fkl_stencil_halfpixel: the details are not valid");
        }
    };
    const auto prepareHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto reads = fk::make_tuple(
            fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
                fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H))),
            fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights));
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalf));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(multiply)>, std::decay_t<decltype(accumulate)>,
                               std::decay_t<decltype(output)>>{reads, multiply, accumulate, output,
                                                               Details{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y}};
    };
    const auto launchPrepared = [&](const auto& a) {
        if (!fk::executeLinearFilter(a.details, a.reads, a.multiply, a.accumulate, a.output, fkStream)) {
            throw std::runtime_error("FKL: the details are not valid");
        }
    };
    // the build-once variants: valid only while the crop (its position and size) does not change
    const fk::Ptr2D<uchar3> cropOnce = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
    const auto readsAsymOnce = fk::make_tuple(
        fk::PerThreadRead<fk::ND::_2D, uchar3>::build(cropOnce).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H))),
        fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights));
    const auto readsHalfOnce = fk::make_tuple(
        fk::PerThreadRead<fk::ND::_2D, uchar3>::build(cropOnce).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H))),
        fk::PerThreadRead<fk::ND::_2D, float3>::build(dWeights));
    const auto multiplyOnce = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
    const auto accumulateOnce = fk::Add<float3, float3, float3, fk::UnaryType>::build();
    const auto outputAsymOnce = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsymOnce));
    const auto outputHalfOnce = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalfOnce));
    const Details detailsOnce{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
    const auto callAsymOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsAsymOnce, multiplyOnce, accumulateOnce, outputAsymOnce, fkStream)) {
            throw std::runtime_error("fkl_stencil_asym_once: the details are not valid");
        }
    };
    const auto callHalfOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsHalfOnce, multiplyOnce, accumulateOnce, outputHalfOnce, fkStream)) {
            throw std::runtime_error("fkl_stencil_halfpixel_once: the details are not valid");
        }
    };
    const auto nothingToPrepare = [] { return 0; };
    {
        // the parameter block of the fused kernel: the sizes of its five arguments
        const auto a = prepareAsym();
        std::cout << "fused kernel arguments (bytes): details " << sizeof(a.details) << ", reads " << sizeof(a.reads) << " (image "
                  << sizeof(fk::get<0>(a.reads)) << ", weights " << sizeof(fk::get<1>(a.reads)) << "), multiply " << sizeof(a.multiply) << ", accumulate "
                  << sizeof(a.accumulate) << ", output " << sizeof(a.output) << std::endl;
    }

    // ---- OpenCV-CUDA: its own copy of the frame (GpuMat::upload), its own intermediates and output
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    cv::cuda::GpuMat cvSrc;
    cvSrc.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, const_cast<unsigned char*>(noise.data())));
    const cv::Size cvOutSize(OUT_W, OUT_H);
    cv::cuda::GpuMat cvF32(CROP_H, CROP_W, CV_32FC3), cvRs(cvOutSize, CV_32FC3), cvRs4(cvOutSize, CV_32FC4), cvSharp4(cvOutSize, CV_32FC4),
        cvU8c4(cvOutSize, CV_8UC4), cvOut(cvOutSize, CV_8UC3);
    const cv::Mat cvKernel = cv::Mat(KERNEL_H, KERNEL_W, CV_32F, const_cast<float*>(&WEIGHTS[0][0])).clone();
    // the filter object is created once, before the calls, as a user creates it once for a stream of frames
    const cv::Ptr<cv::cuda::Filter> cvSharpen =
        cv::cuda::createLinearFilter(CV_32FC4, CV_32FC4, cvKernel, cv::Point(ANCHOR_X, ANCHOR_Y), cv::BORDER_REPLICATE);
    const auto callOpenCV = [&] {
        const cv::cuda::GpuMat roi = cvSrc(cv::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        roi.convertTo(cvF32, CV_32FC3, 1.0, cvs);
        cv::cuda::resize(cvF32, cvRs, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
        cv::cuda::cvtColor(cvRs, cvRs4, cv::COLOR_BGR2BGRA, 0, cvs);
        cvSharpen->apply(cvRs4, cvSharp4, cvs);
        cvSharp4.convertTo(cvU8c4, CV_8UC4, 1.0, cvs);
        cv::cuda::cvtColor(cvU8c4, cvOut, cv::COLOR_BGRA2BGR, 0, cvs);
    };
    // its kernel-only measurement: the ROI as the preparation, the six calls behind the spin
    const auto prepareOpenCV = [&] { return cvSrc(cv::Rect(CROP_X, CROP_Y, CROP_W, CROP_H)); };
    const auto launchOpenCV = [&](const cv::cuda::GpuMat& roi) {
        roi.convertTo(cvF32, CV_32FC3, 1.0, cvs);
        cv::cuda::resize(cvF32, cvRs, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
        cv::cuda::cvtColor(cvRs, cvRs4, cv::COLOR_BGR2BGRA, 0, cvs);
        cvSharpen->apply(cvRs4, cvSharp4, cvs);
        cvSharp4.convertTo(cvU8c4, CV_8UC4, 1.0, cvs);
        cv::cuda::cvtColor(cvU8c4, cvOut, cv::COLOR_BGRA2BGR, 0, cvs);
    };

    // ---- NPP: its own copy of the frame (cudaMallocPitch), its own intermediates and output
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
    Npp8u* nppFrame = nullptr;
    Npp32f *nppF32 = nullptr, *nppRs = nullptr, *nppSharp = nullptr, *nppKernel = nullptr;
    Npp8u* nppOut = nullptr;
    size_t frameStep = 0, f32Step = 0, rsStep = 0, sharpStep = 0, outStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &frameStep, hostRow, SRC_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, frameStep, noise.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppF32), &f32Step, static_cast<size_t>(CROP_W) * 3 * sizeof(Npp32f), CROP_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppRs), &rsStep, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f), OUT_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppSharp), &sharpStep, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f), OUT_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppOut), &outStep, static_cast<size_t>(OUT_W) * 3, OUT_H));
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nppKernel), sizeof(Npp32f) * KERNEL_W * KERNEL_H));
    FKLB_CUDA_CHECK(cudaMemcpy(nppKernel, &WEIGHTS[0][0], sizeof(Npp32f) * KERNEL_W * KERNEL_H, cudaMemcpyHostToDevice));
    const NppiSize nppCropSize{CROP_W, CROP_H}, nppOutSize{OUT_W, OUT_H}, nppKernelSize{KERNEL_W, KERNEL_H};
    const auto callNpp = [&] {
        const Npp8u* roi = nppFrame + static_cast<size_t>(CROP_Y) * frameStep + static_cast<size_t>(CROP_X) * 3;
        FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(roi, static_cast<int>(frameStep), nppF32, static_cast<int>(f32Step), nppCropSize, ctx));
        FKLB_NPP_CHECK(nppiResize_32f_C3R_Ctx(nppF32, static_cast<int>(f32Step), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRs, static_cast<int>(rsStep),
                                              nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
        FKLB_NPP_CHECK(nppiFilterBorder_32f_C3R_Ctx(nppRs, static_cast<int>(rsStep), nppOutSize, NppiPoint{0, 0}, nppSharp, static_cast<int>(sharpStep),
                                                    nppOutSize, nppKernel, nppKernelSize, NppiPoint{ANCHOR_X, ANCHOR_Y}, NPP_BORDER_REPLICATE, ctx));
        FKLB_NPP_CHECK(nppiConvert_32f8u_C3R_Ctx(nppSharp, static_cast<int>(sharpStep), nppOut, static_cast<int>(outStep), nppOutSize, NPP_RND_NEAR, ctx));
    };
    // its kernel-only measurement: the ROI pointer as the preparation, the four calls behind the spin
    const auto prepareNpp = [&] { return static_cast<const Npp8u*>(nppFrame + static_cast<size_t>(CROP_Y) * frameStep + static_cast<size_t>(CROP_X) * 3); };
    const auto launchNpp = [&](const Npp8u* roi) {
        FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(roi, static_cast<int>(frameStep), nppF32, static_cast<int>(f32Step), nppCropSize, ctx));
        FKLB_NPP_CHECK(nppiResize_32f_C3R_Ctx(nppF32, static_cast<int>(f32Step), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRs, static_cast<int>(rsStep),
                                              nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
        FKLB_NPP_CHECK(nppiFilterBorder_32f_C3R_Ctx(nppRs, static_cast<int>(rsStep), nppOutSize, NppiPoint{0, 0}, nppSharp, static_cast<int>(sharpStep),
                                                    nppOutSize, nppKernel, nppKernelSize, NppiPoint{ANCHOR_X, ANCHOR_Y}, NPP_BORDER_REPLICATE, ctx));
        FKLB_NPP_CHECK(nppiConvert_32f8u_C3R_Ctx(nppSharp, static_cast<int>(sharpStep), nppOut, static_cast<int>(outStep), nppOutSize, NPP_RND_NEAR, ctx));
    };

    // The order of the checks and of the timing, the same in every run: the per-call versions, then the build-once variants
    const auto makeKernelOnly = [&](const auto& prepare, const auto& launch) {
        return [&, prepare, launch](fklb::Timing& gpu, fklb::Timing& host) { fklb::timePreparedLaunch(cs, prepare, spin, launch, gpu, host); };
    };
    const auto launchAsymOnce = [&](int) { callAsymOnce(); };
    const auto launchHalfOnce = [&](int) { callHalfOnce(); };
    impls.push_back({"fkl_stencil_asym", fklb::ResizeSampling::ASYMMETRIC, callAsym, makeKernelOnly(prepareAsym, launchPrepared),
                     [&] { return download8u(outAsym.ptr().data, outAsym.ptr().dims.pitch); }});
    impls.push_back({"fkl_stencil_halfpixel", fklb::ResizeSampling::HALF_PIXEL, callHalf, makeKernelOnly(prepareHalf, launchPrepared),
                     [&] { return download8u(outHalf.ptr().data, outHalf.ptr().dims.pitch); }});
    impls.push_back({"opencv_cuda", fklb::ResizeSampling::ASYMMETRIC, callOpenCV, makeKernelOnly(prepareOpenCV, launchOpenCV),
                     [&] { return download8u(cvOut.data, cvOut.step); }});
    impls.push_back({"npp", fklb::ResizeSampling::HALF_PIXEL, callNpp, makeKernelOnly(prepareNpp, launchNpp), [&] { return download8u(nppOut, outStep); }});
    impls.push_back({"fkl_stencil_asym_once", fklb::ResizeSampling::ASYMMETRIC, callAsymOnce, makeKernelOnly(nothingToPrepare, launchAsymOnce),
                     [&] { return download8u(outAsymOnce.ptr().data, outAsymOnce.ptr().dims.pitch); }});
    impls.push_back({"fkl_stencil_halfpixel_once", fklb::ResizeSampling::HALF_PIXEL, callHalfOnce, makeKernelOnly(nothingToPrepare, launchHalfOnce),
                     [&] { return download8u(outHalfOnce.ptr().data, outHalfOnce.ptr().dims.pitch); }});

    // ---- One call of every selected implementation (in the profile mode inside an NVTX range named after it, after the setup has finished)
    std::vector<Output> outputs;
    std::vector<float> resizedOpenCV, resizedNpp;
    for (const Implementation& impl : impls) {
        if (!selected(impl.version)) continue;
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        if (profile) nvtxRangePushA(impl.version.c_str());
        impl.call();
        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        if (profile) nvtxRangePop();
        if (profile) continue;
        outputs.push_back({impl.version, impl.sampling, impl.download()});
        if (impl.version == "opencv_cuda") resizedOpenCV = download32f(cvRs.data, cvRs.step);
        if (impl.version == "npp") resizedNpp = download32f(nppRs, rsStep);
    }
    bool allOk = true;
    if (!profile) {
        // ---- The checks of FKL's path: the resize alone, and LinearFilterDPP with the identity kernel against it
        fklb::CsvWriter checks("e18_stencil_checks.csv", "E18 stencil: checks of FKL's path", {"check", "sampling", "value", "criterion", "ok"});
        std::vector<double> refResizedAsym, refResizedHalf, refSharpAsym, refSharpHalf;
        const std::vector<unsigned char> refAsym = reference(noise, fklb::ResizeSampling::ASYMMETRIC, refResizedAsym, refSharpAsym);
        const std::vector<unsigned char> refHalf = reference(noise, fklb::ResizeSampling::HALF_PIXEL, refResizedHalf, refSharpHalf);
        for (const fklb::ResizeSampling sampling : {fklb::ResizeSampling::ASYMMETRIC, fklb::ResizeSampling::HALF_PIXEL}) {
            fk::Ptr2D<float3> dResized(OUT_W, OUT_H), dIdentityOut(OUT_W, OUT_H);
            const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
            const Details details{OUT_W, OUT_H, KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
            const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
            const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
            bool valid = false;
            if (sampling == fklb::ResizeSampling::ASYMMETRIC) {
                const auto image =
                    fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
                fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream, image,
                                                                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dResized));
                valid = fk::executeLinearFilter(details, fk::make_tuple(image, fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity)), multiply, accumulate,
                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dIdentityOut), fkStream);
            } else {
                const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
                    fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(
                        fk::Size(OUT_W, OUT_H)));
                fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>>(fkStream, image,
                                                                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dResized));
                valid = fk::executeLinearFilter(details, fk::make_tuple(image, fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity)), multiply, accumulate,
                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dIdentityOut), fkStream);
            }
            if (!valid) throw std::runtime_error("identity check: the details are not valid");
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            const std::vector<float> resized = download32f(dResized.ptr().data, dResized.ptr().dims.pitch);
            const std::vector<float> identity = download32f(dIdentityOut.ptr().data, dIdentityOut.ptr().dims.pitch);
            const std::vector<double>& refResized = sampling == fklb::ResizeSampling::ASYMMETRIC ? refResizedAsym : refResizedHalf;
            const double resizeErr = maxAbs(resized, refResized);
            const bool identityOk = bitwiseEqual(identity, resized);
            allOk = allOk && identityOk;
            checks.row("resize alone (fk::executeOperations, float3) against the double-precision resize: max |difference|", name(sampling), resizeErr,
                       "informational (float positions and weights against double)", 1);
            checks.row("LinearFilterDPP with the identity kernel against the resize alone: bitwise identical", name(sampling), identityOk ? 1 : 0, "must be 1",
                       identityOk ? 1 : 0);
            const std::vector<float>& other = sampling == fklb::ResizeSampling::ASYMMETRIC ? resizedOpenCV : resizedNpp;
            if (!other.empty()) {
                checks.row(sampling == fklb::ResizeSampling::ASYMMETRIC ? "OpenCV-CUDA's resized image (CV_32FC3) against FKL's resize: max |difference|"
                                                                        : "NPP's resized image (32f C3) against FKL's resize: max |difference|",
                           name(sampling), maxAbs(other, resized), "informational", 1);
                checks.row(sampling == fklb::ResizeSampling::ASYMMETRIC ? "OpenCV-CUDA's resized image against the double-precision resize: max |difference|"
                                                                        : "NPP's resized image against the double-precision resize: max |difference|",
                           name(sampling), maxAbs(other, refResized), "informational", 1);
            }
            std::cout << "check " << name(sampling) << ": resize alone vs double max |diff| " << resizeErr << "; identity kernel bitwise equal to the resize "
                      << identityOk << std::endl;
        }

        // ---- Agreement of the outputs
        fklb::CsvWriter agreement("e18_stencil_agreement.csv", "E18 stencil: agreement of the 8-bit outputs",
                                  {"version", "sampling", "against", "values", "values_differing", "pixels_differing", "max_diff_levels", "identical",
                                   "max_distance_of_reference_to_half_level"});
        const auto find = [&](const std::string& v) -> const Output* {
            for (const Output& o : outputs) {
                if (o.version == v) return &o;
            }
            return nullptr;
        };
        const auto row = [&](const Output& o, const std::string& against, const std::vector<unsigned char>& other) {
            const Difference d = compare8u(o.image, other, o.sampling == fklb::ResizeSampling::ASYMMETRIC ? refSharpAsym : refSharpHalf);
            agreement.row(o.version, name(o.sampling), against, d.values, d.valuesDiffer, d.pixelsDiffer, d.maxLevels, d.valuesDiffer == 0 ? 1 : 0, d.maxToHalf);
            std::cout << o.version << " vs " << against << ": " << d.valuesDiffer << " values / " << d.pixelsDiffer << " pixels differ, max " << d.maxLevels
                      << " levels; reference within " << d.maxToHalf << " of a half level at every differing value" << std::endl;
            return d;
        };
        for (const Output& o : outputs) {
            row(o, std::string("reference_double_") + name(o.sampling), o.sampling == fklb::ResizeSampling::ASYMMETRIC ? refAsym : refHalf);
        }
        const std::pair<const char*, const char*> pairs[] = {{"fkl_stencil_asym", "opencv_cuda"},
                                                             {"fkl_stencil_halfpixel", "npp"},
                                                             {"fkl_stencil_asym_once", "fkl_stencil_asym"},
                                                             {"fkl_stencil_halfpixel_once", "fkl_stencil_halfpixel"},
                                                             {"fkl_stencil_asym_once", "opencv_cuda"},
                                                             {"fkl_stencil_halfpixel_once", "npp"}};
        for (const auto& p : pairs) {
            const Output* a = find(p.first);
            const Output* b = find(p.second);
            if (a == nullptr || b == nullptr) continue;
            const Difference d = row(*a, p.second, b->image);
            // a build-once variant launches the kernel of its per-call variant with the same arguments: its output must be the same
            if (std::string(p.first).find("_once") != std::string::npos && std::string(p.second).rfind("fkl_", 0) == 0) allOk = allOk && d.valuesDiffer == 0;
        }
        std::cout << (allOk ? "E18 checks passed" : "E18 CHECK FAILED") << std::endl;
    }

    // ---- The timing, after the checks, in the order of the list. For every implementation, two intervals:
    //   whole_call   fklb::timeIt: CUDA events around one call, 100 calls after one warm-up call (E4's interval, Figure 15). The per-call versions
    //                build in the call; the build-once variants hold only the launch path and the kernel; the baselines are called as users call them.
    //                When the GPU waits for the host, the interval also holds the host's work.
    //   kernel_only  fklb::timePreparedLaunch behind gpuSpin (E1's interval, Table III): the preparation (FKL: building the crop view, the IOps and
    //                the details; OpenCV: the ROI; NPP: the ROI pointer; the build-once variants: nothing) runs on the host before the spin is enqueued
    //                and is timed on the host; then the spin, the start event, the whole call (every kernel of it), the stop event. The interval runs
    //                from the end of the spin to the end of the call's last kernel.
    // Each interval of each implementation is inside an NVTX range ("whole_call:<version>", "kernel_only:<version>") for the profiler.
    if (timing && allOk) {
        fklb::CsvWriter t("e18_stencil.csv", "E18 stencil: whole-call and kernel-only times",
                          {"version", "sampling", "interval", "mean_ms", "min_ms", "host_prepare_mean_ms", "host_prepare_min_ms", "iterations", "spin_us"});
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            fklb::Timing whole, gpu, host;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("whole_call:" + impl.version).c_str());
            fklb::timeIt(cs, impl.call, whole);
            nvtxRangePop();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("kernel_only:" + impl.version).c_str());
            impl.kernelOnly(gpu, host);
            nvtxRangePop();
            t.row(impl.version, name(impl.sampling), "whole_call", whole.mean, whole.min, "", "", fklb::ITERATIONS, "");
            t.row(impl.version, name(impl.sampling), "kernel_only", gpu.mean, gpu.min, host.mean, host.min, fklb::ITERATIONS, spinUs());
            std::cout << impl.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                      << gpu.min << "), host preparation mean " << host.mean << " ms" << std::endl;
        }
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    FKLB_CUDA_CHECK(cudaFree(nppFrame));
    FKLB_CUDA_CHECK(cudaFree(nppF32));
    FKLB_CUDA_CHECK(cudaFree(nppRs));
    FKLB_CUDA_CHECK(cudaFree(nppSharp));
    FKLB_CUDA_CHECK(cudaFree(nppOut));
    FKLB_CUDA_CHECK(cudaFree(nppKernel));
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
