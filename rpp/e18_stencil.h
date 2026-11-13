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

// E18 (AMD): a stencil after a resize, FKL's LinearFilterDPP in one kernel, on an AMD GPU (HIP, ParArch::GPU_AMD). The FKL side of
// opencv-cuda/stencil/e18_stencil.cu (E18 on NVIDIA), with the same chain, variants, checks and timing.
//
// The chain, on one 3840 x 2160 8UC3 frame (the seeded source of E4, as in Figure 15):
//   crop       2560 x 1440 at (640, 360)
//   to float   before the resize, so that no intermediate is rounded
//   resize     2560 x 1440 -> 1920 x 1080, bilinear (a uniform scale of 0.75), with the sampling convention of the variant
//   sharpen    the 3 x 3 kernel  0 -1 0 / -1 5 -1 / 0 -1 0  (it is symmetric: correlation and convolution give the same result), border REPLICATE
//   to 8 bits  a saturating conversion to 8UC3: values at or above 255 give 255, at or below 0 give 0, the others round to the nearest integer
//              with ties to even (FKL's SaturateCast)
//   output     1920 x 1080 8UC3
//   fkl_stencil_asym              FKL, one kernel: LinearFilterDPP (fk::executeLinearFilter). Its image Read IOp is the read of the crop view
//                                 (Ptr2D::crop2D) followed by the resize (fk::Resize, ResizeSampling::ASYMMETRIC, the default), which returns float3:
//                                 the DPP stages a tile of resized pixels, with their replicated border, in shared memory. The weights are its second
//                                 Read IOp (a read of a 3 x 3 fk::Ptr2D<float3> with each weight in the three channels). Multiply and accumulate are
//                                 the Unary forms of fk::Mul and fk::Add on float3. The epilogue is fk::SaturateCast<float3, uchar3> followed by the
//                                 write of the output. Every call builds the crop view, every IOp and the details of the DPP, and launches.
//   fkl_stencil_halfpixel         the same with fk::ResizeSampling::HALF_PIXEL (the convention of RPP)
//   fkl_stencil_asym_once         fkl_stencil_asym with the crop view, the IOps and the details built once; every call is fk::executeLinearFilter
//   fkl_stencil_halfpixel_once    the same for fkl_stencil_halfpixel
// The RPP baseline is not here: RPP 3.2.0 has no 2D filter with user weights (its filters are box, median, Gaussian, Sobel and emboss) and no
// conversion from F32 to U8, so the pipeline cannot be written with one documented RPP call per step.
// Sampling convention (the names of fk::ResizeSampling and common/resize_reference.h): ASYMMETRIC (source position x * scale), HALF_PIXEL
// ((x + 0.5) * scale - 0.5).
//
// Differences from the FKL code of opencv-cuda/stencil/e18_stencil.cu, which are the back-end's only:
//   the stream is a hipStream_t in a fk::Stream_<fk::ParArch::GPU_AMD> (CUDA: cudaStream_t, GPU_NVIDIA); the resize-alone check uses
//   fk::TransformDPP<fk::ParArch::GPU_AMD, ...>; the FKL vector types are written fk::uchar3 / fk::float3 (under HIP, ::uchar3 and ::float3 are HIP's
//   types) and the weights are built as fk::float3{v, v, v} (CUDA: make_float3); the copies use the HIP runtime (hipMemcpy2D); the timing is
//   common/hip_timing.h (common/cuda_timing.h with HIP events); the spin waits on the GPU's constant-rate wall clock (wall_clock64) instead of
//   %globaltimer; there is no NVTX and no E18_PROFILE mode.
//
// Every run checks first, then times (the timing is skipped when a check fails):
//   Agreement (e18_stencil_agreement_GPU_AMD.csv): every output against the scalar double-precision reference of its own convention
//   (common/resize_reference.h, the 3 x 3 kernel with the replicated border and the saturating conversion in double), and the build-once variants
//   against their per-call variants: the number of differing values and pixels, the largest difference, and, over the differing values, the largest
//   distance of the unrounded reference to the nearest half level (where the roundings of two nearly equal values can differ by one level).
//   Checks of FKL's path (e18_stencil_checks_GPU_AMD.csv): the resize alone (the same image Read IOp, written as float3 by fk::executeOperations)
//   against the double-precision resize of its convention; LinearFilterDPP with the identity kernel (1 at the anchor) and a float3 write against
//   that resize alone, which must be bitwise identical.
//   Timing (e18_stencil_GPU_AMD.csv), two intervals per variant, in the order of the list above, the same in every run:
//     whole_call   fklb::timeIt (common/hip_timing.h): events around one call, 100 calls after one warm-up call, E4's interval (Figure 15)
//     kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch, E1's interval (Table III): the preparation (the crop view, the IOps and
//                  the details; the build-once variants: nothing) is timed on the host; the call is enqueued behind the spin, so the GPU interval
//                  runs from the end of the spin to the end of the kernel
//   The fused kernel's parameter block: the sizes of its five arguments.
//   E18_ONLY=a,b: only these variants. E18_NO_TIMING=1: the checks only. E18_SPIN_US: the length of the spin (default 5000).
#include <common/csv_writer.h>
#include <common/hip_timing.h>
#include <common/main.h>
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

#include <hip/hip_runtime.h>

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

namespace e18amd {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_X = 640, CROP_Y = 360, CROP_W = 2560, CROP_H = 1440;
constexpr int OUT_W = 1920, OUT_H = 1080;
constexpr int KERNEL_W = 3, KERNEL_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr float WEIGHTS[KERNEL_H][KERNEL_W] = {{0.f, -1.f, 0.f}, {-1.f, 5.f, -1.f}, {0.f, -1.f, 0.f}};

inline bool selected(const std::string& name) {
    const char* e = std::getenv("E18_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// The source of E4 (opencv-cuda/batched-preprocessing/e4_preprocessing.cu): an LCG with seed 12345, bytes 31..24, 23..16, 15..8 of every state,
// row by row
inline std::vector<unsigned char> seededNoise(const int width, const int height) {
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

// The scalar reference in double, as in opencv-cuda/stencil/e18_stencil.cu
inline std::vector<unsigned char> reference(const std::vector<unsigned char>& noise, const fklb::ResizeSampling sampling, std::vector<double>& resized,
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
    size_t values = 0;
    size_t valuesDiffer = 0;
    size_t pixelsDiffer = 0;
    int maxLevels = 0;
    double maxToHalf = -1.;
};

inline Difference compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const std::vector<double>& sharpened) {
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

inline std::vector<unsigned char> download8u(const void* data, const size_t pitch) {
    std::vector<unsigned char> host(static_cast<size_t>(OUT_W) * OUT_H * 3);
    FKLB_HIP_CHECK(hipMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3, data, pitch, static_cast<size_t>(OUT_W) * 3, OUT_H, hipMemcpyDeviceToHost));
    return host;
}

inline std::vector<float> download32f(const void* data, const size_t pitch) {
    std::vector<float> host(static_cast<size_t>(OUT_W) * OUT_H * 3);
    FKLB_HIP_CHECK(hipMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3 * sizeof(float), data, pitch, static_cast<size_t>(OUT_W) * 3 * sizeof(float),
                               OUT_H, hipMemcpyDeviceToHost));
    return host;
}

template <typename A, typename B>
double maxAbs(const std::vector<A>& a, const std::vector<B>& b) {
    double m = 0.;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    return m;
}

inline bool bitwiseEqual(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

inline const char* name(const fklb::ResizeSampling s) { return s == fklb::ResizeSampling::ASYMMETRIC ? "ASYMMETRIC" : "HALF_PIXEL"; }

struct Output {
    std::string version;
    fklb::ResizeSampling sampling;
    std::vector<unsigned char> image;
};

// The GPU spin of the kernel-only interval (E1's role, handwritten-cuda/e1_common.h): one thread waits on the GPU's constant-rate wall clock
// (s_memrealtime, wall_clock64()) and touches no memory. Enqueued before the start event, it keeps the GPU busy while the host records the start
// event and enqueues the call, so that the interval holds the kernel and not the host's dispatch of it. `ticks` = microseconds * the wall clock
// rate (hipDeviceAttributeWallClockRate, kHz) / 1000.
__global__ void gpuSpin(const unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
}

inline unsigned long long spinUs() {
    const char* e = std::getenv("E18_SPIN_US");
    return std::strtoull(e != nullptr ? e : "5000", nullptr, 10);
}

using Details = fk::LinearFilterDPPDetails<fk::float3>;   // the default tile (16 x 8) and largest kernel (7 x 7): the call a user writes

template <typename Reads, typename Multiply, typename Accumulate, typename Write>
struct FilterArguments {
    Reads reads;
    Multiply multiply;
    Accumulate accumulate;
    Write output;
    Details details;
};

struct Implementation {
    std::string version;
    fklb::ResizeSampling sampling;
    std::function<void()> call;
    std::function<void(fklb::Timing& gpu, fklb::Timing& host)> kernelOnly;
    std::function<std::vector<unsigned char>()> download;
};
} // namespace e18amd

int launch() {
    using namespace e18amd;
    using fk::float3;
    using fk::uchar3;
    const bool timing = !(std::getenv("E18_NO_TIMING") != nullptr && std::string(std::getenv("E18_NO_TIMING")) == "1");
    std::cout << "E18 stencil (AMD): " << (timing ? "agreement checks, then the timing" : "agreement checks only (E18_NO_TIMING=1)") << std::endl;
    const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    hipStream_t cs;
    FKLB_HIP_CHECK(hipStreamCreate(&cs));
    fk::Stream_<fk::ParArch::GPU_AMD> fkStream(cs);
    int device = 0, wallClockKHz = 0;
    FKLB_HIP_CHECK(hipGetDevice(&device));
    FKLB_HIP_CHECK(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, device));
    const unsigned long long spinTicks = spinUs() * static_cast<unsigned long long>(wallClockKHz) / 1000ull;
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinTicks); };
    std::vector<Implementation> impls;

    // ---- FKL: its own copy of the frame (fk::Ptr2D), the weights (a 3 x 3 fk::Ptr2D<float3>) and one output (fk::Ptr2D<uchar3>) per variant
    fk::Ptr2D<uchar3> dInput(SRC_W, SRC_H);
    FKLB_HIP_CHECK(hipMemcpy2D(dInput.ptr().data, dInput.ptr().dims.pitch, noise.data(), hostRow, hostRow, SRC_H, hipMemcpyHostToDevice));
    fk::Ptr2D<float3> dWeights(KERNEL_W, KERNEL_H), dIdentity(KERNEL_W, KERNEL_H);
    {
        std::vector<float3> w(KERNEL_W * KERNEL_H), id(KERNEL_W * KERNEL_H);
        for (int ky = 0; ky < KERNEL_H; ++ky) {
            for (int kx = 0; kx < KERNEL_W; ++kx) {
                const float v = WEIGHTS[ky][kx];
                w[ky * KERNEL_W + kx] = float3{v, v, v};
                const float one = (kx == ANCHOR_X && ky == ANCHOR_Y) ? 1.f : 0.f;
                id[ky * KERNEL_W + kx] = float3{one, one, one};
            }
        }
        FKLB_HIP_CHECK(hipMemcpy2D(dWeights.ptr().data, dWeights.ptr().dims.pitch, w.data(), KERNEL_W * sizeof(float3), KERNEL_W * sizeof(float3), KERNEL_H,
                                   hipMemcpyHostToDevice));
        FKLB_HIP_CHECK(hipMemcpy2D(dIdentity.ptr().data, dIdentity.ptr().dims.pitch, id.data(), KERNEL_W * sizeof(float3), KERNEL_W * sizeof(float3), KERNEL_H,
                                   hipMemcpyHostToDevice));
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
    impls.push_back({"fkl_stencil_asym_once", fklb::ResizeSampling::ASYMMETRIC, callAsymOnce, makeKernelOnly(nothingToPrepare, launchAsymOnce),
                     [&] { return download8u(outAsymOnce.ptr().data, outAsymOnce.ptr().dims.pitch); }});
    impls.push_back({"fkl_stencil_halfpixel_once", fklb::ResizeSampling::HALF_PIXEL, callHalfOnce, makeKernelOnly(nothingToPrepare, launchHalfOnce),
                     [&] { return download8u(outHalfOnce.ptr().data, outHalfOnce.ptr().dims.pitch); }});

    // ---- One call of every selected variant
    std::vector<Output> outputs;
    for (const Implementation& impl : impls) {
        if (!selected(impl.version)) continue;
        FKLB_HIP_CHECK(hipStreamSynchronize(cs));
        impl.call();
        FKLB_HIP_CHECK(hipStreamSynchronize(cs));
        outputs.push_back({impl.version, impl.sampling, impl.download()});
    }
    bool allOk = true;
    {
        // ---- The checks of FKL's path: the resize alone, and LinearFilterDPP with the identity kernel against it
        fklb::CsvWriter checks("e18_stencil_checks_GPU_AMD.csv", "E18 stencil (AMD): checks of FKL's path", {"check", "sampling", "value", "criterion", "ok"});
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
                fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, image,
                                                                                             fk::PerThreadWrite<fk::ND::_2D, float3>::build(dResized));
                valid = fk::executeLinearFilter(details, fk::make_tuple(image, fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity)), multiply, accumulate,
                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dIdentityOut), fkStream);
            } else {
                const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(
                    fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(
                        fk::Size(OUT_W, OUT_H)));
                fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, image,
                                                                                             fk::PerThreadWrite<fk::ND::_2D, float3>::build(dResized));
                valid = fk::executeLinearFilter(details, fk::make_tuple(image, fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity)), multiply, accumulate,
                                                fk::PerThreadWrite<fk::ND::_2D, float3>::build(dIdentityOut), fkStream);
            }
            if (!valid) throw std::runtime_error("identity check: the details are not valid");
            FKLB_HIP_CHECK(hipStreamSynchronize(cs));
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
            std::cout << "check " << name(sampling) << ": resize alone vs double max |diff| " << resizeErr << "; identity kernel bitwise equal to the resize "
                      << identityOk << std::endl;
        }

        // ---- Agreement of the outputs
        fklb::CsvWriter agreement("e18_stencil_agreement_GPU_AMD.csv", "E18 stencil (AMD): agreement of the 8-bit outputs",
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
        const std::pair<const char*, const char*> pairs[] = {{"fkl_stencil_asym_once", "fkl_stencil_asym"},
                                                             {"fkl_stencil_halfpixel_once", "fkl_stencil_halfpixel"}};
        for (const auto& p : pairs) {
            const Output* a = find(p.first);
            const Output* b = find(p.second);
            if (a == nullptr || b == nullptr) continue;
            const Difference d = row(*a, p.second, b->image);
            allOk = allOk && d.valuesDiffer == 0;   // the build-once variant launches the kernel of its per-call variant with the same arguments
        }
        std::cout << (allOk ? "E18 checks passed" : "E18 CHECK FAILED") << std::endl;
    }

    // ---- The timing, after the checks, in the order of the list: whole_call (fklb::timeIt) and kernel_only (fklb::timePreparedLaunch behind gpuSpin)
    if (timing && allOk) {
        fklb::CsvWriter t("e18_stencil_GPU_AMD.csv", "E18 stencil (AMD): whole-call and kernel-only times",
                          {"version", "sampling", "interval", "mean_ms", "min_ms", "host_prepare_mean_ms", "host_prepare_min_ms", "iterations", "spin_us"});
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            fklb::Timing whole, gpu, host;
            FKLB_HIP_CHECK(hipStreamSynchronize(cs));
            fklb::timeIt(cs, impl.call, whole);
            FKLB_HIP_CHECK(hipStreamSynchronize(cs));
            impl.kernelOnly(gpu, host);
            t.row(impl.version, name(impl.sampling), "whole_call", whole.mean, whole.min, "", "", fklb::ITERATIONS, "");
            t.row(impl.version, name(impl.sampling), "kernel_only", gpu.mean, gpu.min, host.mean, host.min, fklb::ITERATIONS, spinUs());
            std::cout << impl.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                      << gpu.min << "), host preparation mean " << host.mean << " ms" << std::endl;
        }
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_HIP_CHECK(hipStreamSynchronize(cs));
    FKLB_HIP_CHECK(hipStreamDestroy(cs));
    return allOk ? 0 : 1;
}
