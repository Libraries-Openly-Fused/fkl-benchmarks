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

// E18 box (NVIDIA): a 3 x 3 box filter after a resize, in uchar from start to end. FKL's LinearFilterDPP in one kernel against OpenCV-CUDA and
// NPP, one call per step. The pipeline is one that OpenCV-CUDA, NPP, RPP and FKL can all run.
//
// The pipeline, the same in every library, on one 3840 x 2160 uchar3 frame (the seeded source of E4, as in e18_stencil.cu):
//   crop     2560 x 1440 at (640, 360)
//   resize   2560 x 1440 -> 1920 x 1080, bilinear (a uniform scale of 0.75), uchar in and uchar out, each implementation with its sampling
//            convention
//   box      3 x 3, anchor at the centre, border REPLICATE, uchar in, uchar3 out: the mean of the 9 values, as an integer. OpenCV-CUDA's and
//            NPP's 8-bit box filters truncate it (floor(sum / 9)); RPP's rounds it to the nearest. FKL has a variant of each: fkl_box_* rounds to the
//            nearest, fkl_trunc_* truncates
//   output   1920 x 1080 uchar3
// Every implementation reads its own copy of the same host bytes and writes its own output, each made with its own library. Each crops its own
// way: FKL with Ptr2D::crop2D, OpenCV with a GpuMat ROI, NPP with a pointer into its frame and its step.
//   fkl_box_asym        FKL, one kernel: LinearFilterDPP (one fk::executeLinearFilter) with the arguments that the body of fk::executeBoxFilter
//                       builds for scale 1, and LinearFilterDPPDetails<float3> with its defaults (a 16 x 8 tile, a largest kernel of 7 x 7):
//                         image        the read of the crop view, then fk::Resize (INTER_LINEAR, ResizeSampling::ASYMMETRIC, the default) to float3,
//                                      then fk::SaturateCast<float3, uchar3> and fk::Cast<uchar3, float3>. The resized image is rounded to uchar
//                                      inside the one kernel, as the libraries round it between their two calls. The DPP stages a tile of these
//                                      values, with their replicated border, in shared memory
//                         coefficient  fk::ConstantFilterRead<float3> with 1 / 9 in every channel (scale / (kernel width * kernel height), computed
//                                      as executeBoxFilter computes it)
//                         multiply     fk::Mul<float3, float3, float3, fk::UnaryType>, accumulate fk::Add<float3, float3, float3, fk::UnaryType>
//                         write        fk::SaturateCast<float3, uchar3>, then the write of the output
//                       fk::executeBoxFilter itself does not compile with float3 details at FKL 24478f0: it computes the coefficient as
//                       scale / static_cast<T>(kernelWidth * kernelHeight) (linear_filter.h:325, and :460 and :501 for the other back-ends), and
//                       float3 has no conversion from int. The call here passes the same argument types, so it launches the same kernel.
//                       Every call builds the crop view, every IOp and the details, and launches.
//   fkl_box_halfpixel   the same with fk::ResizeSampling::HALF_PIXEL
//   fkl_trunc_asym      the truncating twin of fkl_box_asym, the box that OpenCV-CUDA and NPP compute; one fk::executeLinearFilter, the same
//                       image Read IOp and details. The coefficient Read IOp fk::ConstantFilterRead<float3> gives the weight 1 in every channel, so the
//                       accumulator holds the exact sum of the 9 values (an integer of at most 2,295, exact in float). The write divides it by 9
//                       (fk::Div<float3>, a correctly rounded division: the target is built without fast math) and truncates it to uchar3
//                       (fk::Cast<float3, uchar3>). A correctly rounded sum / 9 is exactly k when the sum is 9k and lies between k and k + 1
//                       otherwise, so the result is floor(sum / 9)
//   fkl_trunc_halfpixel the same with fk::ResizeSampling::HALF_PIXEL
//   fkl_asym_once       fkl_box_asym with the crop view, the IOps and the details built once; every call is fk::executeLinearFilter
//   fkl_half_once       the same for fkl_box_halfpixel
//   fkl_trunc_asym_once, fkl_trunc_halfpixel_once   the same for fkl_trunc_asym and fkl_trunc_halfpixel
//   opencv_cuda         OpenCV-CUDA, one call per step: GpuMat ROI, cv::cuda::resize (INTER_LINEAR: ASYMMETRIC, 8UC3 to 8UC3), cv::cuda::cvtColor
//                       (BGR to BGRA), the box filter (cv::cuda::createBoxFilter, CV_8UC4, 3 x 3, anchor (-1, -1), BORDER_REPLICATE),
//                       cv::cuda::cvtColor (BGRA to BGR). The box filter of cv::cuda takes only CV_8UC1, CV_8UC4 and CV_32FC1, so the three channels
//                       are given a fourth before the filter and lose it after. The filter object is created once, before the calls, as a user
//                       creates it once for a stream of frames; its apply() pads the image with cv::cuda::copyMakeBorder, then calls NPP's
//                       nppiFilterBox_8u_C4R_Ctx (opencv_contrib, cudafilters/src/filtering.cpp, NPPBoxFilter::apply), so it truncates as NPP does.
//   npp                 NPP, one call per step: nppiResize_8u_C3R_Ctx (NPPI_INTER_LINEAR: HALF_PIXEL; a pointer into the frame and its step),
//                       nppiFilterBoxBorder_8u_C3R_Ctx (mask 3 x 3, anchor (1, 1), NPP_BORDER_REPLICATE, the resized image as the source, offset 0)
// Sampling convention (the names of fk::ResizeSampling and common/resize_reference.h): ASYMMETRIC (source position x * scale): the FKL variants
// named asym, cv::cuda::resize. HALF_PIXEL ((x + 0.5) * scale - 0.5): the FKL variants named halfpixel or half, NPP.
//
// The references are exact. With these sizes every source position is a multiple of 1/3 (ASYMMETRIC) or 1/6 (HALF_PIXEL), so the resized value
// is an integer over 9 or over 36. The program resizes in integers and rounds to the nearest with ties to even. A double resize
// (common/resize_reference.h) cannot do that: HALF_PIXEL has exact ties (weights of 1/2), and the rounding of the positions in double moves them
// to either side. The box reference is the integer sum of the 9 values over 9, rounded to the nearest (sum / 9 is never a tie) or truncated.
//
// Every run checks first, then times (the timing is skipped when a check fails). The checks (e18_box_agreement.csv, every comparison over the
// 1920 x 1080 x 3 = 6,220,800 values: the differing values and pixels, the largest difference, how many differing values are above and below,
// and, where the unrounded value of the step is known, how many differing values are exact ties and their largest distance to a half level):
//   resize   each library's uchar resize (its own intermediate, read after the pipeline's call) against the exact resize of its convention.
//            FKL's read chain of each convention (the image Read IOp above, through fk::executeLinearFilter with the identity weights 0 0 0 / 0 1 0 /
//            0 0 0 as a 3 x 3 fk::Ptr2D<float3>, and the rounding write) against the exact resize, and against the library's resize of its convention
//   box      on one and the same uchar3 image (1920 x 1080 values of E4's generator, seed 12345, so every level occurs next to every other): the
//            OpenCV path (cvtColor, the box filter, cvtColor), NPP's nppiFilterBoxBorder_8u_C3R_Ctx and FKL's two calls above (rounding and
//            truncating) with a plain read of the image (and fk::Cast<uchar3, float3>) as their image Read IOp, against the exact box rounded and
//            truncated
//   pipeline every implementation against the exact pipeline of its convention (exact resize, rounded, exact box rounded; for the truncating ones
//            also truncated), and against the exact box, rounded or truncated as its own box, of its own resize (FKL: its read chain; the
//            libraries: their intermediate): 0 differences there means that every difference from the reference comes from the resize. Every FKL
//            variant against the library of its convention
//   once     the _once variants against their per-call versions: they must be bitwise equal
// e18_box_checks.csv: the number of exact ties of each convention's resize, and the exact references against the double references.
// e18_box_kernel.csv: the four fused kernels (fkl_box_* and fkl_trunc_*; the _once variants launch the same ones) as the CUDA runtime reports
// them (cudaFuncGetAttributes, cudaOccupancyMaxActiveBlocksPerMultiprocessor for the 128-thread block of the launch) and the sizes of their five
// arguments.
//
// Timing (e18_box.csv), two intervals per implementation, in the order of the list above (the per-call versions, then the build-once variants),
// the same in every run:
//   whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call, E4's interval (Figure 15)
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch, E1's interval (Table III): the preparation (FKL: the crop view, the IOps and
//                the details; OpenCV: the ROI; NPP: the ROI pointer; the build-once variants: nothing) is timed on the host; the whole call (every
//                kernel of it) is enqueued behind one spin, so the GPU interval runs from the end of the spin to the end of its last kernel
//   E18_ONLY=a,b: only these implementations. E18_NO_TIMING=1: the checks only. E18_SPIN_US: the length of the spin (default 5000).
//   E18_PROFILE=1: every selected implementation is called once outside any range (a warm-up, so that what a first call allocates is not counted),
//   then E18_PROFILE_CALLS times (default 10) inside an NVTX range named after it, after a synchronised setup; nothing is compared or timed (to count
//   the kernels and the API calls of one call with nsys, or to profile the fused kernel with ncu with E18_PROFILE_CALLS=1).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
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
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_X = 640, CROP_Y = 360, CROP_W = 2560, CROP_H = 1440;
constexpr int OUT_W = 1920, OUT_H = 1080;
constexpr int BOX_W = 3, BOX_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr float SCALE = 1.f;   // the scale argument of fk::executeBoxFilter: the coefficient is SCALE / (BOX_W * BOX_H)
constexpr size_t OUT_VALUES = static_cast<size_t>(OUT_W) * OUT_H * 3;

bool selected(const std::string& name) {
    const char* e = std::getenv("E18_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

// The source of E4 (opencv-cuda/batched-preprocessing/e4_preprocessing.cu), as in e18_stencil.cu: an LCG with seed 12345, bytes 31..24, 23..16,
// 15..8 of every state, row by row
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

const char* name(const fklb::ResizeSampling s) { return s == fklb::ResizeSampling::ASYMMETRIC ? "ASYMMETRIC" : "HALF_PIXEL"; }

// An exact image of 1920 x 1080 x 3 interleaved values: value i is num[i] / den
struct Exact {
    std::vector<int64_t> num;
    int64_t den = 1;
};

// Rounding of an exact non-negative value to the nearest integer, ties to even
int roundHalfEven(const int64_t num, const int64_t den) {
    const int64_t q = num / den, twiceRem = 2 * (num % den);
    if (twiceRem > den) return static_cast<int>(q + 1);
    if (twiceRem < den) return static_cast<int>(q);
    return static_cast<int>(q + (q & 1));
}

std::vector<unsigned char> rounded(const Exact& e) {
    std::vector<unsigned char> out(e.num.size());
    for (size_t i = 0; i < e.num.size(); ++i) out[i] = static_cast<unsigned char>(std::min(255, roundHalfEven(e.num[i], e.den)));
    return out;
}

// The exact value truncated (rounded down): to test whether a library's box truncates
std::vector<unsigned char> truncated(const Exact& e) {
    std::vector<unsigned char> out(e.num.size());
    for (size_t i = 0; i < e.num.size(); ++i) out[i] = static_cast<unsigned char>(std::min<int64_t>(255, e.num[i] / e.den));
    return out;
}

// The source positions of one axis of the bilinear resize, exactly: destination pixel d reads between source pixels i1[d] and i2[d] with the
// weights (den - w[d]) / den and w[d] / den. ASYMMETRIC: position d * src / dst; HALF_PIXEL: (d + 0.5) * src / dst - 0.5. Both clamp a position
// before the first pixel to the first pixel and the second neighbour to the last pixel (as common/resize_reference.h does).
struct Axis {
    std::vector<int> i1, i2;
    std::vector<int64_t> w;
    int64_t den = 1;
};

Axis axis(const int src, const int dst, const fklb::ResizeSampling sampling) {
    const int64_t g = std::gcd(src, dst);
    const int64_t s = src / g, t = dst / g;   // the scale src / dst = s / t
    Axis a;
    a.den = sampling == fklb::ResizeSampling::ASYMMETRIC ? t : 2 * t;
    for (int d = 0; d < dst; ++d) {
        // the position times den: ASYMMETRIC d * s; HALF_PIXEL (2d + 1) * s - t
        int64_t p = sampling == fklb::ResizeSampling::ASYMMETRIC ? d * s : (2 * static_cast<int64_t>(d) + 1) * s - t;
        p = std::max<int64_t>(p, 0);
        int i1 = static_cast<int>(p / a.den);
        int64_t w = p % a.den;
        if (i1 >= src - 1) {
            i1 = src - 1;
            w = 0;
        }
        a.i1.push_back(i1);
        a.i2.push_back(std::min(i1 + 1, src - 1));
        a.w.push_back(w);
    }
    return a;
}

// The exact bilinear resize of the crop of the frame, in integers: every value over den_x * den_y
Exact exactResize(const std::vector<unsigned char>& frame, const fklb::ResizeSampling sampling) {
    const Axis ax = axis(CROP_W, OUT_W, sampling), ay = axis(CROP_H, OUT_H, sampling);
    const auto at = [&](const int x, const int y, const int c) {
        return static_cast<int64_t>(frame[(static_cast<size_t>(CROP_Y + y) * SRC_W + CROP_X + x) * 3 + c]);
    };
    Exact e;
    e.den = ax.den * ay.den;
    e.num.resize(OUT_VALUES);
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                const int64_t top = at(ax.i1[x], ay.i1[y], c) * (ax.den - ax.w[x]) + at(ax.i2[x], ay.i1[y], c) * ax.w[x];
                const int64_t bottom = at(ax.i1[x], ay.i2[y], c) * (ax.den - ax.w[x]) + at(ax.i2[x], ay.i2[y], c) * ax.w[x];
                e.num[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] = top * (ay.den - ay.w[y]) + bottom * ay.w[y];
            }
        }
    }
    return e;
}

// The exact 3 x 3 box of a 1920 x 1080 uchar3 image, anchor at the centre, border REPLICATE: the sum of the 9 values over 9
Exact exactBox(const std::vector<unsigned char>& image) {
    Exact e;
    e.den = BOX_W * BOX_H;
    e.num.resize(OUT_VALUES);
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                int64_t sum = 0;
                for (int ky = 0; ky < BOX_H; ++ky) {
                    for (int kx = 0; kx < BOX_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), OUT_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), OUT_H - 1);
                        sum += image[(static_cast<size_t>(sy) * OUT_W + sx) * 3 + c];
                    }
                }
                e.num[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] = sum;
            }
        }
    }
    return e;
}

struct Difference {
    size_t values = 0;          // compared values (OUT_VALUES)
    size_t valuesDiffer = 0;    // values that differ
    size_t pixelsDiffer = 0;    // pixels with at least one differing channel
    int maxLevels = 0;          // largest |a - b| in levels
    size_t above = 0;           // differing values with a > b
    size_t below = 0;           // differing values with a < b
    size_t atTies = 0;          // differing values whose unrounded exact value is a tie (k + 0.5); only with an exact value
    double maxToHalf = -1.;     // over the other differing values: the largest distance of the unrounded exact value to the nearest half level;
                                // -1 when there is none
    size_t ties = 0;            // all values whose unrounded exact value is a tie, differing or not; only with an exact value
    size_t tiesUp = 0;          // of these, the values where a is rounded up (a = the integer part + 1)
};

// a against b; `exact`: the unrounded exact value of the step at every position, or nullptr when the step has none (the pipeline)
Difference compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const Exact* exact) {
    if (a.size() != b.size()) throw std::runtime_error("compare8u: different sizes");
    Difference d;
    d.values = a.size();
    for (size_t p = 0; p < a.size(); p += 3) {
        bool pixel = false;
        for (size_t c = 0; c < 3; ++c) {
            if (exact != nullptr && 2 * (exact->num[p + c] % exact->den) == exact->den) {
                ++d.ties;
                if (a[p + c] == exact->num[p + c] / exact->den + 1) ++d.tiesUp;
            }
            const int diff = static_cast<int>(a[p + c]) - static_cast<int>(b[p + c]);
            if (diff == 0) continue;
            ++d.valuesDiffer;
            pixel = true;
            d.maxLevels = std::max(d.maxLevels, std::abs(diff));
            ++(diff > 0 ? d.above : d.below);
            if (exact != nullptr) {
                const int64_t rem = exact->num[p + c] % exact->den;
                if (2 * rem == exact->den) {
                    ++d.atTies;
                } else {
                    d.maxToHalf = std::max(d.maxToHalf, std::abs(static_cast<double>(rem) / static_cast<double>(exact->den) - 0.5));
                }
            }
        }
        if (pixel) ++d.pixelsDiffer;
    }
    return d;
}

// a 1920 x 1080 uchar3 image of the device, rows `pitch` bytes apart, as tightly packed host bytes
std::vector<unsigned char> download8u(const void* data, const size_t pitch) {
    std::vector<unsigned char> host(OUT_VALUES);
    FKLB_CUDA_CHECK(cudaMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3, data, pitch, static_cast<size_t>(OUT_W) * 3, OUT_H, cudaMemcpyDeviceToHost));
    return host;
}

// The GPU spin of the kernel-only interval, the one of E1 (handwritten-cuda/e1_common.h) and of e18_stencil.cu: one thread waits on the GPU's
// global timer and touches no memory, so the L2 stays as the previous call left it. Enqueued before the start event (the beforeLaunch hook of
// fklb::timePreparedLaunch), it keeps the GPU busy while the host records the start event and enqueues the whole call: the call is already queued
// when the GPU reaches the start event, and the interval holds its kernels, not the host's dispatch of them. It has to last longer than the slowest
// dispatch of one call.
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
using ResizeAsym = fk::Resize<fk::InterpolationType::INTER_LINEAR>;
using ResizeHalf = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>;
using Multiply = fk::Mul<float3, float3, float3, fk::UnaryType>;
using Accumulate = fk::Add<float3, float3, float3, fk::UnaryType>;
using MultiplyIOp = decltype(Multiply::build());
using AccumulateIOp = decltype(Accumulate::build());

// The arguments of one fk::executeLinearFilter, as the kernel-only interval of the per-call FKL variants prepares them (built in place, then launched)
template <typename Reads, typename Write>
struct FilterArguments {
    Reads reads;
    MultiplyIOp multiply;
    AccumulateIOp accumulate;
    Write output;
    Details details;
};

// One implementation: its whole call, its kernel-only measurement and the download of its output
struct Implementation {
    std::string version;
    fklb::ResizeSampling sampling;
    bool truncating;   // its box truncates (the truncating FKL variants, OpenCV-CUDA and NPP) instead of rounding to the nearest
    std::function<void()> call;
    std::function<void(fklb::Timing& gpu, fklb::Timing& host)> kernelOnly;
    std::function<std::vector<unsigned char>()> download;
};

struct Output {
    std::string version;
    fklb::ResizeSampling sampling;
    bool truncating;
    std::vector<unsigned char> image;
};
} // namespace

int launch() {
    const bool profile = envIs1("E18_PROFILE");
    const bool timing = !profile && !envIs1("E18_NO_TIMING");
    const int profileCalls = std::getenv("E18_PROFILE_CALLS") != nullptr ? std::atoi(std::getenv("E18_PROFILE_CALLS")) : 10;
    std::cout << "E18 box: "
              << (profile ? "profile mode (a warm-up call, then E18_PROFILE_CALLS calls per implementation in its NVTX range; nothing compared, nothing timed)"
                          : (timing ? "checks, then the timing" : "checks only (E18_NO_TIMING=1)"))
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

    // ---- FKL: its own copy of the frame (fk::Ptr2D) and one output (fk::Ptr2D<uchar3>) per variant
    fk::Ptr2D<uchar3> dInput(SRC_W, SRC_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(dInput.ptr().data, dInput.ptr().dims.pitch, noise.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    fk::Ptr2D<uchar3> outAsym(OUT_W, OUT_H), outHalf(OUT_W, OUT_H), outAsymOnce(OUT_W, OUT_H), outHalfOnce(OUT_W, OUT_H);
    fk::Ptr2D<uchar3> outTruncAsym(OUT_W, OUT_H), outTruncHalf(OUT_W, OUT_H), outTruncAsymOnce(OUT_W, OUT_H), outTruncHalfOnce(OUT_W, OUT_H);

    // fkl_box_asym: every call builds the crop view, every IOp and the details, and launches
    const auto callAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsym));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto coefficient = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_box_asym: the details are not valid");
        }
    };
    // its kernel-only measurement: the same building as the preparation (host), the launch alone behind the spin (GPU)
    const auto prepareAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                              .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n)));
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsym));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(output)>>{reads, Multiply::build(), Accumulate::build(), output,
                                                                                            details};
    };
    // fkl_box_halfpixel: the same with ResizeSampling::HALF_PIXEL
    const auto callHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalf));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto coefficient = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_box_halfpixel: the details are not valid");
        }
    };
    const auto prepareHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                              .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n)));
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalf));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(output)>>{reads, Multiply::build(), Accumulate::build(), output,
                                                                                            details};
    };
    // fkl_trunc_asym: the truncating twin of fkl_box_asym, the box that OpenCV-CUDA and NPP compute. The same image Read IOp; the coefficient
    // Read IOp gives the weight 1 (SCALE) in every channel, so the accumulator holds the exact sum of the 9 values (an integer of at most 2,295,
    // exact in float); the write divides it by 9 (fk::Div on float3, a correctly rounded division: the target is built without fast math) and
    // truncates it to uchar3 (fk::Cast<float3, uchar3>). A correctly rounded sum / 9 is k exactly when the sum is 9k and lies in (k, k + 1)
    // otherwise, so the cast gives floor(sum / 9).
    const auto callTruncAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto weights = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncAsym));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_trunc_asym: the details are not valid");
        }
    };
    const auto prepareTruncAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                              .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE)));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncAsym));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(output)>>{reads, Multiply::build(), Accumulate::build(), output,
                                                                                            details};
    };
    // fkl_trunc_halfpixel: the same with ResizeSampling::HALF_PIXEL
    const auto callTruncHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto weights = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncHalf));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_trunc_halfpixel: the details are not valid");
        }
    };
    const auto prepareTruncHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                              .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE)));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncHalf));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(output)>>{reads, Multiply::build(), Accumulate::build(), output,
                                                                                            details};
    };
    const auto launchPrepared = [&](const auto& a) {
        if (!fk::executeLinearFilter(a.details, a.reads, a.multiply, a.accumulate, a.output, fkStream)) {
            throw std::runtime_error("FKL: the details are not valid");
        }
    };
    // the build-once variants: valid only while the crop (its position and size) does not change
    const fk::Ptr2D<uchar3> cropOnce = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
    const Details detailsOnce{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
    const float nOnce = static_cast<float>(detailsOnce.kernelWidth * detailsOnce.kernelHeight);
    const auto coefficientOnce = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(nOnce, nOnce, nOnce));
    const auto readsAsymOnce = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(cropOnce)
                                                  .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                                                  .then(fk::SaturateCast<float3, uchar3>::build())
                                                  .then(fk::Cast<uchar3, float3>::build()),
                                              coefficientOnce);
    const auto readsHalfOnce = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(cropOnce)
                                                  .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                                                  .then(fk::SaturateCast<float3, uchar3>::build())
                                                  .then(fk::Cast<uchar3, float3>::build()),
                                              coefficientOnce);
    const auto multiplyOnce = Multiply::build();
    const auto accumulateOnce = Accumulate::build();
    const auto outputAsymOnce = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsymOnce));
    const auto outputHalfOnce = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outHalfOnce));
    const auto weightsOnce = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE));
    const auto readsTruncAsymOnce = fk::make_tuple(fk::get<0>(readsAsymOnce), weightsOnce);
    const auto readsTruncHalfOnce = fk::make_tuple(fk::get<0>(readsHalfOnce), weightsOnce);
    const auto outputTruncAsymOnce = fk::Div<float3>::build(make_float3(nOnce, nOnce, nOnce))
                                         .then(fk::Cast<float3, uchar3>::build())
                                         .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncAsymOnce));
    const auto outputTruncHalfOnce = fk::Div<float3>::build(make_float3(nOnce, nOnce, nOnce))
                                         .then(fk::Cast<float3, uchar3>::build())
                                         .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outTruncHalfOnce));
    const auto callAsymOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsAsymOnce, multiplyOnce, accumulateOnce, outputAsymOnce, fkStream)) {
            throw std::runtime_error("fkl_asym_once: the details are not valid");
        }
    };
    const auto callHalfOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsHalfOnce, multiplyOnce, accumulateOnce, outputHalfOnce, fkStream)) {
            throw std::runtime_error("fkl_half_once: the details are not valid");
        }
    };
    const auto callTruncAsymOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsTruncAsymOnce, multiplyOnce, accumulateOnce, outputTruncAsymOnce, fkStream)) {
            throw std::runtime_error("fkl_trunc_asym_once: the details are not valid");
        }
    };
    const auto callTruncHalfOnce = [&] {
        if (!fk::executeLinearFilter(detailsOnce, readsTruncHalfOnce, multiplyOnce, accumulateOnce, outputTruncHalfOnce, fkStream)) {
            throw std::runtime_error("fkl_trunc_halfpixel_once: the details are not valid");
        }
    };
    const auto nothingToPrepare = [] { return 0; };

    // ---- The fused kernels as the CUDA runtime reports them, and the sizes of their five arguments (no kernel runs). The kernel is the
    // instantiation that fk::executeLinearFilter launches for these argument types (linear_filter.h, linearFilterDPPKernel).
    {
        int device = 0;
        FKLB_CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props;
        FKLB_CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        fklb::CsvWriter kernelCsv("e18_box_kernel.csv", "E18 box: the fused kernels (cudaFuncGetAttributes, cudaOccupancyMaxActiveBlocksPerMultiprocessor)",
                                  {"version", "registers", "shared_static_bytes", "local_bytes", "const_user_bytes", "block_threads", "blocks_per_sm",
                                   "warps_per_sm", "max_warps_per_sm", "theoretical_occupancy_pct", "arg_details_bytes", "arg_reads_bytes",
                                   "arg_image_bytes", "arg_coefficient_bytes", "arg_multiply_bytes", "arg_accumulate_bytes", "arg_output_bytes",
                                   "arg_total_bytes"});
        const int blockThreads = Details::TILE_WIDTH * Details::TILE_HEIGHT;
        const auto report = [&](const char* version, const auto& a) {
            using A = std::decay_t<decltype(a)>;
            const auto kernel = fk::linearFilterDPPKernel<fk::ParArch::GPU_NVIDIA, Details, decltype(A::reads), decltype(A::multiply),
                                                          decltype(A::accumulate), decltype(A::output)>;
            cudaFuncAttributes attr;
            FKLB_CUDA_CHECK(cudaFuncGetAttributes(&attr, kernel));
            int blocks = 0;
            FKLB_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel, blockThreads, 0));
            const int warps = blocks * ((blockThreads + props.warpSize - 1) / props.warpSize);
            const int maxWarps = props.maxThreadsPerMultiProcessor / props.warpSize;
            const size_t total = sizeof(a.details) + sizeof(a.reads) + sizeof(a.multiply) + sizeof(a.accumulate) + sizeof(a.output);
            kernelCsv.row(version, attr.numRegs, attr.sharedSizeBytes, attr.localSizeBytes, attr.constSizeBytes, blockThreads, blocks, warps, maxWarps,
                          100.0 * warps / maxWarps, sizeof(a.details), sizeof(a.reads), sizeof(fk::get<0>(a.reads)), sizeof(fk::get<1>(a.reads)),
                          sizeof(a.multiply), sizeof(a.accumulate), sizeof(a.output), total);
            std::cout << "fused kernel " << version << ": " << attr.numRegs << " registers, " << attr.sharedSizeBytes << " B static shared, "
                      << attr.localSizeBytes << " B local; " << blocks << " blocks of " << blockThreads << " threads per SM = " << warps << " of "
                      << maxWarps << " warps (" << 100.0 * warps / maxWarps << " %); arguments (bytes): details " << sizeof(a.details) << ", reads "
                      << sizeof(a.reads) << " (image " << sizeof(fk::get<0>(a.reads)) << ", coefficient " << sizeof(fk::get<1>(a.reads))
                      << "), multiply " << sizeof(a.multiply) << ", accumulate " << sizeof(a.accumulate) << ", output " << sizeof(a.output)
                      << ", total " << total << std::endl;
        };
        report("fkl_box_asym", prepareAsym());
        report("fkl_box_halfpixel", prepareHalf());
        report("fkl_trunc_asym", prepareTruncAsym());
        report("fkl_trunc_halfpixel", prepareTruncHalf());
    }

    // ---- OpenCV-CUDA: its own copy of the frame (GpuMat::upload), its own intermediates and output
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    cv::cuda::GpuMat cvSrc;
    cvSrc.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, const_cast<unsigned char*>(noise.data())));
    const cv::Size cvOutSize(OUT_W, OUT_H);
    cv::cuda::GpuMat cvRs(cvOutSize, CV_8UC3), cvRs4(cvOutSize, CV_8UC4), cvBox4(cvOutSize, CV_8UC4), cvOut(cvOutSize, CV_8UC3);
    // the filter object is created once, before the calls, as a user creates it once for a stream of frames
    const cv::Ptr<cv::cuda::Filter> cvBox =
        cv::cuda::createBoxFilter(CV_8UC4, CV_8UC4, cv::Size(BOX_W, BOX_H), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    const auto callOpenCV = [&] {
        const cv::cuda::GpuMat roi = cvSrc(cv::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        cv::cuda::resize(roi, cvRs, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
        cv::cuda::cvtColor(cvRs, cvRs4, cv::COLOR_BGR2BGRA, 0, cvs);
        cvBox->apply(cvRs4, cvBox4, cvs);
        cv::cuda::cvtColor(cvBox4, cvOut, cv::COLOR_BGRA2BGR, 0, cvs);
    };
    // its kernel-only measurement: the ROI as the preparation, the four calls behind the spin
    const auto prepareOpenCV = [&] { return cvSrc(cv::Rect(CROP_X, CROP_Y, CROP_W, CROP_H)); };
    const auto launchOpenCV = [&](const cv::cuda::GpuMat& roi) {
        cv::cuda::resize(roi, cvRs, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
        cv::cuda::cvtColor(cvRs, cvRs4, cv::COLOR_BGR2BGRA, 0, cvs);
        cvBox->apply(cvRs4, cvBox4, cvs);
        cv::cuda::cvtColor(cvBox4, cvOut, cv::COLOR_BGRA2BGR, 0, cvs);
    };

    // ---- NPP: its own copy of the frame (cudaMallocPitch), its own intermediate and output
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
    Npp8u *nppFrame = nullptr, *nppRs = nullptr, *nppOut = nullptr;
    size_t frameStep = 0, rsStep = 0, outStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &frameStep, hostRow, SRC_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, frameStep, noise.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppRs), &rsStep, static_cast<size_t>(OUT_W) * 3, OUT_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppOut), &outStep, static_cast<size_t>(OUT_W) * 3, OUT_H));
    const NppiSize nppCropSize{CROP_W, CROP_H}, nppOutSize{OUT_W, OUT_H}, nppMaskSize{BOX_W, BOX_H};
    const auto callNpp = [&] {
        const Npp8u* roi = nppFrame + static_cast<size_t>(CROP_Y) * frameStep + static_cast<size_t>(CROP_X) * 3;
        FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(roi, static_cast<int>(frameStep), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRs, static_cast<int>(rsStep),
                                             nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
        FKLB_NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(nppRs, static_cast<int>(rsStep), nppOutSize, NppiPoint{0, 0}, nppOut, static_cast<int>(outStep),
                                                      nppOutSize, nppMaskSize, NppiPoint{ANCHOR_X, ANCHOR_Y}, NPP_BORDER_REPLICATE, ctx));
    };
    // its kernel-only measurement: the ROI pointer as the preparation, the two calls behind the spin
    const auto prepareNpp = [&] { return static_cast<const Npp8u*>(nppFrame + static_cast<size_t>(CROP_Y) * frameStep + static_cast<size_t>(CROP_X) * 3); };
    const auto launchNpp = [&](const Npp8u* roi) {
        FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(roi, static_cast<int>(frameStep), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRs, static_cast<int>(rsStep),
                                             nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
        FKLB_NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(nppRs, static_cast<int>(rsStep), nppOutSize, NppiPoint{0, 0}, nppOut, static_cast<int>(outStep),
                                                      nppOutSize, nppMaskSize, NppiPoint{ANCHOR_X, ANCHOR_Y}, NPP_BORDER_REPLICATE, ctx));
    };

    // The order of the checks and of the timing, the same in every run: the per-call versions, then the build-once variants
    const auto makeKernelOnly = [&](const auto& prepare, const auto& launch) {
        return [&, prepare, launch](fklb::Timing& gpu, fklb::Timing& host) { fklb::timePreparedLaunch(cs, prepare, spin, launch, gpu, host); };
    };
    const auto launchAsymOnce = [&](int) { callAsymOnce(); };
    const auto launchHalfOnce = [&](int) { callHalfOnce(); };
    const auto launchTruncAsymOnce = [&](int) { callTruncAsymOnce(); };
    const auto launchTruncHalfOnce = [&](int) { callTruncHalfOnce(); };
    impls.push_back({"fkl_box_asym", fklb::ResizeSampling::ASYMMETRIC, false, callAsym, makeKernelOnly(prepareAsym, launchPrepared),
                     [&] { return download8u(outAsym.ptr().data, outAsym.ptr().dims.pitch); }});
    impls.push_back({"fkl_box_halfpixel", fklb::ResizeSampling::HALF_PIXEL, false, callHalf, makeKernelOnly(prepareHalf, launchPrepared),
                     [&] { return download8u(outHalf.ptr().data, outHalf.ptr().dims.pitch); }});
    impls.push_back({"fkl_trunc_asym", fklb::ResizeSampling::ASYMMETRIC, true, callTruncAsym, makeKernelOnly(prepareTruncAsym, launchPrepared),
                     [&] { return download8u(outTruncAsym.ptr().data, outTruncAsym.ptr().dims.pitch); }});
    impls.push_back({"fkl_trunc_halfpixel", fklb::ResizeSampling::HALF_PIXEL, true, callTruncHalf, makeKernelOnly(prepareTruncHalf, launchPrepared),
                     [&] { return download8u(outTruncHalf.ptr().data, outTruncHalf.ptr().dims.pitch); }});
    impls.push_back({"opencv_cuda", fklb::ResizeSampling::ASYMMETRIC, true, callOpenCV, makeKernelOnly(prepareOpenCV, launchOpenCV),
                     [&] { return download8u(cvOut.data, cvOut.step); }});
    impls.push_back({"npp", fklb::ResizeSampling::HALF_PIXEL, true, callNpp, makeKernelOnly(prepareNpp, launchNpp),
                     [&] { return download8u(nppOut, outStep); }});
    impls.push_back({"fkl_asym_once", fklb::ResizeSampling::ASYMMETRIC, false, callAsymOnce, makeKernelOnly(nothingToPrepare, launchAsymOnce),
                     [&] { return download8u(outAsymOnce.ptr().data, outAsymOnce.ptr().dims.pitch); }});
    impls.push_back({"fkl_half_once", fklb::ResizeSampling::HALF_PIXEL, false, callHalfOnce, makeKernelOnly(nothingToPrepare, launchHalfOnce),
                     [&] { return download8u(outHalfOnce.ptr().data, outHalfOnce.ptr().dims.pitch); }});
    impls.push_back({"fkl_trunc_asym_once", fklb::ResizeSampling::ASYMMETRIC, true, callTruncAsymOnce, makeKernelOnly(nothingToPrepare, launchTruncAsymOnce),
                     [&] { return download8u(outTruncAsymOnce.ptr().data, outTruncAsymOnce.ptr().dims.pitch); }});
    impls.push_back({"fkl_trunc_halfpixel_once", fklb::ResizeSampling::HALF_PIXEL, true, callTruncHalfOnce,
                     makeKernelOnly(nothingToPrepare, launchTruncHalfOnce),
                     [&] { return download8u(outTruncHalfOnce.ptr().data, outTruncHalfOnce.ptr().dims.pitch); }});

    // ---- The profile mode: a warm-up call outside any range, then E18_PROFILE_CALLS calls inside the NVTX range named after the implementation
    if (profile) {
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            impl.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(impl.version.c_str());
            for (int i = 0; i < profileCalls; ++i) impl.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePop();
        }
        std::cout << "profile mode: " << profileCalls << " calls per selected implementation in its NVTX range" << std::endl;
    }

    bool allOk = true;
    if (!profile) {
        // ---- One call of every selected implementation, its output and, for the libraries, their resized image (the intermediate of the call)
        std::vector<Output> outputs;
        std::vector<unsigned char> resizedOpenCV, resizedNpp;
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            impl.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            outputs.push_back({impl.version, impl.sampling, impl.truncating, impl.download()});
            if (impl.version == "opencv_cuda") resizedOpenCV = download8u(cvRs.data, cvRs.step);
            if (impl.version == "npp") resizedNpp = download8u(nppRs, rsStep);
        }

        // ---- The exact references, and the double references against them
        const Exact exactRsAsym = exactResize(noise, fklb::ResizeSampling::ASYMMETRIC);
        const Exact exactRsHalf = exactResize(noise, fklb::ResizeSampling::HALF_PIXEL);
        const std::vector<unsigned char> refRsAsym = rounded(exactRsAsym), refRsHalf = rounded(exactRsHalf);
        const Exact boxOfRefAsym = exactBox(refRsAsym), boxOfRefHalf = exactBox(refRsHalf);
        const std::vector<unsigned char> refAsym = rounded(boxOfRefAsym), refHalf = rounded(boxOfRefHalf);
        const std::vector<unsigned char> refTruncAsym = truncated(boxOfRefAsym), refTruncHalf = truncated(boxOfRefHalf);
        fklb::CsvWriter checks("e18_box_checks.csv", "E18 box: the exact references", {"check", "sampling", "value", "criterion", "ok"});
        for (const fklb::ResizeSampling sampling : {fklb::ResizeSampling::ASYMMETRIC, fklb::ResizeSampling::HALF_PIXEL}) {
            const Exact& e = sampling == fklb::ResizeSampling::ASYMMETRIC ? exactRsAsym : exactRsHalf;
            std::vector<double> dbl(OUT_VALUES);
            const unsigned char* crop = noise.data() + (static_cast<size_t>(CROP_Y) * SRC_W + CROP_X) * 3;
            fklb::resizeBilinear(crop, hostRow, CROP_W, CROP_H, 3, OUT_W, OUT_H, sampling, dbl.data());
            size_t ties = 0, roundingDiffers = 0;
            double maxErr = 0.;
            for (size_t i = 0; i < OUT_VALUES; ++i) {
                if (2 * (e.num[i] % e.den) == e.den) ++ties;
                maxErr = std::max(maxErr, std::abs(dbl[i] - static_cast<double>(e.num[i]) / static_cast<double>(e.den)));
                if (static_cast<int>(std::nearbyint(dbl[i])) != roundHalfEven(e.num[i], e.den)) ++roundingDiffers;
            }
            checks.row("the exact resize: denominator of every value", name(sampling), e.den, "informational", 1);
            checks.row("the exact resize: values that are exact ties (k + 0.5)", name(sampling), ties, "informational", 1);
            checks.row("the double resize (common/resize_reference.h) against the exact resize: max |difference|", name(sampling), maxErr, "informational", 1);
            checks.row("values where the double resize rounded to nearest (std::nearbyint) differs from the exact resize rounded with ties to even",
                       name(sampling), roundingDiffers, "informational", 1);
            std::cout << "exact resize " << name(sampling) << ": denominator " << e.den << ", " << ties << " exact ties; double resize within " << maxErr
                      << ", its rounding differs from the exact one at " << roundingDiffers << " values" << std::endl;
        }

        fklb::CsvWriter agreement("e18_box_agreement.csv", "E18 box: agreement of the uchar outputs",
                                  {"step", "version", "sampling", "against", "values", "values_differing", "pixels_differing", "max_diff_levels",
                                   "values_above", "values_below", "identical", "differing_at_exact_ties", "max_distance_to_half_level_off_ties",
                                   "exact_ties", "exact_ties_where_version_rounds_up"});
        const auto row = [&](const char* step, const std::string& version, const std::string& sampling, const std::string& against,
                             const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const Exact* exact) {
            const Difference d = compare8u(a, b, exact);
            if (exact != nullptr) {
                agreement.row(step, version, sampling, against, d.values, d.valuesDiffer, d.pixelsDiffer, d.maxLevels, d.above, d.below,
                              d.valuesDiffer == 0 ? 1 : 0, d.atTies, d.maxToHalf, d.ties, d.tiesUp);
            } else {
                agreement.row(step, version, sampling, against, d.values, d.valuesDiffer, d.pixelsDiffer, d.maxLevels, d.above, d.below,
                              d.valuesDiffer == 0 ? 1 : 0, "", "", "", "");
            }
            std::cout << step << ": " << version << " vs " << against << ": " << d.valuesDiffer << " values / " << d.pixelsDiffer << " pixels differ, max "
                      << d.maxLevels << " levels (" << d.above << " above, " << d.below << " below)";
            if (exact != nullptr) {
                std::cout << "; " << d.atTies << " at exact ties, the others within " << d.maxToHalf << " of a half level; " << d.tiesUp << " of the "
                          << d.ties << " exact ties rounded up";
            }
            std::cout << std::endl;
            return d;
        };

        // ---- (a) The resize alone. FKL's read chain of each convention, through fk::executeLinearFilter with the identity weights
        fk::Ptr2D<float3> dIdentity(BOX_W, BOX_H);
        {
            std::vector<float3> id(BOX_W * BOX_H);
            for (int ky = 0; ky < BOX_H; ++ky) {
                for (int kx = 0; kx < BOX_W; ++kx) {
                    const float one = (kx == ANCHOR_X && ky == ANCHOR_Y) ? 1.f : 0.f;
                    id[ky * BOX_W + kx] = make_float3(one, one, one);
                }
            }
            FKLB_CUDA_CHECK(cudaMemcpy2D(dIdentity.ptr().data, dIdentity.ptr().dims.pitch, id.data(), BOX_W * sizeof(float3), BOX_W * sizeof(float3), BOX_H,
                                         cudaMemcpyHostToDevice));
        }
        std::vector<unsigned char> chainAsym, chainHalf;
        {
            fk::Ptr2D<uchar3> dChainAsym(OUT_W, OUT_H), dChainHalf(OUT_W, OUT_H);
            const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
            const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
            const auto imageAsym = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                       .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                                       .then(fk::SaturateCast<float3, uchar3>::build())
                                       .then(fk::Cast<uchar3, float3>::build());
            const auto imageHalf = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                       .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                                       .then(fk::SaturateCast<float3, uchar3>::build())
                                       .then(fk::Cast<uchar3, float3>::build());
            const auto weights = fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity);
            const bool okAsym = fk::executeLinearFilter(details, fk::make_tuple(imageAsym, weights), Multiply::build(), Accumulate::build(),
                                                        fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dChainAsym)),
                                                        fkStream);
            const bool okHalf = fk::executeLinearFilter(details, fk::make_tuple(imageHalf, weights), Multiply::build(), Accumulate::build(),
                                                        fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dChainHalf)),
                                                        fkStream);
            if (!okAsym || !okHalf) throw std::runtime_error("identity weights: the details are not valid");
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            chainAsym = download8u(dChainAsym.ptr().data, dChainAsym.ptr().dims.pitch);
            chainHalf = download8u(dChainHalf.ptr().data, dChainHalf.ptr().dims.pitch);
        }
        row("resize", "fkl_read_chain_asym", name(fklb::ResizeSampling::ASYMMETRIC), "exact_resize_ASYMMETRIC", chainAsym, refRsAsym, &exactRsAsym);
        row("resize", "fkl_read_chain_halfpixel", name(fklb::ResizeSampling::HALF_PIXEL), "exact_resize_HALF_PIXEL", chainHalf, refRsHalf, &exactRsHalf);
        if (!resizedOpenCV.empty()) {
            row("resize", "opencv_cuda", name(fklb::ResizeSampling::ASYMMETRIC), "exact_resize_ASYMMETRIC", resizedOpenCV, refRsAsym, &exactRsAsym);
            row("resize", "fkl_read_chain_asym", name(fklb::ResizeSampling::ASYMMETRIC), "opencv_cuda_resize", chainAsym, resizedOpenCV, &exactRsAsym);
        }
        if (!resizedNpp.empty()) {
            row("resize", "npp", name(fklb::ResizeSampling::HALF_PIXEL), "exact_resize_HALF_PIXEL", resizedNpp, refRsHalf, &exactRsHalf);
            row("resize", "fkl_read_chain_halfpixel", name(fklb::ResizeSampling::HALF_PIXEL), "npp_resize", chainHalf, resizedNpp, &exactRsHalf);
        }

        // ---- (b) The box alone, on one and the same uchar3 image: 1920 x 1080 values of E4's generator. Each library has its own copy
        {
            const std::vector<unsigned char> boxImage = seededNoise(OUT_W, OUT_H);
            const size_t boxRow = static_cast<size_t>(OUT_W) * 3;
            const Exact exactBoxImage = exactBox(boxImage);
            const std::vector<unsigned char> refBox = rounded(exactBoxImage);
            // FKL: the same call as the box variants, with a plain read of the image (and the cast to float3) as its image Read IOp
            fk::Ptr2D<uchar3> dBoxIn(OUT_W, OUT_H), dBoxOut(OUT_W, OUT_H);
            FKLB_CUDA_CHECK(cudaMemcpy2D(dBoxIn.ptr().data, dBoxIn.ptr().dims.pitch, boxImage.data(), boxRow, boxRow, OUT_H, cudaMemcpyHostToDevice));
            {
                const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
                const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
                const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dBoxIn).then(fk::Cast<uchar3, float3>::build());
                const auto coefficient = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n));
                const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dBoxOut));
                if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), Multiply::build(), Accumulate::build(), output, fkStream)) {
                    throw std::runtime_error("box alone: the details are not valid");
                }
            }
            // FKL, truncating: the call of the truncating variants with the same plain read
            fk::Ptr2D<uchar3> dBoxTruncOut(OUT_W, OUT_H);
            {
                const Details details{BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
                const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
                const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dBoxIn).then(fk::Cast<uchar3, float3>::build());
                const auto weights = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE));
                const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                        .then(fk::Cast<float3, uchar3>::build())
                                        .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dBoxTruncOut));
                if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), Multiply::build(), Accumulate::build(), output, fkStream)) {
                    throw std::runtime_error("box alone, truncating: the details are not valid");
                }
            }
            // OpenCV: the path of opencv_cuda after the resize, with the same filter object
            cv::cuda::GpuMat cvBoxIn, cvBoxIn4(cvOutSize, CV_8UC4), cvBoxOut4(cvOutSize, CV_8UC4), cvBoxOut(cvOutSize, CV_8UC3);
            cvBoxIn.upload(cv::Mat(OUT_H, OUT_W, CV_8UC3, const_cast<unsigned char*>(boxImage.data())));
            cv::cuda::cvtColor(cvBoxIn, cvBoxIn4, cv::COLOR_BGR2BGRA, 0, cvs);
            cvBox->apply(cvBoxIn4, cvBoxOut4, cvs);
            cv::cuda::cvtColor(cvBoxOut4, cvBoxOut, cv::COLOR_BGRA2BGR, 0, cvs);
            // NPP: nppiFilterBoxBorder_8u_C3R_Ctx as in npp
            Npp8u *nppBoxIn = nullptr, *nppBoxOut = nullptr;
            size_t boxInStep = 0, boxOutStep = 0;
            FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppBoxIn), &boxInStep, boxRow, OUT_H));
            FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppBoxOut), &boxOutStep, boxRow, OUT_H));
            FKLB_CUDA_CHECK(cudaMemcpy2D(nppBoxIn, boxInStep, boxImage.data(), boxRow, boxRow, OUT_H, cudaMemcpyHostToDevice));
            FKLB_NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(nppBoxIn, static_cast<int>(boxInStep), nppOutSize, NppiPoint{0, 0}, nppBoxOut,
                                                          static_cast<int>(boxOutStep), nppOutSize, nppMaskSize, NppiPoint{ANCHOR_X, ANCHOR_Y},
                                                          NPP_BORDER_REPLICATE, ctx));
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            const Difference dFkl = row("box", "fkl_box", "none", "exact_box", download8u(dBoxOut.ptr().data, dBoxOut.ptr().dims.pitch),
                                        refBox, &exactBoxImage);
            const std::vector<unsigned char> cvBoxHost = download8u(cvBoxOut.data, cvBoxOut.step), nppBoxHost = download8u(nppBoxOut, boxOutStep);
            row("box", "opencv_cuda", "none", "exact_box", cvBoxHost, refBox, &exactBoxImage);
            row("box", "npp", "none", "exact_box", nppBoxHost, refBox, &exactBoxImage);
            // the libraries' 8-bit box against the exact box truncated (sum / 9 rounded down)
            const std::vector<unsigned char> truncBox = truncated(exactBoxImage);
            row("box", "opencv_cuda", "none", "exact_box_truncated", cvBoxHost, truncBox, &exactBoxImage);
            row("box", "npp", "none", "exact_box_truncated", nppBoxHost, truncBox, &exactBoxImage);
            const Difference dTrunc = row("box", "fkl_trunc", "none", "exact_box_truncated",
                                          download8u(dBoxTruncOut.ptr().data, dBoxTruncOut.ptr().dims.pitch), truncBox, &exactBoxImage);
            allOk = allOk && dTrunc.valuesDiffer == 0;
            FKLB_CUDA_CHECK(cudaFree(nppBoxIn));
            FKLB_CUDA_CHECK(cudaFree(nppBoxOut));
            // FKL's box computes in float with the coefficient 1/9 in float; its error is far below the 1/18 of a level by which sum / 9 is away
            // from every half level, so it must equal the exact box
            allOk = allOk && dFkl.valuesDiffer == 0;
        }

        // ---- (c) The whole pipeline, and (d) the build-once variants against their per-call versions
        const auto find = [&](const std::string& v) -> const Output* {
            for (const Output& o : outputs) {
                if (o.version == v) return &o;
            }
            return nullptr;
        };
        for (const Output& o : outputs) {
            const bool asym = o.sampling == fklb::ResizeSampling::ASYMMETRIC;
            const bool fkl = o.version.rfind("fkl_", 0) == 0;
            // the exact pipeline of its convention, with the box rounded to the nearest (every implementation) and truncated (the truncating ones)
            row("pipeline", o.version, name(o.sampling), std::string("exact_pipeline_") + name(o.sampling), o.image, asym ? refAsym : refHalf, nullptr);
            if (o.truncating) {
                row("pipeline", o.version, name(o.sampling), std::string("exact_pipeline_truncated_") + name(o.sampling), o.image,
                    asym ? refTruncAsym : refTruncHalf, nullptr);
            }
            // the exact box (rounded or truncated, as the implementation's box) of the implementation's own resize: FKL's read chain of the
            // convention, or the library's intermediate. 0 differences: every difference from the exact pipeline comes from the resize
            const std::vector<unsigned char>& ownResize = fkl ? (asym ? chainAsym : chainHalf) : (asym ? resizedOpenCV : resizedNpp);
            const Exact ownBox = exactBox(ownResize);
            const Difference d = o.truncating
                                     ? row("pipeline", o.version, name(o.sampling), "exact_box_truncated_of_own_resize", o.image, truncated(ownBox), nullptr)
                                     : row("pipeline", o.version, name(o.sampling), "exact_box_of_own_resize", o.image, rounded(ownBox), nullptr);
            if (fkl) allOk = allOk && d.valuesDiffer == 0;
        }
        const std::pair<const char*, const char*> pairs[] = {
            {"fkl_box_asym", "opencv_cuda"},          {"fkl_box_halfpixel", "npp"},         {"fkl_trunc_asym", "opencv_cuda"},
            {"fkl_trunc_halfpixel", "npp"},           {"fkl_asym_once", "opencv_cuda"},     {"fkl_half_once", "npp"},
            {"fkl_trunc_asym_once", "opencv_cuda"},   {"fkl_trunc_halfpixel_once", "npp"},  {"fkl_asym_once", "fkl_box_asym"},
            {"fkl_half_once", "fkl_box_halfpixel"},   {"fkl_trunc_asym_once", "fkl_trunc_asym"}, {"fkl_trunc_halfpixel_once", "fkl_trunc_halfpixel"}};
        for (const auto& p : pairs) {
            const Output* a = find(p.first);
            const Output* b = find(p.second);
            if (a == nullptr || b == nullptr) continue;
            const bool once = std::string(p.second).rfind("fkl_", 0) == 0;
            const Difference d = row(once ? "once" : "pipeline", a->version, name(a->sampling), p.second, a->image, b->image, nullptr);
            // a build-once variant launches the kernel of its per-call version with the same arguments: its output must be the same
            if (once) allOk = allOk && d.valuesDiffer == 0;
        }
        std::cout << (allOk ? "E18 box checks passed" : "E18 box CHECK FAILED") << std::endl;
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
        fklb::CsvWriter t("e18_box.csv", "E18 box: whole-call and kernel-only times",
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
    FKLB_CUDA_CHECK(cudaFree(nppRs));
    FKLB_CUDA_CHECK(cudaFree(nppOut));
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
