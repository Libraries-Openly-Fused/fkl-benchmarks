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

// E18 box split (NVIDIA): the E18 box pipeline of e18_box.cu, and each of its two steps alone, in FKL and in NPP. The fused FKL kernel (the resize
// and the box in one fk::executeLinearFilter) and NPP's two calls are repeated with the code of e18_box.cu, and the resize and the box are timed
// alone: NPP's two calls one by one, and FKL's resize in one fk::executeOperations and FKL's box in one fk::executeLinearFilter.
//
// The frame, the crop, the sizes, the NPP stream context and the timing are those of e18_box.cu, whose code this program copies:
//   frame    3840 x 2160 uchar3, the seeded source of E4 (an LCG with seed 12345)
//   crop     2560 x 1440 at (640, 360)
//   resize   2560 x 1440 -> 1920 x 1080, bilinear, uchar in and uchar out
//   box      3 x 3, anchor at the centre, border REPLICATE, uchar in, uchar3 out
// The rows, in this order, the same in every run:
//   fkl_box_asym         e18_box.cu's fkl_box_asym, with its code: one fk::executeLinearFilter, LinearFilterDPPDetails<float3> with its defaults
//                        (a 16 x 8 tile, a largest kernel of 7 x 7). The image Read IOp is the read of the crop view, fk::Resize (INTER_LINEAR,
//                        ResizeSampling::ASYMMETRIC) to float3, fk::SaturateCast<float3, uchar3> and fk::Cast<uchar3, float3>; the coefficient
//                        fk::ConstantFilterRead<float3> with 1 / 9; the write fk::SaturateCast<float3, uchar3>, then the write of the output: the
//                        box rounded to the nearest. It comes first, as in e18_box.cu
//   fkl_trunc_halfpixel  e18_box.cu's fkl_trunc_halfpixel, with its code: the same call with ResizeSampling::HALF_PIXEL, the weight 1
//                        (fk::ConstantFilterRead<float3>), and the write fk::Div<float3> by 9, then fk::Cast<float3, uchar3>: floor(sum / 9), the box
//                        that NPP computes. The fused kernel whose output equals NPP's (e18_box.cu)
//   npp                  e18_box.cu's npp, with its code: nppiResize_8u_C3R_Ctx (NPPI_INTER_LINEAR: HALF_PIXEL; a pointer into NPP's frame and its
//                        step) to NPP's intermediate, then nppiFilterBoxBorder_8u_C3R_Ctx (mask 3 x 3, anchor (1, 1), NPP_BORDER_REPLICATE)
//   npp_resize           npp's first call alone: nppiResize_8u_C3R_Ctx from the crop (the same pointer and step) to a 1920 x 1080 uchar3 image of
//                        its own
//   fkl_resize           FKL's resize alone, one fk::executeOperations (fk::TransformDPP, thread fusion off): the read of the crop view
//                        (Ptr2D::crop2D, the pitch of the frame), fk::Resize (INTER_LINEAR, HALF_PIXEL) to float3, fk::SaturateCast<float3, uchar3>,
//                        then the write of a 1920 x 1080 uchar3 fk::Ptr2D: the image Read IOp of fkl_trunc_halfpixel without the cast back to float3
//   npp_box              npp's second call alone: nppiFilterBoxBorder_8u_C3R_Ctx with npp's arguments (oSrcSize the whole image, offset (0, 0)), on a
//                        1920 x 1080 uchar3 image of its own
//   fkl_box              fkl_trunc_halfpixel's call with the read of a 1920 x 1080 uchar3 fk::Ptr2D, then fk::Cast<uchar3, float3>, as its image
//                        Read IOp; the weights, the write and the details are those of fkl_trunc_halfpixel
// The input of npp_box and fkl_box is one and the same image: the exact HALF_PIXEL resize of the crop, rounded to the nearest with ties to even (the
// reference below), uploaded once before any call, into an NPP buffer (cudaMallocPitch) and into an fk::Ptr2D. Every row writes its own output.
// fkl_resize and fkl_box build their view, their IOps and their details in every call, as e18_box.cu's per-call variants do. The NPP rows use NPP
// buffers and no FKL object; each library crops its own way, FKL with Ptr2D::crop2D, NPP with a pointer into its frame and its step.
//
// The references are those of e18_box.cu, exact: the resize in integers (every value an integer over 9 for ASYMMETRIC, over 36 for HALF_PIXEL),
// rounded to the nearest with ties to even; the box, the integer sum of the 9 values over 9, rounded to the nearest or truncated.
//
// Every run checks first, then times (the timing is skipped when a check fails). The checks (e18_box_split_agreement.csv, every comparison over the
// 1920 x 1080 x 3 = 6,220,800 values, with the columns of e18_box.cu):
//   resize   FKL's read chain of each convention (the image Read IOp of the fused kernels, through fk::executeLinearFilter with the identity weights,
//            the check of e18_box.cu) and npp's intermediate against the exact resize, and against each other. npp_resize against npp's
//            intermediate (equal: it is the same call). fkl_resize against FKL's read chain of HALF_PIXEL (equal: the same read chain), against the
//            exact resize and against npp_resize
//   box      npp_box and fkl_box against the exact box, truncated, of their input (equal: both compute floor(sum / 9)), and against each other
//   pipeline the three repeated rows as in e18_box.cu: against the exact pipeline of their convention, and against the exact box (rounded or
//            truncated, as their own box) of their own resize; fkl_trunc_halfpixel against npp
// e18_box_split_checks.csv: the number of exact ties of each convention's resize, and the double references against the exact ones (e18_box.cu).
//
// Timing (e18_box_split.csv), two intervals per row, in the order of the list above, as in e18_box.cu:
//   whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch: the preparation (FKL: the view, the IOps and the details; NPP's resize: the ROI
//                pointer; npp_box: nothing) is timed on the host; the whole call is enqueued behind one spin, so the GPU interval runs from the end of
//                the spin to the end of its last kernel
//   E18_ONLY=a,b: only these rows. E18_NO_TIMING=1: the checks only. E18_SPIN_US: the length of the spin (default 5000).
//   E18_PROFILE=1: every selected row is called once outside any range (a warm-up), then E18_PROFILE_CALLS times (default 10) inside an NVTX range
//   named after it, after a synchronised setup; nothing is compared or timed.
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

// The source of E4 (opencv-cuda/batched-preprocessing/e4_preprocessing.cu), as in e18_box.cu: an LCG with seed 12345, bytes 31..24, 23..16,
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

// The exact value truncated (rounded down), the 8-bit box of NPP
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

// The GPU spin of the kernel-only interval, the one of e18_box.cu: one thread waits on the GPU's global timer and touches no memory, so the L2
// stays as the previous call left it. Enqueued before the start event (the beforeLaunch hook of fklb::timePreparedLaunch), it keeps the GPU busy
// while the host records the start event and enqueues the whole call: the interval holds the call's kernels, not the host's dispatch of them. It
// has to last longer than the slowest dispatch of one call.
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
using TransformNoTF = fk::TransformDPP<fk::ParArch::GPU_NVIDIA, fk::TF::DISABLED>;   // thread fusion off, as in E4

// The arguments of one fk::executeLinearFilter, as the kernel-only interval of the FKL filter rows prepares them (built in place, then launched)
template <typename Reads, typename Write>
struct FilterArguments {
    Reads reads;
    MultiplyIOp multiply;
    AccumulateIOp accumulate;
    Write output;
    Details details;
};

// The arguments of fkl_resize's fk::executeOperations, as its kernel-only interval prepares them
template <typename Read, typename Write>
struct TransformArguments {
    Read read;
    Write write;
};

// One row: its whole call, its kernel-only measurement and the download of its output
struct Implementation {
    std::string version;
    std::string sampling;   // the resize convention of the row ("ASYMMETRIC", "HALF_PIXEL"), or "none" for the box rows
    std::string step;       // "pipeline" (the repeated rows), "resize" or "box" (the steps alone)
    bool truncating;        // its box truncates (fkl_trunc_halfpixel, npp, the box rows) instead of rounding to the nearest
    std::function<void()> call;
    std::function<void(fklb::Timing& gpu, fklb::Timing& host)> kernelOnly;
    std::function<std::vector<unsigned char>()> download;
};

struct Output {
    std::string version;
    std::string sampling;
    std::string step;
    bool truncating;
    std::vector<unsigned char> image;
};
} // namespace

int launch() {
    const bool profile = envIs1("E18_PROFILE");
    const bool timing = !profile && !envIs1("E18_NO_TIMING");
    const int profileCalls = std::getenv("E18_PROFILE_CALLS") != nullptr ? std::atoi(std::getenv("E18_PROFILE_CALLS")) : 10;
    std::cout << "E18 box split: "
              << (profile ? "profile mode (a warm-up call, then E18_PROFILE_CALLS calls per row in its NVTX range; nothing compared, nothing timed)"
                          : (timing ? "checks, then the timing" : "checks only (E18_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    const size_t outRow = static_cast<size_t>(OUT_W) * 3;
    // The exact resize of each convention, rounded to the nearest with ties to even. The HALF_PIXEL one is the input of npp_box and fkl_box.
    const Exact exactRsAsym = exactResize(noise, fklb::ResizeSampling::ASYMMETRIC);
    const Exact exactRsHalf = exactResize(noise, fklb::ResizeSampling::HALF_PIXEL);
    const std::vector<unsigned char> refRsAsym = rounded(exactRsAsym), refRsHalf = rounded(exactRsHalf);
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };
    // The rows, in the order in which they are checked and timed (the same in every run)
    std::vector<Implementation> impls;

    // ---- FKL: its own copy of the frame (fk::Ptr2D), its own copy of the input of fkl_box (fk::Ptr2D) and one output (fk::Ptr2D<uchar3>) per row
    fk::Ptr2D<uchar3> dInput(SRC_W, SRC_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(dInput.ptr().data, dInput.ptr().dims.pitch, noise.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    fk::Ptr2D<uchar3> outAsym(OUT_W, OUT_H), outTruncHalf(OUT_W, OUT_H), outResize(OUT_W, OUT_H), outBox(OUT_W, OUT_H);
    fk::Ptr2D<uchar3> dBoxIn(OUT_W, OUT_H);   // the input of fkl_box: the exact HALF_PIXEL resize, uploaded once
    FKLB_CUDA_CHECK(cudaMemcpy2D(dBoxIn.ptr().data, dBoxIn.ptr().dims.pitch, refRsHalf.data(), outRow, outRow, OUT_H, cudaMemcpyHostToDevice));

    // fkl_box_asym (e18_box.cu): every call builds the crop view, every IOp and the details, and launches
    const auto callAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeAsym::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outAsym));
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto coefficient = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE) / make_float3(n, n, n));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_box_asym: the details are not valid");
        }
    };
    // its kernel-only measurement: the same building as the preparation (host), the launch alone behind the spin (GPU)
    const auto prepareAsym = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
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
    // fkl_trunc_halfpixel (e18_box.cu): the weight 1 in every channel, so the accumulator holds the exact sum of the 9 values (an integer of at most
    // 2,295, exact in float); the write divides it by 9 (fk::Div on float3, a correctly rounded division: the target is built without fast math)
    // and truncates it to uchar3 (fk::Cast<float3, uchar3>), which gives floor(sum / 9).
    const auto callTruncHalf = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
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
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
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
    // fkl_resize: FKL's resize alone, the image Read IOp of fkl_trunc_halfpixel without the cast back to float3, and the write of a uchar3 image.
    // Every call builds the crop view and the IOps, and launches.
    const auto callResize = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        fk::executeOperations<TransformNoTF>(fkStream,
                                             fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                                 .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                                                 .then(fk::SaturateCast<float3, uchar3>::build()),
                                             fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outResize));
    };
    const auto prepareResize = [&] {
        const fk::Ptr2D<uchar3> crop = dInput.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto read = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                              .then(ResizeHalf::build(fk::Size(OUT_W, OUT_H)))
                              .then(fk::SaturateCast<float3, uchar3>::build());
        const auto write = fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outResize);
        return TransformArguments<std::decay_t<decltype(read)>, std::decay_t<decltype(write)>>{read, write};
    };
    const auto launchResize = [&](const auto& a) { fk::executeOperations<TransformNoTF>(fkStream, a.read, a.write); };
    // fkl_box: fkl_trunc_halfpixel's call on the uploaded 1920 x 1080 image: its image Read IOp is the read of that image and the cast to float3.
    // Every call builds the IOps and the details, and launches.
    const auto callBox = [&] {
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dBoxIn).then(fk::Cast<uchar3, float3>::build());
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto weights = fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outBox));
        if (!fk::executeLinearFilter(details, fk::make_tuple(image, weights), Multiply::build(), Accumulate::build(), output, fkStream)) {
            throw std::runtime_error("fkl_box: the details are not valid");
        }
    };
    const auto prepareBox = [&] {
        const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dBoxIn).then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(make_float3(SCALE, SCALE, SCALE)));
        const auto output = fk::Div<float3>::build(make_float3(n, n, n))
                                .then(fk::Cast<float3, uchar3>::build())
                                .then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outBox));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(output)>>{reads, Multiply::build(), Accumulate::build(), output,
                                                                                            details};
    };

    // ---- NPP: its own copy of the frame (cudaMallocPitch), its own copy of the input of npp_box, its own intermediate and outputs
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
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppRs), &rsStep, outRow, OUT_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppOut), &outStep, outRow, OUT_H));
    // npp_resize's output, npp_box's input (the exact HALF_PIXEL resize, uploaded once) and npp_box's output
    Npp8u *nppRsAlone = nullptr, *nppBoxIn = nullptr, *nppBoxOut = nullptr;
    size_t rsAloneStep = 0, boxInStep = 0, boxOutStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppRsAlone), &rsAloneStep, outRow, OUT_H));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppBoxIn), &boxInStep, outRow, OUT_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(nppBoxIn, boxInStep, refRsHalf.data(), outRow, outRow, OUT_H, cudaMemcpyHostToDevice));
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppBoxOut), &boxOutStep, outRow, OUT_H));
    const NppiSize nppCropSize{CROP_W, CROP_H}, nppOutSize{OUT_W, OUT_H}, nppMaskSize{BOX_W, BOX_H};
    // npp (e18_box.cu)
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
    // npp_resize: npp's first call alone, to its own image
    const auto callNppResize = [&] {
        const Npp8u* roi = nppFrame + static_cast<size_t>(CROP_Y) * frameStep + static_cast<size_t>(CROP_X) * 3;
        FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(roi, static_cast<int>(frameStep), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRsAlone,
                                             static_cast<int>(rsAloneStep), nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
    };
    // its kernel-only measurement: the ROI pointer as the preparation (prepareNpp), the call behind the spin
    const auto launchNppResize = [&](const Npp8u* roi) {
        FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(roi, static_cast<int>(frameStep), nppCropSize, NppiRect{0, 0, CROP_W, CROP_H}, nppRsAlone,
                                             static_cast<int>(rsAloneStep), nppOutSize, NppiRect{0, 0, OUT_W, OUT_H}, NPPI_INTER_LINEAR, ctx));
    };
    // npp_box: npp's second call alone, on its own copy of the uploaded image
    const auto callNppBox = [&] {
        FKLB_NPP_CHECK(nppiFilterBoxBorder_8u_C3R_Ctx(nppBoxIn, static_cast<int>(boxInStep), nppOutSize, NppiPoint{0, 0}, nppBoxOut,
                                                      static_cast<int>(boxOutStep), nppOutSize, nppMaskSize, NppiPoint{ANCHOR_X, ANCHOR_Y},
                                                      NPP_BORDER_REPLICATE, ctx));
    };
    const auto nothingToPrepare = [] { return 0; };
    const auto launchNppBox = [&](int) { callNppBox(); };

    // The order of the checks and of the timing, the same in every run
    const auto makeKernelOnly = [&](const auto& prepare, const auto& launch) {
        return [&, prepare, launch](fklb::Timing& gpu, fklb::Timing& host) { fklb::timePreparedLaunch(cs, prepare, spin, launch, gpu, host); };
    };
    impls.push_back({"fkl_box_asym", "ASYMMETRIC", "pipeline", false, callAsym, makeKernelOnly(prepareAsym, launchPrepared),
                     [&] { return download8u(outAsym.ptr().data, outAsym.ptr().dims.pitch); }});
    impls.push_back({"fkl_trunc_halfpixel", "HALF_PIXEL", "pipeline", true, callTruncHalf, makeKernelOnly(prepareTruncHalf, launchPrepared),
                     [&] { return download8u(outTruncHalf.ptr().data, outTruncHalf.ptr().dims.pitch); }});
    impls.push_back({"npp", "HALF_PIXEL", "pipeline", true, callNpp, makeKernelOnly(prepareNpp, launchNpp), [&] { return download8u(nppOut, outStep); }});
    impls.push_back({"npp_resize", "HALF_PIXEL", "resize", false, callNppResize, makeKernelOnly(prepareNpp, launchNppResize),
                     [&] { return download8u(nppRsAlone, rsAloneStep); }});
    impls.push_back({"fkl_resize", "HALF_PIXEL", "resize", false, callResize, makeKernelOnly(prepareResize, launchResize),
                     [&] { return download8u(outResize.ptr().data, outResize.ptr().dims.pitch); }});
    impls.push_back({"npp_box", "none", "box", true, callNppBox, makeKernelOnly(nothingToPrepare, launchNppBox),
                     [&] { return download8u(nppBoxOut, boxOutStep); }});
    impls.push_back({"fkl_box", "none", "box", true, callBox, makeKernelOnly(prepareBox, launchPrepared),
                     [&] { return download8u(outBox.ptr().data, outBox.ptr().dims.pitch); }});

    // ---- The profile mode: a warm-up call outside any range, then E18_PROFILE_CALLS calls inside the NVTX range named after the row
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
        std::cout << "profile mode: " << profileCalls << " calls per selected row in its NVTX range" << std::endl;
    }

    bool allOk = true;
    if (!profile) {
        // ---- One call of every selected row, its output and, for npp, its resized image (the intermediate of the call)
        std::vector<Output> outputs;
        std::vector<unsigned char> resizedNpp;
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            impl.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            outputs.push_back({impl.version, impl.sampling, impl.step, impl.truncating, impl.download()});
            if (impl.version == "npp") resizedNpp = download8u(nppRs, rsStep);
        }
        const auto find = [&](const std::string& v) -> const Output* {
            for (const Output& o : outputs) {
                if (o.version == v) return &o;
            }
            return nullptr;
        };

        // ---- The exact references, and the double references against them (e18_box.cu)
        const Exact boxOfRefAsym = exactBox(refRsAsym), boxOfRefHalf = exactBox(refRsHalf);
        const std::vector<unsigned char> refAsym = rounded(boxOfRefAsym), refHalf = rounded(boxOfRefHalf);
        const std::vector<unsigned char> refTruncHalf = truncated(boxOfRefHalf);
        fklb::CsvWriter checks("e18_box_split_checks.csv", "E18 box split: the exact references", {"check", "sampling", "value", "criterion", "ok"});
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

        fklb::CsvWriter agreement("e18_box_split_agreement.csv", "E18 box split: agreement of the uchar outputs",
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
        // a comparison that must find the two images equal
        const auto mustBeEqual = [&](const Difference& d) { allOk = allOk && d.valuesDiffer == 0; };

        // ---- The resize. FKL's read chain of each convention, through fk::executeLinearFilter with the identity weights (e18_box.cu)
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
            const Details details{OUT_W, OUT_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
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
        row("resize", "fkl_read_chain_asym", "ASYMMETRIC", "exact_resize_ASYMMETRIC", chainAsym, refRsAsym, &exactRsAsym);
        row("resize", "fkl_read_chain_halfpixel", "HALF_PIXEL", "exact_resize_HALF_PIXEL", chainHalf, refRsHalf, &exactRsHalf);
        if (!resizedNpp.empty()) {
            row("resize", "npp_intermediate", "HALF_PIXEL", "exact_resize_HALF_PIXEL", resizedNpp, refRsHalf, &exactRsHalf);
            row("resize", "fkl_read_chain_halfpixel", "HALF_PIXEL", "npp_intermediate", chainHalf, resizedNpp, &exactRsHalf);
        }
        // the resize alone: npp_resize is npp's first call, fkl_resize the image Read IOp of fkl_trunc_halfpixel up to its uchar value
        const Output* nppResize = find("npp_resize");
        const Output* fklResize = find("fkl_resize");
        if (nppResize != nullptr) {
            if (!resizedNpp.empty()) mustBeEqual(row("resize", "npp_resize", "HALF_PIXEL", "npp_intermediate", nppResize->image, resizedNpp, &exactRsHalf));
            row("resize", "npp_resize", "HALF_PIXEL", "exact_resize_HALF_PIXEL", nppResize->image, refRsHalf, &exactRsHalf);
        }
        if (fklResize != nullptr) {
            mustBeEqual(row("resize", "fkl_resize", "HALF_PIXEL", "fkl_read_chain_halfpixel", fklResize->image, chainHalf, &exactRsHalf));
            row("resize", "fkl_resize", "HALF_PIXEL", "exact_resize_HALF_PIXEL", fklResize->image, refRsHalf, &exactRsHalf);
            if (nppResize != nullptr) row("resize", "fkl_resize", "HALF_PIXEL", "npp_resize", fklResize->image, nppResize->image, &exactRsHalf);
        }

        // ---- The box alone, on the uploaded image (the exact HALF_PIXEL resize): both compute floor(sum / 9)
        const Output* nppBox = find("npp_box");
        const Output* fklBox = find("fkl_box");
        if (nppBox != nullptr) mustBeEqual(row("box", "npp_box", "none", "exact_box_truncated", nppBox->image, refTruncHalf, &boxOfRefHalf));
        if (fklBox != nullptr) mustBeEqual(row("box", "fkl_box", "none", "exact_box_truncated", fklBox->image, refTruncHalf, &boxOfRefHalf));
        if (nppBox != nullptr && fklBox != nullptr) mustBeEqual(row("box", "fkl_box", "none", "npp_box", fklBox->image, nppBox->image, &boxOfRefHalf));

        // ---- The whole pipeline: the three repeated rows, as in e18_box.cu
        for (const Output& o : outputs) {
            if (o.step != "pipeline") continue;
            const bool asym = o.sampling == "ASYMMETRIC";
            const bool fkl = o.version.rfind("fkl_", 0) == 0;
            // the exact pipeline of its convention, with the box rounded to the nearest (every row) and truncated (the truncating ones)
            row("pipeline", o.version, o.sampling, "exact_pipeline_" + o.sampling, o.image, asym ? refAsym : refHalf, nullptr);
            if (o.truncating) {
                row("pipeline", o.version, o.sampling, "exact_pipeline_truncated_" + o.sampling, o.image, asym ? truncated(boxOfRefAsym) : refTruncHalf,
                    nullptr);
            }
            // the exact box (rounded or truncated, as its own box) of its own resize: FKL's read chain of the convention, or npp's intermediate.
            // 0 differences: every difference from the exact pipeline comes from the resize
            const std::vector<unsigned char>& ownResize = fkl ? (asym ? chainAsym : chainHalf) : resizedNpp;
            const Exact ownBox = exactBox(ownResize);
            const Difference d = o.truncating
                                     ? row("pipeline", o.version, o.sampling, "exact_box_truncated_of_own_resize", o.image, truncated(ownBox), nullptr)
                                     : row("pipeline", o.version, o.sampling, "exact_box_of_own_resize", o.image, rounded(ownBox), nullptr);
            if (fkl) mustBeEqual(d);
        }
        if (find("fkl_trunc_halfpixel") != nullptr && find("npp") != nullptr) {
            row("pipeline", "fkl_trunc_halfpixel", "HALF_PIXEL", "npp", find("fkl_trunc_halfpixel")->image, find("npp")->image, nullptr);
        }
        std::cout << (allOk ? "E18 box split checks passed" : "E18 box split CHECK FAILED") << std::endl;
    }

    // ---- The timing, after the checks, in the order of the list, as in e18_box.cu. For every row, two intervals:
    //   whole_call   fklb::timeIt: CUDA events around one call, 100 calls after one warm-up call. The FKL rows build in the call; the NPP rows are
    //                called as users call them. When the GPU waits for the host, the interval also holds the host's work.
    //   kernel_only  fklb::timePreparedLaunch behind gpuSpin: the preparation (FKL: building the view, the IOps and the details; NPP's resize: the ROI
    //                pointer; npp_box: nothing) runs on the host before the spin is enqueued and is timed on the host; then the spin, the start
    //                event, the whole call (every kernel of it), the stop event. The interval runs from the end of the spin to the end of the call's
    //                last kernel.
    // Each interval of each row is inside an NVTX range ("whole_call:<version>", "kernel_only:<version>") for the profiler.
    if (timing && allOk) {
        fklb::CsvWriter t("e18_box_split.csv", "E18 box split: whole-call and kernel-only times",
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
            t.row(impl.version, impl.sampling, "whole_call", whole.mean, whole.min, "", "", fklb::ITERATIONS, "");
            t.row(impl.version, impl.sampling, "kernel_only", gpu.mean, gpu.min, host.mean, host.min, fklb::ITERATIONS, spinUs());
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
    FKLB_CUDA_CHECK(cudaFree(nppRsAlone));
    FKLB_CUDA_CHECK(cudaFree(nppBoxIn));
    FKLB_CUDA_CHECK(cudaFree(nppBoxOut));
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
