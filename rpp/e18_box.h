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

// E18 (AMD): a 3 x 3 box filter after a crop and a resize, in 8 bits from start to end: RPP (two calls) against FKL (one kernel), on an AMD GPU
// (HIP, ParArch::GPU_AMD). The pipeline is the one every library of E18 can run (OpenCV-CUDA, NPP, RPP and FKL).
//
// The pipeline, on one 3840 x 2160 8UC3 frame (the seeded source of E4, as in Figure 15 and the first E18 program):
//   crop       2560 x 1440 at (640, 360)
//   resize     2560 x 1440 -> 1920 x 1080, bilinear, half-pixel sampling ((x + 0.5) * scale - 0.5), 8-bit input and 8-bit output (rounded to the
//              nearest integer, ties to even)
//   box        3 x 3, anchor at the centre, border REPLICATE: round(sum / 9), 8-bit input and 8-bit output
//   output     1920 x 1080 8UC3
// The variants, in the order of the checks and of the timing:
//   rpp                RPP 3.2.0, HIP back-end, its own buffers and no FKL object. Two calls, as a user writes them:
//                      rppt_resize (U8 NHWC, batch 1, BILINEAR, the crop as the XYWH ROI of the frame, the output size 1920 x 1080) into an
//                      intermediate 1920 x 1080 U8 tensor, then rppt_box_filter (kernelSize 3, REPLICATE, the whole intermediate as its XYWH ROI)
//                      into the output. The ROIs and the output size are in pinned host memory, as in the Figure 15 program
//                      (rpp/benchmark_rpp_preprocessing.h). rppt_resize converts its ROI from XYWH to LTRB in place, so every call first
//                      writes the crop's ROI on the host; that write is part of the call, as the crop may move on every call. rppt_box_filter
//                      converts only an LTRB ROI (box_filter.cpp:2050), so its XYWH ROI is written once.
//   rpp_ltrb           rpp with the crop passed to rppt_resize as an LTRB ROI (640, 360, 3199, 1799), rppt_resize's own form: it converts only
//                      an XYWH ROI (resize.cpp:944), so it launches no conversion kernel. Its ROI is written before every call as well.
//   fkl_box_halfpixel  FKL, one kernel: one fk::executeOperations<fk::LinearFilterDPP<GPU_AMD, Details>>. Its image Read IOp is the read of the crop view (Ptr2D::crop2D) followed by
//                      fk::Resize (INTER_LINEAR, ResizeSampling::HALF_PIXEL, to float3), fk::SaturateCast<float3, uchar3> (the 8-bit resized
//                      image, rounded inside the kernel as RPP rounds it between its two calls) and fk::Cast<uchar3, float3>. The coefficients
//                      are fk::ConstantFilterRead<float3> with 1/9 in every channel; multiply and accumulate are the Unary forms of fk::Mul and
//                      fk::Add on float3; the write is fk::SaturateCast<float3, uchar3> followed by the write of the output. These are the
//                      arguments that fk::executeBoxFilter built (linear_filter.h:452-468 at 24478f0; FKL #435 removed it); it did not compile
//                      for LinearFilterDPPDetails<float3>, because its coefficient scale / static_cast<T>(kernelWidth * kernelHeight)
//                      (linear_filter.h:459-460, GPU_AMD; 324-325 GPU_NVIDIA) has no conversion from int to a vector T. Every call builds
//                      the crop view, every IOp and the details, and launches.
//   fkl_half_once      fkl_box_halfpixel with the crop view, the IOps and the details built once; every call is that fk::executeOperations
//
// Every run checks first, then times (the timing is skipped when a required check fails). Every comparison counts the interior and the last
// two rows and columns separately (RPP's resize never reads the last column and row of its ROI). In the check runs every RPP call is followed by
// hipGetLastError and hipStreamSynchronize, because RPP returns RPP_SUCCESS when a launch fails.
//   a. The resize alone, against the exact reference resize (exactResizeHalfPixel: in integers, rounded to the nearest integer with ties to
//      even exactly): RPP's intermediate tensor, and FKL's image Read IOp written by LinearFilterDPP (fk::executeOperations) with the identity coefficients
//      0 0 0 / 0 1 0 / 0 0 0 (a 3 x 3 fk::Ptr2D<float3>) and the same write; FKL against RPP; and, as a cross-check, the double-precision
//      resize (common/resize_reference.h) rounded with std::nearbyint. The differences are counted at the exact ties of the exact resize, within
//      1/36 of a tie, and at the other values; e18_box_ties_GPU_AMD.csv counts the ties and how each image rounds them. Informational.
//   b. The box alone, on one 1920 x 1080 8UC3 image (E4's generator at 1920 x 1080, the whole image as the ROI): rppt_box_filter, and FKL's
//      LinearFilterDPP (fk::executeOperations) of the box on a plain read of the image (PerThreadRead, Cast<uchar3, float3>), each against round(sum / 9) with the
//      border replicated, in integers. Required: 0 differences.
//   c. The whole pipeline: rpp and fkl_box_halfpixel against the exact reference (the exact resize, then round(sum / 9)), and
//      fkl_box_halfpixel against rpp. Whether every difference comes from step (a): each output against round(sum / 9) of its own 8-bit resized
//      image (RPP's intermediate; FKL's identity output) must give 0 differences, and every differing output value must have a differing resized
//      value in its 3 x 3 window (the differences of the resized images are those of step a). Required: 0 and all.
//   d. fkl_half_once against fkl_box_halfpixel: bitwise equal. Required. rpp_ltrb against rpp (its intermediate and its output): bitwise equal.
//      Required.
//   Also checked: RPP's ROIs after the calls (the crop's is LTRB, the box filter's is unchanged), and the sizes of the fused kernel's arguments.
//   Files: e18_box_agreement_GPU_AMD.csv (a, b, c, d), e18_box_ties_GPU_AMD.csv (a), e18_box_attribution_GPU_AMD.csv (c),
//   e18_box_checks_GPU_AMD.csv (the other checks).
// The timing (e18_box_GPU_AMD.csv), two intervals per variant, in the order of the list, the same in every run:
//   whole_call   fklb::timeIt (common/hip_timing.h): events around one call, 100 calls after one warm-up call (E4's interval, Figure 15)
//   kernel_only  fklb::timePreparedLaunch with gpuSpin (5 ms) as beforeLaunch (E1's interval, Table III): the preparation is timed on the host
//                (rpp, rpp_ltrb: the ROI write; fkl_box_halfpixel: the crop view, the IOps and the details; fkl_half_once: nothing); then the spin, the start
//                event, every kernel of the call (RPP's behind one spin), the stop event
//   Each interval of each variant is in a ROCTx range ("whole_call:<version>", "kernel_only:<version>") for rocprofv3.
//   Before the timing, one gpuSpin of E18_CLOCK_WARMUP_MS (default 2000) brings the GPU's graphics clock up from idle (the RX 9060 idles while
//   the host computes the references of the checks); sclk (fklb::amdGpuClockMHz, common/hip_wallclock.h) is read before and after every
//   interval, outside the timed calls, into the columns sclk_before_mhz and sclk_after_mhz.
// E18_ONLY=a,b: only these variants. E18_NO_TIMING=1: the checks only. E18_SPIN_US: the length of the spin (default 5000).
// E18_PROFILE=1: for rocprofv3, no checks and no timing: after the setup, one warm-up call of every selected variant, then three calls of each,
//   every call in a ROCTx range named after the variant and followed by a synchronisation of its stream.
#include <common/csv_writer.h>
#include <common/hip_timing.h>
#include <common/hip_wallclock.h>
#include <common/main.h>
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

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rpp.h>
#include <rppdefs.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace e18box {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_X = 640, CROP_Y = 360, CROP_W = 2560, CROP_H = 1440;
constexpr int OUT_W = 1920, OUT_H = 1080;
constexpr int KERNEL_W = 3, KERNEL_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr size_t OUT_BYTES = static_cast<size_t>(OUT_W) * OUT_H * 3;
constexpr size_t SRC_BYTES = static_cast<size_t>(SRC_W) * SRC_H * 3;
// RPP's buffers. The tensor the box filter reads starts RPP_OFFSET bytes into its allocation: rppt_box_filter's HIP path returns
// RPP_ERROR_LOW_OFFSET when srcDesc->offsetInBytes < 12 * (kernelSize / 2) (rppt_tensor_filter_augmentations.cpp:133); 64 keeps the image
// 64-byte aligned. Every RPP buffer has RPP_TAIL bytes after its image: RPP's kernels store 8 pixels at a time, up to 7 pixels (21 bytes) past
// the ROI width. RPP's ROI conversion kernels run 256 threads for a batch of 1 and rewrite one ROI each (rpp_hip_roi_conversion.hpp), so every
// ROI array has 256 entries; entry 0 is the image's, the others are padding that only RPP writes.
constexpr unsigned RPP_OFFSET = 64;
constexpr size_t RPP_TAIL = 64;
constexpr size_t RPP_ROI_ENTRIES = 256;

inline bool selected(const std::string& name) {
    const char* e = std::getenv("E18_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

inline bool envIs1(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && std::string(e) == "1";
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

inline size_t at(const int x, const int y, const int c) { return (static_cast<size_t>(y) * OUT_W + x) * 3 + c; }

// The half-pixel bilinear resize of the 8-bit crop to OUT_W x OUT_H in integers, rounded to the nearest integer with ties to even, exactly.
// The destination x samples the source position ((2x + 1) * srcW - dstW) / (2 dstW) (clamped as in common/resize_reference.h: not before the
// first pixel, the second neighbour not after the last), so the weights are multiples of 1 / (2 dstW) and 1 / (2 dstH), and the value times
// D = (2 dstW) (2 dstH) is an integer N. At this scale (4/3, 2560 x 1440 -> 1920 x 1080) the sample positions are multiples of 1/6, the
// weights of each direction are 1/6, 1/2 and 5/6, and every value is a multiple of 1/36 (checked: N is a multiple of D / 36). So many exact
// values lie halfway between two integers; a computation with inexact weights (the double-precision reference, and every float implementation)
// rounds those by the sign of its own error.
// The class of each value: 0 an exact tie (k + 18/36), 1 within 1/36 of a tie (k + 17/36 or k + 19/36), 2 the others; floorValue: floor(value).
enum TieClass : unsigned char { EXACT_TIE = 0, WITHIN_1_36 = 1, OTHER = 2 };
struct ExactResize {
    std::vector<unsigned char> rounded, cls, floorValue;
};
inline ExactResize exactResizeHalfPixel(const unsigned char* src, const size_t pitch, const int srcW, const int srcH) {
    const int64_t denX = 2 * static_cast<int64_t>(OUT_W), denY = 2 * static_cast<int64_t>(OUT_H), D = denX * denY;
    if (D % 36 != 0) throw std::runtime_error("exactResizeHalfPixel: D is not a multiple of 36");
    const auto tap = [](const int d, const int64_t srcSize, const int64_t den, const int dstSize, int& i1, int& i2, int64_t& frac) {
        int64_t num = (2 * static_cast<int64_t>(d) + 1) * srcSize - dstSize;
        num = std::max<int64_t>(num, 0);
        i1 = static_cast<int>(num / den);
        frac = num % den;
        if (i1 >= srcSize - 1) {
            i1 = static_cast<int>(srcSize - 1);
            frac = 0;
        }
        i2 = std::min(i1 + 1, static_cast<int>(srcSize - 1));
    };
    ExactResize e;
    e.rounded.assign(OUT_BYTES, 0);
    e.cls.assign(OUT_BYTES, OTHER);
    e.floorValue.assign(OUT_BYTES, 0);
    for (int y = 0; y < OUT_H; ++y) {
        int y1, y2;
        int64_t fy;
        tap(y, srcH, denY, OUT_H, y1, y2, fy);
        for (int x = 0; x < OUT_W; ++x) {
            int x1, x2;
            int64_t fx;
            tap(x, srcW, denX, OUT_W, x1, x2, fx);
            for (int c = 0; c < 3; ++c) {
                const auto p = [&](const int xx, const int yy) { return static_cast<int64_t>(src[static_cast<size_t>(yy) * pitch + static_cast<size_t>(xx) * 3 + c]); };
                const int64_t n = (p(x1, y1) * (denX - fx) + p(x2, y1) * fx) * (denY - fy) + (p(x1, y2) * (denX - fx) + p(x2, y2) * fx) * fy;
                const int64_t q = n / D, r = n % D;
                if (r % (D / 36) != 0) throw std::runtime_error("exactResizeHalfPixel: a value is not a multiple of 1/36");
                const int64_t k = r / (D / 36);   // the value is q + k / 36
                const int64_t rounded = 2 * r > D ? q + 1 : (2 * r == D ? q + (q & 1) : q);
                e.rounded[at(x, y, c)] = static_cast<unsigned char>(rounded);
                e.floorValue[at(x, y, c)] = static_cast<unsigned char>(q);
                e.cls[at(x, y, c)] = k == 18 ? EXACT_TIE : ((k == 17 || k == 19) ? WITHIN_1_36 : OTHER);
            }
        }
    }
    return e;
}

// The 3 x 3 box of an 8-bit OUT_W x OUT_H image in integers: round(sum / 9) = (sum + 4) / 9 (sum / 9 is never halfway between two integers),
// the border replicated
inline std::vector<unsigned char> boxReference(const std::vector<unsigned char>& image) {
    std::vector<unsigned char> out(OUT_BYTES);
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                int sum = 0;
                for (int ky = 0; ky < KERNEL_H; ++ky) {
                    for (int kx = 0; kx < KERNEL_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), OUT_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), OUT_H - 1);
                        sum += image[at(sx, sy, c)];
                    }
                }
                out[at(x, y, c)] = static_cast<unsigned char>((sum + 4) / 9);
            }
        }
    }
    return out;
}

// The regions of every comparison: the interior, and the last two columns and rows
inline bool inEdge(const size_t value) {
    const size_t pixel = value / 3;
    const int x = static_cast<int>(pixel % OUT_W), y = static_cast<int>(pixel / OUT_W);
    return x >= OUT_W - 2 || y >= OUT_H - 2;
}
inline bool inLastRowOrColumn(const size_t value) {
    const size_t pixel = value / 3;
    const int x = static_cast<int>(pixel % OUT_W), y = static_cast<int>(pixel / OUT_W);
    return x == OUT_W - 1 || y == OUT_H - 1;
}
struct Region {
    size_t values = 0, differ = 0, differLast = 0;   // differLast: in the last row or column
    size_t differByClass[3] = {0, 0, 0};             // by the TieClass of the exact resize at the value (resized images only)
    int maxDiff = 0;
};
struct Comparison {
    Region region[2];   // 0 the interior, 1 the last two columns and rows
};

// exact: the exact resize, for the comparisons of resized images (the differences counted by TieClass); nullptr for the others
inline Comparison compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const ExactResize* exact) {
    if (a.size() != OUT_BYTES || b.size() != OUT_BYTES) throw std::runtime_error("compare8u: not an output image");
    Comparison cmp;
    for (size_t i = 0; i < OUT_BYTES; ++i) {
        Region& r = cmp.region[inEdge(i) ? 1 : 0];
        ++r.values;
        const int diff = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        if (diff != 0) {
            ++r.differ;
            r.differLast += inLastRowOrColumn(i) ? 1 : 0;
            r.maxDiff = std::max(r.maxDiff, diff);
            if (exact != nullptr) ++r.differByClass[exact->cls[i]];
        }
    }
    return cmp;
}

// How an 8-bit resized image rounds the exact ties: per region, the values of each TieClass, and at the exact ties how many of the image's values
// are floor + 1 (rounded up), floor (rounded down) or neither
struct TieRounding {
    size_t values = 0, byClass[3] = {0, 0, 0}, up = 0, down = 0, neither = 0;
};
inline void tieRounding(const std::vector<unsigned char>& image, const ExactResize& exact, TieRounding (&t)[2]) {
    for (size_t i = 0; i < OUT_BYTES; ++i) {
        TieRounding& r = t[inEdge(i) ? 1 : 0];
        ++r.values;
        ++r.byClass[exact.cls[i]];
        if (exact.cls[i] != EXACT_TIE) continue;
        if (image[i] == exact.floorValue[i] + 1) {
            ++r.up;
        } else if (image[i] == exact.floorValue[i]) {
            ++r.down;
        } else {
            ++r.neither;
        }
    }
}

// Step (c): the output values where a and b differ, and how many of them have a value in their 3 x 3 window (border replicated, the same
// channel) where the resized images ra and rb differ
struct Attribution {
    size_t differ[2] = {0, 0}, explained[2] = {0, 0};
};
inline Attribution attribute(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b, const std::vector<unsigned char>& ra,
                             const std::vector<unsigned char>& rb) {
    Attribution att;
    for (int y = 0; y < OUT_H; ++y) {
        for (int x = 0; x < OUT_W; ++x) {
            for (int c = 0; c < 3; ++c) {
                const size_t i = at(x, y, c);
                if (a[i] == b[i]) continue;
                const int r = inEdge(i) ? 1 : 0;
                ++att.differ[r];
                bool resizeDiffers = false;
                for (int ky = 0; ky < KERNEL_H && !resizeDiffers; ++ky) {
                    for (int kx = 0; kx < KERNEL_W && !resizeDiffers; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), OUT_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), OUT_H - 1);
                        resizeDiffers = ra[at(sx, sy, c)] != rb[at(sx, sy, c)];
                    }
                }
                att.explained[r] += resizeDiffers ? 1 : 0;
            }
        }
    }
    return att;
}

// A digest of an 8-bit image (FNV-1a, 64 bits) for the console and the checks file, so that the outputs of two runs or two builds can be compared
inline std::string digest(const std::vector<unsigned char>& image) {
    uint64_t h = 1469598103934665603ull;
    for (const unsigned char b : image) {
        h ^= b;
        h *= 1099511628211ull;
    }
    char text[17];
    std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(h));
    return text;
}

inline std::vector<unsigned char> download8u(const void* data, const size_t pitch) {
    std::vector<unsigned char> host(OUT_BYTES);
    FKLB_HIP_CHECK(hipMemcpy2D(host.data(), static_cast<size_t>(OUT_W) * 3, data, pitch, static_cast<size_t>(OUT_W) * 3, OUT_H, hipMemcpyDeviceToHost));
    return host;
}

// The GPU spin of the kernel-only interval (E1's role, handwritten-cuda/e1_common.h): one thread waits on the GPU's constant-rate wall clock
// (s_memrealtime, wall_clock64()) and touches no memory. Enqueued before the start event, it keeps the GPU busy while the host records the start
// event and enqueues the call, so that the interval holds the kernels and not the host's dispatch of them. `ticks` = microseconds * the wall clock
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

// The clock warm-up before the timing, in ms (E18_CLOCK_WARMUP_MS, default 2000)
inline unsigned long long clockWarmupMs() {
    const char* e = std::getenv("E18_CLOCK_WARMUP_MS");
    return std::strtoull(e != nullptr ? e : "2000", nullptr, 10);
}

// The host's CLOCK_REALTIME in ns, the time base of the window's clock sampler (date +%s%N)
inline long long realtimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
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
    hipStream_t stream;
    std::function<void()> call;
    std::function<void(fklb::Timing& gpu, fklb::Timing& host)> kernelOnly;
    std::function<std::vector<unsigned char>()> download;
};
}  // namespace e18box

int launch() {
    using namespace e18box;
    using fk::float3;
    using fk::uchar3;
    const bool profile = envIs1("E18_PROFILE");
    const bool timing = !profile && !envIs1("E18_NO_TIMING");
    std::cout << "E18 box (AMD): "
              << (profile ? "the profile mode (E18_PROFILE=1): calls in ROCTx ranges, no checks, no timing"
                          : (timing ? "the checks, then the timing" : "the checks only (E18_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> noise = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    int device = 0, wallClockKHz = 0;
    FKLB_HIP_CHECK(hipGetDevice(&device));
    FKLB_HIP_CHECK(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, device));
    const unsigned long long spinTicks = spinUs() * static_cast<unsigned long long>(wallClockKHz) / 1000ull;
    // In the check runs every RPP call is followed by hipGetLastError and hipStreamSynchronize (RPP returns RPP_SUCCESS when a launch fails);
    // in the profile mode and in the timing it is not
    bool checking = !profile;

    // ---- RPP: its own stream, handle, buffers, descriptors, output size and ROIs, all set up once (none depends on the crop but the crop's ROI)
    hipStream_t rppStream;
    FKLB_HIP_CHECK(hipStreamCreate(&rppStream));
    rppHandle_t handle{};
    FKLB_RPP_CHECK(rppCreate(&handle, 1, 0, rppStream, RppBackend::RPP_HIP_BACKEND));
    // the frame; per variant (rpp, rpp_ltrb) its own intermediate and output
    unsigned char *rppFrame = nullptr, *rppResized = nullptr, *rppOut = nullptr, *rppResizedLtrb = nullptr, *rppOutLtrb = nullptr;
    FKLB_HIP_CHECK(hipMalloc(&rppFrame, SRC_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppResized, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppOut, OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppResizedLtrb, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppOutLtrb, OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMemset(rppResized, 0, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMemset(rppResizedLtrb, 0, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMemcpy(rppFrame, noise.data(), SRC_BYTES, hipMemcpyHostToDevice));
    RpptDesc frameDesc(1, 3, SRC_H, SRC_W, RpptDataType::U8, RpptLayout::NHWC);        // the whole frame; the crop is the ROI
    RpptDesc resizedDesc(1, 3, OUT_H, OUT_W, RpptDataType::U8, RpptLayout::NHWC);      // the intermediate: written by the resize, read by the box
    resizedDesc.offsetInBytes = RPP_OFFSET;
    RpptDesc outDesc(1, 3, OUT_H, OUT_W, RpptDataType::U8, RpptLayout::NHWC);
    // The parameter arrays of the RPP calls, in pinned host memory (hipHostMalloc), as in the Figure 15 program
    // (rpp/benchmark_rpp_preprocessing.h): the output size, the crop's ROI of each variant and the box filter's ROI. RPP's kernels read them in
    // place on the GPU, and the XYWH conversion kernel rewrites them in place. RPP 3.2.0's header documents roiTensorPtrSrc as HIP memory for
    // the HIP back-end. The host writes a crop's ROI before every call; no RPP kernel still reads it then, because every call of this program
    // follows a synchronisation (the check run synchronises after every call, fklb::timeIt and fklb::timePreparedLaunch wait for each call's stop
    // event, and the profile mode synchronises after every call).
    RpptImagePatch* rppSizes = nullptr;
    RpptROI *rppResizeRoi = nullptr, *rppLtrbRoi = nullptr, *rppBoxRoi = nullptr;
    FKLB_HIP_CHECK(hipHostMalloc(&rppSizes, sizeof(RpptImagePatch)));
    FKLB_HIP_CHECK(hipHostMalloc(&rppResizeRoi, sizeof(RpptROI) * RPP_ROI_ENTRIES));
    FKLB_HIP_CHECK(hipHostMalloc(&rppLtrbRoi, sizeof(RpptROI) * RPP_ROI_ENTRIES));
    FKLB_HIP_CHECK(hipHostMalloc(&rppBoxRoi, sizeof(RpptROI) * RPP_ROI_ENTRIES));
    rppSizes[0].width = OUT_W;
    rppSizes[0].height = OUT_H;
    std::memset(rppResizeRoi, 0, sizeof(RpptROI) * RPP_ROI_ENTRIES);
    std::memset(rppLtrbRoi, 0, sizeof(RpptROI) * RPP_ROI_ENTRIES);
    std::memset(rppBoxRoi, 0, sizeof(RpptROI) * RPP_ROI_ENTRIES);
    rppBoxRoi[0].xywhROI = {{0, 0}, OUT_W, OUT_H};   // the box filter's: the whole intermediate, written once
    // rpp: the crop as an XYWH ROI (written before every call: rppt_resize converts it to LTRB in place), rppt_resize, rppt_box_filter
    const auto writeRppRoi = [&] { rppResizeRoi[0].xywhROI = {{CROP_X, CROP_Y}, CROP_W, CROP_H}; };
    const auto launchRpp = [&] {
        FKLB_RPP_CHECK(rppt_resize(rppFrame, &frameDesc, rppResized, &resizedDesc, rppSizes, RpptInterpolationType::BILINEAR, rppResizeRoi,
                                   RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
        FKLB_RPP_CHECK(rppt_box_filter(rppResized, &resizedDesc, rppOut, &outDesc, KERNEL_W, RpptImageBorderType::REPLICATE, rppBoxRoi,
                                       RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
    };
    const auto callRpp = [&] {
        writeRppRoi();
        launchRpp();
    };
    // rpp_ltrb: the same two calls with the crop as an LTRB ROI, rppt_resize's own form: it converts only an XYWH ROI (resize.cpp:944 on the
    // single-image path), so this ROI is not rewritten; it is written before every call all the same, as the crop may move on every call
    const auto writeRppLtrbRoi = [&] { rppLtrbRoi[0].ltrbROI = {{CROP_X, CROP_Y}, {CROP_X + CROP_W - 1, CROP_Y + CROP_H - 1}}; };
    const auto launchRppLtrb = [&] {
        FKLB_RPP_CHECK(rppt_resize(rppFrame, &frameDesc, rppResizedLtrb, &resizedDesc, rppSizes, RpptInterpolationType::BILINEAR, rppLtrbRoi,
                                   RpptRoiType::LTRB, handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
        FKLB_RPP_CHECK(rppt_box_filter(rppResizedLtrb, &resizedDesc, rppOutLtrb, &outDesc, KERNEL_W, RpptImageBorderType::REPLICATE, rppBoxRoi,
                                       RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
    };
    const auto callRppLtrb = [&] {
        writeRppLtrbRoi();
        launchRppLtrb();
    };

    // ---- FKL: its own stream, its own copy of the frame (fk::Ptr2D) and one output (fk::Ptr2D<uchar3>) per variant
    hipStream_t fklStream;
    FKLB_HIP_CHECK(hipStreamCreate(&fklStream));
    fk::Stream_<fk::ParArch::GPU_AMD> fkStream(fklStream);
    fk::Ptr2D<uchar3> dFrame(SRC_W, SRC_H);
    FKLB_HIP_CHECK(hipMemcpy2D(dFrame.ptr().data, dFrame.ptr().dims.pitch, noise.data(), hostRow, hostRow, SRC_H, hipMemcpyHostToDevice));
    fk::Ptr2D<uchar3> outBox(OUT_W, OUT_H), outOnce(OUT_W, OUT_H);

    // fkl_box_halfpixel: every call builds the crop view, every IOp and the details, and launches
    const auto callBox = [&] {
        const fk::Ptr2D<uchar3> crop = dFrame.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                               .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType,
                                                fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H)))
                               .then(fk::SaturateCast<float3, uchar3>::build())
                               .then(fk::Cast<uchar3, float3>::build());
        const auto coefficient = fk::ConstantFilterRead<float3>::build(float3{1.f, 1.f, 1.f} / float3{9.f, 9.f, 9.f});
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outBox));
        const Details details{KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
        fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, Details>>(fkStream, details, fk::make_tuple(image, coefficient), multiply,
            accumulate, output);
    };
    const auto prepareBox = [&] {
        const fk::Ptr2D<uchar3> crop = dFrame.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
        const auto reads = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop)
                                              .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType,
                                                               fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(float3{1.f, 1.f, 1.f} / float3{9.f, 9.f, 9.f}));
        const auto multiply = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
        const auto accumulate = fk::Add<float3, float3, float3, fk::UnaryType>::build();
        const auto output = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outBox));
        return FilterArguments<std::decay_t<decltype(reads)>, std::decay_t<decltype(multiply)>, std::decay_t<decltype(accumulate)>,
                               std::decay_t<decltype(output)>>{reads, multiply, accumulate, output,
                                                               Details{KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y}};
    };
    const auto launchPrepared = [&](const auto& a) {
        fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, Details>>(fkStream, a.details, a.reads, a.multiply, a.accumulate, a.output);
    };
    // fkl_half_once: built once, valid only while the crop (its position and size) does not change
    const fk::Ptr2D<uchar3> cropOnce = dFrame.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
    const auto readsOnce = fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(cropOnce)
                                              .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType,
                                                               fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H)))
                                              .then(fk::SaturateCast<float3, uchar3>::build())
                                              .then(fk::Cast<uchar3, float3>::build()),
                                          fk::ConstantFilterRead<float3>::build(float3{1.f, 1.f, 1.f} / float3{9.f, 9.f, 9.f}));
    const auto multiplyOnce = fk::Mul<float3, float3, float3, fk::UnaryType>::build();
    const auto accumulateOnce = fk::Add<float3, float3, float3, fk::UnaryType>::build();
    const auto outputOnce = fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(outOnce));
    const Details detailsOnce{KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
    const auto callOnce = [&] {
        fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, Details>>(fkStream, detailsOnce, readsOnce, multiplyOnce, accumulateOnce,
            outputOnce);
    };

    // ---- The variants, in the order of the checks and of the timing
    const auto makeKernelOnly = [&](const hipStream_t stream, const auto& prepare, const auto& launch) {
        return [&, stream, prepare, launch](fklb::Timing& gpu, fklb::Timing& host) {
            fklb::timePreparedLaunch(stream, prepare, [&, stream] { gpuSpin<<<1, 1, 0, stream>>>(spinTicks); }, launch, gpu, host);
        };
    };
    std::vector<Implementation> impls;
    impls.push_back({"rpp", rppStream, callRpp,
                     makeKernelOnly(rppStream, [&] { writeRppRoi(); return 0; }, [&](int) { launchRpp(); }),
                     [&] { return download8u(rppOut, static_cast<size_t>(OUT_W) * 3); }});
    impls.push_back({"rpp_ltrb", rppStream, callRppLtrb,
                     makeKernelOnly(rppStream, [&] { writeRppLtrbRoi(); return 0; }, [&](int) { launchRppLtrb(); }),
                     [&] { return download8u(rppOutLtrb, static_cast<size_t>(OUT_W) * 3); }});
    impls.push_back({"fkl_box_halfpixel", fklStream, callBox, makeKernelOnly(fklStream, prepareBox, launchPrepared),
                     [&] { return download8u(outBox.ptr().data, outBox.ptr().dims.pitch); }});
    impls.push_back({"fkl_half_once", fklStream, callOnce, makeKernelOnly(fklStream, [] { return 0; }, [&](int) { callOnce(); }),
                     [&] { return download8u(outOnce.ptr().data, outOnce.ptr().dims.pitch); }});

    // ---- The profile mode: a warm-up call of every selected variant, then three calls of each in ROCTx ranges named after it
    if (profile) {
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            impl.call();
            FKLB_HIP_CHECK(hipStreamSynchronize(impl.stream));
        }
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            for (int k = 0; k < 3; ++k) {
                roctxRangePushA(impl.version.c_str());
                impl.call();
                roctxRangePop();
                FKLB_HIP_CHECK(hipStreamSynchronize(impl.stream));
            }
        }
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
        std::cout << "E18 profile mode done" << std::endl;
        return 0;
    }

    // ---- One call of every variant (all of them: the checks need every output), its output downloaded
    std::vector<std::vector<unsigned char>> outputs;
    for (const Implementation& impl : impls) {
        FKLB_HIP_CHECK(hipStreamSynchronize(impl.stream));
        impl.call();
        FKLB_HIP_CHECK(hipStreamSynchronize(impl.stream));
        outputs.push_back(impl.download());
    }
    const std::vector<unsigned char>& outRpp = outputs[0];
    const std::vector<unsigned char>& outRppLtrb = outputs[1];
    const std::vector<unsigned char>& outFklBox = outputs[2];
    const std::vector<unsigned char>& outFklOnce = outputs[3];
    const std::vector<unsigned char> rppResized8 = download8u(rppResized + RPP_OFFSET, static_cast<size_t>(OUT_W) * 3);
    const std::vector<unsigned char> rppResizedLtrb8 = download8u(rppResizedLtrb + RPP_OFFSET, static_cast<size_t>(OUT_W) * 3);

    bool allOk = true;
    {
        fklb::CsvWriter checks("e18_box_checks_GPU_AMD.csv", "E18 box (AMD): other checks", {"check", "value", "criterion", "ok"});
        // RPP's ROIs after the calls (pinned host memory, read in place after the synchronisation): rpp's crop converted to LTRB in place by
        // rppt_resize, rpp_ltrb's crop unchanged (an LTRB ROI is not converted), the box filter's unchanged
        const auto ltrbText = [](const RpptROI& r) {
            return std::to_string(r.ltrbROI.lt.x) + " " + std::to_string(r.ltrbROI.lt.y) + " " + std::to_string(r.ltrbROI.rb.x) + " " +
                   std::to_string(r.ltrbROI.rb.y);
        };
        const auto isCropLtrb = [](const RpptROI& r) {
            return r.ltrbROI.lt.x == CROP_X && r.ltrbROI.lt.y == CROP_Y && r.ltrbROI.rb.x == CROP_X + CROP_W - 1 && r.ltrbROI.rb.y == CROP_Y + CROP_H - 1;
        };
        const RpptROI& box = rppBoxRoi[0];
        const bool cropLtrb = isCropLtrb(rppResizeRoi[0]), ltrbSame = isCropLtrb(rppLtrbRoi[0]);
        const bool boxSame = box.xywhROI.xy.x == 0 && box.xywhROI.xy.y == 0 && box.xywhROI.roiWidth == OUT_W && box.xywhROI.roiHeight == OUT_H;
        const std::string boxText = std::to_string(box.xywhROI.xy.x) + " " + std::to_string(box.xywhROI.xy.y) + " " +
                                    std::to_string(box.xywhROI.roiWidth) + " " + std::to_string(box.xywhROI.roiHeight);
        checks.row("rpp: the crop's ROI after rppt_resize (written as XYWH 640 360 2560 1440)", ltrbText(rppResizeRoi[0]), "LTRB 640 360 3199 1799",
                   cropLtrb ? 1 : 0);
        checks.row("rpp_ltrb: the crop's ROI after rppt_resize (written as LTRB 640 360 3199 1799)", ltrbText(rppLtrbRoi[0]), "unchanged",
                   ltrbSame ? 1 : 0);
        checks.row("rpp, rpp_ltrb: the box filter's ROI after rppt_box_filter (written once as XYWH 0 0 1920 1080)", boxText, "unchanged", boxSame ? 1 : 0);
        allOk = allOk && cropLtrb && ltrbSame && boxSame;
        std::cout << "rpp ROIs after the calls: rpp's crop " << ltrbText(rppResizeRoi[0]) << " (LTRB), rpp_ltrb's crop " << ltrbText(rppLtrbRoi[0])
                  << " (LTRB), box " << boxText << " (XYWH)" << std::endl;
        const auto a = prepareBox();
        const std::string sizes = "details " + std::to_string(sizeof(a.details)) + ", reads " + std::to_string(sizeof(a.reads)) + " (image " +
                                  std::to_string(sizeof(fk::get<0>(a.reads))) + ", coefficient " + std::to_string(sizeof(fk::get<1>(a.reads))) +
                                  "), multiply " + std::to_string(sizeof(a.multiply)) + ", accumulate " + std::to_string(sizeof(a.accumulate)) +
                                  ", output " + std::to_string(sizeof(a.output));
        checks.row("fused kernel arguments (bytes)", sizes, "informational", -1);
        const std::pair<const char*, const std::vector<unsigned char>*> images[] = {
            {"rpp output", &outRpp}, {"rpp_ltrb output", &outRppLtrb}, {"fkl_box_halfpixel output", &outFklBox}, {"fkl_half_once output", &outFklOnce},
            {"rpp intermediate", &rppResized8}, {"rpp_ltrb intermediate", &rppResizedLtrb8}};
        for (const auto& image : images) {
            checks.row(std::string("digest (FNV-1a 64) of the ") + image.first, digest(*image.second), "informational", -1);
            std::cout << "digest " << image.first << ": " << digest(*image.second) << std::endl;
        }
        std::cout << "fused kernel arguments (bytes): " << sizes << std::endl;
    }
    {
        // ---- The references. The exact one: the half-pixel resize of the crop in integers, rounded to nearest with ties to even exactly
        // (exactResizeHalfPixel), then round(sum / 9) in integers. The double-precision resize (common/resize_reference.h) rounded with
        // std::nearbyint (ties to even in the default rounding mode) is a cross-check only: at the exact ties its rounding follows the sign of its
        // own error.
        const unsigned char* crop = noise.data() + (static_cast<size_t>(CROP_Y) * SRC_W + CROP_X) * 3;
        const ExactResize exact = exactResizeHalfPixel(crop, hostRow, CROP_W, CROP_H);
        const std::vector<unsigned char>& refResized8 = exact.rounded;
        const std::vector<unsigned char> refOut = boxReference(refResized8);
        std::vector<double> doubleResized(OUT_BYTES);
        fklb::resizeBilinear(crop, hostRow, CROP_W, CROP_H, 3, OUT_W, OUT_H, fklb::ResizeSampling::HALF_PIXEL, doubleResized.data());
        std::vector<unsigned char> doubleResized8(OUT_BYTES);
        for (size_t i = 0; i < OUT_BYTES; ++i) doubleResized8[i] = static_cast<unsigned char>(std::nearbyint(doubleResized[i]));

        // ---- (a) FKL's image Read IOp, written through LinearFilterDPP with the identity coefficients and the same write
        fk::Ptr2D<float3> dIdentity(KERNEL_W, KERNEL_H);
        {
            std::vector<float3> id(KERNEL_W * KERNEL_H);
            for (int ky = 0; ky < KERNEL_H; ++ky) {
                for (int kx = 0; kx < KERNEL_W; ++kx) {
                    const float one = (kx == ANCHOR_X && ky == ANCHOR_Y) ? 1.f : 0.f;
                    id[ky * KERNEL_W + kx] = float3{one, one, one};
                }
            }
            FKLB_HIP_CHECK(hipMemcpy2D(dIdentity.ptr().data, dIdentity.ptr().dims.pitch, id.data(), KERNEL_W * sizeof(float3), KERNEL_W * sizeof(float3),
                                       KERNEL_H, hipMemcpyHostToDevice));
        }
        fk::Ptr2D<uchar3> dFklResized(OUT_W, OUT_H);
        {
            const fk::Ptr2D<uchar3> crop2 = dFrame.crop2D(fk::Rect(CROP_X, CROP_Y, CROP_W, CROP_H));
            const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop2)
                                   .then(fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType,
                                                    fk::ResizeSampling::HALF_PIXEL>::build(fk::Size(OUT_W, OUT_H)))
                                   .then(fk::SaturateCast<float3, uchar3>::build())
                                   .then(fk::Cast<uchar3, float3>::build());
            const Details details{KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
            fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, Details>>(fkStream, details,
                fk::make_tuple(image, fk::PerThreadRead<fk::ND::_2D, float3>::build(dIdentity)),
                fk::Mul<float3, float3, float3, fk::UnaryType>::build(), fk::Add<float3, float3, float3, fk::UnaryType>::build(),
                fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dFklResized)));
            FKLB_HIP_CHECK(hipStreamSynchronize(fklStream));
        }
        const std::vector<unsigned char> fklResized8 = download8u(dFklResized.ptr().data, dFklResized.ptr().dims.pitch);

        // ---- (b) the box alone on one 1920 x 1080 image: rppt_box_filter, and FKL on a plain read of the image
        const std::vector<unsigned char> boxInput = seededNoise(OUT_W, OUT_H);
        const std::vector<unsigned char> boxInputRef = boxReference(boxInput);
        unsigned char *rppBoxIn = nullptr, *rppBoxOut = nullptr;
        FKLB_HIP_CHECK(hipMalloc(&rppBoxIn, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
        FKLB_HIP_CHECK(hipMalloc(&rppBoxOut, OUT_BYTES + RPP_TAIL));
        FKLB_HIP_CHECK(hipMemset(rppBoxIn, 0, RPP_OFFSET + OUT_BYTES + RPP_TAIL));
        FKLB_HIP_CHECK(hipMemcpy(rppBoxIn + RPP_OFFSET, boxInput.data(), OUT_BYTES, hipMemcpyHostToDevice));
        FKLB_RPP_CHECK(rppt_box_filter(rppBoxIn, &resizedDesc, rppBoxOut, &outDesc, KERNEL_W, RpptImageBorderType::REPLICATE, rppBoxRoi, RpptRoiType::XYWH,
                                       handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        const std::vector<unsigned char> rppBoxAlone = download8u(rppBoxOut, static_cast<size_t>(OUT_W) * 3);
        fk::Ptr2D<uchar3> dBoxIn(OUT_W, OUT_H), dBoxOut(OUT_W, OUT_H);
        FKLB_HIP_CHECK(hipMemcpy2D(dBoxIn.ptr().data, dBoxIn.ptr().dims.pitch, boxInput.data(), static_cast<size_t>(OUT_W) * 3,
                                   static_cast<size_t>(OUT_W) * 3, OUT_H, hipMemcpyHostToDevice));
        {
            const Details details{KERNEL_W, KERNEL_H, ANCHOR_X, ANCHOR_Y};
            fk::executeOperations<fk::LinearFilterDPP<fk::ParArch::GPU_AMD, Details>>(fkStream, details,
                fk::make_tuple(fk::PerThreadRead<fk::ND::_2D, uchar3>::build(dBoxIn).then(fk::Cast<uchar3, float3>::build()),
                               fk::ConstantFilterRead<float3>::build(float3{1.f, 1.f, 1.f} / float3{9.f, 9.f, 9.f})),
                fk::Mul<float3, float3, float3, fk::UnaryType>::build(), fk::Add<float3, float3, float3, fk::UnaryType>::build(),
                fk::SaturateCast<float3, uchar3>::build().then(fk::PerThreadWrite<fk::ND::_2D, uchar3>::build(dBoxOut)));
            FKLB_HIP_CHECK(hipStreamSynchronize(fklStream));
        }
        const std::vector<unsigned char> fklBoxAlone = download8u(dBoxOut.ptr().data, dBoxOut.ptr().dims.pitch);
        FKLB_HIP_CHECK(hipFree(rppBoxIn));
        FKLB_HIP_CHECK(hipFree(rppBoxOut));

        // ---- The agreement rows: every comparison in the interior and in the last two columns and rows
        const char* REGION[2] = {"interior", "last 2 columns and rows"};
        {
            // ---- How each 8-bit resized image rounds the exact ties of the exact resize
            fklb::CsvWriter ties("e18_box_ties_GPU_AMD.csv", "E18 box (AMD): the exact ties of the resize and how each 8-bit resized image rounds them",
                                 {"image", "region", "values", "values_at_exact_ties", "values_within_1_36_of_a_tie", "other_values", "ties_rounded_up",
                                  "ties_rounded_down", "ties_neither"});
            const auto tieRows = [&](const char* image, const std::vector<unsigned char>& x) {
                TieRounding t[2];
                tieRounding(x, exact, t);
                for (int r = 0; r < 2; ++r) {
                    ties.row(image, REGION[r], t[r].values, t[r].byClass[EXACT_TIE], t[r].byClass[WITHIN_1_36], t[r].byClass[OTHER], t[r].up, t[r].down,
                             t[r].neither);
                    std::cout << "ties " << image << " [" << REGION[r] << "]: " << t[r].byClass[EXACT_TIE] << " exact ties, "
                              << t[r].byClass[WITHIN_1_36] << " within 1/36 of a tie, " << t[r].byClass[OTHER] << " other values; ties rounded up "
                              << t[r].up << ", down " << t[r].down << ", neither " << t[r].neither << std::endl;
                }
            };
            tieRows("exact reference resize (ties to even)", refResized8);
            tieRows("double reference resize (std::nearbyint)", doubleResized8);
            tieRows("rpp resize (intermediate)", rppResized8);
            tieRows("fkl read chain (identity)", fklResized8);
        }
        fklb::CsvWriter agreement("e18_box_agreement_GPU_AMD.csv", "E18 box (AMD): agreement of the 8-bit images",
                                  {"check", "image", "against", "region", "values", "values_differing", "differing_at_exact_ties",
                                   "differing_within_1_36_of_a_tie", "differing_other", "differing_in_last_row_or_column", "max_diff_levels", "criterion", "ok"});
        const auto rows = [&](const char* check, const char* image, const char* against, const std::vector<unsigned char>& x,
                              const std::vector<unsigned char>& y, const ExactResize* byClass, const bool requireZero) {
            const Comparison cmp = compare8u(x, y, byClass);
            for (int r = 0; r < 2; ++r) {
                const Region& g = cmp.region[r];
                const bool pass = g.differ == 0;
                const auto cls = [&](const int k) { return byClass != nullptr ? std::to_string(g.differByClass[k]) : std::string(); };
                agreement.row(check, image, against, REGION[r], g.values, g.differ, cls(EXACT_TIE), cls(WITHIN_1_36), cls(OTHER), g.differLast, g.maxDiff,
                              requireZero ? "0 values differing" : "informational", requireZero ? (pass ? 1 : 0) : -1);
                std::cout << check << " " << image << " vs " << against << " [" << REGION[r] << "]: " << g.differ << " of " << g.values << " values differ";
                if (byClass != nullptr) {
                    std::cout << " (" << g.differByClass[EXACT_TIE] << " at exact ties, " << g.differByClass[WITHIN_1_36] << " within 1/36 of a tie, "
                              << g.differByClass[OTHER] << " other)";
                }
                std::cout << ", " << g.differLast << " in the last row or column, max " << g.maxDiff << " levels"
                          << (requireZero ? (pass ? " ok" : " FAIL") : "") << std::endl;
                allOk = allOk && (pass || !requireZero);
            }
            return cmp;
        };
        rows("a", "rpp resize (intermediate)", "exact reference resize", rppResized8, refResized8, &exact, false);
        rows("a", "fkl read chain (identity)", "exact reference resize", fklResized8, refResized8, &exact, false);
        rows("a", "fkl read chain (identity)", "rpp resize (intermediate)", fklResized8, rppResized8, &exact, false);
        rows("a", "double reference resize (cross-check)", "exact reference resize", doubleResized8, refResized8, &exact, false);
        rows("b", "rpp box alone", "round(sum/9)", rppBoxAlone, boxInputRef, nullptr, true);
        rows("b", "fkl box alone", "round(sum/9)", fklBoxAlone, boxInputRef, nullptr, true);
        rows("c", "rpp", "exact reference", outRpp, refOut, nullptr, false);
        rows("c", "fkl_box_halfpixel", "exact reference", outFklBox, refOut, nullptr, false);
        rows("c", "fkl_box_halfpixel", "rpp", outFklBox, outRpp, nullptr, false);
        rows("c", "rpp", "round(sum/9) of its own intermediate", outRpp, boxReference(rppResized8), nullptr, true);
        rows("c", "fkl_box_halfpixel", "round(sum/9) of its own resized image (identity)", outFklBox, boxReference(fklResized8), nullptr, true);
        rows("d", "fkl_half_once", "fkl_box_halfpixel", outFklOnce, outFklBox, nullptr, true);
        rows("rpp_ltrb", "rpp_ltrb resize (intermediate)", "rpp resize (intermediate)", rppResizedLtrb8, rppResized8, nullptr, true);
        rows("rpp_ltrb", "rpp_ltrb", "rpp", outRppLtrb, outRpp, nullptr, true);

        // ---- (c) every differing output value has a differing resized value in its 3 x 3 window
        fklb::CsvWriter attribution("e18_box_attribution_GPU_AMD.csv", "E18 box (AMD): the differences of the outputs and the resized images",
                                    {"output", "against", "resized_images", "region", "values_differing", "with_a_differing_resized_value_in_the_window",
                                     "criterion", "ok"});
        const auto attributionRows = [&](const char* image, const char* against, const char* resized, const std::vector<unsigned char>& x,
                                         const std::vector<unsigned char>& y, const std::vector<unsigned char>& rx, const std::vector<unsigned char>& ry) {
            const Attribution att = attribute(x, y, rx, ry);
            for (int r = 0; r < 2; ++r) {
                const bool pass = att.explained[r] == att.differ[r];
                attribution.row(image, against, resized, REGION[r], att.differ[r], att.explained[r], "all", pass ? 1 : 0);
                std::cout << "c " << image << " vs " << against << " [" << REGION[r] << "]: " << att.explained[r] << " of " << att.differ[r]
                          << " differing values have a differing resized value (" << resized << ") in their window" << (pass ? " ok" : " FAIL")
                          << std::endl;
                allOk = allOk && pass;
            }
        };
        attributionRows("rpp", "exact reference", "rpp intermediate vs exact reference resize", outRpp, refOut, rppResized8, refResized8);
        attributionRows("fkl_box_halfpixel", "exact reference", "fkl identity vs exact reference resize", outFklBox, refOut, fklResized8, refResized8);
        attributionRows("fkl_box_halfpixel", "rpp", "fkl identity vs rpp intermediate", outFklBox, outRpp, fklResized8, rppResized8);
        std::cout << (allOk ? "E18 checks passed" : "E18 CHECK FAILED") << std::endl;
    }
    checking = false;

    // ---- The timing, after the checks, in the order of the list: whole_call (fklb::timeIt) and kernel_only (fklb::timePreparedLaunch behind gpuSpin)
    if (timing && allOk) {
        // The timing file is opened first: its header runs amd-smi twice (common/system_info.h, about 0.07 s each), and the RX 9060's graphics
        // clock falls from 3.2 GHz to about 1.5 GHz within 0.1 s of the GPU going idle, to idle within 0.2 s
        // (results/rx9060/e18-box-phase1/clock-warmup-test/). sclk (fklb::amdGpuClockMHz, hwmon freq1_input) is read before and after each
        // interval, outside the timed calls.
        fklb::CsvWriter t("e18_box_GPU_AMD.csv", "E18 box (AMD): whole-call and kernel-only times",
                          {"version", "interval", "mean_ms", "min_ms", "host_prepare_mean_ms", "host_prepare_min_ms", "iterations", "spin_us",
                           "sclk_before_mhz", "sclk_after_mhz"},
                          "clock warm-up: one gpuSpin of " + std::to_string(clockWarmupMs()) +
                              " ms after this file's header and right before the timing (the sclk before and after it is in the console)");
        // The clock warm-up. The graphics clock idles while the host computes the references of the checks, and it needs about 0.5 s of GPU work to
        // reach 3.1 GHz (results/rx9060/e18-box-window1, the clock-warmup-test). So one gpuSpin of E18_CLOCK_WARMUP_MS (default 2000) runs right
        // before the timing, then a synchronisation, and the first interval starts at once.
        const double sclkBeforeWarmup = fklb::amdGpuClockMHz();
        const long long warmupStartNs = realtimeNs();
        gpuSpin<<<1, 1, 0, rppStream>>>(clockWarmupMs() * static_cast<unsigned long long>(wallClockKHz));
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        const double sclkAfterWarmup = fklb::amdGpuClockMHz();
        const long long timingStartNs = realtimeNs();
        for (const Implementation& impl : impls) {
            if (!selected(impl.version)) continue;
            fklb::Timing whole, gpu, host;
            FKLB_HIP_CHECK(hipDeviceSynchronize());
            const double sclkWholeBefore = fklb::amdGpuClockMHz();
            roctxRangePushA(("whole_call:" + impl.version).c_str());
            fklb::timeIt(impl.stream, impl.call, whole);
            roctxRangePop();
            FKLB_HIP_CHECK(hipDeviceSynchronize());
            const double sclkWholeAfter = fklb::amdGpuClockMHz();
            const double sclkKernelBefore = fklb::amdGpuClockMHz();
            roctxRangePushA(("kernel_only:" + impl.version).c_str());
            impl.kernelOnly(gpu, host);
            roctxRangePop();
            const double sclkKernelAfter = fklb::amdGpuClockMHz();
            t.row(impl.version, "whole_call", whole.mean, whole.min, "", "", fklb::ITERATIONS, "", sclkWholeBefore, sclkWholeAfter);
            t.row(impl.version, "kernel_only", gpu.mean, gpu.min, host.mean, host.min, fklb::ITERATIONS, spinUs(), sclkKernelBefore, sclkKernelAfter);
            std::cout << impl.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                      << gpu.min << "), host preparation mean " << host.mean << " ms; sclk " << sclkWholeBefore << " / " << sclkWholeAfter << " / "
                      << sclkKernelBefore << " / " << sclkKernelAfter << " MHz" << std::endl;
        }
        // after the timing (no console output before it): the clock warm-up, and the host times of the warm-up and the timing (CLOCK_REALTIME, as
        // the clock sampler of the window) for the analysis of gpu_clocks.csv
        const long long timingEndNs = realtimeNs();
        std::cout << "clock warm-up: one gpuSpin of " << clockWarmupMs() << " ms; sclk " << sclkBeforeWarmup << " MHz before it, " << sclkAfterWarmup
                  << " MHz after it" << std::endl;
        std::cout << "E18 timing (realtime ns): warm-up start " << warmupStartNs << ", timing start " << timingStartNs << ", timing end " << timingEndNs
                  << std::endl;
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_HIP_CHECK(hipDeviceSynchronize());
    FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
    FKLB_HIP_CHECK(hipFree(rppFrame));
    FKLB_HIP_CHECK(hipFree(rppResized));
    FKLB_HIP_CHECK(hipFree(rppOut));
    FKLB_HIP_CHECK(hipFree(rppResizedLtrb));
    FKLB_HIP_CHECK(hipFree(rppOutLtrb));
    FKLB_HIP_CHECK(hipHostFree(rppSizes));
    FKLB_HIP_CHECK(hipHostFree(rppResizeRoi));
    FKLB_HIP_CHECK(hipHostFree(rppLtrbRoi));
    FKLB_HIP_CHECK(hipHostFree(rppBoxRoi));
    FKLB_HIP_CHECK(hipStreamDestroy(rppStream));
    FKLB_HIP_CHECK(hipStreamDestroy(fklStream));
    return allOk ? 0 : 1;
}
