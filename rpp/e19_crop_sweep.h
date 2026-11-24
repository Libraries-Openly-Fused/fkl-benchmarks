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

// E19 crop sweep (AMD): E19 (rpp/e19_batched_box.h, unchanged) at N crops of one frame, N = 1, 10, 20, 30, ..., 150 (16 values), one N per
// process: E19_CROPS=<N> selects it, and the program refuses to start for any other value. The sequence, the rows, their calls, the checks and
// the timing are E19's; only N changes. N = 30 is E19 itself. The 16 values are compiled in (fkl_batch's std::array needs N at compile time)
// and E19_CROPS chooses one at run time. Every CSV has the column "crops" (N).
// The crop table for N crops (crop i, 0..N-1): c = i mod 6, r = (i div 6) mod 5, k = i div 30, x = 64 + 600 c + SX[k], y = 64 + 400 r + SY[k],
// SX = {0, 320, 0, 320, 192}, SY = {0, 0, 200, 200, 100}. Crops 0..29 are E19's detection table. The program asserts that crops 0..29 equal
// E19's, that every crop of the 150 lies inside the frame and that the 150 positions are distinct.
//
// E19 batched box (AMD): the 3 x 3 box filter inside a batched pipeline, with a step before it and a step after it, on N crops of one frame,
// on an AMD GPU (HIP, ParArch::GPU_AMD). The AMD side of opencv-cuda/stencil/e19_batched_box.cu (E19 on NVIDIA), with the same frame, the
// same detection table and the same tensor.
//
// The pipeline, per frame, the same in every row:
//   frame    one 3840 x 2160 uchar3 frame, the seeded source of E4 and E18 (an LCG with seed 12345), read as RGB: channel 0 is R
//   crops    N crops of 256 x 256. Crop i (0..N-1) starts at the (x, y) of the crop table above (the detection table). The
//            program treats it as a layout that changes on every frame: every call builds every crop view and every copy from the table
//   grey     RGB to grey, with the library's own call and weights
//   box      3 x 3, anchor at the centre, border REPLICATE at the crop's edges
//   float    out = (v / 255 - 0.45) / 0.225 = v / 57.375 - 2
//   output   plane i of one float tensor of N x 1 x 256 x 256: NCHW, contiguous (a plane stride of 65,536 floats, rows of 1,024 bytes)
// The rows, in the order of the checks and of the timing, the same in every run:
//   fkl_loop   FKL as on NVIDIA (e19_batched_box.cu), op for op: per crop one fk::executeLinearFilter (LinearFilterDPPDetails<float> with its
//              defaults), one kernel. Image read: fk::PerThreadRead<_2D, uchar3> of the crop view (Ptr2D::crop2D), then
//              fk::ColorConversion<COLOR_RGB2GRAY, uchar3, float> (fk::RGB2Gray<uchar3, float>: the luminance rounded to the nearest integer,
//              ties to even, held in a float; see fklReference). Coefficient: fk::ConstantFilterRead<float> with 1 / 9 (scale / (3 * 3), as
//              fk::executeBoxFilter computes it). Multiply and accumulate: fk::Mul and fk::Add on float (UnaryType). Write: fk::Div<float> by
//              255, fk::Sub<float> of 0.45, fk::Div<float> by 0.225, then fk::PerThreadWrite<_3D, float> of the view of plane i of FKL's tensor
//              (fk::Tensor::crop at (0, 0, i), one plane). The box is not rounded to 8 bits, as on NVIDIA. N calls, N kernels per frame.
//   fkl_batch  one fk::executeBoxBlur<1> per frame (MorphologyDPP, already batched at 24478f0: one grid plane per plane of its read), one
//              kernel. Input: fk::PerThreadRead<_2D, uchar3>::build of a std::array of the N crop views (a BatchRead, z = the crop) .then()
//              the same grey operation. Details: MorphologyDPPDetails<float> with its defaults, 256 x 256, mask 3 x 3, anchor (1, 1). The box
//              blur sums the window in float and multiplies the sum by 1 / 9 in float (morphology_detail::boxBlurReducers). Write: the same
//              Div, Sub, Div, then fk::TensorWrite<float> of the whole tensor, which takes the plane from Point.z (no per-plane object).
//   rpp        RPP 3.2.0, HIP back-end, its own buffers and no FKL object. One handle (rppCreate, batch N) and every buffer, ROI and parameter
//              array made once (they depend on the crop size only). Every call: N hipMemcpy2DAsync of the crops (from the detection table)
//              into a dense N x 256 x 256 x 3 U8 NHWC batch (the batch building, in the call), then
//                rppt_color_to_greyscale (RGBtype) into U8 NCHW c = 1 (offsetInBytes 16: the box filter's source needs at least 12),
//                rppt_box_filter (3, REPLICATE, the XYWH ROI {0, 0, 256, 256} of every image) into U8 NCHW c = 1,
//                rppt_crop_mirror_normalize U8 to F32 NCHW (dst = v * multiplier + offset: multiplier 1 / 57.375f, offset -2.0f, mirror 0,
//                  the same ROIs) into RPP's own N x 1 x 256 x 256 tensor.
//              The ROI and parameter arrays are in HIP memory, written once (XYWH ROIs are not converted by these calls). In the check run
//              every RPP call is followed by hipGetLastError and hipStreamSynchronize (RPP returns RPP_SUCCESS when a launch fails).
// The kernels per frame, by reading: fkl_loop N, fkl_batch 1, rpp N copies + 3 kernels.
//
// The checks (e19_batched_box_checks_GPU_AMD.csv), over all N x 256 x 256 values (1,966,080 at N = 30). Each row is called once after every buffer it
// writes is filled with 0xFF bytes (an unwritten tensor value reads as NaN):
//   rpp        grey   against RPP's rule: (R * 0.299f + G * 0.587f) + B * 0.114f in float, then rounded to the nearest integer, ties to even
//                     (V_CVT_PK_U8_F32, results/rx9060/e18-box-phase1/inventory/cvt-test), saturated. If that rule does not give 0 differing
//                     values, the other evaluation orders below are compared and the one with the fewest is reported (the rule found by
//                     comparison); the call is not changed. Required: one rule with 0
//              box    against round(sum / 9) of its own grey image, border REPLICATE at each crop's edges, exactly. Required: 0
//              float  against (v / 255 - 0.45) / 0.225 in double, v its own box value: at most 1e-5 and every value finite. Required
//   fkl_loop, fkl_batch: against FKL's pipeline in double (fklReference): the grey as FKL rounds it, the box as each DPP computes it (the
//              coefficient 1 / 9 in float times the exact sum of the window; for the box blur the float sum is exact, as for the loop's
//              products and sums of integers up to 255 within the stated bound), then the conversion. Required: within 1e-5 and finite. FKL's
//              grey depends on how amdclang++ contracts the luminance into FMAs, so the reference is computed for every evaluation order below
//              and the order with no value beyond 1e-5 is the one FKL's kernel uses (reported; compared with the ISA)
//   reported   rpp against fkl_loop and fkl_batch against fkl_loop: the largest absolute difference
// The evaluation orders of the luminance (GreyOrder): no FMA, left to right (each product and sum rounded); t = 0.587f G, fma(0.299f, R, t),
// fma(0.114f, B, t) (nvcc's, found on NVIDIA); t = 0.299f R, fma(0.587f, G, t), fma(0.114f, B, t); (0.299f R + 0.587f G) rounded, then
// fma(0.114f, B, .); and the exact weighted sum.
// E19_DUMP=<folder>: the checks also write rpp's grey, box and tensor and the FKL rows' tensors there as raw bytes (crop by crop, row by row).
//
// The timing (e19_batched_box_GPU_AMD.csv), two intervals per row, in the order fkl_loop, fkl_batch, rpp, the same in every run, with the E18 RX
// 9060 protocol (rpp/e18_box.h): the timing file is opened first, then one gpuSpin of E19_CLOCK_WARMUP_MS (default 2000), and sclk is read
// before and after every interval, outside the timed calls:
//   whole_call   fklb::timeIt (common/hip_timing.h): events around one call (the whole frame), 100 calls after one warm-up call
//   kernel_only  fklb::timePreparedLaunch with gpuSpin (E19_SPIN_US, default 5000) as beforeLaunch and nothing to prepare: the whole call
//                behind one spin
//   Each interval of each row is in a ROCTx range ("whole_call:<row>", "kernel_only:<row>").
// E19_ONLY=a,b: only these rows. E19_NO_TIMING=1: the checks only. E19_PROFILE=1: for rocprofv3, no checks and no timing: one warm-up call
// of every selected row, then three calls of each, every call in a ROCTx range named after the row and followed by a synchronisation.
#include <common/csv_writer.h>
#include <common/hip_timing.h>
#include <common/hip_wallclock.h>
#include <common/main.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rpp.h>
#include <rppdefs.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace e19amd {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_W = 256, CROP_H = 256;
// The 16 numbers of crops of the sweep, and the crop table's shifts (see the header)
constexpr std::array<int, 16> SWEEP_CROPS = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};
constexpr int MAX_CROPS = 150, E19_CROPS = 30;
constexpr std::array<int, 5> SX = {0, 320, 0, 320, 192}, SY = {0, 0, 200, 200, 100};
constexpr int BOX_W = 3, BOX_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr size_t PLANE = static_cast<size_t>(CROP_W) * CROP_H;   // 65,536 values per crop: the plane stride of the tensor
constexpr size_t SRC_BYTES = static_cast<size_t>(SRC_W) * SRC_H * 3;
constexpr double FLOAT_TOLERANCE = 1e-5;   // rpp's float against the double conversion of its own box
constexpr double FKL_TOLERANCE = 1e-5;     // the FKL rows against fklReference (the bound of e19_batched_box.cu: about 1.7e-6)
constexpr float SCALE = 1.f;               // FKL's box coefficient is SCALE / (BOX_W * BOX_H), as fk::executeBoxFilter computes it
// RPP's buffers: the grey image is the box filter's source, which needs offsetInBytes >= 12 * (kernelSize / 2)
// (rppt_tensor_filter_augmentations.cpp:133); 16 keeps it aligned. Every RPP buffer has RPP_TAIL bytes after its images (RPP stores 8 pixels at
// a time). Every ROI array has 256 entries (RPP's ROI conversion kernels, which XYWH ROIs do not run here, rewrite up to 256).
constexpr unsigned RPP_GREY_OFFSET = 16;
constexpr size_t RPP_TAIL = 64;
constexpr size_t RPP_ROI_ENTRIES = 256;

inline bool selected(const std::string& name) {
    const char* e = std::getenv("E19_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

inline bool envIs1(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && std::string(e) == "1";
}

// The source of E4 and E18 (e18_box.cu): an LCG with seed 12345, bytes 31..24, 23..16, 15..8 of every state, row by row
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

struct Detection {
    int x, y;
};

// The detections of the frame, the crop table: crop i at x = 64 + 600 c + SX[k], y = 64 + 400 r + SY[k], c = i mod 6, r = (i div 6) mod 5,
// k = i div 30
inline std::vector<Detection> detectionsOfTheFrame(const int crops) {
    std::vector<Detection> d;
    for (int i = 0; i < crops; ++i) d.push_back({64 + 600 * (i % 6) + SX[i / 30], 64 + 400 * ((i / 6) % 5) + SY[i / 30]});
    return d;
}

// The three assertions on the table of all 150 crops: crops 0..29 are E19's (x = 64 + 600 (i mod 6), y = 64 + 400 (i div 6)), every crop lies
// inside the frame, and the 150 positions are distinct
inline void checkTheCropTable() {
    const std::vector<Detection> d = detectionsOfTheFrame(MAX_CROPS);
    for (int i = 0; i < E19_CROPS; ++i) {
        if (d[i].x != 64 + 600 * (i % 6) || d[i].y != 64 + 400 * (i / 6)) throw std::runtime_error("the crop table: crop " + std::to_string(i) + " is not E19's");
    }
    for (int i = 0; i < MAX_CROPS; ++i) {
        if (d[i].x < 0 || d[i].y < 0 || d[i].x + CROP_W > SRC_W || d[i].y + CROP_H > SRC_H) {
            throw std::runtime_error("the crop table: crop " + std::to_string(i) + " is not inside the frame");
        }
        for (int j = 0; j < i; ++j) {
            if (d[i].x == d[j].x && d[i].y == d[j].y) throw std::runtime_error("the crop table: crops " + std::to_string(j) + " and " + std::to_string(i) + " are at one position");
        }
    }
}

// ---- The host references
// The evaluation orders of the luminance R * 0.299f + G * 0.587f + B * 0.114f (see the header)
enum class GreyOrder { NO_FMA_LEFT_TO_RIGHT, FMA_G_R_B, FMA_R_G_B, SUM_RG_THEN_FMA_B, EXACT };
inline const char* name(const GreyOrder o) {
    switch (o) {
    case GreyOrder::NO_FMA_LEFT_TO_RIGHT: return "(0.299f R + 0.587f G) + 0.114f B in float, no FMA";
    case GreyOrder::FMA_G_R_B: return "t = 0.587f G; t = fma(0.299f, R, t); t = fma(0.114f, B, t)";
    case GreyOrder::FMA_R_G_B: return "t = 0.299f R; t = fma(0.587f, G, t); t = fma(0.114f, B, t)";
    case GreyOrder::SUM_RG_THEN_FMA_B: return "t = 0.299f R + 0.587f G (rounded); t = fma(0.114f, B, t)";
    case GreyOrder::EXACT: return "0.299f R + 0.587f G + 0.114f B exactly";
    }
    return "?";
}
const std::vector<GreyOrder> ALL_ORDERS = {GreyOrder::NO_FMA_LEFT_TO_RIGHT, GreyOrder::FMA_G_R_B, GreyOrder::FMA_R_G_B, GreyOrder::SUM_RG_THEN_FMA_B,
                                           GreyOrder::EXACT};

// The luminance under one evaluation order; volatile keeps every product and sum a float, std::fma rounds once
inline double luminance(const int r, const int g, const int b, const GreyOrder order) {
    const float R = static_cast<float>(r), G = static_cast<float>(g), B = static_cast<float>(b);
    switch (order) {
    case GreyOrder::NO_FMA_LEFT_TO_RIGHT: {
        volatile float red = 0.299f * R;
        volatile float green = 0.587f * G;
        volatile float blue = 0.114f * B;
        volatile float sum = red + green;
        sum = sum + blue;
        return sum;
    }
    case GreyOrder::FMA_G_R_B: {
        volatile float t = 0.587f * G;
        t = std::fma(0.299f, R, t);
        t = std::fma(0.114f, B, t);
        return t;
    }
    case GreyOrder::FMA_R_G_B: {
        volatile float t = 0.299f * R;
        t = std::fma(0.587f, G, t);
        t = std::fma(0.114f, B, t);
        return t;
    }
    case GreyOrder::SUM_RG_THEN_FMA_B: {
        volatile float red = 0.299f * R;
        volatile float green = 0.587f * G;
        volatile float t = red + green;
        t = std::fma(0.114f, B, t);
        return t;
    }
    case GreyOrder::EXACT: return static_cast<double>(0.299f) * r + static_cast<double>(0.587f) * g + static_cast<double>(0.114f) * b;
    }
    return 0;
}

// The grey of the N crops (crop by crop, row by row): the luminance rounded to the nearest integer, ties to even (std::nearbyint in the
// default rounding mode: RPP's V_CVT_PK_U8_F32 and FKL's __float2int_rn), saturated to [0, 255]
inline std::vector<unsigned char> greyReference(const std::vector<unsigned char>& frame, const std::vector<Detection>& det, const GreyOrder order) {
    std::vector<unsigned char> grey(PLANE * det.size());
    for (size_t i = 0; i < det.size(); ++i) {
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const unsigned char* p = frame.data() + ((static_cast<size_t>(det[i].y) + y) * SRC_W + det[i].x + x) * 3;
                const double v = std::nearbyint(luminance(p[0], p[1], p[2], order));
                grey[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = static_cast<unsigned char>(std::min(255.0, std::max(0.0, v)));
            }
        }
    }
    return grey;
}

// The exact 3 x 3 sums of each crop's 256 x 256 one-channel image, border REPLICATE at the crop's edges
inline std::vector<int> boxSums(const std::vector<unsigned char>& image) {
    std::vector<int> sums(image.size());
    for (size_t i = 0; i < image.size() / PLANE; ++i) {
        const unsigned char* crop = image.data() + i * PLANE;
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                int sum = 0;
                for (int ky = 0; ky < BOX_H; ++ky) {
                    for (int kx = 0; kx < BOX_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), CROP_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), CROP_H - 1);
                        sum += crop[static_cast<size_t>(sy) * CROP_W + sx];
                    }
                }
                sums[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = sum;
            }
        }
    }
    return sums;
}

// FKL's pipeline in double, for one grey order: the grey as FKL rounds it, then the coefficient 1 / 9 in float times the exact sum of the
// window, then (box / 255 - 0.45) / 0.225. The same reference serves fkl_loop (9 products of the coefficient, summed) and fkl_batch (the sum,
// then one product): both are within the bound of e19_batched_box.cu of it
inline std::vector<double> fklReference(const std::vector<unsigned char>& grey) {
    const double coefficient = SCALE / static_cast<float>(BOX_W * BOX_H);
    const std::vector<int> sums = boxSums(grey);
    std::vector<double> out(sums.size());
    for (size_t k = 0; k < sums.size(); ++k) out[k] = (coefficient * sums[k] / 255.0 - 0.45) / 0.225;
    return out;
}

struct Difference {
    size_t differ = 0, nonFinite = 0;
    double maxAbs = 0;
};

inline Difference compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    Difference d;
    for (size_t k = 0; k < a.size(); ++k) {
        const int diff = std::abs(static_cast<int>(a[k]) - static_cast<int>(b[k]));
        if (diff == 0) continue;
        ++d.differ;
        d.maxAbs = std::max(d.maxAbs, static_cast<double>(diff));
    }
    return d;
}

// A tensor against a reference in double: the values farther than the tolerance, the non-finite ones and the largest difference
inline Difference compareToReference(const std::vector<float>& out, const std::vector<double>& reference, const double tolerance) {
    Difference d;
    for (size_t k = 0; k < out.size(); ++k) {
        if (!std::isfinite(out[k])) {
            ++d.nonFinite;
            continue;
        }
        const double diff = std::abs(static_cast<double>(out[k]) - reference[k]);
        d.maxAbs = std::max(d.maxAbs, diff);
        d.differ += diff > tolerance ? 1 : 0;
    }
    return d;
}

inline Difference compareFloats(const std::vector<float>& a, const std::vector<float>& b) {
    Difference d;
    for (size_t k = 0; k < a.size(); ++k) {
        if (!std::isfinite(a[k]) || !std::isfinite(b[k])) {
            ++d.nonFinite;
            continue;
        }
        d.maxAbs = std::max(d.maxAbs, std::abs(static_cast<double>(a[k]) - static_cast<double>(b[k])));
    }
    return d;
}

inline void writeRaw(const std::string& path, const void* data, const size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("cannot write " + path);
}

// The GPU spin of the kernel-only interval, as in rpp/e18_box.h: one thread waits on the GPU's constant-rate wall clock (wall_clock64) and
// touches no memory. `ticks` = microseconds * the wall clock rate (hipDeviceAttributeWallClockRate, kHz) / 1000
__global__ void gpuSpin(const unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
}

inline unsigned long long spinUs() {
    const char* e = std::getenv("E19_SPIN_US");
    return std::strtoull(e != nullptr ? e : "5000", nullptr, 10);
}

inline unsigned long long clockWarmupMs() {
    const char* e = std::getenv("E19_CLOCK_WARMUP_MS");
    return std::strtoull(e != nullptr ? e : "2000", nullptr, 10);
}

inline long long realtimeNs() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

// One row: its stream, its call (the whole frame), the fill of its buffers with 0xFF bytes, and the download of its outputs
struct Row {
    std::string version;
    hipStream_t stream;
    std::function<void()> call;
    std::function<void()> fill;
    std::function<void(std::vector<float>& tensor)> download;
};

// E19's launch() for N = CROPS crops
template <int CROPS>
int runCrops() {
    using fk::uchar3;
    static_assert(RPP_ROI_ENTRIES >= CROPS, "every ROI array has RPP_ROI_ENTRIES entries, one per image at least");
    constexpr size_t VALUES = PLANE * CROPS;
    const bool profile = envIs1("E19_PROFILE");
    const bool timing = !profile && !envIs1("E19_NO_TIMING");
    const char* dumpDir = std::getenv("E19_DUMP");
    std::cout << "E19 batched box (AMD), N = " << CROPS << " crops: "
              << (profile ? "the profile mode (E19_PROFILE=1): calls in ROCTx ranges, no checks, no timing"
                          : (timing ? "the checks, then the timing" : "the checks only (E19_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> frame = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    // Every call reads the N positions from this table and builds every crop view and copy from them
    const std::vector<Detection> detections = detectionsOfTheFrame(CROPS);
    int device = 0, wallClockKHz = 0;
    FKLB_HIP_CHECK(hipGetDevice(&device));
    FKLB_HIP_CHECK(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, device));
    const unsigned long long spinTicks = spinUs() * static_cast<unsigned long long>(wallClockKHz) / 1000ull;
    bool checking = !profile;   // hipGetLastError and hipStreamSynchronize after every RPP call in the check run

    // ---- The host references, computed before the first GPU call
    std::vector<std::vector<unsigned char>> greyRefs;   // per GreyOrder
    std::vector<std::vector<double>> fklRefs;           // per GreyOrder
    if (!profile) {
        for (const GreyOrder o : ALL_ORDERS) {
            greyRefs.push_back(greyReference(frame, detections, o));
            fklRefs.push_back(fklReference(greyRefs.back()));
        }
    }
    std::vector<Row> rows;

    // ---- FKL: its own stream, its own copy of the frame (fk::Ptr2D) and its own tensor (fk::Tensor<float>, 256 x 256 x N planes)
    hipStream_t fklStream;
    FKLB_HIP_CHECK(hipStreamCreate(&fklStream));
    fk::Stream_<fk::ParArch::GPU_AMD> fkStream(fklStream);
    fk::Ptr2D<uchar3> fkFrame(SRC_W, SRC_H);
    FKLB_HIP_CHECK(hipMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, frame.data(), hostRow, hostRow, SRC_H, hipMemcpyHostToDevice));
    fk::Tensor<float> loopTensor(CROP_W, CROP_H, CROPS, 1), batchTensor(CROP_W, CROP_H, CROPS, 1);
    for (const fk::Tensor<float>* t : {&loopTensor, &batchTensor}) {
        if (t->ptr().dims.pitch != CROP_W * sizeof(float) || t->ptr().dims.plane_pitch != PLANE * sizeof(float)) {
            throw std::runtime_error("FKL: the tensor is not contiguous");
        }
    }
    using FklGrey = fk::ColorConversion<fk::ColorConversionCodes::COLOR_RGB2GRAY, uchar3, float>;
    // fkl_loop: per crop, one executeLinearFilter; every call builds the N crop views, the IOps and the details
    using LoopDetails = fk::LinearFilterDPPDetails<float>;   // the default tile (16 x 8) and largest kernel (7 x 7)
    using LoopMultiply = fk::Mul<float, float, float, fk::UnaryType>;
    using LoopAccumulate = fk::Add<float, float, float, fk::UnaryType>;
    const auto callLoop = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const fk::Ptr2D<uchar3> crop = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(FklGrey::build());
            const LoopDetails details{CROP_W, CROP_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
            const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
            const auto coefficient = fk::ConstantFilterRead<float>::build(SCALE / n);
            const fk::Ptr<fk::ND::_3D, float> plane = loopTensor.crop(fk::Point{0, 0, i}, fk::PtrDims<fk::ND::_3D>(CROP_W, CROP_H, 1));
            const auto output = fk::Div<float>::build(255.f)
                                    .then(fk::Sub<float>::build(0.45f))
                                    .then(fk::Div<float>::build(0.225f))
                                    .then(fk::PerThreadWrite<fk::ND::_3D, float>::build(plane));
            if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), LoopMultiply::build(), LoopAccumulate::build(), output, fkStream)) {
                throw std::runtime_error("fkl_loop: the details are not valid");
            }
        }
    };
    rows.push_back({"fkl_loop", fklStream, callLoop, [&] { FKLB_HIP_CHECK(hipMemset(loopTensor.ptr().data, 0xFF, VALUES * sizeof(float))); },
                    [&](std::vector<float>& t) { FKLB_HIP_CHECK(hipMemcpy(t.data(), loopTensor.ptr().data, VALUES * sizeof(float), hipMemcpyDeviceToHost)); }});
    // fkl_batch: one executeBoxBlur<1> per frame on the batch of the N crop views; every call builds the views, the BatchRead and the IOps
    using BatchDetails = fk::MorphologyDPPDetails<float>;   // the default tile (16 x 8) and largest mask (7 x 7)
    const auto callBatch = [&] {
        std::array<fk::Ptr2D<uchar3>, CROPS> crops;
        for (int i = 0; i < CROPS; ++i) crops[i] = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
        const auto input = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(FklGrey::build());
        const BatchDetails details{CROP_W, CROP_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const auto output = fk::Div<float>::build(255.f)
                                .then(fk::Sub<float>::build(0.45f))
                                .then(fk::Div<float>::build(0.225f))
                                .then(fk::TensorWrite<float>::build(batchTensor));
        if (!fk::executeBoxBlur<1>(details, input, output, fkStream)) {
            throw std::runtime_error("fkl_batch: the details are not valid");
        }
    };
    rows.push_back({"fkl_batch", fklStream, callBatch, [&] { FKLB_HIP_CHECK(hipMemset(batchTensor.ptr().data, 0xFF, VALUES * sizeof(float))); },
                    [&](std::vector<float>& t) { FKLB_HIP_CHECK(hipMemcpy(t.data(), batchTensor.ptr().data, VALUES * sizeof(float), hipMemcpyDeviceToHost)); }});

    // ---- RPP: its own stream, handle, frame, batch, intermediates, tensor, ROIs and parameter arrays, all set up once
    hipStream_t rppStream;
    FKLB_HIP_CHECK(hipStreamCreate(&rppStream));
    rppHandle_t handle{};
    FKLB_RPP_CHECK(rppCreate(&handle, CROPS, 0, rppStream, RppBackend::RPP_HIP_BACKEND));
    unsigned char *rppFrame = nullptr, *rppBatch = nullptr, *rppGrey = nullptr, *rppBox = nullptr;
    float* rppTensor = nullptr;
    const size_t batchBytes = VALUES * 3;
    FKLB_HIP_CHECK(hipMalloc(&rppFrame, SRC_BYTES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppBatch, batchBytes + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppGrey, RPP_GREY_OFFSET + VALUES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppBox, VALUES + RPP_TAIL));
    FKLB_HIP_CHECK(hipMalloc(&rppTensor, VALUES * sizeof(float) + RPP_TAIL));
    FKLB_HIP_CHECK(hipMemcpy(rppFrame, frame.data(), SRC_BYTES, hipMemcpyHostToDevice));
    RpptDesc batchDesc(CROPS, 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC);   // the dense batch of the N crops
    RpptDesc greyDesc(CROPS, 1, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NCHW);    // written by the grey call, read by the box call
    greyDesc.offsetInBytes = RPP_GREY_OFFSET;
    RpptDesc boxDesc(CROPS, 1, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NCHW);
    RpptDesc tensorDesc(CROPS, 1, CROP_H, CROP_W, RpptDataType::F32, RpptLayout::NCHW);
    // The ROIs (XYWH {0, 0, 256, 256} for every image) and the parameter arrays of crop_mirror_normalize (one entry per image and channel), in
    // HIP memory, written once: they depend on the crop size only
    RpptROI* rppRoi = nullptr;
    Rpp32f *rppOffset = nullptr, *rppMultiplier = nullptr;
    Rpp32u* rppMirror = nullptr;
    FKLB_HIP_CHECK(hipMalloc(&rppRoi, sizeof(RpptROI) * RPP_ROI_ENTRIES));
    FKLB_HIP_CHECK(hipMalloc(&rppOffset, sizeof(Rpp32f) * CROPS));
    FKLB_HIP_CHECK(hipMalloc(&rppMultiplier, sizeof(Rpp32f) * CROPS));
    FKLB_HIP_CHECK(hipMalloc(&rppMirror, sizeof(Rpp32u) * CROPS));
    {
        std::vector<RpptROI> rois(RPP_ROI_ENTRIES);
        std::memset(rois.data(), 0, sizeof(RpptROI) * RPP_ROI_ENTRIES);
        for (int i = 0; i < CROPS; ++i) rois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
        FKLB_HIP_CHECK(hipMemcpy(rppRoi, rois.data(), sizeof(RpptROI) * RPP_ROI_ENTRIES, hipMemcpyHostToDevice));
        const std::vector<Rpp32f> offset(CROPS, -2.0f), multiplier(CROPS, 1.f / 57.375f);
        const std::vector<Rpp32u> mirror(CROPS, 0u);
        FKLB_HIP_CHECK(hipMemcpy(rppOffset, offset.data(), sizeof(Rpp32f) * CROPS, hipMemcpyHostToDevice));
        FKLB_HIP_CHECK(hipMemcpy(rppMultiplier, multiplier.data(), sizeof(Rpp32f) * CROPS, hipMemcpyHostToDevice));
        FKLB_HIP_CHECK(hipMemcpy(rppMirror, mirror.data(), sizeof(Rpp32u) * CROPS, hipMemcpyHostToDevice));
    }
    // rpp: the batch building (N copies from the detection table) and the three calls
    const auto callRpp = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const unsigned char* crop = rppFrame + (static_cast<size_t>(detections[i].y) * SRC_W + detections[i].x) * 3;
            FKLB_HIP_CHECK(hipMemcpy2DAsync(rppBatch + i * PLANE * 3, static_cast<size_t>(CROP_W) * 3, crop, hostRow, static_cast<size_t>(CROP_W) * 3,
                                            CROP_H, hipMemcpyDeviceToDevice, rppStream));
        }
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
        FKLB_RPP_CHECK(rppt_color_to_greyscale(rppBatch, &batchDesc, rppGrey, &greyDesc, RpptSubpixelLayout::RGBtype, handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
        FKLB_RPP_CHECK(rppt_box_filter(rppGrey, &greyDesc, rppBox, &boxDesc, BOX_W, RpptImageBorderType::REPLICATE, rppRoi, RpptRoiType::XYWH, handle,
                                       RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
        FKLB_RPP_CHECK(rppt_crop_mirror_normalize(rppBox, &boxDesc, rppTensor, &tensorDesc, rppOffset, rppMultiplier, rppMirror, rppRoi, RpptRoiType::XYWH,
                                                  handle, RppBackend::RPP_HIP_BACKEND));
        if (checking) {
            FKLB_HIP_CHECK(hipGetLastError());
            FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        }
    };
    const auto fillRpp = [&] {
        FKLB_HIP_CHECK(hipMemset(rppBatch, 0xFF, batchBytes));
        FKLB_HIP_CHECK(hipMemset(rppGrey, 0xFF, RPP_GREY_OFFSET + VALUES));
        FKLB_HIP_CHECK(hipMemset(rppBox, 0xFF, VALUES));
        FKLB_HIP_CHECK(hipMemset(rppTensor, 0xFF, VALUES * sizeof(float)));
    };
    rows.push_back({"rpp", rppStream, callRpp, fillRpp,
                    [&](std::vector<float>& t) { FKLB_HIP_CHECK(hipMemcpy(t.data(), rppTensor, VALUES * sizeof(float), hipMemcpyDeviceToHost)); }});

    // ---- The profile mode: a warm-up call of every selected row, then three calls of each in ROCTx ranges named after it
    if (profile) {
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            row.call();
            FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
        }
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            for (int k = 0; k < 3; ++k) {
                roctxRangePushA(row.version.c_str());
                row.call();
                roctxRangePop();
                FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
            }
        }
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
        std::cout << "E19 profile mode done" << std::endl;
        return 0;
    }

    // ---- The checks: one call of every row after its buffers are filled with 0xFF bytes
    bool allOk = true;
    {
        std::vector<std::vector<float>> tensors;
        for (const Row& row : rows) {
            row.fill();
            FKLB_HIP_CHECK(hipDeviceSynchronize());
            row.call();
            FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
            FKLB_HIP_CHECK(hipGetLastError());
            tensors.emplace_back(VALUES);
            row.download(tensors.back());
        }
        const std::vector<float>& loopOut = tensors[0];
        const std::vector<float>& batchOut = tensors[1];
        const std::vector<float>& rppOut = tensors[2];
        std::vector<unsigned char> rppGreyHost(VALUES), rppBoxHost(VALUES);
        FKLB_HIP_CHECK(hipMemcpy(rppGreyHost.data(), rppGrey + RPP_GREY_OFFSET, VALUES, hipMemcpyDeviceToHost));
        FKLB_HIP_CHECK(hipMemcpy(rppBoxHost.data(), rppBox, VALUES, hipMemcpyDeviceToHost));
        if (dumpDir != nullptr) {
            const std::string d(dumpDir);
            writeRaw(d + "/rpp_grey.u8", rppGreyHost.data(), VALUES);
            writeRaw(d + "/rpp_box.u8", rppBoxHost.data(), VALUES);
            writeRaw(d + "/rpp_tensor.f32", rppOut.data(), VALUES * sizeof(float));
            writeRaw(d + "/fkl_loop_tensor.f32", loopOut.data(), VALUES * sizeof(float));
            writeRaw(d + "/fkl_batch_tensor.f32", batchOut.data(), VALUES * sizeof(float));
        }

        fklb::CsvWriter checks("e19_batched_box_checks_GPU_AMD.csv", "E19 batched box (AMD): checks",
                               {"crops", "row", "stage", "against", "values", "values_differing_or_beyond", "max_abs_diff", "non_finite", "criterion", "ok"});
        const auto row = [&](const char* r, const char* stage, const std::string& against, const Difference& d, const std::string& criterion, const int ok) {
            checks.row(CROPS, r, stage, against, VALUES, d.differ, d.maxAbs, d.nonFinite, criterion, ok);
            std::cout << r << " " << stage << " vs " << against << ": " << d.differ << " differing/beyond, max " << d.maxAbs << ", non-finite " << d.nonFinite
                      << " [" << criterion << (ok == 1 ? ": ok" : (ok == 0 ? ": FAIL" : "")) << "]" << std::endl;
        };
        // rpp grey: RPP's source rule first, then the other orders (the rule found by comparison)
        size_t bestOrder = 0, bestDiffer = VALUES + 1;
        for (size_t o = 0; o < ALL_ORDERS.size(); ++o) {
            const Difference d = compare8u(rppGreyHost, greyRefs[o]);
            const bool sourceRule = ALL_ORDERS[o] == GreyOrder::NO_FMA_LEFT_TO_RIGHT;
            row("rpp", "grey", std::string(name(ALL_ORDERS[o])) + ", ties to even", d, sourceRule ? "RPP's source rule" : "candidate", -1);
            if (d.differ < bestDiffer) {
                bestDiffer = d.differ;
                bestOrder = o;
            }
        }
        const bool rppGreyOk = bestDiffer == 0;
        row("rpp", "grey", std::string("the rule with the fewest: ") + name(ALL_ORDERS[bestOrder]), compare8u(rppGreyHost, greyRefs[bestOrder]),
            "one rule with 0 differing values", rppGreyOk ? 1 : 0);
        allOk = allOk && rppGreyOk;
        // rpp box: round(sum / 9) of its own grey, exactly
        {
            const std::vector<int> sums = boxSums(rppGreyHost);
            std::vector<unsigned char> expected(VALUES);
            for (size_t k = 0; k < VALUES; ++k) expected[k] = static_cast<unsigned char>((sums[k] + 4) / 9);
            const Difference d = compare8u(rppBoxHost, expected);
            row("rpp", "box", "round(sum / 9) of its own grey, REPLICATE", d, "0 differing values", d.differ == 0 ? 1 : 0);
            allOk = allOk && d.differ == 0;
            std::vector<unsigned char> floorBox(VALUES);
            for (size_t k = 0; k < VALUES; ++k) floorBox[k] = static_cast<unsigned char>(sums[k] / 9);
            row("rpp", "box", "floor(sum / 9) of its own grey (informational)", compare8u(rppBoxHost, floorBox), "informational", -1);
        }
        // rpp float: against the double conversion of its own box
        {
            std::vector<double> expected(VALUES);
            for (size_t k = 0; k < VALUES; ++k) expected[k] = (static_cast<double>(rppBoxHost[k]) / 255.0 - 0.45) / 0.225;
            const Difference d = compareToReference(rppOut, expected, FLOAT_TOLERANCE);
            const bool ok = d.differ == 0 && d.nonFinite == 0;
            row("rpp", "float", "(v / 255 - 0.45) / 0.225 in double, v its own box", d, "within 1e-5, all finite", ok ? 1 : 0);
            allOk = allOk && ok;
        }
        // fkl_loop and fkl_batch: against FKL's pipeline in double, for every grey order; the order with no value beyond 1e-5 is FKL's
        for (const auto& fkl : {std::make_pair("fkl_loop", &loopOut), std::make_pair("fkl_batch", &batchOut)}) {
            size_t best = 0, bestBeyond = VALUES + 1;
            for (size_t o = 0; o < ALL_ORDERS.size(); ++o) {
                const Difference d = compareToReference(*fkl.second, fklRefs[o], FKL_TOLERANCE);
                row(fkl.first, "float", std::string("fklReference, grey ") + name(ALL_ORDERS[o]), d, "candidate", -1);
                if (d.differ + d.nonFinite < bestBeyond) {
                    bestBeyond = d.differ + d.nonFinite;
                    best = o;
                }
            }
            const Difference d = compareToReference(*fkl.second, fklRefs[best], FKL_TOLERANCE);
            const bool ok = d.differ == 0 && d.nonFinite == 0;
            row(fkl.first, "float", std::string("fklReference, the grey order with the fewest: ") + name(ALL_ORDERS[best]), d,
                "one order within 1e-5, all finite", ok ? 1 : 0);
            allOk = allOk && ok;
        }
        // reported
        row("rpp", "float", "fkl_loop", compareFloats(rppOut, loopOut), "reported", -1);
        row("fkl_batch", "float", "fkl_loop", compareFloats(batchOut, loopOut), "reported", -1);
        std::cout << (allOk ? "E19 checks passed" : "E19 CHECK FAILED") << std::endl;
    }
    checking = false;

    // ---- The timing, with the E18 RX 9060 protocol: the file first, then the clock warm-up, then the rows in order
    if (timing && allOk) {
        fklb::CsvWriter t("e19_batched_box_GPU_AMD.csv", "E19 batched box (AMD): whole-call and kernel-only times",
                          {"crops", "version", "interval", "mean_ms", "min_ms", "host_prepare_mean_ms", "host_prepare_min_ms", "iterations", "spin_us",
                           "sclk_before_mhz", "sclk_after_mhz"},
                          "clock warm-up: one gpuSpin of " + std::to_string(clockWarmupMs()) +
                              " ms after this file's header and right before the timing (the sclk before and after it is in the console)");
        const double sclkBeforeWarmup = fklb::amdGpuClockMHz();
        const long long warmupStartNs = realtimeNs();
        gpuSpin<<<1, 1, 0, rppStream>>>(clockWarmupMs() * static_cast<unsigned long long>(wallClockKHz));
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_HIP_CHECK(hipDeviceSynchronize());
        const double sclkAfterWarmup = fklb::amdGpuClockMHz();
        const long long timingStartNs = realtimeNs();
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            fklb::Timing whole, gpu, host;
            const hipStream_t stream = row.stream;
            FKLB_HIP_CHECK(hipDeviceSynchronize());
            const double s0 = fklb::amdGpuClockMHz();
            roctxRangePushA(("whole_call:" + row.version).c_str());
            fklb::timeIt(stream, row.call, whole);
            roctxRangePop();
            FKLB_HIP_CHECK(hipDeviceSynchronize());
            const double s1 = fklb::amdGpuClockMHz();
            const double s2 = fklb::amdGpuClockMHz();
            roctxRangePushA(("kernel_only:" + row.version).c_str());
            fklb::timePreparedLaunch(
                stream, [] { return 0; }, [&, stream] { gpuSpin<<<1, 1, 0, stream>>>(spinTicks); }, [&](int) { row.call(); }, gpu, host);
            roctxRangePop();
            const double s3 = fklb::amdGpuClockMHz();
            t.row(CROPS, row.version, "whole_call", whole.mean, whole.min, "", "", fklb::ITERATIONS, "", s0, s1);
            t.row(CROPS, row.version, "kernel_only", gpu.mean, gpu.min, host.mean, host.min, fklb::ITERATIONS, spinUs(), s2, s3);
            std::cout << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                      << gpu.min << "); sclk " << s0 << " / " << s1 << " / " << s2 << " / " << s3 << " MHz" << std::endl;
        }
        const long long timingEndNs = realtimeNs();
        std::cout << "clock warm-up: one gpuSpin of " << clockWarmupMs() << " ms; sclk " << sclkBeforeWarmup << " MHz before it, " << sclkAfterWarmup
                  << " MHz after it" << std::endl;
        std::cout << "E19 timing (realtime ns): warm-up start " << warmupStartNs << ", timing start " << timingStartNs << ", timing end " << timingEndNs
                  << std::endl;
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_HIP_CHECK(hipDeviceSynchronize());
    FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
    for (void* p : {static_cast<void*>(rppFrame), static_cast<void*>(rppBatch), static_cast<void*>(rppGrey), static_cast<void*>(rppBox),
                    static_cast<void*>(rppTensor), static_cast<void*>(rppRoi), static_cast<void*>(rppOffset), static_cast<void*>(rppMultiplier),
                    static_cast<void*>(rppMirror)}) {
        FKLB_HIP_CHECK(hipFree(p));
    }
    FKLB_HIP_CHECK(hipStreamDestroy(rppStream));
    FKLB_HIP_CHECK(hipStreamDestroy(fklStream));
    return allOk ? 0 : 1;
}

// E19_CROPS=<N>: run N crops, N one of SWEEP_CROPS; any other value (or none) stops the program before any GPU call
template <size_t... K>
int runSelected(const int crops, std::index_sequence<K...>) {
    int result = -1;
    static_cast<void>(((crops == SWEEP_CROPS[K] ? (result = runCrops<SWEEP_CROPS[K]>(), true) : false) || ...));
    return result;
}
}  // namespace e19amd

int launch() {
    using namespace e19amd;
    checkTheCropTable();
    const char* e = std::getenv("E19_CROPS");
    const std::string value = e != nullptr ? e : "";
    int crops = -1;
    for (const int n : SWEEP_CROPS) {
        if (value == std::to_string(n)) crops = n;
    }
    if (crops < 0) {
        std::cerr << "E19 crop sweep: E19_CROPS must be one of 1, 10, 20, ..., 150 (got \"" << value << "\"); not started" << std::endl;
        return 2;
    }
    return runSelected(crops, std::make_index_sequence<SWEEP_CROPS.size()>{});
}
