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

// E19 batched box (NVIDIA): the 3 x 3 box filter inside a batched pipeline, with a step before it and a step after it, on 30 crops of one
// frame. OpenCV-CUDA and NPP, one call per step and per crop: neither library has a batched call for any of these steps (OpenCV-CUDA has no
// batched call at all; NPP's batched calls are its resize and its colour twist, and nppiColorTwistBatch keeps the channel count, so it is not
// a grey conversion), so each library loops over the crops. FKL as it runs at the pin 24478f0, in two rows. fkl_loop: its linear filter
// (LinearFilterDPP) computes one image per call, so it loops too, with the whole pipeline of a crop in one fused call (one kernel per crop);
// it is the reference that a batched LinearFilterDPP must match later. fkl_batch: its box blur (MorphologyDPP, fk::executeBoxBlur) takes a
// batch, so it computes the 30 crops in one call (one kernel per frame).
//
// The pipeline, per frame, the same in every library:
//   frame    one 3840 x 2160 uchar3 frame, the seeded source of E4 and E18 (an LCG with seed 12345), read as RGB: channel 0 is R
//   crops    30 crops of 256 x 256. Crop i (0..29) starts at x = 64 + 600 * (i mod 6), y = 64 + 400 * (i div 6): a made-up layout of 30
//            detections. The program treats it as a layout that changes on every frame: every call reads the positions from the detection
//            table and builds every ROI and every crop pointer from them
//   grey     RGB to grey, with the library's own call and weights: 8 bits in OpenCV-CUDA and NPP; in FKL an integer value held in a float
//            (below)
//   box      3 x 3, anchor at the centre, border REPLICATE at the crop's edges (the crop's grey image is the filter's whole input): 8 bits in
//            OpenCV-CUDA and NPP, float in FKL
//   float    out = (v / 255 - 0.45) / 0.225 = v * ALPHA + BETA, with ALPHA = 1 / 57.375 and BETA = -2
//   output   plane i of one float tensor of 30 x 1 x 256 x 256: NCHW, contiguous, a plane stride of 65,536 floats and a row stride of 1,024 bytes
// Each library reads its own copy of the same host bytes, crops its own way (OpenCV: a GpuMat ROI; NPP: a pointer into its frame and the
// frame's step; FKL: fk::Ptr2D::crop2D), and owns its intermediates and its tensor. The OpenCV-CUDA and NPP rows use no FKL object.
//   opencv_cuda  per crop:
//                  cv::cuda::cvtColor(ROI, grey_i, COLOR_RGB2GRAY): fixed point, grey = (9798 R + 19235 G + 3735 B + 2^14) >> 15
//                    (opencv_contrib cudev: RGB_to_GRAY_func = RGB2Gray<uchar, 3, 2>, with RY15, GY15, BY15 and CV_CUDEV_DESCALE in
//                    functional/detail/color_cvt.hpp)
//                  box->apply(grey_i, box_i), box = cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, Size(3, 3), Point(-1, -1), BORDER_REPLICATE):
//                    cv::cuda::copyMakeBorder pads the grey image by the kernel size (3) on every side into a buffer that the filter object keeps,
//                    then nppiFilterBox_8u_C1R_Ctx runs on the inner ROI of the padded image (cudafilters/src/filtering.cpp, NPPBoxFilter::apply)
//                  box_i.convertTo(plane_i, CV_32F, ALPHA, BETA): saturate_cast<float>(alpha * v + beta), alpha and beta cast to float
//                    (core/src/cuda/gpu_mat.cu, convertToScale). plane_i = rowRange(256 i, 256 (i + 1)) of one continuous 7680 x 256 CV_32F
//                    GpuMat (cv::cuda::createContinuous), so the GpuMat's buffer is the tensor
//   npp          per crop:
//                  nppiRGBToGray_8u_C3C1R_Ctx from the crop pointer and the frame step: 0.299 R + 0.587 G + 0.114 B in float
//                    (nppi_color_conversion.h, "RGBToGray"); the header does not give the rounding to 8 bits
//                  nppiFilterBoxBorder_8u_C1R_Ctx: source size 256 x 256, offset (0, 0), mask 3 x 3, anchor (1, 1), NPP_BORDER_REPLICATE
//                  nppiScale_8u32f_C1R_Ctx into plane i of the tensor (row step 1,024 bytes), nMin = -2, nMax = 0.55 / 0.225. Its header
//                    (nppi_data_exchange_and_initialization.h, "Scale To Higher Bit Depth") documents dst = nMin + (nMax - nMin) / (srcMax - srcMin)
//                    * (src - srcMin), with nMin and nMax as the destination range and the clamp: with the 8-bit range [0, 255] as the source
//                    range, dst = v * (nMax - nMin) / 255 + nMin = v * ALPHA + BETA in one call. The float check below tests that reading
//   fkl_loop     per crop, one fk::executeLinearFilter (FKL's linear_filter.h, its GPU_NVIDIA version), one kernel:
//                  image read   fk::PerThreadRead<_2D, uchar3> of the crop view (crop2D of the rectangle of detection i), then
//                               fk::ColorConversion<COLOR_RGB2GRAY, uchar3, float>, which is fk::RGB2Gray<uchar3, float> (color_conversion.h):
//                               (R * 0.299f) + (G * 0.587f) + (B * 0.114f) in float, then rounded to the nearest integer, ties to even, held in
//                               a float. At 24478f0 its exec() tests std::is_signed_v of the output type before std::is_floating_point_v, and
//                               std::is_signed_v<float> is true, so a float output takes the __float2int_rn branch (see fklReference). These
//                               two are the filter's read chain, as E18's FKL rows put the
//                               resize there: the DPP stages the grey values of each tile with its replicated border in shared memory, the
//                               border at the crop's own edges (readReplicate in neighborhood.h)
//                  coefficient  fk::ConstantFilterRead<float> with 1 / 9 in float (scale / (kernel width * kernel height), computed as
//                               fk::executeBoxFilter computes it), as E18's fkl_box_* rows (e18_box.cu, fkl_box_asym)
//                  multiply     fk::Mul<float, float, float, fk::UnaryType>; accumulate fk::Add<float, float, float, fk::UnaryType>
//                  write        fk::Div<float> by 255, fk::Sub<float> of 0.45, fk::Div<float> by 0.225 (correctly rounded divisions: the target is
//                               built without fast math), then fk::PerThreadWrite<_3D, float> of the view of plane i of FKL's tensor
//                               (fk::Tensor<float> of 256 x 256 x 30 planes, contiguous; the view is fk::Ptr::crop at (0, 0, i) with one plane)
//                  details      fk::LinearFilterDPPDetails<float> with its defaults (a 16 x 8 tile, a largest kernel of 7 x 7)
//                Every call builds the 30 crop views, the IOps and the details, and launches 30 kernels.
//   fkl_batch    per frame, one fk::executeBoxBlur<1> (FKL's morphology.h, MorphologyDPP, its GPU_NVIDIA version), one kernel: one block per
//                16 x 8 tile of a plane, and one grid z plane per plane of the input (morphology_detail::planesOf, the z of the input's active
//                threads: here the batch, 30). Each plane is read and written at Point{x, y, plane}, with its own replicated border:
//                  image read   a BatchRead of the 30 crop views: fk::PerThreadRead<_2D, uchar3>::build of a std::array of the 30 fkFrame.crop2D
//                               (rectangle of detection i), then .then() the grey op of fkl_loop (fk::ColorConversion<COLOR_RGB2GRAY, uchar3,
//                               float>, the same rounding to an integer). FKL's Fuser makes it a BatchRead of the 30 fused (read, grey) IOps
//                               (iop_fuser.h); its active threads are 256 x 256 x 30
//                  box          the box blur's one pass (boxBlurReducers, morphology.h:179-208): an fk::Add reducer sums the 9 values of the
//                               3 x 3 window in float, and its finalizer, fk::Mul, multiplies the sum by 1 / 9 in float
//                               (static_cast<float>(1. / (maskW * maskH)), the same float as fkl_loop's SCALE / 9.f). fkl_loop instead multiplies
//                               each of the 9 values by 1 / 9 and sums the products
//                  write        fkl_loop's fk::Div<float> by 255, fk::Sub<float> of 0.45, fk::Div<float> by 0.225, then one
//                               fk::PerThreadWrite<_3D, float> of FKL's whole tensor (the one fkl_loop writes), which takes the plane from Point.z
//                  details      fk::MorphologyDPPDetails<float> with its defaults (a 16 x 8 tile, a largest mask of 7 x 7): width 256, height
//                               256, mask 3 x 3, anchor (1, 1)
//                Every call builds the array of the 30 crop views and the BatchRead from the detections (as fkl_loop builds its views), the
//                write IOp and the details, and launches 1 kernel.
//
// The checks (e19_batched_box_checks.csv). Every comparison is over the 30 x 256 x 256 = 1,966,080 values. Each row is called once, after every
// buffer it writes is filled with 0xFF bytes, so a value that the call does not write reads as 255 (NaN in the tensor):
//   grey     each library's grey images (its own intermediates, read after the call) against a host reference with that library's weights and
//            rounding. OpenCV: its fixed point above. NPP: its documented weights as floats, with several roundings (the exact weighted sum
//            truncated or rounded half up; the sum evaluated in float, truncated or rounded), two fixed-point rules, and the FMA chain
//            0.587f G, then 0.299f R, then 0.114f B, + 0.5f and truncated; the rule with the fewest differing values is NPP's reference (the
//            rounding found by comparison). The timing needs 0 differing values
//   box      each library's box images against the exact box of its own grey images (border REPLICATE), truncated (floor(sum / 9)) and rounded
//            to the nearest (sum / 9 is never a tie): the rule with 0 differing values is the library's C1 box rounding. The timing needs one
//            rule with 0. Then against the box, with that rounding, of the host reference grey (the reference pipeline): reported
//   float    each library's tensor against (v / 255 - 0.45) / 0.225 in double, v its own box value: the largest absolute difference (the
//            timing needs at most 1e-5, far above the few float roundings of the map and far below the 0.017 of a wrong 8-bit range, and every
//            value finite); then against the same with v the reference pipeline's box value: reported
//   buffers  opencv_cuda: the data pointers of its 30 grey and 30 box images and of its tensor are the same after the call (no call reallocated
//            an output that is set once). The timing needs it
//   between  opencv_cuda against npp: the differing grey and box values and the largest float difference (reported only; the libraries'
//            grey weights differ)
//   fkl_loop has no grey or box image (the pipeline of a crop is one kernel). Its tensor against FKL's pipeline in double (fklReference): the
//            largest absolute difference (the timing needs at most FKL_TOLERANCE = 1e-5, and every value finite); then against the tensors of
//            opencv_cuda and npp (reported only)
//   fkl_batch has no grey or box image either. Its tensor against FKL's box blur in double (fklBatchReference): the largest absolute difference
//            (the timing needs at most FKL_TOLERANCE = 1e-5, and every value finite); the values still holding the fill's bytes (0xFFFFFFFF,
//            a NaN: the timing needs 0 of 1,966,080); then against fkl_loop's tensor (reported only)
// E19_DUMP=<folder>: the checks also write each row's grey images, box images and tensor there as raw bytes (crop by crop, row by row, 256
// values per row; the tensor as float32), so that another rounding rule can be tested without another run.
//
// The timing (e19_batched_box.csv): two intervals per row, in the order fkl_loop, fkl_batch, opencv_cuda, npp, the same in every run. Both hold the whole frame
// (the 30 crops):
//   whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call (E4's and E18's
//                interval). When the GPU waits for the host, the interval holds the host's work too
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin of
//                E19_SPIN_US (default 5,000 us), so the interval runs from the end of the spin to the end of the call's last kernel, if the host
//                has enqueued the whole call before the spin ends. An nsys run of the timing shows whether it has (no gap between the kernels)
//   Each interval of each row is inside an NVTX range ("whole_call:<row>", "kernel_only:<row>").
// E19_ONLY=a,b: only these rows. E19_NO_TIMING=1: the checks only. E19_PROFILE=1: every selected row is called once outside any range (a
// warm-up, so that what a first call allocates is not counted), then E19_PROFILE_CALLS times (default 10), each call in an NVTX range
// "<row>:call" inside the range "<row>"; nothing is compared or timed (to count the kernels of one call with nsys, and to see the host's
// enqueue of one call).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/linear_filter.h>
#include <fused_kernel/algorithms/image_processing/morphology.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <npp.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROPS = 30, CROP_W = 256, CROP_H = 256;
constexpr int BOX_W = 3, BOX_H = 3, ANCHOR_X = 1, ANCHOR_Y = 1;
constexpr size_t PLANE = static_cast<size_t>(CROP_W) * CROP_H;   // 65,536 values per crop: the plane stride of the tensor
constexpr size_t VALUES = PLANE * CROPS;                          // 1,966,080
constexpr double ALPHA = 1.0 / 57.375, BETA = -2.0;               // (v / 255 - 0.45) / 0.225 = v / 57.375 - 2 (0.225 * 255 = 57.375)
constexpr float NPP_MIN = -2.0f, NPP_MAX = 0.55f / 0.225f;        // the values of v = 0 and v = 255
constexpr double FLOAT_TOLERANCE = 1e-5;
constexpr float SCALE = 1.f;   // the scale argument of fk::executeBoxFilter: FKL's coefficient is SCALE / (BOX_W * BOX_H), as in E18
// fkl_loop's tolerance against its reference in double, from FKL's float arithmetic (bounds, one rounding = half a unit in the last place):
//   grey        none: the reference computes FKL's grey value exactly as the kernel does (an integer, see fklReference)
//   box         9 products (at most 28.3) and 8 sums (at most 255): at most 6.9e-5 in grey levels (the reference holds the same coefficient,
//               1 / 9 rounded to float)
//   conversion  the box error over 57.375: at most 1.2e-6; the divisions, the subtraction and the float constants 0.45f and 0.225f: at most
//               5.1e-7
// so FKL is within about 1.7e-6 of the reference (an FMA, where nvcc contracts a product and a sum, rounds once and stays inside the bound).
// 1e-5 is above that and far below the 0.0174 of one grey level after the conversion.
// fkl_batch uses the same tolerance. Its box rounds once (the sum of the 9 integer grey values is exact in float, then one product), and its
// reference makes that same rounding, so only the conversion's 5.1e-7 remains.
constexpr double FKL_TOLERANCE = 1e-5;

bool selected(const std::string& name) {
    const char* e = std::getenv("E19_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

// The source of E4 and E18 (e18_box.cu): an LCG with seed 12345, bytes 31..24, 23..16, 15..8 of every state, row by row
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

// The top-left corner of one crop in the frame
struct Detection {
    int x, y;
};

// The detections of the frame: crop i at x = 64 + 600 * (i mod 6), y = 64 + 400 * (i div 6)
std::vector<Detection> detectionsOfTheFrame() {
    std::vector<Detection> d;
    for (int i = 0; i < CROPS; ++i) d.push_back({64 + 600 * (i % 6), 64 + 400 * (i / 6)});
    return d;
}

// ---- The host references
// The rules of a grey reference: OpenCV's (FIXED15_NEAREST), and the candidates for NPP's rounding of its documented float weights
enum class GreyRule { FIXED15_NEAREST, EXACT_TRUNC, EXACT_NEAREST, FLOAT_EVAL_TRUNC, FLOAT_EVAL_NEAREST, FIXED8_NEAREST, FMA_GRB_NEAREST };

const char* name(const GreyRule r) {
    switch (r) {
    case GreyRule::FIXED15_NEAREST: return "(9798 R + 19235 G + 3735 B + 2^14) >> 15";
    case GreyRule::EXACT_TRUNC: return "0.299f R + 0.587f G + 0.114f B exactly, truncated";
    case GreyRule::EXACT_NEAREST: return "0.299f R + 0.587f G + 0.114f B exactly, rounded half up";
    case GreyRule::FLOAT_EVAL_TRUNC: return "0.299f R + 0.587f G + 0.114f B in float (each product and sum rounded, left to right), truncated";
    case GreyRule::FLOAT_EVAL_NEAREST: return "0.299f R + 0.587f G + 0.114f B in float (each product and sum rounded, left to right), + 0.5f, truncated";
    case GreyRule::FIXED8_NEAREST: return "(77 R + 150 G + 29 B + 128) >> 8";
    case GreyRule::FMA_GRB_NEAREST: return "t = 0.587f G; t = fmaf(0.299f, R, t); t = fmaf(0.114f, B, t); (int)(t + 0.5f)";
    }
    return "?";
}

// A float weight times 2^27, an integer: 0.299f, 0.587f and 0.114f are multiples of 2^-25, 2^-24 and 2^-27
int64_t weight27(const float w) {
    const double scaled = std::ldexp(static_cast<double>(w), 27);
    if (scaled != std::floor(scaled)) throw std::runtime_error("weight27: the weight is not a multiple of 2^-27");
    return static_cast<int64_t>(scaled);
}

int greyOf(const int r, const int g, const int b, const GreyRule rule) {
    static const int64_t wr = weight27(0.299f), wg = weight27(0.587f), wb = weight27(0.114f);
    int v = 0;
    switch (rule) {
    case GreyRule::FIXED15_NEAREST: v = (9798 * r + 19235 * g + 3735 * b + (1 << 14)) >> 15; break;
    case GreyRule::EXACT_TRUNC: v = static_cast<int>((wr * r + wg * g + wb * b) >> 27); break;
    case GreyRule::EXACT_NEAREST: v = static_cast<int>((wr * r + wg * g + wb * b + (int64_t{1} << 26)) >> 27); break;
    case GreyRule::FLOAT_EVAL_TRUNC:
    case GreyRule::FLOAT_EVAL_NEAREST: {
        // volatile: every product and every sum is rounded to float, and none is contracted into an FMA
        volatile float red = 0.299f * static_cast<float>(r);
        volatile float green = 0.587f * static_cast<float>(g);
        volatile float blue = 0.114f * static_cast<float>(b);
        volatile float sum = red + green;
        sum = sum + blue;
        volatile float rounded = rule == GreyRule::FLOAT_EVAL_NEAREST ? sum + 0.5f : sum;
        v = static_cast<int>(rounded);
        break;
    }
    case GreyRule::FIXED8_NEAREST: v = (77 * r + 150 * g + 29 * b + 128) >> 8; break;
    case GreyRule::FMA_GRB_NEAREST: {
        // NPP's rule, found by comparison with the raw dump of the window e19-checks (2026-11-18, results/rtx4090/e19-batched-box-phase1:
        // grey_rule_search.py): 0 of 1,966,080 values differ; of the other 79 rules of the search, the best differs at 64. std::fma rounds
        // once (a correctly rounded fused multiply-add); volatile: the product and the + 0.5f are each rounded to float
        volatile float t = 0.587f * static_cast<float>(g);
        t = std::fma(0.299f, static_cast<float>(r), t);
        t = std::fma(0.114f, static_cast<float>(b), t);
        volatile float rounded = t + 0.5f;
        v = static_cast<int>(rounded);
        break;
    }
    }
    return std::min(255, v);
}

// The grey reference of the 30 crops (crop by crop, row by row) under one rule
std::vector<unsigned char> greyReference(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections, const GreyRule rule) {
    std::vector<unsigned char> grey(VALUES);
    for (int i = 0; i < CROPS; ++i) {
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const unsigned char* p = frame.data() + ((static_cast<size_t>(detections[i].y) + y) * SRC_W + detections[i].x + x) * 3;
                grey[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = static_cast<unsigned char>(greyOf(p[0], p[1], p[2], rule));
            }
        }
    }
    return grey;
}

// The exact 3 x 3 box of each crop's 256 x 256 one-channel image, anchor at the centre, border REPLICATE at the crop's edges: the sum of the
// 9 values (over 9)
std::vector<int> boxSums(const std::vector<unsigned char>& image) {
    std::vector<int> sums(VALUES);
    for (int i = 0; i < CROPS; ++i) {
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

// sum / 9 truncated (floor) or rounded to the nearest ((sum + 4) / 9: sum = 9k + r is rounded up exactly when r >= 5; never a tie)
std::vector<unsigned char> boxRounded(const std::vector<int>& sums, const bool truncate) {
    std::vector<unsigned char> out(sums.size());
    for (size_t k = 0; k < sums.size(); ++k) out[k] = static_cast<unsigned char>(truncate ? sums[k] / 9 : (sums[k] + 4) / 9);
    return out;
}

struct Difference8u {
    size_t differ = 0;
    int maxAbs = 0;
};

Difference8u compare8u(const std::vector<unsigned char>& a, const std::vector<unsigned char>& b) {
    Difference8u d;
    for (size_t k = 0; k < a.size(); ++k) {
        const int diff = std::abs(static_cast<int>(a[k]) - static_cast<int>(b[k]));
        if (diff == 0) continue;
        ++d.differ;
        d.maxAbs = std::max(d.maxAbs, diff);
    }
    return d;
}

struct DifferenceFloat {
    size_t nonFinite = 0;   // values of `out` that are not finite
    double maxAbs = 0;      // over the finite values
    size_t beyond = 0;      // compareToReference only: the finite values farther than the tolerance from the reference
};

// A tensor against (v / 255 - 0.45) / 0.225 in double, v the box value at the same position
DifferenceFloat compareToConversion(const std::vector<float>& out, const std::vector<unsigned char>& box) {
    DifferenceFloat d;
    for (size_t k = 0; k < out.size(); ++k) {
        if (!std::isfinite(out[k])) {
            ++d.nonFinite;
            continue;
        }
        const double reference = (static_cast<double>(box[k]) / 255.0 - 0.45) / 0.225;
        d.maxAbs = std::max(d.maxAbs, std::abs(static_cast<double>(out[k]) - reference));
    }
    return d;
}

DifferenceFloat compareFloats(const std::vector<float>& a, const std::vector<float>& b) {
    DifferenceFloat d;
    for (size_t k = 0; k < a.size(); ++k) {
        if (!std::isfinite(a[k]) || !std::isfinite(b[k])) {
            ++d.nonFinite;
            continue;
        }
        d.maxAbs = std::max(d.maxAbs, std::abs(static_cast<double>(a[k]) - static_cast<double>(b[k])));
    }
    return d;
}

// fkl_loop's reference in double: FKL's pipeline as the kernel computes its grey value, then the box and the conversion without rounding of
// their own.
//   grey   FKL's actual rule at 24478f0. fk::RGB2Gray::exec (color_conversion.h:55-69) tests std::is_unsigned_v of the output type (:57,
//          __float2uint_rn), then std::is_signed_v (:63, __float2int_rn), then std::is_floating_point_v (:67, the value unrounded).
//          std::is_signed_v<float> is true, so with a float output the grey is __float2int_rn of the luminance, held in a float (F2I.NTZ in
//          the kernel's SASS). The luminance is (R * 0.299f) + (G * 0.587f) + (B * 0.114f) as nvcc compiles it in the kernel: t = 0.587f G,
//          t = fma(0.299f, R, t), t = fma(0.114f, B, t) (FMUL, FFMA, FFMA in the SASS). The reference copies that contraction: rounding the
//          exact weighted sum instead differs at 926 of the 1,966,080 grey values (the dump of the window e19-fkl-checks,
//          results/rtx4090/e19-batched-box-fkl-checks). std::fma rounds once; volatile keeps the product a float; std::nearbyint rounds to
//          the nearest integer, ties to even (the default rounding mode), as __float2int_rn
//   box    FKL's coefficient (SCALE / 9, rounded to float as the call computes it) times the exact sum of the 9 grey values of the 3 x 3
//          window, border REPLICATE at the crop's edges
//   output (box / 255 - 0.45) / 0.225
// Crop by crop, row by row, as the tensor.
std::vector<double> fklReference(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    const double coefficient = SCALE / static_cast<float>(BOX_W * BOX_H);
    std::vector<double> grey(VALUES), out(VALUES);
    for (int i = 0; i < CROPS; ++i) {
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const unsigned char* p = frame.data() + ((static_cast<size_t>(detections[i].y) + y) * SRC_W + detections[i].x + x) * 3;
                volatile float t = 0.587f * static_cast<float>(p[1]);
                t = std::fma(0.299f, static_cast<float>(p[0]), t);
                t = std::fma(0.114f, static_cast<float>(p[2]), t);
                grey[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = std::nearbyint(static_cast<float>(t));
            }
        }
    }
    for (int i = 0; i < CROPS; ++i) {
        const double* crop = grey.data() + i * PLANE;
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                double sum = 0;
                for (int ky = 0; ky < BOX_H; ++ky) {
                    for (int kx = 0; kx < BOX_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), CROP_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), CROP_H - 1);
                        sum += crop[static_cast<size_t>(sy) * CROP_W + sx];
                    }
                }
                out[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = (coefficient * sum / 255.0 - 0.45) / 0.225;
            }
        }
    }
    return out;
}

// fkl_batch's reference in double: FKL's pipeline as fk::executeBoxBlur<1> computes it (morphology.h at 24478f0).
//   grey   FKL's grey rule, the lines of fklReference: std::nearbyint of t = 0.587f G, t = fma(0.299f, R, t), t = fma(0.114f, B, t)
//   box    the box blur's one pass: its Add reducer sums the 9 grey values of the 3 x 3 window (border REPLICATE at the crop's edges: the kernel
//          clamps every sample to the plane) in float, which is exact (integers up to 9 * 255 = 2,295); its finalizer, fk::Mul, multiplies the
//          sum by 1 / 9 in float, static_cast<float>(1. / (maskW * maskH)) (boxBlurReducers, morphology.h:189-208), and rounds once to
//          float. The reference rounds the same product once: the product of the two floats is exact in double, then it is cast to float
//   output (box / 255 - 0.45) / 0.225 in double
// Crop by crop, row by row, as the tensor.
std::vector<double> fklBatchReference(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    const float factor = static_cast<float>(1. / (static_cast<double>(BOX_W) * BOX_H));
    std::vector<double> grey(VALUES), out(VALUES);
    for (int i = 0; i < CROPS; ++i) {
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                const unsigned char* p = frame.data() + ((static_cast<size_t>(detections[i].y) + y) * SRC_W + detections[i].x + x) * 3;
                volatile float t = 0.587f * static_cast<float>(p[1]);
                t = std::fma(0.299f, static_cast<float>(p[0]), t);
                t = std::fma(0.114f, static_cast<float>(p[2]), t);
                grey[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = std::nearbyint(static_cast<float>(t));
            }
        }
    }
    for (int i = 0; i < CROPS; ++i) {
        const double* crop = grey.data() + i * PLANE;
        for (int y = 0; y < CROP_H; ++y) {
            for (int x = 0; x < CROP_W; ++x) {
                double sum = 0;
                for (int ky = 0; ky < BOX_H; ++ky) {
                    for (int kx = 0; kx < BOX_W; ++kx) {
                        const int sx = std::min(std::max(x + kx - ANCHOR_X, 0), CROP_W - 1);
                        const int sy = std::min(std::max(y + ky - ANCHOR_Y, 0), CROP_H - 1);
                        sum += crop[static_cast<size_t>(sy) * CROP_W + sx];
                    }
                }
                const float box = static_cast<float>(sum * static_cast<double>(factor));
                out[i * PLANE + static_cast<size_t>(y) * CROP_W + x] = (static_cast<double>(box) / 255.0 - 0.45) / 0.225;
            }
        }
    }
    return out;
}

// The values of a float tensor that still hold the fill's bytes (0xFFFFFFFF, a NaN): values the call did not write
size_t fillLeft(const std::vector<float>& tensor) {
    size_t left = 0;
    for (const float v : tensor) {
        uint32_t bits = 0;
        std::memcpy(&bits, &v, sizeof(bits));
        left += bits == 0xFFFFFFFFu ? 1 : 0;
    }
    return left;
}

// A tensor against a reference in double: the largest absolute difference, and how many values are farther than the tolerance
DifferenceFloat compareToReference(const std::vector<float>& out, const std::vector<double>& reference, const double tolerance) {
    DifferenceFloat d;
    for (size_t k = 0; k < out.size(); ++k) {
        if (!std::isfinite(out[k])) {
            ++d.nonFinite;
            continue;
        }
        const double diff = std::abs(static_cast<double>(out[k]) - reference[k]);
        d.maxAbs = std::max(d.maxAbs, diff);
        d.beyond += diff > tolerance ? 1 : 0;
    }
    return d;
}

// The 256 x 256 one-channel image of crop i on the device (rows `pitch` bytes apart) into plane i of a host buffer of the 30 crops
void downloadCrop(const void* data, const size_t pitch, const int i, std::vector<unsigned char>& host) {
    FKLB_CUDA_CHECK(cudaMemcpy2D(host.data() + i * PLANE, CROP_W, data, pitch, CROP_W, CROP_H, cudaMemcpyDeviceToHost));
}

void writeRaw(const std::string& path, const void* data, const size_t bytes) {
    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    f.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
    if (!f) throw std::runtime_error("cannot write " + path);
}

// The GPU spin of the kernel-only interval, the one of E1 and E18 (e18_box.cu): one thread waits on the GPU's global timer and touches no
// memory, so the L2 stays as the previous call left it. Enqueued before the start event (the beforeLaunch hook of fklb::timePreparedLaunch), it
// keeps the GPU busy while the host records the start event and enqueues the whole call, so the interval holds the call's kernels and not the
// host's dispatch of them, as long as the spin lasts longer than the dispatch of one call.
__global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E19_SPIN_US: the length of gpuSpin in microseconds, default 5000
unsigned long long spinUs() {
    const char* e = std::getenv("E19_SPIN_US");
    return std::strtoull(e != nullptr ? e : "5000", nullptr, 10);
}

// One row: its call (the whole frame), the fill of its buffers with 0xFF bytes, and the download of its intermediates and its tensor
struct Row {
    std::string version;
    std::function<void()> call;
    std::function<void()> fill;
    std::function<void(std::vector<unsigned char>& grey, std::vector<unsigned char>& box, std::vector<float>& tensor)> download;
    bool intermediates = true;   // the row has grey and box images (the FKL rows have none: fkl_loop's crop and fkl_batch's frame are one kernel)
};

struct Result {
    std::string version;
    std::vector<unsigned char> grey, box;
    std::vector<float> tensor;
};
} // namespace

int launch() {
    const bool profile = envIs1("E19_PROFILE");
    const bool timing = !profile && !envIs1("E19_NO_TIMING");
    const int profileCalls = std::getenv("E19_PROFILE_CALLS") != nullptr ? std::atoi(std::getenv("E19_PROFILE_CALLS")) : 10;
    const char* dumpDir = std::getenv("E19_DUMP");
    std::cout << "E19 batched box: "
              << (profile ? "profile mode (a warm-up call, then E19_PROFILE_CALLS calls per row, each in its NVTX range; nothing compared, nothing timed)"
                          : (timing ? "checks, then the timing" : "checks only (E19_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> frame = seededNoise(SRC_W, SRC_H);
    const size_t hostRow = static_cast<size_t>(SRC_W) * 3;
    // The detections of the frame, the input of every call: every call reads the 30 positions from this table and builds every ROI and every
    // crop pointer from them, as it would for a new layout on every frame
    const std::vector<Detection> detections = detectionsOfTheFrame();
    // ---- The checks' host references, computed before the first GPU call: they depend on the frame and the detections only. Built here,
    // they do not leave the GPU idle between the checked calls and the timing (the first timed row follows the checks)
    const std::vector<GreyRule> allGreyRules = {GreyRule::FIXED15_NEAREST,    GreyRule::EXACT_TRUNC,    GreyRule::EXACT_NEAREST,
                                                GreyRule::FLOAT_EVAL_TRUNC,   GreyRule::FLOAT_EVAL_NEAREST, GreyRule::FIXED8_NEAREST,
                                                GreyRule::FMA_GRB_NEAREST};
    std::vector<std::vector<unsigned char>> greyReferences;   // the grey reference of each rule of allGreyRules
    std::vector<std::vector<int>> greyReferenceSums;          // the box sums of each of them
    std::vector<double> fklRef;                               // fkl_loop's reference
    std::vector<double> fklBatchRef;                          // fkl_batch's reference
    if (!profile) {
        for (const GreyRule rule : allGreyRules) {
            greyReferences.push_back(greyReference(frame, detections, rule));
            greyReferenceSums.push_back(boxSums(greyReferences.back()));
        }
        fklRef = fklReference(frame, detections);
        fklBatchRef = fklBatchReference(frame, detections);
    }
    const auto referenceOf = [&](const GreyRule rule) {
        return static_cast<size_t>(std::find(allGreyRules.begin(), allGreyRules.end(), rule) - allGreyRules.begin());
    };
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };
    // The rows (put in the order of the checks and of the timing after the last one)
    std::vector<Row> rows;

    // ---- OpenCV-CUDA
    // Set once: the cv::cuda::Stream of the program's stream (it depends on nothing of a frame)
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    // Set once: OpenCV's own copy of the frame, uploaded before the calls as in E18 (the upload is not a step of the pipeline)
    cv::cuda::GpuMat cvFrame;
    cvFrame.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, const_cast<unsigned char*>(frame.data())));
    // Set once: one grey and one box image per crop. They depend on the crop size, which is the same for every detection, and not on the
    // position; cvtColor and apply() keep an output of the right size and type (GpuMat::create returns at once)
    std::vector<cv::cuda::GpuMat> cvGrey, cvBox;
    for (int i = 0; i < CROPS; ++i) {
        cvGrey.emplace_back(CROP_H, CROP_W, CV_8UC1);
        cvBox.emplace_back(CROP_H, CROP_W, CV_8UC1);
    }
    // Set once: the output tensor, 30 x 1 x 256 x 256 floats as one continuous 7680 x 256 CV_32F GpuMat; its size depends on no position
    cv::cuda::GpuMat cvTensor;
    cv::cuda::createContinuous(CROPS * CROP_H, CROP_W, CV_32F, cvTensor);
    if (!cvTensor.isContinuous() || cvTensor.step != CROP_W * sizeof(float)) throw std::runtime_error("opencv_cuda: the tensor is not contiguous");
    // Set once: the box filter object depends on the type, the kernel size, the anchor and the border only, not on any crop; a user creates it
    // once for a stream of frames, as in E18. It keeps the padded image of its last apply(), one 262 x 262 buffer for every crop of the same size
    const cv::Ptr<cv::cuda::Filter> cvBoxFilter =
        cv::cuda::createBoxFilter(CV_8UC1, CV_8UC1, cv::Size(BOX_W, BOX_H), cv::Point(-1, -1), cv::BORDER_REPLICATE);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callOpenCV = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            cv::cuda::cvtColor(roi, cvGrey[i], cv::COLOR_RGB2GRAY, 0, cvs);
            cvBoxFilter->apply(cvGrey[i], cvBox[i], cvs);
            cv::cuda::GpuMat plane = cvTensor.rowRange(CROP_H * i, CROP_H * (i + 1));
            cvBox[i].convertTo(plane, CV_32F, ALPHA, BETA, cvs);
        }
    };
    // (end of the timed call)
    const auto fillOpenCV = [&] {
        for (int i = 0; i < CROPS; ++i) {
            FKLB_CUDA_CHECK(cudaMemset2D(cvGrey[i].data, cvGrey[i].step, 0xFF, CROP_W, CROP_H));
            FKLB_CUDA_CHECK(cudaMemset2D(cvBox[i].data, cvBox[i].step, 0xFF, CROP_W, CROP_H));
        }
        FKLB_CUDA_CHECK(cudaMemset(cvTensor.data, 0xFF, VALUES * sizeof(float)));
    };
    const auto downloadOpenCV = [&](std::vector<unsigned char>& grey, std::vector<unsigned char>& box, std::vector<float>& tensor) {
        for (int i = 0; i < CROPS; ++i) {
            downloadCrop(cvGrey[i].data, cvGrey[i].step, i, grey);
            downloadCrop(cvBox[i].data, cvBox[i].step, i, box);
        }
        FKLB_CUDA_CHECK(cudaMemcpy(tensor.data(), cvTensor.data, VALUES * sizeof(float), cudaMemcpyDeviceToHost));
    };
    rows.push_back({"opencv_cuda", callOpenCV, fillOpenCV, downloadOpenCV});

    // ---- NPP
    // Set once: the NPP stream context, filled field by field from the device properties and the stream's flags as in E18 (CUDA 13 has no
    // nppGetStreamContext); it depends on the device and the stream only
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
    // Set once: NPP's own copy of the frame (cudaMallocPitch), uploaded before the calls as in E18
    Npp8u* nppFrame = nullptr;
    size_t frameStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &frameStep, hostRow, SRC_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, frameStep, frame.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    // Set once: one grey and one box image per crop; they depend on the crop size, not on the position
    std::vector<Npp8u*> nppGrey(CROPS, nullptr), nppBox(CROPS, nullptr);
    std::vector<size_t> greyStep(CROPS, 0), boxStep(CROPS, 0);
    for (int i = 0; i < CROPS; ++i) {
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppGrey[i]), &greyStep[i], CROP_W, CROP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppBox[i]), &boxStep[i], CROP_W, CROP_H));
    }
    // Set once: the output tensor, 30 x 1 x 256 x 256 floats, contiguous; its size depends on no position
    Npp32f* nppTensor = nullptr;
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&nppTensor), VALUES * sizeof(Npp32f)));
    const NppiSize nppCropSize{CROP_W, CROP_H}, nppMaskSize{BOX_W, BOX_H};
    const int tensorStep = CROP_W * static_cast<int>(sizeof(Npp32f));
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callNpp = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const Npp8u* crop = nppFrame + static_cast<size_t>(detections[i].y) * frameStep + static_cast<size_t>(detections[i].x) * 3;
            FKLB_NPP_CHECK(nppiRGBToGray_8u_C3C1R_Ctx(crop, static_cast<int>(frameStep), nppGrey[i], static_cast<int>(greyStep[i]), nppCropSize, ctx));
            FKLB_NPP_CHECK(nppiFilterBoxBorder_8u_C1R_Ctx(nppGrey[i], static_cast<int>(greyStep[i]), nppCropSize, NppiPoint{0, 0}, nppBox[i],
                                                          static_cast<int>(boxStep[i]), nppCropSize, nppMaskSize, NppiPoint{ANCHOR_X, ANCHOR_Y},
                                                          NPP_BORDER_REPLICATE, ctx));
            FKLB_NPP_CHECK(nppiScale_8u32f_C1R_Ctx(nppBox[i], static_cast<int>(boxStep[i]), nppTensor + PLANE * i, tensorStep, nppCropSize, NPP_MIN,
                                                   NPP_MAX, ctx));
        }
    };
    // (end of the timed call)
    const auto fillNpp = [&] {
        for (int i = 0; i < CROPS; ++i) {
            FKLB_CUDA_CHECK(cudaMemset2D(nppGrey[i], greyStep[i], 0xFF, CROP_W, CROP_H));
            FKLB_CUDA_CHECK(cudaMemset2D(nppBox[i], boxStep[i], 0xFF, CROP_W, CROP_H));
        }
        FKLB_CUDA_CHECK(cudaMemset(nppTensor, 0xFF, VALUES * sizeof(Npp32f)));
    };
    const auto downloadNpp = [&](std::vector<unsigned char>& grey, std::vector<unsigned char>& box, std::vector<float>& tensor) {
        for (int i = 0; i < CROPS; ++i) {
            downloadCrop(nppGrey[i], greyStep[i], i, grey);
            downloadCrop(nppBox[i], boxStep[i], i, box);
        }
        FKLB_CUDA_CHECK(cudaMemcpy(tensor.data(), nppTensor, VALUES * sizeof(Npp32f), cudaMemcpyDeviceToHost));
    };
    rows.push_back({"npp", callNpp, fillNpp, downloadNpp});

    // ---- FKL (fkl_loop), at the pin 24478f0: one fk::executeLinearFilter per crop, the whole pipeline of the crop in its one kernel
    // Set once: FKL's stream object of the program's stream (it depends on nothing of a frame)
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    // Set once: FKL's own copy of the frame (fk::Ptr2D), uploaded before the calls as in E18
    fk::Ptr2D<uchar3> fkFrame(SRC_W, SRC_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, frame.data(), hostRow, hostRow, SRC_H, cudaMemcpyHostToDevice));
    // Set once: the output tensor, fk::Tensor<float> of 256 x 256 x 30 planes (rows of 1,024 bytes, planes of 65,536 floats, contiguous); its
    // size depends on no position
    fk::Tensor<float> fkTensor(CROP_W, CROP_H, CROPS, 1);
    if (fkTensor.ptr().dims.pitch != CROP_W * sizeof(float) || fkTensor.ptr().dims.plane_pitch != PLANE * sizeof(float)) {
        throw std::runtime_error("fkl_loop: the tensor is not contiguous");
    }
    using FklDetails = fk::LinearFilterDPPDetails<float>;   // the default tile (16 x 8) and largest kernel (7 x 7): the call a user writes
    using FklGrey = fk::ColorConversion<fk::ColorConversionCodes::COLOR_RGB2GRAY, uchar3, float>;
    using FklMultiply = fk::Mul<float, float, float, fk::UnaryType>;
    using FklAccumulate = fk::Add<float, float, float, fk::UnaryType>;
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFkl = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const fk::Ptr2D<uchar3> crop = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crop).then(FklGrey::build());
            const FklDetails details{CROP_W, CROP_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
            const float n = static_cast<float>(details.kernelWidth * details.kernelHeight);
            const auto coefficient = fk::ConstantFilterRead<float>::build(SCALE / n);
            const fk::Ptr<fk::ND::_3D, float> plane = fkTensor.crop(fk::Point{0, 0, i}, fk::PtrDims<fk::ND::_3D>(CROP_W, CROP_H, 1));
            const auto output = fk::Div<float>::build(255.f)
                                    .then(fk::Sub<float>::build(0.45f))
                                    .then(fk::Div<float>::build(0.225f))
                                    .then(fk::PerThreadWrite<fk::ND::_3D, float>::build(plane));
            if (!fk::executeLinearFilter(details, fk::make_tuple(image, coefficient), FklMultiply::build(), FklAccumulate::build(), output, fkStream)) {
                throw std::runtime_error("fkl_loop: the details are not valid");
            }
        }
    };
    // (end of the timed call)
    const auto fillFkl = [&] { FKLB_CUDA_CHECK(cudaMemset(fkTensor.ptr().data, 0xFF, VALUES * sizeof(float))); };
    const auto downloadFkl = [&](std::vector<unsigned char>&, std::vector<unsigned char>&, std::vector<float>& tensor) {
        FKLB_CUDA_CHECK(cudaMemcpy(tensor.data(), fkTensor.ptr().data, VALUES * sizeof(float), cudaMemcpyDeviceToHost));
    };
    rows.push_back({"fkl_loop", callFkl, fillFkl, downloadFkl, false});

    // ---- FKL (fkl_batch), at the pin 24478f0: one fk::executeBoxBlur<1> per frame, the 30 crops in its one kernel (grid z = 30). It shares
    // fkl_loop's stream object, frame and tensor (all FKL's), and its fill and download (the whole tensor)
    using FklBatchDetails = fk::MorphologyDPPDetails<float>;   // the default tile (16 x 8) and largest mask (7 x 7): the call a user writes
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFklBatch = [&] {
        // The batch: the 30 crop views of the frame, from the detections (filling the array is the batch's build, part of the call)
        std::array<fk::Ptr2D<uchar3>, CROPS> crops;
        for (int i = 0; i < CROPS; ++i) {
            crops[i] = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
        }
        const auto image = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(FklGrey::build());
        const FklBatchDetails details{CROP_W, CROP_H, BOX_W, BOX_H, ANCHOR_X, ANCHOR_Y};
        const auto output = fk::Div<float>::build(255.f)
                                .then(fk::Sub<float>::build(0.45f))
                                .then(fk::Div<float>::build(0.225f))
                                .then(fk::PerThreadWrite<fk::ND::_3D, float>::build(fkTensor));
        if (!fk::executeBoxBlur<1>(details, image, output, fkStream)) {
            throw std::runtime_error("fkl_batch: the details are not valid");
        }
    };
    // (end of the timed call)
    rows.push_back({"fkl_batch", callFklBatch, fillFkl, downloadFkl, false});

    // The order of the checks and of the timing, the same in every run: fkl_loop, fkl_batch, opencv_cuda, npp. The first row timed after the
    // checks can start at a lower SM clock (in E18's windows it ran at 2,550 MHz), so that row is FKL's, not a baseline's
    {
        const std::vector<std::string> order = {"fkl_loop", "fkl_batch", "opencv_cuda", "npp"};
        std::vector<Row> ordered;
        for (const std::string& version : order) {
            for (const Row& row : rows) {
                if (row.version == version) ordered.push_back(row);
            }
        }
        if (ordered.size() != rows.size()) throw std::runtime_error("the row order does not name every row");
        rows = std::move(ordered);
    }

    // ---- The profile mode: a warm-up call outside any range, then E19_PROFILE_CALLS calls, each in the range "<row>:call" inside "<row>"
    if (profile) {
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            row.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(row.version.c_str());
            for (int k = 0; k < profileCalls; ++k) {
                nvtxRangePushA((row.version + ":call").c_str());
                row.call();
                nvtxRangePop();
            }
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePop();
        }
        std::cout << "profile mode: " << profileCalls << " calls per selected row, each in its NVTX range" << std::endl;
    }

    bool allOk = true;
    if (!profile) {
        // ---- One call of every selected row, after its buffers are filled with 0xFF bytes; then its grey and box images and its tensor
        std::vector<Result> results;
        bool buffersKept = true;
        size_t buffersCompared = 0, buffersMoved = 0;
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            std::vector<const void*> before;
            if (row.version == "opencv_cuda") {
                for (int i = 0; i < CROPS; ++i) {
                    before.push_back(cvGrey[i].data);
                    before.push_back(cvBox[i].data);
                }
                before.push_back(cvTensor.data);
            }
            row.fill();
            FKLB_CUDA_CHECK(cudaDeviceSynchronize());
            row.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            Result r{row.version, std::vector<unsigned char>(VALUES), std::vector<unsigned char>(VALUES), std::vector<float>(VALUES)};
            row.download(r.grey, r.box, r.tensor);
            if (row.version == "opencv_cuda") {
                std::vector<const void*> after;
                for (int i = 0; i < CROPS; ++i) {
                    after.push_back(cvGrey[i].data);
                    after.push_back(cvBox[i].data);
                }
                after.push_back(cvTensor.data);
                buffersCompared = after.size();
                for (size_t k = 0; k < after.size(); ++k) buffersMoved += after[k] != before[k] ? 1 : 0;
                buffersKept = buffersMoved == 0;
            }
            if (dumpDir != nullptr) {
                const std::string base = std::string(dumpDir) + "/" + row.version;
                if (row.intermediates) {
                    writeRaw(base + "_grey.u8", r.grey.data(), VALUES);
                    writeRaw(base + "_box.u8", r.box.data(), VALUES);
                }
                writeRaw(base + "_tensor.f32", r.tensor.data(), VALUES * sizeof(float));
            }
            results.push_back(std::move(r));
        }

        fklb::CsvWriter checks("e19_batched_box_checks.csv", "E19 batched box: the checks (30 crops of 256 x 256: 1,966,080 values)",
                               {"version", "step", "compared_with", "rule", "values", "differing_values", "max_abs_difference", "non_finite",
                                "criterion", "ok"});
        const auto check = [&](const std::string& version, const char* step, const char* comparedWith, const std::string& rule, const size_t values,
                               const size_t differing, const double maxAbs, const std::string& nonFinite, const char* criterion, const bool ok,
                               const bool gates) {
            checks.row(version, step, comparedWith, rule, values, differing, maxAbs, nonFinite, criterion, ok ? "yes" : "no");
            if (gates && !ok) allOk = false;
            std::cout << version << " " << step << " against " << comparedWith << " (" << rule << "): " << differing << " of " << values
                      << " differ, largest difference " << maxAbs << (nonFinite.empty() ? "" : ", not finite " + nonFinite) << "; " << criterion
                      << (gates ? (ok ? ": ok" : ": FAILED") : "") << std::endl;
        };
        const Result* resultOpenCV = nullptr;
        const Result* resultNpp = nullptr;
        const Result* resultFkl = nullptr;
        const Result* resultFklBatch = nullptr;
        for (const Result& r : results) {
            if (r.version == "fkl_loop") {   // no grey or box image: checked below, against FKL's reference
                resultFkl = &r;
                continue;
            }
            if (r.version == "fkl_batch") {   // no grey or box image: checked below, against FKL's box blur reference
                resultFklBatch = &r;
                continue;
            }
            const bool isOpenCV = r.version == "opencv_cuda";
            (isOpenCV ? resultOpenCV : resultNpp) = &r;
            // grey: OpenCV's fixed point; for NPP every candidate rule, the one with the fewest differing values being its reference
            const std::vector<GreyRule> rules =
                isOpenCV ? std::vector<GreyRule>{GreyRule::FIXED15_NEAREST}
                         : std::vector<GreyRule>{GreyRule::EXACT_TRUNC, GreyRule::EXACT_NEAREST, GreyRule::FLOAT_EVAL_TRUNC, GreyRule::FLOAT_EVAL_NEAREST,
                                                 GreyRule::FIXED8_NEAREST, GreyRule::FIXED15_NEAREST, GreyRule::FMA_GRB_NEAREST};
            std::vector<size_t> greyRefs;   // indices into greyReferences
            std::vector<Difference8u> greyDiffs;
            size_t best = 0;
            for (size_t k = 0; k < rules.size(); ++k) {
                greyRefs.push_back(referenceOf(rules[k]));
                greyDiffs.push_back(compare8u(r.grey, greyReferences[greyRefs.back()]));
                if (greyDiffs[k].differ < greyDiffs[best].differ) best = k;
            }
            for (size_t k = 0; k < rules.size(); ++k) {
                const bool chosen = k == best;
                check(r.version, "grey", "host_reference", name(rules[k]), VALUES, greyDiffs[k].differ, greyDiffs[k].maxAbs, "",
                      chosen ? (isOpenCV ? "0 differing values" : "0 differing values (the rule with the fewest: the reference)") : "reported",
                      !chosen || greyDiffs[k].differ == 0, chosen);
            }
            const size_t greyRef = greyRefs[best];
            // box: against the box of its own grey images, truncated and rounded; the rule with 0 differing values is its rounding
            const std::vector<int> ownSums = boxSums(r.grey);
            const std::vector<unsigned char> ownTrunc = boxRounded(ownSums, true), ownNearest = boxRounded(ownSums, false);
            const Difference8u dTrunc = compare8u(r.box, ownTrunc), dNearest = compare8u(r.box, ownNearest);
            const bool truncates = dTrunc.differ == 0, rounds = dNearest.differ == 0;
            check(r.version, "box", "box_of_own_grey", "floor(sum / 9)", VALUES, dTrunc.differ, dTrunc.maxAbs, "", "0 differing values for one of the two rules",
                  truncates || rounds, true);
            check(r.version, "box", "box_of_own_grey", "nearest of sum / 9", VALUES, dNearest.differ, dNearest.maxAbs, "",
                  "0 differing values for one of the two rules", truncates || rounds, true);
            // the reference pipeline: the box, with the library's rounding (truncation when neither rule fits), of the reference grey
            const std::vector<unsigned char> refBox = boxRounded(greyReferenceSums[greyRef], !rounds);
            const Difference8u dRefBox = compare8u(r.box, refBox);
            check(r.version, "box", "box_of_reference_grey", rounds ? "nearest of sum / 9" : "floor(sum / 9)", VALUES, dRefBox.differ, dRefBox.maxAbs, "",
                  "reported", true, false);
            // float: against the conversion in double of its own box values, then of the reference pipeline's
            const DifferenceFloat fOwn = compareToConversion(r.tensor, r.box), fRef = compareToConversion(r.tensor, refBox);
            check(r.version, "float", "conversion_of_own_box", "(v / 255 - 0.45) / 0.225 in double", VALUES, 0, fOwn.maxAbs, std::to_string(fOwn.nonFinite),
                  "largest difference at most 1e-5, every value finite", fOwn.nonFinite == 0 && fOwn.maxAbs <= FLOAT_TOLERANCE, true);
            check(r.version, "float", "conversion_of_reference_box", "(v / 255 - 0.45) / 0.225 in double", VALUES, 0, fRef.maxAbs,
                  std::to_string(fRef.nonFinite), "reported", true, false);
            if (isOpenCV) {
                check(r.version, "buffers", "before_the_call", "data pointers of the 30 grey and 30 box images and of the tensor", buffersCompared,
                      buffersMoved, 0, "", "0 moved", buffersKept, true);
            }
        }
        // between the libraries: reported only
        if (resultOpenCV != nullptr && resultNpp != nullptr) {
            const Difference8u dGrey = compare8u(resultOpenCV->grey, resultNpp->grey), dBox = compare8u(resultOpenCV->box, resultNpp->box);
            const DifferenceFloat dFloat = compareFloats(resultOpenCV->tensor, resultNpp->tensor);
            check("opencv_cuda", "grey", "npp", "their own grey images", VALUES, dGrey.differ, dGrey.maxAbs, "", "reported", true, false);
            check("opencv_cuda", "box", "npp", "their own box images", VALUES, dBox.differ, dBox.maxAbs, "", "reported", true, false);
            check("opencv_cuda", "float", "npp", "their own tensors", VALUES, 0, dFloat.maxAbs, std::to_string(dFloat.nonFinite), "reported", true, false);
        }
        // fkl_loop: its tensor against FKL's pipeline in double; then against the libraries' tensors (reported only)
        if (resultFkl != nullptr) {
            const DifferenceFloat f = compareToReference(resultFkl->tensor, fklRef, FKL_TOLERANCE);
            check("fkl_loop", "float", "fkl_reference",
                  "grey nearbyint(fma(0.114f, B, fma(0.299f, R, 0.587f G))), box (1/9 in float) * sum of 9 in double, (v / 255 - 0.45) / 0.225 in "
                  "double; differing = beyond 1e-5",
                  VALUES, f.beyond, f.maxAbs, std::to_string(f.nonFinite), "largest difference at most 1e-5, every value finite",
                  f.nonFinite == 0 && f.maxAbs <= FKL_TOLERANCE, true);
            for (const Result* other : {resultOpenCV, resultNpp}) {
                if (other == nullptr) continue;
                const DifferenceFloat d = compareFloats(resultFkl->tensor, other->tensor);
                check("fkl_loop", "float", other->version.c_str(), "their own tensors", VALUES, 0, d.maxAbs, std::to_string(d.nonFinite), "reported", true,
                      false);
            }
        }
        // fkl_batch: its tensor against FKL's box blur in double; every value written; then against fkl_loop's tensor (reported only)
        if (resultFklBatch != nullptr) {
            const DifferenceFloat f = compareToReference(resultFklBatch->tensor, fklBatchRef, FKL_TOLERANCE);
            check("fkl_batch", "float", "fkl_batch_reference",
                  "grey nearbyint(fma(0.114f, B, fma(0.299f, R, 0.587f G))), box float(sum of 9 * (1/9 in float)), (v / 255 - 0.45) / 0.225 in "
                  "double; differing = beyond 1e-5",
                  VALUES, f.beyond, f.maxAbs, std::to_string(f.nonFinite), "largest difference at most 1e-5, every value finite",
                  f.nonFinite == 0 && f.maxAbs <= FKL_TOLERANCE, true);
            const size_t left = fillLeft(resultFklBatch->tensor);
            check("fkl_batch", "written", "fill", "values still 0xFFFFFFFF (the fill's bytes)", VALUES, left, 0, "", "0 left", left == 0, true);
            if (resultFkl != nullptr) {
                const DifferenceFloat d = compareFloats(resultFklBatch->tensor, resultFkl->tensor);
                check("fkl_batch", "float", "fkl_loop", "their own tensors", VALUES, 0, d.maxAbs, std::to_string(d.nonFinite), "reported", true, false);
            }
        }
        std::cout << (allOk ? "E19 batched box checks passed" : "E19 batched box CHECK FAILED") << std::endl;
    }

    // ---- The timing, after the checks, in the order of the rows. For every row, two intervals over the whole frame (the 30 crops):
    //   whole_call   fklb::timeIt: CUDA events around one call, 100 calls after one warm-up call (E4's and E18's interval). The call builds every
    //                ROI or crop pointer from the detections; when the GPU waits for the host, the interval holds the host's work too.
    //   kernel_only  fklb::timePreparedLaunch behind gpuSpin, with nothing to prepare: the spin, the start event, the whole call (every kernel of
    //                it), the stop event. The interval runs from the end of the spin to the end of the call's last kernel, if the host has enqueued
    //                the call before the spin ends (an nsys run of the timing checks it).
    // Each interval of each row is inside an NVTX range ("whole_call:<row>", "kernel_only:<row>") for the profiler.
    if (timing && allOk) {
        fklb::CsvWriter t("e19_batched_box.csv", "E19 batched box: whole-call and kernel-only times of one frame (30 crops)",
                          {"version", "interval", "mean_ms", "min_ms", "iterations", "spin_us"});
        const auto nothingToPrepare = [] { return 0; };
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            fklb::Timing whole, gpu, host;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("whole_call:" + row.version).c_str());
            fklb::timeIt(cs, row.call, whole);
            nvtxRangePop();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("kernel_only:" + row.version).c_str());
            fklb::timePreparedLaunch(cs, nothingToPrepare, spin, [&](int) { row.call(); }, gpu, host);
            nvtxRangePop();
            t.row(row.version, "whole_call", whole.mean, whole.min, fklb::ITERATIONS, "");
            t.row(row.version, "kernel_only", gpu.mean, gpu.min, fklb::ITERATIONS, spinUs());
            std::cout << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean << " ms (min "
                      << gpu.min << ")" << std::endl;
        }
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    FKLB_CUDA_CHECK(cudaFree(nppFrame));
    for (int i = 0; i < CROPS; ++i) {
        FKLB_CUDA_CHECK(cudaFree(nppGrey[i]));
        FKLB_CUDA_CHECK(cudaFree(nppBox[i]));
    }
    FKLB_CUDA_CHECK(cudaFree(nppTensor));
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
