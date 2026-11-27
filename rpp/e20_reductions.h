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

// E20 reductions (AMD): crop, resize and reduce, for a batch of N crops of one frame, on an AMD GPU (HIP, ParArch::GPU_AMD). The AMD side of
// opencv-cuda/reductions/e20_reductions.cu (E20 on NVIDIA): the same task, frame, crop table, probe, references, checks and tables, with the
// rows of the libraries of this GPU. Of every one of N crops of an 8-bit, three-channel frame, resized to 224 x 224 (bilinear, as floats),
// per channel: the minimum, the maximum, the sum of the pixels, the number of pixels that are not zero and the sum of their squares, left
// in device memory. From them the host computes the mean, the variance and the standard deviation of every channel of every crop. N is 1,
// 10, 20, ..., 150, the values of the E19 crop sweep, all in one process.
//
// The rows, in the order of the checks and of the timing. Each one is a call that computes the N crops, written with the calls that its
// library has for these operations, as a plain user would write them:
//   fkl   one fk::executeOperations<fk::ReduceDPP<>> (FKL's reduce_dpp.h; ReduceDPP<> is ReduceDPP<ParArch::GPU_AMD> in a HIP build), one
//         kernel for the N crops. The read is a BatchRead of the N crop views (fk::PerThreadRead<_2D, uchar3>::build of a std::array of the
//         N fkFrame.crop2D) .then(fk::Resize<INTER_LINEAR, IGNORE_AR, NullType, ResizeSampling::HALF_PIXEL>::build(Size(224, 224))): one
//         grid z plane per crop, a float3 per resized pixel, interpolated and accumulated in the same kernel, so no resized crop exists in
//         any memory. The sampling is fk::Resize's own template parameter, HALF_PIXEL, the sampling of RPP (see "sampling"), as Oscar
//         decided (2026-11-27: the matched row, with the sampling "as a template parameter of resize"). The five reductions are the ones
//         of the NVIDIA program:
//           minimum         Cast<float3, float3> (nothing),  Min<float3>,   Cast<float3, double3>
//           maximum         Cast<float3, float3>,            Max<float3>,   Cast<float3, double3>
//           sum             Cast<float3, double3>,           Add<double3>,  Cast<double3, double3>
//           non zero        IsNonZero (1u or 0u; below),     Add<uint3>,    Cast<uint3, double3>
//           sum of squares  Square (as double3; below),      Add<double3>,  Cast<double3, double3>
//         IsNonZero and Square are operations of this program (a user's operations), as in the NVIDIA program. The write is
//         fk::TensorWrite<double3> of a tensor of 5 x 1 x N: the five values of crop i in plane i. The thread blocks of the kernel leave their
//         values in a workspace that the caller owns, allocated and zeroed once for each N. Every call builds the array of the N crop views,
//         the read with its resize, the reductions, the write and the details, and launches 1 kernel.
//   rpp   RPP 3.2.0 (HIP back-end), its own stream, handles, frame, buffers, ROIs and parameter arrays, and no FKL object, written with
//         RPP's public calls as Oscar answered the questions of E20-RPP (the coordinator session, 2026-11-27). RPP's batched calls take a
//         dense batch of images (one descriptor with an image stride), so, as in the E19 RX 9060 programs, the batch is built in every call:
//         one hipMemcpy2DAsync per crop, from the crop's position in the detection table, into a dense N x 280 x 280 x 3 U8 NHWC batch (the
//         batch building, timed as part of the call). Then one batched call each:
//           rppt_resize_mirror_normalize   the batch to an N x 3 x 224 x 224 F32 NCHW (planar) batch, BILINEAR, mean 0, std 1 and mirror 0
//                                          for every image: RPP's call that resizes 8-bit images into floats (rppt_resize takes one data
//                                          type in and out). In RPP's source it computes (v - mean) * (1 / std) with no scaling and no
//                                          clamp, so it writes the resized value exactly. Its ROIs are LTRB {0, 0, 279, 279}: an XYWH ROI
//                                          array is converted to LTRB in place by the call (RPP 3.2.0), an LTRB one is read as it is
//           rppt_tensor_min, rppt_tensor_max, rppt_tensor_sum   of the resized batch, per image and channel (XYWH ROIs {0, 0, 224, 224},
//                                          which these calls do not convert), into RPP's own arrays of N x 4 floats in device memory
//           rppt_threshold, rppt_tensor_sum   the count of the pixels that are not zero (RPP has no call for it): of the same planar memory
//                                          seen as 3N one-channel images (rppt_threshold of a three-channel image writes one value for its
//                                          three channels, 1 only when all three are in range), with a second handle of batch 3N (RPP
//                                          launches the batch size of the handle), a mask of 1.0 where a pixel is in [the smallest positive
//                                          float, 255] and 0 elsewhere, and its sum per plane: exact (a sum in float of up to 50,176 ones)
//           rppt_tensor_multiply_tensor, rppt_tensor_sum   the sum of the squares (RPP has no call for it): the resized batch multiplied by
//                                          itself element by element (a generic 4D descriptor, broadcast disabled; one call for the batch,
//                                          which launches one kernel per image), and its sum
//         Its mean, variance and standard deviation are computed on the host from these sums, as for every row (Oscar: no rppt_tensor_mean
//         nor rppt_tensor_stddev). RPP documents F32 data, and the cutoffs of rppt_threshold, as 0 to 1; the resized values are 0 to 255, as in
//         every row (Oscar's answer): in RPP's source neither the resize, the statistics nor the threshold scale or clamp them, and the
//         minimum and the maximum start from the values of each image. RPP accumulates the sums of F32 data in float (Rpp32f partial sums and
//         result, tensor_sum.cpp), and a square in float is not exact, so its sums are not exact: see the checks. Each statistics call clears
//         its partial results (hipMemsetAsync) and synchronises its stream inside the call, and so does the resize (RPP 3.2.0): the host waits
//         for the GPU inside every call of the row. rppt_tensor_multiply_tensor sets up its strides on the host, in an OpenMP loop of N
//         threads, in every call.
// Calls per call of a row (library_calls) and GPU operations (gpu_operations, by reading FKL's and RPP's source: kernels, copies and
// memsets): fkl 1 and 1 kernel; rpp N + 8 (N hipMemcpy2DAsync and 8 RPP calls) and N copies + 6 memsets + N + 12 kernels (the resize 1, the
// threshold 1, the multiply N, one per image, and each of the five sums, the minimum and the maximum 2: a partial kernel and a kernel of the
// per-image results). ROCm runs the copies and the memsets as kernels too (__amd_rocclr_copyBufferRectAligned,
// __amd_rocclr_fillBufferUnAligned): a call of rpp is 2 N + 18 kernels in a rocprofv3 kernel trace (20, 78 and 318 at N = 1, 30 and 150), a
// call of fkl 1 (development runs).
// The sampling of the resize (the names are those of fk::ResizeSampling and of common/resize_reference.h; see E4): RPP's batched resize
// calls (rppt_resize, rppt_resize_mirror_normalize, rppt_resize_crop_mirror) compute the source position of pixel x as x * ratio +
// (ratio - 1) / 2, which is HALF_PIXEL ((x + 0.5) * scale - 0.5), and RPP has no call with the other convention (RpptInterpolationType
// chooses the filter only). The fkl row's fk::Resize samples HALF_PIXEL. (The NVIDIA program's fkl row samples ASYMMETRIC, FKL's default.) At the
// edge the two differ: RPP's bilinear interpolation clamps both neighbours of a source position to the columns 0 .. w - 2 and the rows
// 0 .. h - 2 of its ROI (rpp_hip_roi_range_check in RPP 3.2.0's rpp_hip_interpolation.hpp), so it never reads the last column and the
// last row of a crop, and the last column and the last row of every resized crop (447 of its 50,176 pixels) are not FKL's (E4 found the
// same). The program has a reference for each: HALF_PIXEL for fkl, and HALF_PIXEL_RPP_EDGE (HALF_PIXEL with RPP's clamp) for rpp; the
// column "sampling" of both tables names the row's.
//
// The frame and the crops (those of the NVIDIA program):
//   frame    one 3840 x 2160 8UC3 frame of E4's seeded noise (the LCG of E4, E18 and E19, seed 12345; the bytes 31..24, 23..16 and 15..8 of
//            every state, row by row)
//   crops    N crops of 280 x 280, the first N of a table of 150 at the positions of the E19 crop sweep: crop i (0..149), with c = i mod 6,
//            r = (i div 6) mod 5 and k = i div 30, starts at x = 64 + 600 c + SX[k], y = 64 + 400 r + SY[k], with SX = {0, 320, 0, 320, 192}
//            and SY = {0, 0, 200, 200, 100}. Every call reads the positions from the detection table and builds every crop view or copy from
//            them
//   factors  the crop size is a multiple of 28 in both dimensions, so the factor is a multiple of 1/8 (here 1.25): the positions are exact
//            in float in both conventions (the program verifies it), the weights of a resized pixel are multiples of 1/16 (ASYMMETRIC) or
//            1/64 (HALF_PIXEL, the rows of this program), a resized pixel has at most 14 significant bits and is exact in float, and sums in
//            double are exact in any order. So every statistic of the fkl row can be compared with the host's for equality. (Its square
//            needs up to 28 bits: fkl squares in double; a square in float is not exact.)
//
// The checks (e20_reductions_checks_GPU_AMD.csv), for every N, on two frames: the probe of the NVIDIA program (see its header: values 64..192,
// zeros in blocks of 64 x 64 in channels 0 and 1, and the two corners of every crop above and below every other value) and then the frame
// that is timed. Every library's copy of the frame is uploaded again for each. The host resizes every crop with fklb::resizeBilinear in
// double, in each convention. Each row is called once, after the buffers that EVERY row writes are filled with 0xFF bytes (a NaN in a float
// and in a double), and one line per statistic that the row has compares its 3 N values with the reference of the row's convention:
//   minimum, maximum, non_zero, sum, sum_of_squares   fkl, against HALF_PIXEL: equal. rpp, against HALF_PIXEL_RPP_EDGE: minimum,
//                                maximum and non_zero equal; sum and sum_of_squares at most RPP_SUM_RELATIVE_BOUND relative difference
//                                (Oscar's answer to question 3: a relative bound, measured in a development run and stated in the table)
//   mean                         from the row's sum; fkl at most 1e-9 relative difference, rpp at most RPP_SUM_RELATIVE_BOUND
//   variance, std_dev            from the row's sums; fkl at most 1e-9, rpp at most RPP_VARIANCE_RELATIVE_BOUND (the difference of two sums
//                                that are not exact)
// The bounds of rpp, and what a development run on the RX 9060 measured over the 16 N and both frames (FKL 5a9f35b, RPP 3.2.0): sum 1.23e-7,
// sum_of_squares 1.36e-7 and mean 1.23e-7, bound 1e-6; variance 7.04e-6 and std_dev 3.52e-6, bound 5e-5.
// Every line also has the number of its values that differ from the reference of the OTHER convention (not a criterion): ASYMMETRIC for
// fkl, and for rpp HALF_PIXEL, the fkl row's, so that line counts where rpp computes another result than fkl. In the check run
// every RPP call and the copies are followed by hipGetLastError and hipStreamSynchronize (RPP returns RPP_SUCCESS when a launch fails).
//
// The timing (e20_reductions_GPU_AMD.csv), for every N after its checks: two intervals per row, in the order fkl, rpp, the same in every run,
// with the E18 / E19 RX 9060 protocol: the timing file is opened at the first N that is timed, then one gpuSpin of E20_CLOCK_WARMUP_MS
// (default 2000), and sclk is read before and after every interval, outside the timed calls. Both intervals hold the whole frame (the N
// crops). An N whose check fails is not timed.
//   whole_call   fklb::timeIt (common/hip_timing.h): HIP events around one call, 100 calls after one warm-up call. When the GPU waits for the
//                host, the interval holds the host's work too
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin of
//                E20_SPIN_US (default 5,000 us), so the interval runs from the end of the spin to the end of the call's last GPU operation, if
//                the host has enqueued the whole call before the spin ends. The rpp row has not: its first RPP call synchronises its stream
//                (see "rpp"), so the host waits for the spin inside the call and enqueues the rest of the call after it, and the interval
//                holds that host work too
//   Each interval of each row is in a ROCTx range ("whole_call:<row>:<N>", "kernel_only:<row>:<N>").
// E20_CROPS=a,b: only these N, each one of 1, 10, 20, ..., 150, run as they are listed (default: the 16 of them, in that order). E20_ONLY=a,b:
// only these rows. A value that is no N of the sweep or no row stops the program with the exit status 2. E20_NO_TIMING=1: the checks only.
// E20_PROFILE=1: for rocprofv3, no checks and no timing: every selected row is called once (a warm-up), then E20_PROFILE_CALLS times
// (default 10), each call in a ROCTx range "<row>:call" inside the range "<row>:<N>" and followed by a synchronisation.
#include <common/csv_writer.h>
#include <common/env_args.h>
#include <common/hip_timing.h>
#include <common/hip_wallclock.h>
#include <common/main.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>

#include <hip/hip_runtime.h>
#include <rocprofiler-sdk-roctx/roctx.h>
#include <rpp.h>
#include <rppdefs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---- The user's operations of the fkl row, which FKL's basic operations do not have (the ones of the NVIDIA program)
namespace e20 {
// 1 in every channel of a resized pixel that is not zero: the FirstIOp of the count
struct IsNonZero {
    using Parent = fk::UnaryOperation<fk::float3, fk::uint3, IsNonZero>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {input.x != 0 ? 1u : 0u, input.y != 0 ? 1u : 0u, input.z != 0 ? 1u : 0u};
    }
};

// The square of every channel of a resized pixel, computed in double: the FirstIOp of the sum of squares
struct Square {
    using Parent = fk::UnaryOperation<fk::float3, fk::double3, Square>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {static_cast<double>(input.x) * static_cast<double>(input.x), static_cast<double>(input.y) * static_cast<double>(input.y),
                static_cast<double>(input.z) * static_cast<double>(input.z)};
    }
};
} // namespace e20

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_W = 280, CROP_H = 280;     // 1.25 x 1.25 times the resized size
constexpr int OUT_W = 224, OUT_H = 224;
constexpr int OUT_ELEMENTS = OUT_W * OUT_H;   // 50,176 resized pixels per crop
constexpr int MAX_CROPS = 150;                // the largest N: the crop table has 150 crops, and N crops are its first N
constexpr int STATISTICS = 5;                 // minimum, maximum, sum, non zero, sum of squares
constexpr double RELATIVE_TOLERANCE = 1e-9;   // the mean, the variance and the standard deviation, against the host's in double
// The bounds of the rpp row (Oscar's answer to question 3 of E20-RPP: a relative bound, measured in a development run): its sum and its
// sum of squares, accumulated in float by RPP, and the mean from the sum; and the variance and the standard deviation from both sums. The
// checks table writes them as the criterion
constexpr double RPP_SUM_RELATIVE_BOUND = 1e-6;
constexpr double RPP_VARIANCE_RELATIVE_BOUND = 5e-5;
// RPP's buffers have RPP_TAIL bytes after their images (RPP stores 8 pixels at a time, as in the E19 RX 9060 programs), and every ROI array
// has 256 * ceil(N / 256) entries (RPP's ROI conversion kernels, which these calls do not run with these ROI types, rewrite that many)
constexpr size_t RPP_TAIL = 64;
static_assert(CROP_W % 28 == 0 && CROP_H % 28 == 0, "the crop size is not a multiple of 28 in both dimensions");
static_assert(OUT_W % 8 == 0, "RPP's resize writes 8 pixels per thread and checks only the first against the width");

const std::vector<int> BATCHES = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};

struct Detection {
    int x, y;
};

// ---- The five reductions of the fkl row, each one a Tuple<FirstIOp, AccumulationIOp, FinalIOp>, over the float3 pixels that its read
// gives: the minimum and the maximum accumulated in float3, the count in uint3, and the sum and the sum of squares in double3
auto fklReductions() {
    using Same = fk::Cast<fk::float3, fk::float3>;
    using Min = fk::Min<fk::float3, fk::float3, fk::float3, fk::UnaryType>;
    using Max = fk::Max<fk::float3, fk::float3, fk::float3, fk::UnaryType>;
    using AddCount = fk::Add<fk::uint3, fk::uint3, fk::uint3, fk::UnaryType>;
    using AddSum = fk::Add<fk::double3, fk::double3, fk::double3, fk::UnaryType>;
    return fk::make_tuple(fk::make_tuple(Same::build(), Min::build(), fk::Cast<fk::float3, fk::double3>::build()),
                          fk::make_tuple(Same::build(), Max::build(), fk::Cast<fk::float3, fk::double3>::build()),
                          fk::make_tuple(fk::Cast<fk::float3, fk::double3>::build(), AddSum::build(), fk::Cast<fk::double3, fk::double3>::build()),
                          fk::make_tuple(e20::IsNonZero::build(), AddCount::build(), fk::Cast<fk::uint3, fk::double3>::build()),
                          fk::make_tuple(e20::Square::build(), AddSum::build(), fk::Cast<fk::double3, fk::double3>::build()));
}

// The resize of the fkl row: fk::Resize with HALF_PIXEL sampling, its fourth template parameter (the first three are their defaults)
using FklResize = fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>;

const std::vector<std::string> ROWS = {"fkl", "rpp"};

// The conventions of the references: fk::ResizeSampling's two (common/resize_reference.h), and RPP's bilinear resize, which samples at
// HALF_PIXEL positions but clamps BOTH neighbours of a source position to the columns 0 .. w - 2 and the rows 0 .. h - 2 of its ROI
// (rpp_hip_roi_range_check of RPP 3.2.0's rpp_hip_interpolation.hpp: the index at most roi.z - 1, where z is the last column of the ROI,
// x + w - 1 by RPP's own XYWH to LTRB conversion): it never reads the last column and the last row of a crop, so the last column and the
// last row of every resized crop differ from FKL's HALF_PIXEL resize (E4 found the same: "RPP clamps the source coordinate to size - 2")
enum class Sampling { ASYMMETRIC = 0, HALF_PIXEL = 1, HALF_PIXEL_RPP_EDGE = 2 };
const char* samplingName(const Sampling s) {
    return s == Sampling::ASYMMETRIC ? "ASYMMETRIC" : (s == Sampling::HALF_PIXEL ? "HALF_PIXEL" : "HALF_PIXEL_RPP_EDGE");
}

// RPP's bilinear resize of one crop, in double: the source position (x + 0.5) * scale - 0.5 (RPP computes x * ratio + (ratio - 1) / 2 in
// float, the same value here: positionsAreExact), the weights from it, and both neighbours clamped to 0 .. size - 2. dst as resizeBilinear
void resizeBilinearRppEdge(const unsigned char* src, const size_t srcPitchBytes, double* dst) {
    const double scaleX = static_cast<double>(CROP_W) / OUT_W, scaleY = static_cast<double>(CROP_H) / OUT_H;
    const auto clampTo = [](const int v, const int last) { return v < 0 ? 0 : (v > last ? last : v); };
    for (int y = 0; y < OUT_H; ++y) {
        const double sy = (y + 0.5) * scaleY - 0.5;
        const int fy = static_cast<int>(std::floor(sy));
        const double wy = sy - fy;
        const int y1 = clampTo(fy, CROP_H - 2), y2 = clampTo(fy + 1, CROP_H - 2);
        for (int x = 0; x < OUT_W; ++x) {
            const double sx = (x + 0.5) * scaleX - 0.5;
            const int fx = static_cast<int>(std::floor(sx));
            const double wx = sx - fx;
            const int x1 = clampTo(fx, CROP_W - 2), x2 = clampTo(fx + 1, CROP_W - 2);
            for (int c = 0; c < 3; ++c) {
                const auto at = [&](const int xx, const int yy) { return static_cast<double>(src[static_cast<size_t>(yy) * srcPitchBytes + xx * 3 + c]); };
                dst[(static_cast<size_t>(y) * OUT_W + x) * 3 + c] =
                    (1 - wy) * ((1 - wx) * at(x1, y1) + wx * at(x2, y1)) + wy * ((1 - wx) * at(x1, y2) + wx * at(x2, y2));
            }
        }
    }
}

[[noreturn]] void refuse(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(2);
}

bool selected(const std::string& name) {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

void refuseUnknownRows() {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return;
    std::stringstream list(e);
    std::string name;
    int names = 0;
    while (std::getline(list, name, ',')) {
        if (std::find(ROWS.begin(), ROWS.end(), name) == ROWS.end()) refuse("E20_ONLY: '" + name + "' is not a row of this program");
        ++names;
    }
    if (names == 0) refuse("E20_ONLY: no row");
}

std::vector<int> batchesOfTheRun() {
    const char* e = std::getenv("E20_CROPS");
    if (e == nullptr) return BATCHES;
    std::stringstream list(e);
    std::vector<int> batches;
    std::string item;
    while (std::getline(list, item, ',')) {
        const bool digits = !item.empty() && item.size() <= 3 && item.find_first_not_of("0123456789") == std::string::npos;
        const int n = digits ? std::atoi(item.c_str()) : 0;
        if (std::find(BATCHES.begin(), BATCHES.end(), n) == BATCHES.end() || std::to_string(n) != item) {
            refuse("E20_CROPS: '" + item + "' is not one of 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150");
        }
        batches.push_back(n);
    }
    if (batches.empty()) refuse("E20_CROPS: no N");
    return batches;
}

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

// Whether the source positions of the resized pixels of one dimension are exact in float, in both conventions (as in the NVIDIA program),
// and RPP's (x * ratio + (ratio - 1) / 2, in float) equal to the HALF_PIXEL one
bool positionsAreExact(const int cropSize, const int resizedSize) {
    const float factor = static_cast<float>(1.0 / (static_cast<double>(resizedSize) / static_cast<double>(cropSize)));
    const double exactFactor = static_cast<double>(cropSize) / static_cast<double>(resizedSize);
    const float rppRatio = static_cast<float>(cropSize) / static_cast<float>(resizedSize);
    const float rppOffset = (rppRatio - 1) * 0.5f;
    for (int i = 0; i < resizedSize; ++i) {
        const volatile float asymmetric = static_cast<float>(i) * factor;
        const volatile float scaled = (static_cast<float>(i) + 0.5f) * factor;
        const volatile float halfPixel = scaled - 0.5f;
        const volatile float rpp = static_cast<float>(i) * rppRatio + rppOffset;
        if (static_cast<double>(asymmetric) != i * exactFactor || static_cast<double>(halfPixel) != (i + 0.5) * exactFactor - 0.5 ||
            static_cast<double>(rpp) != (i + 0.5) * exactFactor - 0.5) {
            return false;
        }
    }
    return true;
}

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

// The table of the 150 crops (the NVIDIA program's, with its verifications)
std::vector<Detection> detectionsOfTheFrame() {
    constexpr int SX[5] = {0, 320, 0, 320, 192}, SY[5] = {0, 0, 200, 200, 100};
    static_assert(MAX_CROPS <= 5 * 30, "the crop table has 5 groups of 30 crops");
    for (const int n : BATCHES) {
        if (n < 1 || n > MAX_CROPS) throw std::runtime_error("the crop table has no " + std::to_string(n) + " crops");
    }
    std::vector<Detection> d;
    for (int i = 0; i < MAX_CROPS; ++i) {
        const int c = i % 6, r = (i / 6) % 5, k = i / 30;
        d.push_back({64 + 600 * c + SX[k], 64 + 400 * r + SY[k]});
    }
    for (int i = 0; i < 30 && i < MAX_CROPS; ++i) {
        if (d[i].x != 64 + 600 * (i % 6) || d[i].y != 64 + 400 * (i / 6)) {
            throw std::runtime_error("the crop table: crop " + std::to_string(i) + " is not at E19's position");
        }
    }
    for (int i = 0; i < MAX_CROPS; ++i) {
        if (d[i].x < 0 || d[i].y < 0 || d[i].x + CROP_W > SRC_W || d[i].y + CROP_H > SRC_H) {
            throw std::runtime_error("the crop table: crop " + std::to_string(i) + " is not inside the frame");
        }
        for (int j = 0; j < i; ++j) {
            if (d[j].x == d[i].x && d[j].y == d[i].y) {
                throw std::runtime_error("the crop table: crops " + std::to_string(j) + " and " + std::to_string(i) + " are at the same position");
            }
        }
    }
    return d;
}

int levelOf(const int i) {
    constexpr int OF_GROUP[5] = {0, 0, 2, 3, 1};
    return 4 * ((i / 6) % 5) + OF_GROUP[i / 30];
}

int firstCornerOf(const int i, const int c) { return 255 - levelOf(i) - 21 * c; }
int lastCornerOf(const int i, const int c) { return 20 - levelOf(i) + 21 * c; }

// The probe of the checks (the NVIDIA program's, with its verifications)
std::vector<unsigned char> probeOf(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    std::vector<unsigned char> probe(frame.size());
    for (size_t k = 0; k < frame.size(); ++k) probe[k] = static_cast<unsigned char>(64 + frame[k] * 128 / 255);
    for (int y = 0; y < SRC_H; ++y) {
        for (int x = 0; x < SRC_W; ++x) {
            const int bx = x / 64, by = y / 64;
            unsigned char* pixel = probe.data() + (static_cast<size_t>(y) * SRC_W + x) * 3;
            if ((bx + by) % 4 == 0) pixel[0] = 0;
            if ((bx + 2 * by) % 5 == 0) pixel[1] = 0;
        }
    }
    const int crops = static_cast<int>(detections.size());
    for (int i = 0; i < crops; ++i) {
        for (int dy = 0; dy < 2; ++dy) {
            for (int dx = 0; dx < 2; ++dx) {
                const size_t firstX = static_cast<size_t>(detections[i].x) + dx, firstY = static_cast<size_t>(detections[i].y) + dy;
                unsigned char* first = probe.data() + (firstY * SRC_W + firstX) * 3;
                unsigned char* last = probe.data() + ((firstY + CROP_H - 2) * SRC_W + firstX + CROP_W - 2) * 3;
                for (int c = 0; c < 3; ++c) {
                    first[c] = static_cast<unsigned char>(firstCornerOf(i, c));
                    last[c] = static_cast<unsigned char>(lastCornerOf(i, c));
                }
            }
        }
    }
    const auto holds = [&](const int i, const int x, const int y) {
        const Detection& d = detections[i];
        const bool inside = x >= d.x && x + 1 < d.x + CROP_W && y >= d.y && y + 1 < d.y + CROP_H;
        const bool apart = x + 1 < d.x || x >= d.x + CROP_W || y + 1 < d.y || y >= d.y + CROP_H;
        if (!inside && !apart) throw std::runtime_error("the probe: a corner of a crop is cut by the edge of crop " + std::to_string(i));
        return inside;
    };
    for (int i = 0; i < crops; ++i) {
        for (int j = 0; j < crops; ++j) {
            if (j == i) continue;
            const bool firstOfJ = holds(i, detections[j].x, detections[j].y);
            const bool lastOfJ = holds(i, detections[j].x + CROP_W - 2, detections[j].y + CROP_H - 2);
            if ((firstOfJ && levelOf(j) <= levelOf(i)) || (lastOfJ && levelOf(j) >= levelOf(i))) {
                throw std::runtime_error("the probe: the corners of crops " + std::to_string(i) + " and " + std::to_string(j) + " are not in order");
            }
        }
        for (int c = 0; c < 3; ++c) {
            if (firstCornerOf(i, c) <= 192 || firstCornerOf(i, c) > 255 || lastCornerOf(i, c) >= 64 || lastCornerOf(i, c) < 1) {
                throw std::runtime_error("the probe: a corner value is not outside the values of the frame");
            }
        }
    }
    return probe;
}

// What a row leaves for one crop, per channel (NaN: not computed by the row)
struct Statistics {
    double minimum[3] = {std::nan(""), std::nan(""), std::nan("")}, maximum[3] = {std::nan(""), std::nan(""), std::nan("")};
    double nonZero[3] = {std::nan(""), std::nan(""), std::nan("")};
    double sum[3] = {std::nan(""), std::nan(""), std::nan("")}, sumOfSquares[3] = {std::nan(""), std::nan(""), std::nan("")};
};

struct Reference {
    double minimum[3] = {0, 0, 0}, maximum[3] = {0, 0, 0}, nonZero[3] = {0, 0, 0}, sum[3] = {0, 0, 0}, sumOfSquares[3] = {0, 0, 0};
    double mean[3] = {0, 0, 0}, variance[3] = {0, 0, 0}, stdDev[3] = {0, 0, 0};
};

// The references of one frame: of every crop of the table, for each convention (indexed by Sampling)
struct References {
    std::vector<Reference> of[3];
};

References referencesOf(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    References references;
    std::vector<double> resized(static_cast<size_t>(OUT_ELEMENTS) * 3);
    for (const Detection& detection : detections) {
        const unsigned char* base = frame.data() + (static_cast<size_t>(detection.y) * SRC_W + detection.x) * 3;
        for (int s = 0; s < 3; ++s) {
            if (s == 2) {
                resizeBilinearRppEdge(base, static_cast<size_t>(SRC_W) * 3, resized.data());
            } else {
                fklb::resizeBilinear<unsigned char>(base, static_cast<size_t>(SRC_W) * 3, CROP_W, CROP_H, 3, OUT_W, OUT_H,
                                                    s == 0 ? fklb::ResizeSampling::ASYMMETRIC : fklb::ResizeSampling::HALF_PIXEL, resized.data());
            }
            Reference r;
            for (int c = 0; c < 3; ++c) {
                long double sum = 0, sumOfSquares = 0;
                double minimum = resized[c], maximum = resized[c], nonZero = 0;
                for (int k = 0; k < OUT_ELEMENTS; ++k) {
                    const double v = resized[static_cast<size_t>(k) * 3 + c];
                    minimum = v < minimum ? v : minimum;
                    maximum = v > maximum ? v : maximum;
                    nonZero += v != 0 ? 1 : 0;
                    sum += v;
                    sumOfSquares += static_cast<long double>(v) * v;
                }
                const long double mean = sum / OUT_ELEMENTS;
                long double squares = 0;
                for (int k = 0; k < OUT_ELEMENTS; ++k) {
                    const long double d = resized[static_cast<size_t>(k) * 3 + c] - mean;
                    squares += d * d;
                }
                r.minimum[c] = minimum;
                r.maximum[c] = maximum;
                r.nonZero[c] = nonZero;
                r.sum[c] = static_cast<double>(sum);
                r.sumOfSquares[c] = static_cast<double>(sumOfSquares);
                if (static_cast<long double>(r.sum[c]) != sum || static_cast<long double>(r.sumOfSquares[c]) != sumOfSquares ||
                    static_cast<double>(static_cast<float>(minimum)) != minimum || static_cast<double>(static_cast<float>(maximum)) != maximum) {
                    throw std::runtime_error("the statistics of the reference are not exact");
                }
                r.mean[c] = static_cast<double>(mean);
                r.variance[c] = static_cast<double>(squares / OUT_ELEMENTS);
                r.stdDev[c] = static_cast<double>(std::sqrt(squares / OUT_ELEMENTS));
            }
            references.of[s].push_back(r);
        }
    }
    return references;
}

void verifyTheProbe(const References& references) {
    for (int s = 0; s < 3; ++s) {
        const std::vector<Reference>& of = references.of[s];
        for (int i = 0; i < static_cast<int>(of.size()); ++i) {
            const Reference& r = of[i];
            const bool corners = r.maximum[0] == firstCornerOf(i, 0) && r.maximum[1] == firstCornerOf(i, 1) &&
                                 r.maximum[2] == firstCornerOf(i, 2) && r.minimum[2] == lastCornerOf(i, 2);
            const bool zeros = r.minimum[0] == 0 && r.minimum[1] == 0 && r.nonZero[0] < OUT_ELEMENTS && r.nonZero[1] < OUT_ELEMENTS &&
                               r.nonZero[0] != r.nonZero[1] && r.nonZero[2] == OUT_ELEMENTS;
            if (!corners || !zeros) {
                throw std::runtime_error("the probe: crop " + std::to_string(i) + " has not the corners or the zeros that the checks need");
            }
        }
    }
}

double relativeDifference(const double value, const double reference) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    return reference == 0 ? std::abs(value) : std::abs(value - reference) / std::abs(reference);
}

// The GPU spin of the kernel-only interval, as in rpp/e18_box.h and rpp/e19_crop_sweep.h: one thread waits on the GPU's constant-rate wall
// clock (wall_clock64) and touches no memory. `ticks` = microseconds * the wall clock rate (hipDeviceAttributeWallClockRate, kHz) / 1000
__global__ void gpuSpin(const unsigned long long ticks) {
    const unsigned long long t0 = wall_clock64();
    while (wall_clock64() - t0 < ticks) {
    }
}

unsigned long long spinUs() { return static_cast<unsigned long long>(fklb::envInt("E20_SPIN_US", 5000, 0, 1000000)); }
unsigned long long clockWarmupMs() { return static_cast<unsigned long long>(fklb::envInt("E20_CLOCK_WARMUP_MS", 2000, 0, 60000)); }

// One row: its call (the whole frame: the N crops), its stream, the fill of the buffers that it writes with 0xFF bytes, the read back of the
// statistics that it left for every crop, and which statistics it has
struct Row {
    std::string version;
    Sampling sampling;               // the convention of its resize, which is the one of its reference
    Sampling other;                  // the convention that "differing_from_the_other_sampling" counts against
    int calls;                       // library calls of one call of the row
    std::string gpuOperations;       // the GPU operations of one call, by reading the libraries' source
    size_t intermediateBytes;        // the device memory of the intermediate images of the row (0: none)
    hipStream_t stream;
    double sumBound;                 // the relative bound of its sum and its sum of squares, and of the mean (0: equal, and 1e-9 for the mean)
    double varianceBound;            // the relative bound of the variance and the standard deviation from its sums
    std::function<void()> call;
    std::function<void()> fill;
    std::function<std::vector<Statistics>()> download;
};

template <typename T>
std::vector<T> fromDevice(const void* device, const size_t count) {
    std::vector<T> host(count);
    FKLB_HIP_CHECK(hipMemcpy(host.data(), device, count * sizeof(T), hipMemcpyDeviceToHost));
    return host;
}

// ---- FKL
// What the fkl row uses: FKL's stream object of its own stream and FKL's own copy of the frame (set once), and, set again for each N, the
// output tensor (5 x 1 x N double3: the five values of crop i in plane i) and the workspace of the kernel
struct Fkl {
    fk::Stream_<fk::ParArch::GPU_AMD>& stream;
    fk::Ptr2D<fk::uchar3>& frame;
    fk::Tensor<fk::double3>& out;
    fk::Ptr1D<uchar>& workspace;
};

// The fkl row for N crops. N is a template parameter because the batched read takes it at compile time (the std::array of the N crop
// views). It returns the bytes of workspace that the kernel needs
template <int N>
size_t addFklRow(Fkl& fkl, const hipStream_t stream, const std::vector<Detection>& detections, const std::function<void()>& fill,
                    const std::function<std::vector<Statistics>()>& download, std::vector<Row>& rows) {
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFkl = [&fkl, &detections] {
        // The batch: the N crop views of the frame, from the detections (filling the array is the batch's build, part of the call)
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (int i = 0; i < N; ++i) crops[i] = fkl.frame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(FklResize::build(fk::Size(OUT_W, OUT_H)));
        const auto write = fk::TensorWrite<fk::double3>::build(fkl.out.ptr());
        fk::executeOperations<fk::ReduceDPP<>>(fkl.stream, fk::ReduceDPP<>::build_details(fkl.workspace), read, fklReductions(), write);
    };
    // (end of the timed call)
    rows.push_back({"fkl", Sampling::HALF_PIXEL, Sampling::ASYMMETRIC, 1, "1 kernel", 0, stream, 0, RELATIVE_TOLERANCE, callFkl, fill, download});

    // The workspace that the kernel needs: it depends on the size and the number of the resized crops and on the accumulators, and on
    // nothing of a frame
    std::array<fk::Ptr2D<fk::uchar3>, N> crops;
    for (int i = 0; i < N; ++i) crops[i] = fkl.frame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
    const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(FklResize::build(fk::Size(OUT_W, OUT_H)));
    return fk::ReduceDPP<>::workspaceBytes(read, fklReductions());
}
} // namespace

int launch() {
    const bool profile = envIs1("E20_PROFILE");
    const bool timing = !profile && !envIs1("E20_NO_TIMING");
    const int profileCalls = fklb::envInt("E20_PROFILE_CALLS", 10, 1, 1000000);
    refuseUnknownRows();
    const std::vector<int> batches = batchesOfTheRun();
    std::cout << "E20 reductions (AMD): "
              << (profile ? "profile mode (a warm-up call, then E20_PROFILE_CALLS calls per row, each in its ROCTx range; nothing compared, "
                            "nothing timed)"
                          : (timing ? "for every N, the checks and then the timing" : "checks only (E20_NO_TIMING=1)"))
              << std::endl;
    if (!positionsAreExact(CROP_W, OUT_W) || !positionsAreExact(CROP_H, OUT_H)) {
        throw std::runtime_error("the source positions of the resized pixels are not exact in float");
    }
    const std::vector<unsigned char> frame = seededNoise(SRC_W, SRC_H);
    const std::vector<Detection> table = detectionsOfTheFrame();
    const std::vector<unsigned char> probe = profile ? std::vector<unsigned char>() : probeOf(frame, table);
    References probeReferences, benchmarkReferences;
    if (!profile) {
        probeReferences = referencesOf(probe, table);
        verifyTheProbe(probeReferences);
        benchmarkReferences = referencesOf(frame, table);
    }
    int device = 0, wallClockKHz = 0;
    FKLB_HIP_CHECK(hipGetDevice(&device));
    FKLB_HIP_CHECK(hipDeviceGetAttribute(&wallClockKHz, hipDeviceAttributeWallClockRate, device));
    const unsigned long long spinTicks = spinUs() * static_cast<unsigned long long>(wallClockKHz) / 1000ull;
    const size_t frameRow = static_cast<size_t>(SRC_W) * 3;
    bool checking = false;   // the check run: every RPP call and the copies are followed by hipGetLastError and a synchronisation

    // ---- Set once: every library's own stream and copy of the frame, and its upload (the checks upload the probe and then the timed frame)
    std::vector<std::function<void(const std::vector<unsigned char>&)>> uploads;
    hipStream_t fklStream, rppStream;
    FKLB_HIP_CHECK(hipStreamCreate(&fklStream));
    FKLB_HIP_CHECK(hipStreamCreate(&rppStream));
    fk::Stream_<fk::ParArch::GPU_AMD> fkStream(fklStream);
    fk::Ptr2D<fk::uchar3> fkFrame(SRC_W, SRC_H);
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        FKLB_HIP_CHECK(hipMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, f.data(), frameRow, frameRow, SRC_H, hipMemcpyHostToDevice));
    });
    unsigned char* rppFrame = nullptr;   // RPP: dense, 3 bytes per pixel
    FKLB_HIP_CHECK(hipMalloc(&rppFrame, frame.size() + RPP_TAIL));
    uploads.push_back([&](const std::vector<unsigned char>& f) { FKLB_HIP_CHECK(hipMemcpy(rppFrame, f.data(), f.size(), hipMemcpyHostToDevice)); });
    for (const auto& upload : uploads) upload(frame);

    // FKL's output tensor and workspace, set again for each N, declared before the tables (see the NVIDIA program)
    fk::Tensor<fk::double3> fkOut;
    fk::Ptr1D<uchar> fkWorkspace;

    std::unique_ptr<fklb::CsvWriter> checksTable;
    if (!profile) {
        checksTable = std::make_unique<fklb::CsvWriter>(
            "e20_reductions_checks_GPU_AMD.csv", "E20 reductions (AMD): the checks of every N (N crops of one frame, each resized to 224 x 224)",
            std::initializer_list<const char*>{"crops", "frame", "version", "sampling", "statistic", "compared_with", "values",
                                               "differing_values", "max_relative_difference", "criterion", "ok",
                                               "differing_from_the_other_sampling"});
    }
    std::unique_ptr<fklb::CsvWriter> times;
    bool allOk = true;

    for (const int N : batches) {
        const std::vector<Detection> detections(table.begin(), table.begin() + N);
        std::vector<Row> rows;

        // ---- FKL: one fk::executeOperations<fk::ReduceDPP<>>, one kernel, no intermediate image
        fkOut = fk::Tensor<fk::double3>(STATISTICS, 1, N, 1);
        fkWorkspace = fk::Ptr1D<uchar>();
        Fkl fkl{fkStream, fkFrame, fkOut, fkWorkspace};
        if (fkl.out.ptr().dims.plane_pitch != STATISTICS * sizeof(fk::double3)) throw std::runtime_error("fkl: the output tensor is not contiguous");
        const auto fillFkl = [&] { FKLB_HIP_CHECK(hipMemset(fkl.out.ptr().data, 0xFF, static_cast<size_t>(N) * STATISTICS * sizeof(fk::double3))); };
        // fk::double3 is FKL's vector type, of 32 bytes in a HIP build (aligned to 16; 24 in a CUDA build): value k of plane i starts at
        // byte i * plane_pitch + k * sizeof(fk::double3), and its three channels are its first three doubles
        const auto downloadFkl = [&] {
            const size_t planePitch = fkl.out.ptr().dims.plane_pitch;
            const std::vector<unsigned char> out = fromDevice<unsigned char>(fkl.out.ptr().data, static_cast<size_t>(N) * planePitch);
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                const auto at = [&](const int k, const int c) {
                    double v;
                    std::memcpy(&v, out.data() + static_cast<size_t>(i) * planePitch + k * sizeof(fk::double3) + c * sizeof(double), sizeof(double));
                    return v;
                };
                for (int c = 0; c < 3; ++c) {
                    s[i].minimum[c] = at(0, c);
                    s[i].maximum[c] = at(1, c);
                    s[i].sum[c] = at(2, c);
                    s[i].nonZero[c] = at(3, c);
                    s[i].sumOfSquares[c] = at(4, c);
                }
            }
            return s;
        };
        size_t fkWorkspaceBytes = 0;
        switch (N) {
        case 1: fkWorkspaceBytes = addFklRow<1>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 10: fkWorkspaceBytes = addFklRow<10>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 20: fkWorkspaceBytes = addFklRow<20>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 30: fkWorkspaceBytes = addFklRow<30>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 40: fkWorkspaceBytes = addFklRow<40>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 50: fkWorkspaceBytes = addFklRow<50>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 60: fkWorkspaceBytes = addFklRow<60>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 70: fkWorkspaceBytes = addFklRow<70>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 80: fkWorkspaceBytes = addFklRow<80>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 90: fkWorkspaceBytes = addFklRow<90>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 100: fkWorkspaceBytes = addFklRow<100>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 110: fkWorkspaceBytes = addFklRow<110>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 120: fkWorkspaceBytes = addFklRow<120>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 130: fkWorkspaceBytes = addFklRow<130>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 140: fkWorkspaceBytes = addFklRow<140>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        case 150: fkWorkspaceBytes = addFklRow<150>(fkl, fklStream, detections, fillFkl, downloadFkl, rows); break;
        default: throw std::runtime_error("no FKL batch of " + std::to_string(N) + " crops is compiled in");
        }
        if (fkWorkspaceBytes == 0) throw std::runtime_error("fkl: the row needs no workspace");
        fkl.workspace = fk::Ptr1D<uchar>(static_cast<uint>(fkWorkspaceBytes));
        FKLB_HIP_CHECK(hipMemset(fkl.workspace.ptr().data, 0, fkWorkspaceBytes));

        // ---- RPP: two handles, of batch N (the images) and of batch 3N (their 3N planes, for the count: RPP launches the batch size of the
        // handle, not desc.n), and every buffer, descriptor, ROI and parameter array set once for N (they depend on N and the sizes only); the
        // batch building and the eight calls in every call
        rppHandle_t handle{}, planesHandle{};
        FKLB_RPP_CHECK(rppCreate(&handle, N, 0, rppStream, RppBackend::RPP_HIP_BACKEND));
        FKLB_RPP_CHECK(rppCreate(&planesHandle, 3 * N, 0, rppStream, RppBackend::RPP_HIP_BACKEND));
        const size_t batchBytes = static_cast<size_t>(N) * CROP_W * CROP_H * 3, resizedBytes = static_cast<size_t>(N) * OUT_ELEMENTS * 3 * sizeof(float);
        unsigned char* rppBatch = nullptr;
        float *rppResized = nullptr, *rppMask = nullptr, *rppSquares = nullptr;
        float *rppMin = nullptr, *rppMax = nullptr, *rppSum = nullptr, *rppCount = nullptr, *rppSumOfSquares = nullptr;
        FKLB_HIP_CHECK(hipMalloc(&rppBatch, batchBytes + RPP_TAIL));
        FKLB_HIP_CHECK(hipMalloc(&rppResized, resizedBytes + RPP_TAIL));
        FKLB_HIP_CHECK(hipMalloc(&rppMask, resizedBytes + RPP_TAIL));
        FKLB_HIP_CHECK(hipMalloc(&rppSquares, resizedBytes + RPP_TAIL));
        FKLB_HIP_CHECK(hipMalloc(&rppMin, static_cast<size_t>(N) * 4 * sizeof(float)));
        FKLB_HIP_CHECK(hipMalloc(&rppMax, static_cast<size_t>(N) * 4 * sizeof(float)));
        FKLB_HIP_CHECK(hipMalloc(&rppSum, static_cast<size_t>(N) * 4 * sizeof(float)));
        FKLB_HIP_CHECK(hipMalloc(&rppCount, static_cast<size_t>(N) * 3 * sizeof(float)));
        FKLB_HIP_CHECK(hipMalloc(&rppSumOfSquares, static_cast<size_t>(N) * 4 * sizeof(float)));
        RpptDesc batchDesc(N, 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC);     // the dense batch of the N crops
        RpptDesc resizedDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);    // RPP's resized batch, planar
        // The same planar memory as 3N one-channel images, plane c of image i being image 3 i + c: rppt_threshold of a three-channel image
        // writes one value for its three channels (1 only when all three are in range), so the count of every channel needs its own plane
        RpptDesc planesDesc(3 * N, 1, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
        // The generic descriptor of the resized batch for rppt_tensor_multiply_tensor: dims[0] is the batch
        RpptGenericDesc squaresDesc{};
        squaresDesc.numDims = 4;
        squaresDesc.offsetInBytes = 0;
        squaresDesc.dataType = RpptDataType::F32;
        squaresDesc.layout = RpptLayout::NCHW;
        squaresDesc.dims[0] = N;
        squaresDesc.dims[1] = 3;
        squaresDesc.dims[2] = OUT_H;
        squaresDesc.dims[3] = OUT_W;
        squaresDesc.strides[0] = 3 * OUT_ELEMENTS;
        squaresDesc.strides[1] = OUT_ELEMENTS;
        squaresDesc.strides[2] = OUT_W;
        squaresDesc.strides[3] = 1;
        // The ROIs and the parameter arrays, in pinned host memory: RPP's kernels read them on the GPU, and the statistics calls also read
        // their ROIs on the host (the size checks of rppt_tensor_statistical_operations.cpp)
        const size_t roiEntries = 256 * ((static_cast<size_t>(3 * N) + 255) / 256);
        RpptROI *resizeRoi = nullptr, *statisticsRoi = nullptr, *planesRoi = nullptr;
        RpptImagePatch* resizedSizes = nullptr;
        Rpp32f *mean = nullptr, *stdDev = nullptr, *countFrom = nullptr, *countTo = nullptr;
        Rpp32u *mirror = nullptr, *squaresRoi = nullptr;
        FKLB_HIP_CHECK(hipHostMalloc(&resizeRoi, sizeof(RpptROI) * roiEntries));
        FKLB_HIP_CHECK(hipHostMalloc(&statisticsRoi, sizeof(RpptROI) * roiEntries));
        FKLB_HIP_CHECK(hipHostMalloc(&planesRoi, sizeof(RpptROI) * roiEntries));
        FKLB_HIP_CHECK(hipHostMalloc(&countFrom, sizeof(Rpp32f) * 3 * N));
        FKLB_HIP_CHECK(hipHostMalloc(&countTo, sizeof(Rpp32f) * 3 * N));
        FKLB_HIP_CHECK(hipHostMalloc(&squaresRoi, sizeof(Rpp32u) * 6 * N));
        FKLB_HIP_CHECK(hipHostMalloc(&resizedSizes, sizeof(RpptImagePatch) * N));
        FKLB_HIP_CHECK(hipHostMalloc(&mean, sizeof(Rpp32f) * 3 * N));
        FKLB_HIP_CHECK(hipHostMalloc(&stdDev, sizeof(Rpp32f) * 3 * N));
        FKLB_HIP_CHECK(hipHostMalloc(&mirror, sizeof(Rpp32u) * N));
        std::memset(resizeRoi, 0, sizeof(RpptROI) * roiEntries);
        std::memset(statisticsRoi, 0, sizeof(RpptROI) * roiEntries);
        std::memset(planesRoi, 0, sizeof(RpptROI) * roiEntries);
        for (int k = 0; k < 3 * N; ++k) {
            planesRoi[k].xywhROI = {{0, 0}, OUT_W, OUT_H};
            // the pixels that are not zero: from the smallest positive float to 255, the largest resized value (no pixel is negative)
            countFrom[k] = std::numeric_limits<float>::denorm_min();
            countTo[k] = 255.f;
        }
        for (int i = 0; i < N; ++i) {
            // rppt_tensor_multiply_tensor's ROI of every image: the begin and then the length of every dimension but the batch
            const Rpp32u roi[6] = {0, 0, 0, 3, OUT_H, OUT_W};
            std::memcpy(squaresRoi + 6 * i, roi, sizeof(roi));
            resizeRoi[i].ltrbROI = {{0, 0}, {CROP_W - 1, CROP_H - 1}};   // LTRB, inclusive: the whole crop of the batch
            statisticsRoi[i].xywhROI = {{0, 0}, OUT_W, OUT_H};
            resizedSizes[i].width = OUT_W;
            resizedSizes[i].height = OUT_H;
            mirror[i] = 0;
            for (int c = 0; c < 3; ++c) {
                mean[i * 3 + c] = 0.f;
                stdDev[i * 3 + c] = 1.f;
            }
        }
        const std::vector<RpptROI> resizeRoiCopy(resizeRoi, resizeRoi + roiEntries), statisticsRoiCopy(statisticsRoi, statisticsRoi + roiEntries),
            planesRoiCopy(planesRoi, planesRoi + roiEntries);
        const std::vector<Rpp32f> countFromCopy(countFrom, countFrom + 3 * N), countToCopy(countTo, countTo + 3 * N);
        const std::vector<Rpp32u> squaresRoiCopy(squaresRoi, squaresRoi + 6 * N);
        const auto checkedRpp = [&] {
            if (checking) {
                FKLB_HIP_CHECK(hipGetLastError());
                FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
            }
        };
        // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
        const auto callRpp = [&] {
            // The batch building: one copy per crop, from the detection table, into the dense batch
            for (int i = 0; i < N; ++i) {
                const unsigned char* crop = rppFrame + (static_cast<size_t>(detections[i].y) * SRC_W + detections[i].x) * 3;
                FKLB_HIP_CHECK(hipMemcpy2DAsync(rppBatch + static_cast<size_t>(i) * CROP_W * CROP_H * 3, static_cast<size_t>(CROP_W) * 3, crop,
                                                frameRow, static_cast<size_t>(CROP_W) * 3, CROP_H, hipMemcpyDeviceToDevice, rppStream));
            }
            checkedRpp();
            FKLB_RPP_CHECK(rppt_resize_mirror_normalize(rppBatch, &batchDesc, rppResized, &resizedDesc, resizedSizes, RpptInterpolationType::BILINEAR,
                                                        mean, stdDev, mirror, resizeRoi, RpptRoiType::LTRB, handle, RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            FKLB_RPP_CHECK(rppt_tensor_min(rppResized, &resizedDesc, rppMin, static_cast<Rpp32u>(N) * 4, statisticsRoi, RpptRoiType::XYWH, handle,
                                           RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            FKLB_RPP_CHECK(rppt_tensor_max(rppResized, &resizedDesc, rppMax, static_cast<Rpp32u>(N) * 4, statisticsRoi, RpptRoiType::XYWH, handle,
                                           RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            FKLB_RPP_CHECK(rppt_tensor_sum(rppResized, &resizedDesc, rppSum, static_cast<Rpp32u>(N) * 4, statisticsRoi, RpptRoiType::XYWH, handle,
                                           RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            // The count of the pixels that are not zero: 1.0 where a pixel of a plane is in [smallest positive float, 255], 0 elsewhere, and
            // the sum of every plane's mask
            FKLB_RPP_CHECK(rppt_threshold(rppResized, &planesDesc, rppMask, &planesDesc, countFrom, countTo, planesRoi, RpptRoiType::XYWH,
                                          planesHandle, RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            FKLB_RPP_CHECK(rppt_tensor_sum(rppMask, &planesDesc, rppCount, static_cast<Rpp32u>(N) * 3, planesRoi, RpptRoiType::XYWH, planesHandle,
                                           RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            // The sum of the squares: the resized batch multiplied by itself, element by element, and its sum
            FKLB_RPP_CHECK(rppt_tensor_multiply_tensor(rppResized, rppResized, &squaresDesc, &squaresDesc, rppSquares, &squaresDesc,
                                                       RPP_BROADCAST_DISABLE, squaresRoi, squaresRoi, handle, RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
            FKLB_RPP_CHECK(rppt_tensor_sum(rppSquares, &resizedDesc, rppSumOfSquares, static_cast<Rpp32u>(N) * 4, statisticsRoi, RpptRoiType::XYWH,
                                           handle, RppBackend::RPP_HIP_BACKEND));
            checkedRpp();
        };
        // (end of the timed call)
        const auto fillRpp = [&] {
            FKLB_HIP_CHECK(hipMemset(rppBatch, 0xFF, batchBytes));
            for (float* p : {rppResized, rppMask, rppSquares}) FKLB_HIP_CHECK(hipMemset(p, 0xFF, resizedBytes));
            for (float* p : {rppMin, rppMax, rppSum, rppSumOfSquares}) FKLB_HIP_CHECK(hipMemset(p, 0xFF, static_cast<size_t>(N) * 4 * sizeof(float)));
            FKLB_HIP_CHECK(hipMemset(rppCount, 0xFF, static_cast<size_t>(N) * 3 * sizeof(float)));
        };
        const auto downloadRpp = [&] {
            // RPP must not change its ROIs and parameter arrays
            if (std::memcmp(resizeRoi, resizeRoiCopy.data(), sizeof(RpptROI) * roiEntries) != 0 ||
                std::memcmp(statisticsRoi, statisticsRoiCopy.data(), sizeof(RpptROI) * roiEntries) != 0 ||
                std::memcmp(planesRoi, planesRoiCopy.data(), sizeof(RpptROI) * roiEntries) != 0 ||
                std::memcmp(squaresRoi, squaresRoiCopy.data(), sizeof(Rpp32u) * 6 * N) != 0) {
                throw std::runtime_error("rpp: a call changed its ROIs");
            }
            if (std::memcmp(countFrom, countFromCopy.data(), sizeof(Rpp32f) * 3 * N) != 0 || std::memcmp(countTo, countToCopy.data(), sizeof(Rpp32f) * 3 * N) != 0) {
                throw std::runtime_error("rpp: a call changed the cutoffs of the threshold");
            }
            for (int i = 0; i < N; ++i) {
                if (resizedSizes[i].width != OUT_W || resizedSizes[i].height != OUT_H || mirror[i] != 0 || mean[3 * i] != 0.f || stdDev[3 * i] != 1.f) {
                    throw std::runtime_error("rpp: a call changed its parameter arrays");
                }
            }
            const std::vector<float> minimum = fromDevice<float>(rppMin, static_cast<size_t>(N) * 4);
            const std::vector<float> maximum = fromDevice<float>(rppMax, static_cast<size_t>(N) * 4);
            const std::vector<float> sum = fromDevice<float>(rppSum, static_cast<size_t>(N) * 4);
            const std::vector<float> count = fromDevice<float>(rppCount, static_cast<size_t>(N) * 3);
            const std::vector<float> sumOfSquares = fromDevice<float>(rppSumOfSquares, static_cast<size_t>(N) * 4);
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                for (int c = 0; c < 3; ++c) {   // per image: R, G, B and the whole image; the channels are the first three
                    s[i].minimum[c] = minimum[4 * i + c];
                    s[i].maximum[c] = maximum[4 * i + c];
                    s[i].sum[c] = sum[4 * i + c];
                    s[i].nonZero[c] = count[3 * i + c];   // one value per plane
                    s[i].sumOfSquares[c] = sumOfSquares[4 * i + c];
                }
            }
            return s;
        };
        rows.push_back({"rpp", Sampling::HALF_PIXEL_RPP_EDGE, Sampling::HALF_PIXEL, N + 8,
                        std::to_string(N) + " copies + 6 memsets + " + std::to_string(N + 12) + " kernels", batchBytes + 3 * resizedBytes, rppStream,
                        RPP_SUM_RELATIVE_BOUND,
                        RPP_VARIANCE_RELATIVE_BOUND, callRpp, fillRpp, downloadRpp});

        if (rows.size() != ROWS.size()) throw std::runtime_error("the rows are not the ones of ROWS");
        for (size_t k = 0; k < rows.size(); ++k) {
            if (rows[k].version != ROWS[k]) throw std::runtime_error("the rows are not the ones of ROWS");
        }

        // ---- The profile mode
        if (profile) {
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                FKLB_HIP_CHECK(hipDeviceSynchronize());
                row.call();
                FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
                roctxRangePushA((row.version + ":" + std::to_string(N)).c_str());
                for (int k = 0; k < profileCalls; ++k) {
                    roctxRangePushA((row.version + ":call").c_str());
                    row.call();
                    FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
                    roctxRangePop();
                }
                roctxRangePop();
            }
        }

        // ---- The checks of N, on the probe and then on the frame that is timed
        bool batchOk = true;
        if (!profile) {
            checking = true;
            const double n = static_cast<double>(OUT_ELEMENTS);
            for (const char* frameName : {"probe", "benchmark"}) {
                const bool isProbe = std::string(frameName) == "probe";
                const References& references = isProbe ? probeReferences : benchmarkReferences;
                for (const auto& upload : uploads) upload(isProbe ? probe : frame);
                for (const Row& row : rows) {
                    if (!selected(row.version)) continue;
                    for (const Row& any : rows) any.fill();
                    FKLB_HIP_CHECK(hipDeviceSynchronize());
                    row.call();
                    FKLB_HIP_CHECK(hipStreamSynchronize(row.stream));
                    FKLB_HIP_CHECK(hipGetLastError());
                    const std::vector<Statistics> s = row.download();
                    const std::vector<Reference>& own = references.of[static_cast<int>(row.sampling)];
                    const std::vector<Reference>& other = references.of[static_cast<int>(row.other)];
                    const auto check = [&](const char* statistic, const char* comparedWith, const std::function<double(int, int)>& value,
                                           const std::function<double(const Reference&, int)>& expected, const double tolerance) {
                        std::ostringstream criterion;
                        if (tolerance == 0) {
                            criterion << "equal";
                        } else {
                            criterion << "relative difference at most " << tolerance;
                        }
                        int differing = 0, differingFromOther = 0;
                        double largest = 0;
                        for (int i = 0; i < N; ++i) {
                            for (int c = 0; c < 3; ++c) {
                                const double v = value(i, c);
                                const double difference = relativeDifference(v, expected(own[i], c));
                                differing += difference <= tolerance ? 0 : 1;
                                largest = difference > largest ? difference : largest;
                                differingFromOther += relativeDifference(v, expected(other[i], c)) <= tolerance ? 0 : 1;
                            }
                        }
                        const bool ok = differing == 0;
                        checksTable->row(N, frameName, row.version, samplingName(row.sampling), statistic, comparedWith, 3 * N,
                                         differing, largest, criterion.str(), ok ? "yes" : "no", differingFromOther);
                        batchOk = batchOk && ok;
                        if (!ok) {
                            std::cout << "N " << N << " " << frameName << " " << row.version << " " << statistic << " against " << comparedWith
                                      << ": " << differing << " of " << 3 * N << " values differ, largest relative difference " << largest << "; "
                                      << criterion.str() << ": FAILED" << std::endl;
                        }
                    };
                    check("minimum", "host", [&](int i, int c) { return s[i].minimum[c]; }, [](const Reference& r, int c) { return r.minimum[c]; }, 0);
                    check("maximum", "host", [&](int i, int c) { return s[i].maximum[c]; }, [](const Reference& r, int c) { return r.maximum[c]; }, 0);
                    check("non_zero", "host", [&](int i, int c) { return s[i].nonZero[c]; }, [](const Reference& r, int c) { return r.nonZero[c]; }, 0);
                    check("sum", "host", [&](int i, int c) { return s[i].sum[c]; }, [](const Reference& r, int c) { return r.sum[c]; }, row.sumBound);
                    check("sum_of_squares", "host", [&](int i, int c) { return s[i].sumOfSquares[c]; },
                          [](const Reference& r, int c) { return r.sumOfSquares[c]; }, row.sumBound);
                    const char* fromTheSums = "host; computed on the host from what the row left";
                    check("mean", fromTheSums, [&](int i, int c) { return s[i].sum[c] / n; }, [](const Reference& r, int c) { return r.mean[c]; },
                          row.sumBound == 0 ? RELATIVE_TOLERANCE : row.sumBound);
                    const auto variance = [&](int i, int c) { return s[i].sumOfSquares[c] / n - (s[i].sum[c] / n) * (s[i].sum[c] / n); };
                    check("variance", fromTheSums, variance, [](const Reference& r, int c) { return r.variance[c]; }, row.varianceBound);
                    check("std_dev", fromTheSums, [&](int i, int c) { return std::sqrt(variance(i, c)); },
                          [](const Reference& r, int c) { return r.stdDev[c]; }, row.varianceBound);
                }
            }
            checksTable->flush();
            checking = false;
            std::cout << "N " << N << ": " << (batchOk ? "checks passed" : "CHECK FAILED") << std::endl;
            allOk = allOk && batchOk;
        }

        // ---- The timing of N, after its checks, in the order of the rows: two intervals over the whole frame (the N crops)
        if (timing && batchOk) {
            if (!times) {
                times = std::make_unique<fklb::CsvWriter>(
                    "e20_reductions_GPU_AMD.csv",
                    "E20 reductions (AMD): whole-call and kernel-only times of one frame (the statistics of N crops, each resized to 224 x 224)",
                    std::initializer_list<const char*>{"crops", "version", "sampling", "library_calls", "gpu_operations", "intermediate_image_bytes",
                                                       "interval", "mean_ms", "min_ms", "iterations", "spin_us", "sclk_before_mhz", "sclk_after_mhz"},
                    "clock warm-up: one gpuSpin of " + std::to_string(clockWarmupMs()) + " ms after this file's header, before the first interval");
                const double before = fklb::amdGpuClockMHz();
                gpuSpin<<<1, 1, 0, fklStream>>>(clockWarmupMs() * static_cast<unsigned long long>(wallClockKHz));
                FKLB_HIP_CHECK(hipGetLastError());
                FKLB_HIP_CHECK(hipDeviceSynchronize());
                std::cout << "clock warm-up: one gpuSpin of " << clockWarmupMs() << " ms; sclk " << before << " MHz before it, "
                          << fklb::amdGpuClockMHz() << " MHz after it" << std::endl;
            }
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                const char* sampling = samplingName(row.sampling);
                fklb::Timing whole, gpu, host;
                const hipStream_t stream = row.stream;
                FKLB_HIP_CHECK(hipDeviceSynchronize());
                const double s0 = fklb::amdGpuClockMHz();
                roctxRangePushA(("whole_call:" + row.version + ":" + std::to_string(N)).c_str());
                fklb::timeIt(stream, row.call, whole);
                roctxRangePop();
                FKLB_HIP_CHECK(hipDeviceSynchronize());
                const double s1 = fklb::amdGpuClockMHz();
                const double s2 = fklb::amdGpuClockMHz();
                roctxRangePushA(("kernel_only:" + row.version + ":" + std::to_string(N)).c_str());
                fklb::timePreparedLaunch(
                    stream, [] { return 0; }, [&, stream] { gpuSpin<<<1, 1, 0, stream>>>(spinTicks); }, [&](int) { row.call(); }, gpu, host);
                roctxRangePop();
                const double s3 = fklb::amdGpuClockMHz();
                times->row(N, row.version, sampling, row.calls, row.gpuOperations, row.intermediateBytes, "whole_call", whole.mean, whole.min,
                           fklb::ITERATIONS, "", s0, s1);
                times->row(N, row.version, sampling, row.calls, row.gpuOperations, row.intermediateBytes, "kernel_only", gpu.mean, gpu.min,
                           fklb::ITERATIONS, spinUs(), s2, s3);
                times->flush();
                std::cout << "N " << N << " " << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean "
                          << gpu.mean << " ms (min " << gpu.min << "); sclk " << s0 << " / " << s1 << " / " << s2 << " / " << s3 << " MHz" << std::endl;
            }
        } else if (timing) {
            std::cout << "N " << N << ": not timed: a check failed" << std::endl;
        }

        FKLB_HIP_CHECK(hipDeviceSynchronize());
        FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_RPP_CHECK(rppDestroy(planesHandle, RppBackend::RPP_HIP_BACKEND));
        for (void* p : std::vector<void*>{rppBatch, rppResized, rppMask, rppSquares, rppMin, rppMax, rppSum, rppCount, rppSumOfSquares}) {
            FKLB_HIP_CHECK(hipFree(p));
        }
        for (void* p : std::vector<void*>{resizeRoi, statisticsRoi, planesRoi, resizedSizes, mean, stdDev, mirror, countFrom, countTo, squaresRoi}) {
            FKLB_HIP_CHECK(hipHostFree(p));
        }
    }
    if (profile) {
        std::cout << "profile mode: " << profileCalls << " calls per selected row and N, each in its ROCTx range" << std::endl;
    } else {
        std::cout << (allOk ? "E20 reductions (AMD) checks passed" : "E20 reductions (AMD) CHECK FAILED") << std::endl;
    }
    FKLB_HIP_CHECK(hipDeviceSynchronize());
    FKLB_HIP_CHECK(hipFree(rppFrame));
    FKLB_HIP_CHECK(hipStreamDestroy(rppStream));
    return allOk ? 0 : 1;
}
