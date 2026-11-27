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

// E20 reductions (NVIDIA): crop, resize and reduce, for a batch of N crops of one frame. One benchmark, the same for every library: of
// every one of N crops of an 8-bit, three-channel frame, resized to 224 x 224 (bilinear, as floats: the first two steps of E4's
// pre-processing chain), per channel, the minimum, the maximum, the sum of the pixels, the number of pixels that are not zero and the
// sum of their squares, left in device memory (the thrust row: in host memory). From them the host computes the mean, the variance and
// the standard deviation of every channel of every crop. N is 1, 10, 20, ..., 150, the values of the E19 crop sweep.
//
// The rows, one per library. Each one is a call that computes the N crops, written with the calls that its library has for these
// operations, as a plain user would write them; a library is called once per crop where it has no call for a batch of images, as in E4
// and E19:
//   fkl          one fk::executeOperations<fk::ReduceDPP<>> (FKL's reduce_dpp.h), one kernel for the N crops. The read is a BatchRead of
//                the N crop views (fk::PerThreadRead<_2D, uchar3>::build of a std::array of the N fkFrame.crop2D)
//                .then(fk::Resize<INTER_LINEAR>::build(Size(224, 224))): one grid z plane per crop, a float3 per resized pixel, which
//                is interpolated and accumulated in the same kernel, so no resized crop exists in any memory. The reductions are five
//                Tuple<FirstIOp, AccumulationIOp, FinalIOp>:
//                  minimum         Cast<float3, float3> (nothing),  Min<float3>,   Cast<float3, double3>
//                  maximum         Cast<float3, float3>,            Max<float3>,   Cast<float3, double3>
//                  sum             Cast<float3, double3>,           Add<double3>,  Cast<double3, double3>
//                  non zero        IsNonZero (1u or 0u; below),     Add<uint3>,    Cast<uint3, double3>
//                  sum of squares  Square (as double3; below),      Add<double3>,  Cast<double3, double3>
//                The two sums are accumulated in double, as the ones of OpenCV-CUDA: cv::cuda::calcSum and calcSqrSum accumulate in
//                double (cudaarithm, sum.cu: sumImpl and sumSqrImpl with a double result, the pixel converted to double before it is
//                squared). NPP gives its sums and its dot products as doubles (Npp64f); its accumulators are not public. (A float3
//                accumulator does not hold these sums: the sum of the pixels has up to 32 significant bits, and the sum of their
//                squares up to 48.) IsNonZero and Square are operations of this program (a user's operations): FKL's basic operations
//                have neither. The write is fk::TensorWrite<double3> of a tensor of 5 x 1 x N: the five values of crop i, each with
//                its three channels, in plane i. The thread blocks of the kernel leave their values in a workspace, device memory
//                that the caller owns, allocated and zeroed once for each N, as NPP's scratch buffers. Every call builds the array of
//                the N crop views, the read with its Resize, the reductions, the write and the details, and launches 1 kernel.
//   opencv_cuda  per crop: GpuMat::convertTo (the ROI of the crop to CV_32FC3: cv::cuda::resize of an 8-bit image rounds every resized
//                pixel to 8 bits), cv::cuda::resize (INTER_LINEAR, to 224 x 224), cv::cuda::split (cv::cuda::findMinMax and
//                cv::cuda::countNonZero, OpenCV-CUDA's calls for the minimum, the maximum and the count, take one-channel images
//                only), and of each of the three planes cv::cuda::findMinMax and cv::cuda::countNonZero; then cv::cuda::calcSum and
//                cv::cuda::calcSqrSum of the resized image (CV_64FC3 each). 11 calls per crop: OpenCV-CUDA has no call for a batch of
//                images. Every output is a row of a GpuMat that is set once for each N (a row is a GpuMat of the size and type that
//                the call makes, so getOutputMat keeps it). The float crop, the resized image and its three planes are intermediate
//                images, one of each for the N crops, allocated once.
//   npp          per crop, nppiConvert_8u32f_C3R_Ctx of the crop (from the crop pointer and the frame step; nppiResize of an 8-bit
//                image rounds too) into image i of a stack of N float crops; then ONE nppiResizeBatch_32f_C3R_Advanced_Ctx for the N
//                crops, NPP's batched resize (NPPI_INTER_LINEAR; for N = 1 nppiResize_32f_C3R_Ctx, as in E4), into a stack of N
//                resized images; then per crop, of resized image i (NPP has no batched call for these reductions: its batched
//                statistics are its image quality metrics): nppiMinMax_32f_C3R_Ctx, nppiSum_32f_C3R_Ctx (Npp64f),
//                nppiCountInRange_32f_C3R_Ctx with the smallest float above zero and the largest float as bounds (no pixel is
//                negative, so the range holds the pixels that are not zero) and nppiDotProd_32f64f_C3R_Ctx of the image with itself
//                (Npp64f: the sums of the squares; NPP has no call named for them, and its other call for them,
//                nppiNorm_L2_32f_C3R_Ctx, gives their square roots). 5 N + 1 calls. The descriptors of the batched resize point only
//                at NPP's own stacks and are set once; the two stacks are the intermediate images; the results in device memory, and
//                one scratch buffer per function, allocated once.
//   thrust       the single pass written by hand, per crop: thrust::transform_reduce over a counting iterator (thrust::cuda::par on
//                the stream), the call of Thrust's summary_statistics example. Resized pixel k of the crop is interpolated from the
//                bytes of a contiguous copy of the frame with FKL's arithmetic written out (the same factors, the same order of the
//                operations), and becomes a struct of the fifteen values with the accumulators of the fkl row (float, float, uint,
//                double and double for each channel: 88 bytes). The call returns the result, so it is NOT like the other rows: every
//                one of the N calls waits for its result and copies it to the host, where the row keeps it, and allocates and
//                releases its temporary storage (cudaMalloc and cudaFree), which no other row does in a call. N calls. (What those
//                allocations take depends on the device memory that the process holds: in a version of this program with more rows,
//                a call of Thrust took 119 us at N = 30 and 22 us at the 15 other N.)
// Not rows: forms of a baseline that are faster and need a trick, a batch made by hand, a call used outside its documentation or code
// shaped to suit the inside of a library. Measured on the RTX 4090 in a version of this program with more rows, whose checks they
// passed (whole_call):
//   OpenCV-CUDA  cv::cuda::reduce reduces every row of a matrix in one call. With every crop resized into one row of a continuous
//                N x 50,176 matrix, the minimum, the maximum and the sum of the N crops are one call each, the count is
//                cv::cuda::threshold and a sum, and the squares cv::cuda::multiply and a sum (2 N + 8 calls): 0.34 and 2.00 ms at
//                N = 30 and N = 150 where the row takes 1.6 and 8.1 ms. It makes a batch by hand out of a call for one matrix, and
//                uses the minimum and the maximum of cv::cuda::reduce and cv::cuda::threshold on three channels, which their
//                documentation does not name. In the loop over the crops, the same calls in place of cv::cuda::split,
//                cv::cuda::findMinMax and cv::cuda::countNonZero are 1.35 times faster at every N, with that same use of three
//                channels.
//   Thrust       with thrust::reduce_into, thrust::cuda::par_nosync and a block of the caller as the allocator of the policy, no call
//                waits or allocates: 0.31 ms at N = 30 where the row takes 0.64 ms. CUB gives a thread of its reduction the elements
//                that fit in 64 bytes, of these structs of 88 bytes one (scale_mem_bound in cub/util_arch.cuh); with a run of 2
//                resized pixels as the element of the range, accumulated by the functor, that form takes 0.23 ms; and with all the
//                crops in one thrust::reduce_by_key or one cub::DeviceSegmentedReduce::Reduce, runs of 16 pixels and the positions
//                of the crops in a table in device memory, 0.08 and 0.04 ms (the fkl row: 0.05 ms).
//
// The frame and the crops:
//   frame    one 3840 x 2160 8UC3 frame of E4's seeded noise (the LCG of E4, E18 and E19, seed 12345; the bytes 31..24, 23..16 and 15..8 of
//            every state, row by row)
//   crops    N crops of 280 x 280 (1.25 x 1.25 times the resized size), the first N of a table of 150 at the positions of the E19 crop
//            sweep: crop i (0..149), with c = i mod 6, r = (i div 6) mod 5 and k = i div 30, starts at x = 64 + 600 c + SX[k],
//            y = 64 + 400 r + SY[k], with SX = {0, 320, 0, 320, 192} and SY = {0, 0, 200, 200, 100}: a made-up layout of N detections,
//            in which crops of different k overlap. The program treats it as a layout that changes on every frame: every call reads
//            the positions from the detection table and builds every ROI, crop pointer or crop view from them
//   factors  FKL, cv::cuda::resize and the single pass written by hand compute the source position of a resized pixel in float (their
//            sources; NPP's are not public). The crop size is a multiple of 28 in both dimensions, so the factor is a multiple of 1/8
//            (here 1.25): the positions are exact in float (the program verifies it), the weights of both conventions are multiples
//            of 1/16, a resized pixel has at most 16 significant bits and is exact in float, and the sums of the pixels and of their
//            squares are exact in double in any order. So every statistic that a row leaves can be compared with the host's for
//            equality.
// Each library reads its own copy of the same host bytes, crops its own way (OpenCV: a GpuMat ROI; NPP: a pointer into its frame and the
// frame's step; Thrust: a pointer into a contiguous copy; FKL: fk::Ptr2D::crop2D), and owns its intermediate images and its outputs. The
// OpenCV-CUDA, NPP and Thrust rows use no FKL object.
// The sampling convention of the resize (the names are those of fk::ResizeSampling and of common/resize_reference.h; see E4) is the
// one of each library: ASYMMETRIC (source position x * scale) in FKL, in cv::cuda::resize and in the single pass written by hand, which
// writes out FKL's arithmetic; HALF_PIXEL ((x + 0.5) * scale - 0.5) in NPP. The two read different source positions, so they give
// different statistics: every row is checked against the reference of its OWN convention.
//
// The checks (e20_reductions_checks.csv), for every N, on two frames: a probe and then the frame that is timed. Every library's copy of
// the frame is uploaded again for each. The probe is made so that every statistic depends on what it should:
//   values   every byte v of the frame becomes 64 + v * 128 / 255 (64..192)
//   zeros    the frame in blocks of 64 x 64, block (bx, by): channel 0 is 0 in the blocks with (bx + by) mod 4 = 0, channel 1 in the
//            blocks with (bx + 2 by) mod 5 = 0, and channel 2 has no zero. So the count is the number of pixels in channel 2 and less
//            in the two others, with a different number in each, and the minimum is 0 in two channels and not in the third
//   corners  in every crop of the table, the 2 x 2 pixels at its first corner hold a value above every other one of the crop and the
//            2 x 2 pixels at its last corner a value below every other one (zeros apart): 255 - L - 21 c and 20 - L + 21 c in channel c,
//            with the level L = 4 r + {0, 0, 2, 3, 1}[k] of the crop, which orders the crops that overlap. In both conventions the first
//            resized pixel of a crop reads only the first corner and the last one only the last corner, so the maximum of every
//            channel of a crop is one resized pixel, its first, the minimum of channel 2 is its last, and both are those of no other
//            channel. The corners give the level of a crop, of which there are 20, and not the crop: what tells a crop from every
//            other one is its sums.
// Before any GPU call the program verifies the order of the corners of the crops that overlap, and that the references of the probe
// have these maxima, minima and counts. What the probe does not have: in the convention of NPP no resized pixel of channel 2 is zero in
// either frame (the frame that is timed has no zero in any channel there: every count is the number of pixels). So the lower bound of
// NPP's count is told from a wrong one in channels 0 and 1, and one that is wrong in channel 2 only would pass.
// The host resizes every crop with fklb::resizeBilinear (the scalar bilinear resize in double of common/resize_reference.h), in each
// convention. A reference holds, for every channel of every crop, the five statistics (exact, see "factors"), and the mean, the
// variance and the population standard deviation (from the differences to the mean). Each row is called once, after the buffers that
// EVERY row writes, its intermediate images and its statistics, are filled with 0xFF bytes (a NaN in a float and in a double, -1 in an
// int: never right values). Its statistics are read back, and one line per statistic compares the 3 N values of the row (one per
// channel of every crop) with the reference of the row's convention:
//   minimum, maximum, non_zero, sum, sum_of_squares   equal. The timing needs it.
//   mean, variance, std_dev      computed on the host from the sums that the row left; relative difference at most 1e-9. The timing
//                                needs it.
// Every line also has the number of its values that differ from the reference of the OTHER convention (it is not a criterion: it shows
// that the check tells the conventions apart).
//
// The timing (e20_reductions.csv), for every N after its checks: two intervals per row, in the order fkl, opencv_cuda, npp, thrust, the
// same in every run. The first interval timed in a run starts at a lower SM clock (2,550 MHz in place of 2,700 MHz during the first 0.1
// to 0.25 s of load after an idle GPU: about 5% slower), so the first row is FKL's, not a baseline's: in a complete run it is the
// whole_call of fkl at the first N, and with E20_ONLY it is the one of the first row selected, which is then not comparable with a
// complete run's. Both intervals hold the whole frame (the N crops). An N whose check fails is not timed, and the others are: the timing
// table of a run that ends with CHECK FAILED (exit code 1) holds the N that passed. The column intermediate_image_bytes is the device
// memory of the intermediate images of the row: one set, written again for every crop, in opencv_cuda, and the stacks of the N crops
// in npp.
//   whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call. When the GPU waits
//                for the host, the interval holds the host's work too
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin
//                of E20_SPIN_US (default 5,000 us), so the interval runs from the end of the spin to the end of the call's last kernel,
//                if the host has enqueued the whole call before the spin ends. It has not in two cases, in which the interval also
//                holds what the host does after the spin. The thrust row waits for the result of every crop inside the call. And the
//                host of the calls that make the most launches waits inside the call until the spin ends, whatever its length, as if
//                the stream held a limited number of launches that have not run, and enqueues the rest while the GPU runs: measured
//                behind a spin of 200 ms, with CUDA 13.4.3 and the driver 610.43.02, opencv_cuda waits from N = 50 and npp from
//                N = 100, and fkl at no N.
//   Each interval of each row is inside an NVTX range ("whole_call:<row>:<N>", "kernel_only:<row>:<N>").
// E20_CROPS=a,b: only these N, each one of 1, 10, 20, ..., 150, run as they are listed (default: the 16 of them, in that order; FKL's
// batched read takes N at compile time, so the 16 values are compiled in). E20_ONLY=a,b: only these rows. A value of E20_CROPS that is
// no N of the sweep, or a name in E20_ONLY that is no row, stops the program with the exit status 2. E20_NO_TIMING=1: the checks only.
// E20_PROFILE=1: every selected row is called once outside any range (a warm-up), then E20_PROFILE_CALLS times (default 10), each call
// in an NVTX range "<row>:call" inside the range "<row>:<N>"; nothing is compared or timed (to count the kernels of one call with
// nsys).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/env_args.h>
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

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudawarping.hpp>

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <npp.h>
#include <nvtx3/nvToolsExt.h>

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

// ---- The user's operations of the fkl row, which FKL's basic operations do not have
namespace e20 {
// 1 in every channel of a resized pixel that is not zero: the FirstIOp of the count
struct IsNonZero {
    using Parent = fk::UnaryOperation<float3, uint3, IsNonZero>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {input.x != 0 ? 1u : 0u, input.y != 0 ? 1u : 0u, input.z != 0 ? 1u : 0u};
    }
};

// The square of every channel of a resized pixel, computed in double: the FirstIOp of the sum of squares
struct Square {
    using Parent = fk::UnaryOperation<float3, double3, Square>;
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
// Multiples of 28: the factor of the resize is a multiple of 1/8 (see "factors" in the header)
static_assert(CROP_W % 28 == 0 && CROP_H % 28 == 0, "the crop size is not a multiple of 28 in both dimensions");

// The N of the sweep: the values of the E19 crop sweep. FKL's batched read takes N at compile time, so these are the ones compiled in
// (the switch of launch())
const std::vector<int> BATCHES = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150};

// The top-left corner of one crop in the frame
struct Detection {
    int x, y;
};

// ---- The five reductions of the fkl row, each one a Tuple<FirstIOp, AccumulationIOp, FinalIOp>, over the float3 pixels that its read
// gives: the minimum and the maximum accumulated in float3, the count in uint3, and the sum and the sum of squares in double3
auto fklReductions() {
    using Same = fk::Cast<float3, float3>;
    using Min = fk::Min<float3, float3, float3, fk::UnaryType>;
    using Max = fk::Max<float3, float3, float3, fk::UnaryType>;
    using AddCount = fk::Add<uint3, uint3, uint3, fk::UnaryType>;
    using AddSum = fk::Add<double3, double3, double3, fk::UnaryType>;
    return fk::make_tuple(fk::make_tuple(Same::build(), Min::build(), fk::Cast<float3, double3>::build()),
                          fk::make_tuple(Same::build(), Max::build(), fk::Cast<float3, double3>::build()),
                          fk::make_tuple(fk::Cast<float3, double3>::build(), AddSum::build(), fk::Cast<double3, double3>::build()),
                          fk::make_tuple(e20::IsNonZero::build(), AddCount::build(), fk::Cast<uint3, double3>::build()),
                          fk::make_tuple(e20::Square::build(), AddSum::build(), fk::Cast<double3, double3>::build()));
}

using FklResize = fk::Resize<fk::InterpolationType::INTER_LINEAR>;

// ---- The single pass of the thrust row: the five values of the three channels, with the accumulator types of the fkl row
struct Five {
    float minimum[3], maximum[3];
    unsigned int nonZero[3];
    double sum[3], sumOfSquares[3];
};
static_assert(sizeof(Five) == 88, "the struct of the thrust row is not of 88 bytes");

// The factors of the resize as fk::Resize::build computes them: the reciprocal, in double, of the resized size over the crop size, then
// as a float
constexpr float FACTOR_X = static_cast<float>(1.0 / (static_cast<double>(OUT_W) / static_cast<double>(CROP_W)));
constexpr float FACTOR_Y = static_cast<float>(1.0 / (static_cast<double>(OUT_H) / static_cast<double>(CROP_H)));

// Resized pixel k, row by row, of the crop whose first byte is `crop`, in a contiguous frame of 3 bytes per pixel and SRC_W pixels per
// row, as the five values of one pixel: FKL's bilinear interpolation written out (fk::Resize with the default sampling and
// fk::Interpolate<INTER_LINEAR>: the source position is the pixel times the factor, in float; its four neighbours, the second ones not
// beyond the last pixel of the crop; the four products in the order of FKL)
struct FiveOfCropPixel {
    const unsigned char* crop;   // the first byte of the crop in the frame
    __host__ __device__ Five operator()(const int k) const {
        const int x = k % OUT_W, y = k / OUT_W;
        const float srcX = x * FACTOR_X, srcY = y * FACTOR_Y;
        const int x1 = static_cast<int>(floorf(srcX)), y1 = static_cast<int>(floorf(srcY));
        const int x2 = x1 + 1, y2 = y1 + 1;
        const int x2Read = x2 < CROP_W - 1 ? x2 : CROP_W - 1, y2Read = y2 < CROP_H - 1 ? y2 : CROP_H - 1;
        const unsigned char* p00 = crop + (static_cast<long long>(y1) * SRC_W + x1) * 3;
        const unsigned char* p10 = crop + (static_cast<long long>(y1) * SRC_W + x2Read) * 3;
        const unsigned char* p01 = crop + (static_cast<long long>(y2Read) * SRC_W + x1) * 3;
        const unsigned char* p11 = crop + (static_cast<long long>(y2Read) * SRC_W + x2Read) * 3;
        const float w00 = (x2 - srcX) * (y2 - srcY), w10 = (srcX - x1) * (y2 - srcY), w01 = (x2 - srcX) * (srcY - y1),
                    w11 = (srcX - x1) * (srcY - y1);
        Five f;
        for (int c = 0; c < 3; ++c) {
            const float v = (p00[c] * w00) + (p10[c] * w10) + (p01[c] * w01) + (p11[c] * w11);
            f.minimum[c] = v;
            f.maximum[c] = v;
            f.nonZero[c] = v != 0 ? 1u : 0u;
            f.sum[c] = v;
            f.sumOfSquares[c] = static_cast<double>(v) * v;
        }
        return f;
    }
};

struct FiveOfTwo {
    __host__ __device__ Five operator()(const Five& a, const Five& b) const {
        Five f;
        for (int c = 0; c < 3; ++c) {
            f.minimum[c] = a.minimum[c] < b.minimum[c] ? a.minimum[c] : b.minimum[c];
            f.maximum[c] = a.maximum[c] > b.maximum[c] ? a.maximum[c] : b.maximum[c];
            f.nonZero[c] = a.nonZero[c] + b.nonZero[c];
            f.sum[c] = a.sum[c] + b.sum[c];
            f.sumOfSquares[c] = a.sumOfSquares[c] + b.sumOfSquares[c];
        }
        return f;
    }
};

// The value that changes no other: the initial value of the reduction of thrust
constexpr float NO_MINIMUM = std::numeric_limits<float>::max(), NO_MAXIMUM = std::numeric_limits<float>::lowest();
constexpr Five FIVE_OF_NOTHING{{NO_MINIMUM, NO_MINIMUM, NO_MINIMUM}, {NO_MAXIMUM, NO_MAXIMUM, NO_MAXIMUM}, {0u, 0u, 0u}, {0, 0, 0}, {0, 0, 0}};

// The bounds of nppiCountInRange_32f_C3R_Ctx that hold the pixels that are not zero of an image without negative pixels: the smallest
// float above zero and the largest float
constexpr Npp32f NON_ZERO_FROM = std::numeric_limits<Npp32f>::denorm_min();
constexpr Npp32f NON_ZERO_TO = std::numeric_limits<Npp32f>::max();

// The rows, in the order of the checks and of the timing: FKL's first
const std::vector<std::string> ROWS = {"fkl", "opencv_cuda", "npp", "thrust"};

// A value of E20_ONLY or of E20_CROPS that the program does not take: its message, and the exit status 2 of a refused environment variable
// (as fklb::envInt, common/env_args.h), which is not the 1 of a failed check or of a failed call
[[noreturn]] void refuse(const std::string& message) {
    std::cerr << message << std::endl;
    std::exit(2);
}

bool selected(const std::string& name) {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// E20_ONLY names rows: a name that is no row would leave empty tables behind a "checks passed"
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

// E20_CROPS: the N of the run, each one of BATCHES (default: all of them)
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

// Whether the source positions of the resized pixels of one dimension are exact in float, in both conventions: the factor as fk::Resize
// computes it (the reciprocal, in double, of the resized size over the crop size, then as a float), and every position computed in float
// equal to the position computed in double from the sizes
bool positionsAreExact(const int cropSize, const int resizedSize) {
    const float factor = static_cast<float>(1.0 / (static_cast<double>(resizedSize) / static_cast<double>(cropSize)));
    const double exactFactor = static_cast<double>(cropSize) / static_cast<double>(resizedSize);
    for (int i = 0; i < resizedSize; ++i) {
        const volatile float asymmetric = static_cast<float>(i) * factor;
        const volatile float scaled = (static_cast<float>(i) + 0.5f) * factor;
        const volatile float halfPixel = scaled - 0.5f;
        if (static_cast<double>(asymmetric) != i * exactFactor || static_cast<double>(halfPixel) != (i + 0.5) * exactFactor - 0.5) return false;
    }
    return true;
}

// The frame, as in E4: an LCG with seed 12345, the bytes 31..24, 23..16 and 15..8 of every state, row by row
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

// The table of the 150 crops, at the positions of the E19 crop sweep: crop i, with c = i mod 6, r = (i div 6) mod 5 and k = i div 30, at
// x = 64 + 600 c + SX[k], y = 64 + 400 r + SY[k]. Crops 0-29 are at E19's positions (x = 64 + 600 * (i mod 6), y = 64 + 400 * (i div 6), as
// e19_batched_box.cu), every crop lies inside the frame, the 150 positions are distinct and no N of the sweep is beyond the table:
// otherwise the program stops before any GPU call
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

// The level of crop i of the table, which orders the crops that overlap: a crop holds the first corner of another one only if that one
// has a higher level, and the last corner of another one only if that one has a lower level (probeOf verifies it)
int levelOf(const int i) {
    constexpr int OF_GROUP[5] = {0, 0, 2, 3, 1};
    return 4 * ((i / 6) % 5) + OF_GROUP[i / 30];
}

// The values of the two corners of crop i of the probe in channel c: above every other value of the crop, and below every other one
int firstCornerOf(const int i, const int c) { return 255 - levelOf(i) - 21 * c; }
int lastCornerOf(const int i, const int c) { return 20 - levelOf(i) + 21 * c; }

// The probe of the checks (see "The checks" in the header): the frame with every byte v replaced by 64 + v * 128 / 255 (64..192); zeros
// in blocks of 64 x 64 of channel 0 and of channel 1; and in every crop of the table the 2 x 2 pixels of its first corner and of its last
// corner with firstCornerOf and lastCornerOf. It stops the program unless, in every crop, the first corner holds a value above every
// other corner in the crop and the last corner a value below every other one, and no corner is cut by the edge of a crop
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
    // Whether crop i holds the 2 x 2 pixels that start at (x, y): wholly, or not at all (the table must not cut them)
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

// What a row leaves for one crop, per channel
struct Statistics {
    double minimum[3] = {std::nan(""), std::nan(""), std::nan("")}, maximum[3] = {std::nan(""), std::nan(""), std::nan("")};
    double nonZero[3] = {std::nan(""), std::nan(""), std::nan("")};
    double sum[3] = {std::nan(""), std::nan(""), std::nan("")}, sumOfSquares[3] = {std::nan(""), std::nan(""), std::nan("")};
};

// What the host computes of every channel of one crop resized in double, for one convention: the five statistics (exact), and the mean,
// the variance and the population standard deviation
struct Reference {
    double minimum[3] = {0, 0, 0}, maximum[3] = {0, 0, 0}, nonZero[3] = {0, 0, 0}, sum[3] = {0, 0, 0}, sumOfSquares[3] = {0, 0, 0};
    double mean[3] = {0, 0, 0}, variance[3] = {0, 0, 0}, stdDev[3] = {0, 0, 0};
};

// The references of one frame: of every crop of the table, for each convention ([0] ASYMMETRIC, [1] HALF_PIXEL)
struct References {
    std::vector<Reference> of[2];
};

References referencesOf(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    References references;
    std::vector<double> resized(static_cast<size_t>(OUT_ELEMENTS) * 3);
    for (const Detection& detection : detections) {
        const unsigned char* base = frame.data() + (static_cast<size_t>(detection.y) * SRC_W + detection.x) * 3;
        for (int s = 0; s < 2; ++s) {
            fklb::resizeBilinear<unsigned char>(base, static_cast<size_t>(SRC_W) * 3, CROP_W, CROP_H, 3, OUT_W, OUT_H,
                                                s == 0 ? fklb::ResizeSampling::ASYMMETRIC : fklb::ResizeSampling::HALF_PIXEL, resized.data());
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
                // The sum of the squares of the differences to the mean: no difference of two large numbers
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
                // Exact sums are the same numbers in long double and in double, and an exact pixel is the same number in float and in
                // double: otherwise the statistics of the rows could not be compared for equality
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

// What the header says of the probe, verified on its references, in both conventions: the maximum of every channel of a crop is the
// value of its first corner, the minimum of channel 2 the value of its last corner, channel 2 has no zero, and channels 0 and 1 have
// zeros, a different number of them. Otherwise the program stops before any GPU call
void verifyTheProbe(const References& references) {
    for (int s = 0; s < 2; ++s) {
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

// The GPU spin of the kernel-only interval, the one of E1, E18 and E19: it keeps the stream busy while the host enqueues the call
__global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E20_SPIN_US: the length of gpuSpin in microseconds, default 5000 (0 to 1,000,000: fklb::envInt stops the program at anything else)
unsigned long long spinUs() { return static_cast<unsigned long long>(fklb::envInt("E20_SPIN_US", 5000, 0, 1000000)); }

// One row: its call (the whole frame: the N crops), the fill of the buffers that it writes (its intermediate images and its statistics)
// with 0xFF bytes, and the read back of the statistics that it left for every crop
struct Row {
    std::string version;
    fklb::ResizeSampling sampling;   // the convention of its resize, which is the one of its reference
    int calls;                       // library calls of one call of the row
    size_t intermediateBytes;        // the device memory of the intermediate images of the row (0: none)
    std::function<void()> call;
    std::function<void()> fill;
    std::function<std::vector<Statistics>()> download;
};

template <typename T>
T* deviceAlloc(const size_t count) {
    T* p = nullptr;
    FKLB_CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&p), count * sizeof(T)));
    return p;
}

template <typename T>
std::vector<T> fromDevice(const void* device, const size_t count) {
    std::vector<T> host(count);
    FKLB_CUDA_CHECK(cudaMemcpy(host.data(), device, count * sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// ---- FKL
// What the fkl row uses: FKL's stream object of the program's stream and FKL's own copy of the frame (set once), and, set again for each
// N, the output tensor (5 x 1 x N double3: the five values of crop i in plane i) and the workspace of the kernel
struct Fkl {
    fk::Stream_<fk::ParArch::GPU_NVIDIA>& stream;
    fk::Ptr2D<uchar3>& frame;
    fk::Tensor<double3>& out;
    fk::Ptr1D<uchar>& workspace;
};

// The fkl row for N crops. N is a template parameter because the batched read takes it at compile time (the std::array of the N crop
// views). It returns the bytes of workspace that the kernel needs
template <int N>
size_t addFklRow(Fkl& fkl, const std::vector<Detection>& detections, const std::function<void()>& fill,
                 const std::function<std::vector<Statistics>()>& download, std::vector<Row>& rows) {
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFkl = [&fkl, &detections] {
        // The batch: the N crop views of the frame, from the detections (filling the array is the batch's build, part of the call)
        std::array<fk::Ptr2D<uchar3>, N> crops;
        for (int i = 0; i < N; ++i) crops[i] = fkl.frame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
        const auto read = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(FklResize::build(fk::Size(OUT_W, OUT_H)));
        const auto write = fk::TensorWrite<double3>::build(fkl.out.ptr());
        fk::executeOperations<fk::ReduceDPP<>>(fkl.stream, fk::ReduceDPP<>::build_details(fkl.workspace), read, fklReductions(), write);
    };
    // (end of the timed call)
    rows.push_back({"fkl", fklb::ResizeSampling::ASYMMETRIC, 1, 0, callFkl, fill, download});

    // The workspace that the kernel needs: it depends on the size and the number of the resized crops and on the accumulators, and on
    // nothing of a frame
    std::array<fk::Ptr2D<uchar3>, N> crops;
    for (int i = 0; i < N; ++i) crops[i] = fkl.frame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
    const auto read = fk::PerThreadRead<fk::ND::_2D, uchar3>::build(crops).then(FklResize::build(fk::Size(OUT_W, OUT_H)));
    return fk::ReduceDPP<>::workspaceBytes(read, fklReductions());
}
} // namespace

int launch() {
    const bool profile = envIs1("E20_PROFILE");
    const bool timing = !profile && !envIs1("E20_NO_TIMING");
    const int profileCalls = fklb::envInt("E20_PROFILE_CALLS", 10, 1, 1000000);
    refuseUnknownRows();
    const std::vector<int> batches = batchesOfTheRun();
    std::cout << "E20 reductions: "
              << (profile ? "profile mode (a warm-up call, then E20_PROFILE_CALLS calls per row, each in its NVTX range; nothing compared, "
                            "nothing timed)"
                          : (timing ? "for every N, the checks and then the timing" : "checks only (E20_NO_TIMING=1)"))
              << std::endl;
    if (!positionsAreExact(CROP_W, OUT_W) || !positionsAreExact(CROP_H, OUT_H)) {
        throw std::runtime_error("the source positions of the resized pixels are not exact in float");
    }
    const std::vector<unsigned char> frame = seededNoise(SRC_W, SRC_H);
    // The table of the 150 crops: the detections of a run with N crops are its first N
    const std::vector<Detection> table = detectionsOfTheFrame();
    // The probe of the checks and the host references of both frames, of every crop of the table, computed before the first GPU call
    const std::vector<unsigned char> probe = profile ? std::vector<unsigned char>() : probeOf(frame, table);
    References probeReferences, benchmarkReferences;
    if (!profile) {
        probeReferences = referencesOf(probe, table);
        verifyTheProbe(probeReferences);
        benchmarkReferences = referencesOf(frame, table);
    }
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };

    // Set once: the cv::cuda::Stream and FKL's stream object of the program's stream, and the NPP stream context, filled field by field
    // from the device properties and the stream's flags as in E18 and E19 (CUDA 13 has no nppGetStreamContext)
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
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

    // Set once: every library's own copy of the frame, and its upload: the checks upload the probe and then the frame that is timed
    std::vector<std::function<void(const std::vector<unsigned char>&)>> uploads;
    fk::Ptr2D<uchar3> fkFrame(SRC_W, SRC_H);
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, f.data(), static_cast<size_t>(SRC_W) * 3,
                                     static_cast<size_t>(SRC_W) * 3, SRC_H, cudaMemcpyHostToDevice));
    });
    cv::cuda::GpuMat cvFrame;
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        cvFrame.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, const_cast<unsigned char*>(f.data())));
    });
    Npp8u* nppFrame = nullptr;
    size_t nppFrameStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &nppFrameStep, static_cast<size_t>(SRC_W) * 3, SRC_H));
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, nppFrameStep, f.data(), static_cast<size_t>(SRC_W) * 3, static_cast<size_t>(SRC_W) * 3, SRC_H,
                                     cudaMemcpyHostToDevice));
    });
    unsigned char* flat = deviceAlloc<unsigned char>(frame.size());   // Thrust: contiguous, 3 bytes per pixel, no row padding
    uploads.push_back([&](const std::vector<unsigned char>& f) { FKLB_CUDA_CHECK(cudaMemcpy(flat, f.data(), f.size(), cudaMemcpyHostToDevice)); });
    for (const auto& upload : uploads) upload(frame);

    // FKL's output tensor and the workspace of its kernel, which are set again for each N. They are declared before the two tables, so
    // that a program that stops with an error closes its tables first: after the error of a kernel every CUDA call fails, FKL's buffers
    // throw when they are freed, and the program ends there; a table that is still open then is not marked as incomplete
    // (common/csv_writer.h)
    fk::Tensor<double3> fkOut;
    fk::Ptr1D<uchar> fkWorkspace;

    // The checks table is opened unless nothing is checked (the profile mode writes no table)
    std::unique_ptr<fklb::CsvWriter> checksTable;
    if (!profile) {
        checksTable = std::make_unique<fklb::CsvWriter>(
            "e20_reductions_checks.csv", "E20 reductions: the checks of every N (N crops of one frame, each resized to 224 x 224)",
            std::initializer_list<const char*>{"crops", "frame", "version", "sampling", "statistic", "compared_with", "values",
                                               "differing_values", "max_relative_difference", "criterion", "ok",
                                               "differing_from_the_other_sampling"});
    }
    // The timing table is opened when the first N is timed: a run of checks only, or one in which no N passes its checks, writes none
    std::unique_ptr<fklb::CsvWriter> times;
    bool allOk = true;

    for (const int N : batches) {
        // The detections of the frame, the input of every call: every call reads the N positions from this table and builds every ROI,
        // crop pointer and crop view from them, as it would for a new layout on every frame
        const std::vector<Detection> detections(table.begin(), table.begin() + N);
        std::vector<Row> rows;

        // ---- FKL: one fk::executeOperations<fk::ReduceDPP<>>, one kernel, no intermediate image
        fkOut = fk::Tensor<double3>(STATISTICS, 1, N, 1);
        fkWorkspace = fk::Ptr1D<uchar>();
        Fkl fkl{fkStream, fkFrame, fkOut, fkWorkspace};
        if (fkl.out.ptr().dims.plane_pitch != STATISTICS * sizeof(double3)) throw std::runtime_error("fkl: the output tensor is not contiguous");
        const auto fillFkl = [&] { FKLB_CUDA_CHECK(cudaMemset(fkl.out.ptr().data, 0xFF, static_cast<size_t>(N) * STATISTICS * sizeof(double3))); };
        const auto downloadFkl = [&] {
            const std::vector<double> out = fromDevice<double>(fkl.out.ptr().data, static_cast<size_t>(N) * STATISTICS * 3);
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                const double* five = out.data() + static_cast<size_t>(i) * STATISTICS * 3;
                for (int c = 0; c < 3; ++c) {
                    s[i].minimum[c] = five[c];
                    s[i].maximum[c] = five[3 + c];
                    s[i].sum[c] = five[6 + c];
                    s[i].nonZero[c] = five[9 + c];
                    s[i].sumOfSquares[c] = five[12 + c];
                }
            }
            return s;
        };
        size_t fkWorkspaceBytes = 0;
        switch (N) {
        case 1: fkWorkspaceBytes = addFklRow<1>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 10: fkWorkspaceBytes = addFklRow<10>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 20: fkWorkspaceBytes = addFklRow<20>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 30: fkWorkspaceBytes = addFklRow<30>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 40: fkWorkspaceBytes = addFklRow<40>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 50: fkWorkspaceBytes = addFklRow<50>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 60: fkWorkspaceBytes = addFklRow<60>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 70: fkWorkspaceBytes = addFklRow<70>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 80: fkWorkspaceBytes = addFklRow<80>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 90: fkWorkspaceBytes = addFklRow<90>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 100: fkWorkspaceBytes = addFklRow<100>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 110: fkWorkspaceBytes = addFklRow<110>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 120: fkWorkspaceBytes = addFklRow<120>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 130: fkWorkspaceBytes = addFklRow<130>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 140: fkWorkspaceBytes = addFklRow<140>(fkl, detections, fillFkl, downloadFkl, rows); break;
        case 150: fkWorkspaceBytes = addFklRow<150>(fkl, detections, fillFkl, downloadFkl, rows); break;
        default: throw std::runtime_error("no FKL batch of " + std::to_string(N) + " crops is compiled in");
        }
        // Set once for N: the workspace of the kernel, device memory of the caller, zeroed
        if (fkWorkspaceBytes == 0) throw std::runtime_error("fkl: the row needs no workspace");
        fkl.workspace = fk::Ptr1D<uchar>(static_cast<uint>(fkWorkspaceBytes));
        FKLB_CUDA_CHECK(cudaMemset(fkl.workspace.ptr().data, 0, fkWorkspaceBytes));

        // ---- OpenCV-CUDA
        // Set once for N: the intermediate images (the float crop, the resized image and its three planes: one of each for the N crops),
        // and the outputs, one row per crop (cvSum, cvSqrSum) or per channel of every crop (the others: row 3 i + c), each row with the size
        // and the type that its call makes (getOutputMat keeps a GpuMat that already has them). Continuous, so that the rows are
        // consecutive in device memory
        const cv::Size cvOutSize(OUT_W, OUT_H);
        cv::cuda::GpuMat cvFloat(CROP_H, CROP_W, CV_32FC3), cvResized(cvOutSize, CV_32FC3);
        std::array<cv::cuda::GpuMat, 3> cvPlanes{cv::cuda::GpuMat(cvOutSize, CV_32FC1), cv::cuda::GpuMat(cvOutSize, CV_32FC1),
                                                 cv::cuda::GpuMat(cvOutSize, CV_32FC1)};
        cv::cuda::GpuMat cvMinMax, cvNonZero, cvSum, cvSqrSum;
        cv::cuda::createContinuous(3 * N, 2, CV_32FC1, cvMinMax);
        cv::cuda::createContinuous(3 * N, 1, CV_32SC1, cvNonZero);
        cv::cuda::createContinuous(N, 1, CV_64FC3, cvSum);
        cv::cuda::createContinuous(N, 1, CV_64FC3, cvSqrSum);
        for (const cv::cuda::GpuMat* m : {&cvMinMax, &cvNonZero, &cvSum, &cvSqrSum}) {
            if (!m->isContinuous()) throw std::runtime_error("opencv_cuda: an output is not continuous");
        }
        const size_t cvIntermediateBytes = cvFloat.step * cvFloat.rows + cvResized.step * cvResized.rows + 3 * cvPlanes[0].step * cvPlanes[0].rows;
        // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
        const auto callOpenCV = [&] {
            for (int i = 0; i < N; ++i) {
                const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
                roi.convertTo(cvFloat, CV_32FC3, 1.0, cvs);
                cv::cuda::resize(cvFloat, cvResized, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
                cv::cuda::split(cvResized, cvPlanes.data(), cvs);
                for (int c = 0; c < 3; ++c) {
                    cv::cuda::GpuMat minMax = cvMinMax.row(3 * i + c), nonZero = cvNonZero.row(3 * i + c);
                    cv::cuda::findMinMax(cvPlanes[c], minMax, cv::noArray(), cvs);
                    cv::cuda::countNonZero(cvPlanes[c], nonZero, cvs);
                }
                cv::cuda::GpuMat sum = cvSum.row(i), sqrSum = cvSqrSum.row(i);
                cv::cuda::calcSum(cvResized, sum, cv::noArray(), cvs);
                cv::cuda::calcSqrSum(cvResized, sqrSum, cv::noArray(), cvs);
            }
        };
        // (end of the timed call)
        const auto fillOpenCV = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(cvFloat.data, cvFloat.step, 0xFF, static_cast<size_t>(CROP_W) * 3 * sizeof(float), CROP_H));
            FKLB_CUDA_CHECK(cudaMemset2D(cvResized.data, cvResized.step, 0xFF, static_cast<size_t>(OUT_W) * 3 * sizeof(float), OUT_H));
            for (int c = 0; c < 3; ++c) {
                FKLB_CUDA_CHECK(cudaMemset2D(cvPlanes[c].data, cvPlanes[c].step, 0xFF, static_cast<size_t>(OUT_W) * sizeof(float), OUT_H));
            }
            FKLB_CUDA_CHECK(cudaMemset(cvMinMax.data, 0xFF, static_cast<size_t>(N) * 3 * 2 * sizeof(float)));
            FKLB_CUDA_CHECK(cudaMemset(cvNonZero.data, 0xFF, static_cast<size_t>(N) * 3 * sizeof(int)));
            FKLB_CUDA_CHECK(cudaMemset(cvSum.data, 0xFF, static_cast<size_t>(N) * 3 * sizeof(double)));
            FKLB_CUDA_CHECK(cudaMemset(cvSqrSum.data, 0xFF, static_cast<size_t>(N) * 3 * sizeof(double)));
        };
        const auto downloadOpenCV = [&] {
            const std::vector<float> minMax = fromDevice<float>(cvMinMax.data, static_cast<size_t>(N) * 3 * 2);
            const std::vector<int> nonZero = fromDevice<int>(cvNonZero.data, static_cast<size_t>(N) * 3);
            const std::vector<double> sum = fromDevice<double>(cvSum.data, static_cast<size_t>(N) * 3);
            const std::vector<double> sqrSum = fromDevice<double>(cvSqrSum.data, static_cast<size_t>(N) * 3);
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                for (int c = 0; c < 3; ++c) {
                    s[i].minimum[c] = minMax[2 * (3 * i + c)];
                    s[i].maximum[c] = minMax[2 * (3 * i + c) + 1];
                    s[i].nonZero[c] = nonZero[3 * i + c];
                    s[i].sum[c] = sum[3 * i + c];
                    s[i].sumOfSquares[c] = sqrSum[3 * i + c];
                }
            }
            return s;
        };
        rows.push_back({"opencv_cuda", fklb::ResizeSampling::ASYMMETRIC, 11 * N, cvIntermediateBytes, callOpenCV, fillOpenCV, downloadOpenCV});

        // ---- NPP
        // Set once for N: the intermediate images, a stack of the N float crops and a stack of the N resized images (one cudaMallocPitch
        // image each, image i starting at row i times the image height), the descriptors of the batched resize, which point only at
        // these two stacks (so they do not depend on where the crops are in the frame), the results in device memory (element 3 i + c
        // for channel c of crop i), and the scratch buffer of every function
        Npp8u* nppFloat = nullptr;
        Npp8u* nppResized = nullptr;
        size_t nppFloatStep = 0, nppResizedStep = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFloat), &nppFloatStep, static_cast<size_t>(CROP_W) * 3 * sizeof(Npp32f),
                                        static_cast<size_t>(N) * CROP_H));
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppResized), &nppResizedStep, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f),
                                        static_cast<size_t>(N) * OUT_H));
        const NppiSize nppCropSize{CROP_W, CROP_H}, nppOutSize{OUT_W, OUT_H};
        const NppiRect nppCropRect{0, 0, CROP_W, CROP_H}, nppOutRect{0, 0, OUT_W, OUT_H};
        const int frameStep = static_cast<int>(nppFrameStep), floatStep = static_cast<int>(nppFloatStep);
        const int resizedStep = static_cast<int>(nppResizedStep);
        NppiImageDescriptor* nppBatchSrc = deviceAlloc<NppiImageDescriptor>(N);
        NppiImageDescriptor* nppBatchDst = deviceAlloc<NppiImageDescriptor>(N);
        NppiResizeBatchROI_Advanced* nppBatchRoi = deviceAlloc<NppiResizeBatchROI_Advanced>(N);
        {
            std::vector<NppiImageDescriptor> src(N), dst(N);
            std::vector<NppiResizeBatchROI_Advanced> roi(N);
            for (int i = 0; i < N; ++i) {
                src[i].pData = nppFloat + static_cast<size_t>(i) * CROP_H * nppFloatStep;
                src[i].nStep = floatStep;
                src[i].oSize = nppCropSize;
                dst[i].pData = nppResized + static_cast<size_t>(i) * OUT_H * nppResizedStep;
                dst[i].nStep = resizedStep;
                dst[i].oSize = nppOutSize;
                roi[i].oSrcRectROI = nppCropRect;
                roi[i].oDstRectROI = nppOutRect;
            }
            FKLB_CUDA_CHECK(cudaMemcpy(nppBatchSrc, src.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
            FKLB_CUDA_CHECK(cudaMemcpy(nppBatchDst, dst.data(), N * sizeof(NppiImageDescriptor), cudaMemcpyHostToDevice));
            FKLB_CUDA_CHECK(cudaMemcpy(nppBatchRoi, roi.data(), N * sizeof(NppiResizeBatchROI_Advanced), cudaMemcpyHostToDevice));
        }
        Npp32f* nppMin = deviceAlloc<Npp32f>(static_cast<size_t>(N) * 3);
        Npp32f* nppMax = deviceAlloc<Npp32f>(static_cast<size_t>(N) * 3);
        Npp64f* nppSum = deviceAlloc<Npp64f>(static_cast<size_t>(N) * 3);
        int* nppCount = deviceAlloc<int>(static_cast<size_t>(N) * 3);
        Npp64f* nppDotProd = deviceAlloc<Npp64f>(static_cast<size_t>(N) * 3);
        size_t minMaxBytes = 0, sumBytes = 0, countBytes = 0, dotProdBytes = 0;
        FKLB_NPP_CHECK(nppiMinMaxGetBufferHostSize_32f_C3R_Ctx(nppOutSize, &minMaxBytes, ctx));
        FKLB_NPP_CHECK(nppiSumGetBufferHostSize_32f_C3R_Ctx(nppOutSize, &sumBytes, ctx));
        FKLB_NPP_CHECK(nppiCountInRangeGetBufferHostSize_32f_C3R_Ctx(nppOutSize, &countBytes, ctx));
        FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx(nppOutSize, &dotProdBytes, ctx));
        Npp8u* minMaxBuffer = deviceAlloc<Npp8u>(minMaxBytes > 0 ? minMaxBytes : 1);
        Npp8u* sumBuffer = deviceAlloc<Npp8u>(sumBytes > 0 ? sumBytes : 1);
        Npp8u* countBuffer = deviceAlloc<Npp8u>(countBytes > 0 ? countBytes : 1);
        Npp8u* dotProdBuffer = deviceAlloc<Npp8u>(dotProdBytes > 0 ? dotProdBytes : 1);
        const size_t nppIntermediateBytes = nppFloatStep * N * CROP_H + nppResizedStep * N * OUT_H;
        // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
        const auto callNpp = [&] {
            for (int i = 0; i < N; ++i) {
                const Npp8u* crop = nppFrame + static_cast<size_t>(detections[i].y) * nppFrameStep + static_cast<size_t>(detections[i].x) * 3;
                Npp32f* const floatCrop = reinterpret_cast<Npp32f*>(nppFloat + static_cast<size_t>(i) * CROP_H * nppFloatStep);
                FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(crop, frameStep, floatCrop, floatStep, nppCropSize, ctx));
            }
            if (N == 1) {   // one image: NPP's single-image resize, as in E4
                FKLB_NPP_CHECK(nppiResize_32f_C3R_Ctx(reinterpret_cast<const Npp32f*>(nppFloat), floatStep, nppCropSize, nppCropRect,
                                                      reinterpret_cast<Npp32f*>(nppResized), resizedStep, nppOutSize, nppOutRect,
                                                      NPPI_INTER_LINEAR, ctx));
            } else {
                FKLB_NPP_CHECK(nppiResizeBatch_32f_C3R_Advanced_Ctx(OUT_W, OUT_H, nppBatchSrc, nppBatchDst, nppBatchRoi, N, NPPI_INTER_LINEAR, ctx));
            }
            for (int i = 0; i < N; ++i) {
                const Npp32f* resized = reinterpret_cast<const Npp32f*>(nppResized + static_cast<size_t>(i) * OUT_H * nppResizedStep);
                Npp32f lower[3] = {NON_ZERO_FROM, NON_ZERO_FROM, NON_ZERO_FROM}, upper[3] = {NON_ZERO_TO, NON_ZERO_TO, NON_ZERO_TO};
                FKLB_NPP_CHECK(nppiMinMax_32f_C3R_Ctx(resized, resizedStep, nppOutSize, nppMin + 3 * i, nppMax + 3 * i, minMaxBuffer, ctx));
                FKLB_NPP_CHECK(nppiSum_32f_C3R_Ctx(resized, resizedStep, nppOutSize, sumBuffer, nppSum + 3 * i, ctx));
                FKLB_NPP_CHECK(nppiCountInRange_32f_C3R_Ctx(resized, resizedStep, nppOutSize, nppCount + 3 * i, lower, upper, countBuffer, ctx));
                FKLB_NPP_CHECK(
                    nppiDotProd_32f64f_C3R_Ctx(resized, resizedStep, resized, resizedStep, nppOutSize, nppDotProd + 3 * i, dotProdBuffer, ctx));
            }
        };
        // (end of the timed call)
        const auto fillNpp = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(nppFloat, nppFloatStep, 0xFF, static_cast<size_t>(CROP_W) * 3 * sizeof(Npp32f),
                                         static_cast<size_t>(N) * CROP_H));
            FKLB_CUDA_CHECK(cudaMemset2D(nppResized, nppResizedStep, 0xFF, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f),
                                         static_cast<size_t>(N) * OUT_H));
            FKLB_CUDA_CHECK(cudaMemset(nppMin, 0xFF, static_cast<size_t>(N) * 3 * sizeof(Npp32f)));
            FKLB_CUDA_CHECK(cudaMemset(nppMax, 0xFF, static_cast<size_t>(N) * 3 * sizeof(Npp32f)));
            FKLB_CUDA_CHECK(cudaMemset(nppCount, 0xFF, static_cast<size_t>(N) * 3 * sizeof(int)));
            FKLB_CUDA_CHECK(cudaMemset(nppSum, 0xFF, static_cast<size_t>(N) * 3 * sizeof(Npp64f)));
            FKLB_CUDA_CHECK(cudaMemset(nppDotProd, 0xFF, static_cast<size_t>(N) * 3 * sizeof(Npp64f)));
        };
        const auto downloadNpp = [&] {
            const std::vector<Npp32f> minimum = fromDevice<Npp32f>(nppMin, static_cast<size_t>(N) * 3);
            const std::vector<Npp32f> maximum = fromDevice<Npp32f>(nppMax, static_cast<size_t>(N) * 3);
            const std::vector<int> count = fromDevice<int>(nppCount, static_cast<size_t>(N) * 3);
            const std::vector<Npp64f> sum = fromDevice<Npp64f>(nppSum, static_cast<size_t>(N) * 3);
            const std::vector<Npp64f> dotProd = fromDevice<Npp64f>(nppDotProd, static_cast<size_t>(N) * 3);
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                for (int c = 0; c < 3; ++c) {
                    s[i].minimum[c] = minimum[3 * i + c];
                    s[i].maximum[c] = maximum[3 * i + c];
                    s[i].nonZero[c] = count[3 * i + c];
                    s[i].sum[c] = sum[3 * i + c];
                    s[i].sumOfSquares[c] = dotProd[3 * i + c];
                }
            }
            return s;
        };
        rows.push_back({"npp", fklb::ResizeSampling::HALF_PIXEL, 5 * N + 1, nppIntermediateBytes, callNpp, fillNpp, downloadNpp});

        // ---- Thrust: the single pass written by hand, once per crop; no intermediate image
        // Thrust returns its results to the host: the row keeps them there
        std::vector<Five> thrustOut(N);
        // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
        const auto callThrust = [&] {
            for (int i = 0; i < N; ++i) {
                // The resized pixels of crop i: the crop pointer from the detection
                const FiveOfCropPixel pixels{flat + (static_cast<size_t>(detections[i].y) * SRC_W + detections[i].x) * 3};
                const thrust::counting_iterator<int> first(0);
                thrustOut[i] = thrust::transform_reduce(thrust::cuda::par.on(cs), first, first + OUT_ELEMENTS, pixels, FIVE_OF_NOTHING, FiveOfTwo{});
            }
        };
        // (end of the timed call)
        const auto fillThrust = [&] { std::memset(thrustOut.data(), 0xFF, N * sizeof(Five)); };
        const auto downloadThrust = [&] {
            std::vector<Statistics> s(N);
            for (int i = 0; i < N; ++i) {
                for (int c = 0; c < 3; ++c) {
                    s[i].minimum[c] = thrustOut[i].minimum[c];
                    s[i].maximum[c] = thrustOut[i].maximum[c];
                    s[i].nonZero[c] = thrustOut[i].nonZero[c];
                    s[i].sum[c] = thrustOut[i].sum[c];
                    s[i].sumOfSquares[c] = thrustOut[i].sumOfSquares[c];
                }
            }
            return s;
        };
        rows.push_back({"thrust", fklb::ResizeSampling::ASYMMETRIC, N, 0, callThrust, fillThrust, downloadThrust});

        if (rows.size() != ROWS.size()) throw std::runtime_error("the rows are not the ones of ROWS");
        for (size_t k = 0; k < rows.size(); ++k) {
            if (rows[k].version != ROWS[k]) throw std::runtime_error("the rows are not the ones of ROWS");
        }

        // ---- The profile mode: a warm-up call outside any range, then E20_PROFILE_CALLS calls, each in the range "<row>:call"
        if (profile) {
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                row.call();
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePushA((row.version + ":" + std::to_string(N)).c_str());
                for (int k = 0; k < profileCalls; ++k) {
                    nvtxRangePushA((row.version + ":call").c_str());
                    row.call();
                    nvtxRangePop();
                }
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePop();
            }
        }

        // ---- The checks of N, on the probe and then on the frame that is timed (which stays in the copies of the libraries): one call of
        // every selected row, after the buffers that every row writes are filled
        bool batchOk = true;
        if (!profile) {
            const double n = static_cast<double>(OUT_ELEMENTS);
            // The criterion of the mean, the variance and the standard deviation, as the table has it: written from the constant
            std::ostringstream relativeText;
            relativeText << "relative difference at most " << RELATIVE_TOLERANCE;
            const std::string relativeCriterion = relativeText.str();
            for (const char* frameName : {"probe", "benchmark"}) {
                const bool isProbe = std::string(frameName) == "probe";
                const References& references = isProbe ? probeReferences : benchmarkReferences;
                for (const auto& upload : uploads) upload(isProbe ? probe : frame);
                for (const Row& row : rows) {
                    if (!selected(row.version)) continue;
                    for (const Row& any : rows) any.fill();
                    FKLB_CUDA_CHECK(cudaDeviceSynchronize());
                    row.call();
                    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                    const std::vector<Statistics> s = row.download();
                    const bool asymmetric = row.sampling == fklb::ResizeSampling::ASYMMETRIC;
                    // The reference of the row's convention, and the one of the other convention
                    const std::vector<Reference>& own = references.of[asymmetric ? 0 : 1];
                    const std::vector<Reference>& other = references.of[asymmetric ? 1 : 0];
                    // One line per statistic: value(crop, channel) of the row against expected(reference of the crop, channel), over the
                    // three channels of the N crops
                    const auto check = [&](const char* statistic, const char* comparedWith, const std::function<double(int, int)>& value,
                                           const std::function<double(const Reference&, int)>& expected, const double tolerance,
                                           const char* criterion) {
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
                        checksTable->row(N, frameName, row.version, asymmetric ? "ASYMMETRIC" : "HALF_PIXEL", statistic, comparedWith, 3 * N,
                                         differing, largest, criterion, ok ? "yes" : "no", differingFromOther);
                        batchOk = batchOk && ok;
                        if (!ok) {
                            std::cout << "N " << N << " " << frameName << " " << row.version << " " << statistic << " against " << comparedWith
                                      << ": " << differing << " of " << 3 * N << " values differ, largest relative difference " << largest << "; "
                                      << criterion << ": FAILED" << std::endl;
                        }
                    };
                    check("minimum", "host", [&](int i, int c) { return s[i].minimum[c]; },
                          [](const Reference& r, int c) { return r.minimum[c]; }, 0, "equal");
                    check("maximum", "host", [&](int i, int c) { return s[i].maximum[c]; },
                          [](const Reference& r, int c) { return r.maximum[c]; }, 0, "equal");
                    check("non_zero", "host", [&](int i, int c) { return s[i].nonZero[c]; },
                          [](const Reference& r, int c) { return r.nonZero[c]; }, 0, "equal");
                    check("sum", "host", [&](int i, int c) { return s[i].sum[c]; }, [](const Reference& r, int c) { return r.sum[c]; }, 0, "equal");
                    check("sum_of_squares", "host", [&](int i, int c) { return s[i].sumOfSquares[c]; },
                          [](const Reference& r, int c) { return r.sumOfSquares[c]; }, 0, "equal");
                    // The mean, the variance and the standard deviation of a channel of a crop, from the sums that the row left
                    const char* fromTheSums = "host; computed on the host from what the row left";
                    const auto variance = [&](int i, int c) { return s[i].sumOfSquares[c] / n - (s[i].sum[c] / n) * (s[i].sum[c] / n); };
                    check("mean", fromTheSums, [&](int i, int c) { return s[i].sum[c] / n; }, [](const Reference& r, int c) { return r.mean[c]; },
                          RELATIVE_TOLERANCE, relativeCriterion.c_str());
                    check("variance", fromTheSums, variance, [](const Reference& r, int c) { return r.variance[c]; }, RELATIVE_TOLERANCE,
                          relativeCriterion.c_str());
                    check("std_dev", fromTheSums, [&](int i, int c) { return std::sqrt(variance(i, c)); },
                          [](const Reference& r, int c) { return r.stdDev[c]; }, RELATIVE_TOLERANCE, relativeCriterion.c_str());
                }
            }
            checksTable->flush();
            std::cout << "N " << N << ": " << (batchOk ? "checks passed" : "CHECK FAILED") << std::endl;
            allOk = allOk && batchOk;
        }

        // ---- The timing of N, after its checks, in the order of the rows: two intervals over the whole frame (the N crops)
        if (timing && batchOk) {
            if (!times) {
                times = std::make_unique<fklb::CsvWriter>(
                    "e20_reductions.csv",
                    "E20 reductions: whole-call and kernel-only times of one frame (the five statistics of N crops, each resized to 224 x 224)",
                    std::initializer_list<const char*>{"crops", "version", "sampling", "library_calls", "intermediate_image_bytes", "interval",
                                                       "mean_ms", "min_ms", "iterations", "spin_us"});
            }
            const auto nothingToPrepare = [] { return 0; };
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                const char* sampling = row.sampling == fklb::ResizeSampling::ASYMMETRIC ? "ASYMMETRIC" : "HALF_PIXEL";
                fklb::Timing whole, gpu, host;
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePushA(("whole_call:" + row.version + ":" + std::to_string(N)).c_str());
                fklb::timeIt(cs, row.call, whole);
                nvtxRangePop();
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePushA(("kernel_only:" + row.version + ":" + std::to_string(N)).c_str());
                fklb::timePreparedLaunch(cs, nothingToPrepare, spin, [&](int) { row.call(); }, gpu, host);
                nvtxRangePop();
                times->row(N, row.version, sampling, row.calls, row.intermediateBytes, "whole_call", whole.mean, whole.min, fklb::ITERATIONS, "");
                times->row(N, row.version, sampling, row.calls, row.intermediateBytes, "kernel_only", gpu.mean, gpu.min, fklb::ITERATIONS,
                           spinUs());
                times->flush();
                std::cout << "N " << N << " " << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min
                          << "); kernel only mean " << gpu.mean << " ms (min " << gpu.min << ")" << std::endl;
            }
        } else if (timing) {
            std::cout << "N " << N << ": not timed: a check failed" << std::endl;
        }

        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        for (void* p : std::vector<void*>{nppFloat, nppResized, nppBatchSrc, nppBatchDst, nppBatchRoi, nppMin, nppMax, nppSum, nppCount,
                                          nppDotProd, minMaxBuffer, sumBuffer, countBuffer, dotProdBuffer}) {
            FKLB_CUDA_CHECK(cudaFree(p));
        }
    }
    if (profile) {
        std::cout << "profile mode: " << profileCalls << " calls per selected row and N, each in its NVTX range" << std::endl;
    } else {
        std::cout << (allOk ? "E20 reductions checks passed" : "E20 reductions CHECK FAILED") << std::endl;
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    FKLB_CUDA_CHECK(cudaFree(flat));
    FKLB_CUDA_CHECK(cudaFree(nppFrame));
    // The wrapper of the stream is released before the stream is destroyed
    cvs = cv::cuda::Stream::Null();
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
