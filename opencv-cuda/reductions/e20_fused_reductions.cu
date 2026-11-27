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

// E20 fused reductions (NVIDIA): reductions of an image that is never written. The mean and the standard deviation of every channel of a
// crop of an 8-bit, three-channel frame resized to 224 x 224 (bilinear): what a per-image standardisation needs before it can subtract
// and divide. FKL's ReduceDPP takes any read: its read is the crop view followed by fk::Resize, so one kernel interpolates every pixel of
// the resized crop and accumulates it, and the resized crop exists in no memory. OpenCV-CUDA and NPP write the resized crop into an
// intermediate image, and further calls, each of which reads that image, its planes or a four-channel copy of it again, take its sums,
// its statistics or its histograms. Thrust and CUB are the single pass written by hand: a transform-reduce whose transform is the
// bilinear interpolation of FKL written out.
//
// The rows are of two chains, which reduce different pixels and so give different numbers (the column "pixels" of the tables: float, or
// 8u_ties_to_even and 8u_ties_up, the two roundings of the 8-bit chain). A row is compared with the rows of its chain:
//   float chain  the first two steps of E4's pre-processing chain (8UC3 crop -> float, bilinear resize): the statistics of the resized
//                pixels as floats, the image that a pipeline which goes on in float standardises. cv::cuda::resize and nppiResize of an
//                8-bit image round every resized pixel to 8 bits, so OpenCV-CUDA and NPP convert the whole crop to float before they
//                resize it; FKL's Resize reads the bytes and gives the floats. So the conversion of the baselines grows with the crop, and
//                the rest of every row depends on the 224 x 224 pixels only: the crop sizes of the sweep show both.
//   8-bit chain  the resize of the 8-bit crop: the statistics of the resized pixels rounded to 8 bits. It is what OpenCV-CUDA and NPP offer
//                to a user who only needs the statistics: no conversion to float, no call that reads the whole crop (the resize reads four
//                pixels of the crop for each resized one) and, the image being 8-bit, its histograms as another way to its sums (the
//                hist rows). FKL computes this quantity in one kernel too: its read ends with the rounding. These rows are here so
//                that the conversion of the float chain, which grows with the crop, is not taken for a gain of fusing the resize with
//                the reductions: here no baseline converts the crop.
// How far apart the two chains are depends on the crop and on the rounding of the ties (the resized pixels halfway between two integers,
// which the libraries round differently: see "rounding"). The default crop sizes make many ties, 16% to 37% of the 150,528 resized values
// with ASYMMETRIC and 1% to 12% with HALF_PIXEL, and with them the mean of the rounded pixels is within 4.4e-5 of the mean of the float
// pixels (relative: 0.006 grey levels) with ASYMMETRIC and ties to even, within 9.7e-5 (0.013 grey levels) with HALF_PIXEL and ties to
// even, and within 1.1e-3 (0.14 grey levels) with HALF_PIXEL and ties up, a rule that moves every tie the same way; the standard
// deviation is within 4.8e-5 in the three. (Over the 9,916 crop sizes that the program accepts, the three means stay within 4.5e-4,
// 5.3e-4 and 4e-3, which is half a grey level, and the standard deviations within 1.8e-4.) The checks table has both references in every
// line, and the program prints the largest distances and the ties of every crop.
//
// Every row leaves in device memory (the thrust row: in host memory), per channel, the sum and the sum of the squares of the 50,176 pixels
// of its chain (the hist rows: the 256 counts of the values of a byte, from which the host computes the two sums), or the mean and the
// standard deviation themselves (the msd rows); from the sums, the host computes the mean and the standard deviation (the population
// one, sqrt(sum of squares / N - mean^2)) in double.
//
// The frame and the crops:
//   frame    one 3840 x 2160 8UC3 frame of E4's seeded noise (the LCG of E4, E18 and E19, seed 12345; the bytes 31..24, 23..16 and 15..8 of
//            every state, row by row)
//   crops    the crop starts at (64, 64) and has one of the sizes of E20_CROPS, a list of <width>x<height> (default
//            280x280,560x420,1008x784,1904x1064: 1.25 x 1.25, 2.5 x 1.875, 4.5 x 3.5 and 8.5 x 4.75 times the resized size); every size
//            is checked and then timed before the next one (a size whose check fails is not timed, and the others are: the timing table
//            of a run that ends with CHECK FAILED, exit code 1, holds the sizes that passed). The program treats the crop as one that
//            changes on every frame: every call reads its position from the detection and builds its ROI, crop pointer or crop view
//            from it
//   factors  every implementation computes the source position of a resized pixel in float, so a crop size must be one whose positions
//            are exact in float (as the factor 15/16 of E4c): they are then the positions of the reference, which is in double. That is
//            not enough to compare sums for equality. With the finest weights that such sizes give (multiples of 1/64) the sum of the
//            squares does not fit in a double, and NPP's nppiSum_32f_C3R_Ctx and nppiMean_StdDev_32f_C1R_Ctx (the one inside
//            cv::cuda::meanStdDev) do not return the sum of the floats in double when the pixels have many significant bits (their
//            results are up to 6e-9 and 3e-8 from the exact ones: a part of their accumulation is not in double). So the program refuses
//            a crop size that is not a multiple of 28 in both dimensions (a factor that is a multiple of 1/8): the weights of both
//            conventions are multiples of 1/16, a resized pixel has at most 16 significant bits and is exact in float, the sums of the
//            pixels and of their squares are exact in double in any order, and so are NPP's. In the 8-bit chain the pixel of every
//            library is that exact pixel rounded with its rule (see "rounding"), and the sums are sums of integers. The sums of every
//            row can then be compared for equality (checked for the 9,916 such sizes of the frame, 28 x 28 to 3752 x 2072; of the 576
//            sizes 7a x 7b with a, b <= 24, whose positions are all exact, only 239 pass). (With a size whose positions are not exact,
//            the position in float differs from the one in double by up to 2^-24 of the position, which moves a resized pixel of noise
//            more than a float rounds: the largest differences to the reference of a 1920 x 1080 crop are then about 2e-7 (the sums
//            and the means), 2e-6 (the sums of squares) and 6e-6 (the standard deviations) in the rows of the float chain, and about
//            6e-6, 1.3e-5 and 3e-5 in the rows of the 8-bit chain, where a pixel that moves can round to another integer.)
// Each library reads its own copy of the same host bytes, crops its own way (OpenCV: a GpuMat ROI; NPP: a pointer into its frame and the
// frame's step; CUB and Thrust: a pointer into a contiguous copy; FKL: fk::Ptr2D::crop2D), and owns its intermediate images and its
// outputs.
//
// The sampling convention of the resize (the names are those of fk::ResizeSampling and of common/resize_reference.h; see E4):
//   ASYMMETRIC (source position x * scale): FKL by default (fkl, fkl_8u, fkl_8u_double_sums), cv::cuda::resize (the opencv rows), the
//                                           single pass written by hand (cub, thrust and thrust_reduce_into, which write out FKL's
//                                           arithmetic)
//   HALF_PIXEL ((x + 0.5) * scale - 0.5):   NPP (the npp rows), FKL with fk::ResizeSampling::HALF_PIXEL (fkl_half_pixel and
//                                           fkl_8u_half_pixel)
// The two conventions read different source positions, so they give different statistics: every row is checked against the reference of
// its OWN convention.
// The rounding of the 8-bit chain ("rounding"): every library gives the resized pixel rounded to the nearest integer, and they differ in
// the ties:
//   ties to even   cv::cuda::resize (its kernel ends with saturate_cast<uchar> of the float, the PTX conversion cvt.rni.sat.u8.f32:
//                  cudawarping/src/cuda/resize.cu and core/cuda/saturate_cast.hpp), and fk::SaturateCast<float3, uchar3> of the FKL rows
//                  (cxp::nearbyint)
//   ties up        nppiResize_8u_C3R_Ctx. Its header (nppi_geometry_transforms.h) does not give the rounding: it is the rule that gives
//                  NPP's sums, found by comparison (the checks of npp_8u pass with it at every crop size, and fail with the other rule
//                  at the default ones)
// So fkl_8u, fkl_8u_double_sums, opencv_cuda_8u and opencv_cuda_8u_hist reduce the same pixels, and so do npp_8u and npp_8u_hist;
// fkl_8u_half_pixel reduces the pixels of NPP's convention with FKL's rounding, which are not NPP's pixels. Every row is checked against
// the reference of its OWN convention and rounding.
//
// The rows, each one a call that leaves its statistics in device memory (the hist rows: the counts that give them). Nothing is brought to
// the host inside a call, except in the thrust row, and the host does not wait for the GPU inside a call, except in the thrust row. The
// rows of the float chain:
//   fkl              one fk::executeOperations<fk::ReduceDPP<>>, one kernel. The read is fk::PerThreadRead<_2D, uchar3> of the crop view
//                    .then(fk::Resize<INTER_LINEAR>::build(Size(224, 224))), which gives a float3 per resized pixel. Two reductions with
//                    double3 accumulators:
//                      sum             Cast<float3, double3>,                   Add<double3>,  Cast<double3, double3> (nothing)
//                      sum of squares  SquareAsDouble (as double3; below),      Add<double3>,  Cast<double3, double3>
//                    The accumulators are doubles, as the ones of OpenCV-CUDA: cv::cuda::calcSum and calcSqrSum accumulate in double
//                    (cudaarithm, sum.cu: sumImpl and sumSqrImpl with a double result, the pixel converted to double before it is
//                    squared). NPP gives its sums and its dot products as doubles (Npp64f); its accumulators are not public. calcSum
//                    and calcSqrSum give the exact sums of the floats and of their squares, and nppiDotProd_32f64f_C3R_Ctx the exact
//                    sum of the squares; nppiSum_32f_C3R_Ctx does not give the exact sum for every image (see "factors"). So the fkl
//                    row differs from the opencv_cuda row in its single kernel and in the images that it does not write, and not in
//                    the type of its accumulators. (A float3 accumulator does not hold these sums: the sum of the pixels has up to 32
//                    significant bits, and the sum of their squares up to 48. It would give another, inexact quantity, and is no row.)
//                    SquareAsDouble is an operation of this program (a user's operation): FKL's basic operations have no square. The
//                    write is fk::PerThreadWrite<_1D, double3> of two elements. The workspace of the kernel is device memory of the
//                    caller: one for the FKL rows, allocated and zeroed once for each crop size, with the bytes of the row that needs
//                    most (they depend on the size of the resized crop and on the accumulators, not on the crop, so one for all the
//                    crop sizes would do). Every call builds the crop view, the read with its Resize, the reductions, the write and the
//                    details, and launches 1 kernel. No intermediate image.
//   fkl_half_pixel   the same with fk::ResizeSampling::HALF_PIXEL, the convention of NPP.
//   cub              cub::DeviceReduce::Reduce over a transform iterator on a counting iterator: resized pixel k is interpolated from the
//                    bytes of a contiguous copy of the frame with FKL's arithmetic written out (the same factors, the same order of the
//                    operations), and becomes a struct of six doubles, the three sums and the three sums of squares. The result in
//                    device memory, the temporary storage allocated once for each crop size. No intermediate image.
//   thrust           thrust::transform_reduce over the counting iterator with the same operations (thrust::cuda::par on the stream). It
//                    returns the result, so it is NOT like the other rows: the call waits for the result and copies it to the host,
//                    where the row keeps it, and it allocates and frees its temporary storage in device memory (per call: 1
//                    cudaMalloc, 1 cudaFree, 1 cudaMemcpyAsync and 2 cudaStreamSynchronize). No intermediate image.
//   thrust_reduce_into  thrust::reduce_into (CCCL 3.2, CUDA 13.2) over the range of the cub row, into device memory, with
//                    thrust::cuda::par_nosync on the stream, so the call does not wait, and with a block of device memory of the caller
//                    as the allocator of the policy (allocated once for each crop size, with the bytes that the cub row asks for), so
//                    the call allocates nothing: a call like the ones of the other rows, with the kernels of the cub row. No
//                    intermediate image.
//   opencv_cuda      GpuMat::convertTo (the ROI of the crop to CV_32FC3), cv::cuda::resize (INTER_LINEAR, to 224 x 224),
//                    cv::cuda::calcSum and cv::cuda::calcSqrSum of the resized image (CV_64FC3 each). 4 calls; the float crop and the
//                    resized image are intermediate images, allocated once for each crop size.
//   opencv_cuda_msd  convertTo, cv::cuda::resize, cv::cuda::split, and cv::cuda::meanStdDev of each of the three planes (meanStdDev takes
//                    one-channel images only). 6 calls, and the three planes as further intermediate images. cv::cuda::meanStdDev takes
//                    NPP's scratch buffer from the BufferPool of the stream, which this program turns on before it wraps its stream
//                    (cv::cuda::setBufferPoolUsage(true), once for the run: no call of this row allocates, as no call of the NPP and
//                    FKL rows does, whose scratch buffers and workspace are allocated once for each crop size). With OpenCV's default,
//                    the pool off, every cv::cuda::meanStdDev allocates the buffer and frees it when it returns, and cudaFree waits for
//                    the work of the stream: 3 allocations and 3 waits per call of the row. One other call of the OpenCV rows takes a
//                    buffer from the pool, the cv::cuda::histEven of opencv_cuda_8u_hist (with the pool off: 1 allocation and 1 wait
//                    per call of that row).
//   npp              nppiConvert_8u32f_C3R_Ctx (the crop), nppiResize_32f_C3R_Ctx (NPPI_INTER_LINEAR), nppiSum_32f_C3R_Ctx and
//                    nppiDotProd_32f64f_C3R_Ctx of the resized image with itself (the sums of the squares: NPP has no call named for
//                    them, and its other call for them, nppiNorm_L2_32f_C3R_Ctx, gives their square roots), Npp64f[3] each. 4 calls;
//                    the float crop and the resized image are intermediate images, and the scratch buffer of every call is allocated
//                    once for each crop size.
//   npp_msd          nppiConvert_8u32f_C3R_Ctx, nppiResize_32f_C3R_Ctx and nppiMean_StdDev_32f_C3CR_Ctx for each of the three channels.
//                    5 calls.
// The rows of the 8-bit chain:
//   fkl_8u           one kernel, as fkl. The read is the one of fkl .then(fk::SaturateCast<float3, uchar3>), which rounds the resized
//                    pixel and gives a uchar3. Two reductions with uint3 accumulators, the types a user would choose for bytes (exact:
//                    50,176 squares of at most 255^2 fit in 32 bits), whose results are written as double3:
//                      sum             Cast<uchar3, uint3>,                     Add<uint3>,    Cast<uint3, double3>
//                      sum of squares  SquareOfBytes (as uint3; a user's),      Add<uint3>,    Cast<uint3, double3>
//                    The write, the workspace and what a call builds are the ones of fkl. 1 kernel, no intermediate image.
//   fkl_8u_half_pixel  the same with fk::ResizeSampling::HALF_PIXEL.
//   fkl_8u_double_sums  fkl_8u with double3 accumulators (Cast<uchar3, double3> and SquareOfBytesAsDouble, a user's; Add<double3>), the
//                    ones of opencv_cuda_8u: cv::cuda::calcSum and calcSqrSum accumulate the bytes in double too. fkl_8u differs from
//                    opencv_cuda_8u in two things, its single kernel and the type of its accumulators, and this row in the first only:
//                    the two FKL rows separate what the type of the accumulators costs in FKL's kernel from the rest.
//   opencv_cuda_8u   cv::cuda::resize of the 8UC3 ROI of the crop (INTER_LINEAR), cv::cuda::calcSum and cv::cuda::calcSqrSum of the
//                    resized 8-bit image (CV_64FC3 each). 3 calls; the resized image is the intermediate image.
//   opencv_cuda_8u_hist  the same resize, cv::cuda::cvtColor of the resized image to four channels (COLOR_BGR2BGRA), and cv::cuda::histEven
//                    of the four-channel image with 256 bins from 0 to 256 (bins of one value; NPP's nppiHistogramEven_8u_C4R inside,
//                    with its scratch buffer from the BufferPool of the stream): 256 counts (CV_32SC1) per channel, the four channels
//                    in one call; the counts of the fourth channel, the one that cvtColor adds, are not used. 3 calls, and the
//                    four-channel image as a further intermediate image. The counts are GpuMat headers of memory of the program (the
//                    GpuMat constructor that takes a pointer), at the addresses of the counts of npp_8u_hist (below), because the
//                    time of the call depends on them in the same way: on the RTX 4090, kernel_only is 9.2, 9.3, 9.4 and 10.2 us at
//                    the four default crops there, and 10.3 to 10.9 us at the three smaller ones and 11.3 us at the largest with the
//                    counts where OpenCV allocates them. Not cv::cuda::calcHist: it takes one-channel 8-bit images only, so it needs
//                    cv::cuda::split and three calls (5 calls in all), which take longer than these 3 (measured on the RTX 4090 with
//                    the counts of both where OpenCV allocates them: about 13.5 us against 12 us in whole_call, and about 12.5 us
//                    against 10.5 us in kernel_only).
//   npp_8u           nppiResize_8u_C3R_Ctx of the crop (NPPI_INTER_LINEAR), nppiSum_8u_C3R_Ctx and nppiDotProd_8u64f_C3R_Ctx of the
//                    resized image with itself (Npp64f[3] each). 3 calls; the resized image is the intermediate image.
//   npp_8u_hist      the same resize, and nppiHistogramEven_8u_C3R_Ctx with 257 levels from 0 to 256 (256 bins of one value): 256
//                    counts (Npp32s) per channel, the three channels in one call. 2 calls; its scratch buffer is allocated once for
//                    each crop size. The time of the histogram call depends on the addresses of the counts (see the hist rows of
//                    e20_reductions.cu). On the RTX 4090, kernel_only at the four default crops: 7.8, 8.1, 8.2 and 8.2 us with the
//                    counts of each channel 512 bytes after a multiple of 4,096 bytes, each channel in its own 4,096 bytes; 8.3 to
//                    8.8 us at the three smaller crops and 9.2 us at the largest with the three one after the other from such an
//                    address; and 9.3, 9.3, 9.3 and 9.6 us with them one after the other from a multiple of 1,024 bytes (cudaMalloc
//                    gives such addresses). The program puts them at the first, which no address that was measured beat.
// The hist rows are another route of OpenCV-CUDA and NPP to the same statistics, which the timing compares with their sums: they leave
// the counts of the 256 values of a byte in device memory, NOT the sums, and the host computes the two sums of a channel from its counts
// (256 steps, exact) where the other rows read them back. The route exists for the 8-bit images of this chain only: the float pixels of
// the other chain are not 256 values. The single pass written by hand (cub, thrust, thrust_reduce_into) is a row of the float chain only.
// The OpenCV-CUDA, NPP, Thrust and CUB rows use no FKL object.
//
// The checks (e20_fused_reductions_checks.csv), for every crop size. The host resizes the crop with fklb::resizeBilinear (the scalar
// bilinear resize in double of common/resize_reference.h), in each convention, and makes three references of each: of the resized
// pixels as they are (the float chain), and of the resized pixels rounded to the nearest integer with the ties to even, and with the ties
// up (the 8-bit chain; a resized pixel is exact in double, see "factors", so the ties of the host are the ties). A reference holds, for
// every channel, the sum of its pixels and the sum of their squares (exact, see "factors"), and the mean and the population standard
// deviation (from the differences to the mean). Each row is called once, after the buffers that EVERY row writes, its intermediate
// images and its statistics, are filled with 0xFF bytes (a NaN in a float and in a double and -1 in a count: never right values; 255
// in a byte of an 8-bit image, an image of mean 255, which no crop of this frame resizes to; filling them all shows a row that reads
// what another one left, as an msd row that skips its resize would), its statistics are read back, and for every channel, against the
// reference of the row's convention and pixels:
//   count            the hist rows: the sum of the 256 counts, equal to 50,176 (the pixels of value 0 are in no sum). The timing needs
//                    it.
//   sum              the rows that compute it (all but the msd rows; the hist rows, on the host from their counts): equal. The timing
//                    needs it.
//   sum_of_squares   the rows that compute it (all but the msd rows; the hist rows, on the host from their counts): equal. The timing
//                    needs it.
//   mean, std_dev    every row: computed on the host from the sums that the row left, or as the msd rows left them; relative
//                    difference at most 1e-9. The timing needs it.
// The sums of both chains are checked for equality, the tightest criterion: the 8-bit chain allows it because every library rounds an
// exact resized pixel (see "factors") and its sums are sums of integers.
// The line of each check also holds the same statistic of two other references and their relative distances to the row's value (they are
// not criteria): of the OTHER convention with the pixels of the row, to show that the check tells the conventions apart; and of the float
// pixels of the row's convention (the row's own reference, in the float chain), to show that the check tells the chains apart and how
// far apart they are.
//
// The timing (e20_fused_reductions.csv), for every crop size after its checks: two intervals per row, in the order of the rows, FKL's
// first (fkl, fkl_half_pixel, fkl_8u, fkl_8u_half_pixel, fkl_8u_double_sums, cub, thrust, thrust_reduce_into, opencv_cuda,
// opencv_cuda_msd, npp, npp_msd, opencv_cuda_8u, opencv_cuda_8u_hist, npp_8u, npp_8u_hist), the same in every run:
//   whole_call   fklb::timeIt: CUDA events around one call, 100 calls after one warm-up call
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin
//                of E20_SPIN_US (default 5,000 us). The thrust row waits for its result inside the call, so its interval also holds what
//                its host does after the spin.
//   Each interval of each row is inside an NVTX range ("whole_call:<row>:<crop>", "kernel_only:<row>:<crop>").
// E20_ONLY=a,b: only these rows (a name that is no row stops the program). E20_NO_TIMING=1: the checks only. E20_PROFILE=1: every selected
// row is called once outside any range (a warm-up), then E20_PROFILE_CALLS times (default 10), each call in an NVTX range "<row>:call"
// inside the range "<row>:<crop>"; nothing is compared or timed (to count the kernels of one call with nsys).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/env_args.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/algorithms/image_processing/saturate.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <npp.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---- The user's operations of the FKL rows, which FKL's basic operations do not have
namespace e20 {
// The square of every channel of a resized pixel, in the accumulator type of the sums of squares of the float chain
struct SquareAsDouble {
    using Parent = fk::UnaryOperation<float3, double3, SquareAsDouble>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {static_cast<double>(input.x) * input.x, static_cast<double>(input.y) * input.y, static_cast<double>(input.z) * input.z};
    }
};

// The square of every channel of a resized pixel rounded to 8 bits, in the accumulator type of the sums of squares of fkl_8u and
// fkl_8u_half_pixel
struct SquareOfBytes {
    using Parent = fk::UnaryOperation<uchar3, uint3, SquareOfBytes>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {static_cast<uint>(input.x) * input.x, static_cast<uint>(input.y) * input.y, static_cast<uint>(input.z) * input.z};
    }
};

// The same square, in the accumulator type of the sums of squares of fkl_8u_double_sums
struct SquareOfBytesAsDouble {
    using Parent = fk::UnaryOperation<uchar3, double3, SquareOfBytesAsDouble>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) {
        return {static_cast<double>(input.x) * input.x, static_cast<double>(input.y) * input.y, static_cast<double>(input.z) * input.z};
    }
};
} // namespace e20

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROP_X = 64, CROP_Y = 64;
constexpr int OUT_W = 224, OUT_H = 224;
constexpr int OUT_ELEMENTS = OUT_W * OUT_H;   // 50,176 resized pixels
constexpr double RELATIVE_TOLERANCE = 1e-9;   // the mean and the standard deviation, against the host's in double
// The sums of the 8-bit chain are sums of integers: the largest one fits in the 32-bit accumulators of fkl_8u and fkl_8u_half_pixel
static_assert(static_cast<unsigned long long>(OUT_ELEMENTS) * 255 * 255 < (1ull << 32), "the sums of squares of the 8u rows need more than 32 bits");

// The pixels that a row reduces, which are the ones of its reference: the resized pixels as they are (the float chain), or rounded to
// the nearest integer (the 8-bit chain), a pixel halfway between two integers (a tie) to the even one or up
enum class Rounding { NONE, TIES_TO_EVEN, TIES_UP };

// As the tables name them
const char* pixelsOf(const Rounding rounding) {
    return rounding == Rounding::NONE ? "float" : (rounding == Rounding::TIES_TO_EVEN ? "8u_ties_to_even" : "8u_ties_up");
}

// ---- The single pass of the cub, thrust and thrust_reduce_into rows: the sums and the sums of squares of the three channels, in double
struct Moments {
    double sum[3];
    double sumOfSquares[3];
};

// Resized pixel k, row by row, of a crop of a contiguous frame of 3 bytes per pixel: FKL's bilinear interpolation written out (fk::Resize
// with the default sampling and fk::Interpolate<INTER_LINEAR>: the source position is the pixel times the factor, in float; its four
// neighbours, the second ones not beyond the last pixel of the crop; the four products in the order of FKL), as the moments of one pixel
struct MomentsOfResizedPixel {
    const unsigned char* crop;   // the first byte of the crop in the frame
    int framePitch;              // bytes between two rows of the frame
    int cropW, cropH;
    float factorX, factorY;      // source pixels per resized pixel, as fk::Resize::build computes them
    __host__ __device__ Moments operator()(const int k) const {
        const int x = k % OUT_W, y = k / OUT_W;
        const float srcX = x * factorX, srcY = y * factorY;
        const int x1 = static_cast<int>(floorf(srcX)), y1 = static_cast<int>(floorf(srcY));
        const int x2 = x1 + 1, y2 = y1 + 1;
        const int x2Read = x2 < cropW - 1 ? x2 : cropW - 1, y2Read = y2 < cropH - 1 ? y2 : cropH - 1;
        const unsigned char* p00 = crop + static_cast<long long>(y1) * framePitch + 3 * x1;
        const unsigned char* p10 = crop + static_cast<long long>(y1) * framePitch + 3 * x2Read;
        const unsigned char* p01 = crop + static_cast<long long>(y2Read) * framePitch + 3 * x1;
        const unsigned char* p11 = crop + static_cast<long long>(y2Read) * framePitch + 3 * x2Read;
        const float w00 = (x2 - srcX) * (y2 - srcY), w10 = (srcX - x1) * (y2 - srcY), w01 = (x2 - srcX) * (srcY - y1),
                    w11 = (srcX - x1) * (srcY - y1);
        Moments m;
        for (int c = 0; c < 3; ++c) {
            const float v = (p00[c] * w00) + (p10[c] * w10) + (p01[c] * w01) + (p11[c] * w11);
            m.sum[c] = v;
            m.sumOfSquares[c] = static_cast<double>(v) * v;
        }
        return m;
    }
};

struct MomentsOfTwo {
    __host__ __device__ Moments operator()(const Moments& a, const Moments& b) const {
        return Moments{{a.sum[0] + b.sum[0], a.sum[1] + b.sum[1], a.sum[2] + b.sum[2]},
                       {a.sumOfSquares[0] + b.sumOfSquares[0], a.sumOfSquares[1] + b.sumOfSquares[1], a.sumOfSquares[2] + b.sumOfSquares[2]}};
    }
};

// The value that changes no other: the initial value of the reductions of cub and thrust
constexpr Moments MOMENTS_OF_NOTHING{{0, 0, 0}, {0, 0, 0}};

// The temporary storage of the thrust_reduce_into row: one block of device memory that the caller owns, as the storage of the cub row,
// given to Thrust as the allocator of its execution policy (the idiom of Thrust's custom_temporary_allocation example, with one block).
// Thrust asks for bytes (value_type is char), once in every call, and gives the block back before the call returns
struct BlockOfTheCaller {
    using value_type = char;
    char* block = nullptr;
    size_t bytes = 0;
    bool taken = false;
    char* allocate(const std::ptrdiff_t count) {
        if (taken) throw std::runtime_error("thrust_reduce_into: Thrust asks for a second block of temporary storage");
        if (static_cast<size_t>(count) > bytes) {
            throw std::runtime_error("thrust_reduce_into: Thrust asks for " + std::to_string(count) + " bytes of temporary storage, and its block " +
                                     "of device memory has " + std::to_string(bytes));
        }
        taken = true;
        return block;
    }
    void deallocate(char*, size_t) { taken = false; }
};

// The rows, in the order of the checks and of the timing: FKL's first, then the single pass written by hand, the float chain of
// OpenCV-CUDA and NPP, and their 8-bit chain
const std::vector<std::string> ROWS = {"fkl", "fkl_half_pixel", "fkl_8u", "fkl_8u_half_pixel", "fkl_8u_double_sums", "cub", "thrust",
                                       "thrust_reduce_into", "opencv_cuda", "opencv_cuda_msd", "npp", "npp_msd", "opencv_cuda_8u",
                                       "opencv_cuda_8u_hist", "npp_8u", "npp_8u_hist"};

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
        if (std::find(ROWS.begin(), ROWS.end(), name) == ROWS.end()) {
            throw std::runtime_error("E20_ONLY: '" + name + "' is not a row of this program");
        }
        ++names;
    }
    if (names == 0) throw std::runtime_error("E20_ONLY: no row");
}

// A positive integer of at most 9 digits and nothing else: 0 otherwise ("1e3", "16;32" and " 16" are not sizes)
int positiveInt(const std::string& text) {
    if (text.empty() || text.size() > 9 || text.find_first_not_of("0123456789") != std::string::npos) return 0;
    return std::atoi(text.c_str());
}

// A value of a check as the tables hold it: every digit of a double (fklb::CsvWriter writes 9), so that two values that are compared for
// equality are the numbers that the file shows
std::string exact(const double value) {
    std::ostringstream s;
    s << std::setprecision(17) << value;
    return s.str();
}

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

struct Size {
    int width, height;
};

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

// E20_CROPS: <width>x<height>,<width>x<height>...
std::vector<Size> cropsOfTheRun() {
    const char* e = std::getenv("E20_CROPS");
    std::stringstream list(e != nullptr ? e : "280x280,560x420,1008x784,1904x1064");
    std::vector<Size> sizes;
    std::string item;
    while (std::getline(list, item, ',')) {
        const size_t x = item.find('x');
        if (x == std::string::npos) throw std::runtime_error("E20_CROPS: '" + item + "' is not <width>x<height>");
        const Size size{positiveInt(item.substr(0, x)), positiveInt(item.substr(x + 1))};
        if (size.width < 2 || size.height < 2 || CROP_X + size.width > SRC_W || CROP_Y + size.height > SRC_H) {
            throw std::runtime_error("E20_CROPS: '" + item + "' is not a crop of 2 x 2 or more inside the frame from (64, 64)");
        }
        // Multiples of 28: the factor is a multiple of 1/8 (see "factors" in the header). Their positions are exact in float, which the
        // program also verifies
        if (size.width % 28 != 0 || size.height % 28 != 0 || !positionsAreExact(size.width, OUT_W) || !positionsAreExact(size.height, OUT_H)) {
            throw std::runtime_error("E20_CROPS: '" + item + "' is not a multiple of 28 in both dimensions (224 times a multiple of 1/8, as 280 = "
                                     "224 * 1.25), so its sums cannot be compared for equality");
        }
        sizes.push_back(size);
    }
    if (sizes.empty()) throw std::runtime_error("E20_CROPS: no crop");
    return sizes;
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

// The top-left corner of the crop in the frame
struct Detection {
    int x, y;
};

// What a row leaves, per channel (NaN where it computes no such value)
struct Statistics {
    double sum[3] = {std::nan(""), std::nan(""), std::nan("")};
    double sumOfSquares[3] = {std::nan(""), std::nan(""), std::nan("")};
    double mean[3] = {std::nan(""), std::nan(""), std::nan("")};
    double stdDev[3] = {std::nan(""), std::nan(""), std::nan("")};
    double count[3] = {std::nan(""), std::nan(""), std::nan("")};   // a hist row: the pixels that it counted, the sum of its 256 counts
};

// What a hist row leaves: of every channel, the 256 counts of the values of a byte, one channel after the other. The host computes from
// them the pixels that the row counted, their sum and the sum of their squares (integers below 2^32: exact in a double)
Statistics statisticsOfCounts(const std::vector<int>& counts) {
    Statistics s;
    for (int c = 0; c < 3; ++c) {
        s.count[c] = s.sum[c] = s.sumOfSquares[c] = 0;
        for (int value = 0; value < 256; ++value) {
            const double count = counts[static_cast<size_t>(c) * 256 + value];
            s.count[c] += count;
            s.sum[c] += count * value;
            s.sumOfSquares[c] += count * value * value;
        }
    }
    return s;
}

// Of every channel of the crop resized on the host in double, for one convention and one rounding of the resized pixels: the sum of the
// pixels and the sum of their squares (exact), and the mean and the population standard deviation; and the resized values of the three
// channels that are ties (halfway between two integers), which the two rules of the 8-bit chain round differently
struct Reference {
    double sum[3] = {0, 0, 0}, sumOfSquares[3] = {0, 0, 0}, mean[3] = {0, 0, 0}, stdDev[3] = {0, 0, 0};
    long long ties = 0;
};

Reference referenceOf(const std::vector<unsigned char>& frame, const Detection& detection, const Size& crop, const fklb::ResizeSampling sampling,
                      const Rounding rounding) {
    std::vector<double> resized(static_cast<size_t>(OUT_ELEMENTS) * 3);
    const unsigned char* base = frame.data() + (static_cast<size_t>(detection.y) * SRC_W + detection.x) * 3;
    fklb::resizeBilinear<unsigned char>(base, static_cast<size_t>(SRC_W) * 3, crop.width, crop.height, 3, OUT_W, OUT_H, sampling, resized.data());
    Reference r;
    // The resized values are exact in double (see "factors" in the header), so a tie of the host is a tie: std::nearbyint rounds it to the
    // even integer (the rounding mode of the program is the default one, to the nearest), and std::floor(value + 0.5) rounds it up
    for (double& value : resized) {
        r.ties += value - std::floor(value) == 0.5 ? 1 : 0;
        if (rounding == Rounding::TIES_TO_EVEN) {
            value = std::nearbyint(value);
        } else if (rounding == Rounding::TIES_UP) {
            value = std::floor(value + 0.5);
        }
    }
    for (int c = 0; c < 3; ++c) {
        long double sum = 0, sumOfSquares = 0;
        for (int k = 0; k < OUT_ELEMENTS; ++k) {
            const long double v = resized[static_cast<size_t>(k) * 3 + c];
            sum += v;
            sumOfSquares += v * v;
        }
        const long double mean = sum / OUT_ELEMENTS;
        // The sum of the squares of the differences to the mean: no difference of two large numbers
        long double squares = 0;
        for (int k = 0; k < OUT_ELEMENTS; ++k) {
            const long double d = resized[static_cast<size_t>(k) * 3 + c] - mean;
            squares += d * d;
        }
        r.sum[c] = static_cast<double>(sum);
        r.sumOfSquares[c] = static_cast<double>(sumOfSquares);
        // Exact sums are the same numbers in long double and in double: otherwise the sums of the rows could not be compared for equality
        if (static_cast<long double>(r.sum[c]) != sum || static_cast<long double>(r.sumOfSquares[c]) != sumOfSquares) {
            throw std::runtime_error("the sums of the reference are not exact in double");
        }
        r.mean[c] = static_cast<double>(mean);
        r.stdDev[c] = static_cast<double>(std::sqrt(squares / OUT_ELEMENTS));
    }
    return r;
}

double relativeDifference(const double value, const double reference) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    return reference == 0 ? std::abs(value) : std::abs(value - reference) / std::abs(reference);
}

// The histogram calls of NPP, and of OpenCV-CUDA with NPP inside, take a time that depends on the addresses of the counts of the
// channels (see the hist rows in the header). The fastest ones that were measured: the 256 counts of each channel 512 bytes after a
// multiple of 4,096 bytes, each channel in its own 4,096 bytes. The counts of channel c in a block of 4,096 bytes per channel and
// 4,096 more (COUNTS_BLOCK_INTS ints per channel, and as many more)
constexpr size_t COUNTS_BLOCK_INTS = 1024;
int* countsOfChannel(int* block, const int c) {
    const size_t bytesBefore = (512 + 4096 - reinterpret_cast<std::uintptr_t>(block) % 4096) % 4096;
    return block + bytesBefore / sizeof(int) + COUNTS_BLOCK_INTS * c;
}

// The GPU spin of the kernel-only interval, the one of E1, E18 and E19 (see e20_reductions.cu)
__global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E20_SPIN_US: the length of gpuSpin in microseconds, default 5000 (0 to 1,000,000: fklb::envInt stops the program at anything else)
unsigned long long spinUs() { return static_cast<unsigned long long>(fklb::envInt("E20_SPIN_US", 5000, 0, 1000000)); }

// One row: its call, the fill of the buffers that it writes (its intermediate images and its statistics) with 0xFF bytes, and the read back
// of what it left
struct Row {
    std::string version;
    fklb::ResizeSampling sampling;
    Rounding rounding;        // the pixels that the row reduces (its chain, and its rounding in the 8-bit one): the ones of its reference
    int calls;                // library calls of one call of the row
    size_t intermediateBytes; // the intermediate images that one call writes (0: none)
    std::function<void()> call;
    std::function<void()> fill;
    std::function<Statistics()> download;
    bool leavesMeanAndStdDev = false;   // an msd row: it leaves the mean and the standard deviation, and no sum
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
} // namespace

int launch() {
    const bool profile = envIs1("E20_PROFILE");
    const bool timing = !profile && !envIs1("E20_NO_TIMING");
    const int profileCalls = fklb::envInt("E20_PROFILE_CALLS", 10, 1, 1000000);
    refuseUnknownRows();
    std::cout << "E20 fused reductions: "
              << (profile ? "profile mode (a warm-up call, then E20_PROFILE_CALLS calls per row, each in its NVTX range; nothing compared, "
                            "nothing timed)"
                          : (timing ? "for every crop size, the checks and then the timing" : "checks only (E20_NO_TIMING=1)"))
              << std::endl;
    const std::vector<Size> crops = cropsOfTheRun();
    const std::vector<unsigned char> frame = seededNoise(SRC_W, SRC_H);
    // The detection of the frame, the input of every call: every call reads the position of the crop from it and builds its ROI, crop
    // pointer or crop view from it, as it would for a new crop on every frame
    const Detection detection{CROP_X, CROP_Y};
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };

    // Set once: the cv::cuda::Stream and FKL's stream object of the program's stream, and the NPP stream context, filled field by field
    // from the device properties and the stream's flags as in E18 and E19 (CUDA 13 has no nppGetStreamContext)
    // OpenCV's buffer pool, turned on before the stream is wrapped (a cv::cuda::Stream takes its allocator when it is made): cv::cuda::meanStdDev
    // and cv::cuda::histEven then take NPP's scratch buffer from the pool of the stream. With the pool off (OpenCV's default) each of them
    // allocates the buffer and frees it when it returns, and cudaFree waits for the work of the stream
    cv::cuda::setBufferPoolUsage(true);
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

    // Set once: every library's own copy of the frame, uploaded before the calls
    fk::Ptr2D<uchar3> fkFrame(SRC_W, SRC_H);
    FKLB_CUDA_CHECK(cudaMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, frame.data(), static_cast<size_t>(SRC_W) * 3,
                                 static_cast<size_t>(SRC_W) * 3, SRC_H, cudaMemcpyHostToDevice));
    unsigned char* flat = deviceAlloc<unsigned char>(frame.size());   // CUB and Thrust: contiguous, 3 bytes per pixel, no row padding
    FKLB_CUDA_CHECK(cudaMemcpy(flat, frame.data(), frame.size(), cudaMemcpyHostToDevice));
    cv::cuda::GpuMat cvFrame;
    cvFrame.upload(cv::Mat(SRC_H, SRC_W, CV_8UC3, const_cast<unsigned char*>(frame.data())));
    Npp8u* nppFrame = nullptr;
    size_t nppFrameStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &nppFrameStep, static_cast<size_t>(SRC_W) * 3, SRC_H));
    FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, nppFrameStep, frame.data(), static_cast<size_t>(SRC_W) * 3, static_cast<size_t>(SRC_W) * 3, SRC_H,
                                 cudaMemcpyHostToDevice));

    // The checks table is opened unless nothing is checked (the profile mode writes no table)
    std::unique_ptr<fklb::CsvWriter> checksTable;
    if (!profile) {
        checksTable = std::make_unique<fklb::CsvWriter>(
            "e20_fused_reductions_checks.csv", "E20 fused reductions: the checks of every crop size",
            std::initializer_list<const char*>{"crop_width", "crop_height", "version", "sampling", "pixels", "statistic", "channel", "compared_with",
                                               "value", "reference", "relative_difference", "criterion", "ok", "reference_of_the_other_sampling",
                                               "relative_difference_to_the_other_sampling", "reference_of_the_float_pixels",
                                               "relative_difference_to_the_float_pixels"});
    }
    // The timing table is opened when the first crop size is timed: a run of checks only, or one in which no crop size passes its checks,
    // writes none
    std::unique_ptr<fklb::CsvWriter> times;
    bool allOk = true;

    for (const Size& crop : crops) {
        const int CW = crop.width, CH = crop.height;
        const std::string cropName = std::to_string(CW) + "x" + std::to_string(CH);
        // The references of the crop size: of each convention ([0] ASYMMETRIC, [1] HALF_PIXEL), the one of the float pixels and the ones of
        // the pixels rounded to 8 bits with each rule (indexed by Rounding)
        Reference references[2][3];
        if (!profile) {
            for (const fklb::ResizeSampling sampling : {fklb::ResizeSampling::ASYMMETRIC, fklb::ResizeSampling::HALF_PIXEL}) {
                for (const Rounding rounding : {Rounding::NONE, Rounding::TIES_TO_EVEN, Rounding::TIES_UP}) {
                    references[sampling == fklb::ResizeSampling::ASYMMETRIC ? 0 : 1][static_cast<int>(rounding)] =
                        referenceOf(frame, detection, crop, sampling, rounding);
                }
            }
        }
        const Reference& referenceAsymmetric = references[0][static_cast<int>(Rounding::NONE)];
        const Reference& referenceHalfPixel = references[1][static_cast<int>(Rounding::NONE)];
        std::vector<Row> rows;

        // ---- FKL: one fk::executeOperations<fk::ReduceDPP<>> per row, one kernel, no intermediate image
        // Set once for the crop size: the two double3 of the output of every row (the sums and the sums of squares of the three channels)
        fk::Ptr1D<double3> fkOut(2), fkOutHalfPixel(2), fkOut8(2), fkOut8HalfPixel(2), fkOut8DoubleSums(2);
        using FklAdd = fk::Add<double3, double3, double3, fk::UnaryType>;
        using FklSame = fk::Cast<double3, double3>;
        // The reductions of the float chain: double3 accumulators
        const auto fklReductions = [] {
            return fk::make_tuple(fk::make_tuple(fk::Cast<float3, double3>::build(), FklAdd::build(), FklSame::build()),
                                  fk::make_tuple(e20::SquareAsDouble::build(), FklAdd::build(), FklSame::build()));
        };
        // The reductions of the 8-bit chain: uint3 accumulators (fkl_8u, fkl_8u_half_pixel), and the double3 ones of the float chain
        // (fkl_8u_double_sums)
        using FklAdd32 = fk::Add<uint3, uint3, uint3, fk::UnaryType>;
        const auto fklReductions8 = [] {
            return fk::make_tuple(fk::make_tuple(fk::Cast<uchar3, uint3>::build(), FklAdd32::build(), fk::Cast<uint3, double3>::build()),
                                  fk::make_tuple(e20::SquareOfBytes::build(), FklAdd32::build(), fk::Cast<uint3, double3>::build()));
        };
        const auto fklReductions8DoubleSums = [] {
            return fk::make_tuple(fk::make_tuple(fk::Cast<uchar3, double3>::build(), FklAdd::build(), FklSame::build()),
                                  fk::make_tuple(e20::SquareOfBytesAsDouble::build(), FklAdd::build(), FklSame::build()));
        };
        using FklResize = fk::Resize<fk::InterpolationType::INTER_LINEAR>;
        using FklResizeHalfPixel =
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>;
        // The rounding of the 8-bit chain: to the nearest integer, ties to even, and to the range of a byte
        using FklRound = fk::SaturateCast<float3, uchar3>;
        // The read of a call: the view of the crop of the frame, from the detection, and the resize of it
        const auto fklRead = [&] {
            const fk::Ptr2D<uchar3> view = fkFrame.crop2D(fk::Rect(detection.x, detection.y, CW, CH));
            return fk::PerThreadRead<fk::ND::_2D, uchar3>::build(view).then(FklResize::build(fk::Size(OUT_W, OUT_H)));
        };
        const auto fklReadHalfPixel = [&] {
            const fk::Ptr2D<uchar3> view = fkFrame.crop2D(fk::Rect(detection.x, detection.y, CW, CH));
            return fk::PerThreadRead<fk::ND::_2D, uchar3>::build(view).then(FklResizeHalfPixel::build(fk::Size(OUT_W, OUT_H)));
        };
        // The read of a call of the 8-bit chain: the same read, and the rounding of the resized pixel
        const auto fklRead8 = [&] { return fklRead().then(FklRound::build()); };
        const auto fklRead8HalfPixel = [&] { return fklReadHalfPixel().then(FklRound::build()); };
        // Set once for the crop size: the workspace of the kernels, device memory of the caller, zeroed. The bytes that a row needs depend
        // on the size of the resized crop and on its accumulators, and on nothing of a call: the FKL rows, whose calls never run at the
        // same time, share one workspace with the bytes of the row that needs most
        const size_t fkWorkspaceBytes = std::max({fk::ReduceDPP<>::workspaceBytes(fklRead(), fklReductions()),
                                                  fk::ReduceDPP<>::workspaceBytes(fklReadHalfPixel(), fklReductions()),
                                                  fk::ReduceDPP<>::workspaceBytes(fklRead8(), fklReductions8()),
                                                  fk::ReduceDPP<>::workspaceBytes(fklRead8HalfPixel(), fklReductions8()),
                                                  fk::ReduceDPP<>::workspaceBytes(fklRead8(), fklReductions8DoubleSums())});
        fk::Ptr1D<uchar> fkWorkspace(static_cast<uint>(fkWorkspaceBytes));
        if (fkWorkspaceBytes > 0) {
            FKLB_CUDA_CHECK(cudaMemset(fkWorkspace.ptr().data, 0, fkWorkspaceBytes));
        }
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFkl = [&] {
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double3>::build(fkOut);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), fklRead(), fklReductions(), write);
        };
        // (end of the timed call)
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFklHalfPixel = [&] {
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double3>::build(fkOutHalfPixel);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), fklReadHalfPixel(), fklReductions(),
                                                   write);
        };
        // (end of the timed call)
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFkl8 = [&] {
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double3>::build(fkOut8);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), fklRead8(), fklReductions8(), write);
        };
        // (end of the timed call)
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFkl8HalfPixel = [&] {
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double3>::build(fkOut8HalfPixel);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), fklRead8HalfPixel(), fklReductions8(),
                                                   write);
        };
        // (end of the timed call)
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFkl8DoubleSums = [&] {
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double3>::build(fkOut8DoubleSums);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), fklRead8(), fklReductions8DoubleSums(),
                                                   write);
        };
        // (end of the timed call)
        const auto fillFklOf = [](fk::Ptr1D<double3>& out) {
            return [&out] { FKLB_CUDA_CHECK(cudaMemset(out.ptr().data, 0xFF, 2 * sizeof(double3))); };
        };
        const auto downloadFklOf = [](fk::Ptr1D<double3>& out) {
            return [&out] {
                const std::vector<double> values = fromDevice<double>(out.ptr().data, 6);
                Statistics s;
                for (int c = 0; c < 3; ++c) {
                    s.sum[c] = values[c];
                    s.sumOfSquares[c] = values[3 + c];
                }
                return s;
            };
        };
        rows.push_back({"fkl", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 1, 0, callFkl, fillFklOf(fkOut), downloadFklOf(fkOut)});
        rows.push_back({"fkl_half_pixel", fklb::ResizeSampling::HALF_PIXEL, Rounding::NONE, 1, 0, callFklHalfPixel, fillFklOf(fkOutHalfPixel),
                        downloadFklOf(fkOutHalfPixel)});
        rows.push_back({"fkl_8u", fklb::ResizeSampling::ASYMMETRIC, Rounding::TIES_TO_EVEN, 1, 0, callFkl8, fillFklOf(fkOut8),
                        downloadFklOf(fkOut8)});
        rows.push_back({"fkl_8u_half_pixel", fklb::ResizeSampling::HALF_PIXEL, Rounding::TIES_TO_EVEN, 1, 0, callFkl8HalfPixel,
                        fillFklOf(fkOut8HalfPixel), downloadFklOf(fkOut8HalfPixel)});
        rows.push_back({"fkl_8u_double_sums", fklb::ResizeSampling::ASYMMETRIC, Rounding::TIES_TO_EVEN, 1, 0, callFkl8DoubleSums,
                        fillFklOf(fkOut8DoubleSums), downloadFklOf(fkOut8DoubleSums)});

        // ---- CUB and Thrust: the single pass written by hand, no intermediate image
        // The resized pixels of a call: the crop pointer from the detection, and the factors as fk::Resize::build computes them (the
        // reciprocal, in double, of the resized size over the crop size, then as a float)
        const auto resizedPixels = [&] {
            const float factorX = static_cast<float>(1.0 / (static_cast<double>(OUT_W) / static_cast<double>(CW)));
            const float factorY = static_cast<float>(1.0 / (static_cast<double>(OUT_H) / static_cast<double>(CH)));
            return MomentsOfResizedPixel{flat + (static_cast<size_t>(detection.y) * SRC_W + detection.x) * 3, SRC_W * 3, CW, CH, factorX, factorY};
        };
        // Set once for the crop size: CUB's result in device memory, and its temporary storage (the first call, with no storage, gives
        // its size)
        Moments* cubOut = deviceAlloc<Moments>(1);
        size_t cubStorageBytes = 0;
        FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, cubStorageBytes,
                                                  thrust::make_transform_iterator(thrust::make_counting_iterator(0), resizedPixels()), cubOut,
                                                  OUT_ELEMENTS, MomentsOfTwo{}, MOMENTS_OF_NOTHING, cs));
        void* cubStorage = deviceAlloc<unsigned char>(cubStorageBytes > 0 ? cubStorageBytes : 1);
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callCub = [&] {
            size_t bytes = cubStorageBytes;
            FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(cubStorage, bytes,
                                                      thrust::make_transform_iterator(thrust::make_counting_iterator(0), resizedPixels()), cubOut,
                                                      OUT_ELEMENTS, MomentsOfTwo{}, MOMENTS_OF_NOTHING, cs));
        };
        // (end of the timed call)
        const auto statisticsOfMoments = [](const Moments& m) {
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.sum[c] = m.sum[c];
                s.sumOfSquares[c] = m.sumOfSquares[c];
            }
            return s;
        };
        const auto fillCub = [&] { FKLB_CUDA_CHECK(cudaMemset(cubOut, 0xFF, sizeof(Moments))); };
        const auto downloadCub = [&] { return statisticsOfMoments(fromDevice<Moments>(cubOut, 1)[0]); };
        rows.push_back({"cub", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 1, 0, callCub, fillCub, downloadCub});

        // thrust::transform_reduce returns its result to the host: the row keeps it there
        Moments thrustOut{};
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callThrust = [&] {
            const thrust::counting_iterator<int> first(0);
            thrustOut = thrust::transform_reduce(thrust::cuda::par.on(cs), first, first + OUT_ELEMENTS, resizedPixels(), MOMENTS_OF_NOTHING,
                                                 MomentsOfTwo{});
        };
        // (end of the timed call)
        const auto fillThrust = [&] { std::memset(&thrustOut, 0xFF, sizeof(Moments)); };
        const auto downloadThrust = [&] { return statisticsOfMoments(thrustOut); };
        rows.push_back({"thrust", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 1, 0, callThrust, fillThrust, downloadThrust});

        // thrust::reduce_into leaves its result in device memory. With thrust::cuda::par_nosync the call does not wait for it, and with
        // the allocator the call allocates nothing
        // Set once for the crop size: the result in device memory, and the block of temporary storage. Thrust asks its allocator for the
        // storage of cub::DeviceReduce::Reduce over the range of the cub row: the bytes that the cub row has asked for
        Moments* thrustIntoOut = deviceAlloc<Moments>(1);
        BlockOfTheCaller thrustStorage{deviceAlloc<char>(cubStorageBytes > 0 ? cubStorageBytes : 1), cubStorageBytes};
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callThrustInto = [&] {
            const auto first = thrust::make_transform_iterator(thrust::make_counting_iterator(0), resizedPixels());
            thrust::reduce_into(thrust::cuda::par_nosync(thrustStorage).on(cs), first, first + OUT_ELEMENTS,
                                thrust::device_pointer_cast(thrustIntoOut), MOMENTS_OF_NOTHING, MomentsOfTwo{});
        };
        // (end of the timed call)
        const auto fillThrustInto = [&] { FKLB_CUDA_CHECK(cudaMemset(thrustIntoOut, 0xFF, sizeof(Moments))); };
        const auto downloadThrustInto = [&] { return statisticsOfMoments(fromDevice<Moments>(thrustIntoOut, 1)[0]); };
        rows.push_back({"thrust_reduce_into", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 1, 0, callThrustInto, fillThrustInto,
                        downloadThrustInto});

        // ---- OpenCV-CUDA, the float chain
        // Set once for the crop size: the intermediate images (the float crop, the resized image and its three planes), and the outputs
        // with the size and the type that each call makes (getOutputMat keeps a GpuMat that already has them)
        const cv::Size cvOutSize(OUT_W, OUT_H);
        cv::cuda::GpuMat cvFloat(CH, CW, CV_32FC3), cvResized(cvOutSize, CV_32FC3);
        std::array<cv::cuda::GpuMat, 3> cvPlanes{cv::cuda::GpuMat(cvOutSize, CV_32FC1), cv::cuda::GpuMat(cvOutSize, CV_32FC1),
                                                 cv::cuda::GpuMat(cvOutSize, CV_32FC1)};
        cv::cuda::GpuMat cvSum(1, 1, CV_64FC3), cvSqrSum(1, 1, CV_64FC3);
        std::array<cv::cuda::GpuMat, 3> cvMeanStdDev{cv::cuda::GpuMat(1, 2, CV_64FC1), cv::cuda::GpuMat(1, 2, CV_64FC1),
                                                     cv::cuda::GpuMat(1, 2, CV_64FC1)};
        const size_t cvFloatBytes = cvFloat.step * cvFloat.rows, cvResizedBytes = cvResized.step * cvResized.rows;
        const size_t cvPlanesBytes = 3 * cvPlanes[0].step * cvPlanes[0].rows;
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callOpenCV = [&] {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detection.x, detection.y, CW, CH));
            roi.convertTo(cvFloat, CV_32FC3, 1.0, cvs);
            cv::cuda::resize(cvFloat, cvResized, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
            cv::cuda::calcSum(cvResized, cvSum, cv::noArray(), cvs);
            cv::cuda::calcSqrSum(cvResized, cvSqrSum, cv::noArray(), cvs);
        };
        // (end of the timed call)
        const auto fillOpenCVImages = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(cvFloat.data, cvFloat.step, 0xFF, static_cast<size_t>(CW) * 3 * sizeof(float), CH));
            FKLB_CUDA_CHECK(cudaMemset2D(cvResized.data, cvResized.step, 0xFF, static_cast<size_t>(OUT_W) * 3 * sizeof(float), OUT_H));
        };
        const auto fillOpenCV = [&] {
            fillOpenCVImages();
            FKLB_CUDA_CHECK(cudaMemset(cvSum.data, 0xFF, 3 * sizeof(double)));
            FKLB_CUDA_CHECK(cudaMemset(cvSqrSum.data, 0xFF, 3 * sizeof(double)));
        };
        const auto downloadOpenCV = [&] {
            const std::vector<double> sum = fromDevice<double>(cvSum.data, 3), sqrSum = fromDevice<double>(cvSqrSum.data, 3);
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.sum[c] = sum[c];
                s.sumOfSquares[c] = sqrSum[c];
            }
            return s;
        };
        rows.push_back({"opencv_cuda", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 4, cvFloatBytes + cvResizedBytes, callOpenCV, fillOpenCV,
                        downloadOpenCV});

        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callOpenCVMsd = [&] {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detection.x, detection.y, CW, CH));
            roi.convertTo(cvFloat, CV_32FC3, 1.0, cvs);
            cv::cuda::resize(cvFloat, cvResized, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
            cv::cuda::split(cvResized, cvPlanes.data(), cvs);
            for (int c = 0; c < 3; ++c) cv::cuda::meanStdDev(cvPlanes[c], cvMeanStdDev[c], cvs);
        };
        // (end of the timed call)
        const auto fillOpenCVMsd = [&] {
            fillOpenCVImages();
            for (int c = 0; c < 3; ++c) {
                FKLB_CUDA_CHECK(cudaMemset2D(cvPlanes[c].data, cvPlanes[c].step, 0xFF, static_cast<size_t>(OUT_W) * sizeof(float), OUT_H));
                FKLB_CUDA_CHECK(cudaMemset(cvMeanStdDev[c].data, 0xFF, 2 * sizeof(double)));
            }
        };
        const auto downloadOpenCVMsd = [&] {
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                const std::vector<double> meanStdDev = fromDevice<double>(cvMeanStdDev[c].data, 2);
                s.mean[c] = meanStdDev[0];
                s.stdDev[c] = meanStdDev[1];
            }
            return s;
        };
        rows.push_back({"opencv_cuda_msd", fklb::ResizeSampling::ASYMMETRIC, Rounding::NONE, 6, cvFloatBytes + cvResizedBytes + cvPlanesBytes,
                        callOpenCVMsd, fillOpenCVMsd, downloadOpenCVMsd, true});

        // ---- NPP, the float chain
        // Set once for the crop size: the intermediate images (the float crop and the resized image, cudaMallocPitch), the results in
        // device memory, and the scratch buffer of every call
        Npp32f* nppFloat = nullptr;
        Npp32f* nppResized = nullptr;
        size_t nppFloatStep = 0, nppResizedStep = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFloat), &nppFloatStep, static_cast<size_t>(CW) * 3 * sizeof(Npp32f), CH));
        FKLB_CUDA_CHECK(
            cudaMallocPitch(reinterpret_cast<void**>(&nppResized), &nppResizedStep, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f), OUT_H));
        const NppiSize nppCropSize{CW, CH}, nppOutSize{OUT_W, OUT_H};
        const NppiRect nppCropRect{0, 0, CW, CH}, nppOutRect{0, 0, OUT_W, OUT_H};
        Npp64f* nppSum = deviceAlloc<Npp64f>(3);
        Npp64f* nppDotProd = deviceAlloc<Npp64f>(3);
        Npp64f* nppMean = deviceAlloc<Npp64f>(3);
        Npp64f* nppStdDev = deviceAlloc<Npp64f>(3);
        size_t sumBytes = 0, dotProdBytes = 0, meanStdDevBytes = 0;
        FKLB_NPP_CHECK(nppiSumGetBufferHostSize_32f_C3R_Ctx(nppOutSize, &sumBytes, ctx));
        FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_32f64f_C3R_Ctx(nppOutSize, &dotProdBytes, ctx));
        FKLB_NPP_CHECK(nppiMeanStdDevGetBufferHostSize_32f_C3CR_Ctx(nppOutSize, &meanStdDevBytes, ctx));
        Npp8u* sumBuffer = deviceAlloc<Npp8u>(sumBytes > 0 ? sumBytes : 1);
        Npp8u* dotProdBuffer = deviceAlloc<Npp8u>(dotProdBytes > 0 ? dotProdBytes : 1);
        Npp8u* meanStdDevBuffer = deviceAlloc<Npp8u>(meanStdDevBytes > 0 ? meanStdDevBytes : 1);
        const int frameStep = static_cast<int>(nppFrameStep), floatStep = static_cast<int>(nppFloatStep);
        const int resizedStep = static_cast<int>(nppResizedStep);
        const size_t nppFloatBytes = nppFloatStep * CH, nppResizedBytes = nppResizedStep * OUT_H;
        // The first two calls of both NPP rows: the crop of the frame to float, and the resize of it
        const auto nppConvertAndResize = [&] {
            const Npp8u* nppCrop = nppFrame + static_cast<size_t>(detection.y) * nppFrameStep + static_cast<size_t>(detection.x) * 3;
            FKLB_NPP_CHECK(nppiConvert_8u32f_C3R_Ctx(nppCrop, frameStep, nppFloat, floatStep, nppCropSize, ctx));
            FKLB_NPP_CHECK(nppiResize_32f_C3R_Ctx(nppFloat, floatStep, nppCropSize, nppCropRect, nppResized, resizedStep, nppOutSize, nppOutRect,
                                                  NPPI_INTER_LINEAR, ctx));
        };
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callNpp = [&] {
            nppConvertAndResize();
            FKLB_NPP_CHECK(nppiSum_32f_C3R_Ctx(nppResized, resizedStep, nppOutSize, sumBuffer, nppSum, ctx));
            FKLB_NPP_CHECK(
                nppiDotProd_32f64f_C3R_Ctx(nppResized, resizedStep, nppResized, resizedStep, nppOutSize, nppDotProd, dotProdBuffer, ctx));
        };
        // (end of the timed call)
        const auto fillNppImages = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(nppFloat, nppFloatStep, 0xFF, static_cast<size_t>(CW) * 3 * sizeof(Npp32f), CH));
            FKLB_CUDA_CHECK(cudaMemset2D(nppResized, nppResizedStep, 0xFF, static_cast<size_t>(OUT_W) * 3 * sizeof(Npp32f), OUT_H));
        };
        const auto fillNpp = [&] {
            fillNppImages();
            FKLB_CUDA_CHECK(cudaMemset(nppSum, 0xFF, 3 * sizeof(Npp64f)));
            FKLB_CUDA_CHECK(cudaMemset(nppDotProd, 0xFF, 3 * sizeof(Npp64f)));
        };
        const auto downloadNpp = [&] {
            const std::vector<Npp64f> sum = fromDevice<Npp64f>(nppSum, 3), dotProd = fromDevice<Npp64f>(nppDotProd, 3);
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.sum[c] = sum[c];
                s.sumOfSquares[c] = dotProd[c];
            }
            return s;
        };
        rows.push_back(
            {"npp", fklb::ResizeSampling::HALF_PIXEL, Rounding::NONE, 4, nppFloatBytes + nppResizedBytes, callNpp, fillNpp, downloadNpp});

        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callNppMsd = [&] {
            nppConvertAndResize();
            for (int c = 0; c < 3; ++c) {
                FKLB_NPP_CHECK(nppiMean_StdDev_32f_C3CR_Ctx(nppResized, resizedStep, nppOutSize, c + 1, meanStdDevBuffer, nppMean + c,
                                                            nppStdDev + c, ctx));
            }
        };
        // (end of the timed call)
        const auto fillNppMsd = [&] {
            fillNppImages();
            FKLB_CUDA_CHECK(cudaMemset(nppMean, 0xFF, 3 * sizeof(Npp64f)));
            FKLB_CUDA_CHECK(cudaMemset(nppStdDev, 0xFF, 3 * sizeof(Npp64f)));
        };
        const auto downloadNppMsd = [&] {
            const std::vector<Npp64f> mean = fromDevice<Npp64f>(nppMean, 3), stdDev = fromDevice<Npp64f>(nppStdDev, 3);
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.mean[c] = mean[c];
                s.stdDev[c] = stdDev[c];
            }
            return s;
        };
        rows.push_back({"npp_msd", fklb::ResizeSampling::HALF_PIXEL, Rounding::NONE, 5, nppFloatBytes + nppResizedBytes, callNppMsd, fillNppMsd,
                        downloadNppMsd, true});

        // ---- OpenCV-CUDA, the 8-bit chain: the resize of the 8-bit ROI, and the two reductions of the resized 8-bit image
        // Set once for the crop size: the intermediate image (the resized 8-bit image) and the outputs, as the ones of the float chain
        cv::cuda::GpuMat cvResized8(cvOutSize, CV_8UC3), cvSum8(1, 1, CV_64FC3), cvSqrSum8(1, 1, CV_64FC3);
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callOpenCV8 = [&] {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detection.x, detection.y, CW, CH));
            cv::cuda::resize(roi, cvResized8, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
            cv::cuda::calcSum(cvResized8, cvSum8, cv::noArray(), cvs);
            cv::cuda::calcSqrSum(cvResized8, cvSqrSum8, cv::noArray(), cvs);
        };
        // (end of the timed call)
        const auto fillOpenCV8 = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(cvResized8.data, cvResized8.step, 0xFF, static_cast<size_t>(OUT_W) * 3, OUT_H));
            FKLB_CUDA_CHECK(cudaMemset(cvSum8.data, 0xFF, 3 * sizeof(double)));
            FKLB_CUDA_CHECK(cudaMemset(cvSqrSum8.data, 0xFF, 3 * sizeof(double)));
        };
        const auto downloadOpenCV8 = [&] {
            const std::vector<double> sum = fromDevice<double>(cvSum8.data, 3), sqrSum = fromDevice<double>(cvSqrSum8.data, 3);
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.sum[c] = sum[c];
                s.sumOfSquares[c] = sqrSum[c];
            }
            return s;
        };
        rows.push_back({"opencv_cuda_8u", fklb::ResizeSampling::ASYMMETRIC, Rounding::TIES_TO_EVEN, 3, cvResized8.step * cvResized8.rows, callOpenCV8,
                        fillOpenCV8, downloadOpenCV8});

        // The same resize, the resized 8-bit image to four channels, and the histograms of the four channels in one call:
        // cv::cuda::histEven takes one-channel and four-channel images, and with 256 bins from 0 to 256 every bin is one value. The
        // counts of the fourth channel, the one that cv::cuda::cvtColor adds, are not read back
        // Set once for the crop size: the four-channel image, a further intermediate image, and the 256 counts of each of its channels:
        // GpuMat headers, with the size and the type that histEven makes (GpuMat::create keeps a GpuMat that already has them), of
        // memory of the program at the addresses of countsOfChannel
        cv::cuda::GpuMat cvResized8x4(cvOutSize, CV_8UC4);
        int* cvCountsBlock = deviceAlloc<int>(5 * COUNTS_BLOCK_INTS);
        cv::cuda::GpuMat cvCounts[4];
        for (int c = 0; c < 4; ++c) cvCounts[c] = cv::cuda::GpuMat(1, 256, CV_32SC1, countsOfChannel(cvCountsBlock, c));
        int cvBins[4] = {256, 256, 256, 256}, cvLowerLevel[4] = {0, 0, 0, 0}, cvUpperLevel[4] = {256, 256, 256, 256};
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callOpenCV8Hist = [&] {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detection.x, detection.y, CW, CH));
            cv::cuda::resize(roi, cvResized8, cvOutSize, 0., 0., cv::INTER_LINEAR, cvs);
            cv::cuda::cvtColor(cvResized8, cvResized8x4, cv::COLOR_BGR2BGRA, 0, cvs);
            cv::cuda::histEven(cvResized8x4, cvCounts, cvBins, cvLowerLevel, cvUpperLevel, cvs);
        };
        // (end of the timed call)
        const auto fillOpenCV8Hist = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(cvResized8.data, cvResized8.step, 0xFF, static_cast<size_t>(OUT_W) * 3, OUT_H));
            FKLB_CUDA_CHECK(cudaMemset2D(cvResized8x4.data, cvResized8x4.step, 0xFF, static_cast<size_t>(OUT_W) * 4, OUT_H));
            for (int c = 0; c < 4; ++c) FKLB_CUDA_CHECK(cudaMemset(cvCounts[c].data, 0xFF, 256 * sizeof(int)));
        };
        const auto downloadOpenCV8Hist = [&] {
            // The counts are where the program put them: a histEven that made them again would leave them at other addresses
            for (int c = 0; c < 4; ++c) {
                if (cvCounts[c].data != reinterpret_cast<unsigned char*>(countsOfChannel(cvCountsBlock, c))) {
                    throw std::runtime_error("opencv_cuda_8u_hist: cv::cuda::histEven made again the counts that are set once");
                }
            }
            std::vector<int> counts;
            for (int c = 0; c < 3; ++c) {
                const std::vector<int> ofChannel = fromDevice<int>(cvCounts[c].data, 256);
                counts.insert(counts.end(), ofChannel.begin(), ofChannel.end());
            }
            return statisticsOfCounts(counts);
        };
        rows.push_back({"opencv_cuda_8u_hist", fklb::ResizeSampling::ASYMMETRIC, Rounding::TIES_TO_EVEN, 3,
                        cvResized8.step * cvResized8.rows + cvResized8x4.step * cvResized8x4.rows, callOpenCV8Hist, fillOpenCV8Hist,
                        downloadOpenCV8Hist});

        // ---- NPP, the 8-bit chain: the resize of the 8-bit crop, and the two reductions of the resized 8-bit image
        // Set once for the crop size: the intermediate image (the resized 8-bit image, cudaMallocPitch), the results in device memory, and
        // the scratch buffer of every call
        Npp8u* nppResized8 = nullptr;
        size_t nppResized8Step = 0;
        FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppResized8), &nppResized8Step, static_cast<size_t>(OUT_W) * 3, OUT_H));
        Npp64f* nppSum8 = deviceAlloc<Npp64f>(3);
        Npp64f* nppDotProd8 = deviceAlloc<Npp64f>(3);
        size_t sum8Bytes = 0, dotProd8Bytes = 0;
        FKLB_NPP_CHECK(nppiSumGetBufferHostSize_8u_C3R_Ctx(nppOutSize, &sum8Bytes, ctx));
        FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_8u64f_C3R_Ctx(nppOutSize, &dotProd8Bytes, ctx));
        Npp8u* sum8Buffer = deviceAlloc<Npp8u>(sum8Bytes > 0 ? sum8Bytes : 1);
        Npp8u* dotProd8Buffer = deviceAlloc<Npp8u>(dotProd8Bytes > 0 ? dotProd8Bytes : 1);
        const int resized8Step = static_cast<int>(nppResized8Step);
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callNpp8 = [&] {
            const Npp8u* nppCrop = nppFrame + static_cast<size_t>(detection.y) * nppFrameStep + static_cast<size_t>(detection.x) * 3;
            FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(nppCrop, frameStep, nppCropSize, nppCropRect, nppResized8, resized8Step, nppOutSize, nppOutRect,
                                                 NPPI_INTER_LINEAR, ctx));
            FKLB_NPP_CHECK(nppiSum_8u_C3R_Ctx(nppResized8, resized8Step, nppOutSize, sum8Buffer, nppSum8, ctx));
            FKLB_NPP_CHECK(
                nppiDotProd_8u64f_C3R_Ctx(nppResized8, resized8Step, nppResized8, resized8Step, nppOutSize, nppDotProd8, dotProd8Buffer, ctx));
        };
        // (end of the timed call)
        const auto fillNpp8 = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(nppResized8, nppResized8Step, 0xFF, static_cast<size_t>(OUT_W) * 3, OUT_H));
            FKLB_CUDA_CHECK(cudaMemset(nppSum8, 0xFF, 3 * sizeof(Npp64f)));
            FKLB_CUDA_CHECK(cudaMemset(nppDotProd8, 0xFF, 3 * sizeof(Npp64f)));
        };
        const auto downloadNpp8 = [&] {
            const std::vector<Npp64f> sum = fromDevice<Npp64f>(nppSum8, 3), dotProd = fromDevice<Npp64f>(nppDotProd8, 3);
            Statistics s;
            for (int c = 0; c < 3; ++c) {
                s.sum[c] = sum[c];
                s.sumOfSquares[c] = dotProd[c];
            }
            return s;
        };
        rows.push_back(
            {"npp_8u", fklb::ResizeSampling::HALF_PIXEL, Rounding::TIES_UP, 3, nppResized8Step * OUT_H, callNpp8, fillNpp8, downloadNpp8});

        // The same resize, and the histograms of the three channels of the resized 8-bit image in one call: 257 levels from 0 to 256, which
        // are 256 bins of one value each
        // Set once for the crop size: the 256 counts of each channel in device memory, at the addresses of countsOfChannel, and the
        // scratch buffer of the call
        Npp32s* nppCountsBlock = deviceAlloc<Npp32s>(4 * COUNTS_BLOCK_INTS);
        Npp32s* nppCountsOf[3] = {countsOfChannel(nppCountsBlock, 0), countsOfChannel(nppCountsBlock, 1), countsOfChannel(nppCountsBlock, 2)};
        int histLevels[3] = {257, 257, 257};
        Npp32s histLower[3] = {0, 0, 0}, histUpper[3] = {256, 256, 256};
        size_t histBytes = 0;
        FKLB_NPP_CHECK(nppiHistogramEvenGetBufferSize_8u_C3R_Ctx(nppOutSize, histLevels, &histBytes, ctx));
        Npp8u* histBuffer = deviceAlloc<Npp8u>(histBytes > 0 ? histBytes : 1);
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callNpp8Hist = [&] {
            const Npp8u* nppCrop = nppFrame + static_cast<size_t>(detection.y) * nppFrameStep + static_cast<size_t>(detection.x) * 3;
            FKLB_NPP_CHECK(nppiResize_8u_C3R_Ctx(nppCrop, frameStep, nppCropSize, nppCropRect, nppResized8, resized8Step, nppOutSize, nppOutRect,
                                                 NPPI_INTER_LINEAR, ctx));
            FKLB_NPP_CHECK(nppiHistogramEven_8u_C3R_Ctx(nppResized8, resized8Step, nppOutSize, nppCountsOf, histLevels, histLower, histUpper,
                                                        histBuffer, ctx));
        };
        // (end of the timed call)
        const auto fillNpp8Hist = [&] {
            FKLB_CUDA_CHECK(cudaMemset2D(nppResized8, nppResized8Step, 0xFF, static_cast<size_t>(OUT_W) * 3, OUT_H));
            for (int c = 0; c < 3; ++c) FKLB_CUDA_CHECK(cudaMemset(nppCountsOf[c], 0xFF, 256 * sizeof(Npp32s)));
        };
        const auto downloadNpp8Hist = [&] {
            std::vector<int> counts;
            for (int c = 0; c < 3; ++c) {
                const std::vector<int> ofChannel = fromDevice<int>(nppCountsOf[c], 256);
                counts.insert(counts.end(), ofChannel.begin(), ofChannel.end());
            }
            return statisticsOfCounts(counts);
        };
        rows.push_back({"npp_8u_hist", fklb::ResizeSampling::HALF_PIXEL, Rounding::TIES_UP, 2, nppResized8Step * OUT_H, callNpp8Hist, fillNpp8Hist,
                        downloadNpp8Hist});

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
                nvtxRangePushA((row.version + ":" + cropName).c_str());
                for (int k = 0; k < profileCalls; ++k) {
                    nvtxRangePushA((row.version + ":call").c_str());
                    row.call();
                    nvtxRangePop();
                }
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePop();
            }
        }

        // ---- The checks of the crop size: one call of every selected row, after the buffers that every row writes are filled with 0xFF bytes
        bool cropOk = true;
        if (!profile) {
            const double n = static_cast<double>(OUT_ELEMENTS);
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                for (const Row& any : rows) any.fill();
                FKLB_CUDA_CHECK(cudaDeviceSynchronize());
                row.call();
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                const Statistics s = row.download();
                const bool asymmetric = row.sampling == fklb::ResizeSampling::ASYMMETRIC;
                // The reference of the pixels that the row reduces, in its convention; the one of the same pixels in the other convention,
                // and the one of the float pixels in its convention (its own one, for a row of the float chain)
                const Reference& own = references[asymmetric ? 0 : 1][static_cast<int>(row.rounding)];
                const Reference& other = references[asymmetric ? 1 : 0][static_cast<int>(row.rounding)];
                const Reference& ofFloats = references[asymmetric ? 0 : 1][static_cast<int>(Rounding::NONE)];
                double largest = 0;
                for (int c = 0; c < 3; ++c) {
                    const auto check = [&](const char* statistic, const char* comparedWith, const double value, const double expected,
                                           const double otherExpected, const double floatExpected, const double tolerance,
                                           const char* criterion) {
                        const double difference = relativeDifference(value, expected);
                        const bool ok = difference <= tolerance;
                        checksTable->row(CW, CH, row.version, asymmetric ? "ASYMMETRIC" : "HALF_PIXEL", pixelsOf(row.rounding), statistic, c,
                                         comparedWith, exact(value), exact(expected), difference, criterion, ok ? "yes" : "no", exact(otherExpected),
                                         relativeDifference(value, otherExpected), exact(floatExpected), relativeDifference(value, floatExpected));
                        largest = difference > largest ? difference : largest;
                        if (!ok) {
                            cropOk = false;
                            std::cout << cropName << " " << row.version << " " << statistic << " of channel " << c << ": " << value << " against "
                                      << expected << " (" << comparedWith << "), relative difference " << difference << "; " << criterion
                                      << ": FAILED" << std::endl;
                        }
                    };
                    // The mean and the standard deviation of the channel: as the row left them, or from the sums that it left
                    double mean = s.mean[c], stdDev = s.stdDev[c];
                    if (!row.leavesMeanAndStdDev) {
                        if (!std::isnan(s.count[c])) {
                            // A hist row: its 256 counts add up to the resized pixels (a pixel of value 0 is in no sum)
                            check("count", "the number of resized pixels", s.count[c], n, n, n, 0, "equal");
                        }
                        check("sum", "host", s.sum[c], own.sum[c], other.sum[c], ofFloats.sum[c], 0, "equal");
                        check("sum_of_squares", "host", s.sumOfSquares[c], own.sumOfSquares[c], other.sumOfSquares[c], ofFloats.sumOfSquares[c], 0,
                              "equal");
                        mean = s.sum[c] / n;
                        stdDev = std::sqrt(s.sumOfSquares[c] / n - mean * mean);
                    }
                    const char* from = !row.leavesMeanAndStdDev ? "host; computed on the host from what the row left" : "host";
                    check("mean", from, mean, own.mean[c], other.mean[c], ofFloats.mean[c], RELATIVE_TOLERANCE, "relative difference at most 1e-9");
                    check("std_dev", from, stdDev, own.stdDev[c], other.stdDev[c], ofFloats.stdDev[c], RELATIVE_TOLERANCE,
                          "relative difference at most 1e-9");
                }
                std::cout << cropName << " " << row.version << " (" << (asymmetric ? "ASYMMETRIC" : "HALF_PIXEL") << ", " << pixelsOf(row.rounding)
                          << "): largest relative difference of a statistic " << largest << std::endl;
            }
            checksTable->flush();
            std::cout << cropName << ": " << (cropOk ? "checks passed" : "CHECK FAILED") << " (reference of channel 0: mean "
                      << referenceAsymmetric.mean[0] << " and standard deviation " << referenceAsymmetric.stdDev[0] << " with ASYMMETRIC, "
                      << referenceHalfPixel.mean[0] << " and " << referenceHalfPixel.stdDev[0] << " with HALF_PIXEL)" << std::endl;
            // How far the 8-bit chain is from the float chain: the mean and the standard deviation of the rounded pixels against the ones
            // of the float pixels, for the three pairs of convention and rounding that the rows of the 8-bit chain have
            const auto chainDistance = [&](const int sampling, const Rounding rounding) {
                const Reference& ofBytes = references[sampling][static_cast<int>(rounding)];
                const Reference& ofFloats = references[sampling][static_cast<int>(Rounding::NONE)];
                double mean = 0, stdDev = 0, levels = 0;
                for (int c = 0; c < 3; ++c) {
                    mean = std::max(mean, relativeDifference(ofBytes.mean[c], ofFloats.mean[c]));
                    stdDev = std::max(stdDev, relativeDifference(ofBytes.stdDev[c], ofFloats.stdDev[c]));
                    levels = std::max(levels, std::abs(ofBytes.mean[c] - ofFloats.mean[c]));
                }
                std::ostringstream text;
                text << (sampling == 0 ? "ASYMMETRIC" : "HALF_PIXEL") << ", " << pixelsOf(rounding) << ": mean " << mean << " (" << levels
                     << " grey levels), standard deviation " << stdDev;
                return text.str();
            };
            std::cout << cropName << ": the pixels rounded to 8 bits against the float pixels, largest relative difference of the three channels: "
                      << chainDistance(0, Rounding::TIES_TO_EVEN) << "; " << chainDistance(1, Rounding::TIES_TO_EVEN) << "; "
                      << chainDistance(1, Rounding::TIES_UP) << ". Ties: " << referenceAsymmetric.ties << " (ASYMMETRIC) and "
                      << referenceHalfPixel.ties << " (HALF_PIXEL) of the " << 3 * OUT_ELEMENTS << " resized values" << std::endl;
            allOk = allOk && cropOk;
        }

        // ---- The timing of the crop size, after its checks, in the order of the rows
        if (timing && cropOk) {
            if (!times) {
                times = std::make_unique<fklb::CsvWriter>(
                    "e20_fused_reductions.csv",
                    "E20 fused reductions: whole-call and kernel-only times of the mean and the standard deviation of a crop resized to 224 x 224",
                    std::initializer_list<const char*>{"version", "sampling", "pixels", "crop_width", "crop_height", "library_calls",
                                                       "intermediate_image_bytes", "interval", "mean_ms", "min_ms", "iterations", "spin_us"});
            }
            const auto nothingToPrepare = [] { return 0; };
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                const char* sampling = row.sampling == fklb::ResizeSampling::ASYMMETRIC ? "ASYMMETRIC" : "HALF_PIXEL";
                const char* pixels = pixelsOf(row.rounding);
                fklb::Timing whole, gpu, host;
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePushA(("whole_call:" + row.version + ":" + cropName).c_str());
                fklb::timeIt(cs, row.call, whole);
                nvtxRangePop();
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                nvtxRangePushA(("kernel_only:" + row.version + ":" + cropName).c_str());
                fklb::timePreparedLaunch(cs, nothingToPrepare, spin, [&](int) { row.call(); }, gpu, host);
                nvtxRangePop();
                times->row(row.version, sampling, pixels, CW, CH, row.calls, row.intermediateBytes, "whole_call", whole.mean, whole.min,
                           fklb::ITERATIONS, "");
                times->row(row.version, sampling, pixels, CW, CH, row.calls, row.intermediateBytes, "kernel_only", gpu.mean, gpu.min,
                           fklb::ITERATIONS, spinUs());
                times->flush();
                std::cout << cropName << " " << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min
                          << "); kernel only mean " << gpu.mean << " ms (min " << gpu.min << ")" << std::endl;
            }
        } else if (timing) {
            std::cout << cropName << ": not timed: a check failed" << std::endl;
        }

        FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
        for (void* p : std::vector<void*>{cubOut, cubStorage, thrustIntoOut, thrustStorage.block, nppFloat, nppResized, nppSum, nppDotProd, nppMean,
                                          nppStdDev, sumBuffer, dotProdBuffer, meanStdDevBuffer, nppResized8, nppSum8, nppDotProd8, sum8Buffer,
                                          dotProd8Buffer, nppCountsBlock, histBuffer, cvCountsBlock}) {
            FKLB_CUDA_CHECK(cudaFree(p));
        }
    }
    if (profile) {
        std::cout << "profile mode: " << profileCalls << " calls per selected row and crop size, each in its NVTX range" << std::endl;
    } else {
        std::cout << (allOk ? "E20 fused reductions checks passed" : "E20 fused reductions CHECK FAILED") << std::endl;
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    FKLB_CUDA_CHECK(cudaFree(flat));
    FKLB_CUDA_CHECK(cudaFree(nppFrame));
    // The wrapper of the stream gives its part of the pool back, and synchronises the stream to do it: before the stream is destroyed
    cvs = cv::cuda::Stream::Null();
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
