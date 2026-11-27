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

// E20 reductions (NVIDIA): several reductions of the same matrix. For one one-channel matrix: its minimum, its maximum, the sum of its
// elements, the number of elements that are not zero and the sum of their squares, left in device memory (the thrust row: in host
// memory; the msd rows: the mean and the standard deviation in place of the two sums); the variance is then computed on the host from
// the sum, the sum of the squares and the number of elements (in the msd rows, from the standard deviation). The matrix is of bytes (8u)
// or of floats (32f): two types, each with its rows, its checks and its times.
// FKL's ReduceDPP computes the five in one kernel that reads every element once. OpenCV-CUDA and NPP have one call per reduction (the
// minimum and the maximum share one), and every call reads the whole matrix at least once (OpenCV-CUDA: 4 calls, 4 passes; NPP: 4
// calls, 5 passes, because nppiMinMax launches two kernels over the image, as nppiMean_StdDev does). Thrust and CUB are the single pass
// written by hand: one transform-reduce over a struct with the five values, as Thrust's summary_statistics example does for its
// statistics.
// The accumulators. Between the fkl row of the 8u matrix and its OpenCV-CUDA and NPP rows two things differ, not one: the number of
// passes, and the type in which the sum and the sum of squares are accumulated. fkl, cub and the two thrust rows accumulate them in
// 64-bit integers. OpenCV-CUDA accumulates both in double (cudaarithm/src/cuda/sum.cu: sumImpl<uchar, double, 1> and
// sumSqrImpl<uchar, double, 1>). NPP accumulates the squares in double and the bytes in int, by the names of its kernels (read with
// nsys): of the kernels that read the image, those of the squares (in nppiDotProd_8u64f and in nppiMean_StdDev_8u) are
// ImageReductionKernel<Pixel<double, 1>, ...>, and those of the sum of the bytes (in nppiSum_8u and in nppiMean_StdDev_8u)
// ImageReductionKernel<Pixel<int, 1>, ...>. The fkl_double_sums row, the kernel of fkl with the two sums in double, tells the two
// differences apart. In the rows of the 32f matrix FKL, CUB, Thrust and OpenCV-CUDA accumulate the two sums in double: there the fkl
// row accumulates the two sums as OpenCV-CUDA does. NPP accumulates the squares of the floats in double and their sum in float: its
// kernels of the sum that read the image (in nppiSum_32f and in nppiMean_StdDev_32f) are ImageReductionKernel<Pixel<float, 1>, ...>,
// and the sum that it leaves is equal to the host's up to the width at which one row of the matrix adds up to 2^21, where the multiples
// of 1/8 that a float holds end, and at no width tried beyond (below: the 32f matrix, and the sizes that fail). So in the NPP rows of
// the 32f matrix the sum is not accumulated as in its other rows.
//
// The 8u matrix: width x height bytes, row by row, from the LCG of E4, E18 and E19 (seed 12345, state = state * 1664525 + 1013904223), one
// element per state from its bits 31..24 (b): 0 when b < 8, and 5 + b mod 246 otherwise. So about 3% of the elements are zero, the minimum
// is 0 and the maximum is at most 250: no statistic is the value that a buffer filled with 0xFF bytes holds (below).
// The 32f matrix (CV_32FC1): every byte of the 8u matrix divided by 8, in float. Its zeros are at the same positions, and every element
// is a multiple of 1/8 of at most 31.875 (255 / 8). So any sum of elements, in any order, is a multiple of 1/8, and any sum of squares
// a multiple of 1/64; a double holds every multiple of 1/64 below 2^47, which is more than the sum of the squares of 2^31 such
// elements. A row that accumulates in double rounds nothing, whatever its order, and the checks of the two sums are equalities, as
// for the bytes. The host verifies it: it refuses an element that is not such a multiple, and computes the reference of the matrix in
// integers (of eighths and of sixty-fourths). (The square of one element, at most 65,025 / 64, is exact in a float as well: the
// checks do not tell a row that squares in float from one that squares in double.) One sum is not accumulated in double: NPP's sum
// of the floats, which nppiSum_32f_C1R_Ctx leaves in the npp row and from which nppiMean_StdDev_32f_C1R_Ctx computes the mean of the
// npp_msd and opencv_cuda_msd rows (above). A float holds every multiple of 1/8 up to 2^21 only, and that sum was the host's for every
// matrix tried in which every row adds up to less (those of the default sizes, of 65536x16, 127500x1000, 128500x5 and 8192x131072)
// and for none in which a row adds up to 2^21 or more (the probes of 128000x1000, where one row does, of 129000x5 and of 131072x8;
// both matrices of 4194304x1): NPP holds the sum of a row in a float. A row of up to 65,793 such elements adds up to less than 2^21
// whatever they are (65,793 x 31.875), so the checks of the default sizes are equalities for NPP's sum too.
// Each library reads its own copy of the same host elements, and owns its outputs. The types are E20_TYPES, a list of 8u and 32f (default
// 8u,32f), and the sizes E20_SIZES, a list of <width>x<height> (default 256x256,640x480,1920x1080,3840x2160): for every type, every
// size is checked and then timed before the next one. A size whose check fails is not timed, and the others are: the timing table of
// a run that ends with CHECK FAILED (exit code 1) holds the types and sizes that passed. Sizes that fail (NPP and CUB are the ones of
// CUDA 13.4.3):
//   8u, wider than 8,192 elements     nppiCountInRange_8u_C1R_Ctx loses multiples of 256 of its count, in npp and npp_msd: 8192x8192
//                                     passes, 8193x1 and 16384x16 do not (nppiCountInRange_32f_C1R_Ctx counts right at these sizes).
//   32f, wider than about 128,000     a row of the probe adds up to 2^21 or more, and NPP's sum of the floats is rounded (above):
//                                     127500x1000 and 128500x5 pass, 128000x1000 (where one row does) and 129000x5 do not. At 129000x5
//                                     the sum of npp and the mean of npp_msd and opencv_cuda_msd are 2e-8 of their value away from the
//                                     host's, and at 8388608x1 5e-5.
//   8u and 32f, above 524,280 rows    (8 x 65,535) NPP's reductions return NPP_CUDA_KERNEL_EXECUTION_ERROR, and the program stops in
//                                     opencv_cuda_msd, npp or npp_msd: 1x524280 passes, 1x524281 does not.
//   8u, of 2^31 elements or more      the counts of OpenCV-CUDA and NPP are ints, and at 8192x262144 the mean of
//                                     nppiMean_StdDev_8u_C1R_Ctx changes sign; 8192x262143 passes in every row.
//   32f, larger than the memory       the four device copies of a 32f matrix take 16 bytes per element: 8192x131072 passes in every
//                                     row, and at 8192x262143 the program stops, out of the memory of an RTX 4090 (24 GB).
//
// The rows, each one a call that leaves its statistics in device memory. Nothing is brought to the host
// inside a call, except in the thrust row, and the host does not wait for the GPU inside a call, except in the thrust row (and in the
// opencv_cuda_msd row of a matrix of more than a hundred thousand rows: its description says from which height). T is the type of the
// elements, uchar or float:
//   fkl              one fk::executeOperations<fk::ReduceDPP<>> (FKL's reduce_dpp.h), one kernel. The read is fk::PerThreadRead<_2D, T>
//                    of FKL's copy; the reductions are five Tuple<FirstIOp, AccumulationIOp, FinalIOp>:
//                      minimum         Cast<T, T> (nothing),                 Min<T>,     Cast<T, double>
//                      maximum         Cast<T, T>,                           Max<T>,     Cast<T, double>
//                      sum             Cast<T, A>,                           Add<A>,     Cast<A, double>
//                      non zero        IsNonZero<T> (1u or 0u; below),       Add<uint>,  Cast<uint, double>
//                      sum of squares  Square<T, A> (computed in A; below),  Add<A>,     Cast<A, double>
//                    A is the accumulator of the two sums: ulonglong for the 8u matrix, the type a user would choose (no accumulation
//                    is rounded, whatever the bytes), and double for the 32f matrix, the accumulator of OpenCV-CUDA's sums. IsNonZero
//                    and Square are operations of this program (a user's operations): FKL's basic operations have neither. The write
//                    is fk::PerThreadWrite<_1D, double> of five doubles. The thread blocks of the kernel leave their values in a
//                    workspace, device memory that the caller owns: allocated and zeroed once for each size, as NPP's scratch
//                    buffers. Every call builds the read, the reductions, the write and the details, and launches 1 kernel.
//   fkl_double_sums  8u only: the call of fkl with A = double, so with the two sums accumulated as the opencv_cuda row accumulates
//                    them (its accumulators: uchar, uchar, double, uint, double), with its own five doubles and its own workspace. It
//                    separates the two differences between fkl and the OpenCV-CUDA and NPP rows: the passes, and the accumulators
//                    (measured on an RTX 4090 with CUDA 13.4.3, at 3840 x 2160: about 13 us for fkl, 48 us for this row and 93 us
//                    for opencv_cuda). The integer accumulators of the fkl row are exact and are FKL's to choose: OpenCV-CUDA
//                    writes both sums as CV_64F, and in NPP only the sum has a call with an integer result (nppiSum_8u64s_C1R_Ctx).
//                    The 32f matrix has no such row: its fkl row already accumulates the two sums in double.
//   cub              cub::DeviceReduce::Reduce over a transform iterator (thrust::make_transform_iterator) on the elements of a
//                    contiguous copy (no row padding), with a struct of the five values that has the accumulators of the fkl row (8u:
//                    uchar, uchar, uint, ulonglong, ulonglong, 24 bytes; 32f: float, float, uint, double, double, 32 bytes), the
//                    result in device memory and the temporary storage allocated once for each size (the two-call idiom of CUB). Not
//                    cub::DeviceReduce::TransformReduce on the pointer: with an accumulator of 17 to 32 bytes the CCCL of CUDA 13.3
//                    and of CUDA 13.4 (CUB 3.3, 3.4) reads the bytes of a raw pointer four at a time from addresses that are two or
//                    three bytes apart (cub/agent/agent_reduce.cuh), and its kernel stops with "misaligned address". An input that is
//                    not a raw pointer (the transform iterator of this row, a thrust::device_ptr) has no such load, and takes the
//                    same time. On the floats of the 32f matrix TransformReduce on the pointer runs, in the time of this row.
//   thrust           thrust::transform_reduce on the same copy and with the same operations, the call of Thrust's summary_statistics
//                    example, with thrust::cuda::par on the stream. It returns the struct, so it is NOT like the other rows: the call
//                    waits for the result and copies it to the host, and Thrust allocates and releases its temporary storage in every
//                    call (per call: 1 cudaMalloc, 1 cudaFree, 1 cudaMemcpyAsync, 2 cudaStreamSynchronize). The call stores the
//                    struct in host memory.
//   thrust_reduce_into  thrust::reduce_into (CCCL 3.2, CUDA 13.2) over a transform iterator on the same copy: it writes the result
//                    through an output iterator, here to device memory. With thrust::cuda::par_nosync on the stream the call does not
//                    wait, and with the caller's block as the allocator of the policy (set once for each size, as the storage of the
//                    cub row) it allocates nothing: like the other rows, and with the two kernels of CUB's reduction, as the cub row.
//                    (par_nosync without an allocator takes the storage with cudaMallocAsync and cudaFreeAsync on the stream in every
//                    call.)
//   opencv_cuda      cv::cuda::findMinMax (minimum and maximum: CV_32S for the 8u matrix, CV_32F for the 32f one), cv::cuda::calcSum
//                    (CV_64F), cv::cuda::countNonZero (CV_32S) and cv::cuda::calcSqrSum (CV_64F): the overloads that take a Stream
//                    and write a GpuMat, which is set once with the size and type that each call makes (getOutputMat keeps it; the
//                    read back of the checks refuses an output that a call had to make again). 4 calls, 4 passes over the matrix.
//   opencv_cuda_msd  cv::cuda::findMinMax, cv::cuda::countNonZero and cv::cuda::meanStdDev (the mean and the standard deviation, CV_64F,
//                    which OpenCV computes with NPP's nppiMean_StdDev of the type of the matrix): with the reductions of
//                    OpenCV-CUDA, the fewest calls for the minimum, the maximum, the count and the variance. 3 calls.
//                    cv::cuda::meanStdDev takes NPP's scratch buffer from the BufferPool of the stream, which this program turns on
//                    before it wraps its stream (cv::cuda::setBufferPoolUsage(true): set once, as NPP's scratch buffers and FKL's
//                    workspace are). With OpenCV's default, the pool off, every call allocates the buffer (cudaMalloc) and frees it
//                    when it returns, and cudaFree waits for the work of the stream: the row would wait for the GPU inside every call.
//                    It does in a matrix so tall that the buffer (24 bytes per row of the matrix for NPP, which OpenCV takes as that
//                    many elements of the matrix) is larger than the stack of the pool, 10 MB by default: above 436,906 rows of bytes
//                    or 109,226 rows of floats (the pool serves 64x436906 of bytes and 64x109226 of floats, and not one row more).
//   npp              the calls of the type of the matrix: nppiMinMax_8u_C1R_Ctx or nppiMinMax_32f_C1R_Ctx; nppiSum_8u_C1R_Ctx or
//                    nppiSum_32f_C1R_Ctx (Npp64f); nppiCountInRange_8u_C1R_Ctx with the bounds 1 and 255, or
//                    nppiCountInRange_32f_C1R_Ctx with the smallest float above zero and the largest float (1.4e-45 and 3.4e38: the
//                    matrix has no negative element, so the range holds its elements that are not zero); and
//                    nppiDotProd_8u64f_C1R_Ctx or nppiDotProd_32f64f_C1R_Ctx of the matrix with itself (Npp64f: the sum of the
//                    squares; NPP's other call for it, nppiNorm_L2, gives its square root). Results in device memory, and one scratch
//                    buffer per call, allocated once for each size (nppi...GetBufferHostSize). 4 calls.
//   npp_msd          nppiMinMax and nppiCountInRange as in npp, and nppiMean_StdDev_8u_C1R_Ctx or nppiMean_StdDev_32f_C1R_Ctx: with
//                    the reductions of NPP, the fewest calls for the minimum, the maximum, the count and the variance. 3 calls.
// The OpenCV-CUDA, NPP, Thrust and CUB rows use no FKL object.
//
// The checks (e20_reductions_checks.csv), for every type and size, on two matrices: the probe and then the matrix that is timed. The
// matrix has zeros (the count needs them), so its minimum is 0, the value of an accumulator that starts at zero, and its extremes (0 and
// 250 in bytes) are those of any large part of it (its first two elements are 0). The probe is the matrix with every byte v replaced by
// 253 - v (3 to 253, no zero), a 255 in its first element and a 1 in its last one (for 32f, those bytes divided by 8): its maximum and
// its minimum are single elements at its two ends, so they are right only when the first and the last element are reduced, a minimum
// of 0 is never right, and its maximum is the largest byte, the upper bound of NPP's count. Every library's copy is uploaded again for
// each. The host computes the five statistics of the elements in integers, exactly,
// and from them the mean, the standard deviation and the variance (the population variance, (N * sum of squares - sum^2) / N^2). Each
// row is called once, after the buffers that hold the statistics of EVERY row are filled with 0xFF bytes (a NaN in a double or a float,
// -1 in an int and the largest value in an unsigned integer: never a right value, but for the 255 of a byte, which is the maximum of the
// probe: there the rows that leave a byte are caught by the matrix that is timed, where no statistic is 255. Filling them all shows a
// row that reads what another one left), and its statistics are read back:
//   minimum, maximum, non_zero    equal to the host's. The timing needs it.
//   sum, sum_of_squares           the rows that leave them (fkl, fkl_double_sums, cub, thrust, thrust_reduce_into, opencv_cuda, npp):
//                                 equal to the host's. The host's are exact in a double: integers below 2^53 for 8u, and for 32f as
//                                 said of its matrix. The timing needs it.
//   mean, std_dev                 the msd rows: against the host's, relative difference at most 1e-9. The timing needs it.
//   variance                      every row, computed on the host from what the row left: sum of squares / N - (sum / N)^2 in double from
//                                 the rows that have the sums, and the square of the standard deviation from the msd rows; against the
//                                 host's, relative difference at most 1e-9. The timing needs it.
//
// The timing (e20_reductions.csv), for every type and size after its checks: two intervals per row, in the order fkl, fkl_double_sums,
// cub, thrust, thrust_reduce_into, opencv_cuda, opencv_cuda_msd, npp, npp_msd (the 32f matrix has no fkl_double_sums), the same in
// every run (the first row timed after the checks can start at a lower clock, so that row is FKL's, not a baseline's):
//   whole_call   fklb::timeIt (common/cuda_timing.h): CUDA events around one call, 100 calls after one warm-up call (the interval of E4,
//                E18 and E19). When the GPU waits for the host, the interval holds the host's work too.
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin
//                of E20_SPIN_US (default 5,000 us), so the interval runs from the end of the spin to the end of the call's last kernel,
//                if the host has enqueued the whole call before the spin ends. The thrust row waits for its result inside the call, so
//                its interval also holds what its host does after the spin.
//   Each interval of each row is inside an NVTX range ("whole_call:<row>:<type>:<size>", "kernel_only:<row>:<type>:<size>").
// Both tables have a column with the type, 8u or 32f.
// E20_ONLY=a,b: only these rows (a name that is no row of a type of the run stops the program; a type that has none of the named rows
// is not run, and the program says so). E20_NO_TIMING=1: the checks only. E20_PROFILE=1: every selected row is called once outside any
// range (a warm-up), then E20_PROFILE_CALLS times (default 10), each call in an NVTX range "<row>:call" inside the range
// "<row>:<type>:<size>"; nothing is compared or timed (to count the kernels of one call with nsys).
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/cuda_timing.h>
#include <common/env_args.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/cast.h>
#include <fused_kernel/algorithms/basic_ops/logical.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cub/device/device_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/transform_reduce.h>

#include <npp.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
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
#include <type_traits>
#include <vector>

// ---- The user's operations of the FKL rows, which FKL's basic operations do not have
namespace e20 {
// 1 for an element that is not zero: the FirstIOp of the count
template <typename T>
struct IsNonZero {
    using Parent = fk::UnaryOperation<T, uint, IsNonZero<T>>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) { return input != 0 ? 1u : 0u; }
};

// The square of an element, computed in the accumulator type A of the sum of squares: the FirstIOp of that sum
template <typename T, typename A>
struct Square {
    using Parent = fk::UnaryOperation<T, A, Square<T, A>>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) { return static_cast<A>(input) * static_cast<A>(input); }
};
} // namespace e20

namespace {
constexpr double RELATIVE_TOLERANCE = 1e-9;   // the mean, the standard deviation and the variance, against the host's in double

// The element type T of a matrix: unsigned char (8u) or float (32f)
template <typename T>
constexpr bool IS_32F = std::is_same_v<T, float>;

// Every element is a multiple of 1 / UNITS: an integer in the 8u matrix, and an integer divided by 8 in the 32f matrix
template <typename T>
constexpr int UNITS = IS_32F<T> ? 8 : 1;

// The accumulator of the sum and of the sum of squares in the fkl, cub and thrust rows: ulonglong for the bytes, double for the floats
template <typename T>
using SumOf = std::conditional_t<IS_32F<T>, double, unsigned long long>;

// ---- The five reductions of an FKL row over a matrix of T, each one a Tuple<FirstIOp, AccumulationIOp, FinalIOp>: the minimum and the
// maximum accumulated in T, the count in uint, and the sum and the sum of squares in A
template <typename T, typename A>
auto fklReductions() {
    using Same = fk::Cast<T, T>;
    using Min = fk::Min<T, T, T, fk::UnaryType>;
    using Max = fk::Max<T, T, T, fk::UnaryType>;
    using AddCount = fk::Add<uint, uint, uint, fk::UnaryType>;
    using AddSum = fk::Add<A, A, A, fk::UnaryType>;
    return fk::make_tuple(fk::make_tuple(Same::build(), Min::build(), fk::Cast<T, double>::build()),
                          fk::make_tuple(Same::build(), Max::build(), fk::Cast<T, double>::build()),
                          fk::make_tuple(fk::Cast<T, A>::build(), AddSum::build(), fk::Cast<A, double>::build()),
                          fk::make_tuple(e20::IsNonZero<T>::build(), AddCount::build(), fk::Cast<uint, double>::build()),
                          fk::make_tuple(e20::Square<T, A>::build(), AddSum::build(), fk::Cast<A, double>::build()));
}

// ---- The single pass of the cub and thrust rows: the five values, with the accumulator types of the fkl row of the same matrix
template <typename T>
struct Five {
    T minimum, maximum;
    unsigned int nonZero;
    SumOf<T> sum, sumOfSquares;
};

template <typename T>
struct FiveOfElement {
    __host__ __device__ Five<T> operator()(const T v) const {
        return Five<T>{v, v, v != 0 ? 1u : 0u, static_cast<SumOf<T>>(v), static_cast<SumOf<T>>(v) * static_cast<SumOf<T>>(v)};
    }
};

template <typename T>
struct FiveOfTwo {
    __host__ __device__ Five<T> operator()(const Five<T>& a, const Five<T>& b) const {
        return Five<T>{a.minimum < b.minimum ? a.minimum : b.minimum, a.maximum > b.maximum ? a.maximum : b.maximum, a.nonZero + b.nonZero,
                       a.sum + b.sum, a.sumOfSquares + b.sumOfSquares};
    }
};

// The value that changes no other: the initial value of the reductions of cub and thrust
template <typename T>
constexpr Five<T> FIVE_OF_NOTHING{std::numeric_limits<T>::max(), std::numeric_limits<T>::lowest(), 0u, 0, 0};

// The temporary storage of the thrust_reduce_into row: one block of device memory that the caller owns, as the storage of the cub row,
// given to Thrust as the allocator of its execution policy (the idiom of Thrust's custom_temporary_allocation example, with one block)
struct BlockOfTheCaller {
    using value_type = char;
    char* block = nullptr;
    size_t bytes = 0;
    char* allocate(const std::ptrdiff_t count) {
        if (static_cast<size_t>(count) > bytes) {
            throw std::runtime_error("thrust_reduce_into: Thrust asks for more temporary storage than the block has");
        }
        return block;
    }
    void deallocate(char*, size_t) {}
};

// The bounds of nppiCountInRange_32f_C1R_Ctx that hold the elements that are not zero of a matrix without negative elements: the
// smallest float above zero and the largest float (for bytes they are 1 and 255)
constexpr Npp32f NON_ZERO_FROM = std::numeric_limits<Npp32f>::denorm_min();
constexpr Npp32f NON_ZERO_TO = std::numeric_limits<Npp32f>::max();

// The rows of each matrix type, in the order of the checks and of the timing
const std::vector<std::string> ROWS_8U = {"fkl", "fkl_double_sums", "cub", "thrust", "thrust_reduce_into", "opencv_cuda", "opencv_cuda_msd",
                                          "npp", "npp_msd"};
const std::vector<std::string> ROWS_32F = {"fkl", "cub", "thrust", "thrust_reduce_into", "opencv_cuda", "opencv_cuda_msd", "npp", "npp_msd"};

const std::vector<std::string>& rowsOfType(const std::string& type) { return type == "32f" ? ROWS_32F : ROWS_8U; }

bool selected(const std::string& name) {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// E20_TYPES: 8u, 32f or both, in the order of the run
std::vector<std::string> typesOfTheRun() {
    const char* e = std::getenv("E20_TYPES");
    std::stringstream list(e != nullptr ? e : "8u,32f");
    std::vector<std::string> types;
    std::string name;
    while (std::getline(list, name, ',')) {
        if (name != "8u" && name != "32f") throw std::runtime_error("E20_TYPES: '" + name + "' is not 8u or 32f");
        if (std::find(types.begin(), types.end(), name) != types.end()) throw std::runtime_error("E20_TYPES: '" + name + "' is there twice");
        types.push_back(name);
    }
    if (types.empty()) throw std::runtime_error("E20_TYPES: no type");
    return types;
}

// E20_ONLY names rows: a name that is no row of the types of the run would leave empty tables behind a "checks passed"
void refuseUnknownRows(const std::vector<std::string>& types) {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return;
    std::stringstream list(e);
    std::string name;
    int names = 0;
    while (std::getline(list, name, ',')) {
        const auto hasTheRow = [&](const std::string& type) {
            return std::find(rowsOfType(type).begin(), rowsOfType(type).end(), name) != rowsOfType(type).end();
        };
        if (std::none_of(types.begin(), types.end(), hasTheRow)) {
            throw std::runtime_error("E20_ONLY: '" + name + "' is not a row of this program for the matrix types of the run (E20_TYPES)");
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

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

struct Size {
    int width, height;
};

// E20_SIZES: <width>x<height>,<width>x<height>...
std::vector<Size> sizesOfTheRun() {
    const char* e = std::getenv("E20_SIZES");
    std::stringstream list(e != nullptr ? e : "256x256,640x480,1920x1080,3840x2160");
    std::vector<Size> sizes;
    std::string item;
    while (std::getline(list, item, ',')) {
        const size_t x = item.find('x');
        if (x == std::string::npos) throw std::runtime_error("E20_SIZES: '" + item + "' is not <width>x<height>");
        const Size size{positiveInt(item.substr(0, x)), positiveInt(item.substr(x + 1))};
        if (size.width <= 0 || size.height <= 0) throw std::runtime_error("E20_SIZES: '" + item + "' is not <width>x<height>");
        sizes.push_back(size);
    }
    if (sizes.empty()) throw std::runtime_error("E20_SIZES: no size");
    return sizes;
}

// The bytes of the matrix: one per state of the LCG of E4, E18 and E19 (seed 12345), from its bits 31..24 (b): 0 when b < 8, and
// 5 + b mod 246 otherwise
std::vector<unsigned char> seededBytes(const int width, const int height) {
    std::vector<unsigned char> bytes(static_cast<size_t>(width) * height);
    uint32_t state = 12345u;
    for (unsigned char& v : bytes) {
        state = state * 1664525u + 1013904223u;
        const unsigned int b = state >> 24;
        v = static_cast<unsigned char>(b < 8 ? 0 : 5 + b % 246);
    }
    return bytes;
}

// The bytes of the probe of the checks: every byte v of the matrix replaced by 253 - v (3 to 253: no zero), a 255 in the first one and a
// 1 in the last one. Its maximum and its minimum are single elements at its two ends, its minimum is not the 0 of an empty accumulator,
// and its maximum is the largest byte
std::vector<unsigned char> probeOf(const std::vector<unsigned char>& bytes) {
    std::vector<unsigned char> probe(bytes.size());
    for (size_t k = 0; k < bytes.size(); ++k) probe[k] = static_cast<unsigned char>(253 - bytes[k]);
    probe.front() = 255;
    probe.back() = 1;
    return probe;
}

// A matrix of T from its bytes: the bytes themselves (8u), or every byte divided by 8, which is exact in a float (32f)
template <typename T>
std::vector<T> matrixOf(const std::vector<unsigned char>& bytes) {
    std::vector<T> matrix(bytes.size());
    for (size_t k = 0; k < bytes.size(); ++k) matrix[k] = static_cast<T>(static_cast<T>(bytes[k]) / UNITS<T>);
    return matrix;
}

// A value of a check as the tables hold it: with 17 significant digits, which tell any two doubles apart (fklb::CsvWriter writes 9), so
// that two values that are compared for equality are the same text in the file only when they are equal
std::string exact(const double value) {
    std::ostringstream s;
    s << std::setprecision(17) << value;
    return s.str();
}

// What a row leaves (NaN where it computes no such value)
struct Statistics {
    double minimum = std::nan(""), maximum = std::nan(""), nonZero = std::nan("");
    double sum = std::nan(""), sumOfSquares = std::nan("");
    double mean = std::nan(""), stdDev = std::nan("");
};

// What the host computes from the elements
struct Reference {
    unsigned long long elements = 0;
    double minimum = 0, maximum = 0, nonZero = 0, sum = 0, sumOfSquares = 0;
    double mean = 0, stdDev = 0, variance = 0;
};

// The reference of a matrix, in integers: every element is k / UNITS with k an integer from 0 to 255, which the host verifies here for
// each one. So the sum of the elements is an integer number of 1 / UNITS and the sum of their squares an integer number of 1 / UNITS^2,
// both below 2^53 (verified too): the doubles of the reference are exact, and so is every partial sum of a row that accumulates in double
template <typename T>
Reference referenceOf(const std::vector<T>& matrix) {
    const unsigned long long units = UNITS<T>;
    unsigned long long minimum = 255, maximum = 0, nonZero = 0, sum = 0, sumOfSquares = 0;
    for (const T v : matrix) {
        const double scaled = static_cast<double>(v) * static_cast<double>(units);
        if (!(scaled >= 0 && scaled <= 255) || scaled != std::floor(scaled)) {
            throw std::runtime_error("referenceOf: an element is not an integer from 0 to 255 divided by " + std::to_string(units));
        }
        const unsigned long long k = static_cast<unsigned long long>(scaled);
        minimum = k < minimum ? k : minimum;
        maximum = k > maximum ? k : maximum;
        nonZero += k != 0 ? 1 : 0;
        sum += k;
        sumOfSquares += k * k;
    }
    if (sumOfSquares >= (1ull << 53)) throw std::runtime_error("referenceOf: the sum of squares is not exact in a double");
    Reference r;
    r.elements = matrix.size();
    const double scale = static_cast<double>(units);
    r.minimum = static_cast<double>(minimum) / scale;
    r.maximum = static_cast<double>(maximum) / scale;
    r.nonZero = static_cast<double>(nonZero);
    r.sum = static_cast<double>(sum) / scale;
    r.sumOfSquares = static_cast<double>(sumOfSquares) / (scale * scale);
    // The population variance, (N * sum of squares - sum^2) / N^2: the numerator in integers (below 2^128), then one division; the mean
    // and the variance of the k, divided by UNITS and by UNITS^2 (powers of two)
    const unsigned __int128 n = r.elements;
    const unsigned __int128 numerator = n * sumOfSquares - static_cast<unsigned __int128>(sum) * sum;
    const long double varianceOfTheIntegers = static_cast<long double>(numerator) / (static_cast<long double>(n) * static_cast<long double>(n));
    r.variance = static_cast<double>(varianceOfTheIntegers) / (scale * scale);
    r.mean = static_cast<double>(static_cast<long double>(sum) / static_cast<long double>(n)) / scale;
    r.stdDev = std::sqrt(r.variance);
    return r;
}

double relativeDifference(const double value, const double reference) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    return reference == 0 ? std::abs(value) : std::abs(value - reference) / std::abs(reference);
}

// The GPU spin of the kernel-only interval, the one of E1, E18 and E19: one thread waits on the GPU's global timer and touches no
// memory. Enqueued before the start event (the beforeLaunch hook of fklb::timePreparedLaunch), it keeps the GPU busy while the host
// records the start event and enqueues the whole call, so the interval holds the call's kernels and not the host's dispatch of them, as
// long as the spin lasts longer than the dispatch of one call.
__global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E20_SPIN_US: the length of gpuSpin in microseconds, default 5000 (0 to 1,000,000: fklb::envInt stops the program at anything else)
unsigned long long spinUs() { return static_cast<unsigned long long>(fklb::envInt("E20_SPIN_US", 5000, 0, 1000000)); }

// One row: its call, the fill of the buffers that it writes with 0xFF bytes, and the read back of what it left
struct Row {
    std::string version;
    int calls;   // library calls of one call of the row
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
T fromDevice(const T* device) {
    T host;
    FKLB_CUDA_CHECK(cudaMemcpy(&host, device, sizeof(T), cudaMemcpyDeviceToHost));
    return host;
}

// What launch() sets once and every matrix type and size uses: the mode of the run, the stream as each library takes it, and the tables
struct Run {
    const bool profile, timing;
    const int profileCalls;
    const cudaStream_t cs;
    const unsigned long long spinNs;
    cv::cuda::Stream& cvs;
    fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream;
    const NppStreamContext& ctx;
    std::unique_ptr<fklb::CsvWriter> checksTable, times;
    bool allOk = true;
};

// One size of one matrix type (T: unsigned char for 8u, float for 32f): its rows, and their checks and then their timing (in the profile
// mode, their calls only)
template <typename T>
void checkAndTime(Run& run, const Size& size) {
    const bool profile = run.profile, timing = run.timing;
    const cudaStream_t cs = run.cs;
    cv::cuda::Stream& cvs = run.cvs;
    fk::Stream_<fk::ParArch::GPU_NVIDIA>& fkStream = run.fkStream;
    const NppStreamContext& ctx = run.ctx;
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(run.spinNs); };
    const std::string typeName = IS_32F<T> ? "32f" : "8u";
    const int W = size.width, H = size.height;
    const size_t elements = static_cast<size_t>(W) * H;
    const size_t rowBytes = static_cast<size_t>(W) * sizeof(T);
    const std::string sizeName = std::to_string(W) + "x" + std::to_string(H);
    const std::string name = typeName + " " + sizeName;        // of the lines that the program prints
    const std::string rangeName = typeName + ":" + sizeName;   // of the NVTX ranges
    const std::vector<unsigned char> bytes = seededBytes(W, H);
    const std::vector<T> matrix = matrixOf<T>(bytes);
    const std::vector<T> probe = profile ? std::vector<T>() : matrixOf<T>(probeOf(bytes));
    const Reference benchmarkReference = profile ? Reference{} : referenceOf(matrix);
    const Reference probeReference = profile ? Reference{} : referenceOf(probe);
    std::vector<Row> rows;
    // The upload of a matrix into the copy of every library: the checks upload the probe and then the matrix that is timed
    std::vector<std::function<void(const std::vector<T>&)>> uploads;

    // ---- FKL: one fk::executeOperations<fk::ReduceDPP<>>, one kernel
    // Set once for the size: FKL's own copy of the matrix, uploaded before the calls, and the five doubles of the fkl row
    fk::Ptr2D<T> fkMatrix(W, H);
    uploads.push_back([&](const std::vector<T>& m) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(fkMatrix.ptr().data, fkMatrix.ptr().dims.pitch, m.data(), rowBytes, rowBytes, H, cudaMemcpyHostToDevice));
    });
    uploads.back()(matrix);
    fk::Ptr1D<double> fkOut(5);
    // Set once for the size: the workspace of the kernel, device memory of the caller, zeroed (as NPP's scratch buffers, it depends
    // on the size of the matrix and on the reductions, and on nothing of a call). A matrix of one thread block needs none.
    const size_t fkWorkspaceBytes =
        fk::ReduceDPP<>::workspaceBytes(fk::PerThreadRead<fk::ND::_2D, T>::build(fkMatrix), fklReductions<T, SumOf<T>>());
    fk::Ptr1D<uchar> fkWorkspace(static_cast<uint>(fkWorkspaceBytes));
    if (fkWorkspaceBytes > 0) {
        FKLB_CUDA_CHECK(cudaMemset(fkWorkspace.ptr().data, 0, fkWorkspaceBytes));
    }
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callFkl = [&] {
        const auto read = fk::PerThreadRead<fk::ND::_2D, T>::build(fkMatrix);
        const auto write = fk::PerThreadWrite<fk::ND::_1D, double>::build(fkOut);
        fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), read, fklReductions<T, SumOf<T>>(), write);
    };
    // (end of the timed call)
    const auto fillFklOf = [](fk::Ptr1D<double>& out) {
        return [&out] { FKLB_CUDA_CHECK(cudaMemset(out.ptr().data, 0xFF, 5 * sizeof(double))); };
    };
    const auto downloadFklOf = [](fk::Ptr1D<double>& out) {
        return [&out] {
            double five[5];
            FKLB_CUDA_CHECK(cudaMemcpy(five, out.ptr().data, sizeof(five), cudaMemcpyDeviceToHost));
            Statistics s;
            s.minimum = five[0];
            s.maximum = five[1];
            s.sum = five[2];
            s.nonZero = five[3];
            s.sumOfSquares = five[4];
            return s;
        };
    };
    rows.push_back({"fkl", 1, callFkl, fillFklOf(fkOut), downloadFklOf(fkOut)});

    // ---- FKL with the sum and the sum of squares in double, as the opencv_cuda row accumulates them: a row of the 8u matrix only (the
    // fkl row of the 32f matrix has them in double)
    // Set once for the size: its own five doubles and its own workspace, zeroed (allocated here for the 8u matrix)
    fk::Ptr1D<double> fkOutDoubleSums;
    fk::Ptr1D<uchar> fkWorkspaceDoubleSums;
    if constexpr (!IS_32F<T>) {
        fkOutDoubleSums = fk::Ptr1D<double>(5);
        const size_t bytesOfTheWorkspace =
            fk::ReduceDPP<>::workspaceBytes(fk::PerThreadRead<fk::ND::_2D, T>::build(fkMatrix), fklReductions<T, double>());
        fkWorkspaceDoubleSums = fk::Ptr1D<uchar>(static_cast<uint>(bytesOfTheWorkspace));
        if (bytesOfTheWorkspace > 0) {
            FKLB_CUDA_CHECK(cudaMemset(fkWorkspaceDoubleSums.ptr().data, 0, bytesOfTheWorkspace));
        }
        // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
        const auto callFklDoubleSums = [&] {
            const auto read = fk::PerThreadRead<fk::ND::_2D, T>::build(fkMatrix);
            const auto write = fk::PerThreadWrite<fk::ND::_1D, double>::build(fkOutDoubleSums);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspaceDoubleSums), read, fklReductions<T, double>(),
                                                   write);
        };
        // (end of the timed call)
        rows.push_back({"fkl_double_sums", 1, callFklDoubleSums, fillFklOf(fkOutDoubleSums), downloadFklOf(fkOutDoubleSums)});
    }

    // ---- CUB and Thrust: the single pass written by hand
    // Set once for the size: their contiguous copy of the matrix (no row padding), uploaded before the calls
    T* flat = deviceAlloc<T>(elements);
    uploads.push_back([&](const std::vector<T>& m) {
        FKLB_CUDA_CHECK(cudaMemcpy(flat, m.data(), elements * sizeof(T), cudaMemcpyHostToDevice));
    });
    uploads.back()(matrix);
    // Set once for the size: CUB's result in device memory, and its temporary storage (the first call, with no storage, gives its size)
    Five<T>* cubOut = deviceAlloc<Five<T>>(1);
    size_t cubStorageBytes = 0;
    FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, cubStorageBytes, thrust::make_transform_iterator(flat, FiveOfElement<T>{}), cubOut,
                                              static_cast<long long>(elements), FiveOfTwo<T>{}, FIVE_OF_NOTHING<T>, cs));
    void* cubStorage = deviceAlloc<unsigned char>(cubStorageBytes > 0 ? cubStorageBytes : 1);
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callCub = [&] {
        size_t storageBytes = cubStorageBytes;
        FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(cubStorage, storageBytes, thrust::make_transform_iterator(flat, FiveOfElement<T>{}), cubOut,
                                                  static_cast<long long>(elements), FiveOfTwo<T>{}, FIVE_OF_NOTHING<T>, cs));
    };
    // (end of the timed call)
    const auto statisticsOfFive = [](const Five<T>& five) {
        Statistics s;
        s.minimum = static_cast<double>(five.minimum);
        s.maximum = static_cast<double>(five.maximum);
        s.nonZero = static_cast<double>(five.nonZero);
        s.sum = static_cast<double>(five.sum);
        s.sumOfSquares = static_cast<double>(five.sumOfSquares);
        return s;
    };
    const auto fillCub = [&] { FKLB_CUDA_CHECK(cudaMemset(cubOut, 0xFF, sizeof(Five<T>))); };
    const auto downloadCub = [&] { return statisticsOfFive(fromDevice(cubOut)); };
    rows.push_back({"cub", 1, callCub, fillCub, downloadCub});

    // Thrust's transform_reduce returns its result to the host: the row keeps it there
    Five<T> thrustOut{};
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callThrust = [&] {
        const thrust::device_ptr<const T> first(flat);
        thrustOut = thrust::transform_reduce(thrust::cuda::par.on(cs), first, first + static_cast<long long>(elements), FiveOfElement<T>{},
                                             FIVE_OF_NOTHING<T>, FiveOfTwo<T>{});
    };
    // (end of the timed call)
    const auto fillThrust = [&] { std::memset(&thrustOut, 0xFF, sizeof(Five<T>)); };
    const auto downloadThrust = [&] { return statisticsOfFive(thrustOut); };
    rows.push_back({"thrust", 1, callThrust, fillThrust, downloadThrust});

    // Thrust with its result in device memory: thrust::reduce_into, par_nosync, the caller's block as its temporary storage
    // Set once for the size: the result, and the block. Thrust asks the allocator of its policy for the temporary storage of
    // cub::DeviceReduce::Reduce over its range (thrust/system/cuda/detail/reduce.h): that call, with no storage, gives the size
    Five<T>* thrustIntoOut = deviceAlloc<Five<T>>(1);
    const auto thrustRange = [&] { return thrust::make_transform_iterator(thrust::device_ptr<const T>(flat), FiveOfElement<T>{}); };
    BlockOfTheCaller thrustStorage;
    FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, thrustStorage.bytes, thrustRange(), thrustIntoOut, static_cast<long long>(elements),
                                              FiveOfTwo<T>{}, FIVE_OF_NOTHING<T>, cs));
    thrustStorage.block = deviceAlloc<char>(thrustStorage.bytes > 0 ? thrustStorage.bytes : 1);
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callThrustInto = [&] {
        const auto first = thrustRange();
        thrust::reduce_into(thrust::cuda::par_nosync(thrustStorage).on(cs), first, first + static_cast<long long>(elements),
                            thrust::device_pointer_cast(thrustIntoOut), FIVE_OF_NOTHING<T>, FiveOfTwo<T>{});
    };
    // (end of the timed call)
    const auto fillThrustInto = [&] { FKLB_CUDA_CHECK(cudaMemset(thrustIntoOut, 0xFF, sizeof(Five<T>))); };
    const auto downloadThrustInto = [&] { return statisticsOfFive(fromDevice(thrustIntoOut)); };
    rows.push_back({"thrust_reduce_into", 1, callThrustInto, fillThrustInto, downloadThrustInto});

    // ---- OpenCV-CUDA
    // Set once for the size: OpenCV's own copy of the matrix, uploaded before the calls
    cv::cuda::GpuMat cvMatrix;
    uploads.push_back([&](const std::vector<T>& m) {
        cvMatrix.upload(cv::Mat(H, W, IS_32F<T> ? CV_32FC1 : CV_8UC1, const_cast<T*>(m.data())));
    });
    uploads.back()(matrix);
    // Set once: the outputs, with the size and the type that each call makes for this matrix (getOutputMat keeps a GpuMat that already has
    // them). cv::cuda::findMinMax gives ints for a matrix of bytes and floats for a matrix of floats; the other types do not depend on
    // the matrix
    using CvMinMax = std::conditional_t<IS_32F<T>, float, int>;
    cv::cuda::GpuMat cvMinMax(1, 2, IS_32F<T> ? CV_32FC1 : CV_32SC1), cvSum(1, 1, CV_64FC1), cvNonZero(1, 1, CV_32SC1), cvSqrSum(1, 1, CV_64FC1);
    cv::cuda::GpuMat cvMeanStdDev(1, 2, CV_64FC1);
    // A call that makes its output again (because it is not set with its size and type) leaves it in other memory: these headers keep
    // the memory of the outputs as they are set, so no other GpuMat gets it, and the downloads refuse an output that is elsewhere
    const std::vector<const cv::cuda::GpuMat*> cvOutputs = {&cvMinMax, &cvSum, &cvNonZero, &cvSqrSum, &cvMeanStdDev};
    std::vector<cv::cuda::GpuMat> cvOutputsAsSet;
    for (const cv::cuda::GpuMat* output : cvOutputs) cvOutputsAsSet.push_back(*output);
    const auto refuseAnOutputMadeAgain = [&] {
        for (size_t k = 0; k < cvOutputs.size(); ++k) {
            if (cvOutputs[k]->data != cvOutputsAsSet[k].data) throw std::runtime_error("an OpenCV-CUDA call made again an output that is set once");
        }
    };
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callOpenCV = [&] {
        cv::cuda::findMinMax(cvMatrix, cvMinMax, cv::noArray(), cvs);
        cv::cuda::calcSum(cvMatrix, cvSum, cv::noArray(), cvs);
        cv::cuda::countNonZero(cvMatrix, cvNonZero, cvs);
        cv::cuda::calcSqrSum(cvMatrix, cvSqrSum, cv::noArray(), cvs);
    };
    // (end of the timed call)
    const auto fillOpenCVShared = [&] {
        FKLB_CUDA_CHECK(cudaMemset(cvMinMax.data, 0xFF, 2 * sizeof(CvMinMax)));
        FKLB_CUDA_CHECK(cudaMemset(cvNonZero.data, 0xFF, sizeof(int)));
    };
    const auto fillOpenCV = [&] {
        fillOpenCVShared();
        FKLB_CUDA_CHECK(cudaMemset(cvSum.data, 0xFF, sizeof(double)));
        FKLB_CUDA_CHECK(cudaMemset(cvSqrSum.data, 0xFF, sizeof(double)));
    };
    const auto downloadOpenCVShared = [&] {
        refuseAnOutputMadeAgain();
        CvMinMax minMax[2];
        FKLB_CUDA_CHECK(cudaMemcpy(minMax, cvMinMax.data, sizeof(minMax), cudaMemcpyDeviceToHost));
        Statistics s;
        s.minimum = static_cast<double>(minMax[0]);
        s.maximum = static_cast<double>(minMax[1]);
        s.nonZero = fromDevice(reinterpret_cast<const int*>(cvNonZero.data));
        return s;
    };
    const auto downloadOpenCV = [&] {
        Statistics s = downloadOpenCVShared();
        s.sum = fromDevice(reinterpret_cast<const double*>(cvSum.data));
        s.sumOfSquares = fromDevice(reinterpret_cast<const double*>(cvSqrSum.data));
        return s;
    };
    rows.push_back({"opencv_cuda", 4, callOpenCV, fillOpenCV, downloadOpenCV});

    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callOpenCVMsd = [&] {
        cv::cuda::findMinMax(cvMatrix, cvMinMax, cv::noArray(), cvs);
        cv::cuda::countNonZero(cvMatrix, cvNonZero, cvs);
        cv::cuda::meanStdDev(cvMatrix, cvMeanStdDev, cvs);
    };
    // (end of the timed call)
    const auto fillOpenCVMsd = [&] {
        fillOpenCVShared();
        FKLB_CUDA_CHECK(cudaMemset(cvMeanStdDev.data, 0xFF, 2 * sizeof(double)));
    };
    const auto downloadOpenCVMsd = [&] {
        Statistics s = downloadOpenCVShared();
        double meanStdDev[2];
        FKLB_CUDA_CHECK(cudaMemcpy(meanStdDev, cvMeanStdDev.data, sizeof(meanStdDev), cudaMemcpyDeviceToHost));
        s.mean = meanStdDev[0];
        s.stdDev = meanStdDev[1];
        return s;
    };
    rows.push_back({"opencv_cuda_msd", 3, callOpenCVMsd, fillOpenCVMsd, downloadOpenCVMsd, true});

    // ---- NPP
    // Set once for the size: NPP's own copy of the matrix (cudaMallocPitch), uploaded before the calls
    T* nppMatrix = nullptr;
    size_t nppStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppMatrix), &nppStep, rowBytes, H));
    uploads.push_back([&](const std::vector<T>& m) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(nppMatrix, nppStep, m.data(), rowBytes, rowBytes, H, cudaMemcpyHostToDevice));
    });
    uploads.back()(matrix);
    const NppiSize nppSize{W, H};
    // Set once for the size: the results in device memory, and the scratch buffer of every call
    T* nppMin = deviceAlloc<T>(1);
    T* nppMax = deviceAlloc<T>(1);
    Npp64f* nppSum = deviceAlloc<Npp64f>(1);
    int* nppCount = deviceAlloc<int>(1);
    Npp64f* nppDotProd = deviceAlloc<Npp64f>(1);
    Npp64f* nppMean = deviceAlloc<Npp64f>(1);
    Npp64f* nppStdDev = deviceAlloc<Npp64f>(1);
    size_t minMaxBytes = 0, sumBytes = 0, countBytes = 0, dotProdBytes = 0, meanStdDevBytes = 0;
    if constexpr (IS_32F<T>) {
        FKLB_NPP_CHECK(nppiMinMaxGetBufferHostSize_32f_C1R_Ctx(nppSize, &minMaxBytes, ctx));
        FKLB_NPP_CHECK(nppiSumGetBufferHostSize_32f_C1R_Ctx(nppSize, &sumBytes, ctx));
        FKLB_NPP_CHECK(nppiCountInRangeGetBufferHostSize_32f_C1R_Ctx(nppSize, &countBytes, ctx));
        FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_32f64f_C1R_Ctx(nppSize, &dotProdBytes, ctx));
        FKLB_NPP_CHECK(nppiMeanStdDevGetBufferHostSize_32f_C1R_Ctx(nppSize, &meanStdDevBytes, ctx));
    } else {
        FKLB_NPP_CHECK(nppiMinMaxGetBufferHostSize_8u_C1R_Ctx(nppSize, &minMaxBytes, ctx));
        FKLB_NPP_CHECK(nppiSumGetBufferHostSize_8u_C1R_Ctx(nppSize, &sumBytes, ctx));
        FKLB_NPP_CHECK(nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx(nppSize, &countBytes, ctx));
        FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_8u64f_C1R_Ctx(nppSize, &dotProdBytes, ctx));
        FKLB_NPP_CHECK(nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(nppSize, &meanStdDevBytes, ctx));
    }
    Npp8u* minMaxBuffer = deviceAlloc<Npp8u>(minMaxBytes > 0 ? minMaxBytes : 1);
    Npp8u* sumBuffer = deviceAlloc<Npp8u>(sumBytes > 0 ? sumBytes : 1);
    Npp8u* countBuffer = deviceAlloc<Npp8u>(countBytes > 0 ? countBytes : 1);
    Npp8u* dotProdBuffer = deviceAlloc<Npp8u>(dotProdBytes > 0 ? dotProdBytes : 1);
    Npp8u* meanStdDevBuffer = deviceAlloc<Npp8u>(meanStdDevBytes > 0 ? meanStdDevBytes : 1);
    const int step = static_cast<int>(nppStep);
    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callNpp = [&] {
        if constexpr (IS_32F<T>) {
            FKLB_NPP_CHECK(nppiMinMax_32f_C1R_Ctx(nppMatrix, step, nppSize, nppMin, nppMax, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiSum_32f_C1R_Ctx(nppMatrix, step, nppSize, sumBuffer, nppSum, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_32f_C1R_Ctx(nppMatrix, step, nppSize, nppCount, NON_ZERO_FROM, NON_ZERO_TO, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiDotProd_32f64f_C1R_Ctx(nppMatrix, step, nppMatrix, step, nppSize, nppDotProd, dotProdBuffer, ctx));
        } else {
            FKLB_NPP_CHECK(nppiMinMax_8u_C1R_Ctx(nppMatrix, step, nppSize, nppMin, nppMax, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiSum_8u_C1R_Ctx(nppMatrix, step, nppSize, sumBuffer, nppSum, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_8u_C1R_Ctx(nppMatrix, step, nppSize, nppCount, 1, 255, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiDotProd_8u64f_C1R_Ctx(nppMatrix, step, nppMatrix, step, nppSize, nppDotProd, dotProdBuffer, ctx));
        }
    };
    // (end of the timed call)
    const auto fillNppShared = [&] {
        FKLB_CUDA_CHECK(cudaMemset(nppMin, 0xFF, sizeof(T)));
        FKLB_CUDA_CHECK(cudaMemset(nppMax, 0xFF, sizeof(T)));
        FKLB_CUDA_CHECK(cudaMemset(nppCount, 0xFF, sizeof(int)));
    };
    const auto fillNpp = [&] {
        fillNppShared();
        FKLB_CUDA_CHECK(cudaMemset(nppSum, 0xFF, sizeof(Npp64f)));
        FKLB_CUDA_CHECK(cudaMemset(nppDotProd, 0xFF, sizeof(Npp64f)));
    };
    const auto downloadNppShared = [&] {
        Statistics s;
        s.minimum = static_cast<double>(fromDevice(nppMin));
        s.maximum = static_cast<double>(fromDevice(nppMax));
        s.nonZero = fromDevice(nppCount);
        return s;
    };
    const auto downloadNpp = [&] {
        Statistics s = downloadNppShared();
        s.sum = fromDevice(nppSum);
        s.sumOfSquares = fromDevice(nppDotProd);
        return s;
    };
    rows.push_back({"npp", 4, callNpp, fillNpp, downloadNpp});

    // The timed call (whole_call and kernel_only). Everything in this lambda is inside the interval
    const auto callNppMsd = [&] {
        if constexpr (IS_32F<T>) {
            FKLB_NPP_CHECK(nppiMinMax_32f_C1R_Ctx(nppMatrix, step, nppSize, nppMin, nppMax, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_32f_C1R_Ctx(nppMatrix, step, nppSize, nppCount, NON_ZERO_FROM, NON_ZERO_TO, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiMean_StdDev_32f_C1R_Ctx(nppMatrix, step, nppSize, meanStdDevBuffer, nppMean, nppStdDev, ctx));
        } else {
            FKLB_NPP_CHECK(nppiMinMax_8u_C1R_Ctx(nppMatrix, step, nppSize, nppMin, nppMax, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_8u_C1R_Ctx(nppMatrix, step, nppSize, nppCount, 1, 255, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiMean_StdDev_8u_C1R_Ctx(nppMatrix, step, nppSize, meanStdDevBuffer, nppMean, nppStdDev, ctx));
        }
    };
    // (end of the timed call)
    const auto fillNppMsd = [&] {
        fillNppShared();
        FKLB_CUDA_CHECK(cudaMemset(nppMean, 0xFF, sizeof(Npp64f)));
        FKLB_CUDA_CHECK(cudaMemset(nppStdDev, 0xFF, sizeof(Npp64f)));
    };
    const auto downloadNppMsd = [&] {
        Statistics s = downloadNppShared();
        s.mean = fromDevice(nppMean);
        s.stdDev = fromDevice(nppStdDev);
        return s;
    };
    rows.push_back({"npp_msd", 3, callNppMsd, fillNppMsd, downloadNppMsd, true});

    const std::vector<std::string>& rowsOfTheType = rowsOfType(typeName);
    if (rows.size() != rowsOfTheType.size()) throw std::runtime_error("the rows are not the ones of ROWS");
    for (size_t k = 0; k < rows.size(); ++k) {
        if (rows[k].version != rowsOfTheType[k]) throw std::runtime_error("the rows are not the ones of ROWS");
    }

    // ---- The profile mode: a warm-up call outside any range, then E20_PROFILE_CALLS calls, each in the range "<row>:call"
    if (profile) {
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            row.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA((row.version + ":" + rangeName).c_str());
            for (int k = 0; k < run.profileCalls; ++k) {
                nvtxRangePushA((row.version + ":call").c_str());
                row.call();
                nvtxRangePop();
            }
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePop();
        }
    }

    // ---- The checks of the size, on the probe and then on the matrix that is timed (which stays in the copies of the libraries): one
    // call of every selected row, after the buffers of the statistics of every row are filled with 0xFF bytes
    bool sizeOk = true;
    const auto checkRows = [&](const char* matrixName, const Reference& reference) {
        const double n = static_cast<double>(reference.elements);
        const auto check = [&](const std::string& version, const char* statistic, const char* comparedWith, const double value,
                               const double expected, const double tolerance, const char* criterion) {
            const double difference = relativeDifference(value, expected);
            const bool ok = difference <= tolerance;
            run.checksTable->row(typeName, W, H, matrixName, version, statistic, comparedWith, exact(value), exact(expected), difference, criterion,
                                 ok ? "yes" : "no");
            if (!ok) {
                sizeOk = false;
                std::cout << name << " " << matrixName << " " << version << " " << statistic << ": " << value << " against " << expected << " ("
                          << comparedWith << "), relative difference " << difference << "; " << criterion << ": FAILED" << std::endl;
            }
        };
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            for (const Row& any : rows) any.fill();
            FKLB_CUDA_CHECK(cudaDeviceSynchronize());
            row.call();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            const Statistics s = row.download();
            check(row.version, "minimum", "host", s.minimum, reference.minimum, 0, "equal");
            check(row.version, "maximum", "host", s.maximum, reference.maximum, 0, "equal");
            check(row.version, "non_zero", "host", s.nonZero, reference.nonZero, 0, "equal");
            double variance = std::nan("");
            if (!row.leavesMeanAndStdDev) {
                check(row.version, "sum", "host", s.sum, reference.sum, 0, "equal");
                check(row.version, "sum_of_squares", "host", s.sumOfSquares, reference.sumOfSquares, 0, "equal");
                variance = s.sumOfSquares / n - (s.sum / n) * (s.sum / n);
            } else {
                check(row.version, "mean", "host", s.mean, reference.mean, RELATIVE_TOLERANCE, "relative difference at most 1e-9");
                check(row.version, "std_dev", "host", s.stdDev, reference.stdDev, RELATIVE_TOLERANCE, "relative difference at most 1e-9");
                variance = s.stdDev * s.stdDev;
            }
            check(row.version, "variance", "host; computed on the host from what the row left", variance, reference.variance,
                  RELATIVE_TOLERANCE, "relative difference at most 1e-9");
        }
    };
    if (!profile) {
        const Reference& reference = benchmarkReference;
        for (const auto& upload : uploads) upload(probe);
        checkRows("probe", probeReference);
        for (const auto& upload : uploads) upload(matrix);
        checkRows("benchmark", benchmarkReference);
        run.checksTable->flush();
        std::cout << name << ": " << (sizeOk ? "checks passed" : "CHECK FAILED") << " (" << reference.elements << " elements, minimum "
                  << exact(reference.minimum) << ", maximum " << exact(reference.maximum) << ", non zero " << exact(reference.nonZero) << ", sum "
                  << exact(reference.sum) << ", sum of squares " << exact(reference.sumOfSquares) << ", variance " << reference.variance << ")"
                  << std::endl;
        run.allOk = run.allOk && sizeOk;
    }

    // ---- The timing of the size, after its checks, in the order of the rows. The timing table is opened when the first size is timed: a
    // run of checks only, or one in which no size passes its checks, writes none
    if (timing && sizeOk) {
        if (!run.times) {
            run.times = std::make_unique<fklb::CsvWriter>(
                "e20_reductions.csv", "E20 reductions: whole-call and kernel-only times of the five statistics of one matrix",
                std::initializer_list<const char*>{"version", "type", "width", "height", "elements", "library_calls", "interval", "mean_ms",
                                                   "min_ms", "iterations", "spin_us"});
        }
        const auto nothingToPrepare = [] { return 0; };
        for (const Row& row : rows) {
            if (!selected(row.version)) continue;
            fklb::Timing whole, gpu, host;
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("whole_call:" + row.version + ":" + rangeName).c_str());
            fklb::timeIt(cs, row.call, whole);
            nvtxRangePop();
            FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
            nvtxRangePushA(("kernel_only:" + row.version + ":" + rangeName).c_str());
            fklb::timePreparedLaunch(cs, nothingToPrepare, spin, [&](int) { row.call(); }, gpu, host);
            nvtxRangePop();
            run.times->row(row.version, typeName, W, H, elements, row.calls, "whole_call", whole.mean, whole.min, fklb::ITERATIONS, "");
            run.times->row(row.version, typeName, W, H, elements, row.calls, "kernel_only", gpu.mean, gpu.min, fklb::ITERATIONS, spinUs());
            run.times->flush();
            std::cout << name << " " << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min
                      << "); kernel only mean " << gpu.mean << " ms (min " << gpu.min << ")" << std::endl;
        }
    } else if (timing) {
        std::cout << name << ": not timed: a check failed" << std::endl;
    }

    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    for (void* p : std::vector<void*>{flat, cubOut, cubStorage, thrustIntoOut, thrustStorage.block, nppMatrix, nppMin, nppMax, nppSum, nppCount,
                                      nppDotProd, nppMean, nppStdDev, minMaxBuffer, sumBuffer, countBuffer, dotProdBuffer, meanStdDevBuffer}) {
        FKLB_CUDA_CHECK(cudaFree(p));
    }
}
} // namespace

int launch() {
    const bool profile = envIs1("E20_PROFILE");
    const bool timing = !profile && !envIs1("E20_NO_TIMING");
    const int profileCalls = fklb::envInt("E20_PROFILE_CALLS", 10, 1, 1000000);
    const std::vector<std::string> types = typesOfTheRun();
    refuseUnknownRows(types);
    std::cout << "E20 reductions: "
              << (profile ? "profile mode (a warm-up call, then E20_PROFILE_CALLS calls per row, each in its NVTX range; nothing compared, "
                            "nothing timed)"
                          : (timing ? "for every type and size, the checks and then the timing" : "checks only (E20_NO_TIMING=1)"))
              << std::endl;
    const std::vector<Size> sizes = sizesOfTheRun();
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    const unsigned long long spinNs = 1000ull * spinUs();

    // Set once: the cv::cuda::Stream and FKL's stream object of the program's stream, and the NPP stream context, filled field by field
    // from the device properties and the stream's flags as in E18 and E19 (CUDA 13 has no nppGetStreamContext)
    // OpenCV's buffer pool, turned on before the stream is wrapped (a cv::cuda::Stream takes its allocator when it is made):
    // cv::cuda::meanStdDev then takes NPP's scratch buffer from the pool of the stream. With the pool off (OpenCV's default) every
    // cv::cuda::meanStdDev allocates the buffer and frees it when it returns, and cudaFree waits for the work of the stream
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

    Run run{profile, timing, profileCalls, cs, spinNs, cvs, fkStream, ctx, nullptr, nullptr};
    // The checks table is opened unless nothing is checked (the profile mode writes no table)
    if (!profile) {
        run.checksTable = std::make_unique<fklb::CsvWriter>(
            "e20_reductions_checks.csv", "E20 reductions: the checks of every type and size",
            std::initializer_list<const char*>{"type", "width", "height", "matrix", "version", "statistic", "compared_with", "value", "reference",
                                               "relative_difference", "criterion", "ok"});
    }

    for (const std::string& type : types) {
        // A type none of whose rows is selected has nothing to check or time (E20_ONLY=fkl_double_sums names a row of the 8u matrix only)
        if (std::none_of(rowsOfType(type).begin(), rowsOfType(type).end(), selected)) {
            std::cout << type << ": no row of E20_ONLY is a row of this matrix type: nothing to do" << std::endl;
            continue;
        }
        for (const Size& size : sizes) {
            if (type == "32f") {
                checkAndTime<float>(run, size);
            } else {
                checkAndTime<unsigned char>(run, size);
            }
        }
    }
    if (profile) {
        std::cout << "profile mode: " << profileCalls << " calls per selected row, type and size, each in its NVTX range" << std::endl;
    } else {
        std::cout << (run.allOk ? "E20 reductions checks passed" : "E20 reductions CHECK FAILED") << std::endl;
    }
    // The wrapper of the stream gives its part of the pool back, and synchronises the stream to do it: before the stream is destroyed
    cvs = cv::cuda::Stream::Null();
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return run.allOk ? 0 : 1;
}
