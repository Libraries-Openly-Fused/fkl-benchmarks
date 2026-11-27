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

// E20 batched reductions (NVIDIA): the five statistics of E20 reductions (e20_reductions.cu: minimum, maximum, sum, number of elements that
// are not zero and sum of the squares, left in device memory, and the variance computed on the host from them) for each of 30 crops of
// one frame. FKL's ReduceDPP takes a batch: fkl_batch computes the 30 crops in one kernel, every crop reduced on its own. OpenCV-CUDA and
// NPP have no batched call for any of these reductions (OpenCV-CUDA has no batched call at all; NPP's batched statistics are its image
// quality metrics), so they loop over the crops; fkl_loop is FKL with that loop, one kernel per crop. CUB and Thrust have a batched call,
// with the 30 crops as the 30 parts of one range: cub::DeviceSegmentedReduce::Reduce (the cub_segmented row) and thrust::reduce_by_key
// (the thrust_reduce_by_key row). The cub, thrust and thrust_reduce_into rows loop over the crops.
// The rows differ in one more thing than the number of calls, the accumulators, and it has its row: fkl_batch, fkl_loop and the rows of
// CUB and Thrust accumulate in integers, the two sums in 64 bits. cv::cuda::calcSum and cv::cuda::calcSqrSum accumulate in double
// (cudaarithm's sum.cu: sumImpl<uchar, double, 1> and sumSqrImpl<uchar, double, 1>, on cudev's SumReductor), and NPP gives its sum, its
// dot product, its mean and its standard deviation as Npp64f. fkl_batch_double_sums is fkl_batch with its two sums accumulated in
// double, as the opencv_cuda row has them: what it takes more than fkl_batch is what the type of the accumulators costs in FKL's kernel
// on the GPU of the measurement. It has nothing to do with the number of calls.
//
// The frame and the crops:
//   frame    one 3840 x 2160 one-channel 8-bit frame, the matrix of E20 reductions at that size (the LCG of E4, E18 and E19, seed 12345; an
//            element is 0 when the bits 31..24 of its state, b, are below 8, and 5 + b mod 246 otherwise)
//   crops    30 crops of 256 x 256, E19's layout: crop i (0..29) starts at x = 64 + 600 * (i mod 6), y = 64 + 400 * (i div 6). The program
//            treats it as a layout that changes on every frame: every call reads the positions from the detection table and builds every
//            ROI, crop pointer or crop view from them
// Each library reads its own copy of the same host bytes, crops its own way (OpenCV: a GpuMat ROI; NPP: a pointer into its frame and the
// frame's step; CUB and Thrust: an iterator over the elements of the crop, or of the 30 crops; FKL: fk::Ptr2D::crop2D), and owns its
// outputs.
//
// The rows, each one a call that computes the 30 crops and leaves in device memory their statistics (the thrust row: in host memory).
// The host waits for the GPU inside a call of two rows, thrust (in each of its 30 calls of Thrust) and thrust_reduce_by_key (in its one
// call of Thrust), and of no other row:
//   fkl_batch        one fk::executeOperations<fk::ReduceDPP<>>, one kernel. The read is a BatchRead of the 30 crop views
//                    (fk::PerThreadRead<_2D, uchar>::build of a std::array of the 30 fkFrame.crop2D): one grid z plane per crop. The
//                    reductions are the five of E20 reductions (accumulators uchar, uchar, ulonglong, uint and ulonglong; IsNonZero and
//                    Square are operations of this program), and the write is fk::TensorWrite<double> of a tensor of 5 x 1 x 30: the five
//                    values of crop i in plane i. The workspace of the kernel is allocated and zeroed once.
//                    Every call builds the array of the 30 crop views, the read, the reductions, the write and the details, and launches
//                    1 kernel.
//   fkl_batch_double_sums  fkl_batch with the sum and the sum of squares accumulated in double: fk::Cast<uchar, double> and SquareAsDouble
//                    (an operation of this program) as their first operations, and fk::Add<double>. The minimum, the maximum and the
//                    count as in fkl_batch (uchar, uchar and uint). The same read, the same write into the same tensor, the same
//                    workspace; the sums of a crop are integers below 2^53, so they are exact in a double too. 1 kernel.
//   fkl_loop         per crop, one fk::executeOperations<fk::ReduceDPP<>> with the reductions of fkl_batch, the read of that crop view
//                    and fk::PerThreadWrite<_3D, double> of the view of plane i of the same tensor; one workspace for the 30 calls, which
//                    run one after the other on the stream. 30 kernels.
//   cub              per crop, cub::DeviceReduce::Reduce (see e20_reductions.cu for why not TransformReduce) over a transform iterator
//                    on a counting iterator: element k of the crop is the byte (y + k / 256, x + k mod 256) of a contiguous copy of the
//                    frame (CUB and Thrust reduce ranges; the crop width is a constant of the program, so the division is by a
//                    constant). The result of crop i in element i of a device array; the temporary storage is allocated once. 30 calls.
//   cub_segmented    one cub::DeviceSegmentedReduce::Reduce, the overload with one size for every segment (CCCL 3.2): 30 segments of
//                    65,536 elements of one range, whose element g is element g mod 65,536 of crop g div 65,536. The range is a transform
//                    iterator on a counting iterator, and its functor reads the position of the crop from a table in device memory. Every
//                    call uploads the table (one cudaMemcpyAsync of the 30 positions of the detection table, 240 bytes, which does not
//                    wait for the stream) and then makes the call of CUB; the results in a device array, the temporary storage allocated
//                    once. 1 call. Where the functor has the positions matters: with the 30 of them held in the functor itself, the same
//                    call takes 211 us where this row takes 16 us (RTX 4090). CUB's two other calls for several ranges are slower, with
//                    the same table, and are not rows: the overload of this call that takes the offsets of the segments (28 us) and
//                    cub::DeviceReduce::ReduceByKey (109 us).
//   thrust           per crop, thrust::transform_reduce over the counting iterator with the operations of the cub row (thrust::cuda::par
//                    on the stream). It returns the result, so it is NOT like the other rows: every one of the 30 calls waits for its
//                    result and copies it to the host, where the row keeps it, and allocates and releases its temporary storage. 30 calls.
//   thrust_reduce_into  per crop, thrust::reduce_into (CCCL 3.2) over the range of the cub row, into element i of a device array, with
//                    thrust::cuda::par_nosync on the stream (no call waits) and a block of the caller as the allocator of the policy
//                    (allocated once: no call allocates). Like the other rows, and the kernels of the cub row. 30 calls.
//   thrust_reduce_by_key  one thrust::reduce_by_key over the range of the cub_segmented row (the same table, uploaded in the call), with
//                    the crop of an element (g div 65,536) as its key; the keys of the result are discarded and its values go to a device
//                    array. The policy of thrust_reduce_into. The call returns the ends of its results, so it waits for them. 1 call.
//   opencv_cuda      per crop, on the ROI of the crop: cv::cuda::findMinMax, cv::cuda::calcSum, cv::cuda::countNonZero and
//                    cv::cuda::calcSqrSum, each into row i of a GpuMat of 30 rows that is set once (a row is a GpuMat of the size and
//                    type that the call makes, so getOutputMat keeps it). 120 calls.
//   opencv_cuda_msd  per crop: cv::cuda::findMinMax, cv::cuda::countNonZero and cv::cuda::meanStdDev. 90 calls. cv::cuda::meanStdDev takes
//                    NPP's scratch buffer from the BufferPool of the stream, which this program turns on before it wraps its stream
//                    (cv::cuda::setBufferPoolUsage(true): set once, as NPP's scratch buffers and FKL's workspace are). With OpenCV's
//                    default, the pool off, every cv::cuda::meanStdDev allocates the buffer and frees it when it returns, and cudaFree
//                    waits for the work of the stream: 30 allocations and 30 waits per call of the row.
//   npp              per crop, from the crop pointer and the frame step: nppiMinMax_8u_C1R_Ctx, nppiSum_8u_C1R_Ctx,
//                    nppiCountInRange_8u_C1R_Ctx (bounds 1 and 255) and nppiDotProd_8u64f_C1R_Ctx of the crop with itself (the sum of
//                    the squares, as in e20_reductions.cu), each into element i of a device array; one scratch buffer per function,
//                    allocated once (every crop has the same size). 120 calls.
//   npp_msd          per crop: nppiMinMax_8u_C1R_Ctx, nppiCountInRange_8u_C1R_Ctx and nppiMean_StdDev_8u_C1R_Ctx. 90 calls.
// The OpenCV-CUDA, NPP, CUB and Thrust rows use no FKL object.
//
// The checks (e20_batched_reductions_checks.csv): the checks of E20 reductions, for every crop, on two frames: the probe and then the frame
// that is timed. Every crop of the frame has the minimum 0 and the maximum 250, whatever crop a call reduces. The probe has no zero (every
// value in 32..222) and, in crop i, one 226 + i in its first element and one 1 + i in its last one: the maximum and the minimum of a crop
// are right only for that crop, with its first and its last element, a minimum of 0 is never right, and the maximum of the last crop is
// 255, the largest byte, the upper bound of NPP's count. Every library's copy of the frame is uploaded again for each. The host computes
// the five statistics of each crop in integers, and from them its mean, standard deviation and variance. Each row is called once, after
// the buffers that hold the statistics of EVERY row are filled with 0xFF bytes (never a right value, but for the 255 of a byte, which is
// the maximum of the last crop of the probe: there a row that leaves a byte is caught by the frame that is timed, where no maximum is
// 255) and the table of positions of the cub_segmented and thrust_reduce_by_key rows with zeros (30 times the crop at the corner of the
// frame, which is none of the 30), and the statistics of its 30 crops are read back. One line per row and statistic: the crops whose
// value differs from the host's by more than the criterion allows (equal, for the minimum, the maximum, the count, the sum and the sum
// of squares; relative difference at most 1e-9 for the mean, the standard deviation and the variance), and the largest relative
// difference. The timing needs 0 differing crops in every line.
//
// The timing (e20_batched_reductions.csv): two intervals per row, in the order fkl_batch, fkl_batch_double_sums, fkl_loop, cub,
// cub_segmented, thrust, thrust_reduce_into, thrust_reduce_by_key, opencv_cuda, opencv_cuda_msd, npp, npp_msd, the same in every run.
// Both hold the whole frame (the 30 crops):
//   whole_call   fklb::timeIt: CUDA events around one call, 100 calls after one warm-up call. When the GPU waits for the host, the
//                interval holds the host's work too
//   kernel_only  fklb::timePreparedLaunch with gpuSpin as beforeLaunch and nothing to prepare: the whole call is enqueued behind one spin
//                of E20_SPIN_US (default 5,000 us). The thrust row waits for the result of every crop inside the call, and the
//                thrust_reduce_by_key row for its results, so their intervals also hold what the host does after the spin.
//   Each interval of each row is inside an NVTX range ("whole_call:<row>", "kernel_only:<row>").
// E20_ONLY=a,b: only these rows (a name that is no row stops the program). E20_NO_TIMING=1: the checks only. E20_PROFILE=1: every selected
// row is called once outside any range (a warm-up), then E20_PROFILE_CALLS times (default 10), each call in an NVTX range "<row>:call"
// inside the range "<row>"; nothing is compared or timed (to count the kernels of one call with nsys).
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
#include <fused_kernel/core/data/rect.h>
#include <fused_kernel/core/execution_model/reduce_dpp.h>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_segmented_reduce.cuh>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>
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
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ---- The user's operations of the FKL rows, which FKL's basic operations do not have (the ones of e20_reductions.cu)
namespace e20 {
// 1 for an element that is not zero: the FirstIOp of the count
struct IsNonZero {
    using Parent = fk::UnaryOperation<uchar, uint, IsNonZero>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) { return input != 0 ? 1u : 0u; }
};

// The square of an element, in the accumulator type of the sum of squares
struct Square {
    using Parent = fk::UnaryOperation<uchar, ulonglong, Square>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) { return static_cast<ulonglong>(input) * input; }
};

// The square of an element as a double: the FirstIOp of the sum of squares of the fkl_batch_double_sums row
struct SquareAsDouble {
    using Parent = fk::UnaryOperation<uchar, double, SquareAsDouble>;
    DECLARE_UNARY_PARENT
    FK_HOST_DEVICE_FUSE OutputType exec(const InputType input) { return static_cast<double>(input) * static_cast<double>(input); }
};
} // namespace e20

namespace {
constexpr int SRC_W = 3840, SRC_H = 2160;
constexpr int CROPS = 30, CROP_W = 256, CROP_H = 256;
constexpr long long CROP_ELEMENTS = static_cast<long long>(CROP_W) * CROP_H;   // 65,536 elements per crop
constexpr int STATISTICS = 5;                                                   // minimum, maximum, sum, non zero, sum of squares
constexpr size_t THRUST_BLOCK_BYTES = 1u << 20;                                 // the block of the caller of the Thrust rows: 1 MiB
constexpr double RELATIVE_TOLERANCE = 1e-9;   // the mean, the standard deviation and the variance, against the host's in double

// The top-left corner of one crop in the frame
struct Detection {
    int x, y;
};

// ---- The single pass of the cub and thrust rows: the five values, with the accumulator types of the FKL rows
struct Five {
    unsigned char minimum, maximum;
    unsigned int nonZero;
    unsigned long long sum, sumOfSquares;
};

// Element k of a crop, row by row: the byte (y + k / CROP_W, x + k mod CROP_W) of a contiguous frame of SRC_W bytes per row, as the five
// values of one element
struct FiveOfCropElement {
    const unsigned char* frame;
    int x, y;
    __host__ __device__ Five operator()(const long long k) const {
        const unsigned char v = frame[(y + k / CROP_W) * SRC_W + x + k % CROP_W];
        return Five{v, v, v != 0 ? 1u : 0u, v, static_cast<unsigned long long>(v) * v};
    }
};

struct FiveOfTwo {
    __host__ __device__ Five operator()(const Five& a, const Five& b) const {
        return Five{a.minimum < b.minimum ? a.minimum : b.minimum, a.maximum > b.maximum ? a.maximum : b.maximum, a.nonZero + b.nonZero,
                    a.sum + b.sum, a.sumOfSquares + b.sumOfSquares};
    }
};

// The value that changes no other: the initial value of the reductions of cub and thrust
constexpr Five FIVE_OF_NOTHING{255, 0, 0u, 0ull, 0ull};

// Element g of the 30 crops one after the other, as the five values of one element: element g mod CROP_ELEMENTS of crop g div CROP_ELEMENTS.
// The range of the cub_segmented and thrust_reduce_by_key rows, in unsigned 32-bit arithmetic (the range has 1,966,080 elements and the
// frame 8,294,400 bytes). The positions of the crops are read from a table in device memory, not held in the functor (see cub_segmented in
// the header)
struct FiveOfBatchElement {
    const unsigned char* frame;
    const Detection* detections;
    __host__ __device__ Five operator()(const unsigned int g) const {
        const Detection crop = detections[g / static_cast<unsigned int>(CROP_ELEMENTS)];
        const unsigned int k = g % static_cast<unsigned int>(CROP_ELEMENTS);
        const unsigned char v = frame[(crop.y + k / CROP_W) * SRC_W + crop.x + k % CROP_W];
        return Five{v, v, v != 0 ? 1u : 0u, v, static_cast<unsigned long long>(v) * v};
    }
};

// The crop of element g of that range: the key of the thrust_reduce_by_key row
struct CropOfBatchElement {
    __host__ __device__ int operator()(const unsigned int g) const { return static_cast<int>(g / static_cast<unsigned int>(CROP_ELEMENTS)); }
};

// The temporary storage of the thrust_reduce_into and thrust_reduce_by_key rows: a block of device memory that the caller owns, as the
// storage of the cub rows, given to Thrust as the allocator of its execution policy (the idiom of Thrust's custom_temporary_allocation
// example). A call of Thrust takes its arrays from the block one after the other, each at a multiple of 256 bytes; every call of Thrust
// is given a new BlockOfTheCaller, so it starts at the beginning of the block again
struct BlockOfTheCaller {
    using value_type = char;
    char* block = nullptr;
    size_t bytes = 0, taken = 0;
    char* allocate(const size_t count) {
        const size_t rounded = (count + 255) / 256 * 256;
        if (taken + rounded > bytes) throw std::runtime_error("Thrust asks for more temporary storage than the block of the caller has");
        taken += rounded;
        return block + (taken - rounded);
    }
    void deallocate(char*, size_t) {}
};

// The rows, in the order of the checks and of the timing
const std::vector<std::string> ROWS = {"fkl_batch", "fkl_batch_double_sums", "fkl_loop", "cub", "cub_segmented", "thrust", "thrust_reduce_into",
                                       "thrust_reduce_by_key", "opencv_cuda", "opencv_cuda_msd", "npp", "npp_msd"};

bool selected(const std::string& name) {
    const char* e = std::getenv("E20_ONLY");
    if (e == nullptr) return true;
    return (std::string(",") + e + ",").find("," + name + ",") != std::string::npos;
}

// E20_ONLY names rows: a name that is no row (fkl, the row of the two other programs) would leave empty tables behind a "checks passed"
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

bool envIs1(const char* name) { return std::getenv(name) != nullptr && std::string(std::getenv(name)) == "1"; }

// The matrix of E20 reductions: one element per state of the LCG of E4, E18 and E19 (seed 12345), from its bits 31..24 (b): 0 when
// b < 8, and 5 + b mod 246 otherwise
std::vector<unsigned char> seededMatrix(const int width, const int height) {
    std::vector<unsigned char> matrix(static_cast<size_t>(width) * height);
    uint32_t state = 12345u;
    for (unsigned char& v : matrix) {
        state = state * 1664525u + 1013904223u;
        const unsigned int b = state >> 24;
        v = static_cast<unsigned char>(b < 8 ? 0 : 5 + b % 246);
    }
    return matrix;
}

// The detections of the frame, E19's: crop i at x = 64 + 600 * (i mod 6), y = 64 + 400 * (i div 6)
std::vector<Detection> detectionsOfTheFrame() {
    std::vector<Detection> d;
    for (int i = 0; i < CROPS; ++i) d.push_back({64 + 600 * (i % 6), 64 + 400 * (i / 6)});
    return d;
}

// The probe of the checks: the frame with every value v replaced by 32 + v * 190 / 250 (32..222: no zero), and in crop i a 226 + i in its
// first element and a 1 + i in its last one (the crops do not overlap). The maximum and the minimum of a crop are single elements at its two
// ends, they are those of no other crop, and the maximum of the last crop is 255, the largest byte
std::vector<unsigned char> probeOf(const std::vector<unsigned char>& frame, const std::vector<Detection>& detections) {
    static_assert(226 + (CROPS - 1) == 255, "the first element of the last crop of the probe is the largest byte");
    std::vector<unsigned char> probe(frame.size());
    for (size_t k = 0; k < frame.size(); ++k) probe[k] = static_cast<unsigned char>(32 + frame[k] * 190 / 250);
    for (int i = 0; i < CROPS; ++i) {
        probe[static_cast<size_t>(detections[i].y) * SRC_W + detections[i].x] = static_cast<unsigned char>(226 + i);
        probe[(static_cast<size_t>(detections[i].y) + CROP_H - 1) * SRC_W + detections[i].x + CROP_W - 1] = static_cast<unsigned char>(1 + i);
    }
    return probe;
}

// What a row leaves for one crop (NaN where it computes no such value)
struct Statistics {
    double minimum = std::nan(""), maximum = std::nan(""), nonZero = std::nan("");
    double sum = std::nan(""), sumOfSquares = std::nan("");
    double mean = std::nan(""), stdDev = std::nan("");
};

// What the host computes from the bytes of one crop
struct Reference {
    unsigned long long minimum = 255, maximum = 0, nonZero = 0, sum = 0, sumOfSquares = 0;
    double mean = 0, stdDev = 0, variance = 0;
};

Reference referenceOf(const std::vector<unsigned char>& frame, const Detection& detection) {
    Reference r;
    for (int y = 0; y < CROP_H; ++y) {
        for (int x = 0; x < CROP_W; ++x) {
            const unsigned char v = frame[(static_cast<size_t>(detection.y) + y) * SRC_W + detection.x + x];
            r.minimum = v < r.minimum ? v : r.minimum;
            r.maximum = v > r.maximum ? v : r.maximum;
            r.nonZero += v != 0 ? 1 : 0;
            r.sum += v;
            r.sumOfSquares += static_cast<unsigned long long>(v) * v;
        }
    }
    // The population variance, (N * sum of squares - sum^2) / N^2: the numerator in integers, then one division
    const unsigned __int128 n = CROP_ELEMENTS;
    const unsigned __int128 numerator = n * r.sumOfSquares - static_cast<unsigned __int128>(r.sum) * r.sum;
    r.variance = static_cast<double>(static_cast<long double>(numerator) / (static_cast<long double>(n) * static_cast<long double>(n)));
    r.mean = static_cast<double>(static_cast<long double>(r.sum) / static_cast<long double>(n));
    r.stdDev = std::sqrt(r.variance);
    return r;
}

double relativeDifference(const double value, const double reference) {
    if (!std::isfinite(value)) return std::numeric_limits<double>::infinity();
    return reference == 0 ? std::abs(value) : std::abs(value - reference) / std::abs(reference);
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

// One row: its call (the whole frame), the fill of the buffers that hold what it leaves with 0xFF bytes (and, in the rows that upload it,
// of the table of positions with zeros), and the read back of the statistics that it left for every crop
struct Row {
    std::string version;
    int calls;   // library calls of one call of the row
    std::function<void()> call;
    std::function<void()> fill;
    std::function<std::vector<Statistics>()> download;
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
    std::cout << "E20 batched reductions: "
              << (profile ? "profile mode (a warm-up call, then E20_PROFILE_CALLS calls per row, each in its NVTX range; nothing compared, "
                            "nothing timed)"
                          : (timing ? "checks, then the timing" : "checks only (E20_NO_TIMING=1)"))
              << std::endl;
    const std::vector<unsigned char> frame = seededMatrix(SRC_W, SRC_H);
    // The detections of the frame, the input of every call: every call reads the 30 positions from this table and builds every ROI, crop
    // pointer and crop view from them, as it would for a new layout on every frame
    const std::vector<Detection> detections = detectionsOfTheFrame();
    // The probe of the checks and the host references of both frames, computed before the first GPU call
    const std::vector<unsigned char> probe = profile ? std::vector<unsigned char>() : probeOf(frame, detections);
    std::vector<Reference> benchmarkReferences, probeReferences;
    if (!profile) {
        for (const Detection& detection : detections) {
            benchmarkReferences.push_back(referenceOf(frame, detection));
            probeReferences.push_back(referenceOf(probe, detection));
        }
    }
    // The upload of a frame into the copy of every library: the checks upload the probe and then the frame that is timed
    std::vector<std::function<void(const std::vector<unsigned char>&)>> uploads;
    cudaStream_t cs;
    FKLB_CUDA_CHECK(cudaStreamCreate(&cs));
    const unsigned long long spinNs = 1000ull * spinUs();
    const auto spin = [&] { gpuSpin<<<1, 1, 0, cs>>>(spinNs); };
    std::vector<Row> rows;

    // ---- FKL
    // Set once: FKL's stream object of the program's stream, its own copy of the frame, uploaded before the calls, and its output tensor:
    // 5 x 1 x 30 doubles, the five values of crop i in plane i
    fk::Stream_<fk::ParArch::GPU_NVIDIA> fkStream(cs);
    fk::Ptr2D<uchar> fkFrame(SRC_W, SRC_H);
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(fkFrame.ptr().data, fkFrame.ptr().dims.pitch, f.data(), SRC_W, SRC_W, SRC_H, cudaMemcpyHostToDevice));
    });
    uploads.back()(frame);
    fk::Tensor<double> fkOut(STATISTICS, 1, CROPS, 1);
    if (fkOut.ptr().dims.plane_pitch != STATISTICS * sizeof(double)) throw std::runtime_error("fkl: the output tensor is not contiguous");
    using FklSame = fk::Cast<uchar, uchar>;
    using FklMin = fk::Min<uchar, uchar, uchar, fk::UnaryType>;
    using FklMax = fk::Max<uchar, uchar, uchar, fk::UnaryType>;
    using FklAdd64 = fk::Add<ulonglong, ulonglong, ulonglong, fk::UnaryType>;
    using FklAdd32 = fk::Add<uint, uint, uint, fk::UnaryType>;
    using FklAddDouble = fk::Add<double, double, double, fk::UnaryType>;
    const auto fklReductions = [] {
        return fk::make_tuple(
            fk::make_tuple(FklSame::build(), FklMin::build(), fk::Cast<uchar, double>::build()),
            fk::make_tuple(FklSame::build(), FklMax::build(), fk::Cast<uchar, double>::build()),
            fk::make_tuple(fk::Cast<uchar, ulonglong>::build(), FklAdd64::build(), fk::Cast<ulonglong, double>::build()),
            fk::make_tuple(e20::IsNonZero::build(), FklAdd32::build(), fk::Cast<uint, double>::build()),
            fk::make_tuple(e20::Square::build(), FklAdd64::build(), fk::Cast<ulonglong, double>::build()));
    };
    // The reductions of fkl_batch_double_sums: the sum and the sum of squares accumulated in double, the three others as above
    const auto fklReductionsDoubleSums = [] {
        return fk::make_tuple(
            fk::make_tuple(FklSame::build(), FklMin::build(), fk::Cast<uchar, double>::build()),
            fk::make_tuple(FklSame::build(), FklMax::build(), fk::Cast<uchar, double>::build()),
            fk::make_tuple(fk::Cast<uchar, double>::build(), FklAddDouble::build(), fk::Cast<double, double>::build()),
            fk::make_tuple(e20::IsNonZero::build(), FklAdd32::build(), fk::Cast<uint, double>::build()),
            fk::make_tuple(e20::SquareAsDouble::build(), FklAddDouble::build(), fk::Cast<double, double>::build()));
    };
    const auto cropViews = [&] {
        std::array<fk::Ptr2D<uchar>, CROPS> crops;
        for (int i = 0; i < CROPS; ++i) {
            crops[i] = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
        }
        return crops;
    };
    // Set once: the workspace of the kernels, device memory of the caller, zeroed: the larger of the two that the batch of 30 crops needs,
    // with the reductions of fkl_batch and with those of fkl_batch_double_sums, which the calls of fkl_loop, of one crop each, also use
    // (the calls of a stream run one after the other). It depends on the size and the number of the crops and on the reductions, and on
    // nothing of a frame.
    const size_t fkWorkspaceBytes =
        std::max(fk::ReduceDPP<>::workspaceBytes(fk::PerThreadRead<fk::ND::_2D, uchar>::build(cropViews()), fklReductions()),
                 fk::ReduceDPP<>::workspaceBytes(fk::PerThreadRead<fk::ND::_2D, uchar>::build(cropViews()), fklReductionsDoubleSums()));
    fk::Ptr1D<uchar> fkWorkspace(static_cast<uint>(fkWorkspaceBytes));
    if (fkWorkspaceBytes > 0) {
        FKLB_CUDA_CHECK(cudaMemset(fkWorkspace.ptr().data, 0, fkWorkspaceBytes));
    }
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFklBatch = [&] {
        // The batch: the 30 crop views of the frame, from the detections (filling the array is the batch's build, part of the call)
        const std::array<fk::Ptr2D<uchar>, CROPS> crops = cropViews();
        const auto read = fk::PerThreadRead<fk::ND::_2D, uchar>::build(crops);
        const auto write = fk::TensorWrite<double>::build(fkOut.ptr());
        fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), read, fklReductions(), write);
    };
    // (end of the timed call)
    const auto fillFkl = [&] { FKLB_CUDA_CHECK(cudaMemset(fkOut.ptr().data, 0xFF, CROPS * STATISTICS * sizeof(double))); };
    const auto downloadFkl = [&] {
        const std::vector<double> out = fromDevice<double>(fkOut.ptr().data, CROPS * STATISTICS);
        std::vector<Statistics> s(CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].minimum = out[i * STATISTICS];
            s[i].maximum = out[i * STATISTICS + 1];
            s[i].sum = out[i * STATISTICS + 2];
            s[i].nonZero = out[i * STATISTICS + 3];
            s[i].sumOfSquares = out[i * STATISTICS + 4];
        }
        return s;
    };
    rows.push_back({"fkl_batch", 1, callFklBatch, fillFkl, downloadFkl});

    // fkl_batch with its two sums accumulated in double: the same read, the same write into the same tensor, the same workspace
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFklBatchDoubleSums = [&] {
        const std::array<fk::Ptr2D<uchar>, CROPS> crops = cropViews();
        const auto read = fk::PerThreadRead<fk::ND::_2D, uchar>::build(crops);
        const auto write = fk::TensorWrite<double>::build(fkOut.ptr());
        fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), read, fklReductionsDoubleSums(), write);
    };
    // (end of the timed call)
    rows.push_back({"fkl_batch_double_sums", 1, callFklBatchDoubleSums, fillFkl, downloadFkl});

    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callFklLoop = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const fk::Ptr2D<uchar> crop = fkFrame.crop2D(fk::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            const auto read = fk::PerThreadRead<fk::ND::_2D, uchar>::build(crop);
            const fk::Ptr<fk::ND::_3D, double> plane = fkOut.crop(fk::Point{0, 0, i}, fk::PtrDims<fk::ND::_3D>(STATISTICS, 1, 1));
            const auto write = fk::PerThreadWrite<fk::ND::_3D, double>::build(plane);
            fk::executeOperations<fk::ReduceDPP<>>(fkStream, fk::ReduceDPP<>::build_details(fkWorkspace), read, fklReductions(), write);
        }
    };
    // (end of the timed call)
    rows.push_back({"fkl_loop", CROPS, callFklLoop, fillFkl, downloadFkl});

    // ---- CUB and Thrust: the single pass written by hand, once per crop (cub, thrust, thrust_reduce_into) and once for the 30 crops
    // (cub_segmented, thrust_reduce_by_key)
    // Set once: their contiguous copy of the frame (no row padding), uploaded before the calls
    unsigned char* flat = deviceAlloc<unsigned char>(frame.size());
    uploads.push_back([&](const std::vector<unsigned char>& f) { FKLB_CUDA_CHECK(cudaMemcpy(flat, f.data(), f.size(), cudaMemcpyHostToDevice)); });
    uploads.back()(frame);
    const auto cropRange = [&](const int i) {
        return thrust::make_transform_iterator(thrust::make_counting_iterator(0ll), FiveOfCropElement{flat, detections[i].x, detections[i].y});
    };
    // Set once: CUB's results in device memory, and its temporary storage (the first call, with no storage, gives its size: it depends on
    // the number of elements, which every crop shares)
    Five* cubOut = deviceAlloc<Five>(CROPS);
    size_t cubStorageBytes = 0;
    FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(nullptr, cubStorageBytes, cropRange(0), cubOut, CROP_ELEMENTS, FiveOfTwo{}, FIVE_OF_NOTHING, cs));
    void* cubStorage = deviceAlloc<unsigned char>(cubStorageBytes > 0 ? cubStorageBytes : 1);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callCub = [&] {
        for (int i = 0; i < CROPS; ++i) {
            size_t bytes = cubStorageBytes;
            FKLB_CUDA_CHECK(cub::DeviceReduce::Reduce(cubStorage, bytes, cropRange(i), cubOut + i, CROP_ELEMENTS, FiveOfTwo{}, FIVE_OF_NOTHING, cs));
        }
    };
    // (end of the timed call)
    const auto statisticsOfFives = [](const std::vector<Five>& fives) {
        std::vector<Statistics> s(fives.size());
        for (size_t i = 0; i < fives.size(); ++i) {
            s[i].minimum = fives[i].minimum;
            s[i].maximum = fives[i].maximum;
            s[i].nonZero = fives[i].nonZero;
            s[i].sum = static_cast<double>(fives[i].sum);
            s[i].sumOfSquares = static_cast<double>(fives[i].sumOfSquares);
        }
        return s;
    };
    const auto fillCub = [&] { FKLB_CUDA_CHECK(cudaMemset(cubOut, 0xFF, CROPS * sizeof(Five))); };
    const auto downloadCub = [&] { return statisticsOfFives(fromDevice<Five>(cubOut, CROPS)); };
    rows.push_back({"cub", CROPS, callCub, fillCub, downloadCub});

    // CUB's batched call: the 30 crops as the 30 segments of one range
    // Set once: the table of the positions of the crops in device memory, which every call uploads from the detections, the results in
    // device memory, and the temporary storage (the first call, with no storage, gives its size)
    Detection* batchDetections = deviceAlloc<Detection>(CROPS);
    const auto batchRange = [&] {
        return thrust::make_transform_iterator(thrust::make_counting_iterator(0u), FiveOfBatchElement{flat, batchDetections});
    };
    Five* cubSegmentedOut = deviceAlloc<Five>(CROPS);
    size_t cubSegmentedBytes = 0;
    FKLB_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(nullptr, cubSegmentedBytes, batchRange(), cubSegmentedOut, static_cast<std::int64_t>(CROPS),
                                                       static_cast<int>(CROP_ELEMENTS), FiveOfTwo{}, FIVE_OF_NOTHING, cs));
    void* cubSegmentedStorage = deviceAlloc<unsigned char>(cubSegmentedBytes > 0 ? cubSegmentedBytes : 1);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callCubSegmented = [&] {
        FKLB_CUDA_CHECK(cudaMemcpyAsync(batchDetections, detections.data(), CROPS * sizeof(Detection), cudaMemcpyHostToDevice, cs));
        size_t bytes = cubSegmentedBytes;
        FKLB_CUDA_CHECK(cub::DeviceSegmentedReduce::Reduce(cubSegmentedStorage, bytes, batchRange(), cubSegmentedOut,
                                                           static_cast<std::int64_t>(CROPS), static_cast<int>(CROP_ELEMENTS), FiveOfTwo{},
                                                           FIVE_OF_NOTHING, cs));
    };
    // (end of the timed call)
    // The fill of the rows that read the table (this one and thrust_reduce_by_key) also sets it to zeros, the crop at the corner of the
    // frame 30 times: a call that does not upload it leaves the statistics of that crop, which is none of the 30
    const auto fillBatchDetections = [&] { FKLB_CUDA_CHECK(cudaMemset(batchDetections, 0, CROPS * sizeof(Detection))); };
    const auto fillCubSegmented = [&] {
        FKLB_CUDA_CHECK(cudaMemset(cubSegmentedOut, 0xFF, CROPS * sizeof(Five)));
        fillBatchDetections();
    };
    const auto downloadCubSegmented = [&] { return statisticsOfFives(fromDevice<Five>(cubSegmentedOut, CROPS)); };
    rows.push_back({"cub_segmented", 1, callCubSegmented, fillCubSegmented, downloadCubSegmented});

    // Thrust returns its results to the host: the row keeps them there
    std::vector<Five> thrustOut(CROPS);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callThrust = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const thrust::counting_iterator<long long> first(0ll);
            thrustOut[i] = thrust::transform_reduce(thrust::cuda::par.on(cs), first, first + CROP_ELEMENTS,
                                                    FiveOfCropElement{flat, detections[i].x, detections[i].y}, FIVE_OF_NOTHING, FiveOfTwo{});
        }
    };
    // (end of the timed call)
    const auto fillThrust = [&] { std::memset(thrustOut.data(), 0xFF, CROPS * sizeof(Five)); };
    const auto downloadThrust = [&] { return statisticsOfFives(thrustOut); };
    rows.push_back({"thrust", CROPS, callThrust, fillThrust, downloadThrust});

    // Thrust with its results in device memory: thrust::reduce_into, par_nosync, a block of the caller as its temporary storage
    // Set once: the results, element i for crop i, and the block of the two rows that follow. Thrust has no call that gives the size of
    // what it will take: 1 MiB (THRUST_BLOCK_BYTES) is more than the 92,416 bytes that it takes in a call of thrust::reduce_into and the
    // 177,408 bytes of a call of thrust::reduce_by_key, and BlockOfTheCaller stops the program if Thrust asks for more than the block has
    Five* thrustIntoOut = deviceAlloc<Five>(CROPS);
    char* thrustBlock = deviceAlloc<char>(THRUST_BLOCK_BYTES);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callThrustInto = [&] {
        for (int i = 0; i < CROPS; ++i) {
            BlockOfTheCaller storage{thrustBlock, THRUST_BLOCK_BYTES};
            const auto first = cropRange(i);
            thrust::reduce_into(thrust::cuda::par_nosync(storage).on(cs), first, first + CROP_ELEMENTS,
                                thrust::device_pointer_cast(thrustIntoOut + i), FIVE_OF_NOTHING, FiveOfTwo{});
        }
    };
    // (end of the timed call)
    const auto fillThrustInto = [&] { FKLB_CUDA_CHECK(cudaMemset(thrustIntoOut, 0xFF, CROPS * sizeof(Five))); };
    const auto downloadThrustInto = [&] { return statisticsOfFives(fromDevice<Five>(thrustIntoOut, CROPS)); };
    rows.push_back({"thrust_reduce_into", CROPS, callThrustInto, fillThrustInto, downloadThrustInto});

    // Thrust's batched call: thrust::reduce_by_key over the range of cub_segmented, the crop of an element as its key
    // Set once: the results in device memory, element i for crop i
    Five* thrustByKeyOut = deviceAlloc<Five>(CROPS);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callThrustByKey = [&] {
        FKLB_CUDA_CHECK(cudaMemcpyAsync(batchDetections, detections.data(), CROPS * sizeof(Detection), cudaMemcpyHostToDevice, cs));
        BlockOfTheCaller storage{thrustBlock, THRUST_BLOCK_BYTES};
        const auto keys = thrust::make_transform_iterator(thrust::make_counting_iterator(0u), CropOfBatchElement{});
        thrust::reduce_by_key(thrust::cuda::par_nosync(storage).on(cs), keys, keys + CROPS * CROP_ELEMENTS, batchRange(),
                              thrust::make_discard_iterator(), thrust::device_pointer_cast(thrustByKeyOut), thrust::equal_to<int>(), FiveOfTwo{});
    };
    // (end of the timed call)
    const auto fillThrustByKey = [&] {
        FKLB_CUDA_CHECK(cudaMemset(thrustByKeyOut, 0xFF, CROPS * sizeof(Five)));
        fillBatchDetections();
    };
    const auto downloadThrustByKey = [&] { return statisticsOfFives(fromDevice<Five>(thrustByKeyOut, CROPS)); };
    rows.push_back({"thrust_reduce_by_key", 1, callThrustByKey, fillThrustByKey, downloadThrustByKey});

    // ---- OpenCV-CUDA
    // Set once: the cv::cuda::Stream of the program's stream, and OpenCV's own copy of the frame, uploaded before the calls
    // OpenCV's buffer pool, turned on before the stream is wrapped (a cv::cuda::Stream takes its allocator when it is made): cv::cuda::meanStdDev
    // then takes NPP's scratch buffer from the pool of the stream. With the pool off (OpenCV's default) every cv::cuda::meanStdDev allocates the
    // buffer and frees it when it returns, and cudaFree waits for the work of the stream
    cv::cuda::setBufferPoolUsage(true);
    cv::cuda::Stream cvs = cv::cuda::StreamAccessor::wrapStream(cs);
    cv::cuda::GpuMat cvFrame;
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        cvFrame.upload(cv::Mat(SRC_H, SRC_W, CV_8UC1, const_cast<unsigned char*>(f.data())));
    });
    uploads.back()(frame);
    // Set once: the outputs, one row per crop, each row with the size and the type that its call makes (getOutputMat keeps a GpuMat that
    // already has them). Continuous, so that the rows of the 30 crops are consecutive in device memory
    cv::cuda::GpuMat cvMinMax, cvSum, cvNonZero, cvSqrSum, cvMeanStdDev;
    cv::cuda::createContinuous(CROPS, 2, CV_32SC1, cvMinMax);
    cv::cuda::createContinuous(CROPS, 1, CV_64FC1, cvSum);
    cv::cuda::createContinuous(CROPS, 1, CV_32SC1, cvNonZero);
    cv::cuda::createContinuous(CROPS, 1, CV_64FC1, cvSqrSum);
    cv::cuda::createContinuous(CROPS, 2, CV_64FC1, cvMeanStdDev);
    for (const cv::cuda::GpuMat* m : {&cvMinMax, &cvSum, &cvNonZero, &cvSqrSum, &cvMeanStdDev}) {
        if (!m->isContinuous()) throw std::runtime_error("opencv_cuda: an output is not continuous");
    }
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callOpenCV = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            cv::cuda::GpuMat minMax = cvMinMax.row(i), sum = cvSum.row(i), nonZero = cvNonZero.row(i), sqrSum = cvSqrSum.row(i);
            cv::cuda::findMinMax(roi, minMax, cv::noArray(), cvs);
            cv::cuda::calcSum(roi, sum, cv::noArray(), cvs);
            cv::cuda::countNonZero(roi, nonZero, cvs);
            cv::cuda::calcSqrSum(roi, sqrSum, cv::noArray(), cvs);
        }
    };
    // (end of the timed call)
    const auto fillOpenCVShared = [&] {
        FKLB_CUDA_CHECK(cudaMemset(cvMinMax.data, 0xFF, CROPS * 2 * sizeof(int)));
        FKLB_CUDA_CHECK(cudaMemset(cvNonZero.data, 0xFF, CROPS * sizeof(int)));
    };
    const auto fillOpenCV = [&] {
        fillOpenCVShared();
        FKLB_CUDA_CHECK(cudaMemset(cvSum.data, 0xFF, CROPS * sizeof(double)));
        FKLB_CUDA_CHECK(cudaMemset(cvSqrSum.data, 0xFF, CROPS * sizeof(double)));
    };
    const auto downloadOpenCVShared = [&] {
        const std::vector<int> minMax = fromDevice<int>(cvMinMax.data, CROPS * 2), nonZero = fromDevice<int>(cvNonZero.data, CROPS);
        std::vector<Statistics> s(CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].minimum = minMax[2 * i];
            s[i].maximum = minMax[2 * i + 1];
            s[i].nonZero = nonZero[i];
        }
        return s;
    };
    const auto downloadOpenCV = [&] {
        std::vector<Statistics> s = downloadOpenCVShared();
        const std::vector<double> sum = fromDevice<double>(cvSum.data, CROPS), sqrSum = fromDevice<double>(cvSqrSum.data, CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].sum = sum[i];
            s[i].sumOfSquares = sqrSum[i];
        }
        return s;
    };
    rows.push_back({"opencv_cuda", 4 * CROPS, callOpenCV, fillOpenCV, downloadOpenCV});

    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callOpenCVMsd = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const cv::cuda::GpuMat roi = cvFrame(cv::Rect(detections[i].x, detections[i].y, CROP_W, CROP_H));
            cv::cuda::GpuMat minMax = cvMinMax.row(i), nonZero = cvNonZero.row(i), meanStdDev = cvMeanStdDev.row(i);
            cv::cuda::findMinMax(roi, minMax, cv::noArray(), cvs);
            cv::cuda::countNonZero(roi, nonZero, cvs);
            cv::cuda::meanStdDev(roi, meanStdDev, cvs);
        }
    };
    // (end of the timed call)
    const auto fillOpenCVMsd = [&] {
        fillOpenCVShared();
        FKLB_CUDA_CHECK(cudaMemset(cvMeanStdDev.data, 0xFF, CROPS * 2 * sizeof(double)));
    };
    const auto downloadOpenCVMsd = [&] {
        std::vector<Statistics> s = downloadOpenCVShared();
        const std::vector<double> meanStdDev = fromDevice<double>(cvMeanStdDev.data, CROPS * 2);
        for (int i = 0; i < CROPS; ++i) {
            s[i].mean = meanStdDev[2 * i];
            s[i].stdDev = meanStdDev[2 * i + 1];
        }
        return s;
    };
    rows.push_back({"opencv_cuda_msd", 3 * CROPS, callOpenCVMsd, fillOpenCVMsd, downloadOpenCVMsd, true});

    // ---- NPP
    // Set once: the NPP stream context, filled field by field from the device properties and the stream's flags as in E18 and E19
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
    // Set once: NPP's own copy of the frame (cudaMallocPitch), uploaded before the calls
    Npp8u* nppFrame = nullptr;
    size_t frameStep = 0;
    FKLB_CUDA_CHECK(cudaMallocPitch(reinterpret_cast<void**>(&nppFrame), &frameStep, SRC_W, SRC_H));
    uploads.push_back([&](const std::vector<unsigned char>& f) {
        FKLB_CUDA_CHECK(cudaMemcpy2D(nppFrame, frameStep, f.data(), SRC_W, SRC_W, SRC_H, cudaMemcpyHostToDevice));
    });
    uploads.back()(frame);
    const NppiSize nppCropSize{CROP_W, CROP_H};
    // Set once: the results in device memory, element i for crop i, and the scratch buffer of every function (every crop has the same size)
    Npp8u* nppMin = deviceAlloc<Npp8u>(CROPS);
    Npp8u* nppMax = deviceAlloc<Npp8u>(CROPS);
    Npp64f* nppSum = deviceAlloc<Npp64f>(CROPS);
    int* nppCount = deviceAlloc<int>(CROPS);
    Npp64f* nppDotProd = deviceAlloc<Npp64f>(CROPS);
    Npp64f* nppMean = deviceAlloc<Npp64f>(CROPS);
    Npp64f* nppStdDev = deviceAlloc<Npp64f>(CROPS);
    size_t minMaxBytes = 0, sumBytes = 0, countBytes = 0, dotProdBytes = 0, meanStdDevBytes = 0;
    FKLB_NPP_CHECK(nppiMinMaxGetBufferHostSize_8u_C1R_Ctx(nppCropSize, &minMaxBytes, ctx));
    FKLB_NPP_CHECK(nppiSumGetBufferHostSize_8u_C1R_Ctx(nppCropSize, &sumBytes, ctx));
    FKLB_NPP_CHECK(nppiCountInRangeGetBufferHostSize_8u_C1R_Ctx(nppCropSize, &countBytes, ctx));
    FKLB_NPP_CHECK(nppiDotProdGetBufferHostSize_8u64f_C1R_Ctx(nppCropSize, &dotProdBytes, ctx));
    FKLB_NPP_CHECK(nppiMeanStdDevGetBufferHostSize_8u_C1R_Ctx(nppCropSize, &meanStdDevBytes, ctx));
    Npp8u* minMaxBuffer = deviceAlloc<Npp8u>(minMaxBytes > 0 ? minMaxBytes : 1);
    Npp8u* sumBuffer = deviceAlloc<Npp8u>(sumBytes > 0 ? sumBytes : 1);
    Npp8u* countBuffer = deviceAlloc<Npp8u>(countBytes > 0 ? countBytes : 1);
    Npp8u* dotProdBuffer = deviceAlloc<Npp8u>(dotProdBytes > 0 ? dotProdBytes : 1);
    Npp8u* meanStdDevBuffer = deviceAlloc<Npp8u>(meanStdDevBytes > 0 ? meanStdDevBytes : 1);
    const int step = static_cast<int>(frameStep);
    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callNpp = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const Npp8u* crop = nppFrame + static_cast<size_t>(detections[i].y) * frameStep + detections[i].x;
            FKLB_NPP_CHECK(nppiMinMax_8u_C1R_Ctx(crop, step, nppCropSize, nppMin + i, nppMax + i, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiSum_8u_C1R_Ctx(crop, step, nppCropSize, sumBuffer, nppSum + i, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_8u_C1R_Ctx(crop, step, nppCropSize, nppCount + i, 1, 255, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiDotProd_8u64f_C1R_Ctx(crop, step, crop, step, nppCropSize, nppDotProd + i, dotProdBuffer, ctx));
        }
    };
    // (end of the timed call)
    const auto fillNppShared = [&] {
        FKLB_CUDA_CHECK(cudaMemset(nppMin, 0xFF, CROPS * sizeof(Npp8u)));
        FKLB_CUDA_CHECK(cudaMemset(nppMax, 0xFF, CROPS * sizeof(Npp8u)));
        FKLB_CUDA_CHECK(cudaMemset(nppCount, 0xFF, CROPS * sizeof(int)));
    };
    const auto fillNpp = [&] {
        fillNppShared();
        FKLB_CUDA_CHECK(cudaMemset(nppSum, 0xFF, CROPS * sizeof(Npp64f)));
        FKLB_CUDA_CHECK(cudaMemset(nppDotProd, 0xFF, CROPS * sizeof(Npp64f)));
    };
    const auto downloadNppShared = [&] {
        const std::vector<Npp8u> minimum = fromDevice<Npp8u>(nppMin, CROPS), maximum = fromDevice<Npp8u>(nppMax, CROPS);
        const std::vector<int> count = fromDevice<int>(nppCount, CROPS);
        std::vector<Statistics> s(CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].minimum = minimum[i];
            s[i].maximum = maximum[i];
            s[i].nonZero = count[i];
        }
        return s;
    };
    const auto downloadNpp = [&] {
        std::vector<Statistics> s = downloadNppShared();
        const std::vector<Npp64f> sum = fromDevice<Npp64f>(nppSum, CROPS), dotProd = fromDevice<Npp64f>(nppDotProd, CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].sum = sum[i];
            s[i].sumOfSquares = dotProd[i];
        }
        return s;
    };
    rows.push_back({"npp", 4 * CROPS, callNpp, fillNpp, downloadNpp});

    // The timed call (whole_call and kernel_only): the whole frame. Everything in this lambda is inside the interval
    const auto callNppMsd = [&] {
        for (int i = 0; i < CROPS; ++i) {
            const Npp8u* crop = nppFrame + static_cast<size_t>(detections[i].y) * frameStep + detections[i].x;
            FKLB_NPP_CHECK(nppiMinMax_8u_C1R_Ctx(crop, step, nppCropSize, nppMin + i, nppMax + i, minMaxBuffer, ctx));
            FKLB_NPP_CHECK(nppiCountInRange_8u_C1R_Ctx(crop, step, nppCropSize, nppCount + i, 1, 255, countBuffer, ctx));
            FKLB_NPP_CHECK(nppiMean_StdDev_8u_C1R_Ctx(crop, step, nppCropSize, meanStdDevBuffer, nppMean + i, nppStdDev + i, ctx));
        }
    };
    // (end of the timed call)
    const auto fillNppMsd = [&] {
        fillNppShared();
        FKLB_CUDA_CHECK(cudaMemset(nppMean, 0xFF, CROPS * sizeof(Npp64f)));
        FKLB_CUDA_CHECK(cudaMemset(nppStdDev, 0xFF, CROPS * sizeof(Npp64f)));
    };
    const auto downloadNppMsd = [&] {
        std::vector<Statistics> s = downloadNppShared();
        const std::vector<Npp64f> mean = fromDevice<Npp64f>(nppMean, CROPS), stdDev = fromDevice<Npp64f>(nppStdDev, CROPS);
        for (int i = 0; i < CROPS; ++i) {
            s[i].mean = mean[i];
            s[i].stdDev = stdDev[i];
        }
        return s;
    };
    rows.push_back({"npp_msd", 3 * CROPS, callNppMsd, fillNppMsd, downloadNppMsd, true});

    if (rows.size() != ROWS.size()) throw std::runtime_error("the rows are not the ones of ROWS");
    for (size_t k = 0; k < rows.size(); ++k) {
        if (rows[k].version != ROWS[k]) throw std::runtime_error("the rows are not the ones of ROWS");
    }

    // ---- The profile mode: a warm-up call outside any range, then E20_PROFILE_CALLS calls, each in the range "<row>:call" inside "<row>"
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

    // ---- The checks, on the probe and then on the frame that is timed (which stays in the copies of the libraries): one call of every
    // selected row, after the buffers of the statistics of every row are filled with 0xFF bytes (and the table of positions of the batched
    // calls of CUB and Thrust with zeros)
    bool allOk = true;
    if (!profile) {
        fklb::CsvWriter checks("e20_batched_reductions_checks.csv", "E20 batched reductions: the checks (30 crops of 256 x 256 of one frame)",
                               {"frame", "version", "statistic", "compared_with", "crops", "differing_crops", "max_relative_difference", "criterion",
                                "ok"});
        const double n = static_cast<double>(CROP_ELEMENTS);
        for (const char* frameName : {"probe", "benchmark"}) {
            const bool isProbe = std::string(frameName) == "probe";
            const std::vector<Reference>& references = isProbe ? probeReferences : benchmarkReferences;
            for (const auto& upload : uploads) upload(isProbe ? probe : frame);
            for (const Row& row : rows) {
                if (!selected(row.version)) continue;
                for (const Row& any : rows) any.fill();
                FKLB_CUDA_CHECK(cudaDeviceSynchronize());
                row.call();
                FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
                const std::vector<Statistics> s = row.download();
                // One line per statistic: value(i) of the row against expected(i) of the host, over the 30 crops
                const auto check = [&](const char* statistic, const char* comparedWith, const std::function<double(int)>& value,
                                       const std::function<double(int)>& expected, const double tolerance, const char* criterion) {
                    int differing = 0;
                    double largest = 0;
                    for (int i = 0; i < CROPS; ++i) {
                        const double difference = relativeDifference(value(i), expected(i));
                        differing += difference <= tolerance ? 0 : 1;
                        largest = difference > largest ? difference : largest;
                    }
                    const bool ok = differing == 0;
                    checks.row(frameName, row.version, statistic, comparedWith, CROPS, differing, largest, criterion, ok ? "yes" : "no");
                    allOk = allOk && ok;
                    std::cout << frameName << " " << row.version << " " << statistic << " against " << comparedWith << ": " << differing << " of "
                              << CROPS << " crops differ, largest relative difference " << largest << "; " << criterion
                              << (ok ? ": ok" : ": FAILED") << std::endl;
                };
                check("minimum", "host", [&](int i) { return s[i].minimum; }, [&](int i) { return static_cast<double>(references[i].minimum); }, 0,
                      "equal");
                check("maximum", "host", [&](int i) { return s[i].maximum; }, [&](int i) { return static_cast<double>(references[i].maximum); }, 0,
                      "equal");
                check("non_zero", "host", [&](int i) { return s[i].nonZero; }, [&](int i) { return static_cast<double>(references[i].nonZero); }, 0,
                      "equal");
                std::function<double(int)> variance;
                if (!row.leavesMeanAndStdDev) {
                    check("sum", "host", [&](int i) { return s[i].sum; }, [&](int i) { return static_cast<double>(references[i].sum); }, 0, "equal");
                    check("sum_of_squares", "host", [&](int i) { return s[i].sumOfSquares; },
                          [&](int i) { return static_cast<double>(references[i].sumOfSquares); }, 0, "equal");
                    variance = [&](int i) { return s[i].sumOfSquares / n - (s[i].sum / n) * (s[i].sum / n); };
                } else {
                    check("mean", "host", [&](int i) { return s[i].mean; }, [&](int i) { return references[i].mean; }, RELATIVE_TOLERANCE,
                          "relative difference at most 1e-9");
                    check("std_dev", "host", [&](int i) { return s[i].stdDev; }, [&](int i) { return references[i].stdDev; }, RELATIVE_TOLERANCE,
                          "relative difference at most 1e-9");
                    variance = [&](int i) { return s[i].stdDev * s[i].stdDev; };
                }
                check("variance", "host; computed on the host from what the row left", variance, [&](int i) { return references[i].variance; },
                      RELATIVE_TOLERANCE, "relative difference at most 1e-9");
            }
        }
        std::cout << (allOk ? "E20 batched reductions checks passed" : "E20 batched reductions CHECK FAILED") << std::endl;
    }

    // ---- The timing, after the checks, in the order of the rows: two intervals over the whole frame (the 30 crops)
    if (timing && allOk) {
        fklb::CsvWriter t("e20_batched_reductions.csv", "E20 batched reductions: whole-call and kernel-only times of one frame (30 crops)",
                          {"version", "library_calls", "interval", "mean_ms", "min_ms", "iterations", "spin_us"});
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
            t.row(row.version, row.calls, "whole_call", whole.mean, whole.min, fklb::ITERATIONS, "");
            t.row(row.version, row.calls, "kernel_only", gpu.mean, gpu.min, fklb::ITERATIONS, spinUs());
            std::cout << row.version << ": whole call mean " << whole.mean << " ms (min " << whole.min << "); kernel only mean " << gpu.mean
                      << " ms (min " << gpu.min << ")" << std::endl;
        }
    } else if (timing) {
        std::cout << "not timed: a check failed" << std::endl;
    }
    FKLB_CUDA_CHECK(cudaStreamSynchronize(cs));
    for (void* p : std::vector<void*>{flat, cubOut, cubStorage, batchDetections, cubSegmentedOut, cubSegmentedStorage, thrustIntoOut,
                                      thrustBlock, thrustByKeyOut, nppFrame, nppMin, nppMax, nppSum, nppCount, nppDotProd, nppMean, nppStdDev,
                                      minMaxBuffer, sumBuffer, countBuffer, dotProdBuffer, meanStdDevBuffer}) {
        FKLB_CUDA_CHECK(cudaFree(p));
    }
    // The wrapper of the stream gives its part of the pool back, and synchronises the stream to do it: before the stream is destroyed
    cvs = cv::cuda::Stream::Null();
    FKLB_CUDA_CHECK(cudaStreamDestroy(cs));
    return allOk ? 0 : 1;
}
