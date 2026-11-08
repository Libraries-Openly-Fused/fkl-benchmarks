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

// FKL on HIP against AMD's RPP (Radeon Performance Primitives, the rppt tensor API with the HIP back-end) and against a hand-written fused kernel
// on the pre-processing pipeline (a) of the paper, on an AMD GPU. RPP is the counterpart of NPP.
//
// Pipeline (a): N crops of 60x120 at (i, i) of a 3840x2160 8 bit RGB frame, bilinear resize to 64x128, RGB to BGR, multiply by 0.3, subtract
// {1, 4, 3.2}, divide by {3.2, 0.6, 11.8}, an NCHW float tensor. N = 10 to 300 step 10.
// Every side owns its buffers and its stream, and gets the same host bytes of the frame: FKL a fk::Ptr2D (device memory with a pinned copy) and
// a fk::Stream, RPP a dense hipMalloc buffer and a hipStream_t, the hand-written kernel a hipMallocPitch image (fk::Ptr2D allocates its device
// memory with hipMallocPitch too), a hipMalloc output and a hipStream_t. No FKL object is used on the RPP side or by the hand-written kernel.
//
// Variants. Wall clock on the host, including the synchronisation of the stream (common/hip_wallclock.h): E5_WARMUP (3) warm-up runs, then a
// warm-up of at least E5_WARMUP_SECONDS (2.0 s; 0 to 10, the warm-up ends after 10 s at the latest) that goes on until the mean time of two
// consecutive windows of 0.25 s differs by at most 2 percent, then E5_ITERS (30) timed runs; median, min, max, mean.
//   fkl_fused               FKL, one executeOperations: a BatchRead of the crop views .then(Resize) -> VectorReorder (RGB to BGR) -> Mul -> Sub -> Div
//                           -> TensorSplit, thread fusion off. It is the form of opencv-cuda/batched-preprocessing/e4_preprocessing.cu, so that the
//                           two GPUs run the same FKL code. 1 call
//   fkl_fused_halfpixel     the same with ResizeSampling::HALF_PIXEL, the sampling of cv::resize, NPP and RPP. 1 call
//   fkl_perop_batch         FKL, one executeOperations per operation for all the crops. 6 calls
//   rpp_crop_batch          RPP's documented batched use: the N crops are copied into one dense batch, one hipMemcpy2DAsync per crop, then
//                           rppt_resize_mirror_normalize and rppt_channel_permute run on that batch. N + 2 calls
//   rpp_crop_batch_unsplit  the same calls without the synchronisation between the copies and the RPP calls: the time a user pays. N + 2 calls
//   hand                    the hand-written fused kernel of E1c (handwritten-cuda/e1_preproc.cu: handPreproc, the version hand_batch of its CSV),
//                           ported to HIP (e1c:: below): one kernel for the whole batch, ASYMMETRIC sampling as fkl_fused, the crop pointers in its
//                           parameter block (8 bytes per crop), the crop size, the output size and the constants known at compile time, 32 x 8
//                           threads per block (the block FKL launches for a 64x128 output: checked at compile time). 1 call
// The parts of a run, as the medians build_median_ms and compute_median_ms:
//   FKL: build = the crop views and every operation (on the host); compute = executeOperations and the synchronisation, with no synchronisation
//        between them. The release of the crop views at the end of the call is part of compute.
//   hand: build = the crop pointers, the kernel's parameter block (on the host); compute = the launch, hipGetLastError (executeOperations checks
//        its launch in the same way) and the synchronisation of its stream.
//   rpp_crop_batch: build = the copies and a stream synchronisation; compute = the ROIs, the two RPP calls and a stream synchronisation. Its
//        total therefore includes one stream synchronisation more than rpp_crop_batch_unsplit, which has no parts (NaN). In every call
//        rppt_resize_mirror_normalize itself clears its output and synchronises its stream (RPP 3.2.0), so in rpp_crop_batch_unsplit RPP waits
//        for the copies inside that call.
// RPP has no per-channel multiply: the normalize of rppt_resize_mirror_normalize covers the multiply, the subtraction and the division, with
// mean = sub / 0.3 and std = div / 0.3. Output widths are multiples of 8.
// Environment: E5_ITERS, E5_WARMUP, E5_WARMUP_SECONDS, E5_ROTATE_OFFSET (checked: common/env_args.h), E5_IMAGE (a binary PPM tiled over the
// frame instead of seeded noise), E5_IMAGE_INFO.
// CSVs in the working directory: benchmark_rpp_preprocessing_a_GPU_AMD.csv and benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv.
#include <common/csv_writer.h>
#include <common/env_args.h>
#include <common/hip_wallclock.h>
#include <common/main.h>
#include <common/ppm.h>
#include <common/resize_reference.h>
#include <common/status_check.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/algorithms/basic_ops/arithmetic.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/basic_ops/vector_ops.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/rect.h>

#include <hip/hip_runtime.h>
#include <rpp.h>
#include <rppdefs.h>
#include <rpp_version.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace rppbench {

constexpr int SRC_W = 3840, SRC_H = 2160, CROP_W = 60, CROP_H = 120, OUT_W = 64, OUT_H = 128, PLANE = OUT_W * OUT_H;
constexpr double TOLERANCE = 1e-4;
constexpr float ALPHA = 0.3f;
constexpr float SUB[3] = {1.f, 4.f, 3.2f}, DIV[3] = {3.2f, 0.6f, 11.8f};

// ---- The hand-written fused kernel: E1c's version hand_batch (handwritten-cuda/e1_preproc.cu, handPreproc) ported to HIP. From the static_assert
// to the end of handPreproc the text is E1c's with one HIP change: HIP has no __grid_constant__, and on an AMD GPU every kernel parameter is in
// the constant kernel argument segment, so the qualifier is dropped (the gfx1200 code of the kernel compiled alone is the same with clang's
// grid_constant attribute). The constants are this program's, under E1c's names; their values are E1c's.
namespace e1c {
constexpr int UP_W = OUT_W, UP_H = OUT_H;
constexpr float SUB_X = SUB[0], SUB_Y = SUB[1], SUB_Z = SUB[2];
constexpr float DIV_X = DIV[0], DIV_Y = DIV[1], DIV_Z = DIV[2];
static_assert(fk::getDefaultBlockSize(UP_W, UP_H).x == 32 && fk::getDefaultBlockSize(UP_W, UP_H).y == 8,
              "the hand-written kernels are launched with the block FKL launches for an output image");

template <int BATCH>
struct CropPointers {
    const unsigned char* data[BATCH];
};

__device__ __forceinline__ float3 loadPixel(const unsigned char* base, const size_t pitch, const int x, const int y) {
    const uchar3 p = *reinterpret_cast<const uchar3*>(base + y * pitch + x * sizeof(uchar3));
    return make_float3(static_cast<float>(p.x), static_cast<float>(p.y), static_cast<float>(p.z));
}

// resize (bilinear, ASYMMETRIC: the arithmetic of fk::InterpolateComplete) -> reorder -> * alpha -> - sub -> / div -> planar write.
// srcW and srcH are the size of the crop, imageStride the bytes between two images of the tensor, channelPixels the number of
// elements between two planes of an image (width * height), as fk::TensorSplit computes them.
__device__ __forceinline__ void preprocess(const unsigned char* base, const size_t inPitch, const int srcW, const int srcH, const float fx, const float fy,
                                           const int x, const int y, const int z, float* tensor, const size_t pitch, const size_t imageStride,
                                           const int channelPixels) {
    const float src_x = x * fx;
    const float src_y = y * fy;
    const int x1 = __float2int_rd(src_x);
    const int y1 = __float2int_rd(src_y);
    const int x2 = x1 + 1;
    const int y2 = y1 + 1;
    const int x2_read = min(x2, srcW - 1);
    const int y2_read = min(y2, srcH - 1);
    const float3 r00 = loadPixel(base, inPitch, x1, y1);
    const float3 r10 = loadPixel(base, inPitch, x2_read, y1);
    const float3 r01 = loadPixel(base, inPitch, x1, y2_read);
    const float3 r11 = loadPixel(base, inPitch, x2_read, y2_read);
    const float3 v = make_float3(
        (r00.x * ((x2 - src_x) * (y2 - src_y))) + (r10.x * ((src_x - x1) * (y2 - src_y))) + (r01.x * ((x2 - src_x) * (src_y - y1))) + (r11.x * ((src_x - x1) * (src_y - y1))),
        (r00.y * ((x2 - src_x) * (y2 - src_y))) + (r10.y * ((src_x - x1) * (y2 - src_y))) + (r01.y * ((x2 - src_x) * (src_y - y1))) + (r11.y * ((src_x - x1) * (src_y - y1))),
        (r00.z * ((x2 - src_x) * (y2 - src_y))) + (r10.z * ((src_x - x1) * (y2 - src_y))) + (r01.z * ((x2 - src_x) * (src_y - y1))) + (r11.z * ((src_x - x1) * (src_y - y1))));
    // RGB to BGR, then the three operations, each on the three channels
    const float o0 = ((v.z * ALPHA) - SUB_X) / DIV_X;
    const float o1 = ((v.y * ALPHA) - SUB_Y) / DIV_Y;
    const float o2 = ((v.x * ALPHA) - SUB_Z) / DIV_Z;
    float* const plane = reinterpret_cast<float*>(reinterpret_cast<char*>(tensor) + z * imageStride + y * pitch) + x;
    plane[0] = o0;
    plane[channelPixels] = o1;
    plane[2 * channelPixels] = o2;
}

template <int BATCH>
__global__ void handPreproc(const CropPointers<BATCH> crops, const size_t inPitch, float* __restrict__ tensor, const size_t pitch,
                            const size_t imageStride, const float fx, const float fy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int z = blockIdx.z;
    if (x >= UP_W || y >= UP_H) return;
    preprocess(crops.data[z], inPitch, CROP_W, CROP_H, fx, fy, x, y, z, tensor, pitch, imageStride, UP_W * UP_H);
}
} // namespace e1c

struct Region {
    double maxAbs{0}, meanAbs{0}, fractionOver{0};
    size_t count{0};
};

// max, mean and fraction over TOLERANCE of |a - b| on the values of an NCHW tensor of N images, in the interior of each plane or in its last two columns and rows
inline Region compareTensors(const float* a, const float* b, const int n, const int w, const int h, const bool edge) {
    Region r;
    double sum = 0;
    size_t over = 0;
    bool sawNaN = false;
    for (int i = 0; i < n * 3; ++i) {
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const bool isEdge = x >= w - 2 || y >= h - 2;
                if (isEdge != edge) {
                    continue;
                }
                const size_t k = (static_cast<size_t>(i) * h + y) * w + x;
                const double d = std::fabs(static_cast<double>(a[k]) - static_cast<double>(b[k]));
                if (std::isnan(d)) {
                    sawNaN = true;
                } else if (d > r.maxAbs) {
                    r.maxAbs = d;
                }
                sum += d;
                over += !(d <= TOLERANCE);
                ++r.count;
            }
        }
    }
    if (sawNaN) { // a NaN (also inf - inf: the same infinity in both tensors) fails every tolerance: the maximum says so (the mean is NaN as well)
        r.maxAbs = std::numeric_limits<double>::quiet_NaN();
    }
    r.meanAbs = r.count ? sum / static_cast<double>(r.count) : 0;
    r.fractionOver = r.count ? static_cast<double>(over) / static_cast<double>(r.count) : 0;
    return r;
}

// Scalar reference of pipeline (a) in double precision, from the host bytes of the frame (dense, SRC_W * 3 bytes per row), with the sampling of a resize
inline std::vector<float> referenceA(const std::vector<unsigned char>& frame, const int n, const fklb::ResizeSampling sampling) {
    std::vector<float> out(static_cast<size_t>(n) * 3 * PLANE);
    std::vector<double> resized(static_cast<size_t>(PLANE) * 3);
    for (int i = 0; i < n; ++i) {
        const unsigned char* crop = frame.data() + (static_cast<size_t>(i) * SRC_W + static_cast<size_t>(i)) * 3;
        fklb::resizeBilinear(crop, static_cast<size_t>(SRC_W) * 3, CROP_W, CROP_H, 3, OUT_W, OUT_H, sampling, resized.data());
        for (int p = 0; p < PLANE; ++p) {
            for (int c = 0; c < 3; ++c) {
                const int oc = 2 - c; // RGB -> BGR
                out[(static_cast<size_t>(i) * 3 + oc) * PLANE + p] = static_cast<float>(
                    (resized[static_cast<size_t>(p) * 3 + c] * static_cast<double>(ALPHA) - static_cast<double>(SUB[oc])) / static_cast<double>(DIV[oc]));
            }
        }
    }
    return out;
}

struct Output {
    fklb::CsvWriter* timing;
    fklb::CsvWriter* agreement;
    bool allPass{true};
    int warmup, iters;
    double warmupSeconds{2.0}; // minimum warm-up of a variant, see fklb::measureWall (steady state rule)
    int rotateOffset{0};       // E5_ROTATE_OFFSET: added to the index of the point that selects the first variant of the rotated order
};

struct Variant {
    std::string name;
    int calls;
    std::function<void(fklb::WallClock::time_point&)> run;
};

inline void timeVariant(Output& out, const int batch, const Variant& v) {
    const fklb::WallStats s = fklb::measureWall(v.run, out.warmup, out.iters, out.warmupSeconds, true);
    out.timing->row(batch, v.name, v.calls, s.median, s.min, s.max, s.mean, s.firstMedian, s.secondMedian, s.iters, s.warmupSeconds,
                    s.warmupSteady ? 1 : 0, s.gpuClockStartMHz, s.gpuClockTimedMHz, s.gpuClockEndMHz);
    std::cout << "  " << v.name << ": median " << s.median << " ms (min " << s.min << ", max " << s.max << "; build " << s.firstMedian << ", compute "
              << s.secondMedian << ")" << std::endl;
}

// The variants of one point in a rotated order: the first one is variant (pointIndex mod the number of variants). A GPU that was at rest ramps its
// clock during the first second of work; the warm-up rule of fklb::measureWall covers that, and the rotation makes sure that a ramp that is not
// covered would not always hit the same variant
inline void timeRotated(Output& out, const int batch, const int pointIndex, const std::vector<Variant>& variants) {
    const size_t first = static_cast<size_t>(pointIndex + out.rotateOffset) % variants.size();
    for (size_t k = 0; k < variants.size(); ++k) {
        timeVariant(out, batch, variants[(first + k) % variants.size()]);
    }
}

// required: both regions must be within TOLERANCE; requiredInterior: only the interior must (the last two columns and rows of RPP differ from a reference by design)
inline void agreementRows(Output& out, const int batch, const std::string& what, const float* a, const float* b, const int n, const bool required,
                          const char* note, const bool requiredInterior = false) {
    for (const bool edge : {false, true}) {
        const Region r = compareTensors(a, b, n, OUT_W, OUT_H, edge);
        const bool pass = r.maxAbs <= TOLERANCE;
        const bool req = required || (requiredInterior && !edge);
        out.agreement->row(batch, what, edge ? "last 2 columns and rows" : "interior", r.maxAbs, r.meanAbs, r.fractionOver, req ? (pass ? 1 : 0) : -1, note);
        out.allPass &= pass || !req;
        std::cout << "  " << what << " [" << (edge ? "edge" : "interior") << "]: max " << r.maxAbs << ", mean " << r.meanAbs << ", over 1e-4 " << r.fractionOver
                  << (req ? (pass ? " ok" : " FAIL") : " (informational)") << std::endl;
    }
}

template <int N>
void runA(fk::Stream_<fk::ParArch::GPU_AMD>& fkStream, fk::Ptr2D<fk::uchar3>& fklFrame, const hipStream_t rppStream, const unsigned char* rppFrame,
          const hipStream_t handStream, const unsigned char* handFrame, const size_t handPitch, const std::vector<unsigned char>& hostFrame, Output& out) {
    std::cout << "(a) batch " << N << std::endl;

    // ---- Set up once for this batch size. Nothing here depends on the location of the crops: the output tensors and the intermediate tensors of
    // fkl_perop_batch depend on N and the output size; RPP's handle and descriptors, its buffers and its per-image parameter arrays (output
    // size, mean, standard deviation, mirror, channel order) depend on N, the crop size and the output size.
    fk::Tensor<float> fusedOut(OUT_W, OUT_H, N, 3), halfOut(OUT_W, OUT_H, N, 3), batchOut(OUT_W, OUT_H, N, 3);
    fk::Tensor<fk::float3> tmpA(OUT_W, OUT_H, N, 1), tmpB(OUT_W, OUT_H, N, 1);

    // One RPP handle for both calls, on RPP's stream. rppCreate's batch size is the number of images RPP processes: the N crops.
    rppHandle_t handle{};
    FKLB_RPP_CHECK(rppCreate(&handle, N, 0, rppStream, RppBackend::RPP_HIP_BACKEND));
    // The batch: N dense crops, 8 bit NHWC, one after the other (the RpptDesc constructor sets the batch stride nStride = CROP_H * CROP_W * 3);
    // the output: N float NCHW images
    RpptDesc cropDesc(N, 3, CROP_H, CROP_W, RpptDataType::U8, RpptLayout::NHWC);
    RpptDesc outDesc(N, 3, OUT_H, OUT_W, RpptDataType::F32, RpptLayout::NCHW);
    const size_t cropBytes = static_cast<size_t>(CROP_W) * CROP_H * 3;
    const size_t outBytes = static_cast<size_t>(N) * 3 * PLANE * sizeof(float);
    unsigned char* cropBatch = nullptr;
    float* resized = nullptr;
    float* rppOut = nullptr;
    FKLB_HIP_CHECK(hipMalloc(&cropBatch, cropBytes * N));
    FKLB_HIP_CHECK(hipMalloc(&resized, outBytes));
    FKLB_HIP_CHECK(hipMalloc(&rppOut, outBytes));
    // The parameter arrays of the two RPP calls, in pinned host memory: RPP's kernels read them in place on the GPU, so they must not change until
    // the kernels of a call have finished (every call ends with a synchronisation of RPP's stream)
    RpptImagePatch* outSizes = nullptr;
    RpptROI* rois = nullptr;
    Rpp32f* mean = nullptr;
    Rpp32f* stdDev = nullptr;
    Rpp32u* mirror = nullptr;
    Rpp32u* channelOrder = nullptr;
    FKLB_HIP_CHECK(hipHostMalloc(&outSizes, sizeof(RpptImagePatch) * N));
    FKLB_HIP_CHECK(hipHostMalloc(&rois, sizeof(RpptROI) * N));
    FKLB_HIP_CHECK(hipHostMalloc(&mean, sizeof(Rpp32f) * 3 * N));
    FKLB_HIP_CHECK(hipHostMalloc(&stdDev, sizeof(Rpp32f) * 3 * N));
    FKLB_HIP_CHECK(hipHostMalloc(&mirror, sizeof(Rpp32u) * N));
    FKLB_HIP_CHECK(hipHostMalloc(&channelOrder, sizeof(Rpp32u) * 3 * N));
    for (int i = 0; i < N; ++i) {
        outSizes[i].width = OUT_W;
        outSizes[i].height = OUT_H;
        mirror[i] = 0;
        for (int c = 0; c < 3; ++c) {
            // The permute (2, 1, 0) comes after the normalize, so the output channel 2 - c is (alpha * v_c - sub[2 - c]) / div[2 - c]
            // = (v_c - sub[2 - c] / alpha) / (div[2 - c] / alpha)
            mean[i * 3 + c] = SUB[2 - c] / ALPHA;
            stdDev[i * 3 + c] = DIV[2 - c] / ALPHA;
            channelOrder[i * 3 + c] = 2 - c;
        }
    }
    // The hand-written kernel's output, a dense hipMalloc tensor with the strides of fk::Tensor (checked, as in E1c), and its launch configuration:
    // the resize factors exactly as fk::Resize::build computes them, and E1c's block and grid
    const size_t outPitch = OUT_W * sizeof(float), imageStride = outPitch * OUT_H * 3, tensorBytes = imageStride * N;
    if (static_cast<size_t>(fusedOut.ptr().dims.pitch) != outPitch ||
        static_cast<size_t>(fusedOut.ptr().dims.plane_pitch) * fusedOut.ptr().dims.color_planes != imageStride || fusedOut.sizeInBytes() != tensorBytes) {
        throw std::runtime_error("the strides of fk::Tensor are not those of the hand-written kernel's tensor");
    }
    float* handOut = nullptr;
    FKLB_HIP_CHECK(hipMalloc(&handOut, tensorBytes));
    const double cfx = static_cast<double>(e1c::UP_W) / CROP_W, cfy = static_cast<double>(e1c::UP_H) / CROP_H;
    const float fx = static_cast<float>(1.0 / cfx), fy = static_cast<float>(1.0 / cfy);
    const dim3 block(32, 8);
    const dim3 grid((e1c::UP_W + block.x - 1) / block.x, (e1c::UP_H + block.y - 1) / block.y, N);

    // ---- The timed calls. The location of the crops may change on every image: everything that depends on it is rebuilt in every call, on both
    // sides. For FKL that is the crop views and every operation built from them; for RPP it is the copy of the crops into its batch.
    auto fklFused = [&](fklb::WallClock::time_point& split) {
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (int i = 0; i < N; ++i) {
            crops[i] = fklFrame.crop2D(fk::Rect(i, i, CROP_W, CROP_H));
        }
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
        const auto reorder = fk::VectorReorder<fk::float3, 2, 1, 0>::build();
        const auto mul = fk::Mul<fk::float3>::build(fk::float3{ALPHA, ALPHA, ALPHA});
        const auto sub = fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]});
        const auto div = fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]});
        const auto write = fk::TensorSplit<fk::float3>::build(fusedOut.ptr());
        split = fklb::WallClock::now();
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, read, reorder, mul, sub, div, write);
        fkStream.sync();
    };
    auto fklFusedHalf = [&](fklb::WallClock::time_point& split) {
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (int i = 0; i < N; ++i) {
            crops[i] = fklFrame.crop2D(fk::Rect(i, i, CROP_W, CROP_H));
        }
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR, fk::AspectRatio::IGNORE_AR, NullType, fk::ResizeSampling::HALF_PIXEL>::build(
                fk::Size(OUT_W, OUT_H)));
        const auto reorder = fk::VectorReorder<fk::float3, 2, 1, 0>::build();
        const auto mul = fk::Mul<fk::float3>::build(fk::float3{ALPHA, ALPHA, ALPHA});
        const auto sub = fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]});
        const auto div = fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]});
        const auto write = fk::TensorSplit<fk::float3>::build(halfOut.ptr());
        split = fklb::WallClock::now();
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, read, reorder, mul, sub, div, write);
        fkStream.sync();
    };
    auto fklPerOp = [&](fklb::WallClock::time_point& split) {
        std::array<fk::Ptr2D<fk::uchar3>, N> crops;
        for (int i = 0; i < N; ++i) {
            crops[i] = fklFrame.crop2D(fk::Rect(i, i, CROP_W, CROP_H));
        }
        const auto read = fk::PerThreadRead<fk::ND::_2D, fk::uchar3>::build(crops).then(
            fk::Resize<fk::InterpolationType::INTER_LINEAR>::build(fk::Size(OUT_W, OUT_H)));
        const auto reorder = fk::VectorReorder<fk::float3, 2, 1, 0>::build();
        const auto mul = fk::Mul<fk::float3>::build(fk::float3{ALPHA, ALPHA, ALPHA});
        const auto sub = fk::Sub<fk::float3>::build(fk::float3{SUB[0], SUB[1], SUB[2]});
        const auto div = fk::Div<fk::float3>::build(fk::float3{DIV[0], DIV[1], DIV[2]});
        const auto readA = fk::PerThreadRead<fk::ND::_3D, fk::float3>::build(tmpA.ptr());
        const auto readB = fk::PerThreadRead<fk::ND::_3D, fk::float3>::build(tmpB.ptr());
        const auto writeA = fk::PerThreadWrite<fk::ND::_3D, fk::float3>::build(tmpA.ptr());
        const auto writeB = fk::PerThreadWrite<fk::ND::_3D, fk::float3>::build(tmpB.ptr());
        const auto write = fk::TensorSplit<fk::float3>::build(batchOut.ptr());
        split = fklb::WallClock::now();
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, read, writeA);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, readA, reorder, writeB);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, readB, mul, writeA);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, readA, sub, writeB);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, readB, div, writeA);
        fk::executeOperations<fk::TransformDPP<fk::ParArch::GPU_AMD, fk::TF::DISABLED>>(fkStream, readA, write);
        fkStream.sync();
    };
    // RPP's batched functions take one tensor of images, described by an RpptDesc with a real batch stride (nStride, the distance from one image
    // to the next), and one ROI per image: that is the documented batched use of the rppt tensor API. The N crops of one frame are not such a
    // tensor, so they are copied into one dense batch, one hipMemcpy2DAsync per crop, and those copies are timed as RPP's batch building, as the
    // crop views and the operations are FKL's; then one call each of rppt_resize_mirror_normalize and rppt_channel_permute processes the whole
    // batch. The previous version of this program (rpp/benchmark_rpp_preprocessing.h l.139 at 0fb1f9f) gave RPP the whole frame with a batch
    // stride of 0 and one ROI per crop; RPP does not document a batch stride of 0, so it is not used. One RPP call per crop (a batch of one image)
    // is not used either: it is not the batched API.
    auto rppCropBatch = [&](fklb::WallClock::time_point& split) {
        for (int i = 0; i < N; ++i) {
            FKLB_HIP_CHECK(hipMemcpy2DAsync(cropBatch + cropBytes * i, static_cast<size_t>(CROP_W) * 3,
                                            rppFrame + (static_cast<size_t>(i) * SRC_W + static_cast<size_t>(i)) * 3, static_cast<size_t>(SRC_W) * 3,
                                            static_cast<size_t>(CROP_W) * 3, CROP_H, hipMemcpyDeviceToDevice, rppStream));
        }
        FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
        split = fklb::WallClock::now();
        // rppt_resize_mirror_normalize converts the ROIs from XYWH to LTRB in place, so they are written before each call; each ROI is a whole crop
        for (int i = 0; i < N; ++i) {
            rois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
        }
        FKLB_RPP_CHECK(rppt_resize_mirror_normalize(cropBatch, &cropDesc, resized, &outDesc, outSizes, RpptInterpolationType::BILINEAR, mean, stdDev,
                                                    mirror, rois, RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_RPP_CHECK(rppt_channel_permute(resized, &outDesc, rppOut, &outDesc, channelOrder, handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
    };
    // The same work without the synchronisation between the copies and the RPP calls, which only the measurement of the parts needs (the copies
    // are on RPP's stream, before the RPP calls, and rppt_resize_mirror_normalize synchronises that stream itself)
    auto rppCropBatchUnsplit = [&](fklb::WallClock::time_point&) {
        for (int i = 0; i < N; ++i) {
            FKLB_HIP_CHECK(hipMemcpy2DAsync(cropBatch + cropBytes * i, static_cast<size_t>(CROP_W) * 3,
                                            rppFrame + (static_cast<size_t>(i) * SRC_W + static_cast<size_t>(i)) * 3, static_cast<size_t>(SRC_W) * 3,
                                            static_cast<size_t>(CROP_W) * 3, CROP_H, hipMemcpyDeviceToDevice, rppStream));
        }
        for (int i = 0; i < N; ++i) {
            rois[i].xywhROI = {{0, 0}, CROP_W, CROP_H};
        }
        FKLB_RPP_CHECK(rppt_resize_mirror_normalize(cropBatch, &cropDesc, resized, &outDesc, outSizes, RpptInterpolationType::BILINEAR, mean, stdDev,
                                                    mirror, rois, RpptRoiType::XYWH, handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_RPP_CHECK(rppt_channel_permute(resized, &outDesc, rppOut, &outDesc, channelOrder, handle, RppBackend::RPP_HIP_BACKEND));
        FKLB_HIP_CHECK(hipStreamSynchronize(rppStream));
    };
    // The hand-written kernel as E1c prepares and launches it (prepareHand and launchHand there): its parameter block, the crop pointers into its
    // own frame, is built from the crop locations in every call
    auto hand = [&](fklb::WallClock::time_point& split) {
        e1c::CropPointers<N> crops;
        for (int i = 0; i < N; ++i) crops.data[i] = handFrame + i * handPitch + i * sizeof(uchar3);
        split = fklb::WallClock::now();
        e1c::handPreproc<N><<<grid, block, 0, handStream>>>(crops, handPitch, handOut, outPitch, imageStride, fx, fy);
        FKLB_HIP_CHECK(hipGetLastError());
        FKLB_HIP_CHECK(hipStreamSynchronize(handStream));
    };

    // ---- Agreement: one run of every variant, the outputs on the host
    fklb::WallClock::time_point unused{};
    fklFused(unused);
    fklFusedHalf(unused);
    fklPerOp(unused);
    fusedOut.download(fkStream);
    halfOut.download(fkStream);
    batchOut.download(fkStream);
    fkStream.sync();
    std::vector<float> rppSplitOut(static_cast<size_t>(N) * 3 * PLANE), rppUnsplitOut(rppSplitOut.size());
    rppCropBatch(unused);
    FKLB_HIP_CHECK(hipMemcpy(rppSplitOut.data(), rppOut, outBytes, hipMemcpyDeviceToHost));
    // The unsplit run starts from an empty batch and an empty output, so that its row checks its own copies
    FKLB_HIP_CHECK(hipMemset(cropBatch, 0, cropBytes * N));
    FKLB_HIP_CHECK(hipMemset(rppOut, 0, outBytes));
    rppCropBatchUnsplit(unused);
    FKLB_HIP_CHECK(hipMemcpy(rppUnsplitOut.data(), rppOut, outBytes, hipMemcpyDeviceToHost));
    // The hand-written kernel's run starts from an output of NaNs (every byte 0xFF), so that an element it does not write fails its rows
    std::vector<float> handHost(static_cast<size_t>(N) * 3 * PLANE);
    FKLB_HIP_CHECK(hipMemset(handOut, 0xFF, tensorBytes));
    hand(unused);
    FKLB_HIP_CHECK(hipMemcpy(handHost.data(), handOut, tensorBytes, hipMemcpyDeviceToHost));
    const std::vector<float> refAsym = referenceA(hostFrame, N, fklb::ResizeSampling::ASYMMETRIC);
    const std::vector<float> refHalf = referenceA(hostFrame, N, fklb::ResizeSampling::HALF_PIXEL);
    const float* fused = fusedOut.ptrPinned().data;
    const float* half = halfOut.ptrPinned().data;
    const float* perOp = batchOut.ptrPinned().data;
    agreementRows(out, N, "fkl_fused vs scalar reference ASYMMETRIC", fused, refAsym.data(), N, true, "");
    agreementRows(out, N, "fkl_fused_halfpixel vs scalar reference HALF_PIXEL", half, refHalf.data(), N, true, "");
    agreementRows(out, N, "fkl_perop_batch vs fkl_fused", perOp, fused, N, true, "");
    agreementRows(out, N, "rpp_crop_batch vs scalar reference HALF_PIXEL", rppSplitOut.data(), refHalf.data(), N, false,
                  "RPP clamps the source coordinate to size-2: the last 2 columns and rows differ from the reference; the interior is a required check (RPP's output must be right to be timed)", true);
    agreementRows(out, N, "rpp_crop_batch vs fkl_fused_halfpixel", rppSplitOut.data(), half, N, false,
                  "both HALF_PIXEL; RPP clamps the source coordinate to size-2: the last 2 columns and rows differ");
    agreementRows(out, N, "rpp_crop_batch_unsplit vs rpp_crop_batch", rppUnsplitOut.data(), rppSplitOut.data(), N, true, "the same RPP calls");
    agreementRows(out, N, "hand vs scalar reference ASYMMETRIC", handHost.data(), refAsym.data(), N, true, "");
    // hand against fkl_fused: the differences, informational, and whether the two tensors are the same bits
    size_t handBitsDiffer = 0;
    for (size_t k = 0; k < handHost.size(); ++k) {
        handBitsDiffer += std::memcmp(&handHost[k], &fused[k], sizeof(float)) != 0 ? 1 : 0;
    }
    const std::string handBits = "the whole tensors: " + (handBitsDiffer == 0 ? std::string("bitwise equal")
                                                                              : std::to_string(handBitsDiffer) + " of " + std::to_string(handHost.size()) +
                                                                                    " values differ in their bits");
    agreementRows(out, N, "hand vs fkl_fused", handHost.data(), fused, N, false, handBits.c_str());
    std::cout << "  hand vs fkl_fused: " << handBits << std::endl;

    // ---- Timing
    timeRotated(out, N, N / 10 - 1,
                {{"fkl_fused", 1, fklFused},
                 {"fkl_fused_halfpixel", 1, fklFusedHalf},
                 {"fkl_perop_batch", 6, fklPerOp},
                 {"rpp_crop_batch", N + 2, rppCropBatch},
                 {"rpp_crop_batch_unsplit", N + 2, rppCropBatchUnsplit},
                 {"hand", 1, hand}});
    out.timing->flush();
    out.agreement->flush();

    FKLB_HIP_CHECK(hipFree(handOut));
    FKLB_RPP_CHECK(rppDestroy(handle, RppBackend::RPP_HIP_BACKEND));
    FKLB_HIP_CHECK(hipFree(cropBatch));
    FKLB_HIP_CHECK(hipFree(resized));
    FKLB_HIP_CHECK(hipFree(rppOut));
    FKLB_HIP_CHECK(hipHostFree(outSizes));
    FKLB_HIP_CHECK(hipHostFree(rois));
    FKLB_HIP_CHECK(hipHostFree(mean));
    FKLB_HIP_CHECK(hipHostFree(stdDev));
    FKLB_HIP_CHECK(hipHostFree(mirror));
    FKLB_HIP_CHECK(hipHostFree(channelOrder));
}

template <int... Ns>
void runAll(fk::Stream_<fk::ParArch::GPU_AMD>& fkStream, fk::Ptr2D<fk::uchar3>& fklFrame, const hipStream_t rppStream, const unsigned char* rppFrame,
            const hipStream_t handStream, const unsigned char* handFrame, const size_t handPitch, const std::vector<unsigned char>& hostFrame, Output& out,
            std::integer_sequence<int, Ns...>) {
    (runA<(Ns + 1) * 10>(fkStream, fklFrame, rppStream, rppFrame, handStream, handFrame, handPitch, hostFrame, out), ...);
}

} // namespace rppbench

int launch() {
    using namespace rppbench;
    const char* imagePath = std::getenv("E5_IMAGE");
    const char* imageInfo = std::getenv("E5_IMAGE_INFO");
    const int warmup = fklb::envInt("E5_WARMUP", 3), iters = fklb::envInt("E5_ITERS", 30, 1, 1000000);
    const double warmupSeconds = fklb::envDouble("E5_WARMUP_SECONDS", 2.0, 0.0, 10.0); // the warm-up ends after 10 s whatever the value
    const int rotateOffset = fklb::envInt("E5_ROTATE_OFFSET", 0);

    // The frame, as host bytes (dense RGB, SRC_W * 3 bytes per row): a natural image tiled over it, or seeded noise
    std::vector<unsigned char> hostFrame(static_cast<size_t>(SRC_W) * SRC_H * 3);
    std::vector<unsigned char> ppm;
    int ppmW = 0, ppmH = 0;
    const bool natural = imagePath != nullptr;
    if (natural && !fklb::readPpm(imagePath, ppm, ppmW, ppmH)) {   // a natural-image run must not measure the seeded noise instead
        throw std::runtime_error(std::string("E5_IMAGE=") + imagePath + " cannot be read as a binary PPM (P6, maximum value 255)");
    }
    uint32_t state = 12345u;
    for (int y = 0; y < SRC_H; ++y) {
        for (int x = 0; x < SRC_W; ++x) {
            unsigned char* p = &hostFrame[(static_cast<size_t>(y) * SRC_W + x) * 3];
            if (natural) {
                const unsigned char* q = &ppm[(static_cast<size_t>(y % ppmH) * ppmW + (x % ppmW)) * 3];
                p[0] = q[0];
                p[1] = q[1];
                p[2] = q[2];
            } else {
                state = state * 1664525u + 1013904223u;
                p[0] = static_cast<unsigned char>(state >> 24);
                p[1] = static_cast<unsigned char>(state >> 16);
                p[2] = static_cast<unsigned char>(state >> 8);
            }
        }
    }
    const std::string sourceText = natural ? std::string("natural image ") + imagePath + " (" + std::to_string(ppmW) + "x" + std::to_string(ppmH) +
                                                 ") tiled over the frame" + (imageInfo ? std::string("; ") + imageInfo : "")
                                           : "seeded noise (LCG, seed 12345)";

    // FKL's frame: a fk::Ptr2D (device memory with a pinned host copy), written from the host bytes and uploaded on FKL's stream
    fk::Stream_<fk::ParArch::GPU_AMD> fkStream;
    fk::Ptr2D<fk::uchar3> fklFrame(SRC_W, SRC_H);
    for (int y = 0; y < SRC_H; ++y) {
        for (int x = 0; x < SRC_W; ++x) {
            const unsigned char* p = &hostFrame[(static_cast<size_t>(y) * SRC_W + x) * 3];
            fklFrame.at(x, y) = fk::uchar3{p[0], p[1], p[2]};
        }
    }
    fklFrame.upload(fkStream);
    fkStream.sync();
    // RPP's frame: a dense hipMalloc buffer, copied from the same host bytes; RPP's stream
    hipStream_t rppStream{};
    FKLB_HIP_CHECK(hipStreamCreate(&rppStream));
    unsigned char* rppFrame = nullptr;
    FKLB_HIP_CHECK(hipMalloc(&rppFrame, hostFrame.size()));
    FKLB_HIP_CHECK(hipMemcpy(rppFrame, hostFrame.data(), hostFrame.size(), hipMemcpyHostToDevice));
    // The hand-written kernel's frame: a hipMallocPitch image (cudaMallocPitch in E1c), copied from the same host bytes; its own hipStream_t
    hipStream_t handStream{};
    FKLB_HIP_CHECK(hipStreamCreate(&handStream));
    unsigned char* handFrame = nullptr;
    size_t handPitch = 0;
    FKLB_HIP_CHECK(hipMallocPitch(reinterpret_cast<void**>(&handFrame), &handPitch, SRC_W * sizeof(uchar3), SRC_H));
    FKLB_HIP_CHECK(hipMemcpy2D(handFrame, handPitch, hostFrame.data(), SRC_W * sizeof(uchar3), SRC_W * sizeof(uchar3), SRC_H, hipMemcpyHostToDevice));

    const std::string rppText = "RPP " + std::to_string(RPP_VERSION_MAJOR) + "." + std::to_string(RPP_VERSION_MINOR) + "." + std::to_string(RPP_VERSION_PATCH) +
                                " (ROCm package amdrocm-rpp), rppt tensor API, HIP back-end; rpp_crop_batch: the N crops copied into one dense batch (one "
                                "hipMemcpy2DAsync per crop), then rppt_resize_mirror_normalize (BILINEAR, one ROI XYWH per crop: the whole crop, source U8 "
                                "NHWC with batch stride CROP_H * CROP_W * 3, destination F32 NCHW, mirror off) and rppt_channel_permute (F32 NCHW); "
                                "RPP has its own frame (dense hipMalloc) and its own hipStream_t; output widths are multiples of 8";
    const std::string fklText = "FKL: fk::PerThreadRead::build(crop2D views).then(Resize) -> VectorReorder<float3, 2, 1, 0> -> Mul -> Sub -> Div -> TensorSplit, "
                                "TransformDPP<GPU_AMD, TF::DISABLED>, the crop views and every operation built in every timed call";
    const std::string handText = "hand: the hand-written fused kernel of E1c (handwritten-cuda/e1_preproc.cu: handPreproc) ported to HIP (__grid_constant__ "
                                 "dropped), one launch with 32x8 threads per block, ASYMMETRIC sampling; its own frame (hipMallocPitch, pitch " +
                                 std::to_string(handPitch) + " bytes; FKL's frame: pitch " + std::to_string(fklFrame.ptr().dims.pitch) +
                                 " bytes), output (hipMalloc) and hipStream_t; build = the crop pointers (its parameter block, 8 bytes per crop), "
                                 "compute = the launch, hipGetLastError and the stream synchronisation";
    const std::string notes = "FKL HIP vs RPP and a hand-written kernel; source: " + sourceText + "; " + rppText + "; " + fklText + "; " + handText +
                              "; wall clock on the host including the stream synchronisation; build_median_ms and compute_median_ms are the medians of the "
                              "two parts of a run: FKL build = the crop views and the operations (host), compute = executeOperations and the "
                              "synchronisation (no synchronisation between them); rpp_crop_batch build = the copies and a stream synchronisation, compute = "
                              "the ROIs, the two RPP calls and a stream synchronisation, so its total has one synchronisation more than "
                              "rpp_crop_batch_unsplit, which has no parts (nan); the median of the runs is not the sum of the medians of the parts; "
                              "warm-up: " + std::to_string(warmup) + " runs, at least " +
                              std::to_string(warmupSeconds).substr(0, std::to_string(warmupSeconds).find('.') + 2) +
                              " s and until the mean time of two consecutive windows of 0.25 s differs by at most 2 percent (cap 10 s; the columns "
                              "warmup_s, warmup_steady and sclk_start_mhz, sclk_timed_start_mhz and sclk_end_mhz, the graphics clock at the start of the "
                              "warm-up, at the start of the timed runs and after them, record it); the variants of a point are timed in a rotated order: "
                              "the first one is variant ((index of the point + E5_ROTATE_OFFSET = " + std::to_string(rotateOffset) +
                              ") mod the number of variants)";

    fklb::CsvWriter timingA("benchmark_rpp_preprocessing_a_GPU_AMD.csv", "benchmark_rpp_preprocessing_a",
                            {"batch", "variant", "calls", "median_ms", "min_ms", "max_ms", "mean_ms", "build_median_ms", "compute_median_ms", "iters",
                             "warmup_s", "warmup_steady", "sclk_start_mhz", "sclk_timed_start_mhz", "sclk_end_mhz"},
                            notes);
    fklb::CsvWriter agreeA("benchmark_rpp_preprocessing_a_agreement_GPU_AMD.csv", "benchmark_rpp_preprocessing_a",
                           {"batch", "comparison", "region", "max_abs_diff", "mean_abs_diff", "fraction_over_1e-4", "pass", "note"},
                           notes + "; pass -1 = informational; tolerance 1e-4 absolute");
    Output outA{&timingA, &agreeA, true, warmup, iters, warmupSeconds, rotateOffset};
    runAll(fkStream, fklFrame, rppStream, rppFrame, handStream, handFrame, handPitch, hostFrame, outA, std::make_integer_sequence<int, 30>{});
    FKLB_HIP_CHECK(hipFree(handFrame));
    FKLB_HIP_CHECK(hipStreamDestroy(handStream));
    FKLB_HIP_CHECK(hipFree(rppFrame));
    FKLB_HIP_CHECK(hipStreamDestroy(rppStream));
    std::cout << (outA.allPass ? "all required numerical checks passed" : "NUMERICAL CHECK FAILED") << std::endl;
    return outA.allPass ? 0 : -1;
}
