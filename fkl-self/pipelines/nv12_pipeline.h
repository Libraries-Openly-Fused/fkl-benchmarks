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

// E4b: the NV12 video pre-processing pipeline of the paper, for every back-end (this header is compiled by nvcc,
// amdclang++ and the host compiler; the same code runs on ParArch::GPU_NVIDIA, GPU_AMD, CPU and CPU_MULTITHREAD).
//
//   kernel 1  a 4K NV12 frame -> a full RGB frame (ReadYUV<NV12> + ConvertYUVToRGB, BT.601 studio swing, 8 bit)
//   kernel 2  N detection crops of the RGB frame (one RawPtr2D region of interest per crop, so the interpolation never
//             reads outside its crop) -> bilinear resize to 64x128 -> multiply by 1/255, subtract the mean, divide by the
//             standard deviation -> an NCHW float tensor of N x 3 x 128 x 64; fused into one executeOperations
// The FKL path launches two kernels (the paper says so). The baselines without fusion execute every operation of kernel 2 on
// its own (per batch: one executeOperations per operation for all the crops; per crop: one per operation and per crop).
#pragma once

#include <common/resize_reference.h>

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <utility>
#include <vector>

namespace nv12 {

constexpr int FRAME_W = 3840, FRAME_H = 2160, OUT_W = 64, OUT_H = 128;
constexpr int PLANE = OUT_W * OUT_H;
constexpr uint32_t FRAME_SEED = 424242u, CROPS_SEED = 777u;

using U3 = fk::uchar3;
using F3 = fk::float3;
using Frame = fk::Image<fk::PixelFormat::NV12>;
// BT.601 studio swing (Y 16..235, chroma 16..240) to full range RGB, the conversion of cv::COLOR_YUV2RGB_NV12. In FKL fb9ac3a the
// matrix with the 1.164 gain is ColorRange::Full, bt601; ColorRange::Limited with bt601 has no matrix and gives black.
using ToRgb = fk::ConvertYUVToRGB<fk::PixelFormat::NV12, fk::ColorRange::Full, fk::ColorPrimitives::bt601, false, U3>;
using Rgb2D = fk::RawPtr<fk::ND::_2D, U3>;

// The sampling of the linear resize: FKL's Resize samples the source at x * (src / dst) (ResizeSampling::ASYMMETRIC, the default).
// With -DE5_RESIZE_HALF_PIXEL the resize is ResizeHalfPixel, (x + 0.5) * (src / dst) - 0.5, which needs an FKL with ResizeSampling
// (FKL pull request #409).
#ifdef E5_RESIZE_HALF_PIXEL
using LinearResize = fk::ResizeHalfPixel<fk::InterpolationType::INTER_LINEAR>;
constexpr fklb::ResizeSampling RESIZE_SAMPLING = fklb::ResizeSampling::HALF_PIXEL;
constexpr const char* RESIZE_SAMPLING_NAME = "HALF_PIXEL";
#else
using LinearResize = fk::Resize<fk::InterpolationType::INTER_LINEAR>;
constexpr fklb::ResizeSampling RESIZE_SAMPLING = fklb::ResizeSampling::ASYMMETRIC;
constexpr const char* RESIZE_SAMPLING_NAME = "ASYMMETRIC";
#endif

// Normalisation of a DNN input: (x / 255 - mean) / std
constexpr float MEAN[3] = {0.485f, 0.456f, 0.406f}, STD[3] = {0.229f, 0.224f, 0.225f};

// ------------------------------------------------------------------------------------------------------------------
// Input data
inline uint32_t lcg(uint32_t& state) {
    state = state * 1664525u + 1013904223u;
    return state >> 8;
}

// The host view of the memory of a Ptr: the pinned copy of a device Ptr, the memory itself otherwise
template <typename PtrType>
auto hostRaw(const PtrType& ptr) {
    const auto pinned = ptr.ptrPinned();
    return pinned.data != nullptr ? pinned : ptr.ptr();
}

// A frame of seeded noise: luma in 16..235, chroma in 16..240 (the limited range)
inline void fillNoise(Frame& frame) {
    auto data = frame.getData();
    uint32_t state = FRAME_SEED;
    for (int y = 0; y < FRAME_H * 3 / 2; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            const bool luma = y < FRAME_H;
            data.at(x, y) = static_cast<uchar>(luma ? 16 + lcg(state) % 220 : 16 + lcg(state) % 225);
        }
    }
}

// A binary PPM (P6, 8 bit)
inline bool readPpm(const char* path, std::vector<uchar>& rgb, int& w, int& h) {
    std::ifstream in(path, std::ios::binary);
    std::string magic;
    int maxval = 0;
    in >> magic >> w >> h >> maxval;
    in.get();
    if (!in || magic != "P6" || maxval != 255 || w <= 0 || h <= 0) {
        return false;
    }
    rgb.resize(static_cast<size_t>(w) * h * 3);
    in.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    return static_cast<bool>(in);
}

// An RGB image tiled over the frame, converted to NV12 (BT.601, limited range, chroma averaged over 2x2 pixels)
inline void fillFromRgb(Frame& frame, const std::vector<uchar>& rgb, const int w, const int h) {
    auto data = frame.getData();
    const auto at = [&](const int x, const int y, const int c) {
        return static_cast<double>(rgb[(static_cast<size_t>(y % h) * w + (x % w)) * 3 + c]);
    };
    const auto clampByte = [](const double v) { return static_cast<uchar>(std::min(255.0, std::max(0.0, std::floor(v + 0.5)))); };
    for (int y = 0; y < FRAME_H; ++y) {
        for (int x = 0; x < FRAME_W; ++x) {
            data.at(x, y) = clampByte(16.0 + 0.256788 * at(x, y, 0) + 0.504129 * at(x, y, 1) + 0.097906 * at(x, y, 2));
        }
    }
    for (int y = 0; y < FRAME_H / 2; ++y) {
        for (int x = 0; x < FRAME_W / 2; ++x) {
            double r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; ++dy) {
                for (int dx = 0; dx < 2; ++dx) {
                    r += at(2 * x + dx, 2 * y + dy, 0);
                    g += at(2 * x + dx, 2 * y + dy, 1);
                    b += at(2 * x + dx, 2 * y + dy, 2);
                }
            }
            r /= 4;
            g /= 4;
            b /= 4;
            data.at(2 * x, FRAME_H + y) = clampByte(128.0 - 0.148223 * r - 0.290993 * g + 0.439216 * b);     // U
            data.at(2 * x + 1, FRAME_H + y) = clampByte(128.0 + 0.439216 * r - 0.367788 * g - 0.071427 * b); // V
        }
    }
}

// N detection-like crops (a person is taller than wide): sizes 48..176 x 96..352, anywhere in the frame
template <int N>
std::array<fk::Rect, N> makeCrops() {
    std::array<fk::Rect, N> crops{};
    uint32_t state = CROPS_SEED;
    for (int i = 0; i < N; ++i) {
        const int w = 48 + static_cast<int>(lcg(state) % 129), h = 96 + static_cast<int>(lcg(state) % 257);
        const uint x = lcg(state) % static_cast<uint>(FRAME_W - w), y = lcg(state) % static_cast<uint>(FRAME_H - h);
        crops[i] = fk::Rect(x, y, w, h);
    }
    return crops;
}

// ------------------------------------------------------------------------------------------------------------------
// Operations
inline Rgb2D roiOf(const Rgb2D& frame, const fk::Rect& rect) {
    Rgb2D roi = frame;
    roi.data = reinterpret_cast<U3*>(reinterpret_cast<char*>(frame.data) + static_cast<size_t>(rect.y) * frame.dims.pitch) + rect.x;
    roi.dims.width = static_cast<uint>(rect.width);
    roi.dims.height = static_cast<uint>(rect.height);
    return roi;
}

inline fk::RawPtr<fk::ND::_3D, float> tensorSlice(fk::RawPtr<fk::ND::_3D, float> tensor, const int i) {
    tensor.data = reinterpret_cast<float*>(reinterpret_cast<char*>(tensor.data) +
                                           static_cast<size_t>(i) * tensor.dims.plane_pitch * tensor.dims.color_planes);
    tensor.dims.planes = 1;
    return tensor;
}

inline auto resizeOf(const Rgb2D& roi) {
    return LinearResize::build(fk::PerThreadRead<fk::ND::_2D, U3>::build(roi), fk::Size(OUT_W, OUT_H));
}
inline auto read2D(const fk::RawPtr<fk::ND::_2D, F3>& p) { return fk::PerThreadRead<fk::ND::_2D, F3>::build(p); }
inline auto write2D(const fk::RawPtr<fk::ND::_2D, F3>& p) { return fk::PerThreadWrite<fk::ND::_2D, F3>::build(p); }
inline auto read3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadRead<fk::ND::_3D, F3>::build(p); }
inline auto write3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadWrite<fk::ND::_3D, F3>::build(p); }
inline auto split3D(const fk::RawPtr<fk::ND::_3D, float>& p) { return fk::TensorSplit<F3>::build(p); }
inline auto mulOp() { return fk::Mul<F3>::build(F3{1.f / 255.f, 1.f / 255.f, 1.f / 255.f}); }
inline auto subOp() { return fk::Sub<F3>::build(F3{MEAN[0], MEAN[1], MEAN[2]}); }
inline auto divOp() { return fk::Div<F3>::build(F3{STD[0], STD[1], STD[2]}); }

using ResizeOp = decltype(resizeOf(std::declval<Rgb2D>()));
using Read2DOp = decltype(read2D(std::declval<fk::RawPtr<fk::ND::_2D, F3>>()));
using Write2DOp = decltype(write2D(std::declval<fk::RawPtr<fk::ND::_2D, F3>>()));
using Read3DOp = decltype(read3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using Write3DOp = decltype(write3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using SplitOp = decltype(split3D(std::declval<fk::RawPtr<fk::ND::_3D, float>>()));
using MulOp = decltype(mulOp());
using SubOp = decltype(subOp());
using DivOp = decltype(divOp());

template <int N, typename ResizeT = LinearResize>
auto batchResizeOf(const Rgb2D& frame, const std::array<fk::Rect, N>& crops) {
    std::array<Rgb2D, N> rois;
    for (int i = 0; i < N; ++i) {
        rois[i] = roiOf(frame, crops[i]);
    }
    return ResizeT::build(fk::PerThreadRead<fk::ND::_2D, U3>::build_batch(rois), fk::make_set_std_array<N>(fk::Size(OUT_W, OUT_H)));
}

// ------------------------------------------------------------------------------------------------------------------
// The pipeline of one back-end PA and N crops. The RGB frame is shared by every PA of a program (host memory for the CPU
// back-ends, device memory for the GPU ones). The operations are built once, in the constructor.
// ResizeT: the linear resize of the batch of crops (Resize or ResizeHalfPixel)
template <fk::ParArch PA, int N, typename ResizeT = LinearResize>
class Pipeline {
public:
    using Dpp = fk::TransformDPP<PA>;
    using StreamType = fk::Stream_<PA>;

    Pipeline(StreamType& stream, const Frame& frame, fk::Ptr2D<U3>& rgb, const std::array<fk::Rect, N>& crops)
        : stream_(stream), frame_(frame), rgb_(rgb), crops_(crops),
          fusedOut_(OUT_W, OUT_H, N, 3), batchOut_(OUT_W, OUT_H, N, 3), cropOut_(OUT_W, OUT_H, N, 3),
          tmpA3D_(OUT_W, OUT_H, N, 1), tmpB3D_(OUT_W, OUT_H, N, 1), tmpA_(OUT_W, OUT_H), tmpB_(OUT_W, OUT_H),
          readFrame_(fk::ReadYUV<fk::PixelFormat::NV12>::build(frame)), toRgb_(ToRgb::build()),
          writeRgb_(fk::PerThreadWrite<fk::ND::_2D, U3>::build(rgb)),
          fusedResize_(batchResizeOf<N, ResizeT>(rgb.ptr(), crops)), fusedSplit_(split3D(fusedOut_.ptr())),
          readA3D_(read3D(tmpA3D_.ptr())), readB3D_(read3D(tmpB3D_.ptr())), writeA3D_(write3D(tmpA3D_.ptr())),
          writeB3D_(write3D(tmpB3D_.ptr())), batchSplit_(split3D(batchOut_.ptr())),
          readA_(read2D(tmpA_.ptr())), readB_(read2D(tmpB_.ptr())), writeA_(write2D(tmpA_.ptr())),
          writeB_(write2D(tmpB_.ptr())), mul_(mulOp()), sub_(subOp()), div_(divOp()) {
        for (int i = 0; i < N; ++i) {
            cropResize_.push_back(resizeOf(roiOf(rgb.ptr(), crops[i])));
            cropSplit_.push_back(split3D(tensorSlice(cropOut_.ptr(), i)));
        }
    }

    // Kernel 1: NV12 frame to RGB frame (the frame is uploaded by the caller)
    void convert() { fk::executeOperations<Dpp>(stream_, readFrame_, toRgb_, writeRgb_); }
    // Kernel 2, fused
    void fused() { fk::executeOperations<Dpp>(stream_, fusedResize_, mul_, sub_, div_, fusedSplit_); }
    // Kernel 2 without fusion, an executeOperations per operation for all the crops
    void perOpBatch() {
        fk::executeOperations<Dpp>(stream_, fusedResize_, writeA3D_);
        fk::executeOperations<Dpp>(stream_, readA3D_, mul_, writeB3D_);
        fk::executeOperations<Dpp>(stream_, readB3D_, sub_, writeA3D_);
        fk::executeOperations<Dpp>(stream_, readA3D_, div_, writeB3D_);
        fk::executeOperations<Dpp>(stream_, readB3D_, batchSplit_);
    }
    // Kernel 2 without fusion, an executeOperations per operation and per crop
    void perOpCrop() {
        for (int i = 0; i < N; ++i) {
            fk::executeOperations<Dpp>(stream_, cropResize_[i], writeA_);
            fk::executeOperations<Dpp>(stream_, readA_, mul_, writeB_);
            fk::executeOperations<Dpp>(stream_, readB_, sub_, writeA_);
            fk::executeOperations<Dpp>(stream_, readA_, div_, writeB_);
            fk::executeOperations<Dpp>(stream_, readB_, cropSplit_[i]);
        }
    }
    void sync() { stream_.sync(); }

    // Brings the outputs to the host (after the executions, before sync)
    void download() {
        fusedOut_.download(stream_);
        batchOut_.download(stream_);
        cropOut_.download(stream_);
        rgb_.download(stream_);
    }
    const float* fusedOut() const { return hostRaw(fusedOut_).data; }
    const float* batchOut() const { return hostRaw(batchOut_).data; }
    const float* cropOut() const { return hostRaw(cropOut_).data; }
    static constexpr size_t outValues() { return static_cast<size_t>(N) * 3 * PLANE; }

private:
    StreamType& stream_;
    const Frame& frame_;
    fk::Ptr2D<U3>& rgb_;
    std::array<fk::Rect, N> crops_;
    fk::Tensor<float> fusedOut_, batchOut_, cropOut_;
    fk::Tensor<F3> tmpA3D_, tmpB3D_;
    fk::Ptr2D<F3> tmpA_, tmpB_;
    decltype(fk::ReadYUV<fk::PixelFormat::NV12>::build(std::declval<const Frame&>())) readFrame_;
    decltype(ToRgb::build()) toRgb_;
    decltype(fk::PerThreadWrite<fk::ND::_2D, U3>::build(std::declval<fk::Ptr2D<U3>&>())) writeRgb_;
    decltype(batchResizeOf<N, ResizeT>(std::declval<Rgb2D>(), std::declval<std::array<fk::Rect, N>>())) fusedResize_;
    SplitOp fusedSplit_;
    Read3DOp readA3D_, readB3D_;
    Write3DOp writeA3D_, writeB3D_;
    SplitOp batchSplit_;
    Read2DOp readA_, readB_;
    Write2DOp writeA_, writeB_;
    MulOp mul_;
    SubOp sub_;
    DivOp div_;
    std::vector<ResizeOp> cropResize_;
    std::vector<SplitOp> cropSplit_;
};

// ------------------------------------------------------------------------------------------------------------------
// The scalar reference of kernel 2 (double precision, the sampling of the resize above): from the RGB frame on the host
template <int N>
std::vector<float> referenceTensor(const fk::RawPtr<fk::ND::_2D, U3>& hostRgb, const std::array<fk::Rect, N>& crops,
                                   const fklb::ResizeSampling sampling = RESIZE_SAMPLING) {
    std::vector<float> out(static_cast<size_t>(N) * 3 * PLANE);
    std::vector<double> resized(static_cast<size_t>(PLANE) * 3);
    for (int i = 0; i < N; ++i) {
        const Rgb2D roi = roiOf(hostRgb, crops[i]);
        fklb::resizeBilinear(reinterpret_cast<const uchar*>(roi.data), roi.dims.pitch, crops[i].width, crops[i].height, 3,
                             OUT_W, OUT_H, sampling, resized.data());
        for (int p = 0; p < PLANE; ++p) {
            for (int c = 0; c < 3; ++c) {
                out[(static_cast<size_t>(i) * 3 + c) * PLANE + p] = static_cast<float>(
                    (resized[static_cast<size_t>(p) * 3 + c] * static_cast<double>(1.f / 255.f) - static_cast<double>(MEAN[c])) /
                    static_cast<double>(STD[c]));
            }
        }
    }
    return out;
}

inline double maxAbsDiff(const float* a, const float* b, const size_t n) {
    double worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
        if (!(d <= worst)) { // also catches NaN
            worst = d;
        }
    }
    return worst;
}

// ------------------------------------------------------------------------------------------------------------------
// Wall clock timing of fn() (it must end with the synchronisation of its stream): warmup runs, then iters timed runs
struct Stats {
    double median{0}, min{0}, max{0}, mean{0};
    size_t iters{0};
    double warmupSeconds{0};  // the warm-up that was run, in seconds
    bool warmupSteady{true};  // false: the cap of the warm-up was reached before the time of the runs stopped changing
    // The graphics clock of an AMD GPU (sysfs) at the start of the warm-up, at the start of the timed runs and after them; -1 when it can not be read
    double gpuClockStartMHz{-1}, gpuClockTimedMHz{-1}, gpuClockEndMHz{-1};
};

// The graphics clock of the first AMD GPU in MHz (sysfs hwmon freq1_input of the device with the vendor 0x1002), -1 when it can not be read
inline double amdGpuClockMHz() {
    static const std::string path = [] {
        std::error_code ec;
        for (const auto& card : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            const std::filesystem::path device = card.path() / "device";
            std::ifstream vendor(device / "vendor");
            std::string v;
            if (!(vendor >> v) || v != "0x1002") {
                continue;
            }
            for (const auto& hw : std::filesystem::directory_iterator(device / "hwmon", ec)) {
                if (std::filesystem::exists(hw.path() / "freq1_input", ec)) {
                    return (hw.path() / "freq1_input").string();
                }
            }
        }
        return std::string();
    }();
    if (path.empty()) {
        return -1.0;
    }
    std::ifstream f(path);
    double hz = 0;
    return (f >> hz) ? hz / 1e6 : -1.0;
}

// Warm-up: at least `warmup` runs and, when minWarmupSeconds > 0, at least that long. With steady = true the warm-up goes on, in windows of 0.25 s,
// until the mean time of two consecutive windows differs by at most 2 percent (up to maxWarmupSeconds): a GPU that was at rest raises its clock
// during about a second of continuous work (an RX 9060 runs at 2.0 GHz for the first 0.7 s and at 3.0 GHz after 1 s), and a fused kernel that is
// bound by the core clock is 20 percent slower until then.
inline Stats measure(const std::function<void()>& fn, const int warmup, const int iters, const double minWarmupSeconds = 0.0, const bool steady = false,
                     const double maxWarmupSeconds = 10.0) {
    using Clock = std::chrono::steady_clock;
    const double clockAtStart = amdGpuClockMHz();
    const auto begin = Clock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(Clock::now() - begin).count(); };
    constexpr double WINDOW = 0.25, TOLERANCE_STEADY = 0.02;
    int runs = 0, inWindow = 0;
    double windowStart = 0, windowSum = 0, previousMean = -1;
    bool isSteady = !steady;
    while (!(runs >= warmup && elapsed() >= minWarmupSeconds && (isSteady || elapsed() >= maxWarmupSeconds))) {
        const auto start = Clock::now();
        fn();
        windowSum += std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        ++runs;
        ++inWindow;
        if (elapsed() - windowStart >= WINDOW) {
            const double mean = windowSum / inWindow;
            isSteady = !steady || (previousMean > 0 && std::abs(mean - previousMean) <= TOLERANCE_STEADY * previousMean);
            previousMean = mean;
            windowSum = 0;
            inWindow = 0;
            windowStart = elapsed();
        }
    }
    const double warmupSeconds = elapsed();
    const double clockAtTimedStart = amdGpuClockMHz();
    std::vector<double> ms(static_cast<size_t>(iters));
    for (auto& t : ms) {
        const auto start = Clock::now();
        fn();
        const auto stop = Clock::now();
        t = std::chrono::duration<double, std::milli>(stop - start).count();
    }
    std::sort(ms.begin(), ms.end());
    Stats s;
    s.iters = ms.size();
    s.min = ms.front();
    s.max = ms.back();
    s.median = ms.size() % 2 ? ms[ms.size() / 2] : 0.5 * (ms[ms.size() / 2 - 1] + ms[ms.size() / 2]);
    double sum = 0;
    for (const double t : ms) {
        sum += t;
    }
    s.mean = sum / static_cast<double>(ms.size());
    s.warmupSeconds = warmupSeconds;
    s.warmupSteady = isSteady;
    s.gpuClockStartMHz = clockAtStart;
    s.gpuClockTimedMHz = clockAtTimedStart;
    s.gpuClockEndMHz = amdGpuClockMHz();
    return s;
}

} // namespace nv12
