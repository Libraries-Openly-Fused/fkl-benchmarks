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

// Helpers of the RPP benchmark (rpp/benchmark_rpp_preprocessing.h, HIP): the types of an 8 bit RGB image and of the operations on it, the linear
// resize, the host view of the memory of a Ptr, a PPM reader, and the wall clock timing of a function (the warm-up rule included) with the
// graphics clock of an AMD GPU.
#pragma once

#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/algorithms/basic_ops/memory_operations.h>
#include <fused_kernel/algorithms/image_processing/color_conversion.h>
#include <fused_kernel/algorithms/image_processing/image.h>
#include <fused_kernel/algorithms/image_processing/resize.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
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

namespace preproc {

using U3 = fk::uchar3;
using F3 = fk::float3;
using Rgb2D = fk::RawPtr<fk::ND::_2D, U3>;

// The linear resize: FKL's Resize samples the source at x * (src / dst) (ResizeSampling::ASYMMETRIC, the default)
using LinearResize = fk::Resize<fk::InterpolationType::INTER_LINEAR>;

// The host view of the memory of a Ptr: the pinned copy of a device Ptr, the memory itself otherwise
template <typename PtrType>
auto hostRaw(const PtrType& ptr) {
    const auto pinned = ptr.ptrPinned();
    return pinned.data != nullptr ? pinned : ptr.ptr();
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

inline Rgb2D roiOf(const Rgb2D& frame, const fk::Rect& rect) {
    Rgb2D roi = frame;
    roi.data = reinterpret_cast<U3*>(reinterpret_cast<char*>(frame.data) + static_cast<size_t>(rect.y) * frame.dims.pitch) + rect.x;
    roi.dims.width = static_cast<uint>(rect.width);
    roi.dims.height = static_cast<uint>(rect.height);
    return roi;
}

inline auto read3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadRead<fk::ND::_3D, F3>::build(p); }
inline auto write3D(const fk::RawPtr<fk::ND::_3D, F3>& p) { return fk::PerThreadWrite<fk::ND::_3D, F3>::build(p); }
inline auto split3D(const fk::RawPtr<fk::ND::_3D, float>& p) { return fk::TensorSplit<F3>::build(p); }

using Read3DOp = decltype(read3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using Write3DOp = decltype(write3D(std::declval<fk::RawPtr<fk::ND::_3D, F3>>()));
using SplitOp = decltype(split3D(std::declval<fk::RawPtr<fk::ND::_3D, float>>()));

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

} // namespace preproc
