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

// Wall clock timing on the host (std::chrono::steady_clock) with a warm-up rule, and the graphics clock of an AMD GPU. Timing infrastructure of
// the HIP programs: it calls no library.
//
// fklb::measureWall(run, warmup, iters, minWarmupSeconds, steady, maxWarmupSeconds): run(split) is one timed run of the measured work and must end
// with the synchronisation of the streams it used, so that the interval holds the work. The warm-up is at least `warmup` runs and, when
// minWarmupSeconds > 0, at least that long; with steady = true it goes on, in windows of 0.25 s, until the mean time of two consecutive windows
// differs by at most 2 percent (up to maxWarmupSeconds). Then iters timed runs: median, min, max and mean of their time, in ms.
//   A run may set `split` to the time at which its first part ends, for example after building its batch. The medians of the two parts (from the
//   start of the run to split, and from split to its end) are reported as well; they are NaN when the run leaves split unset. The median of the
//   whole runs is not the sum of the medians of the parts.
//   Why the warm-up rule: a GPU that was at rest raises its clock during about a second of continuous work (an RX 9060 runs at 2.0 GHz for the
//   first 0.7 s and at 3.0 GHz after 1 s), and a kernel that is bound by the core clock is 20 percent slower until then. The graphics clock is
//   read at the start of the warm-up, at the start of the timed runs and after them, so that a file records it.
//
// fklb::amdGpuClockMHz(): the graphics clock of the first AMD GPU in MHz (sysfs hwmon freq1_input of the device with the vendor 0x1002), -1 when
// it cannot be read.
#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace fklb {

using WallClock = std::chrono::steady_clock;

struct WallStats {
    double median{0}, min{0}, max{0}, mean{0};   // of the whole runs, ms
    double firstMedian{std::numeric_limits<double>::quiet_NaN()};    // the medians of the two parts of a run that sets split, ms
    double secondMedian{std::numeric_limits<double>::quiet_NaN()};
    size_t iters{0};
    double warmupSeconds{0};   // the warm-up that was run, in seconds
    bool warmupSteady{true};   // false: the cap of the warm-up was reached before the time of the runs stopped changing
    double gpuClockStartMHz{-1}, gpuClockTimedMHz{-1}, gpuClockEndMHz{-1};
};

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

namespace detail {
inline double medianOfSorted(const std::vector<double>& v) {
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
}  // namespace detail

inline WallStats measureWall(const std::function<void(WallClock::time_point&)>& run, const int warmup, const int iters,
                             const double minWarmupSeconds = 0.0, const bool steady = false, const double maxWarmupSeconds = 10.0) {
    const double clockAtStart = amdGpuClockMHz();
    const auto begin = WallClock::now();
    const auto elapsed = [&] { return std::chrono::duration<double>(WallClock::now() - begin).count(); };
    constexpr double WINDOW = 0.25, TOLERANCE_STEADY = 0.02;
    int runs = 0, inWindow = 0;
    double windowStart = 0, windowSum = 0, previousMean = -1;
    bool isSteady = !steady;
    while (!(runs >= warmup && elapsed() >= minWarmupSeconds && (isSteady || elapsed() >= maxWarmupSeconds))) {
        WallClock::time_point split{};
        const auto start = WallClock::now();
        run(split);
        windowSum += std::chrono::duration<double, std::milli>(WallClock::now() - start).count();
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
    std::vector<double> total(static_cast<size_t>(iters)), first, second;
    bool everySplit = true;
    for (auto& t : total) {
        WallClock::time_point split{};
        const auto start = WallClock::now();
        run(split);
        const auto stop = WallClock::now();
        t = std::chrono::duration<double, std::milli>(stop - start).count();
        if (split == WallClock::time_point{}) {
            everySplit = false;
        } else {
            first.push_back(std::chrono::duration<double, std::milli>(split - start).count());
            second.push_back(std::chrono::duration<double, std::milli>(stop - split).count());
        }
    }
    WallStats s;
    std::sort(total.begin(), total.end());
    s.iters = total.size();
    s.min = total.front();
    s.max = total.back();
    s.median = detail::medianOfSorted(total);
    double sum = 0;
    for (const double t : total) {
        sum += t;
    }
    s.mean = sum / static_cast<double>(total.size());
    if (everySplit && !first.empty()) {
        std::sort(first.begin(), first.end());
        std::sort(second.begin(), second.end());
        s.firstMedian = detail::medianOfSorted(first);
        s.secondMedian = detail::medianOfSorted(second);
    }
    s.warmupSeconds = warmupSeconds;
    s.warmupSteady = isSteady;
    s.gpuClockStartMHz = clockAtStart;
    s.gpuClockTimedMHz = clockAtTimedStart;
    s.gpuClockEndMHz = amdGpuClockMHz();
    return s;
}

}  // namespace fklb
