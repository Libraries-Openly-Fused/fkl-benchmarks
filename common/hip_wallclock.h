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
//   first 0.7 s and at 3.0 GHz after 1 s), and a kernel that is bound by the core clock is 20 percent slower until then. The graphics clock, the
//   memory clock, the GPU function's own PCIe link and the PCIe link to the host (speed and width) are read at the start of the warm-up, at the
//   start of the timed runs and after them, so that a file records them.
//
// The device is the first AMD GPU (sysfs /sys/class/drm/card*/device with the vendor 0x1002) whose hwmon has freq1_input. Each function returns -1
// when it cannot read its value:
//   fklb::amdGpuClockMHz()     the graphics clock in MHz (hwmon freq1_input, labelled sclk);
//   fklb::amdGpuMemClockMHz()  the memory clock in MHz (hwmon freq2_input, labelled mclk);
//   fklb::amdCardLinkGTs()     the speed of the GPU function's own PCIe link in GT/s (current_link_speed of the device, for example
//                              "32.0 GT/s PCIe"). On a card with its own PCIe switch, as the RX 9060, that is the link inside the card,
//                              from the GPU to the switch, not the link to the host;
//   fklb::amdCardLinkWidth()   the width of that link in lanes (current_link_width of the device);
//   fklb::amdHostLinkGTs()     the speed of the link to the host in GT/s: current_link_speed of the root port above the device (the PCI device
//                              directly below the host bridge in the device's sysfs path, for example 0000:00:03.1);
//   fklb::amdHostLinkWidth()   the width of that link in lanes.
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
    double memClockStartMHz{-1}, memClockTimedMHz{-1}, memClockEndMHz{-1};
    double cardLinkGTsStart{-1}, cardLinkGTsTimed{-1}, cardLinkGTsEnd{-1};       // the GPU function's own link (inside the card)
    double cardLinkWidthStart{-1}, cardLinkWidthTimed{-1}, cardLinkWidthEnd{-1};
    double hostLinkGTsStart{-1}, hostLinkGTsTimed{-1}, hostLinkGTsEnd{-1};       // the root port's link (to the host)
    double hostLinkWidthStart{-1}, hostLinkWidthTimed{-1}, hostLinkWidthEnd{-1};
};

namespace detail {
// The device directory of the first AMD GPU whose hwmon has freq1_input, and that hwmon directory; empty strings when there is none
struct AmdGpuPaths {
    std::string device, hwmon, rootPort;
};
// the root port above a device: the PCI device directly below the host bridge (pciDDDD:BB) in the device's canonical sysfs path
inline std::string rootPortOf(const std::filesystem::path& device) {
    std::error_code ec;
    const std::filesystem::path canonical = std::filesystem::canonical(device, ec);
    if (ec) {
        return std::string();
    }
    std::filesystem::path p;
    bool belowBridge = false;
    for (const auto& part : canonical) {
        p /= part;
        if (belowBridge) {
            return p.string();
        }
        belowBridge = part.string().rfind("pci", 0) == 0;
    }
    return std::string();
}
inline const AmdGpuPaths& amdGpuPaths() {
    static const AmdGpuPaths paths = [] {
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
                    return AmdGpuPaths{device.string(), hw.path().string(), rootPortOf(device)};
                }
            }
        }
        return AmdGpuPaths{};
    }();
    return paths;
}
// the number at the start of a sysfs file ("32.0 GT/s PCIe" gives 32), -1 when the file is missing or does not start with a number
inline double readLeadingNumber(const std::string& directory, const char* file) {
    if (directory.empty()) {
        return -1.0;
    }
    std::ifstream f(std::filesystem::path(directory) / file);
    double value = 0;
    return (f >> value) ? value : -1.0;
}
}  // namespace detail

inline double amdGpuClockMHz() {
    const double hz = detail::readLeadingNumber(detail::amdGpuPaths().hwmon, "freq1_input");
    return hz < 0 ? -1.0 : hz / 1e6;
}

inline double amdGpuMemClockMHz() {
    const double hz = detail::readLeadingNumber(detail::amdGpuPaths().hwmon, "freq2_input");
    return hz < 0 ? -1.0 : hz / 1e6;
}

inline double amdCardLinkGTs() {
    return detail::readLeadingNumber(detail::amdGpuPaths().device, "current_link_speed");
}

inline double amdCardLinkWidth() {
    return detail::readLeadingNumber(detail::amdGpuPaths().device, "current_link_width");
}

inline double amdHostLinkGTs() {
    return detail::readLeadingNumber(detail::amdGpuPaths().rootPort, "current_link_speed");
}

inline double amdHostLinkWidth() {
    return detail::readLeadingNumber(detail::amdGpuPaths().rootPort, "current_link_width");
}

namespace detail {
inline double medianOfSorted(const std::vector<double>& v) {
    return v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]);
}
}  // namespace detail

inline WallStats measureWall(const std::function<void(WallClock::time_point&)>& run, const int warmup, const int iters,
                             const double minWarmupSeconds = 0.0, const bool steady = false, const double maxWarmupSeconds = 10.0) {
    const double clockAtStart = amdGpuClockMHz(), memClockAtStart = amdGpuMemClockMHz(), cardGTsAtStart = amdCardLinkGTs(),
                 cardWidthAtStart = amdCardLinkWidth(), hostGTsAtStart = amdHostLinkGTs(), hostWidthAtStart = amdHostLinkWidth();
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
    const double clockAtTimedStart = amdGpuClockMHz(), memClockAtTimedStart = amdGpuMemClockMHz(), cardGTsAtTimedStart = amdCardLinkGTs(),
                 cardWidthAtTimedStart = amdCardLinkWidth(), hostGTsAtTimedStart = amdHostLinkGTs(), hostWidthAtTimedStart = amdHostLinkWidth();
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
    s.memClockStartMHz = memClockAtStart;
    s.memClockTimedMHz = memClockAtTimedStart;
    s.memClockEndMHz = amdGpuMemClockMHz();
    s.cardLinkGTsStart = cardGTsAtStart;
    s.cardLinkGTsTimed = cardGTsAtTimedStart;
    s.cardLinkGTsEnd = amdCardLinkGTs();
    s.cardLinkWidthStart = cardWidthAtStart;
    s.cardLinkWidthTimed = cardWidthAtTimedStart;
    s.cardLinkWidthEnd = amdCardLinkWidth();
    s.hostLinkGTsStart = hostGTsAtStart;
    s.hostLinkGTsTimed = hostGTsAtTimedStart;
    s.hostLinkGTsEnd = amdHostLinkGTs();
    s.hostLinkWidthStart = hostWidthAtStart;
    s.hostLinkWidthTimed = hostWidthAtTimedStart;
    s.hostLinkWidthEnd = amdHostLinkWidth();
    return s;
}

}  // namespace fklb
