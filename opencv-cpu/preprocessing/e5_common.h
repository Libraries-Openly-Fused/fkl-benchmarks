// Timing and statistics shared by the E5 CPU benchmarks, and the CSV header: the system information of
// common/system_info.h (fklb::writeSystemInfoHeader) followed by the facts of the CPU run: governor, OpenCV threading and
// build, FKL thread pool. Read the CSVs with pandas.read_csv(path, comment='#', skipinitialspace=True).
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <common/system_info.h>

#include <fused_kernel/fused_kernel.h>
#include <fused_kernel/core/execution_model/cpu_multithread.h>

namespace e5 {

struct Stats {
    double median{0}, min{0}, max{0}, mean{0};
    size_t iters{0};
};

// Wall clock milliseconds of fn(), after warmup untimed calls, over iters timed calls
inline Stats measure(const std::function<void()>& fn, const int warmup, const int iters) {
    for (int i = 0; i < warmup; ++i) {
        fn();
    }
    std::vector<double> ms(static_cast<size_t>(iters));
    for (auto& t : ms) {
        const auto start = std::chrono::steady_clock::now();
        fn();
        const auto stop = std::chrono::steady_clock::now();
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
    return s;
}

inline std::string readFirstLine(const std::string& path) {
    std::ifstream f(path);
    std::string line;
    if (f && std::getline(f, line)) {
        return line;
    }
    return "unknown";
}

// "MemTotal" of /proc/meminfo, as GiB
inline std::string memTotal() {
    std::ifstream f("/proc/meminfo");
    std::string key;
    long long kb = 0;
    while (f >> key >> kb) {
        if (key == "MemTotal:") {
            return std::to_string(kb / 1024 / 1024) + " GiB visible";
        }
        f.ignore(256, '\n');
    }
    return "unknown";
}

// The lines of cv::getBuildInformation() that say how this OpenCV runs on the CPU
inline std::vector<std::string> openCvBuildLines() {
    const char* const keys[] = {"Version control:", "C++ standard:", "C++ Compiler:", "C++ flags (Release):",
                                "Baseline:", "Dispatched code generation:", "Parallel framework:", "Intel IPP:",
                                "Custom HAL:"};
    std::istringstream in(cv::getBuildInformation());
    std::vector<std::string> out;
    std::string line;
    while (std::getline(in, line)) {
        const auto first = line.find_first_not_of(' ');
        if (first == std::string::npos) {
            continue;
        }
        line = line.substr(first);
        for (const char* key : keys) {
            if (line.rfind(key, 0) == 0) {
                out.push_back(line);
                break;
            }
        }
    }
    return out;
}

struct Context {
    std::string experiment;
    std::string notes;
    int cvThreadsDefault{0};
    size_t fklPoolSize{0};
};

inline void writeSystemInfoHeader(std::ostream& os, const Context& ctx) {
    fklb::writeSystemInfoHeader(os, ctx.experiment, ctx.notes);
    os << "# cpu threads: " << std::thread::hardware_concurrency() << " logical, cpufreq max "
       << readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") << " kHz, governor "
       << readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") << ", driver "
       << readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/scaling_driver") << ", EPP "
       << readFirstLine("/sys/devices/system/cpu/cpu0/cpufreq/energy_performance_preference") << ", boost "
       << readFirstLine("/sys/devices/system/cpu/cpufreq/boost") << "\n";
    // The speed of the memory modules is only readable as root (dmidecode): E5_MEMORY_SPEED=<text> records it when it is known
    const char* memorySpeed = std::getenv("E5_MEMORY_SPEED");
    os << "# memory: " << memTotal() << ", speed " << (memorySpeed ? memorySpeed : "not recorded (dmidecode needs root)") << "\n";
    os << "# opencv default threads: " << ctx.cvThreadsDefault << " (cv::getNumThreads()), useOptimized "
       << cv::useOptimized() << ", useIPP " << cv::ipp::useIPP() << "\n";
    os << "# fkl cpu_multithread pool size: " << ctx.fklPoolSize << " (default hardware concurrency)\n";
    for (const auto& line : openCvBuildLines()) {
        os << "# opencv build: " << line << "\n";
    }
}

} // namespace e5
