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

// Shared by the E1 and E2 programs: the profile switch, the order switch, the loop over the variants of a point, the GPU spin of the kernel-only
// timing, the CSV value of a time and the bit-exact comparison. The timers are those of common/cuda_timing.h, the status checks those of
// common/status_check.h.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace e1 {

// E1_PROFILE=1: no timing loops, every kernel is launched once by the correctness check and nothing else, so that a profiler
// (ncu, for the achieved occupancy of scripts/e1_table.py) sees one launch per kernel. The CSV is then not a timing result.
inline bool profileMode() {
    static const bool profile = std::getenv("E1_PROFILE") != nullptr && std::getenv("E1_PROFILE")[0] == '1';
    return profile;
}

// E1_ORDER=reverse: the variants of a point are timed in the reverse order (an order check: a variant that is timed first after an idle phase might see a
// lower clock or a cooler state; the CSV columns are the same). Default: the order of the list.
inline bool reverseOrder() {
    static const bool reverse = std::getenv("E1_ORDER") != nullptr && std::string(std::getenv("E1_ORDER")) == "reverse";
    return reverse;
}

// Times the variants of a point, one after the other, in the order of the list or in the reverse order (E1_ORDER=reverse): each element times
// one variant, with fklb::timePreparedLaunch, twice: as the launch runs (the dispatch of the launch is in the GPU interval when the GPU waits for
// it) and kernel-only (the launch queued behind gpuSpin before the start event).
inline void timeVariants(std::vector<std::function<void()>> variants) {
    if (reverseOrder()) std::reverse(variants.begin(), variants.end());
    for (auto& v : variants) v();
}

// The GPU spin of the kernel-only timing: one thread waits on the GPU's global timer and touches no memory, so the L2 stays as the previous launch
// left it. Enqueued before the start event (the beforeLaunch hook of fklb::timePreparedLaunch), it keeps the GPU busy while the host records the
// start event and enqueues the timed launch: the launch is already queued when the GPU reaches the start event, and the interval holds the
// kernel, not the host's dispatch of it. It has to last longer than the slowest dispatch of one call.
static __global__ void gpuSpin(const unsigned long long ns) {
    unsigned long long t0 = 0, t = 0;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

// E1_SPIN_US: the length of gpuSpin in microseconds, default 5000 (E1b's iterated version enqueues up to 1191 launches in one call, about 3 ms
// of host time; with that many launches the launch queue can fill before the spin ends, so its kernel-only time may still hold some dispatch).
inline unsigned long long spinNs() {
    static const unsigned long long ns = 1000ull * std::strtoull(std::getenv("E1_SPIN_US") ? std::getenv("E1_SPIN_US") : "5000", nullptr, 10);
    return ns;
}

// A time or a ratio of times as the CSV writes it (9 significant digits, as fklb::CsvWriter), empty in profile mode, where nothing is timed
inline std::string timed(const double value) {
    if (profileMode()) return std::string();
    std::ostringstream s;
    s << std::setprecision(9) << value;
    return s.str();
}

inline bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace e1
