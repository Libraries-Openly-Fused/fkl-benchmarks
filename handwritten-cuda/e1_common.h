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

// Shared by the E1 experiments: the profile switch, the order switch and the bit-exact comparison. The timers are those of
// common/cuda_timing.h, the status checks those of common/status_check.h.
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <functional>
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

// Times the variants of a point, one after the other, in the order of the list or in the reverse order (E1_ORDER=reverse): each element
// times one variant (fklb::timePreparedLaunch).
inline void timeVariants(std::vector<std::function<void()>> variants) {
    if (reverseOrder()) std::reverse(variants.begin(), variants.end());
    for (auto& v : variants) v();
}

inline bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

}  // namespace e1
