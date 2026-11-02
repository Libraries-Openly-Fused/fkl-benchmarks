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

// Numbers from the environment (E5_ITERS, E5_WARMUP, E5_WARMUP_SECONDS, E5_ROTATE_OFFSET, E6_HOST ...) that are checked: an unset or empty variable gives the
// fallback; text, trailing characters, a value outside [lo, hi], and for envDouble a NaN or an infinity, are an error. The program says which variable and
// exits with status 2 (leading white space is accepted, as strtol and strtod do; trailing white space and a value that underflows, "1e-400", are errors), instead of running with the 0 that atoi makes of "abc" (E5_ITERS=0 left an empty vector, a negative value wrapped to a huge size_t, a NaN
// warm-up never ended).

#pragma once

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace fklb {

inline int envInt(const char* name, const int fallback, const int lo = 0, const int hi = 1000000) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    if (errno != 0 || end == value || *end != '\0' || parsed < lo || parsed > hi) {
        std::fprintf(stderr, "%s='%s' is not an integer in [%d, %d]\n", name, value, lo, hi);
        std::exit(2);
    }
    return static_cast<int>(parsed);
}

inline double envDouble(const char* name, const double fallback, const double lo, const double hi) {
    const char* value = std::getenv(name);
    if (value == nullptr || value[0] == '\0') {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    const double parsed = std::strtod(value, &end);
    if (errno != 0 || end == value || *end != '\0' || !std::isfinite(parsed) || parsed < lo || parsed > hi) {
        std::fprintf(stderr, "%s='%s' is not a number in [%g, %g]\n", name, value, lo, hi);
        std::exit(2);
    }
    return parsed;
}

}  // namespace fklb
