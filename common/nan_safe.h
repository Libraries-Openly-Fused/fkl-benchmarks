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

// The maximum of a sequence of absolute differences, with NaN kept. std::max(m, d) returns m when d is NaN, and `if (d > m) m = d;` skips a NaN as well: a NaN in an output
// then never reached the maximum, and a check "maximum <= tolerance" passed with NaN values in the output. Here a NaN difference makes the result NaN for good, and
// NaN <= tolerance is false, so such a check fails.

#pragma once

#include <cmath>
#include <limits>

namespace fklb {

inline double maxKeepNaN(const double m, const double d) {
    return (std::isnan(m) || std::isnan(d)) ? std::numeric_limits<double>::quiet_NaN() : (d > m ? d : m);
}

}  // namespace fklb
