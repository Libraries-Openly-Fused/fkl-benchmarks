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

// Verification of the NPP benchmarks with a non-constant source image.
//
// The benchmarks used to fill the source with a constant colour and require |FastNPP - NPP| < 1e-3. A constant image cannot tell
// two bilinear sampling conventions apart, and with a real image FastNPP (FKL: src = x * scale) and NPP (pixel centre:
// src = (x + 0.5) * scale - 0.5) differ by up to ~14 (see results/rtx4090/e6-resize-convention). So now:
//   - the source is seeded noise (the seed is written in the CSV header of the agreement file),
//   - the whole pipeline (crop, resize, channel swap, multiply, subtract, divide, split) is computed in double precision on the host
//     for each convention,
//   - each implementation must match ONE of the two references (|difference| < 1e-3); which one is recorded, so the check keeps
//     working if FKL adopts the other convention,
//   - the difference between the two implementations is recorded, not required to be small.
// The timing code of the benchmarks is not touched. One row per batch size is written to <function>_agreement.csv.
#pragma once

#include <common/system_info.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace fklb_npp {

constexpr unsigned long long NOISE_SEED = 20261030ull;
constexpr float AGREEMENT_TOLERANCE = 1e-3f;

// The source image: width x height uchar3 pixels (3 bytes each), seeded noise, generated once
inline const std::vector<unsigned char>& seededNoise(const int width, const int height) {
    static std::map<std::pair<int, int>, std::vector<unsigned char>> cache;
    auto it = cache.find({width, height});
    if (it == cache.end()) {
        std::mt19937_64 gen(NOISE_SEED);
        std::uniform_int_distribution<int> dist(0, 255);
        std::vector<unsigned char> img(static_cast<size_t>(width) * height * 3);
        for (auto& v : img) v = static_cast<unsigned char>(dist(gen));
        it = cache.emplace(std::make_pair(width, height), std::move(img)).first;
    }
    return it->second;
}

// Resize of the crop (x0, y0, cropW, cropH) of the noise image to upW x upH, in double precision, 3 channels:
// pixelCentre = false: FKL, src = x * scale, second read clamped, weights from the unclamped x1 + 1
// pixelCentre = true:  src = (x + 0.5) * scale - 0.5, both reads clamped (cv::resize, NPP)
inline std::vector<double> resizeCrop(const std::vector<unsigned char>& img, const int imgW, const int x0, const int y0,
                                      const int cropW, const int cropH, const int upW, const int upH, const bool pixelCentre) {
    auto px = [&](int x, int y, int c) { return static_cast<double>(img[(static_cast<size_t>(y0 + y) * imgW + (x0 + x)) * 3 + c]); };
    std::vector<double> out(static_cast<size_t>(upW) * upH * 3);
    const double fx = static_cast<double>(cropW) / upW, fy = static_cast<double>(cropH) / upH;
    for (int y = 0; y < upH; ++y) {
        for (int x = 0; x < upW; ++x) {
            for (int c = 0; c < 3; ++c) {
                double v;
                if (!pixelCentre) {
                    const double sx = x * fx, sy = y * fy;
                    const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
                    const int x2 = x1 + 1, y2 = y1 + 1;
                    const int x2r = std::min(x2, cropW - 1), y2r = std::min(y2, cropH - 1);
                    v = px(x1, y1, c) * ((x2 - sx) * (y2 - sy)) + px(x2r, y1, c) * ((sx - x1) * (y2 - sy)) +
                        px(x1, y2r, c) * ((x2 - sx) * (sy - y1)) + px(x2r, y2r, c) * ((sx - x1) * (sy - y1));
                } else {
                    const double sx = (x + 0.5) * fx - 0.5, sy = (y + 0.5) * fy - 0.5;
                    const int x1 = static_cast<int>(std::floor(sx)), y1 = static_cast<int>(std::floor(sy));
                    const double ax = sx - x1, ay = sy - y1;
                    const int xa = std::clamp(x1, 0, cropW - 1), xb = std::clamp(x1 + 1, 0, cropW - 1);
                    const int ya = std::clamp(y1, 0, cropH - 1), yb = std::clamp(y1 + 1, 0, cropH - 1);
                    v = px(xa, ya, c) * ((1 - ax) * (1 - ay)) + px(xb, ya, c) * (ax * (1 - ay)) +
                        px(xa, yb, c) * ((1 - ax) * ay) + px(xb, yb, c) * (ax * ay);
                }
                out[(static_cast<size_t>(y) * upW + x) * 3 + c] = v;
            }
        }
    }
    return out;
}

struct Agreement {
    double maxDiff = 0, sumDiff = 0;
    size_t n = 0;
    void add(const double d) { maxDiff = std::max(maxDiff, d); sumDiff += d; ++n; }
    double mean() const { return n ? sumDiff / n : 0.0; }
};

inline const char* convention(const Agreement& fkl, const Agreement& pc) {
    if (fkl.maxDiff < AGREEMENT_TOLERANCE) return "FKL";
    if (pc.maxDiff < AGREEMENT_TOLERANCE) return "pixel-centre";
    return "none";
}

// Checks one batch size. fastAt(j, c, i) / nppAt(j, c, i): the value of plane c, element i (row major, upW x upH) of image j
// from FastNPP / NPP. order, mul, sub, div: the channel order and the constants of the pipeline (output channel k is
// (resized[order[k]] * mul[k] - sub[k]) / div[k]). Returns true if both implementations match one of the references.
inline bool checkBatch(const std::string& function, const int batch, const int imgW, const int cropW, const int cropH, const int upW,
                       const int upH, const int* order, const float* mul, const float* sub, const float* div,
                       const std::function<float(int, int, int)>& fastAt, const std::function<float(int, int, int)>& nppAt,
                       const std::vector<unsigned char>& img) {
    Agreement fastFKL, fastPC, nppFKL, nppPC, cross;
    const int plane = upW * upH;
    for (int j = 0; j < batch; ++j) {
        const std::vector<double> rFKL = resizeCrop(img, imgW, j, j, cropW, cropH, upW, upH, false);
        const std::vector<double> rPC = resizeCrop(img, imgW, j, j, cropW, cropH, upW, upH, true);
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < plane; ++i) {
                const double refFKL = (rFKL[static_cast<size_t>(i) * 3 + order[c]] * mul[c] - sub[c]) / div[c];
                const double refPC = (rPC[static_cast<size_t>(i) * 3 + order[c]] * mul[c] - sub[c]) / div[c];
                const double f = fastAt(j, c, i), n = nppAt(j, c, i);
                fastFKL.add(std::fabs(f - refFKL));
                fastPC.add(std::fabs(f - refPC));
                nppFKL.add(std::fabs(n - refFKL));
                nppPC.add(std::fabs(n - refPC));
                cross.add(std::fabs(f - n));
            }
        }
    }
    const char* fastFollows = convention(fastFKL, fastPC);
    const char* nppFollows = convention(nppFKL, nppPC);
    const bool ok = std::string(fastFollows) != "none" && std::string(nppFollows) != "none";

    static std::map<std::string, std::ofstream> files;
    const std::string name = function + "_agreement.csv";
    auto it = files.find(name);
    if (it == files.end()) {
        it = files.emplace(name, std::ofstream(name)).first;
        fklb::writeSystemInfoHeader(it->second, function + " agreement",
                                    "source: seeded uniform noise (std::mt19937_64 seed " + std::to_string(NOISE_SEED) +
                                    "); references in double precision for the whole pipeline; each implementation must match one "
                                    "convention within 1e-3, the difference between them is informational");
        it->second << "batch,FastNPP_follows,NPP_follows,FastNPP_vs_ref_FKL_max,FastNPP_vs_ref_pixel_centre_max,NPP_vs_ref_FKL_max,"
                      "NPP_vs_ref_pixel_centre_max,FastNPP_vs_NPP_max,FastNPP_vs_NPP_mean\\n";
    }
    it->second << batch << ',' << fastFollows << ',' << nppFollows << ',' << fastFKL.maxDiff << ',' << fastPC.maxDiff << ','
               << nppFKL.maxDiff << ',' << nppPC.maxDiff << ',' << cross.maxDiff << ',' << cross.mean() << '\n';
    it->second.flush();
    std::cout << function << " batch " << batch << ": FastNPP follows the " << fastFollows << " convention, NPP follows the "
              << nppFollows << " convention; max |FastNPP - NPP| = " << cross.maxDiff << " (informational)" << std::endl;
    return ok;
}

}  // namespace fklb_npp
