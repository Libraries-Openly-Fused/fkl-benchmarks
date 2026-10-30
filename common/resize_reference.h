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

// A plain scalar bilinear resize in double precision, for checking the output of a resize (FKL, OpenCV, NPP) against a
// reference that has no vectorisation, no fixed point and no fusion. Header only, host code, no library dependency,
// so it compiles with nvcc, amdclang++ and the host compiler.
//
// The two libraries do not sample at the same positions, so there are two references:
//   ResizeSampling::FKL           the destination pixel x reads the source position x * (src / dst)
//                                 (fused_kernel/algorithms/image_processing/resize.h, exec_resize, in fb9ac3a)
//   ResizeSampling::PIXEL_CENTRE  the centre of the destination pixel reads (x + 0.5) * (src / dst) - 0.5, as cv::resize
//                                 and cv::cuda::resize (INTER_LINEAR on float images) do
// Both clamp a position before the first pixel to the first pixel, and the second neighbour to the last pixel. They give
// the same image only when the input is constant.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace fklb {

enum class ResizeSampling { FKL, PIXEL_CENTRE };

// src: srcW x srcH pixels of `channels` interleaved values of type T, rows srcPitchBytes apart.
// dst: dstW * dstH * channels doubles, interleaved, rows dstW * channels apart.
template <typename T>
void resizeBilinear(const T* src, const size_t srcPitchBytes, const int srcW, const int srcH, const int channels,
                    const int dstW, const int dstH, const ResizeSampling sampling, double* dst) {
    const double scaleX = static_cast<double>(srcW) / dstW;
    const double scaleY = static_cast<double>(srcH) / dstH;
    const auto at = [&](const int x, const int y, const int c) {
        const T* row = reinterpret_cast<const T*>(reinterpret_cast<const char*>(src) + static_cast<size_t>(y) * srcPitchBytes);
        return static_cast<double>(row[static_cast<size_t>(x) * channels + c]);
    };
    for (int y = 0; y < dstH; ++y) {
        double fy = sampling == ResizeSampling::FKL ? y * scaleY : (y + 0.5) * scaleY - 0.5;
        fy = std::max(fy, 0.0);
        int y1 = static_cast<int>(std::floor(fy));
        double ay = fy - y1;
        if (y1 >= srcH - 1) {
            y1 = srcH - 1;
            ay = 0;
        }
        const int y2 = std::min(y1 + 1, srcH - 1);
        for (int x = 0; x < dstW; ++x) {
            double fx = sampling == ResizeSampling::FKL ? x * scaleX : (x + 0.5) * scaleX - 0.5;
            fx = std::max(fx, 0.0);
            int x1 = static_cast<int>(std::floor(fx));
            double ax = fx - x1;
            if (x1 >= srcW - 1) {
                x1 = srcW - 1;
                ax = 0;
            }
            const int x2 = std::min(x1 + 1, srcW - 1);
            for (int c = 0; c < channels; ++c) {
                dst[(static_cast<size_t>(y) * dstW + x) * channels + c] =
                    (at(x1, y1, c) * (1 - ax) + at(x2, y1, c) * ax) * (1 - ay) +
                    (at(x1, y2, c) * (1 - ax) + at(x2, y2, c) * ax) * ay;
            }
        }
    }
}

} // namespace fklb
