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

// Shared by the E1 experiments: CUDA event timing of a launch (100 iterations after a warm-up launch) and the bit-exact comparison.
#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <vector>

#define E1_CHECK(call)                                                                                           \
    {                                                                                                            \
        const cudaError_t err_ = (call);                                                                         \
        if (err_ != cudaSuccess) {                                                                               \
            std::cerr << #call << " failed: " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ \
                      << ")" << std::endl;                                                                       \
            return false;                                                                                        \
        }                                                                                                        \
    }

namespace e1 {
constexpr int ITERATIONS = 100;

struct Timing {
    double mean = 0, min = 1e30;
};

template <typename Launch>
bool timeIt(cudaStream_t stream, const Launch& launch, Timing& t) {
    cudaEvent_t start, stop;
    E1_CHECK(cudaEventCreate(&start));
    E1_CHECK(cudaEventCreate(&stop));
    launch();  // warm-up
    E1_CHECK(cudaStreamSynchronize(stream));
    double sum = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        E1_CHECK(cudaEventRecord(start, stream));
        launch();
        E1_CHECK(cudaEventRecord(stop, stream));
        E1_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        E1_CHECK(cudaEventElapsedTime(&ms, start, stop));
        sum += ms;
        t.min = std::min<double>(t.min, ms);
    }
    t.mean = sum / ITERATIONS;
    E1_CHECK(cudaEventDestroy(start));
    E1_CHECK(cudaEventDestroy(stop));
    return true;
}

bool sameBits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}


}  // namespace e1
