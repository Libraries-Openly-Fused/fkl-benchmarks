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

// HIP event timing of launches: common/cuda_timing.h with the HIP runtime API, the same functions, intervals and constants, so that a HIP program
// measures what its CUDA twin measures. A failed HIP call throws (common/status_check.h). See common/cuda_timing.h for the description:
//   fklb::timeIt(stream, launch, timing): one warm-up launch, then ITERATIONS launches, each between two events recorded on `stream` and
//     synchronised; mean and min in ms (E4's interval).
//   fklb::timePreparedLaunch(stream, prepare, [beforeLaunch,] launch, gpu, host): the GPU interval of the launch alone, and the host time of
//     preparing and releasing its arguments (E1's interval when beforeLaunch enqueues a GPU spin).
#pragma once

#include <common/status_check.h>

#include <hip/hip_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>

namespace fklb {

constexpr int ITERATIONS = 100;

struct Timing {
    double mean = 0, min = 1e30;   // ms
};

template <typename Launch>
void timeIt(hipStream_t stream, const Launch& launch, Timing& t) {
    hipEvent_t start, stop;
    FKLB_HIP_CHECK(hipEventCreate(&start));
    FKLB_HIP_CHECK(hipEventCreate(&stop));
    launch();   // warm-up
    FKLB_HIP_CHECK(hipStreamSynchronize(stream));
    double sum = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        FKLB_HIP_CHECK(hipEventRecord(start, stream));
        launch();
        FKLB_HIP_CHECK(hipEventRecord(stop, stream));
        FKLB_HIP_CHECK(hipEventSynchronize(stop));
        float ms = 0;
        FKLB_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        sum += ms;
        t.min = std::min<double>(t.min, ms);
    }
    t.mean = sum / ITERATIONS;
    FKLB_HIP_CHECK(hipGetLastError());
    FKLB_HIP_CHECK(hipEventDestroy(start));
    FKLB_HIP_CHECK(hipEventDestroy(stop));
}

template <typename T>
inline void preparedBarrier(const T& value) {
    asm volatile("" : : "r"(&value) : "memory");
}

template <typename Prepare, typename BeforeLaunch, typename Launch>
void timePreparedLaunch(hipStream_t stream, const Prepare& prepare, const BeforeLaunch& beforeLaunch, const Launch& launch, Timing& gpu,
                        Timing& host) {
    using Clock = std::chrono::steady_clock;
    hipEvent_t start, stop;
    FKLB_HIP_CHECK(hipEventCreate(&start));
    FKLB_HIP_CHECK(hipEventCreate(&stop));
    {  // warm-up
        const auto args = prepare();
        launch(args);
        FKLB_HIP_CHECK(hipStreamSynchronize(stream));
    }
    double gpuSum = 0, hostSum = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        Clock::time_point h0, h1, h2;
        {
            h0 = Clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const auto args = prepare();   // built in place: no copy or move of the arguments
            preparedBarrier(args);
            h1 = Clock::now();
            beforeLaunch();
            FKLB_HIP_CHECK(hipEventRecord(start, stream));
            launch(args);
            FKLB_HIP_CHECK(hipEventRecord(stop, stream));
            FKLB_HIP_CHECK(hipEventSynchronize(stop));
            h2 = Clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }  // the arguments are released here, inside host 2
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto h3 = Clock::now();
        float ms = 0;
        FKLB_HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
        const double hostMs = std::chrono::duration<double, std::milli>((h1 - h0) + (h3 - h2)).count();
        gpuSum += ms;
        gpu.min = std::min<double>(gpu.min, ms);
        hostSum += hostMs;
        host.min = std::min(host.min, hostMs);
    }
    gpu.mean = gpuSum / ITERATIONS;
    host.mean = hostSum / ITERATIONS;
    FKLB_HIP_CHECK(hipGetLastError());
    FKLB_HIP_CHECK(hipEventDestroy(start));
    FKLB_HIP_CHECK(hipEventDestroy(stop));
}

template <typename Prepare, typename Launch>
void timePreparedLaunch(hipStream_t stream, const Prepare& prepare, const Launch& launch, Timing& gpu, Timing& host) {
    timePreparedLaunch(stream, prepare, [] {}, launch, gpu, host);
}

}  // namespace fklb
