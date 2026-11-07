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

// CUDA event timing of launches. A failed CUDA call throws (common/status_check.h).
//
// fklb::timeIt(stream, launch, timing): e1::timeIt of handwritten-cuda/e1_common.h, moved here. One warm-up launch, then ITERATIONS launches,
//   each between two CUDA events recorded on `stream` and synchronised; mean and min in ms. Whatever launch() does on the host before it
//   enqueues its work is inside the interval: the pipeline experiments build their batch and their operations inside launch() (Oscar's
//   answer 1 of 2026-11-07). Unlike e1::timeIt it always times: there is no early return in profile mode; a program that must not time in
//   a profiling pass says so where it calls.
//
// fklb::timePreparedLaunch(stream, prepare, launch, gpu, host): the kernel comparisons (E1, E2), where each side's host preparation is timed
//   apart from its kernel (Oscar's answer 2 of 2026-11-07). In every iteration the host clock (std::chrono::steady_clock) measures
//   args = prepare(); then two CUDA events bracket launch(args) only, so that no host work sits between the GPU events. prepare() runs in
//   every iteration, so every value that depends on where the crops are is rebuilt for every image (answer 3). One warm-up iteration first.
//   gpu: the time of launch(args); host: the time of prepare(); both mean and min in ms over ITERATIONS. The arguments of an iteration are
//   destroyed after its stop event, outside both intervals. Two compiler barriers keep the compiler from moving the preparation out of its
//   interval. The two clock reads cost a few tens of nanoseconds, so a preparation shorter than that is measured as that.
#pragma once

#include <common/status_check.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>

namespace fklb {

constexpr int ITERATIONS = 100;

struct Timing {
    double mean = 0, min = 1e30;   // ms
};

template <typename Launch>
void timeIt(cudaStream_t stream, const Launch& launch, Timing& t) {
    cudaEvent_t start, stop;
    FKLB_CUDA_CHECK(cudaEventCreate(&start));
    FKLB_CUDA_CHECK(cudaEventCreate(&stop));
    launch();  // warm-up
    FKLB_CUDA_CHECK(cudaStreamSynchronize(stream));
    double sum = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        FKLB_CUDA_CHECK(cudaEventRecord(start, stream));
        launch();
        FKLB_CUDA_CHECK(cudaEventRecord(stop, stream));
        FKLB_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        sum += ms;
        t.min = std::min<double>(t.min, ms);
    }
    t.mean = sum / ITERATIONS;
    FKLB_CUDA_CHECK(cudaGetLastError());
    FKLB_CUDA_CHECK(cudaEventDestroy(start));
    FKLB_CUDA_CHECK(cudaEventDestroy(stop));
}

// the compiler must finish building `value` before this point (GCC and Clang syntax: the host compilers of this repository)
template <typename T>
inline void preparedBarrier(const T& value) {
    asm volatile("" : : "r"(&value) : "memory");
}

template <typename Prepare, typename Launch>
void timePreparedLaunch(cudaStream_t stream, const Prepare& prepare, const Launch& launch, Timing& gpu, Timing& host) {
    using Clock = std::chrono::steady_clock;
    cudaEvent_t start, stop;
    FKLB_CUDA_CHECK(cudaEventCreate(&start));
    FKLB_CUDA_CHECK(cudaEventCreate(&stop));
    {  // warm-up
        const auto args = prepare();
        launch(args);
        FKLB_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    double gpuSum = 0, hostSum = 0;
    for (int i = 0; i < ITERATIONS; ++i) {
        const auto h0 = Clock::now();
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto args = prepare();
        preparedBarrier(args);
        const auto h1 = Clock::now();
        FKLB_CUDA_CHECK(cudaEventRecord(start, stream));
        launch(args);
        FKLB_CUDA_CHECK(cudaEventRecord(stop, stream));
        FKLB_CUDA_CHECK(cudaEventSynchronize(stop));
        float ms = 0;
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double hostMs = std::chrono::duration<double, std::milli>(h1 - h0).count();
        gpuSum += ms;
        gpu.min = std::min<double>(gpu.min, ms);
        hostSum += hostMs;
        host.min = std::min(host.min, hostMs);
    }
    gpu.mean = gpuSum / ITERATIONS;
    host.mean = hostSum / ITERATIONS;
    FKLB_CUDA_CHECK(cudaGetLastError());
    FKLB_CUDA_CHECK(cudaEventDestroy(start));
    FKLB_CUDA_CHECK(cudaEventDestroy(stop));
}

}  // namespace fklb
