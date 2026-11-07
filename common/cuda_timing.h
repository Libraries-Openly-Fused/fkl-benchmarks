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
// fklb::timeIt(stream, launch, timing): one warm-up launch, then ITERATIONS launches, each between two CUDA events recorded on `stream` and
//   synchronised; mean and min in ms. Whatever launch() does on the host before it enqueues its work is inside the interval: in a pipeline
//   experiment, building the batch and the operations is part of the measured time, so launch() builds them. It always times; a program
//   that must not time in a profiling pass says so where it calls.
//
// fklb::timePreparedLaunch(stream, prepare, launch, gpu, host): for the kernel comparisons, which report the GPU time of each side's kernel
//   and the host time of preparing its arguments separately. Every iteration has three intervals:
//     host 1: args = prepare()          (std::chrono::steady_clock)
//     GPU:    launch(args)              (two CUDA events, synchronised; no host work sits between them)
//     host 2: the release of args       (std::chrono::steady_clock)
//   host = host 1 + host 2: the cost of building and releasing the arguments of one image, as a per-image loop pays it (releasing a view
//   of a reference-counted buffer is not free); gpu = the GPU interval. prepare() runs in every iteration because the crop locations may
//   change on every image, so every value that depends on them is rebuilt for every image. One warm-up iteration first; mean and min over
//   ITERATIONS, in ms. prepare() returns the arguments by value; they are a local built in place (guaranteed copy elision), so no copy or
//   move of them is measured, and their scope ends between the GPU interval and the end of host 2. Compiler barriers keep the preparation
//   and the release inside their intervals. The clock reads cost a few tens of nanoseconds, so a preparation shorter than that is
//   measured as that.
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
        Clock::time_point h0, h1, h2;
        {
            h0 = Clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
            const auto args = prepare();   // built in place: no copy or move of the arguments
            preparedBarrier(args);
            h1 = Clock::now();
            FKLB_CUDA_CHECK(cudaEventRecord(start, stream));
            launch(args);
            FKLB_CUDA_CHECK(cudaEventRecord(stop, stream));
            FKLB_CUDA_CHECK(cudaEventSynchronize(stop));
            h2 = Clock::now();
            std::atomic_signal_fence(std::memory_order_seq_cst);
        }  // the arguments are released here, inside host 2
        std::atomic_signal_fence(std::memory_order_seq_cst);
        const auto h3 = Clock::now();
        float ms = 0;
        FKLB_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        const double hostMs = std::chrono::duration<double, std::milli>((h1 - h0) + (h3 - h2)).count();
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
