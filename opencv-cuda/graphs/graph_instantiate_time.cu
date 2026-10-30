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

// E0, CUDA-Graphs range of the VF + HF sweep: how long does it take to build and instantiate the OpenCV-CUDA graph baseline
// of benchmark_vertical_fusionCompareMulAdd for a given number of operations? That baseline captures, for each of the
// REAL_BATCH = 50 crops, convertTo + NumOps / 2 x (multiply, add) into a child graph (mulAdd launcher.cu) and instantiates
// the parent graph, so a point of NumOps operations has 50 x (NumOps + 1) kernel nodes (about 1M at the 19,902 operations of
// the last point). The 2025 figure stopped near 5.4k operations. This probe builds exactly that graph for the NumOps given in
// FKLB_GRAPH_OPS (comma separated, default 502,2002,5002,10002) and writes one CSV row per NumOps: capture time, instantiate
// time (the number that decides how far the sweep can go), one launch, and the device memory the executable graph holds.
#include <common/main.h>
#include <common/csv_writer.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#define CHECK(call)                                                                                              \
    {                                                                                                            \
        const cudaError_t err_ = (call);                                                                         \
        if (err_ != cudaSuccess) {                                                                               \
            std::cerr << #call << " failed: " << cudaGetErrorString(err_) << " (" << __FILE__ << ":" << __LINE__ \
                      << ")" << std::endl;                                                                       \
            return -1;                                                                                           \
        }                                                                                                        \
    }

namespace {
using Clock = std::chrono::steady_clock;
double secondsSince(const Clock::time_point& t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}
}  // namespace

int launch() {
    constexpr int REAL_BATCH = 50;               // as in mul_add/realBatch.h
    constexpr int CROP_W = 60, CROP_H = 120;     // the crops of the VF + HF benchmark
    const cv::Scalar val_mul(0.54f);             // the benchmark's multiplier / addend
    const double alpha = 0.3;

    std::vector<size_t> opsList;
    {
        const char* env = std::getenv("FKLB_GRAPH_OPS");
        std::stringstream ss(env ? env : "502,2002,5002,10002");
        std::string item;
        while (std::getline(ss, item, ',')) {
            opsList.push_back(static_cast<size_t>(std::stoull(item)));
        }
    }

    cv::cuda::GpuMat d_input(2160, 4096, CV_8UC1, cv::Scalar(30));
    std::vector<cv::cuda::GpuMat> crops(REAL_BATCH), d_output(REAL_BATCH);
    for (int i = 0; i < REAL_BATCH; ++i) {
        crops[i] = d_input(cv::Rect(i, i, CROP_W, CROP_H));
        d_output[i].create(CROP_H, CROP_W, CV_32FC1);
    }
    cv::cuda::Stream cv_stream;

    fklb::CsvWriter csv("graph_instantiate_time.csv", "E0 CUDA Graphs: capture + instantiate time of the OpenCV baseline graph",
                        {"ops", "batch", "kernel_nodes", "capture_s", "instantiate_s", "launch_ms", "graph_device_MiB"});

    for (const size_t numOps : opsList) {
        size_t freeBefore = 0, total = 0, freeAfter = 0;
        CHECK(cudaMemGetInfo(&freeBefore, &total));

        const auto tCapture = Clock::now();
        cudaGraph_t mainGraph;
        CHECK(cudaGraphCreate(&mainGraph, 0));
        std::vector<cudaGraphNode_t> subGraphNodes(REAL_BATCH);
        for (int crop_i = 0; crop_i < REAL_BATCH; ++crop_i) {
            cv::cuda::Stream cuStream;
            cudaGraph_t subGraph;
            CHECK(cudaStreamBeginCapture(cv::cuda::StreamAccessor::getStream(cuStream), cudaStreamCaptureModeGlobal));
            crops[crop_i].convertTo(d_output[crop_i], CV_32FC1, alpha, cuStream);
            for (size_t op = 0; op < numOps; op += 2) {
                cv::cuda::multiply(d_output[crop_i], val_mul, d_output[crop_i], 1.0, -1, cuStream);
                cv::cuda::add(d_output[crop_i], val_mul, d_output[crop_i], cv::noArray(), -1, cuStream);
            }
            CHECK(cudaStreamEndCapture(cv::cuda::StreamAccessor::getStream(cuStream), &subGraph));
            CHECK(cudaGraphAddChildGraphNode(&subGraphNodes[crop_i], mainGraph, nullptr, 0, subGraph));
            CHECK(cudaGraphDestroy(subGraph));
        }
        const double captureS = secondsSince(tCapture);

        const size_t kernelNodes = static_cast<size_t>(REAL_BATCH) * (numOps + 1);

        const auto tInst = Clock::now();
        cudaGraphExec_t graphExec;
        CHECK(cudaGraphInstantiate(&graphExec, mainGraph, nullptr, nullptr, 0));
        const double instS = secondsSince(tInst);

        CHECK(cudaMemGetInfo(&freeAfter, &total));
        cudaEvent_t start, stop;
        CHECK(cudaEventCreate(&start));
        CHECK(cudaEventCreate(&stop));
        cudaStream_t stream = cv::cuda::StreamAccessor::getStream(cv_stream);
        CHECK(cudaGraphLaunch(graphExec, stream));  // first launch: uploads the graph
        CHECK(cudaStreamSynchronize(stream));
        CHECK(cudaEventRecord(start, stream));
        CHECK(cudaGraphLaunch(graphExec, stream));
        CHECK(cudaEventRecord(stop, stream));
        CHECK(cudaEventSynchronize(stop));
        float launchMs = 0.f;
        CHECK(cudaEventElapsedTime(&launchMs, start, stop));

        csv.row(numOps, REAL_BATCH, kernelNodes, captureS, instS, launchMs,
                (static_cast<double>(freeBefore) - static_cast<double>(freeAfter)) / (1024.0 * 1024.0));
        csv.flush();
        std::cout << "ops " << numOps << ": " << kernelNodes << " kernel nodes, capture " << captureS << " s, instantiate "
                  << instS << " s, launch " << launchMs << " ms" << std::endl;

        CHECK(cudaGraphExecDestroy(graphExec));
        CHECK(cudaGraphDestroy(mainGraph));
        CHECK(cudaEventDestroy(start));
        CHECK(cudaEventDestroy(stop));
    }
    return 0;
}
