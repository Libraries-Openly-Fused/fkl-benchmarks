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

// E0, baseline fairness check of the CUDA-Graphs series of the VF + HF experiment (benchmark_vertical_fusionCompareMulAdd, mul_add/launcher.cu). That launcher captures, for each of the
// REAL_BATCH = 50 crops, convertTo + NumOps / 2 x (multiply, add) of OpenCV-CUDA into a graph (one stream capture per crop) and adds each of them to a parent graph as a child graph node, without
// dependencies between them. The VF-only launchers (mul_add1, mul1) make one stream capture of the whole chain instead. Does the structure of the graph change the time of the baseline?
//
// The probe runs the same work (the same OpenCV-CUDA calls on the same 60 x 120 crops with the same values) in five structures (common/graph_structures.h):
//   plain          the calls on one stream, no graph (the "OpenCV-CUDA" series of the experiment)
//   child          the structure of the experiment: one child graph node per crop, independent (the "OpenCV-CUDA + CUDA Graphs" series)
//   child_chained  the same child nodes, each depending on the previous one
//   flat_serial    one stream capture of the calls of all the crops, in order: kernel nodes only, one chain of 50 x (NumOps + 1) nodes
//   flat_parallel  one capture with a fork and a join over 50 streams: kernel nodes only, 50 independent chains
// For each NumOps (FKLB_GRAPH_OPS, comma separated, default 502,2502) and structure it writes one row to graph_structure_probe.csv: the nodes of the graph, the capture and the instantiate time, the first
// launch (it uploads the graph), the warm-up, and over FKLB_GRAPH_REPS launches (default 100) the median, mean, smallest and largest of the host submit time (the call of cudaGraphLaunch; all the calls for
// `plain`), the GPU time (events around the submission: what the experiment reports) and the total time including the synchronisation of the stream. Every structure is compared bit by bit with the
// output of `plain`. FKLB_GRAPH_STRUCTURES selects structures; FKLB_GRAPH_PLAIN_REPS, FKLB_GRAPH_WARMUP_MIN_S and FKLB_GRAPH_WARMUP_CAP_S the repetitions of `plain` and the warm-up rule.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/graph_structures.h>

#include <opencv2/core/cuda.hpp>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {
constexpr int REAL_BATCH = 50;               // as in mul_add/realBatch.h
constexpr int CROP_W = 60, CROP_H = 120;     // the crops of the VF + HF experiment (data size of mul_add: 60 x 120, one channel)

// The 50 crops of the experiment: an 8-bit input and a 32-bit float output each
struct Work {
    std::vector<cv::cuda::GpuMat> in, out;
    const cv::Scalar value{0.54f};   // the multiplier and the addend of the experiment
    Work() : in(REAL_BATCH), out(REAL_BATCH) {
        for (int i = 0; i < REAL_BATCH; ++i) {
            in[i] = cv::cuda::GpuMat(cv::Size(CROP_W, CROP_H), CV_8UC1, cv::Scalar(30u));
            out[i].create(cv::Size(CROP_W, CROP_H), CV_32FC1);
        }
    }
    // the calls of one crop, those of the experiment (alpha = 1.0)
    void enqueue(const int crop, cudaStream_t stream, const size_t numOps) {
        cv::cuda::Stream cvStream = cv::cuda::StreamAccessor::wrapStream(stream);
        in[crop].convertTo(out[crop], CV_32FC1, 1.0, cvStream);
        for (size_t op = 0; op < numOps; op += 2) {
            cv::cuda::multiply(out[crop], value, out[crop], 1.0, -1, cvStream);
            cv::cuda::add(out[crop], value, out[crop], cv::noArray(), -1, cvStream);
        }
    }
    void clearOutputs() {
        for (auto& o : out) { o.setTo(cv::Scalar(0.f)); }
        cudaDeviceSynchronize();
    }
    std::vector<float> downloadAll() {
        std::vector<float> all;
        for (auto& o : out) {
            cv::Mat h;
            o.download(h);
            const float* p = reinterpret_cast<const float*>(h.data);
            all.insert(all.end(), p, p + static_cast<size_t>(CROP_W) * CROP_H);
        }
        return all;
    }
};
}  // namespace

int launch() {
    std::vector<size_t> opsList;
    {
        const char* env = std::getenv("FKLB_GRAPH_OPS");
        std::stringstream ss(env != nullptr ? env : "502,2502");
        std::string item;
        while (std::getline(ss, item, ',')) { opsList.push_back(static_cast<size_t>(std::stoull(item))); }
    }
    const fklb::graphs::Options options = fklb::graphs::Options::fromEnvironment();
    int failures = 0;
    try {
        Work work;
        fklb::CsvWriter csv("graph_structure_probe.csv", "E0 CUDA Graphs: structure of the OpenCV-CUDA baseline of the VF + HF experiment (50 crops of 60 x 120, NumOps operations each)",
                            FKLB_GRAPH_CSV_COLUMNS("ops"),
                            "plain = the calls on one stream; child = one child graph node per crop, independent (the experiment's graph); child_chained = the child nodes one after the other; flat_serial = "
                            "one stream capture of all the calls; flat_parallel = one capture with a fork and a join over 50 streams. submit = host time of the launch call(s); gpu = events around the "
                            "launch; total = host time from before the launch to after cudaStreamSynchronize; first_launch_ms = GPU time of the first launch (uploads the graph).");
        for (const size_t numOps : opsList) {
            const long kernelNodes = static_cast<long>(REAL_BATCH) * static_cast<long>(numOps + 1);
            failures += fklb::graphs::runPoint(
                csv, static_cast<long>(numOps), REAL_BATCH, kernelNodes, [&](const int crop, cudaStream_t stream) { work.enqueue(crop, stream, numOps); }, [&] { work.clearOutputs(); },
                [&] { return work.downloadAll(); }, options);
        }
    } catch (const std::exception& e) {
        std::cerr << "graph_structure_probe: " << e.what() << std::endl;
        return -1;
    }
    return failures == 0 ? 0 : -1;
}
