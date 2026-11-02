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

// E0, baseline fairness check of the CUDA-Graphs series of the HF experiment (fkl-self/fusion/benchmark_horizontal_fusion.h, benchmark_Horizontal_Fusion_NO_CPU_OVERHEAD_CUDAGraphs). That benchmark
// builds, for each of the BATCH crops of 60 x 120 of a 3840 x 2160 image, a graph by one stream capture of the launch of the (one-kernel) pipeline of the crop (SaturateCast, Mul, Sub, Div) and adds
// each of them to a parent graph as a child graph node, without dependencies between them. This probe launches the same kernels with the same parameters (the ops are built before the timing, as in the
// benchmark) in the structures of common/graph_structures.h:
//   plain          the BATCH launches on one stream ("Iterated Batch" of the experiment)
//   child          the structure of the experiment: one child graph node per crop, independent
//   child_chained  the same child nodes, each depending on the previous one
//   flat_serial    one stream capture of the BATCH launches, in order: BATCH kernel nodes in one chain
//   flat_parallel  one capture with a fork and a join over BATCH streams: BATCH independent kernel nodes
// For each number of crops (FKLB_GRAPH_CROPS, comma separated, default 11,101,301,601,1191: points of the HF sweep, 1 + 10 k) and structure it writes one row to hf_graph_structure_probe.csv, with the
// columns of opencv-cuda/graphs/graph_structure_probe.cu (nodes, capture, instantiate, first launch, warm-up, and over FKLB_GRAPH_REPS launches the median, mean, smallest and largest of the host submit
// time, the GPU time between two events around the submission, and the total time including the synchronisation). Every structure is compared bit by bit with the output of `plain`.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/graph_structures.h>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/fused_kernel.h>

#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using InputType = unsigned char;
using OutputType = float;
constexpr int CROP_W = 60, CROP_H = 120;
constexpr int IMAGE_W = 3840, IMAGE_H = 2160;   // width x height, as the paper states

using ReadOp = std::decay_t<decltype(fk::PerThreadRead<fk::ND::_2D, InputType>::build(std::declval<fk::Ptr2D<InputType>&>()))>;
using WriteOp = std::decay_t<decltype(fk::PerThreadWrite<fk::ND::_2D, OutputType>::build(std::declval<fk::Ptr2D<OutputType>&>()))>;

// The wrapper of a CUDA stream that the launches of the library take; one is kept while the stream does not change (the plain launches and the flat serial capture use one stream)
struct StreamCache {
    cudaStream_t raw{nullptr};
    std::unique_ptr<fk::Stream> wrapped;
    fk::Stream& get(cudaStream_t stream) {
        if (!wrapped || raw != stream) {
            wrapped = std::make_unique<fk::Stream>(stream);
            raw = stream;
        }
        return *wrapped;
    }
};

// The crops of the experiment and the ops built before the timing
struct Work {
    fk::Stream setup;
    fk::Ptr2D<InputType> input;
    std::vector<fk::Ptr2D<InputType>> crops;
    std::vector<fk::Ptr2D<OutputType>> outputs;
    std::vector<ReadOp> reads;
    std::vector<WriteOp> writes;
    const decltype(fk::SaturateCast<InputType, OutputType>::build()) saturate{fk::SaturateCast<InputType, OutputType>::build()};
    const decltype(fk::Mul<OutputType>::build(1.0f)) mul{fk::Mul<OutputType>::build(1.0f)};
    const decltype(fk::Sub<OutputType>::build(1.f)) sub{fk::Sub<OutputType>::build(1.f)};
    const decltype(fk::Div<OutputType>::build(3.2f)) div{fk::Div<OutputType>::build(3.2f)};
    StreamCache streams;

    explicit Work(const int batch) : input(IMAGE_W, IMAGE_H) {
        fk::setTo(static_cast<InputType>(10u), input, setup);
        crops.resize(batch);
        outputs.resize(batch);
        reads.reserve(batch);
        writes.reserve(batch);
        for (int crop = 0; crop < batch; ++crop) {
            crops[crop] = input.crop(fk::Point{crop, crop, 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(CROP_W), static_cast<uint>(CROP_H), static_cast<uint>(input.dims().pitch)});
            outputs[crop].Alloc(fk::Size(CROP_W, CROP_H));
            reads.push_back(fk::PerThreadRead<fk::ND::_2D, InputType>::build(crops[crop]));
            writes.push_back(fk::PerThreadWrite<fk::ND::_2D, OutputType>::build(outputs[crop]));
        }
        setup.sync();
    }
    // the launch of the pipeline of one crop, as in the experiment
    void enqueue(const int crop, cudaStream_t stream) {
        fk::executeOperations<fk::TransformDPP<>>(streams.get(stream), reads[crop], saturate, mul, sub, div, writes[crop]);
    }
    void clearOutputs() {
        for (auto& output : outputs) {
            FKLB_CUDA(cudaMemset2D(output.ptr().data, output.dims().pitch, 0, static_cast<size_t>(CROP_W) * sizeof(OutputType), CROP_H));
        }
        FKLB_CUDA(cudaDeviceSynchronize());
    }
    std::vector<float> downloadAll() {
        std::vector<float> all(outputs.size() * static_cast<size_t>(CROP_W) * CROP_H);
        float* destination = all.data();
        for (auto& output : outputs) {
            FKLB_CUDA(cudaMemcpy2D(destination, static_cast<size_t>(CROP_W) * sizeof(OutputType), output.ptr().data, output.dims().pitch, static_cast<size_t>(CROP_W) * sizeof(OutputType), CROP_H,
                                   cudaMemcpyDeviceToHost));
            destination += static_cast<size_t>(CROP_W) * CROP_H;
        }
        return all;
    }
};
}  // namespace

int launch() {
    std::vector<int> cropsList;
    {
        const char* env = std::getenv("FKLB_GRAPH_CROPS");
        std::stringstream ss(env != nullptr ? env : "11,101,301,601,1191");
        std::string item;
        while (std::getline(ss, item, ',')) { cropsList.push_back(std::stoi(item)); }
    }
    const fklb::graphs::Options options = fklb::graphs::Options::fromEnvironment();
    int failures = 0;
    try {
        fklb::CsvWriter csv("hf_graph_structure_probe.csv", "E0 CUDA Graphs: structure of the baseline of the HF experiment (crops of 60 x 120 of a 3840 x 2160 image, one kernel per crop)",
                            FKLB_GRAPH_CSV_COLUMNS("crops"),
                            "plain = the launches on one stream; child = one child graph node per crop, independent (the experiment's graph); child_chained = the child nodes one after the other; flat_serial = "
                            "one stream capture of all the launches; flat_parallel = one capture with a fork and a join over one stream per crop. submit = host time of the launch call(s); gpu = events "
                            "around the launch; total = host time from before the launch to after cudaStreamSynchronize; first_launch_ms = GPU time of the first launch (uploads the graph).");
        for (const int crops : cropsList) {
            Work work(crops);
            failures += fklb::graphs::runPoint(
                csv, crops, crops, crops, [&](const int crop, cudaStream_t stream) { work.enqueue(crop, stream); }, [&] { work.clearOutputs(); }, [&] { return work.downloadAll(); }, options);
        }
    } catch (const std::exception& e) {
        std::cerr << "hf_graph_structure_probe: " << e.what() << std::endl;
        return -1;
    }
    return failures == 0 ? 0 : -1;
}
