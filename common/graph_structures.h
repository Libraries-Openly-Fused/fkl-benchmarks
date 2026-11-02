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

#ifndef FKLB_COMMON_GRAPH_STRUCTURES_H
#define FKLB_COMMON_GRAPH_STRUCTURES_H

// The structures of a CUDA graph that holds the work of a number of independent crops, used by the baseline fairness check of the CUDA-Graphs series
// (opencv-cuda/graphs/graph_structure_probe.cu, fkl-self/graphs/hf_graph_structure_probe.cu). The work of one crop is given as enqueue(crop, stream), a callable that launches
// it on the given stream; every structure is built from the same callable:
//   child          the structure of the experiments: one stream capture per crop, each captured graph added to a parent graph as a child graph node, no dependencies between them
//   child_chained  the same child nodes, each depending on the previous one
//   flat_serial    one stream capture of the work of all the crops, in order: only kernel nodes, one chain
//   flat_parallel  one capture with a fork and a join over one stream per crop: only kernel nodes, one independent chain per crop
// plain (no graph) is the work enqueued on a stream, and is timed by the caller.
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fklb {
namespace graphs {

using Clock = std::chrono::steady_clock;

inline void check(const cudaError_t err, const char* call, const char* file, const int line) {
    if (err != cudaSuccess) {
        std::ostringstream ss;
        ss << call << " failed: " << cudaGetErrorString(err) << " (" << file << ":" << line << ")";
        throw std::runtime_error(ss.str());
    }
}
#define FKLB_CUDA(call) ::fklb::graphs::check((call), #call, __FILE__, __LINE__)

inline double secondsBetween(const Clock::time_point& a, const Clock::time_point& b) { return std::chrono::duration<double>(b - a).count(); }
inline double microsecondsBetween(const Clock::time_point& a, const Clock::time_point& b) { return std::chrono::duration<double, std::micro>(b - a).count(); }

struct Graph {
    cudaGraph_t graph{nullptr};
    cudaGraphExec_t exec{nullptr};
    double captureSeconds{0.0};
    double instantiateSeconds{0.0};
    size_t topLevelNodes{0};
    Graph() = default;
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    ~Graph() {
        if (exec != nullptr) { cudaGraphExecDestroy(exec); }
        if (graph != nullptr) { cudaGraphDestroy(graph); }
    }
};

// the instantiation of the experiments: cudaGraphInstantiate(&exec, graph, NULL, NULL, 0)
inline void instantiate(Graph& g) {
    FKLB_CUDA(cudaGraphGetNodes(g.graph, nullptr, &g.topLevelNodes));
    const auto t0 = Clock::now();
    FKLB_CUDA(cudaGraphInstantiate(&g.exec, g.graph, nullptr, nullptr, 0));
    g.instantiateSeconds = secondsBetween(t0, Clock::now());
}

template <class Enqueue>
void buildChild(Graph& g, const int crops, Enqueue&& enqueue, const bool chained) {
    const auto t0 = Clock::now();
    FKLB_CUDA(cudaGraphCreate(&g.graph, 0));
    cudaGraphNode_t previous = nullptr;
    for (int crop = 0; crop < crops; ++crop) {
        cudaStream_t stream;
        FKLB_CUDA(cudaStreamCreate(&stream));
        cudaGraph_t sub;
        FKLB_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        enqueue(crop, stream);
        FKLB_CUDA(cudaStreamEndCapture(stream, &sub));
        cudaGraphNode_t node;
        if (chained && previous != nullptr) {
            FKLB_CUDA(cudaGraphAddChildGraphNode(&node, g.graph, &previous, 1, sub));
        } else {
            FKLB_CUDA(cudaGraphAddChildGraphNode(&node, g.graph, nullptr, 0, sub));
        }
        FKLB_CUDA(cudaGraphDestroy(sub));
        FKLB_CUDA(cudaStreamDestroy(stream));
        previous = node;
    }
    g.captureSeconds = secondsBetween(t0, Clock::now());
    instantiate(g);
}

template <class Enqueue>
void buildFlatSerial(Graph& g, const int crops, Enqueue&& enqueue) {
    cudaStream_t stream;
    FKLB_CUDA(cudaStreamCreate(&stream));
    const auto t0 = Clock::now();
    FKLB_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int crop = 0; crop < crops; ++crop) { enqueue(crop, stream); }
    FKLB_CUDA(cudaStreamEndCapture(stream, &g.graph));
    g.captureSeconds = secondsBetween(t0, Clock::now());
    FKLB_CUDA(cudaStreamDestroy(stream));
    instantiate(g);
}

template <class Enqueue>
void buildFlatParallel(Graph& g, const int crops, Enqueue&& enqueue) {
    cudaStream_t origin;
    FKLB_CUDA(cudaStreamCreate(&origin));
    cudaEvent_t fork;
    FKLB_CUDA(cudaEventCreateWithFlags(&fork, cudaEventDisableTiming));
    std::vector<cudaStream_t> branch(crops);
    std::vector<cudaEvent_t> join(crops);
    for (int crop = 0; crop < crops; ++crop) {
        FKLB_CUDA(cudaStreamCreate(&branch[crop]));
        FKLB_CUDA(cudaEventCreateWithFlags(&join[crop], cudaEventDisableTiming));
    }
    const auto t0 = Clock::now();
    FKLB_CUDA(cudaStreamBeginCapture(origin, cudaStreamCaptureModeGlobal));
    FKLB_CUDA(cudaEventRecord(fork, origin));
    for (int crop = 0; crop < crops; ++crop) {
        FKLB_CUDA(cudaStreamWaitEvent(branch[crop], fork, 0));
        enqueue(crop, branch[crop]);
        FKLB_CUDA(cudaEventRecord(join[crop], branch[crop]));
    }
    for (int crop = 0; crop < crops; ++crop) { FKLB_CUDA(cudaStreamWaitEvent(origin, join[crop], 0)); }
    FKLB_CUDA(cudaStreamEndCapture(origin, &g.graph));
    g.captureSeconds = secondsBetween(t0, Clock::now());
    for (int crop = 0; crop < crops; ++crop) {
        FKLB_CUDA(cudaStreamDestroy(branch[crop]));
        FKLB_CUDA(cudaEventDestroy(join[crop]));
    }
    FKLB_CUDA(cudaEventDestroy(fork));
    FKLB_CUDA(cudaStreamDestroy(origin));
    instantiate(g);
}

inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
inline double mean(const std::vector<double>& v) {
    double s = 0.0;
    for (const double x : v) { s += x; }
    return s / static_cast<double>(v.size());
}

// The times of the launches of one structure: the host time of the call(s) that submit the work, the GPU time between two events recorded on the stream around the submission (what the
// experiments report), and the host time from before the submission to after cudaStreamSynchronize.
struct Series {
    std::vector<double> submitUs, gpuMs, totalMs;
};

// One launch: events around the submission, then the synchronisation of the stream
template <class Submit>
void launchOnce(Submit&& submit, cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, Series& out) {
    FKLB_CUDA(cudaEventRecord(start, stream));
    const auto t0 = Clock::now();
    submit();
    const auto t1 = Clock::now();
    FKLB_CUDA(cudaEventRecord(stop, stream));
    FKLB_CUDA(cudaStreamSynchronize(stream));
    const auto t2 = Clock::now();
    float ms = 0.f;
    FKLB_CUDA(cudaEventElapsedTime(&ms, start, stop));
    out.submitUs.push_back(microsecondsBetween(t0, t1));
    out.gpuMs.push_back(static_cast<double>(ms));
    out.totalMs.push_back(microsecondsBetween(t0, t2) / 1000.0);
}

// The warm-up rule of the other benchmarks of the repository: at least minSeconds, and until two consecutive windows of 0.25 s agree within 2 % (a launch longer than the window is a window
// of its own); at most capSeconds. Returns the seconds it took; steady tells whether the rule was met.
template <class Submit>
double warmUp(Submit&& submit, cudaStream_t stream, cudaEvent_t start, cudaEvent_t stop, const double minSeconds, const double capSeconds, bool& steady) {
    const auto t0 = Clock::now();
    double previousMean = -1.0;
    steady = false;
    for (;;) {
        const auto windowStart = Clock::now();
        Series window;
        do {
            launchOnce(submit, stream, start, stop, window);
        } while (secondsBetween(windowStart, Clock::now()) < 0.25);
        const double windowMean = mean(window.totalMs);
        const double elapsed = secondsBetween(t0, Clock::now());
        if (elapsed >= minSeconds && previousMean > 0.0 && std::fabs(windowMean - previousMean) <= 0.02 * previousMean) {
            steady = true;
            return elapsed;
        }
        if (elapsed >= capSeconds) { return elapsed; }
        previousMean = windowMean;
    }
}

// The columns of the CSV of both probes (POINT names the first one: the operations per crop, or the number of crops)
#define FKLB_GRAPH_CSV_COLUMNS(POINT)                                                                                                                                                    \
    { POINT, "structure", "kernel_nodes", "top_level_nodes", "capture_s", "instantiate_s", "first_launch_ms", "warmup_s", "warmup_steady", "reps", "submit_us_median", "submit_us_min",   \
      "submit_us_max", "gpu_ms_median", "gpu_ms_mean", "gpu_ms_min", "gpu_ms_max", "total_ms_median", "total_ms_mean", "total_ms_min", "total_ms_max", "bit_exact_vs_plain" }

struct Options {
    int reps{100};
    int plainReps{100};
    double warmupMinSeconds{2.0};
    double warmupCapSeconds{10.0};
    std::string structures{"plain,child,child_chained,flat_serial,flat_parallel"};
    // FKLB_GRAPH_INTERLEAVE=1: between two timed launches (outside the timing) the work of crop 0 is launched once and the stream synchronised, the way the loop of the experiments runs
    // the other variant between two launches of the one it times (a host-bound launch can depend on what the host did in between)
    bool interleave{false};
    bool wants(const std::string& name) const { return ("," + structures + ",").find("," + name + ",") != std::string::npos; }
    static std::string env(const char* name, const std::string& fallback) {
        const char* value = std::getenv(name);
        return value != nullptr ? std::string(value) : fallback;
    }
    // FKLB_GRAPH_REPS, FKLB_GRAPH_PLAIN_REPS, FKLB_GRAPH_WARMUP_MIN_S, FKLB_GRAPH_WARMUP_CAP_S, FKLB_GRAPH_STRUCTURES
    static Options fromEnvironment() {
        Options o;
        o.reps = std::atoi(env("FKLB_GRAPH_REPS", "100").c_str());
        o.plainReps = std::atoi(env("FKLB_GRAPH_PLAIN_REPS", std::to_string(o.reps)).c_str());
        o.warmupMinSeconds = std::atof(env("FKLB_GRAPH_WARMUP_MIN_S", "2").c_str());
        o.warmupCapSeconds = std::atof(env("FKLB_GRAPH_WARMUP_CAP_S", "10").c_str());
        o.structures = env("FKLB_GRAPH_STRUCTURES", o.structures);
        o.interleave = env("FKLB_GRAPH_INTERLEAVE", "0") != "0";
        return o;
    }
};

// Times every structure for one point and writes one row per structure.
//   enqueue(crop, stream)   launches the work of one crop on the stream
//   clearOutputs()          sets every output to zero (so that a structure that does not write is caught)
//   downloadAll()           returns every output as one std::vector<float>
// Returns the number of structures that failed (the failure is reported on stderr and the others go on).
template <class Csv, class Enqueue, class ClearOutputs, class DownloadAll>
int runPoint(Csv& csv, const long point, const int crops, const long kernelNodes, Enqueue&& enqueue, ClearOutputs&& clearOutputs, DownloadAll&& downloadAll, const Options& options) {
    int failures = 0;
    cudaStream_t stream;
    FKLB_CUDA(cudaStreamCreate(&stream));
    cudaEvent_t start, stop;
    FKLB_CUDA(cudaEventCreate(&start));
    FKLB_CUDA(cudaEventCreate(&stop));
    const auto plainWork = [&] {
        for (int crop = 0; crop < crops; ++crop) { enqueue(crop, stream); }
    };
    clearOutputs();
    plainWork();
    FKLB_CUDA(cudaStreamSynchronize(stream));
    const std::vector<float> reference = downloadAll();

    for (const std::string name : {"plain", "child", "child_chained", "flat_serial", "flat_parallel"}) {
        if (!options.wants(name)) { continue; }
        try {
            Graph built;
            if (name == "child") { buildChild(built, crops, enqueue, false); }
            else if (name == "child_chained") { buildChild(built, crops, enqueue, true); }
            else if (name == "flat_serial") { buildFlatSerial(built, crops, enqueue); }
            else if (name == "flat_parallel") { buildFlatParallel(built, crops, enqueue); }
            const bool isPlain = name == "plain";
            const auto submit = [&] {
                if (isPlain) { plainWork(); } else { FKLB_CUDA(cudaGraphLaunch(built.exec, stream)); }
            };
            clearOutputs();
            FKLB_CUDA(cudaStreamSynchronize(stream));
            // the first launch uploads the graph
            Series first;
            launchOnce(submit, stream, start, stop, first);
            const bool exact = downloadAll() == reference;
            bool steady = false;
            const double warmupSeconds = warmUp(submit, stream, start, stop, options.warmupMinSeconds, options.warmupCapSeconds, steady);
            const int reps = isPlain ? options.plainReps : options.reps;
            Series timed;
            // a range per structure and point on the Nsight Systems timeline (outside the timed launches; with no tool attached it does nothing)
            const std::string rangeName = name + "@" + std::to_string(point);
            nvtxRangePushA(rangeName.c_str());
            for (int rep = 0; rep < reps; ++rep) {
                launchOnce(submit, stream, start, stop, timed);
                if (options.interleave) {
                    enqueue(0, stream);
                    FKLB_CUDA(cudaStreamSynchronize(stream));
                }
            }
            nvtxRangePop();
            const auto lo = [](const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); };
            const auto hi = [](const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); };
            csv.row(point, name, kernelNodes, built.topLevelNodes, built.captureSeconds, built.instantiateSeconds, first.gpuMs[0], warmupSeconds, steady ? 1 : 0, reps, median(timed.submitUs),
                    lo(timed.submitUs), hi(timed.submitUs), median(timed.gpuMs), mean(timed.gpuMs), lo(timed.gpuMs), hi(timed.gpuMs), median(timed.totalMs), mean(timed.totalMs), lo(timed.totalMs),
                    hi(timed.totalMs), exact ? 1 : 0);
            csv.flush();
            std::cout << "point " << point << " " << name << ": " << kernelNodes << " kernel nodes (" << built.topLevelNodes << " top level), capture " << built.captureSeconds << " s, instantiate "
                      << built.instantiateSeconds << " s, first launch " << first.gpuMs[0] << " ms, warm-up " << warmupSeconds << " s" << (steady ? "" : " (cap)") << "; " << reps
                      << " launches: submit " << median(timed.submitUs) << " us, gpu " << median(timed.gpuMs) << " ms, total " << median(timed.totalMs) << " ms; bit-exact " << exact << std::endl;
        } catch (const std::exception& e) {
            ++failures;
            std::cerr << "point " << point << " " << name << " failed: " << e.what() << std::endl;
            cudaGetLastError();
        }
    }
    FKLB_CUDA(cudaEventDestroy(start));
    FKLB_CUDA(cudaEventDestroy(stop));
    FKLB_CUDA(cudaStreamDestroy(stream));
    return failures;
}

}  // namespace graphs
}  // namespace fklb

#endif  // FKLB_COMMON_GRAPH_STRUCTURES_H
