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

// E0, the time of the fused kernel of the HF experiment (fkl-self/fusion/benchmark_horizontal_fusion.h): what it costs in each loop of the experiment and on its own.
// In the experiment every baseline is timed in a loop that also times the fused kernel (TimeMarkerTwo<GPU_NVIDIA>, fkl-self/fkBenchmarksCommon.h), per iteration:
//     record(start); BASELINE; record(stop); synchronize(stop); first = elapsed(start, stop); record(start); FUSED; record(stop); synchronize(stop); second = elapsed(start, stop)
// and the two columns of the CSV are the means of first and second. The fused kernel is the same in every loop, yet its mean time differs between the CSV files of one run. This program runs the loop of the
// experiment with every structure of the baseline (hf_graph_structure_probe.cu: plain launches, graph of child nodes, chained child nodes, one capture, one capture with a fork and a join), and the fused
// kernel with
//   alone_synced                the loop with an empty baseline: the stream is idle when the fused kernel is launched, the way the loop of the experiment launches it
//   *_pollute2MB / 64MB         a host memory sweep of that size between the synchronisation and the launch (the caches of the host hold something else than the parameters of the launch)
//   *_touch                     the bytes of the operations of the fused kernel (its kernel parameters, 24 bytes per crop) read on the host before the launch (they are in the caches of the host)
//   alone_pipelined10 / 50      10 or 50 launches one after the other between two events, the time divided by the number: the kernel without the launch latency
//   alone_queued_behind_spin    the fused launch enqueued while a kernel of about 1 ms is running, the time between the events around the fused kernel only: no launch latency either
// It records for every iteration the "second" time (the fused kernel as the experiment times it), the "first" time (the baseline) and the host time of the call that launches the fused kernel.
// Crops: FKLB_FUSED_CROPS (comma separated, a subset of 101,301,661,831,1001,1101,1191, default all), iterations FKLB_FUSED_ITERS (500) after FKLB_FUSED_WARMUP (100) that are not recorded, FKLB_FUSED_MODES
// (comma separated prefixes of the mode names, default all). Output: hf_fused_time_probe.csv (a row per crops and mode) and hf_fused_time_probe_raw.csv (a row per iteration). The output of the fused
// kernel is compared bit by bit with the plain launches for three crops. Every timed loop is a range "<mode>@<crops>" on the Nsight Systems timeline.
#include <common/main.h>
#include <common/csv_writer.h>
#include <common/graph_structures.h>

#include <fused_kernel/core/data/ptr_nd.h>
#include <fused_kernel/core/data/ptr_utils.h>
#include <fused_kernel/algorithms/algorithms.h>
#include <fused_kernel/fused_kernel.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
using fklb::graphs::Clock;
using fklb::graphs::microsecondsBetween;
using InputType = unsigned char;
using OutputType = float;
constexpr int CROP_W = 60, CROP_H = 120;
constexpr int IMAGE_W = 3840, IMAGE_H = 2160;   // width x height, as the paper states
constexpr long long SPIN_CYCLES = 2700000;      // about 1 ms at 2.7 GHz

using ReadOp = std::decay_t<decltype(fk::PerThreadRead<fk::ND::_2D, InputType>::build(std::declval<fk::Ptr2D<InputType>&>()))>;
using WriteOp = std::decay_t<decltype(fk::PerThreadWrite<fk::ND::_2D, OutputType>::build(std::declval<fk::Ptr2D<OutputType>&>()))>;

__global__ void spinKernel(const long long cycles) {
    const long long begin = clock64();
    while (clock64() - begin < cycles) {
    }
}

// The wrapper of a CUDA stream that the launches of the library take; one is kept while the stream does not change
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

// A sink for the host memory sweeps and the reads of the parameters, so that the compiler keeps them
volatile unsigned long long g_sink = 0;

struct HostSweep {
    std::vector<unsigned char> buffer;
    explicit HostSweep(const size_t bytes) : buffer(bytes, 1) {}
    void run() {
        unsigned long long sum = 0;
        for (size_t i = 0; i < buffer.size(); i += 64) {
            buffer[i] = static_cast<unsigned char>(buffer[i] + 1);
            sum += buffer[i];
        }
        g_sink = sum;
    }
};

inline void touchBytes(const void* object, const size_t bytes) {
    const unsigned char* p = static_cast<const unsigned char*>(object);
    unsigned long long sum = 0;
    for (size_t i = 0; i < bytes; i += 64) { sum += p[i]; }
    g_sink = sum;
}

struct Samples {
    std::vector<float> firstMs, secondMs;
    std::vector<double> baselineUs, fusedUs;
};

double percentile(std::vector<double> v, const double p) {
    std::sort(v.begin(), v.end());
    return v[static_cast<size_t>(p * static_cast<double>(v.size() - 1))];
}

struct Options {
    int iters{500};
    int warmup{100};
    std::string modes;   // comma separated prefixes, empty = all
    bool wants(const std::string& name) const {
        if (modes.empty()) { return true; }
        std::stringstream ss(modes);
        std::string prefix;
        while (std::getline(ss, prefix, ',')) {
            if (name.rfind(prefix, 0) == 0) { return true; }
        }
        return false;
    }
};

template <size_t BATCH>
class Probe {
    using FusedRead = std::decay_t<decltype(fk::PerThreadRead<fk::ND::_2D, InputType>::build(std::declval<std::array<fk::Ptr2D<InputType>, BATCH>&>()))>;
    using FusedWrite = std::decay_t<decltype(fk::TensorWrite<OutputType>::build(std::declval<fk::Tensor<OutputType>&>()))>;

    fk::Stream m_stream;
    cudaStream_t m_raw;
    fk::Ptr2D<InputType> m_input;
    std::array<fk::Ptr2D<InputType>, BATCH> m_crops;
    std::array<fk::Ptr2D<OutputType>, BATCH> m_outputs;
    std::vector<ReadOp> m_reads;
    std::vector<WriteOp> m_writes;
    const decltype(fk::SaturateCast<InputType, OutputType>::build()) m_saturate{fk::SaturateCast<InputType, OutputType>::build()};
    const decltype(fk::Mul<OutputType>::build(1.0f)) m_mul{fk::Mul<OutputType>::build(1.0f)};
    const decltype(fk::Sub<OutputType>::build(1.f)) m_sub{fk::Sub<OutputType>::build(1.f)};
    const decltype(fk::Div<OutputType>::build(3.2f)) m_div{fk::Div<OutputType>::build(3.2f)};
    fk::Tensor<OutputType> m_tensor;
    std::optional<FusedRead> m_fusedRead;
    std::optional<FusedWrite> m_fusedWrite;
    StreamCache m_streams;
    cudaEvent_t m_start{nullptr}, m_stop{nullptr}, m_mid{nullptr};

public:
    Probe() : m_stream(), m_raw(m_stream), m_input(IMAGE_W, IMAGE_H), m_tensor(CROP_W, CROP_H, static_cast<int>(BATCH)) {
        fk::setTo(static_cast<InputType>(10u), m_input, m_stream);
        m_reads.reserve(BATCH);
        m_writes.reserve(BATCH);
        for (int crop = 0; crop < static_cast<int>(BATCH); ++crop) {
            m_crops[crop] = m_input.crop(fk::Point{crop, crop, 0}, fk::PtrDims<fk::ND::_2D>{static_cast<uint>(CROP_W), static_cast<uint>(CROP_H), static_cast<uint>(m_input.dims().pitch)});
            m_outputs[crop].Alloc(fk::Size(CROP_W, CROP_H));
            m_reads.push_back(fk::PerThreadRead<fk::ND::_2D, InputType>::build(m_crops[crop]));
            m_writes.push_back(fk::PerThreadWrite<fk::ND::_2D, OutputType>::build(m_outputs[crop]));
        }
        m_fusedRead.emplace(fk::PerThreadRead<fk::ND::_2D, InputType>::build(m_crops));
        m_fusedWrite.emplace(fk::TensorWrite<OutputType>::build(m_tensor));
        m_stream.sync();
        FKLB_CUDA(cudaEventCreate(&m_start));
        FKLB_CUDA(cudaEventCreate(&m_stop));
        FKLB_CUDA(cudaEventCreate(&m_mid));
    }
    ~Probe() {
        cudaEventDestroy(m_start);
        cudaEventDestroy(m_stop);
        cudaEventDestroy(m_mid);
    }

    // the launch of the pipeline of one crop, as in the experiment
    void enqueue(const int crop, cudaStream_t stream) {
        fk::Stream& s = stream == m_raw ? m_stream : m_streams.get(stream);
        fk::executeOperations<fk::TransformDPP<>>(s, m_reads[crop], m_saturate, m_mul, m_sub, m_div, m_writes[crop]);
    }
    void plainBaseline() {
        for (int crop = 0; crop < static_cast<int>(BATCH); ++crop) { enqueue(crop, m_raw); }
    }
    // the launch of the fused kernel, as in the experiment
    void fused() { fk::executeOperations<fk::TransformDPP<>>(m_stream, *m_fusedRead, m_saturate, m_mul, m_sub, m_div, *m_fusedWrite); }
    void touchFusedOperations() {
        touchBytes(&*m_fusedRead, sizeof(FusedRead));
        touchBytes(&*m_fusedWrite, sizeof(FusedWrite));
    }

    void clearOutputs() {
        for (auto& output : m_outputs) { FKLB_CUDA(cudaMemset2D(output.ptr().data, output.dims().pitch, 0, static_cast<size_t>(CROP_W) * sizeof(OutputType), CROP_H)); }
        fk::setTo(static_cast<OutputType>(0), m_tensor, m_stream);
        FKLB_CUDA(cudaDeviceSynchronize());
    }
    // the fused kernel against the plain launches, bit by bit, for the first, a middle and the last crop; returns the number of crops that differ
    int checkFused() {
        clearOutputs();
        plainBaseline();
        FKLB_CUDA(cudaStreamSynchronize(m_raw));
        std::vector<float> reference(static_cast<size_t>(CROP_W) * CROP_H * 3);
        const std::array<int, 3> picks{0, static_cast<int>(BATCH / 2), static_cast<int>(BATCH - 1)};
        for (int k = 0; k < 3; ++k) {
            FKLB_CUDA(cudaMemcpy2D(reference.data() + static_cast<size_t>(k) * CROP_W * CROP_H, CROP_W * sizeof(OutputType), m_outputs[picks[k]].ptr().data, m_outputs[picks[k]].dims().pitch,
                                   CROP_W * sizeof(OutputType), CROP_H, cudaMemcpyDeviceToHost));
        }
        fused();
        FKLB_CUDA(cudaStreamSynchronize(m_raw));
        int different = 0;
        for (int k = 0; k < 3; ++k) {
            auto plane = m_tensor.getPlane(picks[k]);
            std::vector<float> got(static_cast<size_t>(CROP_W) * CROP_H);
            FKLB_CUDA(cudaMemcpy2D(got.data(), CROP_W * sizeof(OutputType), plane.ptr().data, plane.dims().pitch, CROP_W * sizeof(OutputType), CROP_H, cudaMemcpyDeviceToHost));
            different += std::memcmp(got.data(), reference.data() + static_cast<size_t>(k) * CROP_W * CROP_H, got.size() * sizeof(float)) != 0 ? 1 : 0;
        }
        return different;
    }

    // The loop of the experiment (TimeMarkerTwo<GPU_NVIDIA>); between() runs after the first synchronisation and before the second start event
    template <class Baseline, class Between>
    Samples loop(const Options& options, Baseline&& baseline, Between&& between) {
        Samples samples;
        for (int it = 0; it < options.warmup + options.iters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t0 = Clock::now();
            baseline();
            const auto t1 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float first = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&first, m_start, m_stop));
            between();
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t2 = Clock::now();
            fused();
            const auto t3 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float second = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&second, m_start, m_stop));
            if (it >= options.warmup) {
                samples.firstMs.push_back(first);
                samples.secondMs.push_back(second);
                samples.baselineUs.push_back(microsecondsBetween(t0, t1));
                samples.fusedUs.push_back(microsecondsBetween(t2, t3));
            }
        }
        return samples;
    }

    // N launches of the fused kernel one after the other between two events; the time of one is the total over N
    Samples pipelined(const Options& options, const int launches) {
        Samples samples;
        for (int it = 0; it < options.warmup + options.iters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            const auto t0 = Clock::now();
            for (int k = 0; k < launches; ++k) { fused(); }
            const auto t1 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float total = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&total, m_start, m_stop));
            if (it >= options.warmup) {
                samples.firstMs.push_back(0.f);
                samples.secondMs.push_back(total / static_cast<float>(launches));
                samples.baselineUs.push_back(0.0);
                samples.fusedUs.push_back(microsecondsBetween(t0, t1) / launches);
            }
        }
        return samples;
    }

    // The fused launch enqueued behind a kernel of about 1 ms: the interval between the events around the fused kernel alone does not wait for the launch call
    Samples queuedBehindSpin(const Options& options) {
        Samples samples;
        for (int it = 0; it < options.warmup + options.iters; ++it) {
            FKLB_CUDA(cudaEventRecord(m_start, m_raw));
            spinKernel<<<1, 1, 0, m_raw>>>(SPIN_CYCLES);
            FKLB_CUDA(cudaEventRecord(m_mid, m_raw));
            const auto t2 = Clock::now();
            fused();
            const auto t3 = Clock::now();
            FKLB_CUDA(cudaEventRecord(m_stop, m_raw));
            FKLB_CUDA(cudaEventSynchronize(m_stop));
            float first = 0.f, second = 0.f;
            FKLB_CUDA(cudaEventElapsedTime(&first, m_start, m_mid));
            FKLB_CUDA(cudaEventElapsedTime(&second, m_mid, m_stop));
            if (it >= options.warmup) {
                samples.firstMs.push_back(first);
                samples.secondMs.push_back(second);
                samples.baselineUs.push_back(0.0);
                samples.fusedUs.push_back(microsecondsBetween(t2, t3));
            }
        }
        return samples;
    }

    int run(fklb::CsvWriter& csv, std::ofstream& raw, const Options& options) {
        int failures = 0;
        const int different = checkFused();
        std::cout << "crops " << BATCH << ": the fused kernel differs from the plain launches in " << different << " of 3 checked crops" << std::endl;
        if (different != 0) { ++failures; }
        const auto report = [&](const std::string& mode, const std::string& kind, const Samples& s) {
            std::vector<double> second(s.secondMs.begin(), s.secondMs.end());
            std::vector<double> first(s.firstMs.begin(), s.firstMs.end());
            const double secondMean = fklb::graphs::mean(second);
            csv.row(BATCH, mode, kind, options.iters, fklb::graphs::mean(first), secondMean, percentile(second, 0.5), percentile(second, 0.0), percentile(second, 1.0), percentile(second, 0.1),
                    percentile(second, 0.9), percentile(s.fusedUs, 0.5), percentile(s.fusedUs, 0.9), percentile(s.baselineUs, 0.5));
            csv.flush();
            for (size_t i = 0; i < s.secondMs.size(); ++i) {
                raw << BATCH << ',' << mode << ',' << i << ',' << s.firstMs[i] << ',' << s.secondMs[i] << ',' << s.fusedUs[i] << '\n';
            }
            std::cout << "crops " << BATCH << " " << mode << ": second mean " << secondMean * 1000.0 << " us, median " << percentile(second, 0.5) * 1000.0 << " us (p10 " << percentile(second, 0.1) * 1000.0
                      << ", p90 " << percentile(second, 0.9) * 1000.0 << "); fused launch call " << percentile(s.fusedUs, 0.5) << " us; first mean " << fklb::graphs::mean(first) << " ms" << std::endl;
        };
        const auto timed = [&](const std::string& mode, const std::string& kind, auto&& body) {
            if (!options.wants(mode)) { return; }
            try {
                const std::string range = mode + "@" + std::to_string(BATCH);
                nvtxRangePushA(range.c_str());
                const Samples s = body();
                nvtxRangePop();
                report(mode, kind, s);
            } catch (const std::exception& e) {
                ++failures;
                std::cerr << "crops " << BATCH << " " << mode << " failed: " << e.what() << std::endl;
                cudaGetLastError();
            }
        };
        const auto none = [] {};
        HostSweep sweep2(2u << 20), sweep64(64u << 20);
        const auto enqueueFn = [this](const int crop, cudaStream_t stream) { this->enqueue(crop, stream); };

        // the loops of the experiment, one per structure of the baseline
        fklb::graphs::Graph child, childChained, flatSerial, flatParallel;
        const int crops = static_cast<int>(BATCH);
        if (options.modes.empty() || options.modes.find("loop") != std::string::npos) {
            fklb::graphs::buildChild(child, crops, enqueueFn, false);
            fklb::graphs::buildChild(childChained, crops, enqueueFn, true);
            fklb::graphs::buildFlatSerial(flatSerial, crops, enqueueFn);
            fklb::graphs::buildFlatParallel(flatParallel, crops, enqueueFn);
        }
        const auto graphLaunch = [this](fklb::graphs::Graph& g) { return [this, &g] { FKLB_CUDA(cudaGraphLaunch(g.exec, m_raw)); }; };
        const auto plain = [this] { plainBaseline(); };
        const auto touch = [this] { touchFusedOperations(); };
        const auto pollute2 = [&] { sweep2.run(); };
        const auto pollute64 = [&] { sweep64.run(); };

        timed("loop_plain", "loop", [&] { return loop(options, plain, none); });
        timed("loop_child", "loop", [&] { return loop(options, graphLaunch(child), none); });
        timed("loop_child_chained", "loop", [&] { return loop(options, graphLaunch(childChained), none); });
        timed("loop_flat_serial", "loop", [&] { return loop(options, graphLaunch(flatSerial), none); });
        timed("loop_flat_parallel", "loop", [&] { return loop(options, graphLaunch(flatParallel), none); });
        timed("loop_flat_serial_pollute2MB", "loop", [&] { return loop(options, graphLaunch(flatSerial), pollute2); });
        timed("loop_flat_serial_pollute64MB", "loop", [&] { return loop(options, graphLaunch(flatSerial), pollute64); });
        timed("loop_plain_touch", "loop", [&] { return loop(options, plain, touch); });
        timed("loop_child_touch", "loop", [&] { return loop(options, graphLaunch(child), touch); });
        timed("alone_synced", "alone", [&] { return loop(options, none, none); });
        timed("alone_synced_pollute2MB", "alone", [&] { return loop(options, none, pollute2); });
        timed("alone_synced_pollute64MB", "alone", [&] { return loop(options, none, pollute64); });
        timed("alone_pipelined10", "alone", [&] { return pipelined(options, 10); });
        timed("alone_pipelined50", "alone", [&] { return pipelined(options, 50); });
        timed("alone_queued_behind_spin", "alone", [&] { return queuedBehindSpin(options); });
        return failures;
    }
};

template <size_t... Batches>
int dispatch(const std::vector<int>& cropsList, fklb::CsvWriter& csv, std::ofstream& raw, const Options& options, std::index_sequence<Batches...>) {
    int failures = 0;
    for (const int crops : cropsList) {
        bool found = false;
        const auto one = [&](auto batchConstant) {
            constexpr size_t batch = decltype(batchConstant)::value;
            if (static_cast<size_t>(crops) == batch) {
                found = true;
                Probe<batch> probe;
                failures += probe.run(csv, raw, options);
            }
        };
        (one(std::integral_constant<size_t, Batches>{}), ...);
        if (!found) {
            std::cerr << "hf_fused_time_probe: " << crops << " crops is not one of the compiled batch sizes" << std::endl;
            ++failures;
        }
    }
    return failures;
}
}  // namespace

int launch() {
    std::vector<int> cropsList;
    {
        const char* env = std::getenv("FKLB_FUSED_CROPS");
        std::stringstream ss(env != nullptr ? env : "101,301,661,831,1001,1101,1191");
        std::string item;
        while (std::getline(ss, item, ',')) { cropsList.push_back(std::stoi(item)); }
    }
    Options options;
    if (const char* env = std::getenv("FKLB_FUSED_ITERS")) { options.iters = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_FUSED_WARMUP")) { options.warmup = std::atoi(env); }
    if (const char* env = std::getenv("FKLB_FUSED_MODES")) { options.modes = env; }
    int failures = 0;
    try {
        fklb::CsvWriter csv("hf_fused_time_probe.csv", "E0 the time of the fused kernel of the HF experiment in the loops of the experiment and on its own (60 x 120 crops of a 3840 x 2160 image)",
                            {"crops", "mode", "kind", "iters", "first_ms_mean", "second_ms_mean", "second_ms_median", "second_ms_min", "second_ms_max", "second_ms_p10", "second_ms_p90",
                             "fused_launch_call_us_median", "fused_launch_call_us_p90", "baseline_launch_us_median"},
                            "second = the interval the experiment reports for the fused kernel (events recorded on the stream: loop_* and alone_synced as TimeMarkerTwo; alone_pipelined*: total over N launches divided by N; "
                            "alone_queued_behind_spin: between the event after a kernel of about 1 ms and the event after the fused kernel); first = the baseline interval (the spin kernel for "
                            "alone_queued_behind_spin); fused_launch_call_us = host time of the call that launches the fused kernel (per launch for alone_pipelined*)");
        std::ofstream raw("hf_fused_time_probe_raw.csv");
        raw << "crops,mode,iteration,first_ms,second_ms,fused_launch_call_us\n";
        raw << std::setprecision(9);
        failures += dispatch(cropsList, csv, raw, options, std::index_sequence<101, 301, 661, 831, 1001, 1101, 1191>{});
    } catch (const std::exception& e) {
        std::cerr << "hf_fused_time_probe: " << e.what() << std::endl;
        return -1;
    }
    return failures == 0 ? 0 : -1;
}
